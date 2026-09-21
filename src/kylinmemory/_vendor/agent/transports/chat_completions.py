"""OpenAI Chat Completions transport.

Handles the default api_mode ('chat_completions') used by ~16 OpenAI-compatible
providers (OpenRouter, Nous, NVIDIA, Qwen, Ollama, DeepSeek, xAI, Kimi, etc.).

Messages and tools are already in OpenAI format — convert_messages and
convert_tools are near-identity.  The complexity lives in build_kwargs
which has provider-specific conditionals for max_tokens defaults,
reasoning configuration, temperature handling, and extra_body assembly.
"""
import copy
from typing import Any, Dict, List, Optional
from kylinmemory._vendor.agent.lmstudio_reasoning import resolve_lmstudio_effort
from kylinmemory._vendor.agent.moonshot_schema import is_moonshot_model, sanitize_moonshot_tools
from kylinmemory._vendor.agent.prompt_builder import DEVELOPER_ROLE_MODELS
from kylinmemory._vendor.agent.transports.base import ProviderTransport
from kylinmemory._vendor.agent.transports.types import NormalizedResponse, ToolCall, Usage

def _build_gemini_thinking_config(model: str, reasoning_config: dict | None) -> dict | None:
    """Translate Hermes/OpenRouter-style reasoning config to Gemini thinkingConfig."""
    if reasoning_config is None or not isinstance(reasoning_config, dict):
        return None
    normalized_model = (model or '').strip().lower()
    if normalized_model.startswith('google/'):
        normalized_model = normalized_model.split('/', 1)[1]
    if not normalized_model.startswith('gemini'):
        return None
    if reasoning_config.get('enabled') is False:
        return {'includeThoughts': False}
    effort = str(reasoning_config.get('effort', 'medium') or 'medium').strip().lower()
    if effort == 'none':
        return {'includeThoughts': False}
    thinking_config: Dict[str, Any] = {'includeThoughts': True}
    if normalized_model.startswith('gemini-2.5-'):
        return thinking_config
    if effort not in {'minimal', 'low', 'medium', 'high', 'xhigh'}:
        effort = 'medium'
    if normalized_model.startswith(('gemini-3', 'gemini-3.1')):
        if 'flash' in normalized_model:
            if effort in {'minimal', 'low'}:
                thinking_config['thinkingLevel'] = 'low'
            elif effort in {'high', 'xhigh'}:
                thinking_config['thinkingLevel'] = 'high'
            else:
                thinking_config['thinkingLevel'] = 'medium'
        elif 'pro' in normalized_model:
            thinking_config['thinkingLevel'] = 'high' if effort in {'high', 'xhigh'} else 'low'
    return thinking_config

def _snake_case_gemini_thinking_config(config: dict | None) -> dict | None:
    """Convert Gemini thinking config keys to the OpenAI-compat field names."""
    if not isinstance(config, dict) or not config:
        return None
    translated: Dict[str, Any] = {}
    if isinstance(config.get('includeThoughts'), bool):
        translated['include_thoughts'] = config['includeThoughts']
    if isinstance(config.get('thinkingLevel'), str) and config['thinkingLevel'].strip():
        translated['thinking_level'] = config['thinkingLevel'].strip().lower()
    if isinstance(config.get('thinkingBudget'), (int, float)):
        translated['thinking_budget'] = int(config['thinkingBudget'])
    return translated or None

def _is_gemini_openai_compat_base_url(base_url: Any) -> bool:
    normalized = str(base_url or '').strip().rstrip('/').lower()
    if not normalized:
        return False
    if 'generativelanguage.googleapis.com' not in normalized:
        return False
    return normalized.endswith('/openai')

class ChatCompletionsTransport(ProviderTransport):
    """Transport for api_mode='chat_completions'.

    The default path for OpenAI-compatible providers.
    """

    @property
    def api_mode(self) -> str:
        return 'chat_completions'

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        """Messages are already in OpenAI format — strip internal fields
        that strict chat-completions providers reject with HTTP 400/422.

        Strips:

        - Codex Responses API fields: ``codex_reasoning_items`` /
          ``codex_message_items`` on the message, ``call_id`` /
          ``response_item_id`` on ``tool_calls`` entries.
        - ``tool_name`` on tool-result messages — written by
          ``make_tool_result_message()`` for the SQLite FTS index, but not
          part of the Chat Completions schema. Strict providers (Fireworks,
          Moonshot/Kimi) reject any payload containing it with
          ``Extra inputs are not permitted, field: 'messages[N].tool_name'``.
          Permissive providers (OpenRouter, MiniMax) silently ignore the
          field, which masked the bug for months.
        """
        needs_sanitize = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if 'codex_reasoning_items' in msg or 'codex_message_items' in msg or 'tool_name' in msg:
                needs_sanitize = True
                break
            tool_calls = msg.get('tool_calls')
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict) and ('call_id' in tc or 'response_item_id' in tc):
                        needs_sanitize = True
                        break
                if needs_sanitize:
                    break
        if not needs_sanitize:
            return messages
        sanitized = copy.deepcopy(messages)
        for msg in sanitized:
            if not isinstance(msg, dict):
                continue
            msg.pop('codex_reasoning_items', None)
            msg.pop('codex_message_items', None)
            msg.pop('tool_name', None)
            tool_calls = msg.get('tool_calls')
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc.pop('call_id', None)
                        tc.pop('response_item_id', None)
        return sanitized

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Tools are already in OpenAI format — identity."""
        return tools

    def build_kwargs(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None=None, **params) -> dict[str, Any]:
        """Build chat.completions.create() kwargs.

        params (all optional):
            timeout: float — API call timeout
            max_tokens: int | None — user-configured max tokens
            ephemeral_max_output_tokens: int | None — one-shot override
            max_tokens_param_fn: callable — returns {max_tokens: N} or {max_completion_tokens: N}
            reasoning_config: dict | None
            request_overrides: dict | None
            session_id: str | None
            model_lower: str — lowercase model name for pattern matching
            # Provider profile path (all per-provider quirks live in providers/)
            provider_profile: ProviderProfile | None — when present, delegates to
                _build_kwargs_from_profile(); all flag params below are bypassed.
            # Legacy-path flags — only used when provider_profile is None
            # (i.e. custom / unregistered providers). Known providers all go
            # through provider_profile.
            is_openrouter: bool
            is_nous: bool
            is_qwen_portal: bool
            is_github_models: bool
            is_nvidia_nim: bool
            is_kimi: bool
            is_tokenhub: bool
            is_lmstudio: bool
            is_custom_provider: bool
            ollama_num_ctx: int | None
            # Provider routing
            provider_preferences: dict | None
            # Qwen-specific
            qwen_prepare_fn: callable | None — runs AFTER codex sanitization
            qwen_prepare_inplace_fn: callable | None — in-place variant for deepcopied lists
            qwen_session_metadata: dict | None
            # Temperature
            fixed_temperature: Any — from _fixed_temperature_for_model()
            omit_temperature: bool
            # Reasoning
            supports_reasoning: bool
            github_reasoning_extra: dict | None
            lmstudio_reasoning_options: list[str] | None  # raw allowed_options from /api/v1/models
            # Claude on OpenRouter/Nous max output
            anthropic_max_output: int | None
            extra_body_additions: dict | None
        """
        sanitized = self.convert_messages(messages)
        _profile = params.get('provider_profile')
        if _profile:
            return self._build_kwargs_from_profile(_profile, model, sanitized, tools, params)
        model_lower = params.get('model_lower', (model or '').lower())
        if sanitized and isinstance(sanitized[0], dict) and (sanitized[0].get('role') == 'system') and any((p in model_lower for p in DEVELOPER_ROLE_MODELS)):
            sanitized = list(sanitized)
            sanitized[0] = {**sanitized[0], 'role': 'developer'}
        api_kwargs: dict[str, Any] = {'model': model, 'messages': sanitized}
        timeout = params.get('timeout')
        if timeout is not None:
            api_kwargs['timeout'] = timeout
        if tools:
            if is_moonshot_model(model):
                tools = sanitize_moonshot_tools(tools)
            api_kwargs['tools'] = tools
        max_tokens_fn = params.get('max_tokens_param_fn')
        ephemeral = params.get('ephemeral_max_output_tokens')
        max_tokens = params.get('max_tokens')
        anthropic_max_out = params.get('anthropic_max_output')
        is_nvidia_nim = params.get('is_nvidia_nim', False)
        is_kimi = params.get('is_kimi', False)
        is_tokenhub = params.get('is_tokenhub', False)
        reasoning_config = params.get('reasoning_config')
        if ephemeral is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(ephemeral))
        elif max_tokens is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(max_tokens))
        elif anthropic_max_out is not None:
            api_kwargs['max_tokens'] = anthropic_max_out
        if is_kimi:
            _kimi_thinking_off = bool(reasoning_config and isinstance(reasoning_config, dict) and (reasoning_config.get('enabled') is False))
            if not _kimi_thinking_off:
                _kimi_effort = 'medium'
                if reasoning_config and isinstance(reasoning_config, dict):
                    _e = (reasoning_config.get('effort') or '').strip().lower()
                    if _e in {'low', 'medium', 'high'}:
                        _kimi_effort = _e
                api_kwargs['reasoning_effort'] = _kimi_effort
        if is_tokenhub:
            _tokenhub_thinking_off = bool(reasoning_config and isinstance(reasoning_config, dict) and (reasoning_config.get('enabled') is False))
            if not _tokenhub_thinking_off:
                _tokenhub_effort = 'high'
                if reasoning_config and isinstance(reasoning_config, dict):
                    _e = (reasoning_config.get('effort') or '').strip().lower()
                    if _e in {'low', 'medium', 'high'}:
                        _tokenhub_effort = _e
                api_kwargs['reasoning_effort'] = _tokenhub_effort
        if params.get('is_lmstudio', False) and params.get('supports_reasoning', False):
            _lm_effort = resolve_lmstudio_effort(reasoning_config, params.get('lmstudio_reasoning_options'))
            if _lm_effort is not None:
                api_kwargs['reasoning_effort'] = _lm_effort
        extra_body: dict[str, Any] = {}
        is_openrouter = params.get('is_openrouter', False)
        is_nous = params.get('is_nous', False)
        is_github_models = params.get('is_github_models', False)
        provider_name = str(params.get('provider_name') or '').strip().lower()
        base_url = params.get('base_url')
        provider_prefs = params.get('provider_preferences')
        if provider_prefs and is_openrouter:
            extra_body['provider'] = provider_prefs
        if is_openrouter and model == 'openrouter/pareto-code':
            _pareto_score = params.get('openrouter_min_coding_score')
            if _pareto_score is not None and _pareto_score != '':
                try:
                    _pareto_score_f = float(_pareto_score)
                except (TypeError, ValueError):
                    _pareto_score_f = None
                if _pareto_score_f is not None and 0.0 <= _pareto_score_f <= 1.0:
                    extra_body['plugins'] = [{'id': 'pareto-router', 'min_coding_score': _pareto_score_f}]
        if is_kimi:
            _kimi_thinking_enabled = True
            if reasoning_config and isinstance(reasoning_config, dict):
                if reasoning_config.get('enabled') is False:
                    _kimi_thinking_enabled = False
            extra_body['thinking'] = {'type': 'enabled' if _kimi_thinking_enabled else 'disabled'}
        if params.get('supports_reasoning', False) and (not params.get('is_lmstudio', False)):
            if is_github_models:
                gh_reasoning = params.get('github_reasoning_extra')
                if gh_reasoning is not None:
                    extra_body['reasoning'] = gh_reasoning
            else:
                extra_body['reasoning'] = {'enabled': True, 'effort': 'medium'}
        if provider_name == 'gemini':
            raw_thinking_config = _build_gemini_thinking_config(model, reasoning_config)
            if _is_gemini_openai_compat_base_url(base_url):
                thinking_config = _snake_case_gemini_thinking_config(raw_thinking_config)
                if thinking_config:
                    openai_compat_extra = extra_body.get('extra_body', {})
                    google_extra = openai_compat_extra.get('google', {})
                    google_extra['thinking_config'] = thinking_config
                    openai_compat_extra['google'] = google_extra
                    extra_body['extra_body'] = openai_compat_extra
            elif raw_thinking_config:
                extra_body['thinking_config'] = raw_thinking_config
        elif provider_name == 'google-gemini-cli':
            thinking_config = _build_gemini_thinking_config(model, reasoning_config)
            if thinking_config:
                extra_body['thinking_config'] = thinking_config
        additions = params.get('extra_body_additions')
        if additions:
            extra_body.update(additions)
        try:
            from kylinmemory._vendor.agent.request_priority import merge_priority_into_extra_body
            extra_body = merge_priority_into_extra_body(extra_body)
        except Exception:
            pass
        if extra_body:
            api_kwargs['extra_body'] = extra_body
        overrides = params.get('request_overrides')
        if overrides:
            for k, v in overrides.items():
                if k == 'extra_body' and isinstance(v, dict):
                    extra_body.update(v)
                else:
                    api_kwargs[k] = v
        try:
            from kylinmemory._vendor.agent.request_priority import merge_priority_into_extra_body
            extra_body = merge_priority_into_extra_body(extra_body)
        except Exception:
            pass
        if extra_body:
            api_kwargs['extra_body'] = extra_body
        return api_kwargs

    def _build_kwargs_from_profile(self, profile, model, sanitized, tools, params):
        """Build API kwargs using a ProviderProfile — single path, no legacy flags.

        This method replaces the entire flag-based kwargs assembly when a
        provider_profile is passed. Every quirk comes from the profile object.
        """
        from kylinmemory._vendor.providers.base import OMIT_TEMPERATURE
        sanitized = profile.prepare_messages(sanitized)
        _model_lower = (model or '').lower()
        if sanitized and isinstance(sanitized[0], dict) and (sanitized[0].get('role') == 'system') and any((p in _model_lower for p in DEVELOPER_ROLE_MODELS)):
            sanitized = list(sanitized)
            sanitized[0] = {**sanitized[0], 'role': 'developer'}
        api_kwargs: dict[str, Any] = {'model': model, 'messages': sanitized}
        if profile.fixed_temperature is OMIT_TEMPERATURE:
            pass
        elif profile.fixed_temperature is not None:
            api_kwargs['temperature'] = profile.fixed_temperature
        else:
            temp = params.get('temperature')
            if temp is not None:
                api_kwargs['temperature'] = temp
        timeout = params.get('timeout')
        if timeout is not None:
            api_kwargs['timeout'] = timeout
        if tools:
            if is_moonshot_model(model):
                tools = sanitize_moonshot_tools(tools)
            api_kwargs['tools'] = tools
        max_tokens_fn = params.get('max_tokens_param_fn')
        ephemeral = params.get('ephemeral_max_output_tokens')
        user_max = params.get('max_tokens')
        anthropic_max = params.get('anthropic_max_output')
        if ephemeral is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(ephemeral))
        elif user_max is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(user_max))
        elif profile.default_max_tokens and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(profile.default_max_tokens))
        elif anthropic_max is not None:
            api_kwargs['max_tokens'] = anthropic_max
        reasoning_config = params.get('reasoning_config')
        extra_body_from_profile, top_level_from_profile = profile.build_api_kwargs_extras(reasoning_config=reasoning_config, supports_reasoning=params.get('supports_reasoning', False), qwen_session_metadata=params.get('qwen_session_metadata'), model=model, ollama_num_ctx=params.get('ollama_num_ctx'), session_id=params.get('session_id'))
        api_kwargs.update(top_level_from_profile)
        extra_body: dict[str, Any] = {}
        profile_body = profile.build_extra_body(session_id=params.get('session_id'), provider_preferences=params.get('provider_preferences'), model=model, base_url=params.get('base_url'), reasoning_config=reasoning_config, openrouter_min_coding_score=params.get('openrouter_min_coding_score'))
        if profile_body:
            extra_body.update(profile_body)
        if extra_body_from_profile:
            extra_body.update(extra_body_from_profile)
        additions = params.get('extra_body_additions')
        if additions:
            extra_body.update(additions)
        try:
            from kylinmemory._vendor.agent.request_priority import merge_priority_into_extra_body
            extra_body = merge_priority_into_extra_body(extra_body)
        except Exception:
            pass
        overrides = params.get('request_overrides')
        if overrides:
            for k, v in overrides.items():
                if k == 'extra_body' and isinstance(v, dict):
                    extra_body.update(v)
                else:
                    api_kwargs[k] = v
        try:
            from kylinmemory._vendor.agent.request_priority import merge_priority_into_extra_body
            extra_body = merge_priority_into_extra_body(extra_body)
        except Exception:
            pass
        if extra_body:
            api_kwargs['extra_body'] = extra_body
        return api_kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize OpenAI ChatCompletion to NormalizedResponse.

        For chat_completions, this is near-identity — the response is already
        in OpenAI format.  extra_content on tool_calls (Gemini thought_signature)
        is preserved via ToolCall.provider_data.  reasoning_details (OpenRouter
        unified format) and reasoning_content (DeepSeek/Moonshot) are also
        preserved for downstream replay.
        """
        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason or 'stop'
        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tc_provider_data: dict[str, Any] = {}
                extra = getattr(tc, 'extra_content', None)
                if extra is None and hasattr(tc, 'model_extra'):
                    extra = (tc.model_extra or {}).get('extra_content')
                if extra is not None:
                    if hasattr(extra, 'model_dump'):
                        try:
                            extra = extra.model_dump()
                        except Exception:
                            pass
                    tc_provider_data['extra_content'] = extra
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments, provider_data=tc_provider_data or None))
        usage = None
        if hasattr(response, 'usage') and response.usage:
            u = response.usage
            usage = Usage(prompt_tokens=getattr(u, 'prompt_tokens', 0) or 0, completion_tokens=getattr(u, 'completion_tokens', 0) or 0, total_tokens=getattr(u, 'total_tokens', 0) or 0)
        reasoning = getattr(msg, 'reasoning', None)
        reasoning_content = getattr(msg, 'reasoning_content', None)
        if reasoning_content is None and hasattr(msg, 'model_extra'):
            model_extra = getattr(msg, 'model_extra', None) or {}
            if isinstance(model_extra, dict) and 'reasoning_content' in model_extra:
                reasoning_content = model_extra['reasoning_content']
        provider_data: Dict[str, Any] = {}
        if reasoning_content is not None:
            provider_data['reasoning_content'] = reasoning_content
        rd = getattr(msg, 'reasoning_details', None)
        if rd:
            provider_data['reasoning_details'] = rd
        return NormalizedResponse(content=msg.content, tool_calls=tool_calls, finish_reason=finish_reason, reasoning=reasoning, usage=usage, provider_data=provider_data or None)

    def validate_response(self, response: Any) -> bool:
        """Check that response has valid choices."""
        if response is None:
            return False
        if not hasattr(response, 'choices') or response.choices is None:
            return False
        if not response.choices:
            return False
        return True

    def extract_cache_stats(self, response: Any) -> dict[str, int] | None:
        """Extract OpenRouter/OpenAI cache stats from prompt_tokens_details."""
        usage = getattr(response, 'usage', None)
        if usage is None:
            return None
        details = getattr(usage, 'prompt_tokens_details', None)
        if details is None:
            return None
        cached = getattr(details, 'cached_tokens', 0) or 0
        written = getattr(details, 'cache_write_tokens', 0) or 0
        if cached or written:
            return {'cached_tokens': cached, 'creation_tokens': written}
        return None
from kylinmemory._vendor.agent.transports import register_transport
register_transport('chat_completions', ChatCompletionsTransport)

