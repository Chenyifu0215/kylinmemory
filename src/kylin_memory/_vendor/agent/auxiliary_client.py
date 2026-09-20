"""Shared auxiliary client router for side tasks.

Provides a single resolution chain so every consumer (context compression,
session search, web extraction, vision analysis, browser vision) picks up
the best available backend without duplicating fallback logic.

Resolution order for text tasks (auto mode):
  1. User's main provider + main model (used regardless of provider type —
     aggregators, direct API-key providers, native Anthropic, Codex, etc.)
  2. OpenRouter  (OPENROUTER_API_KEY)
  3. Nous Portal (~/.kylin-agent-runtime/auth.json active provider)
  4. Custom endpoint (config.yaml model.base_url + OPENAI_API_KEY)
  5. Native Anthropic
  6. Direct API-key providers (z.ai/GLM, Kimi/Moonshot, MiniMax, MiniMax-CN)
  7. None

Resolution order for vision/multimodal tasks (auto mode):
  1. Selected main provider, if it is one of the supported vision backends below
  2. OpenRouter
  3. Nous Portal
  4. Native Anthropic
  5. Custom endpoint (for local vision models: Qwen-VL, LLaVA, Pixtral, etc.)
  6. None

Codex OAuth (ChatGPT-account auth) is intentionally NOT in either
fallback chain: OpenAI gates this endpoint behind an undocumented,
shifting model allow-list, so "just try Codex with a hardcoded model"
rots on its own.  Codex is used only when the user's main provider *is*
openai-codex (Step 1 above) or when a caller explicitly requests it with
a model (auxiliary.<task>.provider + auxiliary.<task>.model).

Per-task overrides are configured in config.yaml under the ``auxiliary:`` section
(e.g. ``auxiliary.vision.provider``, ``auxiliary.compression.model``).
Default "auto" follows the chains above.

Payment / credit exhaustion fallback:
  When a resolved provider returns HTTP 402 or a credit-related error,
  call_llm() automatically retries with the next available provider in the
  auto-detection chain.  This handles the common case where a user depletes
  their OpenRouter balance but has Codex OAuth or another provider available.
"""
import json
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse, parse_qs, urlunparse
if TYPE_CHECKING:
    from openai import OpenAI
_OPENAI_CLS_CACHE: Optional[type] = None

def _load_openai_cls() -> type:
    """Import and cache ``openai.OpenAI``."""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls
        _OPENAI_CLS_CACHE = _cls
    return _OPENAI_CLS_CACHE

class _OpenAIProxy:
    """Module-level proxy that looks like the ``openai.OpenAI`` class.

    Forwards ``OpenAI(...)`` calls and ``isinstance(x, OpenAI)`` checks to the
    real SDK class, importing the SDK lazily on first use.
    """
    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())

    def __repr__(self):
        return '<lazy openai.OpenAI proxy>'
OpenAI = _OpenAIProxy()
from kylin_memory._vendor.agent.credential_pool import load_pool
from kylin_memory._vendor.kylin_agent_runtime_cli.config import get_hermes_home
from kylin_memory._vendor.kylin_agent_runtime_constants import OPENROUTER_BASE_URL
from kylin_memory._vendor.utils import base_url_host_matches, base_url_hostname, normalize_proxy_env_vars
logger = logging.getLogger(__name__)

def _safe_isinstance(obj: Any, maybe_type: Any) -> bool:
    """Return False instead of raising when a patched symbol is not a type."""
    try:
        return isinstance(obj, maybe_type)
    except TypeError:
        return False

def _extract_url_query_params(url: str):
    """Extract query params from URL, return (clean_url, default_query dict or None)."""
    parsed = urlparse(url)
    if parsed.query:
        clean = urlunparse(parsed._replace(query=''))
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        return (clean, params)
    return (url, None)
_stale_base_url_warned = False
_PROVIDER_ALIASES = {'google': 'gemini', 'google-gemini': 'gemini', 'google-ai-studio': 'gemini', 'x-ai': 'xai', 'x.ai': 'xai', 'grok': 'xai', 'glm': 'zai', 'z-ai': 'zai', 'z.ai': 'zai', 'zhipu': 'zai', 'kimi': 'kimi-coding', 'moonshot': 'kimi-coding', 'kimi-cn': 'kimi-coding-cn', 'moonshot-cn': 'kimi-coding-cn', 'gmi-cloud': 'gmi', 'gmicloud': 'gmi', 'minimax-china': 'minimax-cn', 'minimax_cn': 'minimax-cn', 'claude': 'anthropic', 'claude-code': 'anthropic', 'github': 'copilot', 'github-copilot': 'copilot', 'github-model': 'copilot', 'github-models': 'copilot', 'github-copilot-acp': 'copilot-acp', 'copilot-acp-agent': 'copilot-acp', 'tencent': 'tencent-tokenhub', 'tokenhub': 'tencent-tokenhub', 'tencent-cloud': 'tencent-tokenhub', 'tencentmaas': 'tencent-tokenhub'}

def _normalize_aux_provider(provider: Optional[str]) -> str:
    normalized = (provider or 'auto').strip().lower()
    if normalized.startswith('custom:'):
        suffix = normalized.split(':', 1)[1].strip()
        if not suffix:
            return 'custom'
        normalized = suffix
    if normalized == 'codex':
        return 'openai-codex'
    if normalized == 'main':
        main_prov = (_read_main_provider() or '').strip().lower()
        if main_prov and main_prov not in {'auto', 'main', ''}:
            normalized = main_prov
        else:
            return 'custom'
    return _PROVIDER_ALIASES.get(normalized, normalized)
OMIT_TEMPERATURE: object = object()

def _is_kimi_model(model: Optional[str]) -> bool:
    """True for any Kimi / Moonshot model that manages temperature server-side."""
    bare = (model or '').strip().lower().rsplit('/', 1)[-1]
    return bare.startswith('kimi-') or bare == 'kimi'

def _is_arcee_trinity_thinking(model: Optional[str]) -> bool:
    """True for Arcee Trinity Large Thinking (direct or via OpenRouter)."""
    bare = (model or '').strip().lower().rsplit('/', 1)[-1]
    return bare == 'trinity-large-thinking'

def _fixed_temperature_for_model(model: Optional[str], base_url: Optional[str]=None) -> 'Optional[float] | object':
    """Return a temperature directive for models with strict contracts.

    Returns:
        ``OMIT_TEMPERATURE`` — caller must remove the ``temperature`` key so the
            provider chooses its own default.  Used for all Kimi / Moonshot
            models whose gateway selects temperature server-side.
        ``float`` — a specific value the caller must use (reserved for future
            models with fixed-temperature contracts).
        ``None`` — no override; caller should use its own default.
    """
    if _is_kimi_model(model):
        logger.debug('Omitting temperature for Kimi model %r (server-managed)', model)
        return OMIT_TEMPERATURE
    if _is_arcee_trinity_thinking(model):
        return 0.5
    return None

def _is_deepseek_thinking_model(model: Optional[str], base_url: Optional[str]=None, provider: Optional[str]=None) -> bool:
    """Return whether a DeepSeek route defaults to server-side thinking.

    DeepSeek's V4/reasoner OpenAI-compatible endpoint rejects a named
    ``tool_choice`` while thinking is enabled.  Auxiliary structured-memory
    calls must therefore explicitly disable thinking before requesting their
    function tool.  Keep V3 ``deepseek-chat`` unchanged.
    """
    bare = (model or '').strip().lower().rsplit('/', 1)[-1]
    is_thinking_model = bare.startswith('deepseek-v') and (not bare.startswith('deepseek-v3')) or bare == 'deepseek-reasoner'
    if not is_thinking_model:
        return False
    host = base_url_hostname(base_url or '')
    provider_name = str(provider or '').strip().lower()
    return provider_name == 'deepseek' or host in {'api.deepseek.com', 'api.deepseek.com.cn'}

def _compression_threshold_for_model(model: Optional[str]) -> Optional[float]:
    """Return a context-compression threshold override for specific models.

    The threshold is the fraction of the model's context window that must be
    consumed before Hermes triggers summarization.  Higher values delay
    compression and preserve more raw context.

    Returns a float in (0, 1] to override the global ``compression.threshold``
    config value, or ``None`` to leave the user's config value unchanged.
    """
    if _is_arcee_trinity_thinking(model):
        return 0.75
    return None

def _get_aux_model_for_provider(provider_id: str) -> str:
    """Return the cheap auxiliary model for a provider.

    Reads from ProviderProfile.default_aux_model first, falling back to the
    legacy hardcoded dict for providers that predate the profiles system.
    """
    try:
        from kylin_memory._vendor.providers import get_provider_profile
        _p = get_provider_profile(provider_id)
        if _p and _p.default_aux_model:
            return _p.default_aux_model
    except Exception:
        pass
    return _API_KEY_PROVIDER_AUX_MODELS_FALLBACK.get(provider_id, '')
_API_KEY_PROVIDER_AUX_MODELS_FALLBACK: Dict[str, str] = {'gemini': 'gemini-3-flash-preview', 'zai': 'glm-4.5-flash', 'kimi-coding': 'kimi-k2-turbo-preview', 'stepfun': 'step-3.5-flash', 'kimi-coding-cn': 'kimi-k2-turbo-preview', 'gmi': 'google/gemini-3.1-flash-lite-preview', 'minimax': 'MiniMax-M2.7', 'minimax-oauth': 'MiniMax-M2.7-highspeed', 'minimax-cn': 'MiniMax-M2.7', 'anthropic': 'claude-haiku-4-5-20251001', 'ai-gateway': 'google/gemini-3-flash', 'opencode-zen': 'gemini-3-flash', 'opencode-go': 'glm-5', 'kilocode': 'google/gemini-3-flash-preview', 'ollama-cloud': 'nemotron-3-nano:30b', 'tencent-tokenhub': 'hy3-preview'}
_API_KEY_PROVIDER_AUX_MODELS: Dict[str, str] = _API_KEY_PROVIDER_AUX_MODELS_FALLBACK
_PROVIDER_VISION_MODELS: Dict[str, str] = {'xiaomi': 'mimo-v2.5', 'zai': 'glm-5v-turbo'}
_PROVIDERS_WITHOUT_VISION: frozenset = frozenset({'kimi-coding', 'kimi-coding-cn'})
_OR_HEADERS_BASE = {'HTTP-Referer': 'https://hermes-agent.nousresearch.com', 'X-Title': 'Hermes Agent', 'X-OpenRouter-Categories': 'productivity,cli-agent'}
_TRUTHY_ENV_VALUES = frozenset({'1', 'true', 'yes', 'on'})

def build_or_headers(or_config: dict | None=None) -> dict:
    """Build OpenRouter headers, optionally including response-cache headers.

    Precedence for response cache: env var > config.yaml > default (enabled).

    Environment variables:
        ``HERMES_OPENROUTER_CACHE`` — truthy (``1``/``true``/``yes``/``on``)
            enables caching; ``0``/``false``/``no``/``off`` disables.
            Overrides ``openrouter.response_cache`` in config.yaml.
        ``HERMES_OPENROUTER_CACHE_TTL`` — integer seconds (1-86400).
            Overrides ``openrouter.response_cache_ttl`` in config.yaml.

    *or_config* is the ``openrouter`` section from config.yaml.  When *None*,
    falls back to reading config from disk via ``load_config()``.
    """
    headers = dict(_OR_HEADERS_BASE)
    if or_config is None:
        try:
            from kylin_memory._vendor.kylin_agent_runtime_cli.config import load_config
            or_config = load_config().get('openrouter', {})
        except Exception:
            or_config = {}
    env_cache = os.environ.get('HERMES_OPENROUTER_CACHE', '').strip().lower()
    if env_cache:
        cache_enabled = env_cache in _TRUTHY_ENV_VALUES
    else:
        cache_enabled = or_config.get('response_cache', False)
    if not cache_enabled:
        return headers
    headers['X-OpenRouter-Cache'] = 'true'
    env_ttl = os.environ.get('HERMES_OPENROUTER_CACHE_TTL', '').strip()
    if env_ttl:
        if env_ttl.isdigit():
            ttl = int(env_ttl)
            if 1 <= ttl <= 86400:
                headers['X-OpenRouter-Cache-TTL'] = str(ttl)
    else:
        ttl = or_config.get('response_cache_ttl', 300)
        if isinstance(ttl, (int, float)) and 1 <= ttl <= 86400:
            headers['X-OpenRouter-Cache-TTL'] = str(int(ttl))
    return headers
_NVIDIA_NIM_CLOUD_HEADERS = {'X-BILLING-INVOKE-ORIGIN': 'KylinAgent'}

def build_nvidia_nim_headers(base_url: str | None) -> dict:
    """Return NVIDIA NIM cloud attribution headers for build.nvidia.com traffic."""
    if base_url_host_matches(str(base_url or ''), 'integrate.api.nvidia.com'):
        return dict(_NVIDIA_NIM_CLOUD_HEADERS)
    return {}
from kylin_memory._vendor.kylin_agent_runtime_cli import __version__ as _RUNTIME_VERSION
_AI_GATEWAY_HEADERS = {'HTTP-Referer': 'https://www.gitee.com/openkylin/agent-runtime', 'X-Title': 'Kylin Agent', 'User-Agent': f'KylinAgent/{_RUNTIME_VERSION}'}
from kylin_memory._vendor.agent.portal_tags import nous_portal_tags as _nous_portal_tags

def _nous_extra_body() -> dict:
    """Return a fresh Nous Portal ``extra_body`` dict.

    Computed at call time so a hot-reloaded ``kylin_agent_runtime_cli.__version__`` is
    reflected without restarting long-running processes.
    """
    return {'tags': _nous_portal_tags()}
NOUS_EXTRA_BODY = _nous_extra_body()
auxiliary_is_nous: bool = False
_OPENROUTER_MODEL = 'google/gemini-3-flash-preview'
_NOUS_MODEL = 'google/gemini-3-flash-preview'
_NOUS_DEFAULT_BASE_URL = 'https://inference-api.nousresearch.com/v1'
_ANTHROPIC_DEFAULT_BASE_URL = 'https://api.anthropic.com'
_AUTH_JSON_PATH = get_hermes_home() / 'auth.json'
_CODEX_AUX_BASE_URL = 'https://chatgpt.com/backend-api/codex'

def _codex_cloudflare_headers(access_token: str) -> Dict[str, str]:
    """Headers required to avoid Cloudflare 403s on chatgpt.com/backend-api/codex.

    The Cloudflare layer in front of the Codex endpoint whitelists a small set of
    first-party originators (``codex_cli_rs``, ``codex_vscode``, ``codex_sdk_ts``,
    anything starting with ``Codex``). Requests from non-residential IPs (VPS,
    server-hosted agents) that don't advertise an allowed originator are served
    a 403 with ``cf-mitigated: challenge`` regardless of auth correctness.

    We pin ``originator: codex_cli_rs`` to match the upstream codex-rs CLI, set
    ``User-Agent`` to a codex_cli_rs-shaped string (beats SDK fingerprinting),
    and extract ``ChatGPT-Account-ID`` (canonical casing, from codex-rs
    ``auth.rs``) out of the OAuth JWT's ``chatgpt_account_id`` claim.

    Malformed tokens are tolerated — we drop the account-ID header rather than
    raise, so a bad token still surfaces as an auth error (401) instead of a
    crash at client construction.
    """
    headers = {'User-Agent': 'codex_cli_rs/0.0.0 (Hermes Agent)', 'originator': 'codex_cli_rs'}
    if not isinstance(access_token, str) or not access_token.strip():
        return headers
    try:
        import base64
        parts = access_token.split('.')
        if len(parts) < 2:
            return headers
        payload_b64 = parts[1] + '=' * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        acct_id = claims.get('https://api.openai.com/auth', {}).get('chatgpt_account_id')
        if isinstance(acct_id, str) and acct_id:
            headers['ChatGPT-Account-ID'] = acct_id
    except Exception:
        pass
    return headers

def _to_openai_base_url(base_url: str) -> str:
    """Normalize an Anthropic-style base URL to OpenAI-compatible format.

    Some providers (MiniMax, MiniMax-CN) expose an ``/anthropic`` endpoint for
    the Anthropic Messages API and a separate ``/v1`` endpoint for OpenAI chat
    completions.  The auxiliary client uses the OpenAI SDK, so it must hit the
    ``/v1`` surface.  Passing the raw ``inference_base_url`` causes requests to
    land on ``/anthropic/chat/completions`` — a 404.
    """
    url = str(base_url or '').strip().rstrip('/')
    if url.endswith('/anthropic'):
        if 'open.bigmodel.cn' in url or 'bigmodel' in url:
            rewritten = url[:-len('/anthropic')] + '/paas/v4'
            logger.debug('Auxiliary client: rewrote ZAI base URL %s → %s', url, rewritten)
            return rewritten
        rewritten = url[:-len('/anthropic')] + '/v1'
        logger.debug('Auxiliary client: rewrote base URL %s → %s', url, rewritten)
        return rewritten
    if 'api.kimi.com' in url and url.endswith('/coding'):
        rewritten = url + '/v1'
        logger.debug('Auxiliary client: rewrote Kimi base URL %s → %s', url, rewritten)
        return rewritten
    return url

def _select_pool_entry(provider: str) -> Tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug('Auxiliary client: could not load pool for %s: %s', provider, exc)
        return (False, None)
    if not pool or not pool.has_credentials():
        return (False, None)
    try:
        return (True, pool.select())
    except Exception as exc:
        logger.debug('Auxiliary client: could not select pool entry for %s: %s', provider, exc)
        return (True, None)

def _peek_pool_entry(provider: str) -> Optional[Any]:
    """Best-effort current/next pool entry without mutating selection order."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug('Auxiliary client: could not load pool for %s (peek): %s', provider, exc)
        return None
    if not pool or not pool.has_credentials():
        return None
    try:
        current_fn = getattr(pool, 'current', None)
        if callable(current_fn):
            current = current_fn()
            if current is not None:
                return current
        peek_fn = getattr(pool, 'peek', None)
        if callable(peek_fn):
            return peek_fn()
    except Exception as exc:
        logger.debug('Auxiliary client: could not peek pool entry for %s: %s', provider, exc)
    return None

def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ''
    key = getattr(entry, 'runtime_api_key', None) or getattr(entry, 'access_token', '')
    return str(key or '').strip()

def _pool_runtime_base_url(entry: Any, fallback: str='') -> str:
    if entry is None:
        return str(fallback or '').strip().rstrip('/')
    url = getattr(entry, 'runtime_base_url', None) or getattr(entry, 'inference_base_url', None) or getattr(entry, 'base_url', None) or fallback
    return str(url or '').strip().rstrip('/')

def _convert_content_for_responses(content: Any) -> Any:
    """Convert chat.completions content to Responses API format.

    chat.completions uses:
      {"type": "text", "text": "..."}
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

    Responses API uses:
      {"type": "input_text", "text": "..."}
      {"type": "input_image", "image_url": "data:image/png;base64,..."}

    If content is a plain string, it's returned as-is (the Responses API
    accepts strings directly for text-only messages).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ''
    converted: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get('type', '')
        if ptype == 'text':
            converted.append({'type': 'input_text', 'text': part.get('text', '')})
        elif ptype == 'image_url':
            image_data = part.get('image_url', {})
            url = image_data.get('url', '') if isinstance(image_data, dict) else str(image_data)
            entry: Dict[str, Any] = {'type': 'input_image', 'image_url': url}
            detail = image_data.get('detail') if isinstance(image_data, dict) else None
            if detail:
                entry['detail'] = detail
            converted.append(entry)
        elif ptype in {'input_text', 'input_image'}:
            converted.append(part)
        else:
            text = part.get('text', '')
            if text:
                converted.append({'type': 'input_text', 'text': text})
    return converted or ''

class _CodexCompletionsAdapter:
    """Drop-in shim that accepts chat.completions.create() kwargs and
    routes them through the Codex Responses streaming API."""

    def __init__(self, real_client: OpenAI, model: str):
        self._client = real_client
        self._model = model

    def create(self, **kwargs) -> Any:
        messages = kwargs.get('messages', [])
        model = kwargs.get('model', self._model)
        instructions = 'You are a helpful assistant.'
        input_msgs: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content') or ''
            if role == 'system':
                instructions = content if isinstance(content, str) else str(content)
            else:
                input_msgs.append({'role': role, 'content': _convert_content_for_responses(content)})
        resp_kwargs: Dict[str, Any] = {'model': model, 'instructions': instructions, 'input': input_msgs or [{'role': 'user', 'content': ''}], 'store': False}
        timeout = kwargs.get('timeout')
        if timeout is not None:
            resp_kwargs['timeout'] = timeout
        extra_body = kwargs.get('extra_body') or {}
        if isinstance(extra_body, dict):
            reasoning_cfg = extra_body.get('reasoning')
            if isinstance(reasoning_cfg, dict):
                if reasoning_cfg.get('enabled') is False:
                    pass
                else:
                    effort = reasoning_cfg.get('effort') or 'medium'
                    if effort == 'minimal':
                        effort = 'low'
                    resp_kwargs['reasoning'] = {'effort': effort, 'summary': 'auto'}
                    resp_kwargs['include'] = ['reasoning.encrypted_content']
        tools = kwargs.get('tools')
        tool_choice = kwargs.get('tool_choice')
        if tools:
            try:
                from kylin_memory._vendor.tools.schema_sanitizer import strip_pattern_and_format, strip_slash_enum
                tools, _ = strip_pattern_and_format(list(tools))
                tools, _ = strip_slash_enum(tools)
            except Exception as exc:
                logger.warning('Auxiliary client: failed to sanitize tool schemas for Codex/xAI Responses path: %s', exc)
            converted = []
            for t in tools:
                fn = t.get('function', {}) if isinstance(t, dict) else {}
                name = fn.get('name')
                if not name:
                    continue
                converted.append({'type': 'function', 'name': name, 'description': fn.get('description', ''), 'strict': bool(fn.get('strict', False)), 'parameters': fn.get('parameters', {})})
            if converted:
                resp_kwargs['tools'] = converted
                if isinstance(tool_choice, dict):
                    choice_type = str(tool_choice.get('type') or '').lower()
                    if choice_type == 'function':
                        name = (tool_choice.get('function') or {}).get('name')
                        if name:
                            resp_kwargs['tool_choice'] = {'type': 'function', 'name': name}
                elif str(tool_choice or '').lower() == 'required':
                    resp_kwargs['tool_choice'] = {'type': 'function', 'name': converted[0]['name']}
                elif str(tool_choice or '').lower() in {'auto', 'none'}:
                    resp_kwargs['tool_choice'] = str(tool_choice).lower()
        text_parts: List[str] = []
        tool_calls_raw: List[Any] = []
        usage = None
        total_timeout = timeout if isinstance(timeout, (int, float)) and timeout > 0 else None
        deadline = time.monotonic() + float(total_timeout) if total_timeout else None
        timed_out = threading.Event()
        timeout_timer: Optional[threading.Timer] = None

        def _timeout_message() -> str:
            return f'Codex auxiliary Responses stream exceeded {float(total_timeout):.1f}s total timeout'

        def _close_client_on_timeout() -> None:
            timed_out.set()
            close = getattr(self._client, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug('Codex auxiliary: client close during timeout failed', exc_info=True)
            try:
                _evict_cached_client_instance(self._client)
            except Exception:
                logger.debug('Codex auxiliary: cache eviction on timeout failed', exc_info=True)

        def _check_cancelled() -> None:
            if deadline is not None and time.monotonic() >= deadline:
                if not timed_out.is_set():
                    _close_client_on_timeout()
                raise TimeoutError(_timeout_message())
            try:
                from kylin_memory._vendor.tools.interrupt import is_interrupted
                if is_interrupted():
                    raise InterruptedError('Codex auxiliary Responses stream interrupted')
            except InterruptedError:
                raise
            except Exception:
                pass
        try:
            collected_output_items: List[Any] = []
            collected_text_deltas: List[str] = []
            has_function_calls = False
            if total_timeout:
                timeout_timer = threading.Timer(float(total_timeout), _close_client_on_timeout)
                timeout_timer.daemon = True
                timeout_timer.start()
            _check_cancelled()
            with self._client.responses.stream(**resp_kwargs) as stream:
                for _event in stream:
                    _check_cancelled()
                    _etype = getattr(_event, 'type', '')
                    if _etype == 'response.output_item.done':
                        _done = getattr(_event, 'item', None)
                        if _done is not None:
                            collected_output_items.append(_done)
                    elif 'output_text.delta' in _etype:
                        _delta = getattr(_event, 'delta', '')
                        if _delta:
                            collected_text_deltas.append(_delta)
                    elif 'function_call' in _etype:
                        has_function_calls = True
                _check_cancelled()
                final = stream.get_final_response()
            _output = getattr(final, 'output', None)
            if isinstance(_output, list) and (not _output):
                if collected_output_items:
                    final.output = list(collected_output_items)
                    logger.debug('Codex auxiliary: backfilled %d output items from stream events', len(collected_output_items))
                elif collected_text_deltas and (not has_function_calls):
                    assembled = ''.join(collected_text_deltas)
                    final.output = [SimpleNamespace(type='message', role='assistant', status='completed', content=[SimpleNamespace(type='output_text', text=assembled)])]
                    logger.debug('Codex auxiliary: synthesized from %d deltas (%d chars)', len(collected_text_deltas), len(assembled))

            def _item_get(obj: Any, key: str, default: Any=None) -> Any:
                val = getattr(obj, key, None)
                if val is None and isinstance(obj, dict):
                    val = obj.get(key, default)
                return val if val is not None else default
            for item in getattr(final, 'output', []):
                item_type = _item_get(item, 'type')
                if item_type == 'message':
                    for part in _item_get(item, 'content') or []:
                        ptype = _item_get(part, 'type')
                        if ptype in {'output_text', 'text'}:
                            text_parts.append(_item_get(part, 'text', ''))
                elif item_type == 'function_call':
                    tool_calls_raw.append(SimpleNamespace(id=_item_get(item, 'call_id', ''), type='function', function=SimpleNamespace(name=_item_get(item, 'name', ''), arguments=_item_get(item, 'arguments', '{}'))))
            resp_usage = getattr(final, 'usage', None)
            if resp_usage:
                usage = SimpleNamespace(prompt_tokens=getattr(resp_usage, 'input_tokens', 0), completion_tokens=getattr(resp_usage, 'output_tokens', 0), total_tokens=getattr(resp_usage, 'total_tokens', 0))
        except Exception as exc:
            if timed_out.is_set():
                raise TimeoutError(_timeout_message()) from exc
            logger.debug('Codex auxiliary Responses API call failed: %s', exc)
            raise
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()
        content = ''.join(text_parts).strip() or None
        message = SimpleNamespace(role='assistant', content=content, tool_calls=tool_calls_raw or None)
        choice = SimpleNamespace(index=0, message=message, finish_reason='stop' if not tool_calls_raw else 'tool_calls')
        return SimpleNamespace(choices=[choice], model=model, usage=usage)

class _CodexChatShim:
    """Wraps the adapter to provide client.chat.completions.create()."""

    def __init__(self, adapter: _CodexCompletionsAdapter):
        self.completions = adapter

class CodexAuxiliaryClient:
    """OpenAI-client-compatible wrapper that routes through Codex Responses API.

    Consumers can call client.chat.completions.create(**kwargs) as normal.
    Also exposes .api_key and .base_url for introspection by async wrappers.
    """

    def __init__(self, real_client: OpenAI, model: str):
        self._real_client = real_client
        adapter = _CodexCompletionsAdapter(real_client, model)
        self.chat = _CodexChatShim(adapter)
        self.api_key = real_client.api_key
        self.base_url = real_client.base_url

    def close(self):
        self._real_client.close()

class _AsyncCodexCompletionsAdapter:
    """Async version of the Codex Responses adapter.

    Wraps the sync adapter via asyncio.to_thread() so async consumers
    (web tools and other auxiliary consumers) can await it as normal.
    """

    def __init__(self, sync_adapter: _CodexCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)

class _AsyncCodexChatShim:

    def __init__(self, adapter: _AsyncCodexCompletionsAdapter):
        self.completions = adapter

class AsyncCodexAuxiliaryClient:
    """Async-compatible wrapper matching AsyncOpenAI.chat.completions.create()."""

    def __init__(self, sync_wrapper: 'CodexAuxiliaryClient'):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncCodexCompletionsAdapter(sync_adapter)
        self.chat = _AsyncCodexChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        self._real_client = sync_wrapper._real_client

class _AnthropicCompletionsAdapter:
    """OpenAI-client-compatible adapter for Anthropic Messages API."""

    def __init__(self, real_client: Any, model: str, is_oauth: bool=False):
        self._client = real_client
        self._model = model
        self._is_oauth = is_oauth

    def create(self, **kwargs) -> Any:
        from kylin_memory._vendor.agent.anthropic_adapter import build_anthropic_kwargs
        from kylin_memory._vendor.agent.transports import get_transport
        messages = kwargs.get('messages', [])
        model = kwargs.get('model', self._model)
        tools = kwargs.get('tools')
        tool_choice = kwargs.get('tool_choice')
        _skip_mt = kwargs.pop('_skip_zai_max_tokens', False)
        if _skip_mt:
            max_tokens = None
        else:
            max_tokens = kwargs.get('max_tokens') or kwargs.get('max_completion_tokens') or 2000
        temperature = kwargs.get('temperature')
        normalized_tool_choice = None
        if isinstance(tool_choice, str):
            normalized_tool_choice = tool_choice
        elif isinstance(tool_choice, dict):
            choice_type = str(tool_choice.get('type', '')).lower()
            if choice_type == 'function':
                normalized_tool_choice = tool_choice.get('function', {}).get('name')
            elif choice_type in {'auto', 'required', 'none'}:
                normalized_tool_choice = choice_type
        anthropic_kwargs = build_anthropic_kwargs(model=model, messages=messages, tools=tools, max_tokens=max_tokens, reasoning_config=None, tool_choice=normalized_tool_choice, is_oauth=self._is_oauth)
        if temperature is not None:
            from kylin_memory._vendor.agent.anthropic_adapter import _forbids_sampling_params
            if not _forbids_sampling_params(model):
                anthropic_kwargs['temperature'] = temperature
        response = self._client.messages.create(**anthropic_kwargs)
        _transport = get_transport('anthropic_messages')
        _nr = _transport.normalize_response(response, strip_tool_prefix=self._is_oauth)
        assistant_message = SimpleNamespace(content=_nr.content, tool_calls=_nr.tool_calls, reasoning=_nr.reasoning)
        finish_reason = _nr.finish_reason
        usage = None
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = getattr(response.usage, 'input_tokens', 0) or 0
            completion_tokens = getattr(response.usage, 'output_tokens', 0) or 0
            total_tokens = getattr(response.usage, 'total_tokens', 0) or prompt_tokens + completion_tokens
            usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
        choice = SimpleNamespace(index=0, message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], model=model, usage=usage)

class _AnthropicChatShim:

    def __init__(self, adapter: _AnthropicCompletionsAdapter):
        self.completions = adapter

class AnthropicAuxiliaryClient:
    """OpenAI-client-compatible wrapper over a native Anthropic client."""

    def __init__(self, real_client: Any, model: str, api_key: str, base_url: str, is_oauth: bool=False):
        self._real_client = real_client
        adapter = _AnthropicCompletionsAdapter(real_client, model, is_oauth=is_oauth)
        self.chat = _AnthropicChatShim(adapter)
        self.api_key = api_key
        self.base_url = base_url

    def close(self):
        close_fn = getattr(self._real_client, 'close', None)
        if callable(close_fn):
            close_fn()

class _AsyncAnthropicCompletionsAdapter:

    def __init__(self, sync_adapter: _AnthropicCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)

class _AsyncAnthropicChatShim:

    def __init__(self, adapter: _AsyncAnthropicCompletionsAdapter):
        self.completions = adapter

class AsyncAnthropicAuxiliaryClient:

    def __init__(self, sync_wrapper: 'AnthropicAuxiliaryClient'):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncAnthropicCompletionsAdapter(sync_adapter)
        self.chat = _AsyncAnthropicChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        self._real_client = sync_wrapper._real_client

def _endpoint_speaks_anthropic_messages(base_url: str) -> bool:
    """True if the endpoint at ``base_url`` speaks the Anthropic Messages
    protocol instead of OpenAI chat.completions.

    Mirrors ``kylin_agent_runtime_cli.runtime_provider._detect_api_mode_for_url`` so the
    auxiliary client and the main agent stay in sync on transport selection.
    Covers:

    - Any URL ending in ``/anthropic`` (MiniMax, Zhipu GLM, LiteLLM proxies,
      Anthropic-compatible gateways).
    - ``api.kimi.com/coding`` (Kimi Coding Plan — the /coding route only
      speaks Claude-Code's native Anthropic shape; ``chat.completions``
      returns 404 on Anthropic-only model aliases like ``kimi-for-coding``).
    - ``api.anthropic.com`` (native Anthropic).
    """
    normalized = (base_url or '').strip().lower().rstrip('/')
    if not normalized:
        return False
    if normalized.endswith('/anthropic'):
        return True
    hostname = base_url_hostname(normalized)
    if hostname == 'api.anthropic.com':
        return True
    if hostname == 'api.kimi.com' and '/coding' in normalized:
        return True
    return False

def _maybe_wrap_anthropic(client_obj: Any, model: str, api_key: str, base_url: str, api_mode: Optional[str]=None) -> Any:
    """Rewrap a plain OpenAI client in ``AnthropicAuxiliaryClient`` when
    the endpoint actually speaks Anthropic Messages.

    This is the single chokepoint for aux-client transport correction.
    Runs at the end of every ``resolve_provider_client`` branch so that
    api_key providers (Kimi Coding Plan), the ``custom`` endpoint, and
    future /anthropic gateways all land on the right wire format
    regardless of which branch built the client.

    Returns ``client_obj`` unchanged when:

    - It's already an Anthropic/Codex/Gemini/CopilotACP wrapper.
    - The endpoint is an OpenAI-wire endpoint.
    - ``api_mode`` is explicitly set to a non-Anthropic transport.
    - The ``anthropic`` SDK is not installed (falls back to OpenAI wire).
    """
    if _safe_isinstance(client_obj, AnthropicAuxiliaryClient):
        return client_obj
    if _safe_isinstance(client_obj, CodexAuxiliaryClient):
        return client_obj
    try:
        from kylin_memory._vendor.agent.gemini_native_adapter import GeminiNativeClient
        if _safe_isinstance(client_obj, GeminiNativeClient):
            return client_obj
    except ImportError:
        pass
    try:
        from kylin_memory._vendor.agent.copilot_acp_client import CopilotACPClient
        if _safe_isinstance(client_obj, CopilotACPClient):
            return client_obj
    except ImportError:
        pass
    if api_mode and api_mode != 'anthropic_messages':
        return client_obj
    should_wrap = api_mode == 'anthropic_messages' or _endpoint_speaks_anthropic_messages(base_url)
    if not should_wrap:
        return client_obj
    try:
        from kylin_memory._vendor.agent.anthropic_adapter import build_anthropic_client
    except ImportError:
        logger.warning('Endpoint %s speaks Anthropic Messages but the anthropic SDK is not installed — falling back to OpenAI-wire (will likely 404).', base_url)
        return client_obj
    try:
        real_client = build_anthropic_client(api_key, base_url)
    except Exception as exc:
        logger.warning('Failed to build Anthropic client for %s (%s) — falling back to OpenAI-wire client.', base_url, exc)
        return client_obj
    logger.debug('Auxiliary transport: wrapping client in AnthropicAuxiliaryClient (model=%s, base_url=%s, api_mode=%s)', model, base_url[:60] if base_url else '', api_mode or 'auto-detected')
    return AnthropicAuxiliaryClient(real_client, model, api_key, base_url, is_oauth=False)

def _read_nous_auth() -> Optional[dict]:
    """Read and validate ~/.kylin-agent-runtime/auth.json for an active Nous provider.

    Returns the provider state dict if Nous is active with tokens,
    otherwise None.
    """
    pool_present, entry = _select_pool_entry('nous')
    if pool_present:
        if entry is None:
            return None
        return {'access_token': getattr(entry, 'access_token', ''), 'refresh_token': getattr(entry, 'refresh_token', None), 'agent_key': getattr(entry, 'agent_key', None), 'inference_base_url': _pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL), 'portal_base_url': getattr(entry, 'portal_base_url', None), 'client_id': getattr(entry, 'client_id', None), 'scope': getattr(entry, 'scope', None), 'token_type': getattr(entry, 'token_type', 'Bearer'), 'source': 'pool'}
    try:
        if not _AUTH_JSON_PATH.is_file():
            return None
        data = json.loads(_AUTH_JSON_PATH.read_text())
        if data.get('active_provider') != 'nous':
            return None
        provider = data.get('providers', {}).get('nous', {})
        if not provider.get('agent_key') and (not provider.get('access_token')):
            return None
        return provider
    except Exception as exc:
        logger.debug('Could not read Nous auth: %s', exc)
        return None

def _nous_api_key(provider: dict) -> str:
    """Extract the Nous runtime credential from the compatibility field."""
    return provider.get('agent_key') or provider.get('access_token', '')

def _nous_base_url() -> str:
    """Resolve the Nous inference base URL from env or default."""
    return os.getenv('NOUS_INFERENCE_BASE_URL', _NOUS_DEFAULT_BASE_URL)

def _resolve_nous_runtime_api(*, force_refresh: bool=False) -> Optional[tuple[str, str]]:
    """Return fresh Nous runtime credentials when available.

    This mirrors the main agent's 401 recovery path and keeps auxiliary
    clients aligned with the singleton auth store + JWT/mint flow instead of
    relying only on whatever raw tokens happen to be sitting in auth.json
    or the credential pool.
    """
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.auth import NOUS_INFERENCE_AUTH_MODE_AUTO, NOUS_INFERENCE_AUTH_MODE_LEGACY, resolve_nous_runtime_credentials
        creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=max(60, int(os.getenv('HERMES_NOUS_MIN_KEY_TTL_SECONDS', '1800'))), timeout_seconds=float(os.getenv('HERMES_NOUS_TIMEOUT_SECONDS', '15')), inference_auth_mode=NOUS_INFERENCE_AUTH_MODE_LEGACY if force_refresh else NOUS_INFERENCE_AUTH_MODE_AUTO)
    except Exception as exc:
        logger.debug('Auxiliary Nous runtime credential resolution failed: %s', exc)
        return None
    api_key = str(creds.get('api_key') or '').strip()
    base_url = str(creds.get('base_url') or '').strip().rstrip('/')
    if not api_key or not base_url:
        return None
    return (api_key, base_url)

def _resolve_xai_oauth_for_aux() -> Optional[Tuple[str, str]]:
    """Resolve a fresh xAI OAuth (api_key, base_url) for auxiliary clients.

    Prefer the credential pool, matching the main runtime/provider status
    path.  Some xAI OAuth logins live only as pool entries; falling straight
    to the singleton auth-store resolver would make auxiliary tasks such as
    compression report "no provider configured" even though ``kylin-agent-runtime auth
    status`` shows xAI OAuth as logged in.

    Falls back to ``kylin_agent_runtime_cli.auth``'s singleton runtime resolver for older
    auth-store-only logins. Returns ``None`` if the user is not authenticated
    with xAI Grok OAuth.
    """
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL, _xai_validate_inference_base_url
        pool = load_pool('xai-oauth')
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = str(getattr(entry, 'runtime_api_key', None) or getattr(entry, 'access_token', '') or '').strip()
                base_url = _xai_validate_inference_base_url(os.getenv('HERMES_XAI_BASE_URL', '').strip().rstrip('/') or os.getenv('XAI_BASE_URL', '').strip().rstrip('/') or str(getattr(entry, 'runtime_base_url', None) or '').strip().rstrip('/') or str(getattr(entry, 'base_url', None) or '').strip().rstrip('/'), fallback=DEFAULT_XAI_OAUTH_BASE_URL)
                if api_key and base_url:
                    return (api_key, base_url)
    except Exception as exc:
        logger.debug('Auxiliary xAI OAuth pool credential resolution failed: %s', exc)
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.auth import resolve_xai_oauth_runtime_credentials
        creds = resolve_xai_oauth_runtime_credentials()
    except Exception as exc:
        logger.debug('Auxiliary xAI OAuth runtime credential resolution failed: %s', exc)
        return None
    api_key = str(creds.get('api_key') or '').strip()
    base_url = str(creds.get('base_url') or '').strip().rstrip('/')
    if not api_key or not base_url:
        return None
    return (api_key, base_url)

def _read_codex_access_token() -> Optional[str]:
    """Read a valid, non-expired Codex OAuth access token from Hermes auth store.

    If a credential pool exists but currently has no selectable runtime entry
    (for example all pool slots are marked exhausted), fall back to the
    profile's auth.json token instead of hard-failing. This keeps explicit
    fallback-to-Codex working when the pool state is stale but the stored OAuth
    token is still valid.
    """
    pool_present, entry = _select_pool_entry('openai-codex')
    if pool_present:
        token = _pool_runtime_api_key(entry)
        if token:
            return token
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.auth import _read_codex_tokens
        data = _read_codex_tokens()
        tokens = data.get('tokens', {})
        access_token = tokens.get('access_token')
        if not isinstance(access_token, str) or not access_token.strip():
            return None
        try:
            import base64
            payload = access_token.split('.')[1]
            payload += '=' * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            exp = claims.get('exp', 0)
            if exp and time.time() > exp:
                logger.debug('Codex access token expired (exp=%s), skipping', exp)
                return None
        except Exception:
            pass
        return access_token.strip()
    except Exception as exc:
        logger.debug('Could not read Codex auth for auxiliary client: %s', exc)
        return None

def _resolve_api_key_provider() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Try each API-key provider in PROVIDER_REGISTRY order.

    Returns (client, model) for the first provider with usable runtime
    credentials, or (None, None) if none are configured.
    """
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    except ImportError:
        logger.debug('Could not import PROVIDER_REGISTRY for API-key fallback')
        return (None, None)
    for provider_id, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != 'api_key':
            continue
        if provider_id == 'anthropic':
            try:
                from kylin_memory._vendor.kylin_agent_runtime_cli.auth import is_provider_explicitly_configured
                if not is_provider_explicitly_configured('anthropic'):
                    continue
            except ImportError:
                pass
            return _try_anthropic()
        pool_present, entry = _select_pool_entry(provider_id)
        if pool_present:
            api_key = _pool_runtime_api_key(entry)
            if not api_key:
                continue
            raw_base_url = _pool_runtime_base_url(entry, pconfig.inference_base_url) or pconfig.inference_base_url
            base_url = _to_openai_base_url(raw_base_url)
            model = _get_aux_model_for_provider(provider_id) or None
            if model is None:
                continue
            logger.debug('Auxiliary text client: %s (%s) via pool', pconfig.name, model)
            if provider_id == 'gemini':
                from kylin_memory._vendor.agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url
                if is_native_gemini_base_url(base_url):
                    return (GeminiNativeClient(api_key=api_key, base_url=base_url), model)
            extra = {}
            if base_url_host_matches(base_url, 'api.kimi.com'):
                extra['default_headers'] = {'User-Agent': 'claude-code/0.1.0'}
            elif base_url_host_matches(base_url, 'api.githubcopilot.com'):
                from kylin_memory._vendor.kylin_agent_runtime_cli.models import copilot_default_headers
                extra['default_headers'] = copilot_default_headers()
            elif base_url_host_matches(base_url, 'integrate.api.nvidia.com'):
                extra['default_headers'] = build_nvidia_nim_headers(base_url)
            else:
                try:
                    from kylin_memory._vendor.providers import get_provider_profile as _gpf_aux
                    _ph_aux = _gpf_aux(provider_id)
                    if _ph_aux and _ph_aux.default_headers:
                        extra['default_headers'] = dict(_ph_aux.default_headers)
                except Exception:
                    pass
            _client = OpenAI(api_key=api_key, base_url=base_url, **extra)
            _client = _maybe_wrap_anthropic(_client, model, api_key, raw_base_url)
            return (_client, model)
        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = str(creds.get('api_key', '')).strip()
        if not api_key:
            continue
        raw_base_url = str(creds.get('base_url', '')).strip().rstrip('/') or pconfig.inference_base_url
        base_url = _to_openai_base_url(raw_base_url)
        model = _get_aux_model_for_provider(provider_id) or None
        if model is None:
            continue
        logger.debug('Auxiliary text client: %s (%s)', pconfig.name, model)
        if provider_id == 'gemini':
            from kylin_memory._vendor.agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url
            if is_native_gemini_base_url(base_url):
                return (GeminiNativeClient(api_key=api_key, base_url=base_url), model)
        extra = {}
        if base_url_host_matches(base_url, 'api.kimi.com'):
            extra['default_headers'] = {'User-Agent': 'claude-code/0.1.0'}
        elif base_url_host_matches(base_url, 'api.githubcopilot.com'):
            from kylin_memory._vendor.kylin_agent_runtime_cli.models import copilot_default_headers
            extra['default_headers'] = copilot_default_headers()
        elif base_url_host_matches(base_url, 'integrate.api.nvidia.com'):
            extra['default_headers'] = build_nvidia_nim_headers(base_url)
        else:
            try:
                from kylin_memory._vendor.providers import get_provider_profile as _gpf_aux2
                _ph_aux2 = _gpf_aux2(provider_id)
                if _ph_aux2 and _ph_aux2.default_headers:
                    extra['default_headers'] = dict(_ph_aux2.default_headers)
            except Exception:
                pass
        _client = OpenAI(api_key=api_key, base_url=base_url, **extra)
        _client = _maybe_wrap_anthropic(_client, model, api_key, raw_base_url)
        return (_client, model)
    return (None, None)

def _try_openrouter(explicit_api_key: str=None, model: str=None) -> Tuple[Optional[OpenAI], Optional[str]]:
    pool_present, entry = _select_pool_entry('openrouter')
    if pool_present:
        or_key = explicit_api_key or _pool_runtime_api_key(entry)
        if not or_key:
            _mark_provider_unhealthy('openrouter', ttl=60)
            return (None, None)
        base_url = _pool_runtime_base_url(entry, OPENROUTER_BASE_URL) or OPENROUTER_BASE_URL
        logger.debug('Auxiliary client: OpenRouter via pool')
        return (OpenAI(api_key=or_key, base_url=base_url, default_headers=build_or_headers()), model or _OPENROUTER_MODEL)
    or_key = explicit_api_key or os.getenv('OPENROUTER_API_KEY')
    if not or_key:
        _mark_provider_unhealthy('openrouter', ttl=60)
        return (None, None)
    logger.debug('Auxiliary client: OpenRouter')
    return (OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL, default_headers=build_or_headers()), model or _OPENROUTER_MODEL)

def _describe_openrouter_unavailable() -> str:
    """Return a more precise OpenRouter auth failure reason for logs."""
    pool_present, entry = _select_pool_entry('openrouter')
    if pool_present:
        if entry is None:
            return 'OpenRouter credential pool has no usable entries (credentials may be exhausted)'
        if not _pool_runtime_api_key(entry):
            return 'OpenRouter credential pool entry is missing a runtime API key'
    if not str(os.getenv('OPENROUTER_API_KEY') or '').strip():
        return 'OPENROUTER_API_KEY not set'
    return 'no usable OpenRouter credentials found'

def _try_nous(vision: bool=False) -> Tuple[Optional[OpenAI], Optional[str]]:
    try:
        from kylin_memory._vendor.agent.nous_rate_guard import nous_rate_limit_remaining
        _remaining = nous_rate_limit_remaining()
        if _remaining is not None and _remaining > 0:
            logger.debug('Auxiliary: skipping Nous Portal (rate-limited, resets in %.0fs)', _remaining)
            _mark_provider_unhealthy('nous', ttl=_remaining)
            return (None, None)
    except Exception:
        pass
    nous = _read_nous_auth()
    runtime = _resolve_nous_runtime_api(force_refresh=False)
    if runtime is None and (not nous):
        logger.warning('Auxiliary Nous client unavailable: no Nous authentication found (run: kylin-agent-runtime auth).')
        _mark_provider_unhealthy('nous', ttl=60)
        return (None, None)
    if runtime is None and nous:
        logger.debug('Auxiliary Nous: runtime credential mint failed; falling back to stored auth.json token.')
    global auxiliary_is_nous
    auxiliary_is_nous = True
    logger.debug('Auxiliary client: Nous Portal')
    model = _NOUS_MODEL
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.models import get_nous_recommended_aux_model
        recommended = get_nous_recommended_aux_model(vision=vision)
        if recommended:
            model = recommended
            logger.debug('Auxiliary/%s: using Portal-recommended model %s', 'vision' if vision else 'text', model)
        else:
            logger.debug('Auxiliary/%s: no Portal recommendation, falling back to %s', 'vision' if vision else 'text', model)
    except Exception as exc:
        logger.debug('Auxiliary/%s: recommended-models lookup failed (%s); falling back to %s', 'vision' if vision else 'text', exc, model)
    if runtime is not None:
        api_key, base_url = runtime
    else:
        api_key = _nous_api_key(nous or {})
        base_url = str((nous or {}).get('inference_base_url') or _nous_base_url()).rstrip('/')
    return (OpenAI(api_key=api_key, base_url=base_url), model)

def _read_main_model() -> str:
    """Read the user's configured main model from config.yaml.

    config.yaml model.default is the single source of truth for the active
    model. Environment variables are no longer consulted.

    Runtime override: when an AIAgent is active with a CLI/gateway-provided
    model that differs from config.yaml, ``set_runtime_main()`` records the
    override in a process-local global. This is consulted FIRST so tools
    that gate on "the active main model" (e.g. ``vision_analyze``'s native
    fast path) see the live runtime, not the persisted config default.
    """
    override = _RUNTIME_MAIN_MODEL
    if isinstance(override, str) and override.strip():
        return override.strip()
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get('model', {})
        if isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
        if isinstance(model_cfg, dict):
            default = model_cfg.get('default', '')
            if isinstance(default, str) and default.strip():
                return default.strip()
    except Exception:
        pass
    return ''

def _read_main_provider() -> str:
    """Read the user's configured main provider from config.yaml.

    Returns the lowercase provider id (e.g. "alibaba", "openrouter") or ""
    if not configured.

    Runtime override: see ``_read_main_model`` — same mechanism for the
    provider half of the runtime tuple.
    """
    override = _RUNTIME_MAIN_PROVIDER
    if isinstance(override, str) and override.strip():
        return override.strip().lower()
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get('model', {})
        if isinstance(model_cfg, dict):
            provider = model_cfg.get('provider', '')
            if isinstance(provider, str) and provider.strip():
                return provider.strip().lower()
    except Exception:
        pass
    return ''
_RUNTIME_MAIN_PROVIDER: str = ''
_RUNTIME_MAIN_MODEL: str = ''

def set_runtime_main(provider: str, model: str) -> None:
    """Record the live runtime provider/model for the current AIAgent.

    Called by ``run_agent.AIAgent._sync_runtime_main_for_aux_routing`` (or
    equivalent setter) at the top of each turn so that
    ``_read_main_provider`` / ``_read_main_model`` reflect CLI/gateway
    overrides instead of the stale config.yaml default.
    """
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    _RUNTIME_MAIN_PROVIDER = (provider or '').strip().lower()
    _RUNTIME_MAIN_MODEL = (model or '').strip()

def clear_runtime_main() -> None:
    """Clear the runtime override (e.g. on session end)."""
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    _RUNTIME_MAIN_PROVIDER = ''
    _RUNTIME_MAIN_MODEL = ''

def _resolve_custom_runtime() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve the active custom/main endpoint the same way the main CLI does.

    This covers both env-driven OPENAI_BASE_URL setups and config-saved custom
    endpoints where the base URL lives in config.yaml instead of the live
    environment.
    """
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested='custom')
    except Exception as exc:
        logger.debug('Auxiliary client: custom runtime resolution failed: %s', exc)
        runtime = None
    if not isinstance(runtime, dict):
        openai_base = os.getenv('OPENAI_BASE_URL', '').strip().rstrip('/')
        openai_key = os.getenv('OPENAI_API_KEY', '').strip()
        if not openai_base:
            return (None, None, None)
        runtime = {'base_url': openai_base, 'api_key': openai_key}
    custom_base = runtime.get('base_url')
    custom_key = runtime.get('api_key')
    custom_mode = runtime.get('api_mode')
    if not isinstance(custom_base, str) or not custom_base.strip():
        return (None, None, None)
    custom_base = custom_base.strip().rstrip('/')
    if base_url_host_matches(custom_base, 'openrouter.ai'):
        return (None, None, None)
    if not isinstance(custom_key, str) or not custom_key.strip():
        custom_key = 'no-key-required'
    if not isinstance(custom_mode, str) or not custom_mode.strip():
        custom_mode = None
    return (custom_base, custom_key.strip(), custom_mode)

def _current_custom_base_url() -> str:
    custom_base, _, _ = _resolve_custom_runtime()
    return custom_base or ''

def _validate_proxy_env_urls() -> None:
    """Fail fast with a clear error when proxy env vars have malformed URLs.

    Common cause: shell config (e.g. .zshrc) with a typo like
    ``export HTTP_PROXY=http://127.0.0.1:6153export NEXT_VAR=...``
    which concatenates 'export' into the port number.  Without this
    check the OpenAI/httpx client raises a cryptic ``Invalid port``
    error that doesn't name the offending env var.
    """
    from urllib.parse import urlparse
    normalize_proxy_env_vars()
    for key in ('HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'https_proxy', 'http_proxy', 'all_proxy'):
        value = str(os.environ.get(key) or '').strip()
        if not value:
            continue
        try:
            parsed = urlparse(value)
            if parsed.scheme:
                _ = parsed.port
        except ValueError as exc:
            raise RuntimeError(f'Malformed proxy environment variable {key}={value!r}. Fix or unset your proxy settings and try again.') from exc

def _validate_base_url(base_url: str) -> None:
    """Reject obviously broken custom endpoint URLs before they reach httpx."""
    from urllib.parse import urlparse
    candidate = str(base_url or '').strip()
    if not candidate or candidate.startswith('acp://'):
        return
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in {'http', 'https'}:
            _ = parsed.port
    except ValueError as exc:
        raise RuntimeError(f'Malformed custom endpoint URL: {candidate!r}. Run `kylin-agent-runtime setup` or `kylin-agent-runtime model` and enter a valid http(s) base URL.') from exc

def _try_custom_endpoint() -> Tuple[Optional[Any], Optional[str]]:
    runtime = _resolve_custom_runtime()
    if len(runtime) == 2:
        custom_base, custom_key = runtime
        custom_mode = None
    else:
        custom_base, custom_key, custom_mode = runtime
    if not custom_base or not custom_key:
        return (None, None)
    if custom_base.lower().startswith(_CODEX_AUX_BASE_URL.lower()):
        return (None, None)
    model = _read_main_model() or 'gpt-4o-mini'
    logger.debug('Auxiliary client: custom endpoint (%s, api_mode=%s)', model, custom_mode or 'chat_completions')
    _clean_base, _dq = _extract_url_query_params(custom_base)
    _extra = {'default_query': _dq} if _dq else {}
    if custom_mode == 'codex_responses':
        real_client = OpenAI(api_key=custom_key, base_url=_clean_base, **_extra)
        return (CodexAuxiliaryClient(real_client, model), model)
    if custom_mode == 'anthropic_messages':
        try:
            from kylin_memory._vendor.agent.anthropic_adapter import build_anthropic_client
            real_client = build_anthropic_client(custom_key, custom_base)
        except ImportError:
            logger.warning('Custom endpoint declares api_mode=anthropic_messages but the anthropic SDK is not installed — falling back to OpenAI-wire.')
            return (OpenAI(api_key=custom_key, base_url=_clean_base, **_extra), model)
        return (AnthropicAuxiliaryClient(real_client, model, custom_key, custom_base, is_oauth=False), model)
    _fallback_client = OpenAI(api_key=custom_key, base_url=_clean_base, **_extra)
    _fallback_client = _maybe_wrap_anthropic(_fallback_client, model, custom_key, custom_base, custom_mode)
    return (_fallback_client, model)

def _build_xai_oauth_aux_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    """Build a CodexAuxiliaryClient for an xAI Grok OAuth-authenticated session.

    xAI's ``/v1/responses`` endpoint speaks the OpenAI Responses API, so we
    wrap a plain ``OpenAI`` client in ``CodexAuxiliaryClient`` to translate
    ``chat.completions.create()`` calls into ``responses.stream()`` requests.

    The caller must pass an explicit model — pinning a default for Grok
    would silently rot when xAI's allowlist drifts.  Returns ``(None, None)``
    when the user has not authenticated with xAI Grok OAuth.
    """
    if not model:
        logger.warning('Auxiliary client: xai-oauth requested without a model; pass model explicitly (auxiliary.<task>.model in config.yaml).')
        return (None, None)
    resolved = _resolve_xai_oauth_for_aux()
    if resolved is None:
        return (None, None)
    api_key, base_url = resolved
    logger.debug('Auxiliary client: xAI OAuth (%s via Responses API)', model)
    real_client = OpenAI(api_key=api_key, base_url=base_url)
    return (CodexAuxiliaryClient(real_client, model), model)

def _build_codex_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    """Build a CodexAuxiliaryClient for an explicitly-requested model.

    There is no auto-selection of the Codex model: the ChatGPT-account
    Codex endpoint's accepted model list is an undocumented, drifting
    allow-list, so any hardcoded default we pick goes stale.  The caller
    is responsible for passing the model (e.g. from the user's own
    ``model.model`` or ``auxiliary.<task>.model`` config).

    Returns (None, None) when no Codex OAuth token is available.
    """
    if not model:
        logger.warning('Auxiliary client: openai-codex requested without a model; pass model explicitly (auxiliary.<task>.model in config.yaml).')
        return (None, None)
    pool_present, entry = _select_pool_entry('openai-codex')
    if pool_present:
        codex_token = _pool_runtime_api_key(entry)
        if codex_token:
            base_url = _pool_runtime_base_url(entry, _CODEX_AUX_BASE_URL) or _CODEX_AUX_BASE_URL
        else:
            codex_token = _read_codex_access_token()
            if not codex_token:
                return (None, None)
            base_url = _CODEX_AUX_BASE_URL
    else:
        codex_token = _read_codex_access_token()
        if not codex_token:
            return (None, None)
        base_url = _CODEX_AUX_BASE_URL
    logger.debug('Auxiliary client: Codex OAuth (%s via Responses API)', model)
    real_client = OpenAI(api_key=codex_token, base_url=base_url, default_headers=_codex_cloudflare_headers(codex_token))
    return (CodexAuxiliaryClient(real_client, model), model)

def _try_azure_foundry(*, model: Optional[str]=None, explicit_api_key: Optional[str]=None, explicit_base_url: Optional[str]=None, api_mode: Optional[str]=None) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve an Azure Foundry auxiliary client via the runtime resolver.

    Mirrors the ``_try_anthropic`` / ``_try_nous`` shape but delegates to
    :func:`kylin_agent_runtime_cli.runtime_provider._resolve_azure_foundry_runtime` —
    the same resolver the main agent uses — so:

    * ``auth_mode: api_key`` (default) gets the static
      ``AZURE_FOUNDRY_API_KEY`` string.
    * ``auth_mode: entra_id`` gets a callable bearer-token provider
      (``Callable[[], str]`` from
      :mod:`agent.azure_identity_adapter`).
    * Per-model ``api_mode`` auto-routing for GPT-5.x / o-series /
      codex models works.
    * ``model.entra.{tenant_id,client_id,authority,scope}`` config
      fields propagate.
    * Non-default ``model.base_url`` overrides are honored.

    The OpenAI SDK accepts both shapes for ``api_key`` so the caller
    can forward the result without coercion.

    Returns ``(client, model)`` or ``(None, None)`` on failure.
    """
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.runtime_provider import _resolve_azure_foundry_runtime
        from kylin_memory._vendor.kylin_agent_runtime_cli.auth import AuthError
        from kylin_memory._vendor.kylin_agent_runtime_cli.config import load_config
    except ImportError:
        return (None, None)
    try:
        cfg = load_config()
        model_cfg = cfg.get('model') if isinstance(cfg, dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
    except Exception:
        model_cfg = {}
    try:
        runtime = _resolve_azure_foundry_runtime(requested_provider='azure-foundry', model_cfg=model_cfg, explicit_api_key=explicit_api_key, explicit_base_url=explicit_base_url, target_model=model)
    except AuthError as exc:
        logger.debug('Auxiliary azure-foundry: %s', exc)
        return (None, None)
    except Exception as exc:
        logger.debug('Auxiliary azure-foundry runtime error: %s', exc)
        return (None, None)
    api_key = runtime.get('api_key')
    base_url = str(runtime.get('base_url', '') or '')
    runtime_api_mode = api_mode or runtime.get('api_mode') or 'chat_completions'
    _has_key = bool(api_key) if not callable(api_key) else True
    if not _has_key or not base_url:
        return (None, None)
    final_model = _normalize_resolved_model(model or str(model_cfg.get('default') or ''), 'azure-foundry')
    if not final_model:
        logger.debug('Auxiliary azure-foundry: no model resolved (model=%r, default=%r)', model, model_cfg.get('default'))
        return (None, None)
    extra: Dict[str, Any] = {}
    _clean_base, _dq = _extract_url_query_params(base_url)
    if _dq:
        extra['default_query'] = _dq
    client = OpenAI(api_key=api_key, base_url=_clean_base, **extra)
    if runtime_api_mode == 'codex_responses':
        return (CodexAuxiliaryClient(client, final_model), final_model)
    if runtime_api_mode == 'anthropic_messages':
        return (_maybe_wrap_anthropic(client, final_model, api_key, base_url, runtime_api_mode), final_model)
    return (client, final_model)

def _try_anthropic(explicit_api_key: str=None) -> Tuple[Optional[Any], Optional[str]]:
    try:
        from kylin_memory._vendor.agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
    except ImportError:
        return (None, None)
    pool_present, entry = _select_pool_entry('anthropic')
    if pool_present:
        if entry is None:
            return (None, None)
        token = explicit_api_key or _pool_runtime_api_key(entry)
    else:
        entry = None
        token = explicit_api_key or resolve_anthropic_token()
    if not token:
        return (None, None)
    base_url = _pool_runtime_base_url(entry, _ANTHROPIC_DEFAULT_BASE_URL) if pool_present else _ANTHROPIC_DEFAULT_BASE_URL
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get('model')
        if isinstance(model_cfg, dict):
            cfg_provider = str(model_cfg.get('provider') or '').strip().lower()
            if cfg_provider == 'anthropic':
                cfg_base_url = (model_cfg.get('base_url') or '').strip().rstrip('/')
                if cfg_base_url:
                    base_url = cfg_base_url
    except Exception:
        pass
    from kylin_memory._vendor.agent.anthropic_adapter import _is_oauth_token
    is_oauth = _is_oauth_token(token)
    model = _get_aux_model_for_provider('anthropic') or 'claude-haiku-4-5-20251001'
    logger.debug('Auxiliary client: Anthropic native (%s) at %s (oauth=%s)', model, base_url, is_oauth)
    try:
        real_client = build_anthropic_client(token, base_url)
    except ImportError:
        return (None, None)
    return (AnthropicAuxiliaryClient(real_client, model, token, base_url, is_oauth=is_oauth), model)
_AUTO_PROVIDER_LABELS = {'_try_openrouter': 'openrouter', '_try_nous': 'nous', '_try_custom_endpoint': 'local/custom', '_resolve_api_key_provider': 'api-key'}
_MAIN_RUNTIME_FIELDS = ('provider', 'model', 'base_url', 'api_key', 'api_mode', 'auth_mode')

def _normalize_main_runtime(main_runtime: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a sanitized copy of a live main-runtime override.

    Most fields are stripped strings. ``api_key`` may legitimately be a
    zero-arg callable (Azure Foundry Entra ID token provider) — preserve
    those as-is so auxiliary clients inherit the same authentication
    surface as the main agent. The OpenAI SDK accepts ``Callable[[], str]``
    for ``api_key`` and calls it before every request.
    """
    if not isinstance(main_runtime, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for field in _MAIN_RUNTIME_FIELDS:
        value = main_runtime.get(field)
        if field == 'api_key' and callable(value) and (not isinstance(value, str)):
            normalized[field] = value
            continue
        if isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
    provider = normalized.get('provider')
    if isinstance(provider, str):
        normalized['provider'] = provider.lower()
    return normalized

def _get_provider_chain() -> List[tuple]:
    """Return the ordered provider detection chain.

    Built at call time (not module level) so that test patches
    on the ``_try_*`` functions are picked up correctly.

    NOTE: ``openai-codex`` is deliberately NOT in this chain.  The
    ChatGPT-account Codex endpoint only accepts a shifting, undocumented
    allow-list of model IDs, so falling back to it with a guessed model
    fails more often than not.  Codex is used only when the user's main
    provider *is* openai-codex (see Step 1 of ``_resolve_auto``) or when
    a caller explicitly requests it with a model.
    """
    return [('openrouter', _try_openrouter), ('nous', _try_nous), ('local/custom', _try_custom_endpoint), ('api-key', _resolve_api_key_provider)]
_AUX_UNHEALTHY_TTL_SECONDS = 600
_aux_unhealthy_until: Dict[str, float] = {}
_aux_unhealthy_logged_at: Dict[str, float] = {}
_AUX_UNHEALTHY_LABEL_ALIASES = {'openrouter': 'openrouter', 'nous': 'nous', 'custom': 'local/custom', 'local/custom': 'local/custom', 'openai-codex': 'openai-codex', 'codex': 'openai-codex'}

def _normalize_chain_label(provider: str) -> str:
    """Normalize a resolved_provider value to a chain label used by
    ``_get_provider_chain()``. Falls back to the lowercased input for
    direct API-key providers (deepseek, alibaba, minimax, etc.) which
    each report their own provider name from the api-key chain.
    """
    if not provider:
        return ''
    p = str(provider).strip().lower()
    return _AUX_UNHEALTHY_LABEL_ALIASES.get(p, p)

def _mark_provider_unhealthy(provider: str, ttl: Optional[float]=None) -> None:
    """Mark ``provider`` as recently-402'd, hidden from chain iteration
    until the TTL expires. Called from the payment-fallback branches in
    ``call_llm`` and ``acall_llm`` after a confirmed payment error.
    """
    label = _normalize_chain_label(provider)
    if not label:
        return
    expires_at = time.time() + (ttl if ttl is not None else _AUX_UNHEALTHY_TTL_SECONDS)
    _aux_unhealthy_until[label] = expires_at
    logger.warning('Auxiliary: marking %s unhealthy for %ds (payment / credit error). Subsequent auxiliary calls will skip it until %s.', label, int(ttl if ttl is not None else _AUX_UNHEALTHY_TTL_SECONDS), time.strftime('%H:%M:%S', time.localtime(expires_at)))

def _is_provider_unhealthy(label: str) -> bool:
    """True iff ``label`` is in the unhealthy cache and the TTL hasn't expired.
    Lazily evicts expired entries so the cache stays small.
    """
    if not label:
        return False
    expires_at = _aux_unhealthy_until.get(label)
    if expires_at is None:
        return False
    if time.time() >= expires_at:
        _aux_unhealthy_until.pop(label, None)
        _aux_unhealthy_logged_at.pop(label, None)
        return False
    return True

def _log_skip_unhealthy(label: str, task: Optional[str]=None) -> None:
    """Emit a single info-level log per minute when we skip an unhealthy
    provider. Avoids spamming the log on bursty sessions while still
    giving the user a trail.
    """
    now = time.time()
    last = _aux_unhealthy_logged_at.get(label, 0.0)
    if now - last >= 60:
        _aux_unhealthy_logged_at[label] = now
        expires_at = _aux_unhealthy_until.get(label, now)
        logger.info('Auxiliary %s: skipping %s (recently returned payment error, retry in %ds)', task or 'call', label, max(0, int(expires_at - now)))

def _reset_aux_unhealthy_cache() -> None:
    """Clear the unhealthy cache. Used by tests and by a future explicit
    user trigger (e.g. ``kylin-agent-runtime config aux reset``)."""
    _aux_unhealthy_until.clear()
    _aux_unhealthy_logged_at.clear()

def _is_payment_error(exc: Exception) -> bool:
    """Detect payment/credit/quota exhaustion errors.

    Returns True for HTTP 402 (Payment Required) and for 429/other errors
    whose message indicates billing exhaustion or daily quota exhaustion
    rather than transient rate limiting.

    Daily token quota errors (e.g. Bedrock "Too many tokens per day",
    Vertex AI "quota exceeded") are functionally equivalent to credit
    exhaustion — the provider cannot serve the request until the quota
    resets — and should trigger the same provider-fallback logic.
    """
    status = getattr(exc, 'status_code', None)
    if status == 402:
        return True
    err_lower = str(exc).lower()
    if status in {402, 429, None}:
        if any((kw in err_lower for kw in ('credits', 'insufficient funds', 'can only afford', 'billing', 'payment required', 'quota exceeded', 'quota_exceeded', 'too many tokens per day', 'daily limit', 'tokens per day', 'daily quota', 'resource exhausted'))):
            return True
    return False

def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect rate-limit errors that warrant provider fallback.

    Returns True for HTTP 429 errors whose message indicates rate limiting
    (as opposed to billing/quota exhaustion, which _is_payment_error handles).
    Also catches OpenAI SDK RateLimitError instances that may not set
    .status_code on the exception object.
    """
    status = getattr(exc, 'status_code', None)
    err_lower = str(exc).lower()
    if type(exc).__name__ == 'RateLimitError':
        return True
    if status == 429:
        if any((kw in err_lower for kw in ('rate limit', 'rate_limit', 'too many requests', 'try again', 'retry after', 'resets in'))):
            return True
        if not any((kw in err_lower for kw in ('credits', 'insufficient funds', 'billing', 'payment required', 'can only afford'))):
            return True
    return False

def _is_connection_error(exc: Exception) -> bool:
    """Detect connection/network errors that warrant provider fallback.

    Returns True for errors indicating the provider endpoint is unreachable
    (DNS failure, connection refused, TLS errors, timeouts).  These are
    distinct from API errors (4xx/5xx) which indicate the provider IS
    reachable but returned an error.
    """
    try:
        from openai import APIConnectionError, APITimeoutError
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    except ImportError:
        pass
    err_type = type(exc).__name__
    if any((kw in err_type for kw in ('Connection', 'Timeout', 'DNS', 'SSL'))):
        return True
    err_lower = str(exc).lower()
    if any((kw in err_lower for kw in ('connection refused', 'name or service not known', 'no route to host', 'network is unreachable', 'timed out', 'connection reset', 'incomplete chunked read', 'peer closed connection', 'response ended prematurely', 'unexpected eof', 'remoteprotocolerror', 'localprotocolerror'))):
        return True
    return False

def _is_auth_error(exc: Exception) -> bool:
    """Detect auth failures that should trigger provider-specific refresh."""
    status = getattr(exc, 'status_code', None)
    if status == 401:
        return True
    err_lower = str(exc).lower()
    return 'error code: 401' in err_lower or 'authenticationerror' in type(exc).__name__.lower()

def _is_unsupported_parameter_error(exc: Exception, param: str) -> bool:
    """Detect provider 400s for an unsupported request parameter.

    Different OpenAI-compatible endpoints phrase the same class of error a few
    ways: ``Unsupported parameter: X``, ``unsupported_parameter`` with a
    ``param`` field, ``X is not supported``, ``unknown parameter: X``,
    ``unrecognized request argument: X``.  We match on both the parameter
    name and a generic "unsupported/unknown/unrecognized parameter" marker so
    call sites can reactively retry without the offending key instead of
    surfacing a noisy auxiliary failure.

    Generalizes the temperature-specific detector that originally shipped
    with PR #15621 so the same retry strategy can cover ``max_tokens``,
    ``seed``, ``top_p``, and any future quirk. Credit @nicholasrae (PR #15416)
    for the generalization pattern.
    """
    param_lower = (param or '').lower()
    if not param_lower:
        return False
    err_lower = str(exc).lower()
    if param_lower not in err_lower:
        return False
    return any((marker in err_lower for marker in ('unsupported parameter', 'unsupported_parameter', 'not supported', 'does not support', 'unknown parameter', 'unrecognized request argument', 'unrecognized parameter', 'invalid parameter')))

def _is_thinking_tool_choice_error(exc: Exception) -> bool:
    """Detect providers rejecting named tools while thinking is enabled."""
    text = str(exc).lower()
    return 'thinking mode does not support this tool_choice' in text

def _thinking_is_disabled(extra_body: Any) -> bool:
    """Safely inspect a provider-specific thinking switch in request extras."""
    if not isinstance(extra_body, dict):
        return False
    thinking = extra_body.get('thinking')
    return isinstance(thinking, dict) and str(thinking.get('type') or '').lower() == 'disabled'

def _is_unsupported_temperature_error(exc: Exception) -> bool:
    """Back-compat wrapper: detect API errors where the model rejects ``temperature``.

    Delegates to :func:`_is_unsupported_parameter_error`; kept as a separate
    public symbol because existing tests and call sites import it by name.
    """
    return _is_unsupported_parameter_error(exc, 'temperature')

def _evict_cached_clients(provider: str) -> None:
    """Drop cached auxiliary clients for a provider so fresh creds are used."""
    normalized = _normalize_aux_provider(provider)
    with _client_cache_lock:
        stale_keys = [key for key in _client_cache if _normalize_aux_provider(str(key[0])) == normalized]
        for key in stale_keys:
            client = _client_cache.get(key, (None, None, None))[0]
            if client is not None:
                _force_close_async_httpx(client)
                try:
                    close_fn = getattr(client, 'close', None)
                    if callable(close_fn):
                        close_fn()
                except Exception:
                    pass
            _client_cache.pop(key, None)

def _evict_cached_client_instance(target: Any) -> bool:
    """Drop the cache entry whose stored client is *target*.

    Used when a specific cached client has been poisoned (closed httpx
    transport after a timeout, broken streaming session, etc.) so the next
    auxiliary call rebuilds rather than reusing the dead instance.

    Walks both sync and async wrappers (``CodexAuxiliaryClient``,
    ``AnthropicAuxiliaryClient``, ``AsyncCodexAuxiliaryClient``, etc.) via
    their ``_real_client`` attribute so a timeout that closes the underlying
    ``OpenAI`` (or native provider) client evicts every cached shim that
    exposed it. Async wrappers must mirror their sync sibling's
    ``_real_client`` for this to work — otherwise the sync entry is evicted
    but the async entry survives and keeps reusing the dead transport.

    Returns True when at least one entry was evicted.
    """
    if target is None:
        return False
    evicted = False
    with _client_cache_lock:
        for key in list(_client_cache.keys()):
            entry = _client_cache.get(key)
            if entry is None:
                continue
            cached = entry[0]
            if cached is None:
                continue
            real = getattr(cached, '_real_client', None)
            if cached is target or real is target:
                del _client_cache[key]
                evicted = True
    return evicted

def _pool_cache_hint(provider: str, *, main_runtime: Optional[Dict[str, Any]]=None) -> str:
    """Return a stable cache discriminator for pooled providers."""
    normalized = _normalize_aux_provider(provider)
    if normalized == 'auto':
        runtime = _normalize_main_runtime(main_runtime)
        normalized = _normalize_aux_provider(runtime.get('provider') or _read_main_provider())
    if normalized in {'', 'auto', 'custom'}:
        return ''
    entry = _peek_pool_entry(normalized)
    if entry is None:
        return ''
    entry_id = str(getattr(entry, 'id', '') or '').strip()
    if not entry_id:
        return ''
    return f'{normalized}:{entry_id}'

def _pool_error_context(exc: Exception) -> Dict[str, Any]:
    status = getattr(exc, 'status_code', None)
    payload: Dict[str, Any] = {'message': str(exc)}
    if status is not None:
        payload['status_code'] = status
    return payload

def _recoverable_pool_provider(resolved_provider: str, client: Any) -> Optional[str]:
    """Infer which provider pool can recover the current auxiliary client."""
    normalized = _normalize_aux_provider(resolved_provider)
    if normalized not in {'', 'auto', 'custom'}:
        return normalized
    base = str(getattr(client, 'base_url', '') or '')
    if base_url_host_matches(base, 'chatgpt.com'):
        return 'openai-codex'
    if base_url_host_matches(base, 'openrouter.ai'):
        return 'openrouter'
    if base_url_host_matches(base, 'inference-api.nousresearch.com'):
        return 'nous'
    if base_url_host_matches(base, 'api.anthropic.com'):
        return 'anthropic'
    if base_url_host_matches(base, 'api.githubcopilot.com'):
        return 'copilot'
    if base_url_host_matches(base, 'api.kimi.com'):
        return 'kimi-coding'
    return None

def _recover_provider_pool(provider: str, exc: Exception) -> bool:
    """Try same-provider credential-pool recovery for auxiliary calls."""
    normalized = _normalize_aux_provider(provider)
    try:
        pool = load_pool(normalized)
    except Exception as load_exc:
        logger.debug('Auxiliary client: could not load pool for %s recovery: %s', normalized, load_exc)
        return False
    if not pool or not pool.has_credentials():
        return False
    status_code = getattr(exc, 'status_code', None)
    error_context = _pool_error_context(exc)
    if _is_auth_error(exc):
        refreshed = pool.try_refresh_current()
        if refreshed is not None:
            _evict_cached_clients(normalized)
            return True
        next_entry = pool.mark_exhausted_and_rotate(status_code=status_code if status_code is not None else 401, error_context=error_context)
        if next_entry is not None:
            _evict_cached_clients(normalized)
            return True
        return False
    if _is_payment_error(exc) or _is_rate_limit_error(exc):
        fallback_status = 402 if _is_payment_error(exc) else 429
        next_entry = pool.mark_exhausted_and_rotate(status_code=status_code if status_code is not None else fallback_status, error_context=error_context)
        if next_entry is not None:
            _evict_cached_clients(normalized)
            return True
    return False

def _retry_same_provider_sync(*, task: Optional[str], resolved_provider: str, resolved_model: Optional[str], resolved_base_url: Optional[str], resolved_api_key: Optional[str], resolved_api_mode: Optional[str], main_runtime: Optional[Dict[str, Any]], final_model: Optional[str], messages: list, temperature: Optional[float], max_tokens: Optional[int], tools: Optional[list], tool_choice: Any, effective_timeout: float, effective_extra_body: dict) -> Any:
    if task == 'vision':
        _, retry_client, retry_model = resolve_vision_provider_client(provider=resolved_provider, model=final_model, base_url=resolved_base_url, api_key=resolved_api_key, async_mode=False)
    else:
        retry_client, retry_model = _get_cached_client(resolved_provider, resolved_model, base_url=resolved_base_url, api_key=resolved_api_key, api_mode=resolved_api_mode, main_runtime=main_runtime)
    if retry_client is None:
        raise RuntimeError(f"Auxiliary {task or 'call'}: provider {resolved_provider} could not be rebuilt after recovery")
    retry_base = str(getattr(retry_client, 'base_url', '') or '')
    retry_kwargs = _build_call_kwargs(resolved_provider, retry_model or final_model, messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, timeout=effective_timeout, extra_body=effective_extra_body, base_url=retry_base or resolved_base_url)
    if _is_anthropic_compat_endpoint(resolved_provider, retry_base):
        retry_kwargs['messages'] = _convert_openai_images_to_anthropic(retry_kwargs['messages'])
    return _validate_llm_response(retry_client.chat.completions.create(**retry_kwargs), task)

async def _retry_same_provider_async(*, task: Optional[str], resolved_provider: str, resolved_model: Optional[str], resolved_base_url: Optional[str], resolved_api_key: Optional[str], resolved_api_mode: Optional[str], final_model: Optional[str], messages: list, temperature: Optional[float], max_tokens: Optional[int], tools: Optional[list], tool_choice: Any, effective_timeout: float, effective_extra_body: dict) -> Any:
    if task == 'vision':
        _, retry_client, retry_model = resolve_vision_provider_client(provider=resolved_provider, model=final_model, base_url=resolved_base_url, api_key=resolved_api_key, async_mode=True)
    else:
        retry_client, retry_model = _get_cached_client(resolved_provider, resolved_model, async_mode=True, base_url=resolved_base_url, api_key=resolved_api_key, api_mode=resolved_api_mode)
    if retry_client is None:
        raise RuntimeError(f"Auxiliary {task or 'call'}: provider {resolved_provider} could not be rebuilt after recovery")
    retry_base = str(getattr(retry_client, 'base_url', '') or '')
    retry_kwargs = _build_call_kwargs(resolved_provider, retry_model or final_model, messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, timeout=effective_timeout, extra_body=effective_extra_body, base_url=retry_base or resolved_base_url)
    if _is_anthropic_compat_endpoint(resolved_provider, retry_base):
        retry_kwargs['messages'] = _convert_openai_images_to_anthropic(retry_kwargs['messages'])
    return _validate_llm_response(await retry_client.chat.completions.create(**retry_kwargs), task)

def _refresh_provider_credentials(provider: str) -> bool:
    """Refresh short-lived credentials for OAuth-backed auxiliary providers."""
    normalized = _normalize_aux_provider(provider)
    try:
        if normalized == 'openai-codex':
            from kylin_memory._vendor.kylin_agent_runtime_cli.auth import resolve_codex_runtime_credentials
            creds = resolve_codex_runtime_credentials(force_refresh=True)
            if not str(creds.get('api_key', '') or '').strip():
                return False
            _evict_cached_clients(normalized)
            return True
        if normalized == 'nous':
            from kylin_memory._vendor.kylin_agent_runtime_cli.auth import NOUS_INFERENCE_AUTH_MODE_LEGACY, resolve_nous_runtime_credentials
            creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=max(60, int(os.getenv('HERMES_NOUS_MIN_KEY_TTL_SECONDS', '1800'))), timeout_seconds=float(os.getenv('HERMES_NOUS_TIMEOUT_SECONDS', '15')), inference_auth_mode=NOUS_INFERENCE_AUTH_MODE_LEGACY)
            if not str(creds.get('api_key', '') or '').strip():
                return False
            _evict_cached_clients(normalized)
            return True
        if normalized == 'anthropic':
            from kylin_memory._vendor.agent.anthropic_adapter import read_claude_code_credentials, _refresh_oauth_token, resolve_anthropic_token
            creds = read_claude_code_credentials()
            token = _refresh_oauth_token(creds) if isinstance(creds, dict) and creds.get('refreshToken') else None
            if not str(token or '').strip():
                token = resolve_anthropic_token()
            if not str(token or '').strip():
                return False
            _evict_cached_clients(normalized)
            return True
    except Exception as exc:
        logger.debug('Auxiliary provider credential refresh failed for %s: %s', normalized, exc)
        return False
    return False

def _try_payment_fallback(failed_provider: str, task: str=None, reason: str='payment error') -> Tuple[Optional[Any], Optional[str], str]:
    """Try alternative providers after a payment/credit or connection error.

    Iterates the standard auto-detection chain, skipping the provider that
    failed.

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    skip = failed_provider.lower().strip()
    main_provider = _read_main_provider()
    skip_labels = {skip}
    if main_provider and main_provider.lower() in skip:
        skip_labels.add(main_provider.lower())
    _alias_to_label = {'openrouter': 'openrouter', 'nous': 'nous', 'openai-codex': 'openai-codex', 'codex': 'openai-codex', 'custom': 'local/custom', 'local/custom': 'local/custom'}
    skip_chain_labels = {_alias_to_label.get(s, s) for s in skip_labels}
    tried = []
    for label, try_fn in _get_provider_chain():
        if label in skip_chain_labels:
            continue
        if _is_provider_unhealthy(label):
            _log_skip_unhealthy(label, task)
            tried.append(f'{label} (unhealthy)')
            continue
        client, model = try_fn()
        if client is not None:
            logger.info('Auxiliary %s: %s on %s — falling back to %s (%s)', task or 'call', reason, failed_provider, label, model or 'default')
            return (client, model, label)
        tried.append(label)
    logger.warning('Auxiliary %s: %s on %s and no fallback available (tried: %s)', task or 'call', reason, failed_provider, ', '.join(tried))
    return (None, None, '')

def _try_main_agent_model_fallback(failed_provider: str, task: str=None, reason: str='error') -> Tuple[Optional[Any], Optional[str], str]:
    """Last-resort fallback to the user's main agent provider + model.

    Used after the configured fallback_chain is exhausted (or empty) for
    users with an explicit auxiliary provider.  This is the "safety net"
    layer: if nothing the user asked for can serve the request, try the
    main chat model before giving up.

    Skips when the failed provider already IS the main provider (no point
    retrying the same backend that just failed).

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    main_provider = (_read_main_provider() or '').strip()
    main_model = (_read_main_model() or '').strip()
    if not main_provider or not main_model or main_provider.lower() in {'auto', ''}:
        return (None, None, '')
    skip = (failed_provider or '').lower().strip()
    if main_provider.lower() == skip:
        return (None, None, '')
    if _is_provider_unhealthy(main_provider):
        _log_skip_unhealthy(main_provider, task)
        return (None, None, '')
    try:
        client, resolved_model = resolve_provider_client(provider=main_provider, model=main_model)
    except Exception:
        client, resolved_model = (None, None)
    if client is None:
        return (None, None, '')
    label = f'main-agent({main_provider})'
    logger.info('Auxiliary %s: %s on %s — falling back to main agent model %s (%s)', task or 'call', reason, failed_provider, label, resolved_model or main_model)
    return (client, resolved_model or main_model, label)

def _try_configured_fallback_chain(task: str, failed_provider: str, reason: str='error') -> Tuple[Optional[Any], Optional[str], str]:
    """Try user-configured fallback_chain for a specific auxiliary task.

    Reads auxiliary.<task>.fallback_chain from config.yaml and tries each
    entry in order.  Each entry must have at least ``provider``; ``model``,
    ``base_url``, and ``api_key`` are optional.

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    if not task:
        return (None, None, '')
    task_config = _get_auxiliary_task_config(task)
    chain = task_config.get('fallback_chain')
    if not chain or not isinstance(chain, list):
        return (None, None, '')
    skip = failed_provider.lower().strip()
    tried = []
    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        fb_provider = str(entry.get('provider', '')).strip()
        if not fb_provider or fb_provider.lower() == skip:
            continue
        fb_model = str(entry.get('model', '')).strip() or None
        fb_base_url = str(entry.get('base_url', '')).strip() or None
        fb_api_key = str(entry.get('api_key', '')).strip() or None
        label = f'fallback_chain[{i}]({fb_provider})'
        try:
            fb_client = _resolve_single_provider(fb_provider, fb_model, fb_base_url, fb_api_key)
        except Exception:
            fb_client = None
        if fb_client is not None:
            logger.info('Auxiliary %s: %s on %s — configured fallback to %s (%s)', task, reason, failed_provider, label, fb_model or 'default')
            return (fb_client, fb_model, label)
        tried.append(label)
    if tried:
        logger.debug('Auxiliary %s: configured fallback_chain exhausted (tried: %s)', task, ', '.join(tried))
    return (None, None, '')

def _resolve_single_provider(provider: str, model: Optional[str]=None, base_url: Optional[str]=None, api_key: Optional[str]=None) -> Optional[Any]:
    """Resolve a single provider entry from fallback_chain to an OpenAI client.

    Uses the existing provider resolution infrastructure where possible.
    """
    client, resolved_model = resolve_provider_client(provider=provider, model=model, base_url=base_url, api_key=api_key)
    return client

def _resolve_auto(main_runtime: Optional[Dict[str, Any]]=None) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Full auto-detection chain.

    Priority:
      1. User's main provider + main model, regardless of provider type.
         This means auxiliary tasks (compression, vision, web extraction,
         session search, etc.) use the same model the user configured for
         chat.  Users on OpenRouter/Nous get their chosen chat model; users
         on DeepSeek/ZAI/Alibaba get theirs; etc.  Running aux tasks on the
         user's picked model keeps behavior predictable — no surprise
         switches to a cheap fallback model for side tasks.
      2. OpenRouter → Nous → custom → Codex → API-key providers (fallback
         chain, only used when the main provider has no working client).
    """
    global auxiliary_is_nous, _stale_base_url_warned
    auxiliary_is_nous = False
    runtime = _normalize_main_runtime(main_runtime)
    runtime_provider = runtime.get('provider', '')
    runtime_model = str(runtime.get('model') or '')
    runtime_base_url = str(runtime.get('base_url') or '')
    runtime_api_key = runtime.get('api_key', '')
    runtime_api_mode = str(runtime.get('api_mode') or '')
    if not _stale_base_url_warned:
        _env_base = os.getenv('OPENAI_BASE_URL', '').strip()
        _cfg_provider = runtime_provider or _read_main_provider()
        if _env_base and _cfg_provider and (_cfg_provider != 'custom') and (not _cfg_provider.startswith('custom:')):
            logger.warning("OPENAI_BASE_URL is set (%s) but model.provider is '%s'. Auxiliary clients may route to the wrong endpoint. Run: kylin-agent-runtime model to reconfigure, or remove OPENAI_BASE_URL from ~/.kylin-agent-runtime/.env", _env_base, _cfg_provider)
            _stale_base_url_warned = True
    main_provider = str(runtime_provider or _read_main_provider() or '')
    main_model = str(runtime_model or _read_main_model() or '')
    if main_provider and main_model and (main_provider not in {'auto', ''}):
        resolved_provider = main_provider
        explicit_base_url = runtime_base_url or None
        explicit_api_key = runtime_api_key or None
        if runtime_base_url and (main_provider == 'custom' or main_provider.startswith('custom:')):
            resolved_provider = 'custom'
        main_chain_label = _normalize_chain_label(resolved_provider)
        if main_chain_label and _is_provider_unhealthy(main_chain_label):
            _log_skip_unhealthy(main_chain_label)
        else:
            client, resolved = resolve_provider_client(resolved_provider, main_model, explicit_base_url=explicit_base_url, explicit_api_key=explicit_api_key, api_mode=runtime_api_mode or None)
            if client is not None:
                logger.info('Auxiliary auto-detect: using main provider %s (%s)', main_provider, resolved or main_model)
                return (client, resolved or main_model)
    tried = []
    for label, try_fn in _get_provider_chain():
        if _is_provider_unhealthy(label):
            _log_skip_unhealthy(label)
            tried.append(f'{label} (unhealthy)')
            continue
        client, model = try_fn()
        if client is not None:
            if tried:
                logger.info('Auxiliary auto-detect: using %s (%s) — skipped: %s', label, model or 'default', ', '.join(tried))
            else:
                logger.info('Auxiliary auto-detect: using %s (%s)', label, model or 'default')
            return (client, model)
        tried.append(label)
    logger.warning('Auxiliary auto-detect: no provider available (tried: %s). Compression, summarization, and memory flush will not work. Set OPENROUTER_API_KEY or configure a local model in config.yaml.', ', '.join(tried))
    return (None, None)

def _to_async_client(sync_client, model: str, is_vision: bool=False):
    """Convert a sync client to its async counterpart, preserving Codex routing.

    When ``is_vision=True`` and the underlying base URL is Copilot, the
    resulting async client carries the ``Copilot-Vision-Request: true``
    header so the request is routed to Copilot's vision-capable
    infrastructure (otherwise vision payloads silently time out).
    """
    from openai import AsyncOpenAI
    if isinstance(sync_client, CodexAuxiliaryClient):
        return (AsyncCodexAuxiliaryClient(sync_client), model)
    if isinstance(sync_client, AnthropicAuxiliaryClient):
        return (AsyncAnthropicAuxiliaryClient(sync_client), model)
    try:
        from kylin_memory._vendor.agent.gemini_native_adapter import GeminiNativeClient, AsyncGeminiNativeClient
        if isinstance(sync_client, GeminiNativeClient):
            return (AsyncGeminiNativeClient(sync_client), model)
    except ImportError:
        pass
    try:
        from kylin_memory._vendor.agent.copilot_acp_client import CopilotACPClient
        if isinstance(sync_client, CopilotACPClient):
            return (sync_client, model)
    except ImportError:
        pass
    async_kwargs = {'api_key': sync_client.api_key, 'base_url': str(sync_client.base_url)}
    sync_base_url = str(sync_client.base_url)
    if base_url_host_matches(sync_base_url, 'openrouter.ai'):
        async_kwargs['default_headers'] = build_or_headers()
    elif base_url_host_matches(sync_base_url, 'api.githubcopilot.com'):
        from kylin_memory._vendor.kylin_agent_runtime_cli.copilot_auth import copilot_request_headers
        async_kwargs['default_headers'] = copilot_request_headers(is_agent_turn=True, is_vision=is_vision)
    elif base_url_host_matches(sync_base_url, 'api.kimi.com'):
        async_kwargs['default_headers'] = {'User-Agent': 'claude-code/0.1.0'}
    elif base_url_host_matches(sync_base_url, 'integrate.api.nvidia.com'):
        async_kwargs['default_headers'] = build_nvidia_nim_headers(sync_base_url)
    else:
        try:
            from kylin_memory._vendor.agent.model_metadata import _infer_provider_from_url
            from kylin_memory._vendor.providers import get_provider_profile as _gpf_async
            _inferred = _infer_provider_from_url(sync_base_url)
            if _inferred:
                _ph_async = _gpf_async(_inferred)
                if _ph_async and _ph_async.default_headers:
                    async_kwargs['default_headers'] = dict(_ph_async.default_headers)
        except Exception:
            pass
    return (AsyncOpenAI(**async_kwargs), model)

def _normalize_resolved_model(model_name: Optional[str], provider: str) -> Optional[str]:
    """Normalize a resolved model for the provider that will receive it."""
    if not model_name:
        return model_name
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.model_normalize import normalize_model_for_provider
        return normalize_model_for_provider(model_name, provider)
    except Exception:
        return model_name

def resolve_provider_client(provider: str, model: str=None, async_mode: bool=False, raw_codex: bool=False, explicit_base_url: str=None, explicit_api_key: str=None, api_mode: str=None, main_runtime: Optional[Dict[str, Any]]=None, is_vision: bool=False) -> Tuple[Optional[Any], Optional[str]]:
    """Central router: given a provider name and optional model, return a
    configured client with the correct auth, base URL, and API format.

    The returned client always exposes ``.chat.completions.create()`` — for
    Codex/Responses API providers, an adapter handles the translation
    transparently.

    Args:
        provider: Provider identifier.  One of:
            "openrouter", "nous", "openai-codex" (or "codex"),
            "zai", "kimi-coding", "minimax", "minimax-cn",
            "custom" (OPENAI_BASE_URL + OPENAI_API_KEY),
            "auto" (full auto-detection chain).
        model: Model slug override.  If None, uses the provider's default
               auxiliary model.
        async_mode: If True, return an async-compatible client.
        raw_codex: If True, return a raw OpenAI client for Codex providers
            instead of wrapping in CodexAuxiliaryClient.  Use this when
            the caller needs direct access to responses.stream() (e.g.,
            the main agent loop).
        explicit_base_url: Optional direct OpenAI-compatible endpoint.
        explicit_api_key: Optional API key paired with explicit_base_url.
        api_mode: API mode override.  One of "chat_completions",
            "codex_responses", or None (auto-detect).  When set to
            "codex_responses", the client is wrapped in
            CodexAuxiliaryClient to route through the Responses API.

    Returns:
        (client, resolved_model) or (None, None) if auth is unavailable.
    """
    _validate_proxy_env_urls()
    original_provider = (provider or '').strip().lower()
    provider = _normalize_aux_provider(provider)

    def _needs_codex_wrap(client_obj, base_url_str: str, model_str: str) -> bool:
        """Decide if a plain OpenAI client should be wrapped for Responses API.

        Returns True when api_mode is explicitly "codex_responses", or when
        auto-detection (api.openai.com + codex-family model) suggests it.
        Already-wrapped clients (CodexAuxiliaryClient) are skipped.
        """
        if isinstance(client_obj, CodexAuxiliaryClient):
            return False
        if raw_codex:
            return False
        if api_mode == 'codex_responses':
            return True
        if api_mode and api_mode != 'codex_responses':
            return False
        if base_url_hostname(base_url_str) == 'api.openai.com':
            model_lower = (model_str or '').lower()
            if 'codex' in model_lower:
                return True
        return False

    def _wrap_if_needed(client_obj, final_model_str: str, base_url_str: str='', api_key_str: str=''):
        """Wrap a plain OpenAI client in the correct transport adapter.

        Handles two cases:
        - ``CodexAuxiliaryClient`` when the endpoint needs the Responses API
          (explicit ``api_mode=codex_responses`` or api.openai.com + codex
          model name).
        - ``AnthropicAuxiliaryClient`` when the endpoint speaks Anthropic
          Messages (explicit ``api_mode=anthropic_messages``, any ``/anthropic``
          suffix, ``api.kimi.com/coding``, or ``api.anthropic.com``).

        Clients that are already specialized wrappers pass through unchanged.
        """
        if _needs_codex_wrap(client_obj, base_url_str, final_model_str):
            logger.debug('resolve_provider_client: wrapping client in CodexAuxiliaryClient (api_mode=%s, model=%s, base_url=%s)', api_mode or 'auto-detected', final_model_str, base_url_str[:60] if base_url_str else '')
            return CodexAuxiliaryClient(client_obj, final_model_str)
        return _maybe_wrap_anthropic(client_obj, final_model_str, api_key_str, base_url_str, api_mode)
    if provider == 'auto':
        client, resolved = _resolve_auto(main_runtime=main_runtime)
        if client is None:
            return (None, None)
        if model and '/' in model and resolved and ('/' not in resolved):
            logger.debug('Dropping OpenRouter-format model %r for non-OpenRouter auxiliary provider (using %r instead)', model, resolved)
            model = None
        final_model = model or resolved
        return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
    if provider == 'openrouter':
        client, default = _try_openrouter(explicit_api_key=explicit_api_key)
        if client is None:
            logger.warning('resolve_provider_client: openrouter requested but %s', _describe_openrouter_unavailable())
            return (None, None)
        final_model = _normalize_resolved_model(model or default, provider)
        return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
    if provider == 'nous':
        _is_vision = model in _PROVIDER_VISION_MODELS.values() or (model or '').strip().lower() == 'mimo-v2-omni'
        client, default = _try_nous(vision=_is_vision)
        if client is None:
            logger.warning('resolve_provider_client: nous requested but Nous Portal not configured (run: kylin-agent-runtime auth)')
            return (None, None)
        final_model = _normalize_resolved_model(model or default, provider)
        return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
    if provider == 'openai-codex':
        if not model:
            logger.warning('resolve_provider_client: openai-codex requested without a model; pass model explicitly (e.g. model.model in config.yaml or auxiliary.<task>.model for per-task aux routing).')
            return (None, None)
        if raw_codex:
            codex_token = _read_codex_access_token()
            if not codex_token:
                logger.warning('resolve_provider_client: openai-codex requested but no Codex OAuth token found (run: kylin-agent-runtime model)')
                return (None, None)
            final_model = _normalize_resolved_model(model, provider)
            raw_client = OpenAI(api_key=codex_token, base_url=_CODEX_AUX_BASE_URL, default_headers=_codex_cloudflare_headers(codex_token))
            return (raw_client, final_model)
        client, default = _build_codex_client(model)
        if client is None:
            logger.warning('resolve_provider_client: openai-codex requested but no Codex OAuth token found (run: kylin-agent-runtime model)')
            return (None, None)
        final_model = _normalize_resolved_model(model or default, provider)
        return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
    if provider == 'xai-oauth':
        client, default = _build_xai_oauth_aux_client(model)
        if client is None:
            logger.warning('resolve_provider_client: xai-oauth requested but no xAI OAuth token found (run: kylin-agent-runtime model -> xAI Grok OAuth — SuperGrok Subscription)')
            return (None, None)
        final_model = _normalize_resolved_model(model or default, provider)
        return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
    if provider == 'custom':
        if explicit_base_url:
            custom_base = _to_openai_base_url(explicit_base_url).strip()
            custom_key = (explicit_api_key or '').strip() or os.getenv('OPENAI_API_KEY', '').strip() or 'no-key-required'
            if not custom_base:
                logger.warning('resolve_provider_client: explicit custom endpoint requested but base_url is empty')
                return (None, None)
            final_model = _normalize_resolved_model(model or (main_runtime.get('model') if main_runtime else None) or 'gpt-4o-mini', provider)
            extra = {}
            _clean_base, _dq = _extract_url_query_params(custom_base)
            if _dq:
                extra['default_query'] = _dq
            if base_url_host_matches(custom_base, 'api.kimi.com'):
                extra['default_headers'] = {'User-Agent': 'claude-code/0.1.0'}
            elif base_url_host_matches(custom_base, 'api.githubcopilot.com'):
                from kylin_memory._vendor.kylin_agent_runtime_cli.copilot_auth import copilot_request_headers
                extra['default_headers'] = copilot_request_headers(is_agent_turn=True, is_vision=is_vision)
            elif base_url_host_matches(custom_base, 'integrate.api.nvidia.com'):
                extra['default_headers'] = build_nvidia_nim_headers(custom_base)
            else:
                try:
                    from kylin_memory._vendor.providers import get_provider_profile as _gpf_custom
                    _ph_custom = _gpf_custom(provider)
                    if _ph_custom and _ph_custom.default_headers:
                        extra['default_headers'] = dict(_ph_custom.default_headers)
                except Exception:
                    pass
            client = OpenAI(api_key=custom_key, base_url=_clean_base, **extra)
            client = _wrap_if_needed(client, final_model, custom_base, custom_key)
            return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
        for try_fn in (_try_custom_endpoint, _resolve_api_key_provider):
            client, default = try_fn()
            if client is not None:
                final_model = _normalize_resolved_model(model or default, provider)
                _cbase = str(getattr(client, 'base_url', '') or '')
                _raw_ckey = getattr(client, 'api_key', '')
                _ckey = '' if callable(_raw_ckey) and (not isinstance(_raw_ckey, str)) else str(_raw_ckey or '')
                client = _wrap_if_needed(client, final_model, _cbase, _ckey)
                return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
        logger.warning('resolve_provider_client: custom/main requested but no endpoint credentials found')
        return (None, None)
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.runtime_provider import _get_named_custom_provider
        custom_entry = None
        if original_provider and original_provider != provider:
            custom_entry = _get_named_custom_provider(original_provider)
        if custom_entry is None:
            custom_entry = _get_named_custom_provider(provider)
        if custom_entry:
            custom_base = custom_entry.get('base_url', '').strip()
            custom_key = custom_entry.get('api_key', '').strip()
            custom_key_env = (custom_entry.get('key_env') or custom_entry.get('api_key_env') or '').strip()
            if not custom_key and custom_key_env:
                custom_key = os.getenv(custom_key_env, '').strip()
            custom_key = custom_key or 'no-key-required'
            if custom_key == 'no-key-required':
                logger.warning('resolve_provider_client: named custom provider %r has no resolvable api_key — request will be sent with placeholder no-key-required and will 401 on auth-required endpoints', custom_entry.get('name') or provider)
            entry_api_mode = (api_mode or custom_entry.get('api_mode') or '').strip()
            if custom_base:
                final_model = _normalize_resolved_model(model or custom_entry.get('model') or (main_runtime.get('model') if main_runtime else None) or _read_main_model() or 'gpt-4o-mini', provider)
                if entry_api_mode == 'anthropic_messages':
                    openai_base = custom_base
                    raw_base_for_wrap = custom_base
                else:
                    openai_base = _to_openai_base_url(custom_base)
                    raw_base_for_wrap = custom_base
                _clean_base2, _dq2 = _extract_url_query_params(openai_base)
                _extra2 = {'default_query': _dq2} if _dq2 else {}
                logger.debug('resolve_provider_client: named custom provider %r (%s, api_mode=%s)', provider, final_model, entry_api_mode or 'chat_completions')
                if entry_api_mode == 'anthropic_messages':
                    try:
                        from kylin_memory._vendor.agent.anthropic_adapter import build_anthropic_client
                        real_client = build_anthropic_client(custom_key, custom_base)
                    except ImportError:
                        logger.warning('Named custom provider %r declares api_mode=anthropic_messages but the anthropic SDK is not installed — falling back to OpenAI-wire.', provider)
                        _fallback_base = _to_openai_base_url(custom_base)
                        _fb_clean, _fb_dq = _extract_url_query_params(_fallback_base)
                        _fb_extra = {'default_query': _fb_dq} if _fb_dq else {}
                        client = OpenAI(api_key=custom_key, base_url=_fb_clean, **_fb_extra)
                        return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
                    sync_anthropic = AnthropicAuxiliaryClient(real_client, final_model, custom_key, custom_base, is_oauth=False)
                    if async_mode:
                        return (AsyncAnthropicAuxiliaryClient(sync_anthropic), final_model)
                    return (sync_anthropic, final_model)
                client = OpenAI(api_key=custom_key, base_url=_clean_base2, **_extra2)
                if entry_api_mode == 'codex_responses' and (not isinstance(client, CodexAuxiliaryClient)):
                    client = CodexAuxiliaryClient(client, final_model)
                else:
                    client = _wrap_if_needed(client, final_model, raw_base_for_wrap, custom_key)
                return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
            logger.warning('resolve_provider_client: named custom provider %r has no base_url', provider)
            return (None, None)
    except ImportError:
        pass
    if provider == 'azure-foundry':
        client, default_model = _try_azure_foundry(model=model, explicit_api_key=explicit_api_key, explicit_base_url=explicit_base_url, api_mode=api_mode)
        if client is None:
            logger.warning('resolve_provider_client: azure-foundry requested but runtime resolution failed (run: kylin-agent-runtime doctor for diagnostics)')
            return (None, None)
        final_model = _normalize_resolved_model(model or default_model, provider)
        return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials, resolve_external_process_provider_credentials
    except ImportError:
        logger.debug('kylin_agent_runtime_cli.auth not available for provider %s', provider)
        return (None, None)
    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig is None:
        logger.warning('resolve_provider_client: unknown provider %r', provider)
        return (None, None)
    if pconfig.auth_type == 'api_key':
        if provider == 'anthropic':
            client, default_model = _try_anthropic(explicit_api_key=explicit_api_key)
            if client is None:
                logger.warning('resolve_provider_client: anthropic requested but no Anthropic credentials found')
                return (None, None)
            final_model = _normalize_resolved_model(model or default_model, provider)
            return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
        creds = resolve_api_key_provider_credentials(provider)
        api_key = str(creds.get('api_key', '')).strip()
        if explicit_api_key:
            api_key = explicit_api_key.strip() or api_key
        if not api_key:
            tried_sources = list(pconfig.api_key_env_vars)
            if provider == 'copilot':
                tried_sources.append('gh auth token')
            logger.debug('resolve_provider_client: provider %s has no API key configured (tried: %s)', provider, ', '.join(tried_sources))
            return (None, None)
        raw_base_url = str(creds.get('base_url', '')).strip().rstrip('/') or pconfig.inference_base_url
        base_url = _to_openai_base_url(raw_base_url)
        if explicit_base_url:
            base_url = _to_openai_base_url(explicit_base_url.strip().rstrip('/'))
        default_model = _get_aux_model_for_provider(provider)
        final_model = _normalize_resolved_model(model or default_model, provider)
        if provider == 'gemini':
            from kylin_memory._vendor.agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url
            if is_native_gemini_base_url(base_url):
                client = GeminiNativeClient(api_key=api_key, base_url=base_url)
                logger.debug('resolve_provider_client: %s (%s)', provider, final_model)
                return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
        headers = {}
        if base_url_host_matches(base_url, 'api.kimi.com'):
            headers['User-Agent'] = 'claude-code/0.1.0'
        elif base_url_host_matches(base_url, 'api.githubcopilot.com'):
            from kylin_memory._vendor.kylin_agent_runtime_cli.copilot_auth import copilot_request_headers
            headers.update(copilot_request_headers(is_agent_turn=True, is_vision=is_vision))
        elif base_url_host_matches(base_url, 'integrate.api.nvidia.com'):
            headers.update(build_nvidia_nim_headers(base_url))
        else:
            try:
                from kylin_memory._vendor.providers import get_provider_profile as _gpf_main
                _ph_main = _gpf_main(provider)
                if _ph_main and _ph_main.default_headers:
                    headers.update(_ph_main.default_headers)
            except Exception:
                pass
        client = OpenAI(api_key=api_key, base_url=base_url, **{'default_headers': headers} if headers else {})
        if provider == 'copilot' and final_model and (not raw_codex):
            try:
                from kylin_memory._vendor.kylin_agent_runtime_cli.models import _should_use_copilot_responses_api
                if _should_use_copilot_responses_api(final_model):
                    logger.debug('resolve_provider_client: copilot model %s needs Responses API — wrapping with CodexAuxiliaryClient', final_model)
                    client = CodexAuxiliaryClient(client, final_model)
            except ImportError:
                pass
        client = _wrap_if_needed(client, final_model, raw_base_url, api_key)
        logger.debug('resolve_provider_client: %s (%s)', provider, final_model)
        return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
    if pconfig.auth_type == 'external_process':
        creds = resolve_external_process_provider_credentials(provider)
        final_model = _normalize_resolved_model(model or (main_runtime.get('model') if main_runtime else None) or _read_main_model(), provider)
        if provider == 'copilot-acp':
            api_key = str(creds.get('api_key', '')).strip()
            base_url = str(creds.get('base_url', '')).strip()
            command = str(creds.get('command', '')).strip() or None
            args = list(creds.get('args') or [])
            if not final_model:
                logger.warning('resolve_provider_client: copilot-acp requested but no model was provided or configured')
                return (None, None)
            if not api_key or not base_url:
                logger.warning('resolve_provider_client: copilot-acp requested but external process credentials are incomplete')
                return (None, None)
            from kylin_memory._vendor.agent.copilot_acp_client import CopilotACPClient
            client = CopilotACPClient(api_key=api_key, base_url=base_url, command=command, args=args)
            logger.debug('resolve_provider_client: %s (%s)', provider, final_model)
            return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
        logger.warning('resolve_provider_client: external-process provider %s not directly supported', provider)
        return (None, None)
    elif pconfig.auth_type == 'aws_sdk':
        try:
            from kylin_memory._vendor.agent.bedrock_adapter import has_aws_credentials, resolve_bedrock_region
            from kylin_memory._vendor.agent.anthropic_adapter import build_anthropic_bedrock_client
        except ImportError:
            logger.warning('resolve_provider_client: bedrock requested but boto3 or anthropic SDK not installed')
            return (None, None)
        if not has_aws_credentials():
            logger.debug('resolve_provider_client: bedrock requested but no AWS credentials found')
            return (None, None)
        region = resolve_bedrock_region()
        default_model = 'anthropic.claude-haiku-4-5-20251001-v1:0'
        final_model = _normalize_resolved_model(model or default_model, provider)
        try:
            real_client = build_anthropic_bedrock_client(region)
        except ImportError as exc:
            logger.warning('resolve_provider_client: cannot create Bedrock client: %s', exc)
            return (None, None)
        client = AnthropicAuxiliaryClient(real_client, final_model, api_key='aws-sdk', base_url=f'https://bedrock-runtime.{region}.amazonaws.com')
        logger.debug('resolve_provider_client: bedrock (%s, %s)', final_model, region)
        return _to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model)
    elif pconfig.auth_type in {'oauth_device_code', 'oauth_external'}:
        if provider == 'nous':
            return resolve_provider_client('nous', model, async_mode)
        if provider == 'openai-codex':
            return resolve_provider_client('openai-codex', model, async_mode)
        if provider == 'xai-oauth':
            return resolve_provider_client('xai-oauth', model, async_mode)
        logger.warning("resolve_provider_client: OAuth provider %s not directly supported, try 'auto'", provider)
        return (None, None)
    logger.warning('resolve_provider_client: unhandled auth_type %s for %s', pconfig.auth_type, provider)
    return (None, None)

def get_text_auxiliary_client(task: str='', *, main_runtime: Optional[Dict[str, Any]]=None) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for text-only auxiliary tasks.

    Args:
        task: Optional task name ("compression", "web_extract") to check
              for a task-specific provider override.

    Callers may override the returned model via config.yaml
    (e.g. auxiliary.compression.model, auxiliary.web_extract.model).
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(provider, model=model, explicit_base_url=base_url, explicit_api_key=api_key, api_mode=api_mode, main_runtime=main_runtime)

def get_async_text_auxiliary_client(task: str='', *, main_runtime: Optional[Dict[str, Any]]=None):
    """Return (async_client, model_slug) for async consumers.

    For standard providers returns (AsyncOpenAI, model). For Codex returns
    (AsyncCodexAuxiliaryClient, model) which wraps the Responses API.
    Returns (None, None) when no provider is available.
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(provider, model=model, async_mode=True, explicit_base_url=base_url, explicit_api_key=api_key, api_mode=api_mode, main_runtime=main_runtime)
_VISION_AUTO_PROVIDER_ORDER = ('openrouter', 'nous')

def _normalize_vision_provider(provider: Optional[str]) -> str:
    return _normalize_aux_provider(provider)

def _resolve_strict_vision_backend(provider: str, model: Optional[str]=None) -> Tuple[Optional[Any], Optional[str]]:
    provider = _normalize_vision_provider(provider)
    if provider == 'copilot':
        return resolve_provider_client('copilot', model, is_vision=True)
    if provider == 'openrouter':
        return _try_openrouter(model=model)
    if provider == 'nous':
        return _try_nous(vision=True)
    if provider == 'openai-codex':
        return resolve_provider_client('openai-codex', model, is_vision=True)
    if provider == 'anthropic':
        return _try_anthropic()
    if provider == 'custom':
        return _try_custom_endpoint()
    return (None, None)

def _strict_vision_backend_available(provider: str) -> bool:
    return _resolve_strict_vision_backend(provider)[0] is not None

def get_available_vision_backends() -> List[str]:
    """Return the currently available vision backends in auto-selection order.

    Order: active provider → OpenRouter → Nous → stop.  This is the single
    source of truth for setup, tool gating, and runtime auto-routing of
    vision tasks.
    """
    available: List[str] = []
    main_provider = _read_main_provider()
    if main_provider and main_provider not in {'auto', ''}:
        if main_provider in _VISION_AUTO_PROVIDER_ORDER:
            if _strict_vision_backend_available(main_provider):
                available.append(main_provider)
        else:
            client, _ = resolve_provider_client(main_provider, _read_main_model())
            if client is not None:
                available.append(main_provider)
    for p in _VISION_AUTO_PROVIDER_ORDER:
        if p not in available and _strict_vision_backend_available(p):
            available.append(p)
    return available

def resolve_vision_provider_client(provider: Optional[str]=None, model: Optional[str]=None, *, base_url: Optional[str]=None, api_key: Optional[str]=None, async_mode: bool=False) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Resolve the client actually used for vision tasks.

    Direct endpoint overrides take precedence over provider selection. Explicit
    provider overrides still use the generic provider router for non-standard
    backends, so users can intentionally force experimental providers. Auto mode
    stays conservative and only tries vision backends known to work today.
    """
    requested, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model('vision', provider, model, base_url, api_key)
    requested = _normalize_vision_provider(requested)

    def _finalize(resolved_provider: str, sync_client: Any, default_model: Optional[str]):
        if sync_client is None:
            return (resolved_provider, None, None)
        final_model = resolved_model or default_model
        if async_mode:
            async_client, async_model = _to_async_client(sync_client, final_model, is_vision=True)
            return (resolved_provider, async_client, async_model)
        return (resolved_provider, sync_client, final_model)
    if resolved_base_url:
        provider_for_base_override = requested if requested and requested not in {'', 'auto'} else 'custom'
        client, final_model = resolve_provider_client(provider_for_base_override, model=resolved_model, async_mode=async_mode, explicit_base_url=resolved_base_url, explicit_api_key=resolved_api_key, api_mode=resolved_api_mode)
        if client is None:
            return (provider_for_base_override, None, None)
        return (provider_for_base_override, client, final_model)
    if requested == 'auto':
        main_provider = _read_main_provider()
        main_model = _read_main_model()
        if main_provider and main_provider not in {'auto', ''}:
            vision_model = _PROVIDER_VISION_MODELS.get(main_provider, main_model)
            if main_provider == 'nous':
                sync_client, default_model = _resolve_strict_vision_backend(main_provider, vision_model)
                if sync_client is not None:
                    logger.info('Vision auto-detect: using main provider %s (%s)', main_provider, default_model or resolved_model or main_model)
                    return _finalize(main_provider, sync_client, default_model)
            elif main_provider in _PROVIDERS_WITHOUT_VISION:
                logger.debug('Vision auto-detect: skipping main provider %s (no vision support) — falling through to aggregator chain', main_provider)
            else:
                rpc_client, rpc_model = resolve_provider_client(main_provider, vision_model, api_mode=resolved_api_mode, is_vision=True)
                if rpc_client is not None:
                    logger.info('Vision auto-detect: using main provider %s (%s)', main_provider, rpc_model or vision_model)
                    return _finalize(main_provider, rpc_client, rpc_model or vision_model)
        for candidate in _VISION_AUTO_PROVIDER_ORDER:
            if candidate == main_provider:
                continue
            sync_client, default_model = _resolve_strict_vision_backend(candidate)
            if sync_client is not None:
                return _finalize(candidate, sync_client, default_model)
        logger.debug('Auxiliary vision client: none available')
        return (None, None, None)
    if requested in _VISION_AUTO_PROVIDER_ORDER:
        sync_client, default_model = _resolve_strict_vision_backend(requested, resolved_model)
        return _finalize(requested, sync_client, default_model)
    if requested == 'zai' and (not resolved_base_url):
        zai_openai_urls = ['https://open.bigmodel.cn/api/paas/v4', 'https://api.z.ai/api/paas/v4']
        for _zai_url in zai_openai_urls:
            client, final_model = _get_cached_client(requested, resolved_model, async_mode, base_url=_zai_url, api_key=resolved_api_key or None, api_mode='chat_completions', is_vision=True)
            if client is not None:
                return _finalize(requested, client, final_model)
        client, final_model = _get_cached_client(requested, resolved_model, async_mode, api_mode=resolved_api_mode, is_vision=True)
        if client is None:
            return (requested, None, None)
        return (requested, client, final_model)
    client, final_model = _get_cached_client(requested, resolved_model, async_mode, api_mode=resolved_api_mode, is_vision=True)
    if client is None:
        return (requested, None, None)
    return (requested, client, final_model)

def get_auxiliary_extra_body() -> dict:
    """Return extra_body kwargs for auxiliary API calls.
    
    Includes Nous Portal product tags when the auxiliary client is backed
    by Nous Portal. Returns empty dict otherwise.
    """
    return _nous_extra_body() if auxiliary_is_nous else {}

def auxiliary_max_tokens_param(value: int) -> dict:
    """Return the correct max tokens kwarg for the auxiliary client's provider.
    
    OpenRouter and local models use 'max_tokens'. Direct OpenAI with newer
    models (gpt-4o, o-series, gpt-5+) requires 'max_completion_tokens'.
    The Codex adapter translates max_tokens internally, so we use max_tokens
    for it as well.
    """
    custom_base = _current_custom_base_url()
    or_key = os.getenv('OPENROUTER_API_KEY')
    if not or_key and _read_nous_auth() is None and (base_url_hostname(custom_base) in {'api.openai.com', 'api.githubcopilot.com'}):
        return {'max_completion_tokens': value}
    return {'max_tokens': value}
_client_cache: Dict[tuple, tuple] = {}
_client_cache_lock = threading.Lock()
_CLIENT_CACHE_MAX_SIZE = 64

def _client_cache_key(provider: str, *, async_mode: bool, base_url: Optional[str]=None, api_key: Optional[str]=None, api_mode: Optional[str]=None, main_runtime: Optional[Dict[str, Any]]=None, is_vision: bool=False) -> tuple:
    runtime = _normalize_main_runtime(main_runtime)
    runtime_key = tuple((runtime.get(field, '') for field in _MAIN_RUNTIME_FIELDS)) if provider == 'auto' else ()
    pool_hint = _pool_cache_hint(provider, main_runtime=main_runtime)
    return (provider, async_mode, base_url or '', api_key or '', api_mode or '', runtime_key, is_vision, pool_hint)

def _store_cached_client(cache_key: tuple, client: Any, default_model: Optional[str], *, bound_loop: Any=None) -> None:
    with _client_cache_lock:
        old_entry = _client_cache.get(cache_key)
        if old_entry is not None and old_entry[0] is not client:
            _force_close_async_httpx(old_entry[0])
            try:
                close_fn = getattr(old_entry[0], 'close', None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
        _client_cache[cache_key] = (client, default_model, bound_loop)

def _refresh_nous_auxiliary_client(*, cache_provider: str, model: Optional[str], async_mode: bool, base_url: Optional[str]=None, api_key: Optional[str]=None, api_mode: Optional[str]=None, main_runtime: Optional[Dict[str, Any]]=None, is_vision: bool=False) -> Tuple[Optional[Any], Optional[str]]:
    """Refresh Nous runtime creds, rebuild the client, and replace the cache entry."""
    runtime = _resolve_nous_runtime_api(force_refresh=True)
    if runtime is None:
        return (None, model)
    fresh_key, fresh_base_url = runtime
    sync_client = OpenAI(api_key=fresh_key, base_url=fresh_base_url)
    final_model = model
    current_loop = None
    if async_mode:
        try:
            import asyncio as _aio
            current_loop = _aio.get_event_loop()
        except RuntimeError:
            pass
        client, final_model = _to_async_client(sync_client, final_model or '', is_vision=is_vision)
    else:
        client = sync_client
    cache_key = _client_cache_key(cache_provider, async_mode=async_mode, base_url=base_url, api_key=api_key, api_mode=api_mode, main_runtime=main_runtime, is_vision=is_vision)
    _store_cached_client(cache_key, client, final_model, bound_loop=current_loop)
    return (client, final_model)

def neuter_async_httpx_del() -> None:
    """Monkey-patch ``AsyncHttpxClientWrapper.__del__`` to be a no-op.

    The OpenAI SDK's ``AsyncHttpxClientWrapper.__del__`` schedules
    ``self.aclose()`` via ``asyncio.get_running_loop().create_task()``.
    When an ``AsyncOpenAI`` client is garbage-collected while
    prompt_toolkit's event loop is running (the common CLI idle state),
    the ``aclose()`` task runs on prompt_toolkit's loop but the
    underlying TCP transport is bound to a *different* loop (the worker
    thread's loop that the client was originally created on).  If that
    loop is closed or its thread is dead, the transport's
    ``self._loop.call_soon()`` raises ``RuntimeError("Event loop is
    closed")``, which prompt_toolkit surfaces as "Unhandled exception
    in event loop ... Press ENTER to continue...".

    Neutering ``__del__`` is safe because:
    - Cached clients are explicitly cleaned via ``_force_close_async_httpx``
      on stale-loop detection and ``shutdown_cached_clients`` on exit.
    - Uncached clients' TCP connections are cleaned up by the OS when the
      process exits.
    - The OpenAI SDK itself marks this as a TODO (``# TODO(someday):
      support non asyncio runtimes here``).

    Call this once at CLI startup, before any ``AsyncOpenAI`` clients are
    created.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper
        AsyncHttpxClientWrapper.__del__ = lambda self: None
    except (ImportError, AttributeError):
        pass

def _force_close_async_httpx(client: Any) -> None:
    """Mark the httpx AsyncClient inside an AsyncOpenAI client as closed.

    This prevents ``AsyncHttpxClientWrapper.__del__`` from scheduling
    ``aclose()`` on a (potentially closed) event loop, which causes
    ``RuntimeError: Event loop is closed`` → prompt_toolkit's
    "Press ENTER to continue..." handler.

    We intentionally do NOT run the full async close path — the
    connections will be dropped by the OS when the process exits.
    """
    try:
        from httpx._client import ClientState
        inner = getattr(client, '_client', None)
        if inner is not None and (not getattr(inner, 'is_closed', True)):
            inner._state = ClientState.CLOSED
    except Exception:
        pass

def shutdown_cached_clients() -> None:
    """Close all cached clients (sync and async) to prevent event-loop errors.

    Call this during CLI shutdown, *before* the event loop is closed, to
    avoid ``AsyncHttpxClientWrapper.__del__`` raising on a dead loop.
    """
    import inspect
    with _client_cache_lock:
        for key, entry in list(_client_cache.items()):
            client = entry[0]
            if client is None:
                continue
            _force_close_async_httpx(client)
            try:
                close_fn = getattr(client, 'close', None)
                if close_fn and (not inspect.iscoroutinefunction(close_fn)):
                    close_fn()
            except Exception:
                pass
        _client_cache.clear()

def cleanup_stale_async_clients() -> None:
    """Force-close cached async clients whose event loop is closed.

    Call this after each agent turn to proactively clean up stale clients
    before GC can trigger ``AsyncHttpxClientWrapper.__del__`` on them.
    This is defense-in-depth — the primary fix is ``neuter_async_httpx_del``
    which disables ``__del__`` entirely.
    """
    with _client_cache_lock:
        stale_keys = []
        for key, entry in _client_cache.items():
            client, _default, cached_loop = entry
            if cached_loop is not None and cached_loop.is_closed():
                _force_close_async_httpx(client)
                stale_keys.append(key)
        for key in stale_keys:
            del _client_cache[key]

def _is_openrouter_client(client: Any) -> bool:
    for obj in (client, getattr(client, '_client', None), getattr(client, 'client', None)):
        if obj and base_url_host_matches(str(getattr(obj, 'base_url', '') or ''), 'openrouter.ai'):
            return True
    return False

def _cached_client_accepts_slash_models(client: Any, cached_default: Optional[str]) -> bool:
    """Best-effort check for cached clients that accept ``vendor/model`` IDs."""
    if _is_openrouter_client(client):
        return True
    return bool(cached_default and '/' in cached_default)

def _compat_model(client: Any, model: Optional[str], cached_default: Optional[str]) -> Optional[str]:
    """Keep slash-bearing model IDs only for cached clients that support them.

    Mirrors the guard in resolve_provider_client() which is skipped on cache hits.
    """
    if model and '/' in model and (not _cached_client_accepts_slash_models(client, cached_default)):
        return cached_default
    return model or cached_default

def _get_cached_client(provider: str, model: str=None, async_mode: bool=False, base_url: str=None, api_key: str=None, api_mode: str=None, main_runtime: Optional[Dict[str, Any]]=None, is_vision: bool=False) -> Tuple[Optional[Any], Optional[str]]:
    """Get or create a cached client for the given provider.

    Async clients (AsyncOpenAI) use httpx.AsyncClient internally, which
    binds to the event loop that was current when the client was created.
    Using such a client on a *different* loop causes deadlocks or
    RuntimeError.  To prevent cross-loop issues, the cache validates on
    every async hit that the cached loop is the *current, open* loop.
    If the loop changed (e.g. a new gateway worker-thread loop), the stale
    entry is replaced in-place rather than creating an additional entry.

    This keeps cache size bounded to one entry per unique provider config,
    preventing the fd-exhaustion that previously occurred in long-running
    gateways where recycled worker threads created unbounded entries (#10200).
    """
    current_loop = None
    if async_mode:
        try:
            import asyncio as _aio
            current_loop = _aio.get_event_loop()
        except RuntimeError:
            pass
    runtime = _normalize_main_runtime(main_runtime)
    cache_key = _client_cache_key(provider, async_mode=async_mode, base_url=base_url, api_key=api_key, api_mode=api_mode, main_runtime=main_runtime, is_vision=is_vision)
    with _client_cache_lock:
        if cache_key in _client_cache:
            cached_client, cached_default, cached_loop = _client_cache[cache_key]
            if async_mode:
                loop_ok = cached_loop is not None and cached_loop is current_loop and (not cached_loop.is_closed())
                if loop_ok:
                    effective = _compat_model(cached_client, model, cached_default)
                    return (cached_client, effective)
                _force_close_async_httpx(cached_client)
                del _client_cache[cache_key]
            else:
                effective = _compat_model(cached_client, model, cached_default)
                return (cached_client, effective)
    client, default_model = resolve_provider_client(provider, model, async_mode, explicit_base_url=base_url, explicit_api_key=api_key, api_mode=api_mode, main_runtime=runtime, is_vision=is_vision)
    if client is not None:
        bound_loop = current_loop
        with _client_cache_lock:
            if cache_key not in _client_cache:
                while len(_client_cache) >= _CLIENT_CACHE_MAX_SIZE:
                    evict_key, evict_entry = next(iter(_client_cache.items()))
                    _force_close_async_httpx(evict_entry[0])
                    del _client_cache[evict_key]
                _client_cache[cache_key] = (client, default_model, bound_loop)
            else:
                client, default_model, _ = _client_cache[cache_key]
    return (client, model or default_model)

def _resolve_task_provider_model(task: str=None, provider: str=None, model: str=None, base_url: str=None, api_key: str=None) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Determine provider + model for a call.

    Priority:
      1. Explicit provider/model/base_url/api_key args (always win)
      2. Config file (auxiliary.{task}.provider/model/base_url)
      3. "auto" (full auto-detection chain)

    Returns (provider, model, base_url, api_key, api_mode) where model may
    be None (use provider default). When base_url is set, provider is forced
    to "custom" and the task uses that direct endpoint. api_mode is one of
    "chat_completions", "codex_responses", or None (auto-detect).
    """
    cfg_provider = None
    cfg_model = None
    cfg_base_url = None
    cfg_api_key = None
    cfg_api_mode = None
    if task:
        task_config = _get_auxiliary_task_config(task)
        cfg_provider = str(task_config.get('provider', '')).strip() or None
        cfg_model = str(task_config.get('model', '')).strip() or None
        cfg_base_url = str(task_config.get('base_url', '')).strip() or None
        cfg_api_key = str(task_config.get('api_key', '')).strip() or None
        cfg_api_mode = str(task_config.get('api_mode', '')).strip() or None
    resolved_model = model or cfg_model
    resolved_api_mode = cfg_api_mode
    if base_url:
        return ('custom', resolved_model, base_url, api_key, resolved_api_mode)
    if provider:
        return (provider, resolved_model, base_url, api_key, resolved_api_mode)
    if task:
        if cfg_base_url and cfg_api_key:
            return ('custom', resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode)
        if cfg_base_url and cfg_provider and (cfg_provider != 'auto'):
            return (cfg_provider, resolved_model, cfg_base_url, None, resolved_api_mode)
        if cfg_provider and cfg_provider != 'auto':
            return (cfg_provider, resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode)
        return ('auto', resolved_model, None, None, resolved_api_mode)
    return ('auto', resolved_model, None, None, resolved_api_mode)
_DEFAULT_AUX_TIMEOUT = 30.0

def _get_auxiliary_task_config(task: str) -> Dict[str, Any]:
    """Return the config dict for auxiliary.<task>, or {} when unavailable."""
    if not task:
        return {}
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.config import load_config
        config = load_config()
    except ImportError:
        return {}
    aux = config.get('auxiliary', {}) if isinstance(config, dict) else {}
    task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
    return task_config if isinstance(task_config, dict) else {}

def _get_task_timeout(task: str, default: float=_DEFAULT_AUX_TIMEOUT) -> float:
    """Read timeout from auxiliary.{task}.timeout in config, falling back to *default*."""
    if not task:
        return default
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get('timeout')
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    return default

def _get_task_extra_body(task: str) -> Dict[str, Any]:
    """Read auxiliary.<task>.extra_body and return a shallow copy when valid."""
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get('extra_body')
    if isinstance(raw, dict):
        return dict(raw)
    return {}
_ANTHROPIC_COMPAT_PROVIDERS = frozenset({'minimax', 'minimax-oauth', 'minimax-cn'})

def _is_anthropic_compat_endpoint(provider: str, base_url: str) -> bool:
    """Detect if an endpoint expects Anthropic-format content blocks.

    Returns True for known Anthropic-compatible providers (MiniMax) and
    any endpoint whose URL contains ``/anthropic`` in the path.
    """
    if provider in _ANTHROPIC_COMPAT_PROVIDERS:
        return True
    url_lower = (base_url or '').lower()
    return '/anthropic' in url_lower

def _convert_openai_images_to_anthropic(messages: list) -> list:
    """Convert OpenAI ``image_url`` content blocks to Anthropic ``image`` blocks.

    Only touches messages that have list-type content with ``image_url`` blocks;
    plain text messages pass through unchanged.
    """
    converted = []
    for msg in messages:
        content = msg.get('content')
        if not isinstance(content, list):
            converted.append(msg)
            continue
        new_content = []
        changed = False
        for block in content:
            if block.get('type') == 'image_url':
                image_url_val = (block.get('image_url') or {}).get('url', '')
                if image_url_val.startswith('data:'):
                    header, _, b64data = image_url_val.partition(',')
                    media_type = 'image/png'
                    if ':' in header and ';' in header:
                        media_type = header.split(':', 1)[1].split(';', 1)[0]
                    new_content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': b64data}})
                else:
                    new_content.append({'type': 'image', 'source': {'type': 'url', 'url': image_url_val}})
                changed = True
            else:
                new_content.append(block)
        converted.append({**msg, 'content': new_content} if changed else msg)
    return converted

def _build_call_kwargs(provider: str, model: str, messages: list, temperature: Optional[float]=None, max_tokens: Optional[int]=None, tools: Optional[list]=None, timeout: float=30.0, extra_body: Optional[dict]=None, base_url: Optional[str]=None, tool_choice: Any=None) -> dict:
    """Build kwargs for .chat.completions.create() with model/provider adjustments."""
    kwargs: Dict[str, Any] = {'model': model, 'messages': messages, 'timeout': timeout}
    fixed_temperature = _fixed_temperature_for_model(model, base_url)
    if fixed_temperature is OMIT_TEMPERATURE:
        temperature = None
    elif fixed_temperature is not None:
        temperature = fixed_temperature
    if temperature is not None:
        from kylin_memory._vendor.agent.anthropic_adapter import _forbids_sampling_params
        if _forbids_sampling_params(model):
            temperature = None
    if temperature is not None:
        kwargs['temperature'] = temperature
    if max_tokens is not None:
        _model_lower = (model or '').lower()
        _skip_max_tokens = provider == 'zai' and ('4v' in _model_lower or '5v' in _model_lower or '-v' in _model_lower)
        if _skip_max_tokens:
            pass
        elif provider == 'custom':
            custom_base = base_url or _current_custom_base_url()
            if base_url_hostname(custom_base) == 'api.openai.com':
                kwargs['max_completion_tokens'] = max_tokens
            else:
                kwargs['max_tokens'] = max_tokens
        else:
            kwargs['max_tokens'] = max_tokens
    if tools:
        _seen: set = set()
        _deduped: list = []
        for _t in tools:
            _tname = (_t.get('function') or {}).get('name', '')
            if _tname and _tname in _seen:
                logger.warning("_build_call_kwargs: duplicate tool name '%s' removed (provider=%s model=%s)", _tname, provider, model)
                continue
            if _tname:
                _seen.add(_tname)
            _deduped.append(_t)
        kwargs['tools'] = _deduped
        if tool_choice is not None:
            kwargs['tool_choice'] = tool_choice
    merged_extra = dict(extra_body or {})
    _named_tool_choice = False
    if isinstance(tool_choice, dict):
        _named_tool_choice = str(tool_choice.get('type') or '').lower() == 'function' and bool((tool_choice.get('function') or {}).get('name'))
    elif isinstance(tool_choice, str):
        _named_tool_choice = tool_choice.lower() not in {'', 'auto', 'none', 'required'}
    if tools and _named_tool_choice and _is_deepseek_thinking_model(model, base_url, provider):
        merged_extra['thinking'] = {'type': 'disabled'}
        logger.info('Auxiliary structured tool call: disabling DeepSeek thinking for named tool_choice (model=%s)', model)
    if provider == 'nous' or auxiliary_is_nous:
        merged_extra.setdefault('tags', []).extend(_nous_portal_tags())
    try:
        from kylin_memory._vendor.agent.request_priority import merge_priority_into_extra_body
        merged_extra = merge_priority_into_extra_body(merged_extra)
    except Exception:
        pass
    if merged_extra:
        kwargs['extra_body'] = merged_extra
    return kwargs

def _validate_llm_response(response: Any, task: str=None) -> Any:
    """Validate that an LLM response has the expected .choices[0].message shape.

    Fails fast with a clear error instead of letting malformed payloads
    propagate to downstream consumers where they crash with misleading
    AttributeError (e.g. "'str' object has no attribute 'choices'").

    See #7264.
    """
    if response is None:
        raise RuntimeError(f"Auxiliary {task or 'call'}: LLM returned None response")
    try:
        choices = response.choices
        if not choices or not hasattr(choices[0], 'message'):
            raise AttributeError('missing choices[0].message')
    except (AttributeError, TypeError, IndexError) as exc:
        response_type = type(response).__name__
        response_preview = str(response)[:120]
        raise RuntimeError(f"Auxiliary {task or 'call'}: LLM returned invalid response (type={response_type}): {response_preview!r}. Expected object with .choices[0].message — check provider adapter or custom endpoint compatibility.") from exc
    return response

def call_llm(task: str=None, *, provider: str=None, model: str=None, base_url: str=None, api_key: str=None, api_mode: str=None, main_runtime: Optional[Dict[str, Any]]=None, messages: list, temperature: float=None, max_tokens: int=None, tools: list=None, tool_choice: Any=None, timeout: float=None, extra_body: dict=None) -> Any:
    """Centralized synchronous LLM call.

    Resolves provider + model (from task config, explicit args, or auto-detect),
    handles auth, request formatting, and model-specific arg adjustments.

    Args:
        task: Auxiliary task name ("compression", "vision", "web_extract",
              "skills_hub", "mcp", "title_generation").
              Reads provider:model from config/env. Ignored if provider is set.
        provider: Explicit provider override.
        model: Explicit model override.
        api_mode: Optional API transport override.
        messages: Chat messages list.
        temperature: Sampling temperature (None = provider default).
        max_tokens: Max output tokens (handles max_tokens vs max_completion_tokens).
        tools: Tool definitions (for function calling).
        tool_choice: Optional tool selection policy. Accepts the OpenAI
            Chat Completions spelling, including a named function choice.
        timeout: Request timeout in seconds (None = read from auxiliary.{task}.timeout config).
        extra_body: Additional request body fields.

    Returns:
        Response object with .choices[0].message.content

    Raises:
        RuntimeError: If no provider is configured.
    """
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(task, provider, model, base_url, api_key)
    if api_mode:
        resolved_api_mode = str(api_mode).strip() or resolved_api_mode
    effective_extra_body = _get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})
    if task == 'vision':
        effective_provider, client, final_model = resolve_vision_provider_client(provider=resolved_provider if resolved_provider != 'auto' else provider, model=resolved_model or model, base_url=resolved_base_url or base_url, api_key=resolved_api_key or api_key, async_mode=False)
        if client is None and resolved_provider != 'auto' and (not resolved_base_url):
            logger.warning('Vision provider %s unavailable, falling back to auto vision backends', resolved_provider)
            effective_provider, client, final_model = resolve_vision_provider_client(provider='auto', model=resolved_model, async_mode=False)
        if client is None:
            raise RuntimeError(f'No LLM provider configured for task={task} provider={resolved_provider}. Run: kylin-agent-runtime setup')
        resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(resolved_provider, resolved_model, base_url=resolved_base_url, api_key=resolved_api_key, api_mode=resolved_api_mode, main_runtime=main_runtime)
        if client is None:
            _explicit = (resolved_provider or '').strip().lower()
            if _explicit and _explicit not in {'auto', 'openrouter', 'custom'}:
                raise RuntimeError(f"Provider '{_explicit}' is set in config.yaml but no API key was found. Set the {_explicit.upper()}_API_KEY environment variable, or switch to a different provider with `kylin-agent-runtime model`.")
            if not resolved_base_url:
                logger.info('Auxiliary %s: provider %s unavailable, trying auto-detection chain', task or 'call', resolved_provider)
                client, final_model = _get_cached_client('auto', main_runtime=main_runtime)
        if client is None:
            raise RuntimeError(f'No LLM provider configured for task={task} provider={resolved_provider}. Run: kylin-agent-runtime setup')
    effective_timeout = timeout if timeout is not None else _get_task_timeout(task)
    _base_info = str(getattr(client, 'base_url', resolved_base_url) or '')
    if task:
        logger.info('Auxiliary %s: using %s (%s)%s', task, resolved_provider or 'auto', final_model or 'default', f' at {_base_info}' if _base_info and 'openrouter' not in _base_info else '')
    kwargs = _build_call_kwargs(resolved_provider, final_model, messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, timeout=effective_timeout, extra_body=effective_extra_body, base_url=_base_info or resolved_base_url)
    _client_base = str(getattr(client, 'base_url', '') or '')
    if _is_anthropic_compat_endpoint(resolved_provider, _client_base):
        kwargs['messages'] = _convert_openai_images_to_anthropic(kwargs['messages'])
    try:
        return _validate_llm_response(client.chat.completions.create(**kwargs), task)
    except Exception as first_err:
        if tools and tool_choice is not None and _is_thinking_tool_choice_error(first_err) and (not _thinking_is_disabled(kwargs.get('extra_body'))):
            retry_kwargs = dict(kwargs)
            retry_extra = dict(retry_kwargs.get('extra_body') or {})
            retry_extra['thinking'] = {'type': 'disabled'}
            retry_kwargs['extra_body'] = retry_extra
            logger.info('Auxiliary %s: provider rejected named tool_choice in thinking mode; retrying with thinking disabled', task or 'call')
            try:
                return _validate_llm_response(client.chat.completions.create(**retry_kwargs), task)
            except Exception as retry_err:
                first_err = retry_err
                kwargs = retry_kwargs
        if 'temperature' in kwargs and _is_unsupported_temperature_error(first_err):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop('temperature', None)
            logger.info('Auxiliary %s: provider rejected temperature; retrying once without it', task or 'call')
            try:
                return _validate_llm_response(client.chat.completions.create(**retry_kwargs), task)
            except Exception as retry_err:
                retry_err_str = str(retry_err)
                if not (_is_payment_error(retry_err) or _is_connection_error(retry_err) or _is_auth_error(retry_err) or ('max_tokens' in retry_err_str) or ('unsupported_parameter' in retry_err_str)):
                    raise
                first_err = retry_err
                kwargs = retry_kwargs
        err_str = str(first_err)
        _is_zai_param_error = '1210' in err_str and 'bigmodel' in str(getattr(client, 'base_url', ''))
        if max_tokens is not None and ('max_tokens' in err_str or 'unsupported_parameter' in err_str or _is_unsupported_parameter_error(first_err, 'max_tokens') or _is_zai_param_error):
            kwargs.pop('max_tokens', None)
            kwargs.pop('max_completion_tokens', None)
            try:
                return _validate_llm_response(client.chat.completions.create(**kwargs), task)
            except Exception as retry_err:
                if not (_is_payment_error(retry_err) or _is_connection_error(retry_err) or _is_rate_limit_error(retry_err)):
                    raise
                first_err = retry_err
        client_is_nous = resolved_provider == 'nous' or base_url_host_matches(_base_info, 'inference-api.nousresearch.com')
        if _is_auth_error(first_err) and client_is_nous:
            refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(cache_provider=resolved_provider or 'nous', model=final_model, async_mode=False, base_url=resolved_base_url, api_key=resolved_api_key, api_mode=resolved_api_mode, main_runtime=main_runtime, is_vision=task == 'vision')
            if refreshed_client is not None:
                logger.info('Auxiliary %s: refreshed Nous runtime credentials after 401, retrying', task or 'call')
                if refreshed_model and refreshed_model != kwargs.get('model'):
                    kwargs['model'] = refreshed_model
                return _validate_llm_response(refreshed_client.chat.completions.create(**kwargs), task)
        if _is_auth_error(first_err) and resolved_provider not in {'auto', '', None} and (not client_is_nous):
            if _refresh_provider_credentials(resolved_provider):
                logger.info('Auxiliary %s: refreshed %s credentials after auth error, retrying', task or 'call', resolved_provider)
                return _retry_same_provider_sync(task=task, resolved_provider=resolved_provider, resolved_model=resolved_model, resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key, resolved_api_mode=resolved_api_mode, main_runtime=main_runtime, final_model=final_model, messages=messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, effective_timeout=effective_timeout, effective_extra_body=effective_extra_body)
        pool_provider = _recoverable_pool_provider(resolved_provider, client)
        if pool_provider and (_is_auth_error(first_err) or _is_payment_error(first_err) or _is_rate_limit_error(first_err)):
            recovery_err = first_err
            if _is_rate_limit_error(first_err):
                try:
                    return _validate_llm_response(client.chat.completions.create(**kwargs), task)
                except Exception as retry_err:
                    if not (_is_auth_error(retry_err) or _is_payment_error(retry_err) or _is_rate_limit_error(retry_err)):
                        raise
                    recovery_err = retry_err
            if _recover_provider_pool(pool_provider, recovery_err):
                logger.info('Auxiliary %s: recovered %s via credential-pool rotation after %s', task or 'call', pool_provider, type(recovery_err).__name__)
                return _retry_same_provider_sync(task=task, resolved_provider=resolved_provider, resolved_model=resolved_model, resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key, resolved_api_mode=resolved_api_mode, main_runtime=main_runtime, final_model=final_model, messages=messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, effective_timeout=effective_timeout, effective_extra_body=effective_extra_body)
        should_fallback = _is_payment_error(first_err) or _is_connection_error(first_err) or _is_rate_limit_error(first_err)
        is_auto = resolved_provider in {'auto', '', None}
        is_capacity_error = _is_payment_error(first_err) or _is_connection_error(first_err)
        if should_fallback and (is_auto or is_capacity_error):
            if _is_payment_error(first_err):
                reason = 'payment error'
                _mark_provider_unhealthy(_recoverable_pool_provider(resolved_provider, client) or resolved_provider)
            elif _is_rate_limit_error(first_err):
                reason = 'rate limit'
            else:
                reason = 'connection error'
            logger.info('Auxiliary %s: %s on %s (%s), trying fallback', task or 'call', reason, resolved_provider, first_err)
            fb_client, fb_model, fb_label = (None, None, '')
            if is_auto:
                fb_client, fb_model, fb_label = _try_payment_fallback(resolved_provider, task, reason=reason)
            else:
                fb_client, fb_model, fb_label = _try_configured_fallback_chain(task, resolved_provider or 'auto', reason=reason)
                if fb_client is None:
                    fb_client, fb_model, fb_label = _try_main_agent_model_fallback(resolved_provider, task, reason=reason)
            if fb_client is not None:
                fb_kwargs = _build_call_kwargs(fb_label, fb_model, messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, timeout=effective_timeout, extra_body=effective_extra_body, base_url=str(getattr(fb_client, 'base_url', '') or ''))
                return _validate_llm_response(fb_client.chat.completions.create(**fb_kwargs), task)
            logger.warning('Auxiliary %s: %s on %s and all fallbacks exhausted (fallback_chain + main agent model). Raising original error.', task or 'call', reason, resolved_provider)
        if _is_connection_error(first_err):
            try:
                _evict_cached_client_instance(client)
            except Exception:
                logger.debug('Auxiliary: cache eviction after connection error failed', exc_info=True)
        raise

def extract_content_or_reasoning(response) -> str:
    """Extract content from an LLM response, falling back to reasoning fields.

    Mirrors the main agent loop's behavior when a reasoning model (DeepSeek-R1,
    Qwen-QwQ, etc.) returns ``content=None`` with reasoning in structured fields.

    Resolution order:
      1. ``message.content`` — strip inline think/reasoning blocks, check for
         remaining non-whitespace text.
      2. ``message.reasoning`` / ``message.reasoning_content`` — direct
         structured reasoning fields (DeepSeek, Moonshot, NovitaAI, etc.).
      3. ``message.reasoning_details`` — OpenRouter unified array format.

    Returns the best available text, or ``""`` if nothing found.
    """
    import re
    msg = response.choices[0].message
    content = (msg.content or '').strip()
    if content:
        cleaned = re.sub('<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>.*?</(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
        if cleaned:
            return cleaned
    reasoning_parts: list[str] = []
    for field in ('reasoning', 'reasoning_content'):
        val = getattr(msg, field, None)
        if val and isinstance(val, str) and val.strip() and (val not in reasoning_parts):
            reasoning_parts.append(val.strip())
    details = getattr(msg, 'reasoning_details', None)
    if details and isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                summary = detail.get('summary') or detail.get('content') or detail.get('text')
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary.strip() if isinstance(summary, str) else str(summary))
    if reasoning_parts:
        return '\n\n'.join(reasoning_parts)
    return ''

def extract_tool_call_arguments(response, expected_name: str | None=None) -> dict[str, Any] | None:
    """Return arguments from the first matching function tool call.

    Auxiliary structured tasks use a single required function call.  Providers
    and SDK adapters expose the call either as objects or plain dictionaries,
    and ``function.arguments`` may already be decoded or remain a JSON string.
    Keep this normalization in one place so memory layers do not depend on a
    provider-specific response shape.
    """
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return None
    calls = getattr(message, 'tool_calls', None)
    if calls is None and isinstance(message, dict):
        calls = message.get('tool_calls')
    if not isinstance(calls, (list, tuple)):
        return None
    for call in calls:
        function = getattr(call, 'function', None)
        if function is None and isinstance(call, dict):
            function = call.get('function') or call
        name = getattr(function, 'name', None)
        arguments = getattr(function, 'arguments', None)
        if isinstance(function, dict):
            name = function.get('name', name)
            arguments = function.get('arguments', arguments)
        if expected_name is not None and str(name or '') != expected_name:
            continue
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                value = json.loads(arguments)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    return None

async def async_call_llm(task: str=None, *, provider: str=None, model: str=None, base_url: str=None, api_key: str=None, messages: list, temperature: float=None, max_tokens: int=None, tools: list=None, tool_choice: Any=None, timeout: float=None, extra_body: dict=None) -> Any:
    """Centralized asynchronous LLM call.

    Same as call_llm() but async. See call_llm() for full documentation.
    """
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(task, provider, model, base_url, api_key)
    effective_extra_body = _get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})
    if task == 'vision':
        effective_provider, client, final_model = resolve_vision_provider_client(provider=resolved_provider if resolved_provider != 'auto' else provider, model=resolved_model or model, base_url=resolved_base_url or base_url, api_key=resolved_api_key or api_key, async_mode=True)
        if client is None and resolved_provider != 'auto' and (not resolved_base_url):
            logger.warning('Vision provider %s unavailable, falling back to auto vision backends', resolved_provider)
            effective_provider, client, final_model = resolve_vision_provider_client(provider='auto', model=resolved_model, async_mode=True)
        if client is None:
            raise RuntimeError(f'No LLM provider configured for task={task} provider={resolved_provider}. Run: kylin-agent-runtime setup')
        resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(resolved_provider, resolved_model, async_mode=True, base_url=resolved_base_url, api_key=resolved_api_key, api_mode=resolved_api_mode)
        if client is None:
            _explicit = (resolved_provider or '').strip().lower()
            if _explicit and _explicit not in {'auto', 'openrouter', 'custom'}:
                raise RuntimeError(f"Provider '{_explicit}' is set in config.yaml but no API key was found. Set the {_explicit.upper()}_API_KEY environment variable, or switch to a different provider with `kylin-agent-runtime model`.")
            if not resolved_base_url:
                logger.info('Auxiliary %s: provider %s unavailable, trying auto-detection chain', task or 'call', resolved_provider)
                client, final_model = _get_cached_client('auto', async_mode=True)
        if client is None:
            raise RuntimeError(f'No LLM provider configured for task={task} provider={resolved_provider}. Run: kylin-agent-runtime setup')
    effective_timeout = timeout if timeout is not None else _get_task_timeout(task)
    _client_base = str(getattr(client, 'base_url', '') or '')
    kwargs = _build_call_kwargs(resolved_provider, final_model, messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, timeout=effective_timeout, extra_body=effective_extra_body, base_url=_client_base or resolved_base_url)
    if _is_anthropic_compat_endpoint(resolved_provider, _client_base):
        kwargs['messages'] = _convert_openai_images_to_anthropic(kwargs['messages'])
    try:
        return _validate_llm_response(await client.chat.completions.create(**kwargs), task)
    except Exception as first_err:
        if tools and tool_choice is not None and _is_thinking_tool_choice_error(first_err) and (not _thinking_is_disabled(kwargs.get('extra_body'))):
            retry_kwargs = dict(kwargs)
            retry_extra = dict(retry_kwargs.get('extra_body') or {})
            retry_extra['thinking'] = {'type': 'disabled'}
            retry_kwargs['extra_body'] = retry_extra
            logger.info('Auxiliary %s (async): provider rejected named tool_choice in thinking mode; retrying with thinking disabled', task or 'call')
            try:
                return _validate_llm_response(await client.chat.completions.create(**retry_kwargs), task)
            except Exception as retry_err:
                first_err = retry_err
                kwargs = retry_kwargs
        if 'temperature' in kwargs and _is_unsupported_temperature_error(first_err):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop('temperature', None)
            logger.info('Auxiliary %s (async): provider rejected temperature; retrying once without it', task or 'call')
            try:
                return _validate_llm_response(await client.chat.completions.create(**retry_kwargs), task)
            except Exception as retry_err:
                retry_err_str = str(retry_err)
                if not (_is_payment_error(retry_err) or _is_connection_error(retry_err) or _is_auth_error(retry_err) or ('max_tokens' in retry_err_str) or ('unsupported_parameter' in retry_err_str)):
                    raise
                first_err = retry_err
                kwargs = retry_kwargs
        err_str = str(first_err)
        _is_zai_param_error = '1210' in err_str and 'bigmodel' in str(getattr(client, 'base_url', ''))
        if max_tokens is not None and ('max_tokens' in err_str or 'unsupported_parameter' in err_str or _is_unsupported_parameter_error(first_err, 'max_tokens') or _is_zai_param_error):
            kwargs.pop('max_tokens', None)
            kwargs.pop('max_completion_tokens', None)
            try:
                return _validate_llm_response(await client.chat.completions.create(**kwargs), task)
            except Exception as retry_err:
                if not (_is_payment_error(retry_err) or _is_connection_error(retry_err) or _is_rate_limit_error(retry_err)):
                    raise
                first_err = retry_err
        client_is_nous = resolved_provider == 'nous' or base_url_host_matches(_client_base, 'inference-api.nousresearch.com')
        if _is_auth_error(first_err) and client_is_nous:
            refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(cache_provider=resolved_provider or 'nous', model=final_model, async_mode=True, base_url=resolved_base_url, api_key=resolved_api_key, api_mode=resolved_api_mode, is_vision=task == 'vision')
            if refreshed_client is not None:
                logger.info('Auxiliary %s (async): refreshed Nous runtime credentials after 401, retrying', task or 'call')
                if refreshed_model and refreshed_model != kwargs.get('model'):
                    kwargs['model'] = refreshed_model
                return _validate_llm_response(await refreshed_client.chat.completions.create(**kwargs), task)
        if _is_auth_error(first_err) and resolved_provider not in {'auto', '', None} and (not client_is_nous):
            if _refresh_provider_credentials(resolved_provider):
                logger.info('Auxiliary %s (async): refreshed %s credentials after auth error, retrying', task or 'call', resolved_provider)
                return await _retry_same_provider_async(task=task, resolved_provider=resolved_provider, resolved_model=resolved_model, resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key, resolved_api_mode=resolved_api_mode, final_model=final_model, messages=messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, effective_timeout=effective_timeout, effective_extra_body=effective_extra_body)
        pool_provider = _recoverable_pool_provider(resolved_provider, client)
        if pool_provider and (_is_auth_error(first_err) or _is_payment_error(first_err) or _is_rate_limit_error(first_err)):
            recovery_err = first_err
            if _is_rate_limit_error(first_err):
                try:
                    return _validate_llm_response(await client.chat.completions.create(**kwargs), task)
                except Exception as retry_err:
                    if not (_is_auth_error(retry_err) or _is_payment_error(retry_err) or _is_rate_limit_error(retry_err)):
                        raise
                    recovery_err = retry_err
            if _recover_provider_pool(pool_provider, recovery_err):
                logger.info('Auxiliary %s (async): recovered %s via credential-pool rotation after %s', task or 'call', pool_provider, type(recovery_err).__name__)
                return await _retry_same_provider_async(task=task, resolved_provider=resolved_provider, resolved_model=resolved_model, resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key, resolved_api_mode=resolved_api_mode, final_model=final_model, messages=messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, effective_timeout=effective_timeout, effective_extra_body=effective_extra_body)
        should_fallback = _is_payment_error(first_err) or _is_connection_error(first_err) or _is_rate_limit_error(first_err)
        is_auto = resolved_provider in {'auto', '', None}
        is_capacity_error = _is_payment_error(first_err) or _is_connection_error(first_err)
        if should_fallback and (is_auto or is_capacity_error):
            if _is_payment_error(first_err):
                reason = 'payment error'
                _mark_provider_unhealthy(_recoverable_pool_provider(resolved_provider, client) or resolved_provider)
            elif _is_rate_limit_error(first_err):
                reason = 'rate limit'
            else:
                reason = 'connection error'
            logger.info('Auxiliary %s (async): %s on %s (%s), trying fallback', task or 'call', reason, resolved_provider, first_err)
            fb_client, fb_model, fb_label = (None, None, '')
            if is_auto:
                fb_client, fb_model, fb_label = _try_payment_fallback(resolved_provider, task, reason=reason)
            else:
                fb_client, fb_model, fb_label = _try_configured_fallback_chain(task, resolved_provider or 'auto', reason=reason)
                if fb_client is None:
                    fb_client, fb_model, fb_label = _try_main_agent_model_fallback(resolved_provider, task, reason=reason)
            if fb_client is not None:
                fb_kwargs = _build_call_kwargs(fb_label, fb_model, messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, timeout=effective_timeout, extra_body=effective_extra_body, base_url=str(getattr(fb_client, 'base_url', '') or ''))
                async_fb, async_fb_model = _to_async_client(fb_client, fb_model or '', is_vision=task == 'vision')
                if async_fb_model and async_fb_model != fb_kwargs.get('model'):
                    fb_kwargs['model'] = async_fb_model
                return _validate_llm_response(await async_fb.chat.completions.create(**fb_kwargs), task)
            logger.warning('Auxiliary %s (async): %s on %s and all fallbacks exhausted (fallback_chain + main agent model). Raising original error.', task or 'call', reason, resolved_provider)
        if _is_connection_error(first_err):
            try:
                _evict_cached_client_instance(client)
            except Exception:
                logger.debug('Auxiliary (async): cache eviction after connection error failed', exc_info=True)
        raise

