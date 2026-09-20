"""Standalone entry points backed by the complete original auxiliary router."""
from __future__ import annotations

from .config import load_config, get_env_value
from .vendor_bridge import RuntimeRouter, configure_source
from ._vendor.agent import auxiliary_client as upstream

configure_source()
_default_router = None

# Source contracts remain available under the original public import surface.
extract_tool_call_arguments = upstream.extract_tool_call_arguments
_build_call_kwargs = upstream._build_call_kwargs
_resolve_task_provider_model = upstream._resolve_task_provider_model
CodexAuxiliaryClient = upstream.CodexAuxiliaryClient
_CodexCompletionsAdapter = upstream._CodexCompletionsAdapter


def resolve_runtime(runtime=None):
    cfg = load_config().get('model') or {}
    if isinstance(cfg, str):
        cfg = {'default': cfg}
    values = {**cfg, **(runtime or {})}
    if runtime and runtime.get('provider') and runtime['provider'] != cfg.get('provider'):
        for key in ('base_url', 'api_key', 'api_key_env', 'api_mode'):
            values[key] = runtime.get(key) or ''
    values['model'] = values.get('model') or values.get('default') or get_env_value('KYLIN_MEMORY_MODEL')
    # The source's live runtime carries a provider explicitly. Preserve the
    # standalone custom-endpoint shorthand when only URL/client was supplied.
    if not values.get('provider') or values['provider'] == 'auto':
        if values.get('base_url'):
            values['provider'] = 'custom'
    if values.get('api_key_env'):
        values['api_key'] = values.get('api_key') or get_env_value(values['api_key_env'])
    if values.get('base_url') and values.get('provider') == 'custom':
        values['api_key'] = values.get('api_key') or get_env_value('OPENAI_API_KEY')
    return values


def _router(runtime):
    global _default_router
    if runtime.get('_router') is not None:
        return runtime['_router']
    if _default_router is None:
        _default_router = RuntimeRouter()
    return _default_router


def _get_cached_client(provider='auto', model=None, *, base_url=None, api_key=None, api_mode=None, main_runtime=None):
    runtime = resolve_runtime(main_runtime)
    return _router(runtime).module._get_cached_client(provider, model, base_url=base_url,
        api_key=api_key, api_mode=api_mode, main_runtime=runtime)


def call_llm(task=None, *, provider=None, model=None, base_url=None, api_key=None,
             api_mode=None, main_runtime=None, messages, temperature=None,
             max_tokens=None, tools=None, tool_choice=None, timeout=None, extra_body=None):
    runtime = resolve_runtime(main_runtime)
    callback = runtime.pop('call_llm', None)
    kwargs = dict(task=task, provider=provider, model=model, base_url=base_url, api_key=api_key,
        api_mode=api_mode, main_runtime=runtime, messages=messages, temperature=temperature,
        max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, timeout=timeout, extra_body=extra_body)
    if callable(callback):
        return callback(**kwargs)
    return _router(runtime).call(**kwargs)


def __getattr__(name):
    return getattr(upstream, name)
