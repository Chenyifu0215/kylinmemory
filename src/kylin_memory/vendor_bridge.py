"""Bind the unchanged source router to an independent memory instance.

The source owns routing, fallbacks, pools and transports. This boundary only
supplies instance configuration and supports explicitly injected model clients.
"""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
import importlib.util
from types import SimpleNamespace
from pathlib import Path

_active_client = ContextVar('memory_injected_client', default=None)


def configure_source():
    from ._vendor.kylin_agent_runtime_cli import config as source
    if getattr(source, '_memory_context_bound', False):
        return
    original_load = source.load_config
    original_raw = source.read_raw_config
    def load():
        from .config import _context
        current = _context.get()
        return deepcopy(current[1]) if current is not None else original_load()
    def raw():
        from .config import _context
        current = _context.get()
        return deepcopy(current[1]) if current is not None else original_raw()
    source.load_config = load
    source.read_raw_config = raw
    source._memory_context_bound = True


class RuntimeRouter:
    """A private instance of the original router's module-level caches."""
    def __init__(self):
        configure_source()
        path = Path(__file__).parent / '_vendor/agent/auxiliary_client.py'
        spec = importlib.util.spec_from_file_location('kylin_memory._vendor.agent._memory_router', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        original_cached = module._get_cached_client
        original_auto = module._resolve_auto

        def injected(provider, model, base_url=None, api_key=None, api_mode=None, main_runtime=None):
            runtime = main_runtime or _active_client.get() or {}
            client = runtime.get('client')
            if client is None:
                return None
            if provider not in {None, '', 'auto', runtime.get('provider')}:
                return None
            if base_url and base_url.rstrip('/') != str(runtime.get('base_url') or getattr(client, 'base_url', '')).rstrip('/'):
                return None
            if api_key and api_key != runtime.get('api_key'):
                return None
            selected = model or runtime.get('model')
            mode = api_mode or runtime.get('api_mode')
            if mode == 'codex_responses' and hasattr(client, 'responses'):
                client = module.CodexAuxiliaryClient(client, selected)
            elif mode == 'anthropic_messages' and hasattr(client, 'messages'):
                client = module.AnthropicAuxiliaryClient(client, selected)
            return client, selected

        def cached(provider, model=None, async_mode=False, base_url=None, api_key=None,
                   api_mode=None, main_runtime=None, is_vision=False):
            supplied = injected(provider, model, base_url, api_key, api_mode, main_runtime)
            if supplied is not None and not async_mode:
                return supplied
            return original_cached(provider, model, async_mode, base_url, api_key,
                                   api_mode, main_runtime, is_vision)

        def auto(main_runtime=None):
            return injected('auto', None, main_runtime=main_runtime) or original_auto(main_runtime)
        module._get_cached_client = cached
        module._resolve_auto = auto

    def call(self, **kwargs):
        token = _active_client.set(kwargs.get('main_runtime'))
        try:
            return self.module.call_llm(**kwargs)
        finally:
            _active_client.reset(token)

    def close(self):
        self.module.shutdown_cached_clients()
