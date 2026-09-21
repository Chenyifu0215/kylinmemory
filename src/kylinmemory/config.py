"""Standalone configuration; runtime contexts never change process environment."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from copy import deepcopy
import os
from pathlib import Path
import threading

from ._vendor.kylin_agent_runtime_cli.config import DEFAULT_CONFIG as _SOURCE_DEFAULTS
DEFAULT_CONFIG = deepcopy(_SOURCE_DEFAULTS)
_context = ContextVar('kylinmemory_context', default=None)


def get_hermes_home() -> Path:
    context = _context.get()
    if context is not None:
        return context[0]
    return Path(os.environ.get('KYLINMEMORY_HOME') or os.environ.get('HERMES_HOME') or Path.home() / '.kylinmemory').expanduser()


def get_config_path() -> Path:
    return get_hermes_home() / 'config.yaml'


def merge_config(base, overrides):
    result = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict:
    context = _context.get()
    if path is None and context is not None:
        return deepcopy(context[1])
    config_path = Path(path) if path is not None else get_config_path()
    values = {}
    if config_path.exists():
        import yaml
        from ._vendor.kylin_agent_runtime_cli.config import _warn_config_parse_failure
        try:
            values = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
            if not isinstance(values, dict):
                raise ValueError('config.yaml must contain a mapping')
        except Exception as exc:
            _warn_config_parse_failure(config_path, exc)
            values = {}
    return normalize_config(values)


def normalize_config(values, *, base=None):
    from ._vendor.kylin_agent_runtime_cli.config import (
        _deep_merge, _normalize_root_model_keys, _normalize_max_turns_config, _expand_env_vars,
    )
    values = deepcopy(values)
    if 'max_turns' in values:
        agent_config = dict(values.get('agent') or {})
        if agent_config.get('max_turns') is None:
            agent_config['max_turns'] = values['max_turns']
        values['agent'] = agent_config
        values.pop('max_turns')
    merged = _deep_merge(deepcopy(DEFAULT_CONFIG if base is None else base), values)
    return _expand_env_vars(_normalize_root_model_keys(_normalize_max_turns_config(merged)))


def get_env_value(name: str, default: str = '') -> str:
    from ._vendor.kylin_agent_runtime_cli.config import get_env_value as source_get
    from ._vendor.kylin_agent_runtime_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(get_hermes_home())
    try:
        value = source_get(name)
        return default if value is None else value
    finally:
        reset_hermes_home_override(token)


@contextmanager
def runtime_context(root: Path, config: dict):
    from ._vendor.kylin_agent_runtime_constants import set_hermes_home_override, reset_hermes_home_override
    source_token = set_hermes_home_override(root)
    token = _context.set((Path(root), config))
    try:
        yield
    finally:
        _context.reset(token)
        reset_hermes_home_override(source_token)


def context_timer(interval, function, args=None, kwargs=None):
    """Carry per-instance configuration into upstream scheduler callbacks."""
    context = copy_context()
    return threading.Timer(interval, context.run, args=(function, *(args or ())), kwargs=kwargs or {})
