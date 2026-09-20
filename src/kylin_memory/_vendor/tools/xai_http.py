from __future__ import annotations
import os

def get_env_value(name: str, default=None):
    """Read ``name`` from ``~/.kylin-agent-runtime/.env`` first, then ``os.environ``.

    Wraps :func:`kylin_agent_runtime_cli.config.get_env_value` so tests can patch
    ``tools.xai_http.get_env_value`` to inject dotenv-only secrets into the
    xAI credential resolver.
    """
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.config import get_env_value as _hermes_get_env_value
        value = _hermes_get_env_value(name)
        if value is not None:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)

