from contextvars import ContextVar
from typing import Any
_UNSET: Any = object()
_SESSION_PLATFORM: ContextVar = ContextVar('HERMES_SESSION_PLATFORM', default=_UNSET)
_SESSION_CHAT_ID: ContextVar = ContextVar('HERMES_SESSION_CHAT_ID', default=_UNSET)
_SESSION_CHAT_NAME: ContextVar = ContextVar('HERMES_SESSION_CHAT_NAME', default=_UNSET)
_SESSION_THREAD_ID: ContextVar = ContextVar('HERMES_SESSION_THREAD_ID', default=_UNSET)
_SESSION_USER_ID: ContextVar = ContextVar('HERMES_SESSION_USER_ID', default=_UNSET)
_SESSION_USER_NAME: ContextVar = ContextVar('HERMES_SESSION_USER_NAME', default=_UNSET)
_SESSION_KEY: ContextVar = ContextVar('HERMES_SESSION_KEY', default=_UNSET)
_SESSION_ID: ContextVar = ContextVar('HERMES_SESSION_ID', default=_UNSET)
_SESSION_MESSAGE_ID: ContextVar = ContextVar('HERMES_SESSION_MESSAGE_ID', default=_UNSET)
_CRON_AUTO_DELIVER_PLATFORM: ContextVar = ContextVar('HERMES_CRON_AUTO_DELIVER_PLATFORM', default=_UNSET)
_CRON_AUTO_DELIVER_CHAT_ID: ContextVar = ContextVar('HERMES_CRON_AUTO_DELIVER_CHAT_ID', default=_UNSET)
_CRON_AUTO_DELIVER_THREAD_ID: ContextVar = ContextVar('HERMES_CRON_AUTO_DELIVER_THREAD_ID', default=_UNSET)
_VAR_MAP = {'HERMES_SESSION_PLATFORM': _SESSION_PLATFORM, 'HERMES_SESSION_CHAT_ID': _SESSION_CHAT_ID, 'HERMES_SESSION_CHAT_NAME': _SESSION_CHAT_NAME, 'HERMES_SESSION_THREAD_ID': _SESSION_THREAD_ID, 'HERMES_SESSION_USER_ID': _SESSION_USER_ID, 'HERMES_SESSION_USER_NAME': _SESSION_USER_NAME, 'HERMES_SESSION_KEY': _SESSION_KEY, 'HERMES_SESSION_ID': _SESSION_ID, 'HERMES_SESSION_MESSAGE_ID': _SESSION_MESSAGE_ID, 'HERMES_CRON_AUTO_DELIVER_PLATFORM': _CRON_AUTO_DELIVER_PLATFORM, 'HERMES_CRON_AUTO_DELIVER_CHAT_ID': _CRON_AUTO_DELIVER_CHAT_ID, 'HERMES_CRON_AUTO_DELIVER_THREAD_ID': _CRON_AUTO_DELIVER_THREAD_ID}

def get_session_env(name: str, default: str='') -> str:
    """Read a session context variable by its legacy ``HERMES_SESSION_*`` name.

    Drop-in replacement for ``os.getenv("HERMES_SESSION_*", default)``.

    Resolution order:
    1. Context variable (set by the gateway for concurrency-safe access).
       If the variable was explicitly set (even to ``""``) via
       ``set_session_vars`` or ``clear_session_vars``, that value is
       returned — **no fallback to os.environ**.
    2. ``os.environ`` (only when the context variable was never set in
       this context — i.e. CLI, cron scheduler, and test processes that
       don't use ``set_session_vars`` at all).
    3. *default*
    """
    import os
    var = _VAR_MAP.get(name)
    if var is not None:
        value = var.get()
        if value is not _UNSET:
            return value
    return os.getenv(name, default)

