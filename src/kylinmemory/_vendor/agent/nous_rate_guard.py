from __future__ import annotations
import json
import os
import time
from typing import Any, Mapping, Optional
_STATE_SUBDIR = 'rate_limits'
_STATE_FILENAME = 'nous.json'

def _state_path() -> str:
    """Return the path to the Nous rate limit state file."""
    try:
        from kylinmemory._vendor.kylin_agent_runtime_constants import get_hermes_home
        base = get_hermes_home()
    except ImportError:
        base = os.path.join(os.path.expanduser('~'), '.kylin-agent-runtime')
    return os.path.join(base, _STATE_SUBDIR, _STATE_FILENAME)

def nous_rate_limit_remaining() -> Optional[float]:
    """Check if Nous Portal is currently rate-limited.

    Returns:
        Seconds remaining until reset, or None if not rate-limited.
    """
    path = _state_path()
    try:
        with open(path, encoding='utf-8') as f:
            state = json.load(f)
        reset_at = state.get('reset_at', 0)
        remaining = reset_at - time.time()
        if remaining > 0:
            return remaining
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None

