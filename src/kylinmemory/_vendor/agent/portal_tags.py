from __future__ import annotations
from typing import List

def _runtime_version() -> str:
    """Return the current Hermes release version, e.g. ``"0.13.0"``.

    Falls back to ``"unknown"`` if ``kylin_agent_runtime_cli`` cannot be imported (should
    never happen in a real install — guarded for defensive testing).
    """
    try:
        from kylinmemory._vendor.kylin_agent_runtime_cli import __version__
        return __version__
    except Exception:
        return 'unknown'

def kylin_agent_runtime_client_tag() -> str:
    """Return the ``client=...`` tag for Nous Portal requests.

    Format: ``client=hermes-client-v<MAJOR>.<MINOR>.<PATCH>``.
    """
    return f'client=kylin-agent-client-v{_runtime_version()}'

def nous_portal_tags() -> List[str]:
    """Return the canonical list of Nous Portal product tags.

    Always returns a fresh list so callers can mutate it freely
    (e.g. ``merged_extra.setdefault("tags", []).extend(nous_portal_tags())``).
    """
    return ['product=kylin-agent', kylin_agent_runtime_client_tag()]

