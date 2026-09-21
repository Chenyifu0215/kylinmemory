"""Session-scoped request priority propagation.

Some OpenAI-compatible routes accept a body-level ``extra_body.priority``.
The active agent stores that in ``request_overrides``; this module makes the
value available to side LLM calls in the same session without using globals
that would bleed across concurrent gateway requests.
"""
from __future__ import annotations
from contextvars import ContextVar, Token
from typing import Any, Mapping, MutableMapping, Optional
_CURRENT_REQUEST_PRIORITY: ContextVar[Optional[int]] = ContextVar('hermes_current_request_priority', default=None)

def coerce_request_priority(value: Any) -> Optional[int]:
    """Return a positive integer priority, or ``None`` when unset/zero/invalid."""
    if value is None or value is False:
        return None
    if value is True:
        return 1
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return None
    return priority if priority > 0 else None

def priority_from_overrides(overrides: Any) -> Optional[int]:
    """Extract ``extra_body.priority`` from an agent ``request_overrides`` dict."""
    if not isinstance(overrides, Mapping):
        return None
    extra_body = overrides.get('extra_body')
    if not isinstance(extra_body, Mapping):
        return None
    return coerce_request_priority(extra_body.get('priority'))

def current_request_priority() -> Optional[int]:
    """Return the current session priority, if one is bound."""
    return _CURRENT_REQUEST_PRIORITY.get()

def set_current_request_priority(priority: Any) -> Token[Optional[int]]:
    """Bind the current session priority for the active context."""
    return _CURRENT_REQUEST_PRIORITY.set(coerce_request_priority(priority))

def set_current_request_priority_from_overrides(overrides: Any) -> Token[Optional[int]]:
    """Bind priority extracted from ``request_overrides``."""
    return set_current_request_priority(priority_from_overrides(overrides))

def reset_current_request_priority(token: Token[Optional[int]]) -> None:
    """Restore the previous request-priority context."""
    _CURRENT_REQUEST_PRIORITY.reset(token)

def merge_priority_into_extra_body(extra_body: Optional[Mapping[str, Any]], *, priority: Any=None) -> dict[str, Any]:
    """Return a copy of ``extra_body`` with session priority applied when set."""
    merged = dict(extra_body or {})
    effective = coerce_request_priority(priority)
    if effective is None:
        effective = current_request_priority()
    if effective is not None:
        merged['priority'] = effective
    return merged

def merge_priority_into_kwargs(kwargs: MutableMapping[str, Any], *, priority: Any=None) -> MutableMapping[str, Any]:
    """Mutate OpenAI-compatible request kwargs to include ``extra_body.priority``."""
    effective = coerce_request_priority(priority)
    if effective is None:
        effective = current_request_priority()
    if effective is None:
        return kwargs
    extra_body = kwargs.get('extra_body')
    if not isinstance(extra_body, Mapping):
        extra_body = {}
    kwargs['extra_body'] = merge_priority_into_extra_body(extra_body, priority=effective)
    return kwargs

