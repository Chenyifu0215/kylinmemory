from __future__ import annotations
import logging
import threading
from typing import Dict, List, Optional
from kylinmemory._vendor.agent.browser_provider import BrowserProvider
logger = logging.getLogger(__name__)
_providers: Dict[str, BrowserProvider] = {}
_lock = threading.Lock()

def register_provider(provider: BrowserProvider) -> None:
    """Register a cloud browser provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — makes hot-reload scenarios (tests, dev loops) behave
    predictably.
    """
    if not isinstance(provider, BrowserProvider):
        raise TypeError(f'register_provider() expects a BrowserProvider instance, got {type(provider).__name__}')
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Browser provider .name must be a non-empty string')
    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider
    if existing is not None:
        logger.debug("Browser provider '%s' re-registered (was %r)", name, type(existing).__name__)
    else:
        logger.debug("Registered browser provider '%s' (%s)", name, type(provider).__name__)

