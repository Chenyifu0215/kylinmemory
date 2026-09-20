from __future__ import annotations
import logging
import threading
from typing import Dict, List, Optional
from kylin_memory._vendor.agent.web_search_provider import WebSearchProvider
logger = logging.getLogger(__name__)
_providers: Dict[str, WebSearchProvider] = {}
_lock = threading.Lock()

def register_provider(provider: WebSearchProvider) -> None:
    """Register a web search/extract provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — makes hot-reload scenarios (tests, dev loops) behave
    predictably.
    """
    if not isinstance(provider, WebSearchProvider):
        raise TypeError(f'register_provider() expects a WebSearchProvider instance, got {type(provider).__name__}')
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Web provider .name must be a non-empty string')
    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider
    if existing is not None:
        logger.debug("Web provider '%s' re-registered (was %r)", name, type(existing).__name__)
    else:
        logger.debug("Registered web provider '%s' (%s)", name, type(provider).__name__)

