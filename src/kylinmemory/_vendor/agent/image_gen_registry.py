from __future__ import annotations
import logging
import threading
from typing import Dict, List, Optional
from kylinmemory._vendor.agent.image_gen_provider import ImageGenProvider
logger = logging.getLogger(__name__)
_providers: Dict[str, ImageGenProvider] = {}
_lock = threading.Lock()

def register_provider(provider: ImageGenProvider) -> None:
    """Register an image generation provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — this makes hot-reload scenarios (tests, dev loops)
    behave predictably.
    """
    if not isinstance(provider, ImageGenProvider):
        raise TypeError(f'register_provider() expects an ImageGenProvider instance, got {type(provider).__name__}')
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Image gen provider .name must be a non-empty string')
    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider
    if existing is not None:
        logger.debug("Image gen provider '%s' re-registered (was %r)", name, type(existing).__name__)
    else:
        logger.debug("Registered image gen provider '%s' (%s)", name, type(provider).__name__)

