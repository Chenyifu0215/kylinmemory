from __future__ import annotations
import logging
import threading
from typing import Dict, List, Optional
from kylinmemory._vendor.agent.video_gen_provider import VideoGenProvider
logger = logging.getLogger(__name__)
_providers: Dict[str, VideoGenProvider] = {}
_lock = threading.Lock()

def register_provider(provider: VideoGenProvider) -> None:
    """Register a video generation provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — this makes hot-reload scenarios (tests, dev loops)
    behave predictably.
    """
    if not isinstance(provider, VideoGenProvider):
        raise TypeError(f'register_provider() expects a VideoGenProvider instance, got {type(provider).__name__}')
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Video gen provider .name must be a non-empty string')
    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider
    if existing is not None:
        logger.debug("Video gen provider '%s' re-registered (was %r)", name, type(existing).__name__)
    else:
        logger.debug("Registered video gen provider '%s' (%s)", name, type(provider).__name__)

