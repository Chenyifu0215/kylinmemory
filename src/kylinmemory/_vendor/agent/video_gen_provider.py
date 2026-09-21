from __future__ import annotations
import abc
from typing import Any, Dict, List, Optional, Tuple
COMMON_ASPECT_RATIOS: Tuple[str, ...] = ('16:9', '9:16', '1:1', '4:3', '3:4', '3:2', '2:3')
DEFAULT_ASPECT_RATIO = '16:9'
COMMON_RESOLUTIONS: Tuple[str, ...] = ('480p', '540p', '720p', '1080p')
DEFAULT_RESOLUTION = '720p'

class VideoGenProvider(abc.ABC):
    """Abstract base class for a video generation backend.

    Subclasses must implement :meth:`generate`. Everything else has sane
    defaults — override only what your provider needs.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in ``video_gen.provider`` config.

        Lowercase, no spaces. Examples: ``xai``, ``fal``, ``google``.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``kylin-agent-runtime tools``. Defaults to ``name.title()``."""
        return self.name.title()

    def is_available(self) -> bool:
        """Return True when this provider can service calls.

        Typically checks for a required API key and optional-dependency
        import. Default: True.
        """
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        """Return catalog entries for ``kylin-agent-runtime tools`` model picker.

        Each entry represents a **model family** that supports text-to-video
        and/or image-to-video routing internally::

            {
                "id": "veo-3.1",                       # required
                "display": "Veo 3.1",                  # optional; defaults to id
                "speed": "~60s",                       # optional
                "strengths": "...",                    # optional
                "price": "$0.20/s",                    # optional
                "modalities": ["text", "image"],       # optional, advisory
            }

        Default: empty list (provider has no user-selectable models).
        """
        return []

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for the ``kylin-agent-runtime tools`` picker."""
        return {'name': self.display_name, 'badge': '', 'tag': '', 'env_vars': []}

    def default_model(self) -> Optional[str]:
        """Return the default model id, or None if not applicable."""
        models = self.list_models()
        if models:
            return models[0].get('id')
        return None

    def capabilities(self) -> Dict[str, Any]:
        """Return what this provider supports.

        Returned dict (all keys optional)::

            {
                "modalities": ["text", "image"],      # which inputs the backend accepts
                "aspect_ratios": ["16:9", "9:16", ...],
                "resolutions": ["720p", "1080p"],
                "max_duration": 15,                   # seconds
                "min_duration": 1,
                "supports_audio": True,
                "supports_negative_prompt": True,
                "max_reference_images": 7,
            }

        Used by the tool layer for soft validation and by ``kylin-agent-runtime tools``
        for the picker. Default: text-only.
        """
        return {'modalities': ['text'], 'aspect_ratios': list(COMMON_ASPECT_RATIOS), 'resolutions': list(COMMON_RESOLUTIONS), 'max_duration': 10, 'min_duration': 1, 'supports_audio': False, 'supports_negative_prompt': False, 'max_reference_images': 0}

    @abc.abstractmethod
    def generate(self, prompt: str, *, model: Optional[str]=None, image_url: Optional[str]=None, reference_image_urls: Optional[List[str]]=None, duration: Optional[int]=None, aspect_ratio: str=DEFAULT_ASPECT_RATIO, resolution: str=DEFAULT_RESOLUTION, negative_prompt: Optional[str]=None, audio: Optional[bool]=None, seed: Optional[int]=None, **kwargs: Any) -> Dict[str, Any]:
        """Generate a video from a prompt (text-to-video) or animate an image
        (image-to-video).

        Routing: if ``image_url`` is provided, the provider should route to
        its image-to-video endpoint; otherwise text-to-video. The plugin
        is responsible for picking the right underlying endpoint within
        the user's chosen model family.

        Implementations should return the dict from :func:`success_response`
        or :func:`error_response`. ``kwargs`` may contain forward-compat
        parameters future versions of the schema will expose —
        implementations MUST ignore unknown keys (no TypeError).
        """

