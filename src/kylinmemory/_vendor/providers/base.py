"""Provider profile base class.

A ProviderProfile declares everything about an inference provider in one place:
auth, endpoints, client quirks, request-time quirks. The transport reads this
instead of receiving 20+ boolean flags.

Provider profiles are DECLARATIVE — they describe the provider's behavior.
They do NOT own client construction, credential rotation, or streaming.
Those stay on AIAgent.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
logger = logging.getLogger(__name__)
OMIT_TEMPERATURE = object()

def _profile_user_agent() -> str:
    """Return a ``hermes-cli/<version>`` UA string, with a stable fallback.

    Used by ``ProviderProfile.fetch_models`` so the catalog probe is not
    served the default ``Python-urllib/<ver>`` UA — some providers
    (OpenCode Zen, etc.) sit behind a WAF that returns 403 for that.
    """
    try:
        from kylinmemory._vendor.kylin_agent_runtime_cli import __version__ as _ver
        return f'hermes-cli/{_ver}'
    except Exception:
        return 'hermes-cli'

@dataclass
class ProviderProfile:
    """Base provider profile — subclass or instantiate with overrides."""
    name: str
    api_mode: str = 'chat_completions'
    aliases: tuple = ()
    display_name: str = ''
    description: str = ''
    signup_url: str = ''
    env_vars: tuple = ()
    base_url: str = ''
    models_url: str = ''
    auth_type: str = 'api_key'
    supports_health_check: bool = True
    fallback_models: tuple = ()
    hostname: str = ''
    default_headers: dict[str, str] = field(default_factory=dict)
    fixed_temperature: Any = None
    default_max_tokens: int | None = None
    default_aux_model: str = ''

    def get_hostname(self) -> str:
        """Return the provider's base hostname for URL-based detection.

        Uses self.hostname if set explicitly, otherwise derives it from base_url.
        e.g. 'https://api.gmi-serving.com/v1' → 'api.gmi-serving.com'
        """
        if self.hostname:
            return self.hostname
        if self.base_url:
            from urllib.parse import urlparse
            return urlparse(self.base_url).hostname or ''
        return ''

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing.

        Called AFTER codex field sanitization, BEFORE developer role swap.
        Default: pass-through.
        """
        return messages

    def build_extra_body(self, *, session_id: str | None=None, **context: Any) -> dict[str, Any]:
        """Provider-specific extra_body fields.

        Merged into the API kwargs extra_body. Default: empty dict.
        """
        return {}

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None=None, **context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Provider-specific kwargs split between extra_body and top-level api_kwargs.

        Returns (extra_body_additions, top_level_kwargs).
        The transport merges extra_body_additions into extra_body, and
        top_level_kwargs directly into api_kwargs.

        This split exists because some providers put reasoning config in
        extra_body (OpenRouter: extra_body.reasoning) while others put it
        as top-level api_kwargs (Kimi: api_kwargs.reasoning_effort).

        Default: ({}, {}).
        """
        return ({}, {})

    def fetch_models(self, *, api_key: str | None=None, timeout: float=8.0) -> list[str] | None:
        """Fetch the live model list from the provider's models endpoint.

        Returns a list of model ID strings, or None if the fetch failed or
        the provider does not support live model listing.

        Resolution order for the endpoint URL:
          1. self.models_url  (explicit override — use when the models
             endpoint differs from the inference base URL, e.g. OpenRouter
             exposes a public catalog at /api/v1/models while inference is
             at /api/v1)
          2. self.base_url + "/models"  (standard OpenAI-compat fallback)

        The default implementation sends Bearer auth when api_key is given
        and forwards self.default_headers. Override to customise auth, path,
        response shape, or to return None for providers with no REST catalog.

        Callers must always fall back to the static _PROVIDER_MODELS list
        when this returns None.
        """
        url = (self.models_url or '').strip()
        if not url:
            if not self.base_url:
                return None
            url = self.base_url.rstrip('/') + '/models'
        import json
        import urllib.request
        req = urllib.request.Request(url)
        if api_key:
            req.add_header('Authorization', f'Bearer {api_key}')
        req.add_header('Accept', 'application/json')
        req.add_header('User-Agent', _profile_user_agent())
        for k, v in self.default_headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get('data', [])
            return [m['id'] for m in items if isinstance(m, dict) and 'id' in m]
        except Exception as exc:
            logger.debug('fetch_models(%s): %s', self.name, exc)
            return None

