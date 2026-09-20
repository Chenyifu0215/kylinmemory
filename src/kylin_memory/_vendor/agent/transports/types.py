from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolCall:
    """A normalized tool call from any provider.

    ``id`` is the protocol's canonical identifier — what gets used in
    ``tool_call_id`` / ``tool_use_id`` when constructing tool result
    messages.  May be ``None`` when the provider omits it; the agent
    fills it via ``_deterministic_call_id()`` before storing in history.

    ``provider_data`` carries per-tool-call protocol metadata that only
    protocol-aware code reads:

    * Codex: ``{"call_id": "call_XXX", "response_item_id": "fc_XXX"}``
    * Gemini: ``{"extra_content": {"google": {"thought_signature": "..."}}}``
    * Others: ``None``
    """
    id: str | None
    name: str
    arguments: str
    provider_data: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def type(self) -> str:
        return 'function'

    @property
    def function(self) -> ToolCall:
        """Return self so tc.function.name / tc.function.arguments work."""
        return self

    @property
    def call_id(self) -> str | None:
        """Codex call_id from provider_data, accessed via getattr by _build_assistant_message."""
        return (self.provider_data or {}).get('call_id')

    @property
    def response_item_id(self) -> str | None:
        """Codex response_item_id from provider_data."""
        return (self.provider_data or {}).get('response_item_id')

    @property
    def extra_content(self) -> dict[str, Any] | None:
        """Gemini extra_content (thought_signature) from provider_data.

        Gemini 3 thinking models attach ``extra_content`` with a
        ``thought_signature`` to each tool call.  This signature must be
        replayed on subsequent API calls — without it the API rejects the
        request with HTTP 400.  The chat_completions transport stores this
        in ``provider_data["extra_content"]``; this property exposes it so
        ``_build_assistant_message`` can ``getattr(tc, "extra_content")``
        uniformly.
        """
        return (self.provider_data or {}).get('extra_content')

@dataclass
class Usage:
    """Token usage from an API response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

@dataclass
class NormalizedResponse:
    """Normalized API response from any provider.

    Shared fields are truly cross-provider — every caller can rely on
    them without branching on api_mode.  Protocol-specific state goes in
    ``provider_data`` so that only protocol-aware code paths read it.

    Response-level ``provider_data`` examples:

    * Anthropic: ``{"reasoning_details": [...]}``
    * Codex: ``{"codex_reasoning_items": [...], "codex_message_items": [...]}``
    * Others: ``None``
    """
    content: str | None
    tool_calls: list[ToolCall] | None
    finish_reason: str
    reasoning: str | None = None
    usage: Usage | None = None
    provider_data: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def reasoning_content(self) -> str | None:
        pd = self.provider_data or {}
        return pd.get('reasoning_content')

    @property
    def reasoning_details(self):
        pd = self.provider_data or {}
        return pd.get('reasoning_details')

    @property
    def codex_reasoning_items(self):
        pd = self.provider_data or {}
        return pd.get('codex_reasoning_items')

    @property
    def codex_message_items(self):
        pd = self.provider_data or {}
        return pd.get('codex_message_items')

def build_tool_call(id: str | None, name: str, arguments: Any, **provider_fields: Any) -> ToolCall:
    """Build a ``ToolCall``, auto-serialising *arguments* if it's a dict.

    Any extra keyword arguments are collected into ``provider_data``.
    """
    args_str = json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
    pd = dict(provider_fields) if provider_fields else None
    return ToolCall(id=id, name=name, arguments=args_str, provider_data=pd)

def map_finish_reason(reason: str | None, mapping: dict[str, str]) -> str:
    """Translate a provider-specific stop reason to the normalised set.

    Falls back to ``"stop"`` for unknown or ``None`` reasons.
    """
    if reason is None:
        return 'stop'
    return mapping.get(reason, 'stop')

