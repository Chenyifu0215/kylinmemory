"""Diagnostics for the inputs and outputs of layered-memory models.

The L1, L2, and L3 extractors use different provider adapters (and L3 can use
either Chat Completions or Responses).  Keeping the diagnostic event in one
small helper makes the wire input comparable across all three layers.

The events deliberately do not accept or record credentials.  They record the
actual prompt/tool contract and raw provider response.  Known secrets are
always redacted before an event is handed to the logging framework.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)


def _to_plain(value: Any, *, _depth: int = 0) -> Any:
    """Convert SDK/Pydantic responses to JSON-compatible values."""

    if _depth > 30:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {
            str(key): _to_plain(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item, _depth=_depth + 1) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_plain(model_dump(mode="json"), _depth=_depth + 1)
        except TypeError:
            try:
                return _to_plain(model_dump(), _depth=_depth + 1)
            except Exception:
                pass
        except Exception:
            pass

    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        try:
            return _to_plain(as_dict(), _depth=_depth + 1)
        except Exception:
            pass
    try:
        attributes = vars(value)
    except (TypeError, ValueError):
        attributes = None
    if isinstance(attributes, dict):
        return _to_plain(
            {
                key: item
                for key, item in attributes.items()
                if not str(key).startswith("_")
            },
            _depth=_depth + 1,
        )
    return repr(value)


def _log_event(event_name: str, event: Mapping[str, Any]) -> None:
    """Serialize, force-redact, and emit one best-effort diagnostic event."""

    try:
        encoded = json.dumps(
            _to_plain(event),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        try:
            from kylin_memory.redact import redact_sensitive_text

            encoded = redact_sensitive_text(encoded, force=True)
        except Exception:
            pass
        logger.info("%s %s", event_name, encoded)
    except Exception:
        # Diagnostics must never change the extraction outcome.
        logger.debug("Could not log layered-memory model event", exc_info=True)


def log_memory_llm_input(
    layer: str,
    *,
    task: str,
    model: str | None = None,
    api_mode: str | None = None,
    messages: Sequence[Mapping[str, Any]] | None = None,
    instructions: str | None = None,
    input_payload: Any = None,
    tools: Any = None,
    tool_choice: Any = None,
    session_id: str | None = None,
    **metadata: Any,
) -> None:
    """Write the complete layered-memory model input as one searchable event.

    ``messages`` is used for Chat Completions requests.  Responses requests
    expose the equivalent fields as ``instructions`` and ``input``; both are
    retained under their wire names in the event.  ``default=str`` keeps this
    diagnostic best-effort for provider-specific objects without ever letting
    logging interrupt extraction.
    """

    event: dict[str, Any] = {
        "layer": str(layer).upper(),
        "task": task,
        "model": model,
        "api_mode": api_mode,
    }
    if session_id:
        event["session_id"] = session_id
    if messages is not None:
        event["messages"] = list(messages)
    if instructions is not None:
        event["instructions"] = instructions
    if input_payload is not None:
        event["input"] = input_payload
    if tools is not None:
        event["tools"] = tools
    if tool_choice is not None:
        event["tool_choice"] = tool_choice
    event.update({key: value for key, value in metadata.items() if value is not None})

    _log_event("MEMORY_LLM_INPUT", event)


def log_memory_llm_output(
    layer: str,
    *,
    task: str,
    response: Any,
    model: str | None = None,
    api_mode: str | None = None,
    session_id: str | None = None,
    **metadata: Any,
) -> None:
    """Write the complete raw provider response as one searchable event."""

    event: dict[str, Any] = {
        "layer": str(layer).upper(),
        "task": task,
        "model": model,
        "api_mode": api_mode,
        "response": response,
    }
    if session_id:
        event["session_id"] = session_id
    event.update({key: value for key, value in metadata.items() if value is not None})
    _log_event("MEMORY_LLM_OUTPUT", event)


def log_memory_retrieval(
    layer: str,
    *,
    query: Any = "",
    results: Any = None,
    context: Any = None,
    provider: str | None = None,
    session_id: str | None = None,
    task: str = "recall",
    **metadata: Any,
) -> None:
    """Write the result of a layered-memory recall as one searchable event.

    This is intentionally separate from ``MEMORY_LLM_INPUT``.  L1/L2 recall
    is normally local and may not involve an auxiliary model at all, while the
    caller still needs to inspect exactly what was selected for the turn.  The
    event keeps both the structured records (``results``) and the rendered
    context (``context``) so the persisted data can be compared with what was
    injected into the agent prompt.
    """

    event: dict[str, Any] = {
        "layer": str(layer).upper(),
        "task": task,
        "query": query,
    }
    if provider:
        event["provider"] = provider
    if session_id:
        event["session_id"] = session_id
    if results is not None:
        event["results"] = results
        try:
            event["result_count"] = len(results) if not isinstance(results, str) else 1
        except TypeError:
            pass
    if context is not None:
        event["context"] = context
    event.update({key: value for key, value in metadata.items() if value is not None})
    _log_event("MEMORY_RETRIEVAL", event)


def log_memory_user_input(
    content: Any,
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    platform: str | None = None,
    source: str = "conversation",
    **metadata: Any,
) -> None:
    """Write the user input that was used to drive memory retrieval.

    The value is passed through the same best-effort secret redaction as all
    other ``memory_debug`` events.  Keeping this as a dedicated event makes it
    possible to correlate a query with L1/L2 results without relying on a
    truncated conversation preview.
    """

    event: dict[str, Any] = {
        "user_input": content,
        "source": source,
    }
    if session_id:
        event["session_id"] = session_id
    if task_id:
        event["task_id"] = task_id
    if platform:
        event["platform"] = platform
    event.update({key: value for key, value in metadata.items() if value is not None})
    _log_event("MEMORY_USER_INPUT", event)


__all__ = [
    "log_memory_llm_input",
    "log_memory_llm_output",
    "log_memory_retrieval",
    "log_memory_user_input",
]
