"""Session transcript preprocessing and structured L1 memory extraction."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
import re
import unicodedata
from typing import Any, Protocol, Sequence

try:
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - pydantic is a core dependency
    BaseModel = None


if BaseModel is not None:
    class _AtomModel(BaseModel):
        # The LLM-facing schema is the same contract that is persisted by
        # AtomStore. Rejecting unknown keys keeps the boundary exact.
        model_config = ConfigDict(extra="forbid")

    class AtomCandidate(_AtomModel):
        content: str
        type: str
        priority: int
        scene_name: str
        source_message_ids: list[int]
        metadata: dict[str, Any]
        # The extractor no longer asks the model for timestamps: every
        # persisted record is anchored at writer time by ``_apply_decisions``,
        # so a model-supplied value was always discarded.  The field stays for
        # adapters that build candidates from another source.
        timestamps: list[str] = Field(default_factory=list)

    class ExtractionBatch(_AtomModel):
        memories: list[AtomCandidate] = Field(default_factory=list)
else:  # pragma: no cover
    class AtomCandidate: pass
    class ExtractionBatch: pass

logger = logging.getLogger(__name__)

L1_EXTRACTION_TOOL_NAME = "l1_memory_extraction"

# Per-type ``metadata`` keys and ``priority`` floors are enforced by
# ``Atom.from_mapping``; a memory that violates either is dropped entirely.
# JSON Schema cannot express "keys allowed depend on the sibling type value",
# so the rules are restated in the property descriptions, which is where a
# tool-calling model reads them most reliably.
_CHAT_METADATA_RULE = (
    "Per-type allowlist; any other key discards the whole memory. "
    "episodic: activity_start_time, activity_end_time (ISO 8601), only when "
    "derivable from the message timestamps. persona and instruction: always {}."
)
_WORK_METADATA_RULE = (
    "Per-type allowlist; any other key discards the whole memory. Omit keys "
    "you cannot determine; {} when nothing applies. "
    "work_fact: work_object, status, activity_start_time, activity_end_time. "
    "work_task: owner, deadline (ISO 8601), "
    "status (todo|doing|done|blocked|deferred|cancelled). "
    "work_method: scope (project|team|module|agent|workflow), method_type "
    "(sop|principle|constraint|anti_pattern|heuristic|evaluation_criterion). "
    "work_artifact: artifact_type "
    "(doc|pr|issue|repo|branch|design|report|prompt|dataset|meeting_note), "
    "artifact_ref."
)
_CHAT_PRIORITY_RULE = (
    "Importance score. Minimum per type, below which the memory must not be "
    "emitted at all: persona 50, episodic 60, instruction 70. Use 80-100 for "
    "health/taboo/core traits and important events, 90-100 for core behaviour "
    "rules. -1 is reserved for an absolute, never-violable instruction."
)
_WORK_PRIORITY_RULE = (
    "Importance score, minimum 70 for every work type; a memory scoring below "
    "70 must not be emitted at all. Use 90-100 for key decisions, core "
    "requirements, long-term constraints, blocking tasks, cross-task methods "
    "and critical assets."
)
_L1_MEMORY_PROPERTIES = {
    "content": {
        "type": "string",
        "description": (
            "One self-contained memory statement that stays understandable "
            "outside this conversation, with no pronouns referring back to it."
        ),
    },
    "type": {
        "type": "string",
        "enum": [
            "persona", "episodic", "instruction", "work_fact",
            "work_task", "work_method", "work_artifact",
        ],
    },
    "priority": {"type": "integer", "minimum": -1, "maximum": 100},
    "source_message_ids": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 1,
        "description": (
            "Evidence IDs from the new messages only. Must include at least "
            "one user message ID; assistant IDs may be added for context. A "
            "memory with no user evidence is discarded."
        ),
    },
    "metadata": {"type": "object"},
}
_L1_MEMORY_REQUIRED = [
    "content", "type", "priority", "source_message_ids", "metadata",
]


def _l1_extraction_tool(mode: str) -> dict[str, Any]:
    """Build the L1 schema with only the memory types valid for this mode."""
    is_code = str(mode).lower() == "code"
    allowed_types = (
        ["work_fact", "work_task", "work_method", "work_artifact"]
        if is_code
        else ["persona", "episodic", "instruction"]
    )
    memory_properties = dict(_L1_MEMORY_PROPERTIES)
    memory_properties["type"] = {"type": "string", "enum": allowed_types}
    memory_properties["priority"] = {
        **_L1_MEMORY_PROPERTIES["priority"],
        "description": _WORK_PRIORITY_RULE if is_code else _CHAT_PRIORITY_RULE,
    }
    memory_properties["metadata"] = {
        "type": "object",
        "description": _WORK_METADATA_RULE if is_code else _CHAT_METADATA_RULE,
    }
    return {
        "type": "function",
        "function": {
            "name": L1_EXTRACTION_TOOL_NAME,
            "description": (
                "Submit every scene segmented out of the new messages, each "
                "with the long-term memories extracted from it. Call once."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "scenes": {
                        "type": "array",
                        "description": (
                            "One entry per scene, in message order. Together "
                            "the scenes must cover every new message exactly "
                            "once."
                        ),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "scene_name": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": (
                                        "Single sentence naming the scene, in "
                                        "the messages' dominant language. "
                                        "Reuse the previous scene name "
                                        "verbatim when the scene continues."
                                    ),
                                },
                                "message_ids": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 1,
                                    "description": (
                                        "IDs of the new messages belonging to "
                                        "this scene."
                                    ),
                                },
                                "memories": {
                                    "type": "array",
                                    "description": (
                                        "Memories extracted from this scene; "
                                        "empty when nothing is worth keeping."
                                    ),
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": memory_properties,
                                        "required": _L1_MEMORY_REQUIRED,
                                    },
                                },
                            },
                            "required": ["scene_name", "message_ids", "memories"],
                        },
                    },
                },
                "required": ["scenes"],
            },
        },
    }


L1_EXTRACTION_TOOL = _l1_extraction_tool("chat")

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
_MEMORY_CONTEXT_RE = re.compile(
    r"<\s*memory-context\s*>.*?</\s*memory-context\s*>", re.IGNORECASE | re.DOTALL
)
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]{256,}$")
_EXTRACTION_BATCH_LIMIT = 12
_DIAGNOSTIC_SCHEMA_FIELDS = {
    "memories",
    "content",
    "type",
    "priority",
    "scene_name",
    "source_message_ids",
    "metadata",
}


def _json_array_text(text: str) -> str:
    """Extract the first JSON array while tolerating harmless model framing."""
    stripped = _FENCE_RE.sub("", str(text or "")).strip()
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = stripped.find("[", cursor)
        if start < 0:
            return ""
        try:
            value, _ = decoder.raw_decode(stripped[start:])
        except (json.JSONDecodeError, ValueError):
            cursor = start + 1
            continue
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        cursor = start + 1


class SemanticExtractionResponseError(ValueError):
    """Structured-output failure with a payload-free diagnostic code."""

    persistent_error_code = "ValueError"

    def __init__(self, diagnostic_code: str):
        super().__init__("semantic extractor returned invalid structured output")
        self.diagnostic_code = diagnostic_code


def _validation_diagnostic(exc: Exception) -> str:
    """Summarize Pydantic failures without retaining model output or messages."""
    errors_method = getattr(exc, "errors", None)
    if not callable(errors_method):
        return "structured_output_invalid"
    try:
        errors = errors_method(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    except TypeError:
        # Older Pydantic releases do not support all redaction arguments. Do
        # not fall back to ``errors()`` because that can include the rejected
        # model payload.
        return "structured_output_invalid"
    except Exception:
        return "structured_output_invalid"
    if not isinstance(errors, list) or not errors:
        return "structured_output_invalid"

    parts: list[str] = []
    for error in errors[:4]:
        if not isinstance(error, dict):
            continue
        error_type = re.sub(
            r"[^a-zA-Z0-9_.-]",
            "_",
            str(error.get("type") or "invalid"),
        )[:80]
        location = error.get("loc")
        path_parts: list[str] = []
        if isinstance(location, (list, tuple)):
            for item in location[:6]:
                if isinstance(item, int):
                    path_parts.append("[]")
                else:
                    field = str(item)
                    path_parts.append(
                        field if field in _DIAGNOSTIC_SCHEMA_FIELDS else "field"
                    )
        path = ".".join(part for part in path_parts if part)
        parts.append(f"{error_type}@{path}" if path else error_type)
    if not parts:
        return "structured_output_invalid"
    suffix = "+more" if len(errors) > len(parts) else ""
    return "schema:" + ",".join(parts) + suffix


def _parse_extraction_batch(text: str) -> ExtractionBatch:
    """Parse one JSON object while tolerating harmless model framing text."""
    stripped = _FENCE_RE.sub("", text).strip()
    try:
        return ExtractionBatch.model_validate_json(stripped)
    except Exception as direct_exc:
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        cursor = 0
        while True:
            start = stripped.find("{", cursor)
            if start < 0:
                break
            try:
                value, end = decoder.raw_decode(stripped[start:])
            except (json.JSONDecodeError, ValueError):
                cursor = start + 1
                continue
            if isinstance(value, dict) and "memories" in value:
                candidates.append(value)
            # Treat braces inside a decoded value as part of that document,
            # not as additional top-level JSON candidates.
            cursor = start + max(end, 1)

        if len(candidates) > 1:
            raise SemanticExtractionResponseError("multiple_json_objects")
        if not candidates:
            raise SemanticExtractionResponseError(
                _validation_diagnostic(direct_exc)
            ) from direct_exc
        try:
            return ExtractionBatch.model_validate(candidates[0])
        except Exception as exc:
            raise SemanticExtractionResponseError(
                _validation_diagnostic(exc)
            ) from exc


def _parse_scene_extraction(text: str | list[Any]) -> list[AtomCandidate]:
    """Parse a scene-segmented L1 response into Atom candidates.

    Under forced tool calling the caller already holds a decoded ``scenes``
    list, so pass it straight through to the normalizer.  The text path
    remains for adapters and legacy responses that still return free-form
    content, where the tolerant scanning below is doing real work.
    """
    if isinstance(text, list):
        return _normalize_scenes(text)
    # Parse the first complete top-level JSON document.  Looking for the first
    # ``[`` anywhere in the payload is unsafe: in an object response that is
    # commonly the nested ``source_message_ids``/``message_ids`` array, which
    # then gets treated as a list of scenes and silently produces zero
    # candidates.  Compatible models also vary between a scene array, a
    # single scene object, and the legacy flat ``{"memories": [...]}`` object.
    stripped = _FENCE_RE.sub("", str(text or "")).strip()
    decoder = json.JSONDecoder()
    parsed: Any = None
    parse_error: Exception | None = None
    # Scan every possible document start, not just the first ``{``/``[``.
    # Framing text may contain examples such as ``[... ]`` before the real
    # response, and nested arrays (for example source IDs) are not documents
    # we can use.  Prefer semantic top-level objects/scene arrays below.
    decoded: list[Any] = []
    cursor = 0
    while cursor < len(stripped):
        object_start = stripped.find("{", cursor)
        array_start = stripped.find("[", cursor)
        starts = [index for index in (object_start, array_start) if index >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(stripped[start:])
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = exc
            cursor = start + 1
            continue
        decoded.append(value)
        cursor = start + max(end, 1)

    # A flat object is authoritative when present. This avoids interpreting
    # its nested ``source_message_ids`` array as a scene list.
    for value in decoded:
        if isinstance(value, dict) and "memories" in value and "scene_name" not in value and "message_ids" not in value:
            parsed = value
            break
    if parsed is None:
        for value in decoded:
            if isinstance(value, list) and any(
                isinstance(item, dict)
                and ("memories" in item or "content" in item or "scene_name" in item)
                for item in value
            ):
                parsed = value
                break
    if parsed is None:
        for value in decoded:
            if isinstance(value, dict):
                parsed = value
                break
    if parsed is None:
        for value in decoded:
            if isinstance(value, list):
                parsed = value
                break
    if parsed is None:
        # Compatibility with the tolerant flat-object parser, which can find
        # a valid object after arbitrary explanatory framing text.
        try:
            return _parse_extraction_batch(text).memories
        except SemanticExtractionResponseError:
            if parse_error is not None:
                raise SemanticExtractionResponseError("json_document_invalid") from parse_error
            raise

    if isinstance(parsed, dict):
        # Legacy flat object: {"memories": [{...}]}.
        if "memories" in parsed and "scene_name" not in parsed and "message_ids" not in parsed:
            try:
                return _parse_extraction_batch(
                    json.dumps(parsed, ensure_ascii=False)
                ).memories
            except SemanticExtractionResponseError:
                # Continue through the scene normalizer so a permissive
                # response still gets individual valid records retained.
                parsed = [parsed]
        elif isinstance(parsed.get("scenes"), list):
            parsed = parsed["scenes"]
        elif "content" not in parsed and "scene_name" not in parsed:
            # Preserve the strict failure behaviour for unrelated JSON
            # objects instead of silently treating them as an empty scene.
            return _parse_extraction_batch(text).memories
        else:
            # A single scene object is a common model deviation from the
            # requested array contract.
            parsed = [parsed]

    if not isinstance(parsed, list):
        raise SemanticExtractionResponseError("json_array_invalid")
    return _normalize_scenes(parsed)


def _normalize_scenes(parsed: Sequence[Any]) -> list[AtomCandidate]:
    """Turn decoded scene objects into strictly validated candidates."""
    candidates: list[AtomCandidate] = []
    for scene in parsed:
        if not isinstance(scene, dict):
            continue

        # The scene-level ``message_ids`` field is the only evidence some
        # compatible models return for each nested memory.  Use it as a
        # conservative fallback; the extractor/pipeline still validates that
        # the IDs belong to this batch and that at least one is a user row.
        scene_message_ids: list[int] = []
        for value in scene.get("message_ids") or ():
            try:
                scene_message_ids.append(int(value))
            except (TypeError, ValueError):
                continue

        # A few OpenAI-compatible models flatten the requested scene array to
        # ``[{"content": ..., "source_message_ids": [...]}]``. Treat those
        # entries as memories in an implicit scene rather than dropping the
        # entire valid response. The strict AtomCandidate validation below is
        # still applied to every item.
        if "content" in scene and "memories" not in scene:
            scene_name = str(scene.get("scene_name") or "未知情境").strip()
            memories = [scene]
        else:
            scene_name = str(scene.get("scene_name") or "未知情境").strip()
            memories = scene.get("memories")
            if not isinstance(memories, list):
                continue

        for raw in memories:
            if not isinstance(raw, dict) or not str(raw.get("content") or "").strip():
                continue
            source_ids: list[int] = []
            source_values = (
                raw.get("source_message_ids")
                or raw.get("message_ids")
                or scene_message_ids
            )
            for value in source_values:
                try:
                    source_ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            metadata = raw.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            try:
                priority = int(raw.get("priority", 50))
            except (TypeError, ValueError):
                priority = 50
            try:
                candidates.append(AtomCandidate(
                    content=str(raw.get("content")).strip(),
                    type=str(raw.get("type") or "episodic").strip().lower(),
                    priority=priority,
                    scene_name=scene_name,
                    source_message_ids=sorted(set(source_ids)),
                    metadata=metadata,
                ))
            except Exception as exc:
                logger.debug("Skipping malformed scene memory: %s", type(exc).__name__)
    return candidates


class SemanticExtractor(Protocol):
    def extract(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        user_key: str = "",
        scope_key: str = "personal",
        session_id: str = "",
        mode: str = "chat",
    ) -> ExtractionBatch: ...

    def should_extract(
        self, messages: Sequence[dict[str, Any]], **kwargs: Any
    ) -> bool | None: ...


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    if value is None:
        return ""
    return str(value)


def _looks_like_base64_payload(text: str) -> bool:
    if not _BASE64_RE.fullmatch(text):
        return False
    compact = "".join(text.split())
    return (
        len(compact) >= 256
        and len(compact) % 4 == 0
        and len(set(compact)) >= 16
        and any(char.isdigit() or char in "+/=" for char in compact)
    )


def preprocess_messages(
    messages: Sequence[dict[str, Any]], *, max_chars: int = 24000
) -> list[dict[str, Any]]:
    """Keep auditable conversational text while excluding internal prompt data.

    ``max_chars`` is the extraction *chunk* budget, not a transcript-wide
    truncation limit.  Long sessions must reach :func:`chunk_messages` in
    full; otherwise only the first chunk can ever be extracted.  The argument
    remains in the signature for callers written against the original helper.
    """
    result = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role", "")).lower()
        if role not in {"user", "assistant", "tool"}:
            continue
        text = _content(raw.get("content"))
        text = _MEMORY_CONTEXT_RE.sub("", text).strip()
        if not text or _looks_like_base64_payload(text):
            continue
        try:
            from kylinmemory.redact import redact_sensitive_text

            redacted = redact_sensitive_text(text, force=True)
        except Exception:
            # L1 extraction may call a remote auxiliary model. If the central
            # redactor is unavailable, dropping this optional evidence is the
            # only safe fallback; sending the original text would turn a
            # redaction failure into a credential/privacy leak.
            logger.warning("Atom redaction failed; skipping one L0 message")
            continue
        if not redacted.strip():
            continue
        message_id = raw.get("id", raw.get("message_id"))
        try:
            message_id = int(message_id) if message_id is not None else None
        except (TypeError, ValueError):
            message_id = None
        if message_id is None:
            continue
        timestamp = raw.get("timestamp", raw.get("time", ""))
        item = {
            "message_id": message_id,
            "role": role,
            "time": timestamp,
            "content": redacted,
        }
        result.append(item)
    return result


def chunk_messages(
    messages: Sequence[dict[str, Any]],
    *,
    max_chars: int = 24000,
    overlap_chars: int = 1500,
) -> list[list[dict[str, Any]]]:
    chunks = []
    current = []
    size = 0
    for msg in messages:
        header = f"[message_id={msg['message_id']}][role={msg['role']}][time={msg.get('time', '')}]\n"
        encoded = header + msg["content"]
        if len(encoded) > max_chars:
            if current:
                chunks.append(current)
                current = []
                size = 0
            content_budget = max(1, max_chars - len(header))
            fragment_overlap = min(
                max(0, overlap_chars),
                max(0, content_budget // 4),
            )
            step = max(1, content_budget - fragment_overlap)
            content = str(msg.get("content") or "")
            start = 0
            while start < len(content):
                fragment = dict(msg)
                fragment["content"] = content[start : start + content_budget]
                chunks.append([fragment])
                if start + content_budget >= len(content):
                    break
                start += step
            continue
        if current and size + len(encoded) > max_chars:
            chunks.append(current)
            overlap = []
            overlap_size = 0
            for old in reversed(current):
                old_len = len(old.get("content", ""))
                if overlap_size + old_len > overlap_chars:
                    break
                overlap.insert(0, old)
                overlap_size += old_len
            current = overlap
            size = overlap_size
        current.append(msg)
        size += len(encoded)
    if current:
        chunks.append(current)
    return chunks


def _normalized_candidate_content(value: str) -> str:
    """Return a conservative, deterministic key for exact-memory deduping."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _candidate_quality_key(candidate: AtomCandidate) -> tuple[float, ...]:
    """Sort candidates by the normative priority and newest L0 evidence."""
    latest_message_id = max(candidate.source_message_ids or [0])
    return (
        -float(candidate.priority),
        -float(latest_message_id),
    )


def _deduplicate_and_select_candidates(
    candidates: Sequence[AtomCandidate], *, limit: int
) -> list[AtomCandidate]:
    """Deduplicate overlapping chunks, then select a stable bounded result.

    Chunk overlap commonly causes the same memory to be emitted more than
    once. Identity deliberately ignores metadata and timestamps: an identical
    normalized memory statement of the same type should consume
    only one slot even when those auxiliary fields vary between chunk calls.
    The strongest occurrence represents each duplicate group.  Stable input
    order is the final tie-breaker, making retries reproducible.
    """
    if limit <= 0:
        return []

    # key -> (best candidate, first occurrence).  Dict insertion order also
    # remains deterministic, but the explicit index documents the tie-break.
    deduplicated: dict[
        tuple[str, str], tuple[AtomCandidate, int]
    ] = {}
    for position, candidate in enumerate(candidates):
        key = (candidate.type, _normalized_candidate_content(candidate.content))
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = (candidate, position)
            continue
        if _candidate_quality_key(candidate) < _candidate_quality_key(existing[0]):
            # Preserve the group's first occurrence as the stable final
            # tie-break even when a later chunk supplies the better version.
            deduplicated[key] = (candidate, existing[1])

    ranked = sorted(
        deduplicated.values(),
        key=lambda item: (*_candidate_quality_key(item[0]), item[1]),
    )
    return [candidate for candidate, _ in ranked[:limit]]


def _user_anchored_messages(
    messages: Sequence[dict[str, Any]],
    *,
    max_user_turns: int = 10,
) -> list[dict[str, Any]]:
    """Keep each user message with only its nearest assistant context."""
    turns: list[list[dict[str, Any]]] = []
    pending_assistant: dict[str, Any] | None = None
    for message in messages:
        role = str(message.get("role") or "").lower()
        if role == "assistant":
            pending_assistant = message
            continue
        if role != "user":
            continue
        turn = []
        if pending_assistant is not None:
            turn.append(pending_assistant)
        turn.append(message)
        turns.append(turn)
        pending_assistant = None
    selected = turns[-max(1, int(max_user_turns)):]
    return [message for turn in selected for message in turn]


class OpenAICompatibleAtomExtractor:
    """MemoryCore-compatible scene segmentation + L1 extractor."""

    # Bump when the model-facing JSON contract changes. This version is part
    # of durable-job idempotency, so settled sessions can be reprocessed.
    version = "l1-v5-tools"

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_mode: str | None = None,
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
        main_runtime: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
        max_input_chars: int = 24000,
        max_memories: int = 10,
        max_attempts: int = 2,
        precheck_enabled: bool = False,
        previous_scene_name: str = "",
    ):
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.api_mode = api_mode
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self.main_runtime = main_runtime
        self.max_input_chars = max_input_chars
        self.max_memories = max_memories
        self.max_attempts = max(1, int(max_attempts))
        self.precheck_enabled = precheck_enabled
        self.previous_scene_name = str(previous_scene_name or "")

    def _current_main_runtime(self) -> dict[str, Any] | None:
        """Return the live main runtime used for automatic auxiliary routing.

        The semantic provider may outlive the first request that created it,
        and the primary agent can rotate credentials after a 401/402.  A
        startup-time ``main_runtime`` dictionary would therefore keep sending
        queued L1 jobs with the exhausted key.  Runtime construction passes a
        zero-argument callback; retain support for a dictionary for standalone
        callers and older integrations.
        """
        runtime = self.main_runtime
        if callable(runtime):
            try:
                value = runtime()
            except Exception:
                logger.debug(
                    "Could not read live main runtime for semantic extraction",
                    exc_info=True,
                )
                return None
            return value if isinstance(value, dict) else None
        return runtime if isinstance(runtime, dict) else None

    def extract(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        user_key: str = "",
        scope_key: str = "personal",
        session_id: str = "",
        mode: str = "chat",
        previous_scene_name: str = "",
    ) -> ExtractionBatch:
        prepared = preprocess_messages(messages, max_chars=self.max_input_chars)
        if not any(m["role"] == "user" for m in prepared):
            logger.info("L1 semantic extraction skipped: no_user_messages")
            return ExtractionBatch()
        # The extraction boundary counts user turns rather than raw rows.
        # Assistant messages are useful only as the nearest context preceding
        # a user answer; tool output and progress narration are excluded.
        new_messages = _user_anchored_messages(
            prepared, max_user_turns=10
        )
        background_messages: list[dict[str, Any]] = []
        candidates: list[AtomCandidate] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                candidates = self._request(
                    new_messages,
                    background_messages=background_messages,
                    previous_scene_name=previous_scene_name or self.previous_scene_name or "无",
                    mode=mode,
                    session_id=session_id,
                )
                break
            except Exception:
                logger.warning(
                    "L1 semantic extraction request failed; retrying "
                    "structured extraction (%d/%d)",
                    attempt,
                    self.max_attempts,
                )
                if attempt >= self.max_attempts:
                    break
        logger.info(
            "L1 semantic extraction returned %d candidates from %d user turns "
            "(+%d assistant context messages)",
            len(candidates),
            sum(1 for message in new_messages if message["role"] == "user"),
            sum(
                1
                for message in new_messages
                if message["role"] == "assistant"
            ),
        )
        valid_ids = {int(m["message_id"]) for m in new_messages}
        user_ids = {
            int(m["message_id"])
            for m in new_messages
            if m["role"] == "user"
        }
        candidates = [
            candidate for candidate in candidates
            if candidate.source_message_ids
            and set(candidate.source_message_ids) <= valid_ids
            and set(candidate.source_message_ids) & user_ids
        ]
        if candidates:
            self.previous_scene_name = candidates[-1].scene_name
        # Enforce the same bounded per-session output as MemoryCore.  The
        # dedupe pass also protects retries and overlapping capture windows.
        selected = _deduplicate_and_select_candidates(
            candidates,
            limit=min(max(0, int(self.max_memories)), _EXTRACTION_BATCH_LIMIT),
        )
        logger.info(
            "L1 semantic extraction total candidates: raw=%d selected=%d",
            len(candidates),
            len(selected),
        )
        return ExtractionBatch(memories=selected)

    def _request(
        self,
        new_messages: Sequence[dict[str, Any]],
        *,
        background_messages: Sequence[dict[str, Any]] = (),
        previous_scene_name: str = "无",
        mode: str = "chat",
        session_id: str = "",
    ) -> list[AtomCandidate]:
        from kylinmemory.auxiliary_client import call_llm, extract_tool_call_arguments
        from kylinmemory.memory_debug import log_memory_llm_input, log_memory_llm_output
        from kylinmemory.memory_prompts import format_extraction_prompt, get_extract_memories_system_prompt

        mode = "code" if str(mode).lower() == "code" else "chat"
        instructions = get_extract_memories_system_prompt(mode)
        user_prompt = format_extraction_prompt(
            new_messages, background_messages, previous_scene_name
        )
        extraction_tool = _l1_extraction_tool(mode)
        tool_choice = {
            "type": "function",
            "function": {"name": L1_EXTRACTION_TOOL_NAME},
        }
        request_messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_prompt},
        ]
        log_memory_llm_input(
            "L1",
            task="atom_memory",
            model=self.model,
            api_mode=self.api_mode,
            session_id=session_id,
            messages=request_messages,
            tools=[extraction_tool],
            tool_choice=tool_choice,
            mode=mode,
        )
        response = call_llm(
            task="atom_memory",
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            api_mode=self.api_mode,
            main_runtime=self._current_main_runtime(),
            messages=request_messages,
            temperature=0,
            max_tokens=3000,
            timeout=self.timeout,
            extra_body=self.extra_body,
            tools=[extraction_tool],
            tool_choice=tool_choice,
        )
        log_memory_llm_output(
            "L1",
            task="atom_memory",
            model=self.model,
            api_mode=self.api_mode,
            session_id=session_id,
            response=response,
            mode=mode,
        )
        tool_args = extract_tool_call_arguments(response, L1_EXTRACTION_TOOL_NAME)
        if tool_args is None:
            raise SemanticExtractionResponseError("tool_call_missing_or_invalid")
        scenes = tool_args.get("scenes")
        if not isinstance(scenes, list):
            raise SemanticExtractionResponseError("tool_arguments_invalid")
        # The provider already decoded the function arguments, so hand the
        # scene list straight to the strict candidate validation instead of
        # re-serializing it only to parse it back.
        try:
            return _parse_scene_extraction(scenes)
        except SemanticExtractionResponseError:
            raise
        except Exception as exc:
            raise SemanticExtractionResponseError(
                _validation_diagnostic(exc)
            ) from exc
