"""TencentDB-compatible L0 capture backed only by SQLite messages.

Completed turns are cleaned and persisted incrementally in the profile's
``l0_memory.db``. The ``l0_conversations`` projection contains message
text and metadata only: it has no FTS/vector companion table and never
requests an embedding. Conversation recall uses ``state.db`` instead of
maintaining a second searchable transcript in this file.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

_PROCESS_LOCK = threading.RLock()
_CONTEXT_BLOCKS = re.compile(
    r"<\s*(relevant-memories|user-persona|relevant-scenes|scene-navigation|"
    r"current_task_context|history_task_context)[^>]*>[\s\S]*?"
    r"</\s*\1\s*>",
    re.IGNORECASE,
)
_INBOUND_METADATA = re.compile(
    r"(?:Conversation info|Sender|Thread starter|Replied message|"
    r"Forwarded message context|Chat history since last reply)\s*"
    r"\(untrusted[\s\S]*?\):\s*```json\s*[\s\S]*?```",
    re.IGNORECASE,
)
_CODE_BLOCK = re.compile(r"```[^\n]*\n[\s\S]*?```", re.MULTILINE)


@dataclass(frozen=True)
class L0CaptureResult:
    messages: list[dict[str, Any]]
    recorded_count: int
    last_captured_timestamp: int


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    if value is None:
        return ""
    return str(value)


def sanitize_l0_capture_text(value: Any) -> str:
    """Remove runtime-injected context while retaining conversational text."""
    cleaned = _content_text(value)
    cleaned = _CONTEXT_BLOCKS.sub("", cleaned)
    cleaned = _INBOUND_METADATA.sub("", cleaned)
    cleaned = re.sub(r"```json\s*\{[\s\S]*?\"session[\s\S]*?\}\s*```", "", cleaned)
    cleaned = re.sub(r"\[\[reply_to[^\]]*\]\]\s*", "", cleaned)
    cleaned = re.sub(r"¥¥\[[\s\S]*?\]¥¥", "", cleaned)
    cleaned = re.sub(r"^\[[\w\d\-:+ ]+\]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\[media attached:[^\]]*\]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"To send an image back,[\s\S]*?(?:Keep caption in the text body\.)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^System:\s*\[[\s\S]*?$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(
        r"data:image/[a-z+]+;base64,[A-Za-z0-9+/=]+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned.replace("\x00", "")).strip()


def should_capture_l0(text: str) -> bool:
    """Apply MemoryCore's intentionally permissive L0 quality gate."""
    value = str(text or "").strip()
    if not value or value.startswith("/"):
        return False
    if value == "(session bootstrap)" or value.startswith("A new session was started via"):
        return False
    if re.match(r"^✅\s*New session started", value):
        return False
    if value.startswith("Pre-compaction memory flush") or re.fullmatch(r"NO_REPLY\s*", value):
        return False
    return True


def _timestamp_ms(value: Any, fallback: int) -> int:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return fallback
    if timestamp <= 0:
        return fallback
    # SessionDB stores epoch seconds while MemoryCore's L0 contract uses ms.
    return int(timestamp * 1000) if timestamp < 10_000_000_000 else int(timestamp)


def _message_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return time.time_ns() * 1000 + secrets.randbelow(1000)


def _record_id(session_key: str, message: Mapping[str, Any]) -> str:
    identity = "\0".join(
        (
            str(session_key),
            str(message.get("id") or ""),
            str(message.get("timestamp") or ""),
            str(message.get("role") or ""),
        )
    )
    return "l0_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


class L0Recorder:
    """Incremental SQLite L0 recorder with an atomic per-session cursor."""

    def __init__(
        self,
        root: str | Path,
        *,
        plugin_start_ms: int | None = None,
        database: Any | None = None,
    ):
        self.root = Path(root)
        self.plugin_start_ms = int(
            time.time() * 1000 if plugin_start_ms is None else plugin_start_ms
        )
        self._owns_database = database is None
        if database is None:
            from kylin_memory.state import MemoryDB

            database = MemoryDB(self.root / "l0_memory.db")
        self.database = database

    def capture(
        self,
        session_key: str,
        raw_messages: Sequence[Mapping[str, Any]],
        *,
        session_id: str = "",
        team_id: str = "",
        user_id: str = "",
        agent_id: str = "",
        task_id: str = "",
        original_user_text: str = "",
        original_user_message_count: int | None = None,
    ) -> L0CaptureResult:
        """Clean and atomically store messages newer than the SQLite cursor."""
        key = str(session_key or session_id or "default")
        position_slice = (
            original_user_message_count is not None
            and original_user_message_count > 0
            and original_user_message_count <= len(raw_messages)
        )
        source = (
            raw_messages[original_user_message_count:]
            if position_slice
            else raw_messages
        )
        fallback_base = int(time.time() * 1000)
        extracted: list[dict[str, Any]] = []
        for index, raw in enumerate(source):
            if not isinstance(raw, Mapping):
                continue
            role = str(raw.get("role") or "").lower()
            if role not in {"user", "assistant"}:
                continue
            extracted.append(
                {
                    "id": _message_id(raw.get("id", raw.get("message_id"))),
                    "role": role,
                    "content": _content_text(raw.get("content")),
                    "timestamp": _timestamp_ms(
                        raw.get("timestamp", raw.get("time")), fallback_base + index
                    ),
                }
            )

        if original_user_text and position_slice:
            for message in extracted:
                if message["role"] == "user":
                    message["content"] = original_user_text
                    break

        filtered: list[dict[str, Any]] = []
        for message in extracted:
            content = sanitize_l0_capture_text(message["content"])
            if message["role"] == "assistant":
                content = re.sub(r"\n{3,}", "\n\n", _CODE_BLOCK.sub("", content)).strip()
            if not should_capture_l0(content):
                continue
            filtered.append({**message, "content": content})

        records = [
            {
                **message,
                "record_id": _record_id(key, message),
                "session_id": str(session_id or "default"),
                "team_id": str(team_id or "default"),
                "user_id": str(user_id or "default"),
                "agent_id": str(agent_id or "default"),
                "task_id": str(task_id or ""),
            }
            for message in filtered
        ]
        if not records:
            return L0CaptureResult([], 0, self.plugin_start_ms)

        with _PROCESS_LOCK:
            stored, newest = self.database.capture_l0_records(
                key,
                records,
                initial_cursor=self.plugin_start_ms,
            )
        messages = [
            {
                "id": int(record["id"]),
                "role": str(record["role"]),
                "content": str(record["content"]),
                "timestamp": int(record["timestamp"]),
            }
            for record in stored
        ]
        return L0CaptureResult(messages, len(messages), int(newest))

    def read_after(
        self,
        session_key: str,
        *,
        after_recorded_at_ms: int = 0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Read the oldest SQLite L0 user turns after the L1 cursor.

        The turn limit excludes assistant context from the count. Tool-role
        messages are excluded both at capture time and by the database query.
        """
        return self.database.query_l0_for_l1(
            session_key,
            after_recorded_at_ms=after_recorded_at_ms,
            limit=limit,
        )

    def close(self) -> None:
        if self._owns_database and self.database is not None:
            self.database.close()
            self.database = None


__all__ = [
    "L0CaptureResult",
    "L0Recorder",
    "sanitize_l0_capture_text",
    "should_capture_l0",
]
