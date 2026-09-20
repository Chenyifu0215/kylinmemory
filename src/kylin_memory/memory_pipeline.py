"""Small durable L1→L2 pipeline manager.

The manager intentionally has no model dependency.  Applications may provide
an extractor callable returning Atom mappings and a scene consolidator; the
default implementation still provides useful deterministic persistence and
scope isolation for local deployments.
"""

from __future__ import annotations

import hashlib
import logging
import json
import os
import secrets
import threading
from .config import context_timer
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .memory_layers import (
    ALLOWED_METADATA,
    CHAT_TYPES,
    CODE_TYPES,
    Atom,
    AtomStore,
    ScenarioStore,
    normalize_scope,
    should_extract_l1,
    utc_iso,
)

logger = logging.getLogger(__name__)


_L1_CONFLICT_TOOL_NAME = "l1_conflict_decisions"
_CHAT_MERGED_TYPE_RULE = (
    "Type the merged memory belongs to, judged from the merged content; it "
    "need not match the new memory's original type. Required for "
    "update/merge, omitted for store/skip."
)
_WORK_MERGED_TYPE_RULE = _CHAT_MERGED_TYPE_RULE
_CHAT_MERGED_PRIORITY_RULE = (
    "Priority of the merged memory. A merge usually justifies raising it. "
    "Must not fall below the floor for merged_type: persona 50, episodic 60, "
    "instruction 70; maximum 100. -1 is reserved for an absolute, "
    "never-violable instruction. Required for update/merge."
)
_WORK_MERGED_PRIORITY_RULE = (
    "Priority of the merged memory, 70-100. A merge usually justifies raising "
    "it. Values below 70 are rejected. Required for update/merge."
)


def _l1_conflict_tool(mode: str) -> dict[str, Any]:
    """Build the conflict schema with only the types valid for this mode."""
    is_code = str(mode).lower() == "code"
    merged_types = (
        ["work_fact", "work_task", "work_method", "work_artifact"]
        if is_code
        else ["persona", "episodic", "instruction"]
    )
    return {
        "type": "function",
        "function": {
            "name": _L1_CONFLICT_TOOL_NAME,
            "description": (
                "Submit exactly one decision for every new memory record. "
                "Call once."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decisions": {
                        "type": "array",
                        "description": (
                            "One entry per new memory record, no omissions "
                            "and no duplicates."
                        ),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "record_id": {
                                    "type": "string",
                                    "description": (
                                        "record_id of the new memory this "
                                        "decision applies to."
                                    ),
                                },
                                "action": {
                                    "type": "string",
                                    "enum": ["store", "skip", "update", "merge"],
                                    "description": (
                                        "store: keep as new. skip: an existing "
                                        "memory already covers it. update: "
                                        "same fact, the new memory supersedes "
                                        "the targets. merge: complementary "
                                        "information combined into one record."
                                    ),
                                },
                                "target_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Existing record_ids this decision "
                                        "replaces, taken only from this "
                                        "record's own candidate list. One or "
                                        "more for update/merge; empty for "
                                        "store/skip. An update/merge with no "
                                        "valid target is downgraded to store."
                                    ),
                                },
                                "merged_content": {
                                    "type": "string",
                                    "description": (
                                        "Final text of the merged memory, in "
                                        "the existing memories' language. "
                                        "Required for update/merge, omitted "
                                        "for store/skip."
                                    ),
                                },
                                "merged_type": {
                                    "type": "string",
                                    "enum": merged_types,
                                    "description": (
                                        _WORK_MERGED_TYPE_RULE
                                        if is_code
                                        else _CHAT_MERGED_TYPE_RULE
                                    ),
                                },
                                "merged_priority": {
                                    "type": "integer",
                                    "minimum": -1,
                                    "maximum": 100,
                                    "description": (
                                        _WORK_MERGED_PRIORITY_RULE
                                        if is_code
                                        else _CHAT_MERGED_PRIORITY_RULE
                                    ),
                                },
                                "merged_timestamps": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Union of the new memory's and every "
                                        "target's timestamps, deduplicated and "
                                        "sorted ascending, preserving the full "
                                        "timeline. Required for update/merge."
                                    ),
                                },
                            },
                            "required": ["record_id", "action", "target_ids"],
                        },
                    },
                },
                "required": ["decisions"],
            },
        },
    }


_L1_CONFLICT_TOOL = _l1_conflict_tool("chat")


class MemoryPipelineManager:
    def __init__(self, root: str | Path | None = None, *, config: Mapping[str, Any] | None = None,
                 extractor: Callable[..., Any] | None = None,
                 consolidator: Callable[..., Any] | None = None,
                 deduper: Callable[..., Any] | None = None,
                 main_runtime: Mapping[str, Any] | Callable[[], dict[str, Any]] | None = None,
                 l0_reader: Callable[..., Sequence[Mapping[str, Any]]] | None = None):
        self.root = Path(root) if root else None
        self.config = dict(config or {})
        self.extractor = extractor
        self.consolidator = consolidator
        # Optional host supplied conflict resolver.  When absent, the manager
        # uses the configured auxiliary LLM and falls back to ``store`` for all
        # records, matching MemoryCore's fail-open batchDedup contract.
        self.deduper = deduper
        # Resolve at call time so background conflict checks follow model
        # switches and credential rotation just like L1/L2 extraction.
        self.main_runtime = main_runtime
        # TencentDB's L1 runner reads incremental rows from durable L0 instead
        # of treating the scheduler's in-memory message buffer as evidence.
        # The built-in provider wires this to SQLite L0; tests and third-party
        # callers may omit it and retain the legacy buffer path.
        self.l0_reader = l0_reader
        embedding_cfg = self.config.get("embedding") if isinstance(self.config.get("embedding"), Mapping) else {}
        self.enable_dedup = bool(self.config.get("enable_dedup", True))
        self.conflict_recall_top_k = max(
            1, int(embedding_cfg.get("conflict_recall_top_k", 5) or 5)
        )
        self.atoms = AtomStore(self.root, embedding=embedding_cfg)
        scenario = self.config.get("scenario") if isinstance(self.config.get("scenario"), Mapping) else {}
        self.max_scenes = int(scenario.get("max_scenes", 15))
        self.prompt_mode = str(self.config.get("prompt_mode", "chat") or "chat").lower()
        self._scenes: dict[str, ScenarioStore] = {}
        self._cursors: dict[str, str] = {}
        self._locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)
        # MemoryCore-compatible scheduler state.  The public ``notify_*``
        # methods are deliberately synchronous-safe; hosts may call them from
        # their own worker thread without requiring an event loop.
        self._scheduler_state: dict[str, dict[str, Any]] = {}
        self._scheduler_timers: dict[str, threading.Timer] = {}
        self._scheduler_guard = threading.RLock()
        # MemoryCore uses one SerialQueue per derived layer.  Separate global
        # locks preserve that concurrency=1 contract across session timers.
        self._l1_job_lock = threading.RLock()
        self._l2_job_lock = threading.RLock()
        self._closing = False
        self._closed = False
        atom_cfg = self.config
        self.l1_threshold = max(1, int(atom_cfg.get("every_n_conversations", atom_cfg.get("checkpoint_turns", 5)) or 5))
        self.l1_idle_seconds = max(0.0, float(atom_cfg.get("l1_idle_timeout_seconds", atom_cfg.get("idle_timeout_seconds", atom_cfg.get("settle_delay_seconds", 600))) or 0))
        # Warm-up is useful only when it can ramp through more than one
        # smaller checkpoint (1 -> 2 -> 4 -> target).  With a target of 2 it
        # used to flush the first conversation immediately, so the first
        # steady-state run contained only the second conversation.  That made
        # ``every_n_conversations: 2`` behave like two unrelated one-turn
        # extraction jobs.  Keep the configured threshold authoritative for
        # these small windows; larger targets retain MemoryCore warm-up.
        self.l1_warmup = (
            bool(atom_cfg.get("enable_warmup", True))
            and self.l1_threshold > 2
        )
        self.l1_retry_seconds = max(0.0, float(atom_cfg.get("retry_base_delay_seconds", 30) or 0))
        self.l1_max_retries = max(0, int(atom_cfg.get("max_attempts", 5) or 0))
        # L1 batch limits count user turns, not raw L0 rows. Assistant text is
        # retained as context but cannot crowd user evidence out of a batch.
        self.l1_batch_process = max(1, int(atom_cfg.get("l1_batch_process", 10) or 10))
        self.l1_batch_query = max(
            self.l1_batch_process,
            int(atom_cfg.get("l1_batch_query", self.l1_batch_process * 2) or self.l1_batch_process * 2),
        )
        self.l2_delay_seconds = max(0.0, float(scenario.get("l2_delay_after_l1_seconds", 10) or 0))
        self.l2_min_interval_seconds = max(0.0, float(scenario.get("l2_min_interval_seconds", 900) or 0))
        self.l2_max_interval_seconds = max(0.0, float(scenario.get("l2_max_interval_seconds", 3600) or 0))
        self.l2_active_window_seconds = max(0.0, float(scenario.get("session_active_window_hours", 24) or 0) * 3600.0)
        self._scheduler_dir = self.atoms.root / ".metadata" / "pipeline_sessions"
        self._scheduler_dir.mkdir(parents=True, exist_ok=True)

    def notify_conversation(
        self,
        session_key: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        session_id: str = "",
        team_id: str = "",
        user_id: str = "",
        agent_id: str = "",
        task_id: str = "",
        mode: str = "chat",
    ) -> dict[str, Any]:
        """Record one completed conversation and apply MemoryCore triggers.

        This is the Python equivalent of ``PipelineManager.notifyConversation``:
        threshold-triggered work runs immediately; below threshold, a
        resettable idle timer flushes the pending messages.  A per-session lock
        prevents overlapping L1/L2 jobs.
        """
        key = str(session_key or session_id or "default")
        if self._closing or self._closed:
            return {"success": False, "reason": "pipeline_closed"}
        with self._scheduler_guard:
            state = self._scheduler_state.setdefault(key, self._new_scheduler_state())
            # Capture hooks can replay a session transcript at a boundary.
            # Keep scheduler buffering idempotent by ignoring message ids that
            # were already acknowledged for this session.
            captured: list[dict[str, Any]] = []
            last_id = int(state.get("last_notified_message_id") or 0)
            for item in messages:
                if not isinstance(item, Mapping):
                    continue
                value = dict(item)
                try:
                    msg_id = int(value.get("id", value.get("message_id")))
                except (TypeError, ValueError):
                    msg_id = 0
                if msg_id and msg_id <= last_id:
                    continue
                captured.append(value)
                last_id = max(last_id, msg_id)
            if not captured and not callable(self.l0_reader):
                return {"success": True, "queued": False, "reason": "no_new_messages"}
            state["conversation_count"] += 1
            state["last_active"] = time.time()
            state["l1_retry_count"] = 0
            state["last_notified_message_id"] = last_id
            state["buffer"].extend(captured)
            if callable(self.l0_reader):
                # Durable L0 is authoritative for cadence. A stale scheduler
                # checkpoint or assistant/tool rows must not hide/inflate
                # pending user turns after a restart.
                try:
                    self._reconcile_l0_scheduler_state(key, state)
                except Exception:
                    logger.warning("SQLite L0 read failed while accounting pending L1 turns", exc_info=True)
            state["context"] = {
                "session_id": str(session_id or key), "team_id": str(team_id or ""),
                "user_id": str(user_id or ""), "agent_id": str(agent_id or ""),
                "task_id": str(task_id or ""), "mode": str(mode or self.prompt_mode),
            }
            threshold = self._l1_trigger_threshold(state)
            self._persist_scheduler_state()
        if state["conversation_count"] >= threshold:
            self._cancel_scheduler_timer("l1", key)
            self._schedule_idle_flush(
                key, delay=0, session_id=session_id, team_id=team_id,
                user_id=user_id, agent_id=agent_id, task_id=task_id, mode=mode,
            )
        elif self.l1_idle_seconds > 0:
            self._schedule_idle_flush(
                key, session_id=session_id, team_id=team_id, user_id=user_id,
                agent_id=agent_id, task_id=task_id, mode=mode,
            )
        return {"success": True, "queued": True, "conversation_count": state["conversation_count"], "threshold": threshold}

    def _new_scheduler_state(self) -> dict[str, Any]:
        return {
            "conversation_count": 0,
            "buffer": [],
            "warmup_threshold": 1 if self.l1_warmup else 0,
            "last_active": time.time(),
            "last_l2": 0.0,
            "l2_pending": 0,
            "l1_retry_count": 0,
            "context": {},
            "l2_fire_at": 0.0,
            "l2_source": "",
            "last_notified_message_id": 0,
            "last_l1_cursor": 0,
            "l1_backlog_pending": False,
            # The final assistant message from a consumed batch can be the
            # question answered by the next user (for example, "Yes"). Keep
            # that one-message overlap without counting it as another turn.
            "l1_assistant_context": [],
        }

    @staticmethod
    def _l1_user_count(messages: Sequence[Mapping[str, Any]]) -> int:
        return sum(
            1
            for message in messages
            if str(message.get("role") or "").lower() == "user"
        )

    def _l1_trigger_threshold(self, state: Mapping[str, Any]) -> int:
        """Return the active user-turn threshold for one scheduler state."""
        if self.l1_warmup:
            return int(state.get("warmup_threshold") or self.l1_threshold)
        return self.l1_threshold

    def _reconcile_l0_scheduler_state(
        self,
        key: str,
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Rebuild pending cadence and missing context from durable L0.

        Scheduler checkpoints are only an optimization. They may be absent,
        stale, or written before the latest committed L0 turn, so SQLite is
        authoritative whenever an L0 reader is installed.
        """
        if not callable(self.l0_reader):
            return []
        pending_rows = [
            dict(item)
            for item in self.l0_reader(
                key,
                after_recorded_at_ms=int(state.get("last_l1_cursor") or 0),
                # Counting must be able to reach the configured trigger even
                # when it is larger than the extraction query batch.
                limit=max(self.l1_batch_query, self.l1_threshold),
            )
            if isinstance(item, Mapping)
        ]
        state["conversation_count"] = self._l1_user_count(pending_rows)

        if pending_rows:
            evidence = next(
                (
                    item
                    for item in reversed(pending_rows)
                    if str(item.get("role") or "").lower() == "user"
                ),
                pending_rows[-1],
            )
            context = dict(state.get("context") or {})
            defaults = {
                "session_id": evidence.get("session_id") or key,
                "team_id": evidence.get("team_id") or "",
                "user_id": evidence.get("user_id") or "",
                "agent_id": evidence.get("agent_id") or "",
                "task_id": evidence.get("task_id") or "",
                "mode": self.prompt_mode,
            }
            for field, value in defaults.items():
                if context.get(field) in (None, ""):
                    context[field] = str(value or "")
            state["context"] = context
        return pending_rows

    def _slice_l1_user_turns(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        """Select a bounded user-turn batch while retaining assistant context.

        A recorded-at boundary is never split because the durable cursor is a
        millisecond timestamp. This preserves all rows captured atomically,
        even when doing so includes slightly more than the configured number
        of user turns.
        """
        eligible = [
            dict(message)
            for message in messages
            if str(message.get("role") or "").lower()
            in {"user", "assistant"}
        ]
        if not eligible:
            return [], False, False

        user_count = 0
        slice_end = len(eligible)
        for index, message in enumerate(eligible):
            if str(message.get("role") or "").lower() != "user":
                continue
            user_count += 1
            if user_count < self.l1_batch_process:
                continue
            boundary = int(message.get("recorded_at_ms") or 0)
            slice_end = index + 1
            while (
                slice_end < len(eligible)
                and int(eligible[slice_end].get("recorded_at_ms") or 0)
                == boundary
            ):
                slice_end += 1
            break

        batch = eligible[:slice_end]
        has_unprocessed = len(eligible) > slice_end
        queried_users = self._l1_user_count(eligible)
        has_full_backlog = (
            queried_users >= self.l1_batch_query and has_unprocessed
        )
        return batch, has_unprocessed and not has_full_backlog, has_full_backlog

    @staticmethod
    def _trailing_assistant_context(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only the nearest assistant message for the next user turn."""
        for message in reversed(messages):
            role = str(message.get("role") or "").lower()
            if role == "assistant":
                return [dict(message)]
            if role == "user":
                break
        return []

    @staticmethod
    def _l1_semantic_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build user-anchored semantic turns for the extractor.

        Only the nearest assistant message before a user message is retained.
        This preserves the question needed to interpret answers such as
        "yes", while dropping tool output and assistant progress narration.
        """
        result: list[dict[str, Any]] = []
        pending_assistant: dict[str, Any] | None = None
        for raw in messages:
            if not isinstance(raw, Mapping):
                continue
            role = str(raw.get("role") or "").lower()
            if role == "assistant":
                pending_assistant = dict(raw)
                continue
            if role != "user":
                continue
            if pending_assistant is not None:
                result.append(pending_assistant)
            result.append(dict(raw))
            pending_assistant = None
        return result

    @staticmethod
    def _timer_key(layer: str, key: str) -> str:
        return f"{layer}:{key}"

    def _cancel_scheduler_timer(self, layer: str, key: str) -> None:
        timer = self._scheduler_timers.pop(self._timer_key(layer, key), None)
        if timer is not None:
            timer.cancel()

    def _timer_call(self, layer: str, key: str, callback: Callable[[], Any]) -> None:
        self._scheduler_timers.pop(self._timer_key(layer, key), None)
        try:
            callback()
        except Exception:
            logger.exception("Memory pipeline %s timer failed for %s", layer, key)

    def _schedule_idle_flush(self, key: str, *, delay: float | None = None, **kwargs: Any) -> None:
        if self._closing or self._closed:
            return
        self._cancel_scheduler_timer("l1", key)
        seconds = self.l1_idle_seconds if delay is None else max(0.0, float(delay))
        timer = context_timer(
            seconds,
            lambda: self._timer_call("l1", key, lambda: self.flush_conversation(key, **kwargs)),
        )
        timer.daemon = True
        self._scheduler_timers[self._timer_key("l1", key)] = timer
        timer.start()

    def flush_conversation(self, session_key: str, **kwargs: Any) -> dict[str, Any]:
        with self._l1_job_lock, self._locks[f"scheduler:{session_key}"]:
            if self._closed:
                return {"success": False, "reason": "pipeline_closed", "written": []}
            # A threshold timer and an explicit session flush can arrive
            # together. Re-check only after entering both queue/session locks.
            state = self._scheduler_state.get(session_key)
            if not state:
                return {"success": True, "skipped": True, "written": []}
            context = {**dict(state.get("context") or {}), **{k: v for k, v in kwargs.items() if v not in (None, "")}}
            durable_l0_mode = callable(self.l0_reader)
            has_more = False
            has_full_backlog = False
            if durable_l0_mode:
                try:
                    queried = [
                        dict(item)
                        for item in self.l0_reader(
                            session_key,
                            after_recorded_at_ms=int(state.get("last_l1_cursor") or 0),
                            limit=self.l1_batch_query,
                        )
                        if isinstance(item, Mapping)
                    ]
                except Exception as exc:
                    logger.warning("SQLite L0 read failed; retaining L1 cursor", exc_info=True)
                    return {
                        "success": False,
                        "reason": "l0_read_failed",
                        "error": type(exc).__name__,
                        "written": [],
                    }
                queried.sort(key=lambda item: (
                    int(item.get("recorded_at_ms") or 0),
                    int(item.get("timestamp") or 0),
                    int(item.get("id") or 0),
                ))
                batch, has_more, has_full_backlog = (
                    self._slice_l1_user_turns(queried)
                )
            else:
                batch = list(state.get("buffer") or [])
                state["buffer"] = []

            # An idle/backlog poll with no rows is still a successful L1 run
            # in MemoryCore: it resets the conversation counter and advances
            # warm-up just like a runner that returned processedCount=0.
            if not batch and not durable_l0_mode:
                return {"success": True, "skipped": True, "written": []}
            if not batch and durable_l0_mode:
                state["l1_backlog_pending"] = False
                if not int(state.get("conversation_count") or 0):
                    self._persist_scheduler_state()
                    return {"success": True, "skipped": True, "written": []}
            try:
                extraction_batch = batch
                if durable_l0_mode and batch:
                    previous_context = [
                        dict(item)
                        for item in state.get("l1_assistant_context") or []
                        if isinstance(item, Mapping)
                    ]
                    seen_ids = {
                        int(item.get("id", item.get("message_id")))
                        for item in batch
                        if item.get("id", item.get("message_id")) is not None
                    }
                    contextual_batch = [
                        item
                        for item in previous_context
                        if int(item.get("id", item.get("message_id"))) not in seen_ids
                    ] + batch
                    extraction_batch = self._l1_semantic_messages(
                        contextual_batch
                    )
                if durable_l0_mode:
                    result = self._ingest_l0_groups(
                        extraction_batch,
                        session_key=session_key,
                        context=context,
                    )
                else:
                    result = self.ingest(
                        batch, session_key=session_key, session_id=str(context.get("session_id") or session_key),
                        task_id=str(context.get("task_id") or ""), team_id=str(context.get("team_id") or ""),
                        user_id=str(context.get("user_id") or ""), agent_id=str(context.get("agent_id") or ""),
                        mode=str(context.get("mode") or self.prompt_mode),
                    )
            except Exception as exc:
                logger.warning("L1 pipeline run failed; retaining buffer for retry", exc_info=True)
                result = {"success": False, "reason": "l1_extraction_failed", "error": type(exc).__name__, "written": []}
            if result.get("success") is False:
                if not durable_l0_mode:
                    state["buffer"] = batch + list(state.get("buffer") or [])
                state["conversation_count"] = max(1, int(state.get("conversation_count") or 0))
                state["l1_retry_count"] = int(state.get("l1_retry_count") or 0) + 1
                if state["l1_retry_count"] <= self.l1_max_retries:
                    self._schedule_idle_flush(session_key, delay=self.l1_retry_seconds, **context)
                self._persist_scheduler_state()
                return result
            if durable_l0_mode and batch:
                state["last_l1_cursor"] = max(
                    int(item.get("recorded_at_ms") or 0) for item in batch
                )
                state["l1_assistant_context"] = (
                    self._trailing_assistant_context(batch)
                )
            state["l1_backlog_pending"] = bool(has_more or has_full_backlog)
            state["conversation_count"] = 0
            state["l1_retry_count"] = 0
            if self.l1_warmup and state.get("warmup_threshold", 0):
                next_threshold = int(state["warmup_threshold"]) * 2
                state["warmup_threshold"] = 0 if next_threshold >= self.l1_threshold else next_threshold
            state["l2_pending"] = 1
            state["context"] = context
            # MemoryCore advances L2 after every successful L1 run.  The L2
            # reader itself decides whether there are any rows past its cursor.
            self._schedule_l2_flush(session_key, source="delay-after-l1", **context)
            self._persist_scheduler_state()
            if has_full_backlog:
                self._schedule_idle_flush(session_key, delay=0, **context)
            elif has_more:
                self._schedule_idle_flush(session_key, delay=self.l1_idle_seconds, **context)
            return {
                **result,
                "processed_count": self._l1_user_count(batch),
                "processed_message_count": len(batch),
                "has_more": has_more,
                "has_full_backlog": has_full_backlog,
            }

    def _ingest_l0_groups(
        self,
        batch: Sequence[Mapping[str, Any]],
        *,
        session_key: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run L1 per L0 isolation/session group, oldest group first."""
        if not batch:
            return {"success": True, "written": [], "skipped": True}
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for item in batch:
            key = (
                str(item.get("team_id") or context.get("team_id") or ""),
                str(item.get("user_id") or context.get("user_id") or ""),
                str(item.get("agent_id") or context.get("agent_id") or ""),
                str(item.get("session_id") or context.get("session_id") or session_key),
                str(item.get("task_id") or context.get("task_id") or ""),
            )
            grouped.setdefault(key, []).append(dict(item))
        ordered = sorted(
            grouped.items(),
            key=lambda pair: min(
                (int(item.get("recorded_at_ms") or 0), int(item.get("timestamp") or 0))
                for item in pair[1]
            ),
        )
        written: list[str] = []
        for (team_id, user_id, agent_id, session_id, task_id), messages in ordered:
            result = self.ingest(
                messages,
                session_key=session_key,
                session_id=session_id,
                task_id=task_id,
                team_id=team_id,
                user_id=user_id,
                agent_id=agent_id,
                mode=str(context.get("mode") or self.prompt_mode),
            )
            if result.get("success") is False:
                return result
            written.extend(str(value) for value in result.get("written", ()))
        return {"success": True, "written": written, "skipped": not bool(written)}

    def _schedule_l2_flush(self, session_key: str, *, source: str, **kwargs: Any) -> None:
        if self._closing or self._closed or self.l2_max_interval_seconds <= 0 and source == "max-interval":
            return
        state = self._scheduler_state.setdefault(session_key, {})
        now = time.time()
        if source == "max-interval":
            fire_at = now + self.l2_max_interval_seconds
        else:
            last_l2 = float(state.get("last_l2") or 0.0)
            floor = last_l2 + self.l2_min_interval_seconds if last_l2 else 0.0
            fire_at = max(now + self.l2_delay_seconds, floor)
            current_fire = float(state.get("l2_fire_at") or 0.0)
            # L2 delay timers are downward-only: a new L1 event can advance a
            # later schedule, but can never postpone an earlier one.
            if self._timer_key("l2", session_key) in self._scheduler_timers and current_fire and current_fire <= fire_at:
                return
        self._cancel_scheduler_timer("l2", session_key)
        state["l2_fire_at"] = fire_at
        state["l2_source"] = source
        state["context"] = {**dict(state.get("context") or {}), **kwargs}
        timer = context_timer(
            max(0.0, fire_at - now),
            lambda: self._timer_call(
                "l2", session_key,
                lambda: self.run_l2(session_key, source=source, **dict(state.get("context") or {})),
            ),
        )
        timer.daemon = True
        self._scheduler_timers[self._timer_key("l2", session_key)] = timer
        timer.start()

    def run_l2(self, session_key: str, *, source: str = "delay-after-l1", **kwargs: Any) -> dict[str, Any]:
        """Run one L2 job through the global MemoryCore-style serial queue."""
        with self._l2_job_lock:
            return self._run_l2(session_key, source=source, **kwargs)

    def _run_l2(self, session_key: str, *, source: str = "delay-after-l1", **kwargs: Any) -> dict[str, Any]:
        if self._closed:
            return {"success": False, "reason": "pipeline_closed", "changed": []}
        state = self._scheduler_state.setdefault(session_key, self._new_scheduler_state())
        context = {**dict(state.get("context") or {}), **{k: v for k, v in kwargs.items() if v not in (None, "")}}
        now = time.time()
        if (
            source == "max-interval"
            and self.l2_active_window_seconds > 0
            and now - float(state.get("last_active") or 0.0) >= self.l2_active_window_seconds
        ):
            state["l2_fire_at"] = 0.0
            state["l2_source"] = ""
            self._persist_scheduler_state()
            return {"success": True, "skipped": True, "reason": "cold_session", "changed": []}
        result = self.consolidate(
            team_id=str(context.get("team_id") or ""), agent_id=str(context.get("agent_id") or ""),
            user_id=str(context.get("user_id") or ""), session_id=str(context.get("session_id") or session_key),
        )
        if result.get("success"):
            state["l2_pending"] = 0
            # A first empty poll must not rate-limit the first real L1 event.
            if not (not state.get("last_l2") and result.get("skipped")):
                state["last_l2"] = time.time()
            self._schedule_l2_flush(session_key, source="max-interval", **context)
        else:
            # Retry L2 on the maximum interval. Its durable cursor remains
            # unchanged, so no L1 evidence is consumed by the failed run.
            self._schedule_l2_flush(session_key, source="max-interval", **context)
        self._persist_scheduler_state()
        return result

    def resume_scheduled_work(self, session_key: str) -> None:
        """Re-arm durable pending L1/L2 work after provider initialization."""
        if self._closing or self._closed:
            return
        key = str(session_key or "")
        if not key:
            return
        self._load_scheduler_state(key)
        # A scheduler checkpoint is only an optimization. L0 can contain a
        # committed turn even when the process exited before the checkpoint
        # was created or refreshed, so recover the requested key from SQLite
        # even when no state file exists.
        state = self._scheduler_state.get(key)
        if state is None and not callable(self.l0_reader):
            # Direct provider adapters have no durable L0 scheduler to
            # recover.  Do not manufacture an empty state here: the explicit
            # session-boundary path must remain able to ingest its supplied
            # transcript.
            return
        if state is None and callable(self.l0_reader):
            try:
                probe = self.l0_reader(
                    key,
                    after_recorded_at_ms=0,
                    limit=max(self.l1_batch_query, self.l1_threshold),
                )
                if not any(
                    str(item.get("role") or "").lower() == "user"
                    for item in probe
                    if isinstance(item, Mapping)
                ):
                    # Do not create scheduler state for a session that never
                    # committed user evidence. This preserves the provider's
                    # no-synthetic-L0 contract.
                    return
            except Exception:
                logger.warning(
                    "SQLite L0 read failed while probing L1 scheduler state",
                    exc_info=True,
                )
                return
        state = self._scheduler_state.setdefault(key, self._new_scheduler_state())
        if callable(self.l0_reader):
            try:
                self._reconcile_l0_scheduler_state(key, state)
            except Exception:
                logger.warning(
                    "SQLite L0 read failed while recovering L1 scheduler state",
                    exc_info=True,
                )
        self._persist_scheduler_state()

        for key, state in [(key, state)]:
            context = dict(state.get("context") or {})
            threshold = self._l1_trigger_threshold(state)
            if state.get("buffer"):
                delay = (
                    0
                    if int(state.get("conversation_count") or 0) >= threshold
                    else self.l1_idle_seconds
                )
                self._schedule_idle_flush(key, delay=delay, **context)
            elif callable(self.l0_reader) and int(state.get("conversation_count") or 0):
                # SQLite-backed L1 keeps evidence outside the legacy in-memory
                # buffer, so an empty buffer can still have pending work. A
                # recovered threshold runs now rather than waiting for a new
                # turn or for the full idle timeout.
                delay = (
                    0
                    if int(state["conversation_count"]) >= threshold
                    else self.l1_idle_seconds
                )
                self._schedule_idle_flush(key, delay=delay, **context)
            elif callable(self.l0_reader) and state.get("l1_backlog_pending"):
                self._schedule_idle_flush(key, delay=self.l1_idle_seconds, **context)
            if state.get("l2_pending") or state.get("l2_fire_at"):
                source = str(state.get("l2_source") or "delay-after-l1")
                if source == "max-interval":
                    delay = max(0.0, float(state.get("l2_fire_at") or 0.0) - time.time())
                    if delay == 0:
                        delay = self.l2_max_interval_seconds
                    state["l2_fire_at"] = time.time() + delay
                    self._schedule_l2_flush(key, source="max-interval", **context)
                else:
                    self._schedule_l2_flush(key, source="delay-after-l1", **context)

    def flush_session(self, session_key: str, *, run_l2: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Flush one session without disturbing other sessions."""
        state = self._scheduler_state.get(session_key)
        if state is None:
            # No turn has reached the scheduler for this key.  Legacy callers
            # may still use the provider's explicit boundary extraction path.
            return {"success": True, "skipped": True, "scheduler_seen": False, "written": []}
        self._cancel_scheduler_timer("l1", session_key)
        l1_result = self.flush_conversation(session_key, **kwargs)
        if l1_result.get("success") is False:
            return {**l1_result, "scheduler_seen": True}
        if run_l2 and state.get("l2_pending"):
            l2_result = self.run_l2(session_key, source="shutdown", **kwargs)
            return {
                **l1_result,
                "success": l2_result.get("success") is not False,
                "l2": l2_result,
                "scheduler_seen": True,
            }
        return {**l1_result, "scheduler_seen": True}

    @staticmethod
    def _conflict_prompt(matches: Sequence[tuple[Atom, Sequence[Atom]]]) -> str:
        """Build the MemoryCore batch conflict prompt in Python.

        The TypeScript implementation sends one unified candidate pool and a
        list of candidate IDs for each new memory.  Keeping this shape stable
        makes custom resolvers interchangeable with MemoryCore.
        """
        pool: dict[str, Atom] = {}
        parts: list[str] = []
        for index, (atom, candidates) in enumerate(matches):
            ids: list[str] = []
            for candidate in candidates:
                pool.setdefault(candidate.id, candidate)
                ids.append(candidate.id)
            payload = {
                "record_id": atom.id,
                "content": atom.content,
                "type": atom.type,
                "priority": atom.priority,
                "scene_name": atom.scene_name,
            }
            parts.append(
                f"### 第 {index + 1} 条新记忆 (record_id: {atom.id})\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                "【关联候选 ID】"
                + (
                    json.dumps(ids, ensure_ascii=False)
                    if ids
                    else "[]（无相似候选，直接 store）"
                )
            )
        pool_payload = [
            {
                "record_id": atom.id,
                "content": atom.content,
                "type": atom.type,
                "priority": atom.priority,
                "scene_name": atom.scene_name,
                "timestamps": atom.timestamps,
            }
            for atom in pool.values()
        ]
        if pool_payload:
            pool_section = (
                f"## 统一候选记忆池（共 {len(pool_payload)} 条已有记忆）\n\n"
                f"{json.dumps(pool_payload, ensure_ascii=False, indent=2)}"
            )
        else:
            pool_section = "## 统一候选记忆池\n\n（空，没有已有记忆，所有新记忆直接 store）"
        return (
            "**输出语言**：`merged_content` 使用与候选池中已有记忆相同的语言。\n\n"
            f"{pool_section}\n\n"
            f"{'═' * 50}\n\n"
            f"## 待判断的新记忆（共 {len(matches)} 条）\n\n"
            + "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(parts)
            + "\n\n请逐条判断并调用 l1_conflict_decisions 工具提交决策。"
            "当某条新记忆的候选列表为空时，该条直接输出 action=store。"
        )

    def _resolve_conflicts(
        self,
        matches: Sequence[tuple[Atom, Sequence[Atom]]],
        *,
        mode: str,
        session_id: str = "",
    ) -> list[dict[str, Any]]:
        """Run batch conflict detection, degrading to ``store all``.

        ``deduper`` is intentionally injectable for deployments that already
        own an LLM client.  The default path uses the same auxiliary route as
        atom extraction; any transport, parsing, or validation failure is
        non-fatal and stores every new memory, exactly as MemoryCore does.
        """
        memories = [atom for atom, _ in matches]
        store_all = [{"record_id": atom.id, "action": "store", "target_ids": []} for atom in memories]
        if (
            not self.enable_dedup
            or not matches
            or not any(candidates for _, candidates in matches)
        ):
            return store_all
        try:
            if callable(self.deduper):
                raw = self.deduper(matches, mode=mode)
            else:
                from kylin_memory.auxiliary_client import call_llm, extract_tool_call_arguments
                from kylin_memory.memory_prompts import get_conflict_detection_system_prompt
                aux = self.config.get("dedup") if isinstance(self.config.get("dedup"), Mapping) else {}
                if not aux:
                    # atom_memory is the canonical auxiliary task for the
                    # built-in provider; callers may override it with dedup.
                    try:
                        from kylin_memory.config import load_config
                        aux = (((load_config() or {}).get("auxiliary") or {}).get("atom_memory") or {})
                    except Exception:
                        aux = {}
                from kylin_memory.memory_debug import (
                    log_memory_llm_input,
                    log_memory_llm_output,
                )
                conflict_messages = [
                    {
                        "role": "system",
                        "content": get_conflict_detection_system_prompt(mode),
                    },
                    {"role": "user", "content": self._conflict_prompt(matches)},
                ]
                conflict_tools = [_l1_conflict_tool(mode)]
                conflict_tool_choice = {
                    "type": "function",
                    "function": {"name": _L1_CONFLICT_TOOL_NAME},
                }
                log_memory_llm_input(
                    "L1",
                    task="atom_memory_conflict_detection",
                    model=aux.get("model") or None,
                    api_mode=aux.get("api_mode") or None,
                    session_id=session_id,
                    messages=conflict_messages,
                    tools=conflict_tools,
                    tool_choice=conflict_tool_choice,
                    mode=mode,
                )
                response = call_llm(
                    task="atom_memory",
                    provider=aux.get("provider") or None,
                    model=aux.get("model") or None,
                    base_url=aux.get("base_url") or None,
                    api_key=aux.get("api_key") or None,
                    api_mode=aux.get("api_mode") or None,
                    messages=conflict_messages,
                    main_runtime=self.main_runtime() if callable(self.main_runtime) else self.main_runtime,
                    temperature=0,
                    max_tokens=3000,
                    timeout=float(aux.get("timeout", 120) or 120),
                    extra_body=aux.get("extra_body") if isinstance(aux.get("extra_body"), Mapping) else {},
                    tools=conflict_tools,
                    tool_choice=conflict_tool_choice,
                )
                log_memory_llm_output(
                    "L1",
                    task="atom_memory_conflict_detection",
                    model=aux.get("model") or None,
                    api_mode=aux.get("api_mode") or None,
                    session_id=session_id,
                    response=response,
                    mode=mode,
                )
                tool_args = extract_tool_call_arguments(response, _L1_CONFLICT_TOOL_NAME)
                if tool_args is None:
                    raise ValueError("L1 conflict model returned no valid tool call")
                raw = tool_args
            if isinstance(raw, Mapping):
                raw = raw.get("decisions") or raw.get("results") or []
            if isinstance(raw, str):
                text = raw.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
                start, end = text.find("["), text.rfind("]")
                raw = json.loads(text[start:end + 1]) if start >= 0 and end > start else []
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                return store_all
            allowed = {atom.id: {candidate.id for candidate in candidates} for atom, candidates in matches}
            allowed_types = CODE_TYPES if str(mode).lower() == "code" else CHAT_TYPES
            valid_actions = {"store", "skip", "update", "merge"}
            decisions: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                record_id = str(item.get("record_id") or "")
                if record_id not in allowed or record_id in seen:
                    continue
                action = str(item.get("action") or "store").lower()
                if action not in valid_actions:
                    action = "store"
                targets = [str(x) for x in (item.get("target_ids") or []) if str(x) in allowed[record_id]]
                if action in {"update", "merge"} and not targets:
                    action = "store"
                decision = {"record_id": record_id, "action": action, "target_ids": targets}
                content = item.get("merged_content")
                if isinstance(content, str) and content.strip():
                    decision["merged_content"] = content.strip()
                merged_type = str(item.get("merged_type") or "").strip().lower()
                if merged_type in allowed_types:
                    decision["merged_type"] = merged_type
                priority = item.get("merged_priority")
                if isinstance(priority, (int, float)) and not isinstance(priority, bool):
                    priority = int(priority)
                    if -1 <= priority <= 100:
                        decision["merged_priority"] = priority
                timestamps = item.get("merged_timestamps")
                if isinstance(timestamps, list) and all(isinstance(x, str) for x in timestamps):
                    decision["merged_timestamps"] = timestamps
                decisions.append(decision)
                seen.add(record_id)
            decisions.extend(item for item in store_all if item["record_id"] not in seen)
            return decisions
        except Exception:
            logger.warning("L1 batch conflict detection failed; storing all", exc_info=True)
            return store_all

    def _apply_decisions(
        self,
        matches: Sequence[tuple[Atom, Sequence[Atom]]],
        decisions: Sequence[Mapping[str, Any]],
        *,
        mode: str,
    ) -> list[Atom]:
        by_id = {atom.id: atom for atom, _ in matches}
        candidates_by_new = {
            atom.id: {candidate.id: candidate for candidate in candidates}
            for atom, candidates in matches
        }
        written: list[Atom] = []
        for decision in decisions:
            atom = by_id.get(str(decision.get("record_id") or ""))
            if atom is None or str(decision.get("action") or "store") == "skip":
                continue
            action = str(decision.get("action") or "store")
            targets = [
                str(x) for x in (decision.get("target_ids") or [])
                if str(x) in candidates_by_new.get(atom.id, {})
            ]
            final_value = atom.as_dict()
            # MemoryCore anchors every newly persisted record at writer time;
            # extractor-side timestamp suggestions are not copied verbatim.
            final_value["timestamps"] = [utc_iso()]
            if action in {"update", "merge"} and targets:
                existing = [self.atoms.get(target) for target in targets]
                existing = [item for item in existing if item is not None]
                # TencentDB keeps the freshly generated record id/evidence and
                # only carries forward the target version number.  createdAt
                # is the time this replacement record itself was created.
                final_value["version"] = max((item.version for item in existing), default=0) + 1
                final_value["content"] = decision.get("merged_content", atom.content)
                merged_type = decision.get("merged_type", atom.type)
                final_value["type"] = merged_type
                final_value["priority"] = decision.get("merged_priority", atom.priority)
                final_value["timestamps"] = decision.get("merged_timestamps") or [utc_iso()]
                # The conflict tool intentionally has no metadata argument, so
                # the new record's metadata carries over.  A cross-type merge
                # (which the prompt encourages) would then fail the per-type
                # metadata allowlist and silently drop the whole decision, so
                # narrow it to the keys the merged type actually permits.
                if merged_type != atom.type:
                    permitted = ALLOWED_METADATA.get(merged_type, set())
                    final_value["metadata"] = {
                        key: value
                        for key, value in (final_value.get("metadata") or {}).items()
                        if key in permitted
                    }
            # Re-run the complete Atom protocol validation after applying an
            # LLM decision.  A malformed merged field must never delete the
            # candidate targets; fail this one record closed and continue the
            # rest of the independent batch, matching writeMemory isolation.
            final_value.pop("createdAt", None)
            final_value.pop("updatedAt", None)
            try:
                final_atom = Atom.from_mapping(final_value, mode=mode, new_id=False)
            except (TypeError, ValueError):
                logger.warning("Rejected invalid L1 %s decision for %s", action, atom.id)
                continue
            self.atoms.upsert(
                final_atom,
                replace_ids=targets if action in {"update", "merge"} else (),
            )
            written.append(final_atom)
        return written

    def _scope_meta(self, scope: str) -> Path:
        store = self.scene_store(team_id=scope.split("|", 1)[0].removeprefix("team:"),
                                 agent_id=scope.split("|agent:", 1)[-1] if "|agent:" in scope else "default")
        return store.meta_dir

    def _checkpoint_path(self, scope: str) -> Path:
        return self._scope_meta(scope) / "checkpoint.json"

    def _load_cursor(self, key: str) -> str:
        path = self._checkpoint_path(key.split("|session:", 1)[0])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("scopes", {}).get(key, {}).get("last_extraction_updated_time", ""))
        except Exception:
            return ""

    def _scheduler_file(self, key: str) -> Path:
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return self._scheduler_dir / f"{digest}.json"

    def _load_scheduler_state(self, key: str) -> None:
        try:
            payload = json.loads(self._scheduler_file(key).read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                return
            if str(payload.get("session_key") or "") != str(key):
                return
            raw = payload.get("state")
            if not isinstance(raw, Mapping):
                return
            state = self._new_scheduler_state()
            for field in state:
                if field in raw and field != "buffer":
                    state[field] = raw[field]
            if not self.l1_warmup:
                # A checkpoint created before the small-window fix may still
                # contain warmup_threshold=1. Do not let stale scheduler state
                # split a newly configured two-conversation batch after a
                # restart.
                state["warmup_threshold"] = 0
            state["buffer"] = [dict(item) for item in (raw.get("buffer") or []) if isinstance(item, Mapping)]
            self._scheduler_state[str(key)] = state
        except (OSError, ValueError, TypeError):
            return

    def _persist_scheduler_state(self) -> None:
        with self._scheduler_guard:
            for key, state in list(self._scheduler_state.items()):
                path = self._scheduler_file(key)
                active = bool(
                    state.get("buffer")
                    or state.get("conversation_count")
                    or state.get("last_l1_cursor")
                    or state.get("l1_backlog_pending")
                    or state.get("l2_pending")
                    or state.get("l2_fire_at")
                )
                try:
                    if not active:
                        path.unlink(missing_ok=True)
                        continue
                    payload = {
                        "schema_version": 1,
                        "session_key": key,
                        "state": {
                            **{k: v for k, v in state.items() if k != "context"},
                            "context": dict(state.get("context") or {}),
                        },
                    }
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    os.replace(tmp, path)
                except (OSError, TypeError, ValueError):
                    logger.warning("memory pipeline scheduler checkpoint write failed", exc_info=True)

    def _save_cursor(self, key: str, cursor: str) -> None:
        scope = key.split("|session:", 1)[0]
        path = self._checkpoint_path(scope)
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            data = {}
        data.setdefault("schema_version", 1)
        data.setdefault("scopes", {})
        data["scopes"].setdefault(key, {})["last_extraction_updated_time"] = cursor
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _generation_log(self, layer: str, input_refs: Sequence[str], output_refs: Sequence[str], *, scope: str) -> None:
        try:
            now = time.time()
            stamp = time.strftime("%Y-%m-%d/hour=%H", time.gmtime(now))
            root = self.atoms.root / "memory-generation-logs" / "v1" / f"layer={layer}" / f"date={stamp.split('/')[0]}" / stamp.split('/')[1]
            root.mkdir(parents=True, exist_ok=True)
            log_id = secrets.token_hex(8)
            path = root / f"{int(now * 1000)}__mid={len(input_refs)}__lid={log_id}.json"
            path.write_text(json.dumps({"generation_id": log_id, "layer": layer, "scope": scope,
                "input_refs": list(input_refs), "output_refs": list(output_refs), "created_at": time.time()}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("memory generation log write failed", exc_info=True)

    @staticmethod
    def _profile_key(user_id: str) -> str:
        return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _generation_ref(value: Any) -> str:
        if isinstance(value, Mapping):
            return str(value.get("record_id") or value.get("id") or "")
        return str(value or "")

    def _scene_atom_provenance(self, scope: str) -> dict[str, set[str]]:
        """Load the durable L2 -> L1 reference map from generation logs."""
        result: dict[str, set[str]] = defaultdict(set)
        root = self.atoms.root / "memory-generation-logs" / "v1" / "layer=l2"
        if not root.exists():
            return result
        for path in sorted(root.glob("**/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if not isinstance(payload, Mapping) or str(payload.get("scope") or "") != scope:
                continue
            inputs = {
                self._generation_ref(ref)
                for ref in payload.get("input_refs", ())
                if self._generation_ref(ref)
            }
            for output in payload.get("output_refs", ()):
                filename = Path(self._generation_ref(output)).name
                if filename.endswith(".md"):
                    result[filename].update(inputs)
        return result

    def prepare_profile_sources(
        self,
        *,
        team_id: str = "",
        agent_id: str = "",
        user_id: str = "",
    ) -> dict[str, Any]:
        """Build an incremental, evidence-safe L1/L2 batch for L3 extraction.

        L3 remains a per-user encrypted profile while L2 is shared at
        ``team+agent`` scope.  A scene is therefore included only when its full
        durable provenance resolves to current L1 Atoms for this user.  L2 is
        context; the returned L1 messages remain the only evidence surface.
        """
        scope = normalize_scope(team_id, agent_id, user_id)
        profile_key = self._profile_key(user_id)
        with self._locks[scope]:
            store = self.scene_store(
                team_id=team_id, agent_id=agent_id, user_id=user_id
            )
            checkpoint_path = self._checkpoint_path(scope)
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except Exception:
                checkpoint = {}
            state = (
                checkpoint.get("l3_profiles", {}).get(profile_key, {})
                if isinstance(checkpoint, Mapping)
                else {}
            )
            if not isinstance(state, Mapping):
                state = {}
            atom_cursor = state.get("atom_cursor") or {}
            if not isinstance(atom_cursor, Mapping):
                atom_cursor = {}
            incremental = self.atoms.list_after_cursor(
                updated_after=str(atom_cursor.get("updated_at") or ""),
                id_after=str(atom_cursor.get("id") or ""),
                team_id=team_id,
                user_id=user_id,
                agent_id=agent_id,
            )

            index = store.index()
            previous_scenes = state.get("scenes") or {}
            if not isinstance(previous_scenes, Mapping):
                previous_scenes = {}
            current_scenes = {entry.filename: entry.updated for entry in index}
            candidate_scenes = {
                entry.filename
                for entry in index
                if str(previous_scenes.get(entry.filename) or "") != entry.updated
            }

            provenance = self._scene_atom_provenance(scope)
            incremental_ids = {atom.id for atom in incremental}
            for filename, atom_ids in provenance.items():
                if atom_ids & incremental_ids:
                    candidate_scenes.add(filename)

            supporting_atoms: dict[str, Atom] = {}
            scene_contents: list[tuple[str, str]] = []
            indexed_names = {entry.filename for entry in index}
            for filename in sorted(candidate_scenes & indexed_names):
                atom_ids = provenance.get(filename, set())
                if not atom_ids:
                    continue
                atoms = self.atoms.get_many(sorted(atom_ids))
                # Missing/deleted provenance or a mixed-user scene cannot be
                # safely projected into this user's encrypted L3 profile.
                if len(atoms) != len(atom_ids) or any(
                    atom.teamId != team_id
                    or atom.agentId != agent_id
                    or atom.userId != user_id
                    for atom in atoms
                ):
                    continue
                try:
                    content = store.read(filename)
                except (OSError, ValueError):
                    continue
                scene_contents.append((filename, content))
                supporting_atoms.update((atom.id, atom) for atom in atoms)

            evidence_atoms = {
                atom.id: atom for atom in (*incremental, *supporting_atoms.values())
            }
            ordered_atoms = sorted(
                evidence_atoms.values(), key=lambda atom: (atom.updatedAt, atom.id)
            )
            messages = [
                {
                    "role": "assistant",
                    "content": content,
                    "memory_layer": "l2",
                    "source_ref": filename,
                }
                for filename, content in scene_contents
            ]
            messages.extend(
                {
                    "role": "user",
                    "content": atom.content,
                    "memory_layer": "l1",
                    "source_ref": atom.id,
                }
                for atom in ordered_atoms
            )

            if incremental:
                last_atom = incremental[-1]
                next_atom_cursor = {
                    "updated_at": last_atom.updatedAt,
                    "id": last_atom.id,
                }
            else:
                next_atom_cursor = {
                    "updated_at": str(atom_cursor.get("updated_at") or ""),
                    "id": str(atom_cursor.get("id") or ""),
                }
            fingerprint_payload = {
                "l1": [
                    [atom.id, atom.version, atom.updatedAt, atom.content]
                    for atom in ordered_atoms
                ],
                "l2": [
                    [filename, current_scenes.get(filename, ""), content]
                    for filename, content in scene_contents
                ],
            }
            fingerprint = hashlib.sha256(
                json.dumps(
                    fingerprint_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            input_refs = [f"l2:{filename}" for filename, _ in scene_contents]
            input_refs.extend(f"l1:{atom.id}" for atom in ordered_atoms)
            return {
                "scope": scope,
                "profile_key": profile_key,
                "messages": messages,
                "fingerprint": fingerprint,
                "input_refs": input_refs,
                "checkpoint": {
                    "atom_cursor": next_atom_cursor,
                    "scenes": current_scenes,
                },
            }

    def acknowledge_profile_sources(
        self,
        batch: Mapping[str, Any],
        *,
        changed: bool = False,
        output_refs: Sequence[str] = (),
    ) -> None:
        """Advance the L3 checkpoint after a successful or no-change run."""
        scope = str(batch.get("scope") or "")
        profile_key = str(batch.get("profile_key") or "")
        next_state = batch.get("checkpoint")
        if not scope or not profile_key or not isinstance(next_state, Mapping):
            raise ValueError("invalid L3 source checkpoint")
        with self._locks[scope]:
            path = self._checkpoint_path(scope)
            try:
                data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except Exception:
                data = {}
            data.setdefault("schema_version", 1)
            data.setdefault("l3_profiles", {})
            data["l3_profiles"][profile_key] = {
                "atom_cursor": dict(next_state.get("atom_cursor") or {}),
                "scenes": dict(next_state.get("scenes") or {}),
                "last_profile_time": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "last_fingerprint": str(batch.get("fingerprint") or ""),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, path)
        if changed and output_refs:
            self._generation_log(
                "l3",
                [str(ref) for ref in batch.get("input_refs", ())],
                output_refs,
                scope=scope,
            )

    def scene_store(self, *, team_id: str = "", agent_id: str = "", user_id: str = "", global_compat: bool = False) -> ScenarioStore:
        scope = normalize_scope(team_id, agent_id, user_id, global_compat=global_compat)
        if scope not in self._scenes:
            self._scenes[scope] = ScenarioStore(self.root, scope=scope, max_scenes=self.max_scenes, prompt_mode=self.prompt_mode)
        return self._scenes[scope]

    def ingest(self, messages: Sequence[Mapping[str, Any]], *, session_key: str = "", session_id: str = "", task_id: str = "", team_id: str = "", user_id: str = "", agent_id: str = "", mode: str = "chat") -> dict[str, Any]:
        """Extract and persist one bounded L1 batch.

        A supplied extractor may return ``{"memories": [...]}``, a list, or
        Atom instances.  Invalid/low-quality entries are rejected individually.
        """
        clean = [dict(m) for m in messages if isinstance(m, Mapping) and should_extract_l1(m.get("content", ""))]
        def _message_id(m: Mapping[str, Any]) -> int | None:
            try:
                raw = m.get("id", m.get("message_id"))
                return int(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None
        clean = [m for m in clean if _message_id(m) is not None]
        if not clean:
            return {"success": True, "written": [], "skipped": True}
        raw = self.extractor(clean, mode=mode, session_id=session_id) if self.extractor else []
        if isinstance(raw, Mapping): raw = raw.get("memories", raw.get("atoms", []))
        written: list[str] = []
        accepted: list[Atom] = []
        candidate_matches: list[tuple[Atom, Sequence[Atom]]] = []
        for item in list(raw or [])[: int(self.config.get("max_memories_per_job", 20))]:
            try:
                value = item.as_dict() if isinstance(item, Atom) else dict(item)
                value.update(sessionKey=session_key, sessionId=session_id, taskId=task_id, teamId=team_id, userId=user_id, agentId=agent_id)
                value.setdefault("source_message_ids", [_message_id(m) for m in clean if _message_id(m) is not None])
                atom = Atom.from_mapping(value, mode=mode)
                # Source ids must come from this batch; background/unknown ids
                # are rejected rather than silently creating unverifiable L1.
                allowed = {_message_id(m) for m in clean if _message_id(m) is not None}
                user_evidence = {
                    _message_id(m)
                    for m in clean
                    if str(m.get("role") or "").lower() == "user"
                    and _message_id(m) is not None
                }
                if (
                    not set(atom.source_message_ids) & user_evidence
                    or not self.atoms.validate_source_ids(atom.source_message_ids, messages=clean, session_id=session_id, team_id=team_id, user_id=user_id, agent_id=agent_id, task_id=task_id)
                    or not set(atom.source_message_ids) <= allowed
                ):
                    continue
                # Replaying a session boundary must be idempotent.  The
                # current Atom row is the dedupe index; JSONL still retains
                # every accepted version for audit/recovery.
                accepted.append(atom)
                candidates = self.atoms.search(
                    atom.content, limit=self.conflict_recall_top_k,
                    team_id=team_id, user_id=user_id,
                    agent_id=agent_id, session_id=session_id, task_id=task_id,
                )
                candidate_matches.append((atom, candidates))
            except Exception as exc:
                logger.debug("L1 atom rejected: %s", exc)
        if accepted:
            decisions = self._resolve_conflicts(
                candidate_matches, mode=mode, session_id=session_id
            )
            stored = self._apply_decisions(candidate_matches, decisions, mode=mode)
            written.extend(atom.id for atom in stored)
        return {"success": True, "written": written, "skipped": not bool(written)}

    def persist_candidates(
        self,
        candidates: Sequence[Mapping[str, Any] | Atom],
        messages: Sequence[Mapping[str, Any]],
        *,
        session_key: str = "", session_id: str = "", task_id: str = "",
        team_id: str = "", user_id: str = "", agent_id: str = "",
        mode: str = "chat",
    ) -> dict[str, Any]:
        """Persist already-extracted candidates from a legacy/provider adapter.

        This is intentionally a narrow bridge: evidence is checked against the
        supplied L0 batch before any SQLite/JSONL write, and an equivalent
        current row is skipped to make retries idempotent.
        """
        clean = [dict(m) for m in messages if isinstance(m, Mapping)]
        def _message_id(m: Mapping[str, Any]) -> int | None:
            try:
                raw = m.get("id", m.get("message_id"))
                return int(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None
        allowed = {_message_id(m) for m in clean if _message_id(m) is not None}
        user_evidence = {
            _message_id(m)
            for m in clean
            if str(m.get("role") or "").lower() == "user"
            and _message_id(m) is not None
        }
        written: list[str] = []
        accepted: list[Atom] = []
        candidate_matches: list[tuple[Atom, Sequence[Atom]]] = []
        for item in candidates:
            try:
                value = item.as_dict() if isinstance(item, Atom) else dict(item)
                value.update(sessionKey=session_key, sessionId=session_id, taskId=task_id,
                             teamId=team_id, userId=user_id, agentId=agent_id)
                value.setdefault("source_message_ids", sorted(allowed))
                atom = Atom.from_mapping(value, mode=mode)
                if (
                    not atom.source_message_ids
                    or not set(atom.source_message_ids) <= allowed
                    or not set(atom.source_message_ids) & user_evidence
                ):
                    continue
                if not self.atoms.validate_source_ids(atom.source_message_ids, messages=clean,
                        session_id=session_id, team_id=team_id, user_id=user_id,
                        agent_id=agent_id, task_id=task_id):
                    continue
                accepted.append(atom)
                candidate_matches.append((
                    atom,
                    self.atoms.search(atom.content, limit=self.conflict_recall_top_k,
                                      team_id=team_id,
                                      user_id=user_id, agent_id=agent_id,
                                      session_id=session_id, task_id=task_id),
                ))
            except Exception as exc:
                logger.debug("L1 mirror candidate rejected: %s", exc)
        if accepted:
            decisions = self._resolve_conflicts(
                candidate_matches, mode=mode, session_id=session_id
            )
            stored = self._apply_decisions(candidate_matches, decisions, mode=mode)
            written.extend(atom.id for atom in stored)
        return {"success": True, "written": written, "skipped": not bool(written)}

    def consolidate(self, *, team_id: str = "", agent_id: str = "", user_id: str = "", session_id: str = "", scene_name: str = "", summary: str = "", body: str | None = None) -> dict[str, Any]:
        scope = normalize_scope(team_id, agent_id, user_id)
        with self._locks[scope]:
            cursor_key = f"{scope}|session:{session_id}"
            cursor = self._cursors.get(cursor_key) or self._load_cursor(cursor_key)
            atoms = self.atoms.list_updated(
                updated_after=cursor,
                session_id=session_id,
                team_id=team_id,
                user_id=user_id,
                agent_id=agent_id,
            )
            if not atoms:
                return {"success": True, "skipped": True, "changed": [], "latestCursor": cursor}
            store = self.scene_store(team_id=team_id, agent_id=agent_id, user_id=user_id)
            # Snapshot the bounded scene directory before applying an LLM
            # decision.  This is the local equivalent of MemoryCore's L2
            # backup/restore phase and prevents partial MERGE/soft-delete
            # mutations from leaking through a failed job.
            # Include soft-deleted files as well as active index entries.  A
            # MERGE is allowed to mark old files ``[DELETED]`` and a failed
            # retry must restore those physical files byte-for-byte.
            snapshot: dict[str, str] = {}
            for path in store.scene_dir.glob("*.md"):
                try:
                    snapshot[path.name] = path.read_text(encoding="utf-8")
                except OSError:
                    continue
            if body is None and callable(self.consolidator):
                try:
                    try:
                        generated = self.consolidator(
                            atoms,
                            store=store,
                            scene_name=scene_name or atoms[0].scene_name or "general",
                            summary=summary,
                            mode=self.prompt_mode,
                            session_id=session_id,
                        )
                    except TypeError as exc:
                        # Keep third-party/legacy consolidators compatible with
                        # the pre-session_id callable contract.
                        if "session_id" not in str(exc):
                            raise
                        generated = self.consolidator(
                            atoms,
                            store=store,
                            scene_name=scene_name or atoms[0].scene_name or "general",
                            summary=summary,
                            mode=self.prompt_mode,
                        )
                    # Handle both single transaction and multi-scene transactions
                    transactions = []
                    if isinstance(generated, list):
                        transactions = generated
                    elif isinstance(generated, Mapping):
                        transactions = [generated]

                    processed_transactions = []
                    for gen in transactions:
                        if not isinstance(gen, Mapping):
                            continue
                        explicit_action = any(
                            key in gen for key in ("action", "target_files", "targets", "delete_files")
                        )
                        action = str(gen.get("action") or "update").strip().lower()
                        if action not in {"create", "update", "merge"}:
                            action = "update"
                        target_files = gen.get("target_files") or gen.get("targets") or ()
                        if isinstance(target_files, str):
                            target_files = [target_files]
                        target_files = [str(x) for x in target_files if str(x).strip()]
                        proposed_scene_name = str(gen.get("scene_name") or "").strip()
                        txn_scene_name = proposed_scene_name or str(
                            scene_name or atoms[0].scene_name or "general"
                        )
                        # Older/custom consolidators may omit scene_name on an
                        # UPDATE. Preserve the target name in that compatibility
                        # case, but honor an explicit resulting semantic name so
                        # ScenarioStore can rename an evolved scene.
                        if action == "update" and target_files and not proposed_scene_name:
                            txn_scene_name = Path(target_files[0]).stem
                        txn_summary = str(gen.get("summary") or summary)
                        txn_body = gen.get("body")
                        delete_files = gen.get("delete_files", ())
                        if isinstance(delete_files, str):
                            delete_files = [delete_files]
                        if action == "merge":
                            delete_files = list(delete_files or ()) + target_files
                        processed_transactions.append({
                            **dict(gen),
                            "_explicit_action": explicit_action,
                            "action": action,
                            "target_files": target_files,
                            "delete_files": list(delete_files or ()),
                            "scene_name": txn_scene_name,
                            "summary": txn_summary,
                            "body": txn_body,
                        })
                    generated = processed_transactions
                except Exception:
                    logger.warning("LLM L2 consolidation failed; cursor retained for retry", exc_info=True)
                    return {
                        "success": False,
                        "failed": True,
                        "reason": "l2_consolidation_failed",
                        "changed": [],
                        "latestCursor": cursor,
                    }
            try:
                if isinstance(locals().get("generated"), list) and len(generated) > 1:
                    # Multi-scene transactions: apply each one
                    all_changed = []
                    for txn in generated:
                        if not isinstance(txn, Mapping) or not txn.get("_explicit_action"):
                            continue
                        # Use the original L1 scene_name for filtering atoms, not the new L2 scene_name
                        original_scene_name = txn.get("_original_scene_name")
                        txn_atoms = [a for a in atoms if a.scene_name == original_scene_name] if original_scene_name else atoms
                        if not txn_atoms:
                            continue
                        result = store.apply_action(
                            txn_atoms,
                            action=str(txn.get("action") or "update"),
                            scene_name=txn.get("scene_name") or atoms[0].scene_name or "general",
                            target_files=txn.get("target_files") or (),
                            delete_files=txn.get("delete_files") or (),
                            summary=txn.get("summary") or summary,
                            body=txn.get("body"),
                        )
                        all_changed.extend(result.get("changed", []))
                    result = {"changed": all_changed, "latestCursor": atoms[-1].updatedAt}
                elif isinstance(locals().get("generated"), list) and len(generated) == 1:
                    # Single transaction in list format
                    txn = generated[0]
                    if isinstance(txn, Mapping) and txn.get("_explicit_action"):
                        result = store.apply_action(
                            atoms,
                            action=str(txn.get("action") or "update"),
                            scene_name=txn.get("scene_name") or scene_name or atoms[0].scene_name or "general",
                            target_files=txn.get("target_files") or (),
                            delete_files=txn.get("delete_files") or (),
                            summary=txn.get("summary") or summary,
                            body=txn.get("body"),
                        )
                    else:
                        result = store.consolidate(
                            atoms,
                            scene_name=scene_name or atoms[0].scene_name or "general",
                            summary=summary,
                            body=body,
                        )
                else:
                    # Empty list or no generated: fallback to default consolidation
                    result = store.consolidate(
                        atoms,
                        scene_name=scene_name or atoms[0].scene_name or "general",
                        summary=summary,
                        body=body,
                    )
            except Exception:
                logger.warning("L2 scene write failed; restoring snapshot", exc_info=True)
                self._restore_scenes(store, snapshot)
                return {
                    "success": False,
                    "failed": True,
                    "reason": "l2_write_failed",
                    "changed": [],
                    "latestCursor": cursor,
                }
            if result.get("reason") == "scene_limit":
                # MemoryCore treats a capacity violation as an unsuccessful
                # consolidation that must be retried with a MERGE; consuming
                # the L1 cursor here would lose that opportunity.
                self._restore_scenes(store, snapshot)
                return {
                    "success": False,
                    "failed": True,
                    "reason": "scene_limit",
                    "changed": [],
                    "latestCursor": cursor,
                }
            latest = result.get("latestCursor", atoms[-1].updatedAt)
            self._cursors[cursor_key] = latest
            self._save_cursor(cursor_key, latest)
            if result.get("changed"):
                self._generation_log("l2", [a.id for a in atoms], result.get("changed", []), scope=scope)
            return {"success": True, **result}

    @staticmethod
    def _restore_scenes(store: ScenarioStore, snapshot: Mapping[str, str]) -> None:
        """Restore a pre-L2 scene snapshot without advancing pipeline state."""
        current = {path.name for path in store.scene_dir.glob("*.md")}
        original = set(snapshot)
        for filename in current - original:
            path = store.scene_dir / filename
            try:
                path.unlink()
            except OSError:
                pass
        for filename, content in snapshot.items():
            try:
                store.write(filename, content)
            except Exception:
                logger.error("Could not restore L2 scene %s", filename, exc_info=True)
        store.rebuild_index()

    def clean(self, *, before: str | None = None, max_records: int = 20) -> dict[str, int]:
        """Explicit conservative cleaner for old JSONL shards and SQLite rows."""
        cutoff = before or time.strftime("%Y-%m-%d", time.gmtime())
        removed_files = 0
        for path in self.atoms.records_dir.glob("*.jsonl"):
            if path.stem < cutoff:
                try:
                    path.unlink(); removed_files += 1
                except OSError:
                    logger.warning("L1 cleaner could not remove %s", path, exc_info=True)
        # Keep small stores intact as required by the architecture.
        count = int(self.atoms._conn.execute("SELECT COUNT(*) FROM l1_records").fetchone()[0])
        removed_rows = 0
        if count > max_records:
            with self.atoms._conn:
                rows = self.atoms._conn.execute("SELECT id FROM l1_records WHERE updated_time < ?", (cutoff,)).fetchall()
                if len(rows) <= count * 0.8:
                    for row in rows:
                        self.atoms._conn.execute("DELETE FROM l1_fts WHERE id=?", (row[0],))
                        self.atoms._conn.execute("DELETE FROM l1_fts_trigram WHERE id=?", (row[0],))
                        self.atoms._delete_vector(row[0])
                        self.atoms._conn.execute("DELETE FROM l1_records WHERE id=?", (row[0],))
                    removed_rows = len(rows)
        return {"removed_files": removed_files, "removed_rows": removed_rows}

    def delete_session(self, session_id: str, *, team_id: str = "", user_id: str = "", agent_id: str = "") -> int:
        """Remove current L1 rows derived from one L0 session.

        The append-only JSONL audit is intentionally left intact; deleting the
        current SQLite/FTS/vector projections makes the session unavailable to
        recall while preserving an operator-recoverable audit trail.
        """
        if not session_id:
            return 0
        rows = self.atoms._conn.execute(
            "SELECT id FROM l1_records WHERE session_id=?"
            + (" AND team_id=?" if team_id else "")
            + (" AND user_id=?" if user_id else "")
            + (" AND agent_id=?" if agent_id else ""),
            tuple(x for x in (session_id, team_id if team_id else None, user_id if user_id else None, agent_id if agent_id else None) if x is not None),
        ).fetchall()
        removed = 0
        for row in rows:
            removed += int(self.atoms.delete(str(row[0])))
        return removed

    def recall(self, query: str, *, team_id: str = "", user_id: str = "", agent_id: str = "", session_id: str = "", task_id: str = "", limit: int = 5) -> list[Atom]:
        return self.atoms.search(query, limit=limit, team_id=team_id, user_id=user_id, agent_id=agent_id, session_id=session_id, task_id=task_id)

    def close(self) -> None:
        with self._scheduler_guard:
            if self._closed or self._closing:
                return
            self._closing = True
            # Graceful shutdown: MemoryCore flushes L1 then L2.  Failures leave
            # their buffers/cursors checkpointed for the next initialization.
            for timer in list(self._scheduler_timers.values()):
                try:
                    timer.cancel()
                except Exception:
                    pass
            self._scheduler_timers.clear()
            pending = [
                (key, dict(state.get("context") or {}))
                for key, state in self._scheduler_state.items()
            ]
        # Wait for any timer callback already inside a layer queue, then keep
        # both queues blocked until SQLite has been closed and _closed is set.
        with self._l1_job_lock, self._l2_job_lock:
            for key, context in pending:
                state = self._scheduler_state.get(key) or {}
                try:
                    if state.get("buffer") or (
                        callable(self.l0_reader)
                        and (
                            int(state.get("conversation_count") or 0)
                            or state.get("l1_backlog_pending")
                        )
                    ):
                        self.flush_conversation(key, **context)
                    if state.get("l2_pending"):
                        self.run_l2(key, source="shutdown", **context)
                except Exception:
                    logger.warning("memory pipeline shutdown flush failed for %s", key, exc_info=True)
            self._persist_scheduler_state()
            with self._scheduler_guard:
                self._closed = True
            self.atoms.close()


__all__ = ["MemoryPipelineManager"]
