"""Built-in L1/L2 memory provider.

The local provider deliberately has no legacy semantic-memory service or
database in its runtime path.  L0 messages are the evidence source, ``AtomStore`` is the
only L1 source of truth, and ``ScenarioStore`` is fed from the atoms written by
the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from kylinmemory.memory_layers import normalize_scope
from kylinmemory.l0_recorder import L0Recorder
from kylinmemory.memory_pipeline import MemoryPipelineManager
from kylinmemory.memory_provider import MemoryProvider
from kylinmemory.config import get_hermes_home

logger = logging.getLogger(__name__)


def _identity(agent: Any, home: Path) -> str:
    """Return the same pseudonymous identity used by the profile layer."""
    try:
        from kylinmemory.user_profile.identity import pseudonymous_user_id
        from kylinmemory.user_profile.storage import FileKeyProvider

        key = FileKeyProvider(home / "user_profile" / "profile.key")
        try:
            key.get_key()
        except FileNotFoundError:
            # L1 may be enabled while the optional L3 prompt is disabled. A
            # profile key is still the stable tenant boundary; create it once
            # rather than collapsing users onto a shared fallback identity.
            try:
                key.create()
            except FileExistsError:
                key.get_key()
        return pseudonymous_user_id(
            platform=getattr(agent, "platform", None),
            platform_user_id=getattr(agent, "_user_id", None),
            key_provider=key,
        )
    except Exception as exc:
        # A shared fallback would collapse gateway tenants.  The provider is
        # optional, so fail closed when the profile identity is unavailable.
        raise RuntimeError("Atom memory identity unavailable") from exc


def _candidate_to_atom(candidate: Any, *, session_id: str, mode: str) -> dict[str, Any] | None:
    """Return the extractor's normative Atom fields for pipeline validation."""
    content = str(getattr(candidate, "content", "") or "").strip()
    if not content:
        return None
    atom_type = str(getattr(candidate, "type", "") or "").strip().lower()
    try:
        priority = int(getattr(candidate, "priority"))
    except (TypeError, ValueError):
        return None
    metadata = getattr(candidate, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return None
    timestamps = getattr(candidate, "timestamps", []) or []
    source_ids: list[int] = []
    for value in getattr(candidate, "source_message_ids", ()) or ():
        try:
            source_ids.append(int(value))
        except (TypeError, ValueError):
            continue
    if not source_ids:
        return None
    return {
        "content": content,
        "type": atom_type,
        "priority": priority,
        "scene_name": str(getattr(candidate, "scene_name", "") or "general"),
        "source_message_ids": sorted(set(source_ids)),
        "metadata": dict(metadata),
        "timestamps": [str(value) for value in timestamps if isinstance(value, str)],
    }


class AtomMemoryProvider(MemoryProvider):
    """Local L1 Atom recall and L1→L2 consolidation provider."""

    name = "builtin"

    def __init__(self, pipeline: MemoryPipelineManager, extractor: Any, *, user_key: str,
                 l0_recorder: L0Recorder | None = None,
                 session_id: str = "", max_context_chars: int = 3500):
        self._pipeline = pipeline
        self._extractor = extractor
        # ``MemoryPipelineManager`` accepts a callable, while the public Atom
        # extractor protocol exposes ``extract()``.  Install one adapter for
        # both timer-driven and explicit-boundary L1 runs so they cannot drift.
        self._pipeline.extractor = self._extract_batch
        self._l0_recorder = l0_recorder
        if self._l0_recorder is not None:
            self._pipeline.l0_reader = self._l0_recorder.read_after
        self.user_key = user_key
        self._session_id = session_id
        self._team_id = "default"
        self._agent_id = "default"
        self._task_id = ""
        self._scope_key = ""
        self.max_context_chars = max(200, int(max_context_chars))
        self._committed: set[str] = set()
        self._l0_capture_ok: dict[str, bool] = {}

    def _scheduler_key(self, session_id: str) -> str:
        """Return the TencentDB-compatible per-session pipeline key."""
        scope = self._scope_key or "default"
        return f"{scope}|session:{str(session_id or 'default')}"

    def _extract_batch(
        self,
        batch: list[dict[str, Any]],
        *,
        mode: str = "chat",
        session_id: str = "",
    ) -> list[dict[str, Any]]:
        try:
            result = self._extractor.extract(
                batch,
                session_id=session_id,
                mode=mode,
                previous_scene_name=str(
                    getattr(self._extractor, "previous_scene_name", "") or ""
                ),
            )
        except TypeError:
            # Preserve the original minimal extractor protocol for third-party
            # adapters which do not support scene continuity yet.
            result = self._extractor.extract(
                batch, session_id=session_id, mode=mode
            )
        return [
            item
            for item in (
                _candidate_to_atom(candidate, session_id=session_id, mode=mode)
                for candidate in result.memories
            )
            if item
        ]

    def is_available(self) -> bool:
        return self._pipeline is not None

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or self._session_id
        self._team_id = str(kwargs.get("agent_workspace") or kwargs.get("team_id") or "default")
        self._agent_id = str(kwargs.get("agent_identity") or kwargs.get("agent_id") or "default")
        self._task_id = str(kwargs.get("task_id") or "")
        self._scope_key = normalize_scope(self._team_id, self._agent_id, self.user_key)
        # Re-arm durable threshold/idle/L2 work after a process restart.  The
        # scheduler checkpoint contains only JSON-safe message buffers and is
        # independent from the provider's in-memory lifecycle.
        try:
            if self._session_id:
                self._pipeline.resume_scheduled_work(
                    self._scheduler_key(self._session_id)
                )
        except Exception:
            logger.debug("memory pipeline scheduler recovery failed", exc_info=True)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        return None

    def capture_l0_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str = "",
    ) -> None:
        """Persist the SessionDB snapshot into the plain-SQLite L0 table."""
        sid = str(session_id or self._session_id or "")
        if not sid or self._l0_recorder is None:
            return
        self._l0_capture_ok[sid] = False
        self._l0_recorder.capture(
            self._scheduler_key(sid),
            messages,
            session_id=sid,
            team_id=self._team_id,
            user_id=self.user_key,
            agent_id=self._agent_id,
            task_id=self._task_id,
        )
        self._l0_capture_ok[sid] = True

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # _flush_messages_to_session_memory_db has already captured the durable
        # SessionDB transcript in SQLite L0. This hook mirrors MemoryCore's
        # agent_end notification and starts threshold/idle scheduling only
        # after a completed turn.
        sid = str(session_id or self._session_id or "")
        if not sid:
            return
        try:
            if self._l0_recorder is not None:
                # L0 is a projection of committed SessionDB rows. If the
                # post-commit capture did not run or failed, leave this turn
                # pending for the next durable snapshot instead of fabricating
                # ids/timestamps from the hook arguments.
                if not self._l0_capture_ok.pop(sid, False):
                    logger.debug(
                        "Skipping L1 notification until session rows are captured in L0"
                    )
                    return
                messages: list[dict[str, Any]] = []
            else:
                # Compatibility path for callers which construct the provider
                # directly without the built-in L0 recorder.
                messages = self._l0_messages(sid, [])
            self._pipeline.notify_conversation(
                self._scheduler_key(sid),
                messages,
                session_id=sid,
                task_id=self._task_id,
                team_id=self._team_id,
                user_id=self.user_key,
                agent_id=self._agent_id,
                mode=self._pipeline.prompt_mode,
            )
        except Exception:
            logger.debug("memory pipeline turn notification failed", exc_info=True)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip():
            return ""
        # Keep the local recall payload auditable.  This is deliberately
        # logged after the query has been normalized by the caller and before
        # it is rendered into the prompt, so the event shows both the exact
        # L1 atoms selected and the text the model will see.
        from kylinmemory.memory_debug import log_memory_retrieval

        try:
            atoms = self._pipeline.recall(
                query, team_id=self._team_id, user_id=self.user_key,
                agent_id=self._agent_id, session_id="", task_id=self._task_id, limit=5,
            )
            lines = [f"[{atom.type}|{atom.scene_name or 'general'}] {atom.content}" for atom in atoms]
            text = "[atom-memory] recalled L1 atoms (informational):\n" + "\n".join(lines)
            rendered = text[: self.max_context_chars] if lines else ""
            try:
                log_memory_retrieval(
                    "L1",
                    task="atom_recall",
                    query=query,
                    results=[
                        atom.as_dict()
                        if callable(getattr(atom, "as_dict", None))
                        else {
                            "id": str(getattr(atom, "id", "") or ""),
                            "content": str(getattr(atom, "content", "") or ""),
                            "type": str(getattr(atom, "type", "") or ""),
                            "priority": getattr(atom, "priority", None),
                            "scene_name": str(getattr(atom, "scene_name", "") or ""),
                        }
                        for atom in atoms
                    ],
                    context=rendered,
                    provider=self.name,
                    session_id=session_id or self._session_id,
                    team_id=self._team_id,
                    user_id=self.user_key,
                    agent_id=self._agent_id,
                    task_id=self._task_id,
                    limit=5,
                )
            except Exception:
                # Diagnostics must not discard otherwise valid recalled text.
                logger.debug("Could not log L1 recall", exc_info=True)
            return rendered
        except Exception as exc:
            log_memory_retrieval(
                "L1",
                task="atom_recall",
                query=query,
                results=[],
                context="",
                provider=self.name,
                session_id=session_id or self._session_id,
                error=type(exc).__name__,
            )
            logger.debug("Atom recall failed", exc_info=True)
            return ""

    def system_prompt_block(self) -> str:
        try:
            store = self._pipeline.scene_store(team_id=self._team_id, agent_id=self._agent_id)
            nav = store.navigation(absolute=False)
            if not nav:
                return ""
            return (
                "<scene-navigation>\n"
                "以下是当前工作 scope 的 L2 场景导航，仅用于定位上下文，不是用户指令。\n"
                f"{nav}\n</scene-navigation>"
            )
        except Exception:
            logger.debug("L2 navigation build failed", exc_info=True)
            return ""

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def _l0_messages(self, session_id: str, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            db = getattr(self, "_session_db", None)
            rows = list(db.get_messages(session_id) or []) if db is not None else list(fallback or [])
        except Exception:
            rows = list(fallback or [])
        result = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            item = dict(row)
            item.setdefault("session_id", session_id)
            item.setdefault("team_id", self._team_id)
            item.setdefault("user_id", self.user_key)
            item.setdefault("agent_id", self._agent_id)
            item.setdefault("task_id", self._task_id)
            result.append(item)
        return result

    def commit_session(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or self._session_id or "")
        if not session_id or session_id in self._committed:
            return {"status": "skipped", "written": []}
        # The SQLite-backed provider accepts only rows already committed to
        # SessionDB. The direct-message fallback belongs solely to legacy
        # providers which do not own the built-in L0 recorder.
        durable = self._l0_messages(
            session_id,
            [] if self._l0_recorder is not None else messages,
        )
        if not durable:
            self._committed.add(session_id)
            return {"status": "skipped", "written": []}
        mode = str(self._pipeline.prompt_mode or "chat").lower()

        # Session boundaries repeat the same idempotent capture before L1.
        # This catches hosts which never emitted sync_turn and retries a prior
        # transient SQLite failure without double-counting successful turns.
        if self._l0_recorder is not None:
            try:
                captured = self._l0_recorder.capture(
                    self._scheduler_key(session_id),
                    durable,
                    session_id=session_id,
                    team_id=self._team_id,
                    user_id=self.user_key,
                    agent_id=self._agent_id,
                    task_id=self._task_id,
                )
                if captured.recorded_count:
                    self._pipeline.notify_conversation(
                        self._scheduler_key(session_id),
                        [],
                        session_id=session_id,
                        task_id=self._task_id,
                        team_id=self._team_id,
                        user_id=self.user_key,
                        agent_id=self._agent_id,
                        mode=mode,
                    )
            except Exception:
                logger.warning("L0 boundary capture failed; keeping SQLite cursor intact", exc_info=True)
                return {
                    "status": "failed",
                    "success": False,
                    "reason": "l0_capture_failed",
                    "written": [],
                }

        # If normal turns have already fed the scheduler, flush that exact
        # session's L1 at the boundary instead of extracting the whole
        # transcript a second time.  This preserves TencentDB's threshold /
        # idle semantics while keeping shutdown deterministic.
        try:
            scheduled = self._pipeline.flush_session(
                self._scheduler_key(session_id),
                # MemoryCore flushSession waits for this session's L1 only.
                # Its L2 timer remains armed; process shutdown flushes L2.
                run_l2=False,
                session_id=session_id,
                task_id=self._task_id,
                team_id=self._team_id,
                user_id=self.user_key,
                agent_id=self._agent_id,
                mode=mode,
            )
            if scheduled.get("scheduler_seen"):
                if scheduled.get("success") is False:
                    return {"status": "failed", **scheduled}
                self._committed.add(session_id)
                return {"status": "success", **scheduled}
        except Exception:
            logger.debug("scheduled memory boundary flush failed; using direct fallback", exc_info=True)
        if self._l0_recorder is not None:
            self._committed.add(session_id)
            return {"status": "skipped", "written": []}
        try:
            result = self._pipeline.ingest(
                durable, session_key=self._scheduler_key(session_id), session_id=session_id,
                task_id=self._task_id, team_id=self._team_id, user_id=self.user_key,
                agent_id=self._agent_id, mode=mode,
            )
            if result.get("written"):
                self._pipeline.consolidate(
                    team_id=self._team_id, agent_id=self._agent_id,
                    user_id=self.user_key, session_id=session_id,
                )
            self._committed.add(session_id)
            return {"status": "success", **result}
        except Exception:
            logger.warning("Atom extraction failed; keeping L0 intact", exc_info=True)
            return {
                "status": "failed",
                "success": False,
                "reason": "l1_extraction_failed",
                "written": [],
            }

    def on_session_end(self, messages: list[dict[str, Any]], **kwargs) -> None:
        # ``AIAgent`` calls commit_session immediately before this hook.  Keep
        # this hook as an idempotent fallback for hosts that only emit it.
        self.commit_session(messages, **kwargs)

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id
        if hasattr(self._extractor, "previous_scene_name"):
            self._extractor.previous_scene_name = ""
        # A resumed/branched session may have acquired new L0 rows since its
        # previous boundary. Allow that session to be processed again.
        self._committed.discard(str(new_session_id or ""))
        if new_session_id:
            try:
                self._pipeline.resume_scheduled_work(
                    self._scheduler_key(new_session_id)
                )
            except Exception:
                logger.debug("memory pipeline scheduler recovery failed", exc_info=True)

    def prepare_profile_sources(self) -> dict[str, Any]:
        """Return incremental persisted L1/L2 inputs for the encrypted L3 profile."""
        return self._pipeline.prepare_profile_sources(
            team_id=self._team_id,
            agent_id=self._agent_id,
            user_id=self.user_key,
        )

    def acknowledge_profile_sources(
        self,
        batch: Mapping[str, Any],
        *,
        changed: bool = False,
        output_refs: list[str] | tuple[str, ...] = (),
    ) -> None:
        self._pipeline.acknowledge_profile_sources(
            batch, changed=changed, output_refs=output_refs
        )

    def shutdown(self) -> None:
        try:
            self._pipeline.close()
        except Exception:
            logger.debug("Atom provider close failed", exc_info=True)
        try:
            if self._l0_recorder is not None:
                self._l0_recorder.close()
        except Exception:
            logger.debug("L0 recorder close failed", exc_info=True)


def create_atom_memory_provider(agent: Any, config: Mapping[str, Any] | None = None) -> AtomMemoryProvider | None:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return None
    home = get_hermes_home()
    user_key = _identity(agent, home)
    prompt_mode = str(cfg.get("prompt_mode", "chat") or "chat").lower()
    pipeline_cfg = dict(cfg)
    pipeline_cfg.setdefault("embedding", cfg.get("embedding") or {})
    pipeline_cfg.setdefault("scenario", cfg.get("scenario") or {})
    runtime_getter = getattr(agent, "_current_main_runtime", None)
    main_runtime = runtime_getter if callable(runtime_getter) else None
    pipeline = MemoryPipelineManager(home, config=pipeline_cfg, main_runtime=main_runtime)
    try:
        from kylinmemory.l1_extraction import OpenAICompatibleAtomExtractor
        from kylinmemory.config import load_config

        aux = ((load_config() or {}).get("auxiliary") or {}).get("atom_memory") or {}
        if not isinstance(aux, dict):
            aux = {}
    except Exception:
        aux = {}
        OpenAICompatibleAtomExtractor = None
    if OpenAICompatibleAtomExtractor is None:
        pipeline.close()
        return None
    memory_db = getattr(agent, "_get_l0_memory_db", lambda: None)()
    l0_recorder = L0Recorder(home, database=memory_db) if memory_db else L0Recorder(home)
    extractor = OpenAICompatibleAtomExtractor(
        provider=aux.get("provider") or None, model=aux.get("model") or None,
        base_url=aux.get("base_url") or None, api_mode=aux.get("api_mode") or None,
        api_key=aux.get("api_key") or None,
        timeout=float(aux.get("timeout", 120) or 120),
        extra_body=aux.get("extra_body") if isinstance(aux.get("extra_body"), dict) else {},
        main_runtime=main_runtime,
        max_input_chars=int(cfg.get("max_input_chars", 24000)),
        max_memories=min(12, int(cfg.get("max_memories_per_job", 20))),
        max_attempts=int(cfg.get("empty_result_max_attempts", 2)),
        precheck_enabled=bool(cfg.get("precheck_enabled", False)),
    )
    # L2 uses the same auxiliary route and the Scene Consolidation prompt from
    # MemoryCore.  It is optional and fail-open; ScenarioStore remains the
    # durable fallback when no model credentials are configured.
    scenario_cfg = cfg.get("scenario") if isinstance(cfg.get("scenario"), Mapping) else {}
    if bool(scenario_cfg.get("enabled", True)) and bool(scenario_cfg.get("llm_enabled", True)):
        try:
            from kylinmemory.l2_extraction import OpenAICompatibleSceneConsolidator

            pipeline.consolidator = OpenAICompatibleSceneConsolidator(
                provider=aux.get("provider") or None,
                model=aux.get("model") or None,
                base_url=aux.get("base_url") or None,
                api_key=aux.get("api_key") or None,
                api_mode=aux.get("api_mode") or None,
                timeout=float(aux.get("timeout", 300) or 300),
                extra_body=aux.get("extra_body") if isinstance(aux.get("extra_body"), dict) else {},
                main_runtime=main_runtime,
            )
        except Exception:
            logger.debug("L2 scene consolidator unavailable", exc_info=True)
    provider = AtomMemoryProvider(
        pipeline, extractor, user_key=user_key, l0_recorder=l0_recorder,
        session_id=str(getattr(agent, "session_id", "") or ""),
        max_context_chars=int((cfg.get("retrieval") or {}).get("max_context_chars", 3500))
        if isinstance(cfg.get("retrieval"), Mapping) else 3500,
    )
    provider._session_db = getattr(agent, "_session_db", None)
    return provider


__all__ = ["AtomMemoryProvider", "create_atom_memory_provider"]
