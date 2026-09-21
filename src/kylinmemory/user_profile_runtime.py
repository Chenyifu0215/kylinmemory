"""Host adapter between :class:`AIAgent` and the bundled user-profile module."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable, Iterable

from kylinmemory.config import get_hermes_home

from kylinmemory.user_profile import (
    EncryptedFileProfileStore,
    FileKeyProvider,
    InteractionMessage,
    OpenAICompatibleProfileExtractor,
    ProfileService,
)
from kylinmemory.user_profile.identity import pseudonymous_user_id


logger = logging.getLogger(__name__)
_MAX_PROFILE_EXTRACTION_ATTEMPTS = 10
_MAX_RETRY_DELAY_SECONDS = 30.0
ProfileStatusCallback = Callable[[dict[str, Any]], None]


class RuntimeUserProfile:
    """Own encrypted-at-rest storage and plaintext prompt rendering for one user."""

    def __init__(
        self,
        service: ProfileService,
        user_id: str,
        *,
        prompt_max_chars: int = 4_000,
        min_confidence: float = 0.5,
        precheck_enabled: bool = True,
        max_attempts: int = 3,
        retry_base_delay_seconds: float = 1.0,
        extractor_factory: Callable[[], OpenAICompatibleProfileExtractor | None] | None = None,
    ) -> None:
        self.service = service
        self.user_id = user_id
        self.prompt_max_chars = prompt_max_chars
        self.min_confidence = min_confidence
        self.precheck_enabled = precheck_enabled
        self.max_attempts = max(
            1, min(int(max_attempts), _MAX_PROFILE_EXTRACTION_ATTEMPTS)
        )
        self.retry_base_delay_seconds = max(0.0, float(retry_base_delay_seconds))
        self.last_observation: dict[str, Any] | None = None
        self._extractor_factory = extractor_factory

    def profile_prompt(self) -> str:
        """Decrypt, filter, and render the profile as sanitized Markdown text."""
        self.service.get_or_create(self.user_id)
        return self.service.profile_prompt(
            self.user_id,
            max_chars=self.prompt_max_chars,
            min_confidence=self.min_confidence,
        )

    def observe(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        status_callback: ProfileStatusCallback | None = None,
    ) -> bool:
        interactions = _interaction_messages(messages)
        if not interactions:
            logger.info("User-profile extraction skipped: no conversation text")
            self.last_observation = {
                "status": "skipped",
                "reason": "no_conversation",
                "attempts": 0,
            }
            return True
        # The main agent may have switched model/client since initialization.
        # Refresh before both the precheck and structured extraction.
        if self._extractor_factory is not None:
            self.service.extractor = self._extractor_factory()
        if self.service.extractor is None:
            error = "no compatible profile LLM is configured"
            logger.warning("User-profile extraction skipped: %s", error)
            self.last_observation = {
                "status": "unavailable",
                "error": error,
                "attempts": 0,
            }
            _notify_status(status_callback, self.last_observation)
            return False
        precheck = getattr(self.service.extractor, "should_extract", None)
        if self.precheck_enabled and callable(precheck):
            try:
                current_profile = self.service.get_or_create(self.user_id)
                decision = precheck(self.user_id, interactions, current_profile)
            except Exception as exc:
                # The gate is an optimization. Preserve the old behavior when
                # a provider rejects the short request or is temporarily down.
                logger.warning("User-profile precheck failed; extracting anyway: %s", exc)
            else:
                if decision is False:
                    logger.info(
                        "User-profile extraction skipped: precheck found no profile update"
                    )
                    self.last_observation = {
                        "status": "success",
                        "reason": "no_changes",
                        "attempts": 1,
                        "applied": 0,
                        "deleted": 0,
                        "conflicts": 0,
                        "rejected": 0,
                    }
                    _notify_status(status_callback, self.last_observation)
                    return True
                if decision is None:
                    logger.warning(
                        "User-profile precheck returned an unrecognized answer; extracting anyway"
                    )
        for attempt in range(1, self.max_attempts + 1):
            logger.info(
                "User-profile extraction LLM call starting: messages=%d model=%s "
                "attempt=%d/%d",
                len(interactions),
                getattr(self.service.extractor, "model", "unknown"),
                attempt,
                self.max_attempts,
            )
            try:
                result = self.service.observe(self.user_id, interactions)
            except Exception as exc:
                error = _error_summary(exc)
                if attempt >= self.max_attempts:
                    self.last_observation = {
                        "status": "failed",
                        "error": error,
                        "attempts": attempt,
                    }
                    _notify_status(status_callback, self.last_observation)
                    raise

                delay = min(
                    self.retry_base_delay_seconds * (2 ** (attempt - 1)),
                    _MAX_RETRY_DELAY_SECONDS,
                )
                logger.warning(
                    "User-profile extraction attempt %d/%d failed: %s; "
                    "retrying in %.1fs",
                    attempt,
                    self.max_attempts,
                    error,
                    delay,
                )
                _notify_status(
                    status_callback,
                    {
                        "status": "retrying",
                        "error": error,
                        "attempts": attempt,
                        "max_attempts": self.max_attempts,
                        "delay_seconds": delay,
                    },
                )
                if delay > 0:
                    time.sleep(delay)
                continue

            self.last_observation = {
                "status": "success",
                "attempts": attempt,
                "applied": len(result.applied),
                "deleted": len(result.deleted),
                "conflicts": len(result.conflicts),
                "rejected": len(result.rejected),
            }
            logger.info(
                "User-profile extraction LLM call completed: applied=%d deleted=%d "
                "conflicts=%d rejected=%d attempts=%d",
                self.last_observation["applied"],
                self.last_observation["deleted"],
                self.last_observation["conflicts"],
                self.last_observation["rejected"],
                attempt,
            )
            _notify_status(status_callback, self.last_observation)
            return True

        return False  # pragma: no cover - the loop always returns or raises


def initialize_user_profile(agent: Any, config: dict[str, Any]) -> RuntimeUserProfile | None:
    """Build a profile service scoped to the active ``HERMES_HOME``."""

    if not config.get("enabled", True):
        return None

    profile_root = get_hermes_home() / "user_profile"
    key_provider = FileKeyProvider(profile_root / "profile.key")
    _ensure_key(key_provider)
    store = EncryptedFileProfileStore(profile_root / "profiles", key_provider)
    extractor = _extractor_for_agent(agent)
    user_id = _pseudonymous_user_id(agent, key_provider)
    return RuntimeUserProfile(
        ProfileService(store, extractor),
        user_id,
        prompt_max_chars=int(config.get("prompt_max_chars", 4_000)),
        min_confidence=float(config.get("min_confidence", 0.5)),
        precheck_enabled=bool(config.get("precheck_enabled", True)),
        max_attempts=int(config.get("max_attempts", 3)),
        retry_base_delay_seconds=float(
            config.get("retry_base_delay_seconds", 1.0)
        ),
        extractor_factory=lambda: _extractor_for_agent(agent),
    )


def build_user_profile_prompt(agent: Any) -> str:
    runtime = getattr(agent, "_user_profile_runtime", None)
    if runtime is None:
        return ""
    try:
        return runtime.profile_prompt()
    except Exception as exc:
        logger.warning("User-profile prompt unavailable: %s", exc)
        return ""


def commit_user_profile_session(
    agent: Any, messages: Iterable[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """Project persisted L1/L2 memory into L3 at a session boundary."""

    message_list = list(messages or [])
    session_id = str(getattr(agent, "session_id", "") or "")
    manager = getattr(agent, "_memory_manager", None)
    if manager is not None:
        try:
            manager.commit_builtin_session(message_list, session_id=session_id)
        except Exception:
            logger.warning("L1/L2 boundary commit failed before L3", exc_info=True)
    runtime = getattr(agent, "_user_profile_runtime", None)
    if runtime is None:
        return None
    prepare_sources = getattr(manager, "prepare_profile_sources", None)
    if not callable(prepare_sources):
        return {
            "status": "skipped",
            "reason": "layered_memory_unavailable",
            "attempts": 0,
        }
    try:
        source_batch = prepare_sources()
    except Exception:
        logger.warning("L3 source preparation failed", exc_info=True)
        source_batch = None
    if not isinstance(source_batch, dict):
        return {
            "status": "skipped",
            "reason": "layered_memory_unavailable",
            "attempts": 0,
        }
    source_messages = list(source_batch.get("messages") or [])
    fingerprint = str(source_batch.get("fingerprint") or "")
    if not fingerprint:
        fingerprint = _session_fingerprint(source_messages)
    committed = getattr(agent, "_user_profile_committed_sessions", None)
    if not isinstance(committed, dict):
        committed = {}
        agent._user_profile_committed_sessions = committed
    if session_id and committed.get(session_id) == fingerprint:
        return {"status": "skipped", "reason": "already_committed", "attempts": 0}
    if not source_messages:
        acknowledged = _acknowledge_profile_sources(
            manager, source_batch, changed=False
        )
        if session_id and acknowledged:
            committed[session_id] = fingerprint
        return {
            "status": "skipped",
            "reason": "no_new_layered_memory",
            "attempts": 0,
        }
    try:
        if isinstance(runtime, RuntimeUserProfile):
            observed = runtime.observe(
                source_messages,
                status_callback=lambda event: _emit_terminal_status(agent, event),
            )
        else:
            # Keep test doubles and third-party runtime adapters compatible with
            # the original one-positional-argument observe contract.
            observed = runtime.observe(source_messages)
        result = getattr(runtime, "last_observation", None)
        outcome = result if isinstance(result, dict) else {
            "status": "success" if observed else "unavailable",
            "attempts": 1 if observed else 0,
        }
        if observed:
            changed = bool(
                int(outcome.get("applied", 0) or 0)
                or int(outcome.get("deleted", 0) or 0)
            )
            output_refs = []
            if changed:
                profile_user_id = str(getattr(runtime, "user_id", "") or "")
                profile_id = hashlib.sha256(
                    profile_user_id.encode("utf-8")
                ).hexdigest()[:32]
                output_refs = [f"profile:v1:{profile_id}"]
            acknowledged = _acknowledge_profile_sources(
                manager,
                source_batch,
                changed=changed,
                output_refs=output_refs,
            )
            if session_id and acknowledged:
                committed[session_id] = fingerprint
        return outcome
    except Exception as exc:
        logger.warning("User-profile session extraction failed: %s", exc)
        result = getattr(runtime, "last_observation", None)
        if isinstance(result, dict):
            return result
        return {
            "status": "failed",
            "error": _error_summary(exc),
            "attempts": 1,
        }


def _acknowledge_profile_sources(
    manager: Any,
    batch: dict[str, Any],
    *,
    changed: bool,
    output_refs: Iterable[str] = (),
) -> bool:
    """Advance the durable L3 cursor before enabling in-process dedupe."""

    acknowledge = getattr(manager, "acknowledge_profile_sources", None)
    if not callable(acknowledge):
        logger.warning("L3 source checkpoint was not acknowledged: callback unavailable")
        return False
    try:
        result = acknowledge(
            batch,
            changed=changed,
            output_refs=list(output_refs),
        )
    except Exception:
        logger.warning("L3 source checkpoint write failed", exc_info=True)
        return False
    if result is False:
        logger.warning("L3 source checkpoint was not acknowledged")
        return False
    return True


def _notify_status(
    callback: ProfileStatusCallback | None, event: dict[str, Any]
) -> None:
    if not callable(callback):
        return
    try:
        callback(event)
    except Exception:
        logger.debug("User-profile status callback failed", exc_info=True)


def _format_status(event: dict[str, Any]) -> str | None:
    status = event.get("status")
    if status == "retrying":
        return (
            "User profile extraction failed "
            f"(attempt {event['attempts']}/{event['max_attempts']}): "
            f"{event['error']}. Retrying in {event['delay_seconds']:.1f}s..."
        )
    if status == "failed":
        return (
            "User profile extraction failed after "
            f"{event.get('attempts', 1)} attempts: {event.get('error', 'unknown error')}"
        )
    if status == "unavailable":
        return f"User profile was not updated: {event.get('error', 'extractor unavailable')}"
    if status == "success":
        applied = int(event.get("applied", 0) or 0)
        deleted = int(event.get("deleted", 0) or 0)
        attempts = int(event.get("attempts", 1) or 1)
        if applied or deleted:
            retry_note = f" after {attempts} attempts" if attempts > 1 else ""
            return (
                f"User profile updated{retry_note}: "
                f"{applied} added or changed, {deleted} deleted."
            )
        return "User profile checked; no changes found."
    return None


def _emit_terminal_status(agent: Any, event: dict[str, Any]) -> None:
    message = _format_status(event)
    if not message:
        return

    platform = str(getattr(agent, "platform", "") or "").lower()
    if platform == "cli":
        emit = getattr(agent, "_emit_status", None)
        if callable(emit):
            emit(message)
    elif platform == "tui":
        # Calling AIAgent._emit_status() here would also print to stdout and
        # corrupt the TUI's JSON-RPC transport. Send only through its callback.
        callback = getattr(agent, "status_callback", None)
        if callable(callback):
            kind = {
                "retrying": "warn",
                "failed": "error",
                "unavailable": "error",
            }.get(str(event.get("status") or ""), "lifecycle")
            callback(kind, message)


def _error_summary(exc: Exception, *, max_chars: int = 300) -> str:
    summary = " ".join(str(exc).split()) or type(exc).__name__
    try:
        from kylinmemory.redact import redact_sensitive_text

        summary = redact_sensitive_text(summary, force=True)
    except Exception:
        pass
    if len(summary) > max_chars:
        return summary[: max_chars - 3].rstrip() + "..."
    return summary


def _ensure_key(key_provider: FileKeyProvider) -> None:
    try:
        key_provider.get_key()
    except FileNotFoundError:
        try:
            key_provider.create()
        except FileExistsError:
            key_provider.get_key()


def _pseudonymous_user_id(agent: Any, key_provider: FileKeyProvider) -> str:
    return pseudonymous_user_id(
        platform=getattr(agent, "platform", None),
        platform_user_id=getattr(agent, "_user_id", None),
        key_provider=key_provider,
    )


def _extractor_for_agent(agent: Any) -> OpenAICompatibleProfileExtractor | None:
    """Reuse the active agent model and client; never configure a second LLM."""
    client = getattr(agent, "client", None)
    model = str(getattr(agent, "model", "") or "").strip()
    if client is None or not model:
        return None

    api_mode = "chat_completions"
    agent_api_mode = str(getattr(agent, "api_mode", "") or "")
    if agent_api_mode in {"codex_responses", "responses"} and hasattr(
        getattr(client, "responses", None), "create"
    ):
        api_mode = "responses"
    elif not hasattr(getattr(client, "chat", None), "completions"):
        return None
    return OpenAICompatibleProfileExtractor(
        model=model,
        client=client,
        api_mode=api_mode,
    )


def _interaction_messages(
    messages: Iterable[dict[str, Any]],
) -> list[InteractionMessage]:
    interactions: list[InteractionMessage] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        content = _text_content(message.get("content"))
        if not content:
            continue
        source = str(message.get("memory_layer") or message.get("source") or "conversation")
        if source not in {"conversation", "l1", "l2"}:
            source = "conversation"
        source_ref = str(message.get("source_ref") or "")[:512]
        for start in range(0, len(content), 50_000):
            interactions.append(
                InteractionMessage(
                    role=message["role"],
                    content=content[start : start + 50_000],
                    source=source,
                    source_ref=source_ref,
                )
            )
    return interactions


def _session_fingerprint(messages: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for message in _interaction_messages(messages):
        digest.update(message.role.encode("ascii"))
        digest.update(b"\0")
        digest.update(message.source.encode("ascii"))
        digest.update(b"\0")
        digest.update(message.source_ref.encode("utf-8"))
        digest.update(b"\0")
        digest.update(message.content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()
