from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from kylinmemory.l1_memory_provider import AtomMemoryProvider
from kylinmemory.l0_recorder import L0Recorder
from kylinmemory.memory_pipeline import MemoryPipelineManager
from kylinmemory.state import MemoryDB, SessionDB


def test_l0_capture_is_incremental_sanitized_sqlite_without_fts_or_vectors(tmp_path):
    recorder = L0Recorder(tmp_path, plugin_start_ms=0)
    key = "team:t|agent:a|session:s1"
    raw = [
        {
            "id": 1,
            "role": "user",
            "content": "<relevant-memories>old injected memory</relevant-memories>Keep this fact",
            "timestamp": 1.001,
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "Explanation\n```python\nsecret = 1\n```\nDone",
            "timestamp": 1.002,
        },
        {"id": 3, "role": "tool", "content": "internal output", "timestamp": 1.003},
        {"id": 4, "role": "user", "content": "/reset", "timestamp": 1.004},
    ]

    first = recorder.capture(
        key,
        raw,
        session_id="s1",
        team_id="t",
        user_id="u",
        agent_id="a",
        task_id="task",
    )
    replay = recorder.capture(key, raw, session_id="s1")

    assert first.recorded_count == 2
    assert replay.recorded_count == 0
    assert [message["content"] for message in first.messages] == [
        "Keep this fact",
        "Explanation\n\nDone",
    ]
    with sqlite3.connect(tmp_path / "l0_memory.db") as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM l0_conversations ORDER BY timestamp"
        ).fetchall()
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
    assert [row["message_id"] for row in rows] == [1, 2]
    assert all(row["session_key"] == key for row in rows)
    assert all(row["recorded_at_ms"] > 0 for row in rows)
    assert table_names == {"l0_conversations", "l0_capture_checkpoints"}
    assert not (tmp_path / "conversations").exists()
    assert not (tmp_path / "vectors.db").exists()


def test_l0_capture_uses_position_slice_and_clean_original_user_text(tmp_path):
    recorder = L0Recorder(tmp_path, plugin_start_ms=0)
    result = recorder.capture(
        "scope|session:s1",
        [
            {"id": 1, "role": "user", "content": "old", "timestamp": 1.0},
            {"id": 2, "role": "assistant", "content": "old reply", "timestamp": 2.0},
            {
                "id": 3,
                "role": "user",
                "content": "<user-persona>injected</user-persona>polluted",
                "timestamp": 3.0,
            },
            {"id": 4, "role": "assistant", "content": "new reply", "timestamp": 4.0},
        ],
        session_id="s1",
        original_user_text="clean prompt",
        original_user_message_count=2,
    )

    assert [(item["id"], item["content"]) for item in result.messages] == [
        (3, "clean prompt"),
        (4, "new reply"),
    ]


def test_l0_query_limit_counts_user_turns_and_excludes_legacy_tools(tmp_path):
    recorder = L0Recorder(tmp_path, plugin_start_ms=0)
    key = "scope|session:s1"
    messages = [
        {"id": 1, "role": "user", "content": "First", "timestamp": 1},
        {
            "id": 2,
            "role": "assistant",
            "content": "First answer",
            "timestamp": 2,
        },
        {"id": 3, "role": "user", "content": "Second", "timestamp": 3},
        {
            "id": 4,
            "role": "assistant",
            "content": "Second answer",
            "timestamp": 4,
        },
        {"id": 5, "role": "user", "content": "Third", "timestamp": 5},
    ]
    # Separate completed turns receive separate write-time boundaries.
    recorder.capture(key, messages[:2], session_id="s1")
    recorder.capture(key, messages[:4], session_id="s1")
    recorder.capture(key, messages, session_id="s1")
    with recorder.database._conn:
        recorder.database._conn.execute(
            """INSERT INTO l0_conversations (
                   record_id, message_id, session_key, session_id,
                   team_id, task_id, user_id, agent_id, role,
                   message_text, recorded_at, recorded_at_ms, timestamp
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "legacy-tool",
                99,
                key,
                "s1",
                "default",
                "",
                "default",
                "default",
                "tool",
                "internal output",
                "2026-01-01T00:00:00Z",
                1,
                99,
            ),
        )

    assert [
        message["id"] for message in recorder.read_after(key, limit=2)
    ] == [1, 2, 3, 4]


def test_l0_message_batch_and_capture_cursor_are_one_transaction(tmp_path):
    recorder = L0Recorder(tmp_path, plugin_start_ms=0)
    records = [
        {
            "record_id": "valid",
            "id": 1,
            "role": "user",
            "content": "valid",
            "timestamp": 1,
        },
        {
            "record_id": "invalid",
            "id": "not-an-integer",
            "role": "assistant",
            "content": "must roll back",
            "timestamp": 2,
        },
    ]

    with pytest.raises(ValueError):
        recorder.database.capture_l0_records(
            "scope|session:s1", records, initial_cursor=0
        )

    with sqlite3.connect(tmp_path / "l0_memory.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM l0_conversations").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM l0_capture_checkpoints").fetchone()[0] == 0


def test_sqlite_l1_reads_oldest_batch_and_aligns_same_ms_boundary(tmp_path, monkeypatch):
    rows = [
        {
            "id": index,
            "role": "user",
            "content": f"Runtime fact {index}",
            "timestamp": index,
            "recorded_at_ms": recorded,
            "session_id": "s1",
            "team_id": "team",
            "user_id": "user",
            "agent_id": "agent",
            "task_id": "",
        }
        for index, recorded in enumerate((10, 20, 20, 30, 40, 50), start=1)
    ]
    batches: list[list[int]] = []

    def read_after(_key, *, after_recorded_at_ms=0, limit=20):
        return [row for row in rows if row["recorded_at_ms"] > after_recorded_at_ms][:limit]

    def extract(messages, **_kwargs):
        batches.append([message["id"] for message in messages])
        return [
            {
                "content": f"Runtime batch {len(batches)}",
                "type": "work_fact",
                "priority": 80,
                "scene_name": "runtime",
                "source_message_ids": [messages[0]["id"]],
                "metadata": {},
                "timestamps": [],
            }
        ]

    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "code",
            "embedding": {"mode": "disabled"},
            "enable_dedup": False,
            "enable_warmup": False,
            "every_n_conversations": 99,
            "l1_idle_timeout_seconds": 0,
            "l1_batch_process": 2,
            "l1_batch_query": 5,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        extractor=extract,
        l0_reader=read_after,
    )
    key = "team:team|agent:agent|session:s1"
    monkeypatch.setattr(manager, "_schedule_l2_flush", lambda *args, **kwargs: None)
    backlog_schedules = []
    monkeypatch.setattr(
        manager,
        "_schedule_idle_flush",
        lambda session_key, **kwargs: backlog_schedules.append((session_key, kwargs.get("delay"))),
    )
    manager.notify_conversation(
        key, [], session_id="s1", team_id="team", user_id="user", agent_id="agent", mode="code"
    )

    first = manager.flush_conversation(key)
    second = manager.flush_conversation(key)

    assert batches == [[1, 2, 3], [4, 5]]
    assert first["has_full_backlog"] is True
    assert first["processed_count"] == 3
    assert manager._scheduler_state[key]["last_l1_cursor"] == 40
    assert second["has_more"] is True
    assert backlog_schedules[0] == (key, 0.0)
    manager._scheduler_state[key]["l2_pending"] = 0
    manager.close()


def test_sqlite_l1_batch_limits_count_user_turns_not_assistant_rows(
    tmp_path, monkeypatch
):
    rows = [
        {
            "id": 1,
            "role": "user",
            "content": "First user fact",
            "timestamp": 1,
            "recorded_at_ms": 10,
            "session_id": "s1",
            "team_id": "team",
            "user_id": "user",
            "agent_id": "agent",
            "task_id": "",
        }
    ]
    rows.extend(
        {
            **rows[0],
            "id": message_id,
            "role": "assistant",
            "content": f"Progress {message_id}",
            "timestamp": message_id,
            "recorded_at_ms": message_id * 10,
        }
        for message_id in range(2, 12)
    )
    rows.extend(
        [
            {
                **rows[0],
                "id": 12,
                "role": "user",
                "content": "Second user fact",
                "timestamp": 12,
                "recorded_at_ms": 120,
            },
            {
                **rows[0],
                "id": 13,
                "role": "assistant",
                "content": "Do you live in Changsha?",
                "timestamp": 13,
                "recorded_at_ms": 130,
            },
            {
                **rows[0],
                "id": 14,
                "role": "user",
                "content": "Yes.",
                "timestamp": 14,
                "recorded_at_ms": 140,
            },
        ]
    )
    batches = []

    def read_after(_key, *, after_recorded_at_ms=0, limit=20):
        return [
            row for row in rows
            if row["recorded_at_ms"] > after_recorded_at_ms
        ]

    def extract(messages, **_kwargs):
        batches.append([message["id"] for message in messages])
        return []

    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "embedding": {"mode": "disabled"},
            "enable_warmup": False,
            "l1_batch_process": 2,
            "l1_batch_query": 3,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        extractor=extract,
        l0_reader=read_after,
    )
    key = "team:team|agent:agent|session:s1"
    manager._scheduler_state[key] = manager._new_scheduler_state()
    manager._scheduler_state[key]["conversation_count"] = 1
    monkeypatch.setattr(manager, "_schedule_l2_flush", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_schedule_idle_flush", lambda *args, **kwargs: None)

    first = manager.flush_conversation(key)
    manager._scheduler_state[key]["conversation_count"] = 1
    second = manager.flush_conversation(key)

    assert first["processed_count"] == 2
    assert batches == [[1, 11, 12], [13, 14]]
    manager._scheduler_state[key]["l2_pending"] = 0
    manager.close()


def test_sqlite_l1_uses_memorycore_warmup_idle_and_boundary_timing(tmp_path, monkeypatch):
    rows = []
    batches: list[list[int]] = []
    idle_resets = []

    def read_after(_key, *, after_recorded_at_ms=0, limit=20):
        return [row for row in rows if row["recorded_at_ms"] > after_recorded_at_ms][:limit]

    def extract(messages, **_kwargs):
        batches.append([message["id"] for message in messages])
        return []

    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "code",
            "embedding": {"mode": "disabled"},
            "enable_warmup": True,
            "every_n_conversations": 5,
            "l1_idle_timeout_seconds": 600,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        extractor=extract,
        l0_reader=read_after,
    )
    key = "team:team|agent:agent|session:s1"
    monkeypatch.setattr(manager, "_schedule_l2_flush", lambda *args, **kwargs: None)

    def schedule(session_key, **kwargs):
        if kwargs.get("delay") == 0:
            manager.flush_conversation(
                session_key,
                **{name: value for name, value in kwargs.items() if name != "delay"},
            )
        else:
            idle_resets.append((session_key, kwargs.get("delay")))

    monkeypatch.setattr(manager, "_schedule_idle_flush", schedule)

    for message_id in range(1, 4):
        rows.append(
            {
                "id": message_id,
                "role": "user",
                "content": f"Runtime fact {message_id}",
                "timestamp": message_id,
                "recorded_at_ms": message_id * 10,
                "session_id": "s1",
                "team_id": "team",
                "user_id": "user",
                "agent_id": "agent",
                "task_id": "",
            }
        )
        manager.notify_conversation(
            key, [], session_id="s1", team_id="team", user_id="user", agent_id="agent", mode="code"
        )

    # Warm-up threshold 1 runs immediately; threshold 2 resets idle after its
    # first conversation, then runs immediately on its second conversation.
    assert batches == [[1], [2, 3]]
    assert idle_resets == [(key, None)]
    assert manager._scheduler_state[key]["warmup_threshold"] == 4

    # A real session boundary flushes residual rows even below threshold.
    rows.append({**rows[-1], "id": 4, "content": "Runtime fact 4", "recorded_at_ms": 40})
    manager.notify_conversation(
        key, [], session_id="s1", team_id="team", user_id="user", agent_id="agent", mode="code"
    )
    result = manager.flush_session(key, session_id="s1", team_id="team", user_id="user", agent_id="agent", mode="code")

    assert result["scheduler_seen"] is True
    assert batches[-1] == [4]
    manager._scheduler_state[key]["l2_pending"] = 0
    manager.close()


def test_every_three_warmup_counts_only_users_and_reaches_steady_cadence(
    tmp_path, monkeypatch
):
    rows = []
    batches: list[list[dict]] = []

    def read_after(_key, *, after_recorded_at_ms=0, limit=20):
        pending = [
            row
            for row in rows
            if row["recorded_at_ms"] > after_recorded_at_ms
        ]
        selected = []
        user_count = 0
        for row in pending:
            if row["role"] == "tool":
                # Exercise the pipeline's legacy-tool guard even though the
                # real SQLite reader already excludes these rows.
                selected.append(row)
                continue
            if row["role"] == "user":
                if user_count >= limit:
                    break
                user_count += 1
            selected.append(row)
        return selected

    def extract(messages, **_kwargs):
        batches.append([dict(message) for message in messages])
        return []

    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "chat",
            "embedding": {"mode": "disabled"},
            "enable_warmup": True,
            "every_n_conversations": 3,
            "l1_idle_timeout_seconds": 600,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        extractor=extract,
        l0_reader=read_after,
    )
    key = "team:team|agent:agent|session:s1"
    monkeypatch.setattr(manager, "_schedule_l2_flush", lambda *args, **kwargs: None)

    def schedule(session_key, **kwargs):
        if kwargs.get("delay") == 0:
            manager.flush_conversation(
                session_key,
                **{
                    name: value
                    for name, value in kwargs.items()
                    if name != "delay"
                },
            )

    monkeypatch.setattr(manager, "_schedule_idle_flush", schedule)

    for turn in range(1, 7):
        boundary = turn * 10
        rows.extend(
            [
                {
                    "id": turn * 10 + 1,
                    "role": "user",
                    "content": f"User fact {turn}",
                    "timestamp": turn * 10 + 1,
                    "recorded_at_ms": boundary,
                    "session_id": "s1",
                    "team_id": "team",
                    "user_id": "user",
                    "agent_id": "agent",
                    "task_id": "",
                },
                {
                    "id": turn * 10 + 2,
                    "role": "assistant",
                    "content": f"Question or response {turn}",
                    "timestamp": turn * 10 + 2,
                    "recorded_at_ms": boundary,
                    "session_id": "s1",
                    "team_id": "team",
                    "user_id": "user",
                    "agent_id": "agent",
                    "task_id": "",
                },
                {
                    "id": turn * 10 + 3,
                    "role": "tool",
                    "content": f"Internal output {turn}",
                    "timestamp": turn * 10 + 3,
                    "recorded_at_ms": boundary,
                    "session_id": "s1",
                    "team_id": "team",
                    "user_id": "user",
                    "agent_id": "agent",
                    "task_id": "",
                },
            ]
        )
        manager.notify_conversation(
            key,
            [],
            session_id="s1",
            team_id="team",
            user_id="user",
            agent_id="agent",
            mode="chat",
        )

    assert [
        sum(message["role"] == "user" for message in batch)
        for batch in batches
    ] == [1, 2, 3]
    assert all(
        message["role"] != "tool"
        for batch in batches
        for message in batch
    )
    assert manager._scheduler_state[key]["warmup_threshold"] == 0
    assert manager._scheduler_state[key]["conversation_count"] == 0
    manager._scheduler_state[key]["l2_pending"] = 0
    manager.close()


def test_resume_rebuilds_six_pending_user_turns_and_runs_threshold_now(
    tmp_path, monkeypatch
):
    key = "team:team|agent:agent|session:s1"
    rows = [
        {
            "id": message_id,
            "role": "user",
            "content": f"User fact {message_id}",
            "timestamp": message_id,
            "recorded_at_ms": message_id * 10,
            "session_id": "s1",
            "team_id": "team",
            "user_id": "user",
            "agent_id": "agent",
            "task_id": "",
        }
        for message_id in range(1, 7)
    ]

    def read_after(_key, *, after_recorded_at_ms=0, limit=20):
        return [
            row
            for row in rows
            if row["recorded_at_ms"] > after_recorded_at_ms
        ][:limit]

    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "embedding": {"mode": "disabled"},
            "enable_warmup": True,
            "every_n_conversations": 3,
            "l1_idle_timeout_seconds": 600,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        l0_reader=read_after,
    )
    scheduled = []
    monkeypatch.setattr(
        manager,
        "_schedule_idle_flush",
        lambda session_key, **kwargs: scheduled.append(
            (session_key, kwargs.get("delay"), kwargs)
        ),
    )

    manager.resume_scheduled_work(key)

    state = manager._scheduler_state[key]
    assert state["conversation_count"] == 6
    assert state["context"]["session_id"] == "s1"
    assert state["context"]["team_id"] == "team"
    assert state["context"]["user_id"] == "user"
    assert scheduled[0][0:2] == (key, 0)
    manager._scheduler_state[key]["conversation_count"] = 0
    manager.close()


def test_every_two_conversations_extracts_both_turns_as_one_l1_batch(
    tmp_path, monkeypatch
):
    rows = []
    batches: list[list[int]] = []

    def read_after(_key, *, after_recorded_at_ms=0, limit=20):
        return [
            row
            for row in rows
            if row["recorded_at_ms"] > after_recorded_at_ms
        ][:limit]

    def extract(messages, **_kwargs):
        batches.append([message["id"] for message in messages])
        return []

    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "chat",
            "embedding": {"mode": "disabled"},
            "enable_warmup": True,
            "every_n_conversations": 2,
            "l1_idle_timeout_seconds": 600,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        extractor=extract,
        l0_reader=read_after,
    )
    key = "team:team|agent:agent|session:s1"
    idle_resets = []
    monkeypatch.setattr(manager, "_schedule_l2_flush", lambda *args, **kwargs: None)

    def schedule(session_key, **kwargs):
        if kwargs.get("delay") == 0:
            manager.flush_conversation(
                session_key,
                **{
                    name: value
                    for name, value in kwargs.items()
                    if name != "delay"
                },
            )
        else:
            idle_resets.append(session_key)

    monkeypatch.setattr(manager, "_schedule_idle_flush", schedule)

    for turn, contents in enumerate(
        (("first question", "first answer"), ("second question", "second answer")),
        start=1,
    ):
        for offset, (role, content) in enumerate(
            zip(("user", "assistant"), contents),
            start=1,
        ):
            message_id = (turn - 1) * 2 + offset
            rows.append(
                {
                    "id": message_id,
                    "role": role,
                    "content": content,
                    "timestamp": message_id,
                    "recorded_at_ms": turn * 10,
                    "session_id": "s1",
                    "team_id": "team",
                    "user_id": "user",
                    "agent_id": "agent",
                    "task_id": "",
                }
            )
        manager.notify_conversation(
            key,
            [],
            session_id="s1",
            team_id="team",
            user_id="user",
            agent_id="agent",
            mode="chat",
        )

    assert idle_resets == [key]
    # The final assistant reply is retained durably as possible context for
    # the next user turn; it does not enter this user-anchored extraction.
    assert batches == [[1, 2, 3]]
    assert manager._scheduler_state[key]["last_l1_cursor"] == 20
    manager._scheduler_state[key]["l2_pending"] = 0
    manager.close()


def test_every_two_conversations_ignores_stale_warmup_checkpoint(
    tmp_path, monkeypatch
):
    key = "team:team|agent:agent|session:s1"
    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "embedding": {"mode": "disabled"},
            "enable_warmup": True,
            "every_n_conversations": 2,
            "l1_idle_timeout_seconds": 600,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        l0_reader=lambda *_args, **_kwargs: [],
    )
    checkpoint = manager._new_scheduler_state()
    checkpoint["warmup_threshold"] = 1
    checkpoint["last_l1_cursor"] = 10
    manager._scheduler_file(key).write_text(
        json.dumps(
            {"schema_version": 1, "session_key": key, "state": checkpoint}
        ),
        encoding="utf-8",
    )
    immediate = []
    monkeypatch.setattr(
        manager,
        "_schedule_idle_flush",
        lambda session_key, **kwargs: immediate.append(kwargs.get("delay")),
    )

    manager.resume_scheduled_work(key)
    result = manager.notify_conversation(
        key,
        [],
        session_id="s1",
        team_id="team",
        user_id="user",
        agent_id="agent",
    )

    assert manager._scheduler_state[key]["warmup_threshold"] == 0
    assert result["threshold"] == 2
    assert immediate == [None]
    manager._scheduler_state[key]["conversation_count"] = 0
    manager.close()


def test_builtin_provider_captures_sqlite_l0_then_notifies_scheduler(tmp_path, monkeypatch):
    class Extractor:
        def extract(self, messages, *, session_id="", mode="code"):
            return SimpleNamespace(memories=[])

    recorder = L0Recorder(tmp_path, plugin_start_ms=0)
    pipeline = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "code",
            "embedding": {"mode": "disabled"},
            "enable_warmup": False,
            "every_n_conversations": 5,
            "l1_idle_timeout_seconds": 600,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        l0_reader=recorder.read_after,
    )
    provider = AtomMemoryProvider(
        pipeline,
        Extractor(),
        user_key="user",
        l0_recorder=recorder,
        session_id="s1",
    )
    provider.initialize("s1", agent_workspace="team", agent_identity="agent")
    scheduled = []
    monkeypatch.setattr(
        pipeline,
        "_schedule_idle_flush",
        lambda key, **kwargs: scheduled.append((key, kwargs.get("delay"))),
    )

    provider.capture_l0_messages(
        [
            {
                "id": 41,
                "role": "user",
                "content": "The runtime uses Python.",
                "timestamp": 1.001,
            },
            {
                "id": 42,
                "role": "assistant",
                "content": "Confirmed.",
                "timestamp": 1.002,
            },
        ],
        session_id="s1",
    )
    provider.sync_turn("The runtime uses Python.", "Confirmed.", session_id="s1")

    key = provider._scheduler_key("s1")
    assert pipeline._scheduler_state[key]["conversation_count"] == 1
    assert pipeline._scheduler_state[key]["buffer"] == []
    assert scheduled == [(key, None)]
    assert [message["id"] for message in recorder.read_after(key)] == [41, 42]
    with sqlite3.connect(tmp_path / "l0_memory.db") as connection:
        l0_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name LIKE 'l0_%'"
            )
        }
    assert l0_tables == {"l0_conversations", "l0_capture_checkpoints"}
    with sqlite3.connect(tmp_path / "vectors.db") as connection:
        table_names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert not any(name.startswith("l0") for name in table_names)
    pipeline._scheduler_state[key]["conversation_count"] = 0
    provider.shutdown()


def test_builtin_provider_does_not_synthesize_l0_without_committed_capture(
    tmp_path, monkeypatch
):
    recorder = L0Recorder(tmp_path, plugin_start_ms=0)
    pipeline = MemoryPipelineManager(
        tmp_path,
        config={
            "embedding": {"mode": "disabled"},
            "enable_warmup": False,
            "l1_idle_timeout_seconds": 600,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        l0_reader=recorder.read_after,
    )
    provider = AtomMemoryProvider(
        pipeline,
        SimpleNamespace(extract=lambda *_args, **_kwargs: SimpleNamespace(memories=[])),
        user_key="user",
        l0_recorder=recorder,
        session_id="s1",
    )
    provider.initialize("s1", agent_workspace="team", agent_identity="agent")
    scheduled = []
    monkeypatch.setattr(
        pipeline,
        "_schedule_idle_flush",
        lambda key, **kwargs: scheduled.append((key, kwargs.get("delay"))),
    )

    provider.sync_turn("not committed", "not committed", session_id="s1")

    key = provider._scheduler_key("s1")
    assert key not in pipeline._scheduler_state
    assert recorder.read_after(key) == []
    assert scheduled == []
    provider.shutdown()


def test_boundary_only_host_captures_l0_before_l1(tmp_path, monkeypatch):
    seen = []

    class Extractor:
        def extract(self, messages, *, session_id="", mode="code"):
            seen.append([message["id"] for message in messages])
            return SimpleNamespace(memories=[])

    recorder = L0Recorder(tmp_path, plugin_start_ms=0)
    pipeline = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "code",
            "embedding": {"mode": "disabled"},
            "enable_warmup": False,
            "every_n_conversations": 5,
            "l1_idle_timeout_seconds": 0,
            "scenario": {"l2_max_interval_seconds": 0},
        },
        l0_reader=recorder.read_after,
    )
    provider = AtomMemoryProvider(
        pipeline,
        Extractor(),
        user_key="user",
        l0_recorder=recorder,
        session_id="s1",
    )
    provider.initialize("s1", agent_workspace="team", agent_identity="agent")
    durable = [
        {
            "id": 71,
            "role": "user",
            "content": "The runtime uses Python.",
            "timestamp": 1.0,
        },
        {
            "id": 72,
            "role": "assistant",
            "content": "Confirmed.",
            "timestamp": 2.0,
        },
    ]
    provider._session_db = SimpleNamespace(get_messages=lambda _session_id: durable)
    monkeypatch.setattr(pipeline, "_schedule_idle_flush", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_schedule_l2_flush", lambda *args, **kwargs: None)

    result = provider.commit_session(
        [{"id": 999, "role": "user", "content": "uncommitted fallback"}],
        session_id="s1",
    )

    key = provider._scheduler_key("s1")
    assert result["status"] == "success"
    assert seen == [[71]]
    cursor = pipeline._scheduler_state[key]["last_l1_cursor"]
    assert cursor > 0
    assert recorder.read_after(key, after_recorded_at_ms=cursor) == []
    pipeline._scheduler_state[key]["l2_pending"] = 0
    provider.shutdown()


def test_l1_recorded_at_cursor_survives_pipeline_restart(tmp_path, monkeypatch):
    recorder = L0Recorder(tmp_path, plugin_start_ms=0)
    key = "team:team|agent:agent|session:s1"
    extracted = []

    def extractor(messages, **_kwargs):
        extracted.append([message["id"] for message in messages])
        return []

    config = {
        "prompt_mode": "code",
        "embedding": {"mode": "disabled"},
        "enable_warmup": False,
        "every_n_conversations": 5,
        "l1_idle_timeout_seconds": 0,
        "scenario": {"l2_max_interval_seconds": 0},
    }
    first = MemoryPipelineManager(
        tmp_path, config=config, extractor=extractor, l0_reader=recorder.read_after
    )
    monkeypatch.setattr(first, "_schedule_idle_flush", lambda *args, **kwargs: None)
    monkeypatch.setattr(first, "_schedule_l2_flush", lambda *args, **kwargs: None)
    recorder.capture(
        key,
        [{"id": 81, "role": "user", "content": "First durable fact", "timestamp": 1.0}],
        session_id="s1",
        team_id="team",
        user_id="user",
        agent_id="agent",
    )
    first.notify_conversation(
        key, [], session_id="s1", team_id="team", user_id="user", agent_id="agent", mode="code"
    )
    first.flush_conversation(key)
    first._scheduler_state[key]["l2_pending"] = 0
    first.close()

    recorder.capture(
        key,
        [{"id": 82, "role": "user", "content": "Second durable fact", "timestamp": 2.0}],
        session_id="s1",
        team_id="team",
        user_id="user",
        agent_id="agent",
    )
    second = MemoryPipelineManager(
        tmp_path, config=config, extractor=extractor, l0_reader=recorder.read_after
    )
    monkeypatch.setattr(second, "_schedule_idle_flush", lambda *args, **kwargs: None)
    monkeypatch.setattr(second, "_schedule_l2_flush", lambda *args, **kwargs: None)
    second.resume_scheduled_work(key)
    second.notify_conversation(
        key, [], session_id="s1", team_id="team", user_id="user", agent_id="agent", mode="code"
    )
    second.flush_conversation(key)

    assert extracted == [[81], [82]]
    second._scheduler_state[key]["l2_pending"] = 0
    second.close()
