from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import kylinmemory.memory_layers as memory_layers
from kylinmemory.l1_memory_provider import AtomMemoryProvider
from kylinmemory.memory_layers import Atom, ScenarioStore
from kylinmemory.memory_pipeline import MemoryPipelineManager


def _message(message_id: int, content: str = "Python runtime deployment") -> dict:
    return {"id": message_id, "role": "user", "content": content}


def _candidate(
    content: str,
    message_id: int,
    *,
    atom_type: str = "work_fact",
    priority: int = 80,
    scene_name: str = "runtime",
) -> dict:
    return {
        "content": content,
        "type": atom_type,
        "priority": priority,
        "scene_name": scene_name,
        "source_message_ids": [message_id],
        "metadata": {},
        "timestamps": [],
    }


def _persist(
    pipeline: MemoryPipelineManager,
    candidate: dict,
    message: dict,
) -> dict:
    return pipeline.persist_candidates(
        [candidate],
        [message],
        session_key="scope|session:s1",
        session_id="s1",
        team_id="team",
        user_id="user",
        agent_id="agent",
        mode="code",
    )


def _atom(record_id: str, *, updated: str = "2026-01-01T00:00:00Z") -> Atom:
    return Atom.from_mapping(
        {
            "id": record_id,
            "content": f"Fact {record_id}",
            "type": "work_fact",
            "priority": 80,
            "scene_name": "runtime",
            "source_message_ids": [1],
            "metadata": {},
            "timestamps": [updated],
            "createdAt": updated,
            "updatedAt": updated,
            "version": 0,
        },
        mode="code",
        new_id=False,
    )


@pytest.fixture
def pipeline(tmp_path):
    manager = MemoryPipelineManager(
        tmp_path,
        config={"prompt_mode": "code", "embedding": {"mode": "disabled"}},
    )
    yield manager
    manager.close()


@pytest.mark.parametrize("action, expected_delta", [("store", 1), ("skip", 0)])
def test_l1_store_and_skip_actions(pipeline, action, expected_delta):
    first = _persist(
        pipeline,
        _candidate("Python runtime deployment baseline", 1),
        _message(1, "Python runtime deployment baseline"),
    )
    assert len(first["written"]) == 1
    old_id = first["written"][0]
    pipeline.deduper = lambda matches, **_: [
        {
            "record_id": matches[0][0].id,
            "action": action,
            "target_ids": [old_id] if action == "skip" else [],
        }
    ]

    result = _persist(
        pipeline,
        _candidate("Python runtime deployment baseline revised", 2),
        _message(2, "Python runtime deployment baseline revised"),
    )

    assert len(result["written"]) == expected_delta
    assert pipeline.atoms.get(old_id) is not None


def test_l1_update_keeps_new_identity_and_evidence_but_increments_version(pipeline):
    first = _persist(
        pipeline,
        _candidate("Python runtime deployment baseline", 1),
        _message(1, "Python runtime deployment baseline"),
    )
    old = pipeline.atoms.get(first["written"][0])
    pipeline.deduper = lambda matches, **_: [
        {
            "record_id": matches[0][0].id,
            "action": "update",
            "target_ids": [old.id],
            "merged_content": "Python runtime deployment is now containerized",
            "merged_priority": 95,
            "merged_timestamps": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        }
    ]

    result = _persist(
        pipeline,
        _candidate("Python runtime deployment container update", 2),
        _message(2, "Python runtime deployment container update"),
    )
    current = pipeline.atoms.get(result["written"][0])

    assert current.id != old.id
    assert pipeline.atoms.get(old.id) is None
    assert current.version == old.version + 1
    assert current.createdAt != old.createdAt
    assert current.source_message_ids == [2]
    assert current.priority == 95
    assert current.timestamps == ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]


def test_l1_merge_replaces_multiple_targets_and_supports_cross_type(pipeline):
    first = _persist(
        pipeline,
        _candidate("Python runtime deployment uses asyncio", 1),
        _message(1, "Python runtime deployment uses asyncio"),
    )["written"][0]
    second = _persist(
        pipeline,
        _candidate("Python runtime deployment uses containers", 2),
        _message(2, "Python runtime deployment uses containers"),
    )["written"][0]
    pipeline.deduper = lambda matches, **_: [
        {
            "record_id": matches[0][0].id,
            "action": "merge",
            "target_ids": [first, second],
            "merged_content": "Deploy the Python runtime in containers with asyncio",
            "merged_type": "work_method",
            "merged_priority": 90,
        }
    ]

    result = _persist(
        pipeline,
        _candidate("Python runtime deployment combines asyncio containers", 3),
        _message(3, "Python runtime deployment combines asyncio containers"),
    )
    merged = pipeline.atoms.get(result["written"][0])

    assert pipeline.atoms.get(first) is None
    assert pipeline.atoms.get(second) is None
    assert merged.type == "work_method"
    assert merged.version == 1
    assert merged.source_message_ids == [3]


def test_l1_invalid_targets_and_fields_are_filtered(pipeline):
    old_id = _persist(
        pipeline,
        _candidate("Python runtime deployment baseline", 1),
        _message(1, "Python runtime deployment baseline"),
    )["written"][0]
    pipeline.deduper = lambda matches, **_: [
        {
            "record_id": matches[0][0].id,
            "action": "merge",
            "target_ids": ["hallucinated"],
            "merged_type": "not-a-type",
            "merged_priority": 101,
            "merged_timestamps": [1],
        }
    ]

    result = _persist(
        pipeline,
        _candidate("Python runtime deployment baseline update", 2),
        _message(2, "Python runtime deployment baseline update"),
    )
    stored = pipeline.atoms.get(result["written"][0])

    assert pipeline.atoms.get(old_id) is not None
    assert stored.version == 0
    assert stored.type == "work_fact"
    assert stored.priority == 80
    assert all(isinstance(value, str) for value in stored.timestamps)


def test_l1_dedup_failure_stores_every_new_memory(pipeline):
    _persist(
        pipeline,
        _candidate("Python runtime deployment baseline", 1),
        _message(1, "Python runtime deployment baseline"),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("dedup unavailable")

    pipeline.deduper = fail
    result = _persist(
        pipeline,
        _candidate("Python runtime deployment baseline update", 2),
        _message(2, "Python runtime deployment baseline update"),
    )

    assert len(result["written"]) == 1
    assert pipeline.atoms.get(result["written"][0]).version == 0


def test_l1_dedup_can_be_disabled(tmp_path):
    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "code",
            "enable_dedup": False,
            "embedding": {"mode": "disabled"},
        },
        deduper=lambda *args, **kwargs: pytest.fail("deduper must not run"),
    )
    _persist(
        manager,
        _candidate("Python runtime deployment baseline", 1),
        _message(1, "Python runtime deployment baseline"),
    )

    result = _persist(
        manager,
        _candidate("Python runtime deployment baseline", 2),
        _message(2, "Python runtime deployment baseline"),
    )

    assert len(result["written"]) == 1
    manager.close()


def test_l1_batch_prompt_matches_memorycore_protocol():
    new = _atom("new")
    old = _atom("old")

    prompt = MemoryPipelineManager._conflict_prompt(
        [(new, [old]), (_atom("no-match"), [])]
    )

    assert "统一候选记忆池（共 1 条已有记忆）" in prompt
    assert "待判断的新记忆（共 2 条）" in prompt
    assert "### 第 1 条新记忆 (record_id: new)" in prompt
    assert "[]（无相似候选，直接 store）" in prompt
    assert "直接输出 action=store" in prompt


def test_l2_create_update_and_merge_protocol(tmp_path):
    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="code")
    atom = _atom("a1")
    first = store.apply_action(
        [atom], action="create", scene_name="runtime", summary="runtime",
        body="## Work Scene\nPython runtime",
    )
    second = store.apply_action(
        [atom], action="create", scene_name="deploy", summary="deploy",
        body="## Work Scene\nDeployment",
    )
    runtime_file, deploy_file = first["changed"][0], second["changed"][0]
    created = {entry.filename: entry for entry in store.index()}

    updated = store.apply_action(
        [atom], action="update", scene_name="runtime",
        target_files=[runtime_file], summary="runtime updated",
        body="## Work Scene\nUpdated runtime",
    )
    after_update = {entry.filename: entry for entry in store.index()}
    assert updated["changed"] == [runtime_file]
    assert after_update[runtime_file].created == created[runtime_file].created
    assert after_update[runtime_file].heat == created[runtime_file].heat + 1

    renamed = store.apply_action(
        [atom], action="update", scene_name="chengdu-job-search",
        target_files=[runtime_file], summary="成都求职",
        body="## Work Scene\n成都求职",
    )
    assert renamed["changed"] == [runtime_file, "chengdu-job-search.md"]
    assert not (store.scene_dir / runtime_file).exists()
    assert {entry.filename for entry in store.index()} == {
        "chengdu-job-search.md", deploy_file,
    }
    renamed_entry = {entry.filename: entry for entry in store.index()}[
        "chengdu-job-search.md"
    ]

    merged = store.apply_action(
        [atom], action="merge", scene_name="runtime-deploy",
        target_files=["chengdu-job-search.md", deploy_file],
        delete_files=["chengdu-job-search.md", deploy_file], summary="merged",
        body="## Work Scene\nRuntime deployment",
    )
    merged_entry = store.index()[0]
    assert merged["changed"] == [merged_entry.filename]
    assert merged_entry.heat == renamed_entry.heat + created[deploy_file].heat + 1
    assert not (store.scene_dir / runtime_file).exists()
    assert not (store.scene_dir / deploy_file).exists()


def test_l2_metadata_uses_hermes_wall_clock_timezone(tmp_path, monkeypatch):
    times = iter([
        datetime(2026, 8, 31, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2026, 8, 31, 16, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
    ])
    monkeypatch.setattr(memory_layers, "_hermes_now", lambda: next(times))
    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="code")
    atom = _atom("a1")

    created_result = store.apply_action(
        [atom], action="create", scene_name="runtime", summary="runtime",
        body="## Work Scene\nPython runtime",
    )
    filename = created_result["changed"][0]
    created_entry = store.index()[0]

    assert created_entry.created == "2026-08-31T15:00:00+08:00"
    assert created_entry.updated == "2026-08-31T15:00:00+08:00"

    store.apply_action(
        [atom], action="update", scene_name="runtime",
        target_files=[filename], summary="runtime updated",
        body="## Work Scene\nUpdated runtime",
    )
    updated_entry = store.index()[0]

    assert updated_entry.created == "2026-08-31T15:00:00+08:00"
    assert updated_entry.updated == "2026-08-31T16:30:00+08:00"


def test_l2_store_migrates_legacy_utc_metadata_to_local_time(tmp_path, monkeypatch):
    monkeypatch.setattr(
        memory_layers,
        "_hermes_now",
        lambda: datetime(2026, 8, 31, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    scene_dir = tmp_path / "scene_blocks"
    scene_dir.mkdir()
    scene_path = scene_dir / "runtime.md"
    scene_path.write_text(
        "-----META-START-----\n"
        "created: 2026-08-31T07:00:00Z\n"
        "updated: 2026-08-31T08:30:00+00:00\n"
        "summary: runtime\n"
        "heat: 1\n"
        "-----META-END-----\n\n"
        "## Work Scene\nPython runtime",
        encoding="utf-8",
    )

    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="code")

    migrated = scene_path.read_text(encoding="utf-8")
    entry = store.index()[0]
    assert "created: 2026-08-31T15:00:00+08:00" in migrated
    assert "updated: 2026-08-31T16:30:00+08:00" in migrated
    assert entry.created == "2026-08-31T15:00:00+08:00"
    assert entry.updated == "2026-08-31T16:30:00+08:00"


def test_l2_rejects_unknown_target_and_evicts_cold_scene_at_capacity(tmp_path):
    store = ScenarioStore(tmp_path, max_scenes=2, prompt_mode="code")
    atom = _atom("a1")
    with pytest.raises(ValueError, match="unknown scene target"):
        store.apply_action(
            [atom], action="update", scene_name="runtime",
            target_files=["missing.md"], body="## Work Scene\nMissing",
        )
    assert store.apply_action(
        [atom], action="create", scene_name="one",
        body="## Work Scene\nOne",
    )["changed"]

    limited = store.apply_action(
        [atom], action="create", scene_name="two",
        body="## Work Scene\nTwo",
    )

    # Reaching the configured limit is valid; eviction only occurs when a
    # create starts with the store already full.
    assert limited["changed"] == ["two.md"]
    assert {entry.filename for entry in store.index()} == {"one.md", "two.md"}
    assert len(store.index()) == 2


def test_l2_capacity_eviction_prefers_low_heat_then_oldest(tmp_path):
    store = ScenarioStore(tmp_path, max_scenes=4, prompt_mode="code")
    atom = _atom("a1")
    for name in ("cold", "warm", "hot", "full"):
        store.apply_action(
            [atom], action="create", scene_name=name,
            body=f"## Work Scene\n{name}",
        )

    store.apply_action(
        [atom], action="update", scene_name="hot",
        target_files=["hot.md"], body="## Work Scene\nHot updated",
    )
    created = store.apply_action(
        [atom], action="create", scene_name="new",
        body="## Work Scene\nNew",
    )

    assert created["changed"] == ["cold.md", "new.md"]
    assert {entry.filename for entry in store.index()} == {"warm.md", "hot.md", "full.md", "new.md"}


def test_l2_merge_cannot_delete_a_non_target_scene(tmp_path):
    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="code")
    atom = _atom("a1")
    files = [
        store.apply_action(
            [atom], action="create", scene_name=name,
            body=f"## Work Scene\n{name}",
        )["changed"][0]
        for name in ("one", "two", "three")
    ]

    with pytest.raises(ValueError, match="only delete target files"):
        store.apply_action(
            [atom], action="merge", scene_name="merged",
            target_files=files[:2], delete_files=[files[2]],
            body="## Work Scene\nMerged",
        )

    assert {entry.filename for entry in store.index()} == set(files)


def test_l2_consolidator_failure_retains_cursor_and_scenes(tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("LLM unavailable")

    pipeline = MemoryPipelineManager(
        tmp_path,
        config={"prompt_mode": "code", "embedding": {"mode": "disabled"}},
        consolidator=fail,
    )
    written = _persist(
        pipeline,
        _candidate("Python runtime deployment baseline", 1),
        _message(1, "Python runtime deployment baseline"),
    )
    result = pipeline.consolidate(
        team_id="team", user_id="user", agent_id="agent", session_id="s1"
    )

    assert written["written"]
    assert result["success"] is False
    assert result["latestCursor"] == ""
    assert pipeline.scene_store(team_id="team", agent_id="agent").index() == []
    pipeline.close()


def test_l2_pipeline_preserves_explicit_update_scene_rename(tmp_path):
    def update(*args, **kwargs):
        target = kwargs["store"].index()[0].filename
        return {
            "action": "update",
            "target_files": [target],
            "scene_name": "成都求职",
            "summary": "用户改为前往成都求职",
            "body": "## 核心叙事\n用户认为深圳不适合自己，改为前往成都求职。",
            "delete_files": [],
        }

    pipeline = MemoryPipelineManager(
        tmp_path,
        config={"prompt_mode": "chat", "embedding": {"mode": "disabled"}},
        consolidator=update,
    )
    store = pipeline.scene_store(team_id="team", agent_id="agent", user_id="user")
    atom = _atom("seed")
    old_name = store.apply_action(
        [atom], action="create", scene_name="深圳求职",
        body="## 核心叙事\n用户准备前往深圳求职。",
    )["changed"][0]
    _persist(
        pipeline,
        _candidate("用户认为深圳不适合自己，改为前往成都求职。", 1),
        _message(1, "用户认为深圳不适合自己，改为前往成都求职。"),
    )

    result = pipeline.consolidate(
        team_id="team", user_id="user", agent_id="agent", session_id="s1"
    )

    assert result["success"] is True
    assert result["changed"] == [old_name, "成都求职.md"]
    assert not (store.scene_dir / old_name).exists()
    assert [entry.filename for entry in store.index()] == ["成都求职.md"]
    pipeline.close()


def test_l2_partial_writer_failure_restores_snapshot(tmp_path, monkeypatch):
    def merge(*args, **kwargs):
        files = [entry.filename for entry in kwargs["store"].index()]
        return {
            "action": "merge",
            "target_files": files,
            "scene_name": "merged",
            "summary": "merged",
            "body": "## Work Scene\nMerged",
            "delete_files": files,
        }

    pipeline = MemoryPipelineManager(
        tmp_path,
        config={"prompt_mode": "code", "embedding": {"mode": "disabled"}},
        consolidator=merge,
    )
    store = pipeline.scene_store(team_id="team", agent_id="agent", user_id="user")
    atom = _atom("seed")
    store.apply_action([atom], action="create", scene_name="one", body="## Work Scene\nOne")
    store.apply_action([atom], action="create", scene_name="two", body="## Work Scene\nTwo")
    before = {path.name: path.read_text(encoding="utf-8") for path in store.scene_dir.glob("*.md")}
    _persist(
        pipeline,
        _candidate("Python runtime deployment merge", 1),
        _message(1, "Python runtime deployment merge"),
    )
    original_write = store.write
    calls = 0

    def fail_second_write(filename, body):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("partial write")
        return original_write(filename, body)

    monkeypatch.setattr(store, "write", fail_second_write)
    result = pipeline.consolidate(
        team_id="team", user_id="user", agent_id="agent", session_id="s1"
    )
    after = {path.name: path.read_text(encoding="utf-8") for path in store.scene_dir.glob("*.md")}

    assert result["success"] is False
    assert result["latestCursor"] == ""
    assert after == before
    pipeline.close()


def test_scheduler_buffers_are_isolated_per_session(tmp_path):
    class Extractor:
        def extract(self, messages, *, session_id="", mode="code"):
            return SimpleNamespace(
                memories=[
                    SimpleNamespace(
                        content=f"Session {session_id} uses Python runtime",
                        type="work_fact",
                        priority=80,
                        scene_name="runtime",
                        source_message_ids=[messages[0]["id"]],
                        metadata={},
                        timestamps=[],
                    )
                ]
            )

    class SessionReader:
        rows = {
            "s1": [_message(1, "Session s1 uses Python runtime")],
            "s2": [_message(2, "Session s2 uses Python runtime")],
        }

        def get_messages(self, session_id):
            return self.rows[session_id]

    pipeline = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "code",
            "embedding": {"mode": "disabled"},
            "enable_warmup": False,
            "every_n_conversations": 10,
            "l1_idle_timeout_seconds": 0,
            "scenario": {"l2_max_interval_seconds": 0},
        },
    )
    provider = AtomMemoryProvider(
        pipeline, Extractor(), user_key="user", session_id="s1"
    )
    provider._session_db = SessionReader()
    provider.initialize("s1", agent_workspace="team", agent_identity="agent")

    provider.sync_turn("", "", session_id="s1")
    provider.sync_turn("", "", session_id="s2")
    keys = sorted(pipeline._scheduler_state)
    assert keys == [
        "team:team|agent:agent|session:s1",
        "team:team|agent:agent|session:s2",
    ]

    result = provider.commit_session([], session_id="s1")

    assert result["scheduler_seen"] is True
    assert pipeline._scheduler_state[keys[1]]["buffer"][0]["id"] == 2
    assert pipeline.recall(
        "Session s1", team_id="team", user_id="user", agent_id="agent",
        session_id="s1",
    )
    provider.shutdown()


def test_scheduler_warmup_uses_1_2_4_then_steady_threshold(tmp_path, monkeypatch):
    batch_sizes = []

    def extract(messages, **kwargs):
        batch_sizes.append(len(messages))
        return [
            _candidate(
                f"Python runtime batch {len(batch_sizes)}",
                messages[0]["id"],
            )
        ]

    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "code",
            "embedding": {"mode": "disabled"},
            "enable_dedup": False,
            "enable_warmup": True,
            "every_n_conversations": 5,
            "l1_idle_timeout_seconds": 0,
        },
        extractor=extract,
    )
    key = "team:team|agent:agent|session:s1"
    monkeypatch.setattr(manager, "_schedule_l2_flush", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        manager,
        "_schedule_idle_flush",
        lambda session_key, **kwargs: manager.flush_conversation(
            session_key, **{k: v for k, v in kwargs.items() if k != "delay"}
        ),
    )

    for message_id in range(1, 13):
        manager.notify_conversation(
            key,
            [_message(message_id, f"Python runtime event {message_id}")],
            session_id="s1",
            team_id="team",
            user_id="user",
            agent_id="agent",
            mode="code",
        )

    assert batch_sizes == [1, 2, 4, 5]
    assert manager._scheduler_state[key]["warmup_threshold"] == 0
    manager._scheduler_state[key]["l2_pending"] = 0
    manager.close()


def test_scheduler_l1_failure_retains_buffer_and_retries(tmp_path, monkeypatch):
    attempts = 0

    def extract(messages, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary extraction failure")
        return [_candidate("Python runtime recovered", messages[0]["id"])]

    manager = MemoryPipelineManager(
        tmp_path,
        config={
            "prompt_mode": "code",
            "embedding": {"mode": "disabled"},
            "enable_dedup": False,
            "enable_warmup": False,
            "every_n_conversations": 5,
            "l1_idle_timeout_seconds": 0,
            "retry_base_delay_seconds": 30,
            "max_attempts": 5,
        },
        extractor=extract,
    )
    key = "team:team|agent:agent|session:s1"
    retries = []
    monkeypatch.setattr(
        manager,
        "_schedule_idle_flush",
        lambda session_key, **kwargs: retries.append((session_key, kwargs["delay"])),
    )
    monkeypatch.setattr(manager, "_schedule_l2_flush", lambda *args, **kwargs: None)
    manager.notify_conversation(
        key,
        [_message(1)],
        session_id="s1",
        team_id="team",
        user_id="user",
        agent_id="agent",
        mode="code",
    )

    failed = manager.flush_conversation(key)
    assert failed["success"] is False
    assert [message["id"] for message in manager._scheduler_state[key]["buffer"]] == [1]
    assert retries == [(key, 30.0)]

    recovered = manager.flush_conversation(key)
    assert recovered["success"] is True
    assert recovered["written"]
    assert manager._scheduler_state[key]["buffer"] == []
    assert manager._scheduler_state[key]["l1_retry_count"] == 0
    manager._scheduler_state[key]["l2_pending"] = 0
    manager.close()
