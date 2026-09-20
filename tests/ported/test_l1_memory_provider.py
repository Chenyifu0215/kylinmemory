from types import SimpleNamespace

from kylin_memory.l1_memory_provider import AtomMemoryProvider
from kylin_memory.memory_pipeline import MemoryPipelineManager


class _Extractor:
    def extract(self, messages, *, session_id="", mode="chat"):
        candidate = SimpleNamespace(
            content="The project runtime is Python.",
            type="work_fact",
            priority=90,
            scene_name="runtime",
            source_message_ids=[messages[0]["id"]],
            metadata={"work_object": "runtime"},
            timestamps=[],
        )
        return SimpleNamespace(memories=[candidate])


def test_boundary_writes_atom_then_builds_scenario_without_legacy_db(tmp_path):
    pipeline = MemoryPipelineManager(
        tmp_path,
        config={"prompt_mode": "code", "embedding": {"mode": "disabled"}},
    )
    provider = AtomMemoryProvider(pipeline, _Extractor(), user_key="user", session_id="s1")
    provider.initialize("s1", agent_workspace="team", agent_identity="agent")

    result = provider.commit_session(
        [{"id": 1, "session_id": "s1", "role": "user", "content": "This project uses Python."}],
        session_id="s1",
    )

    assert result["written"]
    recalled = pipeline.recall("Python", team_id="team", user_id="user", agent_id="agent")
    assert [atom.type for atom in recalled] == ["work_fact"]
    assert pipeline.scene_store(team_id="team", agent_id="agent").navigation()
    assert (tmp_path / "vectors.db").exists()
    assert not (tmp_path / "semantic_memory.db").exists()
    provider.shutdown()


def test_pipeline_uses_memorycore_l2_consolidator_when_configured(tmp_path):
    seen = {}

    def consolidate(atoms, *, store, scene_name, summary, mode):
        seen.update(scene_name=scene_name, mode=mode, files=[x.filename for x in store.index()])
        return {"scene_name": scene_name, "summary": "LLM summary", "body": "## 任务场景\nRuntime\n\n## 核心 SOP\n- verify"}

    pipeline = MemoryPipelineManager(
        tmp_path,
        config={"prompt_mode": "code", "embedding": {"mode": "disabled"}},
        extractor=lambda messages, **kwargs: [],
        consolidator=consolidate,
    )
    result = pipeline.persist_candidates(
        [{"content": "The runtime uses Python.", "type": "work_fact", "priority": 90,
          "scene_name": "runtime", "source_message_ids": [1], "metadata": {}, "timestamps": []}],
        [{"id": 1, "role": "user", "content": "The runtime uses Python."}],
        session_id="s1", team_id="team", user_id="user", agent_id="agent", mode="code",
    )
    assert result["written"]
    consolidated = pipeline.consolidate(team_id="team", user_id="user", agent_id="agent", session_id="s1")
    assert consolidated["changed"]
    assert seen["mode"] == "code"
    assert "核心 SOP" in pipeline.scene_store(team_id="team", agent_id="agent").read(consolidated["changed"][0])
    pipeline.close()


def test_profile_sources_use_l2_context_and_l1_evidence_once(tmp_path):
    pipeline = MemoryPipelineManager(
        tmp_path,
        config={"prompt_mode": "code", "embedding": {"mode": "disabled"}},
    )
    provider = AtomMemoryProvider(
        pipeline, _Extractor(), user_key="user", session_id="s1"
    )
    provider.initialize("s1", agent_workspace="team", agent_identity="agent")
    provider.commit_session(
        [
            {
                "id": 1,
                "session_id": "s1",
                "role": "user",
                "content": "This project uses Python.",
            }
        ],
        session_id="s1",
    )

    batch = provider.prepare_profile_sources()

    assert [message["memory_layer"] for message in batch["messages"]] == [
        "l2",
        "l1",
    ]
    assert batch["input_refs"][0].startswith("l2:")
    assert batch["input_refs"][1].startswith("l1:")

    provider.acknowledge_profile_sources(batch)
    assert provider.prepare_profile_sources()["messages"] == []
    provider.shutdown()


def test_mixed_user_l2_scene_is_not_projected_into_personal_l3(tmp_path):
    pipeline = MemoryPipelineManager(
        tmp_path,
        config={"prompt_mode": "code", "embedding": {"mode": "disabled"}},
    )
    for user_id, session_id, message_id, content in (
        ("user-a", "s-a", 1, "User A uses Python."),
        ("user-b", "s-b", 2, "User B uses Rust."),
    ):
        persisted = pipeline.persist_candidates(
            [
                {
                    "content": content,
                    "type": "work_fact",
                    "priority": 90,
                    "scene_name": "runtime",
                    "source_message_ids": [message_id],
                    "metadata": {},
                    "timestamps": [],
                }
            ],
            [{"id": message_id, "role": "user", "content": content}],
            session_id=session_id,
            team_id="team",
            user_id=user_id,
            agent_id="agent",
            mode="code",
        )
        assert persisted["written"]
        pipeline.consolidate(
            team_id="team",
            user_id=user_id,
            agent_id="agent",
            session_id=session_id,
        )

    batch = pipeline.prepare_profile_sources(
        team_id="team", agent_id="agent", user_id="user-a"
    )

    assert [message["memory_layer"] for message in batch["messages"]] == ["l1"]
    assert batch["messages"][0]["content"] == "User A uses Python."
    pipeline.close()
