"""sqlite-vec persistence and scope tests for the local L1 Atom store."""

from __future__ import annotations

import importlib.util

import pytest

from kylinmemory.memory_layers import Atom, AtomStore


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec is an optional L1 dependency",
)


class _Embedding:
    provider_name = "test"
    model = "test"
    model_id = "test-model"
    dimensions = 3

    def embed_documents(self, texts):
        return [
            [1.0, 0.0, 0.0]
            if "python" in str(text).lower()
            else [0.0, 1.0, 0.0]
            for text in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def _atom(record_id: str, content: str, *, user_id: str = "u1") -> Atom:
    return Atom.from_mapping(
        {
            "id": record_id,
            "content": content,
            "type": "work_fact",
            "priority": 70,
            "teamId": "team",
            "agentId": "agent",
            "userId": user_id,
        },
        mode="code",
    )


def _store(path):
    store = AtomStore(path, embedding={"mode": "disabled"})
    store.embedding = _Embedding()
    store.embedding_meta = {
        "model_id": "test-model",
        "provider": "test",
        "model": "test",
        "dimensions": 3,
    }
    store._register_embedding_meta()
    return store


def test_sqlite_vec_is_the_only_vector_table_and_survives_reopen(tmp_path):
    store = _store(tmp_path)
    store.upsert_many([_atom("a1", "Python is the runtime")])
    table = store._vec_table
    assert table
    assert store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='l1_vec_blob'"
    ).fetchone() is None
    assert store._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 1
    store.close()

    reopened = _store(tmp_path)
    try:
        results = reopened._search_vector("python", team_id="team", agent_id="agent")
        assert results and results[0][0].id == "a1"
    finally:
        reopened.close()


def test_sqlite_vec_query_respects_scope_and_delete(tmp_path):
    store = _store(tmp_path)
    try:
        store.upsert_many(
            [
                _atom("a1", "Python is the runtime", user_id="u1"),
                _atom("a2", "Python belongs to another user", user_id="u2"),
            ]
        )
        results = store._search_vector(
            "python", team_id="team", agent_id="agent", user_id="u1"
        )
        assert [row[0].id for row in results] == ["a1"]
        assert store.delete("a1") is True
        assert store._search_vector(
            "python", team_id="team", agent_id="agent", user_id="u1"
        ) == []
    finally:
        store.close()
