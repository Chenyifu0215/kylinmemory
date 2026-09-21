from __future__ import annotations

import pytest

from kylinmemory.l1_extraction import (
    AtomCandidate,
    OpenAICompatibleAtomExtractor,
    SemanticExtractionResponseError,
    _parse_extraction_batch,
    _parse_scene_extraction,
    _user_anchored_messages,
)


def test_extractor_accepts_normative_atom_fields():
    batch = _parse_extraction_batch(
        '{"memories":[{"content":"项目使用 Python。","type":"work_fact",'
        '"priority":90,"scene_name":"runtime","source_message_ids":[7],'
        '"metadata":{"work_object":"runtime"},"timestamps":[]}]}'
    )
    atom = batch.memories[0]
    assert atom.type == "work_fact"
    assert atom.priority == 90
    assert atom.source_message_ids == [7]
    assert atom.metadata == {"work_object": "runtime"}


def test_extractor_rejects_legacy_kind_structured_evidence_fields():
    with pytest.raises(SemanticExtractionResponseError):
        _parse_extraction_batch(
            '{"memories":[{"kind":"fact","content":"旧格式",'
            '"structured":{},"evidence":[{"message_id":7}]}]}'
        )


def test_extractor_accepts_memorycore_scene_segmented_array():
    memories = _parse_scene_extraction(
        '[{"scene_name":"团队推进运行时","message_ids":["7"],"memories":['
        '{"content":"团队决定使用 Python。","type":"work_fact",'
        '"priority":90,"source_message_ids":["7"],"metadata":{}}]}]'
    )
    assert len(memories) == 1
    assert memories[0].scene_name == "团队推进运行时"
    assert memories[0].source_message_ids == [7]
    assert memories[0].timestamps == []


def test_scene_parser_accepts_flat_object_without_confusing_nested_source_ids():
    """Legacy flat responses must not be mistaken for a nested ID array."""
    memories = _parse_scene_extraction(
        '{"memories":[{"content":"用户喜欢骑行。","type":"persona",'
        '"priority":70,"scene_name":"个人习惯",'
        '"source_message_ids":[7],"metadata":{},"timestamps":[]}]}'
    )
    assert len(memories) == 1
    assert memories[0].content == "用户喜欢骑行。"
    assert memories[0].source_message_ids == [7]


def test_scene_parser_accepts_framed_flat_object():
    memories = _parse_scene_extraction(
        'Result:\n{"memories":[{"content":"用户喜欢骑行。","type":"persona",'
        '"priority":70,"scene_name":"个人习惯",'
        '"source_message_ids":[7],"metadata":{},"timestamps":[]}]}'
    )
    assert len(memories) == 1


def test_user_anchored_messages_keep_only_nearest_assistant_question():
    messages = [
        {"message_id": 1, "role": "user", "content": "help"},
        {"message_id": 2, "role": "assistant", "content": "progress one"},
        {"message_id": 3, "role": "assistant", "content": "Do you live in Changsha?"},
        {"message_id": 4, "role": "tool", "content": "internal output"},
        {"message_id": 5, "role": "user", "content": "Yes."},
    ]

    assert [
        message["message_id"]
        for message in _user_anchored_messages(messages)
    ] == [1, 3, 5]


def test_extractor_retries_empty_result_and_requires_user_evidence(monkeypatch):
    extractor = OpenAICompatibleAtomExtractor(
        max_attempts=2,
        max_memories=10,
    )
    calls = []

    def request(messages, **_kwargs):
        calls.append([message["message_id"] for message in messages])
        if len(calls) == 1:
            return []
        return [
            AtomCandidate(
                content="The user lives in Changsha.",
                type="persona",
                priority=80,
                scene_name="Residence",
                source_message_ids=[10, 11],
                metadata={},
                timestamps=[],
            ),
            AtomCandidate(
                content="The assistant guessed another residence.",
                type="persona",
                priority=80,
                scene_name="Residence",
                source_message_ids=[10],
                metadata={},
                timestamps=[],
            ),
        ]

    monkeypatch.setattr(extractor, "_request", request)
    batch = extractor.extract(
        [
            {
                "id": 10,
                "role": "assistant",
                "content": "Do you live in Changsha?",
            },
            {"id": 11, "role": "user", "content": "Yes."},
        ]
    )

    assert calls == [[10, 11], [10, 11]]
    assert [memory.content for memory in batch.memories] == [
        "The user lives in Changsha."
    ]
