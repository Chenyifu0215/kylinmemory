from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kylin_memory.user_profile.models import ProfileEntry, ProfileUpdate, UserProfile
from kylin_memory.user_profile import InteractionMessage, OpenAICompatibleProfileExtractor
from kylin_memory.user_profile_runtime import (
    RuntimeUserProfile,
    _error_summary,
    _extractor_for_agent,
    _interaction_messages,
    commit_user_profile_session,
    initialize_user_profile,
)
from kylin_memory.user_profile.extractors import (
    CONSOLIDATION_INSTRUCTIONS,
    PROFILE_EXTRACTION_TOOL_NAME,
    PROFILE_PRECHECK_TOOL_NAME,
    SCHEMA_REVIEW_INSTRUCTIONS,
    SYSTEM_INSTRUCTIONS,
    _chunk_messages,
    _parse_precheck_arguments,
    _profile_extraction_chat_tool,
    _profile_precheck_chat_tool,
)


def _profile_batch(content: str, *, fingerprint: str, source_ref: str) -> dict:
    return {
        "scope": "team:test|agent:test",
        "profile_key": "profile-test",
        "messages": [
            {
                "role": "user",
                "content": content,
                "memory_layer": "l1",
                "source_ref": source_ref,
            }
        ],
        "fingerprint": fingerprint,
        "input_refs": [f"l1:{source_ref}"],
        "checkpoint": {
            "atom_cursor": {"updated_at": fingerprint, "id": source_ref},
            "scenes": {},
        },
    }


def _memory_manager(*batches: dict) -> MagicMock:
    manager = MagicMock()
    manager.prepare_profile_sources.side_effect = list(batches)
    return manager


def _chat_tool_response(
    tool_name: str, arguments: dict, *, content: str | None = None
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name=tool_name,
                                arguments=json.dumps(arguments, ensure_ascii=False),
                            )
                        )
                    ],
                )
            )
        ]
    )


def _chat_profile_tool_response(arguments: dict, *, content: str | None = None):
    return _chat_tool_response(
        PROFILE_EXTRACTION_TOOL_NAME, arguments, content=content
    )


def _chat_precheck_tool_response(has_profile_update: bool):
    return _chat_tool_response(
        PROFILE_PRECHECK_TOOL_NAME, {"has_profile_update": has_profile_update}
    )


def test_initialize_profile_uses_hermes_home_and_pseudonymous_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace(platform="telegram", _user_id="raw-platform-user", client=None)

    runtime = initialize_user_profile(agent, {})

    assert runtime is not None
    assert runtime.user_id.startswith("v1_")
    assert "raw-platform-user" not in runtime.user_id
    assert (tmp_path / "user_profile" / "profile.key").stat().st_mode & 0o777 == 0o600
    prompt = runtime.profile_prompt()
    assert "<user_profile_data>" in prompt
    assert list((tmp_path / "user_profile" / "profiles").glob("*.profile.enc"))


def test_local_frontends_share_one_profile_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cli = initialize_user_profile(SimpleNamespace(platform="cli", _user_id=None, client=None), {})
    tui = initialize_user_profile(SimpleNamespace(platform="tui", _user_id=None, client=None), {})

    assert cli is not None and tui is not None
    assert cli.user_id == tui.user_id


def test_extractor_reuses_active_agent_model_and_client():
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=MagicMock()),
        responses=SimpleNamespace(),
    )
    agent = SimpleNamespace(
        client=client,
        model="runtime-model",
        api_mode="chat_completions",
    )

    extractor = _extractor_for_agent(agent)

    assert extractor is not None
    assert extractor.client is client
    assert extractor.model == "runtime-model"
    assert extractor.api_mode == "chat_completions"


def test_extractor_reuses_active_responses_client_for_tool_calls():
    create = MagicMock()
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    agent = SimpleNamespace(
        client=client,
        model="runtime-model",
        api_mode="codex_responses",
    )

    extractor = _extractor_for_agent(agent)

    assert extractor is not None
    assert extractor.client is client
    assert extractor.api_mode == "responses"


def test_commit_extracts_each_session_once():
    runtime = MagicMock()
    runtime.observe.return_value = True
    batch = _profile_batch("Remembered greeting", fingerprint="fp-1", source_ref="atom-1")
    manager = _memory_manager(batch, batch)
    agent = SimpleNamespace(
        session_id="session-1",
        _memory_manager=manager,
        _user_profile_runtime=runtime,
        _user_profile_committed_sessions={},
    )
    messages = [{"role": "user", "content": "hello"}]

    commit_user_profile_session(agent, messages)
    commit_user_profile_session(agent, messages)

    assert manager.commit_builtin_session.call_count == 2
    manager.commit_builtin_session.assert_called_with(messages, session_id="session-1")
    runtime.observe.assert_called_once_with(batch["messages"])
    manager.acknowledge_profile_sources.assert_called_once_with(
        batch, changed=False, output_refs=[]
    )


def test_resumed_session_with_new_messages_is_extracted_again():
    runtime = MagicMock()
    runtime.observe.return_value = True
    first_batch = _profile_batch("First atom", fingerprint="fp-1", source_ref="atom-1")
    resumed_batch = _profile_batch("New atom", fingerprint="fp-2", source_ref="atom-2")
    manager = _memory_manager(first_batch, resumed_batch)
    agent = SimpleNamespace(
        session_id="session-1",
        _memory_manager=manager,
        _user_profile_runtime=runtime,
        _user_profile_committed_sessions={},
    )

    commit_user_profile_session(agent, [{"role": "user", "content": "first"}])
    resumed = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "new fact"},
    ]
    commit_user_profile_session(agent, resumed)

    assert runtime.observe.call_count == 2
    runtime.observe.assert_called_with(resumed_batch["messages"])
    assert manager.acknowledge_profile_sources.call_count == 2


def test_failed_layered_checkpoint_is_not_marked_committed():
    runtime = MagicMock()
    runtime.observe.return_value = True
    batch = _profile_batch("New atom", fingerprint="fp-1", source_ref="atom-1")
    manager = _memory_manager(batch)
    manager.acknowledge_profile_sources.return_value = False
    agent = SimpleNamespace(
        session_id="session-1",
        _memory_manager=manager,
        _user_profile_runtime=runtime,
        _user_profile_committed_sessions={},
    )

    outcome = commit_user_profile_session(
        agent, [{"role": "user", "content": "raw transcript"}]
    )

    assert outcome == {"status": "success", "attempts": 1}
    assert agent._user_profile_committed_sessions == {}


def test_message_conversion_keeps_only_conversation_text():
    messages = [
        {"role": "system", "content": "secret system prompt"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url"}]},
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "content": "private tool output"},
    ]

    converted = _interaction_messages(messages)

    assert [(item.role, item.content) for item in converted] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]


def test_message_conversion_preserves_layered_source_metadata():
    converted = _interaction_messages(
        [
            {
                "role": "user",
                "content": "Atom evidence",
                "memory_layer": "l1",
                "source_ref": "atom-1",
            },
            {
                "role": "assistant",
                "content": "Scenario context",
                "memory_layer": "l2",
                "source_ref": "scene.md",
            },
        ]
    )

    assert [(item.source, item.source_ref) for item in converted] == [
        ("l1", "atom-1"),
        ("l2", "scene.md"),
    ]


def test_runtime_observe_skips_cleanly_without_extractor():
    service = MagicMock(extractor=None)
    runtime = RuntimeUserProfile(service, "user")

    assert runtime.observe([{"role": "user", "content": "hello"}]) is False
    service.observe.assert_not_called()


def test_runtime_observe_skips_structured_extraction_when_precheck_says_no():
    extractor = SimpleNamespace(
        model="profile-test-model", should_extract=MagicMock(return_value=False)
    )
    service = MagicMock(extractor=extractor)
    current_profile = UserProfile(user_id="user")
    service.get_or_create.return_value = current_profile
    runtime = RuntimeUserProfile(service, "user")

    assert runtime.observe([{"role": "user", "content": "谢谢"}]) is True

    assert extractor.should_extract.call_args.args[2] is current_profile
    service.observe.assert_not_called()


def test_runtime_observe_extracts_when_precheck_is_unknown_or_fails():
    result = SimpleNamespace(applied=[], deleted=[], conflicts=[], rejected=[])
    for precheck in (MagicMock(return_value=None), MagicMock(side_effect=RuntimeError("down"))):
        extractor = SimpleNamespace(model="profile-test-model", should_extract=precheck)
        service = MagicMock(extractor=extractor)
        service.observe.return_value = result
        runtime = RuntimeUserProfile(service, "user")

        assert runtime.observe([{"role": "user", "content": "请帮我查一下天气"}]) is True
        service.observe.assert_called_once()


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"has_profile_update": True}, True),
        ({"has_profile_update": False}, False),
        # Providers may hand back the arguments still encoded.
        ('{"has_profile_update": true}', True),
        ("not json", None),
        # A missing, non-boolean, or absent tool call is not a confident "no";
        # the caller must fail open rather than silently drop the session.
        ({"has_profile_update": "yes"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_parse_precheck_arguments_only_accepts_a_boolean(arguments, expected):
    assert _parse_precheck_arguments(arguments) is expected


def test_extraction_tool_schema_states_rules_the_host_enforces():
    """Rules that silently discard an update belong next to their field.

    ``apply_updates`` drops an inferred update below 0.7, a non-explicit
    delete or boundary, and an upsert whose value is null; ``_validated_updates``
    drops a quote absent from the user's own text. None of that is expressible
    in JSON Schema, so it has to reach the model as field descriptions.
    """
    properties = (
        _profile_extraction_chat_tool()["function"]["parameters"]
        ["properties"]["updates"]["items"]["properties"]
    )

    assert "0.7" in properties["confidence"]["description"]
    assert "boundaries.*" in properties["explicit"]["description"]
    assert "delete" in properties["value"]["description"]
    assert "Verbatim" in properties["evidence_quote"]["description"]
    assert properties["evidence_quote"]["minLength"] == 1


def test_precheck_tool_schema_is_a_single_required_boolean():
    parameters = _profile_precheck_chat_tool()["function"]["parameters"]

    assert parameters["required"] == ["has_profile_update"]
    assert parameters["properties"]["has_profile_update"]["type"] == "boolean"
    assert parameters["additionalProperties"] is False


def test_extractor_precheck_uses_a_forced_tool_call():
    create = MagicMock(return_value=_chat_precheck_tool_response(False))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    extractor = OpenAICompatibleProfileExtractor(
        model="profile-test-model", client=client, api_mode="chat_completions"
    )
    profile = UserProfile(
        user_id="user",
        entries={
            "basic.preferred_name": ProfileEntry(
                value="小王",
                confidence=1.0,
                source="explicit",
                lifecycle="static",
            )
        },
    )

    assert extractor.should_extract(
        "user", [InteractionMessage(role="user", content="我还是叫小王")], profile
    ) is False

    kwargs = create.call_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == PROFILE_PRECHECK_TOOL_NAME
    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": PROFILE_PRECHECK_TOOL_NAME},
    }
    # The old contract capped output at a single token, which a tool call
    # cannot fit in.
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs
    payload = json.loads(kwargs["messages"][1]["content"])
    assert payload["current_profile"]["basic.preferred_name"] == {
        "value": "小王",
        "source": "explicit",
        "confidence": 1.0,
    }


def test_extractor_precheck_fails_open_without_a_tool_call():
    """A model that answers in prose must not be read as "no update"."""

    create = MagicMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="否", tool_calls=None))
            ]
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    extractor = OpenAICompatibleProfileExtractor(
        model="profile-test-model", client=client, api_mode="chat_completions"
    )

    assert extractor.should_extract(
        "user",
        [InteractionMessage(role="user", content="我还是叫小王")],
        UserProfile(user_id="user"),
    ) is None


@pytest.mark.parametrize(
    "template",
    [SYSTEM_INSTRUCTIONS, CONSOLIDATION_INSTRUCTIONS, SCHEMA_REVIEW_INSTRUCTIONS],
)
def test_prompt_templates_render_with_only_the_schema_placeholder(template):
    """Any literal brace added later must be escaped, not read as a field."""

    rendered = template.format(schema="ALLOWED-FIELDS")

    assert "ALLOWED-FIELDS" in rendered
    assert "{schema}" not in rendered


def test_extractor_renders_the_schema_into_the_system_prompt():
    create = MagicMock(return_value=_chat_profile_tool_response({"updates": []}))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    extractor = OpenAICompatibleProfileExtractor(
        model="profile-test-model", client=client, api_mode="chat_completions"
    )

    result = extractor.extract(
        "user",
        [InteractionMessage(role="user", content="删除这个")],
        UserProfile(user_id="user"),
    )

    assert result.updates == []
    instructions = create.call_args.kwargs["messages"][0]["content"]
    assert "submit_user_profile_updates" in instructions
    assert "basic.preferred_name" in instructions
    assert "{schema}" not in instructions
    kwargs = create.call_args.kwargs
    assert "response_format" not in kwargs
    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": PROFILE_EXTRACTION_TOOL_NAME},
    }
    assert kwargs["parallel_tool_calls"] is False
    tool = kwargs["tools"][0]["function"]
    assert tool["name"] == PROFILE_EXTRACTION_TOOL_NAME
    assert tool["strict"] is True
    assert tool["parameters"]["additionalProperties"] is False
    assert "basic.preferred_name" in tool["parameters"]["properties"]["updates"][
        "items"
    ]["properties"]["path"]["enum"]


def test_extractor_ignores_assistant_json_without_required_tool_call():
    create = MagicMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"updates": []}', tool_calls=[])
                )
            ]
        )
    )
    extractor = OpenAICompatibleProfileExtractor(
        model="profile-test-model",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        api_mode="chat_completions",
    )

    with pytest.raises(RuntimeError, match="no user-profile tool call"):
        extractor.extract(
            "user",
            [InteractionMessage(role="user", content="请叫我小王")],
            UserProfile(user_id="user"),
        )


def test_responses_extractor_reads_function_call_arguments():
    arguments = {
        "updates": [
            {
                "action": "upsert",
                "path": "basic.preferred_name",
                "value": "小王",
                "confidence": 1.0,
                "explicit": True,
                "evidence_quote": "请叫我小王",
            }
        ]
    }
    create = MagicMock(
        return_value=SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name=PROFILE_EXTRACTION_TOOL_NAME,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            ]
        )
    )
    extractor = OpenAICompatibleProfileExtractor(
        model="profile-test-model",
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        api_mode="responses",
    )

    result = extractor.extract(
        "user",
        [InteractionMessage(role="user", content="请叫我小王")],
        UserProfile(user_id="user"),
    )

    assert [update.path for update in result.updates] == ["basic.preferred_name"]
    kwargs = create.call_args.kwargs
    assert kwargs["tool_choice"] == {
        "type": "function",
        "name": PROFILE_EXTRACTION_TOOL_NAME,
    }
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["tools"][0]["name"] == PROFILE_EXTRACTION_TOOL_NAME
    assert kwargs["tools"][0]["strict"] is True


def test_layered_extractor_rejects_l2_only_evidence_quote():
    create = MagicMock(
        return_value=_chat_profile_tool_response(
            {
                "updates": [
                    {
                        "action": "upsert",
                        "path": "basic.preferred_name",
                        "value": "小王",
                        "confidence": 1.0,
                        "explicit": True,
                        "evidence_quote": "用户希望称呼为小王",
                    }
                ]
            }
        )
    )
    extractor = OpenAICompatibleProfileExtractor(
        model="profile-test-model",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        api_mode="chat_completions",
    )
    messages = [
        InteractionMessage(
            role="assistant",
            content="用户希望称呼为小王",
            source="l2",
            source_ref="scene.md",
        ),
        InteractionMessage(
            role="user",
            content="该用户提供了稳定的称呼偏好。",
            source="l1",
            source_ref="atom-1",
        ),
    ]

    result = extractor.extract("user", messages, UserProfile(user_id="user"))

    assert result.updates == []
    instructions = create.call_args.kwargs["messages"][0]["content"]
    assert "source=l1" in instructions
    assert "source=l2" in instructions
    # L1 is the only evidence; L2 may inform judgement but cannot support a
    # field or a quote on its own.
    assert "唯一的事实来源" in instructions
    assert "不得单独支撑任何字段" in instructions
    # The conversation-shaped contract must not leak into a layered batch.
    assert "本次输入：已持久化的分层记忆" in instructions
    assert "本次输入：原始会话" not in instructions
    payload = json.loads(create.call_args.kwargs["messages"][1]["content"])
    assert [message["source"] for message in payload["messages"]] == ["l2", "l1"]


def test_conversation_input_gets_its_own_evidence_contract():
    """A raw-conversation batch must not be told to look for L1 atoms.

    Only the CLI still passes a real conversation; the runtime feeds L1/L2.
    Each shape declares what counts as evidence for itself so the shared body
    is never read as if it described the other one.
    """
    create = MagicMock(return_value=_chat_profile_tool_response({"updates": []}))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    extractor = OpenAICompatibleProfileExtractor(
        model="profile-test-model", client=client, api_mode="chat_completions"
    )

    extractor.extract(
        "user",
        [
            InteractionMessage(role="user", content="我是程序员"),
            InteractionMessage(role="assistant", content="了解。"),
        ],
        UserProfile(user_id="user"),
    )

    instructions = create.call_args.kwargs["messages"][0]["content"]
    assert "本次输入：原始会话" in instructions
    assert "role=user 是唯一的事实来源" in instructions
    # Assistant turns disambiguate the user but are not themselves facts. That
    # rule is meaningful only here, so it must not go missing on this path.
    assert "assistant 未被 user 确认的内容不是用户事实" in instructions
    assert "本次输入：已持久化的分层记忆" not in instructions
    assert "source=l1" not in instructions


def test_layered_extractor_accepts_l1_evidence_quote(caplog):
    create = MagicMock(
        return_value=_chat_profile_tool_response(
            {
                "updates": [
                    {
                        "action": "upsert",
                        "path": "basic.preferred_name",
                        "value": "小王",
                        "confidence": 1.0,
                        "explicit": True,
                        "evidence_quote": "请叫我小王",
                    }
                ]
            }
        )
    )
    extractor = OpenAICompatibleProfileExtractor(
        model="profile-test-model",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        api_mode="chat_completions",
    )

    with caplog.at_level("INFO", logger="kylin_memory.memory_debug"):
        result = extractor.extract(
            "user",
            [
                InteractionMessage(
                    role="assistant",
                    content="称呼偏好",
                    source="l2",
                    source_ref="scene.md",
                ),
                InteractionMessage(
                    role="user",
                    content="请叫我小王",
                    source="l1",
                    source_ref="atom-1",
                ),
            ],
            UserProfile(user_id="user"),
        )

    assert [update.path for update in result.updates] == ["basic.preferred_name"]
    event = json.loads(
        next(
            record.getMessage().split(" ", 1)[1]
            for record in caplog.records
            if record.getMessage().startswith("MEMORY_LLM_INPUT ")
        )
    )
    assert event["layer"] == "L3"
    assert event["api_mode"] == "chat_completions"
    assert "source=l1" in event["messages"][0]["content"]
    assert "请叫我小王" in event["messages"][1]["content"]
    assert event["tools"][0]["function"]["name"] == PROFILE_EXTRACTION_TOOL_NAME
    output = json.loads(
        next(
            record.getMessage().split(" ", 1)[1]
            for record in caplog.records
            if record.getMessage().startswith("MEMORY_LLM_OUTPUT ")
        )
    )
    assert output["layer"] == "L3"
    assert output["api_mode"] == "chat_completions"
    assert (
        output["response"]["choices"][0]["message"]["tool_calls"][0]
        ["function"]["name"]
        == PROFILE_EXTRACTION_TOOL_NAME
    )


def test_chunking_preserves_layered_source_metadata():
    chunks = _chunk_messages(
        [
            InteractionMessage(
                role="user",
                content="x" * 300,
                source="l1",
                source_ref="atom-long",
            )
        ],
        max_chars=100,
    )

    pieces = [message for chunk in chunks for message in chunk]
    assert pieces
    assert all(message.source == "l1" for message in pieces)
    assert all(message.source_ref == "atom-long" for message in pieces)


def test_legacy_profile_fields_are_not_part_of_the_current_schema():
    with pytest.raises(ValueError, match="unknown profile paths"):
        UserProfile(
            user_id="user",
            entries={
                "context.current_task": ProfileEntry(
                    value=["调研智能体记忆"],
                    confidence=1.0,
                    source="explicit",
                    lifecycle="dynamic",
                )
            },
        )

    with pytest.raises(ValueError, match="unsupported profile path"):
        ProfileUpdate(
            path="context.current_task",
            value=["新任务"],
            confidence=1.0,
            explicit=True,
            evidence_quote="新任务",
        )


def test_runtime_observe_logs_real_extractor_call(caplog):
    result = SimpleNamespace(
        applied=["basic.preferred_name"],
        deleted=[],
        conflicts=[],
        rejected=[],
    )
    extractor = SimpleNamespace(model="profile-test-model")
    service = MagicMock(extractor=extractor)
    service.observe.return_value = result
    runtime = RuntimeUserProfile(service, "user")

    with caplog.at_level("INFO", logger="kylin_memory.user_profile_runtime"):
        assert runtime.observe([{"role": "user", "content": "叫我小王"}]) is True

    service.observe.assert_called_once()
    assert "extraction LLM call starting" in caplog.text
    assert "model=profile-test-model" in caplog.text
    assert "extraction LLM call completed: applied=1" in caplog.text


def test_runtime_observe_retries_then_reports_success():
    result = SimpleNamespace(
        applied=["basic.preferred_name"],
        deleted=[],
        conflicts=[],
        rejected=[],
    )
    extractor = SimpleNamespace(model="profile-test-model")
    service = MagicMock(extractor=extractor)
    service.observe.side_effect = [
        RuntimeError("provider temporarily unavailable"),
        RuntimeError("invalid profile JSON"),
        result,
    ]
    events = []
    runtime = RuntimeUserProfile(
        service,
        "user",
        max_attempts=3,
        retry_base_delay_seconds=0,
    )

    assert runtime.observe(
        [{"role": "user", "content": "叫我小王"}],
        status_callback=events.append,
    ) is True

    assert service.observe.call_count == 3
    assert [event["status"] for event in events] == [
        "retrying",
        "retrying",
        "success",
    ]
    assert events[-1] == {
        "status": "success",
        "attempts": 3,
        "applied": 1,
        "deleted": 0,
        "conflicts": 0,
        "rejected": 0,
    }


def test_failed_extraction_is_not_marked_committed_and_reports_cli_reason():
    extractor = SimpleNamespace(model="profile-test-model")
    service = MagicMock(extractor=extractor)
    service.observe.side_effect = RuntimeError("upstream 503")
    runtime = RuntimeUserProfile(
        service,
        "user",
        max_attempts=2,
        retry_base_delay_seconds=0,
    )
    emitted = []
    batch = _profile_batch(
        "用户喜欢简洁回答", fingerprint="fp-failed", source_ref="atom-failed"
    )
    manager = _memory_manager(batch)
    agent = SimpleNamespace(
        session_id="session-1",
        platform="cli",
        _emit_status=emitted.append,
        _memory_manager=manager,
        _user_profile_runtime=runtime,
        _user_profile_committed_sessions={},
    )
    messages = [{"role": "user", "content": "我喜欢简洁回答"}]

    outcome = commit_user_profile_session(agent, messages)

    assert outcome == {
        "status": "failed",
        "error": "upstream 503",
        "attempts": 2,
    }
    assert service.observe.call_count == 2
    assert agent._user_profile_committed_sessions == {}
    manager.acknowledge_profile_sources.assert_not_called()
    assert emitted == [
        "User profile extraction failed (attempt 1/2): upstream 503. "
        "Retrying in 0.0s...",
        "User profile extraction failed after 2 attempts: upstream 503",
    ]


def test_successful_commit_reports_update_summary_only_to_terminal_frontends():
    result = SimpleNamespace(
        applied=["basic.preferred_name", "preferences.answer_style"],
        deleted=["interests.short_term"],
        conflicts=[],
        rejected=[],
    )
    messages = [{"role": "user", "content": "叫我小王，回答简洁一些"}]

    def make_agent(platform):
        runtime = RuntimeUserProfile(
            MagicMock(
                extractor=SimpleNamespace(model="profile-test-model"),
                observe=MagicMock(return_value=result),
            ),
            "user",
            retry_base_delay_seconds=0,
        )
        batch = _profile_batch(
            "用户要求叫他小王并简洁回答",
            fingerprint=f"fp-{platform}",
            source_ref=f"atom-{platform}",
        )
        return SimpleNamespace(
            session_id=f"{platform}-session",
            platform=platform,
            _emit_status=MagicMock(),
            status_callback=MagicMock(),
            _memory_manager=_memory_manager(batch),
            _user_profile_runtime=runtime,
            _user_profile_committed_sessions={},
        )

    cli_agent = make_agent("cli")
    tui_agent = make_agent("tui")
    gateway_agent = make_agent("telegram")
    commit_user_profile_session(cli_agent, messages)
    commit_user_profile_session(tui_agent, messages)
    commit_user_profile_session(gateway_agent, messages)

    cli_agent._emit_status.assert_called_once_with(
        "User profile updated: 2 added or changed, 1 deleted."
    )
    tui_agent.status_callback.assert_called_once_with(
        "lifecycle", "User profile updated: 2 added or changed, 1 deleted."
    )
    tui_agent._emit_status.assert_not_called()
    gateway_agent._emit_status.assert_not_called()
    gateway_agent.status_callback.assert_not_called()
    for agent in (cli_agent, tui_agent, gateway_agent):
        agent._memory_manager.acknowledge_profile_sources.assert_called_once()
        ack_kwargs = agent._memory_manager.acknowledge_profile_sources.call_args.kwargs
        assert ack_kwargs["changed"] is True
        assert ack_kwargs["output_refs"] == [
            "profile:v1:04f8996da763b7a969b1028ee3007569"
        ]


def test_profile_error_summary_redacts_credentials():
    summary = _error_summary(
        RuntimeError(
            "request failed Authorization: Bearer sk-test-1234567890abcdef"
        )
    )

    assert summary == "request failed Authorization: Bearer ***"
    assert "sk-test" not in summary
