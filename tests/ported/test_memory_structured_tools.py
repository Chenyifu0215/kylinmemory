from __future__ import annotations
from kylin_memory._vendor.agent.auxiliary_client import call_llm

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kylin_memory.anthropic_adapter import build_anthropic_kwargs
from kylin_memory.l1_extraction import (
    L1_EXTRACTION_TOOL_NAME,
    OpenAICompatibleAtomExtractor,
    SemanticExtractionResponseError,
    _l1_extraction_tool,
)
from kylin_memory.auxiliary_client import (
    _build_call_kwargs,
    _CodexCompletionsAdapter,
    extract_tool_call_arguments,
)
from kylin_memory.l2_extraction import (
    L2_SCENE_TOOL,
    L2_SCENE_TOOL_NAME,
    OpenAICompatibleSceneConsolidator,
)
from kylin_memory.memory_layers import Atom, ScenarioStore
from kylin_memory.memory_prompts import (
    CONFLICT_DETECTION_SYSTEM_PROMPT,
    EXTRACT_MEMORIES_SYSTEM_PROMPT,
    EXTRACT_WORK_MEMORIES_SYSTEM_PROMPT,
    WORK_CONFLICT_DETECTION_SYSTEM_PROMPT,
)
from kylin_memory.memory_pipeline import (
    _L1_CONFLICT_TOOL_NAME,
    MemoryPipelineManager,
    _l1_conflict_tool,
)


def _response(tool_name: str, arguments):
    encoded = (
        json.dumps(arguments, ensure_ascii=False)
        if isinstance(arguments, dict)
        else arguments
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[
            SimpleNamespace(function=SimpleNamespace(
                name=tool_name, arguments=encoded
            ))
        ]))]
    )


def _text_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content,
            tool_calls=None,
        ))]
    )


def _atom(record_id: str, content: str) -> Atom:
    return Atom.from_mapping({
        "id": record_id,
        "content": content,
        "type": "persona",
        "priority": 80,
        "scene_name": "学习偏好",
        "source_message_ids": [1],
        "metadata": {},
        "timestamps": [],
    }, mode="chat", new_id=False)


def test_l1_extractor_requests_and_consumes_required_tool_call(caplog):
    response = _response(L1_EXTRACTION_TOOL_NAME, {
        "scenes": [{
            "scene_name": "学习偏好",
            "message_ids": [1],
            "memories": [{
                "content": "用户喜欢数学。",
                "type": "persona",
                "priority": 80,
                "source_message_ids": [1],
                "metadata": {},
                "timestamps": [],
            }],
        }],
    })
    extractor = OpenAICompatibleAtomExtractor(max_attempts=1)
    with caplog.at_level("INFO", logger="kylin_memory.memory_debug"):
        with patch("kylin_memory.auxiliary_client.call_llm", return_value=response) as call:
            result = extractor.extract(
                [{"id": 1, "role": "user", "content": "我喜欢数学"}],
                session_id="session-l1",
            )

    assert len(result.memories) == 1
    kwargs = call.call_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == L1_EXTRACTION_TOOL_NAME
    assert kwargs["tool_choice"]["type"] == "function"
    event = json.loads(
        next(
            record.getMessage().split(" ", 1)[1]
            for record in caplog.records
            if record.getMessage().startswith("MEMORY_LLM_INPUT ")
        )
    )
    assert event["layer"] == "L1"
    assert event["session_id"] == "session-l1"
    assert "我喜欢数学" in event["messages"][1]["content"]
    assert event["tools"][0]["function"]["name"] == L1_EXTRACTION_TOOL_NAME
    output = json.loads(
        next(
            record.getMessage().split(" ", 1)[1]
            for record in caplog.records
            if record.getMessage().startswith("MEMORY_LLM_OUTPUT ")
        )
    )
    assert output["layer"] == "L1"
    assert output["session_id"] == "session-l1"
    assert (
        output["response"]["choices"][0]["message"]["tool_calls"][0]
        ["function"]["name"]
        == L1_EXTRACTION_TOOL_NAME
    )


def test_l1_extractor_rejects_text_json_without_tool_call():
    response = _text_response('{"scenes": []}')
    extractor = OpenAICompatibleAtomExtractor(max_attempts=1)

    with patch("kylin_memory.auxiliary_client.call_llm", return_value=response):
        with pytest.raises(SemanticExtractionResponseError) as exc_info:
            extractor.extract([{"id": 1, "role": "user", "content": "我喜欢数学"}])

    assert exc_info.value.diagnostic_code == "tool_call_missing_or_invalid"


def test_l1_extractor_rejects_tool_arguments_without_scenes_array():
    extractor = OpenAICompatibleAtomExtractor(max_attempts=1)

    with patch(
        "kylin_memory.auxiliary_client.call_llm",
        return_value=_response(L1_EXTRACTION_TOOL_NAME, {"scenes": {}}),
    ):
        with pytest.raises(SemanticExtractionResponseError) as exc_info:
            extractor.extract([{"id": 1, "role": "user", "content": "我喜欢数学"}])

    assert exc_info.value.diagnostic_code == "tool_arguments_invalid"


def test_l1_tool_schema_limits_types_by_prompt_mode():
    def memory_types(mode: str):
        return (
            _l1_extraction_tool(mode)["function"]["parameters"]
            ["properties"]["scenes"]["items"]["properties"]
            ["memories"]["items"]["properties"]["type"]["enum"]
        )

    assert memory_types("chat") == ["persona", "episodic", "instruction"]
    assert memory_types("code") == [
        "work_fact", "work_task", "work_method", "work_artifact",
    ]


@pytest.mark.parametrize(
    "prompt",
    [
        EXTRACT_MEMORIES_SYSTEM_PROMPT,
        EXTRACT_WORK_MEMORIES_SYSTEM_PROMPT,
        CONFLICT_DETECTION_SYSTEM_PROMPT,
        WORK_CONFLICT_DETECTION_SYSTEM_PROMPT,
    ],
)
def test_every_l1_prompt_stores_the_user_without_a_name(prompt):
    """Stored memories refer to "用户", never to a name picked out of a message.

    A name in the content makes an otherwise reusable memory look
    person-specific and leaks an identifier into every downstream layer, so
    the rule has to hold in the merge prompts too, not just at extraction.
    """
    assert "不要写入姓名、昵称或称谓" in prompt
    assert "（姓名）" not in prompt
    assert "(姓名)" not in prompt


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (EXTRACT_MEMORIES_SYSTEM_PROMPT, "user 消息作为依据"),
        (EXTRACT_WORK_MEMORIES_SYSTEM_PROMPT, "user 消息为依据"),
        # Merging happens after extraction, so its guard is phrased against the
        # inputs it actually has: the candidates and the new memory.
        (
            CONFLICT_DETECTION_SYSTEM_PROMPT,
            "只能重组候选记忆和新记忆中已有的信息",
        ),
        (
            WORK_CONFLICT_DETECTION_SYSTEM_PROMPT,
            "只能重组候选记忆和新记忆中已有的信息",
        ),
    ],
)
def test_every_l1_prompt_limits_content_to_supplied_evidence(prompt, expected):
    assert expected in prompt


def test_l1_tool_schema_does_not_request_writer_owned_timestamps():
    """``_apply_decisions`` anchors every stored record at writer time.

    Asking the model to derive a value that is then always overwritten only
    widens the surface for schema violations, so the field must stay out of
    the extraction contract.
    """
    memory_schema = (
        _l1_extraction_tool("chat")["function"]["parameters"]
        ["properties"]["scenes"]["items"]["properties"]
        ["memories"]["items"]
    )

    assert "timestamps" not in memory_schema["properties"]
    assert "timestamps" not in memory_schema["required"]


def test_l1_tool_schema_states_field_rules_the_host_enforces():
    """Per-type rules are rejected at persistence, so they belong in-schema."""
    for mode, expected in (
        ("chat", "persona 50, episodic 60, instruction 70"),
        ("code", "minimum 70 for every work type"),
    ):
        memory_properties = (
            _l1_extraction_tool(mode)["function"]["parameters"]
            ["properties"]["scenes"]["items"]["properties"]
            ["memories"]["items"]["properties"]
        )
        assert expected in memory_properties["priority"]["description"]
        assert "allowlist" in memory_properties["metadata"]["description"]
        assert memory_properties["source_message_ids"]["minItems"] == 1
        assert (
            "at least one user message ID"
            in memory_properties["source_message_ids"]["description"]
        )


def test_l1_conflict_tool_schema_limits_merged_type_by_prompt_mode():
    def merged_types(mode: str):
        return (
            _l1_conflict_tool(mode)["function"]["parameters"]
            ["properties"]["decisions"]["items"]["properties"]
            ["merged_type"]["enum"]
        )

    assert merged_types("chat") == ["persona", "episodic", "instruction"]
    assert merged_types("code") == [
        "work_fact", "work_task", "work_method", "work_artifact",
    ]


def test_cross_type_merge_narrows_metadata_to_the_merged_type(tmp_path):
    """A cross-type merge must not be discarded by the metadata allowlist.

    The conflict tool has no metadata argument, so the new record's metadata
    carries over.  ``work_task`` permits ``owner`` while ``work_method`` does
    not, so without narrowing, a merge the prompt actively encourages would
    fail validation and silently drop both the memory and its targets.
    """
    manager = MemoryPipelineManager(
        tmp_path,
        config={"prompt_mode": "code", "embedding": {"mode": "disabled"}},
    )
    new_atom = Atom.from_mapping({
        "id": "new",
        "content": "后端团队需要在周五前完成追溯表设计。",
        "type": "work_task",
        "priority": 80,
        "scene_name": "团队推进运行时",
        "source_message_ids": [1],
        "metadata": {"owner": "backend", "status": "todo"},
        "timestamps": [],
    }, mode="code", new_id=False)
    old_atom = Atom.from_mapping({
        "id": "old",
        "content": "追溯表设计应优先保证可回溯性。",
        "type": "work_method",
        "priority": 80,
        "scene_name": "团队推进运行时",
        "source_message_ids": [2],
        "metadata": {},
        "timestamps": [],
    }, mode="code", new_id=False)

    try:
        manager.atoms.upsert(old_atom)
        written = manager._apply_decisions(
            [(new_atom, [old_atom])],
            [{
                "record_id": "new",
                "action": "merge",
                "target_ids": ["old"],
                "merged_content": "追溯表设计以可回溯性优先，由后端团队在周五前完成。",
                "merged_type": "work_method",
                "merged_priority": 90,
                "merged_timestamps": [],
            }],
            mode="code",
        )
    finally:
        manager.close()

    assert len(written) == 1
    assert written[0].type == "work_method"
    # ``owner`` belongs to work_task only and must be dropped rather than
    # taking the whole decision down with it.
    assert written[0].metadata == {}


def test_l2_consolidator_requests_and_consumes_required_tool_call(tmp_path, caplog):
    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="chat")
    atom = Atom.from_mapping({
        "id": "a1", "content": "用户喜欢数学。", "type": "persona",
        "priority": 80, "scene_name": "学习偏好", "source_message_ids": [1],
        "metadata": {}, "timestamps": [],
    }, mode="chat", new_id=False)
    response = _response(L2_SCENE_TOOL_NAME, {
        "action": "create", "target_files": [], "scene_name": "学习偏好",
        "summary": "用户的学习偏好", "body": "## 学习偏好\n- 用户喜欢数学。",
        "delete_files": [],
    })
    consolidator = OpenAICompatibleSceneConsolidator()
    with caplog.at_level("INFO", logger="kylin_memory.memory_debug"):
        with patch("kylin_memory.auxiliary_client.call_llm", return_value=response) as call:
            result = consolidator(
                [atom],
                store=store,
                scene_name="学习偏好",
                session_id="session-l2",
            )

    assert result["action"] == "create"
    assert result["body"].startswith("##")
    kwargs = call.call_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == L2_SCENE_TOOL_NAME
    assert kwargs["tool_choice"]["function"]["name"] == L2_SCENE_TOOL_NAME
    event = json.loads(
        next(
            record.getMessage().split(" ", 1)[1]
            for record in caplog.records
            if record.getMessage().startswith("MEMORY_LLM_INPUT ")
        )
    )
    assert event["layer"] == "L2"
    assert event["session_id"] == "session-l2"
    assert "用户喜欢数学" in event["messages"][1]["content"]
    assert event["tools"][0]["function"]["name"] == L2_SCENE_TOOL_NAME
    output = json.loads(
        next(
            record.getMessage().split(" ", 1)[1]
            for record in caplog.records
            if record.getMessage().startswith("MEMORY_LLM_OUTPUT ")
        )
    )
    assert output["layer"] == "L2"
    assert output["session_id"] == "session-l2"
    assert (
        output["response"]["choices"][0]["message"]["tool_calls"][0]
        ["function"]["name"]
        == L2_SCENE_TOOL_NAME
    )


def test_l2_prompt_uses_atomic_transaction_contract_and_includes_scene_content(tmp_path):
    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="chat")
    atom = _atom("a1", "用户现在偏好应用数学。")
    filename = store.apply_action(
        [atom], action="create", scene_name="学习偏好",
        summary="用户喜欢数学",
        body="## 核心叙事\n用户原本喜欢数学。",
    )["changed"][0]
    response = _response(L2_SCENE_TOOL_NAME, {
        "action": "update", "target_files": [filename],
        "scene_name": "学习偏好", "summary": "用户的数学学习偏好",
        "body": "## 核心叙事\n用户从数学进一步聚焦到应用数学。",
        "delete_files": [],
    })

    with patch("kylin_memory.auxiliary_client.call_llm", return_value=response) as call:
        result = OpenAICompatibleSceneConsolidator()(
            [atom], store=store, scene_name="学习偏好",
        )

    system_prompt = call.call_args.kwargs["messages"][0]["content"]
    user_prompt = call.call_args.kwargs["messages"][1]["content"]
    assert result["action"] == "update"
    assert "必须通过 `l2_scene_transaction` 工具调用" in system_prompt
    assert "不使用 read/write/edit" in system_prompt
    assert "当 UPDATE 和 CREATE 之间难以判断时，选 UPDATE" in system_prompt
    assert "Existing Scene Documents" in user_prompt
    assert '"id": "a1"' in user_prompt
    assert '"created_at":' in user_prompt
    assert "用户原本喜欢数学。" in user_prompt
    assert filename in user_prompt
    assert "忽略上方关于 read/write/edit" not in system_prompt


def test_l2_code_prompt_preserves_work_method_scene_contract(tmp_path):
    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="code")
    atom = Atom.from_mapping({
        "id": "a1", "content": "回归测试应先覆盖高风险路径。", "type": "work_method",
        "priority": 80, "scene_name": "测试方法", "source_message_ids": [1],
        "metadata": {}, "timestamps": [],
    }, mode="code", new_id=False)
    response = _response(L2_SCENE_TOOL_NAME, {
        "action": "create", "target_files": [], "scene_name": "测试方法",
        "summary": "高风险路径优先的回归测试方法",
        "body": "## 任务场景\n回归测试\n\n## 核心 SOP\n先覆盖高风险路径。",
        "delete_files": [],
    })

    with patch("kylin_memory.auxiliary_client.call_llm", return_value=response) as call:
        OpenAICompatibleSceneConsolidator()(
            [atom], store=store, scene_name="测试方法", mode="code",
        )

    system_prompt = call.call_args.kwargs["messages"][0]["content"]
    assert "核心 SOP" in system_prompt
    assert "判断逻辑" in system_prompt
    assert "禁忌与反模式" in system_prompt
    assert "不是项目日报、聊天摘要、任务清单或个人画像" in system_prompt
    # The agent serves one person, so scenes are named after the user's task,
    # not after a collaboration unit.
    assert "只有一位使用者" in system_prompt
    assert "[工作对象][目标活动]" in system_prompt
    assert '禁止"团队""我们""成员""大家"等多人主语' in system_prompt


def test_l2_merge_tool_call_must_delete_every_target(tmp_path):
    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="chat")
    atom = _atom("a1", "整合学习偏好。")
    files = [
        store.apply_action(
            [atom], action="create", scene_name=name, summary=name,
            body=f"## 核心叙事\n{name}",
        )["changed"][0]
        for name in ("数学学习", "物理学习")
    ]
    response = _response(L2_SCENE_TOOL_NAME, {
        "action": "merge", "target_files": files, "scene_name": "理科学习",
        "summary": "理科学习偏好", "body": "## 核心叙事\n用户喜欢理科。",
        "delete_files": files[:1],
    })

    with patch("kylin_memory.auxiliary_client.call_llm", return_value=response):
        with pytest.raises(ValueError, match="must delete every target"):
            OpenAICompatibleSceneConsolidator()(
                [atom], store=store, scene_name="理科学习",
            )


def test_l2_create_tool_call_is_deferred_to_store_capacity_policy(tmp_path):
    store = ScenarioStore(tmp_path, max_scenes=2, prompt_mode="chat")
    atom = _atom("a1", "用户喜欢数学。")
    store.apply_action(
        [atom], action="create", scene_name="数学学习", summary="数学学习",
        body="## 核心叙事\n用户喜欢数学。",
    )
    response = _response(L2_SCENE_TOOL_NAME, {
        "action": "create", "target_files": [], "scene_name": "物理学习",
        "summary": "物理学习", "body": "## 核心叙事\n用户喜欢物理。",
        "delete_files": [],
    })

    with patch("kylin_memory.auxiliary_client.call_llm", return_value=response):
        result = OpenAICompatibleSceneConsolidator()(
            [atom], store=store, scene_name="物理学习",
        )

    assert result["action"] == "create"


@pytest.mark.parametrize("body", [
    "preface\n## 核心叙事\n内容",
    "-----META-START-----\ncreated: now\n## 核心叙事\n内容",
    "## 核心叙事\n```text\n内容\n```",
])
def test_l2_tool_call_rejects_noncanonical_scene_body(tmp_path, body):
    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="chat")
    response = _response(L2_SCENE_TOOL_NAME, {
        "action": "create", "target_files": [], "scene_name": "学习偏好",
        "summary": "学习偏好", "body": body, "delete_files": [],
    })

    with patch("kylin_memory.auxiliary_client.call_llm", return_value=response):
        with pytest.raises(ValueError, match="no usable scene markdown"):
            OpenAICompatibleSceneConsolidator()(
                [_atom("a1", "用户喜欢数学。")],
                store=store,
                scene_name="学习偏好",
            )


def test_l2_consolidator_rejects_text_json_without_tool_call(tmp_path):
    store = ScenarioStore(tmp_path, max_scenes=5, prompt_mode="chat")
    response = _text_response(json.dumps({
        "action": "create",
        "target_files": [],
        "scene_name": "学习偏好",
        "summary": "摘要",
        "body": "## 学习偏好\n- 用户喜欢数学。",
        "delete_files": [],
    }))

    with patch("kylin_memory.auxiliary_client.call_llm", return_value=response):
        with pytest.raises(ValueError, match="no valid l2_scene_transaction tool call"):
            OpenAICompatibleSceneConsolidator()(
                [_atom("a1", "用户喜欢数学。")],
                store=store,
                scene_name="学习偏好",
            )


def test_tool_argument_extractor_supports_dict_calls_and_exact_name():
    response = SimpleNamespace(choices=[SimpleNamespace(message={
        "tool_calls": [{
            "function": {
                "name": L1_EXTRACTION_TOOL_NAME,
                "arguments": {"scenes": []},
            },
        }],
    })])

    assert extract_tool_call_arguments(
        response, L1_EXTRACTION_TOOL_NAME
    ) == {"scenes": []}
    assert extract_tool_call_arguments(response, "another_tool") is None


@pytest.mark.parametrize("arguments", ["{", "[]"])
def test_tool_argument_extractor_rejects_invalid_object_arguments(arguments):
    assert extract_tool_call_arguments(
        _response(L1_EXTRACTION_TOOL_NAME, arguments),
        L1_EXTRACTION_TOOL_NAME,
    ) is None


def test_openai_call_kwargs_forward_named_tool_choice():
    choice = {"type": "function", "function": {"name": L2_SCENE_TOOL_NAME}}
    kwargs = _build_call_kwargs(
        provider="openai",
        model="gpt-5",
        messages=[{"role": "user", "content": "merge"}],
        tools=[L2_SCENE_TOOL],
        tool_choice=choice,
    )

    assert kwargs["tools"] == [L2_SCENE_TOOL]
    assert kwargs["tool_choice"] == choice


def test_deepseek_thinking_is_disabled_for_named_structured_tool_choice():
    choice = {"type": "function", "function": {"name": L2_SCENE_TOOL_NAME}}
    kwargs = _build_call_kwargs(
        provider="auto",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1",
        messages=[{"role": "user", "content": "merge"}],
        tools=[L2_SCENE_TOOL],
        tool_choice=choice,
        extra_body={"thinking": {"type": "enabled"}},
    )

    assert kwargs["tool_choice"] == choice
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


def test_deepseek_v3_does_not_receive_thinking_override_for_tool_choice():
    choice = {"type": "function", "function": {"name": L2_SCENE_TOOL_NAME}}
    kwargs = _build_call_kwargs(
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        messages=[{"role": "user", "content": "merge"}],
        tools=[L2_SCENE_TOOL],
        tool_choice=choice,
    )

    assert "thinking" not in kwargs.get("extra_body", {})


def test_call_llm_retries_deepseek_tool_choice_error_with_thinking_disabled():
    client = MagicMock()
    client.base_url = "https://llm.example/v1"
    error = RuntimeError("Thinking mode does not support this tool_choice")
    ok = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None,
        tool_calls=[],
    ))])
    client.chat.completions.create.side_effect = [error, ok]
    choice = {"type": "function", "function": {"name": L2_SCENE_TOOL_NAME}}

    with (
        patch(
            "kylin_memory._vendor.agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "deepseek-v4-flash", "https://llm.example/v1", None, None),
        ),
        patch(
            "kylin_memory._vendor.agent.auxiliary_client._get_cached_client",
            return_value=(client, "deepseek-v4-flash"),
        ),
    ):
        result = call_llm(
            task=None,
            messages=[{"role": "user", "content": "extract"}],
            tools=[L2_SCENE_TOOL],
            tool_choice=choice,
        )

    assert result is ok
    assert client.chat.completions.create.call_count == 2
    retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
    assert retry_kwargs["tool_choice"] == choice
    assert retry_kwargs["extra_body"]["thinking"] == {"type": "disabled"}


def test_codex_responses_adapter_converts_named_tool_choice():
    arguments = {
        "action": "create",
        "target_files": [],
        "scene_name": "学习偏好",
        "summary": "摘要",
        "body": "## 学习偏好\n- 用户喜欢数学。",
        "delete_files": [],
    }
    final = SimpleNamespace(output=[SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name=L2_SCENE_TOOL_NAME,
        arguments=json.dumps(arguments, ensure_ascii=False),
    )], usage=None)

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(())

        def get_final_response(self):
            return final

    captured = {}

    def stream(**kwargs):
        captured.update(kwargs)
        return _Stream()

    client = MagicMock()
    client.responses.stream = stream
    response = _CodexCompletionsAdapter(client, "gpt-5-codex").create(
        messages=[{"role": "user", "content": "merge"}],
        tools=[L2_SCENE_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": L2_SCENE_TOOL_NAME},
        },
    )

    assert captured["tool_choice"] == {
        "type": "function",
        "name": L2_SCENE_TOOL_NAME,
    }
    assert captured["tools"][0]["name"] == L2_SCENE_TOOL_NAME
    assert extract_tool_call_arguments(response, L2_SCENE_TOOL_NAME) == arguments


def test_anthropic_adapter_converts_named_tool_choice():
    kwargs = build_anthropic_kwargs(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "merge"}],
        tools=[L2_SCENE_TOOL],
        max_tokens=1024,
        reasoning_config=None,
        tool_choice=L2_SCENE_TOOL_NAME,
    )

    assert kwargs["tool_choice"] == {
        "type": "tool",
        "name": L2_SCENE_TOOL_NAME,
    }


def test_anthropic_oauth_prefixes_named_tool_choice_with_schema():
    kwargs = build_anthropic_kwargs(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "merge"}],
        tools=[L2_SCENE_TOOL],
        max_tokens=1024,
        reasoning_config=None,
        tool_choice=L2_SCENE_TOOL_NAME,
        is_oauth=True,
    )

    assert kwargs["tools"][0]["name"] == f"mcp_{L2_SCENE_TOOL_NAME}"
    assert kwargs["tool_choice"] == {
        "type": "tool",
        "name": f"mcp_{L2_SCENE_TOOL_NAME}",
    }


def test_l1_conflict_resolver_requests_and_consumes_tool_call(tmp_path):
    manager = MemoryPipelineManager(
        tmp_path,
        config={"embedding": {"mode": "disabled"}},
    )
    new_atom = _atom("new", "用户现在喜欢应用数学。")
    old_atom = _atom("old", "用户喜欢数学。")
    response = _response(_L1_CONFLICT_TOOL_NAME, {
        "decisions": [{
            "record_id": "new",
            "action": "update",
            "target_ids": ["old"],
            "merged_content": "用户喜欢应用数学。",
            "merged_type": "persona",
            "merged_priority": 90,
            "merged_timestamps": [],
        }],
    })

    try:
        with patch("kylin_memory.auxiliary_client.call_llm", return_value=response) as call:
            decisions = manager._resolve_conflicts(
                [(new_atom, [old_atom])],
                mode="chat",
            )
    finally:
        manager.close()

    assert decisions == [{
        "record_id": "new",
        "action": "update",
        "target_ids": ["old"],
        "merged_content": "用户喜欢应用数学。",
        "merged_type": "persona",
        "merged_priority": 90,
        "merged_timestamps": [],
    }]
    kwargs = call.call_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == _L1_CONFLICT_TOOL_NAME
    assert kwargs["tool_choice"]["function"]["name"] == _L1_CONFLICT_TOOL_NAME


def test_l1_conflict_resolver_ignores_text_json_and_stores_all(tmp_path):
    manager = MemoryPipelineManager(
        tmp_path,
        config={"embedding": {"mode": "disabled"}},
    )
    new_atom = _atom("new", "用户现在喜欢应用数学。")
    old_atom = _atom("old", "用户喜欢数学。")
    response = _text_response(json.dumps({
        "decisions": [{
            "record_id": "new",
            "action": "skip",
            "target_ids": ["old"],
        }],
    }))

    try:
        with patch("kylin_memory.auxiliary_client.call_llm", return_value=response):
            decisions = manager._resolve_conflicts(
                [(new_atom, [old_atom])],
                mode="chat",
            )
    finally:
        manager.close()

    assert decisions == [{
        "record_id": "new",
        "action": "store",
        "target_ids": [],
    }]
