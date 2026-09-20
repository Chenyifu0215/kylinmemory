"""Behavioral regressions at the standalone boundary (no live network calls)."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from kylin_memory import MemorySystem
from kylin_memory.auxiliary_client import _get_cached_client, call_llm, _resolve_task_provider_model
from kylin_memory.config import DEFAULT_CONFIG, get_env_value, merge_config, runtime_context
from kylin_memory.plugin import LayeredMemoryProvider
from kylin_memory.routing import resolve_runtime
from test_plugin import fake_host
from test_runtime import Model, config


def test_switch_commits_old_session_before_changing_id_and_survives_restart(tmp_path):
    model = Model()
    with MemorySystem(tmp_path, config=config(), session_id='old', client=model) as memory:
        memory.observe('以后请叫我小王。', '好的。')
        assert memory.status()['l1'] == 0
        memory.switch_session('new')
        assert memory.status()['l1'] == 1
        assert 'old' in memory.agent._user_profile_committed_sessions
        assert '小王' in memory.context('称呼')['system']
        assert memory.recall('小王')[0]['sessionId'] == 'old'
        assert memory.session_db.get_messages('new') == []
    with MemorySystem(tmp_path, config=config(), session_id='restart', client=model) as memory:
        assert memory.recall('小王')
        assert '小王' in memory.context('称呼')['system']


def test_invalid_switch_does_not_commit_or_change_session(tmp_path):
    model = Model()
    with MemorySystem(tmp_path, config=config(), session_id='old', client=model) as memory:
        memory.observe('以后请叫我小王。')
        with pytest.raises(ValueError):
            memory.switch_session('')
        assert memory.session_id == 'old'
        assert not model.calls


def test_incremental_import_without_timestamps_never_loses_same_tick_rows(tmp_path, monkeypatch):
    model = Model()
    with MemorySystem(tmp_path, config=config(), client=model) as memory:
        instant = memory.provider._l0_recorder.plugin_start_ms / 1000
        monkeypatch.setattr('kylin_memory.runtime.time.time', lambda: instant)
        for _ in range(3):
            memory.ingest([{'role': 'user', 'content': '以后请叫我小王。'},
                           {'role': 'assistant', 'content': '好的。'}])
        assert memory.status()['l0'] == 6
        messages = memory.session_db.get_messages('default')
        timestamps = [int(m['timestamp'] * 1000) for m in messages]
        assert len(set(timestamps)) == 6


def test_disabled_layers_do_not_require_model_credentials(tmp_path, monkeypatch):
    constructor = Mock(side_effect=AssertionError('No client should be created'))
    monkeypatch.setattr(openai, 'OpenAI', constructor)
    cfg = {'model': {'model': 'unused'}, 'memory': {'atom': {'enabled': False}},
           'user_profile': {'enabled': False}}
    with MemorySystem(tmp_path, config=cfg) as memory:
        assert memory.status()['l0'] == 0
        memory.switch_session('next')
    constructor.assert_not_called()


def test_provider_override_uses_own_endpoint_and_key(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-openrouter')
    constructor = Mock()
    monkeypatch.setattr(openai, 'OpenAI', constructor)
    with runtime_context(tmp_path, DEFAULT_CONFIG):
        _, model = _get_cached_client('openrouter', 'aux-model', main_runtime={
            'provider': 'custom', 'model': 'main-model', 'base_url': 'https://main.invalid/v1',
            'api_key': 'test-main'})
    assert model == 'aux-model'
    assert constructor.call_count == 1
    assert constructor.call_args.kwargs['api_key'] == 'test-openrouter'
    assert constructor.call_args.kwargs['base_url'] == 'https://openrouter.ai/api/v1'
    assert 'HTTP-Referer' in constructor.call_args.kwargs['default_headers']


def test_provider_config_and_missing_key_do_not_borrow_main_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with runtime_context(tmp_path, DEFAULT_CONFIG):
        assert _get_cached_client('openrouter', 'aux', main_runtime={'model': 'main', 'api_key': 'private-main'}) == (None, 'aux')
        with pytest.raises(RuntimeError, match='No LLM provider'):
            call_llm(provider='openrouter', model='aux', messages=[])
        assert _get_cached_client('unknown-service', 'aux', main_runtime={'model': 'main', 'api_key': 'private-main'}) == (None, 'aux')
    cfg = merge_config(DEFAULT_CONFIG, {'providers': {'example': {
        'base_url': 'https://example.invalid/v1', 'key_env': 'EXAMPLE_TEST_KEY'}}})
    monkeypatch.setenv('EXAMPLE_TEST_KEY', 'test-example')
    constructor = Mock()
    monkeypatch.setattr(openai, 'OpenAI', constructor)
    with runtime_context(tmp_path, cfg):
        _get_cached_client('example', 'aux', main_runtime={'model': 'main', 'api_key': 'private-main'})
    constructor.assert_called_once_with(api_key='test-example', base_url='https://example.invalid/v1')


def test_main_provider_and_dotenv_credentials_are_isolated(tmp_path, monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    (tmp_path / '.env').write_text('DEEPSEEK_API_KEY="test-dotenv"\n')
    constructor = Mock()
    monkeypatch.setattr(openai, 'OpenAI', constructor)
    cfg = merge_config(DEFAULT_CONFIG, {'model': {'provider': 'deepseek', 'model': 'deepseek-chat'}})
    with runtime_context(tmp_path, cfg):
        _get_cached_client()
        monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-env')
        assert get_env_value('DEEPSEEK_API_KEY') == 'test-env'
    constructor.assert_called_once_with(api_key='test-dotenv', base_url='https://api.deepseek.com/v1')


def test_explicit_route_is_not_mixed_with_task_route(tmp_path):
    cfg = merge_config(DEFAULT_CONFIG, {'auxiliary': {'atom_memory': {
        'provider': 'custom', 'base_url': 'https://task.invalid/v1', 'api_key': 'test-task'}}})
    with runtime_context(tmp_path, cfg):
        route = _resolve_task_provider_model('atom_memory', provider='openrouter', model='chosen')
    assert route[:4] == ('openrouter', 'chosen', None, None)


def test_live_provider_switch_does_not_keep_old_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-new')
    cfg = merge_config(DEFAULT_CONFIG, {'model': {'provider': 'custom', 'model': 'old',
        'base_url': 'https://old.invalid/v1', 'api_key': 'test-old'}})
    with runtime_context(tmp_path, cfg):
        route = resolve_runtime({'provider': 'openrouter', 'model': 'new'})
    assert not route['base_url']
    assert not route['api_key']
    assert route['provider'] == 'openrouter'


def test_source_fallback_argument_mismatch_is_preserved():
    # Source calls resolve_provider_client(base_url=...), whose parameter is
    # explicit_base_url. Preserve the source implementation, not v0.1.1's fix.
    from kylin_memory._vendor.agent.auxiliary_client import _resolve_single_provider
    with pytest.raises(TypeError, match="base_url"):
        _resolve_single_provider('openrouter', 'backup', 'https://backup.invalid/v1', 'test')


@pytest.mark.parametrize('mode', ['anthropic_messages'])
def test_native_sdk_transports_preserve_structured_tools(tmp_path, monkeypatch, mode):
    import json
    from kylin_memory.auxiliary_client import extract_tool_call_arguments
    requests, clients = [], []
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if mode == 'responses':
            data = {'id': 'resp_1', 'object': 'response', 'created_at': 1, 'status': 'completed',
                'model': 'test', 'output': [{'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1',
                    'name': 'extract', 'arguments': '{"ok": true}', 'status': 'completed'}]}
        else:
            data = {'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': 'test',
                'content': [{'type': 'tool_use', 'id': 'call_1', 'name': 'extract', 'input': {'ok': True}}],
                'stop_reason': 'tool_use', 'usage': {'input_tokens': 1, 'output_tokens': 1}}
        return httpx.Response(200, json=data)
    if mode == 'responses':
        owner, name = openai, 'OpenAI'
    else:
        import anthropic
        owner, name = anthropic, 'Anthropic'
    original = getattr(owner, name)
    def create(**kwargs):
        client = original(**kwargs, max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client
    monkeypatch.setattr(owner, name, create)
    try:
        with runtime_context(tmp_path, DEFAULT_CONFIG):
            result = call_llm(main_runtime={'provider': 'custom', 'model': 'test', 'api_mode': mode,
                'base_url': 'https://native.invalid/v1', 'api_key': 'test-native'},
                messages=[{'role': 'user', 'content': 'evidence'}],
                tools=[{'type': 'function', 'function': {'name': 'extract', 'parameters': {'type': 'object'}}}],
                tool_choice={'type': 'function', 'function': {'name': 'extract'}}, max_tokens=100)
        assert extract_tool_call_arguments(result, 'extract') == {'ok': True}
        assert requests[0]['tool_choice']['name'] == 'extract'
        assert requests[0]['tools'][0]['name'] == 'extract'
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize('failure', ['connection', 'payment', 'quota'])
def test_same_provider_fallback_is_skipped_like_source(tmp_path, monkeypatch, failure):
    requests = []
    clients = []
    real_client = openai.OpenAI
    def handle(request):
        import json
        body = json.loads(request.content)
        requests.append((request.url.host, request.headers['authorization'], body))
        if request.url.host == 'primary.invalid':
            if failure == 'connection':
                raise httpx.ConnectError('connection refused', request=request)
            return httpx.Response(402 if failure == 'payment' else 429,
                json={'error': {'message': 'quota exceeded', 'type': 'quota_exceeded'}})
        return httpx.Response(200, json={'id': 'r', 'object': 'chat.completion', 'created': 1,
            'model': body['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'ok'}}]})
    def create(**kwargs):
        client = real_client(**kwargs, max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client
    monkeypatch.setattr(openai, 'OpenAI', create)
    cfg = merge_config(DEFAULT_CONFIG, {'auxiliary': {'atom_memory': {
        'provider': 'custom', 'model': 'primary', 'base_url': 'https://primary.invalid/v1',
        'api_key': 'test-primary', 'fallback_chain': [{'provider': 'custom', 'model': 'backup',
            'base_url': 'https://backup.invalid/v1', 'api_key': 'test-backup'}]}}})
    runtime = {'model': 'main', 'base_url': 'https://main.invalid/v1', '_client_cache': {}}
    tool = {'type': 'function', 'function': {'name': 'extract', 'parameters': {'type': 'object'}}}
    try:
        with runtime_context(tmp_path, cfg):
            expected = openai.APIConnectionError if failure == 'connection' else openai.APIStatusError
            with pytest.raises(expected):
                call_llm(task='atom_memory', main_runtime=runtime,
                    messages=[{'role': 'user', 'content': 'evidence'}], tools=[tool], tool_choice='required')
        assert [r[0] for r in requests] == ['primary.invalid']
        assert requests[0][2]['tools'] == [tool]
    finally:
        for client in clients:
            client.close()


def test_validation_error_does_not_trigger_fallback(tmp_path, monkeypatch):
    request = httpx.Request('POST', 'https://primary.invalid/v1')
    client = Mock()
    client.chat.completions.create.side_effect = openai.BadRequestError('invalid schema',
        response=httpx.Response(400, request=request), body=None)
    constructor = Mock(return_value=client)
    monkeypatch.setattr(openai, 'OpenAI', constructor)
    cfg = merge_config(DEFAULT_CONFIG, {'auxiliary': {'atom_memory': {'fallback_chain': [
        {'provider': 'custom', 'base_url': 'https://backup.invalid/v1', 'model': 'backup'}]}}})
    with runtime_context(tmp_path, cfg), pytest.raises(openai.BadRequestError):
        call_llm(task='atom_memory', main_runtime={'client': client, 'model': 'primary'}, messages=[])
    constructor.assert_not_called()
    assert client.chat.completions.create.call_count == 1


def test_directory_provider_explicit_runtime_tracks_live_host_without_model_kwarg(tmp_path, monkeypatch):
    model = Model()
    fake_host(monkeypatch, tmp_path, model)
    runtime = {'model': 'first', 'client': model}
    provider = LayeredMemoryProvider(main_runtime=lambda: runtime)
    provider.initialize('s1', hermes_home=str(tmp_path))
    try:
        provider.system.observe('以后请叫我小王。', '好的。')
        runtime['model'] = 'second'
        provider.on_turn_start(1, 'test')
        provider.commit_session([])
        provider.system.consolidate()
        assert model.calls
        assert all(call['model'] == 'second' for call in model.calls)
    finally:
        provider.shutdown()


def test_directory_provider_reads_actual_source_live_runtime_without_model_kwarg(tmp_path, monkeypatch):
    import sys
    from kylin_memory._vendor.agent import auxiliary_client as source
    model = Model()
    fake_host(monkeypatch, tmp_path, model)
    host = sys.modules['agent.auxiliary_client']
    monkeypatch.setattr(host, '_read_main_model', source._read_main_model, raising=False)
    monkeypatch.setattr(host, '_read_main_provider', source._read_main_provider, raising=False)
    source.set_runtime_main('custom', 'first')
    provider = LayeredMemoryProvider()
    try:
        provider.initialize('s1', hermes_home=str(tmp_path))
        provider.system.observe('以后请叫我小王。', '好的。')
        source.set_runtime_main('openrouter', 'second')
        provider.on_turn_start(1, 'test')  # Actual host hook has no model/client.
        current = provider.system.current_runtime()
        assert current['model'] == 'second'
        assert current['provider'] == 'openrouter'
        assert not current.get('base_url')
        provider.commit_session([])
        provider.system.consolidate()
        assert model.calls and all(call['model'] == 'second' for call in model.calls)
    finally:
        provider.shutdown()
        source.clear_runtime_main()


@pytest.mark.parametrize('yaml_text', [
    'provider: custom\nbase_url: https://example.invalid/v1\nmax_turns: 23\n',
    'model:\n  default: ${MEMORY_TEST_MODEL}\nagent:\n  max_turns: 19\nmax_turns: 23\n',
    'model: [broken\n',
])
def test_config_normalization_matches_source_loader(tmp_path, monkeypatch, yaml_text):
    from kylin_memory.config import load_config
    from kylin_memory._vendor.kylin_agent_runtime_cli import config as source
    from kylin_memory._vendor.kylin_agent_runtime_constants import set_hermes_home_override, reset_hermes_home_override
    monkeypatch.setenv('MEMORY_TEST_MODEL', 'expanded-model')
    (tmp_path / 'config.yaml').write_text(yaml_text)
    token = set_hermes_home_override(tmp_path)
    try:
        expected = source._load_config_impl(want_deepcopy=True)
        assert load_config(tmp_path / 'config.yaml') == expected
    finally:
        reset_hermes_home_override(token)


def test_interrupt_uses_source_thread_signal():
    from kylin_memory.interrupt import set_interrupt, is_interrupted
    from kylin_memory._vendor.tools import interrupt as source
    set_interrupt(True)
    try:
        assert source.is_interrupted() and is_interrupted()
    finally:
        set_interrupt(False)
    assert not source.is_interrupted()


def test_memory_system_normalizes_programmatic_legacy_overrides(tmp_path):
    with MemorySystem(tmp_path, config={'max_turns': 23, 'memory': {'atom': {'enabled': False}},
                                       'user_profile': {'enabled': False}}) as memory:
        assert memory.config['agent']['max_turns'] == 23
        assert 'max_turns' not in memory.config


def test_dotenv_reader_preserves_original_parser_semantics(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('MEMORY_TEST_OTHER', 'expanded')
    # Original loader treats this as literal, unlike python-dotenv interpolation.
    (tmp_path / '.env').write_text('OPENAI_API_KEY="${MEMORY_TEST_OTHER}"\n')
    with runtime_context(tmp_path, DEFAULT_CONFIG):
        assert get_env_value('OPENAI_API_KEY') == '${MEMORY_TEST_OTHER}'
