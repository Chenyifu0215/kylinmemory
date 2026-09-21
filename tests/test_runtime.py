import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from kylinmemory import MemorySystem
from kylinmemory.config import get_hermes_home, load_config, runtime_context, context_timer


def response(name, arguments):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    ]))])


class Model:
    """Deterministic model output; real extraction, stores and lifecycle run."""
    base_url = 'https://memory.test/v1'

    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs['tools'][0]['function']['name']
        if name == 'l1_memory_extraction':
            # Inspect the exact structured prompt to retain true L0 ids.
            import re
            ids = [int(x) for x in re.findall(r'^\[(\d+)\] \[', kwargs['messages'][-1]['content'], re.MULTILINE)]
            assert ids, kwargs['messages'][-1]['content']
            return response(name, {'scenes': [{'scene_name': '称呼', 'message_ids': ids,
                'memories': [{'content': '用户希望被称呼为小王。', 'type': 'persona', 'priority': 80,
                              'source_message_ids': [ids[0]], 'metadata': {}}]}]})
        if name == 'l1_conflict_decisions':
            return response(name, {'decisions': []})
        if name == 'l2_scene_transaction':
            return response(name, {'action': 'create', 'target_files': [], 'scene_name': '称呼',
                'summary': '用户称呼', 'body': '## 称呼\n- 用户希望被称呼为小王。', 'delete_files': []})
        if name == 'user_profile_precheck':
            return response(name, {'has_profile_update': True})
        return response(name, {'updates': [{'path': 'basic.preferred_name', 'value': '小王',
            'action': 'upsert', 'confidence': 1.0, 'explicit': True, 'evidence_quote': '用户希望被称呼为小王。'}]})


def config():
    return {'model': {'model': 'memory-test', 'base_url': 'https://memory.test/v1'},
        'memory': {'atom': {'embedding': {'mode': 'disabled'}, 'every_n_conversations': 2,
            'enable_warmup': False, 'l1_idle_timeout_seconds': 0,
            'scenario': {'l2_delay_after_l1_seconds': 9999, 'l2_max_interval_seconds': 0}}},
        'user_profile': {'precheck_enabled': False, 'retry_base_delay_seconds': 0}}


def test_full_pipeline_restart_recall_and_encrypted_profile(tmp_path):
    model = Model()
    with MemorySystem(tmp_path, config=config(), session_id='s1', client=model) as memory:
        memory.observe('请记住，以后叫我小王。', '好的，小王。')
        assert memory.status()['l0'] == 2
        assert memory.status()['l1'] == 0
        result = memory.commit()
        assert result['profile']['status'] == 'success'
        assert memory.status()['l1'] == 1
        assert memory.consolidate()['success']
        assert memory.scenes()
        assert '小王' in memory.read_scene(memory.scenes()[0]['filename'])
        context = memory.context('小王')
        assert '小王' in context['system']
        assert '小王' in context['request_context']
        assert memory.recall('小王')[0]['source_message_ids'] == [1]
        assert memory.session_db.get_messages('s1')[0]['content'] == '请记住，以后叫我小王。'
    encrypted = list((tmp_path / 'user_profile/profiles').glob('*.profile.enc'))
    assert encrypted and '小王'.encode() not in encrypted[0].read_bytes()
    with MemorySystem(tmp_path, config=config(), session_id='s2', client=model) as memory:
        assert memory.recall('小王')
        assert '小王' in memory.context('姓名')['system']
        assert memory.status()['l1'] == 1


def test_live_model_switch_is_shared_by_l1_l2_l3(tmp_path):
    model = Model()
    runtime = {'model': 'first', 'client': model}
    with MemorySystem(tmp_path, config=config(), client=model, main_runtime=lambda: runtime) as memory:
        memory.observe('以后请叫我小王。')
        runtime['model'] = 'second'
        memory.commit()
        memory.consolidate()
        assert model.calls
        assert all(call['model'] == 'second' for call in model.calls)


def test_historical_import_preserves_timestamps_and_has_explicit_backfill(tmp_path):
    model = Model()
    with MemorySystem(tmp_path, config=config(), client=model) as memory:
        memory.ingest([{'role': 'user', 'content': '叫我小王。', 'timestamp': 1700000000}], backfill=True)
        assert memory.session_db.get_messages('default')[0]['timestamp'] == 1700000000
        assert memory.status()['l0'] == 1
        assert memory.commit()['profile']['status'] == 'success'
        with pytest.raises(ValueError, match='empty session'):
            memory.ingest([], backfill=True)


def test_user_isolation_and_context_thread_propagation(tmp_path):
    c = config()
    c['user_profile']['enabled'] = False
    first = MemorySystem(tmp_path / 'first', config=c, user_id='a', platform='api', client=Model())
    second = MemorySystem(tmp_path / 'second', config=c, user_id='b', platform='api', client=Model())
    seen, done = [], threading.Event()
    with runtime_context(first.home, first.config):
        def record():
            seen.append((get_hermes_home(), load_config()['model']['model']))
            done.set()
        timer = context_timer(0.01, record)
        timer.start()
    assert done.wait(2)
    timer.join()
    assert seen == [(first.home, 'memory-test')]
    first.observe('以后请叫我小王。')
    first.commit()
    assert first.recall('小王')
    assert second.recall('小王') == []
    first.close()
    second.close()


def test_sdk_http_transport_and_deepseek_named_tool_contract():
    import httpx
    from openai import OpenAI
    from kylinmemory.auxiliary_client import call_llm, extract_tool_call_arguments
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'id': 'c1', 'object': 'chat.completion', 'created': 1,
            'model': 'deepseek-v4-flash', 'choices': [{'index': 0, 'finish_reason': 'tool_calls',
            'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 't1', 'type': 'function',
            'function': {'name': 'extract', 'arguments': '{"ok": true}'}}]}}]})
    client = OpenAI(api_key='test', base_url='https://api.deepseek.com/v1', http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = call_llm(messages=[{'role': 'user', 'content': 'test'}],
        main_runtime={'client': client, 'model': 'deepseek-v4-flash'},
        tools=[{'type': 'function', 'function': {'name': 'extract', 'parameters': {'type': 'object'}}}],
        tool_choice={'type': 'function', 'function': {'name': 'extract'}})
    assert seen[0]['thinking'] == {'type': 'disabled'}
    assert extract_tool_call_arguments(result, 'extract') == {'ok': True}
    client.close()


def test_json_service_and_plugin_install_outside_repository(tmp_path):
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('memory:\n  atom:\n    enabled: false\nuser_profile:\n  enabled: false\n')
    command = [sys.executable, '-m', 'kylinmemory', '--home', str(tmp_path / 'state'), '--config', str(cfg), 'serve']
    requests = '\n'.join([json.dumps({'id': 1, 'method': 'status'}), '{broken', json.dumps({'id': 2, 'method': 'status'})])
    result = subprocess.run(command, input=requests, capture_output=True, text=True, cwd=tmp_path, check=True)
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert lines[0]['result']['l0'] == 0
    assert lines[1]['error']['type'] == 'JSONDecodeError'
    assert lines[2]['id'] == 2
    from kylinmemory.cli import install_plugin
    installed = tmp_path / 'plugin'
    install_plugin(installed)
    assert 'register_memory_provider' in (installed / '__init__.py').read_text()
    (installed / '__init__.py').write_text('existing plugin')
    with pytest.raises(FileExistsError):
        install_plugin(installed)


def test_pipeline_core_does_not_import_host_modules(tmp_path):
    code = '''
import sys
class BlockHost:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'agent', 'tools', 'plugins', 'run_agent', 'kylin_agent_runtime_cli', 'kylin_agent_runtime_constants', 'kylin_agent_runtime_state'}:
            raise AssertionError('host import: ' + fullname)
sys.meta_path.insert(0, BlockHost())
from kylinmemory import MemorySystem
with MemorySystem(sys.argv[1], config={'memory': {'atom': {'enabled': False}}, 'user_profile': {'enabled': False}}) as memory:
    assert memory.status()['l0'] == 0
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path / 'data')], cwd=tmp_path, check=True, capture_output=True)
