import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from kylinmemory.cli import install_plugin
from kylinmemory.config import runtime_context
from kylinmemory.plugin import LayeredMemoryProvider, attach
from kylinmemory.state import SessionDB
from test_runtime import Model, config


def fake_host(monkeypatch, home, model):
    """Supply only the host boundary; the plugin uses its real packaged core."""
    import yaml
    host_config = config()
    host_config['memory']['atom']['enabled'] = False
    host_config['user_profile']['enabled'] = True
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(yaml.safe_dump(host_config))
    for name in ('agent', 'agent.auxiliary_client', 'kylin_agent_runtime_constants',
                 'kylin_agent_runtime_cli', 'kylin_agent_runtime_cli.config', 'kylin_agent_runtime_cli.profiles'):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    auxiliary = sys.modules['agent.auxiliary_client']
    def route(**kwargs):
        kwargs['model'] = kwargs.get('model') or kwargs['main_runtime']['model']
        allowed = {'model', 'messages', 'tools', 'tool_choice', 'temperature', 'max_tokens', 'timeout', 'extra_body'}
        return model.create(**{k: v for k, v in kwargs.items() if k in allowed})
    auxiliary.call_llm = route
    auxiliary._get_cached_client = lambda *args, **kwargs: (model, args[1] or 'memory-test')
    sys.modules['kylin_agent_runtime_constants'].get_hermes_home = lambda: home
    sys.modules['kylin_agent_runtime_cli.config'].load_config = lambda: host_config
    sys.modules['kylin_agent_runtime_cli.profiles'].get_active_profile_name = lambda: 'default'
    return host_config


def test_installed_directory_plugin_drives_all_four_layers(tmp_path, monkeypatch):
    model = Model()
    fake_host(monkeypatch, tmp_path / 'home', model)
    directory = tmp_path / 'plugin'
    install_plugin(directory)
    spec = importlib.util.spec_from_file_location('installed_memory_plugin', directory / '__init__.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    collected = []
    module.register(SimpleNamespace(register_memory_provider=collected.append))
    provider = collected[0]
    assert provider.name == 'builtin'
    assert provider.is_available()
    provider.initialize('s1', hermes_home=str(tmp_path / 'home'), platform='cli', agent_workspace='hermes', agent_identity='default')
    system = provider.system
    # Real host behavior: persist first, then capture snapshot and emit sync.
    system.session_db.append_message('s1', 'user', '以后请叫我小王。', timestamp=(system.provider._l0_recorder.plugin_start_ms + 1) / 1000)
    messages = system.session_db.get_messages('s1')
    provider.capture_l0_messages(messages, session_id='s1')
    provider.sync_turn('以后请叫我小王。', '', session_id='s1')
    # Match the actual host signature; it does NOT supply model=.
    provider.on_turn_start(1, 'test')
    # Source host owns L3: commit via its MemoryManager projection hooks,
    # and refresh profile at every turn independently of cached L2 navigation.
    from kylinmemory.memory_manager import MemoryManager
    from kylinmemory.user_profile_runtime import (
        initialize_user_profile, commit_user_profile_session, build_user_profile_prompt,
    )
    host_manager = MemoryManager()
    host_manager.add_provider(provider)
    host_agent = SimpleNamespace(session_id='s1', platform='cli', _user_id=None,
        client=model, model='memory-test', api_mode='chat_completions',
        _memory_manager=host_manager, _user_profile_committed_sessions={})
    with runtime_context(system.home, system.config):
        host_agent._user_profile_runtime = initialize_user_profile(host_agent, config()['user_profile'])
        assert commit_user_profile_session(host_agent, messages)['status'] == 'success'
        assert '小王' in build_user_profile_prompt(host_agent)
    assert system.agent._user_profile_runtime is None
    system.consolidate()
    assert '小王' in provider.prefetch('小王')
    assert '<scene-navigation>' in provider.system_prompt_block()
    assert '<user_profile_data>' not in provider.system_prompt_block()
    assert all(call['model'] == 'memory-test' for call in model.calls)
    provider.on_session_switch('s2')
    assert system.session_id == 's2'
    provider.shutdown()
    assert list((system.home / 'user_profile/profiles').glob('*.profile.enc'))


def test_attach_keeps_host_message_store_and_live_l3_client(tmp_path, monkeypatch):
    model = Model()
    fake_host(monkeypatch, tmp_path / 'home', model)
    db = SessionDB(tmp_path / 'home/state.db')
    agent = SimpleNamespace(_memory_manager=None, session_id='s1', client=model, model='first',
        api_mode='chat_completions', platform='cli', _user_id=None, _session_db=db,
        _current_main_runtime=lambda: {'model': agent.model, 'client': agent.client})
    system = attach(agent, config=config())
    assert system.session_db is db
    assert agent._user_profile_runtime is system.agent._user_profile_runtime
    system.observe('以后请叫我小王。')
    agent.model = 'second'
    from kylinmemory.user_profile_runtime import commit_user_profile_session
    result = commit_user_profile_session(agent, db.get_messages('s1'))
    assert result['status'] == 'success'
    assert all(call['model'] == 'second' for call in model.calls)
    assert agent._memory_manager.prefetch_all('小王', session_id='s1')
    agent._memory_manager.shutdown_all()
    assert db.get_messages('s1')  # Host still owns its database connection.
    db.close()
