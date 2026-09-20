"""Hermes memory-provider adapter; all memory algorithms remain standalone."""
from __future__ import annotations

from .config import load_config, runtime_context
from .memory_provider import MemoryProvider


class LayeredMemoryProvider(MemoryProvider):
    # The source host recognizes this name as informational local recall and
    # invokes its explicit boundary hook. Disable memory.atom to avoid two
    # local providers. The host keeps its native L3 lifecycle and prompt refresh.
    name = 'builtin'

    def __init__(self, *, main_runtime=None):
        self.system = None
        self._live_model = ''
        self._runtime_source = main_runtime

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        from pathlib import Path
        from .runtime import MemorySystem
        home = Path(kwargs['hermes_home'])
        config = load_config(home / 'config.yaml')
        plugin_config = (config.get('plugins') or {}).get('layered_memory', {})
        # Host flags disable its old built-in; the plugin has independent flags.
        config['memory']['atom']['enabled'] = plugin_config.get('enabled', True)
        # The host owns L3 commit and its per-turn system-prompt refresh.
        # Expose L1/L2 projection hooks below; never create a second L3 writer.
        config['user_profile']['enabled'] = False

        def live_runtime():
            from .auxiliary_client import resolve_runtime
            supplied = self._runtime_source() if callable(self._runtime_source) else self._runtime_source
            with runtime_context(home, config):
                runtime = resolve_runtime(supplied)
            if self._live_model:
                runtime['model'] = self._live_model
            # Explicit host integration preserves all host auth and transport
            # fallback behavior without importing it in standalone execution.
            from agent.auxiliary_client import call_llm, _get_cached_client
            runtime['call_llm'] = call_llm
            if supplied is None:
                # The source publishes these at the beginning of every turn,
                # before it calls on_turn_start without a model argument.
                from agent import auxiliary_client as host_router
                read_model = getattr(host_router, '_read_main_model', None)
                read_provider = getattr(host_router, '_read_main_provider', None)
                if callable(read_model) and read_model():
                    runtime['model'] = read_model()
                if callable(read_provider) and read_provider():
                    live_provider = read_provider()
                    if live_provider != runtime.get('provider'):
                        for key in ('base_url', 'api_key', 'api_mode'):
                            runtime.pop(key, None)
                    runtime['provider'] = live_provider
            if runtime.get('client') is not None:
                return runtime
            client, model = _get_cached_client('auto', runtime.get('model') or None, main_runtime=runtime)
            if client is not None:
                runtime['client'] = client
                runtime['model'] = model or runtime.get('model') or ''
                # Host auxiliary clients already adapt native transports.
                runtime['api_mode'] = 'chat_completions'
            return runtime

        self.system = MemorySystem(home, config=config, session_id=session_id,
            user_id=kwargs.get('user_id'), platform=kwargs.get('platform', 'cli'),
            team_id=kwargs.get('agent_workspace', 'hermes'),
            agent_id=kwargs.get('agent_identity', 'default'),
            task_id=kwargs.get('task_id', ''), main_runtime=live_runtime)

    def on_turn_start(self, turn_number, message, **kwargs):
        if kwargs.get('model'):
            self._live_model = kwargs['model']
        if self.system:
            with runtime_context(self.system.home, self.system.config):
                self.system.manager.on_turn_start(turn_number, message, **kwargs)

    def capture_l0_messages(self, messages, *, session_id=''):
        if self.system:
            with runtime_context(self.system.home, self.system.config):
                self.system.manager.capture_l0_all(messages, session_id=session_id)

    def sync_turn(self, user_content, assistant_content, *, session_id=''):
        if self.system:
            with runtime_context(self.system.home, self.system.config):
                self.system.manager.sync_all(user_content, assistant_content, session_id=session_id)

    def prefetch(self, query, *, session_id=''):
        if not self.system:
            return ''
        with runtime_context(self.system.home, self.system.config):
            return self.system.manager.prefetch_all(query, session_id=session_id)

    def system_prompt_block(self):
        if not self.system:
            return ''
        with runtime_context(self.system.home, self.system.config):
            return self.system.manager.build_system_prompt()

    def commit_session(self, messages, **kwargs):
        if self.system and self.system.provider:
            with runtime_context(self.system.home, self.system.config):
                return self.system.provider.commit_session(messages, **kwargs)

    def prepare_profile_sources(self):
        if self.system:
            with runtime_context(self.system.home, self.system.config):
                return self.system.manager.prepare_profile_sources()

    def acknowledge_profile_sources(self, batch, **kwargs):
        if self.system:
            with runtime_context(self.system.home, self.system.config):
                return self.system.manager.acknowledge_profile_sources(batch, **kwargs)

    def on_session_end(self, messages):
        if self.system:
            with runtime_context(self.system.home, self.system.config):
                self.system.manager.on_session_end(messages)

    def on_session_switch(self, new_session_id, **kwargs):
        if self.system and new_session_id:
            # Host already committed the old boundary. Forward its notification
            # without manufacturing an extra commit or end event.
            with runtime_context(self.system.home, self.system.config):
                self.system.session_id = self.system.agent.session_id = new_session_id
                self.system.session_db.ensure_session(new_session_id, source=self.system.platform,
                                                     user_id=self.system.user_id)
                self.system.manager.on_session_switch(new_session_id, **kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        if self.system and not self.system._closed:
            ScopedManager(self.system).shutdown_all()


def attach(agent, *, config=None):
    """Replace an initialized host's memory using its live client and routing.

    Call before the first turn, with the host's original memory disabled.
    The host continues its original prompt, session and lifecycle ordering.
    """
    from .runtime import MemorySystem
    if getattr(agent, '_memory_manager', None) is not None:
        raise ValueError('Initialize the host with skip_memory=True before attach()')
    from kylin_agent_runtime_constants import get_hermes_home as host_home
    from kylin_agent_runtime_cli.config import load_config as host_config
    from agent.auxiliary_client import call_llm
    from kylin_agent_runtime_cli.profiles import get_active_profile_name

    def live_runtime():
        runtime = dict(agent._current_main_runtime())
        runtime['client'] = agent.client
        runtime['call_llm'] = call_llm
        return runtime

    from .config import DEFAULT_CONFIG, merge_config
    settings = merge_config(DEFAULT_CONFIG, host_config() if config is None else config)
    system = MemorySystem(host_home(), config=settings, session_id=agent.session_id,
        user_id=getattr(agent, '_user_id', None), platform=getattr(agent, 'platform', 'cli'),
        agent_id=get_active_profile_name(), main_runtime=live_runtime,
        session_db=getattr(agent, '_session_db', None))
    agent._memory_manager = ScopedManager(system)
    agent._user_profile_runtime = system.agent._user_profile_runtime
    agent._user_profile_committed_sessions = {}
    # Preserve the dynamic L3 model/client refresh after host model switches.
    if agent._user_profile_runtime:
        from .user_profile_runtime import _extractor_for_agent
        agent._user_profile_runtime._extractor_factory = lambda: _extractor_for_agent(agent)
    agent._cached_system_prompt = None
    agent._standalone_memory = system
    return system


class ScopedManager:
    """Keep host callbacks and their timer jobs inside this instance's config."""

    def __init__(self, system):
        self.system = system

    def __getattr__(self, name):
        target = getattr(self.system.manager, name)
        if not callable(target):
            return target
        def invoke(*args, **kwargs):
            with runtime_context(self.system.home, self.system.config):
                result = target(*args, **kwargs)
                if name == 'on_session_switch':
                    session_id = args[0] if args else kwargs.get('new_session_id')
                    if session_id:
                        self.system.session_id = self.system.agent.session_id = session_id
                if name == 'shutdown_all':
                    self.system.memory_db.close()
                    if self.system._owns_session_db:
                        self.system.session_db.close()
                    self.system._router.close()
                    self.system._closed = True
                return result
        return invoke
