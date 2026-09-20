"""Host-independent lifecycle for the upstream four-layer memory system."""
from __future__ import annotations

from functools import wraps
from pathlib import Path
import math
import threading
import time
from types import SimpleNamespace
from typing import Callable

from .config import DEFAULT_CONFIG, get_hermes_home, load_config, merge_config, normalize_config, runtime_context
from .memory_manager import MemoryManager, build_memory_context_block
from .l1_memory_provider import create_atom_memory_provider
from .state import MemoryDB, SessionDB
from .user_profile_runtime import initialize_user_profile, commit_user_profile_session, build_user_profile_prompt


def scoped(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._lock, runtime_context(self.home, self.config):
            if self._closed:
                raise RuntimeError('MemorySystem is closed')
            return method(self, *args, **kwargs)
    return run


class MemorySystem:
    """One user's memory lifecycle, with independently isolated storage/config.

    Use a persistent instance per active user/session. ``observe`` commits one
    completed turn, ``context`` supplies recall, ``commit`` matches the original
    session boundary, and ``close`` flushes the original shutdown queues.
    Inject ``main_runtime`` to follow a host's live client/model switches.
    """

    def __init__(self, home: str | Path | None = None, *, config: dict | None = None,
                 session_id: str = 'default', user_id: str | None = None,
                 platform: str = 'cli', team_id: str = 'hermes', agent_id: str = 'default',
                 task_id: str = '', client=None, main_runtime: Callable[[], dict] | dict | None = None,
                 session_db=None):
        self.home = Path(home or get_hermes_home()).expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.config = normalize_config(config or {}, base=load_config(self.home / 'config.yaml'))
        self.session_id = session_id
        self.user_id, self.platform = user_id, platform
        self.team_id, self.agent_id, self.task_id = team_id, agent_id, task_id
        self._runtime_source, self._client = main_runtime, client
        self._lock = threading.RLock()
        self._closed = False
        self._owns_session_db = session_db is None
        self._owned_clients = []
        self._client_cache = {}
        self._client_cache_lock = threading.RLock()
        from .vendor_bridge import RuntimeRouter
        self._router = RuntimeRouter()
        with runtime_context(self.home, self.config):
            self.session_db = session_db if session_db is not None else SessionDB(self.home / 'state.db')
            self.memory_db = MemoryDB(self.home / 'l0_memory.db')
            self.session_db.ensure_session(session_id, source=platform, user_id=user_id)
            self.agent = SimpleNamespace(
                platform=platform, _user_id=user_id, session_id=session_id,
                client=None, model='', api_mode='', _session_db=self.session_db,
                _get_l0_memory_db=lambda: self.memory_db,
                _current_main_runtime=self.current_runtime,
                _user_profile_runtime=None, _user_profile_committed_sessions={},
            )
            self.manager = MemoryManager()
            self.agent._memory_manager = self.manager
            self.provider = create_atom_memory_provider(self.agent, self.config['memory']['atom'])
            if self.provider:
                self.manager.add_provider(self.provider)
                self.manager.initialize_all(session_id, hermes_home=str(self.home), agent_workspace=team_id,
                                            agent_identity=agent_id, task_id=task_id, platform=platform)
            self._refresh_profile_client()
            self.agent._user_profile_runtime = initialize_user_profile(self.agent, self.config['user_profile'])

    def current_runtime(self):
        from .auxiliary_client import resolve_runtime
        source = self._runtime_source() if callable(self._runtime_source) else self._runtime_source
        with runtime_context(self.home, self.config):
            values = resolve_runtime(source)
        if self._client is not None:
            values['client'] = self._client
        values['_client_cache'] = self._client_cache
        values['_client_cache_lock'] = self._client_cache_lock
        values['_router'] = self._router
        return values

    def _refresh_profile_client(self):
        if not self.config['user_profile'].get('enabled', True):
            return
        from .auxiliary_client import _get_cached_client
        values = self.current_runtime()
        model = str(values.get('model') or '')
        mode = values.get('api_mode') or 'chat_completions'
        client = values.get('client')
        if client is None and model:
            client, _ = _get_cached_client('auto', model, main_runtime=values)
            # The L1/L2 adapter already exposes chat.completions on every mode.
            mode = 'chat_completions'
        self.agent.client, self.agent.model, self.agent.api_mode = client, model, mode

    def _timestamp_floor(self):
        previous = self.session_db.get_messages(self.session_id)
        floor = max((int(row['timestamp'] * 1000) for row in previous), default=0)
        if self.provider and self.provider._l0_recorder:
            floor = max(floor, self.provider._l0_recorder.plugin_start_ms)
        return floor

    @staticmethod
    def _next_timestamp(floor):
        # Avoid binary rounding back below the strict millisecond cursor.
        return math.nextafter(max(int(time.time() * 1000), floor + 1) / 1000, math.inf)

    @scoped
    def observe(self, user: str, assistant: str = '', *, timestamp: float | None = None):
        """Commit one completed turn, then capture L0 and notify L1 scheduling."""
        if timestamp is None:
            timestamp = self._next_timestamp(self._timestamp_floor())
        ids = [self.session_db.append_message(self.session_id, 'user', user, timestamp=timestamp)]
        if assistant:
            ids.append(self.session_db.append_message(self.session_id, 'assistant', assistant, timestamp=timestamp))
        snapshot = self.session_db.get_messages(self.session_id)
        self.manager.capture_l0_all(snapshot, session_id=self.session_id)
        self.manager.sync_all(user, assistant, session_id=self.session_id)
        return {'session_id': self.session_id, 'message_ids': ids}

    @scoped
    def ingest(self, messages: list[dict], *, backfill: bool = False):
        """Import a transcript into state.db and trigger the same L0/L1 path.

        Backfill is explicit and must be used on a fresh session, allowing an
        offline benchmark to retain historical timestamps without the live
        recorder's startup cutoff. Repeated calls append; they are not snapshots.
        """
        if backfill and self.session_db.get_messages(self.session_id):
            raise ValueError('Backfill requires a new empty session')
        allowed = {'role', 'content', 'tool_name', 'tool_calls', 'tool_call_id', 'timestamp', 'reasoning', 'reasoning_content'}
        records = []
        floor = self._timestamp_floor()
        for item in messages:
            if item.get('role') not in {'user', 'assistant', 'tool', 'system'}:
                raise ValueError('Every message needs a valid role')
            record = {key: value for key, value in item.items() if key in allowed}
            if record.get('timestamp') is None:
                record['timestamp'] = self._next_timestamp(floor)
            floor = max(floor, int(record['timestamp'] * 1000))
            records.append(record)
        ids = [self.session_db.append_message(self.session_id, **record) for record in records]
        recorder = self.provider._l0_recorder if self.provider else None
        old_start = recorder.plugin_start_ms if recorder else None
        try:
            if backfill and recorder:
                recorder.plugin_start_ms = 0
            self.manager.capture_l0_all(self.session_db.get_messages(self.session_id), session_id=self.session_id)
            self.manager.sync_all('', '', session_id=self.session_id)
        finally:
            if recorder:
                recorder.plugin_start_ms = old_start
        return {'session_id': self.session_id, 'message_ids': ids}

    @scoped
    def context(self, query: str):
        """Return distinct static context and request-only informational recall."""
        static = '\n\n'.join(filter(None, [build_user_profile_prompt(self.agent), self.manager.build_system_prompt()]))
        recall = self.manager.prefetch_all(query, session_id=self.session_id)
        return {'system': static, 'recall': recall,
                'request_context': build_memory_context_block(recall, authority='informational') if recall else ''}

    @scoped
    def recall(self, query: str, *, limit: int = 5):
        if not self.provider:
            return []
        return [atom.as_dict() for atom in self.provider._pipeline.recall(query, team_id=self.team_id,
                agent_id=self.agent_id, user_id=self.provider.user_key, task_id=self.task_id, limit=limit)]

    @scoped
    def scenes(self):
        if not self.provider:
            return []
        return [entry.as_dict() for entry in self.provider._pipeline.scene_store(team_id=self.team_id, agent_id=self.agent_id).index()]

    @scoped
    def read_scene(self, filename: str):
        if not self.provider:
            raise RuntimeError('Local memory is disabled')
        return self.provider._pipeline.scene_store(team_id=self.team_id, agent_id=self.agent_id).read(filename)

    @scoped
    def commit(self):
        """Original boundary: flush L1, project available L1/L2 into L3."""
        self._refresh_profile_client()
        result = commit_user_profile_session(self.agent, self.session_db.get_messages(self.session_id))
        return {'profile': result}

    @scoped
    def consolidate(self):
        """Explicit competition/batch operation; does not alter automatic timing."""
        if not self.provider:
            return {'success': True, 'skipped': True}
        pipeline = self.provider._pipeline
        first = pipeline.flush_session(self.provider._scheduler_key(self.session_id), run_l2=False,
            session_id=self.session_id, team_id=self.team_id, agent_id=self.agent_id,
            user_id=self.provider.user_key, task_id=self.task_id, mode=pipeline.prompt_mode)
        if first.get('success') is False:
            return first
        return pipeline.run_l2(self.provider._scheduler_key(self.session_id), source='manual',
            session_id=self.session_id, team_id=self.team_id, agent_id=self.agent_id, user_id=self.provider.user_key)

    @scoped
    def switch_session(self, session_id: str, *, reset: bool = False):
        if not session_id:
            raise ValueError('session_id must not be empty')
        previous = self.session_id
        # The host commits while the old ID is still active, before emitting
        # on_session_switch. The standalone API must own that boundary too.
        self.commit()
        self.manager.on_session_end(self.session_db.get_messages(previous))
        self.session_db.ensure_session(session_id, source=self.platform, user_id=self.user_id)
        self.session_id = self.agent.session_id = session_id
        self.manager.on_session_switch(session_id, parent_session_id=previous, reset=reset)

    @scoped
    def status(self):
        result = {'home': str(self.home), 'session_id': self.session_id, 'l0': 0, 'l1': 0, 'l2': len(self.scenes()), 'profile_enabled': self.agent._user_profile_runtime is not None}
        if self.provider:
            p = self.provider
            result['l0'] = self.memory_db._conn.execute('SELECT COUNT(*) FROM l0_conversations WHERE team_id=? AND agent_id=? AND user_id=?', (self.team_id, self.agent_id, p.user_key)).fetchone()[0]
            result['l1'] = p._pipeline.atoms._conn.execute('SELECT COUNT(*) FROM l1_records WHERE team_id=? AND agent_id=? AND user_id=?', (self.team_id, self.agent_id, p.user_key)).fetchone()[0]
        return result

    def close(self):
        with self._lock, runtime_context(self.home, self.config):
            if self._closed:
                return
            try:
                self.commit()
                self.manager.on_session_end(self.session_db.get_messages(self.session_id))
            finally:
                self.manager.shutdown_all()
                self.memory_db.close()
                if self._owns_session_db:
                    self.session_db.close()
                for client in self._client_cache.values():
                    close = getattr(client, 'close', None)
                    if callable(close):
                        close()
                self._client_cache.clear()
                self._router.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
