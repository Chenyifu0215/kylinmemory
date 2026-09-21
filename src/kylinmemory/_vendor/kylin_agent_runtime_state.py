import json
import logging
import random
import re
import sqlite3
import threading
import time
from pathlib import Path
from kylinmemory._vendor.agent.memory_manager import sanitize_context
from kylinmemory._vendor.kylin_agent_runtime_constants import get_hermes_home
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
logger = logging.getLogger(__name__)
T = TypeVar('T')
DEFAULT_DB_PATH = get_hermes_home() / 'state.db'
SCHEMA_VERSION = 12
_WAL_INCOMPAT_MARKERS = ('locking protocol', 'not authorized', 'disk i/o error')
_last_init_error_lock = threading.Lock()
_wal_fallback_warned_paths: set[str] = set()
_wal_fallback_warned_lock = threading.Lock()

def _set_last_init_error(msg: Optional[str]) -> None:
    """Record (or clear) the most recent state.db init failure.

    Thread-safe via _last_init_error_lock.  Callers pass a message to
    record a failure or None to clear.  SessionDB.__init__ only calls
    this to SET on failure — it deliberately does NOT clear on success,
    because in a multi-threaded caller (e.g. gateway / web_server per-
    request SessionDB() instantiation), a concurrent successful open
    racing past a different thread's failure would erase the cause
    string that thread's /resume handler is about to format.  Explicit
    clears (e.g. test fixtures) are still supported by passing None.
    """
    global _last_init_error
    with _last_init_error_lock:
        _last_init_error = msg

def apply_wal_with_fallback(conn: sqlite3.Connection, *, db_label: str='state.db') -> str:
    """Set ``journal_mode=WAL`` on ``conn``, falling back to DELETE on failure.

    Returns the journal mode actually set (``"wal"`` or ``"delete"``).

    On WAL-incompatible filesystems (NFS, SMB, some FUSE), SQLite raises
    ``OperationalError("locking protocol")`` when setting WAL.  We fall
    back to DELETE mode — the pre-WAL default, which works on NFS — and
    log one WARNING explaining why.

    The WARNING is deduplicated per ``db_label``: repeated connections
    to the same underlying DB (e.g. kanban_db.connect() which is called
    on every kanban operation) log once per process, not once per call.
    Different db_labels log independently, so state.db and kanban.db
    each get one warning on the same NFS mount.

    Shared by :class:`SessionDB` and ``kylin_agent_runtime_cli.kanban_db.connect`` so
    both databases get identical fallback behavior.
    """
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        return 'wal'
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if not any((marker in msg for marker in _WAL_INCOMPAT_MARKERS)):
            raise
        _log_wal_fallback_once(db_label, exc)
        conn.execute('PRAGMA journal_mode=DELETE')
        return 'delete'

def _log_wal_fallback_once(db_label: str, exc: Exception) -> None:
    """Log a single WARNING per (process, db_label) about WAL fallback.

    Without this dedup, NFS users running kanban (which opens a fresh
    connection on every operation — see kylin_agent_runtime_cli/kanban_db.py) would
    fill errors.log with hundreds of identical warnings per hour.
    """
    with _wal_fallback_warned_lock:
        if db_label in _wal_fallback_warned_paths:
            return
        _wal_fallback_warned_paths.add(db_label)
    logger.warning('%s: WAL journal_mode unsupported on this filesystem (%s) — falling back to journal_mode=DELETE (slower rollback-journal mode; reduces concurrency but works on NFS/SMB/FUSE). See https://www.sqlite.org/wal.html for details. This warning fires once per process per database.', db_label, exc)
SCHEMA_SQL = '\nCREATE TABLE IF NOT EXISTS schema_version (\n    version INTEGER NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS sessions (\n    id TEXT PRIMARY KEY,\n    source TEXT NOT NULL,\n    user_id TEXT,\n    model TEXT,\n    model_config TEXT,\n    system_prompt TEXT,\n    parent_session_id TEXT,\n    started_at REAL NOT NULL,\n    ended_at REAL,\n    end_reason TEXT,\n    message_count INTEGER DEFAULT 0,\n    tool_call_count INTEGER DEFAULT 0,\n    input_tokens INTEGER DEFAULT 0,\n    output_tokens INTEGER DEFAULT 0,\n    cache_read_tokens INTEGER DEFAULT 0,\n    cache_write_tokens INTEGER DEFAULT 0,\n    reasoning_tokens INTEGER DEFAULT 0,\n    billing_provider TEXT,\n    billing_base_url TEXT,\n    billing_mode TEXT,\n    estimated_cost_usd REAL,\n    actual_cost_usd REAL,\n    cost_status TEXT,\n    cost_source TEXT,\n    pricing_version TEXT,\n    title TEXT,\n    api_call_count INTEGER DEFAULT 0,\n    handoff_state TEXT,\n    handoff_platform TEXT,\n    handoff_error TEXT,\n    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)\n);\n\nCREATE TABLE IF NOT EXISTS messages (\n    id INTEGER PRIMARY KEY AUTOINCREMENT,\n    session_id TEXT NOT NULL REFERENCES sessions(id),\n    role TEXT NOT NULL,\n    content TEXT,\n    tool_call_id TEXT,\n    tool_calls TEXT,\n    tool_name TEXT,\n    timestamp REAL NOT NULL,\n    token_count INTEGER,\n    finish_reason TEXT,\n    reasoning TEXT,\n    reasoning_content TEXT,\n    reasoning_details TEXT,\n    codex_reasoning_items TEXT,\n    codex_message_items TEXT,\n    platform_message_id TEXT\n);\n\nCREATE TABLE IF NOT EXISTS state_meta (\n    key TEXT PRIMARY KEY,\n    value TEXT\n);\n\nCREATE TABLE IF NOT EXISTS task_runs (\n    id TEXT PRIMARY KEY,\n    session_id TEXT NOT NULL REFERENCES sessions(id),\n    status TEXT NOT NULL,\n    original_user_message TEXT NOT NULL,\n    model TEXT,\n    priority INTEGER DEFAULT 0,\n    started_at REAL NOT NULL,\n    updated_at REAL NOT NULL,\n    completed_at REAL,\n    interruption_reason TEXT,\n    current_location TEXT,\n    current_tool TEXT,\n    current_tool_call_id TEXT,\n    current_tool_args TEXT,\n    current_recovery_policy TEXT,\n    todo_snapshot TEXT,\n    completed_steps TEXT,\n    execution_context TEXT,\n    recovery_attempts INTEGER DEFAULT 0\n);\n\nCREATE TABLE IF NOT EXISTS task_subruns (\n    id TEXT PRIMARY KEY,\n    task_run_id TEXT NOT NULL REFERENCES task_runs(id),\n    parent_subrun_id TEXT,\n    child_session_id TEXT,\n    goal TEXT NOT NULL,\n    status TEXT NOT NULL,\n    model TEXT,\n    started_at REAL NOT NULL,\n    updated_at REAL NOT NULL,\n    completed_at REAL,\n    interruption_reason TEXT,\n    current_tool TEXT,\n    current_tool_call_id TEXT,\n    current_tool_args TEXT,\n    current_recovery_policy TEXT,\n    result_json TEXT,\n    recovery_attempts INTEGER DEFAULT 0\n);\n\nCREATE TABLE IF NOT EXISTS task_checkpoints (\n    id INTEGER PRIMARY KEY AUTOINCREMENT,\n    task_run_id TEXT NOT NULL REFERENCES task_runs(id),\n    subrun_id TEXT,\n    kind TEXT NOT NULL,\n    location TEXT NOT NULL,\n    status TEXT NOT NULL,\n    payload TEXT,\n    created_at REAL NOT NULL\n);\n\n'
FTS_SQL = "\nCREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(\n    content\n);\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN\n    INSERT INTO messages_fts(rowid, content) VALUES (\n        new.id,\n        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')\n    );\nEND;\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN\n    DELETE FROM messages_fts WHERE rowid = old.id;\nEND;\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN\n    DELETE FROM messages_fts WHERE rowid = old.id;\n    INSERT INTO messages_fts(rowid, content) VALUES (\n        new.id,\n        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')\n    );\nEND;\n"
FTS_TRIGRAM_SQL = "\nCREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(\n    content,\n    tokenize='trigram'\n);\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN\n    INSERT INTO messages_fts_trigram(rowid, content) VALUES (\n        new.id,\n        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')\n    );\nEND;\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN\n    DELETE FROM messages_fts_trigram WHERE rowid = old.id;\nEND;\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN\n    DELETE FROM messages_fts_trigram WHERE rowid = old.id;\n    INSERT INTO messages_fts_trigram(rowid, content) VALUES (\n        new.id,\n        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')\n    );\nEND;\n"

class SessionDB:
    """
    SQLite-backed session storage with FTS5 search.

    Thread-safe for the common gateway pattern (multiple reader threads,
    single writer via WAL mode). Each method opens its own cursor.
    """
    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.02
    _WRITE_RETRY_MAX_S = 0.15
    _CHECKPOINT_EVERY_N_WRITES = 50
    _RECORD_SESSION_INIT_ERROR = True

    def __init__(self, db_path: Path=None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._write_count = 0
        try:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=1.0, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            apply_wal_with_fallback(self._conn, db_label=self.db_path.name)
            self._conn.execute('PRAGMA foreign_keys=ON')
            self._init_schema()
        except Exception as exc:
            if self._RECORD_SESSION_INIT_ERROR:
                _set_last_init_error(f'{type(exc).__name__}: {exc}')
            raise

    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Execute a write transaction with BEGIN IMMEDIATE and jitter retry.

        *fn* receives the connection and should perform INSERT/UPDATE/DELETE
        statements.  The caller must NOT call ``commit()`` — that's handled
        here after *fn* returns.

        BEGIN IMMEDIATE acquires the WAL write lock at transaction start
        (not at commit time), so lock contention surfaces immediately.
        On ``database is locked``, we release the Python lock, sleep a
        random 20-150ms, and retry — breaking the convoy pattern that
        SQLite's built-in deterministic backoff creates.

        Returns whatever *fn* returns.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute('BEGIN IMMEDIATE')
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
                return result
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if 'locked' in err_msg or 'busy' in err_msg:
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        jitter = random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
                        time.sleep(jitter)
                        continue
                raise
        raise last_err or sqlite3.OperationalError('database is locked after max retries')

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint.  Never blocks, never raises.

        Flushes committed WAL frames back into the main DB file for any
        frames that no other connection currently needs.  Keeps the WAL
        from growing unbounded when many processes hold persistent
        connections.
        """
        try:
            with self._lock:
                result = self._conn.execute('PRAGMA wal_checkpoint(PASSIVE)').fetchone()
                if result and result[1] > 0:
                    logger.debug('WAL checkpoint: %d/%d pages checkpointed', result[2], result[1])
        except Exception:
            pass

    def close(self):
        """Close the database connection.

        Attempts a PASSIVE WAL checkpoint first so that exiting processes
        help keep the WAL file from growing unbounded.
        """
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute('PRAGMA wal_checkpoint(PASSIVE)')
                except Exception:
                    pass
                self._conn.close()
                self._conn = None

    @staticmethod
    def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Extract expected columns per table from SCHEMA_SQL.

        Uses an in-memory SQLite database to parse the SQL — SQLite itself
        handles all syntax (DEFAULT expressions with commas, inline
        REFERENCES, CHECK constraints, etc.) so there are zero regex
        edge cases.  The in-memory DB is opened, the schema DDL is
        executed, and PRAGMA table_info extracts the column metadata.

        Adding a column to SCHEMA_SQL is all that's needed; the
        reconciliation loop picks it up automatically.
        """
        ref = sqlite3.connect(':memory:')
        try:
            ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            for tbl, in ref.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
                cols: Dict[str, str] = {}
                for row in ref.execute(f'PRAGMA table_info("{tbl}")').fetchall():
                    col_name = row[1]
                    col_type = row[2] or ''
                    notnull = row[3]
                    default = row[4]
                    pk = row[5]
                    parts = [col_type] if col_type else []
                    if notnull and (not pk):
                        parts.append('NOT NULL')
                    if default is not None:
                        parts.append(f'DEFAULT {default}')
                    cols[col_name] = ' '.join(parts)
                table_columns[tbl] = cols
            return table_columns
        finally:
            ref.close()

    def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:
        """Ensure live tables have every column declared in SCHEMA_SQL.

        Follows the Beets/sqlite-utils pattern: the CREATE TABLE definition
        in SCHEMA_SQL is the single source of truth for the desired schema.
        On every startup this method diffs the live columns (via PRAGMA
        table_info) against the declared columns, and ADDs any that are
        missing.

        This makes column additions a declarative operation — just add
        the column to SCHEMA_SQL and it appears on the next startup.
        Version-gated migration blocks are no longer needed for ADD COLUMN.
        """
        expected = self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_cols in expected.items():
            try:
                rows = cursor.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            except sqlite3.OperationalError:
                continue
            live_cols = set()
            for row in rows:
                name = row[1] if isinstance(row, (tuple, list)) else row['name']
                live_cols.add(name)
            for col_name, col_type in declared_cols.items():
                if col_name not in live_cols:
                    safe_name = col_name.replace('"', '""')
                    try:
                        cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_name}" {col_type}')
                    except sqlite3.OperationalError as exc:
                        logger.debug('reconcile %s.%s: %s', table_name, col_name, exc)

    def _init_schema(self):
        """Create tables and FTS if they don't exist, reconcile columns.

        Schema management follows the declarative reconciliation pattern
        (Beets, sqlite-utils): SCHEMA_SQL is the single source of truth.
        On existing databases, _reconcile_columns() diffs live columns
        against SCHEMA_SQL and ADDs any missing ones.  This eliminates
        the version-gated migration chain for column additions, making
        it impossible for reordered or inserted migrations to skip columns.

        The schema_version table is retained for future data migrations
        (transforming existing rows) which cannot be handled declaratively.
        """
        cursor = self._conn.cursor()
        cursor.executescript(SCHEMA_SQL)
        self._reconcile_columns(cursor)
        post_reconcile_indexes = (('idx_sessions_source', 'CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source)'), ('idx_sessions_parent', 'CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id)'), ('idx_sessions_started', 'CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC)'), ('idx_messages_session', 'CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp)'), ('idx_messages_platform_msg_id', 'CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id ON messages(session_id, platform_message_id) WHERE platform_message_id IS NOT NULL'), ('idx_task_runs_session_status', 'CREATE INDEX IF NOT EXISTS idx_task_runs_session_status ON task_runs(session_id, status, updated_at DESC)'), ('idx_task_subruns_parent_status', 'CREATE INDEX IF NOT EXISTS idx_task_subruns_parent_status ON task_subruns(task_run_id, status, updated_at DESC)'), ('idx_task_checkpoints_run', 'CREATE INDEX IF NOT EXISTS idx_task_checkpoints_run ON task_checkpoints(task_run_id, id DESC)'))
        for index_name, statement in post_reconcile_indexes:
            try:
                cursor.execute(statement)
            except sqlite3.OperationalError as exc:
                logger.debug('%s create skipped: %s', index_name, exc)
        cursor.execute('SELECT version FROM schema_version LIMIT 1')
        row = cursor.fetchone()
        if row is None:
            cursor.execute('INSERT INTO schema_version (version) VALUES (?)', (SCHEMA_VERSION,))
        else:
            current_version = row['version'] if isinstance(row, sqlite3.Row) else row[0]
            if current_version < 10:
                try:
                    cursor.execute('SELECT * FROM messages_fts_trigram LIMIT 0')
                    _fts_trigram_exists = True
                except sqlite3.OperationalError:
                    _fts_trigram_exists = False
                if not _fts_trigram_exists:
                    cursor.executescript(FTS_TRIGRAM_SQL)
                    cursor.execute('INSERT INTO messages_fts_trigram(rowid, content) SELECT id, content FROM messages WHERE content IS NOT NULL')
            if current_version < 11:
                for _trig in ('messages_fts_insert', 'messages_fts_delete', 'messages_fts_update', 'messages_fts_trigram_insert', 'messages_fts_trigram_delete', 'messages_fts_trigram_update'):
                    try:
                        cursor.execute(f'DROP TRIGGER IF EXISTS {_trig}')
                    except sqlite3.OperationalError:
                        pass
                for _tbl in ('messages_fts', 'messages_fts_trigram'):
                    try:
                        cursor.execute(f'DROP TABLE IF EXISTS {_tbl}')
                    except sqlite3.OperationalError:
                        pass
                cursor.executescript(FTS_SQL)
                cursor.executescript(FTS_TRIGRAM_SQL)
                cursor.execute("INSERT INTO messages_fts(rowid, content) SELECT id, COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') FROM messages")
                cursor.execute("INSERT INTO messages_fts_trigram(rowid, content) SELECT id, COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') FROM messages")
            if current_version < SCHEMA_VERSION:
                cursor.execute('UPDATE schema_version SET version = ?', (SCHEMA_VERSION,))
        try:
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique ON sessions(title) WHERE title IS NOT NULL')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('SELECT * FROM messages_fts LIMIT 0')
        except sqlite3.OperationalError:
            cursor.executescript(FTS_SQL)
        try:
            cursor.execute('SELECT * FROM messages_fts_trigram LIMIT 0')
        except sqlite3.OperationalError:
            cursor.executescript(FTS_TRIGRAM_SQL)
        self._conn.commit()

    def _insert_session_row(self, session_id: str, source: str, model: str=None, model_config: Dict[str, Any]=None, system_prompt: str=None, user_id: str=None, parent_session_id: str=None) -> None:
        """Shared INSERT OR IGNORE for session rows."""

        def _do(conn):
            conn.execute('INSERT OR IGNORE INTO sessions (id, source, user_id, model, model_config,\n                   system_prompt, parent_session_id, started_at)\n                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (session_id, source, user_id, model, json.dumps(model_config) if model_config else None, system_prompt, parent_session_id, time.time()))
        self._execute_write(_do)

    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create a new session record. Returns the session_id."""
        self._insert_session_row(session_id, source, **kwargs)
        return session_id

    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session as ended.

        No-ops when the session is already ended. The first end_reason wins:
        compression-split sessions must keep their ``end_reason = 'compression'``
        record even if a later stale ``end_session()`` call (e.g. from a
        desynced CLI session_id after ``/resume`` or ``/branch``) targets them
        with a different reason. Use ``reopen_session()`` first if you
        intentionally need to re-end a closed session with a new reason.
        """

        def _do(conn):
            conn.execute('UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL', (time.time(), end_reason, session_id))
        self._execute_write(_do)

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed."""

        def _do(conn):
            conn.execute('UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?', (session_id,))
        self._execute_write(_do)

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Store the full assembled system prompt snapshot."""

        def _do(conn):
            conn.execute('UPDATE sessions SET system_prompt = ? WHERE id = ?', (system_prompt, session_id))
        self._execute_write(_do)

    def update_token_counts(self, session_id: str, input_tokens: int=0, output_tokens: int=0, model: str=None, cache_read_tokens: int=0, cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None, actual_cost_usd: Optional[float]=None, cost_status: Optional[str]=None, cost_source: Optional[str]=None, pricing_version: Optional[str]=None, billing_provider: Optional[str]=None, billing_base_url: Optional[str]=None, billing_mode: Optional[str]=None, api_call_count: int=0, absolute: bool=False) -> None:
        """Update token counters and backfill model if not already set.

        When *absolute* is False (default), values are **incremented** — use
        this for per-API-call deltas (CLI path).

        When *absolute* is True, values are **set directly** — use this when
        the caller already holds cumulative totals (gateway path, where the
        cached agent accumulates across messages).
        """
        self._insert_session_row(session_id, 'unknown', model=model)
        if absolute:
            sql = 'UPDATE sessions SET\n                   input_tokens = ?,\n                   output_tokens = ?,\n                   cache_read_tokens = ?,\n                   cache_write_tokens = ?,\n                   reasoning_tokens = ?,\n                   estimated_cost_usd = COALESCE(?, 0),\n                   actual_cost_usd = CASE\n                       WHEN ? IS NULL THEN actual_cost_usd\n                       ELSE ?\n                   END,\n                   cost_status = COALESCE(?, cost_status),\n                   cost_source = COALESCE(?, cost_source),\n                   pricing_version = COALESCE(?, pricing_version),\n                   billing_provider = COALESCE(billing_provider, ?),\n                   billing_base_url = COALESCE(billing_base_url, ?),\n                   billing_mode = COALESCE(billing_mode, ?),\n                   model = COALESCE(model, ?),\n                   api_call_count = ?\n                   WHERE id = ?'
        else:
            sql = 'UPDATE sessions SET\n                   input_tokens = input_tokens + ?,\n                   output_tokens = output_tokens + ?,\n                   cache_read_tokens = cache_read_tokens + ?,\n                   cache_write_tokens = cache_write_tokens + ?,\n                   reasoning_tokens = reasoning_tokens + ?,\n                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0),\n                   actual_cost_usd = CASE\n                       WHEN ? IS NULL THEN actual_cost_usd\n                       ELSE COALESCE(actual_cost_usd, 0) + ?\n                   END,\n                   cost_status = COALESCE(?, cost_status),\n                   cost_source = COALESCE(?, cost_source),\n                   pricing_version = COALESCE(?, pricing_version),\n                   billing_provider = COALESCE(billing_provider, ?),\n                   billing_base_url = COALESCE(billing_base_url, ?),\n                   billing_mode = COALESCE(billing_mode, ?),\n                   model = COALESCE(model, ?),\n                   api_call_count = COALESCE(api_call_count, 0) + ?\n                   WHERE id = ?'
        params = (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, estimated_cost_usd, actual_cost_usd, actual_cost_usd, cost_status, cost_source, pricing_version, billing_provider, billing_base_url, billing_mode, model, api_call_count, session_id)

        def _do(conn):
            conn.execute(sql, params)
        self._execute_write(_do)

    def ensure_session(self, session_id: str, source: str='unknown', model: str=None, **kwargs) -> str:
        """Ensure a session row exists (INSERT OR IGNORE). Accepts optional kwargs."""
        self._insert_session_row(session_id, source, model=model, **kwargs)
        return session_id

    def prune_empty_ghost_sessions(self, sessions_dir: 'Optional[Path]'=None) -> int:
        """Remove empty TUI ghost sessions (no messages, no title, >24hr old)."""
        cutoff = time.time() - 86400

        def _do(conn):
            rows = conn.execute("\n                SELECT id FROM sessions\n                WHERE source = 'tui'\n                  AND title IS NULL\n                  AND ended_at IS NOT NULL\n                  AND started_at < ?\n                  AND NOT EXISTS (\n                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id\n                  )\n            ", (cutoff,)).fetchall()
            ids = [r[0] if isinstance(r, (tuple, list)) else r['id'] for r in rows]
            if ids:
                placeholders = ','.join('?' * len(ids))
                conn.execute(f'DELETE FROM sessions WHERE id IN ({placeholders})', ids)
            return ids
        removed_ids = self._execute_write(_do) or []
        if sessions_dir and removed_ids:
            for sid in removed_ids:
                self._remove_session_files(sessions_dir, sid)
        return len(removed_ids)

    def finalize_orphaned_compression_sessions(self) -> int:
        """Mark orphaned compression continuation sessions as ended.

        Targets child sessions that were never finalized: parent is ended
        with reason='compression', child has messages but no end_reason/ended_at
        and api_call_count=0.  Non-destructive: preserves all messages and sets
        end_reason='orphaned_compression'.  Fix for #20001.
        """
        cutoff = time.time() - 604800

        def _do(conn):
            now = time.time()
            result = conn.execute("\n                UPDATE sessions\n                SET ended_at = ?,\n                    end_reason = 'orphaned_compression'\n                WHERE api_call_count = 0\n                  AND end_reason IS NULL\n                  AND ended_at IS NULL\n                  AND started_at < ?\n                  AND parent_session_id IS NOT NULL\n                  AND EXISTS (\n                      SELECT 1 FROM sessions p\n                      WHERE p.id = sessions.parent_session_id\n                        AND p.end_reason = 'compression'\n                        AND p.ended_at IS NOT NULL\n                  )\n                  AND EXISTS (\n                      SELECT 1 FROM messages m\n                      WHERE m.session_id = sessions.id\n                  )\n                ", (now, cutoff))
            return result.rowcount
        return self._execute_write(_do) or 0

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        with self._lock:
            cursor = self._conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        """Resolve an exact or uniquely prefixed session ID to the full ID.

        Returns the exact ID when it exists. Otherwise treats the input as a
        prefix and returns the single matching session ID if the prefix is
        unambiguous. Returns None for no matches or ambiguous prefixes.
        """
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return exact['id']
        escaped = session_id_or_prefix.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        with self._lock:
            cursor = self._conn.execute("SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2", (f'{escaped}%',))
            matches = [row['id'] for row in cursor.fetchall()]
        if len(matches) == 1:
            return matches[0]
        return None
    MAX_TITLE_LENGTH = 100

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Validate and sanitize a session title.

        - Strips leading/trailing whitespace
        - Removes ASCII control characters (0x00-0x1F, 0x7F) and problematic
          Unicode control chars (zero-width, RTL/LTR overrides, etc.)
        - Collapses internal whitespace runs to single spaces
        - Normalizes empty/whitespace-only strings to None
        - Enforces MAX_TITLE_LENGTH

        Returns the cleaned title string or None.
        Raises ValueError if the title exceeds MAX_TITLE_LENGTH after cleaning.
        """
        if not title:
            return None
        cleaned = re.sub('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]', '', title)
        cleaned = re.sub('[\\u200b-\\u200f\\u2028-\\u202e\\u2060-\\u2069\\ufeff\\ufffc\\ufff9-\\ufffb]', '', cleaned)
        cleaned = re.sub('\\s+', ' ', cleaned).strip()
        if not cleaned:
            return None
        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(f'Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})')
        return cleaned

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title.

        Returns True if session was found and title was set.
        Raises ValueError if title is already in use by another session,
        or if the title fails validation (too long, invalid characters).
        Empty/whitespace-only strings are normalized to None (clearing the title).
        """
        title = self.sanitize_title(title)

        def _do(conn):
            if title:
                cursor = conn.execute('SELECT id FROM sessions WHERE title = ? AND id != ?', (title, session_id))
                conflict = cursor.fetchone()
                if conflict:
                    raise ValueError(f"Title '{title}' is already in use by session {conflict['id']}")
            cursor = conn.execute('UPDATE sessions SET title = ? WHERE id = ?', (title, session_id))
            return cursor.rowcount
        rowcount = self._execute_write(_do)
        return rowcount > 0

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        with self._lock:
            cursor = self._conn.execute('SELECT title FROM sessions WHERE id = ?', (session_id,))
            row = cursor.fetchone()
        return row['title'] if row else None

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session by exact title. Returns session dict or None."""
        with self._lock:
            cursor = self._conn.execute('SELECT * FROM sessions WHERE title = ?', (title,))
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve a title to a session ID, preferring the latest in a lineage.

        If the exact title exists, returns that session's ID.
        If not, searches for "title #N" variants and returns the latest one.
        If the exact title exists AND numbered variants exist, returns the
        latest numbered variant (the most recent continuation).
        """
        exact = self.get_session_by_title(title)
        escaped = title.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        with self._lock:
            cursor = self._conn.execute("SELECT id, title, started_at FROM sessions WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC", (f'{escaped} #%',))
            numbered = cursor.fetchall()
        if numbered:
            return numbered[0]['id']
        elif exact:
            return exact['id']
        return None

    def get_next_title_in_lineage(self, base_title: str) -> str:
        """Generate the next title in a lineage (e.g., "my session" → "my session #2").

        Strips any existing " #N" suffix to find the base name, then finds
        the highest existing number and increments.
        """
        match = re.match('^(.*?) #(\\d+)$', base_title)
        if match:
            base = match.group(1)
        else:
            base = base_title
        escaped = base.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        with self._lock:
            cursor = self._conn.execute("SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'", (base, f'{escaped} #%'))
            existing = [row['title'] for row in cursor.fetchall()]
        if not existing:
            return base
        max_num = 1
        for t in existing:
            m = re.match('^.* #(\\d+)$', t)
            if m:
                max_num = max(max_num, int(m.group(1)))
        return f'{base} #{max_num + 1}'

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        """Walk the compression-continuation chain forward and return the tip.

        A compression continuation is a child session where:
        1. The parent's ``end_reason = 'compression'``
        2. The child was created AFTER the parent was ended (started_at >= ended_at)

        The second condition distinguishes compression continuations from
        delegate subagents or branch children, which can also have a
        ``parent_session_id`` but were created while the parent was still live.

        Returns the session_id of the latest continuation in the chain, or the
        input ``session_id`` if it isn't part of a compression chain (or if the
        input itself doesn't exist).
        """
        current = session_id
        for _ in range(100):
            with self._lock:
                cursor = self._conn.execute("SELECT id FROM sessions WHERE parent_session_id = ?   AND started_at >= (      SELECT ended_at FROM sessions       WHERE id = ? AND end_reason = 'compression'  ) ORDER BY started_at DESC LIMIT 1", (current, current))
                row = cursor.fetchone()
            if row is None:
                return current
            current = row['id']
        return current

    def list_sessions_rich(self, source: str=None, exclude_sources: List[str]=None, limit: int=20, offset: int=0, include_children: bool=False, project_compression_tips: bool=True, order_by_last_active: bool=False) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active timestamp.

        Returns dicts with keys: id, source, model, title, started_at, ended_at,
        message_count, preview (first 60 chars of first user message),
        last_active (timestamp of last message).

        Uses a single query with correlated subqueries instead of N+2 queries.

        By default, child sessions (subagent runs, compression continuations)
        are excluded.  Pass ``include_children=True`` to include them.

        With ``project_compression_tips=True`` (default), sessions that are
        roots of compression chains are projected forward to their latest
        continuation — one logical conversation = one list entry, showing the
        live continuation's id/message_count/title/last_active. This prevents
        compressed continuations from being invisible to users while keeping
        delegate subagents and branches hidden. Pass ``False`` to return the
        raw root rows (useful for admin/debug UIs).

        Pass ``order_by_last_active=True`` to sort by most-recent activity
        instead of original conversation start time. For compression chains,
        the "most-recent activity" is taken from the live tip (not the root),
        so an old conversation that was compressed and continued recently
        surfaces in the correct slot. Ordering is computed at SQL level via
        a recursive CTE that walks compression-continuation edges, so LIMIT
        and OFFSET still apply efficiently.
        """
        where_clauses = []
        params = []
        if not include_children:
            where_clauses.append("(s.parent_session_id IS NULL OR EXISTS (SELECT 1 FROM sessions p            WHERE p.id = s.parent_session_id            AND p.end_reason = 'branched'            AND s.started_at >= p.ended_at))")
        if source:
            where_clauses.append('s.source = ?')
            params.append(source)
        if exclude_sources:
            placeholders = ','.join(('?' for _ in exclude_sources))
            where_clauses.append(f's.source NOT IN ({placeholders})')
            params.extend(exclude_sources)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ''
        if order_by_last_active:
            query = f"\n                WITH RECURSIVE chain(root_id, cur_id) AS (\n                    SELECT s.id, s.id FROM sessions s {where_sql}\n                    UNION ALL\n                    SELECT c.root_id, child.id\n                    FROM chain c\n                    JOIN sessions parent ON parent.id = c.cur_id\n                    JOIN sessions child ON child.parent_session_id = c.cur_id\n                    WHERE parent.end_reason = 'compression'\n                      AND child.started_at >= parent.ended_at\n                ),\n                chain_max AS (\n                    SELECT\n                        root_id,\n                        MAX(COALESCE(\n                            (SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = cur_id),\n                            (SELECT started_at FROM sessions ss WHERE ss.id = cur_id)\n                        )) AS effective_last_active\n                    FROM chain\n                    GROUP BY root_id\n                )\n                SELECT s.*,\n                    COALESCE(\n                        (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)\n                         FROM messages m\n                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL\n                         ORDER BY m.timestamp, m.id LIMIT 1),\n                        ''\n                    ) AS _preview_raw,\n                    COALESCE(\n                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),\n                        s.started_at\n                    ) AS last_active,\n                    COALESCE(cm.effective_last_active, s.started_at) AS _effective_last_active\n                FROM sessions s\n                LEFT JOIN chain_max cm ON cm.root_id = s.id\n                {where_sql}\n                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC\n                LIMIT ? OFFSET ?\n            "
            params = params + params + [limit, offset]
        else:
            query = f"\n                SELECT s.*,\n                    COALESCE(\n                        (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)\n                         FROM messages m\n                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL\n                         ORDER BY m.timestamp, m.id LIMIT 1),\n                        ''\n                    ) AS _preview_raw,\n                    COALESCE(\n                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),\n                        s.started_at\n                    ) AS last_active\n                FROM sessions s\n                {where_sql}\n                ORDER BY s.started_at DESC\n                LIMIT ? OFFSET ?\n            "
            params.extend([limit, offset])
        with self._lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
        sessions = []
        for row in rows:
            s = dict(row)
            raw = s.pop('_preview_raw', '').strip()
            if raw:
                text = raw[:60]
                s['preview'] = text + ('...' if len(raw) > 60 else '')
            else:
                s['preview'] = ''
            s.pop('_effective_last_active', None)
            sessions.append(s)
        if project_compression_tips and (not include_children):
            projected = []
            for s in sessions:
                if s.get('end_reason') != 'compression':
                    projected.append(s)
                    continue
                tip_id = self.get_compression_tip(s['id'])
                if tip_id == s['id']:
                    projected.append(s)
                    continue
                tip_row = self._get_session_rich_row(tip_id)
                if not tip_row:
                    projected.append(s)
                    continue
                merged = dict(s)
                for key in ('id', 'ended_at', 'end_reason', 'message_count', 'tool_call_count', 'title', 'last_active', 'preview', 'model', 'system_prompt'):
                    if key in tip_row:
                        merged[key] = tip_row[key]
                merged['_lineage_root_id'] = s['id']
                projected.append(merged)
            sessions = projected
        return sessions

    def _get_session_rich_row(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single session with the same enriched columns as
        ``list_sessions_rich`` (preview + last_active). Returns None if the
        session doesn't exist.
        """
        query = "\n            SELECT s.*,\n                COALESCE(\n                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)\n                     FROM messages m\n                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL\n                     ORDER BY m.timestamp, m.id LIMIT 1),\n                    ''\n                ) AS _preview_raw,\n                COALESCE(\n                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),\n                    s.started_at\n                ) AS last_active\n            FROM sessions s\n            WHERE s.id = ?\n        "
        with self._lock:
            cursor = self._conn.execute(query, (session_id,))
            row = cursor.fetchone()
        if not row:
            return None
        s = dict(row)
        raw = s.pop('_preview_raw', '').strip()
        if raw:
            text = raw[:60]
            s['preview'] = text + ('...' if len(raw) > 60 else '')
        else:
            s['preview'] = ''
        return s
    _CONTENT_JSON_PREFIX = '\x00json:'

    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """Serialize structured (list/dict) message content for sqlite.

        sqlite3 can only bind ``str``, ``bytes``, ``int``, ``float``, and ``None``
        to query parameters. Multimodal messages have ``content`` as a list of
        parts (``[{"type": "text", ...}, {"type": "image_url", ...}]``), which
        raises ``ProgrammingError: Error binding parameter N: type 'list' is
        not supported`` when bound directly.

        Returns the value unchanged when it's already a safe scalar, or a
        sentinel-prefixed JSON string for lists/dicts. Paired with
        :meth:`_decode_content` on read.
        """
        if content is None or isinstance(content, (str, bytes, int, float)):
            return content
        try:
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        except (TypeError, ValueError):
            return str(content)

    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        """Reverse :meth:`_encode_content`; returns scalars unchanged."""
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            try:
                return json.loads(content[len(cls._CONTENT_JSON_PREFIX):])
            except (json.JSONDecodeError, TypeError):
                logger.warning('Failed to decode JSON-encoded message content; returning raw string')
                return content
        return content

    def append_message(self, session_id: str, role: str, content: str=None, tool_name: str=None, tool_calls: Any=None, tool_call_id: str=None, token_count: int=None, finish_reason: str=None, reasoning: str=None, reasoning_content: str=None, reasoning_details: Any=None, codex_reasoning_items: Any=None, codex_message_items: Any=None, platform_message_id: str=None, timestamp: float=None) -> int:
        """
        Append a message to a session. Returns the message row ID.

        Also increments the session's message_count (and tool_call_count
        if role is 'tool' or tool_calls is present).

        ``platform_message_id`` is the external messaging platform's own
        message ID (e.g. Telegram update_id, Yuanbao msg_id).  It is
        independent of the SQLite autoincrement primary key and is used by
        platform-specific flows like yuanbao's recall guard to redact a
        message by its platform-side identifier.
        """
        reasoning_details_json = json.dumps(reasoning_details) if reasoning_details else None
        codex_items_json = json.dumps(codex_reasoning_items) if codex_reasoning_items else None
        codex_message_items_json = json.dumps(codex_message_items) if codex_message_items else None
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        stored_content = self._encode_content(content)
        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1
        message_timestamp = timestamp if timestamp is not None and timestamp > 0 else time.time()

        def _do(conn):
            cursor = conn.execute('INSERT INTO messages (session_id, role, content, tool_call_id,\n                   tool_calls, tool_name, timestamp, token_count, finish_reason,\n                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,\n                   codex_message_items, platform_message_id)\n                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (session_id, role, stored_content, tool_call_id, tool_calls_json, tool_name, message_timestamp, token_count, finish_reason, reasoning, reasoning_content, reasoning_details_json, codex_items_json, codex_message_items_json, platform_message_id))
            msg_id = cursor.lastrowid
            if num_tool_calls > 0:
                conn.execute('UPDATE sessions SET message_count = message_count + 1,\n                       tool_call_count = tool_call_count + ? WHERE id = ?', (num_tool_calls, session_id))
            else:
                conn.execute('UPDATE sessions SET message_count = message_count + 1 WHERE id = ?', (session_id,))
            return msg_id
        return self._execute_write(_do)

    def start_task_run(self, run_id: str, session_id: str, original_user_message: str, *, model: Optional[str]=None, priority: int=0, execution_context: Optional[Dict[str, Any]]=None) -> str:
        """Create a durable running task before agent execution starts."""
        now = time.time()

        def _do(conn):
            conn.execute("INSERT INTO task_runs (\n                       id, session_id, status, original_user_message, model,\n                       priority, started_at, updated_at, completed_steps,\n                       execution_context\n                   ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, '[]', ?)", (run_id, session_id, original_user_message, model, max(0, min(int(priority), 2)), now, now, json.dumps(execution_context, ensure_ascii=False, default=str) if execution_context is not None else None))
            return run_id
        return self._execute_write(_do)

    def update_task_execution_context(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Merge resumable runtime metadata without replacing prior fields."""
        if not isinstance(updates, dict) or not updates:
            return False
        now = time.time()

        def _do(conn):
            row = conn.execute('SELECT execution_context FROM task_runs WHERE id = ?', (run_id,)).fetchone()
            if row is None:
                return False
            raw = row['execution_context'] if isinstance(row, sqlite3.Row) else row[0]
            try:
                context = json.loads(raw or '{}')
            except (TypeError, json.JSONDecodeError):
                context = {}
            if not isinstance(context, dict):
                context = {}
            context.update(updates)
            cursor = conn.execute("UPDATE task_runs\n                   SET execution_context = ?, updated_at = ?\n                   WHERE id = ? AND status = 'running'", (json.dumps(context, ensure_ascii=False, default=str), now, run_id))
            return cursor.rowcount == 1
        return self._execute_write(_do)

    def resume_task_run(self, run_id: str) -> bool:
        """Atomically claim an interrupted task for another execution attempt."""
        now = time.time()

        def _do(conn):
            cursor = conn.execute("UPDATE task_runs\n                   SET status = 'running', updated_at = ?,\n                       interruption_reason = NULL,\n                       recovery_attempts = recovery_attempts + 1\n                   WHERE id = ? AND status = 'interrupted'", (now, run_id))
            return cursor.rowcount == 1
        return self._execute_write(_do)

    def update_task_checkpoint(self, run_id: str, *, current_location: Optional[str]=None, current_tool: Optional[str]=None, current_tool_call_id: Optional[str]=None, current_tool_args: Optional[Any]=None, current_recovery_policy: Optional[str]=None, todo_snapshot: Optional[Any]=None, completed_step: Optional[Dict[str, Any]]=None) -> bool:
        """Persist the current tool boundary and an optional completed step."""
        now = time.time()

        def _do(conn):
            row = conn.execute('SELECT completed_steps FROM task_runs WHERE id = ?', (run_id,)).fetchone()
            if row is None:
                return False
            raw_steps = row['completed_steps'] if isinstance(row, sqlite3.Row) else row[0]
            try:
                steps = json.loads(raw_steps or '[]')
            except (TypeError, json.JSONDecodeError):
                steps = []
            if completed_step:
                steps.append(completed_step)
                steps = steps[-200:]
            cursor = conn.execute("UPDATE task_runs\n                   SET updated_at = ?, current_location = ?,\n                       current_tool = ?, current_tool_call_id = ?,\n                       current_tool_args = ?,\n                       current_recovery_policy = ?,\n                       todo_snapshot = COALESCE(?, todo_snapshot),\n                       completed_steps = ?\n                   WHERE id = ? AND status = 'running'", (now, current_location, current_tool, current_tool_call_id, json.dumps(current_tool_args, ensure_ascii=False, default=str) if current_tool_args is not None else None, current_recovery_policy, json.dumps(todo_snapshot, ensure_ascii=False, default=str) if todo_snapshot is not None else None, json.dumps(steps, ensure_ascii=False, default=str), run_id))
            return cursor.rowcount == 1
        return self._execute_write(_do)

    def start_task_subrun(self, subrun_id: str, task_run_id: str, goal: str, *, parent_subrun_id: Optional[str]=None, child_session_id: Optional[str]=None, model: Optional[str]=None) -> str:
        """Create a durable child-agent branch before it begins execution."""
        now = time.time()

        def _do(conn):
            conn.execute("INSERT INTO task_subruns (\n                       id, task_run_id, parent_subrun_id, child_session_id,\n                       goal, status, model, started_at, updated_at\n                   ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)", (subrun_id, task_run_id, parent_subrun_id, child_session_id, goal, model, now, now))
            return subrun_id
        return self._execute_write(_do)

    def resume_task_subrun(self, subrun_id: str) -> bool:
        now = time.time()

        def _do(conn):
            cursor = conn.execute("UPDATE task_subruns\n                   SET status = 'running', updated_at = ?,\n                       interruption_reason = NULL,\n                       recovery_attempts = recovery_attempts + 1\n                   WHERE id = ? AND status = 'interrupted'", (now, subrun_id))
            return cursor.rowcount == 1
        return self._execute_write(_do)

    def update_task_subrun_checkpoint(self, subrun_id: str, *, current_tool: Optional[str], current_tool_call_id: Optional[str], current_tool_args: Optional[Any]=None, current_recovery_policy: Optional[str]=None) -> bool:
        now = time.time()

        def _do(conn):
            cursor = conn.execute("UPDATE task_subruns\n                   SET updated_at = ?, current_tool = ?,\n                       current_tool_call_id = ?, current_tool_args = ?,\n                       current_recovery_policy = ?\n                   WHERE id = ? AND status = 'running'", (now, current_tool, current_tool_call_id, json.dumps(current_tool_args, ensure_ascii=False, default=str) if current_tool_args is not None else None, current_recovery_policy, subrun_id))
            return cursor.rowcount == 1
        return self._execute_write(_do)

    def finish_task_subrun(self, subrun_id: str, status: str, *, result: Optional[Any]=None, reason: Optional[str]=None) -> bool:
        if status not in {'completed', 'failed', 'interrupted'}:
            raise ValueError(f'Invalid task subrun status: {status}')
        now = time.time()
        completed_at = now if status in {'completed', 'failed'} else None

        def _do(conn):
            cursor = conn.execute("UPDATE task_subruns\n                   SET status = ?, updated_at = ?, completed_at = ?,\n                       interruption_reason = ?,\n                       result_json = COALESCE(?, result_json)\n                   WHERE id = ? AND status = 'running'", (status, now, completed_at, reason, json.dumps(result, ensure_ascii=False, default=str) if result is not None else None, subrun_id))
            return cursor.rowcount == 1
        return self._execute_write(_do)

    @staticmethod
    def _decode_task_subrun(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        for field, fallback in (('current_tool_args', None), ('result_json', None)):
            try:
                result[field] = json.loads(result.get(field)) if result.get(field) else fallback
            except (TypeError, json.JSONDecodeError):
                pass
        return result

    def list_task_subruns(self, task_run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute('SELECT * FROM task_subruns\n                   WHERE task_run_id = ? ORDER BY started_at, id', (task_run_id,)).fetchall()
        return [self._decode_task_subrun(row) for row in rows]

    def get_latest_task_subrun(self, task_run_id: str, goal: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute('SELECT * FROM task_subruns\n                   WHERE task_run_id = ? AND goal = ?\n                   ORDER BY updated_at DESC LIMIT 1', (task_run_id, goal)).fetchone()
        return self._decode_task_subrun(row) if row is not None else None

    def add_task_checkpoint(self, task_run_id: str, *, kind: str, location: str, status: str='completed', subrun_id: Optional[str]=None, payload: Optional[Any]=None) -> int:
        """Append an immutable semantic recovery point.

        Checkpoints are intentionally append-only: after abrupt power loss the
        highest committed row is the exact boundary from which work may resume.
        """
        now = time.time()

        def _do(conn):
            cursor = conn.execute('INSERT INTO task_checkpoints (\n                       task_run_id, subrun_id, kind, location,\n                       status, payload, created_at\n                   ) VALUES (?, ?, ?, ?, ?, ?, ?)', (task_run_id, subrun_id, kind, location, status, json.dumps(payload, ensure_ascii=False, default=str) if payload is not None else None, now))
            return int(cursor.lastrowid)
        return self._execute_write(_do)

    def list_task_checkpoints(self, task_run_id: str, *, limit: int=100) -> List[Dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._conn.execute('SELECT * FROM task_checkpoints\n                   WHERE task_run_id = ? ORDER BY id DESC LIMIT ?', (task_run_id, bounded_limit)).fetchall()
        result = []
        for row in reversed(rows):
            item = dict(row)
            try:
                item['payload'] = json.loads(item['payload']) if item.get('payload') else None
            except (TypeError, json.JSONDecodeError):
                pass
            result.append(item)
        return result

    def finish_task_run(self, run_id: str, status: str, *, reason: Optional[str]=None) -> bool:
        """Move a task to completed, failed, interrupted, or cancelled.

        ``cancelled`` is also allowed to replace an ``interrupted`` state so
        an explicit superseding request can win a race with SSE disconnect
        handling and suppress an obsolete recovery offer.
        """
        if status not in {'completed', 'failed', 'interrupted', 'cancelled'}:
            raise ValueError(f'Invalid task run status: {status}')
        now = time.time()
        completed_at = now if status in {'completed', 'failed', 'cancelled'} else None

        def _do(conn):
            cursor = conn.execute("UPDATE task_runs\n                   SET status = ?, updated_at = ?, completed_at = ?,\n                       interruption_reason = ?,\n                       current_location = CASE\n                           WHEN ? = 'completed' THEN 'completed'\n                           ELSE current_location\n                       END\n                   WHERE id = ? AND (\n                       status = 'running' OR\n                       (? = 'cancelled' AND status = 'interrupted')\n                   )", (status, now, completed_at, reason, status, run_id, status))
            if cursor.rowcount == 1 and status in {'interrupted', 'failed', 'cancelled'}:
                conn.execute("UPDATE task_subruns\n                       SET status = ?, updated_at = ?,\n                           completed_at = CASE\n                               WHEN ? IN ('failed', 'cancelled') THEN ?\n                               ELSE completed_at\n                           END,\n                           interruption_reason = COALESCE(?, interruption_reason)\n                       WHERE task_run_id = ? AND status = 'running'", (status, now, status, now, reason, run_id))
            return cursor.rowcount == 1
        return self._execute_write(_do)

    def mark_running_tasks_interrupted(self, reason: str='runtime_restarted') -> int:
        """Recover leases left running by a terminated runtime process."""
        now = time.time()

        def _do(conn):
            cursor = conn.execute("UPDATE task_runs\n                   SET status = 'interrupted', updated_at = ?,\n                       interruption_reason = ?\n                   WHERE status = 'running'", (now, reason))
            root_count = cursor.rowcount
            conn.execute("UPDATE task_subruns\n                   SET status = 'interrupted', updated_at = ?,\n                       interruption_reason = ?\n                   WHERE status = 'running'", (now, reason))
            return root_count
        return self._execute_write(_do)

    def get_task_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute('SELECT * FROM task_runs WHERE id = ?', (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        try:
            result['completed_steps'] = json.loads(result.get('completed_steps') or '[]')
        except (TypeError, json.JSONDecodeError):
            result['completed_steps'] = []
        try:
            result['todo_snapshot'] = json.loads(result.get('todo_snapshot') or '[]')
        except (TypeError, json.JSONDecodeError):
            result['todo_snapshot'] = []
        try:
            result['execution_context'] = json.loads(result.get('execution_context') or '{}')
        except (TypeError, json.JSONDecodeError):
            result['execution_context'] = {}
        return result

    def get_recoverable_task(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM task_runs\n                   WHERE session_id = ?\n                     AND status = 'interrupted'\n                     AND started_at > COALESCE((\n                         SELECT MAX(started_at) FROM task_runs\n                         WHERE session_id = ?\n                           AND status = 'cancelled'\n                     ), 0)\n                   ORDER BY updated_at DESC LIMIT 1", (session_id, session_id)).fetchone()
        if row is None:
            return None
        result = dict(row)
        try:
            result['completed_steps'] = json.loads(result.get('completed_steps') or '[]')
        except (TypeError, json.JSONDecodeError):
            result['completed_steps'] = []
        try:
            result['todo_snapshot'] = json.loads(result.get('todo_snapshot') or '[]')
        except (TypeError, json.JSONDecodeError):
            result['todo_snapshot'] = []
        try:
            result['execution_context'] = json.loads(result.get('execution_context') or '{}')
        except (TypeError, json.JSONDecodeError):
            result['execution_context'] = {}
        return result

    def get_running_task(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the newest active task lease for explicit interruption."""
        with self._lock:
            row = self._conn.execute("SELECT id FROM task_runs\n                   WHERE session_id = ? AND status = 'running'\n                   ORDER BY updated_at DESC LIMIT 1", (session_id,)).fetchone()
        if row is None:
            return None
        return self.get_task_run(str(row['id']))

    def replace_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Atomically replace every message for a session.

        Used by transcript-rewrite flows such as /retry, /undo, and /compress.
        The delete + reinsert sequence must commit as one transaction so a
        mid-rewrite failure does not leave SQLite with a partial transcript.
        """

        def _do(conn):
            conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            conn.execute('UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?', (session_id,))
            now_ts = time.time()
            total_messages = 0
            total_tool_calls = 0
            for msg in messages:
                role = msg.get('role', 'unknown')
                tool_calls = msg.get('tool_calls')
                reasoning_details = msg.get('reasoning_details') if role == 'assistant' else None
                codex_reasoning_items = msg.get('codex_reasoning_items') if role == 'assistant' else None
                codex_message_items = msg.get('codex_message_items') if role == 'assistant' else None
                reasoning_details_json = json.dumps(reasoning_details) if reasoning_details else None
                codex_items_json = json.dumps(codex_reasoning_items) if codex_reasoning_items else None
                codex_message_items_json = json.dumps(codex_message_items) if codex_message_items else None
                tool_calls_json = json.dumps(tool_calls) if tool_calls else None
                platform_msg_id = msg.get('platform_message_id') or msg.get('message_id')
                conn.execute('INSERT INTO messages (session_id, role, content, tool_call_id,\n                       tool_calls, tool_name, timestamp, token_count, finish_reason,\n                       reasoning, reasoning_content, reasoning_details, codex_reasoning_items,\n                       codex_message_items, platform_message_id)\n                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (session_id, role, self._encode_content(msg.get('content')), msg.get('tool_call_id'), tool_calls_json, msg.get('tool_name'), now_ts, msg.get('token_count'), msg.get('finish_reason'), msg.get('reasoning') if role == 'assistant' else None, msg.get('reasoning_content') if role == 'assistant' else None, reasoning_details_json, codex_items_json, codex_message_items_json, platform_msg_id))
                total_messages += 1
                if tool_calls is not None:
                    total_tool_calls += len(tool_calls) if isinstance(tool_calls, list) else 1
                now_ts += 1e-06
            conn.execute('UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?', (total_messages, total_tool_calls, session_id))
        self._execute_write(_do)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all messages for a session, ordered by insertion order."""
        with self._lock:
            cursor = self._conn.execute('SELECT * FROM messages WHERE session_id = ? ORDER BY id', (session_id,))
            rows = cursor.fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            if 'content' in msg:
                msg['content'] = self._decode_content(msg['content'])
            if msg.get('tool_calls'):
                try:
                    msg['tool_calls'] = json.loads(msg['tool_calls'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning('Failed to deserialize tool_calls in get_messages, falling back to []')
                    msg['tool_calls'] = []
            result.append(msg)
        return result

    def get_messages_around(self, session_id: str, around_message_id: int, window: int=5) -> Dict[str, Any]:
        """Load a window of messages anchored on a specific message id.

        Returns a dict with:
          - ``window``: up to ``window`` messages before the anchor, the anchor
            itself, and up to ``window`` messages after, ordered by id ascending.
          - ``messages_before``: count of messages strictly before the anchor
            still in the session (== window unless we hit the start).
          - ``messages_after``: count of messages strictly after the anchor
            still in the session (== window unless we hit the end).

        Used by ``memory`` for both the discovery shape (anchored on the
        FTS5 match) and the scroll shape (anchored on any message id). The
        ``messages_before`` / ``messages_after`` counts let the caller detect
        session boundaries: when either is less than ``window``, the agent has
        reached one end of the session.

        Returns an empty window when ``around_message_id`` is not a real id in
        ``session_id`` — callers decide how to surface that.
        """
        if window < 0:
            window = 0
        with self._lock:
            anchor_exists = self._conn.execute('SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1', (around_message_id, session_id)).fetchone()
            if not anchor_exists:
                return {'window': [], 'messages_before': 0, 'messages_after': 0}
            before_rows = self._conn.execute('SELECT * FROM messages WHERE session_id = ? AND id <= ? ORDER BY id DESC LIMIT ?', (session_id, around_message_id, window + 1)).fetchall()
            after_rows = self._conn.execute('SELECT * FROM messages WHERE session_id = ? AND id > ? ORDER BY id ASC LIMIT ?', (session_id, around_message_id, window)).fetchall()
        rows = list(reversed(before_rows)) + list(after_rows)
        result = []
        for row in rows:
            msg = dict(row)
            if 'content' in msg:
                msg['content'] = self._decode_content(msg['content'])
            if msg.get('tool_calls'):
                try:
                    msg['tool_calls'] = json.loads(msg['tool_calls'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning('Failed to deserialize tool_calls in get_messages_around, falling back to []')
                    msg['tool_calls'] = []
            result.append(msg)
        messages_before = max(0, len(before_rows) - 1)
        messages_after = len(after_rows)
        return {'window': result, 'messages_before': messages_before, 'messages_after': messages_after}

    def get_anchored_view(self, session_id: str, around_message_id: int, window: int=5, bookend: int=3, keep_roles: Optional[Tuple[str, ...]]=('user', 'assistant')) -> Dict[str, Any]:
        """Return an anchored window plus session bookends.

        Built on top of ``get_messages_around``. Three slices:

          - ``window``: messages immediately surrounding the anchor. Filtered
            to ``keep_roles`` (tool-response noise dropped by default), EXCEPT
            the anchor itself is always preserved regardless of role.
          - ``bookend_start``: first ``bookend`` user/assistant messages of the
            session — but only those whose id is strictly before the window's
            first message id. Empty when the window already overlaps the
            session head. Empty-content messages (tool-call-only assistant
            turns) are skipped so they don't crowd out actual prose openings.
          - ``bookend_end``: last ``bookend`` user/assistant messages of the
            session, same non-overlap rule at the tail.

        Bookends let an FTS5 hit anywhere in a long session yield the goal
        (opening) and the resolution (closing) on a single call — without
        loading the whole transcript.

        Returns ``{"window": [], "messages_before": 0, "messages_after": 0,
        "bookend_start": [], "bookend_end": []}`` when the anchor isn't in
        the session.

        ``keep_roles=None`` disables role filtering (raw window + raw
        bookends).
        """
        if bookend < 0:
            bookend = 0
        primitive = self.get_messages_around(session_id, around_message_id, window=window)
        window_rows = primitive['window']
        if not window_rows:
            return {'window': [], 'messages_before': 0, 'messages_after': 0, 'bookend_start': [], 'bookend_end': []}
        if keep_roles is not None:
            keep_set = set(keep_roles)
            filtered_window = [m for m in window_rows if m.get('id') == around_message_id or m.get('role') in keep_set]
        else:
            filtered_window = window_rows
        window_min_id = window_rows[0]['id']
        window_max_id = window_rows[-1]['id']
        bookend_start_rows: List[Any] = []
        bookend_end_rows: List[Any] = []
        if bookend > 0:
            with self._lock:
                role_clause = ''
                role_params: list = []
                if keep_roles is not None:
                    role_placeholders = ','.join(('?' for _ in keep_roles))
                    role_clause = f' AND role IN ({role_placeholders})'
                    role_params = list(keep_roles)
                bookend_start_rows = self._conn.execute(f'SELECT * FROM messages WHERE session_id = ? AND id < ?{role_clause} AND length(content) > 0 ORDER BY id ASC LIMIT ?', (session_id, window_min_id, *role_params, bookend)).fetchall()
                bookend_end_rows = self._conn.execute(f'SELECT * FROM messages WHERE session_id = ? AND id > ?{role_clause} AND length(content) > 0 ORDER BY id DESC LIMIT ?', (session_id, window_max_id, *role_params, bookend)).fetchall()
                bookend_end_rows = list(reversed(bookend_end_rows))

        def _hydrate(row) -> Dict[str, Any]:
            msg = dict(row)
            if 'content' in msg:
                msg['content'] = self._decode_content(msg['content'])
            if msg.get('tool_calls'):
                try:
                    msg['tool_calls'] = json.loads(msg['tool_calls'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning('Failed to deserialize tool_calls in get_anchored_view, falling back to []')
                    msg['tool_calls'] = []
            return msg
        return {'window': filtered_window, 'messages_before': primitive['messages_before'], 'messages_after': primitive['messages_after'], 'bookend_start': [_hydrate(r) for r in bookend_start_rows], 'bookend_end': [_hydrate(r) for r in bookend_end_rows]}

    def resolve_resume_session_id(self, session_id: str) -> str:
        """Redirect a resume target to the descendant session that holds the messages.

        Context compression ends the current session and forks a new child session
        (linked via ``parent_session_id``). The flush cursor is reset, so the
        child is where new messages actually land — the parent ends up with
        ``message_count = 0`` rows unless messages had already been flushed to
        it before compression. See #15000.

        This helper walks ``parent_session_id`` forward from ``session_id`` and
        returns the first descendant in the chain that has at least one message
        row. If the original session already has messages, or no descendant
        has any, the original ``session_id`` is returned unchanged.

        The chain is always walked via the child whose ``started_at`` is
        latest; that matches the single-chain shape that compression creates.
        A depth cap (32) guards against accidental loops in malformed data.
        """
        if not session_id:
            return session_id
        with self._lock:
            try:
                row = self._conn.execute('SELECT 1 FROM messages WHERE session_id = ? LIMIT 1', (session_id,)).fetchone()
            except Exception:
                return session_id
            if row is not None:
                return session_id
            current = session_id
            seen = {current}
            for _ in range(32):
                try:
                    child_row = self._conn.execute('SELECT id FROM sessions WHERE parent_session_id = ? ORDER BY started_at DESC, id DESC LIMIT 1', (current,)).fetchone()
                except Exception:
                    return session_id
                if child_row is None:
                    return session_id
                child_id = child_row['id'] if hasattr(child_row, 'keys') else child_row[0]
                if not child_id or child_id in seen:
                    return session_id
                seen.add(child_id)
                try:
                    msg_row = self._conn.execute('SELECT 1 FROM messages WHERE session_id = ? LIMIT 1', (child_id,)).fetchone()
                except Exception:
                    return session_id
                if msg_row is not None:
                    return child_id
                current = child_id
        return session_id

    def get_messages_as_conversation(self, session_id: str, include_ancestors: bool=False) -> List[Dict[str, Any]]:
        """
        Load messages in the OpenAI conversation format (role + content dicts).
        Used by the gateway to restore conversation history.
        """
        session_ids = [session_id]
        if include_ancestors:
            session_ids = self._session_lineage_root_to_tip(session_id)
        with self._lock:
            placeholders = ','.join(('?' for _ in session_ids))
            rows = self._conn.execute(f'SELECT role, content, tool_call_id, tool_calls, tool_name, finish_reason, reasoning, reasoning_content, reasoning_details, codex_reasoning_items, codex_message_items, platform_message_id FROM messages WHERE session_id IN ({placeholders}) ORDER BY id', tuple(session_ids)).fetchall()
        messages = []
        for row in rows:
            content = self._decode_content(row['content'])
            if row['role'] in {'user', 'assistant'} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg = {'role': row['role'], 'content': content}
            if row['tool_call_id']:
                msg['tool_call_id'] = row['tool_call_id']
            if row['tool_name']:
                msg['tool_name'] = row['tool_name']
            if row['tool_calls']:
                try:
                    msg['tool_calls'] = json.loads(row['tool_calls'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning('Failed to deserialize tool_calls in conversation replay, falling back to []')
                    msg['tool_calls'] = []
            if row['platform_message_id']:
                msg['message_id'] = row['platform_message_id']
            if row['role'] == 'assistant':
                if row['finish_reason']:
                    msg['finish_reason'] = row['finish_reason']
                if row['reasoning']:
                    msg['reasoning'] = row['reasoning']
                if row['reasoning_content'] is not None:
                    msg['reasoning_content'] = row['reasoning_content']
                if row['reasoning_details']:
                    try:
                        msg['reasoning_details'] = json.loads(row['reasoning_details'])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning('Failed to deserialize reasoning_details, falling back to None')
                        msg['reasoning_details'] = None
                if row['codex_reasoning_items']:
                    try:
                        msg['codex_reasoning_items'] = json.loads(row['codex_reasoning_items'])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning('Failed to deserialize codex_reasoning_items, falling back to None')
                        msg['codex_reasoning_items'] = None
                if row['codex_message_items']:
                    try:
                        msg['codex_message_items'] = json.loads(row['codex_message_items'])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning('Failed to deserialize codex_message_items, falling back to None')
                        msg['codex_message_items'] = None
            if include_ancestors and self._is_duplicate_replayed_user_message(messages, msg):
                continue
            messages.append(msg)
        return messages

    def _session_lineage_root_to_tip(self, session_id: str) -> List[str]:
        if not session_id:
            return [session_id]
        chain = []
        current = session_id
        seen = set()
        with self._lock:
            for _ in range(100):
                if not current or current in seen:
                    break
                seen.add(current)
                chain.append(current)
                row = self._conn.execute('SELECT parent_session_id FROM sessions WHERE id = ?', (current,)).fetchone()
                if row is None:
                    break
                current = row['parent_session_id'] if hasattr(row, 'keys') else row[0]
        return list(reversed(chain)) or [session_id]

    @staticmethod
    def _is_duplicate_replayed_user_message(messages: List[Dict[str, Any]], msg: Dict[str, Any]) -> bool:
        if msg.get('role') != 'user':
            return False
        content = msg.get('content')
        if not isinstance(content, str) or not content:
            return False
        for prev in reversed(messages):
            if prev.get('role') == 'user' and prev.get('content') == content:
                return True
            if prev.get('role') == 'assistant' and (prev.get('content') or prev.get('tool_calls')):
                return False
        return False

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for safe use in FTS5 MATCH queries.

        FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
        ``+``, ``*``, ``{``, ``}`` and bare boolean operators (``AND``, ``OR``,
        ``NOT``) have special meaning.  Passing raw user input directly to
        MATCH can cause ``sqlite3.OperationalError``.

        Strategy:
        - Preserve properly paired quoted phrases (``"exact phrase"``)
        - Strip unmatched FTS5-special characters that would cause errors
        - Wrap unquoted hyphenated and dotted terms in quotes so FTS5
          matches them as exact phrases instead of splitting on the
          hyphen/dot (e.g. ``chat-send``, ``P2.2``, ``my-app.config.ts``)
        """
        _quoted_parts: list = []

        def _preserve_quoted(m: re.Match) -> str:
            _quoted_parts.append(m.group(0))
            return f'\x00Q{len(_quoted_parts) - 1}\x00'
        sanitized = re.sub('"[^"]*"', _preserve_quoted, query)
        sanitized = re.sub('[+{}()\\"^]', ' ', sanitized)
        sanitized = re.sub('\\*+', '*', sanitized)
        sanitized = re.sub('(^|\\s)\\*', '\\1', sanitized)
        sanitized = re.sub('(?i)^(AND|OR|NOT)\\b\\s*', '', sanitized.strip())
        sanitized = re.sub('(?i)\\s+(AND|OR|NOT)\\s*$', '', sanitized.strip())
        sanitized = re.sub('\\b(\\w+(?:[._-]\\w+)+)\\b', '"\\1"', sanitized)
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f'\x00Q{i}\x00', quoted)
        return sanitized.strip()

    @staticmethod
    def _is_cjk_codepoint(cp: int) -> bool:
        return 19968 <= cp <= 40959 or 13312 <= cp <= 19903 or 131072 <= cp <= 173791 or (12288 <= cp <= 12351) or (12352 <= cp <= 12447) or (12448 <= cp <= 12543) or (44032 <= cp <= 55215)

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """Check if text contains CJK (Chinese, Japanese, Korean) characters."""
        for ch in text:
            cp = ord(ch)
            if 19968 <= cp <= 40959 or 13312 <= cp <= 19903 or 131072 <= cp <= 173791 or (12288 <= cp <= 12351) or (12352 <= cp <= 12447) or (12448 <= cp <= 12543) or (44032 <= cp <= 55215):
                return True
        return False

    @classmethod
    def _count_cjk(cls, text: str) -> int:
        """Count CJK characters in text."""
        return sum((1 for ch in text if cls._is_cjk_codepoint(ord(ch))))

    def search_messages(self, query: str, source_filter: List[str]=None, exclude_sources: List[str]=None, role_filter: List[str]=None, limit: int=20, offset: int=0, sort: str=None) -> List[Dict[str, Any]]:
        """
        Full-text search across session messages using FTS5.

        Supports FTS5 query syntax:
          - Simple keywords: "docker deployment"
          - Phrases: '"exact phrase"'
          - Boolean: "docker OR kubernetes", "python NOT java"
          - Prefix: "deploy*"

        Returns matching messages with session metadata, content snippet,
        and surrounding context (1 message before and after the match).

        ``sort`` controls temporal ordering:
          - ``None`` (default): FTS5 BM25 relevance only. Time-neutral.
          - ``"newest"``: order by message timestamp DESC, then by rank.
          - ``"oldest"``: order by message timestamp ASC, then by rank.

        The short-CJK LIKE fallback already orders by timestamp DESC and
        ignores ``sort``. The trigram CJK path honours ``sort`` like the main
        FTS5 path.
        """
        if not query or not query.strip():
            return []
        query = self._sanitize_fts5_query(query)
        if not query:
            return []
        if isinstance(sort, str):
            sort_norm = sort.strip().lower()
            if sort_norm not in ('newest', 'oldest'):
                sort_norm = None
        else:
            sort_norm = None
        if sort_norm == 'newest':
            order_by_sql = 'ORDER BY m.timestamp DESC, rank'
        elif sort_norm == 'oldest':
            order_by_sql = 'ORDER BY m.timestamp ASC, rank'
        else:
            order_by_sql = 'ORDER BY rank'
        where_clauses = ['messages_fts MATCH ?']
        params: list = [query]
        if source_filter is not None:
            source_placeholders = ','.join(('?' for _ in source_filter))
            where_clauses.append(f's.source IN ({source_placeholders})')
            params.extend(source_filter)
        if exclude_sources is not None:
            exclude_placeholders = ','.join(('?' for _ in exclude_sources))
            where_clauses.append(f's.source NOT IN ({exclude_placeholders})')
            params.extend(exclude_sources)
        if role_filter:
            role_placeholders = ','.join(('?' for _ in role_filter))
            where_clauses.append(f'm.role IN ({role_placeholders})')
            params.extend(role_filter)
        where_sql = ' AND '.join(where_clauses)
        params.extend([limit, offset])
        sql = f"\n            SELECT\n                m.id,\n                m.session_id,\n                m.role,\n                snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,\n                m.content,\n                m.timestamp,\n                m.tool_name,\n                s.source,\n                s.model,\n                s.started_at AS session_started\n            FROM messages_fts\n            JOIN messages m ON m.id = messages_fts.rowid\n            JOIN sessions s ON s.id = m.session_id\n            WHERE {where_sql}\n            {order_by_sql}\n            LIMIT ? OFFSET ?\n        "
        is_cjk = self._contains_cjk(query)
        if is_cjk:
            raw_query = query.strip('"').strip()
            cjk_count = self._count_cjk(raw_query)
            _tokens_for_check = [t for t in raw_query.split() if t.upper() not in {'AND', 'OR', 'NOT'} and self._contains_cjk(t)]
            _any_short_cjk = any((self._count_cjk(t) < 3 for t in _tokens_for_check))
            if cjk_count >= 3 and (not _any_short_cjk):
                tokens = raw_query.split()
                parts = []
                for tok in tokens:
                    if tok.upper() in {'AND', 'OR', 'NOT'}:
                        parts.append(tok)
                    else:
                        parts.append('"' + tok.replace('"', '""') + '"')
                trigram_query = ' '.join(parts)
                tri_where = ['messages_fts_trigram MATCH ?']
                tri_params: list = [trigram_query]
                if source_filter is not None:
                    tri_where.append(f"s.source IN ({','.join(('?' for _ in source_filter))})")
                    tri_params.extend(source_filter)
                if exclude_sources is not None:
                    tri_where.append(f"s.source NOT IN ({','.join(('?' for _ in exclude_sources))})")
                    tri_params.extend(exclude_sources)
                if role_filter:
                    tri_where.append(f"m.role IN ({','.join(('?' for _ in role_filter))})")
                    tri_params.extend(role_filter)
                tri_sql = f"\n                    SELECT\n                        m.id,\n                        m.session_id,\n                        m.role,\n                        snippet(messages_fts_trigram, 0, '>>>', '<<<', '...', 40) AS snippet,\n                        m.content,\n                        m.timestamp,\n                        m.tool_name,\n                        s.source,\n                        s.model,\n                        s.started_at AS session_started\n                    FROM messages_fts_trigram\n                    JOIN messages m ON m.id = messages_fts_trigram.rowid\n                    JOIN sessions s ON s.id = m.session_id\n                    WHERE {' AND '.join(tri_where)}\n                    {order_by_sql}\n                    LIMIT ? OFFSET ?\n                "
                tri_params.extend([limit, offset])
                with self._lock:
                    try:
                        tri_cursor = self._conn.execute(tri_sql, tri_params)
                    except sqlite3.OperationalError:
                        matches = []
                    else:
                        matches = [dict(row) for row in tri_cursor.fetchall()]
            else:
                non_op_tokens = [t for t in raw_query.split() if t.upper() not in {'AND', 'OR', 'NOT'}] or [raw_query]
                token_clauses = []
                like_params: list = []
                for tok in non_op_tokens:
                    esc = tok.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                    token_clauses.append("(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' OR m.tool_calls LIKE ? ESCAPE '\\')")
                    like_params += [f'%{esc}%', f'%{esc}%', f'%{esc}%']
                like_where = [f"({' OR '.join(token_clauses)})"]
                if source_filter is not None:
                    like_where.append(f"s.source IN ({','.join(('?' for _ in source_filter))})")
                    like_params.extend(source_filter)
                if exclude_sources is not None:
                    like_where.append(f"s.source NOT IN ({','.join(('?' for _ in exclude_sources))})")
                    like_params.extend(exclude_sources)
                if role_filter:
                    like_where.append(f"m.role IN ({','.join(('?' for _ in role_filter))})")
                    like_params.extend(role_filter)
                like_sql = f"\n                    SELECT m.id, m.session_id, m.role,\n                           substr(m.content,\n                                  max(1, instr(m.content, ?) - 40),\n                                  120) AS snippet,\n                           m.content, m.timestamp, m.tool_name,\n                           s.source, s.model, s.started_at AS session_started\n                    FROM messages m\n                    JOIN sessions s ON s.id = m.session_id\n                    WHERE {' AND '.join(like_where)}\n                    ORDER BY m.timestamp DESC\n                    LIMIT ? OFFSET ?\n                "
                like_params.extend([limit, offset])
                like_params = [non_op_tokens[0]] + like_params
                with self._lock:
                    like_cursor = self._conn.execute(like_sql, like_params)
                    matches = [dict(row) for row in like_cursor.fetchall()]
        else:
            with self._lock:
                try:
                    cursor = self._conn.execute(sql, params)
                except sqlite3.OperationalError:
                    return []
                else:
                    matches = [dict(row) for row in cursor.fetchall()]
        for match in matches:
            try:
                with self._lock:
                    ctx_cursor = self._conn.execute('WITH target AS (\n                               SELECT session_id, timestamp, id\n                               FROM messages\n                               WHERE id = ?\n                           )\n                           SELECT role, content\n                           FROM (\n                               SELECT m.id, m.timestamp, m.role, m.content\n                               FROM messages m\n                               JOIN target t ON t.session_id = m.session_id\n                               WHERE (m.timestamp < t.timestamp)\n                                  OR (m.timestamp = t.timestamp AND m.id < t.id)\n                               ORDER BY m.timestamp DESC, m.id DESC\n                               LIMIT 1\n                           )\n                           UNION ALL\n                           SELECT role, content\n                           FROM messages\n                           WHERE id = ?\n                           UNION ALL\n                           SELECT role, content\n                           FROM (\n                               SELECT m.id, m.timestamp, m.role, m.content\n                               FROM messages m\n                               JOIN target t ON t.session_id = m.session_id\n                               WHERE (m.timestamp > t.timestamp)\n                                  OR (m.timestamp = t.timestamp AND m.id > t.id)\n                               ORDER BY m.timestamp ASC, m.id ASC\n                               LIMIT 1\n                           )', (match['id'], match['id']))
                    context_msgs = []
                    for r in ctx_cursor.fetchall():
                        raw = r['content']
                        decoded = self._decode_content(raw)
                        if isinstance(decoded, list):
                            text_parts = [p.get('text', '') for p in decoded if isinstance(p, dict) and p.get('type') == 'text']
                            text = ' '.join((t for t in text_parts if t)).strip()
                            preview = text or '[multimodal content]'
                        elif isinstance(decoded, str):
                            preview = decoded
                        else:
                            preview = ''
                        context_msgs.append({'role': r['role'], 'content': preview[:200]})
                match['context'] = context_msgs
            except Exception:
                match['context'] = []
        for match in matches:
            match.pop('content', None)
        return matches

    def search_sessions(self, source: str=None, limit: int=20, offset: int=0) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source.

        Returns rows enriched with a computed ``last_active`` column (latest
        message timestamp for the session, falling back to ``started_at``),
        ordered by most-recently-used first.
        """
        select_with_last_active = 'SELECT s.*, COALESCE(m.last_active, s.started_at) AS last_active FROM sessions s LEFT JOIN (SELECT session_id, MAX(timestamp) AS last_active FROM messages GROUP BY session_id) m ON m.session_id = s.id '
        with self._lock:
            if source:
                cursor = self._conn.execute(f'{select_with_last_active}WHERE s.source = ? ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?', (source, limit, offset))
            else:
                cursor = self._conn.execute(f'{select_with_last_active}ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?', (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def session_count(self, source: str=None) -> int:
        """Count sessions, optionally filtered by source."""
        with self._lock:
            if source:
                cursor = self._conn.execute('SELECT COUNT(*) FROM sessions WHERE source = ?', (source,))
            else:
                cursor = self._conn.execute('SELECT COUNT(*) FROM sessions')
            return cursor.fetchone()[0]

    def message_count(self, session_id: str=None) -> int:
        """Count messages, optionally for a specific session."""
        with self._lock:
            if session_id:
                cursor = self._conn.execute('SELECT COUNT(*) FROM messages WHERE session_id = ?', (session_id,))
            else:
                cursor = self._conn.execute('SELECT COUNT(*) FROM messages')
            return cursor.fetchone()[0]

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages as a dict."""
        session = self.get_session(session_id)
        if not session:
            return None
        messages = self.get_messages(session_id)
        return {**session, 'messages': messages}

    def export_all(self, source: str=None) -> List[Dict[str, Any]]:
        """
        Export all sessions (with messages) as a list of dicts.
        Suitable for writing to a JSONL file for backup/analysis.
        """
        sessions = self.search_sessions(source=source, limit=100000)
        results = []
        for session in sessions:
            messages = self.get_messages(session['id'])
            results.append({**session, 'messages': messages})
        return results

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""

        def _do(conn):
            conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            conn.execute('UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?', (session_id,))
        self._execute_write(_do)

    @staticmethod
    def _remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
        """Remove on-disk transcript files for a session.

        Cleans up ``{session_id}.json``, ``{session_id}.jsonl``,
        ``{session_id}-details.log``, ``{session_id}-usage.log``, and any
        ``request_dump_{session_id}_*.json`` files left by the gateway.
        Silently skips files that don't exist and swallows OSError so a
        filesystem hiccup never blocks a DB operation.
        """
        if sessions_dir is None:
            return
        for suffix in ('.json', '.jsonl', '-details.log', '-detials.log', '-usage.log'):
            p = sessions_dir / f'{session_id}{suffix}'
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            for p in sessions_dir.glob(f'{session_id}-*'):
                try:
                    if p.is_file():
                        p.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass
        try:
            for p in sessions_dir.glob(f'request_dump_{session_id}_*.json'):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass

    def delete_session(self, session_id: str, sessions_dir: Optional[Path]=None) -> bool:
        """Delete a session and all its messages.

        Child sessions are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted, so they remain accessible independently.
        When *sessions_dir* is provided, also removes on-disk transcript
        files (``.json`` / ``.jsonl`` / ``request_dump_*``) for the deleted
        session. Returns True if the session was found and deleted.
        """

        def _do(conn):
            cursor = conn.execute('SELECT COUNT(*) FROM sessions WHERE id = ?', (session_id,))
            if cursor.fetchone()[0] == 0:
                return False
            conn.execute('UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ?', (session_id,))
            conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            return True
        deleted = self._execute_write(_do)
        if deleted:
            self._remove_session_files(sessions_dir, session_id)
        return deleted

    def prune_sessions(self, older_than_days: int=90, source: str=None, sessions_dir: Optional[Path]=None) -> int:
        """Delete sessions older than N days. Returns count of deleted sessions.

        Only prunes ended sessions (not active ones).  Child sessions outside
        the prune window are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted.  When *sessions_dir* is provided, also removes
        on-disk transcript files (``.json`` / ``.jsonl`` /
        ``request_dump_*``) for every pruned session, outside the DB
        transaction.
        """
        cutoff = time.time() - older_than_days * 86400
        removed_ids: list[str] = []

        def _do(conn):
            if source:
                cursor = conn.execute('SELECT id FROM sessions\n                       WHERE started_at < ? AND ended_at IS NOT NULL AND source = ?', (cutoff, source))
            else:
                cursor = conn.execute('SELECT id FROM sessions WHERE started_at < ? AND ended_at IS NOT NULL', (cutoff,))
            session_ids = {row['id'] for row in cursor.fetchall()}
            if not session_ids:
                return 0
            placeholders = ','.join('?' * len(session_ids))
            conn.execute(f'UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id IN ({placeholders})', list(session_ids))
            for sid in session_ids:
                conn.execute('DELETE FROM messages WHERE session_id = ?', (sid,))
                conn.execute('DELETE FROM sessions WHERE id = ?', (sid,))
                removed_ids.append(sid)
            return len(session_ids)
        count = self._execute_write(_do)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    def get_meta(self, key: str) -> Optional[str]:
        """Read a value from the state_meta key/value store."""
        with self._lock:
            row = self._conn.execute('SELECT value FROM state_meta WHERE key = ?', (key,)).fetchone()
        if row is None:
            return None
        return row['value'] if isinstance(row, sqlite3.Row) else row[0]

    def set_meta(self, key: str, value: str) -> None:
        """Write a value to the state_meta key/value store."""

        def _do(conn):
            conn.execute('INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value', (key, value))
        self._execute_write(_do)

    def apply_telegram_topic_migration(self) -> None:
        """Create Telegram DM topic-mode tables on explicit /topic opt-in.

        This migration is deliberately not part of automatic SessionDB startup
        reconciliation. Operators must be able to upgrade Hermes, keep the old
        Telegram bot behavior running, and only mutate topic-mode state when the
        user executes /topic to opt into the feature.

        Schema versions:
          v1 — initial shape (no ON DELETE CASCADE on session_id FK)
          v2 — session_id FK gets ON DELETE CASCADE so session pruning
               automatically clears bindings.
        """

        def _do(conn):
            conn.executescript("\n                CREATE TABLE IF NOT EXISTS telegram_dm_topic_mode (\n                    chat_id TEXT PRIMARY KEY,\n                    user_id TEXT NOT NULL,\n                    enabled INTEGER NOT NULL DEFAULT 1,\n                    activated_at REAL NOT NULL,\n                    updated_at REAL NOT NULL,\n                    has_topics_enabled INTEGER,\n                    allows_users_to_create_topics INTEGER,\n                    capability_checked_at REAL,\n                    intro_message_id TEXT,\n                    pinned_message_id TEXT\n                );\n\n                CREATE TABLE IF NOT EXISTS telegram_dm_topic_bindings (\n                    chat_id TEXT NOT NULL,\n                    thread_id TEXT NOT NULL,\n                    user_id TEXT NOT NULL,\n                    session_key TEXT NOT NULL,\n                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,\n                    managed_mode TEXT NOT NULL DEFAULT 'auto',\n                    linked_at REAL NOT NULL,\n                    updated_at REAL NOT NULL,\n                    PRIMARY KEY (chat_id, thread_id)\n                );\n\n                CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session\n                ON telegram_dm_topic_bindings(session_id);\n\n                CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user\n                ON telegram_dm_topic_bindings(user_id, chat_id);\n                ")
            current = conn.execute('SELECT value FROM state_meta WHERE key = ?', ('telegram_dm_topic_schema_version',)).fetchone()
            current_version = int(current[0]) if current and str(current[0]).isdigit() else 0
            if current_version < 2:
                fk_rows = conn.execute("PRAGMA foreign_key_list('telegram_dm_topic_bindings')").fetchall()
                needs_rebuild = any((row[2] == 'sessions' and (row[6] or '') != 'CASCADE' for row in fk_rows))
                if needs_rebuild:
                    conn.executescript("\n                        CREATE TABLE telegram_dm_topic_bindings_new (\n                            chat_id TEXT NOT NULL,\n                            thread_id TEXT NOT NULL,\n                            user_id TEXT NOT NULL,\n                            session_key TEXT NOT NULL,\n                            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,\n                            managed_mode TEXT NOT NULL DEFAULT 'auto',\n                            linked_at REAL NOT NULL,\n                            updated_at REAL NOT NULL,\n                            PRIMARY KEY (chat_id, thread_id)\n                        );\n                        INSERT INTO telegram_dm_topic_bindings_new\n                            SELECT chat_id, thread_id, user_id, session_key,\n                                   session_id, managed_mode, linked_at, updated_at\n                            FROM telegram_dm_topic_bindings;\n                        DROP TABLE telegram_dm_topic_bindings;\n                        ALTER TABLE telegram_dm_topic_bindings_new\n                            RENAME TO telegram_dm_topic_bindings;\n                        CREATE UNIQUE INDEX idx_telegram_dm_topic_bindings_session\n                            ON telegram_dm_topic_bindings(session_id);\n                        CREATE INDEX idx_telegram_dm_topic_bindings_user\n                            ON telegram_dm_topic_bindings(user_id, chat_id);\n                        ")
            conn.execute('INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value', ('telegram_dm_topic_schema_version', '2'))
        self._execute_write(_do)

    def enable_telegram_topic_mode(self, *, chat_id: str, user_id: str, has_topics_enabled: Optional[bool]=None, allows_users_to_create_topics: Optional[bool]=None) -> None:
        """Enable Telegram DM topic mode for one private chat/user.

        This method intentionally owns the explicit topic migration. Ordinary
        SessionDB startup must not create these side tables.
        """
        self.apply_telegram_topic_migration()
        now = time.time()

        def _to_int(value: Optional[bool]) -> Optional[int]:
            if value is None:
                return None
            return 1 if value else 0

        def _do(conn):
            conn.execute('\n                INSERT INTO telegram_dm_topic_mode (\n                    chat_id, user_id, enabled, activated_at, updated_at,\n                    has_topics_enabled, allows_users_to_create_topics,\n                    capability_checked_at\n                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?)\n                ON CONFLICT(chat_id) DO UPDATE SET\n                    user_id = excluded.user_id,\n                    enabled = 1,\n                    updated_at = excluded.updated_at,\n                    has_topics_enabled = excluded.has_topics_enabled,\n                    allows_users_to_create_topics = excluded.allows_users_to_create_topics,\n                    capability_checked_at = excluded.capability_checked_at\n                ', (str(chat_id), str(user_id), now, now, _to_int(has_topics_enabled), _to_int(allows_users_to_create_topics), now))
        self._execute_write(_do)

    def disable_telegram_topic_mode(self, *, chat_id: str, clear_bindings: bool=True) -> None:
        """Disable Telegram DM topic mode for one private chat.

        When ``clear_bindings`` is True (default) the (chat_id, thread_id)
        bindings for this chat are also cleared so re-enabling later
        starts from a clean slate. Set to False if the operator wants to
        preserve bindings for a later re-enable.

        Never creates the topic-mode tables from scratch; if they don't
        exist there is nothing to disable and the call is a no-op.
        """

        def _do(conn):
            try:
                conn.execute('UPDATE telegram_dm_topic_mode SET enabled = 0, updated_at = ? WHERE chat_id = ?', (time.time(), str(chat_id)))
                if clear_bindings:
                    conn.execute('DELETE FROM telegram_dm_topic_bindings WHERE chat_id = ?', (str(chat_id),))
            except sqlite3.OperationalError:
                return
        self._execute_write(_do)

    def is_telegram_topic_mode_enabled(self, *, chat_id: str, user_id: str) -> bool:
        """Return whether Telegram DM topic mode is enabled for this chat/user."""
        with self._lock:
            try:
                row = self._conn.execute('\n                    SELECT enabled FROM telegram_dm_topic_mode\n                    WHERE chat_id = ? AND user_id = ?\n                    ', (str(chat_id), str(user_id))).fetchone()
            except sqlite3.OperationalError:
                return False
        if row is None:
            return False
        enabled = row['enabled'] if isinstance(row, sqlite3.Row) else row[0]
        return bool(enabled)

    def get_telegram_topic_binding(self, *, chat_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
        """Return the session binding for a Telegram DM topic, if present."""
        with self._lock:
            try:
                row = self._conn.execute('\n                    SELECT * FROM telegram_dm_topic_bindings\n                    WHERE chat_id = ? AND thread_id = ?\n                    ', (str(chat_id), str(thread_id))).fetchone()
            except sqlite3.OperationalError:
                return None
        return dict(row) if row else None

    def list_telegram_topic_bindings_for_chat(self, *, chat_id: str) -> List[Dict[str, Any]]:
        """All Telegram DM topic bindings for one chat, newest first.

        Read-only; returns [] if the bindings table doesn't exist yet
        (does not trigger the topic-mode migration).
        """
        with self._lock:
            try:
                rows = self._conn.execute('SELECT * FROM telegram_dm_topic_bindings WHERE chat_id = ? ORDER BY updated_at DESC', (str(chat_id),)).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def get_telegram_topic_binding_by_session(self, *, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the Telegram DM topic binding for a given session_id, if present.

        Uses the UNIQUE INDEX on telegram_dm_topic_bindings(session_id) for an
        efficient reverse lookup. Returns None when the session has no binding or
        the table does not exist yet.
        """
        with self._lock:
            try:
                row = self._conn.execute('\n                    SELECT * FROM telegram_dm_topic_bindings\n                    WHERE session_id = ?\n                    ', (str(session_id),)).fetchone()
            except sqlite3.OperationalError:
                return None
        return dict(row) if row else None

    def bind_telegram_topic(self, *, chat_id: str, thread_id: str, user_id: str, session_key: str, session_id: str, managed_mode: str='auto') -> None:
        """Bind one Telegram DM topic thread to one Hermes session.

        A Hermes session may only be linked to one Telegram topic in MVP.
        Rebinding the same topic to the same session is idempotent; trying to
        link the same session to a different topic raises ValueError.
        """
        self.apply_telegram_topic_migration()
        now = time.time()
        chat_id = str(chat_id)
        thread_id = str(thread_id)
        user_id = str(user_id)
        session_key = str(session_key)
        session_id = str(session_id)

        def _do(conn):
            existing_session = conn.execute('\n                SELECT chat_id, thread_id FROM telegram_dm_topic_bindings\n                WHERE session_id = ?\n                ', (session_id,)).fetchone()
            if existing_session is not None:
                linked_chat = existing_session['chat_id'] if isinstance(existing_session, sqlite3.Row) else existing_session[0]
                linked_thread = existing_session['thread_id'] if isinstance(existing_session, sqlite3.Row) else existing_session[1]
                if str(linked_chat) != chat_id or str(linked_thread) != thread_id:
                    raise ValueError('session is already linked to another Telegram topic')
            conn.execute('\n                INSERT INTO telegram_dm_topic_bindings (\n                    chat_id, thread_id, user_id, session_key, session_id,\n                    managed_mode, linked_at, updated_at\n                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)\n                ON CONFLICT(chat_id, thread_id) DO UPDATE SET\n                    user_id = excluded.user_id,\n                    session_key = excluded.session_key,\n                    session_id = excluded.session_id,\n                    managed_mode = excluded.managed_mode,\n                    updated_at = excluded.updated_at\n                ', (chat_id, thread_id, user_id, session_key, session_id, managed_mode, now, now))
        self._execute_write(_do)

    def is_telegram_session_linked_to_topic(self, *, session_id: str) -> bool:
        """Return True if a Hermes session is already bound to any Telegram DM topic.

        Read-only: does NOT trigger the telegram-topic migration. If the
        topic-mode tables have not been created yet (i.e. nobody has run
        ``/topic`` in this profile), the session is by definition unbound
        and we return False.
        """
        with self._lock:
            try:
                row = self._conn.execute('\n                    SELECT 1 FROM telegram_dm_topic_bindings\n                    WHERE session_id = ?\n                    LIMIT 1\n                    ', (str(session_id),)).fetchone()
            except sqlite3.OperationalError:
                return False
        return row is not None

    def list_unlinked_telegram_sessions_for_user(self, *, chat_id: str, user_id: str, limit: int=10) -> List[Dict[str, Any]]:
        """List previous Telegram sessions for this user that are not bound to a topic.

        Read-only: does NOT trigger the telegram-topic migration. If the
        topic-mode tables are absent, fall back to a simpler query that
        just returns this user's Telegram sessions — there can't be any
        bindings yet.
        """
        with self._lock:
            try:
                rows = self._conn.execute("\n                    SELECT s.*,\n                        COALESCE(\n                            (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)\n                             FROM messages m\n                             WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL\n                             ORDER BY m.timestamp, m.id LIMIT 1),\n                            ''\n                        ) AS _preview_raw,\n                        COALESCE(\n                            (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),\n                            s.started_at\n                        ) AS last_active\n                    FROM sessions s\n                    WHERE s.source = 'telegram'\n                      AND s.user_id = ?\n                      AND NOT EXISTS (\n                          SELECT 1 FROM telegram_dm_topic_bindings b\n                          WHERE b.session_id = s.id\n                      )\n                    ORDER BY last_active DESC, s.started_at DESC\n                    LIMIT ?\n                    ", (str(user_id), int(limit))).fetchall()
            except sqlite3.OperationalError:
                rows = self._conn.execute("\n                    SELECT s.*,\n                        COALESCE(\n                            (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)\n                             FROM messages m\n                             WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL\n                             ORDER BY m.timestamp, m.id LIMIT 1),\n                            ''\n                        ) AS _preview_raw,\n                        COALESCE(\n                            (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),\n                            s.started_at\n                        ) AS last_active\n                    FROM sessions s\n                    WHERE s.source = 'telegram'\n                      AND s.user_id = ?\n                    ORDER BY last_active DESC, s.started_at DESC\n                    LIMIT ?\n                    ", (str(user_id), int(limit))).fetchall()
        sessions: List[Dict[str, Any]] = []
        for row in rows:
            session = dict(row)
            raw = str(session.pop('_preview_raw', '') or '').strip()
            session['preview'] = raw[:60] + ('...' if len(raw) > 60 else '') if raw else ''
            sessions.append(session)
        return sessions

    def vacuum(self) -> None:
        """Run VACUUM to reclaim disk space after large deletes.

        SQLite does not shrink the database file when rows are deleted —
        freed pages just get reused on the next insert. After a prune that
        removed hundreds of sessions, the file stays bloated unless we
        explicitly VACUUM.

        VACUUM rewrites the entire DB, so it's expensive (seconds per
        100MB) and cannot run inside a transaction. It also acquires an
        exclusive lock, so callers must ensure no other writers are
        active. Safe to call at startup before the gateway/CLI starts
        serving traffic.
        """
        with self._lock:
            try:
                self._conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            except Exception:
                pass
            self._conn.execute('VACUUM')

    def maybe_auto_prune_and_vacuum(self, retention_days: int=90, min_interval_hours: int=24, vacuum: bool=True, sessions_dir: Optional[Path]=None) -> Dict[str, Any]:
        """Idempotent auto-maintenance: prune old sessions + optional VACUUM.

        Records the last run timestamp in state_meta so subsequent calls
        within ``min_interval_hours`` no-op. Designed to be called once at
        startup from long-lived entrypoints (CLI, gateway, cron scheduler).

        When *sessions_dir* is provided, on-disk transcript files
        (``.json`` / ``.jsonl`` / ``request_dump_*``) for pruned sessions
        are removed as part of the same sweep (issue #3015).

        Never raises. On any failure, logs a warning and returns a dict
        with ``"error"`` set.

        Returns a dict with keys:
          - ``"skipped"`` (bool) — true if within min_interval_hours of last run
          - ``"pruned"`` (int)   — number of sessions deleted
          - ``"vacuumed"`` (bool) — true if VACUUM ran
          - ``"error"`` (str, optional) — present only on failure
        """
        result: Dict[str, Any] = {'skipped': False, 'pruned': 0, 'vacuumed': False}
        try:
            last_raw = self.get_meta('last_auto_prune')
            now = time.time()
            if last_raw:
                try:
                    last_ts = float(last_raw)
                    if now - last_ts < min_interval_hours * 3600:
                        result['skipped'] = True
                        return result
                except (TypeError, ValueError):
                    pass
            pruned = self.prune_sessions(older_than_days=retention_days, sessions_dir=sessions_dir)
            result['pruned'] = pruned
            if vacuum and pruned > 0:
                try:
                    self.vacuum()
                    result['vacuumed'] = True
                except Exception as exc:
                    logger.warning('state.db VACUUM failed: %s', exc)
            self.set_meta('last_auto_prune', str(now))
            if pruned > 0:
                logger.info('state.db auto-maintenance: pruned %d session(s) older than %d days%s', pruned, retention_days, ' + VACUUM' if result['vacuumed'] else '')
        except Exception as exc:
            logger.warning('state.db auto-maintenance failed: %s', exc)
            result['error'] = str(exc)
        return result

    def request_handoff(self, session_id: str, platform: str) -> bool:
        """Mark a session as pending handoff to the given platform.

        Returns True if the row was found and not already in flight; False if
        the session is already in a non-terminal handoff state.
        """

        def _do(conn):
            cur = conn.execute("UPDATE sessions SET handoff_state = 'pending',     handoff_platform = ?,     handoff_error = NULL WHERE id = ? AND (handoff_state IS NULL                   OR handoff_state IN ('completed', 'failed'))", (platform, session_id))
            return cur.rowcount > 0
        return self._execute_write(_do)

    def get_handoff_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Read the current handoff state for a session.

        Returns ``{"state", "platform", "error"}`` or None if the session has
        no handoff record.
        """
        try:
            cur = self._conn.execute('SELECT handoff_state, handoff_platform, handoff_error FROM sessions WHERE id = ?', (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {'state': row['handoff_state'], 'platform': row['handoff_platform'], 'error': row['handoff_error']}
        except Exception:
            return None

    def list_pending_handoffs(self) -> List[Dict[str, Any]]:
        """Return all sessions in handoff_state='pending', oldest first.

        Used by the gateway's handoff watcher.
        """
        try:
            cur = self._conn.execute("SELECT * FROM sessions WHERE handoff_state = 'pending' ORDER BY started_at ASC")
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []

    def claim_handoff(self, session_id: str) -> bool:
        """Atomically transition pending → running. Returns True if claimed."""

        def _do(conn):
            cur = conn.execute("UPDATE sessions SET handoff_state = 'running' WHERE id = ? AND handoff_state = 'pending'", (session_id,))
            return cur.rowcount > 0
        return self._execute_write(_do)

    def complete_handoff(self, session_id: str) -> None:
        """Mark a handoff as completed."""

        def _do(conn):
            conn.execute("UPDATE sessions SET handoff_state = 'completed', handoff_error = NULL WHERE id = ?", (session_id,))
        self._execute_write(_do)

    def fail_handoff(self, session_id: str, error: str) -> None:
        """Mark a handoff as failed and record the reason."""

        def _do(conn):
            conn.execute("UPDATE sessions SET handoff_state = 'failed', handoff_error = ? WHERE id = ?", (error[:500], session_id))
        self._execute_write(_do)

