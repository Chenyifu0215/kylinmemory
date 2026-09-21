import codecs
import logging
import os
import select
import shlex
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import IO, Callable, Protocol
from kylinmemory._vendor.tools.interrupt import is_interrupted
logger = logging.getLogger(__name__)
DEFAULT_PIP_INDEX_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple'
_DEBUG_INTERRUPT = bool(os.getenv('HERMES_DEBUG_INTERRUPT'))
_activity_callback_local = threading.local()

def _get_activity_callback() -> Callable[[str], None] | None:
    return getattr(_activity_callback_local, 'callback', None)

def touch_activity_if_due(state: dict, label: str) -> None:
    """Fire the activity callback at most once every ``state['interval']`` seconds.

    *state* must contain ``last_touch`` (monotonic timestamp) and ``start``
    (monotonic timestamp of the operation start).  An optional ``interval``
    key overrides the default 10 s cadence.

    Swallows all exceptions so callers don't need their own try/except.
    """
    now = time.monotonic()
    interval = state.get('interval', 10.0)
    if now - state['last_touch'] < interval:
        return
    state['last_touch'] = now
    try:
        cb = _get_activity_callback()
        if cb:
            elapsed = int(now - state['start'])
            cb(f'{label} ({elapsed}s elapsed)')
    except Exception:
        pass

class ProcessHandle(Protocol):
    """Duck type that every backend's _run_bash() must return.

    subprocess.Popen satisfies this natively.  SDK backends (Modal, Daytona)
    return _ThreadedProcessHandle which adapts their blocking calls.
    """

    def poll(self) -> int | None:
        ...

    def kill(self) -> None:
        ...

    def wait(self, timeout: float | None=None) -> int:
        ...

    @property
    def stdout(self) -> IO[str] | None:
        ...

    @property
    def returncode(self) -> int | None:
        ...

def _cwd_marker(session_id: str) -> str:
    return f'__HERMES_CWD_{session_id}__'

class BaseEnvironment(ABC):
    """Common interface and unified execution flow for all Hermes backends.

    Subclasses implement ``_run_bash()`` and ``cleanup()``.  The base class
    provides ``execute()`` with session snapshot sourcing, CWD tracking,
    interrupt handling, and timeout enforcement.
    """
    _stdin_mode: str = 'pipe'
    _snapshot_timeout: int = 30

    def get_temp_dir(self) -> str:
        """Return the backend temp directory used for session artifacts.

        Most sandboxed backends use ``/tmp`` inside the target environment.
        LocalEnvironment overrides this on platforms like Termux where ``/tmp``
        may be missing and ``TMPDIR`` is the portable writable location.
        """
        return '/tmp'

    def __init__(self, cwd: str, timeout: int, env: dict=None):
        self.cwd = cwd
        self.timeout = timeout
        self.env = env or {}
        self._session_id = uuid.uuid4().hex[:12]
        temp_dir = self.get_temp_dir().rstrip('/') or '/'
        self._snapshot_path = f'{temp_dir}/hermes-snap-{self._session_id}.sh'
        self._cwd_file = f'{temp_dir}/hermes-cwd-{self._session_id}.txt'
        self._cwd_marker = _cwd_marker(self._session_id)
        self._snapshot_ready = False

    def _run_bash(self, cmd_string: str, *, login: bool=False, timeout: int=120, stdin_data: str | None=None) -> ProcessHandle:
        """Spawn a bash process to run *cmd_string*.

        Returns a ProcessHandle (subprocess.Popen or _ThreadedProcessHandle).
        Must be overridden by every backend.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement _run_bash()')

    @abstractmethod
    def cleanup(self):
        """Release backend resources (container, instance, connection)."""
        ...

    def init_session(self):
        """Capture login shell environment into a snapshot file.

        Called once after backend construction.  On success, sets
        ``_snapshot_ready = True`` so subsequent commands source the snapshot
        instead of running with ``bash -l``.
        """
        _quoted_cwd = shlex.quote(self.cwd)
        _quoted_snap = shlex.quote(self._snapshot_path)
        _quoted_cwd_file = shlex.quote(self._cwd_file)
        bootstrap = f"""export -p > {_quoted_snap}\ndeclare -f | grep -vE '^_[^_]' >> {_quoted_snap}\nalias -p >> {_quoted_snap}\necho 'shopt -s expand_aliases' >> {_quoted_snap}\necho 'set +e' >> {_quoted_snap}\necho 'set +u' >> {_quoted_snap}\nbuiltin cd {_quoted_cwd} 2>/dev/null || true\npwd -P > {_quoted_cwd_file} 2>/dev/null || true\nprintf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' "$(pwd -P)"\n"""
        try:
            proc = self._run_bash(bootstrap, login=True, timeout=self._snapshot_timeout)
            result = self._wait_for_process(proc, timeout=self._snapshot_timeout)
            self._snapshot_ready = True
            self._update_cwd(result)
            logger.info('Session snapshot created (session=%s, cwd=%s)', self._session_id, self.cwd)
        except Exception as exc:
            logger.warning('init_session failed (session=%s): %s — falling back to bash -l per command', self._session_id, exc)
            self._snapshot_ready = False

    @staticmethod
    def _quote_cwd_for_cd(cwd: str) -> str:
        """Quote a ``cd`` target while preserving ``~`` expansion."""
        if cwd == '~':
            return cwd
        if cwd == '~/':
            return '$HOME'
        if cwd.startswith('~/'):
            return f'$HOME/{shlex.quote(cwd[2:])}'
        return shlex.quote(cwd)

    def _wrap_command(self, command: str, cwd: str) -> str:
        """Build the full bash script that sources snapshot, cd's, runs command,
        re-dumps env vars, and emits CWD markers."""
        escaped = command.replace("'", "'\\''")
        _quoted_snap = shlex.quote(self._snapshot_path)
        _quoted_cwd_file = shlex.quote(self._cwd_file)
        parts = []
        if self._snapshot_ready:
            parts.append(f'source {_quoted_snap} >/dev/null 2>&1 || true')
        quoted_cwd = self._quote_cwd_for_cd(cwd)
        parts.append(f'builtin cd -- {quoted_cwd} || exit 126')
        parts.append(f'export PIP_INDEX_URL="${{PIP_INDEX_URL:-{DEFAULT_PIP_INDEX_URL}}}"')
        parts.append(f"eval '{escaped}'")
        parts.append('__hermes_ec=$?')
        if self._snapshot_ready:
            parts.append(f'export -p > {_quoted_snap} 2>/dev/null || true')
        parts.append(f'pwd -P > {_quoted_cwd_file} 2>/dev/null || true')
        parts.append(f'''printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' "$(pwd -P)"''')
        parts.append('exit $__hermes_ec')
        return '\n'.join(parts)

    @staticmethod
    def _embed_stdin_heredoc(command: str, stdin_data: str) -> str:
        """Append stdin_data as a shell heredoc to the command string."""
        delimiter = f'HERMES_STDIN_{uuid.uuid4().hex[:12]}'
        return f"{command} << '{delimiter}'\n{stdin_data}\n{delimiter}"

    def _wait_for_process(self, proc: ProcessHandle, timeout: int=120) -> dict:
        """Poll-based wait with interrupt checking and stdout draining.

        Shared across all backends — not overridden.

        Fires the ``activity_callback`` (if set on this instance) every 10s
        while the process is running so the gateway's inactivity timeout
        doesn't kill long-running commands.

        Also wraps the poll loop in a ``try/finally`` that guarantees we
        call ``self._kill_process(proc)`` if we exit via ``KeyboardInterrupt``
        or ``SystemExit``.  Without this, the local backend (which spawns
        subprocesses with ``os.setsid`` into their own process group) leaves
        an orphan with ``PPID=1`` when python is shut down mid-tool — the
        ``sleep 300``-survives-30-min bug Physikal and I both hit.
        """
        output_chunks: list[str] = []
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        def _drain():
            fd = proc.stdout.fileno()
            if os.name == 'nt':
                try:
                    while True:
                        chunk = os.read(fd, 4096)
                        if not chunk:
                            break
                        output_chunks.append(decoder.decode(chunk))
                except (ValueError, OSError):
                    pass
                finally:
                    try:
                        tail = decoder.decode(b'', final=True)
                        if tail:
                            output_chunks.append(tail)
                    except Exception:
                        pass
                return
            idle_after_exit = 0
            try:
                while True:
                    try:
                        ready, _, _ = select.select([fd], [], [], 0.1)
                    except (ValueError, OSError):
                        break
                    if ready:
                        try:
                            chunk = os.read(fd, 4096)
                        except (ValueError, OSError):
                            break
                        if not chunk:
                            break
                        output_chunks.append(decoder.decode(chunk))
                        idle_after_exit = 0
                    elif proc.poll() is not None:
                        idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
            finally:
                try:
                    tail = decoder.decode(b'', final=True)
                    if tail:
                        output_chunks.append(tail)
                except Exception:
                    pass
        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()
        deadline = time.monotonic() + timeout
        _now = time.monotonic()
        _activity_state = {'last_touch': _now, 'start': _now}
        _tid = threading.current_thread().ident
        _pid = getattr(proc, 'pid', None)
        _iter_count = 0
        _last_heartbeat = _now
        _last_interrupt_state = False
        _cb_was_none = _get_activity_callback() is None
        if _DEBUG_INTERRUPT:
            logger.info('[interrupt-debug] _wait_for_process ENTER tid=%s pid=%s timeout=%ss activity_cb=%s initial_interrupt=%s', _tid, _pid, timeout, 'set' if not _cb_was_none else 'MISSING', is_interrupted())
        try:
            _poll_sleep = 0.005
            while proc.poll() is None:
                _iter_count += 1
                if is_interrupted():
                    if _DEBUG_INTERRUPT:
                        logger.info('[interrupt-debug] _wait_for_process INTERRUPT DETECTED tid=%s pid=%s iter=%d elapsed=%.1fs — killing process group', _tid, _pid, _iter_count, time.monotonic() - _activity_state['start'])
                    self._kill_process(proc)
                    drain_thread.join(timeout=2)
                    return {'output': ''.join(output_chunks) + '\n[Command interrupted]', 'returncode': 130}
                if time.monotonic() > deadline:
                    if _DEBUG_INTERRUPT:
                        logger.info('[interrupt-debug] _wait_for_process TIMEOUT tid=%s pid=%s iter=%d timeout=%ss', _tid, _pid, _iter_count, timeout)
                    self._kill_process(proc)
                    drain_thread.join(timeout=2)
                    partial = ''.join(output_chunks)
                    timeout_msg = f'\n[Command timed out after {timeout}s]'
                    return {'output': partial + timeout_msg if partial else timeout_msg.lstrip(), 'returncode': 124}
                touch_activity_if_due(_activity_state, 'terminal command running')
                if _DEBUG_INTERRUPT and time.monotonic() - _last_heartbeat >= 30.0:
                    _cb_now_none = _get_activity_callback() is None
                    logger.info('[interrupt-debug] _wait_for_process HEARTBEAT tid=%s pid=%s iter=%d elapsed=%.0fs interrupt=%s activity_cb=%s%s', _tid, _pid, _iter_count, time.monotonic() - _activity_state['start'], is_interrupted(), 'set' if not _cb_now_none else 'MISSING', ' (LOST during run)' if _cb_now_none and (not _cb_was_none) else '')
                    _last_heartbeat = time.monotonic()
                    _cb_was_none = _cb_now_none
                time.sleep(_poll_sleep)
                if _poll_sleep < 0.2:
                    _poll_sleep = min(_poll_sleep * 1.5, 0.2)
        except (KeyboardInterrupt, SystemExit):
            if _DEBUG_INTERRUPT:
                logger.info('[interrupt-debug] _wait_for_process EXCEPTION_EXIT tid=%s pid=%s iter=%d elapsed=%.1fs — killing subprocess group before re-raise', _tid, _pid, _iter_count, time.monotonic() - _activity_state['start'])
            try:
                self._kill_process(proc)
                drain_thread.join(timeout=2)
            except Exception:
                pass
            raise
        drain_thread.join(timeout=2)
        try:
            proc.stdout.close()
        except Exception:
            pass
        if _DEBUG_INTERRUPT:
            logger.info('[interrupt-debug] _wait_for_process EXIT (natural) tid=%s pid=%s iter=%d elapsed=%.1fs returncode=%s', _tid, _pid, _iter_count, time.monotonic() - _activity_state['start'], proc.returncode)
        return {'output': ''.join(output_chunks), 'returncode': proc.returncode}

    def _kill_process(self, proc: ProcessHandle):
        """Terminate a process. Subclasses may override for process-group kill."""
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _update_cwd(self, result: dict):
        """Extract CWD from command output. Override for local file-based read."""
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict):
        """Parse the __HERMES_CWD_{session}__ marker from stdout output.

        Updates self.cwd and strips the marker from result["output"].
        Used by remote backends (Docker, SSH, Modal, Daytona, Singularity).
        """
        output = result.get('output', '')
        marker = self._cwd_marker
        last = output.rfind(marker)
        if last == -1:
            return
        search_start = max(0, last - 4096)
        first = output.rfind(marker, search_start, last)
        if first == -1 or first == last:
            return
        cwd_path = output[first + len(marker):last].strip()
        if cwd_path:
            self.cwd = cwd_path
        line_start = output.rfind('\n', 0, first)
        if line_start == -1:
            line_start = first
        line_end = output.find('\n', last + len(marker))
        line_end = line_end + 1 if line_end != -1 else len(output)
        result['output'] = output[:line_start] + output[line_end:]

    def _before_execute(self) -> None:
        """Hook called before each command execution.

        Remote backends (SSH, Modal, Daytona) override this to trigger
        their FileSyncManager.  Bind-mount backends (Docker, Singularity)
        and Local don't need file sync — the host filesystem is directly
        visible inside the container/process.
        """
        pass

    def execute(self, command: str, cwd: str='', *, timeout: int | None=None, stdin_data: str | None=None) -> dict:
        """Execute a command, return {"output": str, "returncode": int}."""
        self._before_execute()
        exec_command, sudo_stdin = self._prepare_command(command)
        from kylinmemory._vendor.tools.terminal_tool import _rewrite_compound_background
        exec_command = _rewrite_compound_background(exec_command)
        effective_timeout = timeout or self.timeout
        effective_cwd = cwd or self.cwd
        if sudo_stdin is not None and stdin_data is not None:
            effective_stdin = sudo_stdin + stdin_data
        elif sudo_stdin is not None:
            effective_stdin = sudo_stdin
        else:
            effective_stdin = stdin_data
        if effective_stdin and self._stdin_mode == 'heredoc':
            exec_command = self._embed_stdin_heredoc(exec_command, effective_stdin)
            effective_stdin = None
        wrapped = self._wrap_command(exec_command, effective_cwd)
        login = not self._snapshot_ready
        proc = self._run_bash(wrapped, login=login, timeout=effective_timeout, stdin_data=effective_stdin)
        result = self._wait_for_process(proc, timeout=effective_timeout)
        self._update_cwd(result)
        return result

    def stop(self):
        """Alias for cleanup (compat with older callers)."""
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def _prepare_command(self, command: str) -> tuple[str, str | None]:
        """Transform sudo commands if SUDO_PASSWORD is available."""
        from kylinmemory._vendor.tools.terminal_tool import _transform_sudo_command
        return _transform_sudo_command(command)

