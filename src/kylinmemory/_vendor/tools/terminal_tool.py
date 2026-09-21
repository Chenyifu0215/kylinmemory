import importlib.util
import json
import logging
import os
import platform
import re
import time
import threading
import shutil
import subprocess
from typing import Optional, Dict, Any, List
from kylinmemory._vendor.utils import env_var_enabled
logger = logging.getLogger(__name__)
from kylinmemory._vendor.tools.tool_backend_helpers import coerce_modal_mode, has_direct_modal_credentials, managed_nous_tools_enabled, resolve_modal_backend_state
_VERCEL_SANDBOX_DEFAULT_CWD = '/vercel/sandbox'
_SUPPORTED_VERCEL_RUNTIMES = ('node24', 'node22', 'python3.13')

def _is_supported_vercel_runtime(runtime: str) -> bool:
    return not runtime or runtime in _SUPPORTED_VERCEL_RUNTIMES

def _check_vercel_sandbox_requirements(config: dict[str, Any]) -> bool:
    """Validate Vercel Sandbox terminal backend requirements."""
    runtime = (config.get('vercel_runtime') or '').strip()
    if not _is_supported_vercel_runtime(runtime):
        supported = ', '.join(_SUPPORTED_VERCEL_RUNTIMES)
        logger.error('Vercel Sandbox runtime %r is not supported. Set TERMINAL_VERCEL_RUNTIME to one of: %s.', runtime, supported)
        return False
    disk = config.get('container_disk', 51200)
    if disk not in {0, 51200}:
        logger.error('Vercel Sandbox does not support custom TERMINAL_CONTAINER_DISK=%s. Use the default shared setting (51200 MB).', disk)
        return False
    if importlib.util.find_spec('vercel') is None:
        logger.error('vercel is required for the Vercel Sandbox terminal backend: pip install vercel')
        return False
    has_oidc = bool(os.getenv('VERCEL_OIDC_TOKEN'))
    has_token = bool(os.getenv('VERCEL_TOKEN'))
    has_project = bool(os.getenv('VERCEL_PROJECT_ID'))
    has_team = bool(os.getenv('VERCEL_TEAM_ID'))
    if has_oidc:
        return True
    if has_token or has_project or has_team:
        if has_token and has_project and has_team:
            return True
        logger.error('Vercel Sandbox backend selected with token auth, but VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID must all be set together. VERCEL_OIDC_TOKEN is supported for one-off local development only.')
        return False
    logger.error('Vercel Sandbox backend selected but no supported auth configuration was found. Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID for normal use. VERCEL_OIDC_TOKEN is supported for one-off local development only.')
    return False
_sudo_password_cache: dict[str, str] = {}
_sudo_password_cache_lock = threading.Lock()
import threading
_callback_tls = threading.local()

def _get_sudo_password_callback():
    return getattr(_callback_tls, 'sudo_password', None)

def _get_sudo_password_cache_scope() -> str:
    """Return the cache scope for interactive sudo passwords."""
    try:
        from kylinmemory._vendor.gateway.session_context import get_session_env
        session_key = get_session_env('HERMES_SESSION_KEY', '')
    except Exception:
        session_key = os.getenv('HERMES_SESSION_KEY', '')
    if session_key:
        return f'session:{session_key}'
    callback = _get_sudo_password_callback()
    if callback is not None:
        owner = getattr(callback, '__self__', None)
        func = getattr(callback, '__func__', None)
        if owner is not None and func is not None:
            return f'callback-owner:{id(owner)}:{id(func)}'
        return f'callback:{id(callback)}'
    return f'thread:{threading.get_ident()}'

def _get_cached_sudo_password() -> str:
    """Return the cached sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        return _sudo_password_cache.get(scope, '')

def _set_cached_sudo_password(password: str) -> None:
    """Persist a sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        if password:
            _sudo_password_cache[scope] = password
        else:
            _sudo_password_cache.pop(scope, None)

def _prompt_for_sudo_password(timeout_seconds: int=45) -> str:
    """
    Prompt user for sudo password with timeout.
    
    Returns the password if entered, or empty string if:
    - User presses Enter without input (skip)
    - Timeout expires (45s default)
    - Any error occurs
    
    Only works in interactive mode (HERMES_INTERACTIVE=1).
    If a _sudo_password_callback is registered (by the CLI), delegates to it
    so the prompt integrates with prompt_toolkit's UI.  Otherwise reads
    directly from /dev/tty with echo disabled.
    """
    import sys
    _sudo_cb = _get_sudo_password_callback()
    if _sudo_cb is not None:
        try:
            return _sudo_cb() or ''
        except Exception:
            return ''
    result = {'password': None, 'done': False}

    def read_password_thread():
        """Read password with echo disabled. Uses msvcrt on Windows, /dev/tty on Unix."""
        tty_fd = None
        old_attrs = None
        try:
            if platform.system() == 'Windows':
                import msvcrt
                chars = []
                while True:
                    c = msvcrt.getwch()
                    if c in {'\r', '\n'}:
                        break
                    if c == '\x03':
                        raise KeyboardInterrupt
                    chars.append(c)
                result['password'] = ''.join(chars)
            else:
                import termios
                tty_fd = os.open('/dev/tty', os.O_RDONLY)
                old_attrs = termios.tcgetattr(tty_fd)
                new_attrs = termios.tcgetattr(tty_fd)
                new_attrs[3] = new_attrs[3] & ~termios.ECHO
                termios.tcsetattr(tty_fd, termios.TCSAFLUSH, new_attrs)
                chars = []
                while True:
                    b = os.read(tty_fd, 1)
                    if not b or b in {b'\n', b'\r'}:
                        break
                    chars.append(b)
                result['password'] = b''.join(chars).decode('utf-8', errors='replace')
        except (EOFError, KeyboardInterrupt, OSError):
            result['password'] = ''
        except Exception:
            result['password'] = ''
        finally:
            if tty_fd is not None and old_attrs is not None:
                try:
                    import termios as _termios
                    _termios.tcsetattr(tty_fd, _termios.TCSAFLUSH, old_attrs)
                except Exception as e:
                    logger.debug('Failed to restore terminal attributes: %s', e)
            if tty_fd is not None:
                try:
                    os.close(tty_fd)
                except Exception as e:
                    logger.debug('Failed to close tty fd: %s', e)
            result['done'] = True
    try:
        os.environ['HERMES_SPINNER_PAUSE'] = '1'
        time.sleep(0.2)
        print()
        print('┌' + '─' * 58 + '┐')
        print('│  🔐 SUDO PASSWORD REQUIRED' + ' ' * 30 + '│')
        print('├' + '─' * 58 + '┤')
        print('│  Enter password below (input is hidden), or:            │')
        print('│    • Press Enter to skip (command fails gracefully)     │')
        print(f'│    • Wait {timeout_seconds}s to auto-skip' + ' ' * 27 + '│')
        print('└' + '─' * 58 + '┘')
        print()
        print('  Password (hidden): ', end='', flush=True)
        password_thread = threading.Thread(target=read_password_thread, daemon=True)
        password_thread.start()
        password_thread.join(timeout=timeout_seconds)
        if result['done']:
            password = result['password'] or ''
            print()
            if password:
                print('  ✓ Password received (cached for this session)')
            else:
                print('  ⏭ Skipped - continuing without sudo')
            print()
            sys.stdout.flush()
            return password
        else:
            print('\n  ⏱ Timeout - continuing without sudo')
            print('    (Press Enter to dismiss)')
            print()
            sys.stdout.flush()
            return ''
    except (EOFError, KeyboardInterrupt):
        print()
        print('  ⏭ Cancelled - continuing without sudo')
        print()
        sys.stdout.flush()
        return ''
    except Exception as e:
        print(f'\n  [sudo prompt error: {e}] - continuing without sudo\n')
        sys.stdout.flush()
        return ''
    finally:
        if 'HERMES_SPINNER_PAUSE' in os.environ:
            del os.environ['HERMES_SPINNER_PAUSE']

def _looks_like_env_assignment(token: str) -> bool:
    """Return True when *token* is a leading shell environment assignment."""
    if '=' not in token or token.startswith('='):
        return False
    name, _value = token.split('=', 1)
    return bool(re.match('^[A-Za-z_][A-Za-z0-9_]*$', name))

def _read_shell_token(command: str, start: int) -> tuple[str, int]:
    """Read one shell token, preserving quotes/escapes, starting at *start*."""
    i = start
    n = len(command)
    while i < n:
        ch = command[i]
        if ch.isspace() or ch in ';|&()':
            break
        if ch == "'":
            i += 1
            while i < n and command[i] != "'":
                i += 1
            if i < n:
                i += 1
            continue
        if ch == '"':
            i += 1
            while i < n:
                inner = command[i]
                if inner == '\\' and i + 1 < n:
                    i += 2
                    continue
                if inner == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == '\\' and i + 1 < n:
            i += 2
            continue
        i += 1
    return (command[start:i], i)

def _rewrite_real_sudo_invocations(command: str, replacement: str="sudo -S -p ''") -> tuple[str, bool]:
    """Rewrite only real unquoted sudo command words, not plain text mentions."""
    out: list[str] = []
    i = 0
    n = len(command)
    command_start = True
    found = False
    while i < n:
        ch = command[i]
        if ch.isspace():
            out.append(ch)
            if ch == '\n':
                command_start = True
            i += 1
            continue
        if ch == '#' and command_start:
            comment_end = command.find('\n', i)
            if comment_end == -1:
                out.append(command[i:])
                break
            out.append(command[i:comment_end])
            i = comment_end
            continue
        if command.startswith('&&', i) or command.startswith('||', i) or command.startswith(';;', i):
            out.append(command[i:i + 2])
            i += 2
            command_start = True
            continue
        if ch in ';|&(':
            out.append(ch)
            i += 1
            command_start = True
            continue
        if ch == ')':
            out.append(ch)
            i += 1
            command_start = False
            continue
        token, next_i = _read_shell_token(command, i)
        if command_start and token == 'sudo':
            out.append(replacement)
            found = True
        else:
            out.append(token)
        if command_start and _looks_like_env_assignment(token):
            command_start = True
        else:
            command_start = False
        i = next_i
    return (''.join(out), found)

def _sudo_nopasswd_works() -> bool:
    """Return True when local sudo currently works without prompting.

    Only probes for the `local` terminal backend; Docker/SSH/Modal/etc. must
    not inherit the host's sudo state. Re-probes every call (no process-level
    cache) so an expired sudo timestamp cannot make a later command silently
    block waiting for a password.
    """
    terminal_env = os.getenv('TERMINAL_ENV', 'local').strip().lower() or 'local'
    if terminal_env != 'local':
        return False
    try:
        probe = subprocess.run(['sudo', '-n', 'true'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=False)
        return probe.returncode == 0
    except Exception:
        return False

def _rewrite_compound_background(command: str) -> str:
    """Wrap `A && B &` (or `A || B &`) to `A && { B & }` at depth 0.

    Bash parses ``A && B &`` with `&&` tighter than `&`, so it forks a
    subshell for the whole `A && B` compound and backgrounds it. Inside
    the subshell, `B` runs foreground, so the subshell waits for `B` to
    finish. When `B` is a long-running process (`python3 -m http.server`,
    `yes > /dev/null`, anything that doesn't naturally exit), the subshell
    never exits. It leaks as a process stuck in ``wait4`` forever — and
    on the way, its open stdout pipe can prevent the terminal tool from
    returning promptly.

    Rewriting the tail to `A && { B & }` preserves `&&`'s error semantics
    (skip B if A fails) while replacing the subshell with a brace group.
    The brace group runs in the current shell (no fork), backgrounds B as
    a simple command (bash doesn't wait for it in non-interactive mode),
    and exits immediately. B runs as a normal backgrounded child, orphaned
    when the parent shell exits.

    Handles redirects (``&>``, ``2>&1``) and skips content inside quoted
    strings and parenthesised subshells. Leaves simple ``cmd &`` alone —
    that construct doesn't have the subshell-wait bug.
    """
    n = len(command)
    i = 0
    paren_depth = 0
    brace_depth = 0
    last_chain_op_end = -1
    rewrites: list[tuple[int, int]] = []
    while i < n:
        ch = command[i]
        if ch == '\n' and paren_depth == 0 and (brace_depth == 0):
            last_chain_op_end = -1
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        if ch == '#':
            nl = command.find('\n', i)
            if nl == -1:
                break
            i = nl
            continue
        if ch == '\\' and i + 1 < n:
            i += 2
            continue
        if ch in {"'", '"'}:
            _, next_i = _read_shell_token(command, i)
            i = max(next_i, i + 1)
            continue
        if ch == '(':
            paren_depth += 1
            i += 1
            continue
        if ch == ')':
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue
        if ch == '{' and i + 1 < n and (command[i + 1].isspace() or command[i + 1] == '\n'):
            brace_depth += 1
            i += 1
            continue
        if ch == '}' and brace_depth > 0:
            brace_depth -= 1
            last_chain_op_end = -1
            i += 1
            continue
        if paren_depth > 0 or brace_depth > 0:
            i += 1
            continue
        if command.startswith('&&', i) or command.startswith('||', i):
            last_chain_op_end = i + 2
            i += 2
            continue
        if ch == ';':
            last_chain_op_end = -1
            i += 1
            continue
        if ch == '|':
            last_chain_op_end = -1
            i += 1
            continue
        if ch == '&':
            if i + 1 < n and command[i + 1] == '>':
                i += 2
                continue
            j = i - 1
            while j >= 0 and command[j].isspace():
                j -= 1
            if j >= 0 and command[j] in '<>':
                i += 1
                continue
            if last_chain_op_end >= 0:
                rewrites.append((last_chain_op_end, i))
            last_chain_op_end = -1
            i += 1
            continue
        _, next_i = _read_shell_token(command, i)
        i = max(next_i, i + 1)
    if not rewrites:
        return command
    result = command
    for chain_end, amp_pos in reversed(rewrites):
        insert_pos = chain_end
        while insert_pos < amp_pos and result[insert_pos].isspace():
            insert_pos += 1
        prefix = result[:insert_pos]
        middle = result[insert_pos:amp_pos]
        suffix = result[amp_pos + 1:]
        result = prefix + '{ ' + middle + '& }' + suffix
    return result

def _transform_sudo_command(command: str | None) -> tuple[str | None, str | None]:
    """
    Transform sudo commands to use -S flag if SUDO_PASSWORD is available.

    This is a shared helper used by all execution environments to provide
    consistent sudo handling across local, SSH, and container environments.

    Returns:
        (transformed_command, sudo_stdin) where:
        - transformed_command has every bare ``sudo`` replaced with
          ``sudo -S -p ''`` so sudo reads its password from stdin.
        - sudo_stdin is the password string with a trailing newline that the
          caller must prepend to the process's stdin stream.  sudo -S reads
          exactly one line (the password) and passes the rest of stdin to the
          child command, so prepending is safe even when the caller also has
          its own stdin_data to pipe.
        - If no password is available, sudo_stdin is None and the command is
          returned unchanged so it fails gracefully with
          "sudo: a password is required".

    Callers that drive a subprocess directly (local, ssh, docker, singularity)
    should prepend sudo_stdin to their stdin_data and pass the merged bytes to
    Popen's stdin pipe.

    Callers that cannot pipe subprocess stdin (modal, daytona,
    vercel_sandbox) must embed the password in the command string
    themselves; see their execute() methods for how they handle the
    non-None sudo_stdin case.

    If SUDO_PASSWORD is not set and in interactive mode (HERMES_INTERACTIVE=1):
      Prompts user for password with 45s timeout, caches for session.

    If SUDO_PASSWORD is not set and NOT interactive:
      Command runs as-is (fails gracefully with "sudo: a password is required").
    """
    if command is None:
        return (None, None)
    transformed, has_real_sudo = _rewrite_real_sudo_invocations(command)
    if not has_real_sudo:
        return (command, None)
    has_configured_password = 'SUDO_PASSWORD' in os.environ
    sudo_password = os.environ.get('SUDO_PASSWORD', '') if has_configured_password else _get_cached_sudo_password()
    if not has_configured_password and (not sudo_password) and _sudo_nopasswd_works():
        return (command, None)
    if not has_configured_password and (not sudo_password) and env_var_enabled('HERMES_INTERACTIVE'):
        sudo_password = _prompt_for_sudo_password(timeout_seconds=45)
        if sudo_password:
            _set_cached_sudo_password(sudo_password)
    if has_configured_password or sudo_password:
        return (transformed, sudo_password + '\n')
    return (command, None)
from kylinmemory._vendor.tools.managed_tool_gateway import is_managed_tool_gateway_ready
import sys

def _parse_env_var(name: str, default: str, converter=int, type_label: str='integer'):
    """Parse an environment variable with *converter*, raising a clear error on bad values.

    Without this wrapper, a single malformed env var (e.g. TERMINAL_TIMEOUT=5m)
    causes an unhandled ValueError that kills every terminal command.
    """
    raw = os.getenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(f'Invalid value for {name}: {raw!r} (expected {type_label}). Check ~/.kylin-agent-runtime/.env or environment variables.')

def _get_env_config() -> Dict[str, Any]:
    """Get terminal environment configuration from environment variables."""
    default_image = 'nikolaik/python-nodejs:python3.11-nodejs20'
    env_type = os.getenv('TERMINAL_ENV', 'local')
    mount_docker_cwd = os.getenv('TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE', 'false').lower() in {'true', '1', 'yes'}
    if env_type == 'local':
        default_cwd = os.getcwd()
    elif env_type == 'ssh':
        default_cwd = '~'
    elif env_type == 'vercel_sandbox':
        default_cwd = _VERCEL_SANDBOX_DEFAULT_CWD
    else:
        default_cwd = '/root'
    cwd = os.getenv('TERMINAL_CWD', default_cwd)
    if cwd:
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    host_prefixes = ('/Users/', '/home/', 'C:\\', 'C:/')
    if env_type == 'docker' and mount_docker_cwd:
        docker_cwd_source = os.getenv('TERMINAL_CWD') or os.getcwd()
        candidate = os.path.abspath(os.path.expanduser(docker_cwd_source))
        if any((candidate.startswith(p) for p in host_prefixes)) or (os.path.isabs(candidate) and os.path.isdir(candidate) and (not candidate.startswith(('/workspace', '/root')))):
            host_cwd = candidate
            cwd = '/workspace'
    elif env_type in {'modal', 'docker', 'singularity', 'daytona', 'vercel_sandbox'} and cwd:
        is_host_path = any((cwd.startswith(p) for p in host_prefixes))
        is_relative = not os.path.isabs(cwd)
        if (is_host_path or is_relative) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend (host/relative path won't work in sandbox). Using %r instead.", cwd, env_type, default_cwd)
            cwd = default_cwd
    return {'env_type': env_type, 'modal_mode': coerce_modal_mode(os.getenv('TERMINAL_MODAL_MODE', 'auto')), 'docker_image': os.getenv('TERMINAL_DOCKER_IMAGE', default_image), 'docker_forward_env': _parse_env_var('TERMINAL_DOCKER_FORWARD_ENV', '[]', json.loads, 'valid JSON'), 'singularity_image': os.getenv('TERMINAL_SINGULARITY_IMAGE', f'docker://{default_image}'), 'modal_image': os.getenv('TERMINAL_MODAL_IMAGE', default_image), 'daytona_image': os.getenv('TERMINAL_DAYTONA_IMAGE', default_image), 'vercel_runtime': os.getenv('TERMINAL_VERCEL_RUNTIME', '').strip(), 'cwd': cwd, 'host_cwd': host_cwd, 'docker_mount_cwd_to_workspace': mount_docker_cwd, 'timeout': _parse_env_var('TERMINAL_TIMEOUT', '180'), 'lifetime_seconds': _parse_env_var('TERMINAL_LIFETIME_SECONDS', '300'), 'ssh_host': os.getenv('TERMINAL_SSH_HOST', ''), 'ssh_user': os.getenv('TERMINAL_SSH_USER', ''), 'ssh_port': _parse_env_var('TERMINAL_SSH_PORT', '22'), 'ssh_key': os.getenv('TERMINAL_SSH_KEY', ''), 'ssh_persistent': os.getenv('TERMINAL_SSH_PERSISTENT', os.getenv('TERMINAL_PERSISTENT_SHELL', 'true')).lower() in {'true', '1', 'yes'}, 'local_persistent': os.getenv('TERMINAL_LOCAL_PERSISTENT', 'false').lower() in {'true', '1', 'yes'}, 'container_cpu': _parse_env_var('TERMINAL_CONTAINER_CPU', '1', float, 'number'), 'container_memory': _parse_env_var('TERMINAL_CONTAINER_MEMORY', '5120'), 'container_disk': _parse_env_var('TERMINAL_CONTAINER_DISK', '51200'), 'container_persistent': os.getenv('TERMINAL_CONTAINER_PERSISTENT', 'true').lower() in {'true', '1', 'yes'}, 'docker_volumes': _parse_env_var('TERMINAL_DOCKER_VOLUMES', '[]', json.loads, 'valid JSON'), 'docker_env': _parse_env_var('TERMINAL_DOCKER_ENV', '{}', json.loads, 'valid JSON'), 'docker_run_as_host_user': os.getenv('TERMINAL_DOCKER_RUN_AS_HOST_USER', 'false').lower() in {'true', '1', 'yes'}, 'docker_extra_args': _parse_env_var('TERMINAL_DOCKER_EXTRA_ARGS', '[]', json.loads, 'valid JSON')}

def _get_modal_backend_state(modal_mode: object | None) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection."""
    return resolve_modal_backend_state(modal_mode, has_direct=has_direct_modal_credentials(), managed_ready=is_managed_tool_gateway_ready('modal'))

def check_terminal_requirements() -> bool:
    """Check if all requirements for the terminal tool are met."""
    try:
        config = _get_env_config()
        env_type = config['env_type']
        if env_type == 'local':
            return True
        elif env_type == 'docker':
            from kylinmemory._vendor.tools.environments.docker import find_docker
            docker = find_docker()
            if not docker:
                logger.error('Docker executable not found in PATH or common install locations')
                return False
            result = subprocess.run([docker, 'version'], capture_output=True, timeout=5)
            return result.returncode == 0
        elif env_type == 'singularity':
            executable = shutil.which('apptainer') or shutil.which('singularity')
            if executable:
                result = subprocess.run([executable, '--version'], capture_output=True, timeout=5)
                return result.returncode == 0
            return False
        elif env_type == 'ssh':
            if not config.get('ssh_host') or not config.get('ssh_user'):
                logger.error("SSH backend selected but TERMINAL_SSH_HOST and TERMINAL_SSH_USER are not both set. Configure both or switch TERMINAL_ENV to 'local'.")
                return False
            return True
        elif env_type == 'modal':
            modal_state = _get_modal_backend_state(config.get('modal_mode'))
            if modal_state['selected_backend'] == 'managed':
                return True
            if modal_state['selected_backend'] != 'direct':
                if modal_state['managed_mode_blocked']:
                    logger.error('Modal backend selected with TERMINAL_MODAL_MODE=managed, but a paid Nous subscription is required for the Tool Gateway and no direct Modal credentials/config were found. Log in with `kylin-agent-runtime model` or choose TERMINAL_MODAL_MODE=direct/auto.')
                    return False
                if modal_state['mode'] == 'managed':
                    logger.error('Modal backend selected with TERMINAL_MODAL_MODE=managed, but the managed tool gateway is unavailable. Configure the managed gateway or choose TERMINAL_MODAL_MODE=direct/auto.')
                    return False
                elif modal_state['mode'] == 'direct':
                    if managed_nous_tools_enabled():
                        logger.error('Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct Modal credentials/config were found. Configure Modal or choose TERMINAL_MODAL_MODE=managed/auto.')
                    else:
                        logger.error('Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct Modal credentials/config were found. Configure Modal or choose TERMINAL_MODAL_MODE=auto.')
                    return False
                else:
                    if managed_nous_tools_enabled():
                        logger.error('Modal backend selected but no direct Modal credentials/config or managed tool gateway was found. Configure Modal, set up the managed gateway, or choose a different TERMINAL_ENV.')
                    else:
                        logger.error('Modal backend selected but no direct Modal credentials/config was found. Configure Modal or choose a different TERMINAL_ENV.')
                    return False
            if importlib.util.find_spec('modal') is None:
                logger.error('modal is required for direct modal terminal backend: pip install modal')
                return False
            return True
        elif env_type == 'vercel_sandbox':
            return _check_vercel_sandbox_requirements(config)
        elif env_type == 'daytona':
            from daytona import Daytona
            return os.getenv('DAYTONA_API_KEY') is not None
        else:
            logger.error("Unknown TERMINAL_ENV '%s'. Use one of: local, docker, singularity, modal, daytona, vercel_sandbox, ssh.", env_type)
            return False
    except Exception as e:
        logger.error('Terminal requirements check failed: %s', e, exc_info=True)
        return False
if __name__ == '__main__':
    print('Terminal Tool Module')
    print('=' * 50)
    config = _get_env_config()
    print('\nCurrent Configuration:')
    print(f"  Environment type: {config['env_type']}")
    print(f"  Docker image: {config['docker_image']}")
    print(f"  Modal image: {config['modal_image']}")
    print(f"  Working directory: {config['cwd']}")
    print(f"  Default timeout: {config['timeout']}s")
    print(f"  Lifetime: {config['lifetime_seconds']}s")
    if not check_terminal_requirements():
        print('\n❌ Requirements not met. Please check the messages above.')
        sys.exit(1)
    print('\n✅ All requirements met!')
    print('\nAvailable Tool:')
    print('  - terminal_tool: Execute commands in sandboxed environments')
    print('\nUsage Examples:')
    print('  # Execute a command')
    print("  result = terminal_tool(command='ls -la')")
    print('  ')
    print('  # Run a background task')
    print("  result = terminal_tool(command='python server.py', background=True)")
    print('\nEnvironment Variables:')
    default_img = 'nikolaik/python-nodejs:python3.11-nodejs20'
    print(f"  TERMINAL_ENV: {os.getenv('TERMINAL_ENV', 'local')} (local/docker/singularity/modal/daytona/vercel_sandbox/ssh)")
    print(f"  TERMINAL_DOCKER_IMAGE: {os.getenv('TERMINAL_DOCKER_IMAGE', default_img)}")
    print(f"  TERMINAL_SINGULARITY_IMAGE: {os.getenv('TERMINAL_SINGULARITY_IMAGE', f'docker://{default_img}')}")
    print(f"  TERMINAL_MODAL_IMAGE: {os.getenv('TERMINAL_MODAL_IMAGE', default_img)}")
    print(f"  TERMINAL_DAYTONA_IMAGE: {os.getenv('TERMINAL_DAYTONA_IMAGE', default_img)}")
    print(f"  TERMINAL_CWD: {os.getenv('TERMINAL_CWD', os.getcwd())}")
    from kylinmemory._vendor.kylin_agent_runtime_constants import display_hermes_home as _dhh
    print(f"  TERMINAL_SANDBOX_DIR: {os.getenv('TERMINAL_SANDBOX_DIR', f'{_dhh()}/sandboxes')}")
    print(f"  TERMINAL_TIMEOUT: {os.getenv('TERMINAL_TIMEOUT', '60')}")
    print(f"  TERMINAL_LIFETIME_SECONDS: {os.getenv('TERMINAL_LIFETIME_SECONDS', '300')}")

