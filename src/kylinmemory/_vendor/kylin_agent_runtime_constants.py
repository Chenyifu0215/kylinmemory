import os
from contextvars import ContextVar, Token
from pathlib import Path
_profile_fallback_warned: bool = False
_UNSET = object()
_HERMES_HOME_OVERRIDE: ContextVar[str | object] = ContextVar('_HERMES_HOME_OVERRIDE', default=_UNSET)
LEGACY_KYLIN_STATE_DB_FILENAME = 'hermes_state.db'
KYLIN_STATE_DB_FILENAME = 'kylin_agent_runtime_state.db'

def set_hermes_home_override(path: str | Path | None) -> Token:
    """Set a context-local Hermes home override and return its reset token.

    This is for in-process, per-task scoping.  It deliberately does not mutate
    ``os.environ`` because that is shared by every thread in the process.
    """
    value: str | object = _UNSET if path is None else str(path)
    return _HERMES_HOME_OVERRIDE.set(value)

def reset_hermes_home_override(token: Token) -> None:
    """Restore the previous context-local Hermes home override."""
    _HERMES_HOME_OVERRIDE.reset(token)

def get_hermes_home_override() -> str | None:
    """Return the active context-local Hermes home override, if any."""
    override = _HERMES_HOME_OVERRIDE.get()
    if override is _UNSET or not override:
        return None
    return str(override)

def get_hermes_home() -> Path:
    """Return the Hermes home directory (default: ~/.kylin-agent-runtime).

    Reads HERMES_HOME env var, falls back to ~/.kylin-agent-runtime.
    This is the single source of truth — all other copies should import this.

    When ``HERMES_HOME`` is unset but an ``active_profile`` file indicates
    a non-default profile is active, logs a loud one-shot warning to
    ``errors.log`` so cross-profile data corruption is diagnosable instead
    of silent.  Behavior is unchanged otherwise — we still return
    ``~/.kylin-agent-runtime`` — because raising here would brick 30+ module-level
    callers that import this at load time.  Subprocess spawners are
    expected to propagate ``HERMES_HOME`` explicitly (see the systemd
    template in ``kylin_agent_runtime_cli/gateway.py`` and the kanban dispatcher in
    ``kylin_agent_runtime_cli/kanban_db.py``).  See https://github.com/NousResearch/hermes-agent/issues/18594.
    """
    override = get_hermes_home_override()
    if override:
        return Path(override)
    val = os.environ.get('HERMES_HOME', '').strip()
    if val:
        return Path(val)
    global _profile_fallback_warned
    if not _profile_fallback_warned:
        try:
            active_path = Path.home() / '.kylin-agent-runtime' / 'active_profile'
            active = active_path.read_text().strip() if active_path.exists() else ''
        except (UnicodeDecodeError, OSError):
            active = ''
        if active and active != 'default':
            _profile_fallback_warned = True
            import sys
            msg = f'[HERMES_HOME fallback] HERMES_HOME is unset but active profile is {active!r}. Falling back to ~/.kylin-agent-runtime, which is the DEFAULT profile — not {active!r}. Any data this process writes will land in the wrong profile. The subprocess spawner should pass HERMES_HOME explicitly (see issue #18594).'
            try:
                sys.stderr.write(msg + '\n')
                sys.stderr.flush()
            except Exception:
                pass
    return Path.home() / '.kylin-agent-runtime'

def ensure_kylin_state_db(home: str | Path | None=None) -> Path:
    """Migrate or create the branded runtime state database file.

    The canonical session database remains ``state.db``.  This helper only
    handles the legacy branded ``hermes_state.db`` file requested by the
    runtime-directory migration.
    """
    root = Path(home) if home is not None else get_hermes_home()
    root.mkdir(parents=True, exist_ok=True)
    legacy_path = root / LEGACY_KYLIN_STATE_DB_FILENAME
    state_path = root / KYLIN_STATE_DB_FILENAME
    if state_path.exists():
        return state_path
    if legacy_path.exists():
        legacy_path.rename(state_path)
    else:
        state_path.touch(exist_ok=True)
    return state_path

def get_default_hermes_root() -> Path:
    """Return the root Hermes directory for profile-level operations.

    In standard deployments this is ``~/.kylin-agent-runtime``.

    In Docker or custom deployments where ``HERMES_HOME`` points outside
    ``~/.kylin-agent-runtime`` (e.g. ``/opt/data``), returns ``HERMES_HOME`` directly
    — that IS the root.

    In profile mode where ``HERMES_HOME`` is ``<root>/profiles/<name>``,
    returns ``<root>`` so that ``profile list`` can see all profiles.
    Works both for standard (``~/.kylin-agent-runtime/profiles/coder``) and Docker
    (``/opt/data/profiles/coder``) layouts.

    Import-safe — no dependencies beyond stdlib.
    """
    native_home = Path.home() / '.kylin-agent-runtime'
    env_home = os.environ.get('HERMES_HOME', '')
    if not env_home:
        return native_home
    env_path = Path(env_home)
    try:
        env_path.resolve().relative_to(native_home.resolve())
        return native_home
    except ValueError:
        pass
    if env_path.parent.name == 'profiles':
        return env_path.parent.parent
    return env_path

def display_hermes_home() -> str:
    """Return a user-friendly display string for the current HERMES_HOME.

    Uses ``~/`` shorthand for readability::

        default:  ``~/.kylin-agent-runtime``
        profile:  ``~/.kylin-agent-runtime/profiles/coder``
        custom:   ``/opt/hermes-custom``

    Use this in **user-facing** print/log messages instead of hardcoding
    ``~/.kylin-agent-runtime``.  For code that needs a real ``Path``, use
    :func:`get_hermes_home` instead.
    """
    home = get_hermes_home()
    try:
        return '~/' + str(home.relative_to(Path.home()))
    except ValueError:
        return str(home)

def get_subprocess_home() -> str | None:
    """Return a per-profile HOME directory for subprocesses, or None.

    When ``{HERMES_HOME}/home/`` exists on disk, subprocesses should use it
    as ``HOME`` so system tools (git, ssh, gh, npm …) write their configs
    inside the Hermes data directory instead of the OS-level ``/root`` or
    ``~/``.  This provides:

    * **Docker persistence** — tool configs land inside the persistent volume.
    * **Profile isolation** — each profile gets its own git identity, SSH
      keys, gh tokens, etc.

    The Python process's own ``os.environ["HOME"]`` and ``Path.home()`` are
    **never** modified — only subprocess environments should inject this value.
    Activation is directory-based: if the ``home/`` subdirectory doesn't
    exist, returns ``None`` and behavior is unchanged.
    """
    hermes_home = get_hermes_home_override() or os.getenv('HERMES_HOME')
    if not hermes_home:
        return None
    profile_home = os.path.join(hermes_home, 'home')
    if os.path.isdir(profile_home):
        return profile_home
    return None
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
AI_GATEWAY_BASE_URL = 'https://ai-gateway.vercel.sh/v1'

