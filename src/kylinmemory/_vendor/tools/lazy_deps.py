from __future__ import annotations
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
logger = logging.getLogger(__name__)
LAZY_DEPS: dict[str, tuple[str, ...]] = {'provider.anthropic': ('anthropic==0.87.0',), 'provider.bedrock': ('boto3==1.42.89',), 'provider.azure_identity': ('azure-identity==1.25.3',), 'search.exa': ('exa-py==2.10.2',), 'search.firecrawl': ('firecrawl-py==4.17.0',), 'search.parallel': ('parallel-web==0.4.2',), 'tts.edge': ('edge-tts==7.2.7',), 'tts.elevenlabs': ('elevenlabs==1.59.0',), 'stt.faster_whisper': ('faster-whisper==1.2.1', 'sounddevice==0.5.5', 'numpy==2.4.3'), 'image.fal': ('fal-client==0.13.1',), 'memory.honcho': ('honcho-ai==2.0.1',), 'memory.hindsight': ('hindsight-client==0.6.1',), 'memory.atom': ('sqlite-vec==0.1.6',), 'platform.telegram': ('python-telegram-bot[webhooks]==22.6',), 'platform.discord': ('discord.py[voice]==2.7.1', 'brotlicffi==1.2.0.1'), 'platform.slack': ('slack-bolt==1.27.0', 'slack-sdk==3.40.1', 'aiohttp==3.13.4'), 'platform.matrix': ('mautrix[encryption]==0.21.0', 'Markdown==3.10.2', 'aiosqlite==0.22.1', 'asyncpg==0.31.0', 'aiohttp-socks==0.11.0'), 'platform.dingtalk': ('dingtalk-stream==0.24.3', 'alibabacloud-dingtalk==2.2.42', 'qrcode==7.4.2'), 'platform.feishu': ('lark-oapi==1.5.3', 'qrcode==7.4.2'), 'terminal.modal': ('modal==1.3.4',), 'terminal.daytona': ('daytona==0.155.0',), 'terminal.vercel': ('vercel==0.5.7',), 'skill.google_workspace': ('google-api-python-client==2.194.0', 'google-auth-oauthlib==1.3.1', 'google-auth-httplib2==0.3.1'), 'skill.youtube': ('youtube-transcript-api==1.2.4',), 'tool.acp': ('agent-client-protocol==0.9.0',), 'tool.dashboard': ('fastapi==0.133.1', 'uvicorn[standard]==0.41.0')}
_SAFE_SPEC = re.compile('^[A-Za-z0-9_][A-Za-z0-9_.\\-]*(?:\\[[A-Za-z0-9_,\\-]+\\])?(?:[<>=!~]=?[A-Za-z0-9_.\\-+,*<>=!~]+)?$')

class FeatureUnavailable(RuntimeError):
    """A lazily-installable feature is missing and cannot be made available.

    Either the deps were never installed and the user has disabled lazy
    installs, or the install attempt failed.
    """

    def __init__(self, feature: str, missing: tuple[str, ...], reason: str):
        self.feature = feature
        self.missing = missing
        self.reason = reason
        super().__init__(self._format())

    def _format(self) -> str:
        spec_list = ' '.join((repr(s) for s in self.missing))
        return f'Feature {self.feature!r} unavailable: {self.reason}. To enable manually: uv pip install {spec_list}  (or: pip install {spec_list}).'

@dataclass(frozen=True)
class _InstallResult:
    success: bool
    stdout: str
    stderr: str

def _allow_lazy_installs() -> bool:
    """Return the ``security.allow_lazy_installs`` config flag.

    Defaults to True. If config is unreadable we fail open (allow), because
    refusing to install would lock people out of their own backends; the
    decision to block is an explicit user opt-in.
    """
    if os.environ.get('HERMES_DISABLE_LAZY_INSTALLS') == '1':
        return False
    try:
        from kylinmemory._vendor.kylin_agent_runtime_cli.config import load_config
        cfg = load_config()
    except Exception:
        return True
    sec = cfg.get('security') or {}
    val = sec.get('allow_lazy_installs', True)
    return bool(val)

def _spec_is_safe(spec: str) -> bool:
    """Reject pip specs that contain URLs, paths, or shell metacharacters."""
    if not spec or len(spec) > 200:
        return False
    if any((ch in spec for ch in (';', '|', '&', '`', '$', '\n', '\r', '\t', '\\'))):
        return False
    if spec.startswith(('-', '/', '.')) or '://' in spec or '@' in spec:
        return False
    return bool(_SAFE_SPEC.match(spec))

def _pkg_name_from_spec(spec: str) -> str:
    """Extract the bare package name from a pip spec.

    ``"slack-bolt>=1.18.0,<2"`` → ``"slack-bolt"``
    ``"mautrix[encryption]>=0.20"`` → ``"mautrix"``
    """
    m = re.match('^([A-Za-z0-9_][A-Za-z0-9_.\\-]*)', spec)
    return m.group(1) if m else spec

def _specifier_from_spec(spec: str) -> str:
    """Extract just the version-specifier portion of a pip spec.

    ``"honcho-ai==2.0.1"`` → ``"==2.0.1"``
    ``"mautrix[encryption]>=0.20,<1"`` → ``">=0.20,<1"``
    ``"package"`` → ``""`` (no version constraint)
    """
    m = re.match('^[A-Za-z0-9_][A-Za-z0-9_.\\-]*(?:\\[[A-Za-z0-9_,\\-]+\\])?', spec)
    if not m:
        return ''
    return spec[m.end():]

def _is_satisfied(spec: str) -> bool:
    """Is ``spec`` already satisfied in the current env?

    Checks both presence AND version. If the package is installed at a
    version outside the spec's range, returns False so the caller will
    upgrade/downgrade to the pinned version. This is what makes
    ``kylin-agent-runtime update`` propagate pin bumps in :data:`LAZY_DEPS` to already-
    installed backends instead of silently leaving stale versions in place.

    If ``packaging`` is unavailable for any reason (it's a transitive of
    pip so this should never happen), we fall back to a presence-only check
    so we err on the side of "don't churn".
    """
    pkg = _pkg_name_from_spec(spec)
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return False
    try:
        installed = version(pkg)
    except PackageNotFoundError:
        return False
    except Exception:
        return False
    spec_tail = _specifier_from_spec(spec)
    if not spec_tail:
        return True
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return True
    try:
        return Version(installed) in SpecifierSet(spec_tail)
    except (InvalidSpecifier, InvalidVersion, Exception):
        return True

def _venv_pip_install(specs: tuple[str, ...], *, timeout: int=300) -> _InstallResult:
    """Install ``specs`` into the active venv using uv → pip → ensurepip ladder.

    Mirrors the strategy in ``kylin_agent_runtime_cli.tools_config._pip_install`` but
    kept independent here so this module has no CLI dependency.
    """
    if not specs:
        return _InstallResult(True, '', '')
    venv_root = Path(sys.executable).parent.parent
    uv_env = {**os.environ, 'VIRTUAL_ENV': str(venv_root)}
    uv_bin = shutil.which('uv')
    if uv_bin:
        try:
            r = subprocess.run([uv_bin, 'pip', 'install', *specs], capture_output=True, text=True, timeout=timeout, env=uv_env)
            if r.returncode == 0:
                return _InstallResult(True, r.stdout or '', r.stderr or '')
            logger.debug('uv pip install failed: %s', r.stderr)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug('uv invocation failed: %s', e)
    pip_cmd = [sys.executable, '-m', 'pip']
    try:
        probe = subprocess.run(pip_cmd + ['--version'], capture_output=True, text=True, timeout=15)
        if probe.returncode != 0:
            raise FileNotFoundError('pip not in venv')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        try:
            subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade', '--default-pip'], capture_output=True, text=True, timeout=120, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return _InstallResult(False, '', f'pip not available and ensurepip failed: {e}')
    try:
        r = subprocess.run(pip_cmd + ['install', *specs], capture_output=True, text=True, timeout=timeout)
        return _InstallResult(r.returncode == 0, r.stdout or '', r.stderr or '')
    except subprocess.TimeoutExpired as e:
        return _InstallResult(False, '', f'pip install timed out: {e}')
    except Exception as e:
        return _InstallResult(False, '', f'pip install failed: {e}')

def feature_specs(feature: str) -> tuple[str, ...]:
    """Return the registered specs for a feature, or raise KeyError."""
    if feature not in LAZY_DEPS:
        raise KeyError(f'Unknown lazy feature: {feature!r}')
    return LAZY_DEPS[feature]

def feature_missing(feature: str) -> tuple[str, ...]:
    """Return the subset of specs for ``feature`` not currently installed."""
    return tuple((s for s in feature_specs(feature) if not _is_satisfied(s)))

def ensure(feature: str, *, prompt: bool=True) -> None:
    """Make sure all packages for ``feature`` are importable.

    If they're missing, attempts to install them in the active venv. Raises
    :class:`FeatureUnavailable` if the user has disabled lazy installs or
    if the install attempt fails.

    ``prompt``: when True (default) and stdin is a TTY, asks the user to
    confirm before installing. Non-interactive callers (gateway, cron,
    batch) get prompt=False and skip the confirmation — config flag is
    the gate in that case.
    """
    if feature not in LAZY_DEPS:
        raise FeatureUnavailable(feature, (), f'feature {feature!r} not in LAZY_DEPS allowlist')
    missing = feature_missing(feature)
    if not missing:
        return
    for spec in missing:
        if not _spec_is_safe(spec):
            raise FeatureUnavailable(feature, missing, f'refusing to install unsafe spec {spec!r}')
    if not _allow_lazy_installs():
        raise FeatureUnavailable(feature, missing, 'lazy installs disabled (security.allow_lazy_installs=false)')
    if prompt and sys.stdin.isatty() and sys.stdout.isatty():
        spec_list = ', '.join(missing)
        try:
            answer = input(f'\nFeature {feature!r} requires: {spec_list}\nInstall into the active venv now? [Y/n] ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = 'n'
        if answer and answer not in {'y', 'yes'}:
            raise FeatureUnavailable(feature, missing, 'user declined install at prompt')
    logger.info('Lazy-installing %s for feature %r', ' '.join(missing), feature)
    result = _venv_pip_install(missing)
    if not result.success:
        snippet = (result.stderr or result.stdout or '').strip()
        if snippet:
            snippet = snippet[-2000:]
        raise FeatureUnavailable(feature, missing, f"pip install failed: {snippet or 'no error output'}")
    try:
        import importlib.metadata as _md
        if hasattr(_md, '_cache_clear'):
            _md._cache_clear()
    except Exception:
        pass
    still_missing = feature_missing(feature)
    if still_missing:
        raise FeatureUnavailable(feature, still_missing, 'install reported success but packages still not importable (may require Python restart)')
    logger.info('Lazy install complete for feature %r', feature)

