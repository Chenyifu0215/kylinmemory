from __future__ import annotations
import contextlib
import json
import logging
import os
import secrets
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from kylin_memory._vendor.kylin_agent_runtime_constants import get_hermes_home
logger = logging.getLogger(__name__)
ENV_CLIENT_ID = 'HERMES_GEMINI_CLIENT_ID'
ENV_CLIENT_SECRET = 'HERMES_GEMINI_CLIENT_SECRET'
_PUBLIC_CLIENT_ID_PROJECT_NUM = '681255809395'
_PUBLIC_CLIENT_ID_HASH = 'oo8ft2oprdrnp9e3aqf6av3hmdib135j'
_PUBLIC_CLIENT_SECRET_SUFFIX = '4uHgMPm-1o7Sk-geV6Cu5clXFsxl'
_DEFAULT_CLIENT_ID = f'{_PUBLIC_CLIENT_ID_PROJECT_NUM}-{_PUBLIC_CLIENT_ID_HASH}.apps.googleusercontent.com'
_DEFAULT_CLIENT_SECRET = f'GOCSPX-{_PUBLIC_CLIENT_SECRET_SUFFIX}'
import re as _re
from kylin_memory._vendor.utils import atomic_replace
_CLIENT_ID_PATTERN = _re.compile('OAUTH_CLIENT_ID\\s*=\\s*[\'\\"]([0-9]+-[a-z0-9]+\\.apps\\.googleusercontent\\.com)[\'\\"]')
_CLIENT_SECRET_PATTERN = _re.compile('OAUTH_CLIENT_SECRET\\s*=\\s*[\'\\"](GOCSPX-[A-Za-z0-9_-]+)[\'\\"]')
_CLIENT_ID_SHAPE = _re.compile('([0-9]{8,}-[a-z0-9]{20,}\\.apps\\.googleusercontent\\.com)')
_CLIENT_SECRET_SHAPE = _re.compile('(GOCSPX-[A-Za-z0-9_-]{20,})')
TOKEN_ENDPOINT = 'https://oauth2.googleapis.com/token'
REFRESH_SKEW_SECONDS = 60
TOKEN_REQUEST_TIMEOUT_SECONDS = 20.0
LOCK_TIMEOUT_SECONDS = 30.0

class GoogleOAuthError(RuntimeError):
    """Raised for any failure in the Google OAuth flow."""

    def __init__(self, message: str, *, code: str='google_oauth_error') -> None:
        super().__init__(message)
        self.code = code

def _credentials_path() -> Path:
    return get_hermes_home() / 'auth' / 'google_oauth.json'

def _lock_path() -> Path:
    return _credentials_path().with_suffix('.json.lock')
_lock_state = threading.local()

@contextlib.contextmanager
def _credentials_lock(timeout_seconds: float=LOCK_TIMEOUT_SECONDS):
    """Cross-process lock around the credentials file (fcntl POSIX / msvcrt Windows)."""
    depth = getattr(_lock_state, 'depth', 0)
    if depth > 0:
        _lock_state.depth = depth + 1
        try:
            yield
        finally:
            _lock_state.depth -= 1
        return
    lock_file_path = _lock_path()
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_file_path), os.O_CREAT | os.O_RDWR, 384)
    acquired = False
    try:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'Timed out acquiring Google OAuth credentials lock at {lock_file_path}.')
                    time.sleep(0.05)
        else:
            try:
                import msvcrt
                deadline = time.monotonic() + max(0.0, float(timeout_seconds))
                while True:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        acquired = True
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f'Timed out acquiring Google OAuth credentials lock at {lock_file_path}.')
                        time.sleep(0.05)
            except ImportError:
                acquired = True
        _lock_state.depth = 1
        yield
    finally:
        try:
            if acquired:
                try:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except ImportError:
                    try:
                        import msvcrt
                        try:
                            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                        except OSError:
                            pass
                    except ImportError:
                        pass
        finally:
            os.close(fd)
            _lock_state.depth = 0
_scraped_creds_cache: Dict[str, str] = {}

def _locate_gemini_cli_oauth_js() -> Optional[Path]:
    """Walk the user's gemini binary install to find its oauth2.js.

    Returns None if gemini isn't installed. Supports both the npm install
    (``node_modules/@google/gemini-cli-core/dist/**/code_assist/oauth2.js``)
    and the Homebrew ``bundle/`` layout.
    """
    import shutil
    gemini = shutil.which('gemini')
    if not gemini:
        return None
    try:
        real = Path(gemini).resolve()
    except OSError:
        return None
    search_dirs: list[Path] = []
    cur = real.parent
    for _ in range(8):
        search_dirs.append(cur)
        if (cur / 'node_modules').exists():
            search_dirs.append(cur / 'node_modules' / '@google' / 'gemini-cli-core')
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    for root in search_dirs:
        if not root.exists():
            continue
        candidates = [root / 'dist' / 'src' / 'code_assist' / 'oauth2.js', root / 'dist' / 'code_assist' / 'oauth2.js', root / 'src' / 'code_assist' / 'oauth2.js']
        for c in candidates:
            if c.exists():
                return c
        try:
            for path in root.rglob('oauth2.js'):
                return path
        except (OSError, ValueError):
            continue
    return None

def _scrape_client_credentials() -> Tuple[str, str]:
    """Extract client_id + client_secret from the local gemini-cli install."""
    if _scraped_creds_cache.get('resolved'):
        return (_scraped_creds_cache.get('client_id', ''), _scraped_creds_cache.get('client_secret', ''))
    oauth_js = _locate_gemini_cli_oauth_js()
    if oauth_js is None:
        _scraped_creds_cache['resolved'] = '1'
        return ('', '')
    try:
        content = oauth_js.read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        logger.debug('Failed to read oauth2.js at %s: %s', oauth_js, exc)
        _scraped_creds_cache['resolved'] = '1'
        return ('', '')
    cid_match = _CLIENT_ID_PATTERN.search(content) or _CLIENT_ID_SHAPE.search(content)
    cs_match = _CLIENT_SECRET_PATTERN.search(content) or _CLIENT_SECRET_SHAPE.search(content)
    client_id = cid_match.group(1) if cid_match else ''
    client_secret = cs_match.group(1) if cs_match else ''
    _scraped_creds_cache['client_id'] = client_id
    _scraped_creds_cache['client_secret'] = client_secret
    _scraped_creds_cache['resolved'] = '1'
    if client_id:
        logger.info('Scraped Gemini OAuth client from %s', oauth_js)
    return (client_id, client_secret)

def _get_client_id() -> str:
    env_val = (os.getenv(ENV_CLIENT_ID) or '').strip()
    if env_val:
        return env_val
    if _DEFAULT_CLIENT_ID:
        return _DEFAULT_CLIENT_ID
    scraped, _ = _scrape_client_credentials()
    return scraped

def _get_client_secret() -> str:
    env_val = (os.getenv(ENV_CLIENT_SECRET) or '').strip()
    if env_val:
        return env_val
    if _DEFAULT_CLIENT_SECRET:
        return _DEFAULT_CLIENT_SECRET
    _, scraped = _scrape_client_credentials()
    return scraped

@dataclass
class RefreshParts:
    refresh_token: str
    project_id: str = ''
    managed_project_id: str = ''

    @classmethod
    def parse(cls, packed: str) -> 'RefreshParts':
        if not packed:
            return cls(refresh_token='')
        parts = packed.split('|', 2)
        return cls(refresh_token=parts[0], project_id=parts[1] if len(parts) > 1 else '', managed_project_id=parts[2] if len(parts) > 2 else '')

    def format(self) -> str:
        if not self.refresh_token:
            return ''
        if not self.project_id and (not self.managed_project_id):
            return self.refresh_token
        return f'{self.refresh_token}|{self.project_id}|{self.managed_project_id}'

@dataclass
class GoogleCredentials:
    access_token: str
    refresh_token: str
    expires_ms: int
    email: str = ''
    project_id: str = ''
    managed_project_id: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {'refresh': RefreshParts(refresh_token=self.refresh_token, project_id=self.project_id, managed_project_id=self.managed_project_id).format(), 'access': self.access_token, 'expires': int(self.expires_ms), 'email': self.email}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoogleCredentials':
        refresh_packed = str(data.get('refresh', '') or '')
        parts = RefreshParts.parse(refresh_packed)
        return cls(access_token=str(data.get('access', '') or ''), refresh_token=parts.refresh_token, expires_ms=int(data.get('expires', 0) or 0), email=str(data.get('email', '') or ''), project_id=parts.project_id, managed_project_id=parts.managed_project_id)

    def expires_unix_seconds(self) -> float:
        return self.expires_ms / 1000.0

    def access_token_expired(self, skew_seconds: int=REFRESH_SKEW_SECONDS) -> bool:
        if not self.access_token or not self.expires_ms:
            return True
        return (time.time() + max(0, skew_seconds)) * 1000 >= self.expires_ms

def load_credentials() -> Optional[GoogleCredentials]:
    """Load credentials from disk. Returns None if missing or corrupt."""
    path = _credentials_path()
    if not path.exists():
        return None
    try:
        with _credentials_lock():
            raw = path.read_text(encoding='utf-8')
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError, IOError) as exc:
        logger.warning('Failed to read Google OAuth credentials at %s: %s', path, exc)
        return None
    if not isinstance(data, dict):
        return None
    creds = GoogleCredentials.from_dict(data)
    if not creds.access_token:
        return None
    return creds

def save_credentials(creds: GoogleCredentials) -> Path:
    """Atomically write creds to disk with 0o600 permissions."""
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 448)
    except OSError:
        pass
    payload = json.dumps(creds.to_dict(), indent=2, sort_keys=True) + '\n'
    with _credentials_lock():
        tmp_path = path.with_suffix(f'.tmp.{os.getpid()}.{secrets.token_hex(4)}')
        try:
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            atomic_replace(tmp_path, path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
    return path

def clear_credentials() -> None:
    """Remove the creds file. Idempotent."""
    path = _credentials_path()
    with _credentials_lock():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning('Failed to remove Google OAuth credentials at %s: %s', path, exc)

def _post_form(url: str, data: Dict[str, str], timeout: float) -> Dict[str, Any]:
    """POST x-www-form-urlencoded and return parsed JSON response."""
    body = urllib.parse.urlencode(data).encode('ascii')
    request = urllib.request.Request(url, data=body, method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode('utf-8', errors='replace')
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = ''
        try:
            detail = exc.read().decode('utf-8', errors='replace')
        except Exception:
            pass
        code = 'google_oauth_token_http_error'
        if 'invalid_grant' in detail.lower():
            code = 'google_oauth_invalid_grant'
        raise GoogleOAuthError(f'Google OAuth token endpoint returned HTTP {exc.code}: {detail or exc.reason}', code=code) from exc
    except urllib.error.URLError as exc:
        raise GoogleOAuthError(f'Google OAuth token request failed: {exc}', code='google_oauth_token_network_error') from exc

def refresh_access_token(refresh_token: str, *, client_id: Optional[str]=None, client_secret: Optional[str]=None, timeout: float=TOKEN_REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Refresh the access token."""
    if not refresh_token:
        raise GoogleOAuthError('Cannot refresh: refresh_token is empty. Re-run OAuth login.', code='google_oauth_refresh_token_missing')
    cid = client_id if client_id is not None else _get_client_id()
    csecret = client_secret if client_secret is not None else _get_client_secret()
    data = {'grant_type': 'refresh_token', 'refresh_token': refresh_token, 'client_id': cid}
    if csecret:
        data['client_secret'] = csecret
    return _post_form(TOKEN_ENDPOINT, data, timeout)
_refresh_inflight: Dict[str, threading.Event] = {}
_refresh_inflight_lock = threading.Lock()

def get_valid_access_token(*, force_refresh: bool=False) -> str:
    """Load creds, refreshing if near expiry, and return a valid bearer token.

    Dedupes concurrent refreshes by refresh_token. On ``invalid_grant``, the
    credential file is wiped and a ``google_oauth_invalid_grant`` error is raised
    (caller is expected to trigger a re-login flow).
    """
    creds = load_credentials()
    if creds is None:
        raise GoogleOAuthError('No Google OAuth credentials found. Run `hermes login --provider google-gemini-cli` first.', code='google_oauth_not_logged_in')
    if not force_refresh and (not creds.access_token_expired()):
        return creds.access_token
    rt = creds.refresh_token
    with _refresh_inflight_lock:
        event = _refresh_inflight.get(rt)
        if event is None:
            event = threading.Event()
            _refresh_inflight[rt] = event
            owner = True
        else:
            owner = False
    if not owner:
        event.wait(timeout=LOCK_TIMEOUT_SECONDS)
        fresh = load_credentials()
        if fresh is not None and (not fresh.access_token_expired()):
            return fresh.access_token
    try:
        try:
            resp = refresh_access_token(rt)
        except GoogleOAuthError as exc:
            if exc.code == 'google_oauth_invalid_grant':
                logger.warning('Google OAuth refresh token invalid (revoked/expired). Clearing credentials at %s — user must re-login.', _credentials_path())
                clear_credentials()
            raise
        new_access = str(resp.get('access_token', '') or '').strip()
        if not new_access:
            raise GoogleOAuthError('Refresh response did not include an access_token.', code='google_oauth_refresh_empty')
        new_refresh = str(resp.get('refresh_token', '') or '').strip() or creds.refresh_token
        expires_in = int(resp.get('expires_in', 0) or 0)
        creds.access_token = new_access
        creds.refresh_token = new_refresh
        creds.expires_ms = int((time.time() + max(60, expires_in)) * 1000)
        save_credentials(creds)
        return creds.access_token
    finally:
        if owner:
            with _refresh_inflight_lock:
                _refresh_inflight.pop(rt, None)
            event.set()

