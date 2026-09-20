import os
import re
_SENSITIVE_QUERY_PARAMS = frozenset({'access_token', 'refresh_token', 'id_token', 'token', 'api_key', 'apikey', 'client_secret', 'password', 'auth', 'jwt', 'session', 'secret', 'key', 'code', 'signature', 'x-amz-signature'})
_REDACT_ENABLED = os.getenv('HERMES_REDACT_SECRETS', 'true').lower() in {'1', 'true', 'yes', 'on'}
_PREFIX_PATTERNS = ['sk-[A-Za-z0-9_-]{10,}', 'ghp_[A-Za-z0-9]{10,}', 'github_pat_[A-Za-z0-9_]{10,}', 'gho_[A-Za-z0-9]{10,}', 'ghu_[A-Za-z0-9]{10,}', 'ghs_[A-Za-z0-9]{10,}', 'ghr_[A-Za-z0-9]{10,}', 'xox[baprs]-[A-Za-z0-9-]{10,}', 'AIza[A-Za-z0-9_-]{30,}', 'pplx-[A-Za-z0-9]{10,}', 'fal_[A-Za-z0-9_-]{10,}', 'fc-[A-Za-z0-9]{10,}', 'bb_live_[A-Za-z0-9_-]{10,}', 'gAAAA[A-Za-z0-9_=-]{20,}', 'AKIA[A-Z0-9]{16}', 'sk_live_[A-Za-z0-9]{10,}', 'sk_test_[A-Za-z0-9]{10,}', 'rk_live_[A-Za-z0-9]{10,}', 'SG\\.[A-Za-z0-9_-]{10,}', 'hf_[A-Za-z0-9]{10,}', 'r8_[A-Za-z0-9]{10,}', 'npm_[A-Za-z0-9]{10,}', 'pypi-[A-Za-z0-9_-]{10,}', 'dop_v1_[A-Za-z0-9]{10,}', 'doo_v1_[A-Za-z0-9]{10,}', 'am_[A-Za-z0-9_-]{10,}', 'sk_[A-Za-z0-9_]{10,}', 'tvly-[A-Za-z0-9]{10,}', 'exa_[A-Za-z0-9]{10,}', 'gsk_[A-Za-z0-9]{10,}', 'syt_[A-Za-z0-9]{10,}', 'retaindb_[A-Za-z0-9]{10,}', 'hsk-[A-Za-z0-9]{10,}', 'mem0_[A-Za-z0-9]{10,}', 'brv_[A-Za-z0-9]{10,}', 'xai-[A-Za-z0-9]{30,}']
_SECRET_ENV_NAMES = '(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)'
_ENV_ASSIGN_RE = re.compile(f"""([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\\s*=\\s*(['\\"]?)(\\S+)\\2""")
_JSON_KEY_NAMES = '(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer|secret_value|raw_secret|secret_input|key_material)'
_JSON_FIELD_RE = re.compile(f'("{_JSON_KEY_NAMES}")\\s*:\\s*"([^"]+)"', re.IGNORECASE)
_AUTH_HEADER_RE = re.compile('(Authorization:\\s*Bearer\\s+)(\\S+)', re.IGNORECASE)
_TELEGRAM_RE = re.compile('(bot)?(\\d{8,}):([-A-Za-z0-9_]{30,})')
_PRIVATE_KEY_RE = re.compile('-----BEGIN[A-Z ]*PRIVATE KEY-----[\\s\\S]*?-----END[A-Z ]*PRIVATE KEY-----')
_DB_CONNSTR_RE = re.compile('((?:postgres(?:ql)?|mysql|mongodb(?:\\+srv)?|redis|amqp)://[^:]+:)([^@]+)(@)', re.IGNORECASE)
_JWT_RE = re.compile('eyJ[A-Za-z0-9_-]{10,}(?:\\.[A-Za-z0-9_=-]{4,}){0,2}')
_DISCORD_MENTION_RE = re.compile('<@!?(\\d{17,20})>')
_SIGNAL_PHONE_RE = re.compile('(\\+[1-9]\\d{6,14})(?![A-Za-z0-9])')
_URL_WITH_QUERY_RE = re.compile('(https?|wss?|ftp)://([^\\s/?#]+)([^\\s?#]*)\\?([^\\s#]+)(#\\S*)?')
_URL_USERINFO_RE = re.compile('(https?|wss?|ftp)://([^/\\s:@]+):([^/\\s@]+)@')
_FORM_BODY_RE = re.compile('^[A-Za-z_][A-Za-z0-9_.-]*=[^&\\s]*(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\\s]*)+$')
_PREFIX_RE = re.compile('(?<![A-Za-z0-9_-])(' + '|'.join(_PREFIX_PATTERNS) + ')(?![A-Za-z0-9_-])')

def mask_secret(value: str, *, head: int=4, tail: int=4, floor: int=12, placeholder: str='***', empty: str='') -> str:
    """Mask a secret for display, preserving ``head`` and ``tail`` characters.

    Canonical helper for display-time redaction across Hermes — used by
    ``kylin-agent-runtime config``, ``kylin-agent-runtime status``, ``hermes dump``, and anywhere
    a secret needs to be shown truncated for debuggability while still
    keeping the bulk hidden.

    Args:
        value:       The secret to mask. ``None``/empty returns ``empty``.
        head:        Leading characters to preserve. Default 4.
        tail:        Trailing characters to preserve. Default 4.
        floor:       Values shorter than ``head + tail + floor_margin`` are
                     fully masked (returns ``placeholder``). Default 12 —
                     matches the existing config/status/dump convention.
        placeholder: Value returned for too-short inputs. Default ``"***"``.
        empty:       Value returned when ``value`` is falsy (None, ""). The
                     caller can override this to e.g. ``color("(not set)",
                     Colors.DIM)`` for user-facing display.

    Examples:
        >>> mask_secret("sk-proj-abcdef1234567890")
        'sk-p...7890'
        >>> mask_secret("short")                         # fully masked
        '***'
        >>> mask_secret("")                              # empty default
        ''
        >>> mask_secret("", empty="(not set)")           # empty override
        '(not set)'
        >>> mask_secret("long-token", head=6, tail=4, floor=18)
        '***'
    """
    if not value:
        return empty
    if len(value) < floor:
        return placeholder
    return f'{value[:head]}...{value[-tail:]}'

def _mask_token(token: str) -> str:
    """Mask a log token — conservative 18-char floor, preserves 6 prefix / 4 suffix."""
    if not token:
        return '***'
    return mask_secret(token, head=6, tail=4, floor=18)

def _redact_query_string(query: str) -> str:
    """Redact sensitive parameter values in a URL query string.

    Handles `k=v&k=v` format. Sensitive keys (case-insensitive) have values
    replaced with `***`. Non-sensitive keys pass through unchanged.
    Empty or malformed pairs are preserved as-is.
    """
    if not query:
        return query
    parts = []
    for pair in query.split('&'):
        if '=' not in pair:
            parts.append(pair)
            continue
        key, _, value = pair.partition('=')
        if key.lower() in _SENSITIVE_QUERY_PARAMS:
            parts.append(f'{key}=***')
        else:
            parts.append(pair)
    return '&'.join(parts)

def _redact_url_query_params(text: str) -> str:
    """Scan text for URLs with query strings and redact sensitive params.

    Catches opaque tokens that don't match vendor prefix regexes, e.g.
    `https://example.com/cb?code=ABC123&state=xyz` → `...?code=***&state=xyz`.
    """

    def _sub(m: re.Match) -> str:
        scheme = m.group(1)
        authority = m.group(2)
        path = m.group(3)
        query = _redact_query_string(m.group(4))
        fragment = m.group(5) or ''
        return f'{scheme}://{authority}{path}?{query}{fragment}'
    return _URL_WITH_QUERY_RE.sub(_sub, text)

def _redact_url_userinfo(text: str) -> str:
    """Strip `user:password@` from HTTP/WS/FTP URLs.

    DB protocols (postgres, mysql, mongodb, redis, amqp) are handled
    separately by `_DB_CONNSTR_RE`.
    """
    return _URL_USERINFO_RE.sub(lambda m: f'{m.group(1)}://{m.group(2)}:***@', text)

def _redact_form_body(text: str) -> str:
    """Redact sensitive values in a form-urlencoded body.

    Only applies when the entire input looks like a pure form body
    (k=v&k=v with no newlines, no other text). Single-line non-form
    text passes through unchanged. This is a conservative pass — the
    `_redact_url_query_params` function handles embedded query strings.
    """
    if not text or '\n' in text or '&' not in text:
        return text
    if not _FORM_BODY_RE.match(text.strip()):
        return text
    return _redact_query_string(text.strip())

def redact_sensitive_text(text: str, *, force: bool=False, code_file: bool=False) -> str:
    """Apply all redaction patterns to a block of text.

    Safe to call on any string -- non-matching text passes through unchanged.
    Disabled by default — enable via security.redact_secrets: true in config.yaml.
    Set force=True for safety boundaries that must never return raw secrets
    regardless of the user's global logging redaction preference.

    Set code_file=True to skip the ENV-assignment and JSON-field regex
    patterns when the text is known to be source code (e.g. MAX_TOKENS=***
    constants, "apiKey": "test" fixtures). Prefix patterns, auth headers,
    private keys, DB connstrings, JWTs, and URL secrets are still redacted.

    Performance: each regex pattern is gated behind a cheap substring
    pre-check (e.g. ``"=" in text`` for ENV assignments, ``"://" in text``
    for URLs, ``"eyJ" in text`` for JWTs). On a typical hermes log line
    (no secrets) this drops the 13-pattern scan from ~5.6us to ~1.8us per
    record (-68%). The pre-checks are conservative — false positives
    still run the full regex, which then doesn't match. False negatives
    are impossible because every regex requires the gated substring to
    match.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    if not (force or _REDACT_ENABLED):
        return text
    if _has_known_prefix_substring(text):
        text = _PREFIX_RE.sub(lambda m: _mask_token(m.group(1)), text)
    if not code_file:
        if '=' in text:

            def _redact_env(m):
                name, quote, value = (m.group(1), m.group(2), m.group(3))
                return f'{name}={quote}{_mask_token(value)}{quote}'
            text = _ENV_ASSIGN_RE.sub(_redact_env, text)
        if ':' in text and '"' in text:

            def _redact_json(m):
                key, value = (m.group(1), m.group(2))
                return f'{key}: "{_mask_token(value)}"'
            text = _JSON_FIELD_RE.sub(_redact_json, text)
    if 'uthorization' in text or 'UTHORIZATION' in text:
        text = _AUTH_HEADER_RE.sub(lambda m: m.group(1) + _mask_token(m.group(2)), text)
    if ':' in text:

        def _redact_telegram(m):
            prefix = m.group(1) or ''
            digits = m.group(2)
            return f'{prefix}{digits}:***'
        text = _TELEGRAM_RE.sub(_redact_telegram, text)
    if 'BEGIN' in text and '-----' in text:
        text = _PRIVATE_KEY_RE.sub('[REDACTED PRIVATE KEY]', text)
    if '://' in text:
        text = _DB_CONNSTR_RE.sub(lambda m: f'{m.group(1)}***{m.group(3)}', text)
    if 'eyJ' in text:
        text = _JWT_RE.sub(lambda m: _mask_token(m.group(0)), text)
    if '://' in text:
        text = _redact_url_userinfo(text)
        if '?' in text:
            text = _redact_url_query_params(text)
    if '&' in text and '=' in text:
        text = _redact_form_body(text)
    if '<@' in text:
        text = _DISCORD_MENTION_RE.sub(lambda m: f"<@{('!' if '!' in m.group(0) else '')}***>", text)
    if '+' in text:

        def _redact_phone(m):
            phone = m.group(1)
            if len(phone) <= 8:
                return phone[:2] + '****' + phone[-2:]
            return phone[:4] + '****' + phone[-4:]
        text = _SIGNAL_PHONE_RE.sub(_redact_phone, text)
    return text

def _extract_literal_prefix(pattern: str) -> str:
    """Return the leading literal characters of a regex pattern.

    Stops at the first regex metacharacter (``[``, ``(``, ``\\``, ``.``,
    ``?``, ``*``, ``+``, ``|``, ``{``, ``^``, ``$``).  Returns the literal
    that any match of the pattern MUST contain as a substring, so the
    pre-screen never produces false negatives.
    """
    meta = '[(\\.?*+|{^$'
    for i, ch in enumerate(pattern):
        if ch in meta:
            return pattern[:i]
    return pattern
_PREFIX_SUBSTRINGS = tuple((_extract_literal_prefix(p) for p in _PREFIX_PATTERNS))

def _has_known_prefix_substring(text: str) -> bool:
    """Return True if ``text`` contains any known credential prefix substring.

    Used as a cheap pre-check before invoking the expensive ``_PREFIX_RE``.
    """
    return any((p in text for p in _PREFIX_SUBSTRINGS))

