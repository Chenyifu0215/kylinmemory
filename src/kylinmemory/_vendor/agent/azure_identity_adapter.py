"""Microsoft Entra ID adapter for Microsoft Foundry.

Provides keyless authentication for Microsoft Foundry deployments using the
`azure-identity` SDK's `DefaultAzureCredential` chain (env service principal
→ workload identity → managed identity → VS Code → Azure CLI → azd →
PowerShell → broker).

Architecture mirrors `agent/bedrock_adapter.py`:

* Lazy import. `azure-identity` is only loaded when ``model.auth_mode =
  entra_id`` is selected. Users who stick with `AZURE_FOUNDRY_API_KEY`
  never pay the import cost.
* SDK-callable contract. The public entry point ``build_token_provider``
  returns a zero-arg callable produced by ``get_bearer_token_provider`` —
  this is exactly the value Microsoft's documented sample plugs into
  ``OpenAI(api_key=token_provider, base_url=...)``. The OpenAI SDK calls
  it before every request, so token refresh is transparent.
* Three explicit consumer-side helpers (display / cache / http-bearer)
  rather than one generic "materialize" function — splitting them by
  purpose prevents accidental token-minting in logging paths or token
  leakage into cache keys / dashboard JSON.
* No persisted JWT. ``azure-identity`` caches in-process and (where
  available) in the OS keychain or ``~/.IdentityService``. Hermes does
  not duplicate that storage in ``auth.json``.

Reference: https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id

Requires: ``azure-identity`` (optional dependency — only needed when
``model.auth_mode = entra_id``).
"""
from __future__ import annotations
import functools
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
logger = logging.getLogger(__name__)
SCOPE_AI_AZURE_DEFAULT = 'https://ai.azure.com/.default'
_AZURE_IDENTITY_FEATURE = 'provider.azure_identity'

def has_azure_identity_installed() -> bool:
    """Return True if `azure-identity` can be imported right now.

    Cheap check — does not walk the credential chain.
    """
    try:
        import azure.identity
        return True
    except Exception:
        return False

def _require_azure_identity():
    """Import ``azure.identity``, lazy-installing it if allowed.

    Raises ``ImportError`` with a clear actionable message when the
    package is missing and lazy installs are disabled.
    """
    try:
        import azure.identity as _ai
        return _ai
    except ImportError:
        try:
            from kylinmemory._vendor.tools.lazy_deps import ensure, FeatureUnavailable
        except ImportError as exc:
            raise ImportError("The 'azure-identity' package is required for Azure AI Foundry Entra ID authentication. Install it with: pip install azure-identity") from exc
        try:
            ensure(_AZURE_IDENTITY_FEATURE, prompt=False)
        except FeatureUnavailable as exc:
            raise ImportError("The 'azure-identity' package is required for Azure AI Foundry Entra ID authentication. " + str(exc)) from exc
        import azure.identity as _ai
        return _ai

def reset_credential_cache() -> None:
    """Clear the cached ``DefaultAzureCredential``. Used by tests and
    profile switches.

    Defensive against tests that ``monkeypatch.setattr`` over
    ``build_credential`` with a plain (non-lru-cached) function — those
    won't expose ``cache_clear()`` until pytest reverts the patch.
    """
    cache_clear = getattr(build_credential, 'cache_clear', None)
    if callable(cache_clear):
        cache_clear()

@dataclass(frozen=True)
class EntraIdentityConfig:
    """Serializable Entra ID config.

    Captures the Hermes-managed Entra knobs we need outside Azure SDK
    environment configuration. Everything else
    (tenant ID, service principal secret, federated token file, sovereign
    cloud authority, etc.) flows through azure-identity's standard
    ``AZURE_*`` env vars — see the Bedrock pattern in
    ``kylin_agent_runtime_cli/runtime_provider.py:1310-1377`` for the analogous
    "let the SDK read env" approach.

    ``scope`` is Microsoft's documented Foundry inference audience. Almost
    everyone uses the default; sovereign-cloud / non-standard tenants can
    override via ``model.entra.scope``. Identity selection (user-assigned
    managed identity, workload identity, service principal, tenant, authority)
    stays in the standard Azure SDK env vars such as ``AZURE_CLIENT_ID``.

    ``exclude_interactive_browser`` is kept as an internal constructor knob
    so probes stay non-interactive by default. It is not written by the setup
    wizard.

    The dataclass is frozen so it's hashable for ``functools.lru_cache``
    keying, and serializable across multiprocessing boundaries (workers
    rebuild the credential inside their own process).
    """
    scope: str = SCOPE_AI_AZURE_DEFAULT
    exclude_interactive_browser: bool = True

    def __post_init__(self) -> None:
        scope = str(self.scope or '').strip() or SCOPE_AI_AZURE_DEFAULT
        object.__setattr__(self, 'scope', scope)

    def to_dict(self) -> Dict[str, Any]:
        return {'scope': self.scope, 'exclude_interactive_browser': self.exclude_interactive_browser}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], *, default_scope: Optional[str]=None) -> 'EntraIdentityConfig':
        data = data or {}
        scope = str(data.get('scope') or '').strip() or default_scope or SCOPE_AI_AZURE_DEFAULT
        exclude_browser = bool(data.get('exclude_interactive_browser', True))
        return cls(scope=scope, exclude_interactive_browser=exclude_browser)

def _build_default_credential(config: EntraIdentityConfig) -> Any:
    """Construct a ``DefaultAzureCredential`` for ``config``.

    Only Hermes-selected knobs are passed as kwargs. Everything else
    (tenant, service principal secret, federated token file, sovereign
    cloud authority, etc.) is read by ``azure-identity`` from the
    standard ``AZURE_*`` environment variables — see Microsoft's
    documented credential resolution chain. Users configure those in
    ``~/.kylin-agent-runtime/.env`` or the deployment environment.
    """
    ai = _require_azure_identity()
    kwargs: Dict[str, Any] = {}
    if not config.exclude_interactive_browser:
        kwargs['exclude_interactive_browser_credential'] = False
    return ai.DefaultAzureCredential(**kwargs)

@functools.lru_cache(maxsize=1)
def build_credential(config: EntraIdentityConfig) -> Any:
    """Return the cached ``DefaultAzureCredential`` for ``config``.

    Hermes processes use exactly one Entra config at a time (the
    ``model.entra.*`` block in config.yaml drives every aux task,
    subagent, and credential probe in the session). ``maxsize=1`` is
    intentional: it reflects the actual usage pattern and keeps the
    cache trivially small.

    ``EntraIdentityConfig`` is a frozen dataclass, so it's hashable and
    safe as an LRU-cache key. ``functools.lru_cache`` is thread-safe in
    CPython.

    If two distinct configs are ever passed (tests do this; production
    rarely), the LRU eviction handles it correctly — each call still
    returns a credential matching its config; only one is cached at a
    time. Use :func:`reset_credential_cache` to clear (e.g. in tests).
    """
    return _build_default_credential(config)

def build_token_provider(scope: Optional[str]=None, *, config: Optional[EntraIdentityConfig]=None, base_url: Optional[str]=None, exclude_interactive_browser: bool=True) -> Callable[[], str]:
    """Return a zero-arg callable that mints a fresh Entra bearer JWT.

    The returned callable is exactly what Microsoft's documented Foundry
    sample expects::

        from openai import OpenAI
        client = OpenAI(
            base_url="https://my-resource.openai.azure.com/openai/v1/",
            api_key=build_token_provider(),
        )

    Scope resolution order:
      1. ``config.scope`` when a config object is supplied
      2. explicit ``scope`` kwarg
      3. ``SCOPE_AI_AZURE_DEFAULT`` (Microsoft's documented Foundry scope)

    ``base_url`` is unused today and kept for back-compat. Tenant /
    service-principal / sovereign-cloud configuration flows through
    ``azure-identity``'s standard ``AZURE_*`` environment variables —
    see :func:`_build_default_credential` for the rationale.

    NOT serializable across process boundaries. For multiprocessing
    workers, serialize the ``EntraIdentityConfig`` and rebuild the
    provider inside the worker.
    """
    ai = _require_azure_identity()
    if config is None:
        config = EntraIdentityConfig(scope=scope or SCOPE_AI_AZURE_DEFAULT, exclude_interactive_browser=exclude_interactive_browser)
    credential = build_credential(config)
    return ai.get_bearer_token_provider(credential, config.scope)

def has_azure_identity_credentials(scope: Optional[str]=None, *, config: Optional[EntraIdentityConfig]=None, timeout_seconds: float=10.0, allow_install: bool=True, **overrides: Any) -> bool:
    """Best-effort probe: can `DefaultAzureCredential` mint a token now?

    Runs ``credential.get_token(scope)`` under a thread-based timeout so
    a slow token service can't hang the caller. Returns False on any
    error — never raises. Use for ``kylin-agent-runtime doctor`` /
    ``kylin-agent-runtime auth status`` / wizard preflight.

    ``allow_install``: when True (default) and ``azure-identity`` is not
    importable, the adapter triggers the standard lazy-install path
    (subject to ``security.allow_lazy_installs``) before probing. Set
    False to make this strictly an "is installed?" check — used on hot
    paths like CLI startup where we never want pip to run.

    NOT used by ``is_provider_configured()`` — that path is structural
    only (no token mint), so CLI startup doesn't pay this latency.
    """
    if not has_azure_identity_installed():
        if not allow_install:
            return False
        try:
            _require_azure_identity()
        except ImportError as exc:
            logger.debug('azure-identity lazy install unavailable: %s', exc)
            return False
    if config is None:
        effective_scope = (scope or '').strip() or SCOPE_AI_AZURE_DEFAULT
        config = EntraIdentityConfig(scope=effective_scope, **overrides)
    result = {'ok': False}

    def _probe() -> None:
        try:
            credential = build_credential(config)
            tok = credential.get_token(config.scope)
            result['ok'] = bool(getattr(tok, 'token', None))
        except Exception as exc:
            logger.debug('Entra credential probe failed: %s', exc)
            result['ok'] = False
    thread = threading.Thread(target=_probe, daemon=True)
    thread.start()
    thread.join(timeout=max(0.01, timeout_seconds))
    if thread.is_alive():
        logger.debug('Entra token service probe timed out after %ss', timeout_seconds)
        return False
    return bool(result.get('ok'))

def describe_active_credential(config: Optional[EntraIdentityConfig]=None, *, scope: Optional[str]=None, timeout_seconds: float=10.0, allow_install: bool=True, **overrides: Any) -> Dict[str, Any]:
    """Return diagnostic info about the active credential chain.

    Best-effort: runs ``get_token()`` and inspects what came back.
    Designed for ``kylin-agent-runtime doctor`` and the wizard preflight — never
    raises, returns ``{"ok": False, "error": ...}`` on failure.

    ``allow_install``: when True (default) and ``azure-identity`` is not
    importable, the adapter triggers the standard lazy-install path
    (subject to ``security.allow_lazy_installs``) before probing. The
    install failure is surfaced as the diagnostic error when it fails.
    Set False for hot CLI paths that should never trigger pip.

    ``azure-identity`` doesn't expose the winning inner credential as
    a public field, so we report a coarse picture (env vars present,
    token expiry, claims-derived tenant) rather than the credential
    class name. Users wanting the precise class can run with
    ``AZURE_LOG_LEVEL=DEBUG``.
    """
    info: Dict[str, Any] = {'ok': False}
    if not has_azure_identity_installed():
        if not allow_install:
            info['error'] = 'azure-identity not installed'
            info['hint'] = 'pip install azure-identity (or rely on lazy install at first use)'
            return info
        try:
            _require_azure_identity()
        except ImportError as exc:
            info['error'] = str(exc) or 'azure-identity not installed'
            info['hint'] = 'pip install azure-identity manually, or enable lazy installs (security.allow_lazy_installs: true in config.yaml).'
            return info
    if config is None:
        effective_scope = (scope or '').strip() or SCOPE_AI_AZURE_DEFAULT
        config = EntraIdentityConfig(scope=effective_scope, **overrides)
    info['scope'] = config.scope
    if os.environ.get('AZURE_TENANT_ID', '').strip():
        info['tenant_id_env'] = os.environ['AZURE_TENANT_ID'].strip()
    env_sources = []
    if os.environ.get('AZURE_FEDERATED_TOKEN_FILE', '').strip():
        env_sources.append('WorkloadIdentityCredential (AZURE_FEDERATED_TOKEN_FILE)')
    if os.environ.get('AZURE_CLIENT_ID', '').strip() and os.environ.get('AZURE_CLIENT_SECRET', '').strip() and os.environ.get('AZURE_TENANT_ID', '').strip():
        env_sources.append('EnvironmentCredential (client secret)')
    if os.environ.get('IDENTITY_ENDPOINT', '').strip() or os.environ.get('MSI_ENDPOINT', '').strip():
        env_sources.append('ManagedIdentityCredential (IDENTITY_ENDPOINT)')
    info['env_sources'] = env_sources
    result: Dict[str, Any] = {}

    def _probe() -> None:
        try:
            credential = build_credential(config)
            tok = credential.get_token(config.scope)
            result['token'] = tok
        except Exception as exc:
            result['error'] = str(exc)
    thread = threading.Thread(target=_probe, daemon=True)
    thread.start()
    thread.join(timeout=max(0.01, timeout_seconds))
    if thread.is_alive():
        info['error'] = f'Token probe timed out after {timeout_seconds:.0f}s'
        info['hint'] = 'DefaultAzureCredential can be slow when the token service is unreachable or when az login state is stale. Try `az login` or set AZURE_CLIENT_ID / AZURE_TENANT_ID / AZURE_CLIENT_SECRET.'
        return info
    if 'error' in result:
        info['error'] = result['error']
        return info
    token = result.get('token')
    if token is None:
        info['error'] = 'credential chain exhausted'
        return info
    info['ok'] = True
    info['expires_on'] = getattr(token, 'expires_on', None)
    return info

def is_token_provider(value: Any) -> bool:
    """Return True when ``value`` is a callable Entra token provider.

    Used at the seams where a consumer must decide between
    string-API-key semantics and bearer-callable semantics.
    """
    return callable(value) and (not isinstance(value, str))

def materialize_bearer_for_http(value: Any) -> str:
    """Return a fresh Bearer JWT for a manual HTTP request.

    Only call this at sites that must construct an ``Authorization``
    header outside the OpenAI SDK (e.g. ``kylin_agent_runtime_cli/azure_detect.py``).
    Calls the callable exactly once and returns the resulting token.

    **Anthropic SDK integration:** the Anthropic Python SDK does not
    accept a ``Callable[[], str]`` for ``auth_token``. Instead,
    :func:`build_bearer_http_client` returns an ``httpx.Client`` whose
    request event hook calls this function and rewrites the
    ``Authorization`` header per request — and that client is passed to
    the Anthropic SDK via ``http_client=...``. See
    :func:`agent.anthropic_adapter.build_anthropic_client` for the
    consumer.

    Raises ``ValueError`` if ``value`` is not a callable token provider
    or non-empty string.
    """
    if is_token_provider(value):
        token = value()
        if not isinstance(token, str) or not token:
            raise ValueError('token provider returned empty value')
        return token
    if isinstance(value, str) and value:
        return value
    raise ValueError('no usable api_key / token provider')

def build_bearer_http_client(token_provider: Callable[[], str], **httpx_kwargs: Any) -> Any:
    """Return an ``httpx.Client`` that mints a fresh Entra bearer JWT
    per outbound request.

    The Anthropic SDK (≤ 0.86.0 at the time of writing) stores
    ``api_key`` / ``auth_token`` as static strings and computes the
    ``Authorization`` header at construction time. To get per-request
    token refresh (the Microsoft-recommended Foundry pattern for
    callable bearer providers), we install an httpx ``request`` event
    hook on a custom client and pass that client to the SDK via
    ``http_client=...``. The hook:

      1. Calls :func:`materialize_bearer_for_http` to mint a fresh JWT
         (azure-identity caches internally — this is cheap when the
         cached token is still valid).
      2. Strips any pre-set ``Authorization`` / ``api-key`` /
         ``x-api-key`` headers the SDK may have added (avoids
         conflicting auth values).
      3. Sets ``Authorization: Bearer <fresh-jwt>``.

    ``token_provider`` must be a zero-arg callable returning a string —
    typically the result of :func:`build_token_provider`.

    ``httpx_kwargs`` are forwarded verbatim to ``httpx.Client(...)`` so
    callers can attach a ``timeout``, ``transport``, ``proxy``, etc.

    Raises ``ImportError`` if ``httpx`` is not installed (it is a
    transitive dependency of both ``openai`` and ``anthropic`` SDKs, so
    in practice always available when this helper is reached).
    """
    if not is_token_provider(token_provider):
        raise ValueError('build_bearer_http_client requires a zero-arg callable token provider')
    try:
        import httpx
    except ImportError as exc:
        raise ImportError('httpx is required for Entra ID bearer auth on Microsoft Foundry Anthropic-style endpoints. It is normally a transitive dependency of the openai/anthropic SDKs.') from exc

    def _inject_bearer(request: 'httpx.Request') -> None:
        try:
            token = materialize_bearer_for_http(token_provider)
        except ValueError as exc:
            logger.warning('Bearer hook: Entra ID token provider returned empty (%s) — stripping Authorization headers. Azure will respond 401. Run `kylin-agent-runtime doctor` or `az login` to recover.', exc)
            for header_name in ('Authorization', 'authorization', 'Api-Key', 'api-key', 'X-Api-Key', 'x-api-key'):
                request.headers.pop(header_name, None)
            return
        for header_name in ('Authorization', 'authorization', 'Api-Key', 'api-key', 'X-Api-Key', 'x-api-key'):
            request.headers.pop(header_name, None)
        request.headers['Authorization'] = f'Bearer {token}'
    return httpx.Client(event_hooks={'request': [_inject_bearer]}, **httpx_kwargs)
__all__ = ['EntraIdentityConfig', 'SCOPE_AI_AZURE_DEFAULT', 'build_bearer_http_client', 'build_credential', 'build_token_provider', 'describe_active_credential', 'has_azure_identity_credentials', 'has_azure_identity_installed', 'is_token_provider', 'materialize_bearer_for_http', 'reset_credential_cache']

