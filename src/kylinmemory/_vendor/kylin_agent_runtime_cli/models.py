from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
import time
from typing import Any, NamedTuple, Optional
from kylinmemory._vendor.kylin_agent_runtime_cli import __version__ as _HERMES_VERSION
_HERMES_USER_AGENT = f'hermes-cli/{_HERMES_VERSION}'
COPILOT_BASE_URL = 'https://api.githubcopilot.com'
COPILOT_MODELS_URL = f'{COPILOT_BASE_URL}/models'
COPILOT_EDITOR_VERSION = 'vscode/1.104.1'
COPILOT_REASONING_EFFORTS_GPT5 = ['minimal', 'low', 'medium', 'high']
COPILOT_REASONING_EFFORTS_O_SERIES = ['low', 'medium', 'high']
OPENROUTER_MODELS: list[tuple[str, str]] = [('anthropic/claude-opus-4.7', ''), ('anthropic/claude-opus-4.6', ''), ('anthropic/claude-sonnet-4.6', ''), ('moonshotai/kimi-k2.6', 'recommended'), ('openrouter/pareto-code', 'auto-routes to cheapest coder meeting openrouter.min_coding_score'), ('qwen/qwen3.6-plus', ''), ('anthropic/claude-haiku-4.5', ''), ('openai/gpt-5.5', ''), ('openai/gpt-5.5-pro', ''), ('openai/gpt-5.4-mini', ''), ('openai/gpt-5.4-nano', ''), ('openai/gpt-5.3-codex', ''), ('xiaomi/mimo-v2.5-pro', ''), ('tencent/hy3-preview', ''), ('google/gemini-3-pro-image-preview', ''), ('google/gemini-3-flash-preview', ''), ('google/gemini-3.1-pro-preview', ''), ('google/gemini-3.1-flash-lite-preview', ''), ('qwen/qwen3.6-35b-a3b', ''), ('stepfun/step-3.5-flash', ''), ('minimax/minimax-m2.7', ''), ('z-ai/glm-5.1', ''), ('x-ai/grok-4.20', ''), ('x-ai/grok-4.3', ''), ('nvidia/nemotron-3-super-120b-a12b', ''), ('deepseek/deepseek-v4-pro', ''), ('openrouter/elephant-alpha', 'free'), ('openrouter/owl-alpha', 'free'), ('tencent/hy3-preview:free', 'free'), ('nvidia/nemotron-3-super-120b-a12b:free', 'free'), ('inclusionai/ring-2.6-1t:free', 'free')]
_openrouter_catalog_cache: list[tuple[str, str]] | None = None

def _codex_curated_models() -> list[str]:
    """Derive the openai-codex curated list from codex_models.py.

    Single source of truth: DEFAULT_CODEX_MODELS + forward-compat synthesis.
    This keeps the gateway /model picker in sync with the CLI `kylin-agent-runtime model`
    flow without maintaining a separate static list.
    """
    from kylinmemory._vendor.kylin_agent_runtime_cli.codex_models import DEFAULT_CODEX_MODELS, _add_forward_compat_models
    return _add_forward_compat_models(list(DEFAULT_CODEX_MODELS))
_XAI_STATIC_FALLBACK: list[str] = ['grok-4.3', 'grok-4.20-0309-reasoning', 'grok-4.20-0309-non-reasoning', 'grok-4.20-multi-agent-0309']
_XAI_TOP_MODEL = 'grok-4.3'

def _xai_promote_top(ids: list[str]) -> list[str]:
    """Pin the headline xAI model to the top of the curated list."""
    if _XAI_TOP_MODEL in ids:
        return [_XAI_TOP_MODEL] + [m for m in ids if m != _XAI_TOP_MODEL]
    return ids

def _xai_curated_models() -> list[str]:
    """Derive the xAI-direct curated list from models.dev disk cache.

    Reads $HERMES_HOME/models_dev_cache.json directly (no network) so this
    runs at import time without blocking. Falls back to ``_XAI_STATIC_FALLBACK``
    when the cache is empty or unreadable. Hermes refreshes the cache from
    https://models.dev/api.json on normal use, so this list self-heals as
    xAI renames models.

    Mirrors ``_codex_curated_models()``'s role for openai-codex.
    """
    try:
        from kylinmemory._vendor.agent.models_dev import _load_disk_cache
        data = _load_disk_cache()
        xai = data.get('xai') if isinstance(data, dict) else None
        models = xai.get('models') if isinstance(xai, dict) else None
        if isinstance(models, dict) and models:
            ids = [mid for mid in models.keys() if isinstance(mid, str)]
            if ids:
                return _xai_promote_top(sorted(ids))
    except Exception:
        pass
    return list(_XAI_STATIC_FALLBACK)
_PROVIDER_MODELS: dict[str, list[str]] = {'nous': ['anthropic/claude-opus-4.7', 'anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6', 'moonshotai/kimi-k2.6', 'qwen/qwen3.6-plus', 'anthropic/claude-haiku-4.5', 'openai/gpt-5.5', 'openai/gpt-5.5-pro', 'openai/gpt-5.4-mini', 'openai/gpt-5.4-nano', 'openai/gpt-5.3-codex', 'xiaomi/mimo-v2.5-pro', 'tencent/hy3-preview', 'google/gemini-3-pro-preview', 'google/gemini-3-flash-preview', 'google/gemini-3.1-pro-preview', 'google/gemini-3.1-flash-lite-preview', 'qwen/qwen3.6-35b-a3b', 'stepfun/step-3.5-flash', 'minimax/minimax-m2.7', 'z-ai/glm-5.1', 'x-ai/grok-4.3', 'nvidia/nemotron-3-super-120b-a12b', 'deepseek/deepseek-v4-pro'], 'openai': ['gpt-5.4', 'gpt-5.4-mini', 'gpt-5-mini', 'gpt-5.3-codex', 'gpt-5.2-codex', 'gpt-4.1', 'gpt-4o', 'gpt-4o-mini'], 'openai-codex': _codex_curated_models(), 'xai-oauth': _xai_curated_models(), 'copilot-acp': ['copilot-acp'], 'copilot': ['gpt-5.4', 'gpt-5.4-mini', 'gpt-5-mini', 'gpt-5.3-codex', 'gpt-5.2-codex', 'gpt-4.1', 'gpt-4o', 'gpt-4o-mini', 'claude-sonnet-4.6', 'claude-sonnet-4', 'claude-sonnet-4.5', 'claude-haiku-4.5', 'gemini-3.1-pro-preview', 'gemini-3-pro-preview', 'gemini-3-flash-preview', 'gemini-2.5-pro'], 'gemini': ['gemini-3.1-pro-preview', 'gemini-3-pro-preview', 'gemini-3-flash-preview', 'gemini-3.1-flash-lite-preview'], 'google-gemini-cli': ['gemini-3.1-pro-preview', 'gemini-3-pro-preview', 'gemini-3-flash-preview'], 'zai': ['glm-5.1', 'glm-5', 'glm-5v-turbo', 'glm-5-turbo', 'glm-4.7', 'glm-4.5', 'glm-4.5-flash'], 'xai': _xai_curated_models(), 'nvidia': ['nvidia/nemotron-3-super-120b-a12b', 'nvidia/nemotron-3-nano-30b-a3b', 'nvidia/llama-3.3-nemotron-super-49b-v1.5', 'qwen/qwen3.5-397b-a17b', 'deepseek-ai/deepseek-v3.2', 'moonshotai/kimi-k2.6', 'minimaxai/minimax-m2.5', 'z-ai/glm5', 'openai/gpt-oss-120b'], 'kimi-coding': ['kimi-k2.6', 'kimi-k2.5', 'kimi-for-coding', 'kimi-k2-thinking', 'kimi-k2-thinking-turbo', 'kimi-k2-turbo-preview', 'kimi-k2-0905-preview'], 'kimi-coding-cn': ['kimi-k2.6', 'kimi-k2.5', 'kimi-k2-thinking', 'kimi-k2-turbo-preview', 'kimi-k2-0905-preview'], 'stepfun': ['step-3.5-flash', 'step-3.5-flash-2603'], 'moonshot': ['kimi-k2.6', 'kimi-k2.5', 'kimi-k2-thinking', 'kimi-k2-turbo-preview', 'kimi-k2-0905-preview'], 'minimax': ['MiniMax-M2.7', 'MiniMax-M2.5', 'MiniMax-M2.1', 'MiniMax-M2'], 'minimax-oauth': ['MiniMax-M2.7', 'MiniMax-M2.7-highspeed'], 'minimax-cn': ['MiniMax-M2.7', 'MiniMax-M2.5', 'MiniMax-M2.1', 'MiniMax-M2'], 'anthropic': ['claude-opus-4-7', 'claude-opus-4-6', 'claude-sonnet-4-6', 'claude-opus-4-5-20251101', 'claude-sonnet-4-5-20250929', 'claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001'], 'deepseek': ['deepseek-v4-pro', 'deepseek-v4-flash', 'deepseek-chat', 'deepseek-reasoner'], 'xiaomi': ['mimo-v2.5-pro', 'mimo-v2.5', 'mimo-v2-pro', 'mimo-v2-omni', 'mimo-v2-flash'], 'tencent-tokenhub': ['hy3-preview'], 'arcee': ['trinity-large-thinking', 'trinity-large-preview', 'trinity-mini'], 'gmi': ['zai-org/GLM-5.1-FP8', 'deepseek-ai/DeepSeek-V3.2', 'moonshotai/Kimi-K2.5', 'google/gemini-3.1-flash-lite-preview', 'anthropic/claude-sonnet-4.6', 'openai/gpt-5.4'], 'opencode-zen': ['kimi-k2.5', 'gpt-5.4-pro', 'gpt-5.4', 'gpt-5.3-codex', 'gpt-5.2', 'gpt-5.2-codex', 'gpt-5.1', 'gpt-5.1-codex', 'gpt-5.1-codex-max', 'gpt-5.1-codex-mini', 'gpt-5', 'gpt-5-codex', 'gpt-5-nano', 'claude-opus-4-6', 'claude-opus-4-5', 'claude-opus-4-1', 'claude-sonnet-4-6', 'claude-sonnet-4-5', 'claude-sonnet-4', 'claude-haiku-4-5', 'claude-3-5-haiku', 'gemini-3.1-pro', 'gemini-3-pro', 'gemini-3-flash', 'minimax-m2.7', 'minimax-m2.5', 'minimax-m2.5-free', 'minimax-m2.1', 'glm-5', 'glm-4.7', 'glm-4.6', 'kimi-k2-thinking', 'kimi-k2', 'qwen3-coder', 'big-pickle'], 'opencode-go': ['kimi-k2.6', 'kimi-k2.5', 'glm-5.1', 'glm-5', 'mimo-v2.5-pro', 'mimo-v2.5', 'mimo-v2-pro', 'mimo-v2-omni', 'minimax-m2.7', 'minimax-m2.5', 'qwen3.6-plus', 'qwen3.5-plus'], 'kilocode': ['anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6', 'openai/gpt-5.4', 'google/gemini-3-pro-preview', 'google/gemini-3-flash-preview'], 'alibaba': ['qwen3.6-plus', 'kimi-k2.5', 'qwen3.5-plus', 'qwen3-coder-plus', 'qwen3-coder-next', 'glm-5', 'glm-4.7', 'MiniMax-M2.5'], 'alibaba-coding-plan': ['qwen3.6-plus', 'qwen3.5-plus', 'qwen3-coder-plus', 'qwen3-coder-next', 'kimi-k2.5', 'glm-5', 'glm-4.7', 'MiniMax-M2.5'], 'huggingface': ['moonshotai/Kimi-K2.5', 'Qwen/Qwen3.5-397B-A17B', 'Qwen/Qwen3.5-35B-A3B', 'deepseek-ai/DeepSeek-V3.2', 'MiniMaxAI/MiniMax-M2.5', 'zai-org/GLM-5', 'XiaomiMiMo/MiMo-V2-Flash', 'moonshotai/Kimi-K2-Thinking', 'moonshotai/Kimi-K2.6'], 'bedrock': ['us.anthropic.claude-sonnet-4-6', 'us.anthropic.claude-opus-4-6-v1', 'us.anthropic.claude-haiku-4-5-20251001-v1:0', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0', 'us.amazon.nova-pro-v1:0', 'us.amazon.nova-lite-v1:0', 'us.amazon.nova-micro-v1:0', 'deepseek.v3.2', 'us.meta.llama4-maverick-17b-instruct-v1:0', 'us.meta.llama4-scout-17b-instruct-v1:0'], 'azure-foundry': [], 'novita': ['moonshotai/kimi-k2.5', 'minimax/minimax-m2.7', 'zai-org/glm-5', 'deepseek/deepseek-v3-0324', 'deepseek/deepseek-r1-0528', 'qwen/qwen3-235b-a22b-fp8']}

def _is_model_free(model_id: str, pricing: dict[str, dict[str, str]]) -> bool:
    """Return True if *model_id* has zero-cost prompt AND completion pricing."""
    p = pricing.get(model_id)
    if not p:
        return False
    try:
        return float(p.get('prompt', '1')) == 0 and float(p.get('completion', '1')) == 0
    except (TypeError, ValueError):
        return False

def fetch_nous_account_tier(access_token: str, portal_base_url: str='') -> dict[str, Any]:
    """Fetch the user's Nous Portal account/subscription info.

    Calls ``<portal>/api/oauth/account`` with the OAuth access token.

    Returns the parsed JSON dict on success, e.g.::

        {
            "subscription": {
                "plan": "Plus",
                "tier": 2,
                "monthly_charge": 20,
                "credits_remaining": 1686.60,
                ...
            },
            ...
        }

    Returns an empty dict on any failure (network, auth, parse).
    """
    base = (portal_base_url or 'https://portal.nousresearch.com').rstrip('/')
    url = f'{base}/api/oauth/account'
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}

def is_nous_free_tier(account_info: dict[str, Any]) -> bool:
    """Return True if the account info indicates a free (unpaid) tier.

    Checks ``subscription.monthly_charge == 0``.  Returns False when
    the field is missing or unparseable (assumes paid — don't block users).
    """
    sub = account_info.get('subscription')
    if not isinstance(sub, dict):
        return False
    charge = sub.get('monthly_charge')
    if charge is None:
        return False
    try:
        return float(charge) == 0
    except (TypeError, ValueError):
        return False

def partition_nous_models_by_tier(model_ids: list[str], pricing: dict[str, dict[str, str]], free_tier: bool) -> tuple[list[str], list[str]]:
    """Split Nous models into (selectable, unavailable) based on user tier.

    For paid-tier users: all models are selectable, none unavailable.

    For free-tier users: only free models are selectable; paid models
    are returned as unavailable (shown grayed out in the menu).
    """
    if not free_tier:
        return (model_ids, [])
    if not pricing:
        return (model_ids, [])
    selectable: list[str] = []
    unavailable: list[str] = []
    for mid in model_ids:
        if _is_model_free(mid, pricing):
            selectable.append(mid)
        else:
            unavailable.append(mid)
    return (selectable, unavailable)

def union_with_portal_free_recommendations(curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str='', *, force_refresh: bool=False) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Augment curated list + pricing with the Portal's ``freeRecommendedModels``.

    The Portal's ``/api/nous/recommended-models`` endpoint advertises which
    models are free *right now* — independent of what the in-repo
    ``_PROVIDER_MODELS["nous"]`` list happens to contain or whether the
    docs-hosted catalog manifest has been rebuilt since the last release.

    For free-tier users this is the source of truth: any model the Portal
    flags as free should be selectable, even if the user is running an
    older Hermes that doesn't ship that model in its hardcoded curated
    list.  This function returns an augmented ``(model_ids, pricing)``
    pair where:

    * Portal free recommendations missing from ``curated_ids`` are
      appended at the front (so the picker shows them first).
    * ``pricing`` gets a synthetic ``{"prompt": "0", "completion": "0"}``
      entry for any free recommendation missing from the live pricing
      map, so :func:`partition_nous_models_by_tier` keeps it.

    Failures (network, parse, missing field) are silent and degrade to
    returning the inputs unchanged.
    """
    try:
        payload = fetch_nous_recommended_models(portal_base_url, force_refresh=force_refresh)
    except Exception:
        return (list(curated_ids), dict(pricing))
    free_block = payload.get('freeRecommendedModels') if isinstance(payload, dict) else None
    if not isinstance(free_block, list) or not free_block:
        return (list(curated_ids), dict(pricing))
    portal_free_ids: list[str] = []
    for entry in free_block:
        name = _extract_model_name(entry)
        if name:
            portal_free_ids.append(name)
    if not portal_free_ids:
        return (list(curated_ids), dict(pricing))
    augmented_pricing = dict(pricing)
    free_synthetic = {'prompt': '0', 'completion': '0'}
    for mid in portal_free_ids:
        if mid not in augmented_pricing:
            augmented_pricing[mid] = dict(free_synthetic)
    augmented_ids = list(curated_ids)
    seen = set(augmented_ids)
    new_ones = [mid for mid in portal_free_ids if mid not in seen]
    if new_ones:
        augmented_ids = new_ones + augmented_ids
    return (augmented_ids, augmented_pricing)

def union_with_portal_paid_recommendations(curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str='', *, force_refresh: bool=False) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Augment curated list with the Portal's ``paidRecommendedModels``.

    Mirror of :func:`union_with_portal_free_recommendations` for paid-tier
    users. The Portal's ``/api/nous/recommended-models`` endpoint advertises
    which paid models are blessed *right now* — independent of what the
    in-repo ``_PROVIDER_MODELS["nous"]`` list happens to contain or whether
    the docs-hosted catalog manifest has been rebuilt since the last release.

    For paid-tier users this lets newly-launched paid models surface in the
    picker even if the user is running an older Hermes that doesn't ship
    them in its hardcoded curated list. This function returns an augmented
    ``(model_ids, pricing)`` pair where:

    * Portal paid recommendations missing from ``curated_ids`` are
      appended at the front (so the picker shows them first).
    * ``pricing`` is left untouched — we deliberately do NOT synthesize
      pricing entries for paid models. Live pricing is fetched separately
      via :func:`get_pricing_for_provider`; if the live endpoint hasn't
      published pricing yet, the picker shows a blank price column rather
      than fabricating numbers. (The free helper synthesizes ``$0`` so
      :func:`partition_nous_models_by_tier` keeps free models selectable;
      no equivalent gating applies on the paid side, so synthesis would
      only mislead the user.)

    Failures (network, parse, missing field) are silent and degrade to
    returning the inputs unchanged — never block the picker on a
    Portal-side hiccup.
    """
    try:
        payload = fetch_nous_recommended_models(portal_base_url, force_refresh=force_refresh)
    except Exception:
        return (list(curated_ids), dict(pricing))
    paid_block = payload.get('paidRecommendedModels') if isinstance(payload, dict) else None
    if not isinstance(paid_block, list) or not paid_block:
        return (list(curated_ids), dict(pricing))
    portal_paid_ids: list[str] = []
    for entry in paid_block:
        name = _extract_model_name(entry)
        if name:
            portal_paid_ids.append(name)
    if not portal_paid_ids:
        return (list(curated_ids), dict(pricing))
    augmented_ids = list(curated_ids)
    seen = set(augmented_ids)
    new_ones = [mid for mid in portal_paid_ids if mid not in seen]
    if new_ones:
        augmented_ids = new_ones + augmented_ids
    return (augmented_ids, dict(pricing))
_FREE_TIER_CACHE_TTL: int = 180
_free_tier_cache: tuple[bool, float] | None = None

def check_nous_free_tier() -> bool:
    """Check if the current Nous Portal user is on a free (unpaid) tier.

    Results are cached for ``_FREE_TIER_CACHE_TTL`` seconds to avoid
    hitting the Portal API on every call.  The cache is short-lived so
    that an account upgrade is reflected within a few minutes.

    Returns False (assume paid) on any error — never blocks paying users.
    """
    global _free_tier_cache
    now = time.monotonic()
    if _free_tier_cache is not None:
        cached_result, cached_at = _free_tier_cache
        if now - cached_at < _FREE_TIER_CACHE_TTL:
            return cached_result
    try:
        from kylinmemory._vendor.kylin_agent_runtime_cli.auth import get_provider_auth_state, resolve_nous_runtime_credentials
        resolve_nous_runtime_credentials(min_key_ttl_seconds=60)
        state = get_provider_auth_state('nous')
        if not state:
            _free_tier_cache = (False, now)
            return False
        access_token = state.get('access_token', '')
        portal_url = state.get('portal_base_url', '')
        if not access_token:
            _free_tier_cache = (False, now)
            return False
        account_info = fetch_nous_account_tier(access_token, portal_url)
        result = is_nous_free_tier(account_info)
        _free_tier_cache = (result, now)
        return result
    except Exception:
        _free_tier_cache = (False, now)
        return False
NOUS_RECOMMENDED_MODELS_PATH = '/api/nous/recommended-models'
_NOUS_RECOMMENDED_CACHE_TTL: int = 600
_nous_recommended_cache: dict[str, tuple[dict[str, Any], float]] = {}

def fetch_nous_recommended_models(portal_base_url: str='', timeout: float=5.0, *, force_refresh: bool=False) -> dict[str, Any]:
    """Fetch the Nous Portal's curated recommended-models payload.

    Hits ``<portal>/api/nous/recommended-models``. The endpoint is public —
    no auth is required. Results are cached per portal URL for
    ``_NOUS_RECOMMENDED_CACHE_TTL`` seconds; pass ``force_refresh=True`` to
    bypass the cache.

    Returns the parsed JSON dict on success, or ``{}`` on any failure
    (network, parse, non-2xx). Callers must treat missing/null fields as
    "no recommendation" and fall back to their own default.
    """
    base = (portal_base_url or 'https://portal.nousresearch.com').rstrip('/')
    now = time.monotonic()
    cached = _nous_recommended_cache.get(base)
    if not force_refresh and cached is not None:
        payload, cached_at = cached
        if now - cached_at < _NOUS_RECOMMENDED_CACHE_TTL:
            return payload
    url = f'{base}{NOUS_RECOMMENDED_MODELS_PATH}'
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    _nous_recommended_cache[base] = (data, now)
    return data

def _resolve_nous_portal_url() -> str:
    """Best-effort lookup of the Portal base URL the user is authed against."""
    try:
        from kylinmemory._vendor.kylin_agent_runtime_cli.auth import DEFAULT_NOUS_PORTAL_URL, get_provider_auth_state
        state = get_provider_auth_state('nous') or {}
        portal = str(state.get('portal_base_url') or '').strip()
        if portal:
            return portal.rstrip('/')
        return str(DEFAULT_NOUS_PORTAL_URL).rstrip('/')
    except Exception:
        return 'https://portal.nousresearch.com'

def _extract_model_name(entry: Any) -> Optional[str]:
    """Pull the ``modelName`` field from a recommended-model entry, else None."""
    if not isinstance(entry, dict):
        return None
    model_name = entry.get('modelName')
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return None

def get_nous_recommended_aux_model(*, vision: bool=False, free_tier: Optional[bool]=None, portal_base_url: str='', force_refresh: bool=False) -> Optional[str]:
    """Return the Portal's recommended model name for an auxiliary task.

    Picks the best field from the Portal's recommended-models payload:

    * ``vision=True``  → ``paidRecommendedVisionModel``  (paid tier) or
                         ``freeRecommendedVisionModel``  (free tier)
    * ``vision=False`` → ``paidRecommendedCompactionModel`` or
                         ``freeRecommendedCompactionModel``

    When ``free_tier`` is ``None`` (default) the user's tier is auto-detected
    via :func:`check_nous_free_tier`. Pass an explicit bool to bypass the
    detection — useful for tests or when the caller already knows the tier.

    For paid-tier users we prefer the paid recommendation but gracefully fall
    back to the free recommendation if the Portal returned ``null`` for the
    paid field (common during the staged rollout of new paid models).

    Returns ``None`` when every candidate is missing, null, or the fetch
    fails — callers should fall back to their own default (currently
    ``google/gemini-3-flash-preview``).
    """
    base = portal_base_url or _resolve_nous_portal_url()
    payload = fetch_nous_recommended_models(base, force_refresh=force_refresh)
    if not payload:
        return None
    if free_tier is None:
        try:
            free_tier = check_nous_free_tier()
        except Exception:
            free_tier = False
    if vision:
        paid_key, free_key = ('paidRecommendedVisionModel', 'freeRecommendedVisionModel')
    else:
        paid_key, free_key = ('paidRecommendedCompactionModel', 'freeRecommendedCompactionModel')
    candidates = [free_key] if free_tier else [paid_key, free_key]
    for key in candidates:
        name = _extract_model_name(payload.get(key))
        if name:
            return name
    return None
_PROVIDER_ALIASES = {'glm': 'zai', 'z-ai': 'zai', 'z.ai': 'zai', 'zhipu': 'zai', 'github': 'copilot', 'github-copilot': 'copilot', 'github-models': 'copilot', 'github-model': 'copilot', 'github-copilot-acp': 'copilot-acp', 'copilot-acp-agent': 'copilot-acp', 'google': 'gemini', 'google-gemini': 'gemini', 'google-ai-studio': 'gemini', 'kimi': 'kimi-coding', 'moonshot': 'kimi-coding', 'kimi-cn': 'kimi-coding-cn', 'moonshot-cn': 'kimi-coding-cn', 'step': 'stepfun', 'stepfun-coding-plan': 'stepfun', 'arcee-ai': 'arcee', 'arceeai': 'arcee', 'gmi-cloud': 'gmi', 'gmicloud': 'gmi', 'minimax-china': 'minimax-cn', 'minimax_cn': 'minimax-cn', 'minimax-portal': 'minimax-oauth', 'minimax-global': 'minimax-oauth', 'minimax_oauth': 'minimax-oauth', 'claude': 'anthropic', 'claude-code': 'anthropic', 'deep-seek': 'deepseek', 'opencode': 'opencode-zen', 'zen': 'opencode-zen', 'go': 'opencode-go', 'opencode-go-sub': 'opencode-go', 'aigateway': 'ai-gateway', 'vercel': 'ai-gateway', 'vercel-ai-gateway': 'ai-gateway', 'kilo': 'kilocode', 'kilo-code': 'kilocode', 'kilo-gateway': 'kilocode', 'dashscope': 'alibaba', 'aliyun': 'alibaba', 'qwen': 'alibaba', 'alibaba-cloud': 'alibaba', 'qwen-portal': 'qwen-oauth', 'gemini-cli': 'google-gemini-cli', 'gemini-oauth': 'google-gemini-cli', 'hf': 'huggingface', 'hugging-face': 'huggingface', 'huggingface-hub': 'huggingface', 'novita-ai': 'novita', 'novitaai': 'novita', 'mimo': 'xiaomi', 'xiaomi-mimo': 'xiaomi', 'tencent': 'tencent-tokenhub', 'tokenhub': 'tencent-tokenhub', 'tencent-cloud': 'tencent-tokenhub', 'tencentmaas': 'tencent-tokenhub', 'aws': 'bedrock', 'aws-bedrock': 'bedrock', 'amazon-bedrock': 'bedrock', 'amazon': 'bedrock', 'grok': 'xai', 'grok-oauth': 'xai-oauth', 'xai-oauth': 'xai-oauth', 'x-ai-oauth': 'xai-oauth', 'xai-grok-oauth': 'xai-oauth', 'x-ai': 'xai', 'x.ai': 'xai', 'nim': 'nvidia', 'nvidia-nim': 'nvidia', 'build-nvidia': 'nvidia', 'nemotron': 'nvidia', 'lmstudio': 'lmstudio', 'lm-studio': 'lmstudio', 'lm_studio': 'lmstudio', 'ollama': 'custom', 'ollama_cloud': 'ollama-cloud'}

def _openrouter_model_is_free(pricing: Any) -> bool:
    """Return True when both prompt and completion pricing are zero."""
    if not isinstance(pricing, dict):
        return False
    try:
        return float(pricing.get('prompt', '0')) == 0 and float(pricing.get('completion', '0')) == 0
    except (TypeError, ValueError):
        return False

def _openrouter_model_supports_tools(item: Any) -> bool:
    """Return True when the model's ``supported_parameters`` advertise tool calling.

    hermes-agent is tool-calling-first — every provider path assumes the model
    can invoke tools. Models that don't advertise ``tools`` in their
    ``supported_parameters`` (e.g. image-only or completion-only models) cannot
    be driven by the agent loop and would fail at the first tool call.

    **Permissive when the field is missing.** Some OpenRouter-compatible gateways
    (Nous Portal, private mirrors, older catalog snapshots) don't populate
    ``supported_parameters`` at all. Treat that as "unknown capability → allow"
    so the picker doesn't silently empty for those users. Only hide models
    whose ``supported_parameters`` is an explicit list that omits ``tools``.

    Ported from Kilo-Org/kilocode#9068.
    """
    if not isinstance(item, dict):
        return True
    params = item.get('supported_parameters')
    if not isinstance(params, list):
        return True
    return 'tools' in params

def fetch_openrouter_models(timeout: float=8.0, *, force_refresh: bool=False) -> list[tuple[str, str]]:
    """Return the curated OpenRouter picker list, refreshed from the live catalog when possible."""
    global _openrouter_catalog_cache
    if _openrouter_catalog_cache is not None and (not force_refresh):
        return list(_openrouter_catalog_cache)
    try:
        from kylinmemory._vendor.kylin_agent_runtime_cli.model_catalog import get_curated_openrouter_models
        remote = get_curated_openrouter_models()
    except Exception:
        remote = None
    fallback = list(remote) if remote else list(OPENROUTER_MODELS)
    preferred_ids = [mid for mid, _ in fallback]
    try:
        req = urllib.request.Request('https://openrouter.ai/api/v1/models', headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return list(_openrouter_catalog_cache or fallback)
    live_items = payload.get('data', [])
    if not isinstance(live_items, list):
        return list(_openrouter_catalog_cache or fallback)
    live_by_id: dict[str, dict[str, Any]] = {}
    for item in live_items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get('id') or '').strip()
        if not mid:
            continue
        live_by_id[mid] = item
    curated: list[tuple[str, str]] = []
    for preferred_id in preferred_ids:
        live_item = live_by_id.get(preferred_id)
        if live_item is None:
            continue
        if not _openrouter_model_supports_tools(live_item):
            continue
        desc = 'free' if _openrouter_model_is_free(live_item.get('pricing')) else ''
        curated.append((preferred_id, desc))
    if not curated:
        return list(_openrouter_catalog_cache or fallback)
    first_id, _ = curated[0]
    curated[0] = (first_id, 'recommended')
    _openrouter_catalog_cache = curated
    return list(curated)

def model_ids(*, force_refresh: bool=False) -> list[str]:
    """Return just the OpenRouter model-id strings."""
    return [mid for mid, _ in fetch_openrouter_models(force_refresh=force_refresh)]

def get_curated_nous_model_ids() -> list[str]:
    """Return the curated Nous Portal model-id list.

    Prefers the remotely-hosted catalog manifest (published under
    ``website/static/api/model-catalog.json``); falls back to the in-repo
    snapshot in ``_PROVIDER_MODELS["nous"]`` when the manifest is
    unreachable. Always returns a list (never None).
    """
    try:
        from kylinmemory._vendor.kylin_agent_runtime_cli.model_catalog import get_curated_nous_models
        remote = get_curated_nous_models()
    except Exception:
        remote = None
    if remote:
        return list(remote)
    return list(_PROVIDER_MODELS.get('nous', []))
_pricing_cache: dict[str, dict[str, dict[str, str]]] = {}

def _format_price_per_mtok(per_token_str: str) -> str:
    """Convert a per-token price string to a human-friendly $/Mtok string.

    Always uses 2 decimal places so that prices align vertically when
    right-justified in a column (the decimal point stays in the same position).

    Examples:
        "0.000003"   → "$3.00"      (per million tokens)
        "0.00003"    → "$30.00"
        "0.00000015" → "$0.15"
        "0.0000001"  → "$0.10"
        "0.00018"    → "$180.00"
        "0"          → "free"
    """
    try:
        val = float(per_token_str)
    except (TypeError, ValueError):
        return '?'
    if val == 0:
        return 'free'
    per_m = val * 1000000
    return f'${per_m:.2f}'

def fetch_models_with_pricing(api_key: str | None=None, base_url: str='https://openrouter.ai/api', timeout: float=8.0, *, force_refresh: bool=False) -> dict[str, dict[str, str]]:
    """Fetch ``/v1/models`` and return ``{model_id: {prompt, completion}}`` pricing.

    Results are cached per *base_url* so repeated calls are free.
    Works with any OpenRouter-compatible endpoint (OpenRouter, Nous Portal).
    """
    cache_key = (base_url or '').rstrip('/')
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]
    url = cache_key.rstrip('/') + '/v1/models'
    headers: dict[str, str] = {'Accept': 'application/json', 'User-Agent': _HERMES_USER_AGENT}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}
    result: dict[str, dict[str, str]] = {}
    for item in payload.get('data', []):
        mid = item.get('id')
        pricing = item.get('pricing')
        if mid and isinstance(pricing, dict):
            entry: dict[str, str] = {'prompt': str(pricing.get('prompt', '')), 'completion': str(pricing.get('completion', ''))}
            if pricing.get('input_cache_read'):
                entry['input_cache_read'] = str(pricing['input_cache_read'])
            if pricing.get('input_cache_write'):
                entry['input_cache_write'] = str(pricing['input_cache_write'])
            result[mid] = entry
    _pricing_cache[cache_key] = result
    return result

def fetch_ai_gateway_pricing(timeout: float=8.0, *, force_refresh: bool=False) -> dict[str, dict[str, str]]:
    """Fetch Vercel AI Gateway /v1/models and return hermes-shaped pricing.

    Vercel uses ``input`` / ``output`` field names; hermes's picker expects
    ``prompt`` / ``completion``. This translates. Cache read/write field names
    already match.
    """
    from kylinmemory._vendor.kylin_agent_runtime_constants import AI_GATEWAY_BASE_URL
    cache_key = AI_GATEWAY_BASE_URL.rstrip('/')
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]
    try:
        req = urllib.request.Request(f'{cache_key}/models', headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}
    result: dict[str, dict[str, str]] = {}
    for item in payload.get('data', []):
        if not isinstance(item, dict):
            continue
        mid = item.get('id')
        pricing = item.get('pricing')
        if not (mid and isinstance(pricing, dict)):
            continue
        entry: dict[str, str] = {'prompt': str(pricing.get('input', '')), 'completion': str(pricing.get('output', ''))}
        if pricing.get('input_cache_read'):
            entry['input_cache_read'] = str(pricing['input_cache_read'])
        if pricing.get('input_cache_write'):
            entry['input_cache_write'] = str(pricing['input_cache_write'])
        result[mid] = entry
    _pricing_cache[cache_key] = result
    return result

def _resolve_openrouter_api_key() -> str:
    """Best-effort OpenRouter API key for pricing fetch."""
    return os.getenv('OPENROUTER_API_KEY', '').strip()
_DEFAULT_NOUS_INFERENCE_BASE = 'https://inference-api.nousresearch.com'

def _resolve_nous_pricing_credentials() -> tuple[str, str]:
    """Return ``(api_key, base_url)`` for Nous Portal pricing.

    The Nous inference ``/v1/models`` endpoint exposes pricing without
    authentication, so the api_key is best-effort: when runtime credential
    resolution fails (expired refresh token, missing auth.json, etc.) we
    still return the default inference base URL so the picker keeps
    working with anonymous pricing data.  Free-tier users in particular
    need this — pricing drives the free/paid partition, and silently
    returning empty pricing because of an auth blip makes the picker
    look broken ("No free models currently available").
    """
    try:
        from kylinmemory._vendor.kylin_agent_runtime_cli.auth import resolve_nous_runtime_credentials
        creds = resolve_nous_runtime_credentials()
        if creds:
            return (creds.get('api_key', ''), creds.get('base_url', ''))
    except Exception:
        pass
    return ('', _DEFAULT_NOUS_INFERENCE_BASE)

def get_pricing_for_provider(provider: str, *, force_refresh: bool=False) -> dict[str, dict[str, str]]:
    """Return live pricing for providers that support it (openrouter, nous, ai-gateway, novita)."""
    normalized = normalize_provider(provider)
    if normalized == 'openrouter':
        return fetch_models_with_pricing(api_key=_resolve_openrouter_api_key(), base_url='https://openrouter.ai/api', force_refresh=force_refresh)
    if normalized == 'ai-gateway':
        return fetch_ai_gateway_pricing(force_refresh=force_refresh)
    if normalized == 'novita':
        return _fetch_novita_pricing(force_refresh=force_refresh)
    if normalized == 'nous':
        api_key, base_url = _resolve_nous_pricing_credentials()
        if base_url:
            stripped = base_url.rstrip('/')
            if stripped.endswith('/v1'):
                stripped = stripped[:-3]
            return fetch_models_with_pricing(api_key=api_key, base_url=stripped, force_refresh=force_refresh)
    return {}

def _fetch_novita_pricing(timeout: float=8.0, *, force_refresh: bool=False) -> dict[str, dict[str, str]]:
    """Fetch pricing from NovitaAI /v1/models.

    NovitaAI returns input/output prices per million tokens in units of
    0.0001 USD. Convert them to the per-token strings used by the shared
    pricing formatter.

    Results are cached in ``_pricing_cache`` keyed on the resolved base URL,
    matching the pattern used by ``fetch_ai_gateway_pricing`` — without this,
    every menu render or pricing lookup re-hits the network.
    """
    api_key = os.getenv('NOVITA_API_KEY', '').strip()
    if not api_key:
        return {}
    base_url = os.getenv('NOVITA_BASE_URL', '').strip() or 'https://api.novita.ai/openai/v1'
    cache_key = base_url.rstrip('/')
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]
    url = cache_key + '/models'
    headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json', 'User-Agent': _HERMES_USER_AGENT}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}
    result: dict[str, dict[str, str]] = {}
    for item in payload.get('data', []):
        if not isinstance(item, dict):
            continue
        mid = item.get('id')
        if not mid:
            continue
        inp = item.get('input_token_price_per_m')
        out = item.get('output_token_price_per_m')
        if inp is None and out is None:
            continue
        result[str(mid)] = {'prompt': str(float(inp or 0) / 10000 / 1000000), 'completion': str(float(out or 0) / 10000 / 1000000)}
    _pricing_cache[cache_key] = result
    return result

def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to Hermes' canonical provider ids.

    Note: ``"auto"`` passes through unchanged — use
    ``kylin_agent_runtime_cli.auth.resolve_provider()`` to resolve it to a concrete
    provider based on credentials and environment.
    """
    normalized = (provider or 'openrouter').strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)

def _payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        data = payload.get('data', [])
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []

def copilot_default_headers() -> dict[str, str]:
    """Standard headers for Copilot API requests.

    Includes Openai-Intent and x-initiator headers that opencode and the
    Copilot CLI send on every request.
    """
    try:
        from kylinmemory._vendor.kylin_agent_runtime_cli.copilot_auth import copilot_request_headers
        return copilot_request_headers(is_agent_turn=True)
    except ImportError:
        return {'Editor-Version': COPILOT_EDITOR_VERSION, 'User-Agent': 'HermesAgent/1.0', 'Openai-Intent': 'conversation-edits', 'x-initiator': 'agent'}

def _copilot_catalog_item_is_text_model(item: dict[str, Any]) -> bool:
    model_id = str(item.get('id') or '').strip()
    if not model_id:
        return False
    if item.get('model_picker_enabled') is False:
        return False
    capabilities = item.get('capabilities')
    if isinstance(capabilities, dict):
        model_type = str(capabilities.get('type') or '').strip().lower()
        if model_type and model_type != 'chat':
            return False
    supported_endpoints = item.get('supported_endpoints')
    if isinstance(supported_endpoints, list):
        normalized_endpoints = {str(endpoint).strip() for endpoint in supported_endpoints if str(endpoint).strip()}
        if normalized_endpoints and (not normalized_endpoints.intersection({'/chat/completions', '/responses', '/v1/messages'})):
            return False
    return True

def fetch_github_model_catalog(api_key: Optional[str]=None, timeout: float=5.0) -> Optional[list[dict[str, Any]]]:
    """Fetch the live GitHub Copilot model catalog for this account."""
    attempts: list[dict[str, str]] = []
    if api_key:
        attempts.append({**copilot_default_headers(), 'Authorization': f'Bearer {api_key}'})
    attempts.append(copilot_default_headers())
    for headers in attempts:
        req = urllib.request.Request(COPILOT_MODELS_URL, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                items = _payload_items(data)
                models: list[dict[str, Any]] = []
                seen_ids: set[str] = set()
                for item in items:
                    if not _copilot_catalog_item_is_text_model(item):
                        continue
                    model_id = str(item.get('id') or '').strip()
                    if not model_id or model_id in seen_ids:
                        continue
                    seen_ids.add(model_id)
                    models.append(item)
                if models:
                    return models
        except Exception:
            continue
    return None
_COPILOT_MODEL_ALIASES = {'openai/gpt-5': 'gpt-5-mini', 'openai/gpt-5-chat': 'gpt-5-mini', 'openai/gpt-5-mini': 'gpt-5-mini', 'openai/gpt-5-nano': 'gpt-5-mini', 'openai/gpt-4.1': 'gpt-4.1', 'openai/gpt-4.1-mini': 'gpt-4.1', 'openai/gpt-4.1-nano': 'gpt-4.1', 'openai/gpt-4o': 'gpt-4o', 'openai/gpt-4o-mini': 'gpt-4o-mini', 'openai/o1': 'gpt-5.2', 'openai/o1-mini': 'gpt-5-mini', 'openai/o1-preview': 'gpt-5.2', 'openai/o3': 'gpt-5.3-codex', 'openai/o3-mini': 'gpt-5-mini', 'openai/o4-mini': 'gpt-5-mini', 'anthropic/claude-opus-4.6': 'claude-opus-4.6', 'anthropic/claude-sonnet-4.6': 'claude-sonnet-4.6', 'anthropic/claude-sonnet-4': 'claude-sonnet-4', 'anthropic/claude-sonnet-4.5': 'claude-sonnet-4.5', 'anthropic/claude-haiku-4.5': 'claude-haiku-4.5', 'claude-opus-4-6': 'claude-opus-4.6', 'claude-sonnet-4-6': 'claude-sonnet-4.6', 'claude-sonnet-4-0': 'claude-sonnet-4', 'claude-sonnet-4-5': 'claude-sonnet-4.5', 'claude-haiku-4-5': 'claude-haiku-4.5', 'anthropic/claude-opus-4-6': 'claude-opus-4.6', 'anthropic/claude-sonnet-4-6': 'claude-sonnet-4.6', 'anthropic/claude-sonnet-4-0': 'claude-sonnet-4', 'anthropic/claude-sonnet-4-5': 'claude-sonnet-4.5', 'anthropic/claude-haiku-4-5': 'claude-haiku-4.5'}

def _copilot_catalog_ids(catalog: Optional[list[dict[str, Any]]]=None, api_key: Optional[str]=None) -> set[str]:
    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)
    if not catalog:
        return set()
    return {str(item.get('id') or '').strip() for item in catalog if str(item.get('id') or '').strip()}

def normalize_copilot_model_id(model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]]=None, api_key: Optional[str]=None) -> str:
    raw = str(model_id or '').strip()
    if not raw:
        return ''
    catalog_ids = _copilot_catalog_ids(catalog=catalog, api_key=api_key)
    alias = _COPILOT_MODEL_ALIASES.get(raw)
    if alias:
        return alias
    candidates = [raw]
    if '/' in raw:
        candidates.append(raw.split('/', 1)[1].strip())
    if raw.endswith('-mini'):
        candidates.append(raw[:-5])
    if raw.endswith('-nano'):
        candidates.append(raw[:-5])
    if raw.endswith('-chat'):
        candidates.append(raw[:-5])
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in _COPILOT_MODEL_ALIASES:
            return _COPILOT_MODEL_ALIASES[candidate]
        if candidate in catalog_ids:
            return candidate
    if '/' in raw:
        return raw.split('/', 1)[1].strip()
    return raw

def _github_reasoning_efforts_for_model_id(model_id: str) -> list[str]:
    raw = (model_id or '').strip().lower()
    if raw.startswith(('openai/o1', 'openai/o3', 'openai/o4', 'o1', 'o3', 'o4')):
        return list(COPILOT_REASONING_EFFORTS_O_SERIES)
    normalized = normalize_copilot_model_id(model_id).lower()
    if normalized.startswith('gpt-5'):
        return list(COPILOT_REASONING_EFFORTS_GPT5)
    return []

def _should_use_copilot_responses_api(model_id: str) -> bool:
    """Decide whether a Copilot model should use the Responses API.

    Replicates opencode's ``shouldUseCopilotResponsesApi`` logic:
    GPT-5+ models use Responses API, except ``gpt-5-mini`` which uses
    Chat Completions.  All non-GPT models (Claude, Gemini, etc.) use
    Chat Completions.
    """
    import re
    match = re.match('^gpt-(\\d+)', model_id)
    if not match:
        return False
    major = int(match.group(1))
    return major >= 5 and (not model_id.startswith('gpt-5-mini'))

def copilot_model_api_mode(model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]]=None, api_key: Optional[str]=None) -> str:
    """Determine the API mode for a Copilot model.

    Uses the model ID pattern (matching opencode's approach) as the
    primary signal.  Falls back to the catalog's ``supported_endpoints``
    only for models not covered by the pattern check.
    """
    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return 'chat_completions'
    if _should_use_copilot_responses_api(normalized):
        return 'codex_responses'
    if catalog:
        catalog_entry = next((item for item in catalog if item.get('id') == normalized), None)
        if isinstance(catalog_entry, dict):
            supported_endpoints = {str(endpoint).strip() for endpoint in catalog_entry.get('supported_endpoints') or [] if str(endpoint).strip()}
            if '/v1/messages' in supported_endpoints and '/chat/completions' not in supported_endpoints:
                return 'anthropic_messages'
    return 'chat_completions'
_AZURE_FOUNDRY_RESPONSES_PREFIXES = ('codex', 'gpt-5', 'o1', 'o3', 'o4')

def azure_foundry_model_api_mode(model_name: Optional[str]) -> Optional[str]:
    """Infer Azure Foundry api_mode from a deployment/model name.

    Returns ``"codex_responses"`` when the model name matches a family that
    only accepts the Responses API on Azure Foundry (GPT-5.x, codex, o1/o3/o4
    reasoning models).  Returns ``None`` otherwise — the caller should fall
    back to the configured/default api_mode (typically ``chat_completions``)
    so GPT-4o, GPT-4 Turbo, Llama, Mistral, etc. keep working.

    Intentionally does NOT return ``anthropic_messages``; Anthropic-style
    Azure endpoints are disambiguated by URL (``/anthropic`` suffix) in
    ``runtime_provider._detect_api_mode_for_url`` and by the user setting
    ``model.api_mode: anthropic_messages`` explicitly.
    """
    raw = str(model_name or '').strip().lower()
    if not raw:
        return None
    if '/' in raw:
        raw = raw.rsplit('/', 1)[-1]
    for prefix in _AZURE_FOUNDRY_RESPONSES_PREFIXES:
        if raw.startswith(prefix):
            return 'codex_responses'
    return None

def normalize_opencode_model_id(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Normalize OpenCode config IDs to the bare model slug used in API requests."""
    provider = normalize_provider(provider_id)
    current = str(model_id or '').strip()
    if not current or provider not in {'opencode-zen', 'opencode-go'}:
        return current
    prefix = f'{provider}/'
    if current.lower().startswith(prefix):
        return current[len(prefix):]
    return current

def opencode_model_api_mode(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Determine the API mode for an OpenCode Zen / Go model.

    OpenCode routes different models behind different API surfaces:

    - GPT-5 / Codex models on Zen use ``/v1/responses``
    - Claude models on Zen use ``/v1/messages``
    - MiniMax models on Go use ``/v1/messages``
    - GLM / Kimi on Go use ``/v1/chat/completions``
    - Other Zen models (Gemini, GLM, Kimi, MiniMax, Qwen, etc.) use
      ``/v1/chat/completions``

    This follows the published OpenCode docs for Zen and Go endpoints.
    """
    provider = normalize_provider(provider_id)
    normalized = normalize_opencode_model_id(provider_id, model_id).lower()
    if not normalized:
        return 'chat_completions'
    if provider == 'opencode-go':
        if normalized.startswith('minimax-'):
            return 'anthropic_messages'
        return 'chat_completions'
    if provider == 'opencode-zen':
        if normalized.startswith('claude-'):
            return 'anthropic_messages'
        if normalized.startswith('gpt-'):
            return 'codex_responses'
        return 'chat_completions'
    return 'chat_completions'

def github_model_reasoning_efforts(model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]]=None, api_key: Optional[str]=None) -> list[str]:
    """Return supported reasoning-effort levels for a Copilot-visible model."""
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return []
    catalog_entry = None
    if catalog is not None:
        catalog_entry = next((item for item in catalog if item.get('id') == normalized), None)
    elif api_key:
        fetched_catalog = fetch_github_model_catalog(api_key=api_key)
        if fetched_catalog:
            catalog_entry = next((item for item in fetched_catalog if item.get('id') == normalized), None)
    if catalog_entry is not None:
        capabilities = catalog_entry.get('capabilities')
        if isinstance(capabilities, dict):
            supports = capabilities.get('supports')
            if isinstance(supports, dict):
                efforts = supports.get('reasoning_effort')
                if isinstance(efforts, list):
                    normalized_efforts = [str(effort).strip().lower() for effort in efforts if str(effort).strip()]
                    return list(dict.fromkeys(normalized_efforts))
            return []
        legacy_capabilities = {str(capability).strip().lower() for capability in catalog_entry.get('capabilities', []) if str(capability).strip()}
        if 'reasoning' not in legacy_capabilities:
            return []
    return _github_reasoning_efforts_for_model_id(str(model_id or normalized))

