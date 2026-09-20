from __future__ import annotations
import re
from typing import Optional
_VENDOR_PREFIXES: dict[str, str] = {'claude': 'anthropic', 'gpt': 'openai', 'o1': 'openai', 'o3': 'openai', 'o4': 'openai', 'gemini': 'google', 'gemma': 'google', 'deepseek': 'deepseek', 'glm': 'z-ai', 'kimi': 'moonshotai', 'minimax': 'minimax', 'grok': 'x-ai', 'qwen': 'qwen', 'mimo': 'xiaomi', 'trinity': 'arcee-ai', 'nemotron': 'nvidia', 'llama': 'meta-llama', 'step': 'stepfun', 'trinity': 'arcee-ai'}
_AGGREGATOR_PROVIDERS: frozenset[str] = frozenset({'openrouter', 'nous', 'ai-gateway', 'kilocode'})
_DOT_TO_HYPHEN_PROVIDERS: frozenset[str] = frozenset({'anthropic'})
_STRIP_VENDOR_ONLY_PROVIDERS: frozenset[str] = frozenset({'copilot', 'copilot-acp', 'openai-codex'})
_AUTHORITATIVE_NATIVE_PROVIDERS: frozenset[str] = frozenset({'gemini', 'huggingface'})
_MATCHING_PREFIX_STRIP_PROVIDERS: frozenset[str] = frozenset({'zai', 'kimi-coding', 'kimi-coding-cn', 'minimax', 'minimax-oauth', 'minimax-cn', 'alibaba', 'qwen-oauth', 'xiaomi', 'arcee', 'ollama-cloud', 'custom'})
_LOWERCASE_MODEL_PROVIDERS: frozenset[str] = frozenset({'xiaomi'})
_DEEPSEEK_REASONER_KEYWORDS: frozenset[str] = frozenset({'reasoner', 'r1', 'think', 'reasoning', 'cot'})
_DEEPSEEK_CANONICAL_MODELS: frozenset[str] = frozenset({'deepseek-chat', 'deepseek-reasoner', 'deepseek-v4-pro', 'deepseek-v4-flash'})
_DEEPSEEK_V_SERIES_RE = re.compile('^deepseek-v\\d+([-.].+)?$')

def _normalize_for_deepseek(model_name: str) -> str:
    """Map a model input to a DeepSeek-accepted identifier.

    Rules:
    - Already a known canonical (``deepseek-chat``/``deepseek-reasoner``/
      ``deepseek-v4-pro``/``deepseek-v4-flash``) -> pass through.
    - Matches the V-series pattern ``deepseek-v<digit>...`` -> pass through
      (covers future ``deepseek-v5-*`` and dated variants without a release).
    - Contains a reasoner keyword (r1, think, reasoning, cot, reasoner)
      -> ``deepseek-reasoner``.
    - Everything else -> ``deepseek-chat``.

    Args:
        model_name: The bare model name (vendor prefix already stripped).

    Returns:
        A DeepSeek-accepted model identifier.
    """
    bare = _strip_vendor_prefix(model_name).lower()
    if bare in _DEEPSEEK_CANONICAL_MODELS:
        return bare
    if _DEEPSEEK_V_SERIES_RE.match(bare):
        return bare
    for keyword in _DEEPSEEK_REASONER_KEYWORDS:
        if keyword in bare:
            return 'deepseek-reasoner'
    return 'deepseek-chat'

def _strip_vendor_prefix(model_name: str) -> str:
    """Remove a ``vendor/`` prefix if present.

    Examples::

        >>> _strip_vendor_prefix("anthropic/claude-sonnet-4.6")
        'claude-sonnet-4.6'
        >>> _strip_vendor_prefix("claude-sonnet-4.6")
        'claude-sonnet-4.6'
        >>> _strip_vendor_prefix("meta-llama/llama-4-scout")
        'llama-4-scout'
    """
    if '/' in model_name:
        return model_name.split('/', 1)[1]
    return model_name

def _dots_to_hyphens(model_name: str) -> str:
    """Replace dots with hyphens in a model name.

    Anthropic's native API uses hyphens where marketing names use dots:
    ``claude-sonnet-4.6`` -> ``claude-sonnet-4-6``.
    """
    return model_name.replace('.', '-')

def _normalize_provider_alias(provider_name: str) -> str:
    """Resolve provider aliases to Hermes' canonical ids."""
    raw = (provider_name or '').strip().lower()
    if not raw:
        return raw
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.models import normalize_provider
        return normalize_provider(raw)
    except Exception:
        return raw

def _strip_matching_provider_prefix(model_name: str, target_provider: str) -> str:
    """Strip ``provider/`` only when the prefix matches the target provider.

    This prevents arbitrary slash-bearing model IDs from being mangled on
    native providers while still repairing manual config values like
    ``zai/glm-5.1`` for the ``zai`` provider.
    """
    if '/' not in model_name:
        return model_name
    prefix, remainder = model_name.split('/', 1)
    if not prefix.strip() or not remainder.strip():
        return model_name
    normalized_prefix = _normalize_provider_alias(prefix)
    normalized_target = _normalize_provider_alias(target_provider)
    if normalized_prefix and normalized_prefix == normalized_target:
        return remainder.strip()
    return model_name

def detect_vendor(model_name: str) -> Optional[str]:
    """Detect the vendor slug from a bare model name.

    Uses the first hyphen-delimited token of the model name to look up
    the corresponding vendor in ``_VENDOR_PREFIXES``.  Also handles
    case-insensitive matching and special patterns.

    Args:
        model_name: A model name, optionally already including a
            ``vendor/`` prefix.  If a prefix is present it is used
            directly.

    Returns:
        The vendor slug (e.g. ``"anthropic"``, ``"openai"``) or ``None``
        if no vendor can be confidently detected.

    Examples::

        >>> detect_vendor("claude-sonnet-4.6")
        'anthropic'
        >>> detect_vendor("gpt-5.4-mini")
        'openai'
        >>> detect_vendor("anthropic/claude-sonnet-4.6")
        'anthropic'
        >>> detect_vendor("my-custom-model")
    """
    name = model_name.strip()
    if not name:
        return None
    if '/' in name:
        return name.split('/', 1)[0].lower() or None
    name_lower = name.lower()
    first_token = name_lower.split('-')[0]
    if first_token in _VENDOR_PREFIXES:
        return _VENDOR_PREFIXES[first_token]
    for prefix, vendor in _VENDOR_PREFIXES.items():
        if name_lower.startswith(prefix):
            return vendor
    return None

def _prepend_vendor(model_name: str) -> str:
    """Prepend the detected ``vendor/`` prefix if missing.

    Used for aggregator providers that require ``vendor/model`` format.
    If the name already contains a ``/``, it is returned as-is.
    If no vendor can be detected, the name is returned unchanged
    (aggregators may still accept it or return an error).

    Examples::

        >>> _prepend_vendor("claude-sonnet-4.6")
        'anthropic/claude-sonnet-4.6'
        >>> _prepend_vendor("anthropic/claude-sonnet-4.6")
        'anthropic/claude-sonnet-4.6'
        >>> _prepend_vendor("my-custom-thing")
        'my-custom-thing'
    """
    if '/' in model_name:
        return model_name
    vendor = detect_vendor(model_name)
    if vendor:
        return f'{vendor}/{model_name}'
    return model_name

def normalize_model_for_provider(model_input: str, target_provider: str) -> str:
    """Translate a model name into the format the target provider's API expects.

    This is the primary entry point for model name normalisation.  It
    accepts any user-facing model identifier and transforms it for the
    specific provider that will receive the API call.

    Args:
        model_input: The model name as provided by the user or config.
            Can be bare (``"claude-sonnet-4.6"``), vendor-prefixed
            (``"anthropic/claude-sonnet-4.6"``), or already in native
            format (``"claude-sonnet-4-6"``).
        target_provider: The canonical Hermes provider id, e.g.
            ``"openrouter"``, ``"anthropic"``, ``"copilot"``,
            ``"deepseek"``, ``"custom"``.  Should already be normalised
            via ``kylin_agent_runtime_cli.models.normalize_provider()``.

    Returns:
        The model identifier string that the target provider's API
        expects.

    Raises:
        No exceptions -- always returns a best-effort string.

    Examples::

        >>> normalize_model_for_provider("claude-sonnet-4.6", "openrouter")
        'anthropic/claude-sonnet-4.6'

        >>> normalize_model_for_provider("anthropic/claude-sonnet-4.6", "anthropic")
        'claude-sonnet-4-6'

        >>> normalize_model_for_provider("anthropic/claude-sonnet-4.6", "copilot")
        'claude-sonnet-4.6'

        >>> normalize_model_for_provider("openai/gpt-5.4", "copilot")
        'gpt-5.4'

        >>> normalize_model_for_provider("claude-sonnet-4.6", "opencode-zen")
        'claude-sonnet-4-6'

        >>> normalize_model_for_provider("minimax-m2.5-free", "opencode-zen")
        'minimax-m2.5-free'

        >>> normalize_model_for_provider("deepseek-v3", "deepseek")
        'deepseek-chat'

        >>> normalize_model_for_provider("deepseek-r1", "deepseek")
        'deepseek-reasoner'

        >>> normalize_model_for_provider("my-model", "custom")
        'my-model'

        >>> normalize_model_for_provider("claude-sonnet-4.6", "zai")
        'claude-sonnet-4.6'

        >>> normalize_model_for_provider("MiMo-V2.5-Pro", "xiaomi")
        'mimo-v2.5-pro'
    """
    name = (model_input or '').strip()
    if not name:
        return name
    provider = _normalize_provider_alias(target_provider)
    if provider in _AGGREGATOR_PROVIDERS:
        return _prepend_vendor(name)
    if provider in {'opencode-zen', 'opencode-go'}:
        if '/' in name:
            _, bare_after_slash = name.split('/', 1)
            name = bare_after_slash.strip() or name
        if provider == 'opencode-zen' and name.lower().startswith('claude-'):
            return _dots_to_hyphens(name)
        return name
    if provider in _DOT_TO_HYPHEN_PROVIDERS:
        bare = _strip_matching_provider_prefix(name, provider)
        if '/' in bare:
            return bare
        return _dots_to_hyphens(bare)
    if provider in {'copilot', 'copilot-acp'}:
        try:
            from kylin_memory._vendor.kylin_agent_runtime_cli.models import normalize_copilot_model_id
            normalized = normalize_copilot_model_id(name)
            if normalized:
                return normalized
        except Exception:
            pass
    if provider in _STRIP_VENDOR_ONLY_PROVIDERS:
        stripped = _strip_matching_provider_prefix(name, provider)
        if stripped == name and name.startswith('openai/'):
            return name.split('/', 1)[1]
        return stripped
    if provider == 'deepseek':
        bare = _strip_matching_provider_prefix(name, provider)
        if '/' in bare:
            return bare
        return _normalize_for_deepseek(bare)
    if provider in _MATCHING_PREFIX_STRIP_PROVIDERS:
        result = _strip_matching_provider_prefix(name, provider)
        if provider in _LOWERCASE_MODEL_PROVIDERS:
            result = result.lower()
        return result
    if provider in _AUTHORITATIVE_NATIVE_PROVIDERS:
        return name
    return name

