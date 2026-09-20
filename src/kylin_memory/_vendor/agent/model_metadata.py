from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
_GROK_EFFORT_CAPABLE_PREFIXES = ('grok-3-mini', 'grok-4.20-multi-agent', 'grok-4.3')

def grok_supports_reasoning_effort(model: str) -> bool:
    """Return True when an xAI Grok model accepts ``reasoning.effort``.

    Allowlist by substring (matches both bare ``grok-3-mini`` and
    aggregator-prefixed ``x-ai/grok-3-mini``). Conservative by design:
    if a future Grok model isn't listed, we send no effort dial rather
    than 400.
    """
    name = (model or '').strip().lower()
    if not name:
        return False
    for sep in ('/',):
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
    return any((name.startswith(prefix) for prefix in _GROK_EFFORT_CAPABLE_PREFIXES))

def _normalize_base_url(base_url: str) -> str:
    return (base_url or '').strip().rstrip('/')
_URL_TO_PROVIDER: Dict[str, str] = {'api.openai.com': 'openai', 'chatgpt.com': 'openai', 'api.anthropic.com': 'anthropic', 'api.z.ai': 'zai', 'open.bigmodel.cn': 'zai', 'api.moonshot.ai': 'kimi-coding', 'api.moonshot.cn': 'kimi-coding-cn', 'api.kimi.com': 'kimi-coding', 'api.stepfun.ai': 'stepfun', 'api.stepfun.com': 'stepfun', 'api.arcee.ai': 'arcee', 'api.minimax': 'minimax', 'dashscope.aliyuncs.com': 'alibaba', 'dashscope-intl.aliyuncs.com': 'alibaba', 'portal.qwen.ai': 'qwen-oauth', 'openrouter.ai': 'openrouter', 'generativelanguage.googleapis.com': 'gemini', 'inference-api.nousresearch.com': 'nous', 'api.deepseek.com': 'deepseek', 'api.githubcopilot.com': 'copilot', 'models.github.ai': 'copilot', 'models.inference.ai.azure.com': 'copilot', 'api.fireworks.ai': 'fireworks', 'opencode.ai': 'opencode-go', 'api.x.ai': 'xai', 'integrate.api.nvidia.com': 'nvidia', 'api.xiaomimimo.com': 'xiaomi', 'xiaomimimo.com': 'xiaomi', 'api.gmi-serving.com': 'gmi', 'api.novita.ai': 'novita', 'tokenhub.tencentmaas.com': 'tencent-tokenhub', 'ollama.com': 'ollama-cloud'}

def _infer_provider_from_url(base_url: str) -> Optional[str]:
    """Infer the models.dev provider name from a base URL.

    This allows context length resolution via models.dev for custom endpoints
    like DashScope (Alibaba), Z.AI, Kimi, etc. without requiring the user to
    explicitly set the provider name in config.
    """
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return None
    parsed = urlparse(normalized if '://' in normalized else f'https://{normalized}')
    host = parsed.netloc.lower() or parsed.path.lower()
    for url_part, provider in _URL_TO_PROVIDER.items():
        if url_part in host:
            return provider
    return None

