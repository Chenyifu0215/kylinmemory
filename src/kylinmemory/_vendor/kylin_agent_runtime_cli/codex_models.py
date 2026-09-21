from __future__ import annotations
from typing import List, Optional
DEFAULT_CODEX_MODELS: List[str] = ['gpt-5.5', 'gpt-5.4-mini', 'gpt-5.4', 'gpt-5.3-codex', 'gpt-5.3-codex-spark', 'gpt-5.2-codex', 'gpt-5.1-codex-max', 'gpt-5.1-codex-mini']
_FORWARD_COMPAT_TEMPLATE_MODELS: List[tuple[str, tuple[str, ...]]] = [('gpt-5.5', ('gpt-5.4', 'gpt-5.4-mini', 'gpt-5.3-codex')), ('gpt-5.4-mini', ('gpt-5.3-codex', 'gpt-5.2-codex')), ('gpt-5.4', ('gpt-5.3-codex', 'gpt-5.2-codex')), ('gpt-5.3-codex', ('gpt-5.2-codex',)), ('gpt-5.3-codex-spark', ('gpt-5.3-codex', 'gpt-5.2-codex'))]

def _add_forward_compat_models(model_ids: List[str]) -> List[str]:
    """Add Clawdbot-style synthetic forward-compat Codex models.

    If a newer Codex slug isn't returned by live discovery, surface it when an
    older compatible template model is present. This mirrors Clawdbot's
    synthetic catalog / forward-compat behavior for GPT-5 Codex variants.
    """
    ordered: List[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        if model_id not in seen:
            ordered.append(model_id)
            seen.add(model_id)
    for synthetic_model, template_models in _FORWARD_COMPAT_TEMPLATE_MODELS:
        if synthetic_model in seen:
            continue
        if any((template in seen for template in template_models)):
            ordered.append(synthetic_model)
            seen.add(synthetic_model)
    return ordered

