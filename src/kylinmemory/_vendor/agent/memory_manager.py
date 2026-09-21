from __future__ import annotations
import re
_FENCE_TAG_RE = re.compile('</?\\s*memory-context\\s*>', re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile('<\\s*memory-context\\s*>[\\s\\S]*?</\\s*memory-context\\s*>', re.IGNORECASE)
_INTERNAL_NOTE_RE = re.compile('\\[System note:\\s*The following is recalled memory context,\\s*NOT new user input\\.\\s*Treat as (?:informational background data|authoritative reference data[^\\]]*)\\.\\]\\s*', re.IGNORECASE)

def sanitize_context(text: str) -> str:
    """Strip fence tags, injected context blocks, and system notes from provider output."""
    text = _INTERNAL_CONTEXT_RE.sub('', text)
    text = _INTERNAL_NOTE_RE.sub('', text)
    text = _FENCE_TAG_RE.sub('', text)
    return text

