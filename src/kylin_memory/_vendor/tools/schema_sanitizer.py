from __future__ import annotations
import logging
from typing import Any
logger = logging.getLogger(__name__)

def strip_nullable_unions(schema: Any, *, keep_nullable_hint: bool=True) -> Any:
    """Collapse ``anyOf`` / ``oneOf`` nullable unions to the non-null branch.

    MCP / Pydantic optional fields commonly arrive as::

        {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}

    Anthropic's tool input-schema validator rejects the null branch. Tool
    optionality is already represented by the parent object's ``required``
    array, so we collapse the union to the single non-null variant.

    Metadata (``title``, ``description``, ``default``, ``examples``) on the
    outer union node is carried over to the replacement variant.

    Args:
        schema: JSON-Schema fragment (dict, list, or scalar).
        keep_nullable_hint: If True, set ``nullable: true`` on the replacement
            to preserve the "this field may be None" signal for downstream
            consumers that care (e.g. runtime argument coercion that maps the
            literal string ``"null"`` to Python ``None``). Anthropic's
            validator accepts ``nullable: true`` but strict producers may
            prefer False.

    Returns:
        The schema with nullable unions collapsed. Non-union nodes are
        returned unchanged.
    """
    if isinstance(schema, list):
        return [strip_nullable_unions(item, keep_nullable_hint=keep_nullable_hint) for item in schema]
    if not isinstance(schema, dict):
        return schema
    stripped = {k: strip_nullable_unions(v, keep_nullable_hint=keep_nullable_hint) for k, v in schema.items()}
    for key in ('anyOf', 'oneOf'):
        variants = stripped.get(key)
        if not isinstance(variants, list):
            continue
        non_null = [item for item in variants if not (isinstance(item, dict) and item.get('type') == 'null')]
        if len(non_null) == 1 and len(non_null) != len(variants):
            replacement = dict(non_null[0]) if isinstance(non_null[0], dict) else {}
            if keep_nullable_hint:
                replacement.setdefault('nullable', True)
            for meta_key in ('title', 'description', 'default', 'examples'):
                if meta_key in stripped and meta_key not in replacement:
                    replacement[meta_key] = stripped[meta_key]
            return strip_nullable_unions(replacement, keep_nullable_hint=keep_nullable_hint)
    return stripped
_STRIP_ON_RECOVERY_KEYS = frozenset({'pattern', 'format'})

def strip_pattern_and_format(tools: list[dict]) -> tuple[list[dict], int]:
    """Strip ``pattern`` and ``format`` JSON Schema keywords from tool schemas.

    This is a *reactive* sanitizer invoked only when llama.cpp's
    ``json-schema-to-grammar`` converter has rejected a tool schema with an
    HTTP 400 grammar-parse error.  llama.cpp's regex engine supports only a
    small subset of ECMAScript regex (literals, ``.``, ``[...]``, ``|``,
    ``*``, ``+``, ``?``, ``{n,m}``) — it rejects escape classes like ``\\d``,
    ``\\w``, ``\\s`` and most ``format`` values.  Cloud providers (OpenAI,
    Anthropic, OpenRouter, Gemini) accept these keywords fine and rely on
    them as prompting hints, so we keep them in the default schema and only
    strip on demand.

    The strip operates on a sibling of ``type`` (so schema keywords are
    removed) — a property literally *named* ``pattern`` (e.g. the first arg
    of the built-in ``search_files`` tool) is not affected because property
    names live in the ``properties`` dict, not as siblings of ``type``.

    Args:
        tools: OpenAI-format tool list, mutated in place for efficiency.
            Callers that need to preserve the original should deep-copy first.

    Returns:
        ``(tools, stripped_count)`` — the same list reference plus a count of
        how many ``pattern``/``format`` keywords were removed across all tools.
    """
    if not tools:
        return (tools, 0)
    stripped = 0

    def _walk(node: Any) -> None:
        nonlocal stripped
        if isinstance(node, dict):
            is_schema_node = 'type' in node or 'anyOf' in node or 'oneOf' in node or ('allOf' in node)
            for key in list(node.keys()):
                if is_schema_node and key in _STRIP_ON_RECOVERY_KEYS:
                    node.pop(key, None)
                    stripped += 1
                    continue
                _walk(node[key])
        elif isinstance(node, list):
            for item in node:
                _walk(item)
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get('function')
        if isinstance(fn, dict):
            params = fn.get('parameters')
            if isinstance(params, dict):
                _walk(params)
                continue
        params = tool.get('parameters')
        if isinstance(params, dict):
            _walk(params)
            continue
    if stripped:
        logger.info('schema_sanitizer: stripped %d pattern/format keyword(s) from tool schemas (llama.cpp grammar-parse recovery)', stripped)
    return (tools, stripped)

def strip_slash_enum(tools: list[dict]) -> tuple[list[dict], int]:
    """Strip ``enum`` keywords whose string values contain a forward slash.

    xAI's ``/v1/responses`` and ``/v1/chat/completions`` endpoints compile
    tool schemas to a grammar that rejects ``enum`` values containing ``/``
    (the request fails with HTTP 400 "Invalid arguments passed to the
    model" before any token is emitted). Most commonly hit by MCP-derived
    tools whose enum lists HuggingFace model IDs (``Qwen/Qwen3.5-0.8B``,
    ``openai/gpt-oss-20b``) or owner/name environment IDs. The constraint
    is purely a prompting hint; dropping it lets the model still see the
    field description and pick a value, without xAI tripping on the slash.

    Args:
        tools: OpenAI-format or Responses-format tool list, mutated in
            place. Callers that need to preserve the original should
            deep-copy first.

    Returns:
        ``(tools, stripped_count)`` — same list reference plus a count of
        how many ``enum`` keywords were removed.
    """
    if not tools:
        return (tools, 0)
    stripped = 0

    def _walk(node: Any) -> None:
        nonlocal stripped
        if isinstance(node, dict):
            enum_val = node.get('enum')
            if isinstance(enum_val, list) and any((isinstance(v, str) and '/' in v for v in enum_val)):
                node.pop('enum', None)
                stripped += 1
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get('function')
        if isinstance(fn, dict):
            params = fn.get('parameters')
            if isinstance(params, dict):
                _walk(params)
                continue
        params = tool.get('parameters')
        if isinstance(params, dict):
            _walk(params)
    if stripped:
        logger.info("schema_sanitizer: stripped %d enum keyword(s) containing '/' from tool schemas (xAI Responses grammar-compile recovery)", stripped)
    return (tools, stripped)

