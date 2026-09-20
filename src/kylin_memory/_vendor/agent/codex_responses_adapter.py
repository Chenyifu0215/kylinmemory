from __future__ import annotations
import hashlib
import json
import logging
import re
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from kylin_memory._vendor.agent.prompt_builder import DEFAULT_AGENT_IDENTITY
logger = logging.getLogger(__name__)
_TOOL_CALL_LEAK_PATTERN = re.compile('(?:^|[\\s>|])to=functions\\.[A-Za-z_][\\w.]*', re.IGNORECASE)

def _chat_content_to_responses_parts(content: Any, *, role: str='user') -> List[Dict[str, Any]]:
    """Convert chat-style multimodal content to Responses API input parts.

    Input:  ``[{"type":"text"|"image_url", ...}]`` (native OpenAI Chat format)
    Output: ``[{"type":"input_text"|"output_text"|"input_image", ...}]`` (Responses format)

    The ``role`` parameter controls the text content type:
    - ``"user"`` (default) → ``"input_text"``
    - ``"assistant"`` → ``"output_text"``

    The Responses API rejects ``input_text`` inside assistant messages and
    ``output_text`` inside user messages, so callers MUST pass the correct
    role for the message being converted.

    Returns an empty list when ``content`` is not a list or contains no
    recognized parts — callers fall back to the string path.
    """
    text_type = 'output_text' if role == 'assistant' else 'input_text'
    if not isinstance(content, list):
        return []
    converted: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            if part:
                converted.append({'type': text_type, 'text': part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = str(part.get('type') or '').strip().lower()
        if ptype in {'text', 'input_text', 'output_text'}:
            text = part.get('text')
            if isinstance(text, str) and text:
                converted.append({'type': text_type, 'text': text})
            continue
        if ptype in {'image_url', 'input_image'}:
            image_ref = part.get('image_url')
            detail = part.get('detail')
            if isinstance(image_ref, dict):
                url = image_ref.get('url')
                detail = image_ref.get('detail', detail)
            else:
                url = image_ref
            if not isinstance(url, str) or not url:
                continue
            image_part: Dict[str, Any] = {'type': 'input_image', 'image_url': url}
            if isinstance(detail, str) and detail.strip():
                image_part['detail'] = detail.strip()
            converted.append(image_part)
    return converted

def _deterministic_call_id(fn_name: str, arguments: str, index: int=0) -> str:
    """Generate a deterministic call_id from tool call content.

    Used as a fallback when the API doesn't provide a call_id.
    Deterministic IDs prevent cache invalidation — random UUIDs would
    make every API call's prefix unique, breaking OpenAI's prompt cache.
    """
    seed = f'{fn_name}:{arguments}:{index}'
    digest = hashlib.sha256(seed.encode('utf-8', errors='replace')).hexdigest()[:12]
    return f'call_{digest}'

def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
    """Split a stored tool id into (call_id, response_item_id)."""
    if not isinstance(raw_id, str):
        return (None, None)
    value = raw_id.strip()
    if not value:
        return (None, None)
    if '|' in value:
        call_id, response_item_id = value.split('|', 1)
        call_id = call_id.strip() or None
        response_item_id = response_item_id.strip() or None
        return (call_id, response_item_id)
    if value.startswith('fc_'):
        return (None, value)
    return (value, None)

def _derive_responses_function_call_id(call_id: str, response_item_id: Optional[str]=None) -> str:
    """Build a valid Responses `function_call.id` (must start with `fc_`)."""
    if isinstance(response_item_id, str):
        candidate = response_item_id.strip()
        if candidate.startswith('fc_'):
            return candidate
    source = (call_id or '').strip()
    if source.startswith('fc_'):
        return source
    if source.startswith('call_') and len(source) > len('call_'):
        return f"fc_{source[len('call_'):]}"
    sanitized = re.sub('[^A-Za-z0-9_-]', '', source)
    if sanitized.startswith('fc_'):
        return sanitized
    if sanitized.startswith('call_') and len(sanitized) > len('call_'):
        return f"fc_{sanitized[len('call_'):]}"
    if sanitized:
        return f'fc_{sanitized[:48]}'
    seed = source or str(response_item_id or '') or uuid.uuid4().hex
    digest = hashlib.sha1(seed.encode('utf-8')).hexdigest()[:24]
    return f'fc_{digest}'

def _responses_tools(tools: Optional[List[Dict[str, Any]]]=None) -> Optional[List[Dict[str, Any]]]:
    """Convert chat-completions tool schemas to Responses function-tool schemas."""
    if not tools:
        return None
    converted: List[Dict[str, Any]] = []
    for item in tools:
        fn = item.get('function', {}) if isinstance(item, dict) else {}
        name = fn.get('name')
        if not isinstance(name, str) or not name.strip():
            continue
        converted.append({'type': 'function', 'name': name, 'description': fn.get('description', ''), 'strict': False, 'parameters': fn.get('parameters', {'type': 'object', 'properties': {}})})
    return converted or None
_RESPONSE_MESSAGE_STATUSES = {'completed', 'incomplete', 'in_progress'}

def _normalize_responses_message_status(value: Any, *, default: str='completed') -> str:
    """Normalize a Responses assistant message status for replay.

    The API accepts completed/incomplete/in_progress on replayed assistant
    output messages.  Preserve those exactly (modulo case/hyphen spelling) so
    incomplete Codex continuation turns don't get falsely marked completed.
    """
    if isinstance(value, str):
        status = value.strip().lower().replace('-', '_').replace(' ', '_')
        if status in _RESPONSE_MESSAGE_STATUSES:
            return status
    return default

def _chat_messages_to_responses_input(messages: List[Dict[str, Any]], *, is_xai_responses: bool=False) -> List[Dict[str, Any]]:
    """Convert internal chat-style messages to Responses input items.

    ``is_xai_responses=True`` strips ``encrypted_content`` from replayed
    reasoning items.  xAI's OAuth/SuperGrok ``/v1/responses`` surface
    rejects encrypted reasoning blobs minted by prior turns: the request
    streams an ``error`` SSE frame before ``response.created`` and the
    OpenAI SDK collapses it into a generic stream-ordering error.  Native
    Codex (chatgpt.com backend-api) DOES accept replayed encrypted_content
    — keep the default off.
    """
    items: List[Dict[str, Any]] = []
    seen_item_ids: set = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        if role == 'system':
            continue
        if role in {'user', 'assistant'}:
            content = msg.get('content', '')
            if isinstance(content, list):
                content_parts = _chat_content_to_responses_parts(content, role=role)
                text_type = 'output_text' if role == 'assistant' else 'input_text'
                content_text = ''.join((p.get('text', '') for p in content_parts if p.get('type') == text_type))
            else:
                content_parts = []
                content_text = str(content) if content is not None else ''
            if role == 'assistant':
                codex_reasoning = msg.get('codex_reasoning_items')
                has_codex_reasoning = False
                if isinstance(codex_reasoning, list) and (not is_xai_responses):
                    for ri in codex_reasoning:
                        if isinstance(ri, dict) and ri.get('encrypted_content'):
                            item_id = ri.get('id')
                            if item_id and item_id in seen_item_ids:
                                continue
                            replay_item = {k: v for k, v in ri.items() if k != 'id'}
                            items.append(replay_item)
                            if item_id:
                                seen_item_ids.add(item_id)
                            has_codex_reasoning = True
                codex_message_items = msg.get('codex_message_items')
                replayed_message_items = 0
                if isinstance(codex_message_items, list):
                    for raw_item in codex_message_items:
                        if not isinstance(raw_item, dict):
                            continue
                        if raw_item.get('type') != 'message' or raw_item.get('role') != 'assistant':
                            continue
                        raw_content_parts = raw_item.get('content')
                        if not isinstance(raw_content_parts, list):
                            continue
                        normalized_content_parts = []
                        for part in raw_content_parts:
                            if not isinstance(part, dict):
                                continue
                            part_type = str(part.get('type') or '').strip()
                            if part_type not in {'output_text', 'text'}:
                                continue
                            text = part.get('text', '')
                            if text is None:
                                text = ''
                            if not isinstance(text, str):
                                text = str(text)
                            normalized_content_parts.append({'type': 'output_text', 'text': text})
                        if not normalized_content_parts:
                            continue
                        replay_item = {'type': 'message', 'role': 'assistant', 'status': _normalize_responses_message_status(raw_item.get('status')), 'content': normalized_content_parts}
                        item_id = raw_item.get('id')
                        if isinstance(item_id, str) and item_id.strip():
                            replay_item['id'] = item_id.strip()
                        phase = raw_item.get('phase')
                        if isinstance(phase, str) and phase.strip():
                            replay_item['phase'] = phase.strip()
                        items.append(replay_item)
                        replayed_message_items += 1
                if replayed_message_items > 0:
                    pass
                elif content_parts:
                    items.append({'role': 'assistant', 'content': content_parts})
                elif content_text.strip():
                    items.append({'role': 'assistant', 'content': content_text})
                elif has_codex_reasoning:
                    items.append({'role': 'assistant', 'content': ''})
                tool_calls = msg.get('tool_calls')
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get('function', {})
                        fn_name = fn.get('name')
                        if not isinstance(fn_name, str) or not fn_name.strip():
                            continue
                        embedded_call_id, embedded_response_item_id = _split_responses_tool_id(tc.get('id'))
                        call_id = tc.get('call_id')
                        if not isinstance(call_id, str) or not call_id.strip():
                            call_id = embedded_call_id
                        if not isinstance(call_id, str) or not call_id.strip():
                            if isinstance(embedded_response_item_id, str) and embedded_response_item_id.startswith('fc_') and (len(embedded_response_item_id) > len('fc_')):
                                call_id = f"call_{embedded_response_item_id[len('fc_'):]}"
                            else:
                                _raw_args = str(fn.get('arguments', '{}'))
                                call_id = _deterministic_call_id(fn_name, _raw_args, len(items))
                        call_id = call_id.strip()
                        arguments = fn.get('arguments', '{}')
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        elif not isinstance(arguments, str):
                            arguments = str(arguments)
                        arguments = arguments.strip() or '{}'
                        items.append({'type': 'function_call', 'call_id': call_id, 'name': fn_name, 'arguments': arguments})
                continue
            if content_parts:
                items.append({'role': role, 'content': content_parts})
            else:
                items.append({'role': role, 'content': content_text})
            continue
        if role == 'tool':
            raw_tool_call_id = msg.get('tool_call_id')
            call_id, _ = _split_responses_tool_id(raw_tool_call_id)
            if not isinstance(call_id, str) or not call_id.strip():
                if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                    call_id = raw_tool_call_id.strip()
            if not isinstance(call_id, str) or not call_id.strip():
                continue
            tool_content = msg.get('content')
            output_value: Any
            if isinstance(tool_content, list):
                converted = _chat_content_to_responses_parts(tool_content, role='user')
                if converted:
                    output_value = converted
                else:
                    output_value = ''
            else:
                output_value = str(tool_content or '')
            items.append({'type': 'function_call_output', 'call_id': call_id, 'output': output_value})
    return items

def _preflight_codex_input_items(raw_items: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise ValueError('Codex Responses input must be a list of input items.')
    normalized: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f'Codex Responses input[{idx}] must be an object.')
        item_type = item.get('type')
        if item_type == 'function_call':
            call_id = item.get('call_id')
            name = item.get('name')
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(f'Codex Responses input[{idx}] function_call is missing call_id.')
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f'Codex Responses input[{idx}] function_call is missing name.')
            arguments = item.get('arguments', '{}')
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = str(arguments)
            arguments = arguments.strip() or '{}'
            normalized.append({'type': 'function_call', 'call_id': call_id.strip(), 'name': name.strip(), 'arguments': arguments})
            continue
        if item_type == 'function_call_output':
            call_id = item.get('call_id')
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(f'Codex Responses input[{idx}] function_call_output is missing call_id.')
            output = item.get('output', '')
            if output is None:
                output = ''
            if isinstance(output, list):
                cleaned: List[Dict[str, Any]] = []
                for part in output:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get('type')
                    if ptype == 'input_text':
                        text = part.get('text')
                        if isinstance(text, str) and text:
                            cleaned.append({'type': 'input_text', 'text': text})
                    elif ptype == 'input_image':
                        url = part.get('image_url')
                        if isinstance(url, str) and url:
                            entry: Dict[str, Any] = {'type': 'input_image', 'image_url': url}
                            detail = part.get('detail')
                            if isinstance(detail, str) and detail.strip():
                                entry['detail'] = detail.strip()
                            cleaned.append(entry)
                normalized.append({'type': 'function_call_output', 'call_id': call_id.strip(), 'output': cleaned if cleaned else ''})
                continue
            if not isinstance(output, str):
                output = str(output)
            normalized.append({'type': 'function_call_output', 'call_id': call_id.strip(), 'output': output})
            continue
        if item_type == 'reasoning':
            encrypted = item.get('encrypted_content')
            if isinstance(encrypted, str) and encrypted:
                item_id = item.get('id')
                if isinstance(item_id, str) and item_id:
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)
                reasoning_item = {'type': 'reasoning', 'encrypted_content': encrypted}
                summary = item.get('summary')
                if isinstance(summary, list):
                    reasoning_item['summary'] = summary
                else:
                    reasoning_item['summary'] = []
                normalized.append(reasoning_item)
            continue
        if item_type == 'message':
            role = item.get('role')
            if role != 'assistant':
                raise ValueError(f"Codex Responses input[{idx}] message items must have role='assistant'.")
            content = item.get('content')
            if not isinstance(content, list):
                raise ValueError(f'Codex Responses input[{idx}] message item must have content list.')
            normalized_content = []
            for part_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    raise ValueError(f'Codex Responses input[{idx}] message content[{part_idx}] must be an object.')
                part_type = part.get('type')
                if part_type not in {'output_text', 'text'}:
                    raise ValueError(f'Codex Responses input[{idx}] message content[{part_idx}] has unsupported type {part_type!r}.')
                text = part.get('text', '')
                if text is None:
                    text = ''
                if not isinstance(text, str):
                    text = str(text)
                normalized_content.append({'type': 'output_text', 'text': text})
            if not normalized_content:
                raise ValueError(f'Codex Responses input[{idx}] message item must contain at least one text part.')
            normalized_item: Dict[str, Any] = {'type': 'message', 'role': 'assistant', 'status': _normalize_responses_message_status(item.get('status')), 'content': normalized_content}
            item_id = item.get('id')
            if isinstance(item_id, str) and item_id.strip():
                normalized_item['id'] = item_id.strip()
            phase = item.get('phase')
            if isinstance(phase, str) and phase.strip():
                normalized_item['phase'] = phase.strip()
            normalized.append(normalized_item)
            continue
        role = item.get('role')
        if role in {'user', 'assistant'}:
            content = item.get('content', '')
            if content is None:
                content = ''
            if isinstance(content, list):
                text_type = 'output_text' if role == 'assistant' else 'input_text'
                validated: List[Dict[str, Any]] = []
                for part_idx, part in enumerate(content):
                    if isinstance(part, str):
                        if part:
                            validated.append({'type': text_type, 'text': part})
                        continue
                    if not isinstance(part, dict):
                        raise ValueError(f'Codex Responses input[{idx}].content[{part_idx}] must be an object or string.')
                    ptype = str(part.get('type') or '').strip().lower()
                    if ptype in {'input_text', 'text', 'output_text'}:
                        text = part.get('text', '')
                        if not isinstance(text, str):
                            text = str(text or '')
                        validated.append({'type': text_type, 'text': text})
                    elif ptype in {'input_image', 'image_url'}:
                        image_ref = part.get('image_url', '')
                        detail = part.get('detail')
                        if isinstance(image_ref, dict):
                            url = image_ref.get('url', '')
                            detail = image_ref.get('detail', detail)
                        else:
                            url = image_ref
                        if not isinstance(url, str):
                            url = str(url or '')
                        image_part: Dict[str, Any] = {'type': 'input_image', 'image_url': url}
                        if isinstance(detail, str) and detail.strip():
                            image_part['detail'] = detail.strip()
                        validated.append(image_part)
                    else:
                        raise ValueError(f"Codex Responses input[{idx}].content[{part_idx}] has unsupported type {part.get('type')!r}.")
                normalized.append({'role': role, 'content': validated})
                continue
            if not isinstance(content, str):
                content = str(content)
            normalized.append({'role': role, 'content': content})
            continue
        raise ValueError(f'Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r}).')
    return normalized

def _preflight_codex_api_kwargs(api_kwargs: Any, *, allow_stream: bool=False) -> Dict[str, Any]:
    if not isinstance(api_kwargs, dict):
        raise ValueError('Codex Responses request must be a dict.')
    required = {'model', 'instructions', 'input'}
    missing = [key for key in required if key not in api_kwargs]
    if missing:
        raise ValueError(f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}.")
    model = api_kwargs.get('model')
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Codex Responses request 'model' must be a non-empty string.")
    model = model.strip()
    instructions = api_kwargs.get('instructions')
    if instructions is None:
        instructions = ''
    if not isinstance(instructions, str):
        instructions = str(instructions)
    instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY
    normalized_input = _preflight_codex_input_items(api_kwargs.get('input'))
    tools = api_kwargs.get('tools')
    normalized_tools = None
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("Codex Responses request 'tools' must be a list when provided.")
        normalized_tools = []
        for idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(f'Codex Responses tools[{idx}] must be an object.')
            if tool.get('type') != 'function':
                raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")
            name = tool.get('name')
            parameters = tool.get('parameters')
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f'Codex Responses tools[{idx}] is missing a valid name.')
            if not isinstance(parameters, dict):
                raise ValueError(f'Codex Responses tools[{idx}] is missing valid parameters.')
            description = tool.get('description', '')
            if description is None:
                description = ''
            if not isinstance(description, str):
                description = str(description)
            strict = tool.get('strict', False)
            if not isinstance(strict, bool):
                strict = bool(strict)
            normalized_tools.append({'type': 'function', 'name': name.strip(), 'description': description, 'strict': strict, 'parameters': parameters})
    store = api_kwargs.get('store', False)
    if store is not False:
        raise ValueError("Codex Responses contract requires 'store' to be false.")
    allowed_keys = {'model', 'instructions', 'input', 'tools', 'store', 'reasoning', 'include', 'max_output_tokens', 'temperature', 'tool_choice', 'parallel_tool_calls', 'prompt_cache_key', 'service_tier', 'extra_headers', 'extra_body'}
    normalized: Dict[str, Any] = {'model': model, 'instructions': instructions, 'input': normalized_input, 'store': False}
    if normalized_tools is not None:
        normalized['tools'] = normalized_tools
    reasoning = api_kwargs.get('reasoning')
    if isinstance(reasoning, dict):
        normalized['reasoning'] = reasoning
    include = api_kwargs.get('include')
    if isinstance(include, list):
        normalized['include'] = include
    service_tier = api_kwargs.get('service_tier')
    if isinstance(service_tier, str) and service_tier.strip():
        normalized['service_tier'] = service_tier.strip()
    max_output_tokens = api_kwargs.get('max_output_tokens')
    if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
        normalized['max_output_tokens'] = int(max_output_tokens)
    temperature = api_kwargs.get('temperature')
    if isinstance(temperature, (int, float)):
        normalized['temperature'] = float(temperature)
    for passthrough_key in ('tool_choice', 'parallel_tool_calls', 'prompt_cache_key'):
        val = api_kwargs.get(passthrough_key)
        if val is not None:
            normalized[passthrough_key] = val
    extra_headers = api_kwargs.get('extra_headers')
    if extra_headers is not None:
        if not isinstance(extra_headers, dict):
            raise ValueError("Codex Responses request 'extra_headers' must be an object.")
        normalized_headers: Dict[str, str] = {}
        for key, value in extra_headers.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Codex Responses request 'extra_headers' keys must be non-empty strings.")
            if value is None:
                continue
            normalized_headers[key.strip()] = str(value)
        if normalized_headers:
            normalized['extra_headers'] = normalized_headers
    extra_body = api_kwargs.get('extra_body')
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise ValueError("Codex Responses request 'extra_body' must be an object.")
        if extra_body:
            normalized['extra_body'] = dict(extra_body)
    if allow_stream:
        stream = api_kwargs.get('stream')
        if stream is not None and stream is not True:
            raise ValueError("Codex Responses 'stream' must be true when set.")
        if stream is True:
            normalized['stream'] = True
        allowed_keys.add('stream')
    elif 'stream' in api_kwargs:
        raise ValueError('Codex Responses stream flag is only allowed in fallback streaming requests.')
    unexpected = sorted((key for key in api_kwargs if key not in allowed_keys))
    if unexpected:
        raise ValueError(f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}.")
    return normalized

def _extract_responses_message_text(item: Any) -> str:
    """Extract assistant text from a Responses message output item."""
    content = getattr(item, 'content', None)
    if not isinstance(content, list):
        return ''
    chunks: List[str] = []
    for part in content:
        ptype = getattr(part, 'type', None)
        if ptype not in {'output_text', 'text'}:
            continue
        text = getattr(part, 'text', None)
        if isinstance(text, str) and text:
            chunks.append(text)
    return ''.join(chunks).strip()

def _extract_responses_reasoning_text(item: Any) -> str:
    """Extract a compact reasoning text from a Responses reasoning item."""
    summary = getattr(item, 'summary', None)
    if isinstance(summary, list):
        chunks: List[str] = []
        for part in summary:
            text = getattr(part, 'text', None)
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return '\n'.join(chunks).strip()
    text = getattr(item, 'text', None)
    if isinstance(text, str) and text:
        return text.strip()
    return ''

def _normalize_codex_response(response: Any) -> tuple[Any, str]:
    """Normalize a Responses API object to an assistant_message-like object."""
    output = getattr(response, 'output', None)
    if not isinstance(output, list) or not output:
        out_text = getattr(response, 'output_text', None)
        if isinstance(out_text, str) and out_text.strip():
            logger.debug('Codex response has empty output but output_text is present (%d chars); synthesizing output item.', len(out_text.strip()))
            output = [SimpleNamespace(type='message', role='assistant', status='completed', content=[SimpleNamespace(type='output_text', text=out_text.strip())])]
            response.output = output
        else:
            raise RuntimeError('Responses API returned no output items')
    response_status = getattr(response, 'status', None)
    if isinstance(response_status, str):
        response_status = response_status.strip().lower()
    else:
        response_status = None
    if response_status in {'failed', 'cancelled'}:
        error_obj = getattr(response, 'error', None)
        if isinstance(error_obj, dict):
            error_msg = error_obj.get('message') or str(error_obj)
        else:
            error_msg = str(error_obj) if error_obj else f"Responses API returned status '{response_status}'"
        raise RuntimeError(error_msg)
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    reasoning_items_raw: List[Dict[str, Any]] = []
    message_items_raw: List[Dict[str, Any]] = []
    tool_calls: List[Any] = []
    has_incomplete_items = response_status in {'queued', 'in_progress', 'incomplete'}
    saw_commentary_phase = False
    saw_final_answer_phase = False
    for item in output:
        item_type = getattr(item, 'type', None)
        item_status = getattr(item, 'status', None)
        if isinstance(item_status, str):
            item_status = item_status.strip().lower()
        else:
            item_status = None
        if item_status in {'queued', 'in_progress', 'incomplete'}:
            has_incomplete_items = True
        if item_type == 'message':
            item_phase = getattr(item, 'phase', None)
            normalized_phase = None
            if isinstance(item_phase, str):
                normalized_phase = item_phase.strip().lower()
                if normalized_phase in {'commentary', 'analysis'}:
                    saw_commentary_phase = True
                elif normalized_phase in {'final_answer', 'final'}:
                    saw_final_answer_phase = True
            message_text = _extract_responses_message_text(item)
            if message_text:
                content_parts.append(message_text)
                raw_message_item: Dict[str, Any] = {'type': 'message', 'role': 'assistant', 'status': _normalize_responses_message_status(item_status), 'content': [{'type': 'output_text', 'text': message_text}]}
                item_id = getattr(item, 'id', None)
                if isinstance(item_id, str) and item_id:
                    raw_message_item['id'] = item_id
                if normalized_phase:
                    raw_message_item['phase'] = normalized_phase
                message_items_raw.append(raw_message_item)
        elif item_type == 'reasoning':
            reasoning_text = _extract_responses_reasoning_text(item)
            if reasoning_text:
                reasoning_parts.append(reasoning_text)
            encrypted = getattr(item, 'encrypted_content', None)
            if isinstance(encrypted, str) and encrypted:
                raw_item = {'type': 'reasoning', 'encrypted_content': encrypted}
                item_id = getattr(item, 'id', None)
                if isinstance(item_id, str) and item_id:
                    raw_item['id'] = item_id
                summary = getattr(item, 'summary', None)
                if isinstance(summary, list):
                    raw_summary = []
                    for part in summary:
                        text = getattr(part, 'text', None)
                        if isinstance(text, str):
                            raw_summary.append({'type': 'summary_text', 'text': text})
                    raw_item['summary'] = raw_summary
                reasoning_items_raw.append(raw_item)
        elif item_type == 'function_call':
            if item_status in {'queued', 'in_progress', 'incomplete'}:
                continue
            fn_name = getattr(item, 'name', '') or ''
            arguments = getattr(item, 'arguments', '{}')
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            raw_call_id = getattr(item, 'call_id', None)
            raw_item_id = getattr(item, 'id', None)
            embedded_call_id, _ = _split_responses_tool_id(raw_item_id)
            call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = _deterministic_call_id(fn_name, arguments, len(tool_calls))
            call_id = call_id.strip()
            response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
            response_item_id = _derive_responses_function_call_id(call_id, response_item_id)
            tool_calls.append(SimpleNamespace(id=call_id, call_id=call_id, response_item_id=response_item_id, type='function', function=SimpleNamespace(name=fn_name, arguments=arguments)))
        elif item_type == 'custom_tool_call':
            fn_name = getattr(item, 'name', '') or ''
            arguments = getattr(item, 'input', '{}')
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            raw_call_id = getattr(item, 'call_id', None)
            raw_item_id = getattr(item, 'id', None)
            embedded_call_id, _ = _split_responses_tool_id(raw_item_id)
            call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = _deterministic_call_id(fn_name, arguments, len(tool_calls))
            call_id = call_id.strip()
            response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
            response_item_id = _derive_responses_function_call_id(call_id, response_item_id)
            tool_calls.append(SimpleNamespace(id=call_id, call_id=call_id, response_item_id=response_item_id, type='function', function=SimpleNamespace(name=fn_name, arguments=arguments)))
    final_text = '\n'.join([p for p in content_parts if p]).strip()
    if not final_text and hasattr(response, 'output_text'):
        out_text = getattr(response, 'output_text', '')
        if isinstance(out_text, str):
            final_text = out_text.strip()
    leaked_tool_call_text = False
    if final_text and (not tool_calls) and _TOOL_CALL_LEAK_PATTERN.search(final_text):
        leaked_tool_call_text = True
        logger.warning('Codex response contains leaked tool-call text in assistant content (no structured function_call items). Treating as incomplete so the continuation path can re-elicit a proper tool call. Leaked snippet: %r', final_text[:300])
        final_text = ''
    assistant_message = SimpleNamespace(content=final_text, tool_calls=tool_calls, reasoning='\n\n'.join(reasoning_parts).strip() if reasoning_parts else None, reasoning_content=None, reasoning_details=None, codex_reasoning_items=reasoning_items_raw or None, codex_message_items=message_items_raw or None)
    if tool_calls:
        finish_reason = 'tool_calls'
    elif leaked_tool_call_text:
        finish_reason = 'incomplete'
    elif has_incomplete_items or (saw_commentary_phase and (not saw_final_answer_phase)):
        finish_reason = 'incomplete'
    elif reasoning_items_raw and (not final_text):
        finish_reason = 'incomplete'
    else:
        finish_reason = 'stop'
    return (assistant_message, finish_reason)

