import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)

def resolve_aws_auth_env_var(env: Optional[Dict[str, str]]=None) -> Optional[str]:
    """Return the name of the AWS auth source that is active, or None.

    Checks environment variables first, then falls back to boto3's credential
    chain for implicit sources (EC2 IMDS, ECS task role, etc.).

    This mirrors OpenClaw's ``resolveAwsSdkEnvVarName()`` — used to detect
    whether the user has any AWS credentials configured without actually
    attempting to authenticate.
    """
    env = env if env is not None else os.environ
    if env.get('AWS_BEARER_TOKEN_BEDROCK', '').strip():
        return 'AWS_BEARER_TOKEN_BEDROCK'
    if env.get('AWS_ACCESS_KEY_ID', '').strip() and env.get('AWS_SECRET_ACCESS_KEY', '').strip():
        return 'AWS_ACCESS_KEY_ID'
    if env.get('AWS_PROFILE', '').strip():
        return 'AWS_PROFILE'
    if env.get('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI', '').strip():
        return 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI'
    if env.get('AWS_WEB_IDENTITY_TOKEN_FILE', '').strip():
        return 'AWS_WEB_IDENTITY_TOKEN_FILE'
    try:
        import botocore.session
        session = botocore.session.get_session()
        credentials = session.get_credentials()
        if credentials is not None:
            resolved = credentials.get_frozen_credentials()
            if resolved and resolved.access_key:
                return 'iam-role'
    except Exception:
        pass
    return None

def has_aws_credentials(env: Optional[Dict[str, str]]=None) -> bool:
    """Return True if any AWS credential source is detected.

    Checks environment variables first (fast, no I/O), then falls back to
    boto3's credential chain which covers EC2 instance roles, ECS task roles,
    Lambda execution roles, and other IMDS-based sources that don't set
    environment variables.

    This two-tier approach mirrors the pattern from OpenClaw PR #62673:
    cloud environments (EC2, ECS, Lambda) provide credentials via instance
    metadata, not environment variables. The env-var check is a fast path
    for local development; the boto3 fallback covers all cloud deployments.
    """
    if resolve_aws_auth_env_var(env) is not None:
        return True
    try:
        import botocore.session
        session = botocore.session.get_session()
        credentials = session.get_credentials()
        if credentials is not None:
            resolved = credentials.get_frozen_credentials()
            if resolved and resolved.access_key:
                return True
    except Exception:
        pass
    return False

def resolve_bedrock_region(env: Optional[Dict[str, str]]=None) -> str:
    """Resolve the AWS region for Bedrock API calls.

    Priority:
      1. AWS_REGION env var
      2. AWS_DEFAULT_REGION env var
      3. boto3/botocore configured region (from ~/.aws/config or SSO profile)
      4. us-east-1 (hard fallback)

    The boto3 fallback is critical for EU/AP users who configure their region
    in ~/.aws/config via a named profile rather than env vars — without it,
    live model discovery would always return us.* profile IDs regardless of
    the user's actual region.
    """
    env = env if env is not None else os.environ
    explicit = env.get('AWS_REGION', '').strip() or env.get('AWS_DEFAULT_REGION', '').strip()
    if explicit:
        return explicit
    try:
        import botocore.session
        region = botocore.session.get_session().get_config_variable('region')
        if region:
            return region
    except Exception:
        pass
    return 'us-east-1'
_NON_TOOL_CALLING_PATTERNS = ['deepseek.r1', 'deepseek-r1', 'stability.', 'cohere.embed', 'amazon.titan-embed']

def _model_supports_tool_use(model_id: str) -> bool:
    """Return True if the model is expected to support tool/function calling.

    Models in the denylist are known to reject toolConfig in the Converse API.
    Unknown models default to True (assume tool support).
    """
    model_lower = model_id.lower()
    return not any((pattern in model_lower for pattern in _NON_TOOL_CALLING_PATTERNS))

def is_anthropic_bedrock_model(model_id: str) -> bool:
    """Return True if the model is an Anthropic Claude model on Bedrock.

    These models should use the AnthropicBedrock SDK path for full feature
    parity (prompt caching, thinking budgets, adaptive thinking).
    Non-Claude models use the Converse API path.

    Matches:
      - ``anthropic.claude-*`` (foundation model IDs)
      - ``us.anthropic.claude-*`` (US inference profiles)
      - ``global.anthropic.claude-*`` (global inference profiles)
      - ``eu.anthropic.claude-*`` (EU inference profiles)
    """
    model_lower = model_id.lower()
    for prefix in ('us.', 'global.', 'eu.', 'ap.', 'jp.'):
        if model_lower.startswith(prefix):
            model_lower = model_lower[len(prefix):]
            break
    return model_lower.startswith('anthropic.claude')

def convert_tools_to_converse(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI-format tool definitions to Bedrock Converse ``toolConfig``.

    OpenAI format::

        {"type": "function", "function": {"name": "...", "description": "...",
         "parameters": {"type": "object", "properties": {...}}}}

    Converse format::

        {"toolSpec": {"name": "...", "description": "...",
         "inputSchema": {"json": {"type": "object", "properties": {...}}}}}
    """
    if not tools:
        return []
    result = []
    for t in tools:
        fn = t.get('function', {})
        name = fn.get('name', '')
        description = fn.get('description', '')
        parameters = fn.get('parameters', {'type': 'object', 'properties': {}})
        result.append({'toolSpec': {'name': name, 'description': description, 'inputSchema': {'json': parameters}}})
    return result

def _convert_content_to_converse(content) -> List[Dict]:
    """Convert OpenAI message content (string or list) to Converse content blocks.

    Handles:
      - Plain text strings → [{"text": "..."}]
      - Content arrays with text/image_url parts → mixed text/image blocks

    Filters out empty text blocks — Bedrock's Converse API rejects messages
    where a text content block has an empty ``text`` field (ValidationException:
    "text content blocks must be non-empty"). Ref: issue #9486.
    """
    if content is None:
        return [{'text': ' '}]
    if isinstance(content, str):
        return [{'text': content}] if content.strip() else [{'text': ' '}]
    if isinstance(content, list):
        blocks = []
        for part in content:
            if isinstance(part, str):
                blocks.append({'text': part})
                continue
            if not isinstance(part, dict):
                continue
            part_type = part.get('type', '')
            if part_type == 'text':
                text = part.get('text', '')
                blocks.append({'text': text if text else ' '})
            elif part_type == 'image_url':
                image_url = part.get('image_url', {})
                url = image_url.get('url', '') if isinstance(image_url, dict) else ''
                if url.startswith('data:'):
                    header, _, data = url.partition(',')
                    media_type = 'image/jpeg'
                    if header.startswith('data:'):
                        mime_part = header[5:].split(';')[0]
                        if mime_part:
                            media_type = mime_part
                    blocks.append({'image': {'format': media_type.split('/')[-1] if '/' in media_type else 'jpeg', 'source': {'bytes': data}}})
                else:
                    blocks.append({'text': f'[Image: {url}]'})
        return blocks if blocks else [{'text': ' '}]
    return [{'text': str(content)}]

def convert_messages_to_converse(messages: List[Dict]) -> Tuple[Optional[List[Dict]], List[Dict]]:
    """Convert OpenAI-format messages to Bedrock Converse format.

    Returns ``(system_prompt, converse_messages)`` where:
      - ``system_prompt`` is a list of system content blocks (or None)
      - ``converse_messages`` is the conversation in Converse format

    Handles:
      - System messages → extracted as system prompt
      - User messages → ``{"role": "user", "content": [...]}``
      - Assistant messages → ``{"role": "assistant", "content": [...]}``
      - Tool calls → ``{"toolUse": {"toolUseId": ..., "name": ..., "input": ...}}``
      - Tool results → ``{"toolResult": {"toolUseId": ..., "content": [...]}}``

    Converse requires strict user/assistant alternation. Consecutive messages
    with the same role are merged into a single message.
    """
    system_blocks: List[Dict] = []
    converse_msgs: List[Dict] = []
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content')
        if role == 'system':
            if isinstance(content, str) and content.strip():
                system_blocks.append({'text': content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        system_blocks.append({'text': part.get('text', '')})
                    elif isinstance(part, str):
                        system_blocks.append({'text': part})
            continue
        if role == 'tool':
            tool_call_id = msg.get('tool_call_id', '')
            result_content = content if isinstance(content, str) else json.dumps(content)
            tool_result_block = {'toolResult': {'toolUseId': tool_call_id, 'content': [{'text': result_content}]}}
            if converse_msgs and converse_msgs[-1]['role'] == 'user':
                converse_msgs[-1]['content'].append(tool_result_block)
            else:
                converse_msgs.append({'role': 'user', 'content': [tool_result_block]})
            continue
        if role == 'assistant':
            content_blocks = []
            if isinstance(content, str) and content.strip():
                content_blocks.append({'text': content})
            elif isinstance(content, list):
                content_blocks.extend(_convert_content_to_converse(content))
            tool_calls = msg.get('tool_calls', [])
            for tc in tool_calls or []:
                fn = tc.get('function', {})
                args_str = fn.get('arguments', '{}')
                try:
                    args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args_dict = {}
                content_blocks.append({'toolUse': {'toolUseId': tc.get('id', ''), 'name': fn.get('name', ''), 'input': args_dict}})
            if not content_blocks:
                content_blocks = [{'text': ' '}]
            if converse_msgs and converse_msgs[-1]['role'] == 'assistant':
                converse_msgs[-1]['content'].extend(content_blocks)
            else:
                converse_msgs.append({'role': 'assistant', 'content': content_blocks})
            continue
        if role == 'user':
            content_blocks = _convert_content_to_converse(content)
            if converse_msgs and converse_msgs[-1]['role'] == 'user':
                converse_msgs[-1]['content'].extend(content_blocks)
            else:
                converse_msgs.append({'role': 'user', 'content': content_blocks})
            continue
    if converse_msgs and converse_msgs[0]['role'] != 'user':
        converse_msgs.insert(0, {'role': 'user', 'content': [{'text': ' '}]})
    if converse_msgs and converse_msgs[-1]['role'] != 'user':
        converse_msgs.append({'role': 'user', 'content': [{'text': ' '}]})
    return (system_blocks if system_blocks else None, converse_msgs)

def _converse_stop_reason_to_openai(stop_reason: str) -> str:
    """Map Bedrock Converse stop reasons to OpenAI finish_reason values."""
    mapping = {'end_turn': 'stop', 'stop_sequence': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length', 'content_filtered': 'content_filter', 'guardrail_intervened': 'content_filter'}
    return mapping.get(stop_reason, 'stop')

def normalize_converse_response(response: Dict) -> SimpleNamespace:
    """Convert a Bedrock Converse API response to an OpenAI-compatible object.

    The agent loop in ``run_agent.py`` expects responses shaped like
    ``openai.ChatCompletion`` — this function bridges the gap.

    Returns a SimpleNamespace with:
      - ``.choices[0].message.content`` — text response
      - ``.choices[0].message.tool_calls`` — tool call list (if any)
      - ``.choices[0].finish_reason`` — stop/tool_calls/length
      - ``.usage`` — token usage stats
    """
    output = response.get('output', {})
    message = output.get('message', {})
    content_blocks = message.get('content', [])
    stop_reason = response.get('stopReason', 'end_turn')
    text_parts = []
    reasoning_parts = []
    tool_calls = []
    for block in content_blocks:
        if 'text' in block:
            text_parts.append(block['text'])
        elif 'reasoningContent' in block:
            reasoning = block['reasoningContent']
            if isinstance(reasoning, dict):
                thinking_text = reasoning.get('text', '')
                if thinking_text:
                    reasoning_parts.append(str(thinking_text))
        elif 'toolUse' in block:
            tu = block['toolUse']
            tool_calls.append(SimpleNamespace(id=tu.get('toolUseId', ''), type='function', function=SimpleNamespace(name=tu.get('name', ''), arguments=json.dumps(tu.get('input', {})))))
    msg = SimpleNamespace(role='assistant', content='\n'.join(text_parts) if text_parts else None, tool_calls=tool_calls if tool_calls else None, reasoning_content='\n\n'.join(reasoning_parts) if reasoning_parts else None)
    usage_data = response.get('usage', {})
    usage = SimpleNamespace(prompt_tokens=usage_data.get('inputTokens', 0), completion_tokens=usage_data.get('outputTokens', 0), total_tokens=usage_data.get('inputTokens', 0) + usage_data.get('outputTokens', 0))
    finish_reason = _converse_stop_reason_to_openai(stop_reason)
    if tool_calls and finish_reason == 'stop':
        finish_reason = 'tool_calls'
    choice = SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, model=response.get('modelId', ''))

def build_converse_kwargs(model: str, messages: List[Dict], tools: Optional[List[Dict]]=None, max_tokens: int=4096, temperature: Optional[float]=None, top_p: Optional[float]=None, stop_sequences: Optional[List[str]]=None, guardrail_config: Optional[Dict]=None) -> Dict[str, Any]:
    """Build kwargs for ``bedrock-runtime.converse()`` or ``converse_stream()``.

    Converts OpenAI-format inputs to Converse API parameters.
    """
    system_prompt, converse_messages = convert_messages_to_converse(messages)
    kwargs: Dict[str, Any] = {'modelId': model, 'messages': converse_messages, 'inferenceConfig': {'maxTokens': max_tokens}}
    if system_prompt:
        kwargs['system'] = system_prompt
    if temperature is not None:
        kwargs['inferenceConfig']['temperature'] = temperature
    if top_p is not None:
        kwargs['inferenceConfig']['topP'] = top_p
    if stop_sequences:
        kwargs['inferenceConfig']['stopSequences'] = stop_sequences
    if tools:
        converse_tools = convert_tools_to_converse(tools)
        if converse_tools:
            if _model_supports_tool_use(model):
                kwargs['toolConfig'] = {'tools': converse_tools}
            else:
                logger.warning('Model %s does not support tool calling — tools stripped. The agent will operate in text-only mode.', model)
    if guardrail_config:
        kwargs['guardrailConfig'] = guardrail_config
    return kwargs

