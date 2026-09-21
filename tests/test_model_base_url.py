import httpx
import openai
import pytest

from kylinmemory.auxiliary_client import call_llm, resolve_runtime
from kylinmemory.config import DEFAULT_CONFIG, merge_config, runtime_context
from kylinmemory.vendor_bridge import RuntimeRouter


@pytest.mark.parametrize(('base', 'expected'), [
    ('https://example.test', 'https://example.test/v1'),
    ('http://localhost:8000/', 'http://localhost:8000/v1'),
    ('https://example.test?api-version=1', 'https://example.test/v1?api-version=1'),
    ('https://example.test/v1', 'https://example.test/v1'),
    ('https://example.test/v2/', 'https://example.test/v2/'),
    ('https://example.test/custom/v1', 'https://example.test/custom/v1'),
])
@pytest.mark.parametrize('mode', ['', 'chat_completions'])
def test_root_model_url_gets_version_prefix(tmp_path, base, expected, mode):
    cfg = merge_config(DEFAULT_CONFIG, {'model': {
        'provider': 'custom', 'base_url': base, 'api_mode': mode}})
    with runtime_context(tmp_path, cfg):
        assert resolve_runtime()['base_url'] == expected
        assert resolve_runtime(resolve_runtime())['base_url'] == expected
    assert cfg['model']['base_url'] == base


@pytest.mark.parametrize(('provider', 'mode', 'base'), [
    ('custom', 'anthropic_messages', 'https://example.test'),
    ('custom', 'codex_responses', 'https://example.test'),
    ('custom', '', 'https://api.anthropic.com'),
    ('gemini', '', 'https://example.test'),
])
def test_other_protocol_urls_are_preserved(tmp_path, provider, mode, base):
    with runtime_context(tmp_path, DEFAULT_CONFIG):
        assert resolve_runtime({'provider': provider, 'api_mode': mode,
                                'base_url': base})['base_url'] == base


def test_root_url_reaches_chat_endpoint_through_router(tmp_path, monkeypatch):
    requests, clients = [], []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion',
            'created': 0, 'model': 'test', 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})
    original = openai.OpenAI
    def create(**kwargs):
        client = original(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client
    monkeypatch.setattr(openai, 'OpenAI', create)
    cfg = merge_config(DEFAULT_CONFIG, {'model': {'provider': 'custom', 'model': 'test',
        'api_mode': 'chat_completions', 'base_url': 'https://example.test', 'api_key': 'test-key'}})
    router = RuntimeRouter()
    try:
        with runtime_context(tmp_path, cfg):
            result = call_llm(task='atom_memory', main_runtime={'_router': router},
                              messages=[{'role': 'user', 'content': 'hello'}])
        assert result.choices[0].message.content == 'ok'
        assert str(requests[0].url) == 'https://example.test/v1/chat/completions'
    finally:
        router.close()
        for client in clients:
            client.close()
