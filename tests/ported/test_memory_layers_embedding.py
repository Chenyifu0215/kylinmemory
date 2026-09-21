import json
import math

import pytest

from kylinmemory.memory_layers import (
    Atom, AtomStore, EmbeddingError, OpenAICompatibleEmbedding, create_atom_embedding,
)
from kylinmemory.config import DEFAULT_CONFIG


class _Response:
    def __init__(self, body):
        self._body = json.dumps(body).encode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._body


def test_embedding_batches_reorder_response_and_authenticate(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        texts = json.loads(request.data)['input']
        return _Response({'data': [
            {'index': i, 'embedding': [3, 4] if t == 'first' else [5, 12]}
            for i, t in reversed(list(enumerate(texts)))
        ]})

    monkeypatch.setattr('kylinmemory.memory_layers.urllib.request.urlopen', fake_urlopen)
    embedding = OpenAICompatibleEmbedding(model='test-model', dimensions='auto',
        base_url='https://example.test/v1', api_key='test-key', batch_size=2, timeout=3)
    vectors = embedding.embed_documents(['first', 'second', 'third'])
    assert len(requests) == 2
    assert [json.loads(r.data) for r, _ in requests] == [
        {'model': 'test-model', 'input': ['first', 'second'], 'encoding_format': 'float'},
        {'model': 'test-model', 'input': ['third'], 'encoding_format': 'float'},
    ]
    assert all(r.full_url == 'https://example.test/v1/embeddings' for r, _ in requests)
    assert all(r.get_header('Authorization') == 'Bearer test-key' for r, _ in requests)
    assert all(t == 3 for _, t in requests)
    assert vectors == [[0.6, 0.8], [5/13, 12/13], [5/13, 12/13]]
    assert embedding.dimensions == 2
    assert all(math.isclose(sum(v*v for v in vector), 1) for vector in vectors)
    assert embedding.embed_documents([]) == []
    assert len(requests) == 2


@pytest.mark.parametrize(('base', 'endpoint'), [
    ('https://example.test', 'https://example.test/v1/embeddings'),
    ('https://example.test/v1/', 'https://example.test/v1/embeddings'),
    ('https://example.test/v1/embeddings', 'https://example.test/v1/embeddings'),
    ('https://example.test/compatible-mode/v1', 'https://example.test/compatible-mode/v1/embeddings'),
    ('https://example.test/custom/embeddings/?api-version=1', 'https://example.test/custom/embeddings?api-version=1'),
])
def test_embedding_endpoints(base, endpoint):
    e = OpenAICompatibleEmbedding(model='test', dimensions='auto', base_url=base)
    assert e.endpoint == endpoint


@pytest.mark.parametrize('data', [
    [],
    [{'index': 0, 'embedding': [1, 0]}, {'index': 0, 'embedding': [0, 1]}],
    [{'index': -1, 'embedding': [1, 0]}, {'index': 1, 'embedding': [0, 1]}],
    [{'index': True, 'embedding': [1, 0]}, {'index': 0, 'embedding': [0, 1]}],
    [{'embedding': [1, 0]}, {'index': 1, 'embedding': [0, 1]}],
    [{'index': 0, 'embedding': [1, 0]}, {'index': 1, 'embedding': [0, 1, 2]}],
    [{'index': 0, 'embedding': [1, 0]}, {'index': 1, 'embedding': [0, 0]}],
    [{'index': 0, 'embedding': [1, 0]}, {'index': 1, 'embedding': [float('nan'), 1]}],
    [{'index': 0, 'embedding': 'base64'}, {'index': 1, 'embedding': [0, 1]}],
])
def test_invalid_response_does_not_set_auto_dimension(monkeypatch, data):
    monkeypatch.setattr('kylinmemory.memory_layers.urllib.request.urlopen',
                        lambda *a, **k: _Response({'data': data}))
    e = OpenAICompatibleEmbedding(model='test', dimensions='auto', base_url='https://example.test')
    with pytest.raises(EmbeddingError):
        e.embed_documents(['first', 'second'])
    assert e.dimensions is None


def test_fixed_dimensions_validate_without_requesting_resize(monkeypatch):
    def respond(request, **kwargs):
        assert 'dimensions' not in json.loads(request.data)
        return _Response({'data': [{'index': 0, 'embedding': [3, 4]}]})
    monkeypatch.setattr('kylinmemory.memory_layers.urllib.request.urlopen', respond)
    e = OpenAICompatibleEmbedding(model='test', dimensions=3, base_url='https://example.test/v1')
    with pytest.raises(EmbeddingError, match='dimension mismatch'):
        e.embed_query('hello')


def test_default_embedding_credentials(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    e = create_atom_embedding(DEFAULT_CONFIG['memory']['atom']['embedding'])
    assert e.model == 'text-embedding-3-small'
    assert e.endpoint == 'https://api.openai.com/v1/embeddings'
    assert e.api_key == 'test-key'


def test_anonymous_endpoint_and_no_model_rewrite():
    e = create_atom_embedding({'mode': 'remote', 'model': 'Qwen3-Embedding-0.6B-Q8_0',
        'base_url': 'http://127.0.0.1:22370/v1/embeddings', 'api_key_env': ''})
    assert e.model == 'Qwen3-Embedding-0.6B-Q8_0'
    assert e.endpoint == 'http://127.0.0.1:22370/v1/embeddings'
    assert e.api_key == ''
    assert e.dimensions is None


def test_standard_embeddings_drive_vector_recall_without_lexical_match(monkeypatch, tmp_path):
    monkeypatch.setattr('kylinmemory.memory_layers.urllib.request.urlopen',
        lambda *a, **k: _Response({'data': [{'index': 0, 'embedding': [3, 4]}]}))
    store = AtomStore(tmp_path, embedding={'mode': 'remote', 'model': 'test',
        'base_url': 'https://example.test/v1', 'api_key_env': '', 'dimensions': 'auto'})
    try:
        assert store._sqlite_vec is not None
        store.upsert(Atom(id='tea', content='用户喜欢喝绿茶。', type='persona', priority=50))
        assert store._search_fts('饮品偏好') == []
        result = store._search_vector('饮品偏好')
        assert result[0][0].id == 'tea'
        assert result[0][1] == pytest.approx(1)
    finally:
        store.close()
