import json
import math

from kylin_memory.memory_layers import OpenAICompatibleEmbedding, create_atom_embedding
from kylin_memory.config import DEFAULT_CONFIG


class _Response:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._body


def test_infer_embedding_request_and_nested_response(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        payload = json.loads(request.data.decode("utf-8"))
        vector = [3.0, 4.0] if payload["input"]["text"] == "first" else [5.0, 12.0]
        return _Response({"output": {"data": [{"embedding": vector, "index": 0}]}})

    monkeypatch.setattr("kylin_memory.memory_layers.urllib.request.urlopen", fake_urlopen)
    embedding = OpenAICompatibleEmbedding(
        model="qwen-embedding",
        dimensions="auto",
        base_url="http://127.0.0.1:18080",
        timeout=3,
    )

    vectors = embedding.embed_documents(["first", "second"])

    assert len(requests) == 2
    assert all(request.full_url == "http://127.0.0.1:18080/v1/infer" for request, _ in requests)
    assert [json.loads(request.data) for request, _ in requests] == [
        {"model": "qwen-embedding", "input": {"text": "first"}},
        {"model": "qwen-embedding", "input": {"text": "second"}},
    ]
    assert [timeout for _, timeout in requests] == [3, 3]
    assert vectors == [[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]]
    assert embedding.dimensions == 2
    assert all(math.isclose(sum(value * value for value in vector), 1.0) for vector in vectors)


def test_infer_embedding_accepts_full_endpoint():
    embedding = OpenAICompatibleEmbedding(
        model="qwen-embedding",
        dimensions=1024,
        base_url="http://127.0.0.1:18080/v1/infer",
    )

    assert embedding.endpoint == "http://127.0.0.1:18080/v1/infer"


def test_default_atom_embedding_config_can_use_anonymous_infer_service():
    config = DEFAULT_CONFIG["memory"]["atom"]["embedding"]
    embedding = create_atom_embedding(config)

    assert embedding is not None
    assert embedding.model == "qwen-embedding"
    assert embedding.endpoint == "http://127.0.0.1:18080/v1/infer"
    assert embedding.api_key == ""


def test_legacy_default_config_uses_current_infer_service_before_migration():
    embedding = create_atom_embedding({
        "mode": "remote",
        "model": "Qwen3-Embedding-0.6B-Q8_0",
        "dimensions": "auto",
        "base_url": "http://127.0.0.1:22370/v1/embeddings",
        "api_key_env": "",
    })

    assert embedding is not None
    assert embedding.model == "qwen-embedding"
    assert embedding.endpoint == "http://127.0.0.1:18080/v1/infer"
