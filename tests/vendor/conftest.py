"""Keep original provider tests away from real credentials and network."""
from pathlib import Path
import httpx
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.name == 'test_resolve_provider_client_cloud_adds_billing_origin_header':
            item.add_marker(pytest.mark.xfail(strict=True,
                reason='Source implementation sends KylinAgent; upstream test still expects HermesAgent'))


@pytest.fixture(autouse=True)
def isolated_account_home(tmp_path, monkeypatch):
    from kylin_memory._vendor.agent import auxiliary_client
    auxiliary_client._reset_aux_unhealthy_cache()
    auxiliary_client.clear_runtime_main()
    auxiliary_client._client_cache.clear()
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    def unexpected_network(*args, **kwargs):
        raise AssertionError('Unmocked network call in migrated source test')
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', unexpected_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', unexpected_network)
