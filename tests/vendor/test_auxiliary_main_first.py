"""Regression tests for the ``auto`` → main-model-first policy.

Prior to this change, aggregator users (OpenRouter / Nous Portal) had aux
tasks routed through a cheap provider-side default (Gemini Flash) while
non-aggregator users got their main model.  This made behavior inconsistent
and surprising — users picked Claude but got Gemini Flash summaries.

The current policy: ``auto`` means "use my main chat model" for every user,
regardless of provider type.  Explicit per-task overrides in ``config.yaml``
(``auxiliary.<task>.provider``) still win.  The cheap fallback chain only
runs when the main provider has no working client.
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest

class TestResolveAutoMainFirst:
    """_resolve_auto() must prefer main provider + main model for every user."""

    def test_openrouter_main_uses_main_model_for_aux(self, monkeypatch):
        """OpenRouter main user → aux uses their picked OR model, not Gemini Flash."""
        monkeypatch.setenv('OPENROUTER_API_KEY', 'or-test-key')
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='openrouter'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='anthropic/claude-sonnet-4.6'), patch('kylinmemory._vendor.agent.auxiliary_client.resolve_provider_client') as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, 'anthropic/claude-sonnet-4.6')
            from kylinmemory._vendor.agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
        assert client is mock_client
        assert model == 'anthropic/claude-sonnet-4.6'
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.args[0] == 'openrouter'
        assert mock_resolve.call_args.args[1] == 'anthropic/claude-sonnet-4.6'

    def test_nous_main_uses_main_model_for_aux(self, monkeypatch):
        """Nous Portal main user → aux uses their picked Nous model, not free-tier MiMo."""
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='nous'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='anthropic/claude-opus-4.6'), patch('kylinmemory._vendor.agent.auxiliary_client.resolve_provider_client') as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, 'anthropic/claude-opus-4.6')
            from kylinmemory._vendor.agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
        assert client is mock_client
        assert model == 'anthropic/claude-opus-4.6'
        assert mock_resolve.call_args.args[0] == 'nous'

    def test_non_aggregator_main_still_uses_main(self, monkeypatch):
        """Non-aggregator main (DeepSeek) → unchanged behavior, main model used."""
        monkeypatch.setenv('DEEPSEEK_API_KEY', 'ds-test')
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='deepseek'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='deepseek-chat'), patch('kylinmemory._vendor.agent.auxiliary_client.resolve_provider_client') as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, 'deepseek-chat')
            from kylinmemory._vendor.agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
        assert client is mock_client
        assert model == 'deepseek-chat'
        assert mock_resolve.call_args.args[0] == 'deepseek'

    def test_main_unavailable_falls_through_to_chain(self, monkeypatch):
        """Main provider with no working client → fall back to aux chain."""
        monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
        chain_client = MagicMock()
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='anthropic'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='claude-opus'), patch('kylinmemory._vendor.agent.auxiliary_client.resolve_provider_client', return_value=(None, None)), patch('kylinmemory._vendor.agent.auxiliary_client._try_openrouter', return_value=(chain_client, 'google/gemini-3-flash-preview')):
            from kylinmemory._vendor.agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
        assert client is chain_client
        assert model == 'google/gemini-3-flash-preview'

    def test_no_main_config_uses_chain_directly(self):
        """No main provider configured → skip step 1, use chain (no regression)."""
        chain_client = MagicMock()
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value=''), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value=''), patch('kylinmemory._vendor.agent.auxiliary_client._try_openrouter', return_value=(chain_client, 'google/gemini-3-flash-preview')):
            from kylinmemory._vendor.agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
        assert client is chain_client

    def test_runtime_override_wins_over_config(self, monkeypatch):
        """main_runtime kwarg overrides config-read main provider/model."""
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='openrouter'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='config-model'), patch('kylinmemory._vendor.agent.auxiliary_client.resolve_provider_client') as mock_resolve:
            mock_resolve.return_value = (MagicMock(), 'runtime-model')
            from kylinmemory._vendor.agent.auxiliary_client import _resolve_auto
            _resolve_auto(main_runtime={'provider': 'anthropic', 'model': 'runtime-model', 'base_url': '', 'api_key': '', 'api_mode': ''})
        assert mock_resolve.call_args.args[0] == 'anthropic'
        assert mock_resolve.call_args.args[1] == 'runtime-model'

class TestResolveVisionMainFirst:
    """Vision auto-detection prefers the main provider first."""

    def test_openrouter_main_vision_uses_main_model(self, monkeypatch):
        """OpenRouter main with vision-capable model → aux vision uses main model."""
        monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='openrouter'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='anthropic/claude-sonnet-4.6'), patch('kylinmemory._vendor.agent.auxiliary_client.resolve_provider_client') as mock_resolve, patch('kylinmemory._vendor.agent.auxiliary_client._resolve_task_provider_model', return_value=('auto', None, None, None, None)):
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, 'anthropic/claude-sonnet-4.6')
            from kylinmemory._vendor.agent.auxiliary_client import resolve_vision_provider_client
            provider, client, model = resolve_vision_provider_client()
        assert provider == 'openrouter'
        assert client is mock_client
        assert model == 'anthropic/claude-sonnet-4.6'
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.args[0] == 'openrouter'
        assert mock_resolve.call_args.args[1] == 'anthropic/claude-sonnet-4.6'
        assert mock_resolve.call_args.kwargs.get('is_vision') is True

    def test_nous_main_vision_uses_paid_nous_vision_backend(self):
        """Paid Nous main → aux vision uses the dedicated Nous vision backend."""
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='nous'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='openai/gpt-5'), patch('kylinmemory._vendor.agent.auxiliary_client._resolve_task_provider_model', return_value=('auto', None, None, None, None)), patch('kylinmemory._vendor.agent.auxiliary_client._resolve_strict_vision_backend', return_value=(MagicMock(), 'google/gemini-3-flash-preview')):
            from kylinmemory._vendor.agent.auxiliary_client import resolve_vision_provider_client
            provider, client, model = resolve_vision_provider_client()
        assert provider == 'nous'
        assert client is not None
        assert model == 'google/gemini-3-flash-preview'

    def test_nous_main_vision_uses_free_tier_nous_vision_backend(self):
        """Free-tier Nous main → aux vision uses MiMo omni, not the text main model."""
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='nous'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='xiaomi/mimo-v2-pro'), patch('kylinmemory._vendor.agent.auxiliary_client._resolve_task_provider_model', return_value=('auto', None, None, None, None)), patch('kylinmemory._vendor.agent.auxiliary_client._resolve_strict_vision_backend', return_value=(MagicMock(), 'xiaomi/mimo-v2-omni')):
            from kylinmemory._vendor.agent.auxiliary_client import resolve_vision_provider_client
            provider, client, model = resolve_vision_provider_client()
        assert provider == 'nous'
        assert client is not None
        assert model == 'xiaomi/mimo-v2-omni'

    def test_exotic_provider_with_vision_override_preserved(self):
        """xiaomi → mimo-v2.5 override still wins over main_model."""
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='xiaomi'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='mimo-v2-pro'), patch('kylinmemory._vendor.agent.auxiliary_client.resolve_provider_client') as mock_resolve, patch('kylinmemory._vendor.agent.auxiliary_client._resolve_task_provider_model', return_value=('auto', None, None, None, None)):
            mock_resolve.return_value = (MagicMock(), 'mimo-v2.5')
            from kylinmemory._vendor.agent.auxiliary_client import resolve_vision_provider_client
            provider, client, model = resolve_vision_provider_client()
        assert provider == 'xiaomi'
        assert mock_resolve.call_args.args[1] == 'mimo-v2.5'
        assert mock_resolve.call_args.kwargs.get('is_vision') is True

    def test_copilot_vision_sets_vision_header(self, monkeypatch):
        """Copilot vision requests include the header required for vision routing."""
        monkeypatch.setenv('COPILOT_GITHUB_TOKEN', 'ghu_test-token')
        captured = {}

        def fake_headers(*, is_agent_turn=False, is_vision=False):
            captured['is_agent_turn'] = is_agent_turn
            captured['is_vision'] = is_vision
            return {'Copilot-Vision-Request': 'true'} if is_vision else {}
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='copilot'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='configured-copilot-model'), patch('kylinmemory._vendor.agent.auxiliary_client._resolve_task_provider_model', return_value=('auto', None, None, None, None)), patch('kylinmemory._vendor.agent.auxiliary_client.OpenAI') as mock_openai, patch('kylinmemory._vendor.kylin_agent_runtime_cli.auth.resolve_api_key_provider_credentials', return_value={'provider': 'copilot', 'api_key': 'copilot-api-token', 'base_url': 'https://api.githubcopilot.com'}), patch('kylinmemory._vendor.kylin_agent_runtime_cli.copilot_auth.copilot_request_headers', side_effect=fake_headers):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            from kylinmemory._vendor.agent.auxiliary_client import resolve_vision_provider_client
            provider, client, model = resolve_vision_provider_client()
        assert provider == 'copilot'
        assert client is mock_client
        assert model == 'configured-copilot-model'
        assert captured == {'is_agent_turn': True, 'is_vision': True}
        assert mock_openai.call_args.kwargs['default_headers']['Copilot-Vision-Request'] == 'true'

    def test_text_copilot_does_not_set_vision_header(self, monkeypatch):
        """Text Copilot requests keep the vision-only header off."""
        monkeypatch.setenv('COPILOT_GITHUB_TOKEN', 'ghu_test-token')
        captured = {}

        def fake_headers(*, is_agent_turn=False, is_vision=False):
            captured['is_agent_turn'] = is_agent_turn
            captured['is_vision'] = is_vision
            return {'Copilot-Vision-Request': 'true'} if is_vision else {}
        with patch('kylinmemory._vendor.agent.auxiliary_client.OpenAI') as mock_openai, patch('kylinmemory._vendor.kylin_agent_runtime_cli.auth.resolve_api_key_provider_credentials', return_value={'provider': 'copilot', 'api_key': 'copilot-api-token', 'base_url': 'https://api.githubcopilot.com'}), patch('kylinmemory._vendor.kylin_agent_runtime_cli.copilot_auth.copilot_request_headers', side_effect=fake_headers):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            from kylinmemory._vendor.agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client('copilot', 'gpt-5-mini')
        assert client is mock_client
        assert model == 'gpt-5-mini'
        assert captured == {'is_agent_turn': True, 'is_vision': False}
        assert 'default_headers' not in mock_openai.call_args.kwargs

    def test_main_unavailable_vision_falls_through_to_aggregators(self):
        """Main provider fails → fall back to OpenRouter/Nous strict backends."""
        fallback_client = MagicMock()
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='deepseek'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='deepseek-chat'), patch('kylinmemory._vendor.agent.auxiliary_client.resolve_provider_client', return_value=(None, None)), patch('kylinmemory._vendor.agent.auxiliary_client._resolve_strict_vision_backend', return_value=(fallback_client, 'google/gemini-3-flash-preview')), patch('kylinmemory._vendor.agent.auxiliary_client._resolve_task_provider_model', return_value=('auto', None, None, None, None)):
            from kylinmemory._vendor.agent.auxiliary_client import resolve_vision_provider_client
            provider, client, model = resolve_vision_provider_client()
        assert client is fallback_client
        assert provider in {'openrouter', 'nous'}

    def test_explicit_provider_override_still_wins(self):
        """Explicit config override bypasses main-first policy."""
        with patch('kylinmemory._vendor.agent.auxiliary_client._read_main_provider', return_value='openrouter'), patch('kylinmemory._vendor.agent.auxiliary_client._read_main_model', return_value='anthropic/claude-opus-4.6'), patch('kylinmemory._vendor.agent.auxiliary_client._resolve_task_provider_model', return_value=('nous', None, None, None, None)), patch('kylinmemory._vendor.agent.auxiliary_client._resolve_strict_vision_backend') as mock_strict:
            mock_strict.return_value = (MagicMock(), 'nous-default-model')
            from kylinmemory._vendor.agent.auxiliary_client import resolve_vision_provider_client
            provider, client, model = resolve_vision_provider_client()
        assert provider == 'nous'
        mock_strict.assert_called_once_with('nous', None)

def test_aggregator_providers_constant_removed():
    """The dead _AGGREGATOR_PROVIDERS constant should no longer live in the module.

    Removed when the main-first policy made the aggregator-skip guard obsolete.
    """
    import kylinmemory._vendor.agent.auxiliary_client as aux_mod
    assert not hasattr(aux_mod, '_AGGREGATOR_PROVIDERS'), '_AGGREGATOR_PROVIDERS was removed when _resolve_auto stopped treating aggregators specially. If you re-added it, the main-first policy may have regressed.'
