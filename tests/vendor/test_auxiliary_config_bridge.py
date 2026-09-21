"""Tests for auxiliary model config bridging — verifies that config.yaml values
are properly mapped to environment variables by both CLI and gateway loaders.

Also tests the vision_tools and browser_tool model override env vars.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def _run_auxiliary_bridge(config_dict, monkeypatch):
    """Simulate the auxiliary config → env var bridging logic shared by CLI and gateway.

    This mirrors the code in cli.py load_cli_config() and gateway/run.py.
    Both use the same pattern; we test it once here.
    """
    for key in ('AUXILIARY_VISION_PROVIDER', 'AUXILIARY_VISION_MODEL', 'AUXILIARY_VISION_BASE_URL', 'AUXILIARY_VISION_API_KEY', 'AUXILIARY_WEB_EXTRACT_PROVIDER', 'AUXILIARY_WEB_EXTRACT_MODEL', 'AUXILIARY_WEB_EXTRACT_BASE_URL', 'AUXILIARY_WEB_EXTRACT_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    auxiliary_cfg = config_dict.get('auxiliary', {})
    if auxiliary_cfg and isinstance(auxiliary_cfg, dict):
        aux_task_env = {'vision': {'provider': 'AUXILIARY_VISION_PROVIDER', 'model': 'AUXILIARY_VISION_MODEL', 'base_url': 'AUXILIARY_VISION_BASE_URL', 'api_key': 'AUXILIARY_VISION_API_KEY'}, 'web_extract': {'provider': 'AUXILIARY_WEB_EXTRACT_PROVIDER', 'model': 'AUXILIARY_WEB_EXTRACT_MODEL', 'base_url': 'AUXILIARY_WEB_EXTRACT_BASE_URL', 'api_key': 'AUXILIARY_WEB_EXTRACT_API_KEY'}}
        for task_key, env_map in aux_task_env.items():
            task_cfg = auxiliary_cfg.get(task_key, {})
            if not isinstance(task_cfg, dict):
                continue
            prov = str(task_cfg.get('provider', '')).strip()
            model = str(task_cfg.get('model', '')).strip()
            base_url = str(task_cfg.get('base_url', '')).strip()
            api_key = str(task_cfg.get('api_key', '')).strip()
            if prov and prov != 'auto':
                os.environ[env_map['provider']] = prov
            if model:
                os.environ[env_map['model']] = model
            if base_url:
                os.environ[env_map['base_url']] = base_url
            if api_key:
                os.environ[env_map['api_key']] = api_key

class TestAuxiliaryConfigBridge:
    """Verify the config.yaml → env var bridging logic used by CLI and gateway."""

    def test_vision_provider_bridged(self, monkeypatch):
        config = {'auxiliary': {'vision': {'provider': 'openrouter', 'model': ''}, 'web_extract': {'provider': 'auto', 'model': ''}}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_PROVIDER') == 'openrouter'
        assert os.environ.get('AUXILIARY_WEB_EXTRACT_PROVIDER') is None

    def test_vision_model_bridged(self, monkeypatch):
        config = {'auxiliary': {'vision': {'provider': 'auto', 'model': 'openai/gpt-4o'}}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_MODEL') == 'openai/gpt-4o'
        assert os.environ.get('AUXILIARY_VISION_PROVIDER') is None

    def test_web_extract_bridged(self, monkeypatch):
        config = {'auxiliary': {'web_extract': {'provider': 'nous', 'model': 'gemini-2.5-flash'}}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_WEB_EXTRACT_PROVIDER') == 'nous'
        assert os.environ.get('AUXILIARY_WEB_EXTRACT_MODEL') == 'gemini-2.5-flash'

    def test_direct_endpoint_bridged(self, monkeypatch):
        config = {'auxiliary': {'vision': {'base_url': 'http://localhost:1234/v1', 'api_key': 'local-key', 'model': 'qwen2.5-vl'}}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_BASE_URL') == 'http://localhost:1234/v1'
        assert os.environ.get('AUXILIARY_VISION_API_KEY') == 'local-key'
        assert os.environ.get('AUXILIARY_VISION_MODEL') == 'qwen2.5-vl'

    def test_empty_values_not_bridged(self, monkeypatch):
        config = {'auxiliary': {'vision': {'provider': 'auto', 'model': ''}}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_PROVIDER') is None
        assert os.environ.get('AUXILIARY_VISION_MODEL') is None

    def test_missing_auxiliary_section_safe(self, monkeypatch):
        """Config without auxiliary section should not crash."""
        config = {'model': {'default': 'test-model'}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_PROVIDER') is None

    def test_non_dict_task_config_ignored(self, monkeypatch):
        """Malformed task config (e.g. string instead of dict) is safely ignored."""
        config = {'auxiliary': {'vision': 'openrouter'}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_PROVIDER') is None

    def test_mixed_tasks(self, monkeypatch):
        config = {'auxiliary': {'vision': {'provider': 'openrouter', 'model': ''}, 'web_extract': {'provider': 'auto', 'model': 'custom-llm'}}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_PROVIDER') == 'openrouter'
        assert os.environ.get('AUXILIARY_VISION_MODEL') is None
        assert os.environ.get('AUXILIARY_WEB_EXTRACT_PROVIDER') is None
        assert os.environ.get('AUXILIARY_WEB_EXTRACT_MODEL') == 'custom-llm'

    def test_all_tasks_with_overrides(self, monkeypatch):
        config = {'auxiliary': {'vision': {'provider': 'openrouter', 'model': 'google/gemini-2.5-flash'}, 'web_extract': {'provider': 'nous', 'model': 'gemini-3-flash'}}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_PROVIDER') == 'openrouter'
        assert os.environ.get('AUXILIARY_VISION_MODEL') == 'google/gemini-2.5-flash'
        assert os.environ.get('AUXILIARY_WEB_EXTRACT_PROVIDER') == 'nous'
        assert os.environ.get('AUXILIARY_WEB_EXTRACT_MODEL') == 'gemini-3-flash'

    def test_whitespace_in_values_stripped(self, monkeypatch):
        config = {'auxiliary': {'vision': {'provider': '  openrouter  ', 'model': '  my-model  '}}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_PROVIDER') == 'openrouter'
        assert os.environ.get('AUXILIARY_VISION_MODEL') == 'my-model'

    def test_empty_auxiliary_dict_safe(self, monkeypatch):
        config = {'auxiliary': {}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get('AUXILIARY_VISION_PROVIDER') is None
        assert os.environ.get('AUXILIARY_WEB_EXTRACT_PROVIDER') is None

class TestDefaultConfigShape:
    """Verify the DEFAULT_CONFIG in kylin_agent_runtime_cli/config.py has correct auxiliary structure."""

    def test_auxiliary_section_exists(self):
        from kylinmemory._vendor.kylin_agent_runtime_cli.config import DEFAULT_CONFIG
        assert 'auxiliary' in DEFAULT_CONFIG

    def test_vision_task_structure(self):
        from kylinmemory._vendor.kylin_agent_runtime_cli.config import DEFAULT_CONFIG
        vision = DEFAULT_CONFIG['auxiliary']['vision']
        assert 'provider' in vision
        assert 'model' in vision
        assert vision['provider'] == 'auto'
        assert vision['model'] == ''

    def test_web_extract_task_structure(self):
        from kylinmemory._vendor.kylin_agent_runtime_cli.config import DEFAULT_CONFIG
        web = DEFAULT_CONFIG['auxiliary']['web_extract']
        assert 'provider' in web
        assert 'model' in web
        assert web['provider'] == 'auto'
        assert web['model'] == ''
