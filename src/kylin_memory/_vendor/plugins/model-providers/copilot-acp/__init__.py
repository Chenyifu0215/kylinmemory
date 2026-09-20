"""GitHub Copilot ACP provider profile.

copilot-acp uses an external ACP subprocess — NOT the standard
transport. api_mode="copilot_acp" is handled separately in run_agent.py.
The profile captures auth + endpoint metadata for registry migration.
"""
from kylin_memory._vendor.providers import register_provider
from kylin_memory._vendor.providers.base import ProviderProfile

class CopilotACPProfile(ProviderProfile):
    """GitHub Copilot ACP — external process, no REST models endpoint."""

    def fetch_models(self, *, api_key: str | None=None, timeout: float=8.0) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None
copilot_acp = CopilotACPProfile(name='copilot-acp', aliases=('github-copilot-acp', 'copilot-acp-agent'), api_mode='chat_completions', env_vars=(), base_url='acp://copilot', auth_type='external_process')
register_provider(copilot_acp)
