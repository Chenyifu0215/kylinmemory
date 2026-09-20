"""xAI (Grok) provider profile."""
from kylin_memory._vendor.providers import register_provider
from kylin_memory._vendor.providers.base import ProviderProfile
xai = ProviderProfile(name='xai', aliases=('grok', 'x-ai', 'x.ai'), api_mode='codex_responses', env_vars=('XAI_API_KEY',), base_url='https://api.x.ai/v1', auth_type='api_key')
register_provider(xai)
