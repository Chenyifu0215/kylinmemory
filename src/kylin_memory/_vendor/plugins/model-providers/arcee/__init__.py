"""Arcee AI provider profile."""
from kylin_memory._vendor.providers import register_provider
from kylin_memory._vendor.providers.base import ProviderProfile
arcee = ProviderProfile(name='arcee', aliases=('arcee-ai', 'arceeai'), env_vars=('ARCEEAI_API_KEY',), base_url='https://api.arcee.ai/api/v1')
register_provider(arcee)
