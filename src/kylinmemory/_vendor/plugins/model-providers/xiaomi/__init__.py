"""Xiaomi MiMo provider profile."""
from kylinmemory._vendor.providers import register_provider
from kylinmemory._vendor.providers.base import ProviderProfile
xiaomi = ProviderProfile(name='xiaomi', aliases=('mimo', 'xiaomi-mimo'), env_vars=('XIAOMI_API_KEY',), base_url='https://api.xiaomimimo.com/v1', supports_health_check=False)
register_provider(xiaomi)
