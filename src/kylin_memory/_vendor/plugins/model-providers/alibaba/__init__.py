"""Alibaba Cloud DashScope provider profile."""
from kylin_memory._vendor.providers import register_provider
from kylin_memory._vendor.providers.base import ProviderProfile
alibaba = ProviderProfile(name='alibaba', aliases=('dashscope', 'alibaba-cloud', 'qwen-dashscope'), env_vars=('DASHSCOPE_API_KEY',), base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1')
register_provider(alibaba)
