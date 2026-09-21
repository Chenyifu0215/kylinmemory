"""OpenAI Codex (Responses API) provider profile."""
from kylinmemory._vendor.providers import register_provider
from kylinmemory._vendor.providers.base import ProviderProfile
openai_codex = ProviderProfile(name='openai-codex', aliases=('codex', 'openai_codex'), api_mode='codex_responses', env_vars=(), base_url='https://chatgpt.com/backend-api/codex', auth_type='oauth_external')
register_provider(openai_codex)
