import os
from typing import Dict, List, Optional, Set
CONFIGURABLE_TOOLSETS = [('web', '🔍 Web Search & Scraping', 'web_search, web_extract'), ('browser', '🌐 Browser Automation', 'navigate, click, type, scroll'), ('terminal', '💻 Terminal & Processes', 'terminal, process'), ('file', '📁 File Operations', 'read, write, patch, search'), ('code_execution', '⚡ Code Execution', 'execute_code'), ('vision', '👁️  Vision / Image Analysis', 'vision_analyze'), ('video', '🎬 Video Analysis', 'video_analyze (requires video-capable model)'), ('image_gen', '🎨 Image Generation', 'image_generate'), ('video_gen', '🎬 Video Generation', 'video_generate (text-to-video + image-to-video)'), ('x_search', '🐦 X (Twitter) Search', 'x_search (requires xAI OAuth or XAI_API_KEY)'), ('moa', '🧠 Mixture of Agents', 'mixture_of_agents'), ('tts', '🔊 Text-to-Speech', 'text_to_speech'), ('skills', '📚 Skills', 'list, view, manage'), ('todo', '📋 Task Planning', 'todo'), ('clarify', '❓ Clarifying Questions', 'clarify'), ('delegation', '👥 Task Delegation', 'delegate_task'), ('cronjob', '⏰ Cron Jobs', 'create/list/update/pause/resume/run, with optional attached skills'), ('messaging', '📨 Cross-Platform Messaging', 'send_message'), ('homeassistant', '🏠 Home Assistant', 'smart home device control'), ('spotify', '🎵 Spotify', 'playback, search, playlists, library'), ('discord', '💬 Discord (read/participate)', 'fetch messages, search members, create thread'), ('discord_admin', '🛡️  Discord Server Admin', 'list channels/roles, pin, assign roles'), ('yuanbao', '🤖 Yuanbao', 'group info, member queries, DM'), ('computer_use', '🖱️  Computer Use (macOS)', 'background desktop control via cua-driver')]
_LEGACY_TOOLSET_ALIASES = {}
_REMOVED_TOOLSETS = {'memory', 'session_search'}
_DEFAULT_OFF_TOOLSETS = {'moa', 'homeassistant', 'spotify', 'discord', 'discord_admin', 'video', 'video_gen', 'x_search'}

def _xai_credentials_present() -> bool:
    """Cheap, side-effect-free check for usable xAI credentials.

    Used to auto-enable the ``x_search`` toolset when the user has either
    completed xAI Grok OAuth (SuperGrok subscription) or set
    ``XAI_API_KEY``. Does NOT hit the network — only inspects the local
    auth store and environment. The tool's runtime ``check_fn`` still
    gates schema registration if creds later expire or get revoked.
    """
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.auth import _read_xai_oauth_tokens
        _read_xai_oauth_tokens()
        return True
    except Exception:
        pass
    try:
        from kylin_memory._vendor.tools.xai_http import get_env_value as _xai_get_env_value
        if str(_xai_get_env_value('XAI_API_KEY') or '').strip():
            return True
    except Exception:
        pass
    return bool(str(os.environ.get('XAI_API_KEY') or '').strip())
_TOOLSET_PLATFORM_RESTRICTIONS: Dict[str, Set[str]] = {'discord': {'discord'}, 'discord_admin': {'discord'}}

def _toolset_allowed_for_platform(ts_key: str, platform: str) -> bool:
    """Return True if ``ts_key`` is configurable on ``platform``.

    Toolsets without a restriction entry are allowed everywhere (the default).
    """
    allowed = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    return allowed is None or platform in allowed

def _get_plugin_toolset_keys() -> set:
    """Return the set of toolset keys provided by plugins."""
    try:
        from kylin_memory._vendor.kylin_agent_runtime_cli.plugins import discover_plugins, get_plugin_toolsets
        discover_plugins()
        return {ts_key for ts_key, _, _ in get_plugin_toolsets()}
    except Exception:
        return set()
from kylin_memory._vendor.kylin_agent_runtime_cli.platforms import PLATFORMS as _PLATFORMS_REGISTRY
PLATFORMS = {k: {'label': info.label, 'default_toolset': info.default_toolset} for k, info in _PLATFORMS_REGISTRY.items()}

def _parse_enabled_flag(value, default: bool=True) -> bool:
    """Parse bool-like config values used by tool/platform settings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'true', '1', 'yes', 'on'}:
            return True
        if lowered in {'false', '0', 'no', 'off'}:
            return False
    return default

def _get_platform_tools(config: dict, platform: str, *, include_default_mcp_servers: bool=True) -> Set[str]:
    """Resolve which individual toolset names are enabled for a platform."""
    from kylin_memory._vendor.toolsets import resolve_toolset, TOOLSETS
    platform_toolsets = config.get('platform_toolsets') or {}
    toolset_names = platform_toolsets.get(platform)
    if toolset_names is None or not isinstance(toolset_names, list):
        plat_info = PLATFORMS.get(platform)
        if plat_info:
            default_ts = plat_info['default_toolset']
        else:
            default_ts = f'hermes-{platform}'
        toolset_names = [default_ts]
    toolset_names = [_LEGACY_TOOLSET_ALIASES.get(str(ts), str(ts)) for ts in toolset_names if str(ts) not in _REMOVED_TOOLSETS]
    configurable_keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    plugin_ts_keys = _get_plugin_toolset_keys()
    platform_default_keys = {p['default_toolset'] for p in PLATFORMS.values()}
    has_explicit_config = any((ts in configurable_keys for ts in toolset_names))
    if has_explicit_config:
        enabled_toolsets = {ts for ts in toolset_names if ts in configurable_keys and _toolset_allowed_for_platform(ts, platform)}
        composite_tools = set()
        for ts_name in toolset_names:
            if ts_name in configurable_keys or ts_name in plugin_ts_keys:
                continue
            if ts_name not in TOOLSETS:
                continue
            composite_tools.update(resolve_toolset(ts_name))
        if composite_tools:
            expanded = set()
            for ts_key, _, _ in CONFIGURABLE_TOOLSETS:
                if not _toolset_allowed_for_platform(ts_key, platform):
                    continue
                ts_tools = set(resolve_toolset(ts_key))
                if ts_tools and ts_tools.issubset(composite_tools):
                    expanded.add(ts_key)
            default_off = set(_DEFAULT_OFF_TOOLSETS)
            if platform in default_off and platform not in _TOOLSET_PLATFORM_RESTRICTIONS:
                default_off.remove(platform)
            if 'homeassistant' in default_off and os.getenv('HASS_TOKEN'):
                default_off.remove('homeassistant')
            expanded -= default_off
            enabled_toolsets |= expanded
    else:
        all_tool_names = set()
        for ts_name in toolset_names:
            all_tool_names.update(resolve_toolset(ts_name))
        enabled_toolsets = set()
        for ts_key, _, _ in CONFIGURABLE_TOOLSETS:
            if not _toolset_allowed_for_platform(ts_key, platform):
                continue
            ts_tools = set(resolve_toolset(ts_key))
            if ts_tools and ts_tools.issubset(all_tool_names):
                enabled_toolsets.add(ts_key)
        x_search_auto_enabled = _toolset_allowed_for_platform('x_search', platform) and _xai_credentials_present()
        if x_search_auto_enabled:
            enabled_toolsets.add('x_search')
        default_off = set(_DEFAULT_OFF_TOOLSETS)
        if platform in default_off and platform not in _TOOLSET_PLATFORM_RESTRICTIONS:
            default_off.remove(platform)
        if 'homeassistant' in default_off and os.getenv('HASS_TOKEN'):
            default_off.remove('homeassistant')
        if x_search_auto_enabled and 'x_search' in default_off:
            default_off.remove('x_search')
        enabled_toolsets -= default_off
    _plat_info = PLATFORMS.get(platform)
    _default_ts = _plat_info['default_toolset'] if _plat_info else f'hermes-{platform}'
    platform_tool_universe = set(resolve_toolset(_default_ts))
    configurable_tool_universe = set()
    for ck in configurable_keys:
        configurable_tool_universe.update(resolve_toolset(ck))
    claimed = set()
    for ts_key in enabled_toolsets:
        claimed.update(resolve_toolset(ts_key))
    skip = configurable_keys | plugin_ts_keys | platform_default_keys
    skip |= {k for k in TOOLSETS if k.startswith('hermes-')}
    skip |= set(_DEFAULT_OFF_TOOLSETS) - {platform}
    for ts_key, ts_def in TOOLSETS.items():
        if ts_key in skip:
            continue
        if ts_def.get('includes'):
            continue
        ts_tools = set(resolve_toolset(ts_key))
        if not ts_tools or not ts_tools.issubset(platform_tool_universe):
            continue
        if ts_tools.issubset(configurable_tool_universe):
            continue
        if not ts_tools.issubset(claimed):
            enabled_toolsets.add(ts_key)
            claimed.update(ts_tools)
    if plugin_ts_keys:
        known_map = config.get('known_plugin_toolsets', {})
        known_for_platform = set(known_map.get(platform, []))
        for pts in plugin_ts_keys:
            if pts in toolset_names:
                enabled_toolsets.add(pts)
            elif pts in _DEFAULT_OFF_TOOLSETS:
                continue
            elif pts not in known_for_platform:
                enabled_toolsets.add(pts)
    explicit_passthrough = {ts for ts in toolset_names if ts not in configurable_keys and ts not in plugin_ts_keys and (ts not in platform_default_keys)}
    mcp_servers = config.get('mcp_servers') or {}
    enabled_mcp_servers = {str(name) for name, server_cfg in mcp_servers.items() if isinstance(server_cfg, dict) and _parse_enabled_flag(server_cfg.get('enabled', True), default=True)}
    if 'no_mcp' in toolset_names:
        explicit_mcp_servers = set()
        enabled_toolsets.update(explicit_passthrough - enabled_mcp_servers - {'no_mcp'})
    else:
        explicit_mcp_servers = explicit_passthrough & enabled_mcp_servers
        enabled_toolsets.update(explicit_passthrough - enabled_mcp_servers)
    if include_default_mcp_servers:
        if explicit_mcp_servers or 'no_mcp' in toolset_names:
            enabled_toolsets.update(explicit_mcp_servers)
        else:
            enabled_toolsets.update(enabled_mcp_servers)
    else:
        enabled_toolsets.update(explicit_mcp_servers)
    agent_cfg = config.get('agent') or {}
    disabled_toolsets = agent_cfg.get('disabled_toolsets') or []
    if disabled_toolsets:
        disabled_set = {str(ts) for ts in disabled_toolsets}
        enabled_toolsets -= disabled_set
    return enabled_toolsets

