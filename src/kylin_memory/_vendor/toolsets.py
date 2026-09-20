from typing import List, Dict, Any, Set, Optional
_HERMES_CORE_TOOLS = ['web_search', 'web_extract', 'terminal', 'process', 'read_file', 'write_file', 'patch', 'search_files', 'vision_analyze', 'image_generate', 'skills_list', 'skill_view', 'skill_manage', 'browser_navigate', 'browser_snapshot', 'browser_click', 'browser_type', 'browser_scroll', 'browser_back', 'browser_press', 'browser_get_images', 'browser_vision', 'browser_console', 'browser_cdp', 'browser_dialog', 'text_to_speech', 'todo', 'clarify', 'execute_code', 'delegate_task', 'cronjob', 'send_message', 'ha_list_entities', 'ha_get_state', 'ha_list_services', 'ha_call_service', 'kanban_show', 'kanban_list', 'kanban_complete', 'kanban_block', 'kanban_heartbeat', 'kanban_comment', 'kanban_create', 'kanban_link', 'kanban_unblock', 'computer_use']
TOOLSETS = {'web': {'description': 'Web research and content extraction tools', 'tools': ['web_search', 'web_extract'], 'includes': []}, 'search': {'description': 'Web search only (no content extraction/scraping)', 'tools': ['web_search'], 'includes': []}, 'x_search': {'description': "Search X (Twitter) posts and threads via xAI's built-in x_search Responses tool. Available when xAI credentials are configured (SuperGrok OAuth or XAI_API_KEY). Off by default; enable in `kylin-agent-runtime tools` → X (Twitter) Search.", 'tools': ['x_search'], 'includes': []}, 'vision': {'description': 'Image analysis and vision tools', 'tools': ['vision_analyze'], 'includes': []}, 'video': {'description': 'Video analysis and understanding tools (opt-in, not in default toolset)', 'tools': ['video_analyze'], 'includes': []}, 'image_gen': {'description': 'Creative generation tools (images)', 'tools': ['image_generate'], 'includes': []}, 'video_gen': {'description': 'Video generation tools. Single ``video_generate`` tool covers text-to-video (prompt only) and image-to-video (prompt + image_url) — the active backend auto-routes. Configure via ``kylin-agent-runtime tools`` → Video Generation.', 'tools': ['video_generate'], 'includes': []}, 'computer_use': {'description': "Background macOS desktop control via cua-driver — screenshots, mouse, keyboard, scroll, drag. Does NOT steal the user's cursor or keyboard focus. Works with any tool-capable model.", 'tools': ['computer_use'], 'includes': []}, 'terminal': {'description': 'Terminal/command execution and process management tools', 'tools': ['terminal', 'process'], 'includes': []}, 'moa': {'description': 'Advanced reasoning and problem-solving tools', 'tools': ['mixture_of_agents'], 'includes': []}, 'skills': {'description': 'Access, create, edit, and manage skill documents with specialized instructions and knowledge', 'tools': ['skills_list', 'skill_view', 'skill_manage'], 'includes': []}, 'browser': {'description': 'Browser automation for web interaction (navigate, click, type, scroll, iframes, hold-click) with web search for finding URLs', 'tools': ['browser_navigate', 'browser_snapshot', 'browser_click', 'browser_type', 'browser_scroll', 'browser_back', 'browser_press', 'browser_get_images', 'browser_vision', 'browser_console', 'browser_cdp', 'browser_dialog', 'web_search'], 'includes': []}, 'cronjob': {'description': 'Cronjob management tool - create, list, update, pause, resume, remove, and trigger scheduled tasks', 'tools': ['cronjob'], 'includes': []}, 'messaging': {'description': 'Cross-platform messaging: send messages to Telegram, Discord, Slack, SMS, etc.', 'tools': ['send_message'], 'includes': []}, 'file': {'description': 'File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files)', 'tools': ['read_file', 'write_file', 'patch', 'search_files'], 'includes': []}, 'tts': {'description': 'Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, OpenAI, or xAI', 'tools': ['text_to_speech'], 'includes': []}, 'todo': {'description': 'Task planning and tracking for multi-step work', 'tools': ['todo'], 'includes': []}, 'clarify': {'description': 'Ask the user clarifying questions (multiple-choice or open-ended)', 'tools': ['clarify'], 'includes': []}, 'code_execution': {'description': 'Run Python scripts that call tools programmatically (reduces LLM round trips)', 'tools': ['execute_code'], 'includes': []}, 'delegation': {'description': 'Spawn subagents with isolated context for complex subtasks', 'tools': ['delegate_task'], 'includes': []}, 'homeassistant': {'description': 'Home Assistant smart home control and monitoring', 'tools': ['ha_list_entities', 'ha_get_state', 'ha_list_services', 'ha_call_service'], 'includes': []}, 'kanban': {'description': 'Kanban multi-agent coordination — only active when the agent is spawned by the kanban dispatcher (HERMES_KANBAN_TASK env set). The dispatcher runs inside the gateway by default; see `kanban.dispatch_in_gateway` in config.yaml. Lets workers mark tasks done with structured handoffs, block for human input, heartbeat during long ops, comment on threads, and (for orchestrators) list, unblock, and fan out tasks.', 'tools': ['kanban_show', 'kanban_list', 'kanban_complete', 'kanban_block', 'kanban_heartbeat', 'kanban_comment', 'kanban_create', 'kanban_link', 'kanban_unblock'], 'includes': []}, 'discord': {'description': 'Discord read and participate tools (fetch messages, search members, create threads)', 'tools': ['discord'], 'includes': []}, 'discord_admin': {'description': 'Discord server management (list channels/roles, pin messages, assign roles)', 'tools': ['discord_admin'], 'includes': []}, 'yuanbao': {'description': 'Yuanbao platform tools - group info, member queries, DM, stickers', 'tools': ['yb_query_group_info', 'yb_query_group_members', 'yb_send_dm', 'yb_search_sticker', 'yb_send_sticker'], 'includes': []}, 'feishu_doc': {'description': 'Read Feishu/Lark document content', 'tools': ['feishu_doc_read'], 'includes': []}, 'feishu_drive': {'description': 'Feishu/Lark document comment operations (list, reply, add)', 'tools': ['feishu_drive_list_comments', 'feishu_drive_list_comment_replies', 'feishu_drive_reply_comment', 'feishu_drive_add_comment'], 'includes': []}, 'spotify': {'description': 'Native Spotify playback, search, playlist, album, and library tools', 'tools': ['spotify_playback', 'spotify_devices', 'spotify_queue', 'spotify_search', 'spotify_playlists', 'spotify_albums', 'spotify_library'], 'includes': []}, 'debugging': {'description': 'Debugging and troubleshooting toolkit', 'tools': ['terminal', 'process'], 'includes': ['web', 'file']}, 'safe': {'description': 'Safe toolkit without terminal access', 'tools': [], 'includes': ['web', 'vision', 'image_gen']}, 'hermes-acp': {'description': 'Editor integration (VS Code, Zed, JetBrains) — coding-focused tools without messaging, audio, or clarify UI', 'tools': ['web_search', 'web_extract', 'terminal', 'process', 'read_file', 'write_file', 'patch', 'search_files', 'vision_analyze', 'skills_list', 'skill_view', 'skill_manage', 'browser_navigate', 'browser_snapshot', 'browser_click', 'browser_type', 'browser_scroll', 'browser_back', 'browser_press', 'browser_get_images', 'browser_vision', 'browser_console', 'browser_cdp', 'browser_dialog', 'todo', 'execute_code', 'delegate_task'], 'includes': []}, 'hermes-api-server': {'description': 'OpenAI-compatible API server — full agent tools accessible via HTTP (no interactive UI tools like clarify or send_message)', 'tools': ['web_search', 'web_extract', 'terminal', 'process', 'read_file', 'write_file', 'patch', 'search_files', 'vision_analyze', 'image_generate', 'skills_list', 'skill_view', 'skill_manage', 'browser_navigate', 'browser_snapshot', 'browser_click', 'browser_type', 'browser_scroll', 'browser_back', 'browser_press', 'browser_get_images', 'browser_vision', 'browser_console', 'browser_cdp', 'browser_dialog', 'todo', 'execute_code', 'delegate_task', 'cronjob', 'ha_list_entities', 'ha_get_state', 'ha_list_services', 'ha_call_service'], 'includes': []}, 'hermes-cli': {'description': 'Full interactive CLI toolset - all default tools plus cronjob management', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-cron': {'description': 'Default cron toolset - same core tools as hermes-cli; gated by `kylin-agent-runtime tools`', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-telegram': {'description': 'Telegram bot toolset - full access for personal use (terminal has safety checks)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-discord': {'description': 'Discord bot toolset - full access (terminal has safety checks via dangerous command approval)', 'tools': _HERMES_CORE_TOOLS + ['discord', 'discord_admin'], 'includes': []}, 'hermes-whatsapp': {'description': 'WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-slack': {'description': 'Slack bot toolset - full access for workspace use (terminal has safety checks)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-signal': {'description': 'Signal bot toolset - encrypted messaging platform (full access)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-bluebubbles': {'description': 'BlueBubbles iMessage bot toolset - Apple iMessage via local BlueBubbles server', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-homeassistant': {'description': 'Home Assistant bot toolset - smart home event monitoring and control', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-email': {'description': 'Email bot toolset - interact with Hermes via email (IMAP/SMTP)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-mattermost': {'description': 'Mattermost bot toolset - self-hosted team messaging (full access)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-matrix': {'description': 'Matrix bot toolset - decentralized encrypted messaging (full access)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-dingtalk': {'description': 'DingTalk bot toolset - enterprise messaging platform (full access)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-feishu': {'description': 'Feishu/Lark bot toolset - enterprise messaging via Feishu/Lark (full access)', 'tools': _HERMES_CORE_TOOLS + ['feishu_doc_read', 'feishu_drive_list_comments', 'feishu_drive_list_comment_replies', 'feishu_drive_reply_comment', 'feishu_drive_add_comment'], 'includes': []}, 'hermes-weixin': {'description': 'Weixin bot toolset - personal WeChat messaging via iLink (full access)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-qqbot': {'description': 'QQBot toolset - QQ messaging via Official Bot API v2 (full access)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-wecom': {'description': 'WeCom bot toolset - enterprise WeChat messaging (full access)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-wecom-callback': {'description': 'WeCom callback toolset - enterprise self-built app messaging (full access)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-yuanbao': {'description': 'Yuanbao Bot 元宝消息平台工具集 - 群信息、成员查询、私聊、贴纸表情', 'tools': _HERMES_CORE_TOOLS + ['yb_query_group_info', 'yb_query_group_members', 'yb_send_dm', 'yb_search_sticker', 'yb_send_sticker'], 'module': 'kylin_memory._vendor.tools.yuanbao_tools', 'includes': []}, 'hermes-sms': {'description': 'SMS bot toolset - interact with Hermes via SMS (Twilio)', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-webhook': {'description': 'Webhook toolset - receive and process external webhook events', 'tools': _HERMES_CORE_TOOLS, 'includes': []}, 'hermes-gateway': {'description': 'Gateway toolset - union of all messaging platform tools', 'tools': [], 'includes': ['hermes-telegram', 'hermes-discord', 'hermes-whatsapp', 'hermes-slack', 'hermes-signal', 'hermes-bluebubbles', 'hermes-homeassistant', 'hermes-email', 'hermes-sms', 'hermes-mattermost', 'hermes-matrix', 'hermes-dingtalk', 'hermes-feishu', 'hermes-wecom', 'hermes-wecom-callback', 'hermes-weixin', 'hermes-qqbot', 'hermes-webhook', 'hermes-yuanbao']}}

def get_toolset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a toolset definition by name.
    
    Args:
        name (str): Name of the toolset
        
    Returns:
        Dict: Toolset definition with description, tools, and includes
        None: If toolset not found
    """
    toolset = TOOLSETS.get(name)
    try:
        from kylin_memory._vendor.tools.registry import registry
    except Exception:
        return toolset if toolset else None
    if toolset:
        merged_tools = sorted(set(toolset.get('tools', [])) | set(registry.get_tool_names_for_toolset(name)))
        return {**toolset, 'tools': merged_tools}
    registry_toolset = name
    description = f'Plugin toolset: {name}'
    alias_target = registry.get_toolset_alias_target(name)
    if name not in _get_plugin_toolset_names():
        registry_toolset = alias_target
        if not registry_toolset:
            return None
        description = f"MCP server '{name}' tools"
    else:
        reverse_aliases = {canonical: alias for alias, canonical in _get_registry_toolset_aliases().items() if alias not in TOOLSETS}
        alias = reverse_aliases.get(name)
        if alias:
            description = f"MCP server '{alias}' tools"
    return {'description': description, 'tools': registry.get_tool_names_for_toolset(registry_toolset), 'includes': []}

def resolve_toolset(name: str, visited: Set[str]=None) -> List[str]:
    """
    Recursively resolve a toolset to get all tool names.
    
    This function handles toolset composition by recursively resolving
    included toolsets and combining all tools.
    
    Args:
        name (str): Name of the toolset to resolve
        visited (Set[str]): Set of already visited toolsets (for cycle detection)
        
    Returns:
        List[str]: List of all tool names in the toolset
    """
    if visited is None:
        visited = set()
    if name in {'all', '*'}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            resolved = resolve_toolset(toolset_name, visited.copy())
            all_tools.update(resolved)
        return sorted(all_tools)
    if name in visited:
        return []
    visited.add(name)
    toolset = get_toolset(name)
    if not toolset:
        if name.startswith('hermes-'):
            platform_name = name[len('hermes-'):]
            try:
                from kylin_memory._vendor.gateway.platform_registry import platform_registry
                if platform_registry.is_registered(platform_name):
                    plugin_tools = set(_HERMES_CORE_TOOLS)
                    try:
                        from kylin_memory._vendor.tools.registry import registry
                        plugin_tools.update((e.name for e in registry._tools.values() if e.toolset == platform_name))
                    except Exception:
                        pass
                    return list(plugin_tools)
            except Exception:
                pass
        return []
    tools = set(toolset.get('tools', []))
    for included_name in toolset.get('includes', []):
        included_tools = resolve_toolset(included_name, visited)
        tools.update(included_tools)
    return sorted(tools)

def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    """
    Resolve multiple toolsets and combine their tools.
    
    Args:
        toolset_names (List[str]): List of toolset names to resolve
        
    Returns:
        List[str]: Combined list of all tool names (deduplicated)
    """
    all_tools = set()
    for name in toolset_names:
        tools = resolve_toolset(name)
        all_tools.update(tools)
    return sorted(all_tools)

def _get_plugin_toolset_names() -> Set[str]:
    """Return toolset names registered by plugins (from the tool registry).

    These are toolsets that exist in the registry but not in the static
    ``TOOLSETS`` dict — i.e. they were added by plugins at load time.
    """
    try:
        from kylin_memory._vendor.tools.registry import registry
        return {toolset_name for toolset_name in registry.get_registered_toolset_names() if toolset_name not in TOOLSETS}
    except Exception:
        return set()

def _get_registry_toolset_aliases() -> Dict[str, str]:
    """Return explicit toolset aliases registered in the live registry."""
    try:
        from kylin_memory._vendor.tools.registry import registry
        return registry.get_registered_toolset_aliases()
    except Exception:
        return {}

def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    """
    Get all available toolsets with their definitions.

    Includes both statically-defined toolsets and plugin-registered ones.
    
    Returns:
        Dict: All toolset definitions
    """
    result = dict(TOOLSETS)
    aliases = _get_registry_toolset_aliases()
    for ts_name in _get_plugin_toolset_names():
        display_name = ts_name
        for alias, canonical in aliases.items():
            if canonical == ts_name and alias not in TOOLSETS:
                display_name = alias
                break
        if display_name in result:
            continue
        toolset = get_toolset(display_name)
        if toolset:
            result[display_name] = toolset
    return result

def get_toolset_names() -> List[str]:
    """
    Get names of all available toolsets (excluding aliases).

    Includes plugin-registered toolset names.
    
    Returns:
        List[str]: List of toolset names
    """
    names = set(TOOLSETS.keys())
    aliases = _get_registry_toolset_aliases()
    for ts_name in _get_plugin_toolset_names():
        for alias, canonical in aliases.items():
            if canonical == ts_name and alias not in TOOLSETS:
                names.add(alias)
                break
        else:
            names.add(ts_name)
    return sorted(names)

def create_custom_toolset(name: str, description: str, tools: List[str]=None, includes: List[str]=None) -> None:
    """
    Create a custom toolset at runtime.
    
    Args:
        name (str): Name for the new toolset
        description (str): Description of the toolset
        tools (List[str]): Direct tools to include
        includes (List[str]): Other toolsets to include
    """
    TOOLSETS[name] = {'description': description, 'tools': tools or [], 'includes': includes or []}

def get_toolset_info(name: str) -> Dict[str, Any]:
    """
    Get detailed information about a toolset including resolved tools.
    
    Args:
        name (str): Toolset name
        
    Returns:
        Dict: Detailed toolset information
    """
    toolset = get_toolset(name)
    if not toolset:
        return None
    resolved_tools = resolve_toolset(name)
    return {'name': name, 'description': toolset['description'], 'direct_tools': toolset['tools'], 'includes': toolset['includes'], 'resolved_tools': resolved_tools, 'tool_count': len(resolved_tools), 'is_composite': bool(toolset['includes'])}
if __name__ == '__main__':
    print('Toolsets System Demo')
    print('=' * 60)
    print('\nAvailable Toolsets:')
    print('-' * 40)
    for name, toolset in get_all_toolsets().items():
        info = get_toolset_info(name)
        composite = '[composite]' if info['is_composite'] else '[leaf]'
        print(f"  {composite} {name:20} - {toolset['description']}")
        print(f"     Tools: {len(info['resolved_tools'])} total")
    print('\nToolset Resolution Examples:')
    print('-' * 40)
    for name in ['web', 'terminal', 'safe', 'debugging']:
        tools = resolve_toolset(name)
        print(f'\n  {name}:')
        print(f"    Resolved to {len(tools)} tools: {', '.join(sorted(tools))}")
    print('\nMultiple Toolset Resolution:')
    print('-' * 40)
    combined = resolve_multiple_toolsets(['web', 'vision', 'terminal'])
    print("  Combining ['web', 'vision', 'terminal']:")
    print(f"    Result: {', '.join(sorted(combined))}")
    print('\nCustom Toolset Creation:')
    print('-' * 40)
    create_custom_toolset(name='my_custom', description='My custom toolset for specific tasks', tools=['web_search'], includes=['terminal', 'vision'])
    custom_info = get_toolset_info('my_custom')
    print("  Created 'my_custom' toolset:")
    print(f"    Description: {custom_info['description']}")
    print(f"    Resolved tools: {', '.join(custom_info['resolved_tools'])}")

