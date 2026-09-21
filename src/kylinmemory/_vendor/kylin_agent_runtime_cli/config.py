import copy
import logging
import os
import re
import stat
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
logger = logging.getLogger(__name__)
_CONFIG_PARSE_WARNED: set = set()

def _warn_config_parse_failure(config_path: Path, exc: Exception) -> None:
    """Surface a config.yaml parse failure to user, log, and stderr.

    A YAML parse error in ``~/.kylin-agent-runtime/config.yaml`` causes ``load_config()``
    to silently fall back to ``DEFAULT_CONFIG``, which means every user
    override (auxiliary providers, fallback chain, model overrides, etc.)
    is dropped. Before this helper that was a one-line ``print(...)`` that
    scrolled off-screen on the first invocation and was never seen again.

    Now: warn once per (path, mtime_ns, size) on stderr **and** in
    ``agent.log`` / ``errors.log`` at WARNING level so ``kylin-agent-runtime logs``
    surfaces it. Re-warns automatically if the file changes (different
    mtime/size), so users editing the config see the next failure.
    """
    try:
        st = config_path.stat()
        key = (str(config_path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(config_path), 0, 0)
    if key in _CONFIG_PARSE_WARNED:
        return
    _CONFIG_PARSE_WARNED.add(key)
    msg = f'Failed to parse {config_path}: {exc}. Falling back to default config — every user override (auxiliary providers, fallback chain, model settings) is being IGNORED. Fix the YAML and restart.'
    logger.warning(msg)
    try:
        sys.stderr.write(f'⚠️  kylin-agent-runtime config: {msg}\n')
        sys.stderr.flush()
    except Exception:
        pass
_ENV_VAR_NAME_RE = re.compile('^[A-Za-z_][A-Za-z0-9_]*$')
_LAST_EXPANDED_CONFIG_BY_PATH: Dict[str, Any] = {}
_LOAD_CONFIG_CACHE: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
_RAW_CONFIG_CACHE: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
_CONFIG_LOCK = threading.RLock()
_EXTRA_ENV_KEYS = frozenset({'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'ANTHROPIC_API_KEY', 'ANTHROPIC_TOKEN', 'DISCORD_HOME_CHANNEL', 'DISCORD_HOME_CHANNEL_NAME', 'TELEGRAM_HOME_CHANNEL', 'TELEGRAM_HOME_CHANNEL_NAME', 'SLACK_HOME_CHANNEL', 'SLACK_HOME_CHANNEL_NAME', 'SIGNAL_ACCOUNT', 'SIGNAL_HTTP_URL', 'SIGNAL_ALLOWED_USERS', 'SIGNAL_GROUP_ALLOWED_USERS', 'SIGNAL_HOME_CHANNEL', 'SIGNAL_HOME_CHANNEL_NAME', 'SMS_HOME_CHANNEL', 'SMS_HOME_CHANNEL_NAME', 'DINGTALK_CLIENT_ID', 'DINGTALK_CLIENT_SECRET', 'DINGTALK_HOME_CHANNEL', 'DINGTALK_HOME_CHANNEL_NAME', 'FEISHU_APP_ID', 'FEISHU_APP_SECRET', 'FEISHU_ENCRYPT_KEY', 'FEISHU_VERIFICATION_TOKEN', 'FEISHU_HOME_CHANNEL', 'FEISHU_HOME_CHANNEL_NAME', 'YUANBAO_HOME_CHANNEL', 'YUANBAO_HOME_CHANNEL_NAME', 'WECOM_BOT_ID', 'WECOM_SECRET', 'WECOM_CALLBACK_CORP_ID', 'WECOM_CALLBACK_CORP_SECRET', 'WECOM_CALLBACK_AGENT_ID', 'WECOM_CALLBACK_TOKEN', 'WECOM_CALLBACK_ENCODING_AES_KEY', 'WECOM_CALLBACK_HOST', 'WECOM_CALLBACK_PORT', 'WECOM_HOME_CHANNEL', 'WECOM_HOME_CHANNEL_NAME', 'WEIXIN_ACCOUNT_ID', 'WEIXIN_TOKEN', 'WEIXIN_BASE_URL', 'WEIXIN_CDN_BASE_URL', 'WEIXIN_HOME_CHANNEL', 'WEIXIN_HOME_CHANNEL_NAME', 'WEIXIN_DM_POLICY', 'WEIXIN_GROUP_POLICY', 'WEIXIN_ALLOWED_USERS', 'WEIXIN_GROUP_ALLOWED_USERS', 'WEIXIN_ALLOW_ALL_USERS', 'BLUEBUBBLES_SERVER_URL', 'BLUEBUBBLES_PASSWORD', 'BLUEBUBBLES_HOME_CHANNEL', 'BLUEBUBBLES_HOME_CHANNEL_NAME', 'QQ_APP_ID', 'QQ_CLIENT_SECRET', 'QQBOT_HOME_CHANNEL', 'QQBOT_HOME_CHANNEL_NAME', 'QQ_HOME_CHANNEL', 'QQ_HOME_CHANNEL_NAME', 'QQ_ALLOWED_USERS', 'QQ_GROUP_ALLOWED_USERS', 'QQ_ALLOW_ALL_USERS', 'QQ_MARKDOWN_SUPPORT', 'QQ_STT_API_KEY', 'QQ_STT_BASE_URL', 'QQ_STT_MODEL', 'IRC_SERVER', 'IRC_PORT', 'IRC_NICKNAME', 'IRC_CHANNEL', 'IRC_USE_TLS', 'IRC_SERVER_PASSWORD', 'IRC_NICKSERV_PASSWORD', 'TERMINAL_ENV', 'TERMINAL_SSH_KEY', 'TERMINAL_SSH_PORT', 'WHATSAPP_MODE', 'WHATSAPP_ENABLED', 'MATTERMOST_HOME_CHANNEL', 'MATTERMOST_HOME_CHANNEL_NAME', 'MATTERMOST_REPLY_MODE', 'MATRIX_PASSWORD', 'MATRIX_ENCRYPTION', 'MATRIX_DEVICE_ID', 'MATRIX_HOME_ROOM', 'MATRIX_REQUIRE_MENTION', 'MATRIX_FREE_RESPONSE_ROOMS', 'MATRIX_AUTO_THREAD', 'MATRIX_DM_AUTO_THREAD', 'MATRIX_RECOVERY_KEY', 'HERMES_LANGFUSE_ENV', 'HERMES_LANGFUSE_RELEASE', 'HERMES_LANGFUSE_SAMPLE_RATE', 'HERMES_LANGFUSE_MAX_CHARS', 'HERMES_LANGFUSE_DEBUG', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_BASE_URL'})
import yaml
from kylinmemory._vendor.kylin_agent_runtime_cli.default_soul import DEFAULT_SOUL_MD
_MANAGED_TRUE_VALUES = ('true', '1', 'yes')
_MANAGED_SYSTEM_NAMES = {'brew': 'Homebrew', 'homebrew': 'Homebrew', 'nix': 'NixOS', 'nixos': 'NixOS'}

def get_managed_system() -> Optional[str]:
    """Return the package manager owning this install, if any."""
    raw = os.getenv('HERMES_MANAGED', '').strip()
    if raw:
        normalized = raw.lower()
        if normalized in _MANAGED_TRUE_VALUES:
            return 'NixOS'
        return _MANAGED_SYSTEM_NAMES.get(normalized, raw)
    managed_marker = get_hermes_home() / '.managed'
    if managed_marker.exists():
        return 'NixOS'
    return None

def is_managed() -> bool:
    """Check if Hermes is running in package-manager-managed mode.

    Two signals: the HERMES_MANAGED env var (set by the systemd service),
    or a .managed marker file in HERMES_HOME (set by the NixOS activation
    script, so interactive shells also see it).
    """
    return get_managed_system() is not None

def format_managed_message(action: str='modify this Hermes installation') -> str:
    """Build a user-facing error for managed installs."""
    managed_system = get_managed_system() or 'a package manager'
    raw = os.getenv('HERMES_MANAGED', '').strip().lower()
    if managed_system == 'NixOS':
        env_hint = 'true' if raw in _MANAGED_TRUE_VALUES else raw or 'true'
        return f'Cannot {action}: this Hermes installation is managed by NixOS (HERMES_MANAGED={env_hint}).\nEdit services.hermes-agent.settings in your configuration.nix and run:\n  sudo nixos-rebuild switch'
    if managed_system == 'Homebrew':
        env_hint = raw or 'homebrew'
        return f'Cannot {action}: this Hermes installation is managed by Homebrew (HERMES_MANAGED={env_hint}).\nUse:\n  brew upgrade hermes-agent'
    return f'Cannot {action}: this Hermes installation is managed by {managed_system}.\nUse your package manager to upgrade or reinstall Hermes.'

def managed_error(action: str='modify configuration'):
    """Print user-friendly error for managed mode."""
    print(format_managed_message(action), file=sys.stderr)
from kylinmemory._vendor.kylin_agent_runtime_constants import ensure_kylin_state_db, get_hermes_home
from kylinmemory._vendor.utils import atomic_replace

def get_config_path() -> Path:
    """Get the main config file path."""
    return get_hermes_home() / 'config.yaml'

def get_env_path() -> Path:
    """Get the .env file path (for API keys)."""
    return get_hermes_home() / '.env'

def _secure_dir(path):
    """Set directory to owner-only access (0700 by default). No-op on Windows.

    Skipped in managed mode — the NixOS module sets group-readable
    permissions (0750) so interactive users in the hermes group can
    share state with the gateway service.

    The mode can be overridden via the HERMES_HOME_MODE environment variable
    (e.g. HERMES_HOME_MODE=0701) for deployments where a web server (nginx,
    caddy, etc.) needs to traverse HERMES_HOME to reach a served subdirectory.
    The execute-only bit on a directory permits cd-through without exposing
    directory listings.
    """
    if is_managed():
        return
    try:
        mode_str = os.environ.get('HERMES_HOME_MODE', '').strip()
        mode = int(mode_str, 8) if mode_str else 448
    except ValueError:
        mode = 448
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        pass

def _is_container() -> bool:
    """Detect if we're running inside a Docker/Podman/LXC container.

    When Hermes runs in a container with volume-mounted config files, forcing
    0o600 permissions breaks multi-process setups where the gateway and
    dashboard run as different UIDs or the volume mount requires broader
    permissions.
    """
    if os.environ.get('HERMES_CONTAINER') or os.environ.get('HERMES_SKIP_CHMOD'):
        return True
    if os.path.exists('/.dockerenv'):
        return True
    try:
        with open('/proc/1/cgroup', 'r', encoding='utf-8') as f:
            cgroup_content = f.read()
        if 'docker' in cgroup_content or 'lxc' in cgroup_content or 'kubepods' in cgroup_content:
            return True
    except (OSError, IOError):
        pass
    return False

def _secure_file(path):
    """Set file to owner-only read/write (0600). No-op on Windows.

    Skipped in managed mode — the NixOS activation script sets
    group-readable permissions (0640) on config files.

    Skipped in containers — Docker/Podman volume mounts often need broader
    permissions.  Set HERMES_SKIP_CHMOD=1 to force-skip on other systems.
    """
    if is_managed() or _is_container():
        return
    try:
        if os.path.exists(str(path)):
            os.chmod(path, 384)
    except (OSError, NotImplementedError):
        pass

def _ensure_default_soul_md(home: Path) -> None:
    """Seed a default SOUL.md into HERMES_HOME if the user doesn't have one yet."""
    soul_path = home / 'SOUL.md'
    if soul_path.exists():
        return
    soul_path.write_text(DEFAULT_SOUL_MD, encoding='utf-8')
    _secure_file(soul_path)

def ensure_hermes_home():
    """Ensure ~/.kylin-agent-runtime directory structure exists with secure permissions.

    In managed mode (NixOS), dirs are created by the activation script with
    setgid + group-writable (2770). We skip mkdir and set umask(0o007) so
    any files created (e.g. SOUL.md) are group-writable (0660).
    """
    home = get_hermes_home()
    if is_managed():
        old_umask = os.umask(7)
        try:
            _ensure_hermes_home_managed(home)
        finally:
            os.umask(old_umask)
    else:
        home.mkdir(parents=True, exist_ok=True)
        _secure_dir(home)
        for subdir in ('cron', 'sessions', 'logs', 'logs/curator', 'pairing', 'hooks', 'image_cache', 'audio_cache', 'skills'):
            d = home / subdir
            d.mkdir(parents=True, exist_ok=True)
            _secure_dir(d)
        _ensure_default_soul_md(home)
    state_path = ensure_kylin_state_db(home)
    _secure_file(state_path)

def _ensure_hermes_home_managed(home: Path):
    """Managed-mode variant: verify dirs exist (activation creates them), seed SOUL.md."""
    if not home.is_dir():
        raise RuntimeError(f"HERMES_HOME {home} does not exist. Run 'sudo nixos-rebuild switch' first.")
    for subdir in ('cron', 'sessions', 'logs'):
        d = home / subdir
        if not d.is_dir():
            raise RuntimeError(f"{d} does not exist. Run 'sudo nixos-rebuild switch' first.")
    (home / 'logs' / 'curator').mkdir(parents=True, exist_ok=True)
    _ensure_default_soul_md(home)
DEFAULT_CONFIG = {'model': '', 'providers': {}, 'fallback_providers': [], 'credential_pool_strategies': {}, 'toolsets': ['hermes-cli'], 'agent': {'max_turns': 90, 'gateway_timeout': 1800, 'restart_drain_timeout': 180, 'api_max_retries': 3, 'verbose_logging': False, 'service_tier': '', 'tool_use_enforcement': 'auto', 'gateway_timeout_warning': 900, 'clarify_timeout': 600, 'gateway_notify_interval': 180, 'gateway_auto_continue_freshness': 3600, 'image_input_mode': 'auto', 'disabled_toolsets': []}, 'terminal': {'backend': 'local', 'modal_mode': 'auto', 'cwd': '.', 'timeout': 180, 'env_passthrough': [], 'shell_init_files': [], 'auto_source_bashrc': True, 'docker_image': 'nikolaik/python-nodejs:python3.11-nodejs20', 'docker_forward_env': [], 'docker_env': {}, 'singularity_image': 'docker://nikolaik/python-nodejs:python3.11-nodejs20', 'modal_image': 'nikolaik/python-nodejs:python3.11-nodejs20', 'daytona_image': 'nikolaik/python-nodejs:python3.11-nodejs20', 'vercel_runtime': 'node24', 'container_cpu': 1, 'container_memory': 5120, 'container_disk': 51200, 'container_persistent': True, 'docker_volumes': [], 'docker_mount_cwd_to_workspace': False, 'docker_extra_args': [], 'docker_run_as_host_user': False, 'persistent_shell': True}, 'web': {'backend': '', 'search_backend': '', 'extract_backend': ''}, 'browser': {'inactivity_timeout': 120, 'command_timeout': 30, 'record_sessions': False, 'allow_private_urls': False, 'engine': 'auto', 'auto_local_for_private_urls': True, 'cdp_url': '', 'dialog_policy': 'must_respond', 'dialog_timeout_s': 300, 'camofox': {'managed_persistence': False, 'user_id': '', 'session_key': '', 'adopt_existing_tab': False}}, 'checkpoints': {'enabled': False, 'max_snapshots': 20, 'max_total_size_mb': 500, 'max_file_size_mb': 10, 'auto_prune': True, 'retention_days': 7, 'delete_orphans': True, 'min_interval_hours': 24}, 'file_read_max_chars': 100000, 'tool_output': {'max_bytes': 50000, 'max_lines': 2000, 'max_line_length': 2000}, 'tool_loop_guardrails': {'warnings_enabled': True, 'hard_stop_enabled': False, 'warn_after': {'exact_failure': 2, 'same_tool_failure': 3, 'idempotent_no_progress': 2}, 'hard_stop_after': {'exact_failure': 5, 'same_tool_failure': 8, 'idempotent_no_progress': 5}}, 'compression': {'enabled': True, 'threshold': 0.5, 'target_ratio': 0.2, 'protect_last_n': 20, 'hygiene_hard_message_limit': 400, 'protect_first_n': 3, 'abort_on_summary_failure': False}, 'prompt_caching': {'cache_ttl': '5m'}, 'openrouter': {'response_cache': True, 'response_cache_ttl': 300, 'min_coding_score': 0.65}, 'bedrock': {'region': '', 'discovery': {'enabled': True, 'provider_filter': [], 'refresh_interval': 3600}, 'guardrail': {'guardrail_identifier': '', 'guardrail_version': '', 'stream_processing_mode': 'async', 'trace': 'disabled'}}, 'auxiliary': {'vision': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 120, 'extra_body': {}, 'download_timeout': 30}, 'web_extract': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 360, 'extra_body': {}}, 'compression': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 120, 'extra_body': {}}, 'skills_hub': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 30, 'extra_body': {}}, 'approval': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 30, 'extra_body': {}}, 'atom_memory': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'api_mode': '', 'timeout': 120, 'extra_body': {}}, 'mcp': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 30, 'extra_body': {}}, 'title_generation': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 30, 'extra_body': {}}, 'triage_specifier': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 120, 'extra_body': {}}, 'kanban_decomposer': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 180, 'extra_body': {}}, 'profile_describer': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 60, 'extra_body': {}}, 'curator': {'provider': 'auto', 'model': '', 'base_url': '', 'api_key': '', 'timeout': 600, 'extra_body': {}}}, 'display': {'compact': False, 'personality': 'kawaii', 'resume_display': 'full', 'busy_input_mode': 'interrupt', 'tui_auto_resume_recent': False, 'bell_on_complete': False, 'show_reasoning': False, 'streaming': False, 'timestamps': False, 'final_response_markdown': 'strip', 'persistent_output': True, 'persistent_output_max_lines': 200, 'inline_diffs': True, 'file_mutation_verifier': True, 'show_cost': False, 'skin': 'default', 'language': 'en', 'tui_status_indicator': 'kaomoji', 'user_message_preview': {'first_lines': 2, 'last_lines': 2}, 'interim_assistant_messages': True, 'tool_progress_command': False, 'tool_progress_overrides': {}, 'tool_preview_length': 0, 'ephemeral_system_ttl': 0, 'platforms': {}, 'runtime_footer': {'enabled': False, 'fields': ['model', 'context_pct', 'cwd']}, 'copy_shortcut': 'auto'}, 'dashboard': {'theme': 'default', 'show_token_analytics': False}, 'privacy': {'redact_pii': False}, 'tts': {'provider': 'edge', 'edge': {'voice': 'en-US-AriaNeural'}, 'elevenlabs': {'voice_id': 'pNInz6obpgDQGcFmaJgB', 'model_id': 'eleven_multilingual_v2'}, 'openai': {'model': 'gpt-4o-mini-tts', 'voice': 'alloy'}, 'xai': {'voice_id': 'eve', 'language': 'en', 'sample_rate': 24000, 'bit_rate': 128000}, 'mistral': {'model': 'voxtral-mini-tts-2603', 'voice_id': 'c69964a6-ab8b-4f8a-9465-ec0925096ec8'}, 'neutts': {'ref_audio': '', 'ref_text': '', 'model': 'neuphonic/neutts-air-q4-gguf', 'device': 'cpu'}, 'piper': {'voice': 'en_US-lessac-medium'}}, 'stt': {'enabled': True, 'provider': 'local', 'local': {'model': 'base', 'language': ''}, 'openai': {'model': 'whisper-1'}, 'mistral': {'model': 'voxtral-mini-latest'}}, 'voice': {'record_key': 'ctrl+b', 'max_recording_seconds': 120, 'auto_tts': False, 'beep_enabled': True, 'silence_threshold': 200, 'silence_duration': 3.0}, 'human_delay': {'mode': 'off', 'min_ms': 800, 'max_ms': 2500}, 'context': {'engine': 'compressor'}, 'memory': {'provider': '', 'atom': {'enabled': True, 'every_n_conversations': 3, 'enable_warmup': True, 'l1_idle_timeout_seconds': 600, 'l1_batch_process': 10, 'l1_batch_query': 20, 'max_input_chars': 24000, 'max_memories_per_job': 20, 'enable_dedup': True, 'precheck_enabled': False, 'max_attempts': 5, 'empty_result_max_attempts': 2, 'retry_base_delay_seconds': 30, 'prompt_mode': 'chat', 'embedding': {'mode': 'remote', 'model': 'text-embedding-3-small', 'dimensions': 'auto', 'base_url': 'https://api.openai.com/v1', 'api_key_env': 'OPENAI_API_KEY', 'conflict_recall_top_k': 5, 'timeout': 10.0, 'batch_size': 32}, 'retrieval': {'mode': 'hybrid', 'top_k': 2, 'candidate_k': 20, 'max_context_chars': 3500, 'score_threshold': 0.5}, 'scenario': {'enabled': True, 'llm_enabled': True, 'max_scenes': 15, 'l2_delay_after_l1_seconds': 10, 'l2_min_interval_seconds': 900, 'l2_max_interval_seconds': 3600, 'session_active_window_hours': 24, 'scene_backup_count': 10}}}, 'user_profile': {'enabled': True, 'prompt_max_chars': 4000, 'min_confidence': 0.5, 'precheck_enabled': True, 'max_attempts': 3, 'retry_base_delay_seconds': 1.0}, 'delegation': {'enabled': True, 'mainline_branch_enabled': False, 'model': '', 'provider': '', 'base_url': '', 'api_key': '', 'api_mode': '', 'inherit_mcp_toolsets': True, 'default_tool_policy': 'inherit', 'router_enabled': False, 'router_min_score': 3, 'orchestrate_uncertain_tasks': False, 'enforce_complex_tasks': False, 'enforcement_max_retries': 2, 'todo_fanout_enabled': False, 'auto_verify_local_evidence': False, 'max_iterations': 50, 'child_timeout_seconds': 600, 'reasoning_effort': '', 'max_concurrent_children': 3, 'max_spawn_depth': 1, 'orchestrator_enabled': True, 'subagent_auto_approve': False}, 'prefill_messages_file': '', 'goals': {'max_turns': 20}, 'skills': {'external_dirs': [], 'template_vars': True, 'inline_shell': False, 'inline_shell_timeout': 10, 'guard_agent_created': False}, 'curator': {'enabled': True, 'interval_hours': 24 * 7, 'min_idle_hours': 2, 'stale_after_days': 30, 'archive_after_days': 90, 'backup': {'enabled': True, 'keep': 5}}, 'honcho': {}, 'timezone': '', 'slack': {'require_mention': True, 'free_response_channels': '', 'allowed_channels': '', 'channel_prompts': {}}, 'discord': {'require_mention': True, 'free_response_channels': '', 'allowed_channels': '', 'auto_thread': True, 'thread_require_mention': False, 'history_backfill': True, 'history_backfill_limit': 50, 'reactions': True, 'channel_prompts': {}, 'dm_role_auth_guild': '', 'server_actions': '', 'allow_any_attachment': False, 'max_attachment_bytes': 33554432}, 'whatsapp': {}, 'telegram': {'reactions': False, 'channel_prompts': {}, 'allowed_chats': ''}, 'mattermost': {'require_mention': True, 'free_response_channels': '', 'allowed_channels': '', 'channel_prompts': {}}, 'matrix': {'require_mention': True, 'free_response_rooms': '', 'allowed_rooms': ''}, 'approvals': {'mode': 'manual', 'timeout': 60, 'cron_mode': 'deny', 'mcp_reload_confirm': True, 'destructive_slash_confirm': True}, 'command_allowlist': [], 'quick_commands': {}, 'hooks': {}, 'hooks_auto_accept': False, 'personalities': {}, 'security': {'allow_private_urls': False, 'redact_secrets': True, 'tirith_enabled': True, 'tirith_path': 'tirith', 'tirith_timeout': 5, 'tirith_fail_open': True, 'website_blocklist': {'enabled': False, 'domains': [], 'shared_files': []}, 'acked_advisories': [], 'allow_lazy_installs': True}, 'cron': {'wrap_response': True, 'max_parallel_jobs': None}, 'kanban': {'dispatch_in_gateway': True, 'dispatch_interval_seconds': 60, 'failure_limit': 2, 'worker_log_rotate_bytes': 2 * 1024 * 1024, 'worker_log_backup_count': 1, 'orchestrator_profile': '', 'default_assignee': '', 'auto_decompose': True, 'auto_decompose_per_tick': 3, 'dispatch_stale_timeout_seconds': 14400}, 'code_execution': {'mode': 'project'}, 'logging': {'level': 'INFO', 'max_size_mb': 5, 'backup_count': 3, 'memory_monitor': {'enabled': True, 'interval_seconds': 300}}, 'model_catalog': {'enabled': True, 'url': 'https://hermes-agent.nousresearch.com/docs/api/model-catalog.json', 'ttl_hours': 24, 'providers': {}}, 'network': {'force_ipv4': False}, 'sessions': {'auto_prune': False, 'retention_days': 90, 'vacuum_after_prune': True, 'min_interval_hours': 24, 'write_json_snapshots': False}, 'onboarding': {'seen': {}}, 'updates': {'pre_update_backup': False, 'backup_keep': 5}, 'lsp': {'enabled': True, 'wait_mode': 'document', 'wait_timeout': 5.0, 'install_strategy': 'auto', 'servers': {}}, 'x_search': {'model': 'grok-4.20-reasoning', 'timeout_seconds': 180, 'retries': 2}, '_config_version': 24}
OPTIONAL_ENV_VARS = {'NOUS_BASE_URL': {'description': 'Nous Portal base URL override', 'prompt': 'Nous Portal base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'OPENROUTER_API_KEY': {'description': 'OpenRouter API key (for vision, web scraping helpers, and MoA)', 'prompt': 'OpenRouter API key', 'url': 'https://openrouter.ai/keys', 'password': True, 'tools': ['vision_analyze', 'mixture_of_agents'], 'category': 'provider', 'advanced': True}, 'GOOGLE_API_KEY': {'description': 'Google AI Studio API key (also recognized as GEMINI_API_KEY)', 'prompt': 'Google AI Studio API key', 'url': 'https://aistudio.google.com/app/apikey', 'password': True, 'category': 'provider', 'advanced': True}, 'GEMINI_API_KEY': {'description': 'Google AI Studio API key (alias for GOOGLE_API_KEY)', 'prompt': 'Gemini API key', 'url': 'https://aistudio.google.com/app/apikey', 'password': True, 'category': 'provider', 'advanced': True}, 'GEMINI_BASE_URL': {'description': 'Google AI Studio base URL override', 'prompt': 'Gemini base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'XAI_API_KEY': {'description': 'xAI API key', 'prompt': 'xAI API key', 'url': 'https://console.x.ai/', 'password': True, 'category': 'provider', 'advanced': True}, 'XAI_BASE_URL': {'description': 'xAI base URL override', 'prompt': 'xAI base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'NVIDIA_API_KEY': {'description': 'NVIDIA NIM API key (build.nvidia.com or local NIM endpoint)', 'prompt': 'NVIDIA NIM API key', 'url': 'https://build.nvidia.com/', 'password': True, 'category': 'provider', 'advanced': True}, 'NVIDIA_BASE_URL': {'description': 'NVIDIA NIM base URL override (e.g. http://localhost:8000/v1 for local NIM)', 'prompt': 'NVIDIA NIM base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'LM_API_KEY': {'description': 'LM Studio bearer token for auth-enabled local servers', 'prompt': 'LM Studio API key / bearer token', 'url': None, 'password': True, 'category': 'provider', 'advanced': True}, 'LM_BASE_URL': {'description': 'LM Studio base URL override', 'prompt': 'LM Studio base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'GLM_API_KEY': {'description': 'Z.AI / GLM API key (also recognized as ZAI_API_KEY / Z_AI_API_KEY)', 'prompt': 'Z.AI / GLM API key', 'url': 'https://z.ai/', 'password': True, 'category': 'provider', 'advanced': True}, 'ZAI_API_KEY': {'description': 'Z.AI API key (alias for GLM_API_KEY)', 'prompt': 'Z.AI API key', 'url': 'https://z.ai/', 'password': True, 'category': 'provider', 'advanced': True}, 'Z_AI_API_KEY': {'description': 'Z.AI API key (alias for GLM_API_KEY)', 'prompt': 'Z.AI API key', 'url': 'https://z.ai/', 'password': True, 'category': 'provider', 'advanced': True}, 'GLM_BASE_URL': {'description': 'Z.AI / GLM base URL override', 'prompt': 'Z.AI / GLM base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'KIMI_API_KEY': {'description': 'Kimi / Moonshot API key', 'prompt': 'Kimi API key', 'url': 'https://platform.moonshot.cn/', 'password': True, 'category': 'provider', 'advanced': True}, 'KIMI_BASE_URL': {'description': 'Kimi / Moonshot base URL override', 'prompt': 'Kimi base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'KIMI_CN_API_KEY': {'description': 'Kimi / Moonshot China API key', 'prompt': 'Kimi (China) API key', 'url': 'https://platform.moonshot.cn/', 'password': True, 'category': 'provider', 'advanced': True}, 'STEPFUN_API_KEY': {'description': 'StepFun Step Plan API key', 'prompt': 'StepFun Step Plan API key', 'url': 'https://platform.stepfun.com/', 'password': True, 'category': 'provider', 'advanced': True}, 'STEPFUN_BASE_URL': {'description': 'StepFun Step Plan base URL override', 'prompt': 'StepFun Step Plan base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'ARCEEAI_API_KEY': {'description': 'Arcee AI API key', 'prompt': 'Arcee AI API key', 'url': 'https://chat.arcee.ai/', 'password': True, 'category': 'provider', 'advanced': True}, 'ARCEE_BASE_URL': {'description': 'Arcee AI base URL override', 'prompt': 'Arcee base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'GMI_API_KEY': {'description': 'GMI Cloud API key', 'prompt': 'GMI Cloud API key', 'url': 'https://www.gmicloud.ai/', 'password': True, 'category': 'provider', 'advanced': True}, 'GMI_BASE_URL': {'description': 'GMI Cloud base URL override', 'prompt': 'GMI Cloud base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'MINIMAX_API_KEY': {'description': 'MiniMax API key (international)', 'prompt': 'MiniMax API key', 'url': 'https://www.minimax.io/', 'password': True, 'category': 'provider', 'advanced': True}, 'MINIMAX_BASE_URL': {'description': 'MiniMax base URL override', 'prompt': 'MiniMax base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'MINIMAX_CN_API_KEY': {'description': 'MiniMax API key (China endpoint)', 'prompt': 'MiniMax (China) API key', 'url': 'https://www.minimaxi.com/', 'password': True, 'category': 'provider', 'advanced': True}, 'MINIMAX_CN_BASE_URL': {'description': 'MiniMax (China) base URL override', 'prompt': 'MiniMax (China) base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'DEEPSEEK_API_KEY': {'description': 'DeepSeek API key for direct DeepSeek access', 'prompt': 'DeepSeek API Key', 'url': 'https://platform.deepseek.com/api_keys', 'password': True, 'category': 'provider'}, 'DEEPSEEK_BASE_URL': {'description': 'Custom DeepSeek API base URL (advanced)', 'prompt': 'DeepSeek Base URL', 'url': '', 'password': False, 'category': 'provider'}, 'DASHSCOPE_API_KEY': {'description': 'Alibaba Cloud DashScope API key (Qwen + multi-provider models)', 'prompt': 'DashScope API Key', 'url': 'https://modelstudio.console.alibabacloud.com/', 'password': True, 'category': 'provider'}, 'DASHSCOPE_BASE_URL': {'description': 'Custom DashScope base URL (default: coding-intl OpenAI-compat endpoint)', 'prompt': 'DashScope Base URL', 'url': '', 'password': False, 'category': 'provider', 'advanced': True}, 'HERMES_QWEN_BASE_URL': {'description': 'Qwen Portal base URL override (default: https://portal.qwen.ai/v1)', 'prompt': 'Qwen Portal base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'HERMES_GEMINI_CLIENT_ID': {'description': "Google OAuth client ID for google-gemini-cli (optional; defaults to Google's public gemini-cli client)", 'prompt': 'Google OAuth client ID (optional — leave empty to use the public default)', 'url': 'https://console.cloud.google.com/apis/credentials', 'password': False, 'category': 'provider', 'advanced': True}, 'HERMES_GEMINI_CLIENT_SECRET': {'description': 'Google OAuth client secret for google-gemini-cli (optional)', 'prompt': 'Google OAuth client secret (optional)', 'url': 'https://console.cloud.google.com/apis/credentials', 'password': True, 'category': 'provider', 'advanced': True}, 'HERMES_GEMINI_PROJECT_ID': {'description': 'GCP project ID for paid Gemini tiers (free tier auto-provisions)', 'prompt': 'GCP project ID for Gemini OAuth (leave empty for free tier)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'OPENCODE_ZEN_API_KEY': {'description': 'OpenCode Zen API key (pay-as-you-go access to curated models)', 'prompt': 'OpenCode Zen API key', 'url': 'https://opencode.ai/auth', 'password': True, 'category': 'provider', 'advanced': True}, 'OPENCODE_ZEN_BASE_URL': {'description': 'OpenCode Zen base URL override', 'prompt': 'OpenCode Zen base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'OPENCODE_GO_API_KEY': {'description': 'OpenCode Go API key ($10/month subscription for open models)', 'prompt': 'OpenCode Go API key', 'url': 'https://opencode.ai/auth', 'password': True, 'category': 'provider', 'advanced': True}, 'OPENCODE_GO_BASE_URL': {'description': 'OpenCode Go base URL override', 'prompt': 'OpenCode Go base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'HF_TOKEN': {'description': 'Hugging Face token for Inference Providers (20+ open models via router.huggingface.co)', 'prompt': 'Hugging Face Token', 'url': 'https://huggingface.co/settings/tokens', 'password': True, 'category': 'provider'}, 'HF_BASE_URL': {'description': 'Hugging Face Inference Providers base URL override', 'prompt': 'HF base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'OLLAMA_API_KEY': {'description': 'Ollama Cloud API key (ollama.com — cloud-hosted open models)', 'prompt': 'Ollama Cloud API key', 'url': 'https://ollama.com/settings', 'password': True, 'category': 'provider', 'advanced': True}, 'OLLAMA_BASE_URL': {'description': 'Ollama Cloud base URL override (default: https://ollama.com/v1)', 'prompt': 'Ollama base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'XIAOMI_API_KEY': {'description': 'Xiaomi MiMo API key for MiMo models (mimo-v2.5-pro, mimo-v2.5, mimo-v2-pro, mimo-v2-omni, mimo-v2-flash)', 'prompt': 'Xiaomi MiMo API Key', 'url': 'https://platform.xiaomimimo.com', 'password': True, 'category': 'provider'}, 'XIAOMI_BASE_URL': {'description': 'Xiaomi MiMo base URL override (default: https://api.xiaomimimo.com/v1)', 'prompt': 'Xiaomi base URL (leave empty for default)', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'AWS_REGION': {'description': 'AWS region for Bedrock API calls (e.g. us-east-1, eu-central-1)', 'prompt': 'AWS Region', 'url': 'https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html', 'password': False, 'category': 'provider', 'advanced': True}, 'AWS_PROFILE': {'description': 'AWS named profile for Bedrock authentication (from ~/.aws/credentials)', 'prompt': 'AWS Profile', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'AZURE_FOUNDRY_API_KEY': {'description': 'Azure Foundry API key for custom Azure endpoints', 'prompt': 'Azure Foundry API Key', 'url': 'https://ai.azure.com/', 'password': True, 'category': 'provider'}, 'AZURE_FOUNDRY_BASE_URL': {'description': "Azure Foundry base URL (set via 'kylin-agent-runtime model' for endpoint-specific config)", 'prompt': 'Azure Foundry base URL', 'url': None, 'password': False, 'category': 'provider', 'advanced': True}, 'EXA_API_KEY': {'description': 'Exa API key for AI-native web search and contents', 'prompt': 'Exa API key', 'url': 'https://exa.ai/', 'tools': ['web_search', 'web_extract'], 'password': True, 'category': 'tool'}, 'PARALLEL_API_KEY': {'description': 'Parallel API key for AI-native web search and extract', 'prompt': 'Parallel API key', 'url': 'https://parallel.ai/', 'tools': ['web_search', 'web_extract'], 'password': True, 'category': 'tool'}, 'FIRECRAWL_API_KEY': {'description': 'Firecrawl API key for web search and scraping', 'prompt': 'Firecrawl API key', 'url': 'https://firecrawl.dev/', 'tools': ['web_search', 'web_extract'], 'password': True, 'category': 'tool'}, 'FIRECRAWL_API_URL': {'description': 'Firecrawl API URL for self-hosted instances (optional)', 'prompt': 'Firecrawl API URL (leave empty for cloud)', 'url': None, 'password': False, 'category': 'tool', 'advanced': True}, 'FIRECRAWL_GATEWAY_URL': {'description': 'Exact Firecrawl tool-gateway origin override for Nous Subscribers only (optional)', 'prompt': 'Firecrawl gateway URL (leave empty to derive from domain)', 'url': None, 'password': False, 'category': 'tool', 'advanced': True}, 'TOOL_GATEWAY_DOMAIN': {'description': 'Shared tool-gateway domain suffix for Nous Subscribers only, used to derive vendor hosts, e.g. nousresearch.com -> firecrawl-gateway.nousresearch.com', 'prompt': 'Tool-gateway domain suffix', 'url': None, 'password': False, 'category': 'tool', 'advanced': True}, 'TOOL_GATEWAY_SCHEME': {'description': 'Shared tool-gateway URL scheme for Nous Subscribers only, used to derive vendor hosts (`https` by default, set `http` for local gateway testing)', 'prompt': 'Tool-gateway URL scheme', 'url': None, 'password': False, 'category': 'tool', 'advanced': True}, 'TOOL_GATEWAY_USER_TOKEN': {'description': 'Explicit Nous Subscriber access token for tool-gateway requests (optional; otherwise read from the Hermes auth store)', 'prompt': 'Tool-gateway user token', 'url': None, 'password': True, 'category': 'tool', 'advanced': True}, 'TAVILY_API_KEY': {'description': 'Tavily API key for AI-native web search, extract, and crawl', 'prompt': 'Tavily API key', 'url': 'https://app.tavily.com/home', 'tools': ['web_search', 'web_extract', 'web_crawl'], 'password': True, 'category': 'tool'}, 'SEARXNG_URL': {'description': 'URL of your SearXNG instance for free self-hosted web search', 'prompt': 'SearXNG URL (e.g. http://localhost:8080)', 'url': 'https://searxng.github.io/searxng/', 'tools': ['web_search'], 'password': False, 'category': 'tool'}, 'BRAVE_SEARCH_API_KEY': {'description': 'Brave Search API subscription token (free tier: 2,000 queries/mo)', 'prompt': 'Brave Search subscription token', 'url': 'https://brave.com/search/api/', 'tools': ['web_search'], 'password': True, 'category': 'tool'}, 'BROWSERBASE_API_KEY': {'description': 'Browserbase API key for cloud browser (optional — local browser works without this)', 'prompt': 'Browserbase API key', 'url': 'https://browserbase.com/', 'tools': ['browser_navigate', 'browser_click'], 'password': True, 'category': 'tool'}, 'BROWSERBASE_PROJECT_ID': {'description': 'Browserbase project ID (optional — only needed for cloud browser)', 'prompt': 'Browserbase project ID', 'url': 'https://browserbase.com/', 'tools': ['browser_navigate', 'browser_click'], 'password': False, 'category': 'tool'}, 'BROWSER_USE_API_KEY': {'description': 'Browser Use API key for cloud browser (optional — local browser works without this)', 'prompt': 'Browser Use API key', 'url': 'https://browser-use.com/', 'tools': ['browser_navigate', 'browser_click'], 'password': True, 'category': 'tool'}, 'FIRECRAWL_BROWSER_TTL': {'description': 'Firecrawl browser session TTL in seconds (optional, default 300)', 'prompt': 'Browser session TTL (seconds)', 'tools': ['browser_navigate', 'browser_click'], 'password': False, 'category': 'tool'}, 'AGENT_BROWSER_ENGINE': {'description': 'Browser engine for local mode: auto (default Chrome), lightpanda (faster, no screenshots), chrome', 'prompt': 'Browser engine (auto/lightpanda/chrome)', 'url': 'https://github.com/vercel-labs/agent-browser', 'tools': ['browser_navigate', 'browser_snapshot', 'browser_click', 'browser_vision'], 'password': False, 'category': 'tool', 'advanced': True}, 'CAMOFOX_URL': {'description': 'Camofox browser server URL for local anti-detection browsing (e.g. http://localhost:9377)', 'prompt': 'Camofox server URL', 'url': 'https://github.com/jo-inc/camofox-browser', 'tools': ['browser_navigate', 'browser_click'], 'password': False, 'category': 'tool'}, 'FAL_KEY': {'description': 'FAL API key for image and video generation', 'prompt': 'FAL API key', 'url': 'https://fal.ai/', 'tools': ['image_generate', 'video_generate'], 'password': True, 'category': 'tool'}, 'VOICE_TOOLS_OPENAI_KEY': {'description': 'OpenAI API key for voice transcription (Whisper) and OpenAI TTS', 'prompt': 'OpenAI API Key (for Whisper STT + TTS)', 'url': 'https://platform.openai.com/api-keys', 'tools': ['voice_transcription', 'openai_tts'], 'password': True, 'category': 'tool'}, 'ELEVENLABS_API_KEY': {'description': 'ElevenLabs API key for premium text-to-speech voices', 'prompt': 'ElevenLabs API key', 'url': 'https://elevenlabs.io/', 'password': True, 'category': 'tool'}, 'MISTRAL_API_KEY': {'description': 'Mistral API key for Voxtral TTS and transcription (STT)', 'prompt': 'Mistral API key', 'url': 'https://console.mistral.ai/', 'password': True, 'category': 'tool'}, 'GITHUB_TOKEN': {'description': 'GitHub token for Skills Hub (higher API rate limits, skill publish)', 'prompt': 'GitHub Token', 'url': 'https://github.com/settings/tokens', 'password': True, 'category': 'tool'}, 'NOTION_API_KEY': {'description': 'Notion integration token (used by the `notion` skill)', 'prompt': 'Notion API key', 'url': 'https://www.notion.so/my-integrations', 'password': True, 'category': 'skill', 'advanced': True}, 'LINEAR_API_KEY': {'description': 'Linear personal API key (used by the `linear` skill)', 'prompt': 'Linear API key', 'url': 'https://linear.app/settings/account/security', 'password': True, 'category': 'skill', 'advanced': True}, 'AIRTABLE_API_KEY': {'description': 'Airtable personal access token (used by the `airtable` skill)', 'prompt': 'Airtable API key', 'url': 'https://airtable.com/create/tokens', 'password': True, 'category': 'skill', 'advanced': True}, 'TENOR_API_KEY': {'description': 'Tenor API key for GIF search (used by the `gif-search` skill)', 'prompt': 'Tenor API key', 'url': 'https://developers.google.com/tenor/guides/quickstart', 'password': True, 'category': 'skill', 'advanced': True}, 'HONCHO_API_KEY': {'description': 'Honcho API key for AI-native persistent memory', 'prompt': 'Honcho API key', 'url': 'https://app.honcho.dev', 'tools': ['honcho_context'], 'password': True, 'category': 'tool'}, 'HONCHO_BASE_URL': {'description': 'Base URL for self-hosted Honcho instances (no API key needed)', 'prompt': 'Honcho base URL (e.g. http://localhost:8000)', 'category': 'tool'}, 'HERMES_LANGFUSE_PUBLIC_KEY': {'description': 'Langfuse project public key (pk-lf-...)', 'prompt': 'Langfuse public key', 'url': 'https://cloud.langfuse.com', 'password': False, 'category': 'tool'}, 'HERMES_LANGFUSE_SECRET_KEY': {'description': 'Langfuse project secret key (sk-lf-...)', 'prompt': 'Langfuse secret key', 'url': 'https://cloud.langfuse.com', 'password': True, 'category': 'tool'}, 'HERMES_LANGFUSE_BASE_URL': {'description': 'Langfuse server URL (default: https://cloud.langfuse.com)', 'prompt': 'Langfuse server URL (leave empty for cloud.langfuse.com)', 'url': None, 'password': False, 'category': 'tool', 'advanced': True}, 'TELEGRAM_BOT_TOKEN': {'description': 'Telegram bot token from @BotFather', 'prompt': 'Telegram bot token', 'url': 'https://t.me/BotFather', 'password': True, 'category': 'messaging'}, 'TELEGRAM_ALLOWED_USERS': {'description': 'Comma-separated Telegram user IDs allowed to use the bot (get ID from @userinfobot)', 'prompt': 'Allowed Telegram user IDs (comma-separated)', 'url': 'https://t.me/userinfobot', 'password': False, 'category': 'messaging'}, 'TELEGRAM_PROXY': {'description': 'Proxy URL for Telegram connections (overrides HTTPS_PROXY). Supports http://, https://, socks5://', 'prompt': 'Telegram proxy URL (optional)', 'password': False, 'category': 'messaging'}, 'DISCORD_BOT_TOKEN': {'description': 'Discord bot token from Developer Portal', 'prompt': 'Discord bot token', 'url': 'https://discord.com/developers/applications', 'password': True, 'category': 'messaging'}, 'DISCORD_ALLOWED_USERS': {'description': 'Comma-separated Discord user IDs allowed to use the bot', 'prompt': 'Allowed Discord user IDs (comma-separated)', 'url': None, 'password': False, 'category': 'messaging'}, 'DISCORD_REPLY_TO_MODE': {'description': "Discord reply threading mode: 'off' (no reply references), 'first' (reply on first message only, default), 'all' (reply on every chunk)", 'prompt': 'Discord reply mode (off/first/all)', 'url': None, 'password': False, 'category': 'messaging'}, 'SLACK_BOT_TOKEN': {'description': 'Slack bot token (xoxb-). Get from OAuth & Permissions after installing your app. Required scopes: chat:write, app_mentions:read, channels:history, groups:history, im:history, im:read, im:write, users:read, files:read, files:write', 'prompt': 'Slack Bot Token (xoxb-...)', 'url': 'https://api.slack.com/apps', 'password': True, 'category': 'messaging'}, 'SLACK_APP_TOKEN': {'description': 'Slack app-level token (xapp-) for Socket Mode. Get from Basic Information → App-Level Tokens. Also ensure Event Subscriptions include: message.im, message.channels, message.groups, app_mention', 'prompt': 'Slack App Token (xapp-...)', 'url': 'https://api.slack.com/apps', 'password': True, 'category': 'messaging'}, 'MATTERMOST_URL': {'description': 'Mattermost server URL (e.g. https://mm.example.com)', 'prompt': 'Mattermost server URL', 'url': 'https://mattermost.com/deploy/', 'password': False, 'category': 'messaging'}, 'MATTERMOST_TOKEN': {'description': 'Mattermost bot token or personal access token', 'prompt': 'Mattermost bot token', 'url': None, 'password': True, 'category': 'messaging'}, 'MATTERMOST_ALLOWED_USERS': {'description': 'Comma-separated Mattermost user IDs allowed to use the bot', 'prompt': 'Allowed Mattermost user IDs (comma-separated)', 'url': None, 'password': False, 'category': 'messaging'}, 'MATTERMOST_REQUIRE_MENTION': {'description': 'Require @mention in Mattermost channels (default: true). Set to false to respond to all messages.', 'prompt': 'Require @mention in channels', 'url': None, 'password': False, 'category': 'messaging'}, 'MATTERMOST_FREE_RESPONSE_CHANNELS': {'description': 'Comma-separated Mattermost channel IDs where bot responds without @mention', 'prompt': 'Free-response channel IDs (comma-separated)', 'url': None, 'password': False, 'category': 'messaging'}, 'MATRIX_HOMESERVER': {'description': 'Matrix homeserver URL (e.g. https://matrix.example.org)', 'prompt': 'Matrix homeserver URL', 'url': 'https://matrix.org/ecosystem/servers/', 'password': False, 'category': 'messaging'}, 'MATRIX_ACCESS_TOKEN': {'description': 'Matrix access token (preferred over password login)', 'prompt': 'Matrix access token', 'url': None, 'password': True, 'category': 'messaging'}, 'MATRIX_USER_ID': {'description': 'Matrix user ID (e.g. @hermes:example.org)', 'prompt': 'Matrix user ID (@user:server)', 'url': None, 'password': False, 'category': 'messaging'}, 'MATRIX_ALLOWED_USERS': {'description': 'Comma-separated Matrix user IDs allowed to use the bot (@user:server format)', 'prompt': 'Allowed Matrix user IDs (comma-separated)', 'url': None, 'password': False, 'category': 'messaging'}, 'MATRIX_REQUIRE_MENTION': {'description': 'Require @mention in Matrix rooms (default: true). Set to false to respond to all messages.', 'prompt': 'Require @mention in rooms (true/false)', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'MATRIX_FREE_RESPONSE_ROOMS': {'description': 'Comma-separated Matrix room IDs where bot responds without @mention', 'prompt': 'Free-response room IDs (comma-separated)', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'MATRIX_AUTO_THREAD': {'description': 'Auto-create threads for messages in Matrix rooms (default: true)', 'prompt': 'Auto-create threads in rooms (true/false)', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'MATRIX_DM_AUTO_THREAD': {'description': 'Auto-create threads for DM messages in Matrix (default: false)', 'prompt': 'Auto-create threads in DMs (true/false)', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'MATRIX_DEVICE_ID': {'description': 'Stable Matrix device ID for E2EE persistence across restarts (e.g. HERMES_BOT)', 'prompt': 'Matrix device ID (stable across restarts)', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'MATRIX_RECOVERY_KEY': {'description': 'Matrix recovery key for cross-signing verification after device key rotation (from Element: Settings → Security → Recovery Key)', 'prompt': 'Matrix recovery key', 'url': None, 'password': True, 'category': 'messaging', 'advanced': True}, 'BLUEBUBBLES_SERVER_URL': {'description': 'BlueBubbles server URL for iMessage integration (e.g. http://192.168.1.10:1234)', 'prompt': 'BlueBubbles server URL', 'url': 'https://bluebubbles.app/', 'password': False, 'category': 'messaging'}, 'BLUEBUBBLES_PASSWORD': {'description': 'BlueBubbles server password (from BlueBubbles Server → Settings → API)', 'prompt': 'BlueBubbles server password', 'url': None, 'password': True, 'category': 'messaging'}, 'BLUEBUBBLES_ALLOWED_USERS': {'description': 'Comma-separated iMessage addresses (email or phone) allowed to use the bot', 'prompt': 'Allowed iMessage addresses (comma-separated)', 'url': None, 'password': False, 'category': 'messaging'}, 'BLUEBUBBLES_ALLOW_ALL_USERS': {'description': 'Allow all BlueBubbles users without allowlist', 'prompt': 'Allow All BlueBubbles Users', 'category': 'messaging'}, 'QQ_APP_ID': {'description': 'QQ Bot App ID from QQ Open Platform (q.qq.com)', 'prompt': 'QQ App ID', 'url': 'https://q.qq.com', 'category': 'messaging'}, 'QQ_CLIENT_SECRET': {'description': 'QQ Bot Client Secret from QQ Open Platform', 'prompt': 'QQ Client Secret', 'password': True, 'category': 'messaging'}, 'QQ_ALLOWED_USERS': {'description': 'Comma-separated QQ user IDs allowed to use the bot', 'prompt': 'QQ Allowed Users', 'category': 'messaging'}, 'QQ_GROUP_ALLOWED_USERS': {'description': 'Comma-separated QQ group IDs allowed to interact with the bot', 'prompt': 'QQ Group Allowed Users', 'category': 'messaging'}, 'QQ_ALLOW_ALL_USERS': {'description': 'Allow all QQ users without an allowlist (true/false)', 'prompt': 'Allow All QQ Users', 'category': 'messaging'}, 'QQBOT_HOME_CHANNEL': {'description': 'Default QQ channel/group for cron delivery and notifications', 'prompt': 'QQ Home Channel', 'category': 'messaging'}, 'QQBOT_HOME_CHANNEL_NAME': {'description': 'Display name for the QQ home channel', 'prompt': 'QQ Home Channel Name', 'category': 'messaging'}, 'QQ_SANDBOX': {'description': 'Enable QQ sandbox mode for development testing (true/false)', 'prompt': 'QQ Sandbox Mode', 'category': 'messaging'}, 'IRC_SERVER': {'description': 'IRC server hostname (e.g. irc.libera.chat)', 'prompt': 'IRC server', 'url': None, 'password': False, 'category': 'messaging'}, 'IRC_CHANNEL': {'description': 'IRC channel to join (e.g. #hermes)', 'prompt': 'IRC channel', 'url': None, 'password': False, 'category': 'messaging'}, 'IRC_NICKNAME': {'description': 'Bot nickname on IRC (default: hermes-bot)', 'prompt': 'IRC nickname', 'url': None, 'password': False, 'category': 'messaging'}, 'IRC_SERVER_PASSWORD': {'description': 'IRC server password (if required)', 'prompt': 'IRC server password', 'url': None, 'password': True, 'category': 'messaging', 'advanced': True}, 'IRC_NICKSERV_PASSWORD': {'description': 'NickServ password for nick identification', 'prompt': 'NickServ password', 'url': None, 'password': True, 'category': 'messaging', 'advanced': True}, 'GATEWAY_ALLOW_ALL_USERS': {'description': 'Allow all users to interact with messaging bots (true/false). Default: false.', 'prompt': 'Allow all users (true/false)', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'API_SERVER_ENABLED': {'description': 'Enable the OpenAI-compatible API server (true/false). Allows frontends like Open WebUI, LobeChat, etc. to connect.', 'prompt': 'Enable API server (true/false)', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'API_SERVER_KEY': {'description': 'Bearer token for API server authentication. Required for non-loopback binding; server refuses to start without it. On loopback (127.0.0.1), all requests are allowed if empty.', 'prompt': 'API server auth key (required for network access)', 'url': None, 'password': True, 'category': 'messaging', 'advanced': True}, 'API_SERVER_PORT': {'description': 'Port for the API server (default: 8642).', 'prompt': 'API server port', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'API_SERVER_HOST': {'description': 'Host/bind address for the API server (default: 127.0.0.1). Use 0.0.0.0 for network access — server refuses to start without API_SERVER_KEY.', 'prompt': 'API server host', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'API_SERVER_MODEL_NAME': {'description': "Model name advertised on /v1/models. Defaults to the profile name (or 'hermes-agent' for the default profile). Useful for multi-user setups with OpenWebUI.", 'prompt': 'API server model name', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'GATEWAY_PROXY_URL': {'description': 'URL of a remote Hermes API server to forward messages to (proxy mode). When set, the gateway handles platform I/O only — all agent work is delegated to the remote server. Use for Docker E2EE containers that relay to a host agent. Also configurable via gateway.proxy_url in config.yaml.', 'prompt': 'Remote Hermes API server URL (e.g. http://192.168.1.100:8642)', 'url': None, 'password': False, 'category': 'messaging', 'advanced': True}, 'GATEWAY_PROXY_KEY': {'description': 'Bearer token for authenticating with the remote Hermes API server (proxy mode). Must match the API_SERVER_KEY on the remote host.', 'prompt': 'Remote API server auth key', 'url': None, 'password': True, 'category': 'messaging', 'advanced': True}, 'WEBHOOK_ENABLED': {'description': 'Enable the webhook platform adapter for receiving events from GitHub, GitLab, etc.', 'prompt': 'Enable webhooks (true/false)', 'url': None, 'password': False, 'category': 'messaging'}, 'WEBHOOK_PORT': {'description': 'Port for the webhook HTTP server (default: 8644).', 'prompt': 'Webhook port', 'url': None, 'password': False, 'category': 'messaging'}, 'WEBHOOK_SECRET': {'description': 'Global HMAC secret for webhook signature validation (overridable per route in config.yaml).', 'prompt': 'Webhook secret', 'url': None, 'password': True, 'category': 'messaging'}, 'SUDO_PASSWORD': {'description': 'Sudo password for terminal commands requiring root access; set to an explicit empty string to try empty without prompting', 'prompt': 'Sudo password', 'url': None, 'password': True, 'category': 'setting'}, 'HERMES_MAX_ITERATIONS': {'description': 'Maximum tool-calling iterations per conversation (default: 90)', 'prompt': 'Max iterations', 'url': None, 'password': False, 'category': 'setting'}, 'HERMES_TOOL_PROGRESS': {'description': '(deprecated) Use display.tool_progress in config.yaml instead', 'prompt': 'Tool progress (deprecated — use config.yaml)', 'url': None, 'password': False, 'category': 'setting'}, 'HERMES_TOOL_PROGRESS_MODE': {'description': '(deprecated) Use display.tool_progress in config.yaml instead', 'prompt': 'Progress mode (deprecated — use config.yaml)', 'url': None, 'password': False, 'category': 'setting'}, 'HERMES_PREFILL_MESSAGES_FILE': {'description': 'Path to JSON file with ephemeral prefill messages for few-shot priming', 'prompt': 'Prefill messages file path', 'url': None, 'password': False, 'category': 'setting'}, 'HERMES_EPHEMERAL_SYSTEM_PROMPT': {'description': 'Ephemeral system prompt injected at API-call time (never persisted to sessions)', 'prompt': 'Ephemeral system prompt', 'url': None, 'password': False, 'category': 'setting'}}

def _normalize_custom_provider_entry(entry: Any, *, provider_key: str='') -> Optional[Dict[str, Any]]:
    """Return a runtime-compatible custom provider entry or ``None``."""
    if not isinstance(entry, dict):
        return None
    _CAMEL_ALIASES: Dict[str, str] = {'apiKey': 'api_key', 'baseUrl': 'base_url', 'apiMode': 'api_mode', 'keyEnv': 'key_env', 'apiKeyEnv': 'key_env', 'defaultModel': 'default_model', 'contextLength': 'context_length', 'rateLimitDelay': 'rate_limit_delay'}
    if 'api_key_env' in entry and 'key_env' not in entry:
        entry['key_env'] = entry['api_key_env']
    _KNOWN_KEYS = {'name', 'api', 'url', 'base_url', 'api_key', 'key_env', 'api_key_env', 'api_mode', 'transport', 'model', 'default_model', 'models', 'context_length', 'rate_limit_delay', 'request_timeout_seconds', 'stale_timeout_seconds', 'discover_models'}
    for camel, snake in _CAMEL_ALIASES.items():
        if camel in entry and snake not in entry:
            logger.warning("providers.%s: camelCase key '%s' auto-mapped to '%s' (use snake_case to avoid this warning)", provider_key or '?', camel, snake)
            entry[snake] = entry[camel]
    unknown = set(entry.keys()) - _KNOWN_KEYS - set(_CAMEL_ALIASES.keys())
    if unknown:
        logger.warning('providers.%s: unknown config keys ignored: %s', provider_key or '?', ', '.join(sorted(unknown)))
    from urllib.parse import urlparse
    base_url = ''
    for url_key in ('base_url', 'url', 'api'):
        raw_url = entry.get(url_key)
        if isinstance(raw_url, str) and raw_url.strip():
            candidate = raw_url.strip()
            parsed = urlparse(candidate)
            if parsed.scheme and parsed.netloc:
                base_url = candidate
                break
            else:
                logger.warning("providers.%s: '%s' value '%s' is not a valid URL (no scheme or host) — skipped", provider_key or '?', url_key, candidate)
    if not base_url:
        return None
    name = ''
    raw_name = entry.get('name')
    if isinstance(raw_name, str) and raw_name.strip():
        name = raw_name.strip()
    elif provider_key.strip():
        name = provider_key.strip()
    if not name:
        return None
    normalized: Dict[str, Any] = {'name': name, 'base_url': base_url}
    provider_key = provider_key.strip()
    if provider_key:
        normalized['provider_key'] = provider_key
    api_key = entry.get('api_key')
    if isinstance(api_key, str) and api_key.strip():
        normalized['api_key'] = api_key.strip()
    key_env = entry.get('key_env')
    if isinstance(key_env, str) and key_env.strip():
        normalized['key_env'] = key_env.strip()
    api_mode = entry.get('api_mode') or entry.get('transport')
    if isinstance(api_mode, str) and api_mode.strip():
        normalized['api_mode'] = api_mode.strip()
    model_name = entry.get('model') or entry.get('default_model')
    if isinstance(model_name, str) and model_name.strip():
        normalized['model'] = model_name.strip()
    models = entry.get('models')
    if isinstance(models, dict) and models:
        normalized['models'] = models
    elif isinstance(models, list) and models:
        normalized['models'] = {str(m): {} for m in models if isinstance(m, str) and m.strip()}
    context_length = entry.get('context_length')
    if isinstance(context_length, int) and context_length > 0:
        normalized['context_length'] = context_length
    rate_limit_delay = entry.get('rate_limit_delay')
    if isinstance(rate_limit_delay, (int, float)) and rate_limit_delay >= 0:
        normalized['rate_limit_delay'] = rate_limit_delay
    discover_models = entry.get('discover_models')
    if isinstance(discover_models, bool):
        normalized['discover_models'] = discover_models
    return normalized

def providers_dict_to_custom_providers(providers_dict: Any) -> List[Dict[str, Any]]:
    """Normalize ``providers`` config entries into the legacy custom-provider shape."""
    if not isinstance(providers_dict, dict):
        return []
    custom_providers: List[Dict[str, Any]] = []
    for key, entry in providers_dict.items():
        normalized = _normalize_custom_provider_entry(entry, provider_key=str(key))
        if normalized is not None:
            custom_providers.append(normalized)
    return custom_providers

def get_compatible_custom_providers(config: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
    """Return a deduplicated custom-provider view across legacy and v12+ config.

    ``custom_providers`` remains the on-disk legacy format, while ``providers``
    is the newer keyed schema.  Runtime and picker flows still need a single
    list-shaped view, but we should not materialise that compatibility layer
    back into config.yaml because it duplicates entries in UIs.
    """
    if config is None:
        config = load_config()
    compatible: List[Dict[str, Any]] = []
    seen_provider_keys: set = set()
    seen_name_url_pairs: set = set()

    def _append_if_new(entry: Optional[Dict[str, Any]]) -> None:
        if entry is None:
            return
        provider_key = str(entry.get('provider_key', '') or '').strip().lower()
        name = str(entry.get('name', '') or '').strip().lower()
        base_url = str(entry.get('base_url', '') or '').strip().rstrip('/').lower()
        model = str(entry.get('model', '') or '').strip().lower()
        pair = (name, base_url, model)
        if provider_key and provider_key in seen_provider_keys:
            return
        if name and base_url and (pair in seen_name_url_pairs):
            return
        compatible.append(entry)
        if provider_key:
            seen_provider_keys.add(provider_key)
        if name and base_url:
            seen_name_url_pairs.add(pair)
    custom_providers = config.get('custom_providers')
    if custom_providers is not None:
        if not isinstance(custom_providers, list):
            return []
        for entry in custom_providers:
            _append_if_new(_normalize_custom_provider_entry(entry))
    for entry in providers_dict_to_custom_providers(config.get('providers')):
        _append_if_new(entry)
    return compatible
_KNOWN_ROOT_KEYS = {'_config_version', 'model', 'providers', 'fallback_model', 'fallback_providers', 'credential_pool_strategies', 'toolsets', 'agent', 'terminal', 'display', 'compression', 'delegation', 'auxiliary', 'custom_providers', 'context', 'memory', 'gateway', 'sessions'}
_CUSTOM_PROVIDER_LIKE_FIELDS = {'base_url', 'api_key', 'rate_limit_delay', 'api_mode'}

@dataclass
class ConfigIssue:
    """A detected config structure problem."""
    severity: str
    message: str
    hint: str

def validate_config_structure(config: Optional[Dict[str, Any]]=None) -> List['ConfigIssue']:
    """Validate config.yaml structure and return a list of detected issues.

    Catches common YAML formatting mistakes that produce confusing runtime
    errors (like "Unknown provider") instead of clear diagnostics.

    Can be called with a pre-loaded config dict, or will load from disk.
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            return [ConfigIssue('error', 'Could not load config.yaml', "Run 'kylin-agent-runtime setup' to create a valid config")]
    issues: List[ConfigIssue] = []
    cp = config.get('custom_providers')
    if cp is not None:
        if isinstance(cp, dict):
            issues.append(ConfigIssue('error', "custom_providers is a dict — it must be a YAML list (items prefixed with '-')", 'Change to:\n  custom_providers:\n    - name: my-provider\n      base_url: https://...\n      api_key: ...'))
            cp_keys = set(cp.keys()) if isinstance(cp, dict) else set()
            suspicious = cp_keys & _CUSTOM_PROVIDER_LIKE_FIELDS
            if suspicious:
                issues.append(ConfigIssue('warning', f'Root-level keys {sorted(suspicious)} look like custom_providers entry fields', "These should be indented under a '- name: ...' list entry, not at root level"))
        elif isinstance(cp, list):
            for i, entry in enumerate(cp):
                if not isinstance(entry, dict):
                    issues.append(ConfigIssue('warning', f'custom_providers[{i}] is not a dict (got {type(entry).__name__})', 'Each entry should have at minimum: name, base_url'))
                    continue
                if not entry.get('name'):
                    issues.append(ConfigIssue('warning', f"custom_providers[{i}] is missing 'name' field", 'Add a name, e.g.: name: my-provider'))
                if not entry.get('base_url'):
                    issues.append(ConfigIssue('warning', f"custom_providers[{i}] is missing 'base_url' field", 'Add the API endpoint URL, e.g.: base_url: https://api.example.com/v1'))
    fb = config.get('fallback_model')
    if fb is not None:
        if isinstance(fb, list):
            for i, entry in enumerate(fb):
                if not isinstance(entry, dict):
                    issues.append(ConfigIssue('error', f'fallback_model[{i}] should be a dict, got {type(entry).__name__}', 'Each entry needs provider + model'))
                else:
                    if not entry.get('provider'):
                        issues.append(ConfigIssue('warning', f"fallback_model[{i}] is missing 'provider' field", 'Add: provider: openrouter (or another provider)'))
                    if not entry.get('model'):
                        issues.append(ConfigIssue('warning', f"fallback_model[{i}] is missing 'model' field", 'Add: model: <model-name>'))
        elif not isinstance(fb, dict):
            issues.append(ConfigIssue('error', f"fallback_model should be a dict with 'provider' and 'model', got {type(fb).__name__}", 'Change to:\n  fallback_model:\n    provider: openrouter\n    model: anthropic/claude-sonnet-4'))
        elif fb:
            if not fb.get('provider'):
                issues.append(ConfigIssue('warning', "fallback_model is missing 'provider' field — fallback will be disabled", 'Add: provider: openrouter (or another provider)'))
            if not fb.get('model'):
                issues.append(ConfigIssue('warning', "fallback_model is missing 'model' field — fallback will be disabled", 'Add: model: anthropic/claude-sonnet-4 (or another model)'))
    if isinstance(cp, dict) and 'fallback_model' not in config and ('fallback_model' in (cp or {})):
        issues.append(ConfigIssue('error', 'fallback_model appears inside custom_providers instead of at root level', 'Move fallback_model to the top level of config.yaml (no indentation)'))
    model_cfg = config.get('model')
    if cp and (not model_cfg):
        issues.append(ConfigIssue('warning', "custom_providers defined but no 'model' section — Hermes won't know which provider to use", 'Add a model section:\n  model:\n    provider: custom\n    default: your-model-name\n    base_url: https://...'))
    for key in config:
        if key.startswith('_'):
            continue
        if key not in _KNOWN_ROOT_KEYS and key in _CUSTOM_PROVIDER_LIKE_FIELDS:
            issues.append(ConfigIssue('warning', f"Root-level key '{key}' looks misplaced — should it be under 'model:' or inside a 'custom_providers' entry?", f"Move '{key}' under the appropriate section"))
    return issues

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, preserving nested defaults.

    Keys in *override* take precedence. If both values are dicts the merge
    recurses, so a user who overrides only ``tts.elevenlabs.voice_id`` will
    keep the default ``tts.elevenlabs.model_id`` intact.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def _expand_env_vars(obj):
    """Recursively expand ``${VAR}`` references in config values.

    Only string values are processed; dict keys, numbers, booleans, and
    None are left untouched.  Unresolved references (variable not in
    ``os.environ``) are kept verbatim so callers can detect them.
    """
    if isinstance(obj, str):
        return re.sub('\\${([^}]+)}', lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj

def _items_by_unique_name(items):
    """Return a name-indexed dict only when all items have unique string names."""
    if not isinstance(items, list):
        return None
    indexed = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('name'), str):
            return None
        name = item['name']
        if name in indexed:
            return None
        indexed[name] = item
    return indexed

def _preserve_env_ref_templates(current, raw, loaded_expanded=None):
    """Restore raw ``${VAR}`` templates when a value is otherwise unchanged.

    ``load_config()`` expands env refs for runtime use. When a caller later
    persists that config after modifying some unrelated setting, keep the
    original on-disk template instead of writing the expanded plaintext
    secret back to ``config.yaml``.

    Prefer preserving the raw template when ``current`` still matches either
    the value previously returned by ``load_config()`` for this config path or
    the current environment expansion of ``raw``. This handles env-var
    rotation between load and save while still treating mixed literal/template
    string edits as caller-owned once their rendered value diverges.
    """
    if isinstance(current, str) and isinstance(raw, str) and re.search('\\${[^}]+}', raw):
        if current == raw:
            return raw
        if isinstance(loaded_expanded, str) and current == loaded_expanded:
            return raw
        if _expand_env_vars(raw) == current:
            return raw
        return current
    if isinstance(current, dict) and isinstance(raw, dict):
        return {key: _preserve_env_ref_templates(value, raw.get(key), loaded_expanded.get(key) if isinstance(loaded_expanded, dict) else None) for key, value in current.items()}
    if isinstance(current, list) and isinstance(raw, list):
        current_by_name = _items_by_unique_name(current)
        raw_by_name = _items_by_unique_name(raw)
        loaded_by_name = _items_by_unique_name(loaded_expanded)
        if current_by_name is not None and raw_by_name is not None:
            return [_preserve_env_ref_templates(item, raw_by_name.get(item.get('name')), loaded_by_name.get(item.get('name')) if loaded_by_name is not None else None) for item in current]
        return [_preserve_env_ref_templates(item, raw[index] if index < len(raw) else None, loaded_expanded[index] if isinstance(loaded_expanded, list) and index < len(loaded_expanded) else None) for index, item in enumerate(current)]
    return current

def _normalize_root_model_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Move stale root-level provider/base_url/context_length into model section.

    Some users (or older code) placed ``provider:``, ``base_url:``, or
    ``context_length:`` at the config root instead of inside ``model:``.
    These root-level keys are only used as a fallback when the corresponding
    ``model.*`` key is empty — they never override an existing value.
    After migration the root-level keys are removed so they can't cause
    confusion on subsequent loads.
    """
    has_root = any((config.get(k) for k in ('provider', 'base_url', 'context_length')))
    if not has_root:
        return config
    config = dict(config)
    model = config.get('model')
    if not isinstance(model, dict):
        model = {'default': model} if model else {}
        config['model'] = model
    for key in ('provider', 'base_url', 'context_length'):
        root_val = config.get(key)
        if root_val and (not model.get(key)):
            model[key] = root_val
        config.pop(key, None)
    return config

def _normalize_max_turns_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy root-level max_turns into agent.max_turns."""
    config = dict(config)
    agent_config = dict(config.get('agent') or {})
    if 'max_turns' in config and 'max_turns' not in agent_config:
        agent_config['max_turns'] = config['max_turns']
    if 'max_turns' not in agent_config:
        agent_config['max_turns'] = DEFAULT_CONFIG['agent']['max_turns']
    config['agent'] = agent_config
    config.pop('max_turns', None)
    return config

def cfg_get(cfg: Optional[Dict[str, Any]], *keys: str, default: Any=None) -> Any:
    """Traverse nested dict keys safely, returning ``default`` on any miss.

    Canonical helper for the ``cfg.get("X", {}).get("Y", default)`` pattern
    that appears 50+ times across the codebase. Handles three common gotchas
    in one place:

      1. Missing intermediate keys (returns ``default``, no KeyError).
      2. An intermediate value that's not a dict (e.g. a user wrote a string
         where a section was expected). Returns ``default`` instead of
         AttributeError on ``.get()``.
      3. ``cfg is None`` (callers sometimes pass ``load_config() or None``).

    Named ``cfg_get`` rather than ``cfg_path`` to avoid shadowing the
    ubiquitous ``cfg_path = _hermes_home / "config.yaml"`` local variable
    that appears in gateway/run.py, cron/scheduler.py, main.py, etc.

    Explicit ``None`` values are returned as-is (matches ``dict.get(key,
    default)`` semantics — ``default`` is only returned when the key is
    *absent*, not when it's present but set to ``None``).

    Examples:
        >>> cfg_get({"agent": {"reasoning_effort": "high"}}, "agent", "reasoning_effort")
        'high'
        >>> cfg_get({}, "agent", "reasoning_effort", default="medium")
        'medium'
        >>> cfg_get({"agent": "oops_a_string"}, "agent", "reasoning_effort", default="low")
        'low'
        >>> cfg_get(None, "anything", default=42)
        42
        >>> cfg_get({"a": {"b": None}}, "a", "b", default="def")  # explicit None preserved
        >>> cfg_get({"a": {"b": False}}, "a", "b", default=True)  # falsy values preserved
        False
    """
    if not isinstance(cfg, dict):
        return default
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        if key not in node:
            return default
        node = node[key]
    return node

def read_raw_config() -> Dict[str, Any]:
    """Read ~/.kylin-agent-runtime/config.yaml as-is, without merging defaults or migrating.

    Returns the raw YAML dict, or ``{}`` if the file doesn't exist or can't
    be parsed.  Use this for lightweight config reads where you just need a
    single value and don't want the overhead of ``load_config()``'s deep-merge
    + migration pipeline.

    Cached on the config file's (mtime_ns, size) — same strategy as
    ``load_config()``. Returns a deepcopy on every call since some callers
    mutate the result before passing to ``save_config()``.
    """
    with _CONFIG_LOCK:
        try:
            config_path = get_config_path()
            st = config_path.stat()
            cache_key = (st.st_mtime_ns, st.st_size)
        except (FileNotFoundError, OSError):
            return {}
        path_key = str(config_path)
        cached = _RAW_CONFIG_CACHE.get(path_key)
        if cached is not None and cached[:2] == cache_key:
            return copy.deepcopy(cached[2])
        try:
            with open(config_path, encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            _warn_config_parse_failure(config_path, e)
            return {}
        if not isinstance(data, dict):
            data = {}
        _RAW_CONFIG_CACHE[path_key] = (cache_key[0], cache_key[1], copy.deepcopy(data))
        return data

def load_config() -> Dict[str, Any]:
    """Load configuration from ~/.kylin-agent-runtime/config.yaml.

    Cached on the config file's (mtime_ns, size). Returns a deepcopy of
    the cached value when unchanged, since most call sites mutate the
    result (e.g. ``cfg["model"]["default"] = ...`` before ``save_config``).
    The cache is keyed on ``str(config_path)`` so profile switches
    (which change ``HERMES_HOME`` and therefore ``get_config_path()``)
    don't collide.

    Read-only callers should use ``load_config_readonly()`` to skip the
    defensive deepcopy — that path matters in agent-loop hot spots like
    ``get_provider_request_timeout`` which is called once per API turn.
    """
    return _load_config_impl(want_deepcopy=True)

def _load_config_impl(*, want_deepcopy: bool) -> Dict[str, Any]:
    with _CONFIG_LOCK:
        ensure_hermes_home()
        config_path = get_config_path()
        path_key = str(config_path)
        try:
            st = config_path.stat()
            cache_key: Optional[Tuple[int, int]] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            cache_key = None
        cached = _LOAD_CONFIG_CACHE.get(path_key)
        if cached is not None and cache_key is not None and (cached[:2] == cache_key):
            return copy.deepcopy(cached[2]) if want_deepcopy else cached[2]
        config = copy.deepcopy(DEFAULT_CONFIG)
        if cache_key is not None:
            try:
                with open(config_path, encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                if 'max_turns' in user_config:
                    agent_user_config = dict(user_config.get('agent') or {})
                    if agent_user_config.get('max_turns') is None:
                        agent_user_config['max_turns'] = user_config['max_turns']
                    user_config['agent'] = agent_user_config
                    user_config.pop('max_turns', None)
                config = _deep_merge(config, user_config)
            except Exception as e:
                _warn_config_parse_failure(config_path, e)
        normalized = _normalize_root_model_keys(_normalize_max_turns_config(config))
        expanded = _expand_env_vars(normalized)
        _LAST_EXPANDED_CONFIG_BY_PATH[path_key] = copy.deepcopy(expanded)
        if cache_key is not None:
            cached_copy = copy.deepcopy(expanded)
            _LOAD_CONFIG_CACHE[path_key] = (cache_key[0], cache_key[1], cached_copy)
            if not want_deepcopy:
                return cached_copy
        else:
            _LOAD_CONFIG_CACHE.pop(path_key, None)
        return expanded
_SECURITY_COMMENT = '\n# ── Security ──────────────────────────────────────────────────────────\n# Secret redaction is ON by default — strings that look like API keys,\n# tokens, and passwords are masked in tool output, logs, and chat\n# responses before the model or user ever sees them. Set redact_secrets\n# to false to disable (e.g. when developing the redactor itself).\n# tirith pre-exec scanning is enabled by default when the tirith binary\n# is available. Configure via security.tirith_* keys or env vars\n# (TIRITH_ENABLED, TIRITH_BIN, TIRITH_TIMEOUT, TIRITH_FAIL_OPEN).\n#\n# security:\n#   redact_secrets: true\n#   tirith_enabled: true\n#   tirith_path: "tirith"\n#   tirith_timeout: 5\n#   tirith_fail_open: true\n'
_FALLBACK_COMMENT = '\n# ── Fallback Model ────────────────────────────────────────────────────\n# Automatic provider failover when primary is unavailable.\n# Uncomment and configure to enable. Triggers on rate limits (429),\n# overload (529), service errors (503), or connection failures.\n#\n# Supported providers:\n#   openrouter   (OPENROUTER_API_KEY)  — routes to any model\n#   openai-codex (OAuth — kylin-agent-runtime auth) — OpenAI Codex\n#   nous         (OAuth — kylin-agent-runtime auth) — Nous Portal\n#   zai          (ZAI_API_KEY)         — Z.AI / GLM\n#   kimi-coding  (KIMI_API_KEY)        — Kimi / Moonshot\n#   kimi-coding-cn (KIMI_CN_API_KEY)   — Kimi / Moonshot (China)\n#   minimax      (MINIMAX_API_KEY)     — MiniMax\n#   minimax-cn   (MINIMAX_CN_API_KEY)  — MiniMax (China)\n#   bedrock      (AWS IAM / boto3)     — AWS Bedrock (Converse API)\n#\n# For custom OpenAI-compatible endpoints, add base_url and key_env.\n#\n# fallback_model:\n#   provider: openrouter\n#   model: anthropic/claude-sonnet-4\n'

def save_config(config: Dict[str, Any]):
    """Save configuration to ~/.kylin-agent-runtime/config.yaml."""
    with _CONFIG_LOCK:
        if is_managed():
            managed_error('save configuration')
            return
        from kylinmemory._vendor.utils import atomic_yaml_write
        ensure_hermes_home()
        config_path = get_config_path()
        current_normalized = _normalize_root_model_keys(_normalize_max_turns_config(config))
        normalized = current_normalized
        raw_existing = _normalize_root_model_keys(_normalize_max_turns_config(read_raw_config()))
        if raw_existing:
            normalized = _preserve_env_ref_templates(normalized, raw_existing, _LAST_EXPANDED_CONFIG_BY_PATH.get(str(config_path)))
        parts = []
        sec = normalized.get('security', {})
        if not sec or sec.get('redact_secrets') is None:
            parts.append(_SECURITY_COMMENT)
        fb = normalized.get('fallback_model', {})
        fb_is_valid = False
        if isinstance(fb, list):
            fb_is_valid = any((isinstance(e, dict) and e.get('provider') and e.get('model') for e in fb))
        elif isinstance(fb, dict):
            fb_is_valid = bool(fb.get('provider') and fb.get('model'))
        if not fb_is_valid:
            parts.append(_FALLBACK_COMMENT)
        atomic_yaml_write(config_path, normalized, extra_content=''.join(parts) if parts else None)
        _secure_file(config_path)
        _LAST_EXPANDED_CONFIG_BY_PATH[str(config_path)] = copy.deepcopy(current_normalized)

def load_env() -> Dict[str, str]:
    """Load environment variables from ~/.kylin-agent-runtime/.env.

    Sanitizes lines before parsing so that corrupted files (e.g.
    concatenated KEY=VALUE pairs on a single line) are handled
    gracefully instead of producing mangled values such as duplicated
    bot tokens.  See #8908.

    The parsed dict is memoised keyed on the .env file mtime, because
    ``get_env_value()`` is called dozens-to-hundreds of times per
    interactive menu render (`kylin-agent-runtime tools`, `kylin-agent-runtime setup`, status
    panels). Sanitisation is O(lines × known-keys), so re-parsing the
    same file on every call was burning ~300ms of CPU per `kylin-agent-runtime tools`
    menu paint on top of the OAuth-refresh slowness. The mtime check
    invalidates the cache when the user edits .env mid-process.
    """
    global _env_cache
    env_path = get_env_path()
    try:
        mtime = env_path.stat().st_mtime
        size = env_path.stat().st_size
        cache_key = (str(env_path), mtime, size)
    except FileNotFoundError:
        cache_key = (str(env_path), None, None)
    except Exception:
        cache_key = None
    if cache_key is not None and _env_cache is not None:
        cached_key, cached_vars = _env_cache
        if cached_key == cache_key:
            return dict(cached_vars)
    env_vars: Dict[str, str] = {}
    if env_path.exists():
        open_kw = {'encoding': 'utf-8-sig', 'errors': 'replace'}
        with open(env_path, **open_kw) as f:
            raw_lines = f.readlines()
        lines = _sanitize_env_lines(raw_lines)
        for line in lines:
            line = line.strip()
            if line and (not line.startswith('#')) and ('=' in line):
                key, _, value = line.partition('=')
                env_vars[key.strip()] = value.strip().strip('"\'')
    if cache_key is not None:
        _env_cache = (cache_key, dict(env_vars))
    return env_vars
_env_cache: Optional[Tuple[Tuple[str, Optional[float], Optional[int]], Dict[str, str]]] = None

def invalidate_env_cache() -> None:
    """Clear the load_env() process-level memo.

    Writers that mutate .env (set_env_value, save_env, etc.) call this
    to guarantee the next load_env() sees their change even on
    filesystems with coarse mtime resolution. Reads invalidate naturally
    via the mtime/size check.
    """
    global _env_cache
    _env_cache = None

def _sanitize_env_lines(lines: list) -> list:
    """Fix corrupted .env lines before reading or writing.

    Handles two known corruption patterns:
    1. Concatenated KEY=VALUE pairs on a single line (missing newline between
       entries, e.g. ``ANTHROPIC_API_KEY=sk-...OPENAI_BASE_URL=https://...``).
    2. Stale ``KEY=***`` placeholder entries left by incomplete setup runs.

    Uses a known-keys set (OPTIONAL_ENV_VARS + _EXTRA_ENV_KEYS) so we only
    split on real Hermes env var names, avoiding false positives from values
    that happen to contain uppercase text with ``=``.
    """
    known_keys = set(OPTIONAL_ENV_VARS.keys()) | _EXTRA_ENV_KEYS
    sanitized: list[str] = []
    for line in lines:
        raw = line.rstrip('\r\n')
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            sanitized.append(raw + '\n')
            continue
        match_ranges: list[tuple[int, int]] = []
        for key_name in known_keys:
            needle = key_name + '='
            idx = stripped.find(needle)
            while idx >= 0:
                match_ranges.append((idx, idx + len(needle)))
                idx = stripped.find(needle, idx + len(needle))
        split_positions = sorted({s for s, e in match_ranges if not any((s2 <= s and e2 >= e and ((s2, e2) != (s, e)) for s2, e2 in match_ranges))})
        if len(split_positions) > 1:
            for i, pos in enumerate(split_positions):
                end = split_positions[i + 1] if i + 1 < len(split_positions) else len(stripped)
                part = stripped[pos:end].strip()
                if part:
                    sanitized.append(part + '\n')
        else:
            sanitized.append(stripped + '\n')
    return sanitized

def _check_non_ascii_credential(key: str, value: str) -> str:
    """Warn and strip non-ASCII characters from credential values.

    API keys and tokens must be pure ASCII — they are sent as HTTP header
    values which httpx/httpcore encode as ASCII.  Non-ASCII characters
    (commonly introduced by copy-pasting from rich-text editors or PDFs
    that substitute lookalike Unicode glyphs for ASCII letters) cause
    ``UnicodeEncodeError: 'ascii' codec can't encode character`` at
    request time.

    Returns the sanitized (ASCII-only) value.  Prints a warning if any
    non-ASCII characters were found and removed.
    """
    try:
        value.encode('ascii')
        return value
    except UnicodeEncodeError:
        pass
    bad_chars: list[str] = []
    for i, ch in enumerate(value):
        if ord(ch) > 127:
            bad_chars.append(f'  position {i}: {ch!r} (U+{ord(ch):04X})')
    sanitized = value.encode('ascii', errors='ignore').decode('ascii')
    print(f'\n  Warning: {key} contains non-ASCII characters that will break API requests.\n  This usually happens when copy-pasting from a PDF, rich-text editor,\n  or web page that substitutes lookalike Unicode glyphs for ASCII letters.\n\n' + '\n'.join((f'  {line}' for line in bad_chars[:5])) + ('\n  ... and more' if len(bad_chars) > 5 else '') + f"\n\n  The non-ASCII characters have been stripped automatically.\n  If authentication fails, re-copy the key from the provider's dashboard.\n", file=sys.stderr)
    return sanitized

def save_env_value(key: str, value: str):
    """Save or update a value in ~/.kylin-agent-runtime/.env."""
    if is_managed():
        managed_error(f'set {key}')
        return
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f'Invalid environment variable name: {key!r}')
    value = value.replace('\n', '').replace('\r', '')
    value = _check_non_ascii_credential(key, value)
    ensure_hermes_home()
    env_path = get_env_path()
    read_kw = {'encoding': 'utf-8-sig', 'errors': 'replace'}
    write_kw = {'encoding': 'utf-8'}
    lines = []
    if env_path.exists():
        with open(env_path, **read_kw) as f:
            lines = f.readlines()
        lines = _sanitize_env_lines(lines)
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f'{key}='):
            lines[i] = f'{key}={value}\n'
            found = True
            break
    if not found:
        if lines and (not lines[-1].endswith('\n')):
            lines[-1] += '\n'
        lines.append(f'{key}={value}\n')
    fd, tmp_path = tempfile.mkstemp(dir=str(env_path.parent), suffix='.tmp', prefix='.env_')
    original_mode = None
    if env_path.exists():
        try:
            original_mode = stat.S_IMODE(env_path.stat().st_mode)
        except OSError:
            pass
    try:
        with os.fdopen(fd, 'w', **write_kw) as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, env_path)
        if original_mode is not None:
            try:
                os.chmod(env_path, original_mode)
            except OSError:
                pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    _secure_file(env_path)
    os.environ[key] = value
    invalidate_env_cache()

def get_env_value(key: str) -> Optional[str]:
    """Get a value from ~/.kylin-agent-runtime/.env or environment."""
    if key in os.environ:
        return os.environ[key]
    env_vars = load_env()
    return env_vars.get(key)

