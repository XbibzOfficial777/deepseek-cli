# DeepSeek CLI v7.7 — Multi-Provider Configuration
# Manages 7 AI providers with YAML config file, API keys, and model selection
# NO TOOL LIMITS — all tools available at all times

import os
import sys
import platform
import socket
import yaml
from pathlib import Path

CONFIG_DIR = Path.home() / '.deepseek-cli'
CONFIG_FILE = CONFIG_DIR / 'config.yaml'
LEGACY_KEY_FILE = Path.home() / '.deepseek_api_key'
CLIENT_VERSION = "7.7"

# Default Gist ID — embedded so every install auto-connects to dashboard backend
# The Gist is public, no secret. PAT stays optional (env/config only, NOT in code).
_DEFAULT_GIST_ID = "55a91f3ee47f659d21a58a80826ca827"

# ══════════════════════════════════════
# PROVIDER DEFINITIONS
# ══════════════════════════════════════

DEFAULT_PROVIDERS = {
    'openrouter': {
        'name': 'OpenRouter',
        'type': 'openai_compatible',
        'base_url': 'https://openrouter.ai/api/v1',
        'api_key_env': 'OPENROUTER_API_KEY',
        'default_model': 'deepseek/deepseek-r1-0528:free',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': True,
        'get_key_url': 'https://openrouter.ai/keys',
        'extra_headers': {
            'HTTP-Referer': 'https://deepseek-cli.local',
            'X-Title': 'DeepSeek CLI v7.7',
        },
        'popular_models': [
            'deepseek/deepseek-r1-0528:free',
            'deepseek/deepseek-chat-v3-0324:free',
            'meta-llama/llama-4-maverick:free',
            'google/gemini-2.5-flash-preview:free',
            'qwen/qwen3-235b-a22b:free',
            'anthropic/claude-sonnet-4',
            'openai/gpt-4o',
            'openai/gpt-4.1-mini',
            'google/gemini-2.5-pro-preview-05-06',
        ],
    },
    'gemini': {
        'name': 'Google Gemini',
        'type': 'gemini',
        'base_url': 'https://generativelanguage.googleapis.com/v1beta',
        'api_key_env': 'GEMINI_API_KEY',
        'default_model': 'gemini-2.5-flash-preview-05-20',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': True,
        'get_key_url': 'https://aistudio.google.com/apikey',
        'popular_models': [
            'gemini-2.5-flash-preview-05-20',
            'gemini-2.5-pro-preview-05-06',
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
        ],
    },
    'huggingface': {
        'name': 'HuggingFace',
        'type': 'huggingface',
        'base_url': 'https://router.huggingface.co',
        'api_key_env': 'HUGGINGFACE_API_KEY',
        'default_model': 'Qwen/Qwen2.5-72B-Instruct',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': True,
        'get_key_url': 'https://huggingface.co/settings/tokens',
        'popular_models': [
            'Qwen/Qwen2.5-72B-Instruct',
            'NousResearch/Hermes-3-Llama-3.1-8B',
            'meta-llama/Llama-3.3-70B-Instruct',
            'mistralai/Mistral-7B-Instruct-v0.3',
            'HuggingFaceH4/zephyr-7b-beta',
            'microsoft/Phi-3-mini-4k-instruct',
            'google/gemma-2-2b-it',
        ],
    },
    'openai': {
        'name': 'OpenAI',
        'type': 'openai_compatible',
        'base_url': 'https://api.openai.com/v1',
        'api_key_env': 'OPENAI_API_KEY',
        'default_model': 'gpt-4.1-mini',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': False,
        'get_key_url': 'https://platform.openai.com/api-keys',
        'popular_models': [
            'gpt-4.1-mini',
            'gpt-4.1',
            'gpt-4o',
            'gpt-4o-mini',
            'o3-mini',
            'o4-mini',
        ],
    },
    'anthropic': {
        'name': 'Anthropic (Claude)',
        'type': 'anthropic',
        'base_url': 'https://api.anthropic.com/v1',
        'api_key_env': 'ANTHROPIC_API_KEY',
        'default_model': 'claude-sonnet-4-20250514',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': False,
        'get_key_url': 'https://console.anthropic.com/settings/keys',
        'popular_models': [
            'claude-sonnet-4-20250514',
            'claude-haiku-4-20250414',
            'claude-3-5-haiku-20241022',
            'claude-3-5-sonnet-20241022',
        ],
    },
    'groq': {
        'name': 'Groq',
        'type': 'openai_compatible',
        'base_url': 'https://api.groq.com/openai/v1',
        'api_key_env': 'GROQ_API_KEY',
        'default_model': 'llama-3.3-70b-versatile',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': True,
        'get_key_url': 'https://console.groq.com/keys',
        'popular_models': [
            'llama-3.3-70b-versatile',
            'llama-3.1-8b-instant',
            'mixtral-8x7b-32768',
            'gemma2-9b-it',
            'qwen-qwq-32b',
        ],
    },
    'together': {
        'name': 'Together AI',
        'type': 'openai_compatible',
        'base_url': 'https://api.together.xyz/v1',
        'api_key_env': 'TOGETHER_API_KEY',
        'default_model': 'meta-llama/Llama-3-70b-chat-hf',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': True,
        'get_key_url': 'https://api.together.xyz/settings/api-keys',
        'popular_models': [
            'meta-llama/Llama-3-70b-chat-hf',
            'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'meta-llama/Llama-3-8b-chat-hf',
            'Qwen/Qwen2.5-72B-Instruct-Turbo',
        ],
    },
    'agnes': {
        'name': 'Agnes AI',
        'type': 'openai_compatible',
        'base_url': 'https://apihub.agnes-ai.com/v1',
        'api_key_env': 'AGNES_API_KEY',
        'default_model': 'agnes-2.0-flash',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': True,
        'get_key_url': 'https://platform.agnes-ai.com',
        'popular_models': [
            'agnes-2.0-flash',
        ],
    },
}

# Agent settings
_stored_config = {}
try:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as _f:
            _stored_config = yaml.safe_load(_f) or {}
except Exception:
    pass

MAX_TOOL_ROUNDS = _stored_config.get('max_tool_rounds', 0)
MAX_TOKENS = _stored_config.get('max_tokens', 16384)
TEMPERATURE = _stored_config.get('temperature', 0.7)
TIMEOUT = None           # No HTTP timeout — AI determines response time
TOOL_TIMEOUT = 0         # 0 = no tool timeout, tools run until completion

# UI
BANNER_COLOR = 'cyan'
ACCENT_COLOR = 'green'
THINKING_VISIBLE = True


# ══════════════════════════════════════
# CONFIG MANAGER
# ══════════════════════════════════════

class ConfigManager:
    """Manages multi-provider configuration with YAML persistence."""

    def __init__(self):
        self.config = self._load()

    def _load(self) -> dict:
        """Load config from YAML file or create from defaults."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict) and 'providers' in data:
                    self._migrate_legacy(data)
                    return data
            except Exception:
                pass

        # Create default config
        config = {
            'version': 5,
            'active_provider': 'openrouter',
            'api_keys': {},
            'models': {},
            'providers': {},
        }
        for pid, pdef in DEFAULT_PROVIDERS.items():
            config['providers'][pid] = dict(pdef)
        return config

    def _migrate_legacy(self, config: dict):
        """Migrate old configs to new format."""
        if LEGACY_KEY_FILE.exists():
            try:
                key = LEGACY_KEY_FILE.read_text().strip()
                if key and not config.get('api_keys', {}).get('openrouter'):
                    if 'api_keys' not in config:
                        config['api_keys'] = {}
                    config['api_keys']['openrouter'] = key
            except Exception:
                pass

    def save(self):
        """Save config to YAML file."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)
            os.chmod(CONFIG_FILE, 0o600)
        except Exception:
            pass

    # ── Provider ────────────────────────

    @property
    def active_provider(self) -> str:
        return self.config.get('active_provider', 'openrouter')

    @active_provider.setter
    def active_provider(self, provider_id: str):
        self.config['active_provider'] = provider_id
        self.save()

    def get_provider_config(self, provider_id: str = None) -> dict:
        """Get full config dict for a provider (merged with defaults)."""
        pid = provider_id or self.active_provider
        stored = self.config.get('providers', {}).get(pid, {})
        defaults = DEFAULT_PROVIDERS.get(pid, {})
        merged = dict(defaults)
        merged.update(stored)
        return merged

    def get_all_providers(self) -> list[dict]:
        """Get list of all provider info dicts."""
        result = []
        for pid in DEFAULT_PROVIDERS:
            pconfig = self.get_provider_config(pid)
            api_key = self.get_api_key(pid)
            result.append({
                'id': pid,
                **pconfig,
                'has_key': bool(api_key),
                'active': pid == self.active_provider,
            })
        return result

    # ── API Key ─────────────────────────

    def get_api_key(self, provider_id: str = None) -> str:
        """Get API key: priority = saved config > env var > empty."""
        pid = provider_id or self.active_provider
        pconfig = self.get_provider_config(pid)

        saved = self.config.get('api_keys', {}).get(pid, '')
        if saved:
            return saved

        env_var = pconfig.get('api_key_env', '')
        if env_var:
            return os.environ.get(env_var, '')

        return ''

    def set_api_key(self, key: str, provider_id: str = None):
        """Save API key for a provider (file + env var)."""
        pid = provider_id or self.active_provider
        key = key.strip()
        if not key:
            return

        if 'api_keys' not in self.config:
            self.config['api_keys'] = {}
        self.config['api_keys'][pid] = key

        pconfig = self.get_provider_config(pid)
        env_var = pconfig.get('api_key_env', '')
        if env_var:
            os.environ[env_var] = key

        self.save()

    def delete_api_key(self, provider_id: str = None) -> bool:
        """Delete saved API key for a provider."""
        pid = provider_id or self.active_provider
        keys = self.config.get('api_keys', {})
        if pid in keys:
            del keys[pid]
            pconfig = self.get_provider_config(pid)
            env_var = pconfig.get('api_key_env', '')
            if env_var:
                os.environ.pop(env_var, None)
            self.save()
            return True
        return False

    # ── Model ───────────────────────────

    def get_provider_model(self, provider_id: str = None) -> str:
        """Get selected model for a provider (saved > default)."""
        pid = provider_id or self.active_provider
        saved = self.config.get('models', {}).get(pid, '')
        if saved:
            return saved
        return self.get_provider_config(pid).get('default_model', '')

    def set_provider_model(self, model: str, provider_id: str = None):
        """Save selected model for a provider."""
        pid = provider_id or self.active_provider
        if 'models' not in self.config:
            self.config['models'] = {}
        self.config['models'][pid] = model
        self.save()

    # ── Connectors ──────────────────────

    def get_connector_config(self, platform: str) -> dict:
        """Get connector config for a platform (telegram/discord)."""
        connectors = self.config.get('connectors', {})
        return connectors.get(platform, {})

    def set_connector_config(self, platform: str, key: str, value):
        """Set a connector config value."""
        if 'connectors' not in self.config:
            self.config['connectors'] = {}
        if platform not in self.config['connectors']:
            self.config['connectors'][platform] = {}
        self.config['connectors'][platform][key] = value
        self.save()

    def get_connector_token(self, platform: str) -> str:
        """Get token for a connector platform."""
        cfg = self.get_connector_config(platform)
        # Priority: saved config > env var
        env_map = {'telegram': 'TELEGRAM_BOT_TOKEN', 'discord': 'DISCORD_BOT_TOKEN'}
        saved = cfg.get('token', '')
        if saved:
            return saved
        return os.environ.get(env_map.get(platform, ''), '')

    def set_connector_token(self, platform: str, token: str):
        """Save connector token."""
        self.set_connector_config(platform, 'token', token.strip())

    # ── MCP Servers ─────────────────────

    def get_mcp_servers(self) -> dict:
        """Get all configured MCP servers."""
        return self.config.get('mcp_servers', {})

    def get_mcp_server(self, server_id: str) -> dict:
        """Get config for a specific MCP server."""
        servers = self.get_mcp_servers()
        return servers.get(server_id, {})

    def set_mcp_server(self, server_id: str, server_config: dict):
        """Add or update an MCP server config."""
        if 'mcp_servers' not in self.config:
            self.config['mcp_servers'] = {}
        self.config['mcp_servers'][server_id] = server_config
        self.save()

    def remove_mcp_server(self, server_id: str) -> bool:
        """Remove an MCP server config."""
        servers = self.get_mcp_servers()
        if server_id in servers:
            del self.config['mcp_servers'][server_id]
            self.save()
            return True
        return False

    def enable_mcp_server(self, server_id: str, enabled: bool = True):
        """Enable or disable an MCP server."""
        servers = self.get_mcp_servers()
        if server_id in servers:
            servers[server_id]['enabled'] = enabled
            self.config['mcp_servers'] = servers
            self.save()


def mask_key(key: str) -> str:
    """Mask API key for display."""
    if not key:
        return '(none)'
    if len(key) <= 10:
        return '****'
    return key[:7] + '****' + key[-4:]


# Global instance
cfg = ConfigManager()

# Cached usage status from enforce_gist() — avoids redundant API calls that can fail on Termux
_cached_usage_status = None

# Cached update info from enforce_gist() — populated at startup, read by the banner.
# Everything is driven by the registry Gist's "latest_version"; nothing is hardcoded
# in the UI, so bumping the Gist instantly changes every client.
#   None              -> not checked yet / check failed
#   {} (empty dict)   -> checked, already up to date
#   {'latest': '7.8'} -> checked, a newer version is available
_update_info = None


def _parse_version(v):
    """Parse a version string like '7.7' or 'v7.7.1' into a tuple of ints.

    Non-numeric/garbage segments are ignored so a malformed remote version
    can never crash the client. Returns () on total failure."""
    if not v:
        return ()
    s = str(v).strip().lstrip('vV').strip()
    parts = []
    for chunk in s.split('.'):
        num = ''.join(ch for ch in chunk if ch.isdigit())
        if num == '':
            break
        parts.append(int(num))
    return tuple(parts)


def is_newer_version(latest, current=CLIENT_VERSION):
    """Return True only when `latest` is strictly greater than `current`.

    Uses tuple comparison with zero-padding so '7.7' == '7.7.0' (NOT an update)
    while '7.8' or '7.7.1' > '7.7' (IS an update). Falls back to a safe string
    compare if either version can't be parsed."""
    lt = _parse_version(latest)
    ct = _parse_version(current)
    if not lt or not ct:
        # Couldn't parse — only flag as update if they differ literally
        return bool(latest) and str(latest).strip().lstrip('vV') != str(current).strip().lstrip('vV')
    n = max(len(lt), len(ct))
    lt = lt + (0,) * (n - len(lt))
    ct = ct + (0,) * (n - len(ct))
    return lt > ct


def get_update_info() -> dict:
    """Return cached update info populated by enforce_gist().

    Returns {} when up to date or not yet checked, or {'latest': <ver>,
    'current': <ver>} when a newer version is available."""
    global _update_info
    return _update_info or {}


def enforce_gist():
    """Fetches resolved Worker API and checks if the current public IP is banned or limited."""
    import os
    import sys
    import urllib.request
    import urllib.error
    import json
    import getpass
    import socket

    # 1. Read Registry Gist ID from environment variables, config file, or built-in default
    registry_gist_id = os.environ.get("DEEPSEEK_GIST_ID", "") or cfg.config.get("gist_id", "") or _DEFAULT_GIST_ID

    # 2. Get public IP address
    # print("\033[93m[*] Checking network permissions against Gist Database...\033[0m")
    client_ip = "127.0.0.1"
    try:
        req = urllib.request.Request("https://api.ipify.org?format=json", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            client_ip = json.loads(response.read().decode()).get("ip", "127.0.0.1")
    except Exception:
        pass

    # 3. Fetch registry to find Cloudflare Worker URL
    api_url = None
    try:
        url = f"https://api.github.com/gists/{registry_gist_id}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        gist_pat = os.environ.get("DEEPSEEK_GIST_PAT", "") or cfg.config.get("gist_pat", "")
        if gist_pat:
            headers["Authorization"] = f"token {gist_pat}"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            gist_content = json.loads(response.read().decode())
            file_data = gist_content.get("files", {}).get("endpoint.json", {})
            if not file_data:
                print("\033[91mError: Registry endpoint.json not found in registry Gist.\033[0m", file=sys.stderr)
                sys.exit(1)
            registry_payload = json.loads(file_data["content"])
            api_url = registry_payload.get("api_url")
            latest_version = registry_payload.get("latest_version")
            # Record update availability so the banner (shown AFTER this check)
            # can render "(Update Available vX.Y)". Printing here would scroll
            # above the big ASCII banner and effectively be invisible. The value
            # comes entirely from the registry Gist, so changing it there updates
            # every client automatically — no manual edits needed.
            global _update_info
            if latest_version and is_newer_version(latest_version, CLIENT_VERSION):
                _update_info = {"latest": str(latest_version).strip().lstrip("vV"),
                                "current": CLIENT_VERSION}
            else:
                _update_info = {}
    except Exception as e:
        print(f"\033[91mFailed to resolve dashboard backend: {e}\033[0m", file=sys.stderr)
        sys.exit(1)

    if not api_url:
        print("\033[91mError: api_url not defined in registry Gist.\033[0m", file=sys.stderr)
        sys.exit(1)

    # 4. Check permissions from Worker API
    try:
        check_url = f"{api_url.rstrip('/')}/api/check?ip={client_ip}"
        req = urllib.request.Request(check_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            result = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"\033[91mFailed to verify access permissions with server (HTTP {e.code}).\033[0m", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\033[91mFailed to verify access permissions with server: {e}\033[0m", file=sys.stderr)
        sys.exit(1)

    is_banned = result.get("banned", False)
    is_limited = result.get("limit_exceeded", False)
    total_tokens = result.get("usage", 0)
    token_limit = result.get("limit", 0)

    # Check Ban state
    if is_banned:
        print("\n\033[1;31m██████████████████████████████████████████████████\033[0m", file=sys.stderr)
        print(f"\033[1;31mACCESS DENIED! IP {client_ip} has been BANNED.\033[0m", file=sys.stderr)
        print("\033[1;31mPlease contact the network administrator to restore access.\033[0m", file=sys.stderr)
        print("\033[1;31m██████████████████████████████████████████████████\n\033[0m", file=sys.stderr)
        sys.exit(1)

    # Check Limit state
    if is_limited:
        print("\n\033[1;31m██████████████████████████████████████████████████\033[0m", file=sys.stderr)
        print("\033[1;31mACCESS DENIED! Token limit has been exceeded.\033[0m", file=sys.stderr)
        print(f"\033[1;31mConsumed: {total_tokens:,} / Limit: {token_limit:,} tokens.\033[0m", file=sys.stderr)
        print("\033[1;31mPlease contact the network administrator to raise your limit.\033[0m", file=sys.stderr)
        print("\033[1;31m██████████████████████████████████████████████████\n\033[0m", file=sys.stderr)
        sys.exit(1)

    # Register client if not found
    if not result.get("found", False):
        try:
            username = f"{getpass.getuser()}@{socket.gethostname()}"
        except Exception:
            username = f"cli_client_{client_ip.replace('.', '_')}"
        
        try:
            try:
                _hostname = socket.gethostname()
            except Exception:
                _hostname = "unknown"
            payload = {
                "ip": client_ip,
                "username": username,
                "input_tokens": 0,
                "output_tokens": 0,
                "last_tool": "initialization",
                "status": "online",
                "version": CLIENT_VERSION,
                "hostname": _hostname,
                "platform": sys.platform,
                "arch": platform.machine(),
                "os_release": platform.release(),
                "device_name": username
            }
            req_update = urllib.request.Request(
                f"{api_url.rstrip('/')}/api/update",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                method="POST"
            )
            with urllib.request.urlopen(req_update, timeout=5) as _:
                pass
        except Exception as reg_err:
            print(f"\033[93m[!] Failed to register client: {reg_err}\033[0m", file=sys.stderr)

    global _cached_usage_status
    _cached_usage_status = {
        "ip": client_ip,
        "usage": result.get("usage", 0),
        "limit": result.get("limit", 0),
        "last_tool": result.get("last_tool", "-"),
        "total_calls": result.get("total_calls", 0),
        "username": result.get("username", "Unknown"),
        "banned": result.get("banned", False),
        "limit_exceeded": result.get("limit_exceeded", False),
        "found": result.get("found", False)
    }

    limit_str = f"{token_limit:,}" if token_limit else "unli"
    # print(f"\033[92m✓ Permissions verified. IP: {client_ip} (Usage: {total_tokens:,} / Limit: {limit_str})\033[0m")


def update_gist_usage(input_tokens: int, output_tokens: int, last_tool: str):
    """Updates the token counts, status, and last tool of the client IP on the Worker backend."""
    import os
    import urllib.request
    import urllib.error
    import json
    import getpass
    import socket

    registry_gist_id = os.environ.get("DEEPSEEK_GIST_ID", "") or cfg.config.get("gist_id", "") or _DEFAULT_GIST_ID

    client_ip = "127.0.0.1"
    try:
        req = urllib.request.Request("https://api.ipify.org?format=json", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            client_ip = json.loads(response.read().decode()).get("ip", "127.0.0.1")
    except Exception:
        pass

    # Fetch registry to find Cloudflare Worker URL
    api_url = None
    try:
        url = f"https://api.github.com/gists/{registry_gist_id}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        gist_pat = os.environ.get("DEEPSEEK_GIST_PAT", "") or cfg.config.get("gist_pat", "")
        if gist_pat:
            headers["Authorization"] = f"token {gist_pat}"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            gist_content = json.loads(response.read().decode())
            file_data = gist_content.get("files", {}).get("endpoint.json", {})
            if file_data:
                api_url = json.loads(file_data["content"]).get("api_url")
    except Exception:
        return

    if not api_url:
        return

    try:
        username = f"{getpass.getuser()}@{socket.gethostname()}"
    except Exception:
        username = f"cli_client_{client_ip.replace('.', '_')}"

    try:
        try:
            _hostname = socket.gethostname()
        except Exception:
            _hostname = "unknown"
        payload = {
            "ip": client_ip,
            "username": username,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "last_tool": last_tool,
            "status": "online",
            "version": CLIENT_VERSION,
            "hostname": _hostname,
            "platform": sys.platform,
            "arch": platform.machine(),
            "os_release": platform.release(),
            "device_name": username
        }
        req_update = urllib.request.Request(
            f"{api_url.rstrip('/')}/api/update",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST"
        )
        with urllib.request.urlopen(req_update, timeout=8) as _:
            pass
    except Exception:
        pass


def get_usage_status() -> dict:
    """Returns cached usage status from startup check."""
    global _cached_usage_status
    return _cached_usage_status

