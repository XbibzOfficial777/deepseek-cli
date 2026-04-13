# DeepSeek CLI v4 — Multi-Provider Configuration
# Manages 7 AI providers with YAML config file, API keys, and model selection

import os
import yaml
from pathlib import Path

CONFIG_DIR = Path.home() / '.deepseek-cli'
CONFIG_FILE = CONFIG_DIR / 'config.yaml'
LEGACY_KEY_FILE = Path.home() / '.deepseek_api_key'

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
            'X-Title': 'DeepSeek CLI v4',
        },
        'popular_models': [
            'deepseek/deepseek-r1-0528:free',
            'deepseek/deepseek-chat-v3-0324:free',
            'meta-llama/llama-4-maverick:free',
            'google/gemini-2.0-flash-exp:free',
            'qwen/qwen3-235b-a22b:free',
            'mistralai/mistral-small-3.1-24b-instruct:free',
            'anthropic/claude-sonnet-4',
            'openai/gpt-4o',
        ],
    },
    'gemini': {
        'name': 'Google Gemini',
        'type': 'gemini',
        'base_url': 'https://generativelanguage.googleapis.com/v1beta',
        'api_key_env': 'GEMINI_API_KEY',
        'default_model': 'gemini-2.0-flash',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': True,
        'get_key_url': 'https://aistudio.google.com/apikey',
        'popular_models': [
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-2.5-pro-preview-05-06',
        ],
    },
    'huggingface': {
        'name': 'HuggingFace',
        'type': 'huggingface',
        'base_url': 'https://router.huggingface.co',
        'api_key_env': 'HUGGINGFACE_API_KEY',
        'default_model': 'mistralai/Mistral-7B-Instruct-v0.3',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': True,
        'get_key_url': 'https://huggingface.co/settings/tokens',
        'popular_models': [
            'mistralai/Mistral-7B-Instruct-v0.3',
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'microsoft/Phi-3-mini-4k-instruct',
            'google/gemma-2-2b-it',
            'HuggingFaceH4/zephyr-7b-beta',
            'Qwen/Qwen2.5-72B-Instruct',
            'NousResearch/Hermes-3-Llama-3.1-8B',
            'meta-llama/Llama-3.3-70B-Instruct',
        ],
    },
    'openai': {
        'name': 'OpenAI',
        'type': 'openai_compatible',
        'base_url': 'https://api.openai.com/v1',
        'api_key_env': 'OPENAI_API_KEY',
        'default_model': 'gpt-4o-mini',
        'enabled': True,
        'supports_tools': True,
        'supports_streaming': True,
        'has_free_models': False,
        'get_key_url': 'https://platform.openai.com/api-keys',
        'popular_models': [
            'gpt-4o-mini',
            'gpt-4o',
            'gpt-4.1-mini',
            'gpt-4.1',
            'o3-mini',
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
}

# Agent settings
MAX_TOOL_ROUNDS = 8
MAX_TOKENS = 4096
TEMPERATURE = 0.7
TIMEOUT = 120

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
            'version': 4,
            'active_provider': 'openrouter',
            'api_keys': {},
            'models': {},
            'providers': {},
        }
        for pid, pdef in DEFAULT_PROVIDERS.items():
            config['providers'][pid] = dict(pdef)
        return config

    def _migrate_legacy(self, config: dict):
        """Migrate v3 ~/.deepseek_api_key to new config."""
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
        # Merge: defaults as base, stored overrides
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


def mask_key(key: str) -> str:
    """Mask API key for display: sk-or-v1-abc...xyz -> sk-or-v1-****...xyz"""
    if not key:
        return '(none)'
    if len(key) <= 10:
        return '****'
    return key[:7] + '****' + key[-4:]


# Global instance
cfg = ConfigManager()
