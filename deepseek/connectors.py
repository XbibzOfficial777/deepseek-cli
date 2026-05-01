# DeepSeek CLI v7.0 — Telegram & Discord Bot Connectors
# Connects the AI agent to Telegram and Discord for remote chat.
# Features:
#   - Telegram Bot: sends/receives messages, supports markdown, long msg splitting
#   - Discord Bot: sends/receives messages, supports markdown, embed fallback
#   - Per-platform token storage in config.yaml
#   - Background thread polling (no async/await complexity)
#   - Message relay: external -> agent.chat() -> reply back
#   - Graceful start/stop with status tracking
#   - Whitelist: restrict to specific user IDs (optional)

import os
import re
import sys
import time
import threading
import traceback
from datetime import datetime

# Telegram support — uses httpx (same HTTP library used by providers.py)
# No separate 'requests' dependency needed
try:
    import httpx as _httpx_client
    _HTTPX_AVAILABLE = True
except ImportError:
    _httpx_client = None
    _HTTPX_AVAILABLE = False

TELEGRAM_LIB_AVAILABLE = _HTTPX_AVAILABLE
DISCORD_LIB_AVAILABLE = _HTTPX_AVAILABLE

# ── Telegram Bot API (pure httpx — no external deps needed) ──

class TelegramBot:
    """
    Telegram Bot using httpx HTTP client (no python-telegram-bot needed).
    Runs in a background thread, polls for updates, and relays messages
    to the agent's chat() method.

    Usage:
        bot = TelegramBot(token='123:ABC', agent_callback=my_func)
        bot.start()
        # ... messages flow ...
        bot.stop()
    """

    API_BASE = 'https://api.telegram.org/bot{token}'

    def __init__(self, token: str, agent_callback=None,
                 allowed_users=None, bot_name: str = ''):
        self.token = token
        self.agent_callback = agent_callback  # callable(user_message) -> str
        self.allowed_users = allowed_users  # list of int user IDs, or None = allow all
        self.bot_name = bot_name
        self._running = False
        self._thread = None
        self._offset = 0
        self._me = None  # bot info cache
        self._message_count = 0
        self._start_time = None
        self._last_error = ''

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def status(self) -> str:
        if not self.token:
            return 'No token'
        if self._running:
            uptime = ''
            if self._start_time:
                elapsed = time.time() - self._start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                uptime = f' ({mins}m {secs}s)'
            return f'Running{uptime} | {self._message_count} msgs'
        return 'Stopped'

    def _api(self, method: str, data: dict = None, files: dict = None,
             timeout: int = 30) -> dict:
        """Make a Telegram Bot API call."""
        if not _HTTPX_AVAILABLE or _httpx_client is None:
            self._last_error = 'httpx library not available. Run: pip install httpx'
            return {'ok': False, 'error': self._last_error}
        url = self.API_BASE.format(token=self.token) + '/' + method
        try:
            if files:
                resp = _httpx_client.post(url, data=data, files=files,
                                              timeout=timeout)
            elif data:
                resp = _httpx_client.post(url, json=data, timeout=timeout)
            else:
                resp = _httpx_client.get(url, timeout=timeout)
            result = resp.json()
            if not result.get('ok'):
                desc = result.get('description', 'Unknown error')
                self._last_error = desc
                return {'ok': False, 'error': desc}
            return result
        except Exception as e:
            self._last_error = str(e)
            return {'ok': False, 'error': str(e)}

    def validate_token(self) -> tuple:
        """Validate the bot token. Returns (True, info_str) or (False, error_str)."""
        if not self.token:
            return False, 'No token provided'
        result = self._api('getMe')
        if result.get('ok'):
            self._me = result.get('result', {})
            name = self._me.get('first_name', 'Bot')
            username = self._me.get('username', '')
            return True, f'@{username} ({name})'
        return False, result.get('error', 'Invalid token')

    def get_me(self) -> dict:
        """Get bot info."""
        if self._me:
            return self._me
        result = self._api('getMe')
        if result.get('ok'):
            self._me = result.get('result', {})
            return self._me
        return {}

    def send_message(self, chat_id: int, text: str,
                     parse_mode: str = 'Markdown') -> bool:
        """Send a message to a Telegram chat."""
        # Telegram has a 4096 char limit per message
        max_len = 4096
        if len(text) <= max_len:
            return self._send_single(chat_id, text, parse_mode)

        # Split into chunks
        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break
            # Find a good split point (newline, then space)
            split_at = remaining.rfind('\n', 0, max_len - 50)
            if split_at < max_len // 2:
                split_at = remaining.rfind(' ', 0, max_len - 50)
            if split_at < max_len // 2:
                split_at = max_len - 50
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip('\n')

        for i, chunk in enumerate(chunks):
            if i == len(chunks) - 1:
                # Last chunk — may need different parse mode
                self._send_single(chat_id, chunk, parse_mode)
            else:
                self._send_single(chat_id, chunk, parse_mode)
            time.sleep(0.3)  # Avoid rate limiting

        return True

    def _send_single(self, chat_id: int, text: str, parse_mode: str) -> bool:
        """Send a single message, with markdown fallback."""
        # Clean markdown for Telegram (remove some unsupported syntax)
        clean_text = self._clean_markdown(text)

        result = self._api('sendMessage', data={
            'chat_id': chat_id,
            'text': clean_text,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True,
        })

        if result.get('ok'):
            return True

        # Fallback: send without parse_mode
        if parse_mode != '':
            result = self._api('sendMessage', data={
                'chat_id': chat_id,
                'text': clean_text,
                'disable_web_page_preview': True,
            })
            return result.get('ok', False)

        return False

    def _clean_markdown(self, text: str) -> str:
        """Clean text for Telegram markdown compatibility."""
        # Remove markdown headers (## etc) — convert to bold
        text = re.sub(r'^#{1,6}\s+', '**', text, flags=re.MULTILINE)
        # Remove ``` code block language tags that Telegram doesn't support well
        text = re.sub(r'```\w*\n', '```\n', text)
        # Escape special chars that could break Telegram markdown
        # But be conservative — only escape if not already in a code block
        return text

    def _is_allowed(self, user_id: int) -> bool:
        """Check if a user is allowed to interact with the bot."""
        if self.allowed_users is None:
            return True  # Allow all
        return user_id in self.allowed_users

    def start(self):
        """Start the bot in a background thread."""
        if self._running:
            return
        if not self.token:
            return

        # Validate token first
        ok, info = self.validate_token()
        if not ok:
            self._last_error = info
            return

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the bot."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _poll_loop(self):
        """Background polling loop for Telegram updates."""
        while self._running:
            try:
                result = self._api('getUpdates', data={
                    'offset': self._offset,
                    'timeout': 30,  # Long polling
                    'allowed_updates': ['message'],
                }, timeout=35)

                if not result.get('ok'):
                    time.sleep(2)
                    continue

                updates = result.get('result', [])
                for update in updates:
                    self._offset = update.get('update_id', 0) + 1
                    self._handle_update(update)

            except Exception as e:
                self._last_error = str(e)
                if self._running:
                    time.sleep(3)

    def _handle_update(self, update: dict):
        """Handle a single Telegram update."""
        message = update.get('message', {})
        if not message:
            return

        # Check if it's a text message
        text = message.get('text', '')
        if not text:
            # Handle non-text (photo, sticker, etc.)
            caption = message.get('caption', '')
            if caption:
                text = caption
            else:
                return

        chat = message.get('chat', {})
        chat_id = chat.get('id', 0)
        from_user = message.get('from', {})
        user_id = from_user.get('id', 0)
        user_name = from_user.get('first_name', 'Unknown')

        # Check whitelist
        if not self._is_allowed(user_id):
            self.send_message(chat_id,
                              'Sorry, you are not authorized to use this bot.')
            return

        # Handle commands
        if text.startswith('/'):
            cmd = text.split()[0].lower()
            if cmd == '/start':
                self.send_message(chat_id,
                    'Hello! I\'m your DeepSeek CLI Agent.\n'
                    'Send me any message and I\'ll respond.\n\n'
                    'Commands:\n'
                    '/start — Show this message\n'
                    '/status — Bot status\n'
                    '/clear — Clear conversation\n'
                    '/help — Show help')
                return
            elif cmd == '/help':
                self.send_message(chat_id,
                    '**DeepSeek CLI Agent**\n\n'
                    'Send any message and I\'ll respond using AI.\n\n'
                    'Commands:\n'
                    '/start — Welcome message\n'
                    '/status — Bot status\n'
                    '/clear — Clear conversation\n'
                    '/help — This message')
                return
            elif cmd == '/status':
                uptime = ''
                if self._start_time:
                    elapsed = time.time() - self._start_time
                    mins = int(elapsed // 60)
                    uptime = f'{mins} minutes'
                self.send_message(chat_id,
                    f'**Bot Status**\n'
                    f'State: Running\n'
                    f'Messages: {self._message_count}\n'
                    f'Uptime: {uptime}')
                return
            elif cmd == '/clear':
                # Trigger clear via callback if supported
                if self.agent_callback and hasattr(self.agent_callback, 'clear_memory'):
                    self.agent_callback.clear_memory()
                    self.send_message(chat_id, 'Conversation cleared.')
                else:
                    self.send_message(chat_id, 'Clear is not supported in this mode.')
                return

        # Regular message — relay to agent
        self._message_count += 1

        if self.agent_callback:
            try:
                # Show "typing" action
                self._api('sendChatAction', data={
                    'chat_id': chat_id,
                    'action': 'typing',
                })

                # Call agent
                display_name = f'{user_name} (TG)'
                response = self.agent_callback(text, source='telegram',
                                               user=display_name)

                # Send response
                if response:
                    self.send_message(chat_id, str(response))
                else:
                    self.send_message(chat_id, '(No response)')

            except Exception as e:
                self.send_message(chat_id, f'Error: {str(e)[:500]}')
        else:
            self.send_message(chat_id,
                'Bot is running but no agent callback is configured.')


# ── Discord Bot (using webhooks / REST — no discord.py dependency) ──

class DiscordBot:
    """
    Discord Bot using pure REST API (no discord.py needed).
    Uses a simple webhook approach or direct channel message sending.

    For receiving messages, it uses a simple polling approach via
    the bot's REST API.

    Usage:
        bot = DiscordBot(token='...', channel_id='...', agent_callback=my_func)
        bot.start()
        bot.stop()
    """

    API_BASE = 'https://discord.com/api/v10'

    def __init__(self, token: str, channel_id: str = '',
                 agent_callback=None, allowed_users=None):
        self.token = token
        self.channel_id = channel_id
        self.agent_callback = agent_callback
        self.allowed_users = allowed_users  # list of str user IDs, or None
        self._running = False
        self._thread = None
        self._me = None
        self._message_count = 0
        self._start_time = None
        self._last_message_id = None  # Track last processed message
        self._last_error = ''

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def status(self) -> str:
        if not self.token:
            return 'No token'
        if self._running:
            uptime = ''
            if self._start_time:
                elapsed = time.time() - self._start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                uptime = f' ({mins}m {secs}s)'
            return f'Running{uptime} | {self._message_count} msgs'
        return 'Stopped'

    def _headers(self) -> dict:
        return {
            'Authorization': f'Bot {self.token}',
            'Content-Type': 'application/json',
            'User-Agent': 'DeepSeekCLI/7.0',
        }

    def _api(self, method: str, endpoint: str, data: dict = None,
             timeout: int = 30) -> dict:
        """Make a Discord API call."""
        url = self.API_BASE + endpoint
        try:
            if data:
                resp = _httpx_client.request(method, url, headers=self._headers(),
                                             json=data, timeout=timeout)
            else:
                resp = _httpx_client.request(method, url, headers=self._headers(),
                                             timeout=timeout)
            if resp.status_code == 204:
                return {'ok': True}
            try:
                result = resp.json()
            except Exception:
                result = {'ok': resp.status_code == 200}
            if resp.status_code >= 400:
                self._last_error = f'{resp.status_code}: {result}'
                return {'ok': False, 'error': f'{resp.status_code}: {result}'}
            result['ok'] = True
            return result
        except Exception as e:
            self._last_error = str(e)
            return {'ok': False, 'error': str(e)}

    def validate_token(self) -> tuple:
        """Validate the bot token. Returns (True, info_str) or (False, error_str)."""
        if not self.token:
            return False, 'No token provided'
        result = self._api('GET', '/users/@me')
        if result.get('ok'):
            self._me = result
            name = result.get('username', 'Bot')
            app_id = result.get('id', '?')
            return True, f'{name} (ID: {app_id})'
        return False, result.get('error', 'Invalid token')

    def send_message(self, text: str, channel_id: str = '') -> bool:
        """Send a message to a Discord channel."""
        ch_id = channel_id or self.channel_id
        if not ch_id:
            return False

        # Discord has a 2000 char limit
        max_len = 2000
        if len(text) <= max_len:
            return self._send_single(ch_id, text)

        # Split into chunks
        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break
            split_at = remaining.rfind('\n', 0, max_len - 50)
            if split_at < max_len // 2:
                split_at = remaining.rfind(' ', 0, max_len - 50)
            if split_at < max_len // 2:
                split_at = max_len - 50
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip('\n')

        for chunk in chunks:
            self._send_single(ch_id, chunk)
            time.sleep(0.5)

        return True

    def _send_single(self, channel_id: str, text: str) -> bool:
        """Send a single Discord message."""
        # Discord uses markdown differently from Telegram
        # Convert some patterns for Discord compatibility
        clean = self._clean_for_discord(text)

        result = self._api('POST', f'/channels/{channel_id}/messages', data={
            'content': clean,
        })
        return result.get('ok', False)

    def _clean_for_discord(self, text: str) -> str:
        """Clean text for Discord markdown."""
        # Remove ```lang, keep ```
        text = re.sub(r'```\w*\n', '```\n', text)
        return text.strip()

    def get_guilds(self) -> list:
        """Get list of guilds the bot is in."""
        result = self._api('GET', '/users/@me/guilds')
        if result.get('ok'):
            return result
        return []

    def get_channels(self, guild_id: str) -> list:
        """Get channels in a guild."""
        result = self._api('GET', f'/guilds/{guild_id}/channels')
        if result.get('ok'):
            return [c for c in result if c.get('type') == 0]  # Text channels only
        return []

    def _is_allowed(self, user_id: str) -> bool:
        """Check if a user is allowed."""
        if self.allowed_users is None:
            return True
        return user_id in self.allowed_users

    def start(self):
        """Start the Discord bot in a background thread."""
        if self._running:
            return
        if not self.token:
            return
        if not self.channel_id:
            return

        ok, info = self.validate_token()
        if not ok:
            self._last_error = info
            return

        self._running = True
        self._start_time = time.time()

        # Initialize: get the last message ID so we don't replay old messages
        init_result = self._api('GET',
            f'/channels/{self.channel_id}/messages?limit=1')
        if init_result.get('ok') and isinstance(init_result, list):
            if init_result:
                self._last_message_id = init_result[0].get('id')

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the Discord bot."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _poll_loop(self):
        """Background polling loop for Discord messages."""
        while self._running:
            try:
                if not self.channel_id:
                    time.sleep(5)
                    continue

                result = self._api('GET',
                    f'/channels/{self.channel_id}/messages?limit=10')

                if not result.get('ok') or not isinstance(result, list):
                    time.sleep(3)
                    continue

                # Process messages in reverse chronological order (oldest first)
                messages = list(reversed(result))
                seen_any_new = False
                for msg in messages:
                    msg_id = msg.get('id', '')
                    author = msg.get('author', {})
                    author_id = author.get('id', '')

                    # Skip if we've already processed this message
                    if self._last_message_id:
                        if msg_id == self._last_message_id:
                            break  # Hit our last seen message, stop processing
                        # Only process messages that come AFTER our last known
                        # Discord snowflake IDs are lexicographically ordered
                        if msg_id < self._last_message_id:
                            continue

                    # Skip bot's own messages
                    if author.get('bot', False):
                        if not self._last_message_id:
                            self._last_message_id = msg_id
                        continue

                    content = msg.get('content', '').strip()
                    if not content:
                        if not self._last_message_id:
                            self._last_message_id = msg_id
                        continue

                    # Update last message ID
                    self._last_message_id = msg_id
                    seen_any_new = True

                    # Check whitelist
                    if not self._is_allowed(author_id):
                        continue

                    # Handle commands
                    if content.startswith('/'):
                        cmd = content.split()[0].lower()
                        if cmd == '/help':
                            self.send_message(
                                '**DeepSeek CLI Agent**\n\n'
                                'Send any message and I\'ll respond using AI.\n\n'
                                'Commands:\n'
                                '/help — This message\n'
                                '/status — Bot status\n'
                                '/clear — Clear conversation')
                            continue

                    # Relay to agent
                    self._message_count += 1
                    if self.agent_callback:
                        try:
                            user_name = author.get('username', 'Unknown')
                            response = self.agent_callback(
                                content, source='discord',
                                user=f'{user_name} (DC)')

                            if response:
                                self.send_message(str(response))
                        except Exception as e:
                            self.send_message(f'Error: {str(e)[:500]}')

                # If no new messages, poll rate is lower
                if not seen_any_new:
                    time.sleep(3)
                else:
                    time.sleep(1)

            except Exception as e:
                self._last_error = str(e)
                if self._running:
                    time.sleep(3)


# ── Connector Manager ──

class ConnectorManager:
    """
    Manages Telegram and Discord bot connectors.
    Provides start/stop/status for both platforms.
    """

    def __init__(self):
        self.telegram: TelegramBot = None
        self.discord: DiscordBot = None
        self._agent_callback = None
        self._agent_memory = None

    def set_agent_callback(self, callback):
        """Set the agent chat callback function."""
        self._agent_callback = callback
        if self.telegram:
            self.telegram.agent_callback = callback
        if self.discord:
            self.discord.agent_callback = callback

    def set_agent_memory(self, memory):
        """Set the agent memory reference for /clear support."""
        self._agent_memory = memory

    def configure_telegram(self, token: str, allowed_users: list = None):
        """Configure and create Telegram bot instance."""
        if self.telegram and self.telegram.is_running:
            self.telegram.stop()

        self.telegram = TelegramBot(
            token=token,
            agent_callback=self._agent_callback,
            allowed_users=allowed_users,
        )
        return self.telegram

    def configure_discord(self, token: str, channel_id: str,
                          allowed_users: list = None):
        """Configure and create Discord bot instance."""
        if self.discord and self.discord.is_running:
            self.discord.stop()

        self.discord = DiscordBot(
            token=token,
            channel_id=channel_id,
            agent_callback=self._agent_callback,
            allowed_users=allowed_users,
        )
        return self.discord

    def start_telegram(self) -> tuple:
        """Start the Telegram bot. Returns (success, message)."""
        if not self.telegram:
            return False, 'Telegram not configured. Set token first.'
        if self.telegram.is_running:
            return False, 'Telegram is already running.'
        if not TELEGRAM_LIB_AVAILABLE:
            return False, 'Install httpx: pip install httpx'
        self.telegram.agent_callback = self._agent_callback
        self.telegram.start()
        if self.telegram.is_running:
            return True, f'Telegram bot started: {self.telegram.status}'
        return False, f'Failed to start: {self.telegram._last_error}'

    def stop_telegram(self) -> tuple:
        """Stop the Telegram bot."""
        if not self.telegram:
            return False, 'Telegram not configured.'
        self.telegram.stop()
        return True, 'Telegram bot stopped.'

    def start_discord(self) -> tuple:
        """Start the Discord bot. Returns (success, message)."""
        if not self.discord:
            return False, 'Discord not configured. Set token and channel ID first.'
        if self.discord.is_running:
            return False, 'Discord is already running.'
        if not DISCORD_LIB_AVAILABLE:
            return False, 'Install httpx: pip install httpx'
        self.discord.agent_callback = self._agent_callback
        self.discord.start()
        if self.discord.is_running:
            return True, f'Discord bot started: {self.discord.status}'
        return False, f'Failed to start: {self.discord._last_error}'

    def stop_discord(self) -> tuple:
        """Stop the Discord bot."""
        if not self.discord:
            return False, 'Discord not configured.'
        self.discord.stop()
        return True, 'Discord bot stopped.'

    def stop_all(self):
        """Stop all running connectors."""
        if self.telegram and self.telegram.is_running:
            self.telegram.stop()
        if self.discord and self.discord.is_running:
            self.discord.stop()

    def get_status(self) -> dict:
        """Get status of all connectors."""
        return {
            'telegram': {
                'configured': self.telegram is not None,
                'running': self.telegram.is_running if self.telegram else False,
                'status': self.telegram.status if self.telegram else 'Not configured',
                'token_set': bool(self.telegram and self.telegram.token),
            },
            'discord': {
                'configured': self.discord is not None,
                'running': self.discord.is_running if self.discord else False,
                'status': self.discord.status if self.discord else 'Not configured',
                'token_set': bool(self.discord and self.discord.token),
                'channel': self.discord.channel_id if self.discord else '',
            },
        }


# ── Global instance ──
connectors = ConnectorManager()
