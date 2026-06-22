# DeepSeek CLI v7.7 — Real MCP Client
# Connects to external MCP servers via stdio or SSE
# Supports popular servers: Canva, Context7, GitHub, Brave Search, etc.
#
# Architecture:
#   MCPClientManager → manages multiple MCPConnection
#   MCPConnection → single server connection (stdio or SSE)
#   Each server exposes tools → registered into ToolRegistry
#
# CRITICAL: All async context managers (stdio_client, ClientSession)
# must live in the SAME asyncio task. We use a persistent background
# coroutine (_session_lifecycle) that keeps the session alive.

import os
import sys
import json
import asyncio
import logging
import threading
import subprocess
from contextlib import contextmanager
from typing import Any, Optional


@contextmanager
def _silence_stderr():
    """Temporarily redirect stderr to devnull during MCP operations."""
    old_stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = old_stderr

# MCP client imports
try:
    from mcp import ClientSession, StdioServerParameters, stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
    # Suppress MCP library's noisy JSON parse errors (servers print banner to stdout)
    import logging as _logging
    _logging.getLogger('mcp.client.stdio').setLevel(_logging.CRITICAL)
    _logging.getLogger('mcp.client.sse').setLevel(_logging.CRITICAL)
    _logging.getLogger('mcp').setLevel(_logging.CRITICAL)
except ImportError:
    MCP_AVAILABLE = False


# ══════════════════════════════════════
# POPULAR MCP SERVER REGISTRY
# ══════════════════════════════════════

POPULAR_MCP_SERVERS = {
    "brave-search": {
        "name": "Brave Search",
        "description": "Web search via Brave Search API",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env_key": "BRAVE_API_KEY",
        "install": "npm install -g @modelcontextprotocol/server-brave-search",
        "tools_hint": ["brave_web_search", "brave_local_search"],
    },
    "github": {
        "name": "GitHub",
        "description": "GitHub repo management, issues, PRs, file operations",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_key": "GITHUB_PERSONAL_ACCESS_TOKEN",
        "install": "npm install -g @modelcontextprotocol/server-github",
        "tools_hint": ["search_repositories", "create_repository", "get_file_contents",
                       "create_issue", "list_issues", "create_pull_request"],
    },
    "filesystem": {
        "name": "Filesystem",
        "description": "Secure filesystem operations with access controls",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", os.path.expanduser("~")],
        "env_key": None,
        "install": "npm install -g @modelcontextprotocol/server-filesystem",
        "tools_hint": ["read_file", "write_file", "list_directory", "create_directory",
                       "move_file", "search_files", "get_file_info"],
    },
    "memory": {
        "name": "Memory",
        "description": "Persistent knowledge graph memory for context retention",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "env_key": None,
        "install": "npm install -g @modelcontextprotocol/server-memory",
        "tools_hint": ["create_entities", "create_relations", "search_nodes",
                       "open_nodes", "delete_entities"],
    },
    "context7": {
        "name": "Context7",
        "description": "Up-to-date library documentation for any package",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp@latest"],
        "env_key": None,
        "install": "npm install -g @upstash/context7-mcp",
        "tools_hint": ["resolve-library-id", "get-library-docs"],
    },
    "canva": {
        "name": "Canva",
        "description": "Create and edit designs, templates, and graphics in Canva",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@canva/cli@latest", "mcp"],
        "env_key": "CANVA_API_KEY",
        "install": "npm install -g @canva/cli",
        "tools_hint": ["create_design", "get_design", "list_designs",
                       "create_design_from_template", "export_design"],
        "oauth_url": "https://www.canva.com/developer/apps/AAHAALoYlkA/credentials",
        "help_text": "Get API key from Canva Developer Console at: https://www.canva.com/developer/apps/AAHAALoYlkA/credentials",
    },
    "fetch": {
        "name": "Fetch",
        "description": "HTTP fetch/web scraping with content extraction",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "env_key": None,
        "install": "npm install -g @modelcontextprotocol/server-fetch",
        "tools_hint": ["fetch", "get_markdown"],
    },
    "sqlite": {
        "name": "SQLite",
        "description": "SQLite database operations and queries",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite"],
        "env_key": None,
        "install": "npm install -g @modelcontextprotocol/server-sqlite",
        "tools_hint": ["query", "list_tables", "describe_table",
                       "create_table", "write_query", "read_query"],
    },
    "postgres": {
        "name": "PostgreSQL",
        "description": "PostgreSQL database operations and queries",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres"],
        "env_key": "DATABASE_URL",
        "install": "npm install -g @modelcontextprotocol/server-postgres",
        "tools_hint": ["query", "list_tables", "describe_table"],
    },
    "puppeteer": {
        "name": "Puppeteer",
        "description": "Browser automation with Puppeteer (screenshots, navigation, JS eval)",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "env_key": None,
        "install": "npm install -g @modelcontextprotocol/server-puppeteer",
        "tools_hint": ["puppeteer_navigate", "puppeteer_screenshot",
                       "puppeteer_click", "puppeteer_fill", "puppeteer_evaluate"],
    },
    "everything": {
        "name": "Everything (Reference)",
        "description": "Reference MCP server with all features for testing",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-everything"],
        "env_key": None,
        "install": "npm install -g @modelcontextprotocol/server-everything",
        "tools_hint": ["echo", "add", "longRunningOperation"],
    },
    "sequential-thinking": {
        "name": "Sequential Thinking",
        "description": "Structured step-by-step reasoning and problem solving",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "env_key": None,
        "install": "npm install -g @modelcontextprotocol/server-sequential-thinking",
        "tools_hint": ["sequentialthinking"],
    },
    "sqlite-npx": {
        "name": "SQLite (NPX)",
        "description": "SQLite database via NPX - no global install needed",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "mcp-server-sqlite"],
        "env_key": None,
        "install": None,
        "tools_hint": ["query", "list_tables"],
    },
}


# ══════════════════════════════════════
# MCP CONNECTION
# ══════════════════════════════════════

class MCPConnection:
    """
    Single MCP server connection.
    The session lifecycle runs as a single persistent coroutine in the
    event loop thread to avoid anyio cancel scope violations.
    """

    def __init__(self, server_id: str, config: dict, loop: asyncio.AbstractEventLoop):
        self.server_id = server_id
        self.config = config
        self.loop = loop
        self.session: Optional[ClientSession] = None
        self.tools: list[dict] = []
        self.connected = False
        self.error: Optional[str] = None
        self._stop_event = threading.Event()
        self._task: Optional[asyncio.Task] = None
        self._tool_futures: dict[str, asyncio.Future] = {}

    def _get_env(self) -> Optional[dict]:
        """Build env dict with required API key."""
        env = dict(os.environ)
        env_key = self.config.get('env_key')
        if env_key:
            if env_key not in env:
                stored = self.config.get('env_value', '')
                if stored:
                    env[env_key] = stored
                else:
                    self.error = f"Missing env var: {env_key}"
                    return None
        return env

    async def _session_lifecycle(self):
        """
        Persistent session coroutine.
        ALL async context managers live here in ONE task.
        Never calls __aenter__/__aexit__ across thread boundaries.
        """
        transport = self.config.get('transport', 'stdio')
        env = self._get_env()
        if env is None:
            self._stop_event.set()
            return

        try:
            if transport == 'sse':
                await self._run_sse_session(env)
            else:
                await self._run_stdio_session(env)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.error = str(e)
        finally:
            self.connected = False
            self.session = None
            self._stop_event.set()

    async def _run_stdio_session(self, env: dict):
        """Run session with stdio transport."""
        command = self.config.get('command', '')
        args = self.config.get('args', [])

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        with _silence_stderr():
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self.connected = True

                    # Discover tools
                    tools_result = await session.list_tools()
                    self.tools = []
                    for tool in tools_result.tools:
                        self.tools.append({
                            'name': tool.name,
                            'description': tool.description or '',
                            'input_schema': tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        })

                    # Serve until told to stop
                    while not self._stop_event.is_set():
                        await asyncio.sleep(0.1)

    async def _run_sse_session(self, env: dict):
        """Run session with SSE transport."""
        url = self.config.get('url', '')

        with _silence_stderr():
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self.connected = True

                    # Discover tools
                    tools_result = await session.list_tools()
                    self.tools = []
                    for tool in tools_result.tools:
                        self.tools.append({
                            'name': tool.name,
                            'description': tool.description or '',
                            'input_schema': tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        })

                    # Serve until told to stop
                    while not self._stop_event.is_set():
                        await asyncio.sleep(0.1)

    async def _call_tool_async(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on this server."""
        if not self.session or not self.connected:
            return f"[ERROR] Not connected to {self.server_id}"

        try:
            result = await self.session.call_tool(tool_name, arguments)
            parts = []
            for item in result.content:
                if hasattr(item, 'text'):
                    parts.append(item.text)
                elif hasattr(item, 'type'):
                    if item.type == 'text':
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return '\n'.join(parts) if parts else "(empty result)"
        except Exception as e:
            return f"[ERROR] MCP tool call failed: {e}"


# ══════════════════════════════════════
# MCP CLIENT MANAGER
# ══════════════════════════════════════

class MCPClientManager:
    """Manages multiple MCP server connections."""

    def __init__(self):
        self.connections: dict[str, MCPConnection] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _ensure_loop(self):
        """Ensure a dedicated event loop runs in a background thread."""
        if self._loop and self._loop.is_running():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        """Run the event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout: int = 120):
        """Run an async coroutine from sync code."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def connect_server(self, server_id: str, config: dict) -> tuple[bool, str]:
        if not MCP_AVAILABLE:
            return False, "mcp package not installed. Run: pip install mcp"

        if server_id in self.connections and self.connections[server_id].connected:
            return True, f"Already connected to {server_id}"

        self._ensure_loop()
        conn = MCPConnection(server_id, config, self._loop)

        if server_id in self.connections:
            old = self.connections[server_id]
            if old.connected:
                old._stop_event.set()

        self.connections[server_id] = conn

        # Start the session lifecycle as a background task on the event loop
        try:
            conn._stop_event.clear()
            conn._task = asyncio.run_coroutine_threadsafe(
                conn._session_lifecycle(), self._loop
            )
            # Wait for connection or failure (up to 30s)
            for _ in range(300):
                if conn.connected:
                    return True, f"Connected to {config.get('name', server_id)} — {len(conn.tools)} tools discovered"
                if conn.error:
                    return False, f"Failed to connect: {conn.error}"
                import time
                time.sleep(0.1)
            return False, f"Connection timeout for {server_id}"
        except Exception as e:
            return False, f"Connection error: {e}"

    def disconnect_server(self, server_id: str) -> tuple[bool, str]:
        conn = self.connections.get(server_id)
        if not conn:
            return False, f"Server '{server_id}' not found"
        if not conn.connected:
            return True, f"Server '{server_id}' already disconnected"
        try:
            conn._stop_event.set()
            conn.connected = False
            return True, f"Disconnected from {server_id}"
        except Exception as e:
            return False, f"Disconnect error: {e}"

    def call_tool(self, server_id: str, tool_name: str, arguments: dict) -> str:
        conn = self.connections.get(server_id)
        if not conn or not conn.connected:
            return f"[ERROR] Not connected to {server_id}"
        try:
            return self._run_async(conn._call_tool_async(tool_name, arguments), timeout=120)
        except Exception as e:
            return f"[ERROR] MCP tool call failed: {e}"

    def get_all_tools(self) -> list[dict]:
        all_tools = []
        for server_id, conn in self.connections.items():
            if conn.connected:
                for tool in conn.tools:
                    all_tools.append({
                        'server_id': server_id,
                        'name': tool['name'],
                        'description': tool['description'],
                        'input_schema': tool['input_schema'],
                    })
        return all_tools

    def get_status(self) -> dict:
        status = {}
        for server_id, conn in self.connections.items():
            status[server_id] = {
                'connected': conn.connected,
                'tools': len(conn.tools),
                'error': conn.error,
                'config': {
                    'name': conn.config.get('name', server_id),
                    'transport': conn.config.get('transport', 'stdio'),
                },
            }
        return status

    def disconnect_all(self):
        for server_id in list(self.connections.keys()):
            try:
                self.disconnect_server(server_id)
            except Exception:
                pass


# ══════════════════════════════════════
# POPULAR SERVER HELPERS
# ══════════════════════════════════════

def get_popular_servers() -> dict:
    return dict(POPULAR_MCP_SERVERS)


def get_server_config(server_id: str, env_value: str = None) -> Optional[dict]:
    if server_id not in POPULAR_MCP_SERVERS:
        return None
    config = dict(POPULAR_MCP_SERVERS[server_id])
    if env_value and config.get('env_key'):
        config['env_value'] = env_value
    return config


def list_popular_servers() -> str:
    lines = ["Popular MCP Servers", "=" * 50]
    for sid, info in POPULAR_MCP_SERVERS.items():
        env_req = f" (requires: {info['env_key']})" if info.get('env_key') else " (no key needed)"
        lines.append(f"\n  {sid}")
        lines.append(f"    {info['description']}{env_req}")
        if info.get('tools_hint'):
            lines.append(f"    Tools: {', '.join(info.get('tools_hint', []))}")
    return '\n'.join(lines)


# ══════════════════════════════════════
# GLOBAL INSTANCE
# ══════════════════════════════════════

mcp_manager = MCPClientManager()
