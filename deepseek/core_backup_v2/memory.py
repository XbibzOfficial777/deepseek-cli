"""
DeepSeek CLI - Conversation Memory

Manages chat history with automatic pruning, message formatting,
and system prompt injection. Messages are stored as dicts and
converted to OpenAI format on demand.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


DEFAULT_SYSTEM_PROMPT = """\
You are DeepSeek CLI, an advanced AI coding assistant with access to powerful tools.

You can help users with:
- Writing, debugging, and analyzing code in any language
- File system operations (read, write, list, search, copy, move, delete)
- Executing shell commands safely
- Web searches and HTTP requests
- Git operations
- Text processing and data manipulation
- Network diagnostics
- System monitoring and process management
- Python package management

When using tools, follow this workflow:
1. Think about what the user needs
2. Choose the appropriate tool(s)
3. Execute the tool(s) and analyze results
4. Provide a clear, actionable response

Be precise, helpful, and security-conscious. If a command might be destructive,
warn the user before executing it. Always explain what you're doing and why.
"""


class ConversationMemory:
    """Thread-safe conversation history with pruning."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_messages: int = 80,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.messages: List[Dict[str, Any]] = []
        self._add_system()

    def _add_system(self) -> None:
        self.messages.insert(0, {
            "role": "system",
            "content": self.system_prompt,
        })

    # ── Add messages ────────────────────────────────────────

    def add_user(self, content: str) -> None:
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": time.time(),
        })
        self._prune()

    def add_assistant(self, content: str = "", tool_calls: Optional[List[Dict]] = None) -> None:
        msg: Dict[str, Any] = {"role": "assistant", "content": content, "timestamp": time.time()}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self._prune()

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
            "timestamp": time.time(),
        })
        self._prune()

    # ── Get messages ────────────────────────────────────────

    def get_openai_messages(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI API format."""
        result = []
        for msg in self.messages:
            m: Dict[str, Any] = {"role": msg["role"]}
            if "content" in msg and msg["content"] is not None:
                m["content"] = msg["content"]
            if "tool_calls" in msg:
                m["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                m["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                m["name"] = msg["name"]
            result.append(m)
        return result

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        entries = []
        for msg in self.messages[1:]:
            if msg["role"] in ("user", "assistant"):
                ts = msg.get("timestamp", 0)
                entries.append({
                    "role": msg["role"],
                    "content": (msg.get("content") or "")[:120],
                    "time": time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "",
                })
        return entries[-limit:]

    # ── Management ──────────────────────────────────────────

    def _prune(self) -> None:
        if len(self.messages) <= self.max_messages:
            return
        system = None
        if self.messages and self.messages[0]["role"] == "system":
            system = self.messages.pop(0)
        self.messages = self.messages[-(self.max_messages - 1):]
        if system:
            self.messages.insert(0, system)

    def clear(self) -> None:
        self.messages.clear()
        self._add_system()

    def compact(self, keep_pairs: int = 5) -> int:
        """Keep system + last N user/assistant pairs. Returns removed count."""
        non_system = [m for m in self.messages if m["role"] != "system"]
        system = [m for m in self.messages if m["role"] == "system"]
        kept = non_system[-(keep_pairs * 2):]
        removed = len(non_system) - len(kept)
        self.messages = system + kept
        return removed

    @property
    def count(self) -> int:
        return len(self.messages)
