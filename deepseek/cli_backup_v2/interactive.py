"""
DeepSeek CLI - Interactive REPL

Professional interactive session with prompt_toolkit, Rich streaming,
slash commands, auto-complete, history, and keyboard shortcuts.

Slash Commands:
  /help              - Show all commands
  /exit, /quit, /q  - Exit session
  /clear             - Clear conversation
  /reset             - Reset agent completely
  /status            - Show agent status
  /tools             - List available tools
  /history           - Show conversation history
  /thinking          - Toggle thinking display
  /model <name>      - Change model
  /save <file>       - Save conversation to JSON
  /compact [N]       - Compact to last N pairs
  /config            - Show current config
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from rich import box as rich_box
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax


class InteractiveSession:
    """Full-featured REPL with streaming, commands, and prompt_toolkit."""

    VERSION = "2.1.0"

    def __init__(self, agent: Any, ui: Any) -> None:
        self.agent = agent
        self.ui = ui
        self._running = True
        self._session: Optional[PromptSession] = None
        self._history_path = Path.home() / ".deepseek" / "history.txt"

        # Ensure history dir exists
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._history_path = Path("/tmp/.deepseek_history.txt")

        # Slash commands for autocomplete
        self._commands = [
            "/help", "/exit", "/quit", "/q",
            "/clear", "/reset", "/status", "/tools",
            "/history", "/thinking", "/model",
            "/save", "/compact", "/config",
        ]

    # ── Main Loop ───────────────────────────────────────────

    async def run(self) -> None:
        """Main REPL loop."""
        self.ui.print_banner()

        while self._running:
            try:
                prompt = self.ui.get_prompt()
                user_input = await self.session.prompt_async(prompt)

                # Ctrl+D in prompt_toolkit returns None
                if user_input is None:
                    await self._cmd_exit()
                    break

                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                else:
                    await self._handle_query(user_input)

            except KeyboardInterrupt:
                self.ui.print_system("Use /exit to quit, /clear to reset")
                continue

            except EOFError:
                await self._cmd_exit()

            except Exception as exc:
                self.ui.print_error(f"Unexpected error: {exc}")

    # ── Prompt Session ──────────────────────────────────────

    def _create_session(self) -> PromptSession:
        """Build prompt_toolkit session with all features."""
        style = Style.from_dict({
            "prompt": "ansicyan bold",
            "": "ansigreen",
            "completion-menu": "bg:#003333 #cccccc",
            "completion-menu.completion": "bg:#003333 #e0e0e0",
            "completion-menu.completion.current": "bg:#00aaaa #000000",
            "scrollbar.background": "bg:#222222",
            "scrollbar.button": "bg:#00aaaa",
        })

        kb = KeyBindings()

        @kb.add("c-c")
        def _ctrl_c(event: Any) -> None:
            event.current_buffer.reset()

        @kb.add("c-d")
        def _ctrl_d(event: Any) -> None:
            event.app.exit(result=None)

        @kb.add("escape", "enter")
        def _alt_enter(event: Any) -> None:
            event.current_buffer.validate_and_handle()

        # Auto-complete words
        tool_names = self.agent.tools.get_names()
        all_words = self._commands + tool_names

        return PromptSession(
            history=FileHistory(str(self._history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(
                words=all_words,
                ignore_case=True,
                sentence=True,
            ),
            key_bindings=kb,
            style=style,
            multiline=True,
            mouse_support=True,
            complete_while_typing=False,
            prompt_continuation=HTML("<dim>  ... </dim>"),
        )

    @property
    def session(self) -> PromptSession:
        if self._session is None:
            self._session = self._create_session()
        return self._session

    # ── Query Handling ──────────────────────────────────────

    async def _handle_query(self, query: str) -> None:
        """Stream agent response in real-time."""
        self.ui.print_user(query)

        # Collect chunks and yield to UI synchronously
        chunks: List[Dict[str, Any]] = []
        try:
            async for chunk in self.agent.process_query(query):
                chunks.append(chunk)
        except asyncio.CancelledError:
            self.ui.print_warning("Query cancelled")
            return

        # Display all chunks via UI
        if chunks:
            self.ui.stream_response(iter(chunks))

    # ── Command Dispatcher ──────────────────────────────────

    async def _handle_command(self, command: str) -> None:
        """Route slash command."""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args_str = parts[1].strip() if len(parts) > 1 else ""
        args = args_str.split() if args_str else []

        dispatch = {
            "/help": self._cmd_help,
            "/exit": self._cmd_exit,
            "/quit": self._cmd_exit,
            "/q": self._cmd_exit,
            "/clear": self._cmd_clear,
            "/reset": self._cmd_reset,
            "/status": self._cmd_status,
            "/tools": self._cmd_tools,
            "/history": self._cmd_history,
            "/thinking": self._cmd_thinking,
            "/model": self._cmd_model,
            "/save": self._cmd_save,
            "/compact": self._cmd_compact,
            "/config": self._cmd_config,
        }

        handler = dispatch.get(cmd)
        if handler:
            if cmd in ("/model", "/save", "/compact"):
                await handler(args, args_str)
            else:
                await handler()
        else:
            self.ui.print_error(f"Unknown command: {cmd}")

    # ── Command Implementations ─────────────────────────────

    async def _cmd_help(self) -> None:
        rows = [
            {"Command": "/help", "Description": "Show this help message"},
            {"Command": "/exit, /quit, /q", "Description": "Exit the session"},
            {"Command": "/clear", "Description": "Clear conversation"},
            {"Command": "/reset", "Description": "Reset agent completely"},
            {"Command": "/status", "Description": "Show agent status & stats"},
            {"Command": "/tools", "Description": "List available tools"},
            {"Command": "/history", "Description": "Show conversation history"},
            {"Command": "/thinking", "Description": "Toggle thinking display"},
            {"Command": "/model <name>", "Description": "Change AI model"},
            {"Command": "/save <file>", "Description": "Save conversation to JSON"},
            {"Command": "/compact [N]", "Description": "Compact to last N pairs (default: 5)"},
            {"Command": "/config", "Description": "Show current configuration"},
        ]
        self.ui.print_table(rows, title="Commands")

    async def _cmd_exit(self) -> None:
        self.ui.print_system("Goodbye!")
        self._running = False
        try:
            await self.agent.close()
        except Exception:
            pass

    async def _cmd_clear(self) -> None:
        self.agent.clear()
        self.ui.clear()
        self.ui.print_banner()
        self.ui.print_success("Conversation cleared")

    async def _cmd_reset(self) -> None:
        self.ui.print_system("Resetting agent...")
        try:
            await self.agent.close()
            from .main import create_agent
            self.agent = create_agent()
            self.ui.model = self.agent.llm.model
            self.ui.tool_count = self.agent.tools.count
            self._session = None  # rebuild prompt session
            self.ui.print_success("Agent reset complete")
        except Exception as exc:
            self.ui.print_error(f"Reset failed: {exc}")

    async def _cmd_status(self) -> None:
        status = self.agent.get_status()
        rows = [
            {"Property": "Model", "Value": status.get("model", "-")},
            {"Property": "Messages", "Value": str(status.get("messages", 0))},
            {"Property": "Tools loaded", "Value": str(status.get("tools", 0))},
            {"Property": "Total tokens", "Value": str(status.get("total_tokens", 0))},
            {"Property": "API requests", "Value": str(status.get("requests", 0))},
            {"Property": "Thinking", "Value": "ON" if self.ui.show_thinking else "OFF"},
        ]
        self.ui.print_table(rows, title="Agent Status")

    async def _cmd_tools(self) -> None:
        tools = self.agent.tools.get_all()
        if not tools:
            self.ui.print_system("No tools loaded")
            return

        rows = []
        for name, tool in tools.items():
            desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
            rows.append({
                "Name": name,
                "Category": tool.category,
                "Risk": tool.risk_level,
                "Description": desc,
            })
        self.ui.print_table(rows, title=f"Tools ({len(rows)} loaded)")

    async def _cmd_history(self) -> None:
        history = self.agent.memory.get_history(limit=30)
        if not history:
            self.ui.print_system("No conversation history yet")
            return

        rows = []
        for i, item in enumerate(history, 1):
            role = "you" if item["role"] == "user" else "ai"
            content = item["content"][:80] + "..." if len(item.get("content", "")) > 80 else item["content"]
            rows.append({
                "#": str(i),
                "Role": role,
                "Time": item.get("time", ""),
                "Content": content,
            })
        self.ui.print_table(rows, title="History")

    async def _cmd_thinking(self) -> None:
        self.ui.show_thinking = not self.ui.show_thinking
        state = "ON" if self.ui.show_thinking else "OFF"
        self.ui.print_success(f"Thinking display: {state}")

    async def _cmd_model(self, args: List[str], args_str: str = "") -> None:
        if not args:
            self.ui.print_system(f"Current model: {self.agent.llm.model}")
            return
        self.agent.llm.model = args_str
        self.ui.model = args_str
        self.ui.print_success(f"Model changed to: {args_str}")

    async def _cmd_save(self, args: List[str], args_str: str = "") -> None:
        if not args:
            self.ui.print_error("Usage: /save <filename>")
            return

        filename = args_str.strip()
        if not filename.endswith(".json"):
            filename += ".json"

        try:
            save_path = Path(filename).expanduser().resolve()
            messages = self.agent.memory.messages

            export = {
                "saved_at": datetime.now().isoformat(),
                "version": self.VERSION,
                "model": self.agent.llm.model,
                "message_count": len(messages),
                "messages": [
                    {"role": m["role"], "content": m.get("content", "")}
                    for m in messages
                ],
            }

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(export, f, indent=2, ensure_ascii=False)

            self.ui.print_success(f"Saved {len(messages)} messages to {save_path}")
        except Exception as exc:
            self.ui.print_error(f"Save failed: {exc}")

    async def _cmd_compact(self, args: List[str], args_str: str = "") -> None:
        n = 5
        if args:
            try:
                n = int(args[0])
            except ValueError:
                self.ui.print_error("Usage: /compact [N]")
                return

        removed = self.agent.compact(keep_pairs=n)
        self.ui.print_success(
            f"Compacted: removed {removed} messages, kept last {n} pairs"
        )

    async def _cmd_config(self) -> None:
        """Show current config."""
        status = self.agent.get_status()
        config_text = json.dumps({
            "model": status.get("model"),
            "tools": status.get("tool_names"),
            "thinking": self.ui.show_thinking,
            "api": {
                "base_url": self.agent.llm.base_url,
                "temperature": self.agent.llm.temperature,
                "max_tokens": self.agent.llm.max_tokens,
            },
        }, indent=2)

        syntax = Syntax(config_text, "json", theme="monokai", background_color="#1E1E1E")
        panel = Panel(
            syntax,
            box=rich_box.ROUNDED,
            border_style="#37474F",
            padding=(1, 2),
            title=Text(" configuration ", style="#00D4FF"),
        )
        self.ui.console.print()
        self.ui.console.print(panel)
        self.ui.console.print()
