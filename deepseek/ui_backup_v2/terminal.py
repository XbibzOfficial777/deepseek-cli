"""
DeepSeek CLI - Professional Terminal UI

Premium Rich-based terminal interface with:
  - Animated banner with live status bar
  - Streaming response display (content + thinking + tool execution)
  - Braille spinner animation
  - Markdown rendering with code syntax highlighting
  - Clean table layout
  - File tree visualization

Color Palette (Dark):
  Accent:      #00D4FF  (electric cyan)
  Secondary:   #7B61FF  (soft purple)
  User:        #00E676  (mint green)
  Assistant:   #00D4FF  (cyan)
  Thinking:    #B388FF  (lavender)
  Error:       #FF5252  (coral red)
  Success:     #00E676  (green)
  Warning:     #FFD740  (amber)
  Text:        #E0E0E0  (light gray)
  Dim:         #616161  (gray)
  Border:      #37474F  (dark blue-gray)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Union

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


# ═══════════════════════════════════════════════════════════════
#  Design Tokens
# ═══════════════════════════════════════════════════════════════

class C:
    """Centralized color palette."""
    ACCENT = "#00D4FF"
    ACCENT_DIM = "#0097A7"
    SECONDARY = "#7B61FF"
    SECONDARY_DIM = "#5C4B99"
    USER = "#00E676"
    USER_DIM = "#00C853"
    ASSISTANT = "#00D4FF"
    ASSISTANT_DIM = "#0097A7"
    THINKING = "#B388FF"
    THINKING_DIM = "#9575CD"
    ERROR = "#FF5252"
    ERROR_DIM = "#E53935"
    SUCCESS = "#00E676"
    SUCCESS_DIM = "#00C853"
    WARNING = "#FFD740"
    WARNING_DIM = "#FFC107"
    TEXT = "#E0E0E0"
    TEXT_DIM = "#616161"
    MUTED = "#37474F"
    BRIGHT = "#FFFFFF"
    BOX = box.ROUNDED


# ═══════════════════════════════════════════════════════════════
#  Braille Spinner
# ═══════════════════════════════════════════════════════════════

_BRAILLE = [
    "\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834",
    "\u2826", "\u2827", "\u2807", "\u280f",
]


def _spin(idx: int) -> str:
    return _BRAILLE[idx % len(_BRAILLE)]


# ═══════════════════════════════════════════════════════════════
#  Terminal UI
# ═══════════════════════════════════════════════════════════════

class TerminalUI:
    """Professional Rich terminal interface."""

    def __init__(
        self,
        model: str = "",
        tool_count: int = 0,
        show_thinking: bool = False,
    ) -> None:
        self.console = Console(
            highlight=False,
            emoji=False,
            legacy_windows=False,
        )
        self.console._color_system = True

        self.model = model
        self.tool_count = tool_count
        self.show_thinking = show_thinking

    # ── Banner ──────────────────────────────────────────────

    def print_banner(self) -> None:
        """Render a clean, modern banner with live status bar."""
        logo = Text()
        logo.append("\n", style="")
        logo.append("  deepseek-cli", style=f"bold {C.ACCENT}")
        logo.append("  v2.1", style=C.TEXT_DIM)
        logo.append("\n", style="")
        logo.append("  AI Coding Agent  |  OpenRouter", style=C.TEXT_DIM)

        panel = Panel(
            logo,
            box=C.BOX,
            border_style=C.MUTED,
            padding=(1, 3),
        )
        self.console.print()
        self.console.print(panel)

        # Status bar
        thinking_state = "on" if self.show_thinking else "off"
        thinking_color = C.SUCCESS if self.show_thinking else C.TEXT_DIM

        bar = Text()
        bar.append(f"  {self.model}", style=C.TEXT_DIM)
        bar.append("  \u2502  ", style=C.MUTED)
        bar.append(f"{self.tool_count} tools", style=C.TEXT_DIM)
        bar.append("  \u2502  ", style=C.MUTED)
        bar.append("thinking ", style=C.TEXT_DIM)
        bar.append(thinking_state, style=thinking_color)

        self.console.print(bar)
        self.console.print(
            Text("  Type /help for commands  \u00b7  Ctrl+D to quit", style=C.MUTED)
        )
        self.console.print()

    # ── Messages ────────────────────────────────────────────

    def print_user(self, message: str) -> None:
        """User message with green accent."""
        body = Text(message, style=C.TEXT)
        header = Text()
        header.append(" you ", style=f"bold {C.USER}")

        panel = Panel(
            Group(header, Text(""), body),
            box=C.BOX,
            border_style=C.USER_DIM,
            padding=(0, 2),
        )
        self.console.print()
        self.console.print(panel)

    def print_assistant(self, message: str) -> None:
        """Assistant message - try markdown rendering."""
        panel = Panel(
            Markdown(message, code_theme="monokai"),
            box=C.BOX,
            border_style=C.MUTED,
            padding=(1, 2),
            title=Text(" response ", style=f"bold {C.ACCENT}"),
            title_align="left",
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_system(self, message: str) -> None:
        """Dim system message."""
        self.console.print()
        self.console.print(Text(f"  {message}", style=C.TEXT_DIM))
        self.console.print()

    def print_error(self, message: str) -> None:
        """Red error panel."""
        panel = Panel(
            Text(message, style=C.ERROR),
            box=C.BOX,
            border_style=C.ERROR_DIM,
            padding=(0, 2),
            title=Text(" error ", style=f"bold {C.ERROR}"),
            title_align="left",
        )
        self.console.print(panel)

    def print_success(self, message: str) -> None:
        """Green success panel."""
        panel = Panel(
            Text(message, style=C.SUCCESS),
            box=C.BOX,
            border_style=C.SUCCESS_DIM,
            padding=(0, 2),
            title=Text(" done ", style=f"bold {C.SUCCESS}"),
            title_align="left",
        )
        self.console.print(panel)

    def print_warning(self, message: str) -> None:
        """Amber warning panel."""
        panel = Panel(
            Text(message, style=C.WARNING),
            box=C.BOX,
            border_style=C.WARNING_DIM,
            padding=(0, 2),
            title=Text(" warning ", style=f"bold {C.WARNING}"),
            title_align="left",
        )
        self.console.print(panel)

    # ── Streaming Display ───────────────────────────────────

    def stream_response(self, chunks: Generator[Dict[str, Any], None, None]) -> None:
        """Stream response with live-updating panels."""
        content = Text()
        thinking = Text()
        frame_idx = 0
        has_thinking = False
        has_content = False

        tool_lines: List[Text] = []
        current_tool_name = ""
        current_tool_args: Dict = {}

        with Live(
            console=self.console,
            refresh_per_second=20,
            transient=True,
        ) as live:
            for chunk in chunks:
                frame = _spin(frame_idx)
                frame_idx += 1

                ctype = chunk.get("type", "")

                if ctype == "thinking":
                    has_thinking = True
                    thinking.append(chunk.get("text", ""), style=C.THINKING_DIM)

                elif ctype == "content":
                    has_content = True
                    content.append(chunk.get("text", ""))

                elif ctype == "tool_start":
                    current_tool_name = chunk.get("name", "?")
                    current_tool_args = chunk.get("args", {})
                    tool_line = Text()
                    tool_line.append(f"  {frame} ", style=C.ACCENT)
                    tool_line.append(f"Executing ", style=C.TEXT_DIM)
                    tool_line.append(current_tool_name, style=f"bold {C.SECONDARY}")
                    args_str = json.dumps(current_tool_args)
                    if len(args_str) > 80:
                        args_str = args_str[:77] + "..."
                    tool_line.append(f"({args_str})", style=C.TEXT_DIM)
                    tool_lines.append(tool_line)
                    live.update(Group(*tool_lines))

                elif ctype == "tool_result":
                    result_text = chunk.get("result", "")
                    if len(result_text) > 200:
                        result_text = result_text[:197] + "..."
                    tl = Text()
                    tl.append(f"  \u2713 ", style=C.SUCCESS)
                    tl.append(chunk.get("name", "?"), style=C.SUCCESS)
                    tl.append(f"  \u2192 {result_text}", style=C.TEXT_DIM)
                    tool_lines.append(tl)
                    live.update(Group(*tool_lines))

                elif ctype == "tool_error":
                    tl = Text()
                    tl.append(f"  \u2717 ", style=C.ERROR)
                    tl.append(chunk.get("name", "?"), style=C.ERROR)
                    tl.append(f"  {chunk.get('error', 'unknown error')}", style=C.ERROR_DIM)
                    tool_lines.append(tl)
                    live.update(Group(*tool_lines))

                elif ctype == "error":
                    content.append(chunk.get("text", ""), style=C.ERROR)
                    has_content = True

                # ── Compose live display ──
                children: List[Any] = []

                if tool_lines:
                    tool_panel = Panel(
                        Group(*tool_lines),
                        box=C.BOX,
                        border_style=C.SECONDARY_DIM,
                        padding=(0, 2),
                        title=Text(f" {frame} tools ", style=C.SECONDARY),
                        title_align="left",
                    )
                    children.append(tool_panel)

                if has_thinking and not has_content:
                    if self.show_thinking:
                        children.append(Text(f"\n  thinking...", style=C.THINKING_DIM))
                    else:
                        children.append(Text(f"\n  {frame} reasoning...", style=C.TEXT_DIM))

                if content.plain:
                    children.append(
                        Panel(
                            content,
                            box=C.BOX,
                            border_style=C.MUTED,
                            padding=(0, 2),
                            title=Text(f" {frame} generating ", style=C.ACCENT_DIM),
                            title_align="left",
                        )
                    )

                if children:
                    live.update(Group(*children))

        # ── Final static render ──

        if tool_lines:
            self.console.print(
                Panel(
                    Group(*tool_lines),
                    box=C.BOX,
                    border_style=C.SECONDARY_DIM,
                    padding=(0, 2),
                    title=Text(" tools ", style=C.SECONDARY),
                    title_align="left",
                )
            )

        if self.show_thinking and has_thinking and thinking.plain:
            self.console.print()
            self.console.print(
                Panel(
                    thinking,
                    box=C.BOX,
                    border_style=C.MUTED,
                    padding=(1, 2),
                    title=Text(" \u25b8 thinking ", style=C.THINKING_DIM),
                    title_align="left",
                )
            )

        if content.plain:
            self.console.print()
            md_text = content.plain
            self.console.print(
                Panel(
                    Markdown(md_text, code_theme="monokai"),
                    box=C.BOX,
                    border_style=C.MUTED,
                    padding=(1, 2),
                    title=Text(" response ", style=f"bold {C.ACCENT}"),
                    title_align="left",
                )
            )
            self.console.print()

    # ── Tables ──────────────────────────────────────────────

    def print_table(
        self,
        data: List[Dict[str, Any]],
        title: str = "",
        columns: Optional[List[str]] = None,
    ) -> None:
        """Clean table with zebra striping."""
        if not data:
            self.print_system("No data to display.")
            return

        table = Table(
            title=title or None,
            box=box.SIMPLE,
            show_header=True,
            header_style=f"bold {C.ACCENT}",
            border_style=C.MUTED,
            row_styles=[f"on {C.MUTED}", ""],
            padding=(0, 2),
            expand=False,
            title_style=f"bold {C.TEXT}" if title else None,
        )

        keys = columns or list(data[0].keys())
        for col in keys:
            table.add_column(
                col.replace("_", " ").title(),
                style=C.TEXT,
                no_wrap=False,
            )

        for item in data:
            row = []
            for key in keys:
                val = str(item.get(key, ""))
                if len(val) > 60:
                    val = val[:57] + "\u2026"
                row.append(val)
            table.add_row(*row)

        self.console.print()
        self.console.print(table)
        self.console.print()

    # ── Code ────────────────────────────────────────────────

    def print_code(self, code: str, language: str = "python", title: str = "") -> None:
        """Syntax-highlighted code block."""
        syntax = Syntax(
            code, language, theme="monokai",
            line_numbers=True, word_wrap=True,
            background_color="#1E1E1E",
        )
        title_text = f" {language} "
        if title:
            title_text = f" {title}  \u00b7  {language} "

        panel = Panel(
            syntax, box=C.BOX, border_style=C.MUTED,
            padding=(0, 1),
            title=Text(title_text, style=C.TEXT_DIM),
            title_align="left",
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    # ── Prompt ──────────────────────────────────────────────

    def get_prompt(self):
        """Return prompt for prompt_toolkit."""
        from prompt_toolkit import HTML
        return HTML(
            f'<ansicyan>\u276f</ansicyan> '
            f'<ansigreen>you</ansigreen>'
            f'<dim> \u00b7 </dim>'
        )

    def get_status_bar(self) -> Text:
        """Return a status bar text."""
        bar = Text()
        bar.append(f"  {self.model}", style=C.TEXT_DIM)
        bar.append("  \u2502  ", style=C.MUTED)
        bar.append(f"{self.tool_count} tools", style=C.TEXT_DIM)
        return bar

    # ── Misc ────────────────────────────────────────────────

    def clear(self) -> None:
        self.console.clear()

    def print_empty(self) -> None:
        self.console.print()

    def timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")
