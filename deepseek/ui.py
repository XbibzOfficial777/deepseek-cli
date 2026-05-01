# DeepSeek CLI v7.7 — UI Components
# Rich-based terminal UI: StreamRenderer, LoadingSpinner, Banner, StatusBar
# Raw-mode interactive: arrow-key select menu, Ctrl+P input
#
# FIXED v5.1.1:
#   - interactive_select: use os.read() instead of sys.stdin.read() for Termux compat
#   - Handle both ESC[A and ESC O A arrow key sequences
#   - Longer escape sequence timeout (0.2s instead of 0.05s)
#   - Flush stdout/stderr before raw mode, restore fully on exit
#   - prompt_input: same fixes for arrow keys + Ctrl+P
#   - Use sys.stderr for debug instead of stdout to avoid interference
#
# FIXED v5.3:
#   - StreamRenderer: REAL-TIME streaming — content via sys.stdout.write()
#   - No more buffering entire response before display
#   - Animated "Thinking..." indicator while model reasons
#   - Thinking panel auto-flushed when first content chunk arrives
#   - Zero-latency display: bytes go directly to file descriptor
#
# FIXED v7.2:
#   - StreamRenderer: Rich Markdown rendering for final responses
#   - Cursor save/restore: raw streamed text replaced with styled Markdown
#   - **bold**, *italic*, `code`, code blocks, headers, lists, blockquotes
#   - Syntax-highlighted code blocks (monokai theme)
#   - Professional AI-style output with colors and formatting
#
# FIXED v7.7:
#   - StreamRenderer: Smooth buffered output (20ms minimum flush interval)
#   - TUIStatusBar: Compact status bar for tool output
#   - Improved tool call/result display with professional formatting
#   - New ASCII art banner v7.7 with feature list
#   - Better visual separators and color coding throughout

import sys
import os
import re
import threading
import time
import tty
import termios
import select as _select
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.syntax import Syntax
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn
from rich.status import Status

console = Console()

# Sentinel value returned by prompt_input() when Ctrl+P is pressed
CTRL_P_SENTINEL = '__CTRL_P_SETTINGS__'

# ══════════════════════════════════════
# LOADING SPINNER
# ══════════════════════════════════════

SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']


class LoadingSpinner:
    """Animated spinner for terminal, usable as context manager."""

    def __init__(self, message: str = 'Loading'):
        self.message = message
        self._running = False
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Clear the spinner line: overwrite with spaces, then newline
        try:
            sys.stdout.write('\r  \033[K')
            sys.stdout.flush()
        except Exception:
            pass

    def _spin(self):
        idx = 0
        while self._running:
            frame = SPINNER_FRAMES[idx % len(SPINNER_FRAMES)]
            line = f'\r  {frame} {self.message}...'
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            idx += 1
            time.sleep(0.08)


def with_spinner(message: str = 'Loading'):
    """Return a LoadingSpinner context manager."""
    return LoadingSpinner(message)


# ══════════════════════════════════════
# LOW-LEVEL RAW TERMINAL HELPERS
# ══════════════════════════════════════

def _raw_read_byte(fd, timeout=0.15):
    """
    Read a single byte from fd in raw mode.
    Uses os.read() which is reliable on Termux/Linux/macOS.
    Returns bytes or empty bytes on timeout.
    """
    ready, _, _ = _select.select([fd], [], [], timeout)
    if ready:
        return os.read(fd, 1)
    return b''


def _read_escape_sequence(fd, timeout=0.15):
    """
    Read an escape sequence starting after ESC (0x1b).
    Returns the decoded key name or None if timeout/incomplete.

    Handles:
      ESC [ A/B/C/D  — standard CSI arrows
      ESC O A/B/C/D  — SS3 arrows (xterm, some Termux)
      ESC [ 1;5 A   — Ctrl+arrow (ignore, return None)
      ESC [ 3 ~      — Delete key
      ESC [ H / ESC O H  — Home
      ESC [ F / ESC O F  — End
    """
    ch2 = _raw_read_byte(fd, timeout)
    if not ch2:
        return None  # Plain ESC (timed out waiting for next char)

    if ch2 == b'[':
        # CSI sequence: ESC [ ...
        ch3 = _raw_read_byte(fd, timeout)
        if not ch3:
            return None

        # Simple single-char codes
        if ch3 in (b'A', b'B', b'C', b'D', b'H', b'F'):
            return ch3.decode('ascii')

        # Multi-byte: ESC [ 3 ~ (Delete)
        if ch3 == b'3':
            ch4 = _raw_read_byte(fd, timeout)
            if ch4 == b'~':
                return 'DELETE'
            return None

        # Multi-byte: ESC [ 1 ; 5 A (Ctrl+arrow) — consume and discard
        if ch3 == b'1':
            ch4 = _raw_read_byte(fd, timeout)
            if ch4 == b';':
                ch5 = _raw_read_byte(fd, timeout)
                if ch5:
                    _raw_read_byte(fd, timeout)  # consume the final letter
                return None
            return None

        # Unknown CSI — consume remaining
        return None

    elif ch2 == b'O':
        # SS3 sequence: ESC O A/B/C/D/H/F
        ch3 = _raw_read_byte(fd, timeout)
        if not ch3:
            return None
        if ch3 in (b'A', b'B', b'C', b'D', b'H', b'F'):
            return ch3.decode('ascii')
        return None

    # Unknown ESC sequence — return None
    return None


def _flush_stdin(fd):
    """Drain any pending bytes from stdin (called before entering raw mode)."""
    try:
        while True:
            ready, _, _ = _select.select([fd], [], [], 0.0)
            if not ready:
                break
            os.read(fd, 4096)
    except Exception:
        pass


# ══════════════════════════════════════
# STREAM RENDERER
# ══════════════════════════════════════

class StreamRenderer:
    """
    Renders streamed LLM output with REAL-TIME display + Rich Markdown final render.

    v7.2: During streaming, raw text goes to terminal for instant feedback.
    On final response (no more tool rounds), raw text is replaced with
    Rich Markdown rendering — **bold**, *italic*, `code`, code blocks,
    headers, lists, blockquotes, tables, links — all styled beautifully.

    Flow:
      1. Thinking chunks -> animated "Thinking..." indicator
      2. First content chunk -> save cursor, flush thinking panel, start streaming
      3. Content chunks -> written directly to stdout (real-time)
      4. render_final() -> restore cursor, clear raw, render Rich Markdown
      5. show_done() -> final newline, cleanup (for intermediate rounds)
    """

    def __init__(self, thinking_visible: bool = True):
        self.thinking_visible = thinking_visible
        self._thinking_text = ''
        self._content_text = ''
        self._thinking_panel_shown = False
        self._stream_started = False
        self._indicator_shown = False
        # Animated thinking indicator
        self._anim_running = False
        self._anim_thread = None
        # v7.2: Cursor save/restore for markdown re-render
        self._cursor_saved = False
        # v7.7: TUI improvements
        self._write_buffer = ''       # Buffer for smooth output
        self._buffer_lock = threading.Lock()
        self._last_flush_time = 0
        self._flush_interval = 0.02  # 20ms minimum between flushes

    # ── Thinking Animation ──

    def _start_thinking_anim(self):
        """Start animated 'Thinking...' indicator in background thread."""
        if self._anim_running:
            return
        self._anim_running = True
        self._anim_thread = threading.Thread(target=self._thinking_anim_loop, daemon=True)
        self._anim_thread.start()

    def _stop_thinking_anim(self):
        """Stop the thinking animation and clear its line."""
        was_running = self._anim_running
        self._anim_running = False
        if self._anim_thread:
            self._anim_thread.join(timeout=0.5)
            self._anim_thread = None
        # Clear the indicator line from terminal
        if was_running:
            try:
                sys.stdout.write('\r\033[K')
                sys.stdout.flush()
            except Exception:
                pass

    def _thinking_anim_loop(self):
        """Background thread: animate 'Thinking...' with pulsing dots."""
        frames = ['   ', '.  ', '.. ', '...', ' ..', '  .', '   ']
        idx = 0
        try:
            while self._anim_running:
                frame = frames[idx % len(frames)]
                line = f'\r  \033[2mThinking{frame}\033[0m'
                sys.stdout.write(line)
                sys.stdout.flush()
                idx += 1
                time.sleep(0.18)
        except Exception:
            pass

    # ── Thinking Panel ──

    def _flush_thinking_panel(self):
        """Display the thinking panel (Rich formatted) if not already shown."""
        if self._thinking_panel_shown:
            return
        self._thinking_panel_shown = True

        text = self._thinking_text.strip()
        if text and self.thinking_visible:
            console.print()
            console.print(Panel(
                text,
                title='thinking',
                title_align='left',
                border_style='dim blue',
                padding=(0, 1),
            ))

    # ── Cursor Save/Restore (v7.2) ──

    def _save_cursor(self):
        """Save current cursor position for markdown re-render."""
        try:
            sys.stdout.write('\033[s')
            sys.stdout.flush()
            self._cursor_saved = True
        except Exception:
            pass

    def _restore_and_clear(self):
        """Restore saved cursor and clear everything below it."""
        if self._cursor_saved:
            try:
                sys.stdout.write('\033[u')  # Restore cursor
                sys.stdout.write('\033[J')  # Clear from cursor to end of screen
                sys.stdout.flush()
            except Exception:
                pass
            self._cursor_saved = False

    # ── Stream End ──

    def _end_stream(self):
        """Write final newline after streamed content."""
        if self._content_text and self._stream_started:
            try:
                sys.stdout.write('\n')
                sys.stdout.flush()
            except Exception:
                pass

    # ── Round Reset (v7.2) ──

    def reset_for_new_round(self):
        """Reset streaming state for a new agent round."""
        self._stream_started = False
        self._cursor_saved = False
        self._indicator_shown = False

    # ── Public API ──

    def append_thinking(self, chunk: str):
        """
        Handle thinking/reasoning chunk.
        Shows animated 'Thinking...' indicator on first chunk.
        """
        if not self.thinking_visible:
            return
        self._thinking_text += chunk
        # Start animated indicator on first thinking chunk
        if not self._stream_started and not self._indicator_shown:
            self._indicator_shown = True
            self._start_thinking_anim()

    def append_content(self, chunk: str):
        """
        Stream content chunk DIRECTLY to terminal — REAL-TIME.
        Uses buffering for smooth display (avoids flickering from too-frequent writes).
        """
        self._content_text += chunk

        # First content chunk: transition from thinking to content
        if not self._stream_started:
            self._stream_started = True
            self._stop_thinking_anim()
            self._flush_thinking_panel()
            console.print()  # blank line before content
            self._save_cursor()

        # Buffer the chunk for smooth output
        with self._buffer_lock:
            self._write_buffer += chunk
            now = time.monotonic()
            # Flush at minimum interval or on newlines
            if '\n' in chunk or (now - self._last_flush_time >= self._flush_interval):
                self._flush_buffer()
                self._last_flush_time = now

    def _flush_buffer(self):
        """Write buffered content to stdout."""
        if not self._write_buffer:
            return
        try:
            sys.stdout.write(self._write_buffer)
            sys.stdout.flush()
        except Exception:
            pass
        self._write_buffer = ''

    def render_final(self, text: str):
        """
        Re-render streamed content as Rich Markdown.
        Call ONLY on the FINAL response (no more tool rounds).

        Replaces raw streamed text with beautifully formatted markdown:
        - **bold** text, *italic* text
        - `inline code` with syntax color
        - Code blocks with syntax highlighting (monokai theme)
        - Headers (# ## ###) with distinct colors
        - Bullet/numbered lists with styled markers
        - Blockquotes with indent and color
        - Links with underline
        - Tables with borders
        - Horizontal rules
        """
        self._stop_thinking_anim()
        self._flush_thinking_panel()
        self._flush_buffer()

        if not text or not text.strip():
            self._end_stream()
            return

        # Restore cursor to start of content area, clear raw text
        self._restore_and_clear()

        # Render as Rich Markdown with syntax highlighting
        md = Markdown(text.strip(), code_theme='monokai')
        console.print(md)
        console.print()

    def show_thinking_as_content(self, thinking_text: str):
        """
        Fallback: display thinking text as content when model returned
        reasoning-only response (no content field). Used for DeepSeek R1
        and other reasoning models that may leave content empty.
        """
        self._stop_thinking_anim()
        if thinking_text and thinking_text.strip():
            # If thinking panel was already shown, content goes after it
            # If thinking panel was NOT shown (thinking_visible=False), show it now
            if not self._thinking_panel_shown and self.thinking_visible:
                self._thinking_panel_shown = True
                console.print()
                console.print(Panel(
                    thinking_text.strip(),
                    title='reasoning (shown as response)',
                    title_align='left',
                    border_style='dim blue',
                    padding=(0, 1),
                ))
            else:
                # Thinking panel already shown, or thinking hidden — just output as content
                self._content_text = thinking_text
                self._stream_started = True
                try:
                    sys.stdout.write(thinking_text.strip())
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                except Exception:
                    pass
        self._end_stream()

    def show_error(self, message: str):
        """Display error message."""
        self._stop_thinking_anim()
        self._flush_thinking_panel()
        self._end_stream()
        console.print(f'\n  [bold red]Error:[/bold red] {message}')

    def show_done(self):
        """Finalize display — cleanup animation, flush thinking, end content.
        Used for intermediate rounds (before tool execution).
        For final response, use render_final() instead."""
        self._flush_buffer()
        self._stop_thinking_anim()
        self._flush_thinking_panel()
        # Discard saved cursor for intermediate rounds (not final)
        self._cursor_saved = False
        self._end_stream()
        sys.stdout.flush()

    def show_tool_call(self, tool_name: str, args: dict):
        """Display tool call info with professional formatting."""
        self._stop_thinking_anim()
        self._flush_thinking_panel()
        self._flush_buffer()
        args_str = ', '.join(f'{k}={v!r}' for k, v in args.items()) if args else ''
        if len(args_str) > 100:
            args_str = args_str[:97] + '...'
        console.print(f'  [bold yellow]>[/bold yellow] [cyan]{tool_name}[/cyan]({args_str})')

    def show_tool_result(self, tool_name: str, result: str):
        """Display tool result with professional formatting."""
        display = result
        if len(display) > 800:
            display = display[:797] + '...'
        lines = display.split('\n')
        if len(lines) > 15:
            display = '\n'.join(lines[:15]) + f'\n  ... ({len(lines) - 15} more lines)'
        console.print(f'  [bold green]<[/bold green] [dim]{display}[/dim] [bold green]<[/bold green]')


# ══════════════════════════════════════
# TOOL PROCESSING INDICATOR (v6.0)
# ══════════════════════════════════════

class ToolProcessingIndicator:
    """
    Professional Rich-based animated indicator shown while processing tool results.

    Uses a Rich Status with spinner + styled message.
    No emojis -- pure Rich formatting for maximum terminal compatibility.

    Usage:
        with ToolProcessingIndicator(round_num, max_rounds, tools_count):
            ... process tools ...
    """

    _SPINNER_NAMES = [
        'dots', 'dots2', 'line', 'arc', 'arrow', 'simpleDots',
        'simpleDotsScrolling', 'bouncingBar', 'star', 'squareCorners',
    ]

    def __init__(self, round_num: int = 1, max_rounds: int = 12, tools_count: int = 0):
        self.round_num = round_num
        self.max_rounds = max_rounds
        self.tools_count = tools_count
        self._status = None
        self._index = 0

    def _spinner_name(self) -> str:
        """Cycle through spinner styles per round for visual variety."""
        return self._SPINNER_NAMES[self._index % len(self._SPINNER_NAMES)]

    def _build_message(self) -> str:
        """Build a professional status message."""
        parts = []
        parts.append(f'Processing tool results')
        parts.append(f'[dim]round {self.round_num}/{self.max_rounds}[/dim]')
        if self.tools_count > 0:
            parts.append(f'[dim]| {self.tools_count} tool(s)[/dim]')
        return '  '.join(parts)

    def __enter__(self):
        self._status = Status(
            self._build_message(),
            spinner=self._spinner_name(),
            spinner_style='bold cyan',
        )
        self._status.start()
        return self

    def __exit__(self, *args):
        if self._status:
            self._status.stop()
            self._status = None

    def update_round(self, round_num: int, tools_count: int = 0):
        """Update the indicator message (call while active)."""
        self.round_num = round_num
        self.tools_count = tools_count
        self._index += 1
        if self._status:
            self._status.update(self._build_message(), spinner=self._spinner_name())


def show_tool_processing(round_num: int = 1, max_rounds: int = 12, tools_count: int = 0) -> ToolProcessingIndicator:
    """Convenience function: create and return a ToolProcessingIndicator."""
    return ToolProcessingIndicator(round_num, max_rounds, tools_count)


# ══════════════════════════════════════
# TUI STATUS BAR (v7.7)
# ══════════════════════════════════════

class TUIStatusBar:
    """Shows a compact status bar at the bottom of tool output."""

    def __init__(self):
        self._visible = False

    def show(self, message: str):
        """Show a status bar message."""
        self._visible = True
        try:
            sys.stdout.write(f'\r  [dim cyan]{message}[/dim cyan]\033[K')
            sys.stdout.flush()
        except Exception:
            pass

    def clear(self):
        """Clear the status bar."""
        if self._visible:
            try:
                sys.stdout.write('\r\033[K')
                sys.stdout.flush()
            except Exception:
                pass
            self._visible = False


# ══════════════════════════════════════
# INTERACTIVE SELECT (Arrow Key Menu)
# ══════════════════════════════════════

def interactive_select(items, title=None, active_index=0):
    """
    Interactive arrow-key selection menu for terminal.

    Uses os.read() for reliable raw I/O on Termux/Linux/macOS.
    Handles both ESC[A and ESC O A arrow key sequences.

    Args:
        items: list of strings to display
        title: optional title line
        active_index: initially selected item index

    Returns:
        int — selected index, or -1 if cancelled (Esc / Ctrl+C)
    """
    if not items:
        return -1

    # Fallback for non-TTY (piped input)
    if not sys.stdin.isatty():
        return _select_fallback(items, title, active_index)

    selected = active_index
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # Ensure stdout is a real terminal too
    out_fd = sys.stdout.fileno()

    # Count total lines drawn (for redraw)
    total_lines = len(items) + (1 if title else 0) + 1  # items + title + hint

    def draw():
        """Draw the full menu from current position."""
        os.write(out_fd, b'\033[?25l')  # Hide cursor
        if title:
            os.write(out_fd, f'\r  \033[1;36m{title}\033[0m\r\n'.encode())
        for i, item in enumerate(items):
            if i == selected:
                os.write(out_fd, f'\r  \033[1;32m>> {item}\033[0m\r\n'.encode())
            else:
                os.write(out_fd, f'\r     {item}\r\n'.encode())
        os.write(out_fd, b'\r  \033[2m  Up/Down: navigate | Enter: select | Esc: cancel\033[0m')
        os.write(out_fd, b'\r\n')
        os.write(out_fd, b'\033[?25h')  # Show cursor again (we reposition with move-up)
        sys.stdout.flush()

    def redraw():
        """Clear previous draw and redraw from scratch."""
        # Move cursor up total_lines, then clear screen from cursor down
        os.write(out_fd, f'\033[{total_lines}A'.encode())
        os.write(out_fd, b'\033[J')
        draw()

    def cleanup():
        """Restore terminal state."""
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
        try:
            os.write(out_fd, b'\033[?25h')  # Show cursor
        except Exception:
            pass
        try:
            sys.stdout.flush()
        except Exception:
            pass

    try:
        # Flush any pending input
        sys.stdout.flush()
        sys.stderr.flush()
        _flush_stdin(fd)

        # Enter raw mode
        tty.setraw(fd)
        draw()

        while True:
            raw = _raw_read_byte(fd, 0.15)
            if not raw:
                continue  # timeout, keep looping

            ch = raw.decode('latin-1')

            if ch == '\r' or ch == '\n':
                # Enter — confirm
                cleanup()
                return selected

            elif ch == '\x1b':
                # Escape sequence
                key = _read_escape_sequence(fd, 0.15)
                if key is None:
                    # Plain Esc or incomplete — cancel
                    cleanup()
                    return -1
                elif key == 'A':
                    # Up arrow
                    selected = (selected - 1) % len(items)
                    redraw()
                elif key == 'B':
                    # Down arrow
                    selected = (selected + 1) % len(items)
                    redraw()
                # Ignore other escape sequences (C/D/H/F etc.)

            elif ch == '\x03':
                # Ctrl+C
                cleanup()
                return -1

            elif ch == 'j' or ch == 'J':
                selected = (selected + 1) % len(items)
                redraw()

            elif ch == 'k' or ch == 'K':
                selected = (selected - 1) % len(items)
                redraw()

    except Exception:
        cleanup()
        raise

    finally:
        cleanup()


def _select_fallback(items, title, active_index):
    """Fallback for non-TTY: numbered list with text input."""
    if title:
        console.print(f'  [bold cyan]{title}[/bold cyan]')
        console.print()
    for i, item in enumerate(items):
        marker = ' [dim]<< active[/dim]' if i == active_index else ''
        console.print(f'    [green]{i+1}.[/green] {item}{marker}')
    console.print()
    try:
        choice = console.input('  Enter number (or 0 to cancel): ').strip()
        idx = int(choice) - 1
        if 0 <= idx < len(items):
            return idx
        return -1
    except (ValueError, KeyboardInterrupt, EOFError):
        return -1


# ══════════════════════════════════════
# CUSTOM PROMPT INPUT (with Ctrl+P)
# ══════════════════════════════════════

def prompt_input() -> str:
    """
    Custom input prompt with full line editing and Ctrl+P detection.

    Uses os.read() for reliable raw I/O on Termux.
    Handles both ESC[A/ESCOA arrow sequences.

    Returns:
        str — user input text, or CTRL_P_SENTINEL, or '/exit'
    """
    if not sys.stdin.isatty():
        try:
            return console.input('[bold green]you > [/bold green]').strip()
        except (KeyboardInterrupt, EOFError):
            return '/exit'

    fd = sys.stdin.fileno()
    out_fd = sys.stdout.fileno()
    old_settings = termios.tcgetattr(fd)

    buffer = ''
    cursor_pos = 0
    history = []
    history_idx = -1
    temp_buffer = ''

    PROMPT_TEXT = 'you > '
    PROMPT_ANSI = '\033[1;32m' + PROMPT_TEXT + '\033[0m'
    PROMPT_BYTES = PROMPT_ANSI.encode()

    # ── Paste detection state ──
    # On Termux/Android: Enter key sends \r (0x0d), paste sends \n (0x0a)
    # Detect rapid char arrival (< 5ms apart = paste, not human typing)
    paste_mode = False
    paste_buffer = ''
    last_char_time = 0.0
    PASTE_THRESHOLD = 0.005  # 5ms between chars = likely paste

    def draw_line():
        """Redraw the input line."""
        os.write(out_fd, b'\r' + PROMPT_BYTES + buffer.encode('utf-8') + b'\033[K')
        if cursor_pos < len(buffer):
            move = len(buffer) - cursor_pos
            os.write(out_fd, f'\033[{move}D'.encode())

    def cleanup():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
        try:
            os.write(out_fd, b'\033[?25h')
            sys.stdout.flush()
        except Exception:
            pass

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        _flush_stdin(fd)

        tty.setraw(fd)
        draw_line()

        while True:
            raw = _raw_read_byte(fd, 0.15)
            if not raw:
                # Timeout — if we were in paste mode, flush the paste buffer
                if paste_mode and paste_buffer:
                    buffer = buffer[:cursor_pos] + paste_buffer + buffer[cursor_pos:]
                    cursor_pos += len(paste_buffer)
                    paste_buffer = ''
                    paste_mode = False
                    draw_line()
                continue

            ch = raw.decode('latin-1')

            # ── Enter (real key = \r, NOT \n which is paste) ──
            if ch == '\r':
                # Flush any pending paste first
                if paste_mode and paste_buffer:
                    buffer = buffer[:cursor_pos] + paste_buffer + buffer[cursor_pos:]
                    cursor_pos += len(paste_buffer)
                    paste_buffer = ''
                    paste_mode = False
                os.write(out_fd, b'\r\n')
                sys.stdout.flush()
                text = buffer.strip()
                if text and (not history or history[-1] != buffer):
                    history.append(buffer)
                return text

            # ── Paste newline (\n) — join into one line with space ──
            if ch == '\n':
                if paste_mode:
                    # In paste mode: add space to paste_buffer, not buffer
                    if paste_buffer and not paste_buffer.endswith(' '):
                        paste_buffer += ' '
                else:
                    # Normal newline in input (rare): append space
                    if buffer and not buffer.endswith(' '):
                        buffer = buffer[:cursor_pos] + ' ' + buffer[cursor_pos:]
                        cursor_pos += 1
                    draw_line()
                continue

            # ── Ctrl+P → Settings Panel ──
            elif ch == '\x10':
                if paste_mode and paste_buffer:
                    buffer = buffer[:cursor_pos] + paste_buffer + buffer[cursor_pos:]
                    paste_buffer = ''
                    paste_mode = False
                os.write(out_fd, b'\r\n')
                sys.stdout.flush()
                return CTRL_P_SENTINEL

            # ── Ctrl+C / Ctrl+D → Exit ──
            elif ch == '\x03' or ch == '\x04':
                if paste_mode and paste_buffer:
                    buffer = buffer[:cursor_pos] + paste_buffer + buffer[cursor_pos:]
                    paste_buffer = ''
                    paste_mode = False
                os.write(out_fd, b'\r\n')
                sys.stdout.flush()
                return '/exit'

            # ── Backspace ──
            elif ch == '\x7f' or ch == '\x08':
                if cursor_pos > 0:
                    buffer = buffer[:cursor_pos - 1] + buffer[cursor_pos:]
                    cursor_pos -= 1
                    draw_line()

            # ── Ctrl+U → Clear line ──
            elif ch == '\x15':
                buffer = buffer[cursor_pos:]
                cursor_pos = 0
                draw_line()

            # ── Ctrl+W → Delete word ──
            elif ch == '\x17':
                saved = cursor_pos
                while cursor_pos > 0 and buffer[cursor_pos - 1] == ' ':
                    buffer = buffer[:cursor_pos - 1] + buffer[cursor_pos:]
                    cursor_pos -= 1
                while cursor_pos > 0 and buffer[cursor_pos - 1] != ' ':
                    buffer = buffer[:cursor_pos - 1] + buffer[cursor_pos:]
                    cursor_pos -= 1
                if cursor_pos != saved:
                    draw_line()

            # ── Ctrl+K → Kill to end ──
            elif ch == '\x0b':
                if cursor_pos < len(buffer):
                    buffer = buffer[:cursor_pos]
                    draw_line()

            # ── Ctrl+A → Home ──
            elif ch == '\x01':
                if cursor_pos != 0:
                    cursor_pos = 0
                    draw_line()

            # ── Ctrl+E → End ──
            elif ch == '\x05':
                if cursor_pos != len(buffer):
                    cursor_pos = len(buffer)
                    draw_line()

            # ── Ctrl+L → Clear screen ──
            elif ch == '\x0c':
                os.write(out_fd, b'\033[2J\033[H')
                draw_line()

            # ── Tab → insert spaces ──
            elif ch == '\x09':
                buffer = buffer[:cursor_pos] + '    ' + buffer[cursor_pos:]
                cursor_pos += 4
                draw_line()

            # ── Escape sequence (arrows, etc.) ──
            elif ch == '\x1b':
                key = _read_escape_sequence(fd, 0.15)
                if key is None:
                    pass  # Plain ESC, ignore

                elif key == 'A':
                    # Up arrow — history back
                    if history:
                        if history_idx == -1:
                            temp_buffer = buffer
                            history_idx = len(history) - 1
                        else:
                            history_idx = max(0, history_idx - 1)
                        buffer = history[history_idx]
                        cursor_pos = len(buffer)
                        draw_line()

                elif key == 'B':
                    # Down arrow — history forward
                    if history_idx >= 0:
                        history_idx += 1
                        if history_idx >= len(history):
                            history_idx = -1
                            buffer = temp_buffer
                        else:
                            buffer = history[history_idx]
                        cursor_pos = len(buffer)
                        draw_line()

                elif key == 'C':
                    # Right arrow
                    if cursor_pos < len(buffer):
                        cursor_pos += 1
                        draw_line()

                elif key == 'D':
                    # Left arrow
                    if cursor_pos > 0:
                        cursor_pos -= 1
                        draw_line()

                elif key == 'H':
                    cursor_pos = 0
                    draw_line()

                elif key == 'F':
                    cursor_pos = len(buffer)
                    draw_line()

                elif key == 'DELETE':
                    if cursor_pos < len(buffer):
                        buffer = buffer[:cursor_pos] + buffer[cursor_pos + 1:]
                        draw_line()

            # ── Printable characters (with paste detection) ──
            elif ord(ch) >= 32:
                now = time.monotonic()
                time_delta = now - last_char_time
                last_char_time = now

                # If we were in paste mode but now char arrived slowly, flush paste first
                if paste_mode and time_delta >= PASTE_THRESHOLD and paste_buffer:
                    buffer = buffer[:cursor_pos] + paste_buffer + buffer[cursor_pos:]
                    cursor_pos += len(paste_buffer)
                    paste_buffer = ''
                    paste_mode = False

                # Detect paste start: rapid character arrival
                if time_delta < PASTE_THRESHOLD and not paste_mode:
                    paste_mode = True
                    paste_buffer = ''

                if paste_mode:
                    # During paste: accumulate chars silently, show at end
                    paste_buffer += ch
                else:
                    # Normal typing: insert char immediately
                    buffer = buffer[:cursor_pos] + ch + buffer[cursor_pos:]
                    cursor_pos += 1
                    draw_line()

    except Exception:
        cleanup()
        raise

    finally:
        cleanup()


# ══════════════════════════════════════
# BANNER & HELPERS
# ══════════════════════════════════════

BANNER = r"""[bold cyan]
________                                            __    
\______ \   ____   ____ ______  ______ ____   ____ |  | __
 |    |  \_/ __ \_/ __ \\____ \/  ___// __ \_/ __ \|  |/ /
 |    `   \  ___/\  ___/|  |_> >___ \\  ___/\  ___/|    < 
/_______  /\___  >\___  >   __/____  >\___  >\___  >__|_ \   [bold red]v7.0[/bold red]
        \/     \/     \/|__|       \/     \/     \/     \/[/bold cyan]

[dim]    DeepSeek CLI Agent v7.0[/dim]
[dim]    Developer : Xbibz Official[/dim]
[dim]    Connectors : Telegram & Discord [/dim]
"""


def show_banner():
    """Display the CLI banner."""
    console.print(BANNER)
    console.print()


def show_welcome(provider_name: str, model: str, has_key: bool):
    """Show welcome message with active provider info."""
    status = '[green]active[/green]' if has_key else '[red]no key[/red]'
    console.print(f'  Provider: [bold]{provider_name}[/bold] ({status})')
    console.print(f'  Model:    [bold]{model}[/bold]')
    console.print()


def show_help():
    """Display help / available commands."""
    table = Table(title='Commands', box=box.ROUNDED, show_lines=False,
                  border_style='cyan', title_style='bold cyan')
    table.add_column('Command', style='bold green', min_width=18)
    table.add_column('Description', style='white')

    commands = [
        ('Ctrl+P',         'Open settings panel'),
        ('/help',          'Show this help message'),
        ('/version',       'Show version and capabilities'),
        ('/tools',         'Show all 65+ available tools'),
        ('/clear',         'Clear conversation history'),
        ('/export',        'Export chat to text file'),
        ('/system <text>', 'Update system prompt'),
        ('/provider',      'Switch AI provider (interactive select)'),
        ('/model',         'Switch model (interactive select)'),
        ('/key',           'Set API key for current provider'),
        ('/models',        'List available models'),
        ('/info',          'Show current config info'),
        ('/thinking',      'Toggle thinking/reasoning visibility'),
        ('/compact',       'Compact conversation (keep system + last 10 msgs)'),
        ('/live_search',   'Live web search (DuckDuckGo + Google News + Bing)'),
        ('/live_models',   'Fetch all models from provider API'),
        ('/search_model',  'Search/filter models from provider API'),
        ('/exit',          'Exit the CLI'),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)
    console.print()


def show_version():
    """Display version information."""
    version_table = Table(box=box.SIMPLE, show_header=False, border_style='cyan')
    version_table.add_column('Key', style='bold cyan', min_width=20)
    version_table.add_column('Value', style='white')
    version_table.add_row('Version', 'DeepSeek CLI Agent v7.7')
    version_table.add_row('Developer', 'Xbibz Official')
    version_table.add_row('TUI', 'Full Real-Time Stream | Rich Markdown | Smooth Buffer')
    version_table.add_row('Features', '90+ Tools | 7 Providers | Smart Loop | OCR | Connectors')
    version_table.add_row('Providers', 'OpenRouter, Gemini, HuggingFace, OpenAI, Anthropic, Groq, Together')
    version_table.add_row('Max Tool Rounds', '12 (smart loop with text-based fallback)')
    version_table.add_row('Tool Categories', 'File, Web, Code, System, Math, Utility, PDF, DOCX, Image, Video, Browser')
    console.print(version_table)
    console.print()
