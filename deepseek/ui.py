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
#   - StreamRenderer: Inline formatting for headers/bold/italic/code during streaming
#   - **bold**, *italic*, `code`, headers rendered in real-time via Rich markup
#   - No Markdown re-render on finalize (streamed content stays as-is)
#
# FIXED v7.7:
#   - StreamRenderer: Smooth buffered output (20ms minimum flush interval)
#   - TUIStatusBar: Compact status bar for tool output
#   - Improved tool call/result display with professional formatting
#   - New ASCII art banner v7.7 with feature list
#   - Better visual separators and color coding throughout
#
# FIXED v7.10:
#   - StreamRenderer: live animated "thinking…" indicator shown the instant a
#     turn starts and auto-cleared on the first streamed token, so the thought
#     process is realtime and the UI never looks frozen/"stuck" during latency
#     (works for every provider, including content-only ones like Agnes AI).
#   - interactive_select: each menu line is clipped to the terminal width (one
#     physical row per line) so the settings panel can no longer wrap and
#     "pile up" / stack on narrow / mobile (Termux) terminals.
#   - prompt_input: the input line is rendered on a single physical row via a
#     horizontally-scrolled window, so a long prompt no longer wraps and the
#     "you > …" prompt no longer re-prints / stacks on every keystroke on
#     narrow / mobile (Termux) terminals.
#
# FIXED v7.11:
#   - Visible thought process for content-only models: a labelled "─── thinking
#     ───" block now streams the model's reasoning live before the answer, even
#     on providers (e.g. Agnes AI) that never emit a separate reasoning field.
#     Driven by a reasoning pre-pass in agent.py.
#   - Tool rounds are now UNLIMITED (agent.py): the agent keeps using tools until
#     it produces a final answer (anti-stuck safety still applies). The tool-round
#     indicator shows just "round N" instead of "N/limit".

import json
import sys
import os
import re
import threading
import time
import tty
import termios
import select as _select
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape

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

# Sentinel value returned by prompt_input() when Ctrl+X is pressed
CTRL_X_SENTINEL = '__CTRL_X__'
CTRL_DOWN_SENTINEL = '__CTRL_DOWN__'
CTRL_LEFT_SENTINEL = '__CTRL_LEFT__'
CTRL_RIGHT_SENTINEL = '__CTRL_RIGHT__'

# ── Persistent input history across prompts ──
_input_history: list = []
_input_history_idx: int = -1
_input_temp_buffer: str = ''

# ── Slash commands list for Tab completion ──
SLASH_COMMANDS = [
    ('/help',          'Show this help message'),
    ('/version',       'Show version and capabilities'),
    ('/k',             'Show token/context usage'),
    ('/context',       'Show token/context usage'),
    ('/init',          'Scan project & create AGENTS.md'),
    ('/install',       'Install a skill (npx skills add)'),
    ('/skills',        'List/manage installed skills'),
    ('/tools',         'Show all available tools'),
    ('/clear',         'Clear conversation history'),
    ('/export',        'Export chat to text file'),
    ('/system',        'Update system prompt'),
    ('/provider',      'Switch AI provider'),
    ('/model',         'Switch model'),
    ('/key',           'Set API key'),
    ('/models',        'List available models'),
    ('/info',          'Show current config info'),
    ('/thinking',      'Toggle thinking/reasoning visibility'),
    ('/compact',       'Compact conversation (keep system + last 10)'),
    ('/live_search',   'Live web search'),
    ('/live_models',   'Fetch all models from provider API'),
    ('/search_model',  'Search/filter models from provider API'),
    ('/agent',         'Switch agent profile'),
    ('/session',       'List/delete saved sessions'),
    ('/connectors',    'Show connectors status'),
    ('/telegram',      'Telegram connector menu'),
    ('/discord',       'Discord connector menu'),
    ('/mcp',           'MCP server management'),
    ('/exit',          'Exit the CLI'),
]

# ══════════════════════════════════════
# LOADING SPINNER
# ══════════════════════════════════════

SPINNER_FRAMES = ['✶', '✸', '✹', '✺', '✻', '✼', '❊']

# Per-tool-call icon cycle
TOOL_ICONS = ['❖', '◆', '◇', '⬟']


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
        # Sleep briefly (50ms) to allow trailing newline characters to arrive in OS buffer
        time.sleep(0.05)
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
    Renders streamed LLM output in REAL-TIME using append-only writes to
    stdout — thinking + response appear token-by-token as they arrive.

    v7.9 (lightweight realtime, no-duplicate):
      - Each chunk is written to stdout exactly ONCE and the cursor is never
        repositioned, so output can never be duplicated regardless of length
        (the old cursor save/restore re-render duplicated text once it
        scrolled; the Rich ``Live`` rewrite re-parsed the whole Markdown on
        every refresh which made long generations feel heavy / "stuck").
      - Thinking streams in a dim style; the response streams in the default
        style. Both are fully real-time and cheap (O(1) work per chunk).
      - render_final() does NOT reprint the answer (it is already on screen);
        it only finalizes the line. This is what guarantees no duplicates.

    Flow per round:
      1. reset_for_new_round()              -> clear per-round state
      2. append_thinking()/append_content() -> stream straight to stdout
      3a. render_final(text)                -> finalize (no reprint), last turn
      3b. show_done()                       -> finalize before tool execution
    """

    _DIM = '\033[2m'
    _RESET = '\033[0m'

    def __init__(self, thinking_visible: bool = True):
        self.thinking_visible = thinking_visible
        self._thinking_text = ''
        self._content_text = ''
        self._thinking_open = False      # currently streaming dim thinking
        self._content_started = False    # have we emitted any content yet
        self._at_line_start = True       # is the cursor at column 0?
        self._streamed_lines = 0         # lines written during this round (for overwrite)
        self._lock = threading.Lock()
        self._line_buffer = ''
        self._table_rows = []
        self._table_has_header = False
        self._tool_call_counter = 0
        self._tool_spinner_stop = None
        self._tool_spinner_thread = None
        self._tool_line = ''
        # ── live "waiting" indicator (pre-first-token) ──
        self._is_tty = False
        try:
            self._is_tty = sys.stdout.isatty()
        except Exception:
            self._is_tty = False
        self._waiting = False
        self._wait_thread = None
        self._wait_stop = None
        self._wait_label = 'thinking…'
        
        # ═══════════════════════════════════════════════════════════
        # MESSAGE HISTORY (v7.12) — Ctrl+Up/Ctrl+Down navigation
        # ═══════════════════════════════════════════════════════════
        self._message_history = []      # List of past messages
        self._history_index = -1        # Current position in history
        self._max_history = 100         # Maximum history entries

    # ── low-level write ──

    def _w(self, s: str):
        if not s:
            return
        sys.stdout.write(s)
        sys.stdout.flush()
        self._at_line_start = s.endswith('\n')
        self._streamed_lines += s.count('\n')

    def _newline_if_needed(self):
        if not self._at_line_start:
            self._w('\n')

    def _flush_table(self) -> str:
        if not self._table_rows:
            return ''
        try:
            try:
                import tabulate
                if self._table_has_header and len(self._table_rows) > 1:
                    result = tabulate.tabulate(
                        self._table_rows[1:], headers=self._table_rows[0],
                        tablefmt='rounded_grid', stralign='left',
                    )
                else:
                    result = tabulate.tabulate(
                        self._table_rows, tablefmt='rounded_grid', stralign='left',
                    )
            except ImportError:
                result = self._render_fallback_table(self._table_rows, self._table_has_header)
            self._table_rows = []
            self._table_has_header = False
            return result
        except Exception:
            self._table_rows = []
            self._table_has_header = False
            return ''

    def _render_fallback_table(self, rows: list[list[str]], has_header: bool) -> str:
        import unicodedata
        if not rows:
            return ""
        # Find max width for each column
        col_widths = []
        num_cols = max(len(r) for r in rows)
        for i in range(num_cols):
            max_w = 0
            for r in rows:
                if i < len(r):
                    clean_cell = re.sub(r'\[/?\w+.*?\]', '', str(r[i]))
                    w = 0
                    for char in clean_cell:
                        status = unicodedata.east_asian_width(char)
                        if status in ('W', 'F', 'A'):
                            w += 2
                        else:
                            w += 1
                    max_w = max(max_w, w)
            col_widths.append(max(max_w, 3))

        lines = []
        # Top border
        top_border = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
        lines.append(top_border)

        for idx, row in enumerate(rows):
            row_cells = []
            for i in range(num_cols):
                val = str(row[i]) if i < len(row) else ""
                clean_val = re.sub(r'\[/?\w+.*?\]', '', val)
                
                w = 0
                for char in clean_val:
                    status = unicodedata.east_asian_width(char)
                    if status in ('W', 'F', 'A'):
                        w += 2
                    else:
                        w += 1
                
                pad_len = col_widths[i] - w
                row_cells.append(val + " " * pad_len)
            
            row_line = "│ " + " │ ".join(row_cells) + " │"
            lines.append(row_line)

            # Header separator
            if idx == 0 and has_header and len(rows) > 1:
                sep = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
                lines.append(sep)
            elif idx < len(rows) - 1:
                sep = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
                lines.append(sep)

        # Bottom border
        bottom_border = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"
        lines.append(bottom_border)
        return "\n".join(lines)

    # ── thinking region ──

    def _open_thinking(self):
        if not self._thinking_open:
            self._newline_if_needed()
            # clear, labelled block header so the reasoning is obviously the
            # model's "thought process"; reasoning then streams dim + indented.
            self._w(self._DIM + '  ─── thinking ───' + self._RESET + '\n' + self._DIM + '  ')
            self._thinking_open = True

    def _close_thinking_if_open(self):
        if self._thinking_open:
            self._w(self._RESET + '\n')
            self._thinking_open = False

    def _flush_buffer(self):
        try:
            sys.stdout.flush()
        except Exception:
            pass

    # ── live waiting indicator (so the UI never looks frozen/stuck) ──

    def _wait_loop(self):
        frames = '✶✸✹✺✻✼❊'
        # brief grace period — instant responses shouldn't flash a spinner
        if self._wait_stop.wait(0.18):
            return
        i = 0
        while not self._wait_stop.is_set():
            frame = frames[i % len(frames)]
            with self._lock:
                if self._wait_stop.is_set():
                    break
                try:
                    sys.stdout.write(f'\r  {self._DIM}{frame} {self._wait_label}{self._RESET}\033[K')
                    sys.stdout.flush()
                except Exception:
                    break
            i += 1
            if self._wait_stop.wait(0.12):
                break

    def begin_stream(self, label: str = 'thinking…', timeout: float = 90):
        """Start an animated indicator with auto-timeout.
        Stops automatically after `timeout` seconds if no chunk arrives."""
        self.stop_waiting()
        if not self._is_tty:
            return
        self._wait_label = label
        self._wait_stop = threading.Event()
        self._waiting = True
        self._wait_thread = threading.Thread(
            target=self._wait_loop_with_timeout,
            args=(timeout,),
            daemon=True
        )
        self._wait_thread.start()

    def _wait_loop_with_timeout(self, timeout: float):
        """Wait loop that auto-stops after timeout."""
        frames = '✶✸✹✺✻✼❊'
        deadline = time.time() + timeout
        i = 0
        while not self._wait_stop.wait(0.12):
            if time.time() > deadline:
                break
            frame = frames[i % len(frames)]
            try:
                sys.stdout.write(f'\r  {self._DIM}{frame} {self._wait_label}{self._RESET}\033[K')
                sys.stdout.flush()
            except Exception:
                break
            i += 1
        self.stop_waiting()

    def stop_waiting(self):
        """Stop the waiting indicator and erase its line (idempotent, safe to
        call on every chunk)."""
        if not self._waiting:
            return
        self._waiting = False
        if self._wait_stop is not None:
            self._wait_stop.set()
        t = self._wait_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=0.6)
        self._wait_thread = None
        if self._is_tty:
            try:
                sys.stdout.write('\r\033[K')
                sys.stdout.flush()
            except Exception:
                pass
            self._at_line_start = True

    def end_stream(self):
        """End the stream and stop any active waiting animation."""
        self.stop_waiting()
        self._close_thinking_if_open()
        self._flush_line_buffer()

    # ── round reset ──

    def reset_for_new_round(self):
        """Reset streaming state for a new agent round."""
        self._close_thinking_if_open()
        with self._lock:
            self._thinking_text = ''
            self._content_text = ''
            self._content_started = False
            self._streamed_lines = 0

    # ── public streaming API ──

    def append_thinking(self, chunk: str):
        """Stream a thinking/reasoning chunk to stdout in real time (dim)."""
        self.stop_waiting()
        if not self.thinking_visible or not chunk:
            return
        with self._lock:
            self._thinking_text += chunk
        self._open_thinking()
        # Strip markdown headers + tool-call XML so they don't pollute the UI
        cleaned = re.sub(r'^#{1,6}\s+', '', chunk, flags=re.MULTILINE)
        cleaned = re.sub(r'</?function[^>]*>|</?tool_call[^>]*>|</?parameter[^>]*>|<parameter=\w+>', '', cleaned)
        cleaned = re.sub(r'<function=\w+>', '', cleaned)
        self._w(cleaned.replace('\n', '\n  '))

    def append_content(self, chunk: str):
        """Stream a response chunk to stdout in real time."""
        self.stop_waiting()
        if not chunk:
            return
        with self._lock:
            self._content_text += chunk
        self._close_thinking_if_open()
        if not self._content_started:
            self._newline_if_needed()
            self._content_started = True

        self._line_buffer += chunk

        # Process complete lines
        while '\n' in self._line_buffer:
            idx = self._line_buffer.index('\n')
            line = self._line_buffer[:idx]
            self._line_buffer = self._line_buffer[idx + 1:]
            self._emit_line(line)

        self._at_line_start = chunk.endswith('\n')
        self._streamed_lines += chunk.count('\n')

    def _emit_line(self, line: str):
        """Format and print one complete line of content."""
        stripped = line.strip()

        # ── Table row ──
        table_match = re.match(r'^.*?(\|.+\|)$', stripped)
        if table_match:
            table_part = table_match.group(1)
            prefix = stripped[:len(stripped) - len(table_part)].strip()

            # Handle concatenated rows: |a|b||c|d| → split on ||
            rows_raw = re.split(r'\|\|', table_part)
            first_row = True
            for row_raw in rows_raw:
                if not row_raw.strip():
                    continue
                row_raw = row_raw.strip()
                if not row_raw.startswith('|'):
                    row_raw = '|' + row_raw
                if not row_raw.endswith('|'):
                    row_raw = row_raw + '|'
                cells = [c.strip() for c in row_raw.strip('|').split('|') if c.strip()]
                if len(cells) >= 2:
                    if all(re.match(r'^-{1,}$', c) for c in cells):
                        if not self._table_rows:
                            self._table_has_header = True
                        continue
                    if first_row and prefix:
                        # Text before table — emit as regular line, start new table
                        if self._table_rows:
                            table_str = self._flush_table()
                            if table_str:
                                console.print(table_str, markup=True)
                                console.print()
                        text = self._apply_inline_formatting(prefix)
                        console.print(text, markup=True, end='')
                        console.print()
                    formatted = []
                    for c in cells:
                        c = re.sub(r'`([^`]+)`', r'[green]\1[/green]', c)
                        c = re.sub(r'\*\*(.+?)\*\*', r'[bold cyan]\1[/bold cyan]', c)
                        c = re.sub(r'~~(.+?)~~', r'[dim red]\1[/dim red]', c)
                        formatted.append(c)
                    self._table_rows.append(formatted)
                    first_row = False
            return

        # ── Flush any pending table ──
        if self._table_rows:
            table_str = self._flush_table()
            if table_str:
                console.print(table_str, markup=True, soft_wrap=True)
                console.print()

        # ── Todo/checklist line ──
        if '[✓]' in stripped:
            text = self._apply_inline_formatting(line)
            txt = text.replace('[✓]', '[green]✓[/green]')
            console.print(txt, markup=True, end='')
            console.print()
            return
        if '[x]' in stripped:
            text = self._apply_inline_formatting(line)
            txt = text.replace('[x]', '[green]✓[/green]')
            console.print(txt, markup=True, end='')
            console.print()
            return
        if '[ ]' in stripped:
            text = self._apply_inline_formatting(line)
            txt = text.replace('[ ]', '[dim]○[/dim]')
            console.print(txt, markup=True, end='')
            console.print()
            return

        # ── Regular line ──
        text = self._apply_inline_formatting(line)
        console.print(text, markup=True, end='')
        console.print()

    def _format_markdown(self, text: str) -> str:
        lines = text.split('\n')
        formatted_lines = []
        in_code_block = False
        code_block_lang = ''
        code_block_lines = []
        table_rows = []
        table_has_header = False

        def flush_table():
            nonlocal table_rows, table_has_header
            if not table_rows:
                return ''
            try:
                try:
                    import tabulate
                    if table_has_header and len(table_rows) > 1:
                        r = tabulate.tabulate(table_rows[1:], headers=table_rows[0],
                                              tablefmt='rounded_grid', stralign='left')
                    else:
                        r = tabulate.tabulate(table_rows, tablefmt='rounded_grid', stralign='left')
                except ImportError:
                    r = self._render_fallback_table(table_rows, table_has_header)
                return r
            except Exception:
                return ''
            finally:
                table_rows = []
                table_has_header = False

        for line in lines:
            stripped = line.strip()

            # Code blocks
            if stripped.startswith('```'):
                if in_code_block:
                    code_content = '\n'.join(code_block_lines)
                    lang = code_block_lang or 'text'
                    formatted_lines.append('')
                    formatted_lines.append(f'  [dim]┌─ {lang} ──────────────────────────────[/dim]')
                    try:
                        from pygments import highlight
                        from pygments.lexers import get_lexer_by_name, guess_lexer
                        from pygments.formatters import Terminal256Formatter
                        lexer = get_lexer_by_name(lang, stripall=True) if lang else guess_lexer(code_content)
                        highlighted = highlight(code_content, lexer, Terminal256Formatter())
                        code_lines = highlighted.rstrip('\n').split('\n')
                    except ImportError:
                        code_lines = [f'[green]{line}[/green]' for line in code_content.split('\n')]
                    for cl in code_lines:
                        formatted_lines.append(f'  [dim]│[/dim] {cl}')
                    formatted_lines.append(f'  [dim]└──────────────────────────────────┘[/dim]')
                    formatted_lines.append('')
                    in_code_block = False
                    code_block_lines = []
                    code_block_lang = ''
                else:
                    in_code_block = True
                    code_block_lang = stripped[3:].strip()
                    continue

            if in_code_block:
                code_block_lines.append(line)
                continue

            # Flush pending table on non-table line
            if table_rows and not re.match(r'^\|.+\|$', stripped):
                t = flush_table()
                if t:
                    formatted_lines.append(t)
                    formatted_lines.append('')

            # Tables
            if re.match(r'^\|.+\|$', stripped):
                cells = [c.strip() for c in stripped.strip('|').split('|')]
                if all(re.match(r'^-{1,}$', c) for c in cells):
                    if not table_rows:
                        table_has_header = True
                    continue
                table_rows.append(cells)
                continue

            # Process inline formatting
            processed = line

            matched = False

            # Headers
            for level in range(6, 0, -1):
                prefix = '#' * level + ' '
                if processed.startswith(prefix):
                    level_text = processed[len(prefix):]
                    level_text = self._clean_inline(level_text)
                    formatted_lines.append(f'\n  [bold cyan]{"═" * (level + 2)}[/bold cyan]')
                    formatted_lines.append(f'  [bold cyan]{level_text}[/bold cyan]')
                    formatted_lines.append(f'  [bold cyan]{"═" * (level + 2)}[/bold cyan]')
                    matched = True
                    break

            if matched:
                continue

            # Blockquotes
            if processed.startswith('> '):
                quote_text = processed[2:]
                quote_text = self._clean_inline(quote_text)
                formatted_lines.append(f'  [dim yellow]▸[/dim yellow] [dim]{quote_text}[/dim]')
                continue

            # Unordered lists
            if re.match(r'^\s*[-*]\s+', processed):
                processed = re.sub(r'^\s*[-*]\s+', '  [green]•[/green] ', processed)
                processed = self._clean_inline(processed)
                formatted_lines.append(f'  {processed}')
                continue

            # Ordered lists
            if re.match(r'^\s*\d+\.\s+', processed):
                processed = re.sub(r'^\s*\d+\.\s+', '  [green]✓[/green] ', processed)
                processed = self._clean_inline(processed)
                formatted_lines.append(f'  {processed}')
                continue

            # Horizontal rules
            if re.match(r'^\s*[-*_]{3,}\s*$', processed):
                formatted_lines.append(f'  [dim]{"─" * 60}[/dim]')
                continue

            # Empty lines
            if not processed.strip():
                formatted_lines.append('')
                continue

            # Regular text
            processed = self._apply_inline_formatting(processed)
            formatted_lines.append(f'  {processed}')

        # Flush trailing table
        if table_rows:
            t = flush_table()
            if t:
                formatted_lines.append(t)

        return '\n'.join(formatted_lines)
    
    def _clean_inline(self, text: str) -> str:
        """Remove inline markdown formatting from text for headers/quotes."""
        # Remove ** and *
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        # Remove `
        text = re.sub(r'`(.+?)`', r'\1', text)
        return text
    
    def _apply_inline_formatting(self, text: str) -> str:
        """Apply Rich markup to a line of text."""
        stripped = text.strip()

        # Horizontal rule
        if re.match(r'^-{3,}$', stripped) or re.match(r'^\*{3,}$', stripped):
            return '[dim]──────────────────────────────────────[/dim]'

        # Headers
        header_match = re.match(r'^#{1,6}\s+(.+)$', text)
        if header_match:
            h = header_match.group(1)
            h = re.sub(r'`([^`]+)`', r'[green]\1[/green]', h)
            h = re.sub(r'\*\*(.+?)\*\*', r'[bold cyan]\1[/bold cyan]', h)
            h = re.sub(r'~~(.+?)~~', r'[dim red]\1[/dim red]', h)
            return f'[bold cyan]{h}[/bold cyan]'

        # Blockquotes
        bq = re.match(r'^>\s+(.+)$', text)
        if bq:
            q = bq.group(1)
            q = re.sub(r'\*\*(.+?)\*\*', r'[bold italic]\1[/bold italic]', q)
            q = re.sub(r'`([^`]+)`', r'[green]\1[/green]', q)
            return f'[dim yellow]▸[/dim yellow] [italic]{q}[/italic]'

        # Unordered lists
        ul = re.match(r'^(\s*)[-*]\s+(.+)$', text)
        if ul:
            rest = ul.group(2)
            rest = re.sub(r'`([^`]+)`', r'[green]\1[/green]', rest)
            rest = re.sub(r'\*\*(.+?)\*\*', r'[bold cyan]\1[/bold cyan]', rest)
            rest = re.sub(r'\*(.+?)\*', r'[italic yellow]\1[/italic yellow]', rest)
            rest = re.sub(r'~~(.+?)~~', r'[dim red]\1[/dim red]', rest)
            rest = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'[blue]\1[/blue]', rest)
            return f'  [green]•[/green] {rest}'

        # Ordered lists
        ol = re.match(r'^(\s*)\d+\.\s+(.+)$', text)
        if ol:
            rest = ol.group(2)
            rest = re.sub(r'`([^`]+)`', r'[green]\1[/green]', rest)
            rest = re.sub(r'\*\*(.+?)\*\*', r'[bold cyan]\1[/bold cyan]', rest)
            rest = re.sub(r'\*(.+?)\*', r'[italic yellow]\1[/italic yellow]', rest)
            rest = re.sub(r'~~(.+?)~~', r'[dim red]\1[/dim red]', rest)
            rest = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'[blue]\1[/blue]', rest)
            return f'  [green]✓[/green] {rest}'

        # Inline formatting
        result = re.sub(r'`([^`]+)`', r'[green]\1[/green]', text)
        result = re.sub(r'\*\*(.+?)\*\*', r'[bold cyan]\1[/bold cyan]', result)
        result = re.sub(r'(?<!\*)\*(.+?)(?<!\*)\*', r'[italic yellow]\1[/italic yellow]', result)
        result = re.sub(r'~~(.+?)~~', r'[dim red]\1[/dim red]', result)
        result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'[blue]\1[/blue]', result)
        return result

    def render_final(self, text: str):
        """Finalize the response."""
        self.stop_waiting()
        self._close_thinking_if_open()
        self._flush_line_buffer()
        self._newline_if_needed()
        console.print()

    def _flush_line_buffer(self):
        if self._table_rows:
            table_str = self._flush_table()
            if table_str:
                console.print(table_str, markup=True, soft_wrap=True)
                console.print()
        if self._line_buffer:
            text = self._apply_inline_formatting(self._line_buffer)
            console.print(text, markup=True, end='', soft_wrap=True)
            console.print()
            self._line_buffer = ''
            self._at_line_start = True

    def show_thinking_as_content(self, thinking_text: str):
        """Reasoning-only fallback (empty content). If thinking was hidden,
        surface it now as the answer; otherwise it already streamed live."""
        self.stop_waiting()
        self._close_thinking_if_open()
        if (not self.thinking_visible) and thinking_text and thinking_text.strip():
            self._newline_if_needed()
            self._w(thinking_text.strip())
        self._newline_if_needed()
        console.print()

    def show_error(self, message: str):
        """Display an error message."""
        self.stop_waiting()
        self._close_thinking_if_open()
        self._newline_if_needed()
        console.print(f'  [bold red]Error:[/bold red] {message}')

    def show_done(self):
        self.stop_waiting()
        self._close_thinking_if_open()
        self._flush_line_buffer()
        self._newline_if_needed()
        self._flush_buffer()

    def _tool_spinner_loop(self):
        frames = TOOL_ICONS
        i = 0
        _PB = '\033[38;5;245m'
        _RESET = '\033[0m'
        stop_event = self._tool_spinner_stop
        if stop_event is None:
            return
        while not stop_event.wait(0.15):
            frame = frames[i % len(frames)]
            sys.stdout.write(f'\r  \033[1;33m{frame}\033[0m {_PB}executing\u2026{_RESET}\033[K')
            sys.stdout.flush()
            i += 1

    def show_tool_call(self, tool_name: str, args: dict):
        self._close_thinking_if_open()
        self._flush_buffer()

        icon = TOOL_ICONS[self._tool_call_counter % len(TOOL_ICONS)]
        self._tool_call_counter += 1

        # ── Read tools ──
        read_tools = {'read_file', 'read_pdf', 'read_docx', 'read_xlsx',
                      'read_pptx', 'read_csv', 'file_info'}
        search_tools = {'grep', 'search_files', 'live_search', 'web_search'}
        write_tools = {'write_file', 'edit_file', 'create_pdf', 'create_docx',
                       'create_xlsx', 'create_csv', 'create_pptx',
                       'edit_pptx', 'edit_xlsx', 'edit_docx', 'convert_document'}
        code_tools = {'run_code', 'run_shell', 'run_python'}
        browse_tools = {'browser_navigate', 'browser_download', 'browser_screenshot'}
        list_tools = {'list_files', 'tree_view'}

        def _path(v):
            if isinstance(v, str):
                return v.split('/')[-1] if '/' in v else v
            return str(v)

        if tool_name in read_tools:
            path = next((_path(v) for k, v in args.items()
                        if k in ('path', 'file_path', 'input_path')), '')
            extra = ''
            if 'offset' in args or 'limit' in args:
                off = args.get('offset', '')
                lim = args.get('limit', '')
                extra = f' [offset={off}, limit={lim}]'
            console.print(f'  [bold blue]→ Read[/bold blue] [cyan]{path}[/cyan]{extra}')

        elif tool_name in search_tools:
            pattern = next((v for k, v in args.items()
                          if k in ('pattern', 'query', 'q')), '')
            path = args.get('path', '')
            if path:
                console.print(f'  [bold yellow]{icon} Grep[/bold yellow] [green]"{pattern}"[/green] in [cyan]{path}[/cyan]')
            else:
                console.print(f'  [bold yellow]{icon} Search[/bold yellow] [green]"{pattern}"[/green]')

        elif tool_name in write_tools:
            path = next((_path(v) for k, v in args.items()
                        if k in ('path', 'output', 'save_path', 'output_path')), '')
            if tool_name == 'edit_file':
                console.print(f'  [bold green]→ Edit[/bold green] [cyan]{path}[/cyan]')
            else:
                console.print(f'  [bold green]→ Write[/bold green] [cyan]{path}[/cyan]')

        elif tool_name in code_tools:
            cmd = next((v for k, v in args.items()
                       if k in ('command', 'code', 'shell')), '')
            preview = cmd[:80] + '...' if len(cmd) > 80 else cmd
            console.print(f'  [bold magenta]→ Run[/bold magenta] [dim]{preview}[/dim]')

        elif tool_name in browse_tools:
            url = next((v for k, v in args.items() if k == 'url'), '')
            console.print(f'  [bold blue]→ Browse[/bold blue] [cyan]{url[:60]}[/cyan]')

        elif tool_name in list_tools:
            path = args.get('path', '.')
            pattern = args.get('pattern', '')
            if pattern:
                console.print(f'  [bold blue]→ List[/bold blue] [cyan]{path}[/cyan] ({pattern})')
            else:
                console.print(f'  [bold blue]→ List[/bold blue] [cyan]{path}[/cyan]')

        else:
            # Compact display for todolist_update: just show the index
            if tool_name == 'todolist_update' and 'index' in args:
                idx = args['index']
                self._tool_line = f'  [bold yellow]{icon}[/bold yellow] [dim][{idx}][/dim]'
            elif tool_name == 'todowrite':
                title = args.get('title', '').lstrip('#').strip()[:60]
                self._tool_line = f'  [bold yellow]{icon} todowrite[/bold yellow] [dim]{escape(title)}[/dim]'
                console.print(self._tool_line)
            else:
                arg_str = ' '.join(f'{k}={_path(v)}' for k, v in args.items())[:80]
                self._tool_line = f'  [bold yellow]{icon} {tool_name}[/bold yellow] [dim]{arg_str}[/dim]'
                console.print(self._tool_line)

        # ── Start tool spinner ──
        self._tool_spinner_stop = threading.Event()
        self._tool_spinner_thread = threading.Thread(target=self._tool_spinner_loop, daemon=True)
        self._tool_spinner_thread.start()

        self._flush_buffer()

    def pause_tool_spinner(self):
        """Temporarily pause the tool spinner animation (e.g. for user confirmation prompts)."""
        if self._tool_spinner_stop:
            self._tool_spinner_stop.set()
        if self._tool_spinner_thread and self._tool_spinner_thread.is_alive():
            self._tool_spinner_thread.join(timeout=0.5)
        self._tool_spinner_stop = None
        self._tool_spinner_thread = None
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()

    def resume_tool_spinner(self):
        """Resume the tool spinner animation."""
        if not self._tool_spinner_thread:
            self._tool_spinner_stop = threading.Event()
            self._tool_spinner_thread = threading.Thread(target=self._tool_spinner_loop, daemon=True)
            self._tool_spinner_thread.start()

    def show_tool_result(self, tool_name: str, result: str):
        self._close_thinking_if_open()

        # ── Stop tool spinner ──
        if self._tool_spinner_stop:
            self._tool_spinner_stop.set()
        if self._tool_spinner_thread and self._tool_spinner_thread.is_alive():
            self._tool_spinner_thread.join(timeout=0.5)
        self._tool_spinner_stop = None
        self._tool_spinner_thread = None
        # Clear the spinner line and move cursor
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()

        self._flush_buffer()

        # Ensure result is a string
        if not isinstance(result, str):
            try:
                result = json.dumps(result, indent=2)
            except Exception:
                result = str(result)

        # ── Diff display ──
        if result.startswith('[DIFF]') and result.endswith('[/DIFF]'):
            diff_body = result[6:-7].strip()
            diff_lines = diff_body.split('\n')
            MAX_DIFF = 20
            shown = 0
            for line in diff_lines:
                if shown >= MAX_DIFF:
                    remaining = len(diff_lines) - shown
                    console.print(f'  [dim]┃ ... ({remaining} more lines)[/dim]')
                    break
                line = line.rstrip()
                if not line:
                    continue
                marker = line[0]
                rest = line[1:].strip()
                if marker == '-':
                    console.print(f'  [red]┃ {escape(rest)}[/red]', soft_wrap=True)
                elif marker == '+':
                    console.print(f'  [green]┃ {escape(rest)}[/green]', soft_wrap=True)
                elif marker == ' ':
                    console.print(f'  [dim]┃ {escape(rest)}[/dim]', soft_wrap=True)
                elif marker == '~':
                    console.print(f'  [dim]┃ {escape(line[1:].strip())}[/dim]', soft_wrap=True)
                else:
                    console.print(f'  [dim]┃ {escape(line)}[/dim]', soft_wrap=True)
                shown += 1
            return

        if result.startswith('[ERROR]'):
            console.print(f'  [dim]  ✗ {escape(result[7:].strip())}[/dim]', soft_wrap=True)
            return

        if result.startswith('[TIMEOUT]'):
            console.print(f'  [yellow]  ⏱ {escape(result[9:].strip())}[/yellow]', soft_wrap=True)
            return

        # ── Success — smart preview ──
        lines = result.split('\n')
        total_lines = len(lines)

        if tool_name in ('read_file', 'read_pdf', 'read_docx', 'read_xlsx',
                         'read_pptx', 'read_csv'):
            preview = '\n'.join(lines[:2])
            if len(preview) > 120:
                preview = preview[:117] + '...'
            if preview:
                console.print(f'    [dim]{escape(preview)}[/dim]', soft_wrap=True)

        elif tool_name in ('edit_file',):
            pass  # diff already shown above

        elif tool_name in ('write_file', 'create_pdf', 'create_docx',
                           'create_xlsx', 'create_csv', 'create_pptx'):
            if not result.startswith('[DIFF]'):
                console.print(f'    [dim]✓ Written to {tool_name.replace("create_","").replace("write_","")}[/dim]', soft_wrap=True)

        elif tool_name in ('list_files', 'tree_view', 'file_info'):
            preview = '\n'.join(lines[:4])
            if len(preview) > 200:
                preview = preview[:197] + '...'
            if preview:
                console.print(f'    [dim]{escape(preview)}[/dim]', soft_wrap=True)

        elif tool_name in ('run_code', 'run_shell', 'run_python'):
            if result.strip():
                preview = result.strip()[:200]
                console.print(f'    [dim]{escape(preview)}[/dim]', soft_wrap=True)

        elif tool_name in ('web_search', 'live_search', 'grep', 'search_files'):
            preview = '\n'.join(lines[:5])
            if len(preview) > 300:
                preview = preview[:297] + '...'
            if preview:
                console.print(f'    [dim]{escape(preview)}[/dim]', soft_wrap=True)

        elif tool_name in ('ocr_read', 'ocr_url'):
            text = result.strip()[:200]
            console.print(f'    [dim]{escape(text)}[/dim]', soft_wrap=True)

        elif tool_name in ('todowrite', 'todolist_get'):
            todo_lines = result.split('\n')
            printed = False
            for tl in todo_lines:
                stripped = tl.strip()
                if not stripped:
                    continue
                if not printed:
                    printed = True
                # Brief sleep to animate ticking off items one by one in real-time
                time.sleep(0.12)
                if stripped.startswith('#'):
                    console.print(f'  [dim]┃ [/dim] [bold]{escape(stripped)}[/bold]')
                elif '[✓]' in stripped or '[x]' in stripped:
                    parts = stripped.split('[✓]' if '[✓]' in stripped else '[x]')
                    task = parts[-1].strip()
                    console.print(f'  [dim]┃ [/dim] [green]✓[/green] [dim]{escape(task)}[/dim]')
                elif '[ ]' in stripped:
                    task = stripped.split('[ ]', 1)[-1].strip()
                    console.print(f'  [dim]┃ [/dim] [ ] [dim]{escape(task)}[/dim]')
                else:
                    console.print(f'  [dim]┃[/dim] [dim]{escape(stripped)}[/dim]')
            if printed:
                console.print('  [dim]┃[/dim]')

        elif tool_name in ('todolist_update',):
            # Combine call + result into one compact line
            result_text = result.strip()
            console.print(f'{self._tool_line} [dim]{escape(result_text)}[/dim]')

        else:
            lines2 = result.split('\n')

            # Detect todo/list content: lines with [✓], [ ], or starts with #
            if any('[✓]' in l or '[ ]' in l or '[x]' in l for l in lines2[:5]):
                printed = False
                for tl in lines2:
                    s = tl.strip()
                    if not s:
                        continue
                    if not printed:
                        printed = True
                    # Brief sleep to animate ticking off items one by one in real-time
                    time.sleep(0.12)
                    if s.startswith('#'):
                        console.print(f'  [dim]┃ [/dim] [bold]{s}[/bold]')
                    elif '[✓]' in s:
                        task = s.split('[✓]', 1)[-1].strip()
                        console.print(f'  [dim]┃ [/dim] [green]✓[/green] [dim]{task}[/dim]')
                    elif '[x]' in s:
                        task = s.split('[x]', 1)[-1].strip()
                        console.print(f'  [dim]┃ [/dim] [green]✓[/green] [dim]{task}[/dim]')
                    elif '[ ]' in s:
                        task = s.split('[ ]', 1)[-1].strip()
                        console.print(f'  [dim]┃ [/dim] [ ] [dim]{task}[/dim]')
                    else:
                        console.print(f'  [dim]┃[/dim] [dim]{s}[/dim]')
                if printed:
                    console.print('  [dim]┃[/dim]')
                return

            # Generic preview
            if len(result) > 500:
                display = result[:497] + '...'
            else:
                display = result
            if len(lines2) > 8:
                display = '\n'.join(lines2[:8]) + f'\n    ... ({len(lines2) - 8} more)'
            if display.strip():
                console.print(f'    [dim]{display}[/dim]')



    # ═══════════════════════════════════════════════════════════
    # MESSAGE HISTORY (v7.12)
    # ═══════════════════════════════════════════════════════════
    
    def add_to_history(self, role: str, content: str):
        """Add a message to history for Ctrl+Up/Ctrl+Down navigation."""
        with self._lock:
            self._message_history.append({
                'role': role,
                'content': content,
                'timestamp': time.time(),
            })
            if len(self._message_history) > self._max_history:
                self._message_history = self._message_history[-self._max_history:]
            self._history_index = len(self._message_history) - 1
    
    def get_history_message(self, direction: str) -> str:
        """Get a message from history based on direction."""
        with self._lock:
            if not self._message_history:
                return ''
            
            if direction == 'up':
                if self._history_index > 0:
                    self._history_index -= 1
                    msg = self._message_history[self._history_index]
                    return f"[{msg['role'].upper()}] {msg['content'][:200]}"
                else:
                    return '[History: Beginning]'
            elif direction == 'down':
                if self._history_index < len(self._message_history) - 1:
                    self._history_index += 1
                    msg = self._message_history[self._history_index]
                    return f"[{msg['role'].upper()}] {msg['content'][:200]}"
                else:
                    return '[History: End]'
            return ''
    
    def show_history_menu(self) -> str:
        """Display a menu of recent messages for selection."""
        self._close_thinking_if_open()
        self._flush_buffer()
        
        if not self._message_history:
            console.print('  [dim]No message history available.[/dim]')
            return ''
        
        console.print('  [bold cyan]─── Message History ───[/bold cyan]')
        console.print()
        
        start_idx = max(0, len(self._message_history) - 10)
        for i in range(start_idx, len(self._message_history)):
            msg = self._message_history[i]
            role_color = {
                'user': 'yellow',
                'assistant': 'green',
                'system': 'dim',
                'tool': 'cyan',
            }.get(msg['role'], 'dim')
            
            preview = msg['content'][:80].replace('\n', ' ')
            if len(msg['content']) > 80:
                preview += '...'
            
            console.print(f'    [{role_color}]•[/ {role_color}] [{msg['role'].upper()}] {preview}')
        
        console.print()
        console.print('  [dim]Use Ctrl+Up/Ctrl+Down to navigate history[/dim]')
        console.print()
        
        return ''

    def show_paste_indicator(self, line_count: int, char_count: int = 0):
        """Display a '[ Pasted ~N Lines ]' indicator."""
        self._close_thinking_if_open()
        self._flush_buffer()
        
        if line_count == 1:
            console.print(f'  [dim][ Pasted ~{line_count} Line ][/dim]')
        else:
            console.print(f'  [dim][ Pasted ~{line_count} Lines ][/dim]')
        
        if char_count > 0:
            if char_count > 1024:
                console.print(f'    [dim]({char_count/1024:.1f} KB)[/dim]')
            else:
                console.print(f'    [dim]({char_count} bytes)[/dim]')
        
        console.print()

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
        if self.max_rounds and self.max_rounds > 0:
            parts.append(f'[dim]round {self.round_num}/{self.max_rounds}[/dim]')
        else:
            parts.append(f'[dim]round {self.round_num}[/dim]')
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
# CONFIRM ACTION (Left/Right Select)
# ══════════════════════════════════════

def show_skill_content(name: str, content: str):
    """Display a skill's SKILL.md content in a formatted box."""
    lines = content.strip().split('\n')
    console.print()
    console.print(f'  [bold cyan]╔══ Skill: {name} {"═" * max(2, 50 - len(name))}╗[/bold cyan]')
    for line in lines:
        if line.startswith('# '):
            console.print(f'  [bold white]  {line[2:].strip()}[/bold white]')
        elif line.startswith('## '):
            console.print(f'  [bold cyan]  {line[3:].strip()}[/bold cyan]')
        elif line.startswith('- '):
            console.print(f'  [dim]  {line}[/dim]')
        elif line.strip() == '':
            console.print()
        else:
            console.print(f'    {line}')
    console.print(f'  [bold cyan]╚{"═" * 55}╝[/bold cyan]')
    console.print()

def confirm_action(tool_name: str, args: dict, verb: str = 'execute') -> str:
    """
    Interactive confirmation with numeric selection.
    Shows tool info and three numbered options.
    Returns 'always_allow', 'allow_once', or 'reject'.
    """
    options = [
        ('Allow Once', 'yellow'),
        ('Always Allow', 'green'),
        ('Reject', 'red'),
    ]
    fd = sys.stdin.fileno()
    out_fd = sys.stdout.fileno()
    old_settings = termios.tcgetattr(fd)
    arg_summary = ', '.join(f'{k}={v}' for k, v in list(args.items())[:3])
    if len(args) > 3:
        arg_summary += '...'

    def _draw():
        line1 = f'\r\033[1;33m⚠ Confirm {verb}:\033[0m \033[1;36m{tool_name}\033[0m \033[2m({arg_summary})\033[0m\033[K\n'
        line2_parts = []
        for i, (label, color) in enumerate(options):
            ccode = {'red': '31', 'yellow': '33', 'green': '32'}.get(color, '0')
            line2_parts.append(f'\033[{ccode}m[{i}] {label}\033[0m')
        line2 = '  '.join(line2_parts) + '\n'
        line3 = f'  number to select  •  Enter = Allow Once  •  Esc/Ctrl+C = cancel\033[K'
        os.write(out_fd, (line1 + line2 + line3).encode())
        sys.stdout.flush()

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        _flush_stdin(fd)
        tty.setraw(fd)
        _draw()
        while True:
            raw = _raw_read_byte(fd, 0.3)
            if not raw:
                continue
            ch = raw.decode('latin-1')

            if ch == '\r' or ch == '\n':
                os.write(out_fd, b'\r\033[2A\033[J')
                sys.stdout.flush()
                return 'allow_once'

            if ch == '\x03' or ch == '\x04' or ch == '\x1b':
                os.write(out_fd, b'\r\033[2A\033[J')
                sys.stdout.flush()
                return 'reject'

            if ch in ('0', '1', '2'):
                os.write(out_fd, b'\r\033[2A\033[J')
                sys.stdout.flush()
                label = options[int(ch)][0]
                return {'Allow Once': 'allow_once', 'Always Allow': 'always_allow', 'Reject': 'reject'}[label]
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
    return 'reject'

# ══════════════════════════════════════
# INTERACTIVE FILTER SELECT
# ══════════════════════════════════════

def interactive_filter_select(items, title=None):
    """
    Interactive selection with real-time character filtering.
    Typing characters filters the list; arrow keys navigate; Enter confirms.
    Returns selected index into original items, or -1 on cancel.
    """
    if not items:
        return -1
    if not sys.stdin.isatty():
        return _select_fallback(items, title, 0)

    fd = sys.stdin.fileno()
    out_fd = sys.stdout.fileno()
    old_settings = termios.tcgetattr(fd)
    filter_text = ''
    filtered = list(items)
    selected = 0

    def _mw():
        try:
            return os.get_terminal_size(out_fd).columns
        except Exception:
            return 80

    def _clip(t):
        w = max(8, _mw() - 1)
        return t[:w - 1] + '…' if len(t) > w else t

    total_lines = len(items) + 2 + 2

    def draw():
        nonlocal total_lines
        total_lines = len(filtered) + (1 if title else 0) + 2
        os.write(out_fd, b'\033[?25l')
        if title:
            t = _clip(f'  {title}' + (f' [{len(filtered)}]' if len(filtered) != len(items) else ''))
            os.write(out_fd, f'\r\033[1;36m{t}\033[0m\033[K\r\n'.encode())
        fline = f'  \033[2mfilter:\033[0m {filter_text}' if filter_text else '  \033[2mtype to filter...\033[0m'
        os.write(out_fd, f'\r{fline}\033[K\r\n'.encode())
        for i, item in enumerate(filtered):
            if i == selected:
                os.write(out_fd, f'\r\033[1;32m  >> {_clip(item)}\033[0m\033[K\r\n'.encode())
            else:
                os.write(out_fd, f'\r     {_clip(item)}\033[K\r\n'.encode())
        hint = _clip('    ↑↓: nav | Enter: select | Esc: cancel')
        os.write(out_fd, f'\r\033[2m{hint}\033[K\r\n'.encode())
        os.write(out_fd, b'\033[?25h')
        sys.stdout.flush()

    def redraw():
        os.write(out_fd, f'\033[{total_lines}A'.encode())
        os.write(out_fd, b'\033[J')
        draw()

    def _rebuild_filter():
        nonlocal filtered, selected
        if filter_text:
            filtered = [it for it in items if filter_text.lower() in it.split('  ')[0].lstrip('/').lower()]
        else:
            filtered = list(items)
        if selected >= len(filtered):
            selected = max(0, len(filtered) - 1)

    try:
        tty.setraw(fd)
        draw()
    except Exception:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return -1

    result = -1
    try:
        while True:
            raw = _raw_read_byte(fd, 0.15)
            if not raw:
                continue
            ch = raw.decode('latin-1')

            if ch == '\r' or ch == '\n':
                if not filtered:
                    continue
                sel = items.index(filtered[selected])
                result = sel
                break

            if ch == '\x03' or ch == '\x04':
                break

            if ch == '\x1b':
                key = _read_escape_sequence(fd, 0.15)
                if key is None:
                    break
                if key == 'A':
                    if filtered:
                        selected = (selected - 1) % len(filtered)
                        redraw()
                elif key == 'B':
                    if filtered:
                        selected = (selected + 1) % len(filtered)
                        redraw()
                continue

            if ch == '\x7f':
                if filter_text:
                    filter_text = filter_text[:-1]
                    _rebuild_filter()
                    redraw()
                continue

            if ch.isprintable():
                filter_text += ch
                _rebuild_filter()
                redraw()
                continue

    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return result

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
    # = items + (optional title) + hint line
    total_lines = len(items) + (1 if title else 0) + 1

    def _menu_width():
        """Current terminal width (columns), with safe fallbacks."""
        for getter in (lambda: os.get_terminal_size(out_fd).columns,
                       lambda: os.get_terminal_size().columns):
            try:
                w = getter()
                if w and w > 0:
                    return w
            except Exception:
                continue
        return 80

    def _clip(text):
        """Clip a plain (ANSI-free) line so it occupies exactly ONE physical
        row. Without this, long items wrap on narrow/mobile terminals while the
        redraw math counts logical lines — leaving stale rows that pile up."""
        w = max(8, _menu_width() - 1)
        if len(text) > w:
            return text[:w - 1] + '…'
        return text

    def draw():
        """Draw the full menu from current position."""
        os.write(out_fd, b'\033[?25l')  # Hide cursor
        if title:
            line = _clip(f'  {title}')
            os.write(out_fd, f'\r\033[1;36m{line}\033[0m\033[K\r\n'.encode())
        for i, item in enumerate(items):
            if i == selected:
                line = _clip(f'  >> {item}')
                os.write(out_fd, f'\r\033[1;32m{line}\033[0m\033[K\r\n'.encode())
            else:
                line = _clip(f'     {item}')
                os.write(out_fd, f'\r{line}\033[0m\033[K\r\n'.encode())
        hint = _clip('    Up/Down: navigate | Enter: select | Esc: cancel')
        os.write(out_fd, f'\r\033[2m{hint}\033[0m\033[K\r\n'.encode())
        os.write(out_fd, b'\033[?25h')  # Show cursor
        sys.stdout.flush()

    def redraw():
        """Clear previous draw and redraw from scratch."""
        os.write(out_fd, f'\033[{total_lines}A\033[J'.encode())
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

_depth = 0  # module-level depth state for Ctrl+X navigation


def _reverse_search(fd, out_fd, saved_buffer):
    """
    Interactive reverse-i-search through _input_history.

    Ctrl+R toggles search mode; user types a query, matches are shown
    in real-time. Ctrl+R cycles backwards, Enter accepts, Esc/Ctrl+G cancels.

    Returns:
        str: the matched history entry, or saved_buffer if cancelled.
    """
    query = ''
    matches = []
    match_pos = -1

    def _update_matches():
        nonlocal matches, match_pos
        if not query:
            matches = []
            match_pos = -1
            return
        q = query.lower()
        matches = [i for i, h in enumerate(_input_history) if q in h.lower()]
        if matches:
            match_pos = len(matches) - 1
        else:
            match_pos = -1

    def _display():
        nonlocal match_pos
        if matches and match_pos >= 0:
            matched = _input_history[matches[match_pos]]
            os.write(out_fd, f'\r\033[2m(reverse-i-search)\033[0m `{query}`: \033[36m{matched}\033[0m\033[K'.encode())
        else:
            label = 'failing-' if query else ''
            os.write(out_fd, f'\r\033[2m({label}reverse-i-search)\033[0m `{query}`:\033[K'.encode())

    _update_matches()
    _display()

    while True:
        raw = _raw_read_byte(fd, 0.15)
        if not raw:
            continue

        ch = raw.decode('latin-1')

        if ch == '\x12':
            if matches and match_pos > 0:
                match_pos -= 1
            _display()

        elif ch in ('\r', '\n'):
            if matches and match_pos >= 0:
                return _input_history[matches[match_pos]]
            return saved_buffer

        elif ch == '\x1b':
            key = _read_escape_sequence(fd, 0.15)
            if key is None:
                return saved_buffer
            _display()

        elif ch in ('\x03', '\x04', '\x07'):
            return saved_buffer

        elif ch in ('\x7f', '\x08'):
            if query:
                query = query[:-1]
                _update_matches()
            _display()

        elif ord(ch) >= 32:
            query += ch
            _update_matches()
            _display()


def prompt_input(depth: int = None) -> str:
    """
    Custom input prompt with full line editing and Ctrl+P detection.

    Uses os.read() for reliable raw I/O on Termux.
    Handles both ESC[A/ESCOA arrow sequences.
    depth: 0 = normal, 1+ = sub-context (Ctrl+X to toggle, Up to exit)

    Returns:
        str — user input text, or CTRL_P_SENTINEL, or CTRL_X_SENTINEL, or '/exit'
    """
    global _depth
    if depth is not None:
        _depth = depth

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
    global _input_history, _input_history_idx, _input_temp_buffer

    PROMPT_TEXT = 'you > ' if _depth == 0 else '\033[1;33m[session]\033[0m > '
    PROMPT_ANSI = '\033[1;32m' + PROMPT_TEXT + '\033[0m'
    PROMPT_BYTES = PROMPT_ANSI.encode()
    PROMPT_VIS_LEN = 6 if _depth == 0 else len('[session] > ')  # visible width (no ANSI)

    # ── Paste detection state ──
    # On Termux/Android: Enter key sends \r (0x0d), paste sends \n (0x0a)
    # Detect rapid char arrival (< 5ms apart = paste, not human typing)
    paste_mode = False
    paste_buffer = ''
    last_char_time = 0.0
    PASTE_THRESHOLD = 0.005  # 5ms between chars = likely paste

    def _cols():
        """Current terminal width (columns), with safe fallbacks."""
        for getter in (lambda: os.get_terminal_size(out_fd).columns,
                       lambda: os.get_terminal_size().columns):
            try:
                c = getter()
                if c and c > 0:
                    return c
            except Exception:
                continue
        return 80

    def _last_line_start():
        idx = buffer.rfind('\n')
        return idx + 1 if idx >= 0 else 0

    def _is_multiline():
        return '\n' in buffer

    def _cont_prompt():
        return b'\033[1;32m... > \033[0m'

    _menu_active = False
    _menu_submit = False

    def _show_command_menu():
        nonlocal _menu_active, _menu_submit, buffer, cursor_pos
        if _menu_active:
            return
        buf = buffer.strip()
        inp = buf[1:] if buf.startswith('/') else ''
        matches = [(c, d) for c, d in SLASH_COMMANDS
                   if not inp or c.startswith('/' + inp)]
        if not matches:
            return
        _menu_active = True
        items = [f'{c}  —  {d}' for c, d in matches]
        idx = interactive_filter_select(items, title=f'Commands')
        if idx >= 0:
            buffer = matches[idx][0]
            cursor_pos = len(buffer)
            _menu_submit = True
        else:
            buffer = ''
            cursor_pos = 0
        _menu_active = False
        draw_line()

    def draw_line():
        """Redraw the input on a SINGLE physical row.

        The buffer is shown through a horizontally-scrolled window sized to the
        terminal width, so the line never wraps. Wrapping was exactly what made
        the prompt re-print and 'stack' on narrow / mobile (Termux) terminals:
        a bare '\\r' only returns to the start of the LAST wrapped row, so once
        prompt+text exceeded the width the prompt got redrawn on every keystroke.

        When the buffer contains newlines (multi-line mode), only the last line
        is shown with a continuation prompt '... > '."""
        cols = _cols()
        if _is_multiline():
            last_start = _last_line_start()
            display_text = buffer[last_start:]
            p_bytes = _cont_prompt()
            p_vis = 6
            rel_cursor = max(0, cursor_pos - last_start)
        else:
            display_text = buffer
            p_bytes = PROMPT_BYTES
            p_vis = PROMPT_VIS_LEN
            rel_cursor = cursor_pos
        avail = max(1, cols - p_vis - 1)
        if len(display_text) <= avail or rel_cursor <= avail:
            start = 0
        else:
            start = rel_cursor - avail
        window = display_text[start:start + avail]
        vis_cursor = rel_cursor - start
        os.write(out_fd, b'\r' + p_bytes + window.encode('utf-8') + b'\033[K')
        back = len(window) - vis_cursor
        if back > 0:
            os.write(out_fd, f'\033[{back}D'.encode())

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
            if _menu_submit:
                _menu_submit = False
                text = buffer.strip()
                if text and (not _input_history or _input_history[-1] != buffer):
                    _input_history.append(buffer)
                os.write(out_fd, b'\r\n')
                sys.stdout.flush()
                return text

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
                if paste_mode:
                    # In paste mode: \r may be part of pasted content
                    # (some terminals use \r\n for pasted newlines).
                    # Treat as another paste character (add space).
                    if paste_buffer and not paste_buffer.endswith(' '):
                        paste_buffer += ' '
                    draw_line()
                    continue
                # Real Enter key - flush any pending paste and submit
                if paste_mode and paste_buffer:
                    buffer = buffer[:cursor_pos] + paste_buffer + buffer[cursor_pos:]
                    cursor_pos += len(paste_buffer)
                    paste_buffer = ''
                    paste_mode = False
                # Multi-line continuation: Enter on a line ending with \\ → insert newline
                stripped = buffer.rstrip()
                if stripped.endswith('\\') and not stripped.endswith('\\\\'):
                    buffer = stripped[:-1] + '\n'
                    cursor_pos = len(buffer)
                    os.write(out_fd, b'\r\n')
                    sys.stdout.flush()
                    draw_line()
                    continue
                os.write(out_fd, b'\r\n')
                sys.stdout.flush()
                text = buffer.strip()
                if text and (not _input_history or _input_history[-1] != buffer):
                    _input_history.append(buffer)
                return text

            # ── Paste newline (\n) — keep as newline for multi-line support ──
            if ch == '\n':
                if paste_mode:
                    # Strip space added by \r handler for \r\n Windows line endings
                    if paste_buffer and paste_buffer[-1] == ' ':
                        paste_buffer = paste_buffer[:-1]
                    paste_buffer += '\n'
                else:
                    buffer = buffer[:cursor_pos] + '\n' + buffer[cursor_pos:]
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

            # ── Ctrl+X → Toggle depth ──
            elif ch == '\x18':
                if paste_mode and paste_buffer:
                    buffer = buffer[:cursor_pos] + paste_buffer + buffer[cursor_pos:]
                    paste_buffer = ''
                    paste_mode = False
                os.write(out_fd, b'\r\n')
                sys.stdout.flush()
                return CTRL_X_SENTINEL

            # ── Ctrl+R → Reverse history search ──
            elif ch == '\x12':
                if paste_mode and paste_buffer:
                    buffer = buffer[:cursor_pos] + paste_buffer + buffer[cursor_pos:]
                    paste_buffer = ''
                    paste_mode = False
                os.write(out_fd, b'\r\n')
                sys.stdout.flush()
                saved = buffer
                result = _reverse_search(fd, out_fd, saved)
                buffer = result
                cursor_pos = len(buffer)
                draw_line()

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
                    last_start = _last_line_start()
                    if cursor_pos > last_start:
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

            # ── Ctrl+A → Home (start of current line) ──
            elif ch == '\x01':
                target = _last_line_start()
                if cursor_pos != target:
                    cursor_pos = target
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

            # ── Tab → command completion (when buffer starts with "/") ──
            elif ch == '\x09':
                if buffer.startswith('/'):
                    inp = buffer.lstrip()
                    if inp:
                        matches = [(c, d) for c, d in SLASH_COMMANDS if c.startswith(inp)]
                    else:
                        matches = list(SLASH_COMMANDS)
                    if matches:
                        items = [f'{c}  —  {d}' for c, d in matches]
                        idx = interactive_select(items, title=f'Commands ({len(matches)})')
                        if idx >= 0:
                            buffer = matches[idx][0]
                            cursor_pos = len(buffer)
                    else:
                        cleanup()
                        console.print('  [yellow]No matching commands. Type /help for list.[/yellow]')
                    tty.setraw(fd)
                    draw_line()
                else:
                    buffer = buffer[:cursor_pos] + '    ' + buffer[cursor_pos:]
                    cursor_pos += 4
                    draw_line()

            # ── Escape sequence (arrows, etc.) ──
            elif ch == '\x1b':
                key = _read_escape_sequence(fd, 0.15)
                if key is None:
                    # Plain ESC — clear current input
                    buffer = ''
                    cursor_pos = 0
                    draw_line()

                elif key == 'A':
                    if _depth > 0:
                        # Up arrow while in sub-context → go back up
                        os.write(out_fd, b'\r\n')
                        sys.stdout.flush()
                        return CTRL_X_SENTINEL
                    # Up arrow — history back
                    if _input_history:
                        if _input_history_idx == -1:
                            _input_temp_buffer = buffer
                            _input_history_idx = len(_input_history) - 1
                        else:
                            _input_history_idx = max(0, _input_history_idx - 1)
                        buffer = _input_history[_input_history_idx]
                        cursor_pos = len(buffer)
                        draw_line()

                elif key == 'B':
                    if _depth > 0:
                        os.write(out_fd, b'\r\n')
                        sys.stdout.flush()
                        return CTRL_DOWN_SENTINEL
                    # Down arrow — history forward
                    if _input_history_idx >= 0:
                        _input_history_idx += 1
                        if _input_history_idx >= len(_input_history):
                            _input_history_idx = -1
                            buffer = _input_temp_buffer
                        else:
                            buffer = _input_history[_input_history_idx]
                        cursor_pos = len(buffer)
                        draw_line()

                elif key == 'C':
                    if _depth > 0:
                        os.write(out_fd, b'\r\n')
                        sys.stdout.flush()
                        return CTRL_RIGHT_SENTINEL
                    # Right arrow (stays within current line in multi-line mode)
                    if cursor_pos < len(buffer):
                        last_start = _last_line_start()
                        next_newline = buffer.find('\n', last_start)
                        line_end = next_newline if next_newline >= 0 else len(buffer)
                        if cursor_pos < line_end:
                            cursor_pos += 1
                            draw_line()

                elif key == 'D':
                    if _depth > 0:
                        os.write(out_fd, b'\r\n')
                        sys.stdout.flush()
                        return CTRL_LEFT_SENTINEL
                    # Left arrow (stays within current line in multi-line mode)
                    if cursor_pos > 0:
                        last_start = _last_line_start()
                        if cursor_pos > last_start:
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
                    was_slash = buffer.strip().startswith('/')
                    buffer = buffer[:cursor_pos] + ch + buffer[cursor_pos:]
                    cursor_pos += 1
                    draw_line()
                    now_slash = buffer.strip().startswith('/')
                    if now_slash and not was_slash:
                        _show_command_menu()
                        tty.setraw(fd)
                    elif now_slash and was_slash:
                        _show_command_menu()
                        tty.setraw(fd)

    except BaseException:
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
/_______  /\___  >\___  >   __/____  >\___  >\___  >__|_ \   [bold red]v7.7[/bold red]
        \/     \/     \/|__|       \/     \/     \/     \/[/bold cyan]

[dim]    DeepSeek CLI Agent v7.7[/dim]
[dim]    Developer : Xbibz Official[/dim]
[dim]    Connectors : Telegram & Discord | Tools: 90+ | Smart Loop[/dim]
"""


def show_banner():
    """Display the CLI banner.

    When the startup version check (enforce_gist) detected a newer release in the
    registry Gist, an "(Update Available vX.Y)" line is rendered right under the
    banner so it's actually visible to the user."""
    import os, sys
    # Skip the banner when dscli is being called recursively (e.g. via agent bash tool)
    if not sys.stdout.isatty() or os.environ.get('DEEPSEEK_NESTED') == '1':
        return
    console.print(BANNER)

    # ── Update Available notice (driven by registry Gist's latest_version) ──
    try:
        from .config import get_update_info, CLIENT_VERSION
        info = get_update_info()
        if info and info.get("latest"):
            latest = info["latest"]
            current = info.get("current", CLIENT_VERSION)
            console.print(
                f"  [black on yellow] Update Available v{latest} [/black on yellow]"
                f"  [dim]you are on v{current}[/dim]"
            )
            console.print(
                "  [yellow]Run [bold]bash install.sh[/bold] (or the one-line installer) to update.[/yellow]"
            )
    except Exception:
        pass

    console.print()


def show_welcome(provider_name: str, model: str, has_key: bool):
    """Show welcome message with active provider info."""
    # Skip if non-TTY (recursive call via bash tool) or DEEPSEEK_NESTED is set
    import os, sys
    if not sys.stdout.isatty() or os.environ.get('DEEPSEEK_NESTED') == '1':
        return
    status = '[green]active[/green]' if has_key else '[red]no key[/red]'
    # Pull the saved session (set by auth.ensure_authenticated) so we can show
    # the logged-in account on the same line as the provider info.
    # Pull the saved session (set by auth.ensure_authenticated) so we can show
    # the logged-in account on the same line as the provider info.
    sess = {}
    try:
        from . import auth as _auth_mod
        sess = getattr(_auth_mod, '_current_session', {}) or {}
    except Exception:
        pass
    who = sess.get('username') or sess.get('email') or ''
    if who:
        console.print(f'  Account:  [bold cyan]{who}[/bold cyan]  |  Provider: [bold]{provider_name}[/bold] ({status})')
    else:
        console.print(f'  Provider: [bold]{provider_name}[/bold] ({status})')
    console.print(f'  Model:    [bold]{model}[/bold]')
    console.print()


def show_help():
    """Display help / available commands."""
    table = Table(title='Commands', box=box.ROUNDED, show_lines=False,
                  border_style='cyan', title_style='bold cyan')
    table.add_column('Command', style='bold green', min_width=18)
    table.add_column('Description', style='white')

    table.add_row('Ctrl+P', 'Open settings panel')
    for cmd, desc in SLASH_COMMANDS:
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
    version_table.add_row('Features', '90+ Tools | 8 Providers | Smart Loop | OCR | Connectors')
    version_table.add_row('Providers', 'OpenRouter, Gemini, HuggingFace, OpenAI, Anthropic, Groq, Together, Agnes AI')
    version_table.add_row('Max Tool Rounds', '12 (smart loop with text-based fallback)')
    version_table.add_row('Tool Categories', 'File, Web, Code, System, Math, Utility, PDF, DOCX, Image, Video, Browser')
    console.print(version_table)
    console.print()
