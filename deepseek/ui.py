# DeepSeek CLI v4 — UI Components
# ═══════════════════════════════════════════════════════════════════
# ARCHITECTURE: Raw sys.stdout for interactive screens, Rich for static screens
#
# TERMINAL INPUT STRATEGY:
#   - _RawMode context manager sets raw terminal mode ONCE per interactive function
#   - _read_key() reads keys WITHOUT changing terminal mode (no set/restore per key)
#   - This eliminates ALL race conditions where escape sequence bytes get echoed
#     during the gap between tty.tcsetattr(restore) and tty.setraw(next call)
# ═══════════════════════════════════════════════════════════════════

import sys
import os
import time
import select
import termios
import tty
import threading
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ══════════════════════════════════════
# ANSI HELPERS (for raw terminal only)
# ══════════════════════════════════════

def _c(code: str) -> str:
    """ANSI color code shorthand."""
    codes = {
        'reset':   '\033[0m',
        'bold':    '\033[1m',
        'dim':     '\033[2m',
        'red':     '\033[31m',
        'green':   '\033[32m',
        'yellow':  '\033[33m',
        'blue':    '\033[34m',
        'cyan':    '\033[36m',
        'white':   '\033[37m',
        'magenta': '\033[35m',
    }
    return codes.get(code, '\033[0m')


def _w(text: str = ''):
    """Write to stdout (raw, no Rich)."""
    sys.stdout.write(text)
    sys.stdout.flush()


def _wl(text: str = ''):
    """Write line to stdout (raw, no Rich).
    CRITICAL: Uses \\r\\n (not \\n) because in raw mode, \\n is ONLY a line feed
    and does NOT return the cursor to column 0. Without \\r, text staircases.
    """
    sys.stdout.write(text + '\r\n')
    sys.stdout.flush()


def _hide_cursor():
    _w('\033[?25l')


def _show_cursor():
    _w('\033[?25h')


def _clear_lines(n: int):
    """Clear n lines above cursor. After call, cursor is n lines above start."""
    for _ in range(n):
        _w('\033[2K')
        _w('\033[1A')


def _draw_and_pad(lines: list, prev_total: int) -> int:
    """Draw lines and pad with blank lines if content shrank."""
    for line in lines:
        _w('\r\033[2K' + line + '\r\n')
    pad_count = prev_total - len(lines)
    for _ in range(pad_count):
        _w('\r\033[2K\r\n')
    return len(lines)


# ══════════════════════════════════════
# RAW TERMINAL MODE MANAGER
# ══════════════════════════════════════

class _RawMode:
    """Context manager that sets raw terminal mode ONCE on entry."""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self):
        self._old = termios.tcgetattr(self._fd)
        new = list(self._old)
        new[0] &= ~(termios.BRKINT | termios.ICRNL |
                     termios.INPCK | termios.ISTRIP | termios.IXON)
        new[1] &= ~(termios.OPOST)
        new[3] &= ~(termios.ECHO | termios.ICANON |
                     termios.ISIG | termios.IEXTEN)
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSANOW, new)
        return self

    def __exit__(self, *args):
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._old = None


def _drain_input():
    """Drain all pending input bytes from stdin (must be in raw mode)."""
    fd = sys.stdin.fileno()
    while True:
        r, _, _ = select.select([fd], [], [], 0.02)
        if not r:
            break
        try:
            os.read(fd, 4096)
        except OSError:
            break


# ══════════════════════════════════════
# SINGLE KEY READER (no mode changes!)
# ══════════════════════════════════════

def _read_key() -> str:
    """Read a single keypress. Assumes raw mode is ALREADY set."""
    fd = sys.stdin.fileno()
    ch = os.read(fd, 1)

    if ch == b'\x1b':
        r, _, _ = select.select([fd], [], [], 0.1)
        if not r:
            return 'ESC'
        ch2 = os.read(fd, 1)
        if ch2 == b'[':
            buf = b''
            for _ in range(10):
                r2, _, _ = select.select([fd], [], [], 0.05)
                if not r2:
                    return 'UNKNOWN'
                b = os.read(fd, 1)
                if 0x40 <= b[0] <= 0x7E:
                    arrow_map = {
                        0x41: 'UP', 0x42: 'DOWN', 0x43: 'RIGHT', 0x44: 'LEFT',
                        0x48: 'HOME', 0x46: 'END',
                    }
                    return arrow_map.get(b[0], 'UNKNOWN')
                elif 0x30 <= b[0] <= 0x3F:
                    buf += b
                    continue
                else:
                    return 'UNKNOWN'
            return 'UNKNOWN'
        elif ch2 == b'O':
            r2, _, _ = select.select([fd], [], [], 0.05)
            if r2:
                os.read(fd, 1)
            return 'UNKNOWN'
        else:
            return 'ESC'

    if ch[0] & 0x80:
        byte0 = ch[0]
        if byte0 & 0xE0 == 0xC0:
            remaining = 1
        elif byte0 & 0xF0 == 0xE0:
            remaining = 2
        elif byte0 & 0xF8 == 0xF0:
            remaining = 3
        else:
            return 'UNKNOWN'
        buf = ch
        for _ in range(remaining):
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                buf += os.read(fd, 1)
            else:
                break
        try:
            s = buf.decode('utf-8')
            if s.isprintable():
                return s
        except (UnicodeDecodeError, ValueError):
            pass
        return 'UNKNOWN'

    try:
        c = ch.decode('ascii')
    except (UnicodeDecodeError, ValueError):
        return 'UNKNOWN'

    ctrl_map = {
        '\x0d': 'ENTER', '\x0a': 'ENTER', '\x09': 'TAB',
        '\x7f': 'BACKSPACE', '\x08': 'BACKSPACE',
        '\x03': 'CTRL_C', '\x04': 'CTRL_D',
        '\x01': 'CTRL_A', '\x02': 'CTRL_B', '\x05': 'CTRL_E',
        '\x06': 'CTRL_F', '\x07': 'CTRL_G', '\x0b': 'CTRL_K',
        '\x0c': 'CTRL_L', '\x0e': 'CTRL_N', '\x0f': 'CTRL_O',
        '\x10': 'CTRL_P', '\x11': 'CTRL_Q', '\x12': 'CTRL_R',
        '\x13': 'CTRL_S', '\x14': 'CTRL_T', '\x15': 'CTRL_U',
        '\x16': 'CTRL_V', '\x17': 'CTRL_W', '\x18': 'CTRL_X',
        '\x19': 'CTRL_Y', '\x1a': 'CTRL_Z',
        '\x1c': 'CTRL_BACKSLASH', '\x1d': 'CTRL_BRACKET_RIGHT',
        '\x1e': 'CTRL_CARET', '\x1f': 'CTRL_UNDERSCORE',
    }
    if c in ctrl_map:
        return ctrl_map[c]
    if c.isprintable():
        return c
    return 'UNKNOWN'


# ══════════════════════════════════════
# LINE INPUT (replaces Python's input())
# ══════════════════════════════════════

def read_input_line(prompt: str = '', palette_commands: list = None):
    """Read a full line of input using RAW terminal mode."""
    with _RawMode():
        _drain_input()
        line_buffer = ''
        cursor_pos = 0
        _w(prompt)

        while True:
            key = _read_key()

            if key == 'ENTER':
                _w('\r\n')
                return line_buffer
            elif key == 'CTRL_C':
                _wl('')
                raise KeyboardInterrupt()
            elif key == 'CTRL_D':
                _wl('')
                raise EOFError()
            elif key == 'BACKSPACE':
                if line_buffer and cursor_pos > 0:
                    line_buffer = line_buffer[:cursor_pos - 1] + line_buffer[cursor_pos:]
                    cursor_pos -= 1
                    tail = line_buffer[cursor_pos:]
                    _w('\b \b' + tail + ' ')
                    if tail:
                        _w(f'\033[{len(tail) + 1}D')
            elif key == 'CTRL_P':
                if palette_commands:
                    _show_cursor()
                    _wl('')
                    cmd = command_palette(palette_commands)
                    if cmd:
                        return cmd
                    _w(prompt + line_buffer)
                    if cursor_pos < len(line_buffer):
                        _w(f'\033[{len(line_buffer) - cursor_pos}D')
            elif key == 'CTRL_A':
                if cursor_pos > 0:
                    _w(f'\033[{cursor_pos}D')
                    cursor_pos = 0
            elif key == 'CTRL_E':
                if cursor_pos < len(line_buffer):
                    _w(f'\033[{len(line_buffer) - cursor_pos}C')
                    cursor_pos = len(line_buffer)
            elif key == 'CTRL_U':
                if cursor_pos > 0:
                    deleted = line_buffer[:cursor_pos]
                    line_buffer = line_buffer[cursor_pos:]
                    cursor_pos = 0
                    _w(f'\033[{len(deleted)}D')
                    _w(line_buffer + ' ')
                    _w(f'\033[{len(line_buffer) + 1}D')
            elif key == 'CTRL_K':
                if cursor_pos < len(line_buffer):
                    deleted_len = len(line_buffer) - cursor_pos
                    line_buffer = line_buffer[:cursor_pos]
                    _w(' ' * deleted_len)
                    _w(f'\033[{deleted_len}D')
            elif key == 'CTRL_W':
                if cursor_pos > 0:
                    i = cursor_pos - 1
                    while i > 0 and line_buffer[i - 1] != ' ':
                        i -= 1
                    deleted = line_buffer[i:cursor_pos]
                    line_buffer = line_buffer[:i] + line_buffer[cursor_pos:]
                    cursor_pos = i
                    _w(f'\033[{len(deleted)}D')
                    _w(line_buffer[cursor_pos:] + ' ')
                    _w(f'\033[{len(line_buffer) - cursor_pos + 1}D')
            elif key in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
                if key == 'LEFT' and cursor_pos > 0:
                    cursor_pos -= 1
                    _w('\033[1D')
                elif key == 'RIGHT' and cursor_pos < len(line_buffer):
                    cursor_pos += 1
                    _w('\033[1C')
            elif key == 'HOME':
                if cursor_pos > 0:
                    _w(f'\033[{cursor_pos}D')
                    cursor_pos = 0
            elif key == 'END':
                if cursor_pos < len(line_buffer):
                    _w(f'\033[{len(line_buffer) - cursor_pos}C')
                    cursor_pos = len(line_buffer)
            elif key == 'ESC':
                pass
            elif key in ('UNKNOWN', 'CTRL_G', 'CTRL_L', 'CTRL_N', 'CTRL_O',
                         'CTRL_Q', 'CTRL_R', 'CTRL_S', 'CTRL_T', 'CTRL_V',
                         'CTRL_X', 'CTRL_Y', 'CTRL_Z', 'CTRL_BACKSLASH',
                         'CTRL_BRACKET_RIGHT', 'CTRL_CARET', 'CTRL_UNDERSCORE',
                         'CTRL_B', 'CTRL_F'):
                pass
            elif key == 'TAB':
                pass
            elif len(key) >= 1 and key.isprintable():
                line_buffer = (line_buffer[:cursor_pos] + key +
                               line_buffer[cursor_pos:])
                cursor_pos += 1
                tail = line_buffer[cursor_pos:]
                _w(key + tail)
                if tail:
                    _w(f'\033[{len(tail)}D')


# ══════════════════════════════════════
# PASSWORD/MASKED INPUT
# ══════════════════════════════════════

def read_password(prompt: str = '') -> str:
    """Read input with characters masked as *."""
    with _RawMode():
        _drain_input()
        line_buffer = ''
        _w(prompt)
        while True:
            key = _read_key()
            if key == 'ENTER':
                _wl('')
                return line_buffer if line_buffer else ''
            elif key in ('ESC', 'CTRL_C', 'CTRL_D'):
                _wl('')
                return None
            elif key == 'BACKSPACE':
                if line_buffer:
                    line_buffer = line_buffer[:-1]
                    _w('\b \b')
            elif key == 'CTRL_U':
                if line_buffer:
                    _w(f'\033[{len(line_buffer)}D')
                    _w(' ' * len(line_buffer))
                    _w(f'\033[{len(line_buffer)}D')
                    line_buffer = ''
            elif key in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'HOME', 'END',
                         'TAB', 'UNKNOWN', 'CTRL_A', 'CTRL_B', 'CTRL_E',
                         'CTRL_F', 'CTRL_K', 'CTRL_L', 'CTRL_N', 'CTRL_P',
                         'CTRL_Q', 'CTRL_R', 'CTRL_S', 'CTRL_T', 'CTRL_V',
                         'CTRL_X', 'CTRL_Y', 'CTRL_Z', 'CTRL_G',
                         'CTRL_BACKSLASH', 'CTRL_BRACKET_RIGHT',
                         'CTRL_CARET', 'CTRL_UNDERSCORE', 'CTRL_W'):
                pass
            elif len(key) >= 1 and key.isprintable():
                line_buffer += key
                _w('*')


# ══════════════════════════════════════
# LOADING ANIMATION
# ══════════════════════════════════════

class LoadingSpinner:
    """Animated loading spinner using raw terminal."""
    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, message: str = 'Loading'):
        self.message = message
        self._running = False
        self._thread = None

    def __enter__(self):
        self._running = True
        _hide_cursor()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        _clear_lines(1)
        _show_cursor()

    def _spin(self):
        i = 0
        while self._running:
            frame = self.FRAMES[i % len(self.FRAMES)]
            _w(f'\r  {_c("cyan")}{frame}{_c("reset")} {self.message}')
            time.sleep(0.08)
            i += 1


# ══════════════════════════════════════
# SPINNER HELPER (for repl.py)
# ══════════════════════════════════════

def with_spinner(message: str = 'Loading'):
    """Return a LoadingSpinner context manager. Usage: with ui.with_spinner('msg'):"""
    return LoadingSpinner(message)


# ══════════════════════════════════════
# BANNER (Rich — static, no interaction)
# ══════════════════════════════════════

def show_banner(provider_name: str, model: str, thinking_visible: bool) -> None:
    """Display startup banner using Rich."""
    console.clear()
    t = Text()
    t.append('\n', style='reset')
    t.append('  ╔═══════════════════════════════════════════╗\n', style='bold cyan')
    t.append('  ║                                           ║\n', style='bold cyan')
    t.append('  ║          ', style='bold cyan')
    t.append('🤖  DeepSeek CLI Agent v4.0', style='bold white')
    t.append('          ║\n', style='bold cyan')
    t.append('  ║          ', style='bold cyan')
    t.append('Multi-Provider · Agentic Loop · 26+ Tools', style='dim')
    t.append('║\n', style='bold cyan')
    t.append('  ║                                           ║\n', style='bold cyan')
    t.append('  ╚═══════════════════════════════════════════╝\n', style='bold cyan')
    t.append(f'\n  Provider: ', style='dim')
    t.append(provider_name, style='magenta bold')
    t.append('\n  Model: ', style='dim')
    t.append(model, style='green bold')
    t.append('\n  Thinking: ', style='dim')
    t.append('ON' if thinking_visible else 'OFF',
             style='bold green' if thinking_visible else 'bold red')
    t.append('\n  Type ', style='dim')
    t.append('/help', style='bold cyan')
    t.append(' for commands · ', style='dim')
    t.append('CTRL+P', style='bold cyan')
    t.append(' command palette\n', style='dim')
    console.print(t)
    console.print()


# ══════════════════════════════════════
# STREAMING OUTPUT
# ══════════════════════════════════════

class StreamRenderer:
    """Renders streaming output using Rich console."""

    def __init__(self, thinking_visible: bool = True):
        self.thinking_visible = thinking_visible
        self._in_thinking = False

    def append_thinking(self, text: str):
        if not self.thinking_visible or not text:
            return
        if not self._in_thinking:
            self._in_thinking = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def append_content(self, text: str):
        if not text:
            return
        if self._in_thinking:
            self._in_thinking = False
            sys.stdout.write('\n')
        sys.stdout.write(text)
        sys.stdout.flush()

    def show_tool_call(self, tool_name: str, args: dict):
        if self._in_thinking:
            self._in_thinking = False
            sys.stdout.write('\n')
        args_str = str(args)
        if len(args_str) > 80:
            args_str = args_str[:80] + '...'
        console.print(f'  ⚙️  Tool: [bold yellow]{tool_name}[/bold yellow]({args_str})')

    def show_tool_result(self, tool_name: str, result: str):
        if len(result) > 300:
            result = result[:300] + '...'
        console.print(f'  ✅ [green]Result:[/green] [dim]{result}[/dim]')

    def show_error(self, error: str):
        console.print(f'\n  ❌ [bold red]Error:[/bold red] {error}')

    def show_done(self):
        if self._in_thinking:
            self._in_thinking = False
        console.print()


# ══════════════════════════════════════
# INTERACTIVE HELP (100% raw terminal)
# ══════════════════════════════════════

def show_help_with_search(commands: list):
    """Interactive help with live search."""
    with _RawMode():
        _drain_input()
        filter_text = ''
        prev_total = 0
        _hide_cursor()
        try:
            while True:
                if filter_text:
                    ft = filter_text.lower()
                    filtered = [c for c in commands
                                if ft in c['name'].lower() or ft in c['desc'].lower()]
                else:
                    filtered = list(commands)
                lines = []
                lines.append(
                    f'  {_c("bold")}{_c("cyan")}╔══════════════════════════════════════════╗{_c("reset")}')
                lines.append(
                    f'  {_c("bold")}{_c("cyan")}║{_c("reset")}       {_c("bold")}Help — Commands{_c("reset")}                    {_c("bold")}{_c("cyan")}║{_c("reset")}')
                if filter_text:
                    pad = 33 - len(filter_text)
                    if pad < 1: pad = 1
                    lines.append(
                        f'  {_c("bold")}{_c("cyan")}║{_c("reset")} {_c("yellow")}Filter: {filter_text}{" " * pad}{_c("bold")}{_c("cyan")}║{_c("reset")}')
                else:
                    lines.append(
                        f'  {_c("bold")}{_c("cyan")}║{_c("reset")}                                          {_c("bold")}{_c("cyan")}║{_c("reset")}')
                lines.append(
                    f'  {_c("bold")}{_c("cyan")}║{_c("reset")}                                          {_c("bold")}{_c("cyan")}║{_c("reset")}')
                for cmd in filtered:
                    name = cmd['name']
                    desc = cmd['desc']
                    max_desc = 39 - len(name)
                    if max_desc < 5: max_desc = 5
                    if len(desc) > max_desc:
                        desc = desc[:max_desc - 2] + '..'
                    lines.append(
                        f'  {_c("bold")}{_c("cyan")}║{_c("reset")}  {_c("bold")}{_c("green")}{name:<14}{_c("reset")} {_c("dim")}{desc}{_c("reset")}')
                if len(filtered) < len(commands):
                    hidden = len(commands) - len(filtered)
                    lines.append(
                        f'  {_c("bold")}{_c("cyan")}║{_c("reset")}  {_c("dim")}... {hidden} hidden (type to filter){_c("reset")}             {_c("bold")}{_c("cyan")}║{_c("reset")}')
                lines.append(
                    f'  {_c("bold")}{_c("cyan")}╠══════════════════════════════════════════╣{_c("reset")}')
                lines.append(
                    f'  {_c("bold")}{_c("cyan")}║{_c("reset")} {_c("dim")}Type to filter · ESC/q to close{_c("reset")}           {_c("bold")}{_c("cyan")}║{_c("reset")}')
                lines.append(
                    f'  {_c("bold")}{_c("cyan")}╚══════════════════════════════════════════╝{_c("reset")}')
                _clear_lines(prev_total)
                prev_total = _draw_and_pad(lines, prev_total)
                key = _read_key()
                if key in ('ESC', 'CTRL_C', 'q'):
                    break
                elif key == 'BACKSPACE':
                    filter_text = filter_text[:-1]
                elif key == 'ENTER':
                    break
                elif key in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'HOME', 'END',
                             'UNKNOWN', 'CTRL_P', 'TAB'):
                    pass
                elif len(key) >= 1 and key.isprintable():
                    filter_text += key
        finally:
            _clear_lines(prev_total)
            _show_cursor()


# ══════════════════════════════════════
# INTERACTIVE SELECTOR (generic, 100% raw)
# ══════════════════════════════════════

def _interactive_selector(title: str, items: list[dict],
                          id_key: str = 'id',
                          name_key: str = 'name',
                          current_value: str = None,
                          max_visible: int = 10,
                          extra_info: callable = None) -> str:
    """
    Generic interactive selector with arrow keys.
    Returns selected item[id_key] or current_value if cancelled.
    extra_info(item, is_selected) -> str for extra display per item.
    """
    if not items:
        console.print('[red]No items available.[/red]')
        return current_value

    with _RawMode():
        _drain_input()
        current_idx = 0
        for i, item in enumerate(items):
            if item.get(id_key) == current_value:
                current_idx = i
                break
        selected_idx = current_idx
        scroll_offset = max(0, selected_idx - 5)
        filter_text = ''
        prev_total = 0
        _hide_cursor()

        try:
            while True:
                if filter_text:
                    ft = filter_text.lower()
                    filtered = [item for item in items
                                if ft in item.get(id_key, '').lower() or
                                ft in item.get(name_key, '').lower()]
                else:
                    filtered = list(items)

                if not filtered:
                    lines = [
                        f'  {_c("red")}No matches for "{filter_text}"{_c("reset")}',
                        f'  {_c("dim")}Backspace to edit · ESC to cancel{_c("reset")}',
                    ]
                    _clear_lines(prev_total)
                    prev_total = _draw_and_pad(lines, prev_total)
                    key = _read_key()
                    if key == 'BACKSPACE':
                        filter_text = filter_text[:-1]
                    elif key in ('ESC', 'CTRL_C'):
                        return current_value
                    elif key in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'UNKNOWN',
                                 'HOME', 'END', 'CTRL_P', 'TAB'):
                        pass
                    elif len(key) >= 1 and key.isprintable():
                        filter_text += key
                    continue

                if selected_idx >= len(filtered):
                    selected_idx = len(filtered) - 1
                max_vis = min(max_visible, len(filtered))
                if selected_idx < scroll_offset:
                    scroll_offset = selected_idx
                if selected_idx >= scroll_offset + max_vis:
                    scroll_offset = selected_idx - max_vis + 1
                visible = filtered[scroll_offset:scroll_offset + max_vis]

                lines = []
                if filter_text:
                    lines.append(
                        f'  {_c("bold")}{_c("cyan")}{title}{_c("reset")}  '
                        f'{_c("yellow")}Filter: {filter_text}{_c("reset")}')
                else:
                    lines.append(
                        f'  {_c("bold")}{_c("cyan")}{title}{_c("reset")}  '
                        f'{_c("dim")}(↑↓ navigate · type filter · ENTER select · ESC cancel){_c("reset")}')
                lines.append('')

                for i, item in enumerate(visible):
                    real_idx = scroll_offset + i
                    is_sel = real_idx == selected_idx
                    is_cur = item.get(id_key) == current_value
                    marker = '❯' if is_sel else ' '
                    iid = item.get(id_key, '')
                    iname = item.get(name_key, iid)
                    if len(iid) > 55:
                        iid = iid[:52] + '...'

                    extra = ''
                    if extra_info:
                        extra = extra_info(item, is_sel)

                    if is_sel:
                        line = f'  {_c("bold")}{_c("cyan")}{marker} {iid}{_c("reset")}{extra}'
                        if is_cur:
                            line += f'  {_c("bold")}{_c("green")}← current{_c("reset")}'
                    else:
                        line = f'  {marker} {_c("dim")}{iid}{_c("reset")}{extra}'
                        if is_cur:
                            line += f'  {_c("green")}← current{_c("reset")}'
                    lines.append(line)

                lines.append('')
                if scroll_offset > 0:
                    lines.append(f'  {_c("dim")}↑ more above{_c("reset")}')
                if scroll_offset + max_vis < len(filtered):
                    lines.append(f'  {_c("dim")}↓ more below ({len(filtered)} total){_c("reset")}')
                else:
                    lines.append(f'  {_c("dim")}({len(filtered)} items){_c("reset")}')

                _clear_lines(prev_total)
                prev_total = _draw_and_pad(lines, prev_total)

                key = _read_key()
                if key == 'UP':
                    selected_idx = max(0, selected_idx - 1)
                elif key == 'DOWN':
                    selected_idx = min(len(filtered) - 1, selected_idx + 1)
                elif key == 'ENTER':
                    _clear_lines(prev_total)
                    return filtered[selected_idx].get(id_key, current_value)
                elif key in ('ESC', 'CTRL_C'):
                    _clear_lines(prev_total)
                    return current_value
                elif key == 'BACKSPACE':
                    filter_text = filter_text[:-1]
                elif key == 'HOME':
                    selected_idx = 0
                elif key == 'END':
                    selected_idx = len(filtered) - 1
                elif key in ('LEFT', 'RIGHT', 'UNKNOWN', 'CTRL_P', 'TAB'):
                    pass
                elif len(key) >= 1 and key.isprintable():
                    filter_text += key
        finally:
            _show_cursor()
    return current_value


def select_model_interactive(models: list[dict], current_model: str) -> str:
    """Interactive model selector."""
    def extra(item, sel):
        if item.get('free'):
            return f' {_c("green")}[FREE]{_c("reset")}'
        return ''
    return _interactive_selector(
        'Select Model', models, 'id', 'name',
        current_model, max_visible=10, extra_info=extra)


# ══════════════════════════════════════
# PROVIDER SELECTOR (100% raw terminal)
# ══════════════════════════════════════

def select_provider_interactive(providers: list[dict],
                                current_provider: str) -> str:
    """
    Interactive provider selector with arrow keys.
    Shows provider name, API key status, free tier info.
    """
    if not providers:
        console.print('[red]No providers available.[/red]')
        return current_provider

    # Build display items
    items = []
    for p in providers:
        items.append({
            'id': p['id'],
            'name': p['name'],
            'has_key': p.get('has_key', False),
            'has_free': p.get('has_free_models', False),
            'active': p.get('active', False),
            'supports_tools': p.get('supports_tools', False),
            'type': p.get('type', ''),
        })

    def extra(item, sel):
        tags = []
        if item['has_free']:
            tags.append(f'{_c("green")}FREE{_c("reset")}')
        if item['has_key']:
            tags.append(f'{_c("green")}key ✓{_c("reset")}')
        else:
            tags.append(f'{_c("red")}no key{_c("reset")}')
        if not item['supports_tools']:
            tags.append(f'{_c("yellow")}no tools{_c("reset")}')
        return f'  [{" | ".join(tags)}]'

    return _interactive_selector(
        'Select Provider', items, 'id', 'name',
        current_provider, max_visible=8, extra_info=extra)


# ══════════════════════════════════════
# COMMAND PALETTE (100% raw terminal)
# ══════════════════════════════════════

def command_palette(commands: list) -> str:
    """Interactive command palette (CTRL+P). Returns command name or ''."""
    with _RawMode():
        _drain_input()
        selected_idx = 0
        filter_text = ''
        prev_total = 0
        _hide_cursor()
        try:
            while True:
                if filter_text:
                    ft = filter_text.lower()
                    filtered = [c for c in commands
                                if ft in c['name'].lower() or ft in c['desc'].lower()]
                else:
                    filtered = list(commands)
                if not filtered:
                    lines = [
                        f'  {_c("red")}No commands match "{filter_text}"{_c("reset")}',
                        f'  {_c("dim")}Backspace to edit · ESC to close{_c("reset")}',
                    ]
                    _clear_lines(prev_total)
                    prev_total = _draw_and_pad(lines, prev_total)
                    key = _read_key()
                    if key == 'BACKSPACE':
                        filter_text = filter_text[:-1]
                    elif key in ('ESC', 'CTRL_C', 'CTRL_P'):
                        break
                    elif key in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'UNKNOWN',
                                 'HOME', 'END', 'TAB'):
                        pass
                    elif len(key) >= 1 and key.isprintable():
                        filter_text += key
                    continue
                if selected_idx >= len(filtered):
                    selected_idx = len(filtered) - 1
                max_visible = min(8, len(filtered))
                lines = []
                lines.append(
                    f'  {_c("bold")}{_c("cyan")}Command Palette{_c("reset")}  '
                    f'{_c("dim")}(↑↓ navigate · type filter · ENTER select · ESC close){_c("reset")}')
                if filter_text:
                    lines.append(f'  {_c("yellow")}Filter: {filter_text}{_c("reset")}')
                else:
                    lines.append('')
                for i in range(min(max_visible, len(filtered))):
                    idx = selected_idx + i
                    if idx >= len(filtered):
                        break
                    cmd = filtered[idx]
                    marker = '❯' if i == 0 else ' '
                    if i == 0:
                        lines.append(
                            f'  {_c("bold")}{_c("white")}{marker} '
                            f'{cmd["name"]:<14}{_c("reset")} '
                            f'{_c("dim")}{cmd["desc"]}{_c("reset")}')
                    else:
                        lines.append(
                            f'  {marker} {_c("dim")}{cmd["name"]:<14}  {cmd["desc"]}{_c("reset")}')
                _clear_lines(prev_total)
                prev_total = _draw_and_pad(lines, prev_total)
                key = _read_key()
                if key == 'UP':
                    selected_idx = max(0, selected_idx - 1)
                elif key == 'DOWN':
                    selected_idx = min(len(filtered) - 1, selected_idx + 1)
                elif key == 'ENTER':
                    _clear_lines(prev_total)
                    return filtered[selected_idx]['name']
                elif key in ('ESC', 'CTRL_C', 'CTRL_P'):
                    _clear_lines(prev_total)
                    return ''
        finally:
            _show_cursor()
    return ''


# ══════════════════════════════════════
# STATIC DISPLAYS (Rich)
# ══════════════════════════════════════

def show_all_tools(tool_list: list[dict]):
    """Display all tools in a Rich table."""
    table = Table(
        title=f'  🛠️  Available Tools ({len(tool_list)})',
        box=box.ROUNDED,
        show_lines=True,
        title_style='bold cyan',
        header_style='bold magenta',
    )
    table.add_column('#', style='dim', width=4)
    table.add_column('Tool Name', style='bold green')
    table.add_column('Description', style='white')
    for i, tool in enumerate(tool_list, 1):
        table.add_row(str(i), tool['name'], tool['description'])
    console.print()
    console.print(table)
    console.print()


def show_status(provider_name: str, model: str, thinking: bool, msg_count: int):
    """Compact status bar."""
    console.print(
        f'  [dim]{'─' * 40}[/dim]\n'
        f'  [magenta]{provider_name}[/magenta] · '
        f'[green]{model}[/green] · '
        f'{"[bold green]think[/bold green]" if thinking else "[red]think[/red]"} · '
        f'[dim]{msg_count} msgs[/dim]'
    )


def show_providers_table(providers: list[dict]):
    """Display all providers in a Rich table."""
    table = Table(
        title='  🌐  AI Providers',
        box=box.ROUNDED,
        show_lines=True,
        title_style='bold cyan',
        header_style='bold magenta',
    )
    table.add_column('Provider', style='bold white', width=20)
    table.add_column('Type', style='dim', width=18)
    table.add_column('API Key', width=18)
    table.add_column('Tools', width=6)
    table.add_column('Free', width=5)
    table.add_column('Status', width=10)
    for p in providers:
        key_status = ('[bold green]✓ Set[/bold green]' if p.get('has_key')
                      else '[red]✗ Missing[/red]')
        tools_str = ('[green]✓[/green]' if p.get('supports_tools')
                     else '[yellow]✗[/yellow]')
        free_str = ('[green]✓[/green]' if p.get('has_free_models')
                    else '[dim]✗[/dim]')
        active_str = ('[bold cyan]● Active[/bold cyan]' if p.get('active')
                      else '[dim]  [/dim]')
        table.add_row(
            p['name'],
            p.get('type', ''),
            key_status,
            tools_str,
            free_str,
            active_str,
        )
    console.print()
    console.print(table)
    console.print()
