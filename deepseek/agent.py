# DeepSeek CLI v7.7 — Smart Agentic Loop (FIXED + Enhanced UI + OCR + Rich MD)
# ═══════════════════════════════════════════════════════════════
# FIXED v5.5 — 8-Point Agent Improvement Plan:
#   1. Smart loop stop: max_rounds=12, max_same_tool=3
#   2. Pydantic validation (in toolkit.py)
#   3. No silent fails: retry JSON parse 2x, raise ValueError
#   4. Optional: message summarization (deferred)
#   5. Logging & Metrics: latency, tool_calls, errors -> JSON file (WAJIB)
#   6. safe_execute: threading-based timeout wrapper (signal.SIGALRM fails on Termux)
#   7. Anti-stuck: detect repeated content outputs, stop if no progress
#   8. Prompt control: system prompt instructions to stop when done
# ═══════════════════════════════════════════════════════════════

import json
import re
import time
import os
import itertools
import traceback
import sys
import threading
import typing as t
from datetime import datetime
from enum import Enum
from rich.console import Console
import select as _select
import tty
import termios

from .config import MAX_TOKENS, TEMPERATURE, MAX_TOOL_ROUNDS
SMART_MAX_ROUNDS = MAX_TOOL_ROUNDS
from .providers import BaseProvider
from .memory import Memory
from .toolkit import ToolRegistry
from .ui import StreamRenderer, ToolProcessingIndicator, confirm_action

console = Console()


class ErrorSeverity(Enum):
    TRANSIENT  = 'transient'   # Retryable (network, timeout, rate limit)
    PERMANENT  = 'permanent'   # Not retryable (bad args, unknown tool)
    CRITICAL   = 'critical'    # System-level (import fail, disk full)


class ToolResult:
    """Structured result from tool execution with error context."""
    def __init__(self, success: bool = True, data: str = '',
                 error: str = '', severity: ErrorSeverity = ErrorSeverity.PERMANENT,
                 trace: str = '', tool_name: str = ''):
        self.success = success
        self.data = data
        self.error = error
        self.severity = severity
        self.trace = trace
        self.tool_name = tool_name

    def to_str(self) -> str:
        if self.success:
            return self.data
        return f'[ERROR] {self.error}'

    @classmethod
    def ok(cls, data: str) -> 'ToolResult':
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str, severity: ErrorSeverity = ErrorSeverity.PERMANENT,
             trace: str = '', tool_name: str = '') -> 'ToolResult':
        return cls(success=False, error=error, severity=severity,
                   trace=trace, tool_name=tool_name)

    @classmethod
    def timeout(cls, seconds: int, tool_name: str = '') -> 'ToolResult':
        return cls(success=False,
                   error=f'Execution exceeded {seconds}s',
                   severity=ErrorSeverity.TRANSIENT,
                   tool_name=tool_name)

    @classmethod
    def unknown_tool(cls, name: str) -> 'ToolResult':
        return cls(success=False, error=f"Unknown tool '{name}'",
                   severity=ErrorSeverity.PERMANENT, tool_name=name)

    @classmethod
    def json_error(cls, tool_name: str, detail: str) -> 'ToolResult':
        return cls(success=False, error=f'Invalid arguments: {detail}',
                   severity=ErrorSeverity.PERMANENT, tool_name=tool_name)


TRANSIENT_KEYWORDS = [
    'connection', 'timeout', 'network', 'connect', 'refused', 'reset',
    'eof', 'name resolution', 'dns', 'ssl', 'certificate', 'handshake',
    'rate limit', 'too many requests', 429, 503, 502, 500,
    'service unavailable', 'bad gateway', 'internal server',
    'temporarily', 'try again', 'retry', ' throttl', 'quota',
    'zipimport', 'bad local file header',
]


def classify_error(error_text: str) -> ErrorSeverity:
    """Classify an error as transient or permanent based on keywords."""
    lower = error_text.lower()
    for kw in TRANSIENT_KEYWORDS:
        if isinstance(kw, str) and kw in lower:
            return ErrorSeverity.TRANSIENT
    return ErrorSeverity.PERMANENT

# ══════════════════════════════════════════════════
# SMART LOOP LIMITS (v5.5)
# ══════════════════════════════════════════════════
# Tool rounds are UNLIMITED (user request): the loop only ends when the model
# returns a final answer with no tool calls, or when anti-stuck safety triggers.
# Loop is truly unlimited — AI decides when to stop calling tools
TOOL_TIMEOUT_DEFAULT = 0    # 0 = no timeout, AI determines execution time

# ══════════════════════════════════════════════════
# VISIBLE THINKING (provider-agnostic)
# ══════════════════════════════════════════════════
# Many providers (e.g. Agnes AI / plain OpenAI-compatible endpoints) only return
# `content` and never emit a separate `reasoning` field, so there is no native
# thought process to display — and tests show such models ignore a "wrap your
# reasoning in <think> tags" instruction. So, when thinking is ON, we run a short
# REASONING PRE-PASS: one streaming call (no tools) whose only job is to surface a
# concise plan/thought process, which we render live as the dim "thinking" block
# before the real answer. A user-turn nudge + assistant prefill reliably forces a
# reasoning-only response even on models that ignore the system role.
REASONING_NUDGE = (
    "\n\n[INSTRUCTION: Before answering or calling any tool, first write ONLY your "
    "brief step-by-step reasoning/plan (2-6 short sentences) in the SAME language "
    "as my message above. Do NOT write the final answer or perform the task in "
    "this part — just your thought process.]"
)
REASONING_PREFILL = "Baik, berikut proses berpikir saya sebelum menjawab:\n"

# Kept as a harmless safety net: if a model DOES emit <think> tags inside content
# (some do), we still split them out so the tags never reach the stored answer.


class ThinkTagStreamParser:
    """Splits a streamed *content* flow into ('thinking', text) / ('content', text)
    segments based on <think> ... </think> tags. Robust to a tag being split
    across multiple chunks (a partial tag tail is held back until completed)."""

    OPEN = '<think>'
    CLOSE = '</think>'

    def __init__(self):
        self.in_think = False
        self._pending = ''  # held-back tail that might be the start of a tag

    @staticmethod
    def _partial_tail(data: str, tag: str) -> int:
        """Length of the longest suffix of `data` that is a prefix of `tag`."""
        maxk = min(len(tag) - 1, len(data))
        for k in range(maxk, 0, -1):
            if data.endswith(tag[:k]):
                return k
        return 0

    def feed(self, chunk: str):
        out = []
        data = self._pending + chunk
        self._pending = ''
        while data:
            tag = self.CLOSE if self.in_think else self.OPEN
            idx = data.find(tag)
            if idx != -1:
                before = data[:idx]
                if before:
                    out.append(('thinking' if self.in_think else 'content', before))
                self.in_think = not self.in_think
                data = data[idx + len(tag):]
                continue
            hold = self._partial_tail(data, tag)
            if hold:
                self._pending = data[len(data) - hold:]
                emit = data[:len(data) - hold]
            else:
                emit = data
            if emit:
                out.append(('thinking' if self.in_think else 'content', emit))
            data = ''
        return out

    def flush(self):
        out = []
        if self._pending:
            out.append(('thinking' if self.in_think else 'content', self._pending))
            self._pending = ''
        return out

# ══════════════════════════════════════════════════
# LOGGING CONFIG
# ══════════════════════════════════════════════════
LOG_DIR = os.path.join(os.path.expanduser('~'), '.deepseek-cli', 'logs')


class AgentMetrics:
    """Track and persist agent metrics for every chat turn."""

    def __init__(self):
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.total_tool_calls = 0
        self.total_errors = 0
        self.total_latencies = []
        self.tool_usage = {}       # tool_name -> count
        self.turn_history = []     # list of turn dicts
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        os.makedirs(LOG_DIR, exist_ok=True)

    def _log_file_path(self):
        return os.path.join(LOG_DIR, f'session_{self.session_id}.json')

    def record_turn(self, turn_data: dict):
        """Record a single agent turn and append to log file."""
        self.turn_history.append(turn_data)
        self.total_tool_calls += turn_data.get('tool_calls', 0)
        self.total_errors += turn_data.get('errors', 0)
        if turn_data.get('latency'):
            self.total_latencies.append(turn_data['latency'])
        for tool in turn_data.get('tools_used', []):
            self.tool_usage[tool] = self.tool_usage.get(tool, 0) + 1
        # Persist to JSON file immediately
        self._save_log()

    def _save_log(self):
        """Save current metrics to JSON log file."""
        try:
            log_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'total_turns': len(self.turn_history),
                'total_tool_calls': self.total_tool_calls,
                'total_errors': self.total_errors,
                'avg_latency': (sum(self.total_latencies) / len(self.total_latencies)) if self.total_latencies else 0,
                'tool_usage': self.tool_usage,
                'turns': self.turn_history[-50:],  # Keep last 50 turns in log
            }
            with open(self._log_file_path(), 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
        except Exception:
            pass  # Logging should never crash the agent

    def get_summary(self) -> dict:
        """Get a summary of current metrics."""
        return {
            'session_id': self.session_id,
            'total_turns': len(self.turn_history),
            'total_tool_calls': self.total_tool_calls,
            'total_errors': self.total_errors,
            'avg_latency': (sum(self.total_latencies) / len(self.total_latencies)) if self.total_latencies else 0,
            'tool_usage': self.tool_usage,
        }


# ══════════════════════════════════════════════════
# SAFE EXECUTE (Threading-based timeout)
# Works on Termux/Android where signal.SIGALRM fails
# ══════════════════════════════════════════════════

def safe_execute(func, args: dict, timeout: int = TOOL_TIMEOUT_DEFAULT,
                 tool_name: str = '', retries: int = 2) -> str:
    """
    Execute a tool function. No forced timeout - AI determines execution time.
    Auto-retries transient errors (network, rate limit, etc.).
    """
    last_result = ToolResult.fail('Unknown error', tool_name=tool_name)

    for attempt in range(retries + 1):
        result_container: t.Dict[str, t.Any] = {'result': None, 'error': None, 'done': False}

        def worker():
            try:
                result_container['result'] = func(args)
            except Exception as e:
                tb = traceback.format_exc()
                result_container['error'] = str(e)
                result_container['trace'] = tb
            finally:
                result_container['done'] = True

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join()  # No timeout - AI decides execution duration

        if not result_container['done']:
            last_result = ToolResult.fail('Execution failed unexpectedly', tool_name=tool_name)
        elif result_container['error'] is not None:
            error_text = result_container['error'][:500]
            severity = classify_error(error_text)
            trace = result_container.get('trace', '')[:300]
            last_result = ToolResult.fail(error_text, severity, trace, tool_name)
        else:
            return result_container['result']

        # Retry only transient errors
        if last_result.severity == ErrorSeverity.TRANSIENT and attempt < retries:
            time.sleep(1.0 * (attempt + 1))  # Backoff
            continue
        break

    return last_result.to_str()


# ══════════════════════════════════════════════════
# JSON PARSER (with retry)
# ══════════════════════════════════════════════════

def safe_parse_json(raw_str: str, max_retries: int = 2) -> dict:
    """
    Parse JSON string with retry. Raises ValueError on final failure.
    No more silent empty dict on parse failure!
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return json.loads(raw_str)
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries:
                # Try to fix common issues
                fixed = raw_str.strip()
                if not fixed.endswith('}'):
                    fixed += '}'
                if not fixed.startswith('{'):
                    fixed = '{' + fixed
                try:
                    raw_str = fixed
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"JSON parse failed after {max_retries + 1} attempts: {last_error}")


def sanitize_json_args(raw: str) -> str:
    """Ensure a tool-call arguments string is valid JSON.
    Handles single quotes, trailing commas, unquoted keys, None/null."""
    if not raw or not raw.strip():
        return '{}'
    s = raw.strip()
    # Already valid
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    # Replace single quotes with double quotes (but not inside strings)
    # Strategy: try to fix common patterns
    s = re.sub(r"'", '"', s)
    # Fix Python literals
    s = s.replace('None', 'null').replace('True', 'true').replace('False', 'false')
    # Remove trailing comma before }
    s = re.sub(r',\s*}', '}', s)
    # Wrap bare keys: {foo: "bar"} -> {"foo": "bar"}
    s = re.sub(r'\{(\s*)([a-zA-Z_]\w*)(\s*):', r'{\1"\2"\3:', s)
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    return '{}'


# ══════════════════════════════════════════════════
# TEXT-BASED TOOL CALL PARSER
# ══════════════════════════════════════

def parse_text_tool_calls(content: str, available_tools: dict) -> list:
    """
    Parse tool calls from model text output when the model doesn't use
    proper structured tool calling format.

    Detects patterns like:
      <function=name> <param=k> v </function>
      <function=name>({"key": "value"})
      browser_navigate(url="https://...")
      Let me call tool_name with {"arg": "val"}
      tool_name({"arg": "val"})
    """
    if not content or not available_tools:
        return [], content

    tool_calls = []
    cleaned = content

    # Pattern 1: <function=name> <param=key> value </function>
    pattern1 = re.compile(
        r'<function=(\w+)\s*>(.*?)</function>',
        re.DOTALL
    )
    for m in pattern1.finditer(content):
        tool_name = m.group(1)
        args_str = m.group(2).strip()
        if tool_name in available_tools:
            args = {}
            # Parse <param=key> value </param>
            param_pattern = re.compile(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', re.DOTALL)
            for pm in param_pattern.finditer(args_str):
                args[pm.group(1)] = pm.group(2).strip()
            # Also try JSON-like parsing
            if not args:
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    pass
            tool_calls.append({
                'id': f'text_{tool_name}_{len(tool_calls)}',
                'type': 'function',
                'function': {
                    'name': tool_name,
                    'arguments': json.dumps(args) if args else '{}'
                }
            })
            cleaned = cleaned.replace(m.group(0), '', 1)

    # Pattern 2: tool_name({"arg": "value"}) or tool_name({json})
    # Uses brace counting to find the matching ')' — handles nested JSON
    pattern2_head = re.compile(r'\b(\w+)\s*\(\s*(\{)')
    for m in pattern2_head.finditer(cleaned):
        tool_name = m.group(1)
        if tool_name not in available_tools:
            continue
        start = m.start()
        brace_count = 1
        i = m.end(2)  # position after the '{'
        while i < len(cleaned) and brace_count > 0:
            if cleaned[i] == '{':
                brace_count += 1
            elif cleaned[i] == '}':
                brace_count -= 1
            elif cleaned[i] == ')' and brace_count == 0:
                break
            i += 1
        if brace_count == 0 and i < len(cleaned) and cleaned[i] == ')':
            json_str = cleaned[m.end(2)-1:i+1]
            try:
                args = json.loads(json_str)
                tool_calls.append({
                    'id': f'text_{tool_name}_{len(tool_calls)}',
                    'type': 'function',
                    'function': {
                        'name': tool_name,
                        'arguments': json.dumps(args)
                    }
                })
                cleaned = cleaned[:m.start()] + cleaned[i+1:]
            except (json.JSONDecodeError, TypeError):
                pass

    # Pattern 3: tool_name(arg="value", arg2="value")
    pattern3 = re.compile(
        r'\b(\w+)\s*\(([^)]+)\)',
        re.DOTALL
    )
    for m in pattern3.finditer(cleaned):
        tool_name = m.group(1)
        if tool_name in available_tools:
            args_str = m.group(2).strip()
            args = {}
            # Parse keyword arguments: key="value" or key='value'
            kw_pattern = re.compile(r'(\w+)\s*=\s*["\']([^"\']*)["\']')
            for km in kw_pattern.finditer(args_str):
                args[km.group(1)] = km.group(2).strip()
            if args:
                tool_calls.append({
                    'id': f'text_{tool_name}_{len(tool_calls)}',
                    'type': 'function',
                    'function': {
                        'name': tool_name,
                        'arguments': json.dumps(args)
                    }
                })
                cleaned = cleaned.replace(m.group(0), '', 1)

    # Clean up residual text
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    return tool_calls, cleaned


# ══════════════════════════════════════
# AGENT
# ══════════════════════════════════════

class Agent:
    """Smart Agentic Loop: sends messages, handles tool calls, feeds results back.
    v5.5: Loop detection, anti-stuck, safe execute, metrics logging."""

    def __init__(self, memory: Memory, tools: ToolRegistry,
                 provider: BaseProvider, model: str,
                 thinking_visible: bool = True):
        self.memory = memory
        self.tools = tools
        self.provider = provider
        self.model = model
        self.thinking_visible = thinking_visible
        self.renderer = StreamRenderer(thinking_visible=thinking_visible)
        self._tool_functions = tools.get_openai_tools()
        # v5.5: Metrics
        self.metrics = AgentMetrics()
        
        # Double-ESC interrupt (v7.12) - background thread monitors stdin
        self._interrupted = False
        self._interrupt_monitor = None
        self._interrupt_monitor_running = False
        self._always_allow_tools = set()  # Tools user auto-approved this session
        self.created_files = []  # Files created during last chat() call
        
        try:
            from .planner import Planner
            self.planner = Planner(self.provider)
        except Exception:
            self.planner = None

    def _run_thinking_pass(self, user_message: str):
        """Reasoning pre-pass: one short, tool-less streaming call that surfaces
        the model's plan/thought process, rendered live as the dim thinking block
        before the real answer. Best-effort — any failure is swallowed so the
        answer still proceeds normally. Returns the reasoning text (may be '')."""
        # Build context from real history, nudging the latest user turn to reason
        # first, then prefill the assistant turn so even instruction-ignoring
        # models (e.g. Agnes) reliably produce reasoning-only output.
        history = [dict(m) for m in self.memory.get_messages()
                   if m.get('role') != 'system']
        if history and history[-1].get('role') == 'user':
            history[-1]['content'] = (history[-1].get('content') or '') + REASONING_NUDGE
        else:
            history.append({'role': 'user',
                            'content': user_message + REASONING_NUDGE})
        think_msgs = history + [{'role': 'assistant', 'content': REASONING_PREFILL}]

        reasoning = ''
        self.renderer.begin_stream('thinking…')
        try:
            for chunk in self.provider.chat_stream(
                    messages=think_msgs, model=self.model, tools=None,
                    max_tokens=400):
                ctype = chunk.get('type')
                cdata = chunk.get('data') or ''
                if ctype in ('content', 'thinking') and cdata:
                    self.renderer.stop_waiting()
                    reasoning += cdata
                    self.renderer.append_thinking(cdata)
                elif ctype == 'error':
                    break
        except Exception:
            pass
        finally:
            self.renderer.stop_waiting()
        # Close the dim block on its own line so the answer renders cleanly below.
        self.renderer._close_thinking_if_open()
        return reasoning.strip()

    def _cleanup_plan(self, success: bool = True):
        if not hasattr(self, 'active_plan') or not self.active_plan:
            return
        for step in self.active_plan.steps:
            if success:
                if step.status in ('pending', 'in_progress'):
                    step.status = 'done'
            else:
                if step.status == 'in_progress':
                    step.status = 'failed'
                elif step.status == 'pending':
                    step.status = 'skipped'



    def _display_git_status(self, files_changed: list):
        """Display git status with color-coded file changes."""
        if not files_changed:
            return

        console.print()
        console.print('  [bold cyan]─── File Changes ───[/bold cyan]')
        console.print()

        for change in files_changed:
            status = change.get('status', '?')
            path = change.get('path', '')
            lines = change.get('lines', 0)

            if status == 'modified':
                color = 'yellow'
                symbol = '~'
                display = f'  [bold {color}]{symbol}[/bold {color}] [cyan]{path}[/cyan]'
                if lines != 0:
                    display += f'  [dim]({lines} lines)[/dim]'
                console.print(display)

            elif status == 'deleted':
                color = 'red'
                symbol = '-'
                display = f'  [bold {color}]{symbol}[/bold {color}] [red]{path}[/red]'
                console.print(display)

            elif status == 'added':
                color = 'green'
                symbol = '+'
                display = f'  [bold {color}]{symbol}[/bold {color}] [green]{path}[/green]'
                if lines != 0:
                    display += f'  [dim]({lines} lines)[/dim]'
                console.print(display)

            elif status == 'renamed':
                color = 'magenta'
                symbol = '→'
                display = f'  [bold {color}]{symbol}[/bold {color}] [cyan]{path}[/cyan]'
                console.print(display)

            else:
                display = f'  [dim]?[/dim] [dim]{path}[/dim]'
                console.print(display)

        console.print()

    def _handle_connection_error(self, error_msg: str, max_retries: int = 15) -> bool:
        for attempt in range(1, max_retries + 1):
            wait = 2 ** attempt  # exponential backoff: 2s, 4s, 8s, 16s, 32s, ...
            console.print(f'  [bold yellow]\u21bb Retry {attempt}/{max_retries} in {wait}s\u2026[/bold yellow]')
            time.sleep(wait)
            try:
                if hasattr(self.provider, 'validate_key'):
                    ok, msg = self.provider.validate_key()
                    if ok:
                        console.print(f'  [bold green]\u2713 Reconnected[/bold green]')
                        return True
            except Exception:
                pass
        console.print(f'  [bold red]\u2717 Failed after {max_retries} retries[/bold red]')
        return False

    def _interrupt_monitor_worker(self):
        """Background thread: reads raw stdin for double-ESC sequence.
        Sets self._interrupted = True when detected.
        
        FIXED: Only triggers on actual consecutive ESC presses (< 0.15s),
        NOT on single ESC within 0.5s of history. This prevents accidental
        interrupts when user presses ESC for menu or other UI navigation.
        """
        fd = sys.stdin.fileno()
        old_flags = None
        try:
            old_flags = termios.tcgetattr(fd)
            new_flags = termios.tcgetattr(fd)
            new_flags[3] = new_flags[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, new_flags)

            while self._interrupt_monitor_running and not self._interrupted:
                ready, _, _ = _select.select([fd], [], [], 0.05)
                if ready:
                    key = os.read(fd, 1)
                    if key == b'\x1b':
                        # Check for second ESC immediately after first
                        ready2, _, _ = _select.select([fd], [], [], 0.15)
                        if ready2:
                            key2 = os.read(fd, 1)
                            if key2 == b'\x1b':
                                # Confirmed: Actual double-ESC received
                                self._interrupted = True
                                # Drain any remaining input
                                while True:
                                    r, _, _ = _select.select([fd], [], [], 0.0)
                                    if not r:
                                        break
                                    os.read(fd, 4096)
                                return
                        # Single ESC only - do NOT interrupt
                        # (User might be using it for menu navigation)
        except Exception:
            pass
        finally:
            if old_flags is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, old_flags)
                except Exception:
                    pass

    def _start_interrupt_monitor(self):
        if self._interrupt_monitor_running:
            return
        self._interrupted = False
        self._interrupt_monitor_running = True
        self._interrupt_monitor = threading.Thread(
            target=self._interrupt_monitor_worker,
            daemon=True
        )
        self._interrupt_monitor.start()

    def _stop_interrupt_monitor(self):
        self._interrupt_monitor_running = False
        # Join with timeout so we don't block forever
        if self._interrupt_monitor and self._interrupt_monitor.is_alive():
            self._interrupt_monitor.join(timeout=0.5)

    def _check_interrupt(self):
        """Check if user pressed ESC twice to interrupt streaming.
        Non-blocking — just returns the flag set by the monitor thread."""
        if self._interrupted:
            return True
        # Fallback: try reading stdin directly (covers the case where
        # monitor thread hasn't started yet)
        fd = sys.stdin.fileno()
        old_flags = None
        try:
            old_flags = termios.tcgetattr(fd)
            new_flags = termios.tcgetattr(fd)
            new_flags[3] = new_flags[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, new_flags)
            ready, _, _ = _select.select([fd], [], [], 0.0)
            if ready:
                key = os.read(fd, 1)
                if key == b'\x1b':
                    now = time.time()
                    ready2, _, _ = _select.select([fd], [], [], 0.0)
                    if ready2:
                        key2 = os.read(fd, 1)
                        if key2 == b'\x1b':
                            self._interrupted = True
                            while True:
                                r, _, _ = _select.select([fd], [], [], 0.0)
                                if not r:
                                    break
                                os.read(fd, 4096)
                            return True
                    if now - self._interrupt_last_time < 0.5:
                        self._interrupted = True
                        while True:
                            r, _, _ = _select.select([fd], [], [], 0.0)
                            if not r:
                                break
                            os.read(fd, 4096)
                        return True
                    self._interrupt_last_time = now
        except Exception:
            pass
        finally:
            if old_flags is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, old_flags)
                except Exception:
                    pass
        return False

    def _trigger_interrupt(self):
        """Mark the agent as interrupted and clean up."""
        self._interrupted = True
        self._stop_interrupt_monitor()
        self.renderer.stop_waiting()
        self.renderer._close_thinking_if_open()
    def chat(self, user_message: str) -> dict:
        """
        Process a user message through the SMART agentic loop.
        Returns {'content': str, 'tool_rounds': int, 'error': str|None,
                 'stopped_by': str|None, 'metrics': dict}
        """
        from .config import enforce_gist
        enforce_gist()
        
        try:
            res = self._chat_impl(user_message)
            return res
        finally:
            try:
                content = ""
                tools_used_list = []
                if hasattr(self, 'metrics') and self.metrics.turn_history:
                    last_turn = self.metrics.turn_history[-1]
                    content = last_turn.get('content_preview', '')
                    tools_used_list = last_turn.get('tools_used', [])
                
                input_est = len(user_message) // 3 + 1000
                output_est = len(content) // 3 if content else 100
                last_tool = tools_used_list[-1] if tools_used_list else "none"
                
                from .config import update_gist_usage
                import threading
                threading.Thread(
                    target=update_gist_usage,
                    args=(input_est, output_est, last_tool),
                    daemon=True
                ).start()
            except Exception:
                pass

    def _chat_impl(self, user_message: str) -> dict:
        self.memory.add_user(user_message)
        self.created_files.clear()

        # Inisialisasi dan jalankan Planner jika diperlukan
        self.active_plan = None
        self.memory.active_plan = None
        if hasattr(self, 'planner') and self.planner and self.planner.should_plan(user_message, self.memory.count()):
            self.renderer.begin_stream('planning task steps…')
            try:
                plan = self.planner.create_plan(user_message, self.memory.count())
                if plan and plan.steps and (len(plan.steps) > 1 or (plan.steps and plan.steps[0].description != f'Process: {user_message[:100]}')):
                    self.active_plan = plan
                    self.memory.active_plan = plan
                    console.print("\n  [bold cyan]Plan generated:[/bold cyan]")
                    for idx, step in enumerate(self.active_plan.steps):
                        priority = f" [high]" if step.priority == 'high' else ""
                        tool_hint = f" (tool: {step.tool_hint})" if step.tool_hint else ""
                        console.print(f"    [dim]{idx+1}.[/dim] [ ] {step.description}{tool_hint}{priority}")
                    console.print()
            except Exception:
                pass
            self.renderer.end_stream()

        full_content = ''
        thinking_text = ''
        tool_rounds = 0
        total_errors = 0
        tools_used = []
        start_time = time.time()

        # Visible thought process (provider-agnostic). Runs once per user turn,
        # before the answer/tool loop, so even content-only models (Agnes AI)
        # show their reasoning live in the dim thinking block. Kept separate from
        # the per-round thinking_text (it is already streamed to the screen here).
        if self.thinking_visible:
            self._run_thinking_pass(user_message)

        stopped_by = None

        # Start background ESC monitor so double-ESC works even during network I/O
        self._start_interrupt_monitor()

        # Always send tools
        send_tools = self._tool_functions if self.provider.supports_tools else None

        for round_num in itertools.count(0):
            if SMART_MAX_ROUNDS > 0 and round_num >= SMART_MAX_ROUNDS:
                break
            if self.active_plan:
                current_step = self.active_plan.get_next_pending()
                if current_step:
                    current_step.status = 'in_progress'
            messages = self.memory.get_messages()
            full_content = ''
            thinking_text = ''
            tool_calls_list = []
            has_error = False
            needs_retry = False
            self.renderer.reset_for_new_round()
            think_parser = ThinkTagStreamParser()
            self.renderer.begin_stream('thinking…')

            def _emit_content(text: str):
                """Route streamed content through the <think> parser so reasoning
                renders as the dim thinking block and only the answer is kept.
                Strips tool-call XML tags from displayed content so the raw
                <function=...> / <tool_call> ... </tool_call> markup never
                appears on screen — only the clean tool call via show_tool_call()."""
                nonlocal full_content, thinking_text
                for kind, seg in think_parser.feed(text):
                    if kind == 'thinking':
                        thinking_text += seg
                        self.renderer.append_thinking(seg)
                    else:
                        full_content += seg
                        # Strip inline tool-call XML so it doesn't pollute the UI
                        clean = re.sub(
                            r'</?function[^>]*>|</?tool[^>]*>|</?parameter[^>]*>|<parameter=\w+>'
                            r'|<invoke>|</invoke>|<result>|</result>|<action>|</action>',
                            '', seg
                        )
                        if clean:
                            self.renderer.append_content(clean)

            try:
                for chunk in self.provider.chat_stream(
                    messages=messages,
                    model=self.model,
                    tools=send_tools
                ):
                    # ── DOUBLE-ESC INTERRUPT CHECK ──
                    if self._check_interrupt():
                        console.print(f'\n  [bold yellow]  [INTERRUPTED] Agent stopped by user (double-ESC)[/bold yellow]')
                        self.renderer.show_done()
                        latency = time.time() - start_time
                        self.metrics.record_turn({
                            'user_message': user_message[:200],
                            'round': round_num,
                            'tool_rounds': tool_rounds,
                            'tool_calls': len(tool_calls_list),
                            'errors': total_errors,
                            'tools_used': tools_used[-10:],
                            'latency': round(latency, 2),
                            'stopped_by': 'user_interrupt',
                            'content_preview': full_content[:200] if full_content else thinking_text[:200],
                        })
                        self._cleanup_plan(success=False)
                        self._interrupted = False
                        self._stop_interrupt_monitor()
                        return {'content': full_content or thinking_text,
                                'tool_rounds': tool_rounds, 'error': None,
                                'stopped_by': 'user_interrupt', 'metrics': self.metrics.get_summary()}
                    
                    self.renderer.stop_waiting()
                    chunk_type = chunk['type']
                    chunk_data = chunk['data']

                    if chunk_type == 'thinking':
                        thinking_text += chunk_data
                        self.renderer.append_thinking(chunk_data)
                    elif chunk_type == 'content':
                        _emit_content(chunk_data)
                    elif chunk_type == 'tool_calls':
                        tool_calls_list = chunk_data
                    elif chunk_type == 'error':
                        error_text = str(chunk_data) if chunk_data else ''

                        if any(kw in error_text.lower() for kw in ['connection', 'timeout', 'network', 'connect', 'refused', 'reset']):
                            console.print(f'  [bold red]\u2717 {error_text}[/bold red]')
                            restored = self._handle_connection_error(error_text)
                            if restored:
                                needs_retry = True
                                break
                            else:
                                has_error = True
                                total_errors += 1
                                break
                        else:
                            self.renderer.show_error(error_text)
                            has_error = True
                            total_errors += 1
                            break
                    elif chunk_type == 'done':
                        pass
            except Exception as e:
                error_text = str(e)
                console.print(f'  [bold red]\u2717 Stream error: {error_text[:100]}[/bold red]')
                if any(kw in error_text.lower() for kw in ['connection', 'timeout', 'network', 'connect', 'refused', 'reset', 'read', 'write', 'eof']):
                    restored = self._handle_connection_error(error_text)
                    if restored:
                        needs_retry = True
                    else:
                        has_error = True
                        total_errors += 1
                else:
                    has_error = True
                    total_errors += 1
            finally:
                self.renderer.stop_waiting()

            # Flush any content held back by the <think> parser (partial tag tail)
            for kind, seg in think_parser.flush():
                if kind == 'thinking':
                    thinking_text += seg
                    self.renderer.append_thinking(seg)
                else:
                    full_content += seg
                    self.renderer.append_content(seg)

            # Retry round after successful reconnection — don't fall through
            # to empty-response handling with no streamed content.
            if needs_retry:
                console.print(f'  [dim]Retrying request after reconnection...[/dim]')
                continue

            if has_error:
                self.renderer.show_done()
                latency = time.time() - start_time
                self.metrics.record_turn({
                    'user_message': user_message[:200],
                    'round': round_num,
                    'tool_rounds': tool_rounds,
                    'tool_calls': 0,
                    'errors': total_errors,
                    'tools_used': tools_used[-10:],
                    'latency': round(latency, 2),
                    'stopped_by': 'stream_error',
                    'content_preview': full_content[:200] if full_content else thinking_text[:200],
                })
                self._cleanup_plan(success=False)
                self._stop_interrupt_monitor()
                return {'content': full_content or thinking_text,
                        'tool_rounds': tool_rounds, 'error': 'Stream error',
                        'stopped_by': 'stream_error', 'metrics': self.metrics.get_summary()}

            # ── FALLBACK: Parse text-based tool calls from content ──
            if not tool_calls_list and full_content.strip():
                text_calls, cleaned_content = parse_text_tool_calls(
                    full_content, self.tools.tools
                )
                if text_calls:
                    tool_calls_list = text_calls
                    full_content = cleaned_content
                    console.print(f'\n  [dim cyan](Detected {len(text_calls)} text-based tool call(s))[/dim cyan]')

            # Final safety: strip any remaining tool-call XML artifacts
            full_content = re.sub(
                r'</?function[^>]*>|</?tool[^>]*>|</?parameter[^>]*>|<parameter=\w+>'
                r'|<invoke>|</invoke>|<result>|</result>|<action>|</action>',
                '', full_content
            )

            # ── NO TOOL CALLS → Agent is done speaking ──
            if not tool_calls_list:
                # BUG FIX: If content is empty but thinking has text, use thinking as content
                # DeepSeek R1 and other reasoning models may send all text as reasoning_content
                # and leave the 'content' field empty, causing blank responses.
                display_content = full_content
                if not full_content.strip() and thinking_text.strip():
                    # Model sent reasoning-only — display thinking as the actual response
                    display_content = thinking_text.strip()
                    self.renderer.show_thinking_as_content(thinking_text)
                    console.print(f'  [dim yellow](Reasoning-only response — thinking shown as answer)[/dim yellow]')
                    console.print()
                elif not full_content.strip():
                    # Model returned completely empty — this is a real bug scenario
                    display_content = '(No response received from model. Try switching provider/model with /provider or /model)'
                    self.renderer.show_done()
                    console.print(f'  [bold yellow]Warning: Model returned an empty response![/bold yellow]')
                    console.print(f'  [dim]Possible fixes: Switch model (/model), switch provider (/provider), or check your API key (/key)[/dim]')
                    console.print()
                else:
                    # v7.2: Render final response as Rich Markdown (replaces raw streamed text)
                    self.renderer.render_final(full_content)

                self.memory.add_assistant(display_content)
                latency = time.time() - start_time
                self.metrics.record_turn({
                    'user_message': user_message[:200],
                    'round': round_num,
                    'tool_rounds': tool_rounds,
                    'tool_calls': 0,
                    'errors': total_errors,
                    'tools_used': tools_used[-10:],
                    'latency': round(latency, 2),
                    'stopped_by': 'natural' if tool_rounds > 0 else 'no_tools',
                    'content_preview': display_content[:200],
                })
                self._cleanup_plan(success=True)
                self._stop_interrupt_monitor()
                return {'content': display_content,
                        'tool_rounds': tool_rounds, 'error': None,
                        'stopped_by': 'natural', 'metrics': self.metrics.get_summary()}

            tool_rounds += 1
            self.renderer.show_done()

            # ── DOUBLE-ESC INTERRUPT CHECK before tool execution ──
            if self._check_interrupt():
                console.print(f'\n  [bold yellow]  [INTERRUPTED] Agent stopped by user (double-ESC)[/bold yellow]')
                self.renderer.show_done()
                latency = time.time() - start_time
                self.metrics.record_turn({
                    'user_message': user_message[:200],
                    'round': round_num,
                    'tool_rounds': tool_rounds,
                    'tool_calls': len(tool_calls_list),
                    'errors': total_errors,
                    'tools_used': tools_used[-10:],
                    'latency': round(latency, 2),
                    'stopped_by': 'user_interrupt',
                    'content_preview': full_content[:200] if full_content else thinking_text[:200],
                })
                self._cleanup_plan(success=False)
                self._interrupted = False
                self._stop_interrupt_monitor()
                return {'content': full_content or thinking_text,
                        'tool_rounds': tool_rounds, 'error': None,
                        'stopped_by': 'user_interrupt', 'metrics': self.metrics.get_summary()}

            # ── EXECUTE TOOL CALLS ──
            assistant_content = full_content or thinking_text
            memory_tool_calls = []
            for tc in tool_calls_list:
                fn = tc.get('function', {})
                memory_tool_calls.append({
                    'id': tc.get('id', ''),
                    'type': 'function',
                    'function': {
                        'name': fn.get('name', ''),
                        'arguments': sanitize_json_args(fn.get('arguments', '{}'))
                    }
                })
            self.memory.add_assistant_tool_calls(assistant_content, memory_tool_calls)

            _round_label = f'Tool Round {tool_rounds}'
            console.print(f'\n  [bold cyan]  {_round_label}[/bold cyan]')
            console.print()

            round_tool_count = 0
            for tc in tool_calls_list:
                fn = tc.get('function', {})
                tool_name = fn.get('name', 'unknown')
                tc_id = tc.get('id', '')
                raw_args = fn.get('arguments', '{}')

                # ── POINT 3: No silent JSON parse fails ──
                raw_args = sanitize_json_args(raw_args)
                try:
                    args = safe_parse_json(raw_args)
                except ValueError as e:
                    console.print(f'  [bold red]JSON parse error:[/bold red] {e}')
                    result = f"[ERROR] Invalid JSON arguments for {tool_name}: {e}"
                    total_errors += 1
                    self.renderer.show_tool_call(tool_name, {'raw': raw_args})
                    self.renderer.show_tool_result(tool_name, result)
                    self.memory.add_tool_result(tc_id, tool_name, result)
                    continue

                self.renderer.show_tool_call(tool_name, args)
                round_tool_count += 1
                tools_used.append(tool_name)

                # ── POINT 6: Safe execute with timeout + auto-retry ──
                if tool_name not in self.tools.tools:
                    result = ToolResult.unknown_tool(tool_name).to_str()
                    total_errors += 1
                    self.renderer.show_tool_result(tool_name, result)
                    self.memory.add_tool_result(tc_id, tool_name, result)
                    continue

                # ── User confirmation for dangerous tools ──
                _dangerous_tools = (
                    'write_file', 'edit_file', 'run_shell', 'run_code',
                    'create_pdf', 'create_docx', 'edit_docx', 'create_xlsx',
                    'edit_xlsx', 'create_pptx', 'edit_pptx', 'create_csv',
                    'edit_csv', 'todowrite'
                )
                if tool_name in _dangerous_tools and tool_name not in self._always_allow_tools:
                    # Pause the tool spinner animation so it doesn't overwrite the prompt
                    self.renderer.pause_tool_spinner()
                    
                    # Pause interrupt monitor to avoid stdin race condition
                    monitor_was_running = self._interrupt_monitor_running
                    if monitor_was_running:
                        self._stop_interrupt_monitor()
                    try:
                        verb = 'write' if any(x in tool_name for x in ('write', 'edit', 'create')) else 'execute'
                        ans = confirm_action(tool_name, args, verb=verb)
                    finally:
                        if monitor_was_running:
                            self._start_interrupt_monitor()
                    if ans == 'reject':
                        result = f"[Rejected by user] {tool_name} not executed."
                        console.print(f'  [dim]{result}[/dim]')
                        self.memory.add_tool_result(tc_id, tool_name, result)
                        continue
                    elif ans == 'always_allow':
                        self._always_allow_tools.add(tool_name)
                    
                    # Resume the tool spinner animation for actual execution
                    self.renderer.resume_tool_spinner()

                handler = self.tools.tools[tool_name]['handler']
                try:
                    result = safe_execute(handler, args,
                                          timeout=TOOL_TIMEOUT_DEFAULT,
                                          tool_name=tool_name)
                except Exception as e:
                    tb = traceback.format_exc()
                    result = ToolResult.fail(str(e)[:300],
                                             severity=ErrorSeverity.CRITICAL,
                                             trace=tb[:300],
                                             tool_name=tool_name).to_str()

                if result.startswith('[ERROR]') or result.startswith('[TIMEOUT]'):
                    total_errors += 1
                    if self.active_plan:
                        for step in self.active_plan.steps:
                            if step.status == 'in_progress':
                                step.status = 'failed'
                                try:
                                    self.active_plan = self.planner.refine_plan(
                                        self.active_plan, f"Tool {tool_name} failed: {result}"
                                    )
                                    self.memory.active_plan = self.active_plan
                                except Exception:
                                    pass
                                break
                else:
                    if self.active_plan:
                        for step in self.active_plan.steps:
                            if step.status == 'in_progress':
                                step.status = 'done'
                                break

                self.renderer.show_tool_result(tool_name, result)
                self.memory.add_tool_result(tc_id, tool_name, result)

                # Track created files for connector auto-send
                if tool_name == 'write_file' and result.startswith('Written '):
                    fpath = args.get('path', '')
                    if fpath:
                        self.created_files.append(fpath)

            console.print()

            # ── PROFESSIONAL ANIMATED INDICATOR: Processing tool results ──
            with ToolProcessingIndicator(round_num=tool_rounds,
                                         tools_count=round_tool_count):
                # Brief pause so the spinner is visible to the user
                time.sleep(0.15)

        # ── MAX ROUNDS REACHED ──
        console.print(f'\n  [bold yellow]  [MAX ROUNDS] Reached {SMART_MAX_ROUNDS} tool rounds — forcing stop[/bold yellow]')
        self.memory.add_assistant(full_content + "\n\n[System: Stopped — max tool rounds reached]")
        self.renderer.show_done()
        latency = time.time() - start_time
        self.metrics.record_turn({
            'user_message': user_message[:200],
            'round': SMART_MAX_ROUNDS,
            'tool_rounds': tool_rounds,
            'tool_calls': round_tool_count,
            'errors': total_errors,
            'tools_used': tools_used[-10:],
            'latency': round(latency, 2),
            'stopped_by': 'max_rounds',
            'content_preview': full_content[:200],
        })
        self._cleanup_plan(success=False)
        self._stop_interrupt_monitor()
        return {'content': full_content, 'tool_rounds': tool_rounds,
                'error': 'Max tool rounds reached', 'stopped_by': 'max_rounds',
                'metrics': self.metrics.get_summary()}

    def set_model(self, model: str):
        self.model = model

    def set_thinking(self, visible: bool):
        self.thinking_visible = visible
        self.renderer = StreamRenderer(thinking_visible=visible)

    def set_provider(self, provider: BaseProvider):
        """Switch to a different provider."""
        self.provider = provider

    def chat_with_files(self, user_message: str, files: list[dict]) -> dict:
        """
        Process a user message with file attachments from connectors (Telegram/Discord).
        Files are described as dicts: {'filename': str, 'url': str|None, 'path': str|None,
                                       'mime_type': str, 'size': int, 'caption': str|None}
        The agent will use tools to process the files and respond.

        Returns same dict as chat().
        """
        # Build enriched message with file info
        file_descriptions = []
        for f in files:
            desc_parts = []
            desc_parts.append(f"File: {f.get('filename', 'unknown')}")
            if f.get('mime_type'):
                desc_parts.append(f"Type: {f['mime_type']}")
            if f.get('size'):
                size_kb = f['size'] / 1024
                desc_parts.append(f"Size: {size_kb:.1f} KB")
            if f.get('url'):
                desc_parts.append(f"URL: {f['url']}")
            if f.get('path'):
                desc_parts.append(f"Local path: {f['path']}")
            if f.get('caption'):
                desc_parts.append(f"Caption: {f['caption']}")
            file_descriptions.append(' | '.join(desc_parts))

        # Create enriched user message
        if file_descriptions:
            enriched = (
                f"{user_message}\n\n"
                f"[FILE ATTACHMENTS from connector ({len(files)} file(s))]\n"
                + '\n'.join(file_descriptions)
                + "\n\nIMPORTANT: Use the appropriate tool to process these files "
                "(read_file for local paths, web_fetch for URLs, read_pdf for PDFs, "
                "read_docx for DOCX, image_view/image_info for images, ocr_read for OCR, "
                "video_info for videos). Analyze the file content and respond to the user's question."
            )
        else:
            enriched = user_message

        # If files have local paths, verify they exist and provide info
        file_paths = [f.get('path') for f in files if f.get('path')]
        file_urls = [f.get('url') for f in files if f.get('url')]

        # For file URLs, we can download them first if needed
        if file_urls:
            try:
                import httpx
                for f in files:
                    url = f.get('url')
                    filename = f.get('filename', '')
                    if not url or not filename:
                        continue
                    # Download to temp directory
                    save_dir = os.path.join(os.path.expanduser('~'), '.deepseek-cli', 'uploads')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, filename)
                    try:
                        with httpx.Client(timeout=30, follow_redirects=True) as client:
                            r = client.get(url)
                            if r.status_code == 200:
                                with open(save_path, 'wb') as out_f:
                                    out_f.write(r.content)
                                f['path'] = save_path
                                file_paths.append(save_path)
                    except Exception:
                        pass
            except Exception:
                pass

        return self.chat(enriched)


def safe_tool_call(func, *args, **kwargs):
    """Execute with error handling (lightweight wrapper for quick calls)."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        sev = classify_error(str(e))
        return ToolResult.fail(str(e)[:200], severity=sev, trace=tb[:200]).to_str()
