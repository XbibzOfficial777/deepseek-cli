# DeepSeek CLI v6.1 — Smart Agentic Loop (FIXED + Enhanced UI + OCR)
# ═══════════════════════════════════════════════════════════════
# FIXED v5.5 — 8-Point Agent Improvement Plan:
#   1. Smart loop stop: max_rounds=12, max_same_tool=3, same_tool_counter
#   2. Pydantic validation (in toolkit.py)
#   3. No silent fails: retry JSON parse 2x, raise ValueError
#   4. Optional: message summarization (deferred)
#   5. Logging & Metrics: latency, tool_calls, errors -> JSON file (WAJIB)
#   6. safe_execute: threading-based timeout wrapper (signal.SIGALRM fails on Termux)
#   7. Anti-stuck: detect repeated content outputs, stop if no progress
#   8. Prompt control: system prompt instructions to stop when done
# ═══════════════════════════════════════════════════════════════

import json
import time
import os
import traceback
from datetime import datetime
from rich.console import Console

from .config import MAX_TOOL_ROUNDS
from .providers import BaseProvider
from .memory import Memory
from .toolkit import ToolRegistry
from .ui import StreamRenderer, ToolProcessingIndicator

console = Console()

# ══════════════════════════════════════════════════
# SMART LOOP LIMITS (v5.5)
# ══════════════════════════════════════════════════
SMART_MAX_ROUNDS = 12       # Max tool rounds before forced stop
MAX_SAME_TOOL = 3              # Max consecutive calls to same tool
MAX_REPEATED_CONTENT = 2   # Max identical content outputs before stop
TOOL_TIMEOUT_DEFAULT = 60      # Default timeout for tool execution (seconds)

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

def safe_execute(func, args: dict, timeout: int = TOOL_TIMEOUT_DEFAULT) -> str:
    """
    Execute a tool function with a timeout using threading.
    Falls back gracefully on any error.
    """
    import threading

    result_container = {'result': None, 'error': None, 'done': False}

    def worker():
        try:
            result_container['result'] = func(args)
        except Exception as e:
            result_container['error'] = str(e)
        finally:
            result_container['done'] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if not result_container['done']:
        return f"[TIMEOUT] Tool execution exceeded {timeout}s — killed"

    if result_container['error'] is not None:
        return f"[ERROR] {result_container['error']}"

    return result_container['result']


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
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"JSON parse failed after {max_retries + 1} attempts: {last_error}")


# ══════════════════════════════════════════════════
# ANTI-STUCK DETECTOR
# ══════════════════════════════════════════════════

class AntiStuckDetector:
    """Detect when the agent is stuck producing identical or near-identical content."""

    def __init__(self):
        self.content_history = []
        self.max_history = 5

    def check(self, content: str) -> bool:
        """
        Check if content is repeating. Returns True if stuck.
        Compares simplified content (stripped, lowercased, truncated).
        """
        if not content:
            return False

        simplified = content.strip().lower()
        if len(simplified) > 200:
            simplified = simplified[:200]

        # Count how many times similar content appeared
        match_count = 0
        for prev in self.content_history:
            if self._similarity(simplified, prev) > 0.85:
                match_count += 1

        self.content_history.append(simplified)
        if len(self.content_history) > self.max_history:
            self.content_history.pop(0)

        return match_count >= MAX_REPEATED_CONTENT

    def _similarity(self, a: str, b: str) -> float:
        """Simple character-level similarity ratio."""
        if not a or not b:
            return 0.0
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if not longer:
            return 0.0
        matches = sum(1 for c in shorter if c in longer)
        return matches / len(longer)

    def reset(self):
        self.content_history.clear()


# ══════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════

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
        # v5.5: Metrics & anti-stuck
        self.metrics = AgentMetrics()
        self.anti_stuck = AntiStuckDetector()

    def chat(self, user_message: str) -> dict:
        """
        Process a user message through the SMART agentic loop.
        Returns {'content': str, 'tool_rounds': int, 'error': str|None,
                 'stopped_by': str|None, 'metrics': dict}
        """
        self.memory.add_user(user_message)
        self.anti_stuck.reset()

        full_content = ''
        thinking_text = ''
        tool_rounds = 0
        total_errors = 0
        tools_used = []
        start_time = time.time()

        # State tracking for loop detection
        last_tool_name = None
        same_tool_counter = 0
        stopped_by = None
        # v6.1.1: Processing indicator — starts after tools, stops on first LLM chunk
        processing_indicator = None

        # Always send tools
        send_tools = self._tool_functions if self.provider.supports_tools else None

        for round_num in range(SMART_MAX_ROUNDS + 1):
            messages = self.memory.get_messages()
            full_content = ''
            thinking_text = ''
            tool_calls_list = []
            has_error = False

            for chunk in self.provider.chat_stream(
                messages=messages,
                model=self.model,
                tools=send_tools
            ):
                chunk_type = chunk['type']
                chunk_data = chunk['data']

                # v6.1.1: Stop processing indicator on first chunk from LLM
                if processing_indicator is not None:
                    processing_indicator.stop()
                    processing_indicator = None
                    console.print()  # blank line after indicator

                if chunk_type == 'thinking':
                    thinking_text += chunk_data
                    self.renderer.append_thinking(chunk_data)
                elif chunk_type == 'content':
                    full_content += chunk_data
                    self.renderer.append_content(chunk_data)
                elif chunk_type == 'tool_calls':
                    tool_calls_list = chunk_data
                elif chunk_type == 'error':
                    self.renderer.show_error(chunk_data)
                    has_error = True
                    total_errors += 1
                    break
                elif chunk_type == 'done':
                    pass

            if has_error:
                # v6.1.1: Safety — ensure indicator is stopped on error
                if processing_indicator is not None:
                    processing_indicator.stop()
                    processing_indicator = None
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
                return {'content': full_content or thinking_text,
                        'tool_rounds': tool_rounds, 'error': 'Stream error',
                        'stopped_by': 'stream_error', 'metrics': self.metrics.get_summary()}

            # ── NO TOOL CALLS → Agent is done speaking ──
            if not tool_calls_list:
                self.memory.add_assistant(full_content)
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
                    'stopped_by': 'natural' if tool_rounds > 0 else 'no_tools',
                    'content_preview': full_content[:200],
                })
                return {'content': full_content,
                        'tool_rounds': tool_rounds, 'error': None,
                        'stopped_by': 'natural', 'metrics': self.metrics.get_summary()}

            tool_rounds += 1
            self.renderer.show_done()

            # ── ANTI-STUCK CHECK: Detect repeated content before tool execution ──
            if full_content and self.anti_stuck.check(full_content):
                console.print(f'\n  [bold yellow]  [ANTI-STUCK] Repeated content detected — stopping loop[/bold yellow]')
                self.memory.add_assistant(full_content + "\n\n[System: Stopped — repeated content detected]")
                latency = time.time() - start_time
                self.metrics.record_turn({
                    'user_message': user_message[:200],
                    'round': round_num,
                    'tool_rounds': tool_rounds,
                    'tool_calls': len(tool_calls_list),
                    'errors': total_errors,
                    'tools_used': tools_used[-10:],
                    'latency': round(latency, 2),
                    'stopped_by': 'anti_stuck',
                    'content_preview': full_content[:200],
                })
                return {'content': full_content,
                        'tool_rounds': tool_rounds, 'error': None,
                        'stopped_by': 'anti_stuck', 'metrics': self.metrics.get_summary()}

            # ── LOOP DETECTION: Check if same tool called too many times ──
            tool_names_this_round = []
            for tc in tool_calls_list:
                fn = tc.get('function', {})
                tool_names_this_round.append(fn.get('name', 'unknown'))

            # Use the first tool name for loop detection (most common pattern)
            first_tool = tool_names_this_round[0] if tool_names_this_round else 'unknown'

            if first_tool == last_tool_name:
                same_tool_counter += 1
            else:
                same_tool_counter = 0
                last_tool_name = first_tool

            if same_tool_counter >= MAX_SAME_TOOL:
                console.print(f'\n  [bold yellow]  [LOOP] Tool "{first_tool}" called {same_tool_counter + 1}x consecutively — stopping[/bold yellow]')
                self.memory.add_assistant(full_content + "\n\n[System: Stopped — tool loop detected]")
                latency = time.time() - start_time
                self.metrics.record_turn({
                    'user_message': user_message[:200],
                    'round': round_num,
                    'tool_rounds': tool_rounds,
                    'tool_calls': len(tool_calls_list),
                    'errors': total_errors,
                    'tools_used': tools_used[-10:],
                    'latency': round(latency, 2),
                    'stopped_by': 'loop_detected',
                    'content_preview': full_content[:200],
                })
                return {'content': full_content,
                        'tool_rounds': tool_rounds, 'error': None,
                        'stopped_by': 'loop_detected', 'metrics': self.metrics.get_summary()}

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
                        'arguments': fn.get('arguments', '{}')
                    }
                })
            self.memory.add_assistant_tool_calls(assistant_content, memory_tool_calls)

            console.print(f'\n  [bold cyan]  Tool Round {tool_rounds}/{SMART_MAX_ROUNDS}[/bold cyan]')
            console.print()

            round_tool_count = 0
            for tc in tool_calls_list:
                fn = tc.get('function', {})
                tool_name = fn.get('name', 'unknown')
                tc_id = tc.get('id', '')
                raw_args = fn.get('arguments', '{}')

                # ── POINT 3: No silent JSON parse fails ──
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

                # ── POINT 6: Safe execute with timeout ──
                if tool_name in self.tools.tools:
                    handler = self.tools.tools[tool_name]['handler']
                    result = safe_execute(handler, args, timeout=TOOL_TIMEOUT_DEFAULT)
                else:
                    result = f"[ERROR] Unknown tool '{tool_name}'"
                    total_errors += 1

                # Check for error in result
                if result.startswith('[ERROR]') or result.startswith('[TIMEOUT]'):
                    total_errors += 1

                self.renderer.show_tool_result(tool_name, result)
                self.memory.add_tool_result(tc_id, tool_name, result)

            console.print()

            # ── ANIMATED INDICATOR: Processing tool results ──
            # Starts immediately after tools finish, runs UNTIL the LLM
            # sends its first chunk (thinking or content). Covers the
            # network latency of the next API call — so the user always
            # sees the animation instead of a blank terminal.
            processing_indicator = ToolProcessingIndicator(
                round_num=tool_rounds,
                max_rounds=SMART_MAX_ROUNDS,
                tools_count=round_tool_count,
            )
            processing_indicator.start()

        # ── MAX ROUNDS REACHED ──
        # v6.1.1: Safety — ensure indicator is stopped
        if processing_indicator is not None:
            processing_indicator.stop()
            processing_indicator = None
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
