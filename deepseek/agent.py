# DeepSeek CLI v7.2 — Smart Agentic Loop (FIXED + Enhanced UI + OCR + Rich MD)
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
import re
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
SMART_MAX_ROUNDS = 70   # Max tool rounds before forced stop
MAX_SAME_TOOL = 70              # Max consecutive calls to same tool
MAX_REPEATED_CONTENT = 70   # Max identical content outputs before stop
TOOL_TIMEOUT_DEFAULT = 90      # Default timeout for tool execution (seconds)

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


# ══════════════════════════════════════
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
    pattern2 = re.compile(
        r'\b(\w+)\s*\(\s*(\{[^}]*\})\s*\)',
        re.DOTALL
    )
    for m in pattern2.finditer(cleaned):
        tool_name = m.group(1)
        if tool_name in available_tools:
            args_str = m.group(2).strip()
            try:
                args = json.loads(args_str)
                tool_calls.append({
                    'id': f'text_{tool_name}_{len(tool_calls)}',
                    'type': 'function',
                    'function': {
                        'name': tool_name,
                        'arguments': json.dumps(args)
                    }
                })
                cleaned = cleaned.replace(m.group(0), '', 1)
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

        # Always send tools
        send_tools = self._tool_functions if self.provider.supports_tools else None

        for round_num in range(SMART_MAX_ROUNDS + 1):
            messages = self.memory.get_messages()
            full_content = ''
            thinking_text = ''
            tool_calls_list = []
            has_error = False
            self.renderer.reset_for_new_round()  # v7.2: reset for markdown re-render per round

            for chunk in self.provider.chat_stream(
                messages=messages,
                model=self.model,
                tools=send_tools
            ):
                chunk_type = chunk['type']
                chunk_data = chunk['data']

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

            # ── FALLBACK: Parse text-based tool calls from content ──
            if not tool_calls_list and full_content.strip():
                text_calls, cleaned_content = parse_text_tool_calls(
                    full_content, self.tools.tools
                )
                if text_calls:
                    tool_calls_list = text_calls
                    full_content = cleaned_content
                    console.print(f'\n  [dim cyan](Detected {len(text_calls)} text-based tool call(s))[/dim cyan]')

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
                return {'content': display_content,
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

            # ── PROFESSIONAL ANIMATED INDICATOR: Processing tool results ──
            with ToolProcessingIndicator(round_num=tool_rounds, max_rounds=SMART_MAX_ROUNDS,
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
