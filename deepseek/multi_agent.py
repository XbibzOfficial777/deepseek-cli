# DeepSeek CLI v7.7 — Multi-Agent System
# Agent delegation, specialized profiles, and concurrent execution

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import cfg
from .memory import Memory
from .toolkit import ToolRegistry
from .providers import create_provider


# ══════════════════════════════════════
# AGENT PROFILES
# ══════════════════════════════════════

AGENT_PROFILES = {
    'general': {
        'name': 'General Assistant',
        'emoji': '',
        'description': 'All-purpose assistant with full tool access',
        'system_prompt_extra': '',
    },
    'coder': {
        'name': 'Code Specialist',
        'emoji': '',
        'description': 'Specialized for programming, debugging, and code review',
        'system_prompt_extra': (
            'You are a Code Specialist.\n'
            'Focus on writing clean, efficient, well-structured code.\n'
            'Always explain your reasoning before writing code.\n'
            'Prefer robust solutions over quick hacks.\n'
            'When debugging, be methodical: isolate the problem first, then fix.\n'
            'Use run_code and run_shell tools to test your solutions.\n'
        ),
    },
    'researcher': {
        'name': 'Research Agent',
        'emoji': '',
        'description': 'Focused on web search, fact-finding, and research',
        'system_prompt_extra': (
            'You are a Research Agent.\n'
            'Your primary role is to find accurate, up-to-date information.\n'
            'Always use live_search for current events and recent information.\n'
            'Cross-reference information from multiple sources when possible.\n'
            'Cite your sources and indicate confidence levels.\n'
            'Distinguish between facts, opinions, and speculation.\n'
        ),
    },
    'filesystem': {
        'name': 'File Operations Agent',
        'emoji': '',
        'description': 'Specialized for file management and data processing',
        'system_prompt_extra': (
            'You are a File Operations Agent.\n'
            'Your specialty is managing files, directories, and data.\n'
            'Be careful with destructive operations (delete, overwrite).\n'
            'Always verify file paths before operations.\n'
            'Prefer reading before modifying existing files.\n'
            'Use file tools efficiently: batch operations when possible.\n'
        ),
    },
    'reasoner': {
        'name': 'Deep Reasoner',
        'emoji': '',
        'description': 'Step-by-step analytical reasoning and problem solving',
        'system_prompt_extra': (
            'You are a Deep Reasoner.\n'
            'Break down complex problems into step-by-step reasoning.\n'
            'Use structured thinking: define the problem, list constraints,\n'
            'explore approaches, evaluate trade-offs, then conclude.\n'
            'Show your work clearly at each step.\n'
            'Consider edge cases and assumptions explicitly.\n'
            'When uncertain, acknowledge limitations rather than guessing.\n'
        ),
    },
}


def get_profile_names() -> list[str]:
    return list(AGENT_PROFILES.keys())


def get_profile_info(profile_id: str) -> dict:
    return AGENT_PROFILES.get(profile_id, AGENT_PROFILES['general'])


# ══════════════════════════════════════
# AGENT WORKER
# ══════════════════════════════════════

class AgentWorker:
    """
    A lightweight agent that runs independently with its own memory.
    Used for delegation and concurrent task execution.
    """

    def __init__(self, profile_id: str, task: str, tools: ToolRegistry,
                 provider_id: str = None, model: str = None):
        self.profile_id = profile_id
        self.profile = AGENT_PROFILES.get(profile_id, AGENT_PROFILES['general'])
        self.task = task
        self.tools = tools

        # Create isolated memory
        self.memory = Memory()
        # Inject profile-specific system prompt
        extra = self.profile.get('system_prompt_extra', '')
        if extra:
            self.memory.system_prompt += '\n' + extra
            self.memory.messages[0]['content'] = self.memory.system_prompt

        # Setup provider
        pid = provider_id or cfg.active_provider
        provider_config = cfg.get_provider_config(pid)
        api_key = cfg.get_api_key(pid)
        self.provider = create_provider(pid, provider_config, api_key)
        self.model = model or cfg.get_provider_model(pid)

        self.result: str = ''
        self.error: str = ''
        self.live_output: str = ''

    def run(self) -> str:
        """Execute the task and return result."""
        try:
            self.memory.add_user(self.task)

            full_content = ''
            send_tools = self.tools.get_openai_tools() if self.provider.supports_tools else None

            # Full loop: call LLM, execute tools, feed results back to LLM
            max_rounds = 6
            for round_num in range(max_rounds):
                messages = self.memory.get_messages()
                round_content = ''
                tool_calls_list = []

                for chunk in self.provider.chat_stream(
                    messages, self.model, tools=send_tools
                ):
                    if chunk['type'] == 'content':
                        full_content += chunk['data']
                        round_content += chunk['data']
                        self.live_output = full_content
                    elif chunk['type'] == 'tool_calls':
                        tool_calls_list = chunk.get('data', [])
                    elif chunk['type'] == 'error':
                        err_text = f'\n[Error: {chunk["data"]}]\n'
                        full_content += err_text
                        self.live_output = full_content

                if tool_calls_list:
                    # Format tool calls for memory
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
                    self.memory.add_assistant_tool_calls(round_content, memory_tool_calls)

                    for tc in tool_calls_list:
                        fn = tc.get('function', {})
                        tool_name = fn.get('name', '')
                        raw_args = fn.get('arguments', '{}')
                        
                        self.live_output += f"\n  [Call Tool] {tool_name} with args: {raw_args}\n"
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            args = {}
                        
                        result = self.tools.execute(tool_name, args)
                        if result is None:
                            result = '[ERROR] Tool execution failed'
                        
                        self.live_output += f"  [Tool Result] {tool_name} returned: {str(result)[:200]}...\n"
                        
                        tool_call_id = tc.get('id', f'tc_{round_num}')
                        self.memory.add_tool_result(tool_call_id, tool_name, result)
                else:
                    self.memory.add_assistant(round_content)
                    break

            self.result = full_content
            self.live_output = full_content
            return full_content

        except Exception as e:
            self.error = str(e)
            self.live_output += f"\n[ERROR] {e}\n"
            return f'[ERROR] {e}'

class MultiAgentManager:
    """
    Manages agent profiles, delegation, and concurrent execution.
    Supports both blocking and background (non-blocking) modes.
    """

    def __init__(self):
        self.active_profile = 'general'
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.history = []
        self.running_tasks: dict[str, dict] = {}  # profile_id -> {worker, future, output, status}
        import threading
        self._checker_thread = threading.Thread(target=self._global_checker, daemon=True)
        self._checker_thread.start()

    def _global_checker(self):
        import time
        while True:
            try:
                for pid, info in list(self.running_tasks.items()):
                    if info.get('status') in ('done', 'error'):
                        continue
                    fut = info.get('future')
                    if fut:
                        if fut.done():
                            try:
                                res = fut.result()
                                info['output'] = res
                                info['status'] = 'done'
                            except Exception as e:
                                info['output'] = f"[ERROR] {e}"
                                info['status'] = 'error'
                        else:
                            w = info.get('worker')
                            if w and hasattr(w, 'live_output') and w.live_output:
                                info['output'] = w.live_output
            except Exception:
                pass
            time.sleep(0.3)

    def set_profile(self, profile_id: str) -> bool:
        if profile_id in AGENT_PROFILES:
            self.active_profile = profile_id
            return True
        return False

    def get_system_extra(self) -> str:
        profile = AGENT_PROFILES.get(self.active_profile, AGENT_PROFILES['general'])
        return profile.get('system_prompt_extra', '')

    def delegate(self, profile_id: str, task: str,
                 tools: ToolRegistry, provider_id: str = None,
                 model: str = None, timeout: int = 120) -> str:
        """Delegate a task to a specialized agent (blocking/interactive progress)."""
        import sys
        import os
        import select
        import termios
        import time
        from rich.console import Console

        console = Console()
        worker = AgentWorker(profile_id, task, tools, provider_id, model)
        self.history.append({'profile': profile_id, 'task': task, 'worker': worker})

        console.print(f"\n[bold yellow]┌── Sub-agent delegation ──────────────────────────[/bold yellow]")
        console.print(f"[bold yellow]│[/bold yellow] Profile: [cyan]{profile_id}[/cyan]")
        console.print(f"[bold yellow]│[/bold yellow] Task: [dim]{task}[/dim]")
        console.print(f"[bold yellow]└──────────────────────────────────────────────────[/bold yellow]")
        console.print("Choose action:")
        console.print("  [1] View Progress (default)")
        console.print("  [0] Run in Background (detach)")

        fd = sys.stdin.fileno()
        old_flags = None
        choice = '1'
        try:
            old_flags = termios.tcgetattr(fd)
            new_flags = termios.tcgetattr(fd)
            new_flags[3] = new_flags[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, new_flags)

            console.print("Your choice [0/1, timeout 5s]: ", end="")
            sys.stdout.flush()
            start_time = time.time()
            while time.time() - start_time < 5.0:
                ready, _, _ = select.select([fd], [], [], 0.1)
                if ready:
                    key = os.read(fd, 1)
                    if key in (b'1', b'\n', b'\r'):
                        choice = '1'
                        console.print("1 (View Progress)")
                        break
                    elif key == b'0':
                        choice = '0'
                        console.print("0 (Run in Background)")
                        break
        except Exception:
            pass
        finally:
            if old_flags is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, old_flags)
                except Exception:
                    pass

        # Start worker execution in a thread
        future = self._executor.submit(worker.run)
        self.running_tasks[profile_id] = {
            'worker': worker,
            'future': future,
            'status': 'running',
            'output': '',
        }

        if choice == '0':
            console.print(f"\n[bold green][INFO] Sub-agent delegated and running in background.[/bold green]")
            return f"[INFO] Agent {profile_id} delegated and running in background."

        # Choice was 1 (View Progress)
        console.print("\n[bold cyan]── Live Output (Press '0' to detach/exit to main chat) ──[/bold cyan]\n")
        
        last_len = 0
        detached = False
        try:
            old_flags = termios.tcgetattr(fd)
            new_flags = termios.tcgetattr(fd)
            new_flags[3] = new_flags[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, new_flags)

            while not future.done():
                live = getattr(worker, 'live_output', '')
                if len(live) > last_len:
                    new_chunk = live[last_len:]
                    sys.stdout.write(new_chunk)
                    sys.stdout.flush()
                    last_len = len(live)

                # Check if user wants to detach by pressing '0'
                ready, _, _ = select.select([fd], [], [], 0.05)
                if ready:
                    key = os.read(fd, 1)
                    if key == b'0':
                        detached = True
                        break
        except Exception:
            # Fallback loop
            while not future.done():
                live = getattr(worker, 'live_output', '')
                if len(live) > last_len:
                    new_chunk = live[last_len:]
                    sys.stdout.write(new_chunk)
                    sys.stdout.flush()
                    last_len = len(live)
                time.sleep(0.1)
        finally:
            if old_flags is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, old_flags)
                except Exception:
                    pass

        if detached:
            console.print("\n[bold yellow]\n[INFO] Detached from sub-agent view. Sub-agent continues running in background.[/bold yellow]\n")
            return f"[INFO] Agent {profile_id} is running in background. Task: {task}"

        # Otherwise wait for complete output
        try:
            result = future.result()
            live = getattr(worker, 'live_output', '')
            if len(live) > last_len:
                sys.stdout.write(live[last_len:])
                sys.stdout.flush()
            
            # Print newline after completion
            console.print()
            
            if worker.error:
                return f'[ERROR] Agent {profile_id} failed: {worker.error}'
            return result
        except Exception as e:
            return f'[ERROR] Agent {profile_id} failed: {e}'

    def delegate_async(self, profile_id: str, task: str,
                       tools: ToolRegistry, provider_id: str = None,
                       model: str = None) -> threading.Thread:
        """Delegate a task to a specialized agent (non-blocking)."""
        worker = AgentWorker(profile_id, task, tools, provider_id, model)
        self.history.append({'profile': profile_id, 'task': task, 'worker': worker})
        t = threading.Thread(target=worker.run, daemon=True)
        t.start()
        return t

    def run_concurrent(self, tasks: list[tuple[str, str]],
                       tools: ToolRegistry) -> dict[str, str]:
        """
        Run multiple agents concurrently.
        tasks: list of (profile_id, task_description)
        Returns: dict of profile_id -> result
        """
        futures = {}
        results = {}

        for profile_id, task in tasks:
            future = self._executor.submit(
                self.delegate, profile_id, task, tools
            )
            futures[future] = profile_id

        for future in as_completed(futures):
            profile_id = futures[future]
            try:
                results[profile_id] = future.result(timeout=180)
            except Exception as e:
                results[profile_id] = f'[ERROR] {e}'

        return results

    def run_concurrent_async(self, tasks: list[tuple[str, str]],
                             tools: ToolRegistry) -> list[str]:
        """
        Start multiple agents in the background (non-blocking).
        Tasks run via ThreadPoolExecutor and results are stored in history + running_tasks.
        Returns list of profile_ids for reference.
        """
        ids = []
        for profile_id, task in tasks:
            worker = AgentWorker(profile_id, task, tools)
            future = self._executor.submit(worker.run)
            self.history.append({'profile': profile_id, 'task': task, 'worker': worker})
            self.running_tasks[profile_id] = {
                'worker': worker,
                'future': future,
                'status': 'running',
                'output': '',
            }
            ids.append(profile_id)

        # Background checker to update status when done
        def _check():
            import time
            while True:
                done = True
                for pid, info in list(self.running_tasks.items()):
                    if info['future'].done():
                        info['status'] = 'done'
                        try:
                            info['output'] = info['future'].result(timeout=1)
                        except Exception as e:
                            info['output'] = f'[ERROR] {e}'
                            info['status'] = 'error'
                    else:
                        done = False
                        # Collect partial output
                        w = info['worker']
                        if hasattr(w, 'live_output') and w.live_output:
                            info['output'] = w.live_output
                if done:
                    break
                time.sleep(0.3)

        checker = threading.Thread(target=_check, daemon=True)
        checker.start()
        return ids


# Global instance
multi_agent_manager = MultiAgentManager()
