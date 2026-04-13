# DeepSeek CLI v4 — Agentic Loop
# Send → Parse tool_calls → Execute → Feed back → Repeat

import json
from rich.console import Console

from .config import MAX_TOOL_ROUNDS
from .providers import BaseProvider
from .memory import Memory
from .toolkit import ToolRegistry
from .ui import StreamRenderer

console = Console()


class Agent:
    """Agentic loop: sends messages, handles tool calls, feeds results back."""

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

    def chat(self, user_message: str) -> dict:
        """
        Process a user message through the agentic loop.
        Returns {'content': str, 'tool_rounds': int, 'error': str|None}
        """
        self.memory.add_user(user_message)
        full_content = ''
        thinking_text = ''
        tool_rounds = 0

        # Determine if tools should be sent
        send_tools = self._tool_functions if self.provider.supports_tools else None

        for round_num in range(MAX_TOOL_ROUNDS + 1):
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
                    break
                elif chunk_type == 'done':
                    pass

            if has_error:
                self.renderer.show_done()
                return {'content': full_content or thinking_text,
                        'tool_rounds': tool_rounds, 'error': 'Stream error'}

            if not tool_calls_list:
                self.memory.add_assistant(full_content)
                self.renderer.show_done()
                return {'content': full_content,
                        'tool_rounds': tool_rounds, 'error': None}

            tool_rounds += 1
            self.renderer.show_done()

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

            console.print(f'\n  [bold cyan]⚙ Tool Round {tool_rounds}/{MAX_TOOL_ROUNDS}[/bold cyan]')
            console.print()

            for tc in tool_calls_list:
                fn = tc.get('function', {})
                tool_name = fn.get('name', 'unknown')
                tc_id = tc.get('id', '')
                try:
                    args = json.loads(fn.get('arguments', '{}'))
                except json.JSONDecodeError:
                    args = {}
                self.renderer.show_tool_call(tool_name, args)
                result = self.tools.execute(tool_name, args)
                self.renderer.show_tool_result(tool_name, result)
                self.memory.add_tool_result(tc_id, tool_name, result)

            console.print()
            console.print('  [dim]→ Processing tool results...[/dim]')

        self.memory.add_assistant(full_content + "\n\n[Max tool rounds reached]")
        self.renderer.show_done()
        return {'content': full_content, 'tool_rounds': tool_rounds,
                'error': 'Max tool rounds reached'}

    def set_model(self, model: str):
        self.model = model

    def set_thinking(self, visible: bool):
        self.thinking_visible = visible
        self.renderer = StreamRenderer(thinking_visible=visible)

    def set_provider(self, provider: BaseProvider):
        """Switch to a different provider."""
        self.provider = provider
