# DeepSeek CLI v3 — Conversation Memory

from datetime import datetime
from typing import Optional


class Memory:
    """Manages conversation history with token-aware trimming."""

    def __init__(self, system_prompt: str = None):
        self.messages = []
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant with access to tools. "
            "Use tools when needed to help the user. "
            "Always respond in the same language the user uses."
        )
        self.created_at = datetime.now()

    def add_system(self, text: str = None):
        """Add or replace system message."""
        content = text or self.system_prompt
        if self.messages and self.messages[0].get('role') == 'system':
            self.messages[0]['content'] = content
        else:
            self.messages.insert(0, {'role': 'system', 'content': content})

    def add_user(self, text: str):
        self.messages.append({'role': 'user', 'content': text})

    def add_assistant(self, text: str):
        self.messages.append({'role': 'assistant', 'content': text})

    def add_tool_result(self, tool_call_id: str, name: str, result: str):
        self.messages.append({
            'role': 'tool',
            'tool_call_id': tool_call_id,
            'name': name,
            'content': str(result)
        })

    def add_assistant_tool_calls(self, content: str, tool_calls: list):
        """Add assistant message with tool_calls."""
        self.messages.append({
            'role': 'assistant',
            'content': content or '',
            'tool_calls': tool_calls
        })

    def get_messages(self) -> list:
        return list(self.messages)

    def clear(self):
        self.messages.clear()
        self.add_system()

    def count(self) -> int:
        return len(self.messages)

    def get_last_exchange(self, n: int = 4) -> list:
        """Get last n messages for context display."""
        return self.messages[-n:] if self.messages else []

    def export_text(self) -> str:
        """Export conversation as readable text."""
        lines = [
            f"Conversation exported: {datetime.now().isoformat()}",
            "=" * 50
        ]
        for msg in self.messages:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            if role == 'TOOL':
                name = msg.get('name', 'tool')
                lines.append(f"[{role}: {name}] {content}")
            elif role == 'ASSISTANT' and msg.get('tool_calls'):
                lines.append(f"[{role}] {content}")
                for tc in msg['tool_calls']:
                    fn = tc.get('function', {})
                    lines.append(f"  → Tool: {fn.get('name', '?')}({fn.get('arguments', '')})")
            else:
                lines.append(f"[{role}] {content}")
            lines.append("-" * 30)
        return "\n".join(lines)
