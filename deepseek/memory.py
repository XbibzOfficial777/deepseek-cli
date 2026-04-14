# DeepSeek CLI v5.5 — Conversation Memory
# Stores message history with tool call support

class Memory:
    """Manages conversation message history."""

    def __init__(self):
        self.messages: list[dict] = []
        self.system_prompt = (
            "You are DeepSeek CLI Agent v5.5, a powerful AI assistant running in the terminal.\n"
            "You have access to 65+ tools including file operations, LIVE web search, code execution,\n"
            "system info, math, PDF reader/editor, DOCX creator/reader, image viewer,\n"
            "video info, APK analyzer, live model search, and web browser automation.\n"
            "Use tools freely — there are NO usage limits.\n"
            "\n"
            "IMPORTANT RULES:\n"
            "1. When users ask about current events, news, weather, stock prices, or anything\n"
            "   that requires real-time information, ALWAYS use the 'live_search' tool.\n"
            "2. Be helpful, direct, and concise. Execute tools when needed to accomplish tasks.\n"
            "3. STOP calling tools once you have enough information to answer the user's question.\n"
            "   Do NOT keep calling the same tool repeatedly — this wastes resources and time.\n"
            "4. Do NOT repeat tool calls you have already made. If a tool returned useful results,\n"
            "   use those results directly instead of calling the tool again.\n"
            "5. When your task is complete, provide your final answer WITHOUT calling any more tools.\n"
            "6. If a tool fails, try a different approach instead of retrying the same tool endlessly.\n"
            "7. You have a MAXIMUM of 12 tool rounds per response. Plan your tool usage wisely.\n"
            "\n"
            "RESPONSE GUIDELINES:\n"
            "- Answer directly when you know the answer — no need to always use tools.\n"
            "- If you cannot complete a task after trying, explain what went wrong.\n"
            "- Format responses clearly with markdown when appropriate.\n"
            "- Keep responses focused and avoid unnecessary verbosity."
        )
        # Add initial system message
        self.messages.append({'role': 'system', 'content': self.system_prompt})

    def add_user(self, content: str):
        self.messages.append({'role': 'user', 'content': content})

    def add_assistant(self, content: str):
        self.messages.append({'role': 'assistant', 'content': content})

    def add_assistant_tool_calls(self, content: str, tool_calls: list):
        msg = {'role': 'assistant', 'content': content, 'tool_calls': tool_calls}
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, name: str, result: str):
        self.messages.append({
            'role': 'tool',
            'tool_call_id': tool_call_id,
            'name': name,
            'content': result
        })

    def add_system(self, content: str):
        self.system_prompt = content
        # Update the first system message
        if self.messages and self.messages[0]['role'] == 'system':
            self.messages[0]['content'] = content
        else:
            self.messages.insert(0, {'role': 'system', 'content': content})

    def get_messages(self) -> list[dict]:
        return list(self.messages)

    def clear(self):
        self.messages = [{'role': 'system', 'content': self.system_prompt}]

    def count(self) -> int:
        return len(self.messages) - 1  # Exclude system message

    def export_text(self) -> str:
        """Export conversation as readable text."""
        lines = []
        lines.append("DeepSeek CLI v5.5 — Chat Export")
        lines.append(f"Messages: {self.count()}")
        lines.append("=" * 50)
        for msg in self.messages:
            role = msg['role'].upper()
            content = msg.get('content', '')
            if role == 'SYSTEM':
                continue
            if role == 'TOOL':
                name = msg.get('name', '?')
                lines.append(f"\n[Tool Result: {name}]")
                lines.append(content)
            elif role == 'ASSISTANT':
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get('function', {})
                        lines.append(f"\n[Tool Call: {fn.get('name', '?')}]")
                        lines.append(fn.get('arguments', '{}'))
                if content:
                    lines.append(f"\n[Assistant]")
                    lines.append(content)
            elif role == 'USER':
                lines.append(f"\n[User]")
                lines.append(content)
        return '\n'.join(lines)
