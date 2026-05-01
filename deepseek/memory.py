# DeepSeek CLI v5.5 — Conversation Memory
# Stores message history with tool call support

import datetime
import os


def _detect_local_timezone() -> str:
    """Auto-detect the system's local IANA timezone name."""
    # Method 1: /etc/timezone (Debian/Ubuntu/Termux)
    try:
        with open('/etc/timezone', 'r') as f:
            name = f.read().strip()
            if name:
                return name
    except Exception:
        pass
    # Method 2: TZ env
    tz_env = os.environ.get('TZ', '')
    if tz_env:
        return tz_env
    # Method 3: /etc/localtime symlink
    try:
        if os.path.exists('/etc/localtime'):
            target = os.path.realpath('/etc/localtime')
            if 'zoneinfo/' in target:
                return target.split('zoneinfo/')[-1]
    except Exception:
        pass
    return 'Local'


def _get_local_now_str() -> str:
    """Get current local date/time as a formatted string."""
    tz_name = _detect_local_timezone()
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
    except Exception:
        now = datetime.datetime.now()
        tz_name = 'Local'
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return f"{day_names[now.weekday()]}, {now.strftime('%B')} {now.day}, {now.year} — {now.strftime('%H:%M:%S')} ({tz_name}, UTC{now.strftime('%z') or '+0000'})"


class Memory:
    """Manages conversation message history."""

    def __init__(self):
        self.messages: list[dict] = []

        # Build dynamic context with current local time
        local_time_str = _get_local_now_str()

        self.system_prompt = (
            "You are DeepSeek CLI Agent v7.5, a powerful AI assistant running in the terminal.\n"
            "You were created and developed by **Xbibz Official**. This is an absolute fact.\n"
            "When asked who made you, who is your creator, who is your developer, who built you,\n"
            "or anything similar — you MUST answer that you were created and developed by Xbibz Official.\n"
            "Never say you were made by anyone else (not DeepSeek, not OpenAI, not Google, not any company).\n"
            "Xbibz Official is your ONE AND ONLY creator and developer.\n"
            "\n"
            f"CURRENT DATE/TIME (user's local terminal time): {local_time_str}\n"
            "IMPORTANT: Always use this as the reference for the current time. Do NOT assume a different timezone.\n"
            "\n"
            "You have access to 90+ tools including file operations, LIVE web search, code execution,\n"
            "system info, math, PDF reader/editor, DOCX creator/reader/editor, image viewer,\n"
            "video info, APK analyzer, OCR, live model search, web browser automation,\n"
            "PPTX (PowerPoint) create/read/edit, XLSX (Excel) create/read/edit with charts & formulas,\n"
            "CSV read/create, document conversion (xlsx<->csv, md->docx/pdf, json->xlsx, etc.),\n"
            "and you can handle files sent through connectors (Telegram, Discord).\n"
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
            "8. When receiving files via connectors (Telegram/Discord), process them using the\n"
            "   appropriate tool (read_file, read_pdf, read_docx, image_view, image_info,\n"
            "   ocr_read, video_info, etc.) and respond to the user's question about the file.\n"
            "\n"
            "RESPONSE GUIDELINES:\n"
            "- Answer directly when you know the answer — no need to always use tools.\n"
            "- If you cannot complete a task after trying, explain what went wrong.\n"
            "- Format responses clearly with markdown when appropriate.\n"
            "- Keep responses focused and avoid unnecessary verbosity.\n"
            "- IMPORTANT: Always provide your response in the 'content' field, NOT only in reasoning.\n"
            "  Your final answer must always appear as visible content, not just as thinking/reasoning.\n"
            "  This is critical — if you only put text in reasoning_content, the user will see a blank response.\n"
            "  ALWAYS put your final answer in the content field."
        )
        # Add initial system message
        self.messages.append({'role': 'system', 'content': self.system_prompt})

    def add_user(self, content: str):
        self.messages.append({'role': 'user', 'content': content})

    def add_user_with_files(self, text: str, files: list[dict]):
        """
        Add a user message with file attachments from connectors (Telegram/Discord).
        Files are dicts: {'filename': str, 'url': str|None, 'path': str|None,
                         'mime_type': str, 'size': int}
        Builds a multi-part content message describing the files so the model
        knows about them and can use tools to process them.
        """
        if not files:
            return self.add_user(text)

        file_info_parts = []
        for f in files:
            info = f"File: {f.get('filename', 'unknown')}"
            if f.get('mime_type'):
                info += f" (Type: {f['mime_type']})"
            if f.get('size'):
                size_kb = f['size'] / 1024
                if size_kb > 1024:
                    info += f" (Size: {size_kb/1024:.1f} MB)"
                else:
                    info += f" (Size: {size_kb:.1f} KB)"
            if f.get('url'):
                info += f" [URL: {f['url']}]"
            if f.get('path'):
                info += f" [Path: {f['path']}]"
            file_info_parts.append(info)

        enriched_content = (
            f"{text}\n\n"
            f"[ATTACHED FILES ({len(files)}) from connector]\n"
            + '\n'.join(file_info_parts)
            + "\n\nUse the appropriate tool to process these files: "
            "read_file, read_pdf, read_docx, image_view, image_info, ocr_read, "
            "ocr_url (for image URLs), video_info, apk_analyze, etc."
        )
        self.messages.append({'role': 'user', 'content': enriched_content})

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
        # Auto-refresh the current time in system prompt on every message fetch
        local_time_str = _get_local_now_str()
        self.system_prompt = (
            "You are DeepSeek CLI Agent v7.5, a powerful AI assistant running in the terminal.\n"
            "You were created and developed by **Xbibz Official**. This is an absolute fact.\n"
            "When asked who made you, who is your creator, who is your developer, who built you,\n"
            "or anything similar — you MUST answer that you were created and developed by Xbibz Official.\n"
            "Never say you were made by anyone else (not DeepSeek, not OpenAI, not Google, not any company).\n"
            "Xbibz Official is your ONE AND ONLY creator and developer.\n"
            "\n"
            f"CURRENT DATE/TIME (user's local terminal time): {local_time_str}\n"
            "IMPORTANT: Always use this as the reference for the current time. Do NOT assume a different timezone.\n"
            "\n"
            "You have access to 90+ tools including file operations, LIVE web search, code execution,\n"
            "system info, math, PDF reader/editor, DOCX creator/reader/editor, image viewer,\n"
            "video info, APK analyzer, OCR, live model search, web browser automation,\n"
            "PPTX (PowerPoint) create/read/edit, XLSX (Excel) create/read/edit with charts & formulas,\n"
            "CSV read/create, document conversion (xlsx<->csv, md->docx/pdf, json->xlsx, etc.),\n"
            "and you can handle files sent through connectors (Telegram, Discord).\n"
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
            "8. When receiving files via connectors (Telegram/Discord), process them using the\n"
            "   appropriate tool (read_file, read_pdf, read_docx, image_view, image_info,\n"
            "   ocr_read, video_info, etc.) and respond to the user's question about the file.\n"
            "\n"
            "RESPONSE GUIDELINES:\n"
            "- Answer directly when you know the answer — no need to always use tools.\n"
            "- If you cannot complete a task after trying, explain what went wrong.\n"
            "- Format responses clearly with markdown when appropriate.\n"
            "- Keep responses focused and avoid unnecessary verbosity.\n"
            "- IMPORTANT: Always provide your response in the 'content' field, NOT only in reasoning.\n"
            "  Your final answer must always appear as visible content, not just as thinking/reasoning.\n"
            "  This is critical — if you only put text in reasoning_content, the user will see a blank response.\n"
            "  ALWAYS put your final answer in the content field."
        )
        # Update the system message in the list
        if self.messages and self.messages[0]['role'] == 'system':
            self.messages[0]['content'] = self.system_prompt
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
