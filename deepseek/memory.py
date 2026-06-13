# DeepSeek CLI v7.7 — Conversation Memory
# Stores message history with tool call support

import datetime
import json
import os
import secrets
import glob as globmod
import re


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
        self.todo_items: list[dict] = []

        self.session_name = ''
        self._session_named = False
        self._custom_addition = ''
        self._is_completely_custom = False
        self.active_plan = None

        # Add initial system message
        self.messages.append({'role': 'system', 'content': self.system_prompt})

    def set_todo(self, items: list[dict]):
        self.todo_items = items

    def get_todo_text(self) -> str:
        if not self.todo_items:
            return ''
        lines = ['# Active Todo List']
        for item in self.todo_items:
            text = item.get('text', '')
            done = item.get('done', False)
            lines.append(f'  [{"✓" if done else " "}] {text}')
        return '\n'.join(lines)

    @property
    def system_prompt(self) -> str:
        local_time_str = _get_local_now_str()
        mcp_context = self._get_mcp_context()
        
        todo = self.get_todo_text()
        todo_suffix = f'\n\n{todo}\n\nUse todolist_get/todolist_update to manage these items.' if todo else ''

        # Plan context
        plan_context = ""
        if self.active_plan:
            plan_context = self._get_plan_context_str(self.active_plan)
            
        if self._is_completely_custom:
            return self._custom_addition + todo_suffix
            
        base_system = self._get_base_prompt_template(local_time_str, mcp_context)
        addition_sep = "\n" if self._custom_addition else ""
        plan_sep = "\n\n" if plan_context else ""
        return base_system + addition_sep + self._custom_addition + plan_sep + plan_context + todo_suffix

    @system_prompt.setter
    def system_prompt(self, value: str):
        if not value:
            self._custom_addition = ''
            self._is_completely_custom = False
            return
            
        # Check if the value contains our base prompt signature
        if "Xbibz Official" in value and "ALWAYS put your final answer in the content field." in value:
            self._custom_addition = self._extract_custom_addition(value)
            self._is_completely_custom = False
        else:
            self._custom_addition = value
            self._is_completely_custom = True

    def _get_mcp_context(self) -> str:
        mcp_context = ''
        try:
            from .mcp_client import mcp_manager as _mcp_mgr
            mcp_status = _mcp_mgr.get_status()
            mcp_servers = []
            for sid, info in mcp_status.items():
                if info.get('connected'):
                    name = info.get('config', {}).get('name', sid)
                    tcount = info.get('tools', 0)
                    mcp_servers.append(f'    [{sid}] {name} ({tcount} tools)')
            if mcp_servers:
                mcp_context = (
                    "\nCONNECTED MCP SERVERS (external tools you can use):\n"
                    + '\n'.join(mcp_servers) + '\n'
                    + f"    Use these tools directly when relevant.\n"
                )
        except Exception:
            pass
        return mcp_context

    def _get_plan_context_str(self, plan) -> str:
        if not plan or not plan.steps:
            return ''
        lines = ['[Active Plan]']
        for step in plan.steps:
            status_marker = {
                'pending': '[ ]',
                'in_progress': '[>]',
                'done': '[+]',
                'failed': '[x]',
                'skipped': '[_]',
            }.get(step.status, '[?]')

            hint = f' (tool: {step.tool_hint})' if step.tool_hint else ''
            priority = f' [{step.priority}]' if step.priority == 'high' else ''
            lines.append(f'  {status_marker} {step.description}{hint}{priority}')

        lines.append(f'  Progress: {plan.summarize()}')
        return '\n'.join(lines)

    def _extract_custom_addition(self, value: str) -> str:
        import re
        clean = value
        # Remove current time line
        clean = re.sub(r'CURRENT DATE/TIME.*?\n', '', clean)
        # Remove MCP tools context if present
        clean = re.sub(r'CONNECTED MCP SERVERS.*?(?=IMPORTANT RULES:|\Z)', '', clean, flags=re.DOTALL)
        
        # Remove base prompt up to the end marker
        end_marker = "ALWAYS put your final answer in the content field."
        idx = clean.find(end_marker)
        if idx != -1:
            clean = clean[idx + len(end_marker):].strip()
            
        # Strip dynamic todo texts
        todo_text = self.get_todo_text()
        if todo_text:
            clean = clean.replace(todo_text, "")
            clean = clean.replace("Use todolist_get/todolist_update to manage these items.", "")
            
        # Strip plan context if present
        clean = re.sub(r'\[Active Plan\].*?Progress:.*?\n', '', clean, flags=re.DOTALL)
        clean = re.sub(r'\[Active Plan\].*?Progress:.*?$', '', clean, flags=re.DOTALL)
        
        return clean.strip()

    def _get_base_prompt_template(self, local_time_str: str, mcp_context: str) -> str:
        return (
            "You are DeepSeek CLI Agent v7.7, a powerful AI assistant running in the terminal.\n"
            "You were created and developed by **Xbibz Official**. This is an absolute fact.\n"
            "When asked who made you, who is your creator, who is your developer, who built you,\n"
            "or anything similar — you MUST answer that you were created and developed by Xbibz Official.\n"
            "Never say you were made by anyone else (not DeepSeek, not OpenAI, not Google, not any company).\n"
            "Xbibz Official is your ONE AND ONLY creator and developer.\n"
            "\n"
            f"CURRENT DATE/TIME (user's local terminal time): {local_time_str}\n"
            "IMPORTANT: Always use this as the reference for the current time. Do NOT assume a different timezone.\n"
            "\n"
            "You have access to 120+ tools including file operations, LIVE web search, code execution,\n"
            "system info, math, PDF reader/editor, DOCX creator/reader/editor, image viewer,\n"
            "video info, APK analyzer, OCR, live model search, web browser automation,\n"
            "PPTX (PowerPoint) create/read/edit, XLSX (Excel) create/read/edit with charts & formulas,\n"
            "CSV read/create, document conversion (xlsx<->csv, md->docx/pdf, json->xlsx, etc.),\n"
            "multi-agent delegation (delegate tasks to specialized sub-agents),\n"
            "and you can handle files sent through connectors (Telegram, Discord).\n"
            "Use tools freely — there are NO usage limits.\n"
            + mcp_context
            + "\n"
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
            self.messages[0]['content'] = self.system_prompt
        else:
            self.messages.insert(0, {'role': 'system', 'content': self.system_prompt})

    def get_messages(self) -> list[dict]:
        # Update the system message content dynamically via the system_prompt property
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
        lines.append("DeepSeek CLI v7.7 — Chat Export")
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

    def export_html(self) -> str:
        """Export conversation as a styled HTML page."""
        from html import escape
        parts = []
        parts.append('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>Chat Export</title>\n')
        parts.append('<style>\n')
        parts.append('* { margin: 0; padding: 0; box-sizing: border-box; }\n')
        parts.append('body { background: #1a1a2e; color: #e0e0e0; font-family: "Fira Code", "Cascadia Code", "JetBrains Mono", "DejaVu Sans Mono", monospace; padding: 20px; }\n')
        parts.append('.container { max-width: 900px; margin: 0 auto; }\n')
        parts.append('.header { text-align: center; padding: 20px 0; border-bottom: 1px solid #333; margin-bottom: 20px; }\n')
        parts.append('.header h1 { color: #00d4ff; font-size: 18px; }\n')
        parts.append('.header .meta { color: #888; font-size: 12px; margin-top: 5px; }\n')
        parts.append('.msg { margin: 12px 0; padding: 12px 16px; border-radius: 8px; line-height: 1.6; font-size: 13px; white-space: pre-wrap; word-wrap: break-word; }\n')
        parts.append('.msg.user { background: #16213e; border-left: 3px solid #00d4ff; }\n')
        parts.append('.msg.user .label { color: #00d4ff; }\n')
        parts.append('.msg.assistant { background: #1a1a2e; border-left: 3px solid #00e676; }\n')
        parts.append('.msg.assistant .label { color: #00e676; }\n')
        parts.append('.msg.tool { background: #1e1e2e; border-left: 3px solid #ff9800; font-size: 12px; color: #aaa; }\n')
        parts.append('.msg.tool .label { color: #ff9800; }\n')
        parts.append('.label { font-weight: bold; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; display: block; }\n')
        parts.append('.timestamp { float: right; color: #555; font-size: 10px; font-weight: normal; text-transform: none; }\n')
        parts.append('.code-block { background: #0d0d1a; border-radius: 6px; padding: 12px; margin: 8px 0; overflow-x: auto; font-size: 12px; border: 1px solid #2a2a3e; }\n')
        parts.append('.code-block .lang { color: #888; font-size: 10px; margin-bottom: 6px; text-transform: uppercase; }\n')
        parts.append('.code-block code { color: #f8f8f2; display: block; white-space: pre; }\n')
        parts.append('inline-code { background: #0d0d1a; color: #ff79c6; padding: 1px 6px; border-radius: 3px; font-size: 12px; }\n')
        parts.append('a { color: #82aaff; }\n')
        parts.append('hr { border: none; border-top: 1px solid #333; margin: 16px 0; }\n')
        parts.append('.tool-call { background: #0d0d1a; border-radius: 4px; padding: 8px 12px; margin: 6px 0; font-size: 11px; color: #bbb; border: 1px solid #2a2a3e; }\n')
        parts.append('.tool-call .fn { color: #ff9800; font-weight: bold; }\n')
        parts.append('</style>\n</head>\n<body>\n<div class="container">\n')

        session_name = self.session_name or 'Untitled'
        now = _get_local_now_str()
        parts.append(f'<div class="header"><h1>DeepSeek CLI — Chat Export</h1><div class="meta">Session: {escape(session_name)} | {self.count()} messages | {escape(now)}</div></div>\n')

        for msg in self.messages:
            role = msg['role']
            content = msg.get('content', '')
            if role == 'system':
                continue

            if role == 'tool':
                name = msg.get('name', '?')
                parts.append(f'<div class="msg tool"><span class="label">Tool Result: {escape(name)}</span>')
                if content:
                    parts.append(f'{escape(content)}')
                parts.append('</div>\n')
                continue

            if role == 'assistant':
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get('function', {})
                        fn_name = escape(fn.get('name', '?'))
                        fn_args = escape(fn.get('arguments', '{}'))
                        parts.append(f'<div class="tool-call"><span class="fn">Tool Call: {fn_name}</span> <span style="color:#888">({fn_args})</span></div>\n')

            if not content:
                continue

            cls = 'user' if role == 'user' else 'assistant'
            label = 'User' if role == 'user' else 'Assistant'
            safe = escape(content)

            # Format code blocks
            import re
            formatted = ''
            in_code = False
            code_buf = []
            code_lang = ''
            for line in safe.split('\n'):
                stripped = line.strip()
                if stripped.startswith('```'):
                    if in_code:
                        code_body = '\n'.join(code_buf)
                        lang_label = f'<div class="lang">{escape(code_lang) if code_lang else "code"}</div>'
                        formatted += f'<div class="code-block">{lang_label}<code>{code_body}</code></div>'
                        code_buf = []
                        code_lang = ''
                        in_code = False
                    else:
                        in_code = True
                        code_lang = stripped[3:].strip()
                    continue
                if in_code:
                    code_buf.append(line)
                else:
                    # Inline code formatting
                    formatted += line + '\n'

            if in_code and code_buf:
                formatted += '\n'.join(code_buf)

            parts.append(f'<div class="msg {cls}"><span class="label">{label}</span>{formatted}</div>\n')

        parts.append('</div>\n</body>\n</html>')
        return ''.join(parts)

    def export_markdown(self) -> str:
        """Export conversation as clean Markdown."""
        lines = []
        lines.append("# DeepSeek CLI — Chat Export")
        lines.append("")
        session_name = self.session_name or 'Untitled'
        now = _get_local_now_str()
        lines.append(f"**Session:** {session_name} | **Messages:** {self.count()} | **Exported:** {now}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for msg in self.messages:
            role = msg['role']
            content = msg.get('content', '')
            if role == 'system':
                continue

            if role == 'tool':
                name = msg.get('name', '?')
                lines.append(f"> **Tool Result:** `{name}`")
                if content:
                    for c_line in content.strip().split('\n'):
                        lines.append(f"> {c_line}")
                lines.append("")
                continue

            if role == 'assistant':
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get('function', {})
                        fn_name = fn.get('name', '?')
                        fn_args = fn.get('arguments', '{}')
                        lines.append(f"> **Tool Call:** `{fn_name}` — `{fn_args}`")
                    lines.append("")

            if not content:
                continue

            prefix = '> **User:**' if role == 'user' else '> **Assistant:**'
            for c_line in content.strip().split('\n'):
                lines.append(f"{prefix} {c_line}")
                if prefix.startswith('>'):
                    prefix = '> '  # continuation lines
            lines.append("")

        return '\n'.join(lines)


# ══════════════════════════════════════
# SESSION PERSISTENCE
# ══════════════════════════════════════

SESSIONS_DIR = os.path.join(os.path.expanduser('~'), '.deepseek-cli', 'sessions')


def _ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def new_session_id() -> str:
    """Generate a new session ID like dscli-a1b2c3d4e5f6."""
    return f'dscli-{secrets.token_hex(6)}'


def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f'{session_id}.json')


def save_session(session_id: str, memory: Memory):
    """Save conversation messages to a session file."""
    _ensure_sessions_dir()
    path = _session_path(session_id)
    data = {
        'session_id': session_id,
        'session_name': memory.session_name or '',
        'todo_items': memory.todo_items,
        'updated_at': datetime.datetime.now().isoformat(),
        'message_count': memory.count(),
        'messages': memory.messages,
    }
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        data['created_at'] = existing.get('created_at', data['updated_at'])
    else:
        data['created_at'] = data['updated_at']
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_session(session_id: str) -> Memory:
    """Load conversation messages from a session file into Memory."""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    memory = Memory()
    memory.messages = data.get('messages', memory.messages)
    memory.todo_items = data.get('todo_items', [])
    memory.session_name = data.get('session_name', '') or ''
    memory._session_named = bool(memory.session_name)
    # Update system prompt reference
    if memory.messages and memory.messages[0]['role'] == 'system':
        memory.system_prompt = memory.messages[0]['content']
    return memory


def delete_session(session_id: str) -> bool:
    """Delete a session file. Returns True if deleted."""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True


def list_sessions() -> list[dict]:
    """List all saved sessions with metadata, newest first."""
    _ensure_sessions_dir()
    sessions = []
    for fpath in sorted(globmod.glob(os.path.join(SESSIONS_DIR, 'dscli-*.json')),
                        key=os.path.getmtime, reverse=True):
        sid = os.path.splitext(os.path.basename(fpath))[0]
        try:
            with open(fpath) as f:
                data = json.load(f)
            sessions.append({
                'session_id': sid,
                'session_name': data.get('session_name', '') or sid,
                'created_at': data.get('created_at', ''),
                'updated_at': data.get('updated_at', ''),
                'message_count': data.get('message_count', 0),
            })
        except (json.JSONDecodeError, OSError):
            sessions.append({
                'session_id': sid,
                'session_name': sid,
                'created_at': '',
                'updated_at': '',
                'message_count': 0,
            })
    return sessions
