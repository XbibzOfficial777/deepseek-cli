# DeepSeek CLI v3 — Tool Registry (26+ Tools)
# All tools displayed and available for the agent

import os
import sys
import json
import subprocess
import math
import random
import datetime
import time
import shutil
import platform
from typing import Any
from pathlib import Path


class ToolRegistry:
    """Registry for all agent tools."""

    def __init__(self):
        self.tools: dict[str, dict] = {}
        self._register_all()

    def register(self, name: str, description: str, parameters: dict, handler):
        self.tools[name] = {
            'description': description,
            'parameters': parameters,
            'handler': handler,
        }

    def get_openai_tools(self) -> list[dict]:
        """Return tools in OpenAI function-calling format."""
        result = []
        for name, tool in self.tools.items():
            result.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': tool['description'],
                    'parameters': tool['parameters'],
                }
            })
        return result

    def get_tool_list(self) -> list[dict]:
        """Return list of all tools with names and descriptions."""
        return [
            {'name': name, 'description': t['description']}
            for name, t in self.tools.items()
        ]

    def execute(self, name: str, arguments: dict) -> str:
        if name not in self.tools:
            return f"Error: Unknown tool '{name}'"
        try:
            return self.tools[name]['handler'](arguments)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    def _register_all(self):
        self._register_file_tools()
        self._register_web_tools()
        self._register_code_tools()
        self._register_system_tools()
        self._register_math_tools()
        self._register_utility_tools()

    # ══════════════════════════════════════
    # FILE TOOLS
    # ══════════════════════════════════════

    def _register_file_tools(self):
        self.register(
            'read_file',
            'Read contents of a file from filesystem. Use absolute or relative paths.',
            {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'File path to read'}
                },
                'required': ['path']
            },
            lambda args: self._read_file(args['path'])
        )

        self.register(
            'write_file',
            'Write content to a file. Creates parent dirs if needed. Overwrites existing.',
            {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'File path to write'},
                    'content': {'type': 'string', 'description': 'Content to write'}
                },
                'required': ['path', 'content']
            },
            lambda args: self._write_file(args['path'], args['content'])
        )

        self.register(
            'list_files',
            'List files and directories at a given path. Like ls/dir command.',
            {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Directory path (default: current)'},
                    'pattern': {'type': 'string', 'description': 'Optional glob filter (e.g. *.py)'}
                },
                'required': []
            },
            lambda args: self._list_files(args.get('path', '.'), args.get('pattern', ''))
        )

        self.register(
            'delete_file',
            'Delete a file or empty directory from filesystem.',
            {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'File or directory path to delete'}
                },
                'required': ['path']
            },
            lambda args: self._delete_file(args['path'])
        )

        self.register(
            'file_info',
            'Get detailed information about a file (size, modified, type, etc).',
            {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'File path'}
                },
                'required': ['path']
            },
            lambda args: self._file_info(args['path'])
        )

    # ══════════════════════════════════════
    # WEB TOOLS
    # ══════════════════════════════════════

    def _register_web_tools(self):
        self.register(
            'web_search',
            'Search the web using DuckDuckGo. Returns titles, URLs, and snippets.',
            {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query'},
                    'max_results': {'type': 'integer', 'description': 'Max results (default 5)'}
                },
                'required': ['query']
            },
            lambda args: self._web_search(args['query'], args.get('max_results', 5))
        )

        self.register(
            'web_fetch',
            'Fetch and extract text content from a URL. For reading web pages.',
            {
                'type': 'object',
                'properties': {
                    'url': {'type': 'string', 'description': 'URL to fetch'}
                },
                'required': ['url']
            },
            lambda args: self._web_fetch(args['url'])
        )

    # ══════════════════════════════════════
    # CODE TOOLS
    # ══════════════════════════════════════

    def _register_code_tools(self):
        self.register(
            'run_code',
            'Execute Python code and return stdout/stderr output. Useful for calculations, data processing.',
            {
                'type': 'object',
                'properties': {
                    'code': {'type': 'string', 'description': 'Python code to execute'},
                    'timeout': {'type': 'integer', 'description': 'Timeout in seconds (default 30)'}
                },
                'required': ['code']
            },
            lambda args: self._run_code(args['code'], args.get('timeout', 30))
        )

        self.register(
            'run_shell',
            'Execute a shell/bash command and return output. Use for system operations.',
            {
                'type': 'object',
                'properties': {
                    'command': {'type': 'string', 'description': 'Shell command to execute'},
                    'timeout': {'type': 'integer', 'description': 'Timeout in seconds (default 30)'}
                },
                'required': ['command']
            },
            lambda args: self._run_shell(args['command'], args.get('timeout', 30))
        )

        self.register(
            'install_package',
            'Install a Python package via pip. Use when code needs a missing library.',
            {
                'type': 'object',
                'properties': {
                    'package': {'type': 'string', 'description': 'Package name to install'}
                },
                'required': ['package']
            },
            lambda args: self._install_package(args['package'])
        )

    # ══════════════════════════════════════
    # SYSTEM TOOLS
    # ══════════════════════════════════════

    def _register_system_tools(self):
        self.register(
            'system_info',
            'Get system information: OS, CPU, memory, disk, Python version.',
            {
                'type': 'object',
                'properties': {},
                'required': []
            },
            lambda args: self._system_info()
        )

        self.register(
            'process_list',
            'List running processes on the system.',
            {
                'type': 'object',
                'properties': {
                    'filter_name': {'type': 'string', 'description': 'Optional process name filter'}
                },
                'required': []
            },
            lambda args: self._process_list(args.get('filter_name', ''))
        )

        self.register(
            'disk_usage',
            'Get disk usage information for filesystem.',
            {
                'type': 'object',
                'properties': {},
                'required': []
            },
            lambda args: self._disk_usage()
        )

        self.register(
            'network_info',
            'Get network information: hostname, IP, interfaces.',
            {
                'type': 'object',
                'properties': {},
                'required': []
            },
            lambda args: self._network_info()
        )

        self.register(
            'env_vars',
            'Get environment variables. Optionally filter by name pattern.',
            {
                'type': 'object',
                'properties': {
                    'filter': {'type': 'string', 'description': 'Optional name filter (e.g. PATH)'}
                },
                'required': []
            },
            lambda args: self._env_vars(args.get('filter', ''))
        )

    # ══════════════════════════════════════
    # MATH TOOLS
    # ══════════════════════════════════════

    def _register_math_tools(self):
        self.register(
            'calculate',
            'Evaluate a mathematical expression. Supports +, -, *, /, ^, sqrt, sin, cos, tan, log, pi, e.',
            {
                'type': 'object',
                'properties': {
                    'expression': {'type': 'string', 'description': 'Math expression to evaluate'}
                },
                'required': ['expression']
            },
            lambda args: self._calculate(args['expression'])
        )

        self.register(
            'unit_convert',
            'Convert between units: length, weight, temperature, data, time.',
            {
                'type': 'object',
                'properties': {
                    'value': {'type': 'number', 'description': 'Value to convert'},
                    'from_unit': {'type': 'string', 'description': 'Source unit'},
                    'to_unit': {'type': 'string', 'description': 'Target unit'}
                },
                'required': ['value', 'from_unit', 'to_unit']
            },
            lambda args: self._unit_convert(args['value'], args['from_unit'], args['to_unit'])
        )

    # ══════════════════════════════════════
    # UTILITY TOOLS
    # ══════════════════════════════════════

    def _register_utility_tools(self):
        self.register(
            'timestamp',
            'Get current date/time. Optionally convert Unix timestamp or format.',
            {
                'type': 'object',
                'properties': {
                    'unix_ts': {'type': 'number', 'description': 'Optional Unix timestamp to convert'},
                    'timezone': {'type': 'string', 'description': 'Timezone (default: local)'}
                },
                'required': []
            },
            lambda args: self._timestamp(args.get('unix_ts'), args.get('timezone', 'local'))
        )

        self.register(
            'text_transform',
            'Transform text: uppercase, lowercase, reverse, word_count, char_count, base64 encode/decode.',
            {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string', 'description': 'Text to transform'},
                    'operation': {'type': 'string', 'description': 'Operation: upper, lower, reverse, word_count, char_count, base64_encode, base64_decode, title, strip, slug'}
                },
                'required': ['text', 'operation']
            },
            lambda args: self._text_transform(args['text'], args['operation'])
        )

        self.register(
            'json_parse',
            'Parse, format, or query JSON data. Can pretty-print or extract fields.',
            {
                'type': 'object',
                'properties': {
                    'json_string': {'type': 'string', 'description': 'JSON string to parse'},
                    'query': {'type': 'string', 'description': 'Optional dot-notation field to extract'},
                    'pretty': {'type': 'boolean', 'description': 'Pretty print (default true)'}
                },
                'required': ['json_string']
            },
            lambda args: self._json_parse(args['json_string'], args.get('query'), args.get('pretty', True))
        )

        self.register(
            'generate_uuid',
            'Generate random UUIDs. Useful for creating unique identifiers.',
            {
                'type': 'object',
                'properties': {
                    'count': {'type': 'integer', 'description': 'Number of UUIDs to generate (default 1)'}
                },
                'required': []
            },
            lambda args: self._generate_uuid(args.get('count', 1))
        )

        self.register(
            'random_number',
            'Generate random numbers within a range. Can generate integers or floats.',
            {
                'type': 'object',
                'properties': {
                    'min_val': {'type': 'number', 'description': 'Minimum value (default 1)'},
                    'max_val': {'type': 'number', 'description': 'Maximum value (default 100)'},
                    'integer': {'type': 'boolean', 'description': 'Integer only (default true)'}
                },
                'required': []
            },
            lambda args: self._random_number(
                args.get('min_val', 1), args.get('max_val', 100),
                args.get('integer', True)
            )
        )

        self.register(
            'base64_tool',
            'Encode or decode Base64 strings and files.',
            {
                'type': 'object',
                'properties': {
                    'data': {'type': 'string', 'description': 'String to encode/decode'},
                    'mode': {'type': 'string', 'description': 'encode or decode'}
                },
                'required': ['data', 'mode']
            },
            lambda args: self._base64_tool(args['data'], args['mode'])
        )

        self.register(
            'regex_test',
            'Test regular expressions against text. Shows matches and groups.',
            {
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string', 'description': 'Regex pattern'},
                    'text': {'type': 'string', 'description': 'Text to test against'},
                    'flags': {'type': 'string', 'description': 'Flags: i=ignorecase, m=multiline, s=dotall'}
                },
                'required': ['pattern', 'text']
            },
            lambda args: self._regex_test(args['pattern'], args['text'], args.get('flags', ''))
        )

        self.register(
            'sort_data',
            'Sort lines of text. Can sort alphabetically, numerically, reverse, unique.',
            {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string', 'description': 'Text with lines to sort'},
                    'mode': {'type': 'string', 'description': 'alpha, numeric, reverse, unique, length'},
                    'reverse': {'type': 'boolean', 'description': 'Reverse order'}
                },
                'required': ['text']
            },
            lambda args: self._sort_data(args['text'], args.get('mode', 'alpha'), args.get('reverse', False))
        )

        self.register(
            'count_text',
            'Count occurrences: lines, words, characters, specific pattern matches in text.',
            {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string', 'description': 'Text to analyze'},
                    'pattern': {'type': 'string', 'description': 'Optional pattern to count'}
                },
                'required': ['text']
            },
            lambda args: self._count_text(args['text'], args.get('pattern'))
        )

    # ══════════════════════════════════════
    # TOOL IMPLEMENTATIONS
    # ══════════════════════════════════════

    def _read_file(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"
        try:
            content = p.read_text(encoding='utf-8', errors='replace')
            lines = content.split('\n')
            if len(lines) > 500:
                preview = '\n'.join(lines[:250])
                return f"[Showing first 250 of {len(lines)} lines]\n\n{preview}\n\n... ({len(lines) - 250} more lines)"
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    def _write_file(self, path: str, content: str) -> str:
        p = Path(path).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            lines = content.count('\n') + 1
            return f"Written {len(content)} chars ({lines} lines) to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _list_files(self, path: str, pattern: str) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: Path not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"
        try:
            items = list(p.iterdir())
            if pattern:
                items = [i for i in items if i.match(pattern)]
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            lines = [f"📁 {i.name}/" if i.is_dir() else f"📄 {i.name}" for i in items]
            if not lines:
                return f"No items found in {path}" + (f" matching '{pattern}'" if pattern else "")
            return f"Contents of {path} ({len(lines)} items):\n" + "\n".join(lines)
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error: {e}"

    def _delete_file(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: Not found: {path}"
        try:
            if p.is_dir():
                shutil.rmtree(p)
                return f"Deleted directory: {path}"
            else:
                p.unlink()
                return f"Deleted file: {path}"
        except Exception as e:
            return f"Error deleting: {e}"

    def _file_info(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: Not found: {path}"
        stat = p.stat()
        size = stat.st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.2f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} B"
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        ftype = 'directory' if p.is_dir() else 'file'
        return (f"Path: {p}\n"
                f"Type: {ftype}\n"
                f"Size: {size_str}\n"
                f"Modified: {mtime}\n"
                f"Permissions: {oct(stat.st_mode)[-3:]}")

    def _web_search(self, query: str, max_results: int) -> str:
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(f"[{r.get('title', '')}]({r.get('href', '')})\n  {r.get('body', '')}")
            if not results:
                return f"No results found for: {query}"
            return f"Search results for '{query}':\n\n" + "\n\n".join(results)
        except ImportError:
            return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
        except Exception as e:
            return f"Search error: {e}"

    def _web_fetch(self, url: str) -> str:
        try:
            import httpx
            r = httpx.get(url, timeout=20, follow_redirects=True, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; DeepSeekCLI/3.0)'
            })
            r.raise_for_status()
            content_type = r.headers.get('content-type', '')
            if 'text' not in content_type and 'json' not in content_type:
                return f"Binary content (type: {content_type}), size: {len(r.content)} bytes"
            text = r.text[:5000]
            if len(r.text) > 5000:
                text += f"\n\n... (truncated, total {len(r.text)} chars)"
            return text
        except Exception as e:
            return f"Fetch error: {e}"

    def _run_code(self, code: str, timeout: int) -> str:
        banned = ['os.system', 'subprocess.call', 'subprocess.run', 'exec', 'eval',
                   '__import__', 'shutil.rmtree', 'open(']
        for b in banned:
            if b in code:
                return f"Blocked: {b} not allowed in run_code. Use run_shell instead."
        try:
            result = subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True, text=True, timeout=timeout
            )
            output = ''
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Code timed out after {timeout}s"
        except Exception as e:
            return f"Code execution error: {e}"

    def _run_shell(self, command: str, timeout: int) -> str:
        dangerous = ['rm -rf /', 'mkfs', 'dd if=', ':(){ :|:& };:']
        for d in dangerous:
            if d in command:
                return f"Blocked: dangerous command detected"
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout
            )
            output = ''
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Shell error: {e}"

    def _install_package(self, package: str) -> str:
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', package],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return f"Package '{package}' installed successfully."
            return f"Install failed: {result.stderr}"
        except Exception as e:
            return f"Install error: {e}"

    def _system_info(self) -> str:
        uname = platform.uname()
        mem = ''
        try:
            total = os.popen('free -h 2>/dev/null || cat /proc/meminfo 2>/dev/null | head -3').read()
            mem = f"\nMemory:\n{total.strip()}" if total.strip() else ''
        except Exception:
            pass
        return (f"System: {uname.system} {uname.release}\n"
                f"Node: {uname.node}\n"
                f"Machine: {uname.machine}\n"
                f"Processor: {uname.processor}\n"
                f"Python: {sys.version}\n"
                f"CWD: {os.getcwd()}{mem}")

    def _process_list(self, filter_name: str) -> str:
        try:
            cmd = 'ps aux 2>/dev/null || ps -ef 2>/dev/null'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            lines = result.stdout.strip().split('\n')
            if filter_name:
                lines = [l for l in lines if filter_name.lower() in l.lower()]
            if len(lines) > 20:
                lines = lines[:20]
                lines.append(f"... (showing 20 of many, use filter to narrow)")
            return '\n'.join(lines) if lines else "No processes found"
        except Exception as e:
            return f"Error: {e}"

    def _disk_usage(self) -> str:
        try:
            result = subprocess.run('df -h 2>/dev/null', shell=True, capture_output=True, text=True, timeout=10)
            return result.stdout.strip() if result.stdout.strip() else "Disk info not available"
        except Exception as e:
            return f"Error: {e}"

    def _network_info(self) -> str:
        try:
            hostname = platform.node()
            ip = ''
            try:
                s = __import__('socket')
                ip = s.gethostbyname(hostname)
            except Exception:
                pass
            interfaces = ''
            try:
                result = subprocess.run('ifconfig 2>/dev/null || ip addr 2>/dev/null || ipconfig 2>/dev/null',
                                       shell=True, capture_output=True, text=True, timeout=10)
                interfaces = f"\nInterfaces:\n{result.stdout.strip()[:1000]}"
            except Exception:
                pass
            return f"Hostname: {hostname}\nIP: {ip}{interfaces}"
        except Exception as e:
            return f"Error: {e}"

    def _env_vars(self, filter_name: str) -> str:
        env = dict(os.environ)
        if filter_name:
            env = {k: v for k, v in env.items() if filter_name.upper() in k.upper()}
        if not env:
            return f"No env vars matching '{filter_name}'"
        lines = [f"{k}={v}" for k, v in sorted(env.items())]
        return '\n'.join(lines)

    def _calculate(self, expression: str) -> str:
        safe_map = {
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
            'tan': math.tan, 'log': math.log, 'log10': math.log10,
            'pi': math.pi, 'e': math.e, 'abs': abs,
            'ceil': math.ceil, 'floor': math.floor, 'pow': pow,
            'round': round,
        }
        try:
            result = eval(expression, {"__builtins__": {}}, safe_map)
            return str(result)
        except Exception as e:
            return f"Math error: {e}"

    def _unit_convert(self, value, from_unit: str, to_unit: str) -> str:
        conversions = {
            ('km', 'mi'): lambda v: v * 0.621371,
            ('mi', 'km'): lambda v: v * 1.60934,
            ('kg', 'lb'): lambda v: v * 2.20462,
            ('lb', 'kg'): lambda v: v * 0.453592,
            ('c', 'f'): lambda v: v * 9/5 + 32,
            ('f', 'c'): lambda v: (v - 32) * 5/9,
            ('kb', 'mb'): lambda v: v / 1024,
            ('mb', 'gb'): lambda v: v / 1024,
            ('gb', 'tb'): lambda v: v / 1024,
            ('m', 'ft'): lambda v: v * 3.28084,
            ('ft', 'm'): lambda v: v * 0.3048,
            ('cm', 'in'): lambda v: v * 0.393701,
            ('in', 'cm'): lambda v: v * 2.54,
            ('hour', 'min'): lambda v: v * 60,
            ('min', 'sec'): lambda v: v * 60,
        }
        key = (from_unit.lower(), to_unit.lower())
        if key not in conversions:
            available = ', '.join(f"{a}->{b}" for a, b in conversions.keys())
            return f"Conversion {from_unit} -> {to_unit} not supported.\nAvailable: {available}"
        try:
            result = conversions[key](float(value))
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        except Exception as e:
            return f"Conversion error: {e}"

    def _timestamp(self, unix_ts=None, tz='local') -> str:
        try:
            if unix_ts:
                dt = datetime.datetime.fromtimestamp(float(unix_ts))
            else:
                dt = datetime.datetime.now()
            return dt.strftime(f"%Y-%m-%d %H:%M:%S ({tz})")
        except Exception as e:
            return f"Error: {e}"

    def _text_transform(self, text: str, operation: str) -> str:
        ops = {
            'upper': lambda t: t.upper(),
            'lower': lambda t: t.lower(),
            'title': lambda t: t.title(),
            'strip': lambda t: t.strip(),
            'reverse': lambda t: t[::-1],
            'word_count': lambda t: f"Word count: {len(t.split())}",
            'char_count': lambda t: f"Character count: {len(t)}",
            'line_count': lambda t: f"Line count: {len(t.splitlines())}",
            'slug': lambda t: '-'.join(t.lower().split()),
            'base64_encode': lambda t: __import__('base64').b64encode(t.encode()).decode(),
            'base64_decode': lambda t: __import__('base64').b64decode(t.encode()).decode(errors='replace'),
        }
        if operation not in ops:
            return f"Unknown operation: {operation}\nAvailable: {', '.join(ops.keys())}"
        try:
            return ops[operation](text)
        except Exception as e:
            return f"Error: {e}"

    def _json_parse(self, json_string: str, query=None, pretty=True) -> str:
        try:
            data = json.loads(json_string)
            if query:
                keys = query.split('.')
                for key in keys:
                    if isinstance(data, dict):
                        data = data.get(key, f"Key '{key}' not found")
                    elif isinstance(data, list):
                        try:
                            data = data[int(key)]
                        except (ValueError, IndexError):
                            return f"Invalid index: {key}"
                    else:
                        return f"Cannot navigate into {type(data).__name__}"
            indent = 2 if pretty else None
            return json.dumps(data, indent=indent, ensure_ascii=False)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"
        except Exception as e:
            return f"Error: {e}"

    def _generate_uuid(self, count: int) -> str:
        import uuid
        uuids = [str(uuid.uuid4()) for _ in range(count)]
        return '\n'.join(uuids)

    def _random_number(self, min_val, max_val, integer=True) -> str:
        try:
            min_v, max_v = float(min_val), float(max_val)
            if integer:
                return str(random.randint(int(min_v), int(max_v)))
            return str(random.uniform(min_v, max_v))
        except Exception as e:
            return f"Error: {e}"

    def _base64_tool(self, data: str, mode: str) -> str:
        import base64
        try:
            if mode == 'encode':
                return base64.b64encode(data.encode()).decode()
            elif mode == 'decode':
                return base64.b64decode(data.encode()).decode(errors='replace')
            else:
                return "Mode must be 'encode' or 'decode'"
        except Exception as e:
            return f"Error: {e}"

    def _regex_test(self, pattern: str, text: str, flags: str) -> str:
        import re
        try:
            flag_val = 0
            if 'i' in flags:
                flag_val |= re.IGNORECASE
            if 'm' in flags:
                flag_val |= re.MULTILINE
            if 's' in flags:
                flag_val |= re.DOTALL
            matches = list(re.finditer(pattern, text, flag_val))
            if not matches:
                return f"No matches for pattern '{pattern}'"
            lines = [f"Pattern '{pattern}' found {len(matches)} matches:"]
            for i, m in enumerate(matches[:10]):
                groups = [f"Group {j}: {g}" for j, g in enumerate(m.groups())]
                group_str = ' | '.join(groups) if groups else 'No groups'
                lines.append(f"  Match {i+1}: '{m.group()}' at pos {m.span()[0]}-{m.span()[1]} ({group_str})")
            if len(matches) > 10:
                lines.append(f"  ... and {len(matches) - 10} more matches")
            return '\n'.join(lines)
        except re.error as e:
            return f"Regex error: {e}"

    def _sort_data(self, text: str, mode: str, reverse: bool) -> str:
        lines = text.strip().split('\n')
        if not lines:
            return "No text to sort"
        if mode == 'numeric':
            try:
                lines.sort(key=lambda x: float(x.strip()), reverse=reverse)
            except ValueError:
                lines.sort(key=lambda x: float(''.join(c for c in x if c.isdigit() or c == '.').strip() or '0'), reverse=reverse)
        elif mode == 'length':
            lines.sort(key=len, reverse=reverse)
        elif mode == 'unique':
            seen = set()
            unique = []
            for l in lines:
                if l not in seen:
                    seen.add(l)
                    unique.append(l)
            lines = unique
        elif mode == 'reverse':
            lines = lines[::-1]
        else:  # alpha
            lines.sort(key=str.lower, reverse=reverse)
        return '\n'.join(lines)

    def _count_text(self, text: str, pattern: str = None) -> str:
        lines = text.split('\n')
        words = text.split()
        chars = len(text)
        result = f"Lines: {len(lines)}\nWords: {len(words)}\nCharacters: {chars}"
        if pattern:
            count = text.lower().count(pattern.lower())
            result += f"\nPattern '{pattern}' occurrences: {count}"
        return result
