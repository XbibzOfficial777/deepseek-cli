"""
Tool Registry - 25+ built-in tools for the AI agent.
Each tool has a JSON schema for function calling and a real Python handler.
"""
import os
import sys
import json
import hashlib
import base64
import re
import subprocess
import shutil
import tempfile
import math
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path


class ToolRegistry:
    """Manages all available tools for the agent."""

    def __init__(self):
        self.tools: dict = {}
        self._register_all()

    def register(self, name: str, description: str, parameters: dict, handler):
        """Register a single tool."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler,
        }

    def get_schemas(self) -> list:
        """Return OpenAI-compatible tool schemas for API."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self.tools.values()
        ]

    def get_tool_names(self) -> list:
        """Return list of tool names."""
        return list(self.tools.keys())

    def get_tool_info(self, name: str) -> dict:
        """Get info about a specific tool."""
        return self.tools.get(name, {})

    def execute(self, name: str, args: dict) -> str:
        """Execute a tool by name with given arguments."""
        if name not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"Error: Unknown tool '{name}'. Available: {available}"

        tool = self.tools[name]
        try:
            result = tool["handler"](args)
            if result is None:
                return "Tool executed successfully (no output)."
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, ensure_ascii=False, default=str)
            return str(result)
        except Exception as e:
            return f"Error executing '{name}': {type(e).__name__}: {e}"

    # ────────────────────────────────────────────────────────────
    #  Tool Implementations
    # ────────────────────────────────────────────────────────────

    def _register_all(self):
        """Register all built-in tools."""

        # ── FILE OPERATIONS (9 tools) ──────────────────────────

        self.register(
            name="read_file",
            description="Read the contents of a file. Returns the full text content.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file"},
                    "encoding": {"type": "string", "description": "File encoding (default: utf-8)", "default": "utf-8"},
                },
                "required": ["path"],
            },
            handler=self._read_file,
        )

        self.register(
            name="write_file",
            description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
            handler=self._write_file,
        )

        self.register(
            name="edit_file",
            description="Edit a file by replacing a specific search string with new content. Only replaces the first occurrence unless replace_all is true.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "old_text": {"type": "string", "description": "Text to search for"},
                    "new_text": {"type": "string", "description": "Text to replace with"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false)", "default": False},
                },
                "required": ["path", "old_text", "new_text"],
            },
            handler=self._edit_file,
        )

        self.register(
            name="list_files",
            description="List files and directories in a given path. Shows names, types, and sizes.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current directory)"},
                    "show_hidden": {"type": "boolean", "description": "Show hidden files (default: false)", "default": False},
                },
                "required": [],
            },
            handler=self._list_files,
        )

        self.register(
            name="delete_file",
            description="Delete a file or an empty directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file/directory to delete"},
                    "recursive": {"type": "boolean", "description": "Delete directory recursively (default: false)", "default": False},
                },
                "required": ["path"],
            },
            handler=self._delete_file,
        )

        self.register(
            name="search_files",
            description="Search for files matching a name pattern (glob) in a directory.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', '**/*.txt')"},
                    "path": {"type": "string", "description": "Search directory (default: current)"},
                },
                "required": ["pattern"],
            },
            handler=self._search_files,
        )

        self.register(
            name="file_info",
            description="Get detailed metadata about a file: size, type, permissions, modification time.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                },
                "required": ["path"],
            },
            handler=self._file_info,
        )

        self.register(
            name="create_directory",
            description="Create a directory, including any necessary parent directories.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to create"},
                },
                "required": ["path"],
            },
            handler=self._create_directory,
        )

        self.register(
            name="tree_view",
            description="Display directory tree structure (up to 3 levels deep).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Root directory path"},
                    "max_depth": {"type": "integer", "description": "Max depth (default: 3)", "default": 3},
                },
                "required": [],
            },
            handler=self._tree_view,
        )

        # ── WEB TOOLS (2 tools) ───────────────────────────────

        self.register(
            name="web_search",
            description="Search the web using DuckDuckGo. Returns titles, URLs, and snippets.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results (default: 5, max: 10)", "default": 5},
                },
                "required": ["query"],
            },
            handler=self._web_search,
        )

        self.register(
            name="web_fetch",
            description="Fetch and extract text content from a URL. Returns the page title and main text.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_length": {"type": "integer", "description": "Max characters to return (default: 3000)", "default": 3000},
                },
                "required": ["url"],
            },
            handler=self._web_fetch,
        )

        # ── CODE TOOLS (3 tools) ───────────────────────────────

        self.register(
            name="execute_python",
            description="Execute Python code and return stdout + stderr. Use for calculations, data processing, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)", "default": 30},
                },
                "required": ["code"],
            },
            handler=self._execute_python,
        )

        self.register(
            name="install_package",
            description="Install a Python package via pip.",
            parameters={
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "Package name (e.g. 'requests', 'numpy')"},
                    "version": {"type": "string", "description": "Specific version (optional, e.g. '==2.0.0')"},
                },
                "required": ["package"],
            },
            handler=self._install_package,
        )

        self.register(
            name="run_command",
            description="Run a shell command and return stdout + stderr. Use with caution.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)", "default": 30},
                },
                "required": ["command"],
            },
            handler=self._run_command,
        )

        # ── SYSTEM TOOLS (4 tools) ─────────────────────────────

        self.register(
            name="system_info",
            description="Get system information: OS, architecture, Python version, disk, memory, CPU.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._system_info,
        )

        self.register(
            name="check_internet",
            description="Check internet connectivity by testing multiple endpoints.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._check_internet,
        )

        self.register(
            name="process_list",
            description="List running processes (top 20 by CPU usage on Linux/Termux).",
            parameters={
                "type": "object",
                "properties": {
                    "sort_by": {"type": "string", "description": "Sort by: cpu, mem, pid (default: cpu)", "default": "cpu"},
                },
                "required": [],
            },
            handler=self._process_list,
        )

        self.register(
            name="env_variables",
            description="List all or specific environment variables.",
            parameters={
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "description": "Filter variables containing this string (optional)"},
                },
                "required": [],
            },
            handler=self._env_variables,
        )

        # ── UTILITY TOOLS (7 tools) ────────────────────────────

        self.register(
            name="calculate",
            description="Evaluate a mathematical expression safely. Supports +, -, *, /, **, sqrt, sin, cos, log, pi, e.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression (e.g. 'sqrt(144) + 2**10')"},
                },
                "required": ["expression"],
            },
            handler=self._calculate,
        )

        self.register(
            name="get_datetime",
            description="Get current date, time, timezone, and formatted variants.",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone (default: system timezone)"},
                    "format": {"type": "string", "description": "strftime format (optional)"},
                },
                "required": [],
            },
            handler=self._get_datetime,
        )

        self.register(
            name="json_format",
            description="Parse, format, or minify JSON data.",
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "JSON string to process"},
                    "action": {"type": "string", "enum": ["format", "minify", "validate", "extract"],
                               "description": "Action: format (pretty-print), minify, validate, or extract (get keys)"},
                },
                "required": ["data"],
            },
            handler=self._json_format,
        )

        self.register(
            name="base64_tool",
            description="Encode or decode Base64 strings.",
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "String to process"},
                    "action": {"type": "string", "enum": ["encode", "decode"], "description": "encode or decode"},
                },
                "required": ["data", "action"],
            },
            handler=self._base64_tool,
        )

        self.register(
            name="hash_tool",
            description="Generate hash digests (MD5, SHA1, SHA256, SHA512) of a string or file.",
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "String to hash"},
                    "algorithm": {"type": "string", "enum": ["md5", "sha1", "sha256", "sha512"],
                                  "description": "Hash algorithm (default: sha256)"},
                    "is_file": {"type": "boolean", "description": "If true, 'data' is a file path (default: false)"},
                },
                "required": ["data"],
            },
            handler=self._hash_tool,
        )

        self.register(
            name="url_parser",
            description="Parse a URL into its components: scheme, host, path, query, fragment, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to parse"},
                },
                "required": ["url"],
            },
            handler=self._url_parser,
        )

        self.register(
            name="text_transform",
            description="Transform text: uppercase, lowercase, title, reverse, word count, char count, slug, strip.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to transform"},
                    "action": {"type": "string",
                               "enum": ["uppercase", "lowercase", "title", "reverse", "word_count",
                                        "char_count", "line_count", "slug", "strip", "snake_case", "camel_case"],
                               "description": "Transformation to apply"},
                },
                "required": ["text", "action"],
            },
            handler=self._text_transform,
        )

        self.register(
            name="regex_search",
            description="Search for regex pattern matches in text or a file. Returns all matches with positions.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression pattern"},
                    "text": {"type": "string", "description": "Text to search in"},
                    "file_path": {"type": "string", "description": "Alternatively, path to a file to search in"},
                    "flags": {"type": "string", "description": "Regex flags: i=ignorecase, m=multiline, s=dotall"},
                },
                "required": ["pattern"],
            },
            handler=self._regex_search,
        )

    # ── HANDLER IMPLEMENTATIONS ─────────────────────────────────

    def _read_file(self, args):
        path = args.get("path", "")
        enc = args.get("encoding", "utf-8")
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"
        size = p.stat().st_size
        if size > 500_000:
            return f"Error: File too large ({size:,} bytes). Max 500KB for reading."
        try:
            content = p.read_text(encoding=enc)
            if not content.strip():
                return "(empty file)"
            # Show first portion if very long
            lines = content.split("\n")
            if len(lines) > 2000:
                content = "\n".join(lines[:2000])
                content += f"\n\n... (truncated, {len(lines) - 2000} more lines)"
            return content
        except UnicodeDecodeError:
            return f"Error: Cannot decode file as {enc}. Try binary-safe reading."

    def _write_file(self, args):
        path = args.get("path", "")
        content = args.get("content", "")
        p = Path(path).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            size = p.stat().st_size
            return f"Written {size:,} bytes to {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error: {e}"

    def _edit_file(self, args):
        path = args.get("path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        replace_all = args.get("replace_all", False)
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: File not found: {path}"
        try:
            content = p.read_text(encoding="utf-8")
            if old_text not in content:
                # Show context
                lines = content.split("\n")[:5]
                preview = "\n".join(lines)
                return f"Error: Search text not found in {path}.\nFile preview (first 5 lines):\n{preview}"
            if replace_all:
                count = content.count(old_text)
                content = content.replace(old_text, new_text)
            else:
                content = content.replace(old_text, new_text, 1)
                count = 1
            p.write_text(content, encoding="utf-8")
            return f"Replaced {count} occurrence(s) in {path}"
        except Exception as e:
            return f"Error: {e}"

    def _list_files(self, args):
        path = args.get("path", ".")
        show_hidden = args.get("show_hidden", False)
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: Path not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"

        entries = []
        for item in sorted(p.iterdir()):
            if not show_hidden and item.name.startswith("."):
                continue
            try:
                stat = item.stat()
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                ftype = "DIR " if item.is_dir() else "FILE"
                entries.append(f"  {ftype}  {size:>10,}  {mtime}  {item.name}")
            except PermissionError:
                entries.append(f"  ????            ??????????  {item.name}")

        if not entries:
            return "(empty directory)"
        header = f"  {'TYPE':<5}  {'SIZE':>10}  {'MODIFIED':<17}  NAME"
        return f"Directory: {p.resolve()}\n{header}\n" + "\n".join(entries)

    def _delete_file(self, args):
        path = args.get("path", "")
        recursive = args.get("recursive", False)
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: Not found: {path}"
        try:
            if p.is_dir() and recursive:
                shutil.rmtree(p)
                return f"Deleted directory recursively: {path}"
            elif p.is_dir():
                p.rmdir()
                return f"Deleted empty directory: {path}"
            else:
                p.unlink()
                return f"Deleted file: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except OSError as e:
            return f"Error: {e}"

    def _search_files(self, args):
        pattern = args.get("pattern", "*")
        path = args.get("path", ".")
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: Path not found: {path}"
        try:
            matches = sorted(p.glob(pattern))
            if not matches:
                return f"No files matching '{pattern}' in {path}"
            result = [str(m.relative_to(p)) for m in matches[:100]]
            if len(result) < len(matches):
                result.append(f"... and {len(matches) - 100} more files")
            return f"Found {len(matches)} file(s):\n" + "\n".join(result)
        except Exception as e:
            return f"Error: {e}"

    def _file_info(self, args):
        path = args.get("path", "")
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: Not found: {path}"
        try:
            stat = p.stat()
            info = {
                "path": str(p.resolve()),
                "name": p.name,
                "type": "directory" if p.is_dir() else "file",
                "size_bytes": stat.st_size,
                "size_human": self._human_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "permissions": oct(stat.st_mode),
            }
            if p.suffix:
                info["extension"] = p.suffix
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error: {e}"

    def _create_directory(self, args):
        path = args.get("path", "")
        p = Path(path).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            return f"Created directory: {p.resolve()}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error: {e}"

    def _tree_view(self, args):
        path = args.get("path", ".")
        max_depth = min(args.get("max_depth", 3), 5)
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: Path not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"

        lines = []
        self._build_tree(p, lines, "", max_depth, 0)
        total = len(lines)
        return f"Tree: {p.resolve()}\n\n" + "\n".join(lines) + f"\n\n({total} items)"

    def _build_tree(self, path, lines, prefix, max_depth, current_depth):
        if current_depth >= max_depth:
            return
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            items = [i for i in items if not i.name.startswith(".")]
        except PermissionError:
            return

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "`-- " if is_last else "|-- "
            icon = "[DIR]" if item.is_dir() else "[FILE]"
            lines.append(f"{prefix}{connector}{icon} {item.name}")
            if item.is_dir():
                extension = "    " if is_last else "|   "
                self._build_tree(item, lines, prefix + extension, max_depth, current_depth + 1)

    # ── WEB HANDLERS ───────────────────────────────────────────

    def _web_search(self, args):
        query = args.get("query", "")
        max_results = min(args.get("max_results", 5), 10)
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"No results found for: {query}"
            output = []
            for i, r in enumerate(results, 1):
                output.append(f"[{i}] {r.get('title', 'No title')}")
                output.append(f"    URL: {r.get('href', '')}")
                output.append(f"    {r.get('body', '')}\n")
            return "\n".join(output)
        except ImportError:
            return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
        except Exception as e:
            return f"Search error: {e}"

    def _web_fetch(self, args):
        url = args.get("url", "")
        max_len = args.get("max_length", 3000)
        try:
            import httpx
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/121.0.0.0",
                }
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type:
                    # Simple HTML to text
                    text = resp.text
                    # Remove scripts and styles
                    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                    # Remove tags
                    text = re.sub(r"<[^>]+>", " ", text)
                    # Clean whitespace
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) > max_len:
                        text = text[:max_len] + "..."
                    return f"Title: {url}\nLength: {len(resp.text):,} chars\n\n{text}"
                else:
                    text = resp.text[:max_len]
                    return f"Content-Type: {content_type}\n\n{text}"
        except httpx.TimeoutException:
            return f"Error: Request timed out for {url}"
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} for {url}"
        except Exception as e:
            return f"Fetch error: {e}"

    # ── CODE HANDLERS ──────────────────────────────────────────

    def _execute_python(self, args):
        code = args.get("code", "")
        timeout = args.get("timeout", 30)
        if not code.strip():
            return "Error: No code provided"
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )
            output = ""
            if result.stdout.strip():
                output += result.stdout
            if result.stderr.strip():
                output += ("\n[stderr] " if output else "[stderr] ") + result.stderr
            if not output:
                return "(no output)"
            if len(output) > 10000:
                output = output[:10000] + "\n... (truncated)"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Code timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    def _install_package(self, args):
        package = args.get("package", "")
        version = args.get("version", "")
        if not package.strip():
            return "Error: No package name provided"
        pkg_spec = f"{package}{version}" if version else package
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg_spec],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return f"Successfully installed {pkg_spec}"
            return f"Install failed:\n{result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return "Error: pip install timed out"
        except Exception as e:
            return f"Error: {e}"

    def _run_command(self, args):
        command = args.get("command", "")
        timeout = args.get("timeout", 30)
        if not command.strip():
            return "Error: No command provided"
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = ""
            if result.stdout.strip():
                output += result.stdout
            if result.stderr.strip():
                output += ("\n[stderr] " if output else "[stderr] ") + result.stderr
            if not output:
                output = f"(exit code: {result.returncode})"
            if len(output) > 10000:
                output = output[:10000] + "\n... (truncated)"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    # ── SYSTEM HANDLERS ────────────────────────────────────────

    def _system_info(self, args):
        import platform
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or "N/A",
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "python_path": sys.executable,
            "cwd": os.getcwd(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "home": str(Path.home()),
        }
        # Disk info
        try:
            disk = shutil.disk_usage("/")
            info["disk_total"] = self._human_size(disk.total)
            info["disk_used"] = self._human_size(disk.used)
            info["disk_free"] = self._human_size(disk.free)
        except Exception:
            pass
        # Memory info
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        total_kb = int(line.split()[1])
                        info["memory_total"] = self._human_size(total_kb * 1024)
                    elif "MemAvailable" in line:
                        avail_kb = int(line.split()[1])
                        info["memory_available"] = self._human_size(avail_kb * 1024)
        except Exception:
            pass
        return json.dumps(info, indent=2)

    def _check_internet(self, args):
        import httpx
        results = []
        endpoints = [
            ("Google DNS", "https://dns.google/resolve?name=example.com"),
            ("Cloudflare", "https://1.1.1.1/cdn-cgi/trace"),
            ("OpenRouter", "https://openrouter.ai/"),
        ]
        for name, url in endpoints:
            try:
                with httpx.Client(timeout=5.0) as client:
                    r = client.get(url)
                    results.append(f"  {name}: {'OK' if r.status_code < 500 else f'HTTP {r.status_code}'}")
            except Exception:
                results.append(f"  {name}: FAILED")
        connected = sum(1 for r in results if "OK" in r)
        status = "CONNECTED" if connected > 0 else "OFFLINE"
        return f"Internet Status: {status} ({connected}/{len(endpoints)} reachable)\n\n" + "\n".join(results)

    def _process_list(self, args):
        sort_by = args.get("sort_by", "cpu")
        try:
            # Try ps command (works on Termux/Linux)
            if sort_by == "cpu":
                cmd = "ps aux --sort=-%cpu 2>/dev/null || ps aux"
            elif sort_by == "mem":
                cmd = "ps aux --sort=-%mem 2>/dev/null || ps aux"
            else:
                cmd = "ps aux"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                # Show header + top 20
                output_lines = lines[:1]  # header
                for line in lines[1:21]:
                    parts = line.split(None, 10)
                    if len(parts) >= 11:
                        output_lines.append(
                            f"  PID: {parts[1]:<8} CPU: {parts[2]:>5}%  MEM: {parts[3]:>5}%  CMD: {parts[10]}"
                        )
                    else:
                        output_lines.append(f"  {line}")
                if len(lines) > 21:
                    output_lines.append(f"  ... ({len(lines) - 21} more processes)")
                return "Top Processes:\n" + "\n".join(output_lines)
            return "Could not read process list."
        except Exception as e:
            return f"Error: {e}"

    def _env_variables(self, args):
        filter_str = args.get("filter", "").lower()
        vars_list = []
        for key, value in sorted(os.environ.items()):
            if filter_str and filter_str not in key.lower() and filter_str not in value.lower():
                continue
            # Mask sensitive values
            sensitive_keys = ["key", "token", "secret", "password", "auth"]
            is_sensitive = any(s in key.lower() for s in sensitive_keys)
            display_val = value[:8] + "..." if is_sensitive and len(value) > 8 else value
            vars_list.append(f"  {key} = {display_val}")
        if not vars_list:
            return f"No variables matching '{filter_str}'"
        return f"Environment Variables ({len(vars_list)} shown):\n" + "\n".join(vars_list)

    # ── UTILITY HANDLERS ───────────────────────────────────────

    def _calculate(self, args):
        expr = args.get("expression", "").strip()
        if not expr:
            return "Error: No expression provided"
        # Allowed math functions and constants
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "len": len, "int": int, "float": float,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
            "factorial": math.factorial, "gcd": math.gcd,
        }
        try:
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return f"{expr} = {result}"
        except ZeroDivisionError:
            return f"Error: Division by zero in '{expr}'"
        except Exception as e:
            return f"Error evaluating '{expr}': {e}"

    def _get_datetime(self, args):
        tz_name = args.get("timezone", "")
        fmt = args.get("format", "")
        now = datetime.now()
        try:
            if tz_name:
                from zoneinfo import ZoneInfo
                now = now.astimezone(ZoneInfo(tz_name))
                tz_display = tz_name
            else:
                tz_display = "local"
        except ImportError:
            tz_display = "local"
        except Exception:
            tz_display = "local"

        if fmt:
            return now.strftime(fmt)

        return json.dumps({
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": tz_display,
            "unix_timestamp": int(now.timestamp()),
            "weekday": now.strftime("%A"),
            "day_of_year": now.timetuple().tm_yday,
            "week_number": now.isocalendar()[1],
        }, indent=2)

    def _json_format(self, args):
        data_str = args.get("data", "")
        action = args.get("action", "format")
        try:
            parsed = json.loads(data_str)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        if action == "format":
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        elif action == "minify":
            return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
        elif action == "validate":
            return "Valid JSON!"
        elif action == "extract":
            if isinstance(parsed, dict):
                keys = list(parsed.keys())
                return json.dumps({"keys": keys, "count": len(keys)}, indent=2)
            return f"JSON is {type(parsed).__name__}, not an object"
        return f"Unknown action: {action}"

    def _base64_tool(self, args):
        data = args.get("data", "")
        action = args.get("action", "encode")
        try:
            if action == "encode":
                encoded = base64.b64encode(data.encode("utf-8")).decode("utf-8")
                return encoded
            elif action == "decode":
                decoded = base64.b64decode(data).decode("utf-8")
                return decoded
        except Exception as e:
            return f"Base64 {action} error: {e}"
        return f"Unknown action: {action}"

    def _hash_tool(self, args):
        data = args.get("data", "")
        algorithm = args.get("algorithm", "sha256").lower()
        is_file = args.get("is_file", False)

        hash_map = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
        }
        if algorithm not in hash_map:
            return f"Unknown algorithm: {algorithm}. Use: md5, sha1, sha256, sha512"

        try:
            h = hash_map[algorithm]()
            if is_file:
                p = Path(data).expanduser()
                if not p.exists():
                    return f"Error: File not found: {data}"
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
            else:
                h.update(data.encode("utf-8"))
            return f"{algorithm.upper()}: {h.hexdigest()}"
        except Exception as e:
            return f"Hash error: {e}"

    def _url_parser(self, args):
        url = args.get("url", "")
        try:
            parsed = urlparse(url)
            info = {
                "url": url,
                "scheme": parsed.scheme or "none",
                "netloc": parsed.netloc or "none",
                "host": parsed.hostname or "none",
                "port": parsed.port or "default",
                "path": parsed.path or "/",
                "query": parsed.query or "none",
                "fragment": parsed.fragment or "none",
            }
            # Parse query params
            if parsed.query:
                params = {}
                for part in parsed.query.split("&"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        params[k] = v
                    else:
                        params[part] = ""
                info["query_params"] = params
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Parse error: {e}"

    def _text_transform(self, args):
        text = args.get("text", "")
        action = args.get("action", "")

        transforms = {
            "uppercase": lambda t: t.upper(),
            "lowercase": lambda t: t.lower(),
            "title": lambda t: t.title(),
            "reverse": lambda t: t[::-1],
            "word_count": lambda t: f"Words: {len(t.split())}",
            "char_count": lambda t: f"Characters: {len(t)}",
            "line_count": lambda t: f"Lines: {len(t.splitlines())}",
            "slug": lambda t: re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-"),
            "strip": lambda t: t.strip(),
            "snake_case": lambda t: re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", t).lower().replace(" ", "_").replace("-", "_"),
            "camel_case": lambda t: "".join(word.capitalize() for word in re.split(r"[_\-\s]+", t)),
        }
        if action not in transforms:
            return f"Unknown action: {action}. Use: {', '.join(transforms.keys())}"
        return transforms[action](text)

    def _regex_search(self, args):
        pattern = args.get("pattern", "")
        text = args.get("text", "")
        file_path = args.get("file_path", "")

        # Build regex flags
        flags = 0
        flag_str = args.get("flags", "")
        if "i" in flag_str:
            flags |= re.IGNORECASE
        if "m" in flag_str:
            flags |= re.MULTILINE
        if "s" in flag_str:
            flags |= re.DOTALL

        if file_path and not text:
            p = Path(file_path).expanduser()
            if not p.exists():
                return f"Error: File not found: {file_path}"
            try:
                text = p.read_text(encoding="utf-8")
            except Exception as e:
                return f"Error reading file: {e}"

        if not text:
            return "Error: No text to search in"

        try:
            matches = list(re.finditer(pattern, text, flags))
            if not matches:
                return f"No matches found for pattern: {pattern}"
            results = []
            for i, m in enumerate(matches, 1):
                line_num = text[:m.start()].count("\n") + 1
                start_col = m.start() - text.rfind("\n", 0, m.start())
                results.append(
                    f"  Match {i}: line {line_num}, col {start_col}\n"
                    f"    Text: '{m.group()}'\n"
                    f"    Span: [{m.start()}:{m.end()}]"
                )
            return f"Found {len(matches)} match(es) for '{pattern}':\n\n" + "\n".join(results)
        except re.error as e:
            return f"Regex error: {e}"

    # ── HELPERS ────────────────────────────────────────────────

    @staticmethod
    def _human_size(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
