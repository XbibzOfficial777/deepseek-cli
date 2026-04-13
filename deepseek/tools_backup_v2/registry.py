"""
DeepSeek CLI - Complete Tool Registry

20 professional tools with OpenAI function-calling JSON schemas.
Each tool executes independently and returns JSON strings.

Categories:
  system      - Shell, system info, process manager, package manager
  filesystem  - File read/write/list/search/copy/move/delete
  web         - Web search, HTTP requests
  code        - Code analysis, code generator
  git         - Git operations
  text        - Text processing, grep in files
  network     - Network diagnostics
  utility     - JSON processing, encoding, calculator, timestamp, environment
"""

from __future__ import annotations

import json
import subprocess
import os
import shutil
import time
import re
import socket
import base64
import hashlib
import urllib.request
import urllib.error
import urllib.parse
import html as html_mod
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  Base
# ═══════════════════════════════════════════════════════════════

class Tool(ABC):
    name: str = "unnamed"
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    risk_level: str = "safe"
    category: str = "general"

    @abstractmethod
    def execute(self, **kwargs) -> str:
        ...

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_all(self) -> Dict[str, Tool]:
        return dict(self._tools)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_names(self) -> List[str]:
        return list(self._tools.keys())

    @property
    def count(self) -> int:
        return len(self._tools)


# ═══════════════════════════════════════════════════════════════
#  1. Shell Execution
# ═══════════════════════════════════════════════════════════════

class ShellTool(Tool):
    name = "execute_shell"
    description = (
        "Execute a shell/bash command and return stdout/stderr. "
        "Supports any terminal command. Destructive commands like 'rm -rf /' are blocked."
    )
    risk_level = "moderate"
    category = "system"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)."},
        },
        "required": ["command"],
    }

    _DANGEROUS = [
        r"rm\s+-rf\s+/", r"rm\s+-rf\s+\*", r"mkfs\.", r"dd\s+.*of=/dev/",
        r">\s+/dev/sd[a-z]", r"chmod\s+-R\s+777\s+/", r":\(\)\{.*\};:",
        r"wget\s+.*\|\s*(sh|bash)", r"curl\s+.*\|\s*(sh|bash)",
    ]

    def execute(self, command: str, timeout: int = 30) -> str:
        for pat in self._DANGEROUS:
            if re.search(pat, command, re.IGNORECASE):
                return json.dumps({"error": f"Blocked: dangerous command pattern", "command": command})
        try:
            t0 = time.time()
            r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
            out = {
                "return_code": r.returncode,
                "stdout": r.stdout[-8000:] if r.stdout else "",
                "stderr": r.stderr[-4000:] if r.stderr else "",
                "duration": round(time.time() - t0, 2),
            }
            if len(r.stdout) > 8000: out["stdout_truncated"] = True
            if len(r.stderr) > 4000: out["stderr_truncated"] = True
            return json.dumps(out, indent=2)
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"Timed out after {timeout}s"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  2. File System
# ═══════════════════════════════════════════════════════════════

class FileSystemTool(Tool):
    name = "file_system"
    description = (
        "Read, write, list, search, copy, move, delete files and directories. "
        "Primary tool for all file operations."
    )
    risk_level = "low"
    category = "filesystem"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "append", "list", "search", "delete",
                         "move", "copy", "create_dir", "create_file", "info", "tree"],
                "description": "File operation to perform.",
            },
            "path": {"type": "string", "description": "File or directory path."},
            "content": {"type": "string", "description": "Content for write/append."},
            "destination": {"type": "string", "description": "Destination for move/copy."},
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py')."},
            "recursive": {"type": "boolean", "description": "Recursive search/list."},
        },
        "required": ["operation", "path"],
    }

    def execute(self, operation: str, path: str, content: str = "",
                destination: str = "", pattern: str = "*", recursive: bool = False) -> str:
        try:
            p = Path(path).expanduser().resolve()
            ops = {
                "read": lambda: self._read(p),
                "write": lambda: self._write(p, content),
                "append": lambda: self._append(p, content),
                "list": lambda: self._list(p, pattern, recursive),
                "search": lambda: self._search(p, pattern, recursive),
                "info": lambda: self._info(p),
                "delete": lambda: self._delete(p),
                "create_dir": lambda: self._mkdir(p),
                "create_file": lambda: self._touch(p),
                "tree": lambda: self._tree(p),
            }
            if operation == "move" and destination:
                d = Path(destination).expanduser().resolve()
                shutil.move(str(p), str(d))
                return json.dumps({"ok": True, "from": str(p), "to": str(d)})
            if operation == "copy" and destination:
                d = Path(destination).expanduser().resolve()
                d.parent.mkdir(parents=True, exist_ok=True)
                (shutil.copytree if p.is_dir() else shutil.copy2)(str(p), str(d))
                return json.dumps({"ok": True, "from": str(p), "to": str(d)})
            h = ops.get(operation)
            return h() if h else json.dumps({"error": f"Unknown: {operation}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _read(self, p):
        if not p.exists(): return json.dumps({"error": f"Not found: {p}"})
        c = p.read_text(encoding="utf-8", errors="replace")
        if len(c) > 30000: c = c[:30000] + "\n... [truncated]"
        return json.dumps({"content": c, "size": len(c), "path": str(p)})

    def _write(self, p, c):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(c, encoding="utf-8")
        return json.dumps({"ok": True, "path": str(p), "size": len(c)})

    def _append(self, p, c):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f: f.write(c)
        return json.dumps({"ok": True, "path": str(p)})

    def _list(self, p, pat, rec):
        if not p.is_dir(): return json.dumps({"error": f"Not a dir: {p}"})
        items = []
        for item in sorted((p.rglob if rec else p.glob)(pat)):
            try:
                st = item.stat()
                items.append({"name": item.name, "type": "dir" if item.is_dir() else "file",
                              "size": st.st_size, "path": str(item.relative_to(p))})
            except (PermissionError, OSError): pass
            if len(items) >= 500: break
        return json.dumps({"items": items, "count": len(items)})

    def _search(self, p, pat, rec):
        if not p.is_dir(): return json.dumps({"error": f"Not a dir: {p}"})
        matches = [str(i) for i in sorted((p.rglob if rec else p.glob)(pat)) if i.is_file()][:200]
        return json.dumps({"matches": matches, "count": len(matches)})

    def _info(self, p):
        if not p.exists(): return json.dumps({"error": f"Not found: {p}"})
        st = p.stat()
        return json.dumps({"name": p.name, "type": "dir" if p.is_dir() else "file",
                           "size": st.st_size,
                           "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                           "permissions": oct(st.st_mode)[-3:]})

    def _delete(self, p):
        if not p.exists(): return json.dumps({"error": f"Not found: {p}"})
        (shutil.rmtree if p.is_dir() else p.unlink)(p)
        return json.dumps({"ok": True, "deleted": str(p)})

    def _mkdir(self, p):
        p.mkdir(parents=True, exist_ok=True)
        return json.dumps({"ok": True, "created": str(p)})

    def _touch(self, p):
        p.parent.mkdir(parents=True, exist_ok=True); p.touch()
        return json.dumps({"ok": True, "created": str(p)})

    def _tree(self, p, max_depth=4):
        if not p.is_dir(): return json.dumps({"error": f"Not a dir: {p}"})
        lines = [p.name]
        def walk(d, prefix, depth):
            if depth >= max_depth: return
            try:
                items = sorted(d.iterdir())
            except PermissionError: return
            for i, item in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "`-- " if is_last else "|-- "
                if item.is_dir():
                    lines.append(f"{prefix}{connector}{item.name}/")
                    walk(item, prefix + ("    " if is_last else "|   "), depth + 1)
                else:
                    size = item.stat().st_size
                    lines.append(f"{prefix}{connector}{item.name} ({size}b)")
        walk(p, "", 0)
        return json.dumps({"tree": "\n".join(lines), "entries": len(lines)})


# ═══════════════════════════════════════════════════════════════
#  3. Web Search
# ═══════════════════════════════════════════════════════════════

class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web via DuckDuckGo. Returns titles, URLs, and snippets."
    risk_level = "low"
    category = "web"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "num_results": {"type": "integer", "description": "Results count (default 5, max 10)."},
        },
        "required": ["query"],
    }

    def execute(self, query: str, num_results: int = 5) -> str:
        try:
            num_results = min(max(num_results, 1), 10)
            params = urllib.parse.urlencode({"q": query})
            url = f"https://html.duckduckgo.com/html/?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            results = []
            titles = re.findall(r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', raw, re.DOTALL)
            snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', raw, re.DOTALL)
            for i in range(min(len(titles), num_results)):
                title = html_mod.unescape(re.sub(r"<[^>]+>", "", titles[i])).strip()
                snippet = html_mod.unescape(re.sub(r"<[^>]+>", "", snippets[i])).strip() if i < len(snippets) else ""
                results.append({"title": title, "snippet": snippet[:300]})
            if not results:
                for m in re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>', raw)[:num_results]:
                    results.append({"title": m[1].strip(), "snippet": ""})
            return json.dumps({"query": query, "results": results, "count": len(results)}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "query": query})


# ═══════════════════════════════════════════════════════════════
#  4. Code Analysis
# ═══════════════════════════════════════════════════════════════

class CodeAnalysisTool(Tool):
    name = "analyze_code"
    description = (
        "Analyze code: count lines, find functions, classes, imports, comments. "
        "Works on files or code snippets."
    )
    risk_level = "safe"
    category = "code"
    parameters = {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to code file."},
            "code": {"type": "string", "description": "Code snippet (instead of filepath)."},
        },
    }

    def execute(self, filepath: str = "", code: str = "") -> str:
        try:
            if filepath:
                p = Path(filepath).expanduser().resolve()
                if not p.exists(): return json.dumps({"error": f"Not found: {filepath}"})
                content = p.read_text(encoding="utf-8", errors="replace")
                fname, ext = p.name, p.suffix
            elif code:
                content, fname, ext = code, "snippet", ".txt"
            else:
                return json.dumps({"error": "Provide filepath or code"})
            lines = content.split("\n")
            funcs, classes, imports, comments = [], [], [], []
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if s.startswith(("import ", "from ")): imports.append(s[:100])
                if s.startswith(("#", "//", "/*")): comments.append({"line": i, "text": s[:80]})
                m = re.match(r"(?:async\s+)?def\s+(\w+)", s)
                if m: funcs.append({"name": m.group(1), "line": i})
                m = re.match(r"class\s+(\w+)", s)
                if m: classes.append({"name": m.group(1), "line": i})
            return json.dumps({
                "file": fname, "ext": ext, "lines": len(lines),
                "blank": sum(1 for l in lines if not l.strip()),
                "comment_lines": len(comments),
                "functions": funcs, "classes": classes,
                "imports": imports[:15], "import_count": len(imports),
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  5. System Info
# ═══════════════════════════════════════════════════════════════

class SystemInfoTool(Tool):
    name = "system_info"
    description = "Get OS, CPU, memory, disk, Python, and environment info."
    risk_level = "safe"
    category = "system"
    parameters = {
        "type": "object",
        "properties": {
            "info_type": {
                "type": "string",
                "enum": ["all", "os", "cpu", "memory", "disk", "python", "env", "network"],
                "description": "Info category (default: all).",
            },
        },
    }

    def execute(self, info_type: str = "all") -> str:
        r = {}
        if info_type in ("all", "os"):
            import platform
            r["os"] = {"system": platform.system(), "release": platform.release(),
                       "machine": platform.machine(), "hostname": platform.node()}
        if info_type in ("all", "python"):
            import sys
            r["python"] = {"version": sys.version.split()[0], "platform": sys.platform,
                           "executable": sys.executable}
        if info_type in ("all", "cpu"):
            try:
                import multiprocessing
                r["cpu"] = {"cores": multiprocessing.cpu_count()}
            except Exception:
                r["cpu"] = {"cores": "unknown"}
        if info_type in ("all", "memory"):
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            kb = int(line.split()[1])
                            r["memory"] = {"total_mb": round(kb/1024), "total_gb": round(kb/1024/1024, 1)}
                            break
            except Exception: pass
        if info_type in ("all", "disk"):
            try:
                total, _, free = shutil.disk_usage("/")
                r["disk"] = {"total_gb": round(total/1e9, 1), "free_gb": round(free/1e9, 1)}
            except Exception: pass
        if info_type in ("all", "env"):
            r["env"] = {k: os.environ.get(k, "")[:100] for k in
                        ["HOME", "USER", "SHELL", "LANG", "TERM", "PWD", "PATH"]}
        if info_type in ("all", "network"):
            r["network"] = {"hostname": socket.gethostname() if hasattr(socket, 'gethostname') else "unknown"}
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                r["network"]["local_ip"] = s.getsockname()[0]
                s.close()
            except Exception: pass
        return json.dumps(r, indent=2)


# ═══════════════════════════════════════════════════════════════
#  6. Git Operations
# ═══════════════════════════════════════════════════════════════

class GitTool(Tool):
    name = "git_operations"
    description = (
        "Git operations: status, log, diff, branch list, remote, "
        "stash, show, blame, tag. Works on any git repository."
    )
    risk_level = "low"
    category = "git"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["status", "log", "diff", "branch", "remote", "stash",
                         "show", "blame", "tag"],
                "description": "Git operation.",
            },
            "args": {"type": "string", "description": "Additional arguments (e.g. 'HEAD~5' for log)."},
            "path": {"type": "string", "description": "Repository path (default: current dir)."},
        },
        "required": ["operation"],
    }

    def execute(self, operation: str, args: str = "", path: str = ".") -> str:
        try:
            cmd_map = {
                "status": "git status --short",
                "log": f"git log --oneline -20 {args}",
                "diff": f"git diff {args}",
                "branch": "git branch -a",
                "remote": "git remote -v",
                "stash": "git stash list",
                "show": f"git show {args}",
                "blame": f"git blame {args}",
                "tag": "git tag -l",
            }
            cmd = cmd_map.get(operation)
            if not cmd: return json.dumps({"error": f"Unknown git op: {operation}"})
            cwd = Path(path).expanduser().resolve() if path else None
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=str(cwd) if cwd else None)
            return json.dumps({
                "operation": operation,
                "return_code": r.returncode,
                "output": r.stdout[-6000:] if r.stdout else "",
                "error": r.stderr[-2000:] if r.stderr else "",
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  7. HTTP Requests
# ═══════════════════════════════════════════════════════════════

class HttpRequestTool(Tool):
    name = "http_request"
    description = (
        "Make HTTP requests (GET, POST, PUT, DELETE). Returns status code, "
        "headers, and response body. For fetching web content, testing APIs, etc."
    )
    risk_level = "moderate"
    category = "web"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to request."},
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                       "description": "HTTP method (default: GET)."},
            "headers": {"type": "string", "description": "JSON string of headers."},
            "body": {"type": "string", "description": "Request body (for POST/PUT)."},
            "timeout": {"type": "integer", "description": "Timeout seconds (default: 15)."},
        },
        "required": ["url"],
    }

    def execute(self, url: str, method: str = "GET", headers: str = "",
                body: str = "", timeout: int = 15) -> str:
        try:
            hdrs = {"User-Agent": "DeepSeek-CLI/2.1"}
            if headers:
                try: hdrs.update(json.loads(headers))
                except Exception: pass
            data = body.encode("utf-8") if body else None
            req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_body = resp.read().decode("utf-8", errors="replace")[:10000]
                resp_headers = dict(resp.headers)
            return json.dumps({
                "status": resp.status,
                "method": method,
                "url": resp.url,
                "duration": round(time.time() - t0, 2),
                "headers": {k: v[:100] for k, v in list(resp_headers.items())[:15]},
                "body_length": len(resp_body),
                "body": resp_body[:3000],
            }, indent=2)
        except urllib.error.HTTPError as e:
            return json.dumps({"error": f"HTTP {e.code}: {e.reason}", "url": url})
        except Exception as e:
            return json.dumps({"error": str(e), "url": url})


# ═══════════════════════════════════════════════════════════════
#  8. Text Processing
# ═══════════════════════════════════════════════════════════════

class TextProcessTool(Tool):
    name = "text_process"
    description = (
        "Text processing: grep/search in text, regex replace, count lines/words/chars, "
        "extract data with regex, find & replace, sort, uniq, head, tail."
    )
    risk_level = "safe"
    category = "text"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["grep", "replace", "count", "extract", "split", "sort", "uniq", "head", "tail"],
                "description": "Text operation.",
            },
            "text": {"type": "string", "description": "Input text to process."},
            "pattern": {"type": "string", "description": "Regex pattern or delimiter."},
            "replacement": {"type": "string", "description": "Replacement string (for replace)."},
            "filepath": {"type": "string", "description": "Read text from this file instead."},
            "count": {"type": "integer", "description": "Number of lines (for head/tail)."},
        },
        "required": ["operation"],
    }

    def execute(self, operation: str, text: str = "", pattern: str = "",
                replacement: str = "", filepath: str = "", count: int = 10) -> str:
        try:
            if filepath:
                p = Path(filepath).expanduser().resolve()
                if not p.exists(): return json.dumps({"error": f"Not found: {filepath}"})
                text = p.read_text(encoding="utf-8", errors="replace")
            if not text and operation != "count":
                return json.dumps({"error": "Provide text or filepath"})

            if operation == "grep":
                matches = [(i+1, line.strip()) for i, line in enumerate(text.split("\n"))
                           if re.search(pattern, line, re.IGNORECASE)]
                return json.dumps({"matches": matches[:100], "count": len(matches)})
            elif operation == "replace":
                result = re.sub(pattern, replacement, text)
                return json.dumps({"result": result[:10000], "replacements": len(re.findall(pattern, text))})
            elif operation == "count":
                lines = text.split("\n") if text else []
                words = text.split() if text else []
                return json.dumps({"lines": len(lines), "words": len(words),
                                   "chars": len(text), "bytes": len(text.encode("utf-8"))})
            elif operation == "extract":
                matches = re.findall(pattern, text)
                return json.dumps({"matches": [str(m) for m in matches[:100]], "count": len(matches)})
            elif operation == "split":
                parts = text.split(pattern) if pattern else text.split("\n")
                return json.dumps({"parts": parts[:200], "count": len(parts)})
            elif operation == "sort":
                lines = sorted(text.split("\n"))
                return json.dumps({"lines": lines[:500], "count": len(lines)})
            elif operation == "uniq":
                seen, unique = set(), []
                for line in text.split("\n"):
                    if line not in seen: seen.add(line); unique.append(line)
                return json.dumps({"unique": unique[:500], "count": len(unique)})
            elif operation == "head":
                lines = text.split("\n")[:count]
                return json.dumps({"lines": lines, "count": len(lines)})
            elif operation == "tail":
                lines = text.split("\n")[-count:]
                return json.dumps({"lines": lines, "count": len(lines)})
            return json.dumps({"error": f"Unknown op: {operation}"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  9. Process Manager
# ═══════════════════════════════════════════════════════════════

class ProcessManagerTool(Tool):
    name = "process_manager"
    description = (
        "List running processes, check port usage, find processes by name, "
        "kill processes. Works on Linux, macOS, Termux."
    )
    risk_level = "moderate"
    category = "system"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list", "find", "kill", "port", "cpu", "memory"],
                "description": "Process operation.",
            },
            "name": {"type": "string", "description": "Process name to find/kill."},
            "pid": {"type": "integer", "description": "PID to kill."},
            "port": {"type": "integer", "description": "Port number to check."},
        },
        "required": ["operation"],
    }

    def execute(self, operation: str, name: str = "", pid: int = 0, port: int = 0) -> str:
        try:
            if operation == "list":
                r = subprocess.run("ps aux 2>/dev/null || ps -ef 2>/dev/null", shell=True,
                                   capture_output=True, text=True, timeout=10)
                lines = [l for l in r.stdout.strip().split("\n") if l]
                return json.dumps({"processes": lines[:50], "count": len(lines)})
            elif operation == "find" and name:
                safe_name = name.replace("'", "'\\''")
                r = subprocess.run(f"ps aux 2>/dev/null | grep -i '{safe_name}' | grep -v grep || ps -ef 2>/dev/null | grep -i '{safe_name}' | grep -v grep",
                                   shell=True, capture_output=True, text=True, timeout=10)
                lines = [l for l in r.stdout.strip().split("\n") if l]
                return json.dumps({"processes": lines, "count": len(lines), "query": name})
            elif operation == "kill":
                if pid:
                    r = subprocess.run(f"kill {pid} 2>&1", shell=True, capture_output=True, text=True, timeout=5)
                elif name:
                    safe_name = name.replace("'", "'\\''")
                    r = subprocess.run(f"pkill -f '{safe_name}' 2>&1", shell=True, capture_output=True, text=True, timeout=5)
                else:
                    return json.dumps({"error": "Provide pid or name"})
                return json.dumps({"output": r.stdout + r.stderr, "return_code": r.returncode})
            elif operation == "port" and port:
                r = subprocess.run(f"ss -tlnp 2>/dev/null | grep ':{port}' || netstat -tlnp 2>/dev/null | grep ':{port}'",
                                   shell=True, capture_output=True, text=True, timeout=10)
                return json.dumps({"port": port, "output": r.stdout.strip() or "No process on this port"})
            elif operation == "cpu":
                r = subprocess.run("ps aux --sort=-%cpu 2>/dev/null | head -11 || top -bn1 -o %CPU 2>/dev/null | head -11",
                                   shell=True, capture_output=True, text=True, timeout=10)
                return json.dumps({"top_cpu": r.stdout.strip().split("\n")})
            elif operation == "memory":
                r = subprocess.run("ps aux --sort=-%mem 2>/dev/null | head -11 || top -bn1 -o %MEM 2>/dev/null | head -11",
                                   shell=True, capture_output=True, text=True, timeout=10)
                return json.dumps({"top_memory": r.stdout.strip().split("\n")})
            return json.dumps({"error": f"Unknown op: {operation}"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  10. Package Manager
# ═══════════════════════════════════════════════════════════════

class PackageManagerTool(Tool):
    name = "package_manager"
    description = "Manage Python packages: install, uninstall, list, show info, check outdated."
    risk_level = "moderate"
    category = "system"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["install", "uninstall", "list", "info", "outdated"],
                "description": "Package operation.",
            },
            "package": {"type": "string", "description": "Package name."},
            "version": {"type": "string", "description": "Version constraint (e.g. '>=1.0')."},
        },
        "required": ["operation"],
    }

    def execute(self, operation: str, package: str = "", version: str = "") -> str:
        try:
            pkg = package.strip()
            if operation == "install" and pkg:
                spec = f"{pkg}{version}" if version else pkg
                r = subprocess.run(f"pip install --quiet {spec} 2>&1", shell=True,
                                   capture_output=True, text=True, timeout=120)
                return json.dumps({"installed": pkg, "output": r.stderr[-2000:] if r.stderr else "OK",
                                   "return_code": r.returncode})
            elif operation == "uninstall" and pkg:
                r = subprocess.run(f"pip uninstall -y {pkg} 2>&1", shell=True,
                                   capture_output=True, text=True, timeout=30)
                return json.dumps({"uninstalled": pkg, "output": r.stdout + r.stderr,
                                   "return_code": r.returncode})
            elif operation == "list":
                r = subprocess.run("pip list --format=json 2>/dev/null", shell=True,
                                   capture_output=True, text=True, timeout=15)
                try:
                    pkgs = json.loads(r.stdout)[:100]
                    return json.dumps({"packages": [{"name": p["name"], "version": p["version"]} for p in pkgs],
                                       "count": len(pkgs)})
                except Exception:
                    return json.dumps({"output": r.stdout[:5000]})
            elif operation == "info" and pkg:
                r = subprocess.run(f"pip show {pkg} 2>&1", shell=True,
                                   capture_output=True, text=True, timeout=10)
                return json.dumps({"package": pkg, "info": r.stdout.strip()})
            elif operation == "outdated":
                r = subprocess.run("pip list --outdated --format=json 2>/dev/null", shell=True,
                                   capture_output=True, text=True, timeout=15)
                try:
                    pkgs = json.loads(r.stdout)
                    return json.dumps({"outdated": [{"name": p["name"], "current": p["version"],
                                                      "latest": p["latest_version"]} for p in pkgs[:50]],
                                       "count": len(pkgs)})
                except Exception:
                    return json.dumps({"output": r.stdout[:3000]})
            return json.dumps({"error": f"Provide package for {operation}"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  11. Network Diagnostics
# ═══════════════════════════════════════════════════════════════

class NetworkTool(Tool):
    name = "network_diagnostics"
    description = (
        "Network tools: ping hosts, check DNS, test port connectivity, "
        "check internet connection, get IP info."
    )
    risk_level = "low"
    category = "network"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["ping", "dns", "port_check", "internet", "ip_info"],
                "description": "Network operation.",
            },
            "host": {"type": "string", "description": "Hostname or IP."},
            "port": {"type": "integer", "description": "Port number."},
            "timeout": {"type": "integer", "description": "Timeout seconds."},
        },
        "required": ["operation"],
    }

    def execute(self, operation: str, host: str = "", port: int = 0, timeout: int = 5) -> str:
        try:
            if operation == "internet":
                urls = ["https://openrouter.ai", "https://httpbin.org/get", "https://www.google.com"]
                for url in urls:
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "DeepSeek-CLI/2.1"})
                        t0 = time.time()
                        with urllib.request.urlopen(req, timeout=timeout) as resp:
                            return json.dumps({"online": True, "url": url,
                                               "status": resp.status,
                                               "latency_ms": round((time.time()-t0)*1000)})
                    except Exception: continue
                return json.dumps({"online": False})
            elif operation == "ping" and host:
                r = subprocess.run(f"ping -c 3 -W {timeout} {host} 2>&1 || ping -n 3 {host} 2>&1",
                                   shell=True, capture_output=True, text=True, timeout=20)
                return json.dumps({"host": host, "output": r.stdout.strip() or r.stderr.strip()})
            elif operation == "dns" and host:
                try:
                    ip = socket.gethostbyname(host)
                    return json.dumps({"host": host, "ip": ip})
                except socket.gaierror:
                    return json.dumps({"host": host, "error": "DNS resolution failed"})
            elif operation == "port_check" and host and port:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                t0 = time.time()
                result = s.connect_ex((host, port))
                latency = round((time.time() - t0) * 1000)
                s.close()
                return json.dumps({"host": host, "port": port,
                                   "open": result == 0, "latency_ms": latency})
            elif operation == "ip_info":
                info = {"hostname": socket.gethostname()}
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    info["local_ip"] = s.getsockname()[0]
                    s.close()
                except Exception: pass
                try:
                    req = urllib.request.Request("https://httpbin.org/ip", headers={"User-Agent": "DeepSeek-CLI"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read())
                        info["public_ip"] = data.get("origin", "unknown")
                except Exception: pass
                return json.dumps(info)
            return json.dumps({"error": "Provide host/port for this operation"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  12. Grep in Files
# ═══════════════════════════════════════════════════════════════

class GrepTool(Tool):
    name = "grep_files"
    description = (
        "Search for text patterns across files in a directory. Like the Unix 'grep' command. "
        "Supports regex patterns, file filtering, and context lines."
    )
    risk_level = "safe"
    category = "search"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex or text pattern to search."},
            "path": {"type": "string", "description": "Directory to search (default: .)."},
            "file_pattern": {"type": "string", "description": "File glob filter (e.g. '*.py')."},
            "include_hidden": {"type": "boolean", "description": "Include hidden files (default: false)."},
            "max_results": {"type": "integer", "description": "Max matches (default: 50)."},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".", file_pattern: str = "*",
                include_hidden: bool = False, max_results: int = 50) -> str:
        try:
            root = Path(path).expanduser().resolve()
            if not root.is_dir():
                return json.dumps({"error": f"Not a directory: {path}"})
            matches = []
            regex = re.compile(pattern, re.IGNORECASE)
            for filepath in sorted(root.rglob(file_pattern)):
                if not include_hidden and any(p.startswith(".") for p in filepath.parts):
                    continue
                if not filepath.is_file(): continue
                try:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(content.split("\n"), 1):
                        if regex.search(line):
                            matches.append({
                                "file": str(filepath.relative_to(root)),
                                "line": i,
                                "text": line.strip()[:200],
                            })
                            if len(matches) >= max_results:
                                break
                except (PermissionError, OSError): continue
                if len(matches) >= max_results: break
            return json.dumps({"pattern": pattern, "matches": matches, "count": len(matches)})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  13. JSON Processing
# ═══════════════════════════════════════════════════════════════

class JsonTool(Tool):
    name = "json_process"
    description = (
        "Parse, validate, format, query JSON data. Can extract values by key, "
        "flatten nested structures, compare two JSON objects."
    )
    risk_level = "safe"
    category = "utility"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["parse", "format", "query", "keys", "flatten", "count"],
                "description": "JSON operation.",
            },
            "data": {"type": "string", "description": "JSON string to process."},
            "key": {"type": "string", "description": "Key to query/extract."},
            "filepath": {"type": "string", "description": "Read JSON from file."},
            "indent": {"type": "integer", "description": "Indentation spaces (default: 2)."},
        },
        "required": ["operation"],
    }

    def execute(self, operation: str, data: str = "", key: str = "",
                filepath: str = "", indent: int = 2) -> str:
        try:
            if filepath:
                p = Path(filepath).expanduser().resolve()
                if not p.exists(): return json.dumps({"error": f"Not found: {filepath}"})
                data = p.read_text(encoding="utf-8", errors="replace")
            if not data:
                return json.dumps({"error": "Provide data or filepath"})

            if operation == "parse":
                parsed = json.loads(data)
                return json.dumps({"valid": True, "type": type(parsed).__name__,
                                   "keys": list(parsed.keys()) if isinstance(parsed, dict) else "N/A",
                                   "length": len(parsed) if isinstance(parsed, (list, dict)) else "N/A"})
            elif operation == "format":
                parsed = json.loads(data)
                return json.dumps({"formatted": json.dumps(parsed, indent=indent, ensure_ascii=False)})
            elif operation == "query" and key:
                parsed = json.loads(data)
                keys = key.split(".")
                val = parsed
                for k in keys:
                    if isinstance(val, dict):
                        val = val.get(k)
                    elif isinstance(val, list) and k.isdigit():
                        val = val[int(k)]
                    else:
                        return json.dumps({"error": f"Key not found: {k}"})
                return json.dumps({"key": key, "value": val})
            elif operation == "keys":
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return json.dumps({"keys": list(parsed.keys()), "count": len(parsed.keys())})
                return json.dumps({"error": "Not a JSON object"})
            elif operation == "flatten":
                parsed = json.loads(data)
                def flat(obj, prefix=""):
                    items = {}
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            items.update(flat(v, f"{prefix}.{k}" if prefix else k))
                    elif isinstance(obj, list):
                        for i, v in enumerate(obj):
                            items.update(flat(v, f"{prefix}[{i}]"))
                    else:
                        items[prefix] = obj
                    return items
                return json.dumps({"flattened": flat(parsed), "count": len(flat(parsed))})
            elif operation == "count":
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return json.dumps({"type": "object", "keys": len(parsed),
                                       "values": sum(len(v) if isinstance(v, (list, dict)) else 1 for v in parsed.values())})
                elif isinstance(parsed, list):
                    return json.dumps({"type": "array", "items": len(parsed)})
                return json.dumps({"type": type(parsed).__name__})
            return json.dumps({"error": f"Unknown op: {operation}"})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  14. Encoding & Hashing
# ═══════════════════════════════════════════════════════════════

class EncodingTool(Tool):
    name = "encoding_utils"
    description = (
        "Encode/decode base64, URL-encode, hash strings (MD5, SHA1, SHA256, SHA512). "
        "Useful for data transformation and security."
    )
    risk_level = "safe"
    category = "utility"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["base64_encode", "base64_decode", "url_encode", "url_decode",
                         "md5", "sha1", "sha256", "sha512"],
                "description": "Encoding/hashing operation.",
            },
            "text": {"type": "string", "description": "Text to encode/decode/hash."},
        },
        "required": ["operation", "text"],
    }

    def execute(self, operation: str, text: str) -> str:
        try:
            if operation == "base64_encode":
                encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
                return json.dumps({"input_length": len(text), "encoded": encoded})
            elif operation == "base64_decode":
                decoded = base64.b64decode(text).decode("utf-8")
                return json.dumps({"decoded": decoded, "length": len(decoded)})
            elif operation == "url_encode":
                encoded = urllib.parse.quote(text, safe="")
                return json.dumps({"encoded": encoded})
            elif operation == "url_decode":
                decoded = urllib.parse.unquote(text)
                return json.dumps({"decoded": decoded})
            elif operation in ("md5", "sha1", "sha256", "sha512"):
                h = hashlib.new(operation.replace("sha", "sha"))
                if operation == "md5":
                    h = hashlib.md5(text.encode("utf-8"))
                else:
                    h = hashlib.new(operation, text.encode("utf-8"))
                return json.dumps({"algorithm": operation, "hash": h.hexdigest()})
            return json.dumps({"error": f"Unknown: {operation}"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  15. Calculator & Math
# ═══════════════════════════════════════════════════════════════

class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate mathematical expressions safely. Supports basic math, "
        "trigonometric functions, logarithms, constants (pi, e)."
    )
    risk_level = "safe"
    category = "utility"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate."},
        },
        "required": ["expression"],
    }

    # Only allow safe math operations
    _SAFE_NAMES = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sum": sum, "len": len,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "log2": math.log2,
        "ceil": math.ceil, "floor": math.floor,
        "pi": math.pi, "e": math.e,
        "radians": math.radians, "degrees": math.degrees,
        "int": int, "float": float,
    }

    def execute(self, expression: str) -> str:
        try:
            # Replace common math notation
            expr = expression.replace("^", "**")
            # Only allow alphanumeric, math operators, dots, parens, spaces
            if re.search(r"[a-zA-Z_]", expr):
                # Check for safe names only
                names_used = set(re.findall(r"[a-zA-Z_]\w*", expr))
                for name in names_used:
                    if name not in self._SAFE_NAMES:
                        return json.dumps({"error": f"Disallowed: {name}"})
                result = eval(expr, {"__builtins__": {}}, self._SAFE_NAMES)
            else:
                result = eval(expr, {"__builtins__": {}}, {})

            if isinstance(result, float):
                if result == float('inf') or result == float('-inf') or result != result:
                    return json.dumps({"error": "Result is infinity or NaN"})
            return json.dumps({"expression": expression, "result": result, "type": type(result).__name__})
        except ZeroDivisionError:
            return json.dumps({"error": "Division by zero"})
        except Exception as e:
            return json.dumps({"error": f"Evaluation error: {e}"})


# ═══════════════════════════════════════════════════════════════
#  16. Timestamp & Date
# ═══════════════════════════════════════════════════════════════

class TimestampTool(Tool):
    name = "timestamp_utils"
    description = (
        "Get current timestamp, convert between timestamps and human-readable dates, "
        "calculate date differences, format dates."
    )
    risk_level = "safe"
    category = "utility"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["now", "from_unix", "to_unix", "diff", "format"],
                "description": "Timestamp operation.",
            },
            "timestamp": {"type": "number", "description": "Unix timestamp."},
            "date_string": {"type": "string", "description": "Date string (e.g. '2024-01-15')."},
            "format": {"type": "string", "description": "Date format string (default: '%Y-%m-%d %H:%M:%S')."},
            "timezone": {"type": "string", "description": "Timezone offset (e.g. '+07:00', default: local)."},
        },
        "required": ["operation"],
    }

    def execute(self, operation: str, timestamp: float = 0, date_string: str = "",
                fmt: str = "%Y-%m-%d %H:%M:%S", timezone: str = "") -> str:
        try:
            if operation == "now":
                now = datetime.now()
                return json.dumps({
                    "datetime": now.strftime(fmt),
                    "unix": int(now.timestamp()),
                    "iso": now.isoformat(),
                    "day_of_week": now.strftime("%A"),
                })
            elif operation == "from_unix" and timestamp is not None:
                dt = datetime.fromtimestamp(timestamp)
                return json.dumps({
                    "unix": timestamp,
                    "datetime": dt.strftime(fmt),
                    "iso": dt.isoformat(),
                    "relative": self._relative_time(timestamp),
                })
            elif operation == "to_unix" and date_string:
                for f in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"]:
                    try:
                        dt = datetime.strptime(date_string, f)
                        return json.dumps({"date": date_string, "unix": int(dt.timestamp())})
                    except ValueError: continue
                return json.dumps({"error": f"Cannot parse date: {date_string}"})
            elif operation == "diff":
                if not timestamp:
                    return json.dumps({"error": "Provide timestamp"})
                dt = datetime.fromtimestamp(timestamp)
                now = datetime.now()
                delta = now - dt if now > dt else dt - now
                days = delta.days
                hours = delta.seconds // 3600
                mins = (delta.seconds % 3600) // 60
                return json.dumps({"diff": f"{days}d {hours}h {mins}m",
                                   "past": now > dt, "total_hours": round(delta.total_seconds() / 3600, 1)})
            elif operation == "format" and date_string:
                for f in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                    try:
                        dt = datetime.strptime(date_string, f)
                        return json.dumps({"input": date_string, "formatted": dt.strftime(fmt)})
                    except ValueError: continue
                return json.dumps({"error": f"Cannot parse date: {date_string}"})
            return json.dumps({"error": f"Missing params for {operation}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @staticmethod
    def _relative_time(ts: float) -> str:
        diff = time.time() - ts
        if diff < 60: return f"{int(diff)}s ago"
        if diff < 3600: return f"{int(diff/60)}m ago"
        if diff < 86400: return f"{int(diff/3600)}h ago"
        return f"{int(diff/86400)}d ago"


# ═══════════════════════════════════════════════════════════════
#  17. Environment Variables
# ═══════════════════════════════════════════════════════════════

class EnvironmentTool(Tool):
    name = "environment"
    description = "Get, set, or list environment variables."
    risk_level = "moderate"
    category = "system"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["get", "set", "list", "path"],
                "description": "Environment operation.",
            },
            "name": {"type": "string", "description": "Variable name."},
            "value": {"type": "string", "description": "Variable value (for set)."},
        },
        "required": ["operation"],
    }

    def execute(self, operation: str, name: str = "", value: str = "") -> str:
        try:
            if operation == "get" and name:
                val = os.environ.get(name)
                if val is None:
                    return json.dumps({"error": f"Variable not set: {name}"})
                return json.dumps({"name": name, "value": val})
            elif operation == "set" and name:
                os.environ[name] = value
                return json.dumps({"ok": True, "name": name, "value": value})
            elif operation == "list":
                env_vars = {}
                for k in sorted(os.environ.keys()):
                    v = os.environ[k]
                    env_vars[k] = v[:200] if len(v) > 200 else v
                return json.dumps({"variables": env_vars, "count": len(env_vars)})
            elif operation == "path":
                paths = os.environ.get("PATH", "").split(os.pathsep)
                return json.dumps({"path": paths, "count": len(paths)})
            return json.dumps({"error": f"Missing params for {operation}"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  18. Diff Tool
# ═══════════════════════════════════════════════════════════════

class DiffTool(Tool):
    name = "diff"
    description = (
        "Compare two texts or two files and show the differences. "
        "Returns unified diff output."
    )
    risk_level = "safe"
    category = "text"
    parameters = {
        "type": "object",
        "properties": {
            "file1": {"type": "string", "description": "Path to first file."},
            "file2": {"type": "string", "description": "Path to second file."},
            "text1": {"type": "string", "description": "First text (if not using files)."},
            "text2": {"type": "string", "description": "Second text (if not using files)."},
            "context_lines": {"type": "integer", "description": "Context lines around changes (default: 3)."},
        },
    }

    def execute(self, file1: str = "", file2: str = "", text1: str = "",
                text2: str = "", context_lines: int = 3) -> str:
        try:
            import difflib

            t1 = text1
            t2 = text2
            name1 = "text1"
            name2 = "text2"

            if file1:
                p1 = Path(file1).expanduser().resolve()
                if not p1.exists(): return json.dumps({"error": f"Not found: {file1}"})
                t1 = p1.read_text(encoding="utf-8", errors="replace")
                name1 = p1.name
            if file2:
                p2 = Path(file2).expanduser().resolve()
                if not p2.exists(): return json.dumps({"error": f"Not found: {file2}"})
                t2 = p2.read_text(encoding="utf-8", errors="replace")
                name2 = p2.name

            if t1 is None or t2 is None:
                return json.dumps({"error": "Provide text or files to compare"})

            lines1 = t1.splitlines(keepends=True)
            lines2 = t2.splitlines(keepends=True)

            diff = list(difflib.unified_diff(lines1, lines2, fromfile=name1, tofile=name2, n=context_lines))
            diff_text = "".join(diff)

            if not diff_text:
                return json.dumps({"same": True, "message": "Files/texts are identical"})

            added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))

            return json.dumps({
                "same": False,
                "added_lines": added,
                "removed_lines": removed,
                "diff": diff_text[-8000:] if len(diff_text) > 8000 else diff_text,
                "truncated": len(diff_text) > 8000,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  19. File Permissions
# ═══════════════════════════════════════════════════════════════

class PermissionsTool(Tool):
    name = "file_permissions"
    description = "Get or change file/directory permissions (chmod)."
    risk_level = "moderate"
    category = "filesystem"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File or directory path."},
            "mode": {"type": "string", "description": "Permission mode (octal, e.g. '755')."},
        },
        "required": ["path"],
    }

    def execute(self, path: str, mode: str = "") -> str:
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists(): return json.dumps({"error": f"Not found: {path}"})
            st = p.stat()
            current_mode = oct(st.st_mode)[-3:]

            if mode:
                new_mode = int(mode, 8)
                p.chmod(new_mode)
                st = p.stat()
                updated_mode = oct(st.st_mode)[-3:]
                return json.dumps({"path": str(p), "old_mode": current_mode, "new_mode": updated_mode, "ok": True})
            return json.dumps({"path": str(p), "mode": current_mode, "type": "dir" if p.is_dir() else "file",
                               "readable": os.access(p, os.R_OK), "writable": os.access(p, os.W_OK),
                               "executable": os.access(p, os.X_OK)})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  20. Regex Tester
# ═══════════════════════════════════════════════════════════════

class RegexTool(Tool):
    name = "regex_tester"
    description = (
        "Test regular expressions against text. Shows all matches with groups, "
        "positions, and match details. Useful for debugging regex patterns."
    )
    risk_level = "safe"
    category = "text"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression pattern."},
            "text": {"type": "string", "description": "Text to test against."},
            "flags": {"type": "string", "description": "Regex flags: i=ignorecase, m=multiline, s=dotall (comma-separated)."},
        },
        "required": ["pattern", "text"],
    }

    def execute(self, pattern: str, text: str, flags: str = "") -> str:
        try:
            flag_val = 0
            if "i" in flags: flag_val |= re.IGNORECASE
            if "m" in flags: flag_val |= re.MULTILINE
            if "s" in flags: flag_val |= re.DOTALL

            regex = re.compile(pattern, flag_val)
            matches = []
            for m in regex.finditer(text):
                entry = {
                    "match": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                    "groups": list(m.groups()) if m.groups() else [],
                    "groupdict": m.groupdict() if m.groupdict() else {},
                }
                matches.append(entry)

            return json.dumps({
                "pattern": pattern,
                "flags": flags,
                "match_count": len(matches),
                "matches": matches[:50],
                "is_valid": True,
            })
        except re.error as e:
            return json.dumps({"pattern": pattern, "is_valid": False, "error": str(e)})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  Registry Builder
# ═══════════════════════════════════════════════════════════════

def build_registry() -> ToolRegistry:
    """Create registry with all 20 tools."""
    reg = ToolRegistry()
    for tool_cls in [
        ShellTool,            # 1  execute_shell
        FileSystemTool,       # 2  file_system
        WebSearchTool,        # 3  web_search
        CodeAnalysisTool,     # 4  analyze_code
        SystemInfoTool,       # 5  system_info
        GitTool,              # 6  git_operations
        HttpRequestTool,      # 7  http_request
        TextProcessTool,      # 8  text_process
        ProcessManagerTool,   # 9  process_manager
        PackageManagerTool,   # 10 package_manager
        NetworkTool,          # 11 network_diagnostics
        GrepTool,             # 12 grep_files
        JsonTool,             # 13 json_process
        EncodingTool,         # 14 encoding_utils
        CalculatorTool,       # 15 calculator
        TimestampTool,        # 16 timestamp_utils
        EnvironmentTool,      # 17 environment
        DiffTool,             # 18 diff
        PermissionsTool,      # 19 file_permissions
        RegexTool,            # 20 regex_tester
    ]:
        reg.register(tool_cls())
    return reg
