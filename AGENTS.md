# DeepSeek CLI Agent — Repo Guide (v7.7)

## Structure

Single `deepseek/` Python package (Python 3.8+). **No `pyproject.toml`** — only `requirements.txt`.

- `deepseek/__main__.py` — CLI entrypoint (`python -m deepseek` or `dscli`)
- `deepseek/agent.py` — Agentic loop, AntiStuckDetector, tool-calling loop
- `deepseek/repl.py` — Interactive REPL (main loop, slash commands, Ctrl+P panel)
- `deepseek/providers.py` — 8 AI provider adapters
- `deepseek/toolkit.py` — 120+ tool registry with Pydantic validation
- `deepseek/doc_tools.py` — PDF, DOCX, PPTX, XLSX tools
- `deepseek/selenium_browser.py` — Selenium Firefox automation (~2700 lines)
- `deepseek/mcp_client.py` / `deepseek/mcp_tools.py` — MCP protocol client
- `deepseek/multi_agent.py` — Sub-agent delegation (5 profiles)
- `deepseek/connectors.py` — Telegram & Discord connectors
- `deepseek/tools.py` — **Deprecated** (warns, delegates to `toolkit.py`)
- `deepseek/llm.py` — Backward compat shim (re-exports `create_provider` only)

## Commands

| Command | Action |
|---------|--------|
| `dscli` or `python -m deepseek` | Launch REPL |
| `dscli -s <session_id>` | Resume session |
| `dscli -l` / `dscli list session` | List sessions |
| `dscli -d <session_id>` | Delete session |
| `dscli install <package>` | Install skill via npx |
| `bash install.sh` | Install |
| `bash install.sh --uninstall` | Uninstall |

## Architecture quirks

- **No tests directory exists in this repo** — VERIFICATION_REPORT.md references `tests/` but it was never committed. `python -m pytest` will fail.
- **`setup_deploy.py` contains hardcoded secrets** — GH_TOKEN, CF_TOKEN, ADMIN_PASSCODE are baked in. Listed in `.gitignore` but still in working tree. **Do not commit.**
- **`dashboard/`** — separate Cloudflare Workers project (React + Vite + React Native Web). Listed in `.gitignore`.
- **`dscli` wrapper** — sets `DEEPSEEK_ORIGINAL_CWD` to PWD before `cd` to install dir.
- **`tools.py`** — deprecated. Import `ToolRegistry` from `toolkit.py` instead.

## Config & Storage

- Config: `~/.deepseek-cli/config.yaml` — provider config, API keys, model selection
- Sessions: `~/.deepseek-cli/sessions/` — auto-saved every turn
- Metrics: `~/.deepseek-cli/logs/` — JSON log per session
- `MAX_TOOL_ROUNDS` from config (default `0` = unlimited)
- `TOOL_TIMEOUT_DEFAULT = 0` (no timeout; AI determines execution time)

## .gitignore gotchas

- `*test` — matches any file/dir ending in "test" (including `tests/`)
- `AGENTS.MD` (uppercase) vs `AGENTS.md` (lowercase)
- `setup_deploy.py` and `dashboard/` are gitignored but present in working tree

## Dependencies

Core: `httpx`, `rich`, `pyyaml`, `pydantic`, `duckduckgo-search`. Optional: `PyPDF2`, `reportlab`, `python-docx`, `Pillow`, `pytesseract`, `selenium`, `webdriver-manager`, `mcp`, `openpyxl`, `matplotlib`, `python-pptx`.

## All VERIFICATION_REPORT.md fixes are applied

The 22 fixes (Gemini `_convert_tools()`, Pydantic v1/v2 fallback, `_live_search` URL key, `AgentWorker.run()` loop, Emu/Pt/Font imports, Selenium redefinitions, dead code cleanup) are **already in the committed code**. Do not re-apply.

## Dashboard deployment

The Cloudflare Workers dashboard (`dashboard/`) is at https://deepseek-dashboard.bibzflow.workers.dev.

Deploy flow: `npm run build` → `python3 setup_deploy.py` → `npx wrangler deploy`.

Previously fixed bugs:
1. **`setup_deploy.py` placeholder names mismatched** — replaced `__HTML_CONTENT__` but template uses `__HTML_B64__`. Now fixed to use base64-encoded replacements.
2. **`vite.config.js` broken `Platform.select` define** — `JSON.stringify(function(){})` returned `undefined` (no-op). Removed (unnecessary with react-native-web).
