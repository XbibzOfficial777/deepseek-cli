# DeepSeek CLI Agent — Repo Guide (v7.7)

## Structure

Single `deepseek/` Python package (Python 3.8+). Not a monorepo.

- `deepseek/__main__.py` — CLI entrypoint (`python -m deepseek` or `dscli`)
- `deepseek/agent.py` — Agentic loop, AntiStuckDetector, tool-calling loop
- `deepseek/repl.py` — Interactive REPL
- `deepseek/providers.py` — 8 AI provider adapters (OpenRouter, Gemini, Anthropic, Groq, etc.)
- `deepseek/toolkit.py` — 120+ tool registry
- `deepseek/config.py` — YAML config at `~/.deepseek-cli/config.yaml`
- `deepseek/memory.py` — Session persistence to `~/.deepseek-cli/sessions/`
- `deepseek/doc_tools.py` — PDF, DOCX, PPTX, XLSX tools
- `deepseek/multi_agent.py` — Sub-agent delegation
- `deepseek/mcp_client.py` / `deepseek/mcp_tools.py` — MCP protocol client
- `deepseek/selenium_browser.py` — Browser automation
- `deepseek/connectors.py` — Telegram & Discord connectors

## Key commands

- `python -m deepseek` or `dscli` — launch REPL
- `dscli -s <session_id>` — resume session
- `dscli -l` or `dscli list session` — list sessions
- `dscli -d <session_id>` — delete session
- `python -m pytest tests/ -v` — run all tests (37 fix tests + agent tests)
- `bash install.sh` — install; `bash install.sh --uninstall` — uninstall
- `python -m deepseek` — launch directly without install

## Working tree note

Source files were deleted from working tree (git showed `D`); restored from HEAD during bugfix session.

## Dependencies (from `requirements.txt`)

Core: `httpx`, `rich`, `pyyaml`, `pydantic`. Optional: `PyPDF2`, `reportlab`, `python-docx`, `Pillow`, `pytesseract`, `beautifulsoup4`, `lxml`, `selenium`, `webdriver-manager`, `mcp`.

## Testing quirks

- `tests/test_fixes.py` (33 tests) covers bug fixes from `VERIFICATION_REPORT.md`. Each class maps to one FIX ID.
- `tests/test_agent.py` tests `safe_parse_json` and `AntiStuckDetector`.
- Tests set `sys.path` to parent dir before importing `deepseek.*`.

## Known bugs / fixes (from VERIFICATION_REPORT.md)

- Gemini tool calling: `_convert_tools()` must return one `{"functionDeclarations": [...]}` object, not multiple
- `validate_args()` needs Pydantic v1 fallback (`.dict()` if `.model_dump()` missing)
- `_live_search()` news: use `r.get('url', '') or r.get('href', '')`
- `AgentWorker.run()` must loop (max 6 rounds) with `memory.add_tool_result()`
- `doc_tools`: `Emu`/`Pt`/`Font` must be imported at module level, not local scope
- Selenium: no duplicate method definitions (`switch_to_frame`, `handle_popup`, etc.)

## Config

- `~/.deepseek-cli/config.yaml` — stores provider config, API keys, model selection
- `MAX_TOOL_ROUNDS` from config (but actual loop is unlimited; anti-stuck at 50 same-tool / 3 repeated-content)
- Sessions auto-saved to `~/.deepseek-cli/sessions/`
- Metrics logged to `~/.deepseek-cli/logs/`
