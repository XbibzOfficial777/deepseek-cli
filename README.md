<div align="center">

# 🧠 DeepSeek CLI Agent v7.7
### *The Ultimate Multi-Provider AI Agent for Developers*
[![deepseek-cli](https://imgbs.com/uploads/670312-c65bfd00.png)](https://deepseek-cli.pages.dev)

[![Version](https://img.shields.io/badge/Version-7.7.0-00FFA3.svg?style=for-the-badge)](https://github.com/XbibzOfficial777/deepseek-cli)
[![License](https://img.shields.io/badge/License-MIT-white.svg?style=for-the-badge)](https://github.com/XbibzOfficial777/deepseek-cli/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Termux-FF6F00.svg?style=for-the-badge)](https://github.com/XbibzOfficial777/deepseek-cli)
[![Tools](https://img.shields.io/badge/Tools-120%2B-8A2BE2.svg?style=for-the-badge)](https://github.com/XbibzOfficial777/deepseek-cli)

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-session-management">Sessions</a> •
  <a href="#-multi-agent-system">Multi-Agent</a> •
  <a href="#-mcp-servers">MCP</a> •
  <a href="#-supported-providers">Providers</a> •
  <a href="#-command-reference">Commands</a> •
  <a href="#-architecture">Architecture</a>
</p>

</div>

---

**DeepSeek CLI** is a production-grade, autonomous AI Agent engineered to streamline development workflows. Powered by an advanced **agentic loop**, it doesn't just chat — it **reasons, plans, delegates, and executes** tasks using a suite of **120+ built-in tools** and **external MCP servers**.

> ⚡ *Designed for developers who demand speed, autonomy, and elegance in their terminal.*

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🧠 Autonomous Reasoning** | Multi-step planning engine with anti-stuck detection, tool loop prevention, and smart round limits |
| **🛠️ 120+ Integrated Tools** | File ops, web search, code execution, OCR, PDF, DOCX, PPTX, XLSX, image processing, browser automation, and more |
| **🌐 Multi-Provider** | Native support for 8 AI providers — switch between OpenRouter, Gemini, Anthropic, Groq, OpenAI, Together AI, HuggingFace, and Agnes AI |
| **🎨 Rich Markdown UI** | Professional terminal interface with real-time streaming, syntax-highlighted code blocks, and styled output |
| **📱 Mobile Ready** | Fully optimized for Android Termux and low-latency environments (threading-based timeouts, paste detection) |
| **🔗 Telegram & Discord** | Built-in connectors to run your AI agent through Telegram bots and Discord channels |
| **🔄 Smart Loop Protection** | Anti-stuck detection, tool loop prevention, automatic timeout safeguards, and repeated-content detection |
| **💾 Session Management** | Auto-save conversations, resume with `dscli -s`, delete with `dscli -d` |
| **🧪 Multi-Agent System** | 5 specialized agent profiles (coder, researcher, filesystem, reasoner, general) with delegation & concurrent execution |
| **🔌 MCP Client** | Built-in Model Context Protocol client — connect to external MCP servers (Context7, GitHub, Canva, Brave Search, SQLite, Puppeteer, etc.) |
| **📊 Session Logging** | Automatic metrics logging with tool usage statistics, latency tracking, and turn history |
| **🌐 Web Browser Automation** | Two browser engines — HTTP-based (browser_navigate, browser_login, etc.) and Selenium Firefox-based (full DOM interaction, screenshots, login flows) |
| **🤖 Auth Automation** | Google OAuth, generic login flows, 2FA/OTP handling, CAPTCHA detection, GUI fallback |
| **📄 Full Document Suite** | Read/create/edit for PDF, DOCX, PPTX, XLSX, CSV, plus document format conversion |
| **🔍 OCR Support** | Text extraction from images via Tesseract + EasyOCR fallback |

---

## 🚀 Quick Start

### One-Line Installation

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/XbibzOfficial777/deepseek-cli/master/install.sh)"
```

### Manual Installation

```bash
git clone https://github.com/XbibzOfficial777/deepseek-cli.git
cd deepseek-cli
bash install.sh
```

### Uninstall

```bash
bash install.sh --uninstall
```

### First Launch

```bash
dscli
```

On first run, set your API key:

```bash
# Inside the REPL:
/key                             # Set API key for current provider
/provider openrouter              # Switch provider
/model deepseek/deepseek-r1-0528:free  # Select a model
```

> 💡 **Pro tip:** Press `Ctrl+P` to open the interactive settings panel with arrow-key navigation.

---

## 💾 Session Management

Auto-named sessions with full save/resume/delete lifecycle:

```bash
dscli                          # Auto-creates session: dscli-xxxxxxxxxxxx
dscli list session             # List all saved sessions
dscli -l                       # Same as above
dscli -s dscli-xxxxxxxxxxxx    # Resume a session
dscli -d dscli-xxxxxxxxxxxx    # Delete a session
```

From inside the REPL:
- `/session` — list all sessions
- `/session delete <id>` — delete a session
- **Ctrl+X** — toggle session info view
- **Up arrow** (while in session view) — go back
- Sessions auto-saved to `~/.deepseek-cli/sessions/` after every turn

---

## 🧪 Multi-Agent System

5 specialized agent profiles with delegation and concurrent execution:

### Profiles

| Profile | Focus Area |
| :--- | :--- |
| `general` | All-purpose assistant with full tool access |
| `coder` | Programming, debugging, code review |
| `researcher` | Web search, fact-finding, research |
| `filesystem` | File management and data processing |
| `reasoner` | Step-by-step analytical reasoning |

### Commands

```bash
/agent              # Interactive profile selector
/agent list         # List all profiles
/agent coder        # Switch to Code Specialist
```

### Tool-Based Delegation

The AI can delegate tasks to sub-agents using tools:

- `delegate(profile, task)` — spawns a sub-agent, returns result (blocking)
- `delegate_concurrent(tasks[])` — runs multiple agents in parallel via thread pool

### Architecture

```
deepseek/multi_agent.py
├── AGENT_PROFILES     — 5 profile definitions with system prompt extras
├── AgentWorker        — Isolated agent with own memory, provider, model
│   └── run()          — Full LLM→tool→result loop (max 6 rounds)
├── MultiAgentManager  — Singleton managing profiles, delegation, concurrent exec
│   ├── delegate()     — Blocking delegation
│   ├── delegate_async() — Non-blocking (thread-based)
│   └── run_concurrent() — Parallel execution via ThreadPoolExecutor
```

---

## 🔌 MCP Servers

Built-in real MCP (Model Context Protocol) client. Connect to external servers and use their tools automatically.

### Popular Servers

| Server | Description | Auth |
|--------|-------------|------|
| `context7` | Live library documentation | None |
| `canva` | Create/edit Canva designs | CANVA_API_KEY |
| `github` | Repos, issues, PRs | GITHUB_TOKEN |
| `brave-search` | Web search | BRAVE_API_KEY |
| `filesystem` | Secure file access | None |
| `memory` | Knowledge graph | None |
| `fetch` | HTTP scraping | None |
| `sqlite` | Database queries | None |
| `postgres` | PostgreSQL | DATABASE_URL |
| `puppeteer` | Browser automation | None |
| `sequential-thinking` | Reasoning | None |

### Architecture

```
deepseek/mcp_client.py
├── POPULAR_MCP_SERVERS  — Registry of 13+ pre-configured servers
├── MCPConnection        — Single server connection (stdio/SSE transport)
│   ├── _run_stdio_session()  — stdio transport via asyncio
│   ├── _run_sse_session()    — SSE transport
│   └── _call_tool_async()    — Tool invocation
├── MCPClientManager     — Singleton managing multiple connections
│   ├── connect_server()   — Establish connection (async in background thread)
│   ├── call_tool()        — Route tool calls to correct server
│   ├── get_all_tools()    — Aggregate all available tools
│   └── get_status()       — Connection status overview
└── mcp_manager          — Global singleton instance
```

### Usage

```bash
/mcp                          # Open MCP management menu
/mcp list                     # List popular servers
/mcp connect context7         # Connect to Context7
/mcp connect canva            # Connect to Canva (enter CANVA_API_KEY)
/mcp status                   # Show connected servers
/mcp disconnect context7      # Disconnect
```

Connected MCP servers automatically register their tools into the agent's toolset.

---

## 🔌 Supported Providers

| Provider | Type | Base URL | Default Model | Free Models |
| :--- | :--- | :--- | :--- | :--- |
| **OpenRouter** | OpenAI-compatible | `openrouter.ai` | `deepseek/deepseek-r1-0528:free` | ✅ |
| **Google Gemini** | Gemini API | `generativelanguage.googleapis.com` | `gemini-2.5-flash-preview-05-20` | ✅ |
| **Anthropic** | Anthropic API | `api.anthropic.com` | `claude-sonnet-4-20250514` | ❌ |
| **Groq** | OpenAI-compatible | `api.groq.com` | `llama-3.3-70b-versatile` | ✅ |
| **Together AI** | OpenAI-compatible | `api.together.xyz` | `meta-llama/Llama-3-70b-chat-hf` | ✅ |
| **HuggingFace** | HuggingFace Router | `router.huggingface.co` | `Qwen/Qwen2.5-72B-Instruct` | ✅ |
| **OpenAI** | OpenAI API | `api.openai.com` | `gpt-4.1-mini` | ❌ |
| **Agnes AI** | OpenAI-compatible | `apihub.agnes-ai.com` | `agnes-2.0-flash` | ✅ |

> All 8 providers support **tool/function calling** and **real-time streaming** (thinking + response).

### Provider Architecture

```
deepseek/providers.py
├── BaseProvider              — Abstract base (chat_stream, fetch_models, validate_key)
├── OpenAICompatibleProvider  — OpenRouter, OpenAI, Groq, Together (shared impl)
├── GeminiProvider            — Google Gemini (generateContent + _convert_tools)
├── AnthropicProvider         — Anthropic Claude (messages API)
├── HuggingFaceProvider       — HuggingFace (prompt-based tool calling)
└── create_provider()         — Factory function for provider instantiation
```

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────────┐
│   User      │ ──▶ │  Agent   │ ──▶ │  Multi-Agent │ ──▶ │  External    │
│  Input      │     │  Engine  │     │   Manager    │     │  MCP Servers │
└─────────────┘     └──────────┘     └──────────────┘     └──────────────┘
                          │                  │
                          ▼                  ▼
                    ┌──────────┐     ┌──────────────┐
                    │  Memory  │     │  Tool Registry│
                    │ (Context)│     │  (120+ Tools) │
                    └──────────┘     └──────────────┘
                          │                  │
                          ▼                  ▼
                    ┌──────────┐     ┌──────────────┐
                    │ Provider │     │  Executor    │
                    │   API    │     │ (Safe Timeout)│
                    └──────────┘     └──────────────┘
```

### File Structure

```
deepseek/
├── __main__.py          — CLI entrypoint (argparse, session commands, uninstall)
├── agent.py             — Agentic loop, AntiStuckDetector, ThinkTagStreamParser,
│                          AgentMetrics logging, safe_parse_json, safe_execute
├── repl.py              — Interactive REPL (main loop, slash commands, Ctrl+P panel)
├── providers.py         — 8 AI provider adapters (OpenAI-compatible, Gemini, Anthropic, HF)
├── toolkit.py           — 120+ tool registry with Pydantic validation layer
├── config.py            — YAML config manager, multi-provider settings, API keys
├── memory.py            — Conversation Memory (system prompt, messages, session persistence)
├── multi_agent.py       — Agent profiles, AgentWorker, MultiAgentManager
├── mcp_client.py        — MCP protocol client (stdio/SSE, connection management)
├── mcp_tools.py         — MCP tool registration into ToolRegistry
├── doc_tools.py         — PDF, DOCX, PPTX, XLSX, CSV tools + document conversion
├── selenium_browser.py  — Selenium Firefox browser automation (2700+ lines)
├── webcontrol.py        — HTTP-based browser automation (navigate, login, click, extract)
├── connectors.py        — Telegram & Discord bot connectors (httpx-based, no extra deps)
├── ui.py                — Rich terminal UI (StreamRenderer, banner, spinner, raw input)
└── llm.py               — Backward-compatibility shim
```

### Execution Flow

1. **🧠 Perception** — Agent receives user intent and analyzes context
2. **📋 Planning** — LLM generates a multi-step execution strategy (optional planner layer)
3. **🧪 Delegation** — Agent can delegate sub-tasks to specialized sub-agents
4. **⚡ Action** — Agent invokes specialized tools with thread-safe timeout protection
5. **👁️ Observation** — Results are fed back into memory for self-correction
6. **🔁 Loop** — Repeat until task is complete or anti-stuck safety triggers

### Smart Loop Controls (v5.5+)

| Setting | Default | Description |
|---------|---------|-------------|
| `UNLIMITED_TOOL_ROUNDS` | `True` | No hard cap on tool rounds — loop ends when model returns final answer |
| `MAX_SAME_TOOL` | `50` | Anti-stuck: stop if same tool called 50 times without progress |
| `MAX_REPEATED_CONTENT` | `3` | Anti-stuck: stop if 3 consecutive identical content outputs |
| `TOOL_TIMEOUT_DEFAULT` | `90s` | Per-tool timeout via threading (Termux-compatible) |
| `safe_parse_json` | `2 retries` | JSON parse with auto-fix (missing braces) and retry |

---

## ⌨️ Command Reference

### Navigation

| Key | Action |
| :--- | :--- |
| `Ctrl+P` | Open interactive settings panel |
| `Ctrl+X` | Toggle session info view (press again or Up to go back) |

### Slash Commands

| Command | Description |
| :--- | :--- |
| `/help` | Show help and tool documentation |
| `/provider` | Switch AI provider (interactive menu) |
| `/model` | Change active LLM model (interactive menu) |
| `/agent` | Switch multi-agent profile (coder, researcher, etc.) |
| `/key` | Set API key for current provider |
| `/tools` | List all 120+ available tools |
| `/thinking` | Toggle visibility of internal reasoning |
| `/clear` | Reset conversation memory |
| `/export` | Export chat session to text file |
| `/info` | Show current configuration |
| `/compact` | Compact conversation (keep system + last 10) |
| `/session` | List/manage saved sessions |
| `/live_search <query>` | Real-time web search (multi-source) |
| `/live_models` | Fetch all models from provider API |
| `/search_model` | Search/filter models from provider API |
| `/mcp` | Manage MCP server connections |
| `/telegram` | Manage Telegram bot connector |
| `/discord` | Manage Discord bot connector |
| `/exit` | Exit the application |

---

## 🛠️ Tool Ecosystem

### 📁 File Operations
`read_file` · `write_file` · `edit_file` · `list_files` · `delete_file` · `file_info` · `search_files` · `tree_view`

### 🌐 Web & Search
`web_search` · `web_fetch` · `live_search` · `browser_navigate` · `browser_click` · `browser_fill_form` · `browser_snapshot` · `browser_login` · `browser_screenshot` · `browser_cookies`

### 💻 Code Execution
`run_code` · `run_shell` · `install_package`

### 🖼️ Image & OCR
`image_view` · `image_info` · `ocr_read` · `ocr_url`

### 📄 Documents
`read_pdf` · `create_pdf` · `pdf_edit` · `read_docx` · `create_docx` · `docx_info` · `edit_docx` · `read_pptx` · `create_pptx` · `edit_pptx` · `pptx_info` · `read_xlsx` · `create_xlsx` · `edit_xlsx` · `xlsx_info` · `read_csv` · `create_csv` · `convert_document`

### 🔬 System & Analysis
`system_info` · `process_list` · `disk_usage` · `network_info` · `env_vars` · `apk_analyze` · `video_info`

### 🔧 Utilities
`calculate` · `unit_convert` · `timestamp` · `text_transform` · `json_parse` · `regex_test` · `base64_tool` · `generate_uuid` · `random_number` · `sort_data` · `count_text` · `hash_tool`

### 🌤️ Real-Time Data
`get_datetime` · `get_calendar` · `get_weather` · `get_news` · `get_stock_price` · `get_crypto_price` · `get_currency_rate` · `get_holidays` · `get_qibla` · `get_countdown` · `get_sun_times` · `get_day_info` · `get_ip_info` · `get_random_fact`

### 🤖 Multi-Agent Delegation
`delegate(profile, task)` · `delegate_concurrent(tasks[])`

### 🧪 Advanced Automation
**Selenium Browser:** Navigate, click, type, scroll, login, screenshot, cookie management, file upload, frame switching, popup handling, tab switching, auth automation

**Auth Automation:** Google OAuth, generic login, 2FA/OTP handling, CAPTCHA detection, GUI fallback, cookie export/import

### 🤖 Connectors
**Telegram Bot:** Message relay, markdown support, user whitelist, long message splitting, file handling

**Discord Bot:** Channel messaging, webhook integration, user whitelist

---

## 📊 Session Metrics

Automatic logging saves detailed metrics to `~/.deepseek-cli/logs/`:

```json
{
  "session_id": "20260602_121114",
  "timestamp": "2026-06-02T12:11:14",
  "total_turns": 15,
  "total_tool_calls": 42,
  "total_errors": 0,
  "avg_latency": 1.84,
  "tool_usage": {
    "web_search": 8,
    "read_file": 12,
    "run_code": 5
  },
  "turns": [...]
}
```
Full conversation persistence at `~/.deepseek-cli/sessions/` — resume any session with `dscli -s <session_id>`.

---

## 📱 Termux Optimization

DeepSeek CLI is fully optimized for Android Termux:

- ✅ Threading-based timeouts (no `signal.SIGALRM` — works on Termux/Android)
- ✅ Paste detection with intelligent line joining
- ✅ Arrow-key and Ctrl shortcuts fully supported
- ✅ Low-bandwidth streaming compatible
- ✅ No system dependencies required
- ✅ Raw terminal I/O with escape sequence handling for both CSI and SS3 formats
- ✅ Real-time streaming with 20ms buffered flush intervals

---

## 🧪 Testing

```bash
cd deepseek-cli
python -m pytest tests/ -v
```

| Test File | Coverage |
|-----------|----------|
| `tests/test_fixes.py` | 33+ bug-fix tests — each class maps to a FIX ID in VERIFICATION_REPORT.md |
| `tests/test_agent.py` | `safe_parse_json` and `AntiStuckDetector` unit tests |

### Key Bug Fixes (v5.5+)

| Fix | Area | Description |
|-----|------|-------------|
| Gemini tool calling | `providers.py` | `_convert_tools()` returns single `functionDeclarations` object |
| Pydantic validation | `toolkit.py` | `validate_args()` supports both v1 (`.dict()`) and v2 (`.model_dump()`) |
| Live search URL | `toolkit.py` | News results use `'url'` key with `'href'` fallback |
| AgentWorker loop | `multi_agent.py` | Full tool-result feedback loop (max 6 rounds) |
| Doc tools imports | `doc_tools.py` | `Emu`/`Pt`/`Font` at module level, not local scope |
| Selenium duplicates | `selenium_browser.py` | No duplicate method definitions |

---

## 📁 Configuration

Config file: `~/.deepseek-cli/config.yaml`

| Setting | Description |
|---------|-------------|
| `active_provider` | Current AI provider ID |
| `api_keys` | Per-provider API key storage |
| `models` | Per-provider model selection |
| `providers` | Full provider configurations |
| `connectors` | Telegram/Discord bot tokens |
| `mcp_servers` | Connected MCP server configs |

---

## 🤝 Contributing

Contributions are what make the open-source community amazing. Any contributions you make are **greatly appreciated**.

1. 🍴 Fork the Project
2. 🌿 Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. ✅ Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the Branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📦 Dependencies

### Core (required)
`httpx` · `rich` · `pyyaml` · `pydantic`

### Documents
`PyPDF2` · `reportlab` · `python-docx`

### Images & OCR
`Pillow` · `pytesseract`

### Web Scraping
`beautifulsoup4` · `lxml`

### Browser Automation
`selenium` · `webdriver-manager`

### MCP
`mcp`

### Optional
`easyocr` (OCR fallback — install manually if needed)

---

<div align="center">

**Created with 💚 by [XbibzOfficial](https://github.com/XbibzOfficial777)**

[![GitHub Stars](https://img.shields.io/github/stars/XbibzOfficial777/deepseek-cli?style=social)](https://github.com/XbibzOfficial777/deepseek-cli)
[![GitHub Forks](https://img.shields.io/github/forks/XbibzOfficial777/deepseek-cli?style=social)](https://github.com/XbibzOfficial777/deepseek-cli)

## Stargazers over time
[![Stargazers over time](https://starchart.cc/XbibzOfficial777/deepseek-cli.svg?variant=adaptive)](https://starchart.cc/XbibzOfficial777/deepseek-cli)

[⬆ Back to Top](#-deepseek-cli-agent-v77)

</div>
