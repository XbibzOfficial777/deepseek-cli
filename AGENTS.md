# DeepSeek CLI Agent — Repository Guide (v7.7)

> **Panduan lengkap** untuk kontributor, AI agents (Claude/Codex/GPT), dan maintainer
> yang bekerja di repo ini. Versi: **v7.7** · Update: 2026-06-23

---

## 📑 Daftar Isi

1. [Arsitektur & Workflow Logic](#-arsitektur--workflow-logic)
2. [Struktur File Lengkap](#-struktur-file-lengkap)
3. [Data Flow End-to-End](#-data-flow-end-to-end)
4. [Sistem Update & Versi](#-sistem-update--versi)
5. [Autentikasi (Firebase)](#-autentikasi-firebase)
6. [Dashboard Cloudflare Worker](#-dashboard-cloudflare-worker)
7. [Multi-Agent System](#-multi-agent-system)
8. [MCP (Model Context Protocol)](#-mcp-model-context-protocol)
9. [Providers (8 AI)](#-providers-8-ai)
10. [Toolkit (120+ Tools)](#-toolkit-120-tools)
11. [Session & Memory](#-session--memory)
12. [Build / Deploy](#-build--deploy)
13. [Risiko & Performa](#-risiko--performa)
14. [Aturan Contribution](#-aturan-contribution)
15. [Troubleshooting](#-troubleshooting)

---

## 🏗 Arsitektur & Workflow Logic

**DeepSeek CLI** (`dscli`) adalah **autonomous terminal AI agent** yang berjalan secara
**agentic loop**: menerima intent user → reasoning → eksekusi tools → observasi hasil →
loop sampai selesai. Pasangannya adalah **Cloudflare Worker dashboard** (`dashboard-react/`)
yang menampilkan data usage + manajemen user secara real-time.

### Arsitektur 3-Layer

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1 — User Interface (REPL)                                  │
│  • Input: prompt + slash commands + raw key bindings              │
│  • Output: streaming rich markdown (think + content + tools)      │
│  • File: deepseek/repl.py (2186 lines)                            │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2 — Agent Engine (the brain)                                │
│  • agent.py: agentic loop, AntiStuckDetector, metrics             │
│  • memory.py: conversation history + session persistence          │
│  • multi_agent.py: 5 specialized profiles + delegation            │
│  • planner.py: task decomposition (optional pre-pass)             │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3 — Capabilities (tools & external)                         │
│  • toolkit.py: 120+ built-in tools (Pydantic-validated)           │
│  • providers.py: 8 AI providers (OpenAI/Gemini/Anthropic/etc)     │
│  • mcp_client.py: external MCP servers (Context7, GitHub, etc)    │
│  • doc_tools.py: PDF/DOCX/PPTX/XLSX/CSV                           │
│  • selenium_browser.py + webcontrol.py: browser automation        │
│  • connectors.py: Telegram & Discord bot relays                   │
└──────────────────────────────────────────────────────────────────┘
```

### Workflow Logic (sequence diagram)

```
User → REPL.prompt()
  │
  ▼
auth.login() ─── DEEPSEEK_SKIP_AUTH=1 → skip
  │ (Firebase Identity Toolkit + RTDB mirror)
  ▼
__main__.enforce_gist() ─────▶ ban? → sys.exit(1)
  │                              limit? → sys.exit(1)
  │ (registry Gist: endpoint.json)
  ▼
ui.show_banner() ─────▶ "Update Available vX.Y" notice
  │
  ▼
repl.main() ──── loop forever:
  │                  │
  │   ┌──────────────┴────────────────┐
  │   ▼                               │
  │ Agent.chat(user_msg)              │
  │   ├─ memory.add_user(msg)         │
  │   ├─ reasoning pre-pass (optional)│
  │   │                               │
  │   │ ╔══ AGENTIC LOOP ══════╗      │
  │   │ ║ while not done:      ║      │
  │   │ ║   1. provider.chat_  ║      │
  │   │ ║      stream()        ║◀─────┤ streaming
  │   │ ║   2. parse: think /  ║      │
  │   │ ║      content / tools ║      │
  │   │ ║   3. if tools → exec ║──────┼──▶ toolkit.execute(tool)
  │   │ ║   4. feed results    ║      │    (timeout via threading)
  │   │ ║      back to memory  ║◀─────┘    (Pydantic validate args)
  │   │ ║   5. anti-stuck check║
  │   │ ╚══════════════════════╝
  │   └─ metrics.log(turn)
  ▼
ui.render_stream()
  │
  ▼
save session → ~/.deepseek-cli/sessions/<id>.json
  │
  ▼
loop back to User prompt
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Threading, not signal.SIGALRM** | Termux/Android compatibility — SIGALRM tidak reliable di mobile |
| **Pure stdlib auth** | No firebase-admin SDK — Termux-friendly, no extra deps |
| **Two-Gist split** | Registry Gist (public) + Usage Gist (private) — clear separation |
| **Merge-only registry writes** | Avoid "tertumpuk" duplicate Gists when admin POSTs version |
| **Pydantic v1 + v2 compat** | `.dict()` + `.model_dump()` — works with both versions |
| **React + Vite SPA** | Dashboard built with React + TypeScript; built into `./dist/` and served via Cloudflare Worker `[assets]` binding |
| **`/api/admin/data` polled** | Dashboard polls every 15s (configurable 5/15/30s) for near-real-time without WebSocket complexity |

---

## 📁 Struktur File Lengkap

```
deepseek-cli/
├── README.md                          # Public-facing docs (badge-heavy)
├── AGENTS.md                          # ← file ini (panduan kontributor)
├── CLAUDE.md                          # Quick ref untuk Claude Code / AI agents
├── ERROR_HANDLING.md                  # Error contract documentation
├── PROJECT_ANALYSIS.md                # Internal analysis & bug audit
├── VERIFICATION_REPORT.md             # 22 bug fixes (all applied)
│
├── install.sh                         # Cross-platform installer (Linux/macOS/Termux)
├── requirements.txt                   # Python deps (pip install)
├── converted_from_csv.xlsx            # Sample data for doc_tools tests
│
├── deepseek/                          # Main Python package (Python 3.8+)
│   ├── __init__.py                    # 18 lines — cleanup legacy dirs
│   ├── __main__.py                    # 160 lines — CLI entrypoint, argparse
│   ├── agent.py                       # 1214 lines — agentic loop, anti-stuck
│   ├── repl.py                        # 2186 lines — interactive REPL
│   ├── providers.py                   # 847 lines — 8 AI providers
│   ├── toolkit.py                     # 4550 lines — 120+ tools + Pydantic
│   ├── memory.py                      # 556 lines — conversation + sessions
│   ├── config.py                      # 425 lines — YAML config manager
│   ├── ui.py                          # 1946 lines — Rich terminal UI
│   ├── selenium_browser.py            # 2751 lines — Selenium Firefox automation
│   ├── doc_tools.py                   # 2507 lines — PDF/DOCX/PPTX/XLSX/CSV
│   ├── mcp_tools.py                   # 1482 lines — MCP tool registration
│   ├── connectors.py                  # 767 lines — Telegram & Discord
│   ├── mcp_client.py                  # 471 lines — MCP protocol client
│   ├── multi_agent.py                 # 247 lines — 5-agent delegation
│   ├── planner.py                     # 300 lines — task decomposition
│   ├── webcontrol.py                  # 939 lines — HTTP browser automation
│   ├── auth.py                        # 132 lines — Firebase auth gate
│   ├── llm.py                         # 3 lines — backward-compat shim
│   └── tools.py                       # 15 lines — DEPRECATED shim ⚠️
│
└── dashboard-react/                   # Cloudflare Worker + React SPA (gitignored root)
    ├── public/                        # Static assets served as-is
    │   ├── favicon.svg
    │   └── icons.svg
    ├── src/
    │   ├── main.tsx                   # React entrypoint
    │   ├── App.tsx                    # 531 lines — main dashboard shell
    │   ├── App.css                    # legacy styles (deprecated)
    │   ├── index.css                  # legacy styles (deprecated)
    │   ├── api/client.ts              # API helpers (passcode header, fetch)
    │   ├── hooks/useDashboardData.ts  # TanStack Query hooks
    │   ├── lib/types.ts               # TypeScript types matching Worker API
    │   ├── lib/format.ts              # Number, time, admin-id helpers
    │   ├── lib/theme.ts               # 4 theme presets (blue/violet/emerald/rose)
    │   ├── lib/themeVars.ts           # CSS variable application
    │   ├── styles/globals.css         # 541 lines — the actual stylesheet
    │   ├── components/
    │   │   ├── AdminChip.tsx          # Admin session badge (header)
    │   │   ├── Charts.tsx             # Status + tool analytics
    │   │   ├── CliUsersModal.tsx      # Firebase user management modal
    │   │   ├── ConfirmModal.tsx       # Generic confirmation dialog
    │   │   ├── Drawer.tsx             # Right-side detail drawer
    │   │   ├── FilterBar.tsx          # Search + sort + status filter
    │   │   ├── LimitModal.tsx         # Token limit setter
    │   │   ├── LoginOverlay.tsx       # Passcode entry screen
    │   │   ├── PasscodeModal.tsx      # Change admin passcode
    │   │   ├── StatsCards.tsx         # 4-card summary row
    │   │   ├── ThemeDots.tsx          # 4-theme switcher
    │   │   ├── Toast.tsx              # Toast context provider
    │   │   ├── UserTable.tsx          # Main user table with actions
    │   │   └── VersionModal.tsx       # Registry version publisher
    │   └── assets/                    # Bundled images
    ├── worker.js                      # 510 lines — Cloudflare Worker handler
    ├── wrangler.toml                  # CF Worker config (secrets not committed)
    ├── package.json                   # React/Vite deps + wrangler (dev)
    ├── vite.config.ts                 # Vite build config
    ├── tsconfig*.json                 # TypeScript configs
    ├── eslint.config.js               # ESLint flat config
    └── dist/                          # vite build output → deployed via [assets]
```

> ⚠️ **The deployed dashboard** is the **React SPA** built from `dashboard-react/src/`
> into `dashboard-react/dist/`. Cloudflare Worker serves it via the `[assets]` binding
> in `wrangler.toml`. `worker.js` handles both static asset routing AND all `/api/*`
> endpoints in a single fetch handler.

---

## 🔄 Data Flow End-to-End

### CLI Client → Dashboard

```
CLI (dscli)                              Dashboard Browser
   │                                           │
   │  POST /api/update                         │
   │  {                                        │
   │    ip, username,                          │
   │    input_tokens, output_tokens,           │
   │    last_tool, status, version,            │
   │    hostname, platform, arch, ...          │
   │  }                                        │
   │ ────────────────────────────────────────▶ │
   │                                           │
   │                              Worker.js:    │
   │                              ├─ getGistData() → GitHub API
   │                              ├─ find user by IP
   │                              ├─ cycle reset (24h)
   │                              ├─ update tokens + last_tool
   │                              └─ saveGistData() → PATCH Gist
   │                                           │
   │ ◀──────────────────────────────────────── │
   │  { success: true, banned: false }         │
   │                                           │
   │                              Admin opens dashboard:
   │                              GET /api/admin/data
   │                              (X-Admin-Passcode header)
   │                                           │
   │                              Renders table of all CLI users
   │                              with their tokens, tools, status
```

### Auth Gate Flow

```
CLI launch
   │
   ▼
__main__.py → auth.main()
   │
   ├─ ~/.deepseek-cli/auth.json exists?
   │  YES → silent token refresh → check banned flag
   │  NO  → interactive menu
   │       │
   │       ├─ Login (email+password)
   │       ├─ Register (username+email+password) → email verification
   │       └─ Forgot password (sends reset email)
   │
   ▼
Banned? → sys.exit(1) with friendly message
   │
   ▼
enforce_gist() → check global ban/limit
   │
   ▼
REPL
```

### Update Notification Flow (the famous one)

```
dscli launch
   │
   ▼
enforce_gist() (config.py)
   │
   ├─ fetch registry Gist 55a91f3e…/endpoint.json
   │     { latest_version: "7.7", api_url: "..." }
   │
   ├─ compare with CLIENT_VERSION (config.py) = "7.7"
   │     using is_newer_version() — proper semver tuple compare
   │
   ▼
ui.show_banner()
   │
   ├─ prints big ASCII banner
   │
   └─ if newer → prints "Update Available vX.Y" RIGHT BELOW banner
   │
   ▼
REPL main loop
```

> **To publish a new version to every client:** change `latest_version` in the
> registry Gist (via the dashboard Version button — `POST /api/admin/version` —
> or PATCH the Gist manually). All clients pick it up on next launch. The
> client only *displays* the notice; users upgrade by running `bash install.sh`.

---

## 🔔 Sistem Update & Versi

### Critical: Why the old code "didn't work"

The previous code had two bugs:
1. Printed the "Update Available" notice to **stderr** *before* the ASCII banner,
   so it scrolled off-screen on small terminals.
2. Used naive `!=` string compare: `"7.7" != "7.7.0"` returned true (wrong).

Both fixed:
- Notice is now in **stdout** printed **after** the banner.
- `is_newer_version()` does a real semver tuple compare.

### The version contract

| Location | Field | Purpose |
|----------|-------|---------|
| `deepseek/config.py` | `CLIENT_VERSION = "7.7"` | The version baked into this build |
| Registry Gist `55a91f3e…` | `endpoint.json → latest_version` | Authoritative "current latest" |
| `dashboard-react/worker.js` | `/api/version` GET (public) | Reads registry, returns `{latest_version, api_url}` |
| `dashboard-react/worker.js` | `/api/admin/version` POST | Merge-only write of `latest_version` |
| `deepseek/install.sh` | hardcoded v7.7 strings | Banner text in installer |

### Version bump checklist

When releasing v7.8:
- [ ] Bump `CLIENT_VERSION` in `deepseek/config.py`
- [ ] Bump `VERSION`/`VERSION_BANNER` in `deepseek/repl.py`
- [ ] Update v7.7 strings in `deepseek/ui.py` and `install.sh`
- [ ] Publish via dashboard **Version** button (or PATCH registry Gist directly)
- [ ] Update README badges
- [ ] Update VERIFICATION_REPORT.md

> ⚠️ **Never change `_DEFAULT_GIST_ID`** (`55a91f3ee47f659d21a58a80826ca827`)
> in `config.py` — existing installs would lose their update channel.

---

## 🔐 Autentikasi (Firebase)

`dscli` requires login before the REPL starts. Implemented in `deepseek/auth.py`
and invoked from `__main__.py` **before** `enforce_gist()`.

### Flow

```
Launch → auth.main()
  │
  ├─ ~/.deepseek-cli/auth.json exists?
  │  YES → silently refresh token via securetoken.googleapis.com
  │  NO  → interactive menu
  │
  ├─ Login (email + password)
  ├─ Register (username + email + password)
  │     → Identity Toolkit createUser
  │     → sendOobCode (email verification link)
  │     → user must verify before next launch
  └─ Forgot password
        → sendOobCode (password reset email)
```

### What gets stored

- **Firebase Auth** (Identity Toolkit): `localId` (UID), `email`, `idToken`, `refreshToken`, `expiresIn`
- **RTDB mirror** at `/dscliUsers/<uid>`: `{ username, email, banned, created_at, last_login }`
- **Local**: `~/.deepseek-cli/auth.json` (chmod 600) — `idToken`, `refreshToken`, expiresAt

### Dashboard ↔ CLI user management

The Worker exposes (admin-only):
- `GET /api/admin/users` — list all `/dscliUsers` from RTDB
- `POST /api/admin/user_action { action: ban|unban|delete, uid }`

UI: header **"CLI Users"** button → modal with search + ban/delete actions.

**For advanced ban/unban** (Firebase Auth-level disable), the Worker can use a
**Firebase Service Account** (set as `FIREBASE_SERVICE_ACCOUNT` secret in
`wrangler.toml`). Without it, only RTDB-level ban flag is set; with it, the user
is also disabled in Firebase Auth.

Worker needs env vars: `FIREBASE_API_KEY`, `FIREBASE_DB_URL`, `FIREBASE_USERS_PATH`.

### Firebase project

- Project: `xbibzstorage`
- RTDB region: `asia-southeast1`
- Web config (apiKey) is a **public client config** by design.

> ⚠️ **RTDB security is currently open.** The path `/dscliUsers` keeps CLI data
> separate from anything else but is NOT locked down. **Tighten Firebase rules
> before relying on this for anything sensitive.**

### Escape hatch

- `DEEPSEEK_SKIP_AUTH=1` — bypass login (offline/dev only)
- `DEEPSEEK_FIREBASE_API_KEY` — override apiKey
- `DEEPSEEK_FIREBASE_DB_URL` — override DB URL

---

## 🌐 Dashboard Cloudflare Worker

### Stack

- **Frontend**: React 19 + TypeScript + Vite 8 (SPA, built to `dist/`)
- **Backend**: Cloudflare Workers (single `worker.js` fetch handler, ES modules)
- **Storage**: two GitHub Gists
- **Auth**: passcode (header `X-Admin-Passcode`); optional Firebase Admin (service account)

### Gist contract

| Gist ID | File | Contents | Public? |
|---------|------|----------|---------|
| `339448cf1b84118482f9af646f242791` | `usage.json` | `[ {ip, username, tokens, ...}, ... ]` | No |
| `339448cf1b84118482f9af646f242791` | `secrets.json` | `{ admin_passcode }` | No |
| `55a91f3ee47f659d21a58a80826ca827` | `endpoint.json` | `{ api_url, latest_version }` | Yes |

> ⚠️ **"Tertumpuk" footgun (FIXED):** the original deploy script called
> `create_gist()` on every run, spawning dozens of duplicate registry + DB
> Gists. The deploy is now **idempotent**: it reuses the canonical Gist IDs
> (registry `55a91f3e…`, live DB `339448cf…`) and PATCH-merges instead of
> recreating. Never point the CLI default at a new registry Gist.

### Auth & UI

- **Admin Console (`/admin`)**: Passcode-gated dashboard to manage CLI users, see analytics, and execute admin actions (Ban/Limit).
- **User Dashboard (`/account`)**: Firebase Auth integrated view (`AuthView`, `.uda-*` classes). Users can register/login, sync their username, and view their individual token usages. The UI uses modern glassmorphic designs with glow animations.

### Build & serve flow

```
dashboard-react/src/  ──[vite build]──▶  dashboard-react/dist/
                                                  │
                                                  ▼
                                  Cloudflare Worker [assets] binding
                                  (serves /, /assets/*, SPA fallback)
                                                  +
                                  worker.js handles /api/* endpoints
```

### API surface

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/api/check?ip=X` | public | CLI self-check (banned + limit) |
| GET | `/api/version` | public | Latest release version |
| POST | `/api/update` | public | CLI usage telemetry |
| GET | `/api/admin/data` | admin | Full records array |
| POST | `/api/admin/action` | admin | toggle_ban / delete / update_limit |
| POST | `/api/admin/change_password` | admin | Rotate admin passcode |
| GET | `/api/admin/version` | admin | Full registry payload |
| POST | `/api/admin/version` | admin | Merge-only `latest_version` write |
| GET | `/api/admin/users` | admin | List Firebase RTDB users |
| POST | `/api/admin/user_action` | admin | ban / unban / delete user |

### Worker env vars (`wrangler.toml [vars]`)

| Variable | Purpose |
|----------|---------|
| `GIST_ID` | Private DB Gist ID (usage + secrets) |
| `REGISTRY_GIST_ID` | Public registry Gist ID |
| `REGISTRY_FILENAME` | Default `endpoint.json` |
| `GIST_FILENAME` | Default `usage.json` |
| `FIREBASE_API_KEY` | Firebase web API key (public) |
| `FIREBASE_DB_URL` | Firebase RTDB URL |
| `FIREBASE_USERS_PATH` | Default `dscliUsers` |

### Worker secrets (`wrangler secret put`)

| Secret | Required? | Purpose |
|--------|-----------|---------|
| `GITHUB_PAT` | **Yes** | Read+write the Gists (no GitHub login otherwise) |
| `ADMIN_PASSCODE` | **Yes** | Gate for `/api/admin/*` endpoints |
| `FIREBASE_SERVICE_ACCOUNT` | Optional | JSON service account for Firebase Admin (Auth-level ban/delete) |

---

## 🧪 Multi-Agent System

5 specialized profiles, each with own system prompt, model, and tool access:

| Profile | Focus | System prompt extra |
|---------|-------|---------------------|
| `general` | All-purpose | (default) |
| `coder` | Programming, debugging, code review | Code-first reasoning |
| `researcher` | Web search, fact-finding | Always cite sources |
| `filesystem` | File management, data processing | Sandbox-safe file ops |
| `reasoner` | Step-by-step analytical reasoning | Slow down, justify each step |

### Delegation API

```python
# Blocking delegation
result = multi_agent_manager.delegate(profile="coder", task="Fix this bug")

# Concurrent delegation
results = multi_agent_manager.run_concurrent([
    {"profile": "researcher", "task": "Find X"},
    {"profile": "coder", "task": "Refactor Y"},
    {"profile": "filesystem", "task": "List Z"},
])
```

Each `AgentWorker` has **isolated memory**, **own provider**, **own model**,
and runs the full LLM → tool → result loop (max 6 rounds).

---

## 🔌 MCP (Model Context Protocol)

Built-in real MCP client. Connect to external servers and use their tools automatically.

### Popular servers

| Server | Auth | Use case |
|--------|------|----------|
| `context7` | none | Live library docs |
| `canva` | `CANVA_API_KEY` | Create/edit designs |
| `github` | `GITHUB_TOKEN` | Repos, issues, PRs |
| `brave-search` | `BRAVE_API_KEY` | Web search |
| `filesystem` | none | Secure file access |
| `memory` | none | Knowledge graph |
| `fetch` | none | HTTP scraping |
| `sqlite` | none | Database queries |
| `postgres` | `DATABASE_URL` | PostgreSQL |
| `puppeteer` | none | Browser automation |
| `sequential-thinking` | none | Reasoning |

### Architecture

```
deepseek/mcp_client.py
├── POPULAR_MCP_SERVERS  — Registry of 13+ pre-configured servers
├── MCPConnection        — Single server (stdio or SSE transport)
│   ├── _run_stdio_session()   — stdio transport via asyncio
│   ├── _run_sse_session()     — SSE transport
│   └── _call_tool_async()     — Tool invocation
├── MCPClientManager     — Singleton managing multiple connections
│   ├── connect_server()       — Async connect (in background thread)
│   ├── call_tool()            — Route tool calls to correct server
│   ├── get_all_tools()        — Aggregate all available tools
│   └── get_status()           — Connection status overview
└── mcp_manager          — Global singleton instance
```

### Commands

```
/mcp                     # Open MCP management menu
/mcp list                # List popular servers
/mcp connect context7    # Connect to Context7
/mcp connect canva       # Connect to Canva (asks for API key)
/mcp status              # Show connected servers
/mcp disconnect context7  # Disconnect
```

Connected MCP servers auto-register their tools into the agent's toolset.

---

## 🤖 Providers (8 AI)

| Provider | Type | Base URL | Default Model | Free? | Tools | Streaming |
|----------|------|----------|---------------|-------|-------|-----------|
| OpenRouter | OpenAI-compat | `openrouter.ai/api/v1` | `deepseek/deepseek-r1-0528:free` | ✅ | ✅ | ✅ |
| Google Gemini | Gemini API | `generativelanguage.googleapis.com` | `gemini-2.5-flash-preview-05-20` | ✅ | ✅ | ✅ |
| Anthropic | Anthropic API | `api.anthropic.com` | `claude-sonnet-4-20250514` | ❌ | ✅ | ✅ |
| Groq | OpenAI-compat | `api.groq.com` | `llama-3.3-70b-versatile` | ✅ | ✅ | ✅ |
| Together AI | OpenAI-compat | `api.together.xyz` | `meta-llama/Llama-3-70b-chat-hf` | ✅ | ✅ | ✅ |
| HuggingFace | HF Router | `router.huggingface.co` | `Qwen/Qwen2.5-72B-Instruct` | ✅ | ⚠️ prompt-based | ✅ |
| OpenAI | OpenAI API | `api.openai.com` | `gpt-4.1-mini` | ❌ | ✅ | ✅ |
| Agnes AI | OpenAI-compat | `apihub.agnes-ai.com` | `agnes-2.0-flash` | ✅ | ✅ | ✅ |

> HuggingFace uses **prompt-based** tool calling (not native structured tools)
> because the HF router doesn't expose OpenAI-style tool_calls.

### Architecture

```
deepseek/providers.py
├── BaseProvider              — Abstract base (chat_stream, fetch_models, validate_key)
├── OpenAICompatibleProvider  — OpenRouter, OpenAI, Groq, Together, Agnes (shared impl)
├── GeminiProvider            — Google Gemini (generateContent + _convert_tools)
├── AnthropicProvider         — Anthropic Claude (messages API)
├── HuggingFaceProvider       — HF router (prompt-based tools)
└── create_provider()         — Factory function
```

> Bug-fix history:
> - Gemini `_convert_tools()` was returning an **array** instead of a single
>   `functionDeclarations` object. **Fixed.**
> - HuggingFace tool-call parser handles **JSON-in-text** fallback.

---

## 🛠 Toolkit (120+ Tools)

Categories:
- 📁 File ops: `read_file`, `write_file`, `edit_file`, `list_files`, `delete_file`, `file_info`, `search_files`, `tree_view`
- 🌐 Web: `web_search`, `web_fetch`, `live_search`, `browser_navigate`, `browser_click`, `browser_fill_form`, `browser_snapshot`, `browser_login`, `browser_screenshot`, `browser_cookies`
- 💻 Code: `run_code`, `run_shell`, `install_package`
- 🖼 Image/OCR: `image_view`, `image_info`, `ocr_read`, `ocr_url`
- 📄 Documents: `read_pdf`, `create_pdf`, `pdf_edit`, `read_docx`, `create_docx`, `docx_info`, `edit_docx`, `read_pptx`, `create_pptx`, `edit_pptx`, `pptx_info`, `read_xlsx`, `create_xlsx`, `edit_xlsx`, `xlsx_info`, `read_csv`, `create_csv`, `convert_document`
- 🔬 System: `system_info`, `process_list`, `disk_usage`, `network_info`, `env_vars`, `apk_analyze`, `video_info`
- 🔧 Utility: `calculate`, `unit_convert`, `timestamp`, `text_transform`, `json_parse`, `regex_test`, `base64_tool`, `generate_uuid`, `random_number`, `sort_data`, `count_text`, `hash_tool`
- 🌤️ Real-time data: `get_datetime`, `get_calendar`, `get_weather`, `get_news`, `get_stock_price`, `get_crypto_price`, `get_currency_rate`, `get_holidays`, `get_qibla`, `get_countdown`, `get_sun_times`, `get_day_info`, `get_ip_info`, `get_random_fact`
- 🤖 Multi-agent: `delegate`, `delegate_concurrent`

### Validation

All tools use **Pydantic** for argument validation. Both v1 (`.dict()`) and
v2 (`.model_dump()`) are supported.

### Timeout

`TOOL_TIMEOUT_DEFAULT = 0` (no hard timeout by default — AI determines execution
time). When set > 0, uses **threading** (not SIGALRM) for Termux compat.

---

## 💾 Session & Memory

### Session lifecycle

```
dscli launch
  → memory.create_session()
  → session_id = "dscli-YYYYMMDD-HHMMSS"
  → save to ~/.deepseek-cli/sessions/<id>.json

every turn
  → memory.add_user/assistant/tool(msg)
  → auto-save to disk

commands:
  dscli -l             # list sessions
  dscli -s <id>        # resume
  dscli -d <id>        # delete
```

### Metrics

JSON log per session at `~/.deepseek-cli/logs/<id>.json`:
```json
{
  "session_id": "...",
  "timestamp": "...",
  "total_turns": 15,
  "total_tool_calls": 42,
  "total_errors": 0,
  "avg_latency": 1.84,
  "tool_usage": { "web_search": 8, "read_file": 12, "run_code": 5 },
  "turns": [...]
}
```

### Loop safety

| Setting | Default | Effect |
|---------|---------|--------|
| `UNLIMITED_TOOL_ROUNDS` | `True` | No hard cap — ends when model says "done" |
| `MAX_SAME_TOOL` | `50` | Anti-stuck: stop if same tool 50× no progress |
| `MAX_REPEATED_CONTENT` | `3` | Anti-stuck: stop if 3× identical outputs |
| `TOOL_TIMEOUT_DEFAULT` | `0` | Per-tool timeout via threading (0 = no timeout) |
| `safe_parse_json` | 2 retries | Auto-fix missing braces + retry |

---

## 🚀 Build / Deploy

### CLI (no build step)

```bash
# Edit → run (Python is interpreted)
python -m py_compile deepseek/*.py    # fast syntax gate
python -m deepseek                    # run from source
bash install.sh                       # install (venv + dscli wrapper)
```

### Dashboard (React SPA + Worker)

```bash
cd dashboard-react
npm install                           # one-time
npm run build                         # → ./dist/ (SPA)
npx wrangler deploy --compatibility-date=2023-01-01
```

The deploy:
1. Compiles React + TS into `./dist/` (single-page app, hashed assets).
2. Uploads `worker.js` + `./dist/` to Cloudflare Workers.
3. Worker uses `[assets]` binding in `wrangler.toml` to serve static files
   AND handles all `/api/*` endpoints in the same fetch handler.

### Secrets setup (one-time after cloning)

```bash
cd dashboard-react
npx wrangler secret put GITHUB_PAT              # GitHub PAT with gist scope
npx wrangler secret put ADMIN_PASSCODE          # dashboard admin passcode
# Optional: Firebase Admin (for Auth-level ban/delete)
npx wrangler secret put FIREBASE_SERVICE_ACCOUNT  # JSON string
```

### Live URL

`https://deepseek-dash.bibzflow.workers.dev`

### GitHub Gists (canonical)

| Purpose | Gist ID | File |
|---------|---------|------|
| Usage DB (private) | `339448cf1b84118482f9af646f242791` | `usage.json` |
| Usage DB secrets (private) | `339448cf1b84118482f9af646f242791` | `secrets.json` |
| Public registry | `55a91f3ee47f659d21a58a80826ca827` | `endpoint.json` |

---

## ⚠️ Risiko & Performa

### Critical risks

| Risk | Mitigation |
|------|------------|
| 🔴 **Secrets committed in wrangler.toml + setup_deploy.py** (GH PAT, CF token, Firebase keys, admin passcode) | **Rotate them.** Use `wrangler secret put` instead of `[vars]` for sensitive values |
| 🔴 **Firebase RTDB open rules** | Set `.read` / `.write` rules to authenticated-only before going public |
| 🟡 **Anti-stuck threshold (`MAX_SAME_TOOL = 50`)** | Could let model loop on same tool 50 times — consider lowering to 10 |
| 🟡 **TOOL_TIMEOUT_DEFAULT = 0** | If a tool hangs, user waits forever — set > 0 for production |
| 🟡 **AgentMetrics logs grow unbounded** | Rotate logs older than 30 days |
| 🟢 **Gist ID changes break clients** | Never change `_DEFAULT_GIST_ID` in config.py |
| 🟢 **Worker bundling `dist/`** | If `dist/` is missing, dashboard 404s on JS chunks — always run `npm run build` before deploy |

### Performance

| Concern | Mitigation |
|---------|------------|
| Dashboard polling every 5-30s | Default 15s; reduces Gist API calls from 5/min to 4/min |
| Tool result in memory grows large | Use `/compact` to keep system + last 10 messages |
| Gist file edit race condition | Last-write-wins (rare; only relevant if 2 admins act simultaneously) |
| Worker cold start | <5ms typically; not a concern |
| IndexedDB / localStorage passcode | Stored client-side only; passcode never sent to server except for auth |
| React bundle size | TanStack Query + lucide-react icons; total JS gzipped ~150KB |
| Cloudflare Worker bundle limit | `worker.js` + assets bound separately — well under 1MB / 10MB limits |

### Pre-commit checklist

Before committing any change:

- [ ] `python -m py_compile deepseek/*.py` — no syntax errors
- [ ] `node --check dashboard-react/worker.js` — JS syntax OK
- [ ] `cd dashboard-react && npm run build` — TypeScript compiles + Vite builds
- [ ] No new mandatory deps (Termux compat)
- [ ] No SIGALRM anywhere
- [ ] No new Gist creation logic (only merge)
- [ ] No change to `_DEFAULT_GIST_ID`
- [ ] No secrets in diff (`wrangler.toml` should be unchanged or only `[vars]` updated)
- [ ] If UI changed: visual review in Chrome + Firefox + Safari
- [ ] If API changed: backward-compat (old clients still work)

---

## 🤝 Aturan Contribution

### Git workflow

```bash
git checkout -b feature/amazing-thing
# ... make changes ...
python -m py_compile deepseek/*.py
node --check dashboard-react/worker.js
cd dashboard-react && npm run build
git add -p
git commit -m "feat: add amazing thing"
git push origin feature/amazing-thing
# Open PR
```

### Commit message format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
Scopes: `cli`, `dashboard`, `worker`, `auth`, `agents`, `providers`, `toolkit`, `mcp`

Example:
```
feat(dashboard): add admin profile chip with logged-in indicator

- Shows the admin's passcode hash in header when authenticated
- Improves UI/UX with refined typography and breathing room
- Preserves all existing API contracts

Closes #42
```

### Code style

- **Python**: PEP 8, type hints encouraged, docstrings on public functions
- **TypeScript**: strict mode, no `any` in committed code, prefer named exports
- **CSS**: vanilla CSS variables, BEM-ish naming, no Tailwind/CSS-in-JS
- **Naming**: snake_case (Python), camelCase (JS/TS), kebab-case (CSS classes)
- **Max line length**: 120 (Python), 100 (TS/JS)

---

## 🩺 Troubleshooting

### "Module not found" on launch

```bash
bash install.sh              # re-install (handles pip + venv)
# or
pip install -e .             # dev mode
```

### "Invalid passcode" on dashboard

- Check `X-Admin-Passcode` header matches what's in Gist `secrets.json`
- Or matches `ADMIN_PASSCODE` env var in `wrangler.toml`
- Default passcode: `XbibzOfficial777` (CHANGE IT via dashboard Passcode modal)

### "Cannot connect to Gist"

```bash
# Test manually
curl -H "Authorization: token $GITHUB_PAT" \
  https://api.github.com/gists/339448cf1b84118482f9af646f242791
```

If 401: PAT expired → rotate at https://github.com/settings/tokens

### Dashboard shows old UI / 404 on JS chunks

```bash
cd dashboard-react
rm -rf dist node_modules/.vite
npm run build
npx wrangler deploy
```

### "Update Available v7.8" but I'm on 7.7

- Registry Gist `latest_version` is set to `7.8`
- You actually are on 7.7 — to "upgrade", run `bash install.sh` from the new release

### CLI hangs mid-tool

- Check `TOOL_TIMEOUT_DEFAULT` in `~/.deepseek-cli/config.yaml`
- If 0, set to e.g. 90 for safety

### "Ban" / "Limit exceeded" but I'm not banned

- Open dashboard → search your IP → verify `banned` is `false` and tokens are within limit
- Check cycle_start timestamp — tokens reset every 24h

---

## 📜 License & Credits

MIT License. Built by [@XbibzOfficial777](https://github.com/XbibzOfficial777).

See [README.md](./README.md) for full project docs.
See [VERIFICATION_REPORT.md](./VERIFICATION_REPORT.md) for the 22-bug fix log.
See [PROJECT_ANALYSIS.md](./PROJECT_ANALYSIS.md) for the codebase audit.

**Last updated:** 2026-06-23 · **Version:** 7.7 · **Maintainer:** DeepSeek CLI team
