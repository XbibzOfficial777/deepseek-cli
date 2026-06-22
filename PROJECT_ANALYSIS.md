# DeepSeek CLI v7.7 — Full Project Analysis & Bug Fixes

## 📁 Project Overview

**DeepSeek CLI** adalah AI agent CLI yang powerful untuk terminal. Mendukung 8 provider AI (OpenRouter, Gemini, Anthropic, Groq, Together AI, HuggingFace, OpenAI, Agnes AI), 120+ tools, streaming real-time, browser automation, OCR, document tools, multi-agent delegation, Telegram/Discord connectors, dan MCP protocol.

**Total lines:** ~21,380 baris Python

---

## 🏗️ Arsitektur & Data Flow

```
User Input (REPL)
    │
    ▼
┌─────────────────────────────────────┐
│  repl.py — main()                  │
│  - Init Memory, ToolRegistry       │
│  - Init Provider (YAML config)     │
│  - Init Agent(memory, tools, prov) │
│  - Main loop: prompt → agent.chat()│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  agent.py — Agent.chat()           │
│  - Add user message to memory       │
│  - Reasoning pre-pass (optional)    │
│  ┌───────────────────────────────┐  │
│  │ AGENTIC LOOP (UNLIMITED)      │  │
│  │ 1. Call provider.chat_stream()│  │
│  │ 2. Stream: thinking/content/  │  │
│  │    tool_calls/done/error      │  │
│  │ 3. Parse tool calls (native   │  │
│  │    or text-based fallback)    │  │
│  │ 4. If tool calls → execute    │  │
│  │    each tool via toolkit      │  │
│  │ 5. Feed results back to memory│  │
│  │ 6. Anti-stuck / loop detection│  │
│  │ 7. Repeat or stop             │  │
│  └───────────────────────────────┘  │
│  - Metrics logging                  │
│  - Return result dict               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  ui.py — StreamRenderer            │
│  - Real-time streaming display      │
│  - Thinking block rendering         │
│  - Tool call/result formatting      │
│  - Status bar, spinner              │
└─────────────────────────────────────┘
```

### Komponen Utama:

| File | Lines | Fungsi |
|------|-------|--------|
| `__main__.py` | 160 | CLI entrypoint, argparse, session management, uninstall |
| `agent.py` | 1214 | **Core agentic loop** — streaming, tool execution, anti-stuck, metrics |
| `repl.py` | 2186 | Interactive REPL, slash commands, Ctrl+P settings, history |
| `providers.py` | 847 | 8 AI provider adapters (OpenAI-compatible, Gemini, Anthropic, HuggingFace) |
| `toolkit.py` | 4550 | 120+ tool registry + Pydantic validation |
| `memory.py` | 556 | Session persistence, message history, todo management |
| `config.py` | 425 | YAML config manager, 8 provider definitions |
| `ui.py` | 1946 | StreamRenderer, terminal raw input, banner, spinner |
| `selenium_browser.py` | 2751 | Selenium browser automation (12 tools) |
| `doc_tools.py` | 2507 | PDF/DOCX/PPTX/XLSX create/edit/read tools |
| `mcp_tools.py` | 1482 | MCP protocol tools |
| `connectors.py` | 767 | Telegram & Discord bot connectors |
| `mcp_client.py` | 471 | MCP protocol client |
| `multi_agent.py` | 247 | Sub-agent delegation (5 profiles) |
| `planner.py` | 300 | LLM-based task decomposition |
| `webcontrol.py` | 939 | Web control utilities |
| `llm.py` | 3 | Backward compat shim |
| `__init__.py` | 18 | Cleanup old dirs |
| `tools.py` | 15 | Deprecated shim |

---

## 🐛 Bugs & Issues Found

### CRITICAL

#### BUG-1: `safe_execute` timeout terlalu lama (90 detik)
**File:** `agent.py:331`
**Masalah:** `TOOL_TIMEOUT_DEFAULT = 90` — tool execution timeout 90 detik terlalu lama. Jika tool hang, user harus tunggu 90 detik.
**Fix:** Kurangi ke 30 detik, atau buat configurable.

#### BUG-2: `MAX_SAME_TOOL = 50` terlalu tinggi
**File:** `agent.py:43`
**Masalah:** Anti-stuck threshold 50x sama tool call hampir tidak pernah trigger. Model bisa stuck infinite loop dengan tool yang sama 50+ kali.
**Fix:** Turunkan ke 10 atau kurangi bertahap.

#### BUG-3: `AgentWorker.run()` tidak loop dengan benar
**File:** `multi_agent.py:110`
**Masalah:** AgentWorker hanya melakukan 1 LLM call per round, tidak ada loop untuk multi-round tool execution. `has_tool_calls` flag tidak digunakan dengan benar — setelah tool executed, langsung break tanpa LLM call lagi untuk round berikutnya.
**Fix:** Perlu while loop yang benar.

### HIGH

#### BUG-4: `repl.py` — Ctrl+P settings panel tidak ada implementasi
**File:** `repl.py`
**Masalah:** `open_settings_panel()` dipanggil tapi tidak terdefinisi di file yang saya baca (2186 lines, mungkin ada di bagian yang tidak ter-read). Perlu verifikasi.

#### BUG-5: `tools.py` masih ada (deprecated)
**File:** `deepseek/tools.py`
**Masalah:** File deprecated masih ada dan bisa menyebabkan kebingungan. Seharusnya dihapus atau di-import dengan benar.

#### BUG-6: `llm.py` re-exports `chat_stream` tapi tidak terdefinisi
**File:** `deepseek/llm.py`
**Masalah:** `from .providers import create_provider, fetch_models, chat_stream` — `chat_stream` dan `fetch_models` tidak ada di providers.py sebagai standalone functions. Hanya ada class methods.
**Fix:** Hapus import yang tidak ada.

### MEDIUM

#### BUG-7: `memory.py` — system prompt duplikasi
**File:** `memory.py`
**Masalah:** System prompt didefinisikan di `__init__` DAN di `get_messages()`. Saat `get_messages()` dipanggil, system prompt di-rebuild dari awal, kehilangan custom additions dari REPL.

#### BUG-8: `planner.py` — planning tidak diintegrasikan ke agent
**File:** `planner.py`
**Masalah:** Planner class ada tapi tidak dipanggil dari `agent.py` atau `repl.py`. Fitur planning tidak aktif.

#### BUG-9: `config.py` — `MAX_TOOL_ROUNDS = 12` tapi agent loop unlimited
**File:** `config.py:63` vs `agent.py:40`
**Masalah:** Config bilang 12 rounds, tapi `UNLIMITED_TOOL_ROUNDS = True` di agent.py. Tidak konsisten.

### LOW

#### BUG-10: `AGENTS.md` — versi tidak sinkron
**File:** `AGENTS.md`
**Masalah:** Mencantumkan "FIXED v5.5" tapi versi aktual adalah v7.7. Banyak fix sudah di-merge tapi dokumentasi tidak di-update.

#### BUG-11: `requirements.txt` dihapus dari repo
**File:** `requirements.txt`
**Masalah:** File requirements.txt dihapus dari working tree (git showed `D`). Install script harus bisa install dependencies tanpa file ini.

---

## ✅ Fixes yang Sudah Diterapkan

1. **Pydantic validation** di toolkit.py — validasi input tool
2. **safe_parse_json** dengan retry 2x — tidak ada silent fail
3. **Threading-based safe_execute** — kompatibel dengan Termux (no SIGALRM)
4. **AntiStuckDetector** — detect repeated content (threshold 3x)
5. **ThinkTagStreamParser** — split <think> tags dari content
6. **Text-based tool call parser** — fallback untuk model tanpa structured tool calls
7. **Double-ESC interrupt** — user bisa interrupt streaming
8. **Connection error retry** — auto-retry 3x saat koneksi terputus
9. **Reasoning pre-pass** — visible thinking untuk semua provider
10. **Metrics logging** — JSON log per session
11. **StreamRenderer** — real-time streaming display
12. **Raw terminal input** — arrow keys, Ctrl+P, history

---

## 🔧 Rekomendasi Perbaikan

### Priority 1 (Urgent):
1. Fix `AgentWorker.run()` — perlu while loop yang benar untuk multi-round tool execution
2. Fix `llm.py` — hapus import yang tidak ada
3. Kurangi `TOOL_TIMEOUT_DEFAULT` dari 90 ke 30 detik
4. Kurangi `MAX_SAME_TOOL` dari 50 ke 10

### Priority 2 (Important):
5. Integrasikan `Planner` ke agent loop
6. Fix system prompt duplication di memory.py
7. Restore `requirements.txt`
8. Hapus `tools.py` (deprecated)

### Priority 3 (Nice to have):
9. Update `AGENTS.md` ke v7.7
10. Tambah unit tests untuk AgentWorker
11. Tambah config untuk adjustable thresholds
