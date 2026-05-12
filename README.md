<div align="center">

# 🧠 DeepSeek CLI Agent v7.7
### *The Ultimate Multi-Provider AI Agent for Developers*

[![Version](https://img.shields.io/badge/Version-7.7.0-00FFA3.svg?style=for-the-badge)](https://github.com/XbibzOfficial777/deepseek-cli)
[![License](https://img.shields.io/badge/License-MIT-white.svg?style=for-the-badge)](https://github.com/XbibzOfficial777/deepseek-cli/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Termux-FF6F00.svg?style=for-the-badge)](https://github.com/XbibzOfficial777/deepseek-cli)
[![Tools](https://img.shields.io/badge/Tools-90%2B-8A2BE2.svg?style=for-the-badge)](https://github.com/XbibzOfficial777/deepseek-cli)

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-supported-providers">Providers</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-command-reference">Commands</a> •
  <a href="#-tool-ecosystem">Tools</a>
</p>

</div>

---

**DeepSeek CLI** is a production-grade, autonomous AI Agent engineered to streamline development workflows. Powered by an advanced **agentic loop**, it doesn't just chat — it **reasons, plans, and executes** tasks using a suite of **90+ built-in professional tools**.

> ⚡ *Designed for developers who demand speed, autonomy, and elegance in their terminal.*

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🧠 Autonomous Reasoning** | Multi-step planning engine that decomposes complex tasks into executable actions |
| **🛠️ 90+ Integrated Tools** | File operations, web search, code execution, OCR, PDF, DOCX, image processing, browser automation, and more |
| **🌐 Multi-Provider** | Native support for 7 AI providers — switch between OpenRouter, Gemini, Anthropic, Groq, and more |
| **🎨 Rich Markdown UI** | Professional terminal interface with real-time streaming, syntax-highlighted code blocks, and styled output |
| **📱 Mobile Ready** | Fully optimized for Android Termux and low-latency environments |
| **🔗 Telegram & Discord** | Built-in connectors to run your AI agent through Telegram bots and Discord channels |
| **🔄 Smart Loop Protection** | Anti-stuck detection, tool loop prevention, and automatic timeout safeguards |
| **📊 Session Logging** | Automatic metrics logging with tool usage statistics and latency tracking |

---

## 🚀 Quick Start

### One-Line Installation

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/XbibzOfficial777/deepseek-cli/main/install.sh)"
```

### Manual Installation

```bash
git clone https://github.com/XbibzOfficial777/deepseek-cli.git
cd deepseek-cli
bash install.sh
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

## 🔌 Supported Providers

| Provider | Status | Best For |
| :--- | :--- | :--- |
| **OpenRouter** | ✅ Active | Universal access to 200+ models including DeepSeek-R1, Llama, Claude |
| **Google Gemini** | ✅ Active | High-speed reasoning with 1M+ token context window |
| **Anthropic** | ✅ Active | Precision coding with Claude Sonnet 4 & Haiku 4 |
| **Groq** | ✅ Active | Ultra-low latency inference for production workloads |
| **Together AI** | ✅ Active | Diverse open-source model ecosystem |
| **HuggingFace** | ✅ Active | Access to cutting-edge community models |
| **OpenAI** | ✅ Active | Industry-standard GPT-4o and o-series reasoning models |

---

## 🏗️ Architecture

The system is built on a modular **Agentic Loop** architecture:

```
┌─────────────┐     ┌──────────┐     ┌──────────────┐
│   User      │ ──▶ │  Agent   │ ──▶ │   Planner    │
│  Input      │     │  Engine  │     │  (optional)  │
└─────────────┘     └──────────┘     └──────────────┘
                          │                  │
                          ▼                  ▼
                    ┌──────────┐     ┌──────────────┐
                    │  Memory  │     │  Tool Registry│
                    │ (Context)│     │  (90+ Tools)  │
                    └──────────┘     └──────────────┘
                          │                  │
                          ▼                  ▼
                    ┌──────────┐     ┌──────────────┐
                    │ Provider │     │  Executor    │
                    │   API    │     │ (Safe Timeout)│
                    └──────────┘     └──────────────┘
```

**Execution Flow:**

1. **🧠 Perception** — Agent receives user intent and analyzes context
2. **📋 Planning** — LLM generates a multi-step execution strategy (optional planner layer)
3. **⚡ Action** — Agent invokes specialized tools with thread-safe timeout protection
4. **👁️ Observation** — Results are fed back into memory for self-correction
5. **🔁 Loop** — Repeat until task is complete or max rounds reached

---

## ⌨️ Command Reference

| Command | Description |
| :--- | :--- |
| `Ctrl+P` | Open interactive settings panel |
| `/help` | Show help and tool documentation |
| `/provider` | Switch AI provider (interactive menu) |
| `/model` | Change active LLM model (interactive menu) |
| `/key` | Set API key for current provider |
| `/tools` | List all 90+ available tools |
| `/thinking` | Toggle visibility of internal reasoning |
| `/clear` | Reset conversation memory |
| `/export` | Export chat session to text file |
| `/info` | Show current configuration |
| `/compact` | Compact conversation (keep system + last 10) |
| `/live_search <query>` | Real-time web search (multi-source) |
| `/live_models` | Fetch all models from provider API |
| `/telegram` | Manage Telegram bot connector |
| `/discord` | Manage Discord bot connector |
| `/quit` | Exit the application |

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
`read_pdf` · `create_pdf` · `pdf_edit` · `read_docx` · `create_docx` · `docx_info` · `read_pptx` · `create_pptx` · `edit_pptx` · `read_xlsx` · `create_xlsx` · `edit_xlsx`

### 🔬 System & Analysis
`system_info` · `process_list` · `disk_usage` · `network_info` · `env_vars` · `apk_analyze` · `video_info`

### 🔧 Utilities
`calculate` · `unit_convert` · `timestamp` · `text_transform` · `json_parse` · `regex_test` · `base64_tool` · `generate_uuid` · `sort_data` · `hash_tool`

### 🌤️ Real-Time Data (MCP)
`get_datetime` · `get_weather` · `get_news` · `get_stock_price` · `get_crypto_price` · `get_currency_rate` · `get_holidays` · `get_qibla` · `get_countdown`

### 🧪 Advanced Automation
**Selenium Browser:** Navigate, click, type, scroll, login, screenshot, cookie management, file upload

**Auth Automation:** Google OAuth, login flows, 2FA handling, CAPTCHA detection

### 🤖 Connectors
**Telegram Bot:** Message relay, markdown support, user whitelist, long message splitting

**Discord Bot:** Channel messaging, webhook integration, user whitelist

---

## 📊 Session Metrics

Automatic logging saves detailed metrics to `~/.deepseek-cli/logs/`:

```json
{
  "session_id": "20260512_051128",
  "total_turns": 15,
  "total_tool_calls": 42,
  "total_errors": 0,
  "avg_latency": 1.84,
  "tool_usage": {
    "web_search": 8,
    "read_file": 12,
    "run_code": 5
  }
}
```

---

## 📱 Termux Optimization

DeepSeek CLI is fully optimized for Android Termux:

- ✅ Threading-based timeouts (no `signal.SIGALRM`)
- ✅ Paste detection with intelligent line joining
- ✅ Arrow-key and Ctrl shortcuts fully supported
- ✅ Low-bandwidth streaming compatible
- ✅ No system dependencies required

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

<div align="center">

**Created with 💚 by [XbibzOfficial](https://github.com/XbibzOfficial777)**

[![GitHub Stars](https://img.shields.io/github/stars/XbibzOfficial777/deepseek-cli?style=social)](https://github.com/XbibzOfficial777/deepseek-cli)
[![GitHub Forks](https://img.shields.io/github/forks/XbibzOfficial777/deepseek-cli?style=social)](https://github.com/XbibzOfficial777/deepseek-cli)

[⬆ Back to Top](#-deepseek-cli-agent-v77)

</div>
