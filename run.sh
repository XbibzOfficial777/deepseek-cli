#!/usr/bin/env bash
# ═══════════════════════════════════════════
#  DeepSeek CLI v4 — Setup & Launcher
#  Android Termux Compatible
#  Multi-Provider: OpenRouter, Gemini, HuggingFace,
#                  OpenAI, Anthropic, Groq, Together
# ═══════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export PYTHONIOENCODING=utf-8

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

banner() {
    clear
    echo -e "${CYAN}"
    echo "  ╔═══════════════════════════════════════════╗"
    echo "  ║                                           ║"
    echo "  ║    [ + ] DeepSeek CLI Agent v6.0            ║"
    echo "  ║     Multi-Provider · 7 AI Services        ║"
    echo "  ║     Agentic Loop · 26+ Tools              ║"
    echo "  ║                                           ║"
    echo "  ╚═══════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_deps() {
    echo -e "${YELLOW}[1/4] Checking dependencies...${NC}"
    if ! command -v python &>/dev/null; then
        echo -e "${RED}Python not found! Install: pkg install python${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ Python $(python --version 2>&1 | awk '{print $2}')${NC}"
}

install_deps() {
    echo -e "${YELLOW}[2/4] Installing packages...${NC}"
    pip install -q httpx rich duckduckgo-search pyyaml 2>/dev/null
    echo -e "${GREEN}  ✓ All packages installed${NC}"
}

check_config() {
    echo -e "${YELLOW}[3/4] Checking configuration...${NC}"
    CONFIG_DIR="$HOME/.deepseek-cli"
    CONFIG_FILE="$CONFIG_DIR/config.yaml"
    LEGACY_KEY="$HOME/.deepseek_api_key"

    # Migrate legacy API key
    if [ -f "$LEGACY_KEY" ] && [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${CYAN}  Migrating v3 API key to v4 config...${NC}"
        mkdir -p "$CONFIG_DIR"
        LEGACY_KEY_VAL=$(cat "$LEGACY_KEY")
        # Create minimal config with migrated key
        cat > "$CONFIG_FILE" <<EOF
version: 4
active_provider: openrouter
api_keys:
  openrouter: "$LEGACY_KEY_VAL"
models: {}
providers: {}
EOF
        chmod 600 "$CONFIG_FILE"
        echo -e "${GREEN}  ✓ Migrated OpenRouter API key${NC}"
    elif [ -f "$CONFIG_FILE" ]; then
        echo -e "${GREEN}  ✓ Config found at $CONFIG_FILE${NC}"
    else
        echo -e "${YELLOW}  No config found — will create on first run${NC}"
        # Check for any env var API keys
        FOUND_KEY=""
        if [ -n "$OPENROUTER_API_KEY" ]; then
            FOUND_KEY="OPENROUTER_API_KEY"
        elif [ -n "$GEMINI_API_KEY" ]; then
            FOUND_KEY="GEMINI_API_KEY"
        elif [ -n "$GROQ_API_KEY" ]; then
            FOUND_KEY="GROQ_API_KEY"
        fi
        if [ -n "$FOUND_KEY" ]; then
            echo -e "${GREEN}  ✓ Found $FOUND_KEY in environment${NC}"
        fi
    fi
}

cleanup_old() {
    echo -e "${YELLOW}[4/4] Cleaning old files...${NC}"
    cd "$SCRIPT_DIR/deepseek" 2>/dev/null || true
    for d in tools ui agent memory config llm providers; do
        if [ -d "$d" ] && [ ! -f "$d/__init__.py" ]; then
            rm -rf "$d" 2>/dev/null && echo -e "  Cleaned: $d/"
        fi
    done
    rm -rf __pycache__ 2>/dev/null
    cd "$SCRIPT_DIR" 2>/dev/null || true
    echo -e "${GREEN}  ✓ Clean${NC}"
}

launch() {
    echo ""
    echo -e "${CYAN}  Launching DeepSeek CLI v4...${NC}"
    echo -e "${MAGENTA}  Supported providers:${NC}"
    echo -e "    ${GREEN}OpenRouter${NC} (free models) · ${GREEN}Gemini${NC} (free) · ${GREEN}HuggingFace${NC} (free)"
    echo -e "    OpenAI · Anthropic · ${GREEN}Groq${NC} (free) · ${GREEN}Together AI${NC} (free)"
    echo -e "${CYAN}  ───────────────────────────────────────────${NC}"
    echo ""
    cd "$SCRIPT_DIR"
    exec python -m deepseek
}

banner
check_deps
install_deps
check_config
cleanup_old
launch
