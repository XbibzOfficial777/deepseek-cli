#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  DeepSeek CLI v7.7 — Installer / Uninstaller
#  Multi-Provider AI Agent | 8 Providers | 90+ Tools | Smart Loop | OCR
#  Features: Live Search, Browser Automation, Telegram & Discord Connectors
#  Document Tools (PPTX/XLSX/DOCX/CSV/PDF), Selenium, Rich Markdown UI
#
#  Usage:
#    bash install.sh                         # Install
#    bash install.sh --uninstall             # Uninstall
#    bash install.sh --clean                 # Deep clean (remove all stale code)
#    bash -c "$(curl -fsSL RAW_URL)"         # Install via curl pipe
#    bash -c "$(curl -fsSL RAW_URL)" --uninstall  # Uninstall via curl pipe
#
#  After install, run:  dscli
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colors ──────────────────────────────────────────────
R="\033[0m"
B="\033[1m"
D="\033[2m"
CY="\033[36m"
GR="\033[32m"
RD="\033[31m"
YE="\033[33m"
PU="\033[35m"
BL="\033[34m"

info()  { echo -e "${CY}${B}  ▸${R} ${D}$1${R}"; }
ok()    { echo -e "${GR}${B}  ✓${R} ${D}$1${R}"; }
warn()  { echo -e "${YE}${B}  ⚠${R} ${D}$1${R}"; }
err()   { echo -e "${RD}${B}  ✗${R} ${D}$1${R}"; }
head()  { echo -e "\n${PU}${B}  $1${R}"; echo -e "${PU}  ──────────────────────────────────${R}"; }
step()  { echo -e "\n${BL}${B}  [$1/$2]${R} ${D}$3${R}"; }

TOTAL_STEPS=6

# ═══════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════

echo ""
echo -e "${CY}${B}  ╔══════════════════════════════════════════╗${R}"
echo -e "${CY}${B}  ║      DeepSeek CLI v7.7  Installer       ║${R}"
echo -e "${CY}${B}  ║         Developer : @XbibzOfficial777   ║${R}"
echo -e "${CY}${B}  ╚══════════════════════════════════════════╝${R}"
echo ""

# ═══════════════════════════════════════════════════════════════
# UNINSTALL MODE
# ═══════════════════════════════════════════════════════════════

UNINSTALL_MODE=false
CLEAN_MODE=false
case " ${0:-} ${1:-} ${*} " in
    *\ --uninstall\ *|*\ uninstall\ *) UNINSTALL_MODE=true ;;
    *\ --clean\ *) CLEAN_MODE=true ;;
esac
if $UNINSTALL_MODE; then
    echo -e "${YE}${B}  ╔══════════════════════════════════════════╗${R}"
    echo -e "${YE}${B}  ║     DeepSeek CLI v7.7  Uninstaller     ║${R}"
    echo -e "${YE}${B}  ╚══════════════════════════════════════════╝${R}"
    echo ""

    FOUND=false
    DEEPSEEK_CONFIG="$HOME/.deepseek-cli"

    # 1. Remove package directories
    for d in "$HOME/.local/lib/deepseek-cli" "$DEEPSEEK_CONFIG"; do
        if [ -d "$d" ]; then
            echo -e "  ${CY}▸${R} Removing package: ${D}$d${R}"
            rm -rf "$d" 2>/dev/null || true
            FOUND=true
        fi
    done

    # 2. Remove dscli wrapper binary
    for d in /usr/local/bin "$HOME/.local/bin" "${PREFIX:-/data/data/com.termux/files/usr}/bin"; do
        if [ -f "$d/dscli" ]; then
            echo -e "  ${CY}▸${R} Removing wrapper: ${D}$d/dscli${R}"
            rm -f "$d/dscli"
            FOUND=true
        fi
    done

    # 3. Remove logs
    if [ -d "$HOME/.deepseek-cli/logs" ]; then
        echo -e "  ${CY}▸${R} Removing logs: ${D}$HOME/.deepseek-cli/logs${R}"
        rm -rf "$HOME/.deepseek-cli/logs"
        FOUND=true
    fi

    # 4. Remove legacy key file
    if [ -f "$HOME/.deepseek_api_key" ]; then
        echo -e "  ${CY}▸${R} Removing legacy key file${R}"
        rm -f "$HOME/.deepseek_api_key"
        FOUND=true
    fi

    # 5. Remove all ~/.deepseek-cli/ contents EXCEPT config.yaml (API key)
    if [ -d "$DEEPSEEK_CONFIG" ]; then
        echo -e "  ${CY}▸${R} Cleaning: ${D}$DEEPSEEK_CONFIG (keeping config.yaml with API key)${R}"
        for item in "$DEEPSEEK_CONFIG"/*; do
            [ -e "$item" ] || break
            basename_item="$(basename "$item")"
            if [ "$basename_item" != "config.yaml" ]; then
                rm -rf "$item"
            fi
        done
        FOUND=true
    fi

    # 6. Clean PATH entries from shell configs
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile"; do
        if [ -f "$rc" ]; then
            if grep -q "deepseek-cli\|dscli" "$rc" 2>/dev/null; then
                echo -e "  ${CY}▸${R} Cleaning PATH in: ${D}$rc${R}"
                sed '/# DeepSeek CLI/d' "$rc" > "$rc.tmp" && mv "$rc.tmp" "$rc"
                sed '/deepseek-cli/d' "$rc" > "$rc.tmp" && mv "$rc.tmp" "$rc"
                FOUND=true
            fi
        fi
    done

    # 7. Remove any stray cache/history files
    rm -f "$HOME/.deepseek-cli-history" 2>/dev/null || true

    # ── CLEAN MODE: also remove any pip-installed deepseek package ──
    if $CLEAN_MODE; then
        echo ""
        echo -e "  ${YE}${B}--clean mode: removing ALL stale installations${R}"
        if command -v pip3 &>/dev/null; then
            pip3 uninstall -y deepseek-cli deepseek-cli-agent deepseek 2>/dev/null | grep -E "Successfully|Skipping|not installed" || true
        elif command -v pip &>/dev/null; then
            pip uninstall -y deepseek-cli deepseek-cli-agent deepseek 2>/dev/null | grep -E "Successfully|Skipping|not installed" || true
        fi
        # Find any rogue deepseek packages in site-packages
        for sp in $($PYTHON -c "import site; print('\n'.join(site.getsitepackages()))" 2>/dev/null) /usr/lib/python3*/dist-packages /usr/local/lib/python3*/dist-packages; do
            if [ -d "$sp/deepseek" ]; then
                echo -e "  ${CY}▸${R} Removing pip package: ${D}$sp/deepseek${R}"
                rm -rf "$sp/deepseek" 2>/dev/null || true
                FOUND=true
            fi
        done
    fi

    echo ""
    if $FOUND; then
        echo -e "  ${GR}${B}✓ DeepSeek CLI has been uninstalled.${R}"
        echo -e "  ${GR}${B}✓ API key kept in ${D}$DEEPSEEK_CONFIG/config.yaml${R}"
    else
        echo -e "  ${YE}DeepSeek CLI is not installed or already removed.${R}"
    fi
    echo -e "  ${D}Run 'bash install.sh' to reinstall anytime.${R}"
    echo ""
    echo -e "  ${YE}${B}⚠ If you still see 'Signed in as' (old code), run:${R}"
    echo -e "  ${D}    bash install.sh --clean${R}"
    echo -e "  ${D}    hash -r${R}"
    echo -e "  ${D}    exec \$SHELL${R}"
    echo ""
    exit 0
fi

# ═══════════════════════════════════════════════════════════════
# DETECT ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

step 1 $TOTAL_STEPS "Detecting environment"

PREFIX="${PREFIX:-}"
IS_TERMUX=false
IS_MACOS=false
IS_LINUX=false
INSTALL_DIR=""
BIN_DIR=""

if [ -d "/data/data/com.termux" ] && [ -n "${PREFIX:-}" ]; then
    IS_TERMUX=true
    INSTALL_DIR="$HOME/.local/lib/deepseek-cli"
    BIN_DIR="$PREFIX/bin"
    info "Detected: Android Termux"
elif [ "$(uname)" = "Darwin" ]; then
    IS_MACOS=true
    INSTALL_DIR="$HOME/.local/lib/deepseek-cli"
    BIN_DIR="/usr/local/bin"
    info "Detected: macOS"
else
    IS_LINUX=true
    INSTALL_DIR="$HOME/.local/lib/deepseek-cli"
    BIN_DIR="$HOME/.local/bin"
    info "Detected: Linux"
fi

ok "Install dir: $INSTALL_DIR"
ok "Bin dir:     $BIN_DIR"

# ═══════════════════════════════════════════════════════════════
# CHECK PYTHON
# ═══════════════════════════════════════════════════════════════

step 2 $TOTAL_STEPS "Checking Python"

PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    err "Python 3 not found!"
    if $IS_TERMUX; then
        echo -e "${D}  Run:  pkg install python${R}"
    else
        echo -e "${D}  Install Python 3.10+ from your package manager${R}"
    fi
    exit 1
fi

PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]; }; then
    err "Python 3.8+ required (found ${PY_MAJOR}.${PY_MINOR})"
    exit 1
fi

ok "Python ${PY_MAJOR}.${PY_MINOR} ($PYTHON)"

# ═══════════════════════════════════════════════════════════════
# INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════

step 3 $TOTAL_STEPS "Installing Python dependencies"

if ! $PYTHON -m pip --version &>/dev/null 2>&1; then
    info "Installing pip..."
    if $IS_TERMUX; then
        pkg install -y python-pip 2>/dev/null || $PYTHON -m ensurepip --upgrade 2>/dev/null || true
    else
        curl -fsSL https://bootstrap.pypa.io/get-pip.py | $PYTHON - --quiet 2>/dev/null || true
    fi
fi

$PYTHON -m pip install --quiet --upgrade pip 2>/dev/null || true

MISSING=""
$PYTHON -c "import httpx" 2>/dev/null || MISSING="$MISSING httpx"
$PYTHON -c "import rich" 2>/dev/null || MISSING="$MISSING rich"
$PYTHON -c "import yaml" 2>/dev/null || MISSING="$MISSING pyyaml"
$PYTHON -c "import duckduckgo_search" 2>/dev/null || MISSING="$MISSING duckduckgo-search"
$PYTHON -c "import PyPDF2" 2>/dev/null || MISSING="$MISSING PyPDF2"
$PYTHON -c "import reportlab" 2>/dev/null || MISSING="$MISSING reportlab"
$PYTHON -c "import docx" 2>/dev/null || MISSING="$MISSING python-docx"
$PYTHON -c "from PIL import Image" 2>/dev/null || MISSING="$MISSING Pillow"
$PYTHON -c "import bs4" 2>/dev/null || MISSING="$MISSING beautifulsoup4"
$PYTHON -c "import lxml" 2>/dev/null || MISSING="$MISSING lxml"
$PYTHON -c "import mcp" 2>/dev/null || MISSING="$MISSING mcp"
$PYTHON -c "import pydantic" 2>/dev/null || MISSING="$MISSING pydantic"
$PYTHON -c "import pytesseract" 2>/dev/null || MISSING="$MISSING pytesseract"

if [ -n "$MISSING" ]; then
    info "Installing:${MISSING}"
    $PYTHON -m pip install --quiet $MISSING 2>&1 | while IFS= read -r line; do
        if echo "$line" | grep -qi "error\|fail"; then
            warn "$line"
        fi
    done
    ok "Dependencies installed"
else
    ok "All dependencies already installed"
fi

DEPS_OK=true
$PYTHON -c "import httpx; import rich; import yaml; import duckduckgo_search" 2>/dev/null || DEPS_OK=false

if ! $DEPS_OK; then
    err "Failed to install core dependencies. Try manually:"
    echo -e "${D}  $PYTHON -m pip install httpx rich pyyaml duckduckgo-search${R}"
    exit 1
fi

OPT_MISSING=""
$PYTHON -c "import PyPDF2" 2>/dev/null || OPT_MISSING="$OPT_MISSING PyPDF2"
$PYTHON -c "import reportlab" 2>/dev/null || OPT_MISSING="$OPT_MISSING reportlab"
$PYTHON -c "import docx" 2>/dev/null || OPT_MISSING="$OPT_MISSING python-docx"
$PYTHON -c "from PIL import Image" 2>/dev/null || OPT_MISSING="$OPT_MISSING Pillow"
$PYTHON -c "import bs4" 2>/dev/null || OPT_MISSING="$OPT_MISSING beautifulsoup4"
$PYTHON -c "import lxml" 2>/dev/null || OPT_MISSING="$OPT_MISSING lxml"
$PYTHON -c "import mcp" 2>/dev/null || OPT_MISSING="$OPT_MISSING mcp"
$PYTHON -c "import pydantic" 2>/dev/null || OPT_MISSING="$OPT_MISSING pydantic"
$PYTHON -c "import pytesseract" 2>/dev/null || OPT_MISSING="$OPT_MISSING pytesseract"
$PYTHON -c "import selenium" 2>/dev/null || OPT_MISSING="$OPT_MISSING selenium"
$PYTHON -c "import openpyxl" 2>/dev/null || OPT_MISSING="$OPT_MISSING openpyxl"
$PYTHON -c "from pptx import Presentation" 2>/dev/null || OPT_MISSING="$OPT_MISSING python-pptx"
$PYTHON -c "from webdriver_manager.firefox import GeckoDriverManager" 2>/dev/null || OPT_MISSING="$OPT_MISSING webdriver-manager"

if [ -n "$OPT_MISSING" ]; then
    warn "Optional tools missing:${OPT_MISSING}"
    echo -e "${D}  Install manually:  $PYTHON -m pip install PyPDF2 reportlab python-docx Pillow beautifulsoup4 lxml mcp pydantic pytesseract selenium openpyxl python-pptx webdriver-manager${R}"
    warn "Some tools will be limited without these"
fi

# ═══════════════════════════════════════════════════════════════
# DOWNLOAD / COPY PACKAGE
# ═══════════════════════════════════════════════════════════════

step 4 $TOTAL_STEPS "Setting up DeepSeek CLI package"

mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"

# ── Source detection ──
LOCAL_SOURCE=false
if [ -n "$SCRIPT_DIR" ] && [ -d "$SCRIPT_DIR/deepseek" ] && [ -f "$SCRIPT_DIR/deepseek/__init__.py" ]; then
    LOCAL_SOURCE=true
    info "Installing from local source..."
    rm -rf "$INSTALL_DIR/deepseek" 2>/dev/null || true
    # Clear any cached bytecode that could shadow the new source
    find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    cp -r "$SCRIPT_DIR/deepseek" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    ok "Moved from $SCRIPT_DIR"
fi

if ! $LOCAL_SOURCE; then
    # Download from GitHub — handles: curl pipe, wget pipe, no local source
    GITHUB_RAW="${GITHUB_RAW_URL:-https://raw.githubusercontent.com/XbibzOfficial777/deepseek-cli/main}"
    GITHUB_API="${GITHUB_API_URL:-https://api.github.com/repos/XbibzOfficial777/deepseek-cli/contents/deepseek?ref=main}"
    info "Downloading from GitHub: $GITHUB_RAW"

    TEMP_DIR=$(mktemp -d)
    mkdir -p "$TEMP_DIR/deepseek"

    # ── File list: AUTO-DETECT via GitHub API (preferred) ──────────────
    # Falls back to a hardcoded list (kept in sync with the repo) if the
    # API is rate-limited or unreachable. This is the only way to guarantee
    # no file gets missed (the original bug: auth.py was missing).
    FILES=()
    AUTO_DETECTED=false
    if command -v curl >/dev/null 2>&1; then
        API_RESP=$(curl -fsSL --max-time 10 \
            -H "Accept: application/vnd.github.v3+json" \
            -H "User-Agent: deepseek-cli-installer" \
            "$GITHUB_API" 2>/dev/null || true)
        if [ -n "$API_RESP" ] && echo "$API_RESP" | grep -q '"name"'; then
            # Parse JSON — use python if available, otherwise regex
            if command -v python3 >/dev/null 2>&1; then
                PY_NAMES=$(printf '%s' "$API_RESP" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for item in data:
        if isinstance(item, dict) and item.get('type') == 'file':
            n = item.get('name', '')
            if n.endswith('.py'):
                print('deepseek/' + n)
except Exception:
    pass
" 2>/dev/null || true)
            elif command -v python >/dev/null 2>&1; then
                PY_NAMES=$(printf '%s' "$API_RESP" | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for item in data:
        if isinstance(item, dict) and item.get('type') == 'file':
            n = item.get('name', '')
            if n.endswith('.py'):
                print('deepseek/' + n)
except Exception:
    pass
" 2>/dev/null || true)
            else
                # Pure-sh fallback: extract "name":"X.py" pairs
                PY_NAMES=$(printf '%s' "$API_RESP" | grep -oE '"name"[[:space:]]*:[[:space:]]*"[^"]+\.py"' | sed -n 's/"name"[[:space:]]*:[[:space:]]*"\([^"]*\)"/deepseek\/\1/p')
            fi
            if [ -n "$PY_NAMES" ]; then
                while IFS= read -r f; do
                    [ -n "$f" ] && FILES+=("$f")
                done <<< "$PY_NAMES"
                if [ "${#FILES[@]}" -ge 10 ]; then
                    AUTO_DETECTED=true
                    # Always include requirements.txt (not in deepseek/ dir)
                    FILES+=("requirements.txt")
                fi
            fi
        fi
    fi

    if ! $AUTO_DETECTED; then
        # ── Fallback hardcoded list — must stay in sync with repo ──────
        # Kept as defensive backup so install still works when GitHub API
        # is rate-limited (60 req/h unauthenticated).
        warn "GitHub API auto-detect failed — using hardcoded file list"
        FILES=(
            "deepseek/__init__.py"
            "deepseek/__main__.py"
            "deepseek/agent.py"
            "deepseek/auth.py"
            "deepseek/config.py"
            "deepseek/connectors.py"
            "deepseek/doc_tools.py"
            "deepseek/llm.py"
            "deepseek/mcp_client.py"
            "deepseek/mcp_tools.py"
            "deepseek/memory.py"
            "deepseek/multi_agent.py"
            "deepseek/planner.py"
            "deepseek/providers.py"
            "deepseek/repl.py"
            "deepseek/selenium_browser.py"
            "deepseek/toolkit.py"
            "deepseek/tools.py"
            "deepseek/ui.py"
            "deepseek/webcontrol.py"
            "requirements.txt"
        )
    fi

    info "File list: ${#FILES[@]} files$($AUTO_DETECTED && echo ' (auto)' || echo ' (fallback)')"

    DOWNLOADED=0
    FAILED_FILES=()
    for f in "${FILES[@]}"; do
        url="${GITHUB_RAW}/${f}"
        if curl -fsSL "$url" -o "$TEMP_DIR/$f" 2>/dev/null; then
            DOWNLOADED=$((DOWNLOADED + 1))
        elif wget -qO "$TEMP_DIR/$f" "$url" 2>/dev/null; then
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            FAILED_FILES+=("$f")
            warn "Failed: $f"
        fi
    done

    # Hard fail if we got fewer than 10 files — something is very wrong.
    if [ $DOWNLOADED -lt 10 ]; then
        err "Download failed ($DOWNLOADED/$(( ${#FILES[@]} )) files)"
        echo -e "${D}  Make sure the GitHub repo has all files in main branch${R}"
        echo -e "${D}  Or download the zip and run:  bash install.sh  (from extracted dir)${R}"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    # CRITICAL: wipe the destination first so stale .py files don't linger.
    # `cp -r` only overwrites — it does NOT delete files absent from source.
    # This was the root cause of the "Signed in as Xbibzzz" bug: the old
    # auth.py from a previous install was never deleted because the
    # installer didn't fetch a new one.
    rm -rf "$INSTALL_DIR/deepseek" 2>/dev/null || true
    find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    mkdir -p "$INSTALL_DIR/deepseek"
    cp -r "$TEMP_DIR/deepseek/." "$INSTALL_DIR/deepseek/"
    cp "$TEMP_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    rm -rf "$TEMP_DIR"
    ok "Downloaded $DOWNLOADED files"
    if [ ${#FAILED_FILES[@]} -gt 0 ]; then
        warn "Some files failed: ${FAILED_FILES[*]}"
    fi
fi

# Clear any cached bytecode that could shadow the new source
find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Verify
VERIFY_OK=false
VERIFY_OUT=$($PYTHON -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
try:
    from deepseek.providers import create_provider
    from deepseek.agent import Agent
    from deepseek.toolkit import ToolRegistry
    from deepseek.config import cfg
    t = ToolRegistry()
    print(f'{len(t.tools)} tools')
except Exception as e:
    print(f'Error: {e}')
" 2>&1) && VERIFY_OK=true

if echo "$VERIFY_OUT" | grep -q "Error"; then
    warn "Package verification: $VERIFY_OUT"
else
    ok "Package verified: $VERIFY_OUT"
fi

# Sanity check: ensure no leftover 'Signed in as' (old removed line) in installed code
if grep -rn "console.print.*Signed in as\|print.*Signed in as" "$INSTALL_DIR/deepseek" 2>/dev/null; then
    warn "Stale code detected in install dir. The 'Signed in as' line was removed in v7.7."
    warn "If you still see it after install, run:  bash install.sh --clean"
fi

# ═══════════════════════════════════════════════════════════════
# CREATE 'dscli' COMMAND
# ═══════════════════════════════════════════════════════════════

step 5 $TOTAL_STEPS "Creating 'dscli' command"

WRAPPER="$BIN_DIR/dscli"

cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/usr/bin/env bash
# DeepSeek CLI v7.7 — Launcher
# Generated by install.sh

set -euo pipefail

INSTALL_DIR="__INSTALL_DIR_PLACEHOLDER__"

if [ "$INSTALL_DIR" = "__INSTALL_DIR_PLACEHOLDER__" ]; then
    for d in "$HOME/.local/lib/deepseek-cli" "$HOME/.deepseek-cli" "/usr/local/lib/deepseek-cli"; do
        if [ -f "$d/deepseek/__init__.py" ]; then
            INSTALL_DIR="$d"
            break
        fi
    done
fi

if [ ! -f "$INSTALL_DIR/deepseek/__init__.py" ]; then
    echo -e "\033[31m  ✗ DeepSeek CLI not found!\033[0m"
    echo -e "\033[2m  Run the installer again: bash install.sh\033[0m"
    exit 1
fi

PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo -e "\033[31m  ✗ Python 3 not found!\033[0m"
    exit 1
fi

export DEEPSEEK_ORIGINAL_CWD="$PWD"
cd "$INSTALL_DIR"
exec $PYTHON -m deepseek "$@"
WRAPPER_EOF

sed "s|__INSTALL_DIR_PLACEHOLDER__|$INSTALL_DIR|g" "$WRAPPER" > "$WRAPPER.tmp" && mv "$WRAPPER.tmp" "$WRAPPER"
chmod +x "$WRAPPER"

PATH_NEED_FIX=false
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) PATH_NEED_FIX=true ;;
esac

if $PATH_NEED_FIX; then
    _PATH_INJECTED_BASHRC=false
    _PATH_INJECTED_ZSHRC=false

    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "deepseek-cli" "$HOME/.bashrc" 2>/dev/null; then
            echo "" >> "$HOME/.bashrc"
            echo "# DeepSeek CLI - Auto-added by installer" >> "$HOME/.bashrc"
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.bashrc"
            _PATH_INJECTED_BASHRC=true
            info "Added to PATH in ~/.bashrc"
        fi
    elif [ -f "$HOME/.bash_profile" ]; then
        if ! grep -q "deepseek-cli" "$HOME/.bash_profile" 2>/dev/null; then
            echo "" >> "$HOME/.bash_profile"
            echo "# DeepSeek CLI - Auto-added by installer" >> "$HOME/.bash_profile"
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.bash_profile"
            _PATH_INJECTED_BASHRC=true
            info "Added to PATH in ~/.bash_profile (no .bashrc)"
        fi
    else
        echo "# DeepSeek CLI - Auto-added by installer" > "$HOME/.bashrc"
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.bashrc"
        _PATH_INJECTED_BASHRC=true
        info "Created ~/.bashrc with PATH entry"
    fi

    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "deepseek-cli" "$HOME/.zshrc" 2>/dev/null; then
            echo "" >> "$HOME/.zshrc"
            echo "# DeepSeek CLI - Auto-added by installer" >> "$HOME/.zshrc"
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.zshrc"
            _PATH_INJECTED_ZSHRC=true
            info "Added to PATH in ~/.zshrc"
        fi
    else
        echo "# DeepSeek CLI - Auto-added by installer" > "$HOME/.zshrc"
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.zshrc"
        _PATH_INJECTED_ZSHRC=true
        info "Created ~/.zshrc with PATH entry"
    fi

    export PATH="$BIN_DIR:$PATH"
    info "PATH exported for current session"
fi

if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc" 2>/dev/null || true
fi
if [ -f "$HOME/.zshrc" ]; then
    . "$HOME/.zshrc" 2>/dev/null || true
fi

if command -v dscli &>/dev/null; then
    ok "Command 'dscli' created at $WRAPPER"
else
    ok "Command 'dscli' created at $WRAPPER"
    if $PATH_NEED_FIX; then
        warn "PATH not updated for current session. Restart terminal or run:"
        echo -e "${D}    export PATH=\"$BIN_DIR:\$PATH\"${R}"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# CLEANUP — remove cache, pycache, temp files
# ═══════════════════════════════════════════════════════════════

step 6 $TOTAL_STEPS "Cleaning up cache & temporary files"

CLEANED=0

# Remove __pycache__ directories
if [ -d "$INSTALL_DIR/deepseek/__pycache__" ]; then
    rm -rf "$INSTALL_DIR/deepseek/__pycache__" 2>/dev/null
    CLEANED=$((CLEANED + 1))
fi

# Remove any .pyc files (compiled bytecode)
find "$INSTALL_DIR/deepseek" -name "*.pyc" -delete 2>/dev/null || true

# Remove pip cache if this was a fresh install
$PYTHON -m pip cache purge 2>/dev/null || true

# Remove temp download directory (if any was left)
rm -rf /tmp/tmp.*deepseek* /tmp/tmp.*dscli* 2>/dev/null || true

if [ $CLEANED -gt 0 ]; then
    ok "Cache cleaned ($CLEANED dirs)"
else
    ok "No cache files found"
fi

# ═══════════════════════════════════════════════════════════════
# SETUP CONFIG (first run)
# ═══════════════════════════════════════════════════════════════

CONFIG_DIR="$HOME/.deepseek-cli"
CONFIG_FILE="$CONFIG_DIR/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    info "Creating default config..."
    $PYTHON -c "
import sys; sys.path.insert(0, '$INSTALL_DIR')
from deepseek.config import cfg
cfg.save()
print('  Config saved to ~/.deepseek-cli/config.yaml')
" 2>/dev/null || true
fi

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════

echo ""
echo -e "${GR}${B}  ╔══════════════════════════════════════════╗${R}"
echo -e "${GR}${B}  ║         Install Complete!                ║${R}"
echo -e "${GR}${B}  ╚══════════════════════════════════════════╝${R}"
echo ""
echo -e "${D}  Run:  ${CY}${B}dscli${R}${D}  to start DeepSeek CLI${R}"
echo ""
echo -e "${D}  Quick start:${R}"
echo -e "${D}    dscli                      ${D}# Launch REPL${R}"
echo -e "${D}    dscli install <package>    ${D}# Install a skill (npx skills add)${R}"
echo -e "${D}    Ctrl+P                     ${D}# Open settings panel${R}"
echo -e "${D}    /provider                   ${D}# Switch provider (arrow keys)${R}"
echo -e "${D}    /model                      ${D}# Switch model (arrow keys)${R}"
echo -e "${D}    /key                        ${D}# Set API key${R}"
echo -e "${D}    /skills                     ${D}# Manage installed skills${R}"
echo -e "${D}    /init                       ${D}# Scan project & create AGENTS.md${R}"
echo ""
echo -e "${D}  Supported providers:${R}"
echo -e "${D}    OpenRouter · Gemini · HuggingFace · OpenAI · Anthropic · Groq · Together${R}"
echo ""
echo -e "${D}  90+ Tools: File Ops, Web Search, Code, System, Math, Utility,${R}"
echo -e "${D}    PDF, DOCX, Image, Video, OCR, APK, Live Search, Web Browser,${R}"
echo -e "${D}    Selenium Automation, PPTX, XLSX, CSV, Document Conversion,${R}"
echo -e "${D}    Telegram & Discord Connectors${R}"
echo -e "${D}  v7.7: Rich Markdown, Smooth Buffer, TUI Status Bar, Auth Automation${R}"
echo ""

if [ $# -eq 0 ] && tty -s; then
    echo -en "${CY}${B}  Launch DeepSeek CLI now? [Y/n]${R} "
    read -r ANSWER </dev/tty 2>/dev/null || ANSWER="y"
    case "$ANSWER" in
        n*|N*) ;;
        *) echo ""; exec dscli ;;
    esac
fi
