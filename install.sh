#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  DeepSeek CLI v7.7 — Installer / Uninstaller (Modernized)
#  Multi-Provider AI Agent | 8 Providers | 90+ Tools | Smart Loop | OCR
#  Features: Live Search, Browser Automation, Telegram & Discord Connectors
#  Document Tools (PPTX/XLSX/DOCX/CSV/PDF), Selenium, Rich Markdown UI
#
#  Usage:
#    bash install.sh                         # Install
#    bash install.sh --uninstall             # Uninstall
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

TOTAL_STEPS=7  # increased because we added venv step

# ── Spinner (arc rotate + flading text) ────────────────
SPINNER_PID=""
SPINNER_TMP=""

_cleanup_spinner() {
    if [ -n "$SPINNER_PID" ] && kill -0 "$SPINNER_PID" 2>/dev/null; then
        kill "$SPINNER_PID" 2>/dev/null || true
        wait "$SPINNER_PID" 2>/dev/null || true
    fi
    [ -f "$SPINNER_TMP" ] && rm -f "$SPINNER_TMP"
}

start_spinner() {
    local msg="${1:-Installation In Progress ...}"
    # Create temporary Python script for the animation
    SPINNER_TMP="$(mktemp)"
    cat > "$SPINNER_TMP" << 'PYEOF'
import sys, time, math
msg = sys.argv[1] if len(sys.argv) > 1 else "Installation In Progress ..."
frames = ['◜', '◝', '◞', '◟']
idx = 0
while True:
    t = time.time() * 3
    wave = ''
    for i, ch in enumerate(msg):
        val = (math.sin(i * 1.5 - t) + 1) / 2
        code = 232 + int(val * 23)
        wave += f"\033[38;5;{code}m{ch}\033[0m"
    f = frames[idx % 4]
    idx += 1
    sys.stdout.write(f'\r  {f}  {wave}')
    sys.stdout.flush()
    time.sleep(0.08)
PYEOF
    # Start spinner in background
    $PYTHON "$SPINNER_TMP" "$msg" &
    SPINNER_PID=$!
    # Ensure spinner is killed on exit
    trap _cleanup_spinner EXIT INT TERM
}

stop_spinner() {
    _cleanup_spinner
    trap - EXIT INT TERM
    # Clear the line
    printf '\r\033[2K'
}

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
case " ${0:-} ${1:-} ${*} " in
    *\ --uninstall\ *|*\ uninstall\ *) UNINSTALL_MODE=true ;;
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

    # 5. Remove all ~/.deepseek-cli/ contents EXCEPT config.yaml and venv (if user wants to keep venv, we skip)
    if [ -d "$DEEPSEEK_CONFIG" ]; then
        echo -e "  ${CY}▸${R} Cleaning: ${D}$DEEPSEEK_CONFIG (keeping config.yaml and venv if present)${R}"
        for item in "$DEEPSEEK_CONFIG"/*; do
            [ -e "$item" ] || break
            basename_item="$(basename "$item")"
            if [ "$basename_item" != "config.yaml" ] && [ "$basename_item" != "venv" ]; then
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

    echo ""
    if $FOUND; then
        echo -e "  ${GR}${B}✓ DeepSeek CLI has been uninstalled.${R}"
        echo -e "  ${GR}${B}✓ API key and venv kept in ${D}$DEEPSEEK_CONFIG/${R}"
    else
        echo -e "  ${YE}DeepSeek CLI is not installed or already removed.${R}"
    fi
    echo -e "  ${D}Run 'bash install.sh' to reinstall anytime.${R}"
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
# SET UP VIRTUAL ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

step 3 $TOTAL_STEPS "Creating virtual environment"

# Customizable venv directory via env var
DEEPSEEK_VENV_DIR="${DEEPSEEK_VENV_DIR:-$HOME/.deepseek-cli/venv}"
export DEEPSEEK_VENV_DIR

if [ ! -d "$DEEPSEEK_VENV_DIR" ]; then
    info "Creating venv at: $DEEPSEEK_VENV_DIR"
    # Ensure parent dir exists
    mkdir -p "$(dirname "$DEEPSEEK_VENV_DIR")"
    start_spinner "Creating virtual environment ..."
    $PYTHON -m venv "$DEEPSEEK_VENV_DIR" --without-pip 2>/dev/null || {
        stop_spinner
        err "Failed to create virtual environment. Try: $PYTHON -m venv $DEEPSEEK_VENV_DIR"
        exit 1
    }
    stop_spinner
    ok "Virtual environment created"
else
    ok "Using existing venv at: $DEEPSEEK_VENV_DIR"
fi

# Ensure pip is present in venv
VENV_PYTHON="$DEEPSEEK_VENV_DIR/bin/python"
VENV_PIP="$DEEPSEEK_VENV_DIR/bin/pip"

if [ ! -f "$VENV_PYTHON" ]; then
    err "Virtual environment python not found: $VENV_PYTHON"
    exit 1
fi

# Bootstrap pip if missing
if ! "$VENV_PYTHON" -m pip --version &>/dev/null; then
    info "Installing pip inside venv..."
    start_spinner "Bootstrapping pip ..."
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VENV_PYTHON" - --quiet 2>/dev/null || {
        stop_spinner
        err "Failed to install pip. Try manually: $VENV_PYTHON -m ensurepip"
        exit 1
    }
    stop_spinner
    ok "pip installed"
else
    ok "pip already available"
fi

# Upgrade pip
"$VENV_PIP" install --quiet --upgrade pip 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# INSTALL DEPENDENCIES (inside venv)
# ═══════════════════════════════════════════════════════════════

step 4 $TOTAL_STEPS "Installing Python dependencies"

# Determine required packages (core + optional)
REQUIRED_PKGS="httpx rich pyyaml duckduckgo-search"
OPTIONAL_PKGS="PyPDF2 reportlab python-docx Pillow beautifulsoup4 lxml mcp pydantic pytesseract selenium openpyxl python-pptx webdriver-manager"

# Check which are already installed in venv
MISSING=""
for pkg in $REQUIRED_PKGS; do
    if ! "$VENV_PYTHON" -c "import $(echo $pkg | sed 's/-/_/g')" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    info "Installing core packages:${MISSING}"
    start_spinner "Installing dependencies ..."
    # Use venv pip
    if ! "$VENV_PIP" install --quiet $MISSING 2>/dev/null; then
        stop_spinner
        err "Failed to install core dependencies. Retrying with verbose..."
        "$VENV_PIP" install $MISSING || exit 1
    fi
    stop_spinner
    ok "Core dependencies installed"
else
    ok "All core dependencies already installed"
fi

# Verify core dependencies
if ! "$VENV_PYTHON" -c "import httpx, rich, yaml, duckduckgo_search" 2>/dev/null; then
    err "Core dependencies still missing. Manual install:"
    echo -e "${D}  $VENV_PIP install $REQUIRED_PKGS${R}"
    exit 1
fi

# Optional packages: install if missing (but don't fail)
OPT_MISSING=""
for pkg in $OPTIONAL_PKGS; do
    # Convert package name to import name (e.g., python-docx -> docx, Pillow -> PIL)
    import_name=$(echo $pkg | sed 's/-/_/g' | sed 's/^python-//')
    if [ "$pkg" = "Pillow" ]; then import_name="PIL"; fi
    if [ "$pkg" = "python-pptx" ]; then import_name="pptx"; fi
    if [ "$pkg" = "webdriver-manager" ]; then import_name="webdriver_manager"; fi
    if ! "$VENV_PYTHON" -c "import $import_name" 2>/dev/null; then
        OPT_MISSING="$OPT_MISSING $pkg"
    fi
done

if [ -n "$OPT_MISSING" ]; then
    warn "Optional tools missing:${OPT_MISSING}"
    echo -e "${D}  Installing them now (this may take a moment)...${R}"
    start_spinner "Installing optional packages ..."
    "$VENV_PIP" install --quiet $OPT_MISSING 2>/dev/null || warn "Some optional packages failed to install"
    stop_spinner
    ok "Optional packages installed (if available)"
fi

# ═══════════════════════════════════════════════════════════════
# DOWNLOAD / COPY PACKAGE
# ═══════════════════════════════════════════════════════════════

step 5 $TOTAL_STEPS "Setting up DeepSeek CLI package"

mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"

# ── Source detection ──
LOCAL_SOURCE=false
if [ -n "$SCRIPT_DIR" ] && [ -d "$SCRIPT_DIR/deepseek" ] && [ -f "$SCRIPT_DIR/deepseek/__init__.py" ]; then
    LOCAL_SOURCE=true
    info "Installing from local source..."
    rm -rf "$INSTALL_DIR/deepseek" 2>/dev/null || true
    cp -r "$SCRIPT_DIR/deepseek" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    ok "Moved from $SCRIPT_DIR"
fi

if ! $LOCAL_SOURCE; then
    GITHUB_RAW="${GITHUB_RAW_URL:-https://raw.githubusercontent.com/XbibzOfficial777/deepseek-cli/main}"
    info "Downloading from GitHub: $GITHUB_RAW"

    TEMP_DIR=$(mktemp -d)
    mkdir -p "$TEMP_DIR/deepseek"

    FILES=(
        "deepseek/__init__.py"
        "deepseek/__main__.py"
        "deepseek/llm.py"
        "deepseek/auth.py"
        "deepseek/config.py"
        "deepseek/providers.py"
        "deepseek/agent.py"
        "deepseek/toolkit.py"
        "deepseek/memory.py"
        "deepseek/ui.py"
        "deepseek/repl.py"
        "deepseek/mcp_tools.py"
        "deepseek/mcp_client.py"
        "deepseek/multi_agent.py"
        "deepseek/doc_tools.py"
        "deepseek/webcontrol.py"
        "deepseek/selenium_browser.py"
        "deepseek/connectors.py"
        "deepseek/planner.py"
        "deepseek/tools.py"
        "requirements.txt"
    )

    start_spinner "Downloading package files ..."
    DOWNLOADED=0
    for f in "${FILES[@]}"; do
        url="${GITHUB_RAW}/${f}"
        if curl -fsSL "$url" -o "$TEMP_DIR/$f" 2>/dev/null; then
            DOWNLOADED=$((DOWNLOADED + 1))
        elif wget -qO "$TEMP_DIR/$f" "$url" 2>/dev/null; then
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            warn "Failed: $f"
        fi
    done
    stop_spinner

    if [ $DOWNLOADED -lt 10 ]; then
        err "Download failed ($DOWNLOADED/$(( ${#FILES[@]} )) files)"
        echo -e "${D}  Make sure the GitHub repo has all files in main branch${R}"
        echo -e "${D}  Or download the zip and run:  bash install.sh  (from extracted dir)${R}"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    cp -r "$TEMP_DIR/deepseek" "$INSTALL_DIR/"
    cp "$TEMP_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    rm -rf "$TEMP_DIR"
    ok "Downloaded $DOWNLOADED files"
fi

# Verify
VERIFY_OK=false
VERIFY_OUT=$("$VENV_PYTHON" -c "
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

# ═══════════════════════════════════════════════════════════════
# CREATE 'dscli' COMMAND (using venv)
# ═══════════════════════════════════════════════════════════════

step 6 $TOTAL_STEPS "Creating 'dscli' command"

WRAPPER="$BIN_DIR/dscli"

cat > "$WRAPPER" << WRAPPER_EOF
#!/usr/bin/env bash
# DeepSeek CLI v7.7 — Launcher (venv)
# Generated by install.sh

set -euo pipefail

# Use the venv from installation
DEEPSEEK_VENV_DIR="__DEEPSEEK_VENV_DIR_PLACEHOLDER__"
if [ ! -d "\$DEEPSEEK_VENV_DIR" ]; then
    # Fallback: try to locate
    for d in "\$HOME/.deepseek-cli/venv" "\$HOME/.local/lib/deepseek-cli/venv" "/usr/local/lib/deepseek-cli/venv"; do
        if [ -d "\$d" ]; then
            DEEPSEEK_VENV_DIR="\$d"
            break
        fi
    done
fi

if [ ! -d "\$DEEPSEEK_VENV_DIR" ]; then
    echo -e "\033[31m  ✗ Virtual environment not found!\033[0m"
    echo -e "\033[2m  Reinstall with: bash install.sh\033[0m"
    exit 1
fi

PYTHON="\$DEEPSEEK_VENV_DIR/bin/python"
if [ ! -f "\$PYTHON" ]; then
    echo -e "\033[31m  ✗ Python not found in venv!\033[0m"
    exit 1
fi

# Ensure the package is in PYTHONPATH
INSTALL_DIR="__INSTALL_DIR_PLACEHOLDER__"
if [ ! -d "\$INSTALL_DIR/deepseek" ]; then
    # Fallback: try common locations
    for d in "\$HOME/.local/lib/deepseek-cli" "\$HOME/.deepseek-cli/lib" "/usr/local/lib/deepseek-cli"; do
        if [ -d "\$d/deepseek" ]; then
            INSTALL_DIR="\$d"
            break
        fi
    done
fi

if [ ! -f "\$INSTALL_DIR/deepseek/__init__.py" ]; then
    echo -e "\033[31m  ✗ DeepSeek CLI package not found!\033[0m"
    exit 1
fi

export DEEPSEEK_ORIGINAL_CWD="\$PWD"
cd "\$INSTALL_DIR"
exec "\$PYTHON" -m deepseek "\$@"
WRAPPER_EOF

# Replace placeholders
sed -i "s|__DEEPSEEK_VENV_DIR_PLACEHOLDER__|$DEEPSEEK_VENV_DIR|g" "$WRAPPER"
sed -i "s|__INSTALL_DIR_PLACEHOLDER__|$INSTALL_DIR|g" "$WRAPPER"
chmod +x "$WRAPPER"

# Add to PATH if needed
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

step 7 $TOTAL_STEPS "Cleaning up cache & temporary files"

CLEANED=0

# Remove __pycache__ directories
if [ -d "$INSTALL_DIR/deepseek/__pycache__" ]; then
    rm -rf "$INSTALL_DIR/deepseek/__pycache__" 2>/dev/null
    CLEANED=$((CLEANED + 1))
fi

# Remove any .pyc files
find "$INSTALL_DIR/deepseek" -name "*.pyc" -delete 2>/dev/null || true

# Remove pip cache (inside venv)
"$VENV_PIP" cache purge 2>/dev/null || true

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
    "$VENV_PYTHON" -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
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
echo -e "${D}  Virtual environment: ${CY}${B}$DEEPSEEK_VENV_DIR${R}"
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