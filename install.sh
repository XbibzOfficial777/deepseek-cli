#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  DeepSeek CLI v6.1 — Installer
#  Multi-Provider AI Agent | 7 Providers | 67+ Tools | Smart Loop | OCR
#  Features: Live Search, Live Model Search, Ctrl+P, Arrow-Key Select
#  v6.1: OCR support (pytesseract + easyocr), Professional Rich UI animations
#
#  Install methods:
#    1) bash install.sh                      (from downloaded file)
#    2) bash -c "$(curl -fsSL RAW_URL)"      (from GitHub raw)
#    3) bash -c "$(wget -qO- RAW_URL)"       (from GitHub raw, wget)
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

TOTAL_STEPS=5

# ═══════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════

echo ""
echo -e "${CY}${B}  ╔══════════════════════════════════════════╗${R}"
echo -e "${CY}${B}  ║      DeepSeek CLI v6.1  Installer       ║${R}"
echo -e "${CY}${B}  ║  Multi-Provider AI Agent · 7 Services   ║${R}"
echo -e "${CY}${B}  ║  67+ Tools · Smart Loop · OCR · Rich    ║${R}"
echo -e "${CY}${B}  ╚══════════════════════════════════════════╝${R}"
echo ""

# ═══════════════════════════════════════════════════════════════
# DETECT ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

step 1 $TOTAL_STEPS "Detecting environment"

# Fix: initialize PREFIX for set -u (nounset) safety
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

# Ensure pip is available
if ! $PYTHON -m pip --version &>/dev/null 2>&1; then
    info "Installing pip..."
    if $IS_TERMUX; then
        pkg install -y python-pip 2>/dev/null || $PYTHON -m ensurepip --upgrade 2>/dev/null || true
    else
        curl -fsSL https://bootstrap.pypa.io/get-pip.py | $PYTHON - --quiet 2>/dev/null || true
    fi
fi

# Upgrade pip
$PYTHON -m pip install --quiet --upgrade pip 2>/dev/null || true

# Install packages — check each one
MISSING=""
$PYTHON -c "import httpx" 2>/dev/null || MISSING="$MISSING httpx"
$PYTHON -c "import rich" 2>/dev/null || MISSING="$MISSING rich"
$PYTHON -c "import yaml" 2>/dev/null || MISSING="$MISSING pyyaml"
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

# Verify core deps
DEPS_OK=true
$PYTHON -c "import httpx; import rich; import yaml" 2>/dev/null || DEPS_OK=false

if ! $DEPS_OK; then
    err "Failed to install core dependencies. Try manually:"
    echo -e "${D}  $PYTHON -m pip install httpx rich pyyaml${R}"
    exit 1
fi

# Optional deps warning (not blocking)
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

if [ -n "$OPT_MISSING" ]; then
    warn "Optional tools missing:${OPT_MISSING}"
    echo -e "${D}  Install manually:  $PYTHON -m pip install PyPDF2 reportlab python-docx Pillow lxml mcp pydantic pytesseract selenium${R}"
    warn "Some tools will be limited without these"
fi

# ═══════════════════════════════════════════════════════════════
# DOWNLOAD / COPY PACKAGE
# ═══════════════════════════════════════════════════════════════

step 4 $TOTAL_STEPS "Setting up DeepSeek CLI package"

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR" 2>/dev/null || true

# If deepseek/ folder exists in current dir, use it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"

if [ -d "$SCRIPT_DIR/deepseek" ] && [ -f "$SCRIPT_DIR/deepseek/__init__.py" ]; then
    info "Installing from local source..."
    cp -r "$SCRIPT_DIR/deepseek" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    ok "Copied from $SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/deepseek/__init__.py" ]; then
    # Script is inside the package dir (e.g. GitHub raw single file)
    info "Installing from GitHub source..."
    GITHUB_RAW="${GITHUB_RAW_URL:-https://raw.githubusercontent.com/XbibzOfficial777/deepseek-cli/main}"
    info "Downloading from: $GITHUB_RAW"

    TEMP_DIR=$(mktemp -d)
    mkdir -p "$TEMP_DIR/deepseek"

    FILES=(
        "deepseek/__init__.py"
        "deepseek/__main__.py"
        "deepseek/llm.py"
        "deepseek/config.py"
        "deepseek/providers.py"
        "deepseek/agent.py"
        "deepseek/toolkit.py"
        "deepseek/memory.py"
        "deepseek/ui.py"
        "deepseek/repl.py"
        "deepseek/mcp_tools.py"
        "requirements.txt"
    )

    DOWNLOADED=0
    FAILED=0
    for f in "${FILES[@]}"; do
        url="${GITHUB_RAW}/${f}"
        if curl -fsSL "$url" -o "$TEMP_DIR/$f" 2>/dev/null; then
            DOWNLOADED=$((DOWNLOADED + 1))
        elif wget -qO "$TEMP_DIR/$f" "$url" 2>/dev/null; then
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            FAILED=$((FAILED + 1))
            warn "Failed: $f"
        fi
    done

    if [ $DOWNLOADED -lt 8 ]; then
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
else
    # Already installed — check if update needed
    if [ -f "$INSTALL_DIR/deepseek/__init__.py" ]; then
        info "DeepSeek CLI already installed at $INSTALL_DIR"
        echo -e "${D}  Re-run to update. Current version:${R}"
        $PYTHON -c "
import sys; sys.path.insert(0, '$INSTALL_DIR')
try:
    from deepseek.config import DEFAULT_PROVIDERS
    print(f'  {len(DEFAULT_PROVIDERS)} providers configured')
except Exception as e:
    print(f'  Error: {e}')
" 2>/dev/null || warn "Could not detect version"
    else
        err "No source found! Download the release zip and run install.sh from inside."
        echo -e "${D}  1) Download:  https://github.com/XbibzOfficial777/deepseek-cli${R}"
        echo -e "${D}  2) Extract:   unzip deepseek-cli-v5.1.zip${R}"
        echo -e "${D}  3) Install:   cd deepseek-cli-v5.1 && bash install.sh${R}"
        exit 1
    fi
fi

# Verify the package works
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
    VERIFY_OK = True
except Exception as e:
    print(f'Error: {e}')
    VERIFY_OK = False
" 2>&1) && VERIFY_OK=true

if echo "$VERIFY_OUT" | grep -q "Error"; then
    warn "Package verification: $VERIFY_OUT"
else
    ok "Package verified: $VERIFY_OUT"
fi

# ═══════════════════════════════════════════════════════════════
# CREATE 'dscli' COMMAND
# ═══════════════════════════════════════════════════════════════

step 5 $TOTAL_STEPS "Creating 'dscli' command"

WRAPPER="$BIN_DIR/dscli"

cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/usr/bin/env bash
# DeepSeek CLI v5.2 — Launcher
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

cd "$INSTALL_DIR"
exec $PYTHON -m deepseek "$@"
WRAPPER_EOF

sed -i "s|__INSTALL_DIR_PLACEHOLDER__|$INSTALL_DIR|g" "$WRAPPER"
chmod +x "$WRAPPER"

# Ensure BIN_DIR is in PATH — inject into BOTH .bashrc AND .zshrc
PATH_NEED_FIX=false
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) PATH_NEED_FIX=true ;;
esac

if $PATH_NEED_FIX; then
    # Track which files were modified
    _PATH_INJECTED_BASHRC=false
    _PATH_INJECTED_ZSHRC=false

    # --- Inject into .bashrc ---
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
        # Create .bashrc with the PATH entry
        echo "# DeepSeek CLI - Auto-added by installer" > "$HOME/.bashrc"
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.bashrc"
        _PATH_INJECTED_BASHRC=true
        info "Created ~/.bashrc with PATH entry"
    fi

    # --- Inject into .zshrc ---
    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "deepseek-cli" "$HOME/.zshrc" 2>/dev/null; then
            echo "" >> "$HOME/.zshrc"
            echo "# DeepSeek CLI - Auto-added by installer" >> "$HOME/.zshrc"
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.zshrc"
            _PATH_INJECTED_ZSHRC=true
            info "Added to PATH in ~/.zshrc"
        fi
    else
        # Create .zshrc with the PATH entry
        echo "# DeepSeek CLI - Auto-added by installer" > "$HOME/.zshrc"
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.zshrc"
        _PATH_INJECTED_ZSHRC=true
        info "Created ~/.zshrc with PATH entry"
    fi

    # Export for current session
    export PATH="$BIN_DIR:$PATH"
    info "PATH exported for current session"
fi

# Always source the config for immediate availability
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
echo -e "${D}    Ctrl+P                     ${D}# Open settings panel${R}"
echo -e "${D}    /provider                   ${D}# Switch provider (arrow keys)${R}"
echo -e "${D}    /model                      ${D}# Switch model (arrow keys)${R}"
echo -e "${D}    /key                        ${D}# Set API key${R}"
echo ""
echo -e "${D}  Supported providers:${R}"
echo -e "${D}    OpenRouter · Gemini · HuggingFace · OpenAI · Anthropic · Groq · Together${R}"
echo ""
echo -e "${D}  All providers support 67+ tools/skills with smart loop protection!${R}"
echo -e "${D}  NEW v6.1: OCR support (pytesseract + easyocr), Professional Rich UI${R}"
echo ""

# Offer to launch
if [ $# -eq 0 ] && tty -s; then
    echo -en "${CY}${B}  Launch DeepSeek CLI now? [Y/n]${R} "
    read -r ANSWER </dev/tty 2>/dev/null || ANSWER="y"
    case "$ANSWER" in
        n*|N*) ;;
        *) echo ""; exec dscli ;;
    esac
fi
