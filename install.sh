#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  DeepSeek CLI v7.7 — Modern Installer
#
#  Features:
#    • Modern animations: arc_rotate spinner + flading_text gradient wave
#    • Dedicated virtualenv: zero pip conflicts, isolated dependencies
#    • Custom venv name: set DEEPSEEK_VENV_NAME=myenv or DEEPSEEK_VENV_DIR=/path
#    • Auto-detect all files from GitHub Contents API
#    • Cross-platform: Linux, macOS, Termux
#
#  Usage:
#    bash install.sh                          # Install
#    bash install.sh --uninstall              # Uninstall
#    bash install.sh --clean                  # Deep clean (remove all + venv)
#    bash install.sh --rebuild                # Recreate venv from scratch
#    DEEPSEEK_VENV_NAME=myenv bash install.sh  # Custom venv name
#    DEEPSEEK_VENV_DIR=/custom bash install.sh  # Custom venv path
#    bash -c "$(curl -fsSL RAW_URL)"           # Install via curl pipe
#
#  After install, run:  dscli
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
#  COLORS & SYMBOLS
# ═══════════════════════════════════════════════════════════════

R="\033[0m"; B="\033[1m"; D="\033[2m"; CY="\033[36m"
GR="\033[32m"; RD="\033[31m"; YE="\033[33m"; PU="\033[35m"
BL="\033[34m"; MG="\033[35m"

# ═══════════════════════════════════════════════════════════════
#  ANIMATION FRAMES
# ═══════════════════════════════════════════════════════════════

# Arc rotate — 4-frame smooth rotating arc
ARC_FRAMES=('◜' '◝' '◞' '◟')

# Dots — classic loading spinner
DOT_FRAMES=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')

# ═══════════════════════════════════════════════════════════════
#  ANIMATION ENGINE — pure bash, no python dependency
# ═══════════════════════════════════════════════════════════════

# ANSI 256-color grayscale range (232–255) for the flading text
FLAD_MIN=232
FLAD_MAX=255

# Get the visible width of a string (strip ANSI codes)
# Usage: vis_len "string"
vis_len() {
    local s="$*"
    # Strip ANSI escape sequences (ESC[...m)
    local stripped
    stripped=$(printf '%s' "$s" | sed -r "s/\x1b\[[0-9;]*[mGKHJ]//g")
    echo "${#stripped}"
}

# Pad a string to a given width with spaces
# Usage: pad_str "string" width
pad_str() {
    local s="$1"
    local width="$2"
    local current
    current=$(vis_len "$s")
    local pad=$((width - current))
    if [ "$pad" -gt 0 ]; then
        printf '%s%*s' "$s" "$pad" ""
    else
        printf '%s' "$s"
    fi
}

# Render the flading text gradient for a given phase
# Usage: render_flading "text" phase
render_flading() {
    local text="$1"
    local phase="$2"
    local len=${#text}
    local out=""
    for ((j=0; j<len; j++)); do
        # Each character has a phase offset; total phase drives the wave
        local idx=$(( (j * 2 + phase) % 24 ))
        local code=$((FLAD_MIN + idx))
        local ch="${text:$j:1}"
        out+="\033[38;5;${code}m${ch}\033[0m"
    done
    printf '%s' "$out"
}

# Background animation loop — runs until ANIM_STOP file is removed
# Usage: start_animation "label"
start_animation() {
    local label="$1"
    ANIM_LABEL="$label"
    ANIM_STOP_FILE=$(mktemp -u)
    # Create the stop file
    touch "$ANIM_STOP_FILE"

    (
        local i=0
        while [ -f "$ANIM_STOP_FILE" ]; do
            local arc="${ARC_FRAMES[$((i % 4))]}"
            # Build: "{arc_rotate} {flading_text label}"
            local out="  ${B}${YE}${arc}${R}  "
            out+="$(render_flading "$ANIM_LABEL" "$i")"
            # Clear to end of line so any leftover from previous writes is gone
            printf '\r%s\033[K' "$out" >&2
            sleep 0.06
            i=$((i + 1))
        done
    ) &
    ANIM_PID=$!
}

# Stop the background animation and clear the line
# Usage: stop_animation
stop_animation() {
    if [ -n "${ANIM_STOP_FILE:-}" ] && [ -f "$ANIM_STOP_FILE" ]; then
        rm -f "$ANIM_STOP_FILE"
    fi
    if [ -n "${ANIM_PID:-}" ]; then
        kill "$ANIM_PID" 2>/dev/null || true
        wait "$ANIM_PID" 2>/dev/null || true
        ANIM_PID=""
    fi
    # Clear the animation line
    printf '\r\033[K' >&2
}

# Run a command with the modern animation showing label during execution
# Usage: run_animated "label" command [args...]
run_animated() {
    local label="$1"
    shift
    local log
    log=$(mktemp)

    # Run the animation in background
    local saved_tty_settings=""
    if [ -t 1 ]; then
        saved_tty_settings=$(stty -g 2>/dev/null || echo "")
    fi

    start_animation "$label"

    # Run the actual command, capturing output to log file
    set +e
    "$@" > "$log" 2>&1
    local rc=$?
    set -e

    stop_animation

    if [ "$rc" -ne 0 ]; then
        echo -e "  ${RD}${B}✗${R} ${B}Failed:${R} ${label}" >&2
        if [ -s "$log" ]; then
            echo -e "  ${D}─── error output ───${R}" >&2
            head -20 "$log" | sed 's/^/    /' >&2
            echo -e "  ${D}─────────────────────${R}" >&2
        fi
        rm -f "$log"
        return "$rc"
    fi

    rm -f "$log"
    return 0
}

# Cleanup animation on exit
trap 'stop_animation 2>/dev/null || true' EXIT INT TERM

# ═══════════════════════════════════════════════════════════════
#  LOG HELPERS
# ═══════════════════════════════════════════════════════════════

info()  { echo -e "  ${CY}${B}▸${R} ${D}$1${R}"; }
ok()    { echo -e "  ${GR}${B}✓${R} ${D}$1${R}"; }
warn()  { echo -e "  ${YE}${B}⚠${R} ${D}$1${R}"; }
err()  { echo -e "  ${RD}${B}✗${R} ${D}$1${R}"; }
step()  { echo -e "\n  ${BL}${B}▸ $1${R} ${D}$2${R}"; }
head()  { echo -e "\n  ${PU}${B}═══ $1 ═══${R}"; }

TOTAL_STEPS=6

# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════

# ASCII art logo
cat <<'BANNER' >&2

  ╔══════════════════════════════════════════╗
  ║      DeepSeek CLI v7.7  Installer       ║
  ║         Developer : @XbibzOfficial777   ║
  ╚══════════════════════════════════════════╝

BANNER

# ═══════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════

UNINSTALL_MODE=false
CLEAN_MODE=false
REBUILD_MODE=false
case " ${0:-} ${1:-} ${*} " in
    *\ --uninstall\ *|*\ uninstall\ *) UNINSTALL_MODE=true ;;
    *\ --clean\ *) CLEAN_MODE=true ;;
    *\ --rebuild\ *) REBUILD_MODE=true ;;
esac

# ═══════════════════════════════════════════════════════════════
#  UNINSTALL MODE
# ═══════════════════════════════════════════════════════════════

if $UNINSTALL_MODE; then
    echo -e "  ${YE}${B}Uninstall mode${R}" >&2
    echo "" >&2

    FOUND=false

    # 1. Remove package directory + wrapper
    for d in "$HOME/.local/lib/deepseek-cli" "$HOME/.deepseek-cli" "/usr/local/lib/deepseek-cli"; do
        if [ -d "$d" ]; then
            echo -e "  ${CY}▸${R} Removing package: ${D}$d${R}" >&2
            rm -rf "$d" 2>/dev/null || true
            FOUND=true
        fi
    done

    for d in /usr/local/bin "$HOME/.local/bin" "${PREFIX:-/data/data/com.termux/files/usr}/bin"; do
        if [ -f "$d/dscli" ]; then
            echo -e "  ${CY}▸${R} Removing wrapper: ${D}$d/dscli${R}" >&2
            rm -f "$d/dscli"
            FOUND=true
        fi
    done

    # 2. Remove user config (keep API key file)
    DEEPSEEK_CONFIG="$HOME/.deepseek-cli"
    if [ -d "$DEEPSEEK_CONFIG" ]; then
        echo -e "  ${CY}▸${R} Cleaning: ${D}$DEEPSEEK_CONFIG (keeping config.yaml)${R}" >&2
        for item in "$DEEPSEEK_CONFIG"/*; do
            [ -e "$item" ] || break
            base_item="$(basename "$item")"
            if [ "$base_item" != "config.yaml" ]; then
                rm -rf "$item"
            fi
        done
        FOUND=true
    fi

    # 3. --clean: also remove venv + pip package + rogue site-packages
    if $CLEAN_MODE; then
        echo "" >&2
        echo -e "  ${YE}${B}--clean mode: removing venv + rogue installations${R}" >&2

        # Remove the venv (default location)
        DEEPSEEK_VENV_DIR="${DEEPSEEK_VENV_DIR:-}"
        DEEPSEEK_VENV_NAME="${DEEPSEEK_VENV_NAME:-dscli-env}"
        if [ -z "$DEEPSEEK_VENV_DIR" ]; then
            DEEPSEEK_VENV_DIR="$HOME/.local/share/deepseek-cli/$DEEPSEEK_VENV_NAME"
        fi
        if [ -d "$DEEPSEEK_VENV_DIR" ]; then
            echo -e "  ${CY}▸${R} Removing venv: ${D}$DEEPSEEK_VENV_DIR${R}" >&2
            rm -rf "$DEEPSEEK_VENV_DIR" 2>/dev/null || true
            FOUND=true
        fi
        # Also remove the parent dir if empty
        if [ -d "$HOME/.local/share/deepseek-cli" ]; then
            rmdir "$HOME/.local/share/deepseek-cli" 2>/dev/null || true
        fi

        # Remove pip-installed packages
        for pip_cmd in pip3 pip; do
            if command -v "$pip_cmd" >/dev/null 2>&1; then
                "$pip_cmd" uninstall -y deepseek-cli deepseek-cli-agent deepseek 2>/dev/null \
                    | grep -E "Successfully|Skipping|not installed" || true
            fi
        done

        # Remove rogue deepseek dirs from site-packages
        if command -v python3 >/dev/null 2>&1; then
            for sp in $(python3 -c "import site; print('\n'.join(site.getsitepackages()))" 2>/dev/null); do
                if [ -d "$sp/deepseek" ]; then
                    echo -e "  ${CY}▸${R} Removing pip package: ${D}$sp/deepseek${R}" >&2
                    rm -rf "$sp/deepseek" 2>/dev/null || true
                    FOUND=true
                fi
            done
        fi
    fi

    # 4. Clean PATH entries
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile"; do
        if [ -f "$rc" ] && grep -q "deepseek-cli\|dscli" "$rc" 2>/dev/null; then
            echo -e "  ${CY}▸${R} Cleaning PATH in: ${D}$rc${R}" >&2
            sed '/# DeepSeek CLI/d' "$rc" > "$rc.tmp" && mv "$rc.tmp" "$rc"
            sed '/deepseek-cli/d' "$rc" > "$rc.tmp" && mv "$rc.tmp" "$rc"
            FOUND=true
        fi
    done

    echo "" >&2
    if $FOUND; then
        echo -e "  ${GR}${B}✓ DeepSeek CLI uninstalled.${R}" >&2
        if ! $CLEAN_MODE; then
            echo -e "  ${YE}${B}Note: API key kept at ${D}~/.deepseek-cli/config.yaml${R}" >&2
            echo -e "  ${YE}${B}Run 'bash install.sh --clean' for full removal (incl. venv).${R}" >&2
        fi
    else
        echo -e "  ${YE}DeepSeek CLI is not installed.${R}" >&2
    fi
    echo -e "  ${D}Run 'bash install.sh' to reinstall anytime.${R}" >&2
    exit 0
fi

# ═══════════════════════════════════════════════════════════════
#  DETECT ENVIRONMENT
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
#  CHECK PYTHON
# ═══════════════════════════════════════════════════════════════

PYTHON=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    err "Python 3 not found!"
    if $IS_TERMUX; then
        echo -e "  ${D}Run:  pkg install python${R}"
    else
        echo -e "  ${D}Install Python 3.10+ from your package manager${R}"
    fi
    exit 1
fi

PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]; }; then
    err "Python 3.8+ required (found ${PY_MAJOR}.${PY_MINOR})"
    exit 1
fi

ok "Python ${PY_MAJOR}.${PY_MINOR} ($PYTHON)"

# ═══════════════════════════════════════════════════════════════
#  VENV SETUP — the new dedicated virtualenv for dscli
# ═══════════════════════════════════════════════════════════════

step 2 $TOTAL_STEPS "Setting up dedicated virtualenv"

# Venv configuration:
#   DEEPSEEK_VENV_DIR  — full path to venv (overrides VENV_NAME)
#   DEEPSEEK_VENV_NAME — short name (default: dscli-env)
#   Default location: $HOME/.local/share/deepseek-cli/$DEEPSEEK_VENV_NAME
DEEPSEEK_VENV_DIR="${DEEPSEEK_VENV_DIR:-}"
DEEPSEEK_VENV_NAME="${DEEPSEEK_VENV_NAME:-dscli-env}"
if [ -z "$DEEPSEEK_VENV_DIR" ]; then
    DEEPSEEK_VENV_DIR="$HOME/.local/share/deepseek-cli/$DEEPSEEK_VENV_NAME"
fi

mkdir -p "$(dirname "$DEEPSEEK_VENV_DIR")" 2>/dev/null || true

info "Venv name: $DEEPSEEK_VENV_NAME"
info "Venv path: $DEEPSEEK_VENV_DIR"

# Check if venv exists
VENV_OK=false
if [ -d "$DEEPSEEK_VENV_DIR/bin" ] && [ -x "$DEEPSEEK_VENV_DIR/bin/python" ]; then
    if $REBUILD_MODE; then
        warn "Rebuild requested — removing old venv"
        rm -rf "$DEEPSEEK_VENV_DIR"
    else
        VENV_OK=true
        ok "Existing venv detected"
    fi
fi

if ! $VENV_OK; then
    info "Creating new venv..."
    if ! run_animated "Creating virtual environment..." \
        "$PYTHON" -m venv "$DEEPSEEK_VENV_DIR" 2>&1; then
        # Fallback: virtualenv
        warn "venv module failed, trying virtualenv..."
        if ! "$PYTHON" -m pip install --quiet virtualenv 2>/dev/null; then
            run_animated "Installing virtualenv..." \
                "$PYTHON" -m pip install --quiet virtualenv 2>/dev/null || true
        fi
        if ! run_animated "Creating virtualenv..." \
            "$PYTHON" -m virtualenv "$DEEPSEEK_VENV_DIR" 2>&1; then
            err "Failed to create venv"
            exit 1
        fi
    fi
    ok "Venv created at $DEEPSEEK_VENV_DIR"
fi

# Activate venv paths
VENV_PYTHON="$DEEPSEEK_VENV_DIR/bin/python"
VENV_PIP="$DEEPSEEK_VENV_DIR/bin/pip"
VENV_BIN="$DEEPSEEK_VENV_DIR/bin"

# Verify venv is functional
if ! "$VENV_PYTHON" -c "import sys; assert sys.prefix != sys.base_prefix" 2>/dev/null; then
    err "Venv at $DEEPSEEK_VENV_DIR is not functional"
    exit 1
fi
ok "Venv Python: $VENV_PYTHON"

# Upgrade pip inside venv
run_animated "Upgrading pip..." \
    "$VENV_PIP" install --quiet --upgrade pip 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
#  INSTALL DEPENDENCIES — into the venv (no conflicts!)
# ═══════════════════════════════════════════════════════════════

step 3 $TOTAL_STEPS "Installing dependencies into venv"

CORE_DEPS=(
    "httpx"
    "rich"
    "pyyaml"
    "duckduckgo-search"
)

OPT_DEPS=(
    "PyPDF2"
    "reportlab"
    "python-docx"
    "Pillow"
    "beautifulsoup4"
    "lxml"
    "mcp"
    "pydantic"
    "pytesseract"
    "selenium"
    "openpyxl"
    "python-pptx"
    "webdriver-manager"
)

# Check which core deps are missing
MISSING_CORE=()
for dep in "${CORE_DEPS[@]}"; do
    if ! "$VENV_PYTHON" -c "import $dep" 2>/dev/null; then
        # Handle import name vs package name differences
        case "$dep" in
            pyyaml) pkg_name="pyyaml" ;;
            duckduckgo-search) pkg_name="duckduckgo-search" ;;
            *) pkg_name="$dep" ;;
        esac
        MISSING_CORE+=("$pkg_name")
    fi
done

if [ ${#MISSING_CORE[@]} -gt 0 ]; then
    info "Installing core: ${MISSING_CORE[*]}"
    if ! run_animated "Installing core dependencies..." \
        "$VENV_PIP" install --quiet "${MISSING_CORE[@]}" 2>/dev/null; then
        err "Failed to install core dependencies"
        exit 1
    fi
    ok "Core dependencies installed"
else
    ok "All core dependencies already installed"
fi

# Check optional deps
MISSING_OPT=()
for dep in "${OPT_DEPS[@]}"; do
    if ! "$VENV_PYTHON" -c "import $dep" 2>/dev/null; then
        MISSING_OPT+=("$dep")
    fi
done

if [ ${#MISSING_OPT[@]} -gt 0 ]; then
    info "Installing optional: ${MISSING_OPT[*]}"
    run_animated "Installing optional dependencies..." \
        "$VENV_PIP" install --quiet "${MISSING_OPT[@]}" 2>/dev/null || \
        warn "Some optional tools failed to install (non-fatal)"
    if [ ${#MISSING_OPT[@]} -gt 0 ]; then
        ok "Optional dependencies installed"
    fi
else
    ok "All optional dependencies already installed"
fi

# ═══════════════════════════════════════════════════════════════
#  DOWNLOAD / COPY PACKAGE — auto-detect via GitHub API
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
    find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    cp -r "$SCRIPT_DIR/deepseek" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    ok "Moved from $SCRIPT_DIR"
fi

if ! $LOCAL_SOURCE; then
    GITHUB_RAW="${GITHUB_RAW_URL:-https://raw.githubusercontent.com/XbibzOfficial777/deepseek-cli/main}"
    GITHUB_API="${GITHUB_API_URL:-https://api.github.com/repos/XbibzOfficial777/deepseek-cli/contents/deepseek?ref=main}"
    info "Downloading from GitHub: $GITHUB_RAW"

    TEMP_DIR=$(mktemp -d)
    mkdir -p "$TEMP_DIR/deepseek"

    # ── Auto-detect files via GitHub API (preferred) ──
    FILES=()
    AUTO_DETECTED=false
    if command -v curl >/dev/null 2>&1; then
        API_RESP=$(curl -fsSL --max-time 10 \
            -H "Accept: application/vnd.github.v3+json" \
            -H "User-Agent: deepseek-cli-installer" \
            "$GITHUB_API" 2>/dev/null || true)
        if [ -n "$API_RESP" ] && echo "$API_RESP" | grep -q '"name"'; then
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
            else
                PY_NAMES=$(printf '%s' "$API_RESP" | grep -oE '"name"[[:space:]]*:[[:space:]]*"[^"]+\.py"' | sed -n 's/"name"[[:space:]]*:[[:space:]]*"\([^"]*\)"/deepseek\/\1/p')
            fi
            if [ -n "$PY_NAMES" ]; then
                while IFS= read -r f; do
                    [ -n "$f" ] && FILES+=("$f")
                done <<< "$PY_NAMES"
                if [ "${#FILES[@]}" -ge 10 ]; then
                    AUTO_DETECTED=true
                    FILES+=("requirements.txt")
                fi
            fi
        fi
    fi

    if ! $AUTO_DETECTED; then
        warn "GitHub API auto-detect failed — using hardcoded file list"
        FILES=(
            "deepseek/__init__.py" "deepseek/__main__.py" "deepseek/agent.py"
            "deepseek/auth.py" "deepseek/config.py" "deepseek/connectors.py"
            "deepseek/doc_tools.py" "deepseek/llm.py" "deepseek/mcp_client.py"
            "deepseek/mcp_tools.py" "deepseek/memory.py" "deepseek/multi_agent.py"
            "deepseek/planner.py" "deepseek/providers.py" "deepseek/repl.py"
            "deepseek/selenium_browser.py" "deepseek/toolkit.py" "deepseek/tools.py"
            "deepseek/ui.py" "deepseek/webcontrol.py" "requirements.txt"
        )
    fi

    info "File list: ${#FILES[@]} files$($AUTO_DETECTED && echo ' (auto)' || echo ' (fallback)')"

    # Download with animation
    DOWNLOADED=0
    FAILED_FILES=()
    for f in "${FILES[@]}"; do
        url="${GITHUB_RAW}/${f}"
        if curl -fsSL "$url" -o "$TEMP_DIR/$f" 2>/dev/null; then
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            FAILED_FILES+=("$f")
        fi
    done

    if [ $DOWNLOADED -lt 10 ]; then
        err "Download failed ($DOWNLOADED/$(( ${#FILES[@]} )) files)"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    ok "Downloaded $DOWNLOADED files"

    # CRITICAL: wipe destination first so stale files don't linger
    rm -rf "$INSTALL_DIR/deepseek" 2>/dev/null || true
    find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    mkdir -p "$INSTALL_DIR/deepseek"
    cp -r "$TEMP_DIR/deepseek/." "$INSTALL_DIR/deepseek/"
    cp "$TEMP_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    rm -rf "$TEMP_DIR"

    if [ ${#FAILED_FILES[@]} -gt 0 ]; then
        warn "Some files failed: ${FAILED_FILES[*]}"
    fi
fi

# Clear any cached bytecode
find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Verify (using venv python so deps are available)
VERIFY_OUT=$(run_animated "Verifying package..." \
    "$VENV_PYTHON" -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
try:
    from deepseek.providers import create_provider
    from deepseek.agent import Agent
    from deepseek.toolkit import ToolRegistry
    t = ToolRegistry()
    print(f'{len(t.tools)} tools')
except Exception as e:
    print(f'Error: {e}')
" 2>&1) || VERIFY_OUT=""

if echo "$VERIFY_OUT" | grep -q "Error"; then
    warn "Package verification: $VERIFY_OUT"
else
    ok "Package verified: $VERIFY_OUT"
fi

# Sanity check: ensure no leftover ACTIVE 'Signed in as' print calls
if grep -rnE "^\s*[^#]*console\.print.*Signed in as\b|^\s*[^#]*print\s*\(.*Signed in as" \
    "$INSTALL_DIR/deepseek" 2>/dev/null | grep -v "^\s*#"; then
    warn "Stale 'Signed in as' print call detected."
    warn "Run: bash install.sh --clean && bash install.sh"
fi

# ═══════════════════════════════════════════════════════════════
#  CREATE 'dscli' WRAPPER — uses venv python directly
# ═══════════════════════════════════════════════════════════════

step 5 $TOTAL_STEPS "Creating 'dscli' command"

WRAPPER="$BIN_DIR/dscli"

# Wrapper uses the venv's Python directly — no more conflicts with
# active virtualenvs, system Python, or PATH ordering. The venv is
# ALWAYS at $DEEPSEEK_VENV_DIR so this is deterministic.
cat > "$WRAPPER" << WRAPPER_EOF
#!/usr/bin/env bash
# DeepSeek CLI v7.7 — Launcher
# Generated by install.sh — uses the dedicated venv at:
#   \${DEEPSEEK_VENV_DIR:-__VENV_DIR_PLACEHOLDER__}
# Override at runtime with: DEEPSEEK_VENV_DIR=/path/to/venv dscli ...

set -euo pipefail

# Defaults baked in at install time. User can override via env var.
INSTALL_DIR="\${INSTALL_DIR_OVERRIDE:-__INSTALL_DIR_PLACEHOLDER__}"
VENV_DIR="\${DEEPSEEK_VENV_DIR:-__VENV_DIR_PLACEHOLDER__}"
VENV_PYTHON="\$VENV_DIR/bin/python"

if [ ! -f "\$INSTALL_DIR/deepseek/__init__.py" ]; then
    echo -e "\033[31m  ✗ DeepSeek CLI package not found at \$INSTALL_DIR\033[0m"
    echo -e "\033[2m  Run: bash install.sh --clean && bash install.sh\033[0m"
    exit 1
fi

if [ ! -x "\$VENV_PYTHON" ]; then
    echo -e "\033[31m  ✗ Venv python not found at \$VENV_PYTHON\033[0m"
    echo -e "\033[2m  Run: bash install.sh --clean && bash install.sh\033[0m"
    echo -e "\033[2m  Or set: DEEPSEEK_PYTHON=/usr/bin/python3 dscli\033[0m"
    exit 1
fi

# Allow override via DEEPSEEK_PYTHON env var (escape hatch)
PYTHON="\${DEEPSEEK_PYTHON:-\$VENV_PYTHON}"

export DEEPSEEK_ORIGINAL_CWD="\$PWD"
export PYTHONPATH="\$INSTALL_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
cd "\$INSTALL_DIR"
exec "\$PYTHON" -m deepseek "\$@"
WRAPPER_EOF

# Replace placeholders
sed "s|__INSTALL_DIR_PLACEHOLDER__|$INSTALL_DIR|g; s|__VENV_DIR_PLACEHOLDER__|$DEEPSEEK_VENV_DIR|g" \
    "$WRAPPER" > "$WRAPPER.tmp" && mv "$WRAPPER.tmp" "$WRAPPER"
chmod +x "$WRAPPER"

ok "Wrapper created at $WRAPPER"

# PATH setup
PATH_NEED_FIX=false
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) PATH_NEED_FIX=true ;;
esac

if $PATH_NEED_FIX; then
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile"; do
        if [ -f "$rc" ]; then
            if ! grep -q "deepseek-cli" "$rc" 2>/dev/null; then
                echo "" >> "$rc"
                echo "# DeepSeek CLI - Auto-added by installer" >> "$rc"
                echo "export PATH=\"\$BIN_DIR:\$PATH\"" >> "$rc"
                info "Added to PATH in $rc"
            fi
        fi
    done
    # Ensure at least one rc file exists
    if [ ! -f "$HOME/.bashrc" ] && [ ! -f "$HOME/.zshrc" ] && [ ! -f "$HOME/.bash_profile" ]; then
        echo "# DeepSeek CLI - Auto-added by installer" > "$HOME/.bashrc"
        echo "export PATH=\"\$BIN_DIR:\$PATH\"" >> "$HOME/.bashrc"
    fi
    export PATH="$BIN_DIR:$PATH"
    info "PATH updated for current session"
fi

# ═══════════════════════════════════════════════════════════════
#  CLEANUP
# ═══════════════════════════════════════════════════════════════

step 6 $TOTAL_STEPS "Cleaning up"

# Clear __pycache__ again
find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
ok "Cache cleared"

# Write env info for the user
cat > "$INSTALL_DIR/.env-info" <<EOF
# Generated by install.sh on $(date)
DEEPSEEK_VENV_DIR=$DEEPSEEK_VENV_DIR
DEEPSEEK_INSTALL_DIR=$INSTALL_DIR
DEEPSEEK_WRAPPER=$WRAPPER
DEEPSEEK_PYTHON=$VENV_PYTHON
EOF
ok "Env info saved to $INSTALL_DIR/.env-info"

# ═══════════════════════════════════════════════════════════════
#  CONFIG SETUP
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
" 2>/dev/null && ok "Config saved" || true
fi

# ═══════════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════════

cat << 'DONE' >&2

  ╔══════════════════════════════════════════╗
  ║         Install Complete!                ║
  ╚══════════════════════════════════════════╝

DONE

echo -e "  ${D}Venv:      ${CY}${B}$DEEPSEEK_VENV_DIR${R}" >&2
echo -e "  ${D}Package:   ${CY}${B}$INSTALL_DIR/deepseek/${R}" >&2
echo -e "  ${D}Wrapper:   ${CY}${B}$WRAPPER${R}" >&2
echo "" >&2
echo -e "  ${GR}${B}Run:${R}  ${CY}${B}dscli${R} ${D}to start DeepSeek CLI${R}" >&2
echo "" >&2

echo -e "  ${D}Customize venv (optional):${R}" >&2
echo -e "  ${D}  DEEPSEEK_VENV_NAME=myenv bash install.sh${R}" >&2
echo -e "  ${D}  DEEPSEEK_VENV_DIR=/custom/path bash install.sh${R}" >&2
echo "" >&2

# Auto-launch if TTY
if [ $# -eq 0 ] && tty -s 2>/dev/null; then
    echo -en "  ${CY}${B}Launch DeepSeek CLI now? [Y/n]${R} " >&2
    read -r ANSWER </dev/tty 2>/dev/null || ANSWER="y"
    case "$ANSWER" in
        n*|N*) ;;
        *) exec dscli ;;
    esac
fi
