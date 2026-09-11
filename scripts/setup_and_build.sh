#!/usr/bin/env bash
# OpenWorker dev build setup + build script.
#
# Checks all prerequisites, offers to install missing ones, then builds the DMG.
# Works on both Apple Silicon (M1/M2/M3/M4) and Intel Macs.
#
# Usage:
#   bash setup_and_build.sh          # interactive: check + install + build
#   bash setup_and_build.sh --check  # only check, don't install or build
#   bash setup_and_build.sh --build  # skip check, just build (assumes deps installed)
#
# Prerequisites this script manages:
#   - Homebrew (package manager)
#   - Rust (rustup) — for Tauri native build
#   - Node.js + npm — for GUI build
#   - Python 3.10+ — for the sidecar server
#   - cmake — required by whisper-rs-sys (Rust dependency)
#   - Python venv deps: pip install -e '.[dev,messaging,browser,bedrock]' pyinstaller typer

set -euo pipefail

# -- Colors -------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# -- Root directory -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GUI_DIR="$PROJECT_ROOT/surfaces/gui"
VENV_DIR="$PROJECT_ROOT/.venv"

# -- Parse args ---------------------------------------------------------------
MODE="full"  # full | check | build
for arg in "$@"; do
  case "$arg" in
    --check) MODE="check" ;;
    --build) MODE="build" ;;
    --help|-h)
      echo "Usage: bash setup_and_build.sh [--check|--build]"
      echo "  --check  Only check prerequisites, don't install or build"
      echo "  --build  Skip check, just build (assumes deps installed)"
      exit 0
      ;;
  esac
done

# -- Helper: yes/no prompt ----------------------------------------------------
ask_yes_no() {
  local prompt="$1"
  local default="${2:-y}"
  if [ "$default" = "y" ]; then
    echo -n "$prompt [Y/n] "
  else
    echo -n "$prompt [y/N] "
  fi
  read -r answer
  answer="${answer:-$default}"
  case "$answer" in
    [yY]|[yY][eE][sS]) return 0 ;;
    *) return 1 ;;
  esac
}

# -- Check functions ----------------------------------------------------------
check_brew() {
  if command -v brew &>/dev/null; then
    log_ok "Homebrew: $(brew --version | head -1)"
    return 0
  fi
  log_warn "Homebrew is not installed"
  return 1
}

check_rust() {
  if command -v rustc &>/dev/null; then
    log_ok "Rust: $(rustc --version)"
    return 0
  fi
  # Also check ~/.cargo/bin/rustc
  if [ -x "$HOME/.cargo/bin/rustc" ]; then
    log_ok "Rust: $($HOME/.cargo/bin/rustc --version) (in ~/.cargo/bin)"
    return 0
  fi
  log_warn "Rust is not installed"
  return 1
}

check_node() {
  if command -v node &>/dev/null; then
    log_ok "Node.js: $(node --version)"
    if command -v npm &>/dev/null; then
      log_ok "npm: $(npm --version)"
    else
      log_warn "npm not found (Node.js installed but npm missing)"
      return 1
    fi
    return 0
  fi
  log_warn "Node.js is not installed"
  return 1
}

check_python() {
  if command -v python3 &>/dev/null; then
    local ver
    ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    local major minor
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
      log_ok "Python: $ver"
      return 0
    fi
    log_warn "Python $ver found but 3.10+ required"
    return 1
  fi
  log_warn "Python 3 is not installed"
  return 1
}

check_cmake() {
  if command -v cmake &>/dev/null; then
    log_ok "CMake: $(cmake --version | head -1)"
    return 0
  fi
  log_warn "CMake is not installed (required by whisper-rs-sys)"
  return 1
}

check_venv() {
  if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
    log_ok "Python venv: $VENV_DIR"
    # Check key packages
    local missing=()
    for pkg in pyinstaller typer; do
      if ! "$VENV_DIR/bin/python" -c "import $pkg" &>/dev/null; then
        missing+=("$pkg")
      fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
      log_warn "venv missing packages: ${missing[*]}"
      return 1
    fi
    # Check the coworker package is installed
    if ! "$VENV_DIR/bin/python" -c "import coworker" &>/dev/null; then
      log_warn "coworker package not installed in venv"
      return 1
    fi
    return 0
  fi
  log_warn "Python venv not found at $VENV_DIR"
  return 1
}

check_gui_deps() {
  if [ -d "$GUI_DIR/node_modules" ]; then
    log_ok "GUI node_modules: present"
    return 0
  fi
  log_warn "GUI node_modules not found (npm install needed)"
  return 1
}

# -- Install functions --------------------------------------------------------
install_brew() {
  log_info "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add to PATH for Apple Silicon
  if [ -f /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
  log_ok "Homebrew installed"
}

install_rust() {
  log_info "Installing Rust via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # Source the env
  if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
  fi
  log_ok "Rust installed: $(rustc --version)"
}

install_node() {
  log_info "Installing Node.js via Homebrew..."
  brew install node
  log_ok "Node.js installed: $(node --version)"
}

install_cmake() {
  log_info "Installing CMake via Homebrew..."
  brew install cmake
  log_ok "CMake installed: $(cmake --version | head -1)"
}

setup_venv() {
  log_info "Setting up Python venv..."
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --quiet --upgrade pip
  log_info "Installing project dependencies (this may take a few minutes)..."
  "$VENV_DIR/bin/pip" install --quiet -e "$PROJECT_ROOT[dev,messaging,browser,bedrock]"
  log_info "Installing build-only deps (pyinstaller, typer)..."
  "$VENV_DIR/bin/pip" install --quiet pyinstaller typer
  # Verify
  "$VENV_DIR/bin/python" -c "import aisuite, coworker" || {
    log_error "Failed to import aisuite or coworker after install"
    return 1
  }
  log_ok "Python venv ready"
}

setup_gui() {
  log_info "Installing GUI dependencies..."
  (cd "$GUI_DIR" && npm install)
  log_ok "GUI dependencies installed"
}

# -- Build function -----------------------------------------------------------
build_dmg() {
  log_info "Building OpenWorker DMG..."
  
  # Ensure cargo env is loaded
  if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
  fi
  
  # Run the build script
  (cd "$PROJECT_ROOT" && bash packaging/build_dmg.sh)
  
  local dmg_path="$GUI_DIR/src-tauri/target/release/bundle/dmg/OpenWorker_0.2.1_aarch64.dmg"
  if [ -f "$dmg_path" ]; then
    log_ok "DMG built successfully!"
    log_info "Location: $dmg_path"
    log_info ""
    log_info "To install:"
    log_info "  1. Open the DMG: open \"$dmg_path\""
    log_info "  2. Drag OpenWorker.app to /Applications"
    log_info "  3. Since this is an unsigned dev build, right-click the app"
    log_info "     in Finder and select 'Open' to bypass Gatekeeper"
    return 0
  else
    log_error "DMG not found after build"
    return 1
  fi
}

# -- Main ---------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  OpenWorker Dev Build Setup"
echo "=========================================="
echo ""

if [ "$MODE" != "build" ]; then
  log_info "Checking prerequisites..."
  echo ""
  
  missing=()
  
  check_brew || missing+=("brew")
  check_rust || missing+=("rust")
  check_node || missing+=("node")
  check_python || missing+=("python")
  check_cmake || missing+=("cmake")
  check_venv || missing+=("venv")
  check_gui_deps || missing+=("gui_deps")
  
  echo ""
  
  if [ ${#missing[@]} -eq 0 ]; then
    log_ok "All prerequisites are installed!"
  else
    log_warn "Missing: ${missing[*]}"
    echo ""
    
    if [ "$MODE" = "check" ]; then
      log_info "Run without --check to install missing dependencies"
      exit 1
    fi
    
    # Offer to install
    for dep in "${missing[@]}"; do
      case "$dep" in
        brew)
          if ask_yes_no "Install Homebrew?"; then
            install_brew
          fi
          ;;
        rust)
          if ask_yes_no "Install Rust?"; then
            install_rust
          fi
          ;;
        node)
          if ask_yes_no "Install Node.js?"; then
            install_node
          fi
          ;;
        cmake)
          if ask_yes_no "Install CMake?"; then
            install_cmake
          fi
          ;;
        venv)
          if ask_yes_no "Set up Python venv and install dependencies?"; then
            setup_venv
          fi
          ;;
        gui_deps)
          if ask_yes_no "Install GUI dependencies (npm install)?"; then
            setup_gui
          fi
          ;;
      esac
    done
    
    echo ""
    log_info "Re-checking prerequisites..."
    echo ""
    
    still_missing=()
    check_brew || still_missing+=("brew")
    check_rust || still_missing+=("rust")
    check_node || still_missing+=("node")
    check_python || still_missing+=("python")
    check_cmake || still_missing+=("cmake")
    check_venv || still_missing+=("venv")
    check_gui_deps || still_missing+=("gui_deps")
    
    if [ ${#still_missing[@]} -gt 0 ]; then
      log_error "Still missing: ${still_missing[*]}"
      log_error "Please install these manually and re-run the script"
      exit 1
    fi
    
    log_ok "All prerequisites are now installed!"
  fi
fi

if [ "$MODE" != "check" ]; then
  echo ""
  echo "=========================================="
  echo "  Building OpenWorker DMG"
  echo "=========================================="
  echo ""
  
  build_dmg
fi

echo ""
log_ok "Done!"
echo ""
