#!/bin/bash
set -e

# ── Config ────────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/tnkrai/Open_Duck_Mini_Runtime.git"
REPO_BRANCH="v2"
INSTALL_DIR="$HOME/Open_Duck_Mini_Runtime"
CONFIG_FILE="$HOME/duck_config.json"
SERVICE_NAME="tnkr-robot"
SERVER_PORT=8000
TOTAL_STEPS=6

# ── Clean flag ────────────────────────────────────────────────────────────────

if [ "$1" = "--clean" ]; then
    echo ""
    echo -e "\033[1;33m  Cleaning previous installation...\033[0m"
    echo ""

    # Stop and remove systemd service
    sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    sudo rm -f /etc/systemd/system/tnkr-robot.service
    sudo systemctl daemon-reload 2>/dev/null || true
    echo -e "  \033[0;32m✓\033[0m Service removed"

    # Remove runtime directory (venv + repo)
    if [ -d "$INSTALL_DIR" ]; then
        rm -rf "$INSTALL_DIR"
        echo -e "  \033[0;32m✓\033[0m Runtime directory removed"
    fi

    # Remove config
    if [ -f "$CONFIG_FILE" ]; then
        rm -f "$CONFIG_FILE"
        echo -e "  \033[0;32m✓\033[0m Config file removed"
    fi

    echo -e "  \033[0;32m✓\033[0m Clean complete — running fresh install"
    echo ""
fi

# ── Colors & Symbols ─────────────────────────────────────────────────────────

BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'

CHECK="${GREEN}✓${RESET}"
CROSS="${RED}✗${RESET}"
ARROW="${CYAN}→${RESET}"
DOT="${DIM}·${RESET}"

# ── Spinner ───────────────────────────────────────────────────────────────────

SPINNER_CHARS='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
SPINNER_PID=""

start_spinner() {
    local msg="$1"
    (
        i=0
        while true; do
            printf "\r  ${CYAN}${SPINNER_CHARS:$i:1}${RESET} ${DIM}%s${RESET}" "$msg"
            i=$(( (i + 1) % ${#SPINNER_CHARS} ))
            sleep 0.08
        done
    ) &
    SPINNER_PID=$!
}

stop_spinner() {
    local success="${1:-true}"
    if [ -n "$SPINNER_PID" ]; then
        kill "$SPINNER_PID" 2>/dev/null
        wait "$SPINNER_PID" 2>/dev/null || true
        SPINNER_PID=""
    fi
    if [ "$success" = "true" ]; then
        printf "\r  ${CHECK} %-60s\n" "$2"
    else
        printf "\r  ${CROSS} %-60s\n" "$2"
    fi
}

# ── Step header ───────────────────────────────────────────────────────────────

step() {
    local num="$1"
    local title="$2"
    echo ""
    printf "  ${WHITE}[%d/%d]${RESET} ${BOLD}%s${RESET}\n" "$num" "$TOTAL_STEPS" "$title"
}

# ── Error handler ─────────────────────────────────────────────────────────────

on_error() {
    stop_spinner false "Failed"
    echo ""
    echo -e "  ${RED}Setup failed. Check the error above and try again.${RESET}"
    echo -e "  ${DIM}If the issue persists, visit: https://docs.tnkr.com/setup${RESET}"
    echo ""
    exit 1
}
trap on_error ERR

# ══════════════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════════════

clear
echo ""
echo -e "${CYAN}"
cat << 'LOGO'

    ████████╗███╗   ██╗██╗  ██╗██████╗
    ╚══██╔══╝████╗  ██║██║ ██╔╝██╔══██╗
       ██║   ██╔██╗ ██║█████╔╝ ██████╔╝
       ██║   ██║╚██╗██║██╔═██╗ ██╔══██╗
       ██║   ██║ ╚████║██║  ██╗██║  ██║
       ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝

LOGO
echo -e "${RESET}"
echo -e "  ${BOLD}Open Duck Mini ${DIM}— Robot Setup${RESET}"
echo -e "  ${DIM}────────────────────────────────────────${RESET}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: System dependencies
# ══════════════════════════════════════════════════════════════════════════════

step 1 "System dependencies"

start_spinner "Updating package lists..."
sudo apt-get update -qq > /dev/null 2>&1
stop_spinner true "Package lists updated"

start_spinner "Installing git, python3, venv..."
sudo apt-get install -y -qq git python3-pip python3-venv > /dev/null 2>&1
stop_spinner true "Dependencies installed"

# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: Enable I2C
# ══════════════════════════════════════════════════════════════════════════════

step 2 "Hardware interfaces"

start_spinner "Enabling I2C for IMU sensor... 📡"
sudo raspi-config nonint do_i2c 0
stop_spinner true "I2C enabled"

# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: USB serial latency
# ══════════════════════════════════════════════════════════════════════════════

step 3 "Motor communication"

start_spinner "Setting USB serial latency rule... 📡"
echo 'SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"' | \
  sudo tee /etc/udev/rules.d/99-ftdi-latency.rules > /dev/null
sudo udevadm control --reload-rules
stop_spinner true "USB latency optimized for motor control"

# ══════════════════════════════════════════════════════════════════════════════
#  Step 4: Clone and install runtime
# ══════════════════════════════════════════════════════════════════════════════

step 4 "Runtime"

if [ -d "$INSTALL_DIR" ]; then
    start_spinner "Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull origin "$REPO_BRANCH" > /dev/null 2>&1
    stop_spinner true "Runtime updated 🔄"
else
    start_spinner "Cloning Runtime..."
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR" > /dev/null 2>&1
    cd "$INSTALL_DIR"
    stop_spinner true "Runtime cloned 📥"
fi

if [ ! -d "$INSTALL_DIR/.venv" ]; then
    start_spinner "Creating Python environment..."
    python3 -m venv "$INSTALL_DIR/.venv"
    stop_spinner true "Python environment created"
fi

start_spinner "Installing packages 📦 (this may take a few minutes)..."
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip > /dev/null 2>&1
"$INSTALL_DIR/.venv/bin/pip" install -e . > /dev/null 2>&1
stop_spinner true "All packages installed"

# ══════════════════════════════════════════════════════════════════════════════
#  Step 5: Default config
# ══════════════════════════════════════════════════════════════════════════════

step 5 "Robot configuration"

if [ ! -f "$CONFIG_FILE" ]; then
    cp "$INSTALL_DIR/example_config.json" "$CONFIG_FILE"
    echo -e "  ${CHECK} Default config created at ${DIM}$CONFIG_FILE${RESET}"
else
    echo -e "  ${DOT} Config already exists at ${DIM}$CONFIG_FILE${RESET} ${DIM}(keeping existing)${RESET}"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Step 6: Systemd service
# ══════════════════════════════════════════════════════════════════════════════

step 6 "TNKR server"

start_spinner "Installing system service..."
sed -e "s|TNKR_USER|$(whoami)|g" \
    -e "s|TNKR_INSTALL_DIR|$INSTALL_DIR|g" \
    "$INSTALL_DIR/tnkr-robot.service.template" | \
    sudo tee /etc/systemd/system/tnkr-robot.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME" > /dev/null 2>&1
sudo systemctl restart "$SERVICE_NAME"
stop_spinner true "Server installed and running"

# ══════════════════════════════════════════════════════════════════════════════
#  Done
# ══════════════════════════════════════════════════════════════════════════════

SERVER_URL="http://$(hostname).local:$SERVER_PORT"

echo ""
echo -e "  ${DIM}────────────────────────────────────────${RESET}"
echo ""
echo -e "  ${GREEN}${BOLD}Setup complete!${RESET}"
echo ""
echo -e "  ${ARROW} Server running at:"
echo -e "    ${WHITE}${SERVER_URL}${RESET}"
echo ""
echo -e "  ${ARROW} Next steps:"
echo -e "    ${DIM}1.${RESET} Open the URL above in your browser 🌐"
echo -e "    ${DIM}2.${RESET} Check motors 🔌"
echo -e "    ${DIM}3.${RESET} Calibrate joints 🔧"
echo -e "    ${DIM}4.${RESET} Configure features 🔧"
echo -e "    ${DIM}5.${RESET} Start walking! 🚶"
echo ""
