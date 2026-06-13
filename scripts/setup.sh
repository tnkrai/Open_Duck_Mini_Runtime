#!/bin/bash
# shellcheck disable=SC2059  # We intentionally use color variables in printf format strings
# ──────────────────────────────────────────────────────────────────────────────
#  TNKR Open Duck Mini — Raspberry Pi Setup
#
#  Designed for Pi Zero 2W (512 MB RAM, aarch64).
#  Follows patterns from Pi-hole, Klipper, and rustup installers:
#    - State-file resumability (safe to re-run after interruption)
#    - OOM protection (swap expansion, single-threaded builds)
#    - Split pip phases (core, rustypot, optional)
#    - Proper signal traps (no orphan spinners, swap always restored)
#
#  Usage:
#    bash setup.sh            # Install or resume
#    bash setup.sh --clean    # Wipe everything and reinstall
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/tnkrai/Open_Duck_Mini_Runtime.git"
REPO_BRANCH="v2"
INSTALL_DIR="$HOME/Open_Duck_Mini_Runtime"
CONFIG_FILE="$HOME/duck_config.json"
SERVICE_NAME="tnkr-robot"
SERVER_PORT=8000
STATE_DIR="$HOME/.tnkr-setup"
LOG_FILE="$STATE_DIR/setup.log"
TOTAL_STEPS=12
MIN_DISK_MB=2048
SWAP_SIZE_MB=2048

# Anonymous usage telemetry (see telemetry_init below for the user notice).
# Key is write-only (can send events, cannot read data). Key/host and the
# property names must match mini_bdx_runtime/mini_bdx_runtime/telemetry.py.
TELEMETRY_FILE="$HOME/.tnkr-telemetry.json"
POSTHOG_KEY="phc_FarYZWwIbyZFV2iUKyl8WyRRdFFuw2MH3NZat4zPmEK"
POSTHOG_HOST="https://us.i.posthog.com"
TELEMETRY_PROMPT_TIMEOUT_S=15
TELEMETRY_CURL_MAX_TIME_S=3

# ── Colors & Symbols ─────────────────────────────────────────────────────────

BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'

CHECK="${GREEN}✓${RESET}"
CROSS="${RED}✗${RESET}"
ARROW="${CYAN}→${RESET}"
DOT="${DIM}·${RESET}"

# ── State ─────────────────────────────────────────────────────────────────────

SPINNER_PID=""
ORIGINAL_SWAP_SIZE=""
SWAP_EXPANDED=false
TEST_MODE=false
CLEAN_INSTALL=false
# Durations use bash's $SECONDS (seconds since shell start), not wall-clock:
# the Pi has no RTC, so NTP jumping the clock mid-install would otherwise
# produce negative or absurd durations.

# Telemetry state (set by telemetry_init; referenced by the EXIT trap, so
# initialized here for `set -u` safety)
TELEMETRY_ENABLED="false"
DEVICE_ID=""
CURRENT_STEP_NUM=""
CURRENT_STEP_ID=""
STEP_START=""

# ── Helpers ───────────────────────────────────────────────────────────────────

die() {
    printf "\n  ${CROSS} %s\n\n" "$1" >&2
    exit 1
}

info() {
    printf "  ${CHECK} %s\n" "$1"
}

warn() {
    printf "  ${YELLOW}!${RESET} %s\n" "$1"
}

step_done() {
    [ -f "$STATE_DIR/step_${1}.done" ]
}

mark_done() {
    touch "$STATE_DIR/step_${1}.done"
}

# ── Telemetry ─────────────────────────────────────────────────────────────────
#
# Anonymous, opt-out usage telemetry. Sent via curl because steps 1-7 run
# before the Python venv exists. Never blocks setup (3s cap, always || true),
# never sends: motion data, names, hostnames, or location (GeoIP disabled).

telemetry_init() {
    if [ -f "$TELEMETRY_FILE" ]; then
        # Existing device: reuse id + respect saved preference. Never re-prompt.
        DEVICE_ID=$(sed -n 's/.*"device_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$TELEMETRY_FILE" | head -1)
        if grep -q '"enabled"[[:space:]]*:[[:space:]]*false' "$TELEMETRY_FILE"; then
            TELEMETRY_ENABLED="false"
        else
            TELEMETRY_ENABLED="true"
        fi
    else
        DEVICE_ID=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "")
        TELEMETRY_ENABLED="true"
        echo ""
        printf "  ${DIM}────────────────────────────────────────${RESET}\n"
        printf "  ${BOLD}Anonymous usage telemetry${RESET}\n"
        printf "  Helps us fix setup and robot failures for everyone.\n"
        printf "  ${DIM}We collect:${RESET} setup step outcomes, API errors, hardware model.\n"
        printf "  ${DIM}Never:${RESET} motion/joint data, names, precise location.\n"
        printf "  ${DIM}Disable any time:${RESET} TNKR_TELEMETRY=0, or \"enabled\": false in %s\n" "$TELEMETRY_FILE"
        printf "  ${DIM}────────────────────────────────────────${RESET}\n"
        if [ "$TTY_OUT" = "/dev/tty" ]; then
            local ans=""
            read -r -t "$TELEMETRY_PROMPT_TIMEOUT_S" -p "  Press Enter to continue, or type 'n' to opt out: " ans < /dev/tty || true
            echo ""
            case "$ans" in
                n|N|no|NO) TELEMETRY_ENABLED="false"; info "Telemetry disabled" ;;
            esac
        fi
    fi

    # Env var is a hard override in both directions. Token set must match
    # telemetry.py: set-but-empty ("") also disables.
    if [ -n "${TNKR_TELEMETRY+x}" ]; then
        case "${TNKR_TELEMETRY,,}" in
            ""|0|false|off) TELEMETRY_ENABLED="false" ;;
            *)              TELEMETRY_ENABLED="true" ;;
        esac
    fi

    # Persist only when the file doesn't exist yet — never clobber user edits
    if [ ! -f "$TELEMETRY_FILE" ] && [ -n "$DEVICE_ID" ]; then
        printf '{\n  "device_id": "%s",\n  "enabled": %s,\n  "notice_version": 1,\n  "created_at": "%s"\n}\n' \
            "$DEVICE_ID" "$TELEMETRY_ENABLED" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            > "$TELEMETRY_FILE" 2>/dev/null || true
    fi

    # No uuid source (non-Linux?) — stay silent rather than mislabel devices
    [ -z "$DEVICE_ID" ] && TELEMETRY_ENABLED="false"

    if [ "$TELEMETRY_ENABLED" = "true" ]; then
        printf "  ${DOT} Telemetry: enabled ${DIM}(device %.8s…, disable: TNKR_TELEMETRY=0)${RESET}\n" "$DEVICE_ID"
    fi
}

# ph_capture <event_name> [<json_props_fragment>] [fg]
# Fragment example: "step_id":"08_pip_core","step_num":8
# Runs in the background by default so a blackholed network never slows the
# install; pass "fg" for events that must be delivered before the script
# exits (the EXIT trap's failure event and setup_completed).
ph_capture() {
    [ "$TELEMETRY_ENABLED" = "true" ] || return 0
    [ -n "$DEVICE_ID" ] || return 0
    local event="$1" props="${2:-}" mode="${3:-bg}"
    local payload="{\"api_key\":\"$POSTHOG_KEY\",\"event\":\"$event\",\"distinct_id\":\"$DEVICE_ID\",\"properties\":{\"source\":\"openduck-runtime\",\"setup_script\":true,\"\$geoip_disable\":true${props:+,$props}}}"
    if [ "$mode" = "fg" ]; then
        curl -s --max-time "$TELEMETRY_CURL_MAX_TIME_S" -o /dev/null -X POST "$POSTHOG_HOST/i/v0/e/" \
            -H 'Content-Type: application/json' -d "$payload" 2>/dev/null || true
    else
        ( curl -s --max-time "$TELEMETRY_CURL_MAX_TIME_S" -o /dev/null -X POST "$POSTHOG_HOST/i/v0/e/" \
            -H 'Content-Type: application/json' -d "$payload" 2>/dev/null || true ) &
    fi
}

# Hardware props as a JSON fragment — property names are a contract with
# mini_bdx_runtime/telemetry.py (pi_model, arch, ram_mb, os_release).
# JSON-escaped via python3 when available (device-tree model / PRETTY_NAME
# strings with quotes would otherwise silently break the whole payload).
telemetry_hw_props() {
    local arch model ram_mb os_release
    arch=$(uname -m 2>/dev/null || echo "")
    model=""
    [ -f /proc/device-tree/model ] && model=$(tr -d '\0' < /proc/device-tree/model)
    ram_mb=$(( $(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0) / 1024 ))
    os_release=$(sed -n 's/^PRETTY_NAME="\(.*\)"/\1/p' /etc/os-release 2>/dev/null | head -1)
    if command -v python3 > /dev/null 2>&1; then
        python3 -c '
import json, sys
arch, model, ram, osr = sys.argv[1:5]
print(",".join(f"{json.dumps(k)}:{json.dumps(v)}" for k, v in
      [("arch", arch), ("pi_model", model), ("ram_mb", int(ram)), ("os_release", osr)]))
' "$arch" "$model" "$ram_mb" "$os_release" 2>/dev/null && return 0
    fi
    printf '"arch":"%s","pi_model":"%s","ram_mb":%s,"os_release":"%s"' \
        "$arch" "$model" "$ram_mb" "$os_release"
}

# ── Spinner ───────────────────────────────────────────────────────────────────

SPINNER_CHARS='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

# Detect usable TTY once at startup (actual write test, not just -w check).
# Spinner animation writes here; result lines go to stdout (through tee → log).
TTY_OUT="/dev/null"
if (printf "" > /dev/tty) 2>/dev/null; then
    TTY_OUT="/dev/tty"
fi

start_spinner() {
    local msg="$1"
    if [ "$TTY_OUT" = "/dev/null" ]; then
        # No TTY — just print a static line (no animation)
        printf "  ${DIM}%s${RESET}\n" "$msg"
        return
    fi
    (
        i=0
        while true; do
            printf "\r  ${CYAN}${SPINNER_CHARS:$i:1}${RESET} ${DIM}%s${RESET}" "$msg" > "$TTY_OUT"
            i=$(( (i + 1) % ${#SPINNER_CHARS} ))
            sleep 0.08
        done
    ) &
    SPINNER_PID=$!
}

stop_spinner() {
    local success="${1:-true}"
    local msg="${2:-}"
    if [ -n "$SPINNER_PID" ] && kill -0 "$SPINNER_PID" 2>/dev/null; then
        kill "$SPINNER_PID" 2>/dev/null
        wait "$SPINNER_PID" 2>/dev/null || true
    fi
    SPINNER_PID=""
    if [ -n "$msg" ]; then
        # Clear the spinner line on the real terminal (if available)
        if [ "$TTY_OUT" != "/dev/null" ]; then
            printf "\r%-70s\r" "" > "$TTY_OUT"
        fi
        # Print the result to stdout (goes through tee into the log)
        if [ "$success" = "true" ]; then
            printf "  ${CHECK} %s\n" "$msg"
        else
            printf "  ${CROSS} %s\n" "$msg"
        fi
    fi
}

# ── Step runner ───────────────────────────────────────────────────────────────

run_step() {
    local step_num="$1"
    local step_id="$2"
    local step_title="$3"
    local step_func="$4"
    local always="${5:-false}"   # if "true", run every time (never skipped by .done)

    if [ "$always" != "true" ] && step_done "$step_id"; then
        printf "\n  ${WHITE}[%d/%d]${RESET} ${BOLD}%s${RESET} ${DIM}... skipped${RESET}\n" \
            "$step_num" "$TOTAL_STEPS" "$step_title"
        return 0
    fi

    printf "\n  ${WHITE}[%d/%d]${RESET} ${BOLD}%s${RESET}\n" \
        "$step_num" "$TOTAL_STEPS" "$step_title"
    # Record which step is running so the EXIT trap can attribute a failure
    # to it (set -e aborts mid-step; the trap sends setup_step_failed).
    CURRENT_STEP_NUM="$step_num"
    CURRENT_STEP_ID="$step_id"
    STEP_START=$SECONDS
    "$step_func"
    mark_done "$step_id"
    ph_capture setup_step_completed \
        "\"step_id\":\"$step_id\",\"step_num\":$step_num,\"duration_s\":$(( SECONDS - STEP_START ))"
    CURRENT_STEP_ID=""
}

# ── Swap management ──────────────────────────────────────────────────────────

expand_swap() {
    if ! command -v dphys-swapfile > /dev/null 2>&1; then
        warn "dphys-swapfile not found — skipping swap expansion"
        return 0
    fi

    ORIGINAL_SWAP_SIZE=$(grep -E '^CONF_SWAPSIZE=' /etc/dphys-swapfile 2>/dev/null | cut -d= -f2 || echo "100")
    ORIGINAL_SWAP_SIZE="${ORIGINAL_SWAP_SIZE:-100}"

    if [ "$ORIGINAL_SWAP_SIZE" -ge "$SWAP_SIZE_MB" ] 2>/dev/null; then
        info "Swap already ${ORIGINAL_SWAP_SIZE}MB (>= ${SWAP_SIZE_MB}MB)"
        ORIGINAL_SWAP_SIZE=""
        return 0
    fi

    start_spinner "Expanding swap to ${SWAP_SIZE_MB}MB for compilation..."
    sudo dphys-swapfile swapoff 2>/dev/null || true
    sudo sed -i "s/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=${SWAP_SIZE_MB}/" /etc/dphys-swapfile
    sudo dphys-swapfile setup > /dev/null 2>&1
    sudo dphys-swapfile swapon
    SWAP_EXPANDED=true
    stop_spinner true "Swap expanded to ${SWAP_SIZE_MB}MB (was ${ORIGINAL_SWAP_SIZE}MB)"
}

restore_swap() {
    if [ "$SWAP_EXPANDED" != "true" ] || [ -z "$ORIGINAL_SWAP_SIZE" ]; then
        return 0
    fi
    if ! command -v dphys-swapfile > /dev/null 2>&1; then
        return 0
    fi

    sudo dphys-swapfile swapoff 2>/dev/null || true
    sudo sed -i "s/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=${ORIGINAL_SWAP_SIZE}/" /etc/dphys-swapfile
    sudo dphys-swapfile setup > /dev/null 2>&1
    sudo dphys-swapfile swapon 2>/dev/null || true
    SWAP_EXPANDED=false
    printf "  ${CHECK} Swap restored to %sMB\n" "$ORIGINAL_SWAP_SIZE"
}

# ── Trap handler ─────────────────────────────────────────────────────────────

cleanup() {
    local exit_code=$?

    # Kill spinner if still running
    if [ -n "$SPINNER_PID" ] && kill -0 "$SPINNER_PID" 2>/dev/null; then
        kill "$SPINNER_PID" 2>/dev/null
        wait "$SPINNER_PID" 2>/dev/null || true
        SPINNER_PID=""
    fi

    # Restore swap if we expanded it
    restore_swap

    if [ $exit_code -ne 0 ]; then
        # Report which step died and why. error_tail = last log lines,
        # ANSI-stripped, home-paths/username redacted (privacy contract:
        # usernames are never sent), JSON-escaped via python3 (skipped if
        # unavailable). Ctrl-C/SIGTERM are tagged so dashboards can separate
        # interruptions from real failures.
        if [ -n "$CURRENT_STEP_ID" ]; then
            local fail_duration tail_json interrupted
            fail_duration=$(( SECONDS - ${STEP_START:-$SECONDS} ))
            interrupted=false
            case "$exit_code" in 130|143) interrupted=true ;; esac
            tail_json="null"
            if command -v python3 > /dev/null 2>&1 && [ -f "$LOG_FILE" ]; then
                sleep 0.2; sync 2>/dev/null || true  # let tee drain the pipe
                # Redact home paths; redact the username only when it's >= 4
                # chars — substituting "pi" would garble "pip install" in the
                # very error text this event exists to carry.
                local scrub_expr scrub_user
                scrub_expr="s|$HOME|~|g; s|/home/[^/ \"']*|/home/<user>|g"
                scrub_user=$(whoami)
                [ "${#scrub_user}" -ge 4 ] && scrub_expr="$scrub_expr; s|$scrub_user|<user>|g"
                tail_json=$(tail -n 80 "$LOG_FILE" | sed 's/\x1b\[[0-9;]*m//g' | \
                    sed "$scrub_expr" | \
                    python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[-6000:]))' \
                    2>/dev/null || echo "null")
            fi
            ph_capture setup_step_failed \
                "\"step_id\":\"$CURRENT_STEP_ID\",\"step_num\":${CURRENT_STEP_NUM:-0},\"exit_code\":$exit_code,\"interrupted\":$interrupted,\"duration_s\":$fail_duration,\"error_tail\":$tail_json" \
                fg
        fi
        echo ""
        printf "  ${RED}Setup interrupted or failed.${RESET}\n"
        printf "  ${DIM}Re-run the script to resume from where it left off.${RESET}\n"
        printf "  ${DIM}Full log: %s${RESET}\n" "$LOG_FILE"
        printf "  ${DIM}To start fresh: bash setup.sh --clean${RESET}\n"
        echo ""
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ══════════════════════════════════════════════════════════════════════════════
#  Step functions
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 1: Pre-flight checks ────────────────────────────────────────────────

do_preflight() {
    # Architecture
    local arch
    arch=$(uname -m)
    if [ "$arch" != "aarch64" ] && [ "$arch" != "armv7l" ]; then
        die "Unsupported architecture: $arch (expected aarch64 or armv7l)"
    fi
    info "Architecture: $arch"

    # Pi detection (non-fatal)
    if [ -f /proc/device-tree/model ]; then
        local model
        model=$(tr -d '\0' < /proc/device-tree/model)
        info "Board: $model"
    else
        warn "Not a Raspberry Pi — hardware steps (I2C) will be skipped"
    fi

    # Disk space
    local avail_kb
    avail_kb=$(df --output=avail "$HOME" | tail -1 | tr -d ' ')
    local avail_mb=$((avail_kb / 1024))
    if [ "$avail_mb" -lt "$MIN_DISK_MB" ]; then
        die "Insufficient disk space: ${avail_mb}MB available, need ${MIN_DISK_MB}MB"
    fi
    info "Disk space: ${avail_mb}MB available"

    # Internet
    if ! curl -sI --connect-timeout 5 https://pypi.org > /dev/null 2>&1; then
        die "No internet connection (cannot reach pypi.org)"
    fi
    info "Internet: connected"

    # Memory
    local mem_kb
    mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local mem_mb=$((mem_kb / 1024))
    info "Memory: ${mem_mb}MB"
}

# ── Step 2: System dependencies ──────────────────────────────────────────────

do_system_deps() {
    start_spinner "Updating package lists..."
    sudo apt-get update -qq > /dev/null 2>&1
    stop_spinner true "Package lists updated"

    start_spinner "Installing system packages..."
    sudo apt-get install -y -qq \
        git python3-pip python3-venv python3-dev build-essential swig cargo \
        libsdl2-dev libsdl2-mixer-dev libsdl2-image-dev libsdl2-ttf-dev \
        > /dev/null 2>&1
    stop_spinner true "System packages installed (git, python3, build tools, swig, cargo, SDL2)"
}

# ── Step 3: Enable I2C ───────────────────────────────────────────────────────

do_i2c() {
    if ! command -v raspi-config > /dev/null 2>&1; then
        warn "raspi-config not found — skipping I2C enable (not a Raspberry Pi?)"
        return 0
    fi

    start_spinner "Enabling I2C for IMU sensor..."
    sudo raspi-config nonint do_i2c 0
    stop_spinner true "I2C enabled"
}

# ── Step 4: USB serial latency ───────────────────────────────────────────────
#
# The robot talks to the Feetech STS3215 bus servos through a USB-to-serial
# adapter plugged into the Pi Zero 2 W's micro-USB data port. Two adapter chip
# variants exist in the field; both do the same job (serial bridge at 1 Mbaud)
# and are interchangeable — only the chip differs:
#
#   Chip   | USB VID | Kernel driver | Device node    | Notes
#   -------|---------|---------------|----------------|-------------------------
#   CH343  | 0x1a86  | cdc_acm       | /dev/ttyACM*   | QinHeng; current (v3).
#          |         |               |                | No latency_timer knob;
#          |         |               |                | can't do bulk multi-servo
#          |         |               |                | sync at 1Mbaud (per-servo
#          |         |               |                | IO + retries in HWI).
#   FTDI   | 0x0403  | ftdi_sio      | /dev/ttyUSB*   | Older adapter. Honors the
#          |         |               |                | latency_timer rule below.
#
# The runtime does NOT hardcode the device path — HWI.find_servo_port()
# (mini_bdx_runtime/rustypot_position_hwi.py) auto-detects the adapter by USB
# vendor id, so any robot/cable works regardless of which ttyACM*/ttyUSB* it
# enumerates as. The latency rule below only affects FTDI (ftdi_sio); it is a
# no-op on CH343/cdc_acm, which has no latency_timer attribute.

do_usb_latency() {
    start_spinner "Setting USB serial latency rule..."
    sudo mkdir -p /etc/udev/rules.d
    echo 'SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"' | \
        sudo tee /etc/udev/rules.d/99-ftdi-latency.rules > /dev/null
    sudo udevadm control --reload-rules 2>/dev/null || true
    stop_spinner true "USB latency optimized for motor control"
}

# ── Step 5: Expand swap ──────────────────────────────────────────────────────

do_swap() {
    expand_swap
}

# ── Step 6: Clone / update runtime ───────────────────────────────────────────

do_clone() {
    # Stop the service if running (we're about to modify its files)
    if systemctl is-active "$SERVICE_NAME" > /dev/null 2>&1; then
        start_spinner "Stopping running service..."
        sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
        stop_spinner true "Service stopped"
    fi

    if [ -d "$INSTALL_DIR/.git" ]; then
        start_spinner "Checking for updates..."
        cd "$INSTALL_DIR"
        local old_head new_head
        old_head=$(git rev-parse HEAD 2>/dev/null || echo none)
        # Handle dirty state: stash local changes
        if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet HEAD 2>/dev/null; then
            git stash --include-untracked > /dev/null 2>&1 || true
        fi
        git fetch origin "$REPO_BRANCH" > /dev/null 2>&1
        git checkout "$REPO_BRANCH" > /dev/null 2>&1 || true
        git reset --hard "origin/$REPO_BRANCH" > /dev/null 2>&1
        new_head=$(git rev-parse HEAD)
        if [ "$old_head" = "$new_head" ]; then
            stop_spinner true "Already up to date ($(git rev-parse --short HEAD))"
        else
            # New code pulled — clear downstream step flags so new pip deps,
            # config, and the systemd service are re-applied on this same run
            # (and the service restarts onto the new code via step 12).
            rm -f "$STATE_DIR"/step_0[789]_*.done "$STATE_DIR"/step_1[012]_*.done
            stop_spinner true "Updated $(git rev-parse --short "$old_head" 2>/dev/null || echo new)→$(git rev-parse --short HEAD) — re-running install steps"
        fi
    else
        # Remove non-git directory if it exists (broken state)
        if [ -d "$INSTALL_DIR" ]; then
            rm -rf "$INSTALL_DIR"
        fi
        start_spinner "Cloning runtime..."
        git clone --branch "$REPO_BRANCH" --depth 1 "$REPO_URL" "$INSTALL_DIR" > /dev/null 2>&1
        stop_spinner true "Runtime cloned"
    fi
}

# ── Step 7: Python venv ──────────────────────────────────────────────────────

do_venv() {
    if [ -d "$INSTALL_DIR/.venv" ] && [ -x "$INSTALL_DIR/.venv/bin/python" ]; then
        info "Virtual environment already exists"
    else
        start_spinner "Creating Python virtual environment..."
        python3 -m venv "$INSTALL_DIR/.venv"
        stop_spinner true "Virtual environment created"
    fi

    # Always upgrade pip + setuptools
    start_spinner "Upgrading pip and setuptools..."
    "$INSTALL_DIR/.venv/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel > /dev/null 2>&1
    stop_spinner true "pip and setuptools up to date"
}

# ── Pip environment (shared by steps 8-10) ────────────────────────────────────

setup_pip_env() {
    # TMPDIR on SD card, not /tmp (which may be a small tmpfs)
    mkdir -p "$INSTALL_DIR/.pip-tmp"
    export TMPDIR="$INSTALL_DIR/.pip-tmp"

    # Single-threaded builds to prevent OOM on 512MB Pi
    export MAKEFLAGS="-j1"
    export CARGO_BUILD_JOBS=1

    # Ensure piwheels is configured for pre-built ARM wheels
    if ! grep -q "piwheels" /etc/pip.conf 2>/dev/null; then
        warn "piwheels not configured — adding for faster ARM installs"
        sudo mkdir -p /etc
        printf '[global]\nextra-index-url=https://www.piwheels.org/simple\n' | \
            sudo tee /etc/pip.conf > /dev/null
    fi
}

# Run `pip install` with FULL output captured to LOG_FILE while showing only a
# short tail on the console. The setup_step_failed telemetry reads error_tail
# from LOG_FILE, so the old `... 2>&1 | tail -5` truncated build output before it
# reached the log — PostHog (and the on-disk log) only ever saw pip's final
# summary, never the actual compiler error. Writing the full build log to disk
# fixes both local debugging and telemetry. Exit code is preserved for `set -e`.
pip_install() {
    local rc=0
    "$PIP" install "$@" >> "$LOG_FILE" 2>&1 || rc=$?
    tail -n 8 "$LOG_FILE"
    return "$rc"
}

PIP=""  # set once in main after venv step

# ── Step 8: Core pip packages ────────────────────────────────────────────────

do_pip_core() {
    setup_pip_env

    echo "  ${DIM}Installing core packages (pre-built wheels where possible)...${RESET}"

    if [ "$TEST_MODE" = "true" ]; then
        # Test mode: skip hardware-specific packages that can't install in Docker
        pip_install --no-cache-dir --prefer-binary \
            "numpy>=1.26.4" \
            "websockets>=12.0" \
            "fastapi>=0.115.0" \
            "uvicorn>=0.30.0" \
            "pydantic>=2.0.0" \
            "posthog>=3.0"

        "$INSTALL_DIR/.venv/bin/python" -c \
            "import numpy; import fastapi; import posthog; print('Core packages verified (test mode)')" \
            || die "Core package verification failed — check log at $LOG_FILE"
    else
        # Full build output goes to LOG_FILE (see pip_install); console stays short.
        pip_install --no-cache-dir --prefer-binary \
            "numpy>=1.26.4" \
            "onnxruntime>=1.18.1" \
            "adafruit-circuitpython-bno055>=5.4.13" \
            "lgpio>=0.2.2.0" \
            "supabase>=2.0.0" \
            "websockets>=12.0" \
            "fastapi>=0.115.0" \
            "uvicorn>=0.30.0" \
            "pydantic>=2.0.0" \
            "posthog>=3.0" \
            "pypot @ git+https://github.com/pollen-robotics/pypot@support-feetech-sts3215"

        "$INSTALL_DIR/.venv/bin/python" -c \
            "import numpy; import onnxruntime; import fastapi; import lgpio; import supabase; import posthog; print('Core packages verified')" \
            || die "Core package verification failed — check log at $LOG_FILE"
    fi

    info "Core packages installed and verified"
}

# ── Step 9: rustypot (Rust compilation — the slow step) ──────────────────────

do_pip_rustypot() {
    if [ "$TEST_MODE" = "true" ]; then
        info "Skipping rustypot (test mode)"
        return 0
    fi

    setup_pip_env

    echo ""
    printf "  ${YELLOW}!${RESET} ${BOLD}Compiling rustypot (Rust → Python extension)${RESET}\n"
    printf "  ${DIM}This is the slowest step — expect 10-20 minutes on Pi Zero 2W.${RESET}\n"
    printf "  ${DIM}The script is safe to interrupt and resume.${RESET}\n"
    echo ""

    CARGO_BUILD_JOBS=1 pip_install --no-cache-dir "rustypot==0.1.0"

    # Verify
    "$INSTALL_DIR/.venv/bin/python" -c "import rustypot; print('rustypot verified')" \
        || die "rustypot verification failed — check log at $LOG_FILE"

    info "rustypot compiled and verified"
}

# ── Step 10: Optional packages ───────────────────────────────────────────────

do_pip_optional() {
    setup_pip_env

    if [ "$TEST_MODE" = "true" ]; then
        echo "  ${DIM}Installing optional packages (test mode — lightweight only)...${RESET}"
        pip_install --no-cache-dir --prefer-binary \
            "openai>=1.70.0" \
            || warn "Some optional packages failed"
    else
        echo "  ${DIM}Installing optional packages (non-fatal if they fail)...${RESET}"

        # scipy — only needed by imu.py (not raw_imu.py which the walk script uses)
        # pygame — only needed for Xbox controller (not remote mode from dashboard)
        # openai — not used by walk script or server currently
        pip_install --no-cache-dir --prefer-binary \
            "scipy>=1.15.1" \
            "pygame>=2.6.0" \
            "openai>=1.70.0" \
            || warn "Some optional packages failed — robot will still work in remote mode"
    fi

    # Install the project itself in editable mode (deps already handled above)
    cd "$INSTALL_DIR"
    pip_install --no-cache-dir --no-deps -e .

    info "Package installation complete"

    # Clean up pip temp files to reclaim disk
    rm -rf "$INSTALL_DIR/.pip-tmp" 2>/dev/null || true
}

# ── Step 11: Default config ──────────────────────────────────────────────────

do_config() {
    if [ -f "$CONFIG_FILE" ]; then
        printf "  ${DOT} Config already exists at ${DIM}%s${RESET} ${DIM}(keeping existing)${RESET}\n" "$CONFIG_FILE"
    else
        cp "$INSTALL_DIR/example_config.json" "$CONFIG_FILE"
        info "Default config created at $CONFIG_FILE"
    fi
}

# ── Step 12: Systemd service ─────────────────────────────────────────────────

do_systemd() {
    if [ ! -f "$INSTALL_DIR/tnkr-robot.service.template" ]; then
        die "Service template not found at $INSTALL_DIR/tnkr-robot.service.template"
    fi

    start_spinner "Installing system service..."
    sed -e "s|TNKR_USER|$(whoami)|g" \
        -e "s|TNKR_INSTALL_DIR|$INSTALL_DIR|g" \
        "$INSTALL_DIR/tnkr-robot.service.template" | \
        sudo tee /etc/systemd/system/tnkr-robot.service > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME" > /dev/null 2>&1
    stop_spinner true "Service installed"

    # Restore swap before starting the service (service shouldn't need 2GB swap)
    restore_swap

    start_spinner "Starting server..."
    sudo systemctl restart "$SERVICE_NAME"
    # Give it a moment to start
    sleep 2
    if systemctl is-active "$SERVICE_NAME" > /dev/null 2>&1; then
        stop_spinner true "Server running"
    else
        stop_spinner false "Server failed to start"
        printf "  ${DIM}Check logs: sudo journalctl -u %s -n 30${RESET}\n" "$SERVICE_NAME"
    fi
}

# ── Success banner ───────────────────────────────────────────────────────────

print_success() {
    local server_url
    server_url="http://$(hostname).local:$SERVER_PORT"

    echo ""
    printf "  ${DIM}────────────────────────────────────────${RESET}\n"
    echo ""
    printf "  ${GREEN}${BOLD}Setup complete!${RESET}\n"
    echo ""
    printf "  ${ARROW} Server running at:\n"
    printf "    ${WHITE}%s${RESET}\n" "$server_url"
    echo ""
    printf "  ${ARROW} Next steps:\n"
    printf "    ${DIM}1.${RESET} Check motors\n"
    printf "    ${DIM}2.${RESET} Calibrate joints\n"
    printf "    ${DIM}3.${RESET} Configure features\n"
    printf "    ${DIM}4.${RESET} Start walking!\n"
    echo ""
    printf "  ${ARROW} Server commands:\n"
    printf "    ${DIM}sudo systemctl status %s${RESET}   ${DIM}— check status${RESET}\n" "$SERVICE_NAME"
    printf "    ${DIM}sudo systemctl stop %s${RESET}     ${DIM}— stop${RESET}\n" "$SERVICE_NAME"
    printf "    ${DIM}sudo systemctl start %s${RESET}    ${DIM}— start${RESET}\n" "$SERVICE_NAME"
    printf "    ${DIM}sudo systemctl restart %s${RESET}  ${DIM}— restart${RESET}\n" "$SERVICE_NAME"
    printf "    ${DIM}sudo journalctl -u %s -f${RESET}  ${DIM}— live logs${RESET}\n" "$SERVICE_NAME"
    echo ""
    printf "  ${DIM}Setup log: %s${RESET}\n" "$LOG_FILE"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

main() {
    # ── Parse flags ───────────────────────────────────────────────────────
    for arg in "$@"; do
        case "$arg" in
            --test) TEST_MODE=true ;;
        esac
    done

    if [ "$TEST_MODE" = "true" ]; then
        warn "Test mode — hardware packages (onnxruntime, rustypot, adafruit) will be skipped"
    fi

    if [ "${1:-}" = "--clean" ]; then
        cd "$HOME"
        echo ""
        printf "  ${YELLOW}${BOLD}Cleaning previous installation...${RESET}\n"
        echo ""

        # Stop and remove service
        sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
        sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true
        sudo rm -f /etc/systemd/system/tnkr-robot.service
        sudo systemctl daemon-reload 2>/dev/null || true
        info "Service removed"

        # Remove runtime directory
        if [ -d "$INSTALL_DIR" ]; then
            rm -rf "$INSTALL_DIR"
            info "Runtime directory removed"
        fi

        # Remove config
        if [ -f "$CONFIG_FILE" ]; then
            rm -f "$CONFIG_FILE"
            info "Config file removed"
        fi

        # Remove state directory
        if [ -d "$STATE_DIR" ]; then
            rm -rf "$STATE_DIR"
            info "State directory removed"
        fi

        # NOTE: $TELEMETRY_FILE is deliberately preserved — the anonymous
        # device id should stay stable across reinstalls.

        CLEAN_INSTALL=true
        echo ""
        info "Clean complete — running fresh install"
        echo ""
    fi

    # ── Init state directory and logging ──────────────────────────────────
    mkdir -p "$STATE_DIR"
    # Log all output to file while still showing on terminal
    exec > >(tee -a "$LOG_FILE") 2>&1

    # Check for resumed install
    local done_count
    done_count=$(find "$STATE_DIR" -name "*.done" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$done_count" -gt 0 ]; then
        echo ""
        printf "  ${CYAN}Resuming setup (${done_count}/${TOTAL_STEPS} steps already complete)${RESET}\n"
    fi

    # ── Telemetry consent + install-started event ─────────────────────────
    telemetry_init
    local resumed="false" hw_props
    [ "$done_count" -gt 0 ] && resumed="true"
    hw_props=$(telemetry_hw_props)
    ph_capture setup_started \
        "$hw_props,\"resumed\":$resumed,\"steps_already_done\":$done_count,\"clean_install\":$CLEAN_INSTALL,\"\$set\":{$hw_props}"

    # ── Header ────────────────────────────────────────────────────────────
    echo ""
    printf "${CYAN}"
    cat << 'LOGO'

    ████████╗███╗   ██╗██╗  ██╗██████╗
    ╚══██╔══╝████╗  ██║██║ ██╔╝██╔══██╗
       ██║   ██╔██╗ ██║█████╔╝ ██████╔╝
       ██║   ██║╚██╗██║██╔═██╗ ██╔══██╗
       ██║   ██║ ╚████║██║  ██╗██║  ██║
       ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝

LOGO
    printf "${RESET}"
    printf "  ${BOLD}Open Duck Mini ${DIM}— Robot Setup${RESET}\n"
    printf "  ${DIM}────────────────────────────────────────${RESET}\n"

    # ── Run steps ─────────────────────────────────────────────────────────

    run_step  1 "01_preflight"    "Pre-flight checks"                      do_preflight
    run_step  2 "02_system_deps"  "System dependencies"                    do_system_deps
    run_step  3 "03_i2c"          "Hardware interfaces (I2C)"              do_i2c
    run_step  4 "04_usb_latency"  "Motor communication (USB latency)"      do_usb_latency
    run_step  5 "05_swap"         "Expand swap for compilation"            do_swap
    run_step  6 "06_clone"        "Clone / update runtime"                 do_clone        true
    run_step  7 "07_venv"         "Python virtual environment"             do_venv

    # Set PIP path now that venv is guaranteed to exist
    PIP="$INSTALL_DIR/.venv/bin/pip"

    run_step  8 "08_pip_core"     "Core Python packages"                   do_pip_core
    run_step  9 "09_pip_rustypot" "Compile rustypot (Rust extension)"      do_pip_rustypot
    run_step 10 "10_pip_optional" "Optional packages"                      do_pip_optional
    run_step 11 "11_config"       "Robot configuration"                    do_config
    run_step 12 "12_systemd"      "TNKR server"                           do_systemd

    # ── Done ──────────────────────────────────────────────────────────────
    ph_capture setup_completed \
        "\"total_duration_s\":$SECONDS,\"clean_install\":$CLEAN_INSTALL" fg
    print_success
}

# Wrap in main() — prevents partial execution when piped from curl
main "$@"
