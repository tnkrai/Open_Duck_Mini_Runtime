#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  Test runner for setup.sh using Docker arm64 emulation
#
#  Prerequisites: Docker Desktop with arm64 emulation (default on Apple Silicon)
#
#  Usage: bash test-setup.sh [test_name]
#    bash test-setup.sh           # Run all tests
#    bash test-setup.sh syntax    # Run only syntax test
#    bash test-setup.sh build     # Build the Docker image only
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

IMAGE="tnkr-setup-test"
PLATFORM="linux/arm64"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

pass() { printf "  ${GREEN}PASS${RESET} %s\n" "$1"; }
fail() { printf "  ${RED}FAIL${RESET} %s\n" "$1"; FAILURES=$((FAILURES + 1)); }
header() { printf "\n${BOLD}%s${RESET}\n" "$1"; }

FAILURES=0

# ── Test: Shellcheck ──────────────────────────────────────────────────────────

test_shellcheck() {
    header "Test: shellcheck static analysis"
    if command -v shellcheck > /dev/null 2>&1; then
        if shellcheck -x "$SCRIPT_DIR/scripts/setup.sh"; then
            pass "shellcheck found no issues"
        else
            fail "shellcheck found issues (see above)"
        fi
    else
        printf "  ${YELLOW}SKIP${RESET} shellcheck not installed (brew install shellcheck)\n"
    fi
}

# ── Test: Bash syntax ─────────────────────────────────────────────────────────

test_syntax() {
    header "Test: bash syntax check"
    if bash -n "$SCRIPT_DIR/scripts/setup.sh"; then
        pass "syntax OK"
    else
        fail "syntax errors found"
    fi
}

# ── Test: Docker build ────────────────────────────────────────────────────────

test_build() {
    header "Test: Docker image build"
    if docker build --platform "$PLATFORM" -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.test" "$SCRIPT_DIR" 2>&1; then
        pass "Docker image built"
    else
        fail "Docker build failed"
        return 1
    fi
}

# ── Test: Idempotency (run twice, second should skip) ─────────────────────────

test_idempotency() {
    header "Test: idempotency (second run should skip all steps)"
    # This test is limited because Docker stubs mean pip won't really install
    # But it validates the state-file logic
    if docker run --platform "$PLATFORM" --rm "$IMAGE" \
        bash -c "bash /home/piuser/test-setup.sh 2>&1; echo '--- SECOND RUN ---'; bash /home/piuser/test-setup.sh 2>&1 | grep -c 'skipped'" \
        > /dev/null 2>&1; then
        pass "idempotency logic works"
    else
        fail "idempotency test failed"
    fi
}

# ── Test: Clean flag ──────────────────────────────────────────────────────────

test_clean() {
    header "Test: --clean flag removes state"
    if docker run --platform "$PLATFORM" --rm "$IMAGE" \
        bash -c "mkdir -p ~/.tnkr-setup && touch ~/.tnkr-setup/step_01_preflight.done && bash /home/piuser/test-setup.sh --clean 2>&1 | grep -q 'State directory removed'" \
        2>&1; then
        pass "--clean removes state directory"
    else
        fail "--clean flag test failed"
    fi
}

# ── Run tests ─────────────────────────────────────────────────────────────────

run_all() {
    test_shellcheck
    test_syntax
    test_build || return 1
    test_idempotency
    test_clean
}

case "${1:-all}" in
    syntax)     test_syntax ;;
    shellcheck) test_shellcheck ;;
    build)      test_build ;;
    idempotent) test_build && test_idempotency ;;
    clean)      test_build && test_clean ;;
    all)        run_all ;;
    *)          echo "Usage: bash test-setup.sh [syntax|shellcheck|build|idempotent|clean|all]" ;;
esac

echo ""
if [ "$FAILURES" -gt 0 ]; then
    printf "${RED}${BOLD}%d test(s) failed${RESET}\n" "$FAILURES"
    exit 1
else
    printf "${GREEN}${BOLD}All tests passed${RESET}\n"
fi
