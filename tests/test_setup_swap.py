"""scripts/setup.sh swap expansion.

The Zero 2W has 512MB of RAM and this script compiles a Rust extension on it, so
swap decides whether the install finishes or thrashes for an hour. It silently
did nothing on current Pi OS: `expand_swap` only knew dphys-swapfile, and Trixie
replaced that with rpi-swap (zram). These tests run the real function text out of
setup.sh against stubbed system tools, so the shell logic is checked rather than
described.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SETUP_SH = Path(__file__).parent.parent / "scripts" / "setup.sh"

# The stubs stand in for everything that would touch the machine. Each records
# what it was asked to do in $CALLS so a test can assert on the sequence.
HARNESS = r"""
set -euo pipefail
CALLS="$WORK/calls"; : > "$CALLS"

info()  { echo "INFO: $*"; }
warn()  { echo "WARN: $*"; }
start_spinner() { echo "SPIN: $*"; }
stop_spinner()  { echo "DONE: $*"; }
CHECK="+"; RESET=""

sudo() {
    echo "sudo $*" >> "$CALLS"
    case "$1" in
        rm)        shift; command rm "$@" ;;
        chmod)     shift; command chmod "$@" ;;
        fallocate) [ "${FALLOCATE_FAILS:-0}" = "1" ] && return 1
                   command : > "$SWAP_FILE" ; return 0 ;;
        dd)        command : > "$SWAP_FILE" ; return 0 ;;
        mkswap)    [ "${MKSWAP_FAILS:-0}" = "1" ] && return 1 ; return 0 ;;
        swapon)    [ "${SWAPON_FAILS:-0}" = "1" ] && return 1 ; return 0 ;;
        swapoff)   return 0 ;;
        *)         return 0 ;;
    esac
}
command_v_dphys="${HAS_DPHYS:-0}"
command() {
    if [ "$1" = "-v" ] && [ "$2" = "dphys-swapfile" ]; then
        [ "$command_v_dphys" = "1" ] && { echo /sbin/dphys-swapfile; return 0; }
        return 1
    fi
    builtin command "$@"
}
df() { echo "Avail"; echo "$(( ${FREE_MB:-100000} * 1024 ))"; }

SWAP_SIZE_MB="${TARGET_MB:-2048}"
SWAP_INSTALL_HEADROOM_MB=1024
SWAP_MIN_USEFUL_MB=256
ORIGINAL_SWAP_SIZE=""
SWAP_EXPANDED=false
"""


def _run(script: str, env: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    """Source setup.sh's swap block with stubs, then run `script`."""
    text = SETUP_SH.read_text()
    start = text.index("# ── Swap management")
    block = text[start : text.index("# ── Trap handler")]
    # the harness defines SWAP_FILE_FAKE; point the real variable at it
    block = block.replace('SWAP_FILE="/var/swap.tnkr-setup"', 'SWAP_FILE="$WORK/swapfile"')
    full = f'WORK="{tmp_path}"\n' + HARNESS + block + "\n" + script
    return subprocess.run(
        ["bash", "-c", full], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "WORK": str(tmp_path), **env},
    )


def _row(filename: str, kind: str, size_kb: int, prio: int = -2) -> str:
    """One /proc/swaps row in the kernel's real format.

    Verified against a live Pi Zero 2W on Trixie with `cat -A /proc/swaps`:

        /dev/zram0                              partition^I424956^I^I0^I^I100$

    The kernel space-PADS the filename to column 40 (or emits a single space if
    the path is longer), then tab-separates the rest. An earlier version of this
    fixture put a tab straight after the filename, which no kernel produces, and
    it made a correct grep look broken.
    """
    pad = " " * (40 - len(filename)) if len(filename) < 40 else " "
    return f"{filename}{pad}{kind}\t{size_kb}\t\t0\t\t{prio}\n"


def _swaps(tmp_path: Path, *lines: str) -> str:
    p = tmp_path / "swaps"
    p.write_text("Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n" + "".join(lines))
    return str(p)


def test_zram_does_not_count_as_swap(tmp_path):
    """The trap this bug hides behind. Trixie's rpi-swap is zram, so `free` shows
    swap and a naive check passes — but zram is backed by the same RAM we are
    trying not to exhaust and cannot hold a cargo build."""
    proc = _swaps(tmp_path, _row("/dev/zram0", "partition", 424956, 100))
    r = _run('echo "MB=$(active_disk_swap_mb)"', {"SWAP_PROC": proc}, tmp_path)
    assert "MB=0" in r.stdout, r.stdout + r.stderr


def test_a_real_swapfile_does_count(tmp_path):
    proc = _swaps(tmp_path, _row("/var/swap", "file", 2097148))
    r = _run('echo "MB=$(active_disk_swap_mb)"', {"SWAP_PROC": proc}, tmp_path)
    assert "MB=2047" in r.stdout or "MB=2048" in r.stdout, r.stdout


def test_mixed_zram_and_file_counts_only_the_file(tmp_path):
    proc = _swaps(
        tmp_path,
        _row("/dev/zram0", "partition", 424956, 100),
        _row("/var/swap", "file", 1048576),
    )
    r = _run('echo "MB=$(active_disk_swap_mb)"', {"SWAP_PROC": proc}, tmp_path)
    assert "MB=1024" in r.stdout, r.stdout


def test_without_dphys_it_creates_a_swapfile(tmp_path):
    """The reported bug. Before the fix this printed 'dphys-swapfile not found —
    skipping swap expansion' and returned, leaving the compile with no swap."""
    proc = _swaps(tmp_path)  # no swap at all
    r = _run(
        'expand_swap; echo "EXPANDED=$SWAP_EXPANDED"; cat "$WORK/calls"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "8000"},
        tmp_path,
    )
    assert "EXPANDED=true" in r.stdout, r.stdout + r.stderr
    assert "mkswap" in r.stdout and "swapon" in r.stdout
    assert "chmod 600" in r.stdout, "swapon refuses a group/world-readable file"
    assert "skipping swap expansion" not in r.stdout


def test_dphys_is_still_preferred_when_present(tmp_path):
    """Older images in the field already run this path; the fix must not move them."""
    proc = _swaps(tmp_path)
    r = _run(
        'expand_swap; cat "$WORK/calls"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "1", "FREE_MB": "8000"},
        tmp_path,
    )
    assert "dphys-swapfile setup" in r.stdout, r.stdout
    assert "mkswap" not in r.stdout


def test_enough_disk_swap_already_is_a_no_op(tmp_path):
    proc = _swaps(tmp_path, _row("/var/swap", "file", 3145728))
    r = _run(
        'expand_swap; echo "EXPANDED=$SWAP_EXPANDED"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "8000"},
        tmp_path,
    )
    assert "EXPANDED=false" in r.stdout
    assert "already" in r.stdout


def test_the_swapfile_is_sized_to_leave_room_for_the_install(tmp_path):
    """MIN_DISK_MB and SWAP_SIZE_MB are both 2048, so a Pi that just scraped past
    preflight has nothing spare. Taking the full 2048 would fill the card and
    fail pip later with a confusing ENOSPC."""
    proc = _swaps(tmp_path)
    r = _run("expand_swap", {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "1800"}, tmp_path)
    assert "776MB swap file" in r.stdout, r.stdout  # 1800 - 1024 headroom


def test_no_room_warns_instead_of_filling_the_card(tmp_path):
    proc = _swaps(tmp_path)
    r = _run(
        'expand_swap; echo "EXPANDED=$SWAP_EXPANDED"; cat "$WORK/calls"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "1100"},
        tmp_path,
    )
    assert "EXPANDED=false" in r.stdout
    assert "mkswap" not in r.stdout
    assert "may be slow or run out of memory" in r.stdout


def test_a_stale_swapfile_from_a_killed_run_is_replaced_not_reused(tmp_path):
    """A run killed hard enough to skip the EXIT trap leaves the file behind, and
    its size is unknown, so it is replaced rather than reused: mkswap on a file
    that is still swapped on corrupts it.

    This pins the swapoff-before-mkswap ORDER, not the parsing. Both the awk
    field match and a plain grep read the real format correctly.
    """
    (tmp_path / "swapfile").write_text("stale")
    proc = _swaps(tmp_path, _row(f"{tmp_path}/swapfile", "file", 524288))
    r = _run(
        'expand_swap; cat "$WORK/calls"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "8000"},
        tmp_path,
    )
    calls = r.stdout
    assert "swapoff" in calls, "must swapoff the stale file before touching it"
    assert calls.index("swapoff") < calls.index("mkswap"), "swapoff has to precede mkswap"


def test_dd_takes_over_when_fallocate_fails(tmp_path):
    """fallocate can leave holes on some filesystems and mkswap refuses those."""
    proc = _swaps(tmp_path)
    r = _run(
        'expand_swap; cat "$WORK/calls"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "8000", "FALLOCATE_FAILS": "1"},
        tmp_path,
    )
    assert "dd if=/dev/zero" in r.stdout, r.stdout
    assert "EXPANDED" not in r.stdout or "mkswap" in r.stdout


def test_a_failed_swapon_warns_and_does_not_abort_the_install(tmp_path):
    proc = _swaps(tmp_path)
    r = _run(
        'expand_swap; echo "RC=$?"; echo "EXPANDED=$SWAP_EXPANDED"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "8000", "SWAPON_FAILS": "1"},
        tmp_path,
    )
    assert "RC=0" in r.stdout, "swap is best-effort; it must never fail the install"
    assert "EXPANDED=false" in r.stdout
    assert "Could not create a swap file" in r.stdout


def test_restore_removes_the_swapfile(tmp_path):
    """Temporary on purpose. Nothing needs swap after the build and a 2GB file
    left on an SD card forever is a real cost."""
    proc = _swaps(tmp_path)
    r = _run(
        'expand_swap; restore_swap; echo "EXISTS=$([ -e "$WORK/swapfile" ] && echo yes || echo no)"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "8000"},
        tmp_path,
    )
    assert "EXISTS=no" in r.stdout, r.stdout
    assert "Swap file removed" in r.stdout


def test_restore_leaves_swap_alone_when_we_did_not_touch_it(tmp_path):
    proc = _swaps(tmp_path, _row("/var/swap", "file", 3145728))
    r = _run(
        'expand_swap; restore_swap; cat "$WORK/calls"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "8000"},
        tmp_path,
    )
    assert "swapoff" not in r.stdout


def test_step_5_reruns_on_resume():
    """restore_swap runs in the EXIT trap, so an interrupted run tears the swap
    down. Without always=true the resumed run — the one after an OOM, on the
    machine that just proved it needs swap — skips step 5 as done and compiles
    with none. Independent of the dphys bug: this bit old Pi OS too.
    """
    line = next(
        l for l in SETUP_SH.read_text().splitlines() if 'run_step  5 "05_swap"' in l
    )
    assert line.split()[-1] == "true", f"step 5 must pass always=true, got: {line.strip()}"


def test_the_real_kernel_format_parses(tmp_path):
    """Anchored on a line copied verbatim off a Pi Zero 2W running Trixie:

        /dev/zram0                              partition\t424956\t\t0\t\t100

    415MB of zram and no disk swap at all, which is what `free` on that machine
    reports as "Swap: 414". The whole bug lives in the gap between those two
    readings, so the fixture format is worth pinning.
    """
    p = tmp_path / "swaps"
    p.write_text(
        "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n"
        "/dev/zram0                              partition\t424956\t\t0\t\t100\n"
    )
    r = _run('echo "MB=$(active_disk_swap_mb)"', {"SWAP_PROC": str(p)}, tmp_path)
    assert "MB=0" in r.stdout, r.stdout


def test_our_own_leftover_swapfile_is_adopted_so_it_still_gets_cleaned_up(tmp_path):
    """A run killed hard enough to skip the EXIT trap leaves OUR swapfile active.

    The next run sees it in /proc/swaps, counts it toward the target and returns
    early "already enough" — correct as far as the build goes, but it never takes
    ownership, so restore_swap leaves the file on the card. Adopting it means the
    run that benefits from it is also the run that cleans it up.
    """
    (tmp_path / "swapfile").write_text("leftover")
    proc = _swaps(tmp_path, _row(f"{tmp_path}/swapfile", "file", 2097152))
    r = _run(
        'expand_swap; echo "EXPANDED=$SWAP_EXPANDED"; restore_swap;'
        ' echo "EXISTS=$([ -e "$WORK/swapfile" ] && echo yes || echo no)"',
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "8000"},
        tmp_path,
    )
    assert "EXPANDED=true" in r.stdout, r.stdout
    assert "EXISTS=no" in r.stdout, "our own leftover swapfile should not survive the run"


def test_nearly_enough_swap_is_not_reported_as_an_out_of_disk_warning(tmp_path):
    """A 512MB file measures 511MB (524284 KB), so an exact-size swapfile always
    reads a megabyte short of its own target. That tiny deficit used to fall into
    the out-of-disk branch and tell an operator with 22GB free and a working
    swapfile that their build might run out of memory. Seen on a real Pi."""
    proc = _swaps(tmp_path, _row("/var/swap.tnkr-setup", "file", 524284))
    r = _run(
        "expand_swap",
        {"SWAP_PROC": proc, "HAS_DPHYS": "0", "FREE_MB": "22675", "TARGET_MB": "512"},
        tmp_path,
    )
    assert "run out of memory" not in r.stdout, r.stdout
    assert "Only 22675MB free" not in r.stdout
    assert "within" in r.stdout and "target" in r.stdout, r.stdout
