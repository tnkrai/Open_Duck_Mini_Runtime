"""scripts/setup.sh address discovery — the IP printed beside the `.local` name.

The installer used to end with `Server running at: http://<name>.local:8000` and
nothing else. `.local` is mDNS, which is dependable on macOS, needs avahi on
Linux, and on Windows resolves erratically enough that a tester gave up and
scanned the subnet to find the Pi by hand. `lan_ip` supplies the address that
line was missing.

These run the real function text out of setup.sh against stubbed system tools,
so the shell logic is checked rather than described.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SETUP_SH = Path(__file__).parent.parent / "scripts" / "setup.sh"

# Stubs for everything that would ask the actual machine where it lives. Each
# reads its answer out of the environment so a test can pick the scenario.
#
# `ip` and `hostname` are shell functions rather than files on $PATH: the real
# ones exist on the developer's machine as well as on the Pi, and a test that
# silently fell through to them would pass by describing the laptop it ran on.
HARNESS = r"""
set -euo pipefail

ip() {
    [ "${IP_ROUTE_OUT:-}" = "" ] && return 1
    printf '%s\n' "$IP_ROUTE_OUT"
}

hostname() {
    if [ "${1:-}" = "-I" ]; then
        [ "${HOSTNAME_I_OUT:-}" = "" ] && return 1
        printf '%s\n' "$HOSTNAME_I_OUT"
        return 0
    fi
    printf '%s\n' "${HOSTNAME_OUT:-duck}"
}

SERVER_PORT=8000
SERVICE_NAME="tnkr-robot"
LOG_FILE="/tmp/setup.log"
FROM_CLI="${FROM_CLI:-false}"
BOLD=""; DIM=""; RESET=""; GREEN=""; WHITE=""; ARROW="->"
"""

# A realistic `ip -4 route get 1.1.1.1` line, copied from a Pi: the source
# address is not in a fixed column, which is why the awk scans for `src`.
ROUTE_LINE = "1.1.1.1 via 192.168.1.1 dev wlan0 src 192.168.1.42 uid 1000"


def _run(script: str, env: dict, *, through: str = "# ── Success banner") -> subprocess.CompletedProcess:
    """Source setup.sh's address block with stubs, then run `script`."""
    text = SETUP_SH.read_text()
    start = text.index("# ── Address discovery")
    # `index(through, start)`: the file opens with a banner drawn in the same
    # box characters, so an unanchored search for the Main header finds that one
    # and slices backwards to nothing.
    block = text[start : text.index(through, start)]
    full = HARNESS + block + "\n" + script
    return subprocess.run(
        ["bash", "-c", full],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", **env},
    )


def _banner(env: dict) -> subprocess.CompletedProcess:
    """Same, but far enough down the file to include print_success itself."""
    return _run("print_success", env, through="# ═══")


# ── lan_ip: which source wins ────────────────────────────────────────────────


def test_prefers_the_address_the_operator_reached_us_on():
    """$SSH_CONNECTION field 3 is the server side of the operator's own
    connection, so it is the one address already proven to work from where they
    are sitting. Field 1 is *their* address, and picking it would send them to
    their own machine."""
    r = _run(
        'lan_ip',
        {"SSH_CONNECTION": "192.168.1.9 51344 192.168.1.42 22", "IP_ROUTE_OUT": ROUTE_LINE},
    )
    assert r.stdout == "192.168.1.42", r.stdout + r.stderr


def test_a_pi_on_two_networks_does_not_fall_back_to_the_routing_table():
    """The trap this ordering exists for. A Pi with Wi-Fi and Ethernet both up
    has a default route that may leave by the interface the operator's laptop
    cannot see; the connection they are already holding cannot be wrong."""
    r = _run(
        'lan_ip',
        {"SSH_CONNECTION": "10.0.0.5 51344 10.0.0.77 22", "IP_ROUTE_OUT": ROUTE_LINE},
    )
    assert r.stdout == "10.0.0.77", r.stdout + r.stderr


def test_ipv6_ssh_connection_falls_through_to_the_routing_table():
    """Reached over IPv6, field 3 is an IPv6 address. Printing it would need
    bracketing and would not help the Windows case this exists for, so it is
    skipped rather than mangled."""
    r = _run(
        'lan_ip',
        {"SSH_CONNECTION": "fe80::1 51344 fe80::dead:beef 22", "IP_ROUTE_OUT": ROUTE_LINE},
    )
    assert r.stdout == "192.168.1.42", r.stdout + r.stderr


def test_sitting_at_the_robot_uses_the_routing_table():
    """No SSH_CONNECTION means a keyboard plugged into the Pi."""
    r = _run('lan_ip', {"IP_ROUTE_OUT": ROUTE_LINE})
    assert r.stdout == "192.168.1.42", r.stdout + r.stderr


def test_falls_back_to_hostname_when_there_is_no_default_route():
    """A Pi on a switch with no gateway has no route to 1.1.1.1 and is still
    perfectly reachable from the laptop next to it."""
    r = _run('lan_ip', {"HOSTNAME_I_OUT": "192.168.1.42 "})
    assert r.stdout == "192.168.1.42", r.stdout + r.stderr


def test_hostname_fallback_skips_ipv6_to_find_the_ipv4():
    """`hostname -I` mixes both families on one line and orders them however the
    kernel feels; taking $1 blindly yields an IPv6 address about half the time."""
    r = _run('lan_ip', {"HOSTNAME_I_OUT": "fe80::dead:beef 192.168.1.42 "})
    assert r.stdout == "192.168.1.42", r.stdout + r.stderr


# ── lan_ip: when there is nothing worth printing ─────────────────────────────


def test_loopback_is_not_an_address_and_is_refused():
    """Sending the operator to 127.0.0.1 is worse than sending them nowhere: it
    resolves, it connects to their own machine, and it fails confusingly."""
    r = _run('lan_ip && echo "PRINTED" || echo "REFUSED"', {"HOSTNAME_I_OUT": "127.0.0.1 "})
    assert "REFUSED" in r.stdout, r.stdout + r.stderr
    assert "127.0.0.1" not in r.stdout, r.stdout


def test_no_address_anywhere_returns_non_zero_and_prints_nothing():
    r = _run('lan_ip && echo "PRINTED" || echo "REFUSED"', {})
    assert r.stdout.strip() == "REFUSED", r.stdout + r.stderr


# ── _is_ipv4 ─────────────────────────────────────────────────────────────────


def test_is_ipv4_accepts_and_rejects():
    cases = {
        "192.168.1.42": True,
        "10.0.0.1": True,
        "255.255.255.255": True,
        "0.0.0.0": True,
        "256.1.1.1": False,      # octet out of range
        "1.2.3": False,          # too few
        "1.2.3.4.5": False,      # too many
        "1.2.3.four": False,
        "fe80::1": False,
        "1.2.3.4444": False,     # length guard
        "": False,
    }
    script = "\n".join(
        f'_is_ipv4 "{value}" && echo "{value} YES" || echo "{value} NO"' for value in cases
    )
    r = _run(script, {})
    for value, expected in cases.items():
        want = f"{value} {'YES' if expected else 'NO'}"
        assert want in r.stdout, f"{value!r}: {r.stdout}{r.stderr}"


# ── print_success ────────────────────────────────────────────────────────────


def test_banner_prints_the_ip_beside_the_name():
    r = _banner({"HOSTNAME_OUT": "new1", "IP_ROUTE_OUT": ROUTE_LINE})
    assert "http://new1.local:8000" in r.stdout, r.stdout + r.stderr
    assert "http://192.168.1.42:8000" in r.stdout, r.stdout + r.stderr
    # Order matters: the name survives a DHCP lease changing, the address does not.
    assert r.stdout.index("new1.local") < r.stdout.index("192.168.1.42"), r.stdout


def test_banner_still_finishes_when_no_address_can_be_found():
    """The `|| lan=""` guard under `set -e`. Without it a finished install exits
    non-zero because a cosmetic line had nothing to say."""
    r = _banner({"HOSTNAME_OUT": "new1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "http://new1.local:8000" in r.stdout, r.stdout
    assert "Setup complete!" in r.stdout, r.stdout


def test_from_cli_prints_nothing_at_all():
    """--from-cli means the tnkr CLI owns the ending; this banner would tell an
    operator at a laptop to run systemctl against their laptop."""
    r = _banner({"HOSTNAME_OUT": "new1", "IP_ROUTE_OUT": ROUTE_LINE, "FROM_CLI": "true"})
    assert r.stdout.strip() == "", r.stdout
