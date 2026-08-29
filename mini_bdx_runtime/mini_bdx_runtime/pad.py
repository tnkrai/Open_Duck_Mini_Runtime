"""The Xbox controller that ships with the duck, as the Pi sees it.

Studio never talks to this radio. The agent wraps `bluetoothctl` so Connect can
list My devices (paired) vs Nearby (scan), connect / disconnect, and walk start
can refuse a controller walk when pygame has no joystick.

Pairing needs a BlueZ agent on the Pi. Headless images have none, so
`bluetoothctl pair` alone fails with AuthenticationFailed / "No agent available
for request type 2". We drive an interactive `bluetoothctl` over a PTY with
`agent DisplayYesNo` + `default-agent` and auto-confirm the passkey.
"""

from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import time
from typing import Optional

_MAC = re.compile(r"^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")
_XBOX_HINTS = ("xbox", "wireless controller", "gamepad")
_DEVICE_LINE = re.compile(r"^Device\s+([0-9A-Fa-f:]{17})\s+(.*)$")
_BATTERY_PCT = re.compile(r"\((\d{1,3})\)")
# `bluetoothctl show` with no radio at all -- an rfkill block looks like this too,
# because a blocked adapter is not exposed as a controller.
_NO_CONTROLLER = re.compile(r"no default controller", re.I)

# How long a dead stick is allowed to keep the last command (O3).
SILENCE_S = 0.3
# Xbox often advertises as a bare MAC for several seconds before the Name lands.
_DEFAULT_SCAN_S = 18.0
# HID often lands a beat after Bonded:yes.
_JOYSTICK_WAIT_S = 4.0


def is_xbox_name(name: str) -> bool:
    n = name.lower()
    return any(hint in n for hint in _XBOX_HINTS)


def is_controller_device(*, name: str, icon: str = "") -> bool:
    """True for an Xbox / gamepad the operator should be offered."""
    if is_xbox_name(name):
        return True
    # Name can still be the MAC while Icon already says gamepad.
    return icon.strip().lower() == "input-gaming"


def valid_address(address: str) -> bool:
    return bool(_MAC.match(address))


def walk_flags(walk_input: str) -> list[str]:
    """Flags after the walk script path. keyboard (or anything else) keeps --remote."""
    if walk_input == "pad":
        return ["--commands"]
    return ["--remote", "--commands"]


def should_zero_commands(alive: bool, last_ok: float, now: float, silence_s: float = SILENCE_S) -> bool:
    """True when the walk should emit a zero command (pad gone or silent)."""
    return (not alive) and (now - last_ok >= silence_s)


def joystick_present() -> bool:
    try:
        import pygame

        pygame.init()
        pygame.joystick.init()
        return pygame.joystick.get_count() > 0
    except Exception:
        return False


def _bt(*args: str, timeout: float = 12.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bluetoothctl", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ── The radio itself ─────────────────────────────────────────────────────────
# A powered-off adapter used to be indistinguishable from "no controller
# nearby": `bluetoothctl scan` returns an empty list either way, Studio rendered
# "turn the controller on, then hold sync", and an operator held the sync button
# until they gave up. The previous `_ensure_adapter_on` ran `power on`, threw the
# result away and returned None, so nothing downstream -- not the agent, not
# Studio, not analytics -- could tell a dead radio from an empty room.
#
# `power on` also cannot clear an rfkill soft block, which is the state a fresh
# Pi OS image most often boots into. So the wake attempt has to unblock first,
# and then it has to CHECK, because the useful output of this function is not
# the side effect but the answer.
#
# The wake is `wake_adapter()` and NOTHING here calls it. Scanning, pairing,
# disconnecting and forgetting all read the radio with `adapter_state()` and
# refuse early when it is down. Powering a radio on is a thing the operator
# asks for, on the one route that does it, so the screen can say the radio was
# off before it changes it and can report what happened when it will not.

#: `reason` on the adapter dict. Closed, and ordered by what the operator has to
#: do about it: a hard block is a physical switch, a soft block is one command,
#: `off` is ours to fix, `missing` means there is no Bluetooth on this Pi at all.
ADAPTER_OK = None
ADAPTER_MISSING = "missing"
ADAPTER_HARD_BLOCKED = "hard_blocked"
ADAPTER_BLOCKED = "blocked"
ADAPTER_OFF = "off"


def _adapter_dict(*, present: bool, powered: bool, soft: bool, hard: bool) -> dict:
    if hard:
        reason = ADAPTER_HARD_BLOCKED
    elif soft:
        reason = ADAPTER_BLOCKED
    elif not present:
        reason = ADAPTER_MISSING
    elif not powered:
        reason = ADAPTER_OFF
    else:
        reason = ADAPTER_OK
    return {
        "present": present,
        "powered": powered,
        "blocked": bool(soft or hard),
        "hardBlocked": bool(hard),
        "reason": reason,
        # Filled in by wake_adapter. `wokeVia` is which escalation step
        # actually worked, and `wakeError` is the sentence BlueZ printed when
        # none did -- the one string that turns "the pad never appeared" into a
        # diagnosis. A plain read leaves both unset.
        "wokeVia": None,
        "wakeError": None,
    }


def _rfkill(*args: str, timeout: float = 5.0) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(
            ["rfkill", *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _rfkill_state() -> tuple[bool, bool]:
    """`(soft_blocked, hard_blocked)` for bluetooth. Both False if rfkill is absent."""
    out = _rfkill("list", "bluetooth")
    if out is None or out.returncode != 0:
        return False, False
    soft = hard = False
    for line in out.stdout.splitlines():
        s = line.strip().lower()
        if s.startswith("soft blocked:"):
            soft = soft or s.endswith("yes")
        elif s.startswith("hard blocked:"):
            hard = hard or s.endswith("yes")
    return soft, hard


def _rfkill_unblock() -> bool:
    """Clear a soft block. True if a command reported success.

    Unblocking needs CAP_NET_ADMIN and the agent runs as the login user (see
    tnkr-robot.service.template: `User=TNKR_USER`), so the plain call usually
    fails. Pi OS gives that user passwordless sudo, so `sudo -n` is the one that
    normally lands. `-n` matters: without it a missing sudo rule blocks on a
    password prompt that nobody is there to answer.
    """
    for argv in (("unblock", "bluetooth"),):
        out = _rfkill(*argv)
        if out is not None and out.returncode == 0:
            return True
    try:
        out = subprocess.run(
            ["sudo", "-n", "rfkill", "unblock", "bluetooth"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return out.returncode == 0


def adapter_state() -> dict:
    """What the radio is doing, independent of any controller being near it."""
    soft, hard = _rfkill_state()
    present = False
    powered = False
    try:
        shown = _bt("show", timeout=5.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        shown = None
    if shown is not None and shown.returncode == 0 and not _NO_CONTROLLER.search(shown.stdout):
        present = True
        for line in shown.stdout.splitlines():
            s = line.strip()
            if s.startswith("Powered:"):
                powered = "yes" in s.lower()
                break
    return _adapter_dict(present=present, powered=powered, soft=soft, hard=hard)


#: Longest BlueZ sentence we keep. Long enough for
#: "Failed to set power on: org.bluez.Error.Blocked", short enough that a
#: surprise never becomes a paragraph in an analytics property.
_WAKE_ERROR_MAX = 120

#: bluetoothctl colours its output; the PTY path sees the escapes raw.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _first_failure(text: str) -> str:
    """The first line that reads like a refusal, cleaned and clipped."""
    for line in _ANSI.sub("", text or "").splitlines():
        s = line.strip()
        low = s.lower()
        if "failed" in low or "org.bluez.error" in low or "not available" in low:
            return s[:_WAKE_ERROR_MAX]
    return ""


def _bt_try(*args: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Run bluetoothctl and KEEP the verdict.

    `_bt` passes `check=False` and every caller threw the result away. That is
    how a refusal, a permission error and a five-second hang all came to look
    exactly like success.
    """
    try:
        done = _bt(*args, timeout=timeout)
    except FileNotFoundError:
        return False, "bluetoothctl is not installed"
    except subprocess.TimeoutExpired:
        return False, "bluetoothctl %s timed out after %.0fs" % (" ".join(args), timeout)
    if done.returncode != 0:
        return False, (
            _first_failure(done.stderr) or _first_failure(done.stdout) or "exit %d" % done.returncode
        )
    failure = _first_failure(done.stdout)
    return (not failure), failure


def _power_on_via_pty() -> tuple[bool, str]:
    """`power on` TYPED INTO bluetoothctl rather than piped at it.

    `_pair_via_agent` already had to do this: a bluetoothctl with no controlling
    terminal behaves differently, and on some BlueZ builds it exits before its
    D-Bus call lands. Pairing got the PTY treatment; `power on` was left on the
    piped path. That is the difference between an operator typing the command
    once and it working, and the agent running the same command dozens of times
    with no effect.
    """
    try:
        pid, fd = pty.fork()
    except OSError as exc:
        return False, ("cannot fork a pty: %s" % exc)[:_WAKE_ERROR_MAX]
    if pid == 0:  # pragma: no cover - replaced by exec
        os.execvp("bluetoothctl", ["bluetoothctl"])
    text = ""
    try:
        _pty_read(fd, 1.5)
        text += _pty_send(fd, "power on", wait=2.0)
        text += _pty_send(fd, "pairable on", wait=1.0)
        text += _pty_send(fd, "show", wait=1.5)
        os.write(fd, b"quit\n")
        _pty_read(fd, 0.5)
    except OSError as exc:
        return False, ("pty write failed: %s" % exc)[:_WAKE_ERROR_MAX]
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
    clean = _ANSI.sub("", text).lower()
    worked = "powered: yes" in clean or "changing power on succeeded" in clean
    return worked, _first_failure(text)


def wake_adapter() -> dict:
    """Wake the radio, verify it, and say which step worked.

    Called from exactly one place: `POST /api/pad/adapter/on`, which is the
    operator pressing Turn on. It is deliberately not wired into scan or pair.
    A radio that powers itself up behind the operator cannot be reported to
    them, and the half hour that produced this file was spent on a screen that
    was quietly retrying something it never mentioned.

    Three things can hold the adapter down and the old code could not tell them
    apart, because it ran one command, discarded the result and returned None:

        rfkill soft block   `power on` cannot clear one
        headless invocation a piped bluetoothctl is not an interactive one
        genuinely absent    no Bluetooth on this board at all

    So try them in cost order, re-read the adapter after each, and stop at the
    first that works. `wokeVia` then names the cause in the field data without
    anyone having to reproduce it, and `wakeError` carries BlueZ's own sentence
    when nothing worked at all.
    """
    state = adapter_state()
    if state["reason"] is ADAPTER_OK:
        return dict(state, wokeVia="already_on")
    if state["reason"] == ADAPTER_HARD_BLOCKED:
        # A physical kill switch. Nothing we run here can clear it, and
        # pretending otherwise would just cost the operator another 30 minutes.
        return dict(state, wakeError="bluetooth is hard blocked (physical switch)")

    error = ""

    # 1. rfkill. Cheapest, and the one thing `power on` provably cannot do.
    if state["blocked"]:
        if _rfkill_unblock():
            time.sleep(0.5)
            state = adapter_state()
            if state["reason"] is ADAPTER_OK:
                return dict(state, wokeVia="rfkill")
        else:
            error = "rfkill unblock was refused (needs root)"

    # 2. the piped one-shot -- what the agent has always done.
    ok, message = _bt_try("power", "on", timeout=5.0)
    _bt_try("pairable", "on", timeout=5.0)
    state = adapter_state()
    if state["reason"] is ADAPTER_OK:
        return dict(state, wokeVia="bluetoothctl")
    if not ok:
        error = error or message

    # 3. the PTY, which is how pairing already talks to BlueZ on a headless Pi.
    ok, message = _power_on_via_pty()
    state = adapter_state()
    if state["reason"] is ADAPTER_OK:
        return dict(state, wokeVia="pty")
    error = error or message

    return dict(state, wakeError=(error or None))


def _stop_discovery() -> None:
    """Nearby scan holds Discovering:yes; connect then fails with InProgress."""
    try:
        _bt("scan", "off", timeout=5.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    time.sleep(0.4)


def _parse_devices(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _DEVICE_LINE.match(line.strip())
        if m:
            found.append((m.group(1), m.group(2).strip()))
    return found


def _info_fields(address: str) -> dict:
    try:
        out = _bt("info", address, timeout=6.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {
            "name": "",
            "icon": "",
            "connected": False,
            "paired": False,
            "battery_percent": None,
        }
    name = ""
    icon = ""
    connected = False
    paired = False
    battery_percent: int | None = None
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.startswith("Name:"):
            name = s[5:].strip()
        elif s.startswith("Alias:") and not name:
            name = s[6:].strip()
        elif s.startswith("Icon:"):
            icon = s[5:].strip()
        elif s.startswith("Connected:"):
            connected = "yes" in s.lower()
        elif s.startswith("Paired:") or s.startswith("Bonded:"):
            if "yes" in s.lower():
                paired = True
        elif s.startswith("Battery Percentage:"):
            m = _BATTERY_PCT.search(s)
            if m:
                battery_percent = max(0, min(100, int(m.group(1))))
    return {
        "name": name,
        "icon": icon,
        "connected": connected,
        "paired": paired,
        "battery_percent": battery_percent,
    }


def _device_dict(addr: str, listed_name: str, info: dict) -> dict:
    name = str(info["name"] or listed_name or addr)
    return {
        "address": addr,
        "name": name,
        "connected": bool(info["connected"]),
        "paired": bool(info["paired"]),
        "batteryPercent": info["battery_percent"],
    }


def list_xbox_devices() -> list[dict]:
    """Every Xbox-like device bluetoothctl currently knows about."""
    try:
        listed = _bt("devices")
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []

    devices: list[dict] = []
    seen: set[str] = set()
    for addr, listed_name in _parse_devices(listed.stdout):
        info = _info_fields(addr)
        name = str(info["name"] or listed_name or addr)
        icon = str(info["icon"] or "")
        if not is_controller_device(name=name, icon=icon):
            continue
        devices.append(_device_dict(addr, listed_name, info))
        seen.add(addr.upper())

    # Paired controllers that dropped out of `devices` still belong in My devices.
    try:
        paired_listed = _bt("devices", "Paired")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return devices
    for addr, listed_name in _parse_devices(paired_listed.stdout):
        if addr.upper() in seen:
            continue
        info = _info_fields(addr)
        name = str(info["name"] or listed_name or addr)
        icon = str(info["icon"] or "")
        if not is_controller_device(name=name, icon=icon):
            continue
        info = {**info, "paired": True}
        devices.append(_device_dict(addr, listed_name, info))
    return devices


def _status_from_devices(devices: list[dict], adapter: Optional[dict] = None) -> dict:
    # `adapter` always travels with the device list, including -- especially --
    # when that list is empty. An empty list plus `adapter.reason == "blocked"`
    # is a different sentence from an empty list plus `adapter.reason == None`,
    # and Studio renders them differently.
    state = adapter if adapter is not None else adapter_state()
    if not devices:
        return {
            "present": False,
            "connected": False,
            "name": None,
            "address": None,
            "devices": [],
            "adapter": state,
        }
    connected = next((d for d in devices if d["connected"]), None)
    primary = connected or next((d for d in devices if d["paired"]), None) or devices[0]
    return {
        "present": True,
        "connected": bool(connected),
        "name": primary["name"],
        "address": primary["address"],
        "devices": devices,
        "adapter": state,
    }


def pad_status(adapter: Optional[dict] = None) -> dict:
    """`{present, connected, name, address, devices, adapter}` as the Pi sees it.

    Pass `adapter` when the caller has already read the radio this request, so
    one call costs one `bluetoothctl show` rather than two.
    """
    return _status_from_devices(list_xbox_devices(), adapter)


def scan_pad(timeout: float = _DEFAULT_SCAN_S) -> dict:
    """Look for controllers (README: bluetoothctl scan on), then return the list."""
    adapter = adapter_state()
    if adapter["reason"] is not ADAPTER_OK:
        # Eighteen seconds spent scanning a radio that is off finds exactly
        # nothing, and to the operator that is indistinguishable from a
        # controller which refuses to pair. Answer now, and say what is wrong.
        return _status_from_devices([], adapter)
    try:
        # --timeout runs scan for N seconds then exits; plain "scan on" never returns.
        # Xbox needs longer than a few seconds for the Name: to land (otherwise it
        # only appears as a MAC and we filter it out).
        _bt("--timeout", str(int(timeout)), "scan", "on", timeout=timeout + 6.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return pad_status(adapter)


def _pty_read(fd: int, timeout: float) -> str:
    buf = b""
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], max(0.0, end - time.time()))
        if not r:
            break
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        # Drain bursts that arrive right after the first read.
        end = max(end, time.time() + 0.12)
    return buf.decode("utf-8", errors="replace")


def _pty_send(fd: int, cmd: str, wait: float = 0.8) -> str:
    os.write(fd, (cmd + "\n").encode())
    time.sleep(0.15)
    return _pty_read(fd, wait)


def _pair_via_agent(address: str) -> None:
    """Interactive bluetoothctl: agent → pair (auto-yes) → trust → connect."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("bluetoothctl", ["bluetoothctl"])

    try:
        _pty_read(fd, 1.5)
        for cmd in ("power on", "pairable on", "agent DisplayYesNo", "default-agent"):
            _pty_send(fd, cmd, wait=1.0)

        os.write(fd, f"pair {address}\n".encode())
        deadline = time.time() + 28.0
        while time.time() < deadline:
            chunk = _pty_read(fd, 1.0)
            if not chunk:
                continue
            low = chunk.lower()
            if "confirm passkey" in low or ("confirm" in low and "passkey" in low):
                os.write(fd, b"yes\n")
            if (
                "pairing successful" in low
                or "already exists" in low
                or "failed to pair" in low
                or "authenticationfailed" in low
                or "not available" in low
            ):
                break

        _pty_send(fd, f"trust {address}", wait=2.0)
        _pty_send(fd, f"connect {address}", wait=8.0)
        os.write(fd, b"quit\n")
        _pty_read(fd, 1.0)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def _wait_for_joystick(timeout: float = _JOYSTICK_WAIT_S) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if joystick_present():
            return True
        time.sleep(0.25)
    return joystick_present()


def _resolve_address(
    address: Optional[str], adapter: Optional[dict] = None
) -> tuple[Optional[str], dict]:
    if address and not valid_address(address):
        raise ValueError(f"invalid pad address: {address}")
    status = pad_status(adapter)
    if address:
        return address, status
    if not status["present"]:
        scanned = scan_pad()
        if not scanned["present"]:
            return None, scanned
        status = scanned
    devices = status.get("devices") or []
    if not devices:
        return None, status
    return devices[0]["address"], status


def _connect_bonded(address: str) -> None:
    """Reconnect a My-devices controller. Retries after stopping discovery."""
    _bt("trust", address, timeout=8.0)
    for _ in range(4):
        _bt("connect", address, timeout=12.0)
        if _info_fields(address).get("connected"):
            return
        time.sleep(0.8)


def pair_pad(address: Optional[str] = None) -> dict:
    """pair → trust → connect for one controller (README steps), with a BlueZ agent."""
    adapter = adapter_state()
    if adapter["reason"] is not ADAPTER_OK:
        return _status_from_devices([], adapter)
    # Studio's Nearby poll leaves the adapter discovering; BlueZ then rejects connect.
    _stop_discovery()
    address, status = _resolve_address(address, adapter)
    if not address:
        return status
    info = _info_fields(address)
    # Already linked — do not re-run pair (that is what confuses a bonded pad).
    if info.get("connected"):
        _wait_for_joystick()
        return pad_status(adapter)
    try:
        if info.get("paired"):
            _connect_bonded(address)
        else:
            _pair_via_agent(address)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return pad_status(adapter)
    # Bond can succeed a moment before uhid/js0 appears; Drive needs the stick.
    status = pad_status(adapter)
    if status.get("connected"):
        _wait_for_joystick()
        return pad_status(adapter)
    return status


def disconnect_pad(address: Optional[str] = None) -> dict:
    """Drop the HID link; keep the bond so it stays under My devices."""
    adapter = adapter_state()
    if adapter["reason"] is not ADAPTER_OK:
        return _status_from_devices([], adapter)
    _stop_discovery()
    address, status = _resolve_address(address, adapter)
    if not address:
        return status
    try:
        _bt("disconnect", address, timeout=10.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Give BlueZ a beat so Connected:no lands before the next status read.
    time.sleep(0.4)
    return pad_status(adapter)


def forget_pad(address: Optional[str] = None) -> dict:
    """Remove the bond (bluetoothctl remove). Device leaves My devices."""
    adapter = adapter_state()
    if adapter["reason"] is not ADAPTER_OK:
        return _status_from_devices([], adapter)
    _stop_discovery()
    address, status = _resolve_address(address, adapter)
    if not address:
        return status
    try:
        _bt("disconnect", address, timeout=8.0)
        _bt("remove", address, timeout=10.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    time.sleep(0.3)
    return pad_status(adapter)
