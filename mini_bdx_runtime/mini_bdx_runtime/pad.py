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


def _ensure_adapter_on() -> None:
    """Pi images often leave the adapter powered off / blocked until something wakes it."""
    try:
        _bt("power", "on", timeout=5.0)
        _bt("pairable", "on", timeout=5.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


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


def _status_from_devices(devices: list[dict]) -> dict:
    if not devices:
        return {"present": False, "connected": False, "name": None, "address": None, "devices": []}
    connected = next((d for d in devices if d["connected"]), None)
    primary = connected or next((d for d in devices if d["paired"]), None) or devices[0]
    return {
        "present": True,
        "connected": bool(connected),
        "name": primary["name"],
        "address": primary["address"],
        "devices": devices,
    }


def pad_status() -> dict:
    """`{present, connected, name, address, devices}` for Xbox controllers the Pi sees."""
    return _status_from_devices(list_xbox_devices())


def scan_pad(timeout: float = _DEFAULT_SCAN_S) -> dict:
    """Look for controllers (README: bluetoothctl scan on), then return the list."""
    _ensure_adapter_on()
    try:
        # --timeout runs scan for N seconds then exits; plain "scan on" never returns.
        # Xbox needs longer than a few seconds for the Name: to land (otherwise it
        # only appears as a MAC and we filter it out).
        _bt("--timeout", str(int(timeout)), "scan", "on", timeout=timeout + 6.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return pad_status()


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


def _resolve_address(address: Optional[str]) -> tuple[Optional[str], dict]:
    if address and not valid_address(address):
        raise ValueError(f"invalid pad address: {address}")
    status = pad_status()
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
    _ensure_adapter_on()
    # Studio's Nearby poll leaves the adapter discovering; BlueZ then rejects connect.
    _stop_discovery()
    address, status = _resolve_address(address)
    if not address:
        return status
    info = _info_fields(address)
    # Already linked — do not re-run pair (that is what confuses a bonded pad).
    if info.get("connected"):
        _wait_for_joystick()
        return pad_status()
    try:
        if info.get("paired"):
            _connect_bonded(address)
        else:
            _pair_via_agent(address)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return pad_status()
    # Bond can succeed a moment before uhid/js0 appears; Drive needs the stick.
    status = pad_status()
    if status.get("connected"):
        _wait_for_joystick()
        return pad_status()
    return status


def disconnect_pad(address: Optional[str] = None) -> dict:
    """Drop the HID link; keep the bond so it stays under My devices."""
    _ensure_adapter_on()
    _stop_discovery()
    address, status = _resolve_address(address)
    if not address:
        return status
    try:
        _bt("disconnect", address, timeout=10.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Give BlueZ a beat so Connected:no lands before the next status read.
    time.sleep(0.4)
    return pad_status()


def forget_pad(address: Optional[str] = None) -> dict:
    """Remove the bond (bluetoothctl remove). Device leaves My devices."""
    _ensure_adapter_on()
    _stop_discovery()
    address, status = _resolve_address(address)
    if not address:
        return status
    try:
        _bt("disconnect", address, timeout=8.0)
        _bt("remove", address, timeout=10.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    time.sleep(0.3)
    return pad_status()
