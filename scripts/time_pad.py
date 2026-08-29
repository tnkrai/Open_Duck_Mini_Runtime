#!/usr/bin/env python3
"""Time every bluetoothctl call the agent makes, on connect and on disconnect.

Run ON THE DUCK. Nothing here changes any code: it wraps pad._bt with a stopwatch,
runs the two real paths against the real radio, and prints where the seconds went.

    python3 scripts/time_pad.py                # uses the first bonded controller
    python3 scripts/time_pad.py AA:BB:CC:DD:EE:FF

The question it answers: connecting feels fast and disconnecting feels slow, yet
connect issues 16 bluetoothctl calls and disconnect issues 10. So it is not the
number of calls. One call is blocking, and this says which.
"""

import sys
import time

sys.path.insert(0, "/home/%s/Open_Duck_Mini_Runtime/mini_bdx_runtime" % __import__("getpass").getuser())

try:
    from mini_bdx_runtime import pad
except ImportError:
    sys.path.insert(0, "mini_bdx_runtime")
    from mini_bdx_runtime import pad

calls = []
_real_bt = pad._bt
_real_rfkill = pad._rfkill
_real_sleep = time.sleep


def timed_bt(*args, timeout=12.0):
    began = time.monotonic()
    try:
        return _real_bt(*args, timeout=timeout)
    finally:
        calls.append((" ".join(args), time.monotonic() - began))


def timed_rfkill(*args, timeout=5.0):
    began = time.monotonic()
    try:
        return _real_rfkill(*args, timeout=timeout)
    finally:
        calls.append(("rfkill " + " ".join(args), time.monotonic() - began))


def timed_sleep(seconds):
    calls.append(("sleep(%.1f)" % seconds, seconds))
    _real_sleep(seconds)


pad._bt = timed_bt
pad._rfkill = timed_rfkill
pad.time.sleep = timed_sleep


def run(label, fn):
    calls.clear()
    began = time.monotonic()
    status = fn()
    total = time.monotonic() - began
    print("\n=== %s: %.2fs wall ===" % (label, total))
    for i, (name, seconds) in enumerate(calls, 1):
        bar = "#" * min(60, int(seconds * 20))
        flag = "   <-- THIS ONE" if seconds > 1.0 else ""
        print("  %2d. %6.2fs %-34s %s%s" % (i, seconds, name, bar, flag))
    accounted = sum(s for _, s in calls)
    print("  accounted %.2fs of %.2fs (%.2fs elsewhere)" % (accounted, total, total - accounted))
    return status, total


address = sys.argv[1] if len(sys.argv) > 1 else None
if not address:
    devices = pad.pad_status().get("devices") or []
    if not devices:
        sys.exit("no bonded controller found -- pair one first, or pass an address")
    address = devices[0]["address"]
    calls.clear()
print("controller: %s" % address)

connected = pad._info_fields(address).get("connected")
calls.clear()
print("currently connected: %s" % connected)

if connected:
    _, t_off = run("DISCONNECT", lambda: pad.disconnect_pad(address))
    _real_sleep(2.0)
    _, t_on = run("CONNECT", lambda: pad.pair_pad(address))
else:
    _, t_on = run("CONNECT", lambda: pad.pair_pad(address))
    _real_sleep(2.0)
    _, t_off = run("DISCONNECT", lambda: pad.disconnect_pad(address))

print("\n" + "=" * 62)
print("connect %.2fs   disconnect %.2fs" % (t_on, t_off))
print("Any line flagged THIS ONE is where the wait actually is.")
