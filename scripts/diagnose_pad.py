#!/usr/bin/env python3
"""Why is the Bluetooth radio off, and why did the agent not fix it?

Run this ON THE DUCK, over SSH:

    python3 scripts/diagnose_pad.py

It answers one question the shipped code cannot: an operator reported that
`bluetoothctl power on` typed into their own shell worked first try, while the
agent had already run the same command 47 times without effect. Something about
HOW the agent runs it is different, and the agent threw away every clue --
`_bt()` passes `check=False` and `_ensure_adapter_on()` never read a returncode,
so a refusal, a permission error and a timeout all looked identical to success.

This prints the returncode, stdout and stderr the agent never looked at, for
each candidate cause:

  1. rfkill soft/hard block   -- `power on` cannot clear one
  2. D-Bus / group permission -- the service user may not manage the adapter
  3. headless bluetoothctl    -- no TTY, stdout a pipe, which is exactly how
                                 subprocess.run() invokes it and NOT how a
                                 human types it

Nothing here changes state except the explicitly-labelled WAKE section, which
does what the agent does. Read the output top to bottom; the first section that
reports a non-zero returncode or a block is the answer.
"""

from __future__ import annotations

import grp
import os
import pty
import pwd
import select
import shutil
import subprocess
import sys
import time


def hr(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def run(argv: list[str], timeout: float = 8.0) -> None:
    """Exactly how the agent shells out: no TTY, stdout and stderr are pipes."""
    print("\n$ " + " ".join(argv))
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError:
        print("  !! not installed")
        return
    except subprocess.TimeoutExpired:
        # The shipped _ensure_adapter_on swallows precisely this and returns
        # None, which the caller reads as "the radio is fine".
        print("  !! TIMED OUT after %ss  <-- the agent silently ignores this" % timeout)
        return
    print("  returncode: %d%s" % (done.returncode, "" if done.returncode == 0 else "   <-- NON-ZERO"))
    for label, stream in (("stdout", done.stdout), ("stderr", done.stderr)):
        text = (stream or "").strip()
        if text:
            for line in text.splitlines():
                print("  %s: %s" % (label, line))


def run_pty(commands: list[str], settle: float = 1.0) -> None:
    """The same commands typed into an interactive bluetoothctl, over a PTY.

    This is how the OPERATOR ran them, and how pad.py's `_pair_via_agent`
    already drives pairing. If this works and the piped calls above do not,
    the bug is headless invocation, not the radio.
    """
    print("\n$ bluetoothctl (interactive, over a pty)")
    print("  typing: " + " ; ".join(commands))
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("bluetoothctl", ["bluetoothctl"])
    out = []
    try:
        time.sleep(settle)
        for cmd in commands:
            os.write(fd, (cmd + "\n").encode())
            deadline = time.time() + 1.5
            while time.time() < deadline:
                r, _, _ = select.select([fd], [], [], 0.2)
                if not r:
                    continue
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                out.append(chunk.decode(errors="replace"))
        os.write(fd, b"quit\n")
        time.sleep(0.4)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
    text = "".join(out)
    for line in text.splitlines():
        s = line.strip()
        # keep the lines that carry a verdict; drop the banner and the prompts
        if any(k in s for k in ("Failed", "Powered", "succeeded", "Error", "not available")):
            print("  | " + s)


hr("0. WHO AND WHERE")
user = pwd.getpwuid(os.getuid()).pw_name
groups = sorted(g.gr_name for g in grp.getgrall() if user in g.gr_mem)
print("user            : %s (uid %d)" % (user, os.getuid()))
print("groups          : %s" % (", ".join(groups) or "(none)"))
print("in 'bluetooth'  : %s" % ("bluetooth" in groups))
print("has a tty       : %s   <-- the agent runs with False" % sys.stdin.isatty())
print("bluetoothctl    : %s" % (shutil.which("bluetoothctl") or "NOT FOUND"))
print("rfkill          : %s" % (shutil.which("rfkill") or "NOT FOUND"))

hr("1. IS THE RADIO BLOCKED?  (power on cannot clear a block)")
run(["rfkill", "list"])

hr("2. IS bluetoothd EVEN RUNNING?")
run(["systemctl", "is-active", "bluetooth"])
run(["systemctl", "is-enabled", "bluetooth"])

hr("3. WHAT DOES THE ADAPTER SAY RIGHT NOW?")
run(["bluetoothctl", "show"])
run(["hciconfig", "-a"], timeout=6.0)

hr("4. WAKE IT THE WAY THE AGENT DOES  (piped, no tty)")
print("This is the exact call _ensure_adapter_on() makes, with the returncode")
print("and stderr it discards. If this is where it breaks, this is the bug.")
run(["bluetoothctl", "power", "on"], timeout=5.0)
run(["bluetoothctl", "show"])

hr("5. WAKE IT THE WAY THE OPERATOR DID  (interactive, over a pty)")
run_pty(["power on", "show"])

hr("6. CAN IT SEE ANYTHING?  (10s scan, piped -- put the pad in pairing mode)")
run(["bluetoothctl", "--timeout", "10", "scan", "on"], timeout=18.0)
run(["bluetoothctl", "devices"])

hr("VERDICT")
print("Read back up. In order:")
print("  section 1 shows 'Soft blocked: yes'  -> rfkill; needs `rfkill unblock bluetooth`")
print("  section 2 shows inactive             -> bluetoothd is down; nothing else matters")
print("  section 4 non-zero or TIMED OUT")
print("     but section 5 prints 'Powered: yes' -> headless bluetoothctl is the bug")
print("  section 4 and 5 both fail            -> permission or firmware; send both outputs")
