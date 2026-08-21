"""The walk→server telemetry file — the steering pattern, pointed the other way.

While the walk subprocess owns the servo bus + IMU, the server can't read
hardware — but the walk loop already reads every joint at 50 Hz for its policy
observations. So the runner drops each tick's pose into a one-slot shared-memory
file (temp + atomic rename, latest wins, never grows) and the server's
/api/state serves that instead of a stale pre-walk cache. Mirror of the
/dev/shm command file the walk already reads for steering.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

# tmpfs on the Pi (RAM — no SD-card wear at 50 writes/s); tempdir off-Pi so the
# code stays importable/testable on dev machines without /dev/shm.
TELEMETRY_FILE = (
    "/dev/shm/tnkr_walk_telemetry.json"
    if os.path.isdir("/dev/shm")
    else os.path.join(tempfile.gettempdir(), "tnkr_walk_telemetry.json")
)

# Why the abort reason is a SECOND file rather than a key in the snapshot above:
# the snapshot is deliberately deleted on the way out (clear(), so a dead walk's
# pose is never served as live), and the abort reason has to OUTLIVE the process
# that wrote it — the server only asks "why did it stop?" after the exit. So this
# one is latched, like walk_pause: it stays until someone clears it, and the walk
# clears it at startup so a previous abort can never be read as this walk's.
ABORT_FILE = (
    "/dev/shm/tnkr_walk_abort.json"
    if os.path.isdir("/dev/shm")
    else os.path.join(tempfile.gettempdir(), "tnkr_walk_abort.json")
)

# Older than this and the snapshot is a dead walk's leftovers, not a live pose.
FRESH_S = 1.0


def write_snapshot(
    joints: dict[str, float],
    imu: dict | None = None,
    envelope: dict | None = None,
) -> None:
    """Publish one tick's pose.

    ``envelope`` is the safety envelope's aggregated clamp counts, and it is omitted
    from the file entirely when absent — a built-in-policy walk writes exactly the
    bytes it wrote before the envelope existed (amendment A8).
    """
    data = {"timestamp": time.time(), "joints": joints, "imu": imu}
    if envelope is not None:
        data["envelope"] = envelope
    tmp = TELEMETRY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, TELEMETRY_FILE)


def read_snapshot(max_age_s: float = FRESH_S) -> dict | None:
    """The latest snapshot, or None if missing/unreadable/stale — the caller falls
    back to its cached pose rather than serving a dead snapshot as live."""
    try:
        with open(TELEMETRY_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if time.time() - float(data.get("timestamp", 0)) > max_age_s:
        return None
    return data


def clear() -> None:
    try:
        os.remove(TELEMETRY_FILE)
    except FileNotFoundError:
        pass


def write_abort(code: str, reason: str, detail: str, operator: str) -> None:
    """Record why a guard stopped the walk, before the process exits.

    Without this the server can only report that the walk died, which is the same
    thing it reports for a crash, an OOM kill and a power cut. ``code`` is the
    machine-readable reason (``POLICY_ABORTED``), ``operator`` the one sentence a
    person sees, ``detail`` the tick counts and milliseconds for the log.
    """
    tmp = ABORT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "code": code,
                "reason": reason,
                "detail": detail,
                "operator": operator,
            },
            f,
        )
    os.replace(tmp, ABORT_FILE)


def read_abort() -> dict | None:
    """The last recorded abort, or None. Latched: no freshness window, because the
    reader is asking about a process that has already exited."""
    try:
        with open(ABORT_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def clear_abort() -> None:
    try:
        os.remove(ABORT_FILE)
    except FileNotFoundError:
        pass
