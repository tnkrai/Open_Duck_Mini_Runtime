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

# Older than this and the snapshot is a dead walk's leftovers, not a live pose.
FRESH_S = 1.0


def write_snapshot(joints: dict[str, float], imu: dict | None = None) -> None:
    data = {"timestamp": time.time(), "joints": joints, "imu": imu}
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
