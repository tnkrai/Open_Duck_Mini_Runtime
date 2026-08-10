"""Live joint trim — move joints_offsets while the gait is running.

The stance wizard trims a *standing* duck, but a sagging hip or a toed-in ankle
only shows itself in the gait, so tuning it there otherwise costs a
stop/adjust/start cycle per guess. This is the shared file that removes the
cycle: the server writes trim targets as the user nudges sliders, and the walk
reads them each control tick and slews toward them.

Direction of flow: the server owns this file, the walk only reads it (the
mirror of walk_telemetry, which the walk owns and the server reads). That's
what makes delta mode correct — a nudge adds to the last *commanded* value, so
two quick taps land as two increments instead of one getting lost because the
ramp hadn't arrived yet.

Targets are clamped by the caller to the saved trim ± TRIM_LIMIT_RAD, so a
whole session of nudging can't walk a joint somewhere it can't come back from.
The file lives in RAM and dies with the walk — /api/walk/offsets/save is what
promotes a trim into duck_config.json.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time

# tmpfs on the Pi (RAM — no SD-card wear); tempdir off-Pi so the code stays
# importable/testable on dev machines without /dev/shm. Mirrors walk_telemetry.
OFFSETS_FILE = (
    "/dev/shm/tnkr_walk_offsets.json"
    if os.path.isdir("/dev/shm")
    else os.path.join(tempfile.gettempdir(), "tnkr_walk_offsets.json")
)

# How far a live trim may stray from the saved baseline, per joint (~8.6°).
# Enough to pull out a visible sag; short of the range where a mid-gait change
# stops being a trim and starts being a different stance.
TRIM_LIMIT_RAD = 0.15

# How fast the walk slews to a new target. At 0.25 rad/s a full-limit move
# takes ~0.6s — fast enough to feel responsive on a slider, slow enough that
# the step in progress absorbs it instead of the duck lurching.
RAMP_RATE_RAD_S = 0.25


def write(offsets: dict[str, float]) -> None:
    """Publish trim targets. Temp + atomic rename, so the walk reading mid-tick
    sees one complete set of offsets — never a mix of old and new joints."""
    data = {"timestamp": time.time(), "offsets": {k: float(v) for k, v in offsets.items()}}
    tmp = OFFSETS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, OFFSETS_FILE)


def read() -> dict[str, float] | None:
    """The current trim targets, or None if no live trim is published.

    No freshness window: unlike a telemetry pose, a trim stays in effect until
    it's changed or the walk ends, so an old timestamp is still the truth. The
    walk's start and stop paths call clear(), which is what keeps one session's
    experiment out of the next one.

    Non-finite values are dropped rather than returned — inf/NaN survive JSON
    and would otherwise reach a goal position as garbage.
    """
    try:
        with open(OFFSETS_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    raw = data.get("offsets")
    if not isinstance(raw, dict):
        return None
    out = {}
    for name, value in raw.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out[name] = value
    return out


def ramp_toward(
    current: dict[str, float],
    target: dict[str, float],
    dt: float,
    rate: float = RAMP_RATE_RAD_S,
) -> dict[str, float]:
    """One tick of slew from `current` toward `target`, capped at `rate` rad/s.

    Lives here so the walk loop stays a one-liner and the ramp math is testable
    without a robot. Joints absent from `target` hold their current value — a
    partial trim must not drag every other joint to zero.
    """
    step = abs(rate) * dt
    out = dict(current)
    for name, goal in target.items():
        now = current.get(name, goal)
        delta = goal - now
        out[name] = goal if abs(delta) <= step else now + math.copysign(step, delta)
    return out


def clear() -> None:
    try:
        os.remove(OFFSETS_FILE)
    except FileNotFoundError:
        pass
