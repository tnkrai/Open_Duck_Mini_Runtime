"""Pre-walk hardware checks: joints, calibration offsets, IMU, feet switches.

This implements the open TODO at the bottom of ``checklist.md``::

    TODO (Antoine)
    Make a script that goes through all this automatically
    - joints positions and offsets
    - imu orientation
    - feet switches

Why it matters more once custom policies exist: with an unfamiliar policy AND an
uncalibrated joint, the operator has two suspects and no way to separate them. Preflight
removes one, so a policy is never blamed for a mechanical fault.

Design notes
------------
**Collaborators are injected, never constructed.** ``run_preflight`` takes the HWI, IMU,
feet-switch and config objects it needs. That is what makes it testable without hardware
(the suite's convention -- see ``tests/test_hwi_adapter.py``), and it keeps bus ownership
the caller's problem: the walk subprocess owns the serial bus while it runs, so only
``tnkr_server`` knows whether a read is currently legal.

**Joints are read one at a time, not via ``get_present_positions()``.** That method
returns ``None`` for the entire read if any single servo fails
(``rustypot_position_hwi.py:236``), so it cannot say *which* joint is dead -- and "right
knee is not responding" is the whole point. Reading per-servo through ``_io_retry`` also
inherits its retry-and-name behaviour for free.

**It reads. It never writes.** No offsets are set, no joint is moved, no torque is
enabled. A check that changes the thing it checks is not a check.

Each result carries two strings, per ``tnkr-studio/app/DESIGN.md#errors``: ``operator`` is
one short sentence for a person, ``detail`` is for the log.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

# Anything beyond this from level, at rest, means the IMU is not mounted or configured the
# way the config claims. Generous: we are catching an inverted axis, not a wonky table.
MAX_RESTING_TILT_DEG: float = 30.0

# A joint reporting a position this far from zero is not plausibly a calibrated duck at
# rest -- either the offset is wrong or the servo is reporting nonsense.
MAX_PLAUSIBLE_POSITION_RAD: float = 4.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str  # developer-facing: ids, values, exceptions
    operator: str  # one sentence, < 100 chars, no diagnostics

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "operator": self.operator,
        }


@dataclass
class PreflightReport:
    ok: bool
    checks: list[CheckResult] = field(default_factory=list)
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [c.as_dict() for c in self.checks],
            "duration_ms": self.duration_ms,
        }


def _label(joint_name: str) -> str:
    """'right_hip_pitch' -> 'Right hip pitch', for operator copy."""
    return joint_name.replace("_", " ").capitalize()


# ── The checks ──────────────────────────────────────────────────────────────────


def check_joints(hwi) -> CheckResult:
    """Every joint answers a position read, with a plausible value.

    Reads per-servo so a failure names the joint. ``_io_retry`` raises with the joint name
    and servo id already in the message.
    """
    dead: list[str] = []
    implausible: list[tuple[str, float]] = []
    errors: list[str] = []

    for name, servo_id in hwi.joints.items():
        try:
            pos = hwi._io_retry(
                lambda i=servo_id: hwi.io.read_present_position([i])[0],
                name,
                "read_present_position",
            )
        except Exception as exc:  # OSError from _io_retry, or anything the io raises
            dead.append(name)
            errors.append(f"{name}(id {servo_id}): {exc}")
            continue

        if pos is None or not math.isfinite(pos):
            dead.append(name)
            errors.append(f"{name}(id {servo_id}): read returned {pos!r}")
        elif abs(pos) > MAX_PLAUSIBLE_POSITION_RAD:
            implausible.append((name, pos))

    if dead:
        first = dead[0]
        return CheckResult(
            "joints",
            False,
            f"{len(dead)}/{len(hwi.joints)} joints did not respond: " + "; ".join(errors),
            f"{_label(first)} is not responding."
            if len(dead) == 1
            else f"{len(dead)} joints are not responding.",
        )

    if implausible:
        name, pos = implausible[0]
        return CheckResult(
            "joints",
            False,
            "implausible positions: "
            + ", ".join(f"{n}={p:.3f}rad" for n, p in implausible),
            f"{_label(name)} is reporting an impossible position.",
        )

    return CheckResult(
        "joints", True, f"{len(hwi.joints)}/{len(hwi.joints)} joints responded", ""
    )


def check_offsets(hwi, config) -> CheckResult:
    """Every joint has a calibration offset recorded.

    A missing offset is a failure, not a warning: the walk loop subtracts it on every
    read (``rustypot_position_hwi.py:235``), so an absent one silently shifts that joint's
    entire observation and the policy walks on bad data.
    """
    offsets = getattr(config, "joints_offset", None)
    if not isinstance(offsets, dict):
        return CheckResult(
            "offsets",
            False,
            f"joints_offset is {type(offsets).__name__}, expected dict",
            "This robot has no calibration saved.",
        )

    missing = [name for name in hwi.joints if name not in offsets]
    if missing:
        return CheckResult(
            "offsets",
            False,
            f"{len(missing)}/{len(hwi.joints)} offsets missing: {', '.join(missing)}",
            f"{_label(missing[0])} is not calibrated."
            if len(missing) == 1
            else f"{len(missing)} joints are not calibrated.",
        )

    bad = [n for n in hwi.joints if not isinstance(offsets[n], (int, float))]
    if bad:
        return CheckResult(
            "offsets",
            False,
            f"non-numeric offsets: {', '.join(bad)}",
            f"{_label(bad[0])} has a bad calibration value.",
        )

    if getattr(config, "default", False):
        return CheckResult(
            "offsets",
            False,
            "duck_config fell back to defaults, so every offset is 0.0 -- "
            "no calibration file was loaded",
            "This robot has no calibration saved.",
        )

    return CheckResult(
        "offsets", True, f"{len(hwi.joints)}/{len(hwi.joints)} offsets present", ""
    )


def _tilt_deg(quaternion) -> float | None:
    """Angle between the duck's own up-axis and world up, in degrees.

    Returns None when the quaternion is unusable. ``raw_imu`` hands back identity
    ``[1,0,0,0]`` before its first real reading, which is genuinely level, so identity is
    treated as valid rather than as missing data.
    """
    if quaternion is None or len(quaternion) != 4:
        return None
    w, x, y, z = (float(v) for v in quaternion)

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-6 or not math.isfinite(norm):
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # World-frame z component of the body z-axis. 1.0 == perfectly upright.
    cos_tilt = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.degrees(math.acos(cos_tilt))


def check_imu(imu, config) -> CheckResult:
    """The IMU responds, its quaternion is usable, and the duck reads as upright.

    An inverted ``imu_upside_down`` flag is a real, previously-observed failure in this
    repo (``imu-debug-logs.md``): the gravity cue the policy sees is inverted and the duck
    walks as if the floor were the ceiling. It is a hard fail, not a warning, because it
    looks exactly like a bad policy from the outside.
    """
    try:
        data = imu.get_data()
    except Exception as exc:
        return CheckResult(
            "imu", False, f"get_data raised: {exc}", "The orientation sensor is not responding."
        )

    if not data:
        return CheckResult(
            "imu", False, f"get_data returned {data!r}", "The orientation sensor is not responding."
        )

    tilt = _tilt_deg(data.get("quaternion"))
    if tilt is None:
        return CheckResult(
            "imu",
            False,
            f"unusable quaternion: {data.get('quaternion')!r}",
            "The orientation sensor is not responding.",
        )

    if tilt > MAX_RESTING_TILT_DEG:
        upside_down = bool(getattr(config, "imu_upside_down", False))
        return CheckResult(
            "imu",
            False,
            f"resting tilt {tilt:.1f}deg exceeds {MAX_RESTING_TILT_DEG:.0f}deg "
            f"(imu_upside_down={upside_down}) -- sensor is mis-mounted, the config flag "
            f"is wrong, or the duck is not on a level surface",
            "The robot does not think it is upright.",
        )

    return CheckResult("imu", True, f"upright, resting tilt {tilt:.1f}deg", "")


def check_feet(feet) -> CheckResult:
    """Both switches read, and report a state.

    Stuck-closed is not detectable from a single sample without knowing whether the duck
    is being held up, so this checks readability and shape only. Naming that limit is
    better than a check that guesses.

    ``feet=None`` means the caller could not construct the reader at all -- no GPIO,
    which is the normal case off-Pi. That is reported as a failure rather than skipped,
    because a walk without foot contacts feeds the policy two constant zeros where it
    expects real switch state.
    """
    if feet is None:
        return CheckResult(
            "feet",
            False,
            "foot-switch reader unavailable (no GPIO on this machine)",
            "The foot sensors are not available.",
        )

    try:
        contacts = feet.get()
    except Exception as exc:
        return CheckResult(
            "feet", False, f"get raised: {exc}", "The foot sensors are not responding."
        )

    if contacts is None or len(contacts) != 2:
        return CheckResult(
            "feet",
            False,
            f"expected 2 contacts, got {contacts!r}",
            "The foot sensors are not responding.",
        )

    left, right = (bool(c) for c in contacts)
    return CheckResult(
        "feet",
        True,
        f"left {'closed' if left else 'open'}, right {'closed' if right else 'open'}",
        "",
    )


# ── Orchestration ───────────────────────────────────────────────────────────────


def run_preflight(hwi, imu, feet, config) -> PreflightReport:
    """Run every check. Reports; never decides whether a walk may start.

    Each check is independent and all of them run, so one dead servo does not hide an
    inverted IMU -- an operator fixing faults one round trip at a time is a worse
    experience than one list.
    """
    started = time.monotonic()
    checks = [
        check_joints(hwi),
        check_offsets(hwi, config),
        check_imu(imu, config),
        check_feet(feet),
    ]
    return PreflightReport(
        ok=all(c.ok for c in checks),
        checks=checks,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
