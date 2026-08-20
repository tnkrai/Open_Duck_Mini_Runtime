"""The safety envelope: bound what a policy commands, and stop when it stops working.

Why this exists
---------------
The built-in policy never needed any of this — it was trusted by construction, one file
that shipped with the robot. A policy a user trained last night or downloaded from Discord
is not, and architecture Decision 1 removed the pre-hardware simulation gate, so the last
line of defence is on the robot. Two halves:

* ``ActionEnvelope`` bounds *where* a joint may go — joint-limit clamp, then velocity clamp.
* ``AbortMonitor`` decides *whether the loop should still be running* — sustained tilt (the
  duck fell and the policy is now commanding a gait into the carpet) or sustained missed
  deadlines (a heavy policy running at 31 Hz on a loop trained for 50 Hz falls over).

Neither half decides *when* it applies. Per amendment **A8** the envelope arms only for a
custom policy, and that ``if`` lives at the single call site in
``scripts/v2_rl_walk_mujoco.py`` — never in here. This module is unconditional and pure so
that it can be unit-tested directly, which is what pays for the gating: the code is
exercised in CI even on the ducks that never load a custom policy.

Design notes
------------
**Nothing here allocates per tick.** Every scratch buffer is preallocated in ``__init__``
and every numpy op writes through ``out=``. ``clamp()`` is called 50 times a second inside
a 20 ms budget that another guard in this same module aborts on; it must not be the reason
the budget is missed.

**Counters, not log lines.** A saturating policy trips a clamp on every tick of every
joint. Printing that would blow the budget on its own, so clamping is counted and the
caller flushes an aggregate at most once a second.

**Aborts fire on sustained conditions, never single samples.** A single missed deadline
happens on any loaded Linux box, and a single tilt spike happens when the duck steps off a
rug. A guard that aborts on one sample is a nuisance, and a nuisance guard gets disabled —
which is strictly worse than not shipping it. Both counters reset on the first good sample.

**Missing IMU data counts toward the tilt abort.** Treating an unreadable orientation as
"upright" is a silent fall (failure mode F4 in the eng review). ``check_tilt(None)`` is not
a no-op — and because the driver never actually hands back ``None`` (it repeats its last
fused quaternion, starting at identity), ``imu_quaternion`` is what turns a stale reading
into the ``None`` the guard was written for. Without it the guard is unreachable code.

See ``tnkr-studio/docs/plans/custom-policy/_architecture.md``, Decision 9 and amendment A8.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from typing import Mapping, Sequence

import numpy as np

from mini_bdx_runtime.policy_contract import CONTROL_HZ

# The documented fallback when a joint has no declared travel limit, matching
# tnkr-studio/server/trlc_studio/robots/openduck_mini/manifest.py:18 ("None -> +/-2 rad").
# Deliberately generous: this is the number that must never *stop* a working policy, only
# stop a joint being driven through its mechanical stop.
DEFAULT_JOINT_LIMIT_RAD: float = 2.0

# ── Abort thresholds ────────────────────────────────────────────────────────────
#
# Defaults, overridable per robot through duck_config.json (see duck_config.py). A missing
# or nonsense key falls back to these rather than disarming the guard.
#
#   tilt_limit_deg        pitch or roll past this is not walking. 60 deg is well past any
#                         gait the reference motion produces and well short of "lying down".
#   tilt_abort_ticks      8 ticks = 0.16 s at 50 Hz. Long enough to ride out a stumble.
#   budget_overrun_ticks  10 ticks = 0.2 s of missed deadlines, i.e. the story's number:
#                         a policy that cannot keep time for a fifth of a second is not
#                         going to start.
DEFAULT_TILT_LIMIT_DEG: float = 60.0
DEFAULT_TILT_ABORT_TICKS: int = 8
DEFAULT_BUDGET_OVERRUN_TICKS: int = 10

# How old a fused-orientation reading may be and still answer the question "is the duck
# upright". 0.2 s is ten samples at 50 Hz — long enough that a couple of dropped reads
# are not an event, short enough that the tilt streak still needs its own 8 ticks on top
# before anything aborts (so ~0.36 s from "the IMU went quiet" to "torque off").
DEFAULT_IMU_MAX_AGE_S: float = 0.2

# The one reason code this module raises. Mirrors the ErrorCode of the same name that
# story 2.5 adds on the Studio side; the runtime emits the code, Studio owns the sentence.
ABORT_CODE: str = "POLICY_ABORTED"


class PolicyAbort(SystemExit):
    """A guard tripped: stop walking, deliberately.

    Subclasses ``SystemExit`` on purpose. ``RLWalk.run``'s ``finally`` block is the only
    teardown that exists — it disables torque, clears the telemetry snapshot and stops the
    expression features — and it already runs on ``SystemExit`` because that is how
    ``handle_sigterm`` ends a walk. Reusing that path means the abort cannot forget a step,
    and there is no second copy of cleanup to drift.

    Carries the same two strings as ``preflight.CheckResult``: ``operator`` is one short
    sentence for a person, ``detail`` is for the log (see ``tnkr-studio/app/DESIGN.md``).

    ``code`` deliberately shadows ``SystemExit.code`` — it holds the reason code, not an
    exit status. That is safe because ``RLWalk.run`` always catches this, so the process
    exits 0 the way a stopped walk does; a walk that aborted did its job and stopped, and
    reporting it as a crash would put it in the same bucket as an OOM kill.
    """

    def __init__(self, reason: str, detail: str, operator: str) -> None:
        super().__init__(0)
        self.code: str = ABORT_CODE
        self.reason: str = reason
        self.detail: str = detail
        self.operator: str = operator

    def __str__(self) -> str:
        return f"{self.code} {self.reason}: {self.detail}"


# ── Arming ──────────────────────────────────────────────────────────────────────

# Set to "1" to arm the envelope for a policy that is NOT custom. For deliberately
# exercising the guards against a policy already known to be good.
FORCE_ENVELOPE_ENV: str = "TNKR_FORCE_ENVELOPE"

# The policies that shipped with the robot, by file name. These are the only files
# trusted by construction (architecture Decision 11): one of them has run on every duck
# sold, which is the whole argument for leaving their code path alone. Antoine published
# exactly two, BEST_WALK_ONNX.onnx and BEST_WALK_ONNX_2.onnx, and this repo ships the
# second in scripts/; anything else is somebody's upload.
BUILTIN_POLICY_FILENAMES: frozenset[str] = frozenset(
    {"BEST_WALK_ONNX.onnx", "BEST_WALK_ONNX_2.onnx"}
)


def is_builtin_policy(onnx_model_path: str | None) -> bool:
    """Whether this path names a policy that shipped with the robot.

    By file name, not by digest: the two published files have no digest anybody wrote
    down, and a name is what the one production caller has. A user who renames a
    downloaded model to ``BEST_WALK_ONNX_2.onnx`` defeats it — and gets the same
    behaviour they get today by overwriting the built-in, which is a deliberate act
    rather than the accident this guard is for.

    A path we know nothing about is **not** the built-in. That is the direction the
    error has to fall: an unrecognised policy running with the guards armed costs a
    clamp that never trips, and the reverse costs a duck.
    """
    if not onnx_model_path:
        return False
    return os.path.basename(str(onnx_model_path)) in BUILTIN_POLICY_FILENAMES


def is_armed(
    custom_policy: bool,
    onnx_model_path: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Whether the envelope applies to this run (amendment A8).

    A predicate, not a branch inside the envelope: the ``if`` still lives at the single
    call site in the walk loop, and this exists so both of its answers can be tested
    without a robot. The envelope itself never asks who loaded the policy.

    **The flag is not trusted on its own, because nothing in production sets it.**
    ``/api/walk/start`` globs ``scripts/*.onnx``, takes the first hit and spawns the walk
    with no ``--custom_policy`` (``tnkr_server.py``), which is exactly how a user runs a
    downloaded policy today: drop the file in ``scripts/`` and press Walk. Trusting the
    flag would leave that run with every guard disarmed and nothing logged — fail-open
    safety with no detection. So provenance comes from the artifact: a policy this repo
    did not ship is custom, whoever spawned it and however they spelled the arguments.
    """
    if env is None:
        env = os.environ
    if env.get(FORCE_ENVELOPE_ENV) == "1":
        return True
    if custom_policy:
        return True
    return not is_builtin_policy(onnx_model_path)


# ── Joint limits ────────────────────────────────────────────────────────────────


def joint_limits_from_urdf(
    path: str | None,
    joint_names: Sequence[str],
    fallback_rad: float = DEFAULT_JOINT_LIMIT_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint ``(lower, upper)`` travel limits in radians, aligned to ``joint_names``.

    Never raises and never refuses. A Pi with no URDF (the common case today — nothing in
    this repo ships one) gets ``+/-fallback_rad`` for every joint and one line on stdout at
    startup. Refusing to walk because a description file is absent would turn a safety
    feature into an outage.

    Joints are matched **by name**, never by index: the URDF carries antennas and fixed
    frame joints that ``hwi.joints`` does not, so position in the file means nothing.
    A revolute joint with no ``<limit lower= upper=>`` (a continuous joint) falls back too.
    """
    n = len(joint_names)
    lower = np.full(n, -abs(float(fallback_rad)))
    upper = np.full(n, abs(float(fallback_rad)))

    if not path:
        print(
            f"[envelope] no URDF supplied, clamping every joint to "
            f"+/-{fallback_rad} rad"
        )
        return lower, upper

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        print(
            f"[envelope] could not read the URDF at {path} ({exc}), clamping every "
            f"joint to +/-{fallback_rad} rad"
        )
        return lower, upper

    declared: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        raw_lower, raw_upper = limit.get("lower"), limit.get("upper")
        if raw_lower is None or raw_upper is None:
            continue
        try:
            lo, hi = float(raw_lower), float(raw_upper)
        except ValueError:
            continue
        # An inverted or non-finite range would clamp every command to nothing, which is
        # a worse failure than the fallback it replaces.
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            continue
        declared[name] = (lo, hi)

    missing: list[str] = []
    for i, name in enumerate(joint_names):
        if name in declared:
            lower[i], upper[i] = declared[name]
        else:
            missing.append(name)

    if missing:
        print(
            f"[envelope] URDF at {path} declares no limit for {', '.join(missing)}, "
            f"clamping those to +/-{fallback_rad} rad"
        )
    return lower, upper


# ── Clamping ────────────────────────────────────────────────────────────────────


class ActionEnvelope:
    """Bounds one tick's commanded joint targets, and counts how often it had to.

    ``clamp()`` applies the joint-limit clamp first and the velocity clamp second. The
    order is load-bearing and is documented at the call site: the velocity clamp must see
    the final commanded value, including the operator's head command, or a head command
    can be delivered as a step no servo should be asked to take.
    """

    def __init__(
        self,
        joint_names: Sequence[str],
        lower: np.ndarray,
        upper: np.ndarray,
        max_velocity: float,
        control_hz: int = CONTROL_HZ,
    ) -> None:
        self.joint_names: list[str] = list(joint_names)
        n = len(self.joint_names)
        if n == 0:
            raise ValueError("ActionEnvelope needs at least one joint")

        self.lower = np.array(lower, dtype=float).reshape(-1).copy()
        self.upper = np.array(upper, dtype=float).reshape(-1).copy()
        if self.lower.size != n or self.upper.size != n:
            raise ValueError(
                f"bounds do not match the joints: {n} names, "
                f"{self.lower.size} lower, {self.upper.size} upper"
            )
        if np.any(self.lower > self.upper):
            bad = [
                self.joint_names[i]
                for i in np.flatnonzero(self.lower > self.upper).tolist()
            ]
            raise ValueError(f"inverted joint limits for: {', '.join(bad)}")

        self.max_velocity = float(max_velocity)
        self.control_hz = float(control_hz)
        if self.max_velocity <= 0 or self.control_hz <= 0:
            raise ValueError(
                f"max_velocity and control_hz must be positive, got "
                f"{max_velocity} and {control_hz}"
            )
        # rad per control tick — the same expression the disabled clamp used at
        # v2_rl_walk_mujoco.py:350-356 (max_motor_velocity * (1 / control_freq)).
        self.max_step = self.max_velocity * (1.0 / self.control_hz)

        # Preallocated scratch. The tick allocates nothing.
        self._before = np.zeros(n)
        self._vel_lower = np.zeros(n)
        self._vel_upper = np.zeros(n)
        self._changed = np.zeros(n, dtype=bool)
        self._limit_counts = np.zeros(n, dtype=np.int64)
        self._velocity_counts = np.zeros(n, dtype=np.int64)

    def clamp(self, targets: np.ndarray, prev: np.ndarray) -> np.ndarray:
        """Clamp ``targets`` in place against the joint limits, then against ``prev``.

        ``prev`` is the previous tick's *commanded* value, not the measured position: the
        velocity limit is a bound on what we ask of a servo, and a servo that is already
        lagging must not have that lag turned into permission for a bigger step.

        Returns the same array it was handed (converted once if it was not float64), so the
        caller can assign it back without caring which happened.
        """
        targets = np.asarray(targets, dtype=float)

        # 1. Joint limits — the mechanical stops.
        np.copyto(self._before, targets)
        np.clip(targets, self.lower, self.upper, out=targets)
        np.not_equal(targets, self._before, out=self._changed)
        self._limit_counts += self._changed

        # 2. Velocity — how far a target may move in one tick, from what we last commanded.
        np.copyto(self._before, targets)
        np.subtract(prev, self.max_step, out=self._vel_lower)
        np.add(prev, self.max_step, out=self._vel_upper)
        np.clip(targets, self._vel_lower, self._vel_upper, out=targets)
        np.not_equal(targets, self._before, out=self._changed)
        self._velocity_counts += self._changed

        return targets

    def clamp_counts(self, kind: str = "any") -> dict[str, int]:
        """Per-joint clamp counts since the last ``reset_counts()``, joints that never
        tripped omitted.

        ``kind`` is ``"any"`` (the default — either clamp), ``"limit"`` or ``"velocity"``.
        Telemetry reports the total; the two kinds are separated in the log because they
        mean different things: a joint-limit clamp says the policy asked for an impossible
        pose, a velocity clamp says it asked for an impossible *move*.
        """
        if kind == "limit":
            counts = self._limit_counts
        elif kind == "velocity":
            counts = self._velocity_counts
        elif kind == "any":
            counts = self._limit_counts + self._velocity_counts
        else:
            raise ValueError(f"kind must be 'any', 'limit' or 'velocity', got {kind!r}")
        return {
            name: int(counts[i])
            for i, name in enumerate(self.joint_names)
            if counts[i]
        }

    def reset_counts(self) -> None:
        self._limit_counts.fill(0)
        self._velocity_counts.fill(0)


def format_counts(counts: dict[str, int], limit: int = 4) -> str:
    """``"right_knee x47, left_ankle x3"``, worst first, for the once-a-second log line.

    Truncated because a policy that saturates everything would otherwise print all
    fourteen joints every second, and the first few are the ones worth reading.
    """
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{name} x{count}" for name, count in ranked[:limit])
    if len(ranked) > limit:
        shown += f", and {len(ranked) - limit} more"
    return shown


# ── Aborting ────────────────────────────────────────────────────────────────────


def imu_quaternion(
    reading: Mapping[str, object] | None,
    now_s: float,
    max_age_s: float = DEFAULT_IMU_MAX_AGE_S,
) -> Sequence[float] | None:
    """The orientation out of one ``raw_imu.Imu.get_data()`` dict, or ``None``.

    This exists because "the quaternion is missing" is not a state the driver can
    produce, and the tilt guard was written as if it were. ``raw_imu``'s worker wraps the
    fused read in its own ``try`` and keeps the previous value on failure — deliberately,
    because the policy's gyro and accelerometer must keep flowing — and that previous
    value starts life as identity. So a BNO055 whose fused-orientation register stops
    answering hands out a perfectly level quaternion forever, and a worker thread that
    stalls hands out the same dict forever. Both read as "upright" to anything that only
    checks for ``None``, which is failure mode F4 with the guard still switched on.

    What separates the two cases is *age*, so the driver stamps each sample with when its
    fused read last succeeded and this asks how long ago that was. Unknown means: no
    reading, no quaternion in it, a driver that does not stamp, a fused read that has
    never once succeeded, or a stamp older than ``max_age_s``. Every one of those is an
    orientation nobody knows, which is what ``AbortMonitor.check_tilt(None)`` is for.

    ``now_s`` and the stamp must come from the same clock (``time.monotonic()``); the
    walk and the IMU worker share a process, so they do.
    """
    if not isinstance(reading, Mapping):
        return None
    quaternion = reading.get("quaternion")
    if quaternion is None:
        return None
    stamped_at = reading.get("quaternion_t")
    if not isinstance(stamped_at, (int, float)) or isinstance(stamped_at, bool):
        return None
    if not math.isfinite(float(stamped_at)):
        return None
    if now_s - float(stamped_at) > max_age_s:
        return None
    return quaternion  # type: ignore[return-value]


def tilt_deg(quaternion: Sequence[float] | None) -> tuple[float, float] | None:
    """``(pitch, roll)`` in degrees from a ``(w, x, y, z)`` quaternion, or ``None``.

    ``None`` means the orientation is unknown — absent, wrong length, non-numeric, or a
    zero/degenerate quaternion that cannot be normalised. The caller must treat that as
    unsafe, not as level.

    Identity ``[1, 0, 0, 0]`` is genuinely level and is reported as ``(0, 0)``. A duck
    standing still really does read identity, so refusing it here would abort every walk
    in its first 8 ticks; identity that is merely *left over* from a fused read that
    never happened is ``imu_quaternion``'s problem, one layer up, and never reaches
    here.

    Standard ZYX (yaw-pitch-roll) convention, matching ``imu.py``'s ``as_euler("xyz")``
    for the pitch/roll pair. A disagreement with the physical axis remap could only swap
    which of the two names appears in the log — both are checked against the same limit,
    so it cannot make a fall go unnoticed.
    """
    if quaternion is None:
        return None
    try:
        if len(quaternion) != 4:
            return None
        w, x, y, z = (float(v) for v in quaternion)
    except (TypeError, ValueError):
        return None

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-6:
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return math.degrees(pitch), math.degrees(roll)


class AbortMonitor:
    """Counts consecutive bad ticks and says when the walk must stop.

    Stateful on purpose, and the state is a streak length: both guards trip on a sustained
    condition. Every ``check_*`` must be called on **every** tick, including good ones —
    that is what resets the streak, and it is why a transient stumble or one slow tick does
    not abort.
    """

    def __init__(
        self,
        tilt_limit_deg: float = DEFAULT_TILT_LIMIT_DEG,
        tilt_ticks: int = DEFAULT_TILT_ABORT_TICKS,
        budget_s: float = 1.0 / CONTROL_HZ,
        budget_ticks: int = DEFAULT_BUDGET_OVERRUN_TICKS,
    ) -> None:
        self.tilt_limit_deg = float(tilt_limit_deg)
        self.budget_s = float(budget_s)
        # A config that said "0 ticks" would abort on the first sample, and a negative one
        # would abort before any sample. Neither is a guard; both floor at one tick.
        self.tilt_ticks = max(1, int(tilt_ticks))
        self.budget_ticks = max(1, int(budget_ticks))

        self.tilt_streak: int = 0
        self.budget_streak: int = 0
        self._overrun_sum_s: float = 0.0
        self._worst: str = ""

    def check_tilt(self, quaternion: Sequence[float] | None) -> str | None:
        """The abort detail once tilt has been out of bounds for ``tilt_ticks`` ticks.

        ``None`` from the IMU is not "upright": it increments the streak. An IMU that has
        stopped answering during a walk is a robot whose orientation nobody knows, which is
        exactly the situation the guard exists for.
        """
        angles = tilt_deg(quaternion)
        if angles is None:
            self.tilt_streak += 1
            self._worst = "orientation unknown"
        else:
            pitch, roll = angles
            if max(abs(pitch), abs(roll)) <= self.tilt_limit_deg:
                self.tilt_streak = 0
                return None
            self.tilt_streak += 1
            axis, value = (
                ("pitch", pitch) if abs(pitch) >= abs(roll) else ("roll", roll)
            )
            self._worst = f"{axis} {value:.0f} deg"

        if self.tilt_streak < self.tilt_ticks:
            return None
        return (
            f"{self._worst} for {self.tilt_streak} ticks "
            f"(limit {self.tilt_limit_deg:.0f})"
        )

    def check_budget(self, took_s: float) -> str | None:
        """The abort detail once the loop has overrun ``budget_ticks`` ticks in a row.

        The overrun test is ``took > budget``, byte-for-byte the condition the existing
        print at ``v2_rl_walk_mujoco.py:434`` uses, so the printed lines and the abort can
        never disagree about what an overrun is.
        """
        if took_s <= self.budget_s:
            self.budget_streak = 0
            self._overrun_sum_s = 0.0
            return None

        self.budget_streak += 1
        self._overrun_sum_s += took_s - self.budget_s
        if self.budget_streak < self.budget_ticks:
            return None

        mean_over_ms = (self._overrun_sum_s / self.budget_streak) * 1000.0
        return (
            f"{self.budget_streak} consecutive overruns, mean {mean_over_ms:.1f}ms "
            f"over a {self.budget_s * 1000.0:.1f}ms budget"
        )

    # Operator copy lives next to the guard that produces it, one short sentence each, no
    # diagnostics — the tick counts and milliseconds are in `detail`, for the log.
    # (tnkr-studio/app/DESIGN.md#errors)
    OPERATOR: dict[str, str] = {
        "tilt": "The robot fell over, so walking stopped.",
        "budget": "This policy is too slow to keep the robot walking.",
    }

    def abort(self, reason: str, detail: str) -> PolicyAbort:
        """Build the exception for a tripped guard. Raising is the caller's job."""
        return PolicyAbort(
            reason, detail, self.OPERATOR.get(reason, "Walking stopped.")
        )
