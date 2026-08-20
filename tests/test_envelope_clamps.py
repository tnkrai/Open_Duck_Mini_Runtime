"""The clamp half of the safety envelope, and the gate that decides when it applies.

No robot: ``ActionEnvelope`` is a pure function of arrays, which is the whole reason the
gating in amendment A8 is affordable — the code is exercised here even though the ducks in
the field never run it.

The gating itself is covered two ways, because the two failure modes are different:

* **behaviour** -- ``is_armed`` is a predicate, so both of its answers are testable
  directly (``test_builtin_policy_is_not_clamped``, ``test_force_envelope_arms_it``);
* **structure** -- an ``ast`` read of the walk script asserts the clamp call really is
  behind that gate and nowhere else, since a behavioural test cannot reach ``RLWalk.run``
  without a servo bus, an IMU and GPIO.
"""

from __future__ import annotations

import ast
import timeit
from pathlib import Path

import numpy as np
import pytest

from mini_bdx_runtime.envelope import (
    DEFAULT_JOINT_LIMIT_RAD,
    ActionEnvelope,
    format_counts,
    is_armed,
    joint_limits_from_urdf,
)
from mini_bdx_runtime.policy_contract import ACT_DIM, CONTROL_HZ

WALK_SCRIPT = Path(__file__).parent.parent / "scripts" / "v2_rl_walk_mujoco.py"

# The real joint order, from rustypot_position_hwi.HWI.joints. 5:9 is the head.
JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

MAX_MOTOR_VELOCITY = 5.24  # rad/s, from RLWalk.__init__


def make_envelope(lower=-1.0, upper=1.0, names=None, max_velocity=MAX_MOTOR_VELOCITY):
    names = list(names or JOINT_NAMES)
    n = len(names)
    return ActionEnvelope(
        names,
        np.full(n, lower),
        np.full(n, upper),
        max_velocity,
        CONTROL_HZ,
    )


# ── Joint-limit clamp: the boundaries the story names ───────────────────────────


def test_within_range_is_untouched() -> None:
    env = make_envelope()
    prev = np.zeros(len(JOINT_NAMES))
    targets = np.full(len(JOINT_NAMES), 0.05)  # inside both limit and one tick's travel

    out = env.clamp(targets, prev)

    assert np.allclose(out, 0.05)
    assert env.clamp_counts() == {}, "a well-behaved policy must trip nothing"


# A velocity so large the second clamp cannot fire, so these two tests isolate the
# joint-limit clamp. The two clamps interacting is its own test below.
UNLIMITED_VELOCITY = 1000.0


def test_below_min_is_raised_to_the_limit() -> None:
    env = make_envelope(lower=-0.5, upper=0.5, max_velocity=UNLIMITED_VELOCITY)
    prev = np.zeros(len(JOINT_NAMES))
    targets = np.zeros(len(JOINT_NAMES))
    targets[3] = -4.0  # left_knee, way past its stop

    env.clamp(targets, prev)

    assert targets[3] == pytest.approx(-0.5)
    assert env.clamp_counts("limit") == {"left_knee": 1}


def test_above_max_is_lowered_to_the_limit() -> None:
    env = make_envelope(lower=-0.5, upper=0.5, max_velocity=UNLIMITED_VELOCITY)
    prev = np.zeros(len(JOINT_NAMES))
    targets = np.zeros(len(JOINT_NAMES))
    targets[12] = 4.71  # right_knee, the story's own example

    env.clamp(targets, prev)

    assert targets[12] == pytest.approx(0.5)
    assert env.clamp_counts("limit") == {"right_knee": 1}


def test_asymmetric_limits_are_per_joint_and_by_name() -> None:
    """One joint's stop must not be applied to another's — the head is not the knee."""
    n = len(JOINT_NAMES)
    lower = np.full(n, -2.0)
    upper = np.full(n, 2.0)
    lower[6], upper[6] = -0.1, 0.1  # head_pitch, the fragile one
    env = ActionEnvelope(JOINT_NAMES, lower, upper, UNLIMITED_VELOCITY, CONTROL_HZ)

    prev = np.zeros(n)
    targets = np.full(n, 1.5)
    env.clamp(targets, prev)

    assert targets[6] == pytest.approx(0.1)
    # every other joint was only velocity-limited, not limit-clamped
    assert env.clamp_counts("limit") == {"head_pitch": 1}


def test_no_declared_limit_falls_back_to_two_radians() -> None:
    lower, upper = joint_limits_from_urdf(None, JOINT_NAMES)
    env = ActionEnvelope(JOINT_NAMES, lower, upper, MAX_MOTOR_VELOCITY, CONTROL_HZ)

    prev = np.zeros(len(JOINT_NAMES))
    targets = np.full(len(JOINT_NAMES), 9.0)
    env.clamp(targets, prev)

    # the limit clamp pinned it to the fallback before the velocity clamp saw it
    assert np.allclose(env.lower, -DEFAULT_JOINT_LIMIT_RAD)
    assert np.allclose(env.upper, DEFAULT_JOINT_LIMIT_RAD)
    assert env.clamp_counts("limit") == {name: 1 for name in JOINT_NAMES}


# ── Velocity clamp ──────────────────────────────────────────────────────────────


def test_velocity_clamp_bounds_one_ticks_travel() -> None:
    env = make_envelope(lower=-10.0, upper=10.0)
    step = MAX_MOTOR_VELOCITY / CONTROL_HZ
    prev = np.zeros(len(JOINT_NAMES))
    targets = np.full(len(JOINT_NAMES), 5.0)

    env.clamp(targets, prev)

    assert np.allclose(targets, step)
    assert env.clamp_counts("velocity") == {name: 1 for name in JOINT_NAMES}
    assert env.clamp_counts("limit") == {}, "10 rad is inside the joint limits here"


def test_exactly_at_the_velocity_limit_is_not_clamped() -> None:
    """The boundary case the story calls out by name. A target exactly one tick's travel
    away is legal, and counting it would make the telemetry report saturation on a policy
    that is merely moving at full speed."""
    env = make_envelope(lower=-10.0, upper=10.0)
    step = env.max_step
    prev = np.zeros(len(JOINT_NAMES))
    targets = np.full(len(JOINT_NAMES), step)

    env.clamp(targets, prev)

    assert np.allclose(targets, step)
    assert env.clamp_counts() == {}


def test_first_tick_is_bounded_from_the_rest_pose_not_pinned_to_zero() -> None:
    """``prev`` on the first tick is ``init_pos``, so the first commanded step must be a
    real step away from the rest pose — not zero motion."""
    env = make_envelope(lower=-10.0, upper=10.0)
    init_pos = np.linspace(-0.4, 0.4, len(JOINT_NAMES))
    prev = init_pos.copy()
    targets = init_pos + 1.0  # a big first step

    env.clamp(targets, prev)

    assert np.all(targets > init_pos), "the first tick may still move"
    assert np.allclose(targets, init_pos + env.max_step)


def test_velocity_clamp_is_symmetric() -> None:
    env = make_envelope(lower=-10.0, upper=10.0)
    prev = np.zeros(len(JOINT_NAMES))
    targets = np.full(len(JOINT_NAMES), -5.0)

    env.clamp(targets, prev)

    assert np.allclose(targets, -env.max_step)


def test_prev_is_read_only() -> None:
    """The caller owns the baseline buffer; clamping must not write through it."""
    env = make_envelope()
    prev = np.full(len(JOINT_NAMES), 0.3)
    before = prev.copy()

    env.clamp(np.full(len(JOINT_NAMES), 5.0), prev)

    assert np.array_equal(prev, before)


def test_limit_clamp_runs_before_the_velocity_clamp() -> None:
    """Order is load-bearing: a target far past a stop must end up one tick's travel from
    prev, not one tick's travel from the absurd value it asked for."""
    env = make_envelope(lower=-0.5, upper=0.5)
    prev = np.zeros(len(JOINT_NAMES))
    targets = np.full(len(JOINT_NAMES), 50.0)

    env.clamp(targets, prev)

    assert np.allclose(targets, env.max_step)
    assert env.clamp_counts("limit") and env.clamp_counts("velocity")


# ── Counters ────────────────────────────────────────────────────────────────────


def test_counts_accumulate_across_ticks_and_reset() -> None:
    env = make_envelope(lower=-0.5, upper=0.5)
    prev = np.zeros(len(JOINT_NAMES))
    for _ in range(7):
        targets = np.zeros(len(JOINT_NAMES))
        targets[12] = 4.0
        env.clamp(targets, prev)

    assert env.clamp_counts("limit") == {"right_knee": 7}
    env.reset_counts()
    assert env.clamp_counts() == {}


def test_counts_separate_limit_from_velocity() -> None:
    env = make_envelope(lower=-0.5, upper=0.5)
    prev = np.zeros(len(JOINT_NAMES))
    targets = np.zeros(len(JOINT_NAMES))
    targets[0] = 4.0  # limit-clamped, then velocity-clamped
    targets[1] = 0.4  # inside the limit, outside one tick's travel

    env.clamp(targets, prev)

    assert env.clamp_counts("limit") == {"left_hip_yaw": 1}
    assert env.clamp_counts("velocity") == {"left_hip_yaw": 1, "left_hip_roll": 1}
    assert env.clamp_counts() == {"left_hip_yaw": 2, "left_hip_roll": 1}


def test_unknown_count_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_envelope().clamp_counts("sideways")


def test_format_counts_is_worst_first_and_truncated() -> None:
    line = format_counts({"a": 1, "b": 9, "c": 5, "d": 7, "e": 3}, limit=2)
    assert line.startswith("b x9, d x7")
    assert "and 3 more" in line


# ── Construction is validated, not trusted ──────────────────────────────────────


def test_mismatched_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        ActionEnvelope(JOINT_NAMES, np.zeros(3), np.ones(3), 1.0, CONTROL_HZ)


def test_inverted_limits_are_rejected_by_name() -> None:
    n = len(JOINT_NAMES)
    lower = np.full(n, -1.0)
    upper = np.full(n, 1.0)
    lower[12], upper[12] = 1.0, -1.0
    with pytest.raises(ValueError, match="right_knee"):
        ActionEnvelope(JOINT_NAMES, lower, upper, 1.0, CONTROL_HZ)


def test_nonpositive_velocity_or_rate_is_rejected() -> None:
    n = len(JOINT_NAMES)
    with pytest.raises(ValueError):
        ActionEnvelope(JOINT_NAMES, np.full(n, -1.0), np.full(n, 1.0), 0.0, CONTROL_HZ)
    with pytest.raises(ValueError):
        ActionEnvelope(JOINT_NAMES, np.full(n, -1.0), np.full(n, 1.0), 5.24, 0)


def test_envelope_covers_every_commanded_dof() -> None:
    """One bound per action the policy emits: 14 in, 14 clamped, none unguarded."""
    env = make_envelope()
    assert len(env.joint_names) == ACT_DIM
    assert env.lower.size == ACT_DIM and env.upper.size == ACT_DIM


# ── URDF limits ─────────────────────────────────────────────────────────────────

# A trimmed URDF: real names and real numbers from
# tnkr-studio/app/public/robots/open-duck-mini/open_duck_mini_v2.urdf, plus the three
# shapes that must fall back. Joint order is deliberately NOT hwi order.
URDF = """<?xml version="1.0"?>
<robot name="open_duck_mini_v2">
  <joint name="trunk_frame" type="fixed">
    <parent link="trunk"/><child link="frame"/>
  </joint>
  <joint name="right_knee" type="revolute">
    <limit effort="1" velocity="20" lower="-1.5707963" upper="1.5707963"/>
  </joint>
  <joint name="left_knee" type="revolute">
    <limit effort="1" velocity="20" lower="-1.5707963" upper="1.5707963"/>
  </joint>
  <joint name="head_pitch" type="revolute">
    <limit effort="1" velocity="20" lower="-0.7853982" upper="0.7853982"/>
  </joint>
  <joint name="left_antenna" type="revolute">
    <limit effort="1" velocity="20" lower="-1.5707963" upper="1.5707963"/>
  </joint>
  <joint name="head_yaw" type="continuous"/>
  <joint name="head_roll" type="revolute">
    <limit effort="1" velocity="20" lower="0.5" upper="-0.5"/>
  </joint>
  <joint name="neck_pitch" type="revolute">
    <limit effort="1" velocity="20" lower="not-a-number" upper="1.0"/>
  </joint>
</robot>
"""


def test_urdf_limits_are_matched_by_name(tmp_path) -> None:
    path = tmp_path / "duck.urdf"
    path.write_text(URDF)

    lower, upper = joint_limits_from_urdf(str(path), JOINT_NAMES)

    knee = JOINT_NAMES.index("right_knee")
    assert lower[knee] == pytest.approx(-1.5707963)
    assert upper[knee] == pytest.approx(1.5707963)
    head_pitch = JOINT_NAMES.index("head_pitch")
    assert upper[head_pitch] == pytest.approx(0.7853982)


def test_urdf_joints_absent_from_the_robot_are_ignored(tmp_path) -> None:
    """The URDF has antennas and fixed frames the walk never commands. Aligning by index
    instead of name would shift every limit onto the wrong joint."""
    path = tmp_path / "duck.urdf"
    path.write_text(URDF)

    lower, upper = joint_limits_from_urdf(str(path), JOINT_NAMES)

    assert "left_antenna" not in JOINT_NAMES
    assert len(lower) == len(JOINT_NAMES) == len(upper)
    # left_hip_yaw is first in hwi order but absent from this URDF -> fallback
    assert lower[0] == pytest.approx(-DEFAULT_JOINT_LIMIT_RAD)


def test_urdf_shapes_that_cannot_be_used_fall_back(tmp_path) -> None:
    path = tmp_path / "duck.urdf"
    path.write_text(URDF)

    lower, upper = joint_limits_from_urdf(str(path), JOINT_NAMES)

    for name in ("head_yaw", "head_roll", "neck_pitch"):
        i = JOINT_NAMES.index(name)
        assert lower[i] == pytest.approx(-DEFAULT_JOINT_LIMIT_RAD), name
        assert upper[i] == pytest.approx(DEFAULT_JOINT_LIMIT_RAD), name


def test_missing_urdf_falls_back_and_never_raises(tmp_path) -> None:
    lower, upper = joint_limits_from_urdf(str(tmp_path / "nope.urdf"), JOINT_NAMES)
    assert np.allclose(lower, -DEFAULT_JOINT_LIMIT_RAD)
    assert np.allclose(upper, DEFAULT_JOINT_LIMIT_RAD)


def test_unparseable_urdf_falls_back_and_never_raises(tmp_path) -> None:
    path = tmp_path / "truncated.urdf"
    path.write_text("<robot><joint name='left_knee'")
    lower, upper = joint_limits_from_urdf(str(path), JOINT_NAMES)
    assert np.allclose(lower, -DEFAULT_JOINT_LIMIT_RAD)
    assert np.allclose(upper, DEFAULT_JOINT_LIMIT_RAD)


# ── Amendment A8: the gate ──────────────────────────────────────────────────────


def test_builtin_policy_is_not_clamped() -> None:
    """No custom policy and no force flag -> the envelope is never built, so the loop
    runs the instructions every duck in the field runs today."""
    assert is_armed(False, {}) is False


def test_custom_policy_arms_it() -> None:
    assert is_armed(True, {}) is True


def test_force_envelope_arms_it() -> None:
    assert is_armed(False, {"TNKR_FORCE_ENVELOPE": "1"}) is True


def test_force_envelope_needs_exactly_one() -> None:
    """A stale ``TNKR_FORCE_ENVELOPE=0`` in a service file must not arm anything."""
    assert is_armed(False, {"TNKR_FORCE_ENVELOPE": "0"}) is False
    assert is_armed(False, {"TNKR_FORCE_ENVELOPE": ""}) is False


def test_force_envelope_reads_the_real_environment(monkeypatch) -> None:
    monkeypatch.delenv("TNKR_FORCE_ENVELOPE", raising=False)
    assert is_armed(False) is False
    monkeypatch.setenv("TNKR_FORCE_ENVELOPE", "1")
    assert is_armed(False) is True


# ── The gate is where the story says it is ──────────────────────────────────────
#
# Structural, because RLWalk cannot be constructed without a bus, an IMU and GPIO — the
# same reason test_policy_contract.py reads this file with ast rather than importing it.


def _run_body() -> ast.FunctionDef:
    tree = ast.parse(WALK_SCRIPT.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLWalk"
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )


def _guards_of(tree: ast.AST, call_name: str) -> list[str]:
    """The `if` tests, unparsed, that enclose each call to ``call_name``."""
    guards: list[str] = []

    def walk(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                walk(child, enclosing + [ast.unparse(child.test)])
                for handler in child.orelse:
                    walk(handler, enclosing)
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == call_name
            ):
                guards.append(" and ".join(enclosing))
            walk(child, enclosing)

    walk(tree, [])
    return guards


def test_the_clamp_call_is_behind_the_custom_policy_gate() -> None:
    guards = _guards_of(_run_body(), "clamp")
    assert guards, "no envelope.clamp call found in RLWalk.run"
    for guard in guards:
        assert "self.envelope is not None" in guard, (
            "a clamp call in the walk loop is not gated on the envelope being armed. "
            "With the built-in policy the loop must execute exactly the code it does "
            "today (amendment A8)."
        )


def test_the_envelope_is_only_built_when_armed() -> None:
    tree = ast.parse(WALK_SCRIPT.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLWalk"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )

    constructed: list[str] = []

    def walk(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                walk(child, enclosing + [ast.unparse(child.test)])
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in ("ActionEnvelope", "AbortMonitor")
            ):
                constructed.append(" and ".join(enclosing))
            walk(child, enclosing)

    walk(init, [])
    assert len(constructed) == 2, f"expected one of each guard, got {constructed}"
    for guard in constructed:
        assert "is_armed" in guard, (
            "the safety envelope is constructed unconditionally. It must arm only for a "
            "custom policy or under TNKR_FORCE_ENVELOPE (amendment A8)."
        )


# ── Cost ────────────────────────────────────────────────────────────────────────


def test_clamping_costs_well_under_the_tick_budget() -> None:
    """The story's own instruction: measure it, do not assume.

    Budget is 0.5 ms of the 20 ms tick. Timed as the best of several runs, because the
    question is what the code costs, not what a loaded CI box costs.
    """
    env = make_envelope()
    prev = np.zeros(len(JOINT_NAMES))
    targets = np.full(len(JOINT_NAMES), 0.3)

    per_call = min(timeit.repeat(lambda: env.clamp(targets, prev), repeat=5, number=200))
    per_call /= 200

    assert per_call < 0.5e-3, f"clamp costs {per_call * 1e3:.3f}ms of a 0.5ms budget"


# ── Replaying the loop's own arithmetic ─────────────────────────────────────────


def test_clamps_the_array_the_loop_actually_builds() -> None:
    """The loop hands us ``init_pos + action * action_scale`` where ``action`` is whatever
    onnxruntime returned (float32). Clamping must happen in place on that array, or the
    "no allocation in the tick" requirement is quietly false."""
    init_pos = [0.0] * len(JOINT_NAMES)
    action = np.full(len(JOINT_NAMES), 2.0, dtype=np.float32)  # a policy at full deflection
    targets = init_pos + action * 0.25  # action_scale

    # the head override, exactly as the loop writes it
    last_commands = [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
    targets[5:9] = last_commands[3:] + targets[5:9]

    env = make_envelope(lower=-0.4, upper=0.4)
    prev = np.array(init_pos, dtype=float)
    out = env.clamp(targets, prev)

    assert out is targets, "clamped a copy, not the array the loop will send"
    assert np.all(out <= 0.4 + 1e-12)
    assert env.clamp_counts("limit"), "0.5 rad past a 0.4 rad stop must be clamped"


def test_a_held_head_command_is_not_capped_at_one_ticks_travel() -> None:
    """Why the baseline is the last COMMANDED value.

    ``prev_motor_targets`` is re-assigned before the head override, so using it here would
    make a held head command look like a fresh 0.5 rad jump on every single tick, pinning
    the head at 0.1 rad for as long as the operator held it. Against the last commanded
    value the command ramps in over a few ticks and then holds.
    """
    env = make_envelope(lower=-2.0, upper=2.0)
    prev = np.zeros(len(JOINT_NAMES))
    targets = prev

    for _ in range(20):
        targets = np.zeros(len(JOINT_NAMES))
        targets[6] = 0.5  # head_pitch, held
        targets = env.clamp(targets, prev)
        np.copyto(prev, targets)

    assert targets[6] == pytest.approx(0.5)
