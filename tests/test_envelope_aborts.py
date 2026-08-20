"""The abort half of the safety envelope: sustained tilt, sustained missed deadlines.

No robot and no IMU — ``AbortMonitor`` is fed quaternions and durations directly, which is
the point of keeping it a pure state machine.

The thing these tests are really defending is the word *sustained*. A guard that aborts on
one sample would fire on any loaded Linux box, operators would turn it off, and the duck
would be worse protected than if it had never shipped. Every "transient recovers" test
below is that requirement.
"""

from __future__ import annotations

import ast
import json
import math
import timeit
from pathlib import Path

import pytest

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.envelope import (
    ABORT_CODE,
    DEFAULT_BUDGET_OVERRUN_TICKS,
    DEFAULT_TILT_ABORT_TICKS,
    DEFAULT_TILT_LIMIT_DEG,
    AbortMonitor,
    PolicyAbort,
    is_armed,
    tilt_deg,
)
from mini_bdx_runtime.policy_contract import CONTROL_HZ
from mini_bdx_runtime import walk_telemetry

WALK_SCRIPT = Path(__file__).parent.parent / "scripts" / "v2_rl_walk_mujoco.py"

BUDGET_S = 1.0 / CONTROL_HZ  # 20 ms
UPRIGHT = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z — raw_imu's identity


def quat_from_pitch_roll(pitch_deg: float, roll_deg: float = 0.0) -> list[float]:
    """A (w, x, y, z) quaternion with the given pitch and roll, no yaw."""
    p, r = math.radians(pitch_deg) / 2.0, math.radians(roll_deg) / 2.0
    # ZYX composition with yaw = 0
    return [
        math.cos(r) * math.cos(p),
        math.sin(r) * math.cos(p),
        math.cos(r) * math.sin(p),
        -math.sin(r) * math.sin(p),
    ]


def monitor(**kwargs) -> AbortMonitor:
    return AbortMonitor(
        kwargs.pop("tilt_limit_deg", DEFAULT_TILT_LIMIT_DEG),
        kwargs.pop("tilt_ticks", DEFAULT_TILT_ABORT_TICKS),
        kwargs.pop("budget_s", BUDGET_S),
        kwargs.pop("budget_ticks", DEFAULT_BUDGET_OVERRUN_TICKS),
        **kwargs,
    )


# ── Budget ──────────────────────────────────────────────────────────────────────


def test_sustained_overrun_aborts_on_the_nth_tick() -> None:
    m = monitor(budget_ticks=10)
    for tick in range(9):
        assert m.check_budget(0.033) is None, f"aborted early, at tick {tick + 1}"

    detail = m.check_budget(0.033)

    assert detail is not None
    assert "10 consecutive overruns" in detail
    assert "13.0ms" in detail, detail  # 33ms - 20ms budget
    assert "20.0ms budget" in detail


def test_transient_overrun_recovers() -> None:
    """One slow tick is a loaded machine, not a failing policy."""
    m = monitor(budget_ticks=10)
    for _ in range(9):
        assert m.check_budget(0.033) is None
    assert m.check_budget(0.010) is None, "a good tick must not abort"
    assert m.budget_streak == 0

    # and the streak really restarted, rather than resuming near the threshold
    for _ in range(9):
        assert m.check_budget(0.033) is None


def test_a_tick_exactly_on_budget_is_not_an_overrun() -> None:
    """Matches the existing print at v2_rl_walk_mujoco.py:434, which fires on
    ``1 / control_freq - took < 0`` — strictly over, never equal."""
    m = monitor(budget_ticks=1)
    assert m.check_budget(BUDGET_S) is None
    assert m.budget_streak == 0


def test_a_single_tick_over_budget_aborts_only_if_configured_to() -> None:
    m = monitor(budget_ticks=1)
    assert m.check_budget(BUDGET_S + 0.001) is not None


def test_overrun_mean_is_the_mean_of_the_overruns() -> None:
    m = monitor(budget_ticks=2)
    m.check_budget(BUDGET_S + 0.010)
    detail = m.check_budget(BUDGET_S + 0.020)
    assert "mean 15.0ms" in detail, detail


def test_recovery_forgets_the_previous_overrun_sizes() -> None:
    m = monitor(budget_ticks=2)
    m.check_budget(BUDGET_S + 0.100)  # one enormous outlier
    m.check_budget(0.001)  # recovered
    m.check_budget(BUDGET_S + 0.002)
    detail = m.check_budget(BUDGET_S + 0.002)
    assert "mean 2.0ms" in detail, detail


# ── Tilt ────────────────────────────────────────────────────────────────────────


def test_upright_never_aborts() -> None:
    m = monitor(tilt_ticks=2)
    for _ in range(100):
        assert m.check_tilt(UPRIGHT) is None


def test_sustained_tilt_aborts_and_names_the_axis() -> None:
    m = monitor(tilt_limit_deg=60.0, tilt_ticks=8)
    tilted = quat_from_pitch_roll(71.0)
    for tick in range(7):
        assert m.check_tilt(tilted) is None, f"aborted early, at tick {tick + 1}"

    detail = m.check_tilt(tilted)

    assert detail is not None
    assert "pitch 71 deg" in detail, detail
    assert "for 8 ticks" in detail
    assert "limit 60" in detail


def test_sustained_roll_aborts_too() -> None:
    m = monitor(tilt_limit_deg=60.0, tilt_ticks=3)
    rolled = quat_from_pitch_roll(0.0, -75.0)
    m.check_tilt(rolled)
    m.check_tilt(rolled)
    detail = m.check_tilt(rolled)
    assert detail is not None and "roll -75 deg" in detail, detail


def test_transient_tilt_recovers() -> None:
    """The duck stepped off a rug. That is not a fall."""
    m = monitor(tilt_limit_deg=60.0, tilt_ticks=8)
    tilted = quat_from_pitch_roll(80.0)
    for _ in range(7):
        assert m.check_tilt(tilted) is None
    assert m.check_tilt(UPRIGHT) is None
    assert m.tilt_streak == 0
    for _ in range(7):
        assert m.check_tilt(tilted) is None


def test_exactly_at_the_tilt_limit_is_not_a_fall() -> None:
    m = monitor(tilt_limit_deg=60.0, tilt_ticks=1)
    assert m.check_tilt(quat_from_pitch_roll(60.0)) is None
    assert m.check_tilt(quat_from_pitch_roll(60.5)) is not None


# ── The near-critical one: a missing IMU reading is not "upright" (F4) ───────────


def test_imu_none_counts_toward_the_abort() -> None:
    """Failure mode F4. If a missing orientation read as level, an IMU that died mid-walk
    would leave the policy driving a fallen duck with no guard at all."""
    m = monitor(tilt_ticks=4)
    for _ in range(3):
        assert m.check_tilt(None) is None
    detail = m.check_tilt(None)
    assert detail is not None
    assert "orientation unknown" in detail, detail


def test_imu_recovering_from_none_resets_the_streak() -> None:
    m = monitor(tilt_ticks=4)
    m.check_tilt(None)
    m.check_tilt(None)
    assert m.check_tilt(UPRIGHT) is None
    assert m.tilt_streak == 0


@pytest.mark.parametrize(
    "quaternion",
    [
        None,
        [],
        [1.0, 0.0, 0.0],  # short
        [1.0, 0.0, 0.0, 0.0, 0.0],  # long
        [0.0, 0.0, 0.0, 0.0],  # zero — degenerate, not level
        ["a", "b", "c", "d"],
        [float("nan"), 0.0, 0.0, 0.0],
        42,
    ],
    ids=["none", "empty", "short", "long", "zero", "strings", "nan", "not-a-sequence"],
)
def test_unusable_quaternions_are_unknown_not_level(quaternion) -> None:
    assert tilt_deg(quaternion) is None
    m = monitor(tilt_ticks=1)
    assert m.check_tilt(quaternion) is not None


def test_identity_quaternion_is_level() -> None:
    """raw_imu hands back identity before its first real reading. Calling that unknown
    would abort every walk in its first few ticks."""
    pitch, roll = tilt_deg(UPRIGHT)
    assert pitch == pytest.approx(0.0)
    assert roll == pytest.approx(0.0)


def test_unnormalised_quaternions_are_normalised_first() -> None:
    scaled = [4.0 * q for q in quat_from_pitch_roll(30.0)]
    pitch, roll = tilt_deg(scaled)
    assert pitch == pytest.approx(30.0, abs=0.01)
    assert roll == pytest.approx(0.0, abs=0.01)


def test_pitch_and_roll_are_recovered_independently() -> None:
    pitch, roll = tilt_deg(quat_from_pitch_roll(20.0, -35.0))
    assert pitch == pytest.approx(20.0, abs=0.01)
    assert roll == pytest.approx(-35.0, abs=0.01)


# ── The exception itself ────────────────────────────────────────────────────────


def test_abort_carries_a_code_a_reason_and_two_audiences() -> None:
    m = monitor()
    abort = m.abort("tilt", "pitch 71 deg for 8 ticks (limit 60)")

    assert abort.code == ABORT_CODE == "POLICY_ABORTED"
    assert abort.reason == "tilt"
    assert "71" in abort.detail
    assert abort.operator and "71" not in abort.operator, (
        "the operator sentence must not carry diagnostics — the tick counts and degrees "
        "belong in the log (tnkr-studio/app/DESIGN.md#errors)"
    )


@pytest.mark.parametrize("reason", ["tilt", "budget"])
def test_operator_copy_is_one_short_sentence(reason) -> None:
    operator = AbortMonitor.OPERATOR[reason]
    assert len(operator) <= 100, operator
    assert operator.count(".") == 1 and operator.endswith("."), operator
    assert ";" not in operator, "a semicolon means it is explaining, not telling"


def test_abort_is_a_system_exit_so_the_existing_teardown_runs() -> None:
    """The walk's ``finally`` is the only cleanup that disables torque, and it already runs
    for SystemExit because that is how SIGTERM ends a walk. A second teardown would be a
    second thing to forget a step in."""
    abort = monitor().abort("budget", "10 consecutive overruns")
    assert isinstance(abort, SystemExit)
    assert abort.code == "POLICY_ABORTED"  # not an exit status
    with pytest.raises(SystemExit):
        raise abort


def test_unknown_reason_still_has_something_to_show_an_operator() -> None:
    assert monitor().abort("gremlins", "?").operator


# ── Thresholds come from config, and a bad config never disarms the guard ───────


def write_config(tmp_path, **keys) -> DuckConfig:
    path = tmp_path / "duck_config.json"
    path.write_text(json.dumps({"joints_offsets": {}, **keys}))
    return DuckConfig(config_json_path=str(path))


def test_missing_keys_use_the_safe_defaults(tmp_path) -> None:
    config = write_config(tmp_path)
    assert config.tilt_limit_deg == DEFAULT_TILT_LIMIT_DEG
    assert config.tilt_abort_ticks == DEFAULT_TILT_ABORT_TICKS
    assert config.budget_overrun_ticks == DEFAULT_BUDGET_OVERRUN_TICKS


def test_configured_thresholds_are_honoured(tmp_path) -> None:
    config = write_config(
        tmp_path, tilt_limit_deg=45, tilt_abort_ticks=3, budget_overrun_ticks=25
    )
    assert config.tilt_limit_deg == 45.0
    assert config.tilt_abort_ticks == 3
    assert config.budget_overrun_ticks == 25


@pytest.mark.parametrize(
    "keys",
    [
        {"tilt_abort_ticks": 0},
        {"tilt_abort_ticks": -5},
        {"tilt_abort_ticks": "soon"},
        {"tilt_abort_ticks": None},
        {"budget_overrun_ticks": 0},
        {"budget_overrun_ticks": "never"},
        {"tilt_limit_deg": 0},
        {"tilt_limit_deg": -60},
        {"tilt_limit_deg": 1000},
        {"tilt_limit_deg": "sixty"},
        {"tilt_limit_deg": None},
    ],
    ids=str,
)
def test_a_nonsense_threshold_falls_back_instead_of_disarming(tmp_path, keys) -> None:
    """"Configurable" must not become "switch-off-able". A zero tick count would abort on
    the first sample; a 1000-degree limit would never abort at all. Both are worse than
    the default."""
    config = write_config(tmp_path, **keys)
    assert config.tilt_limit_deg == DEFAULT_TILT_LIMIT_DEG or "tilt_limit_deg" not in keys
    assert config.tilt_abort_ticks >= 1
    assert config.budget_overrun_ticks >= 1
    for key in keys:
        assert getattr(config, key) == {
            "tilt_limit_deg": DEFAULT_TILT_LIMIT_DEG,
            "tilt_abort_ticks": DEFAULT_TILT_ABORT_TICKS,
            "budget_overrun_ticks": DEFAULT_BUDGET_OVERRUN_TICKS,
        }[key]


def test_the_monitor_floors_tick_counts_at_one() -> None:
    """Belt and braces: even handed a zero directly, the guard is a guard."""
    m = monitor(tilt_ticks=0, budget_ticks=-3)
    assert m.tilt_ticks == 1
    assert m.budget_ticks == 1


# ── The reason survives the process (story: "before the process exits") ─────────


def test_the_abort_record_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        walk_telemetry, "ABORT_FILE", str(tmp_path / "tnkr_walk_abort.json")
    )
    abort = monitor().abort("budget", "10 consecutive overruns, mean 13.4ms")

    walk_telemetry.write_abort(
        abort.code, abort.reason, abort.detail, abort.operator
    )
    record = walk_telemetry.read_abort()

    assert record["code"] == "POLICY_ABORTED"
    assert record["reason"] == "budget"
    assert "13.4ms" in record["detail"]
    assert record["operator"] == AbortMonitor.OPERATOR["budget"]


def test_no_abort_record_reads_as_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(walk_telemetry, "ABORT_FILE", str(tmp_path / "absent.json"))
    assert walk_telemetry.read_abort() is None


def test_clearing_the_abort_record_is_idempotent(tmp_path, monkeypatch) -> None:
    """The walk clears it at startup, so a previous abort is never read as this walk's —
    and a startup with nothing to clear must not raise."""
    monkeypatch.setattr(walk_telemetry, "ABORT_FILE", str(tmp_path / "a.json"))
    walk_telemetry.clear_abort()
    walk_telemetry.write_abort("POLICY_ABORTED", "tilt", "d", "o")
    walk_telemetry.clear_abort()
    assert walk_telemetry.read_abort() is None


def test_the_pose_snapshot_omits_the_envelope_key_for_a_builtin_walk(
    tmp_path, monkeypatch
) -> None:
    """Amendment A8 down to the bytes: a built-in walk writes what it wrote before the
    envelope existed."""
    monkeypatch.setattr(
        walk_telemetry, "TELEMETRY_FILE", str(tmp_path / "telemetry.json")
    )
    walk_telemetry.write_snapshot({"left_knee": 0.0})
    assert "envelope" not in walk_telemetry.read_snapshot()

    walk_telemetry.write_snapshot({"left_knee": 0.0}, envelope={"clamped": {}})
    assert walk_telemetry.read_snapshot()["envelope"] == {"clamped": {}}


# ── Amendment A8: the built-in policy never arms the aborts ─────────────────────


def test_builtin_policy_does_not_arm_the_aborts() -> None:
    assert is_armed(False, {}) is False


def _run_body() -> ast.FunctionDef:
    tree = ast.parse(WALK_SCRIPT.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLWalk"
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )


def test_every_abort_check_is_gated_on_the_monitor_existing() -> None:
    """Structural, because RLWalk needs a bus, an IMU and GPIO to construct. With the
    built-in policy the loop's overrun handling must stay the bare print it is today."""
    run = _run_body()
    found: list[str] = []

    def walk(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                walk(child, enclosing + [ast.unparse(child.test)])
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in ("check_tilt", "check_budget")
            ):
                found.append(" and ".join(enclosing))
            walk(child, enclosing)

    walk(run, [])
    assert len(found) == 2, f"expected one tilt and one budget check, got {found}"
    for guard in found:
        assert "self.abort_monitor is not None" in guard, guard


def test_the_abort_handler_precedes_the_generic_system_exit_handler() -> None:
    """PolicyAbort subclasses SystemExit, so ordering is what makes the reason get
    written. Swapping the two handlers would silently turn every abort back into "the
    process died"."""
    run = _run_body()
    tries = [n for n in ast.walk(run) if isinstance(n, ast.Try)]
    assert tries, "RLWalk.run has no try block"
    handlers = tries[0].handlers
    caught = [ast.unparse(h.type) if h.type else "bare" for h in handlers]

    assert caught[0] == "PolicyAbort", caught
    assert any("SystemExit" in c for c in caught[1:]), caught


def test_the_previous_abort_is_cleared_at_startup() -> None:
    """Otherwise a built-in walk started after a bad custom one reports the custom one's
    abort as its own."""
    run = _run_body()
    calls = [
        ast.unparse(n)
        for n in ast.walk(run)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "clear_abort"
    ]
    assert calls == ["local_telemetry.clear_abort()"], calls


# ── Cost ────────────────────────────────────────────────────────────────────────


def test_the_per_tick_checks_cost_under_a_tenth_of_a_millisecond() -> None:
    """Both guards run on every tick of a 20 ms budget, and one of them aborts when that
    budget is missed — so they must not be what misses it. Best of several runs: the
    question is what the code costs, not what a loaded CI box costs."""
    m = monitor()
    tilted = quat_from_pitch_roll(10.0)

    def tick():
        m.check_tilt(tilted)
        m.check_budget(0.001)

    per_tick = min(timeit.repeat(tick, repeat=5, number=500)) / 500
    assert per_tick < 0.1e-3, f"checks cost {per_tick * 1e3:.4f}ms of a 0.1ms budget"
