"""The supervised bench run: bounded, stoppable, torque off on every way out.

Story 4.3, architecture Decision 10. Three things are being defended here, in rising order
of how bad it is to get them wrong.

**Bench is a mode of the existing loop.** A second control loop would be a second, untested
safety envelope, and the whole argument for the bench being safe to watch is that the clamps
and aborts guarding it are the ones story 1.2 and 1.3 tested. ``test_the_walk_has_exactly_one
_control_loop`` and the AST tests beside it are that guarantee.

**Torque comes off on all five exits.** Timer, operator stop, abort, SIGTERM, unhandled
exception. These are tested by *running the real loop* against a stub self, rather than by
reading it, because the claim is about what happens and not about what the code says. The
walk script cannot be imported off a Pi -- ``raw_imu`` imports ``board``, which raises on any
machine Blinka does not recognise -- so ``walk_module`` loads it under a private name with
the six hardware wheels faked, and takes them back out afterwards.

**Nothing computes a pass.** "Did not abort" is not "walked well": a policy can hold every
joint inside every limit, keep every deadline, and produce a gait that faceplants the moment
it bears weight. The only judge is the operator, which is what Decision 1 accepted when the
sim gate was dropped, so ``test_nothing_in_the_runtime_records_a_pass_without_an_operator``
asserts that no code path in the server writes a passing verdict on its own.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import signal
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

import tnkr_server
from conftest import fake_fetch, spawned_argv_or_none, write_walk_script
from mini_bdx_runtime import bench, policy_store
from mini_bdx_runtime import walk_telemetry
from mini_bdx_runtime.envelope import AbortMonitor, PolicyAbort
from mini_bdx_runtime.policy_contract import ACT_DIM, OBS_DIM

REPO = Path(__file__).parent.parent
WALK_SCRIPT = REPO / "scripts" / "v2_rl_walk_mujoco.py"
SERVER_SCRIPT = REPO / "scripts" / "tnkr_server.py"


class Clock:
    """A monotonic clock a test drives by hand. Bench timing must not need real seconds."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def run(seconds=10.0, policy_id="9f2a", clock=None, stop=False) -> bench.BenchRun:
    return bench.BenchRun(
        seconds,
        policy_id,
        clock=clock or Clock(),
        stop_check=(lambda: stop) if isinstance(stop, bool) else stop,
    )


# ── The duration ────────────────────────────────────────────────────────────────


def test_the_default_bench_is_ten_seconds() -> None:
    assert bench.clamp_seconds(None) == 10.0
    assert bench.BenchRun().seconds == 10.0


@pytest.mark.parametrize(
    "asked, got",
    [
        (10.0, 10.0),
        (3.0, 3.0),
        (0.0, bench.MIN_BENCH_SECONDS),
        (-5.0, bench.MIN_BENCH_SECONDS),
        (1e9, bench.MAX_BENCH_SECONDS),
        (None, bench.DEFAULT_BENCH_SECONDS),
        ("nonsense", bench.DEFAULT_BENCH_SECONDS),
        (float("nan"), bench.DEFAULT_BENCH_SECONDS),
        (float("inf"), bench.MAX_BENCH_SECONDS),
    ],
)
def test_the_duration_is_clamped_not_refused(asked, got) -> None:
    """This arrives over an API with no authentication, so ``benchSeconds: 1e9`` is a value
    somebody can send. The answer to it is ten seconds of torque, not an unbounded run."""
    assert bench.clamp_seconds(asked) == got


def test_a_bench_always_has_an_upper_bound() -> None:
    """The property, not the number: whatever arrives, the run ends on its own."""
    for asked in (None, -1, 0, 1e30, "", float("nan")):
        assert bench.clamp_seconds(asked) <= bench.MAX_BENCH_SECONDS


# ── The mode flag ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value, expected", [("free", "free"), ("bench", "bench"), (None, "free")])
def test_parse_mode_accepts_the_two_modes(value, expected) -> None:
    assert bench.parse_mode(value) == expected


@pytest.mark.parametrize("value", ["Bench", "BENCH", "benchmark", "", "walk", "free "])
def test_an_unrecognised_mode_is_refused_rather_than_defaulted(value) -> None:
    """A caller who asked for a bench, was silently given free walking, and had the gate
    skipped is the one outcome this story cannot allow."""
    with pytest.raises(ValueError):
        bench.parse_mode(value)


# ── The five endings, as state ───────────────────────────────────────────────────


def test_the_timer_ends_the_run() -> None:
    clock = Clock()
    r = run(seconds=10.0, clock=clock)
    for _ in range(5):
        clock.advance(1.0)
        assert r.tick(0.01) is None
    clock.advance(5.0)
    assert r.tick(0.01) == bench.ENDED_TIMER
    assert r.report().ended == bench.ENDED_TIMER


def test_the_operator_stop_ends_the_run_on_the_next_tick() -> None:
    stopped = {"yes": False}
    r = run(seconds=10.0, stop=lambda: stopped["yes"])
    assert r.tick(0.01) is None
    stopped["yes"] = True
    assert r.tick(0.01) == bench.ENDED_OPERATOR


def test_a_stop_on_the_last_tick_reports_the_operator_not_the_timer() -> None:
    """Which ending it was decides whether a pass can be recorded, and an operator who hit
    stop because the duck was thrashing must not be handed a completed-run prompt."""
    clock = Clock()
    r = run(seconds=1.0, clock=clock, stop=True)
    clock.advance(5.0)  # the deadline has also passed
    assert r.tick(0.01) == bench.ENDED_OPERATOR


def test_an_abort_ends_the_run_and_carries_the_reason() -> None:
    r = run()
    abort = PolicyAbort("tilt", "pitch 71 deg for 8 ticks", "The robot fell over.")
    assert r.aborted(abort) == bench.ENDED_ABORT

    report = r.report()
    assert report.aborted is True
    assert report.ended == bench.ENDED_ABORT
    assert report.abort_reason == "tilt"
    assert report.abort_code == "POLICY_ABORTED"
    assert "71" in report.detail


def test_an_abort_outranks_an_ending_already_recorded() -> None:
    """A guard that trips on the same tick the deadline arrives has still fired."""
    r = run()
    r.end(bench.ENDED_TIMER, "completed")
    r.aborted(PolicyAbort("budget", "10 overruns", "Too slow."))
    assert r.report().ended == bench.ENDED_ABORT


def test_the_first_ordinary_ending_wins() -> None:
    """A bench that finished its deadline and then unwound through the signal handler kept
    walking for exactly as long as it was asked to. Relabelling it 'signal' would make a
    completed run unpassable."""
    r = run()
    r.end(bench.ENDED_TIMER, "completed")
    r.end(bench.ENDED_SIGNAL, "stopped")
    assert r.report().ended == bench.ENDED_TIMER


def test_ticks_stop_changing_the_ending_once_it_is_set() -> None:
    clock = Clock()
    r = run(seconds=1.0, clock=clock)
    clock.advance(2.0)
    assert r.tick(0.01) == bench.ENDED_TIMER
    assert r.tick(0.01) == bench.ENDED_TIMER
    assert r.ticks == 2, "a tick after the ending must still be counted"


# ── What the report carries ─────────────────────────────────────────────────────


def test_the_report_counts_ticks_and_averages_their_cost() -> None:
    r = run()
    r.tick(0.010)
    r.tick(0.020)
    report = r.report()
    assert report.ticks == 2
    assert report.mean_tick_ms == pytest.approx(15.0)


def test_a_run_with_no_ticks_reports_no_timing_rather_than_zero() -> None:
    """Zero milliseconds a tick would read as an impossibly fast policy. A bench that never
    ticked (the process died during startup) knows nothing about timing."""
    assert run().report().mean_tick_ms is None


def test_clamp_counts_are_summed_across_the_windows() -> None:
    """The envelope's counters are reset every second when the walk publishes telemetry, so
    a bench total has to be accumulated as the windows go by."""
    r = run()
    r.add_clamps({"head_pitch": 3, "left_knee": 1})
    r.add_clamps({"head_pitch": 2})
    report = r.report()
    assert report.clamped == {"head_pitch": 5, "left_knee": 1}
    assert report.clamp_events == 6


def test_clamp_counts_are_reported_not_judged() -> None:
    """A saturating policy is not automatically a failed bench: the clamp is doing its job,
    and whether the gait was any good is still the operator's call."""
    r = run()
    r.add_clamps({"head_pitch": 500})
    r.end(bench.ENDED_TIMER, "completed")
    assert r.report().as_dict()["passable"] is True


@pytest.mark.parametrize(
    "ending, passable",
    [
        (bench.ENDED_TIMER, True),
        (bench.ENDED_OPERATOR, True),
        (bench.ENDED_ABORT, False),
        (bench.ENDED_SIGNAL, False),
        (bench.ENDED_ERROR, False),
    ],
)
def test_only_a_watched_ending_can_be_passed(ending, passable) -> None:
    assert (ending in bench.PASSABLE_ENDINGS) is passable
    assert bench.BenchReport(ended=ending).as_dict()["passable"] is passable


def test_the_report_is_camel_case_for_studio() -> None:
    report = run().report()
    payload = report.as_dict()
    assert "policyId" in payload and "clampEvents" in payload and "meanTickMs" in payload
    json.dumps(payload)  # it has to survive being a response body


# ── The report file ─────────────────────────────────────────────────────────────


def test_the_report_survives_the_process_that_wrote_it() -> None:
    """Latched, like the abort file: the server only asks what happened after the walk has
    exited, so a freshness window would throw away the only record of the run."""
    r = run(seconds=4.0)
    r.tick(0.01)
    r.end(bench.ENDED_TIMER, "completed")
    bench.write_report(r.report())

    loaded = bench.read_report()
    assert loaded is not None
    assert loaded.ended == bench.ENDED_TIMER
    assert loaded.policy_id == "9f2a"
    assert loaded.ticks == 1


def test_no_report_reads_as_no_bench_run() -> None:
    assert bench.read_report() is None


@pytest.mark.parametrize("contents", ["", "{", "null", "[]", '"ok"', '{"ended":'])
def test_an_unreadable_report_reads_as_no_bench_run(contents) -> None:
    """Fail closed. A truncated report must not become a passable run -- a power cut during
    the write would otherwise clear the gate."""
    Path(bench.REPORT_FILE).write_text(contents)
    assert bench.read_report() is None


def test_a_report_with_a_mangled_ending_is_not_passable() -> None:
    Path(bench.REPORT_FILE).write_text(json.dumps({"ended": "tim", "policyId": "9f2a"}))
    loaded = bench.read_report()
    assert loaded is not None
    assert loaded.ended == bench.ENDED_ERROR
    assert loaded.as_dict()["passable"] is False


def test_a_report_with_a_mangled_clamp_map_still_loads() -> None:
    Path(bench.REPORT_FILE).write_text(
        json.dumps({"ended": "timer", "clamped": "lots", "ticks": "many"})
    )
    loaded = bench.read_report()
    assert loaded is not None
    assert loaded.clamped == {}
    assert loaded.ticks == 0


def test_clearing_a_report_that_is_not_there_is_not_an_error() -> None:
    bench.clear_report()
    bench.clear_report()
    assert bench.read_report() is None


# ── The stop flag ───────────────────────────────────────────────────────────────


def test_the_stop_flag_is_a_marker_and_is_idempotent() -> None:
    assert bench.stop_requested() is False
    bench.request_stop()
    bench.request_stop()
    assert bench.stop_requested() is True
    bench.clear_stop()
    bench.clear_stop()
    assert bench.stop_requested() is False


def test_a_stale_stop_request_does_not_end_the_next_bench_before_it_starts() -> None:
    """The failure this prevents: the operator holds the duck, presses start, and nothing
    ever moves because the previous run's stop flag was still on disk."""
    bench.request_stop()
    bench.clear_stop()
    r = bench.BenchRun(4.0, "9f2a")
    assert r.tick(0.01) is None


# ── The store's record of it ─────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "BEST_WALK_ONNX_2.onnx").write_bytes(b"builtin")
    return policy_store.PolicyStore(root=tmp_path / "policies", scripts_dir=scripts)


def install(store, onnx_specs, policy_id="9f2a", payload=b"model"):
    import hashlib

    store.fetch = fake_fetch(onnx_specs, payload=payload)
    result = store.install(
        policy_id, "https://example.test/m.onnx", hashlib.sha256(payload).hexdigest()
    )
    assert result.ok, result.detail
    return result


def test_a_policy_that_has_never_been_benched_is_gated(store, onnx_specs) -> None:
    install(store, onnx_specs)
    status = store.bench_status("9f2a")
    assert status["required"] is True
    assert status["passed"] is None


def test_the_builtin_never_requires_a_bench_run(store) -> None:
    """Decision 11: it is the policy every duck sold walks on. Gating it would gate the
    robot's own factory behaviour."""
    status = store.bench_status(policy_store.BUILTIN_ID)
    assert status["required"] is False
    assert status["exempt"] is True


@pytest.mark.parametrize("value", [None, "", "builtin", " builtin "])
def test_every_spelling_of_the_builtin_is_exempt(store, value) -> None:
    assert store.bench_status(value)["required"] is False


def test_a_pass_clears_the_gate_and_persists(store, onnx_specs) -> None:
    """Per policy id, on disk, so the gate is imposed once rather than every session."""
    install(store, onnx_specs)
    store.mark_bench("9f2a", True, "looked like walking")

    fresh = policy_store.PolicyStore(root=store.root, scripts_dir=store.scripts_dir)
    status = fresh.bench_status("9f2a")
    assert status["passed"] is True
    assert status["required"] is False
    assert status["reason"] == "looked like walking"
    assert status["at"] is not None


def test_a_failure_is_recorded_and_still_gated(store, onnx_specs) -> None:
    install(store, onnx_specs)
    status = store.mark_bench("9f2a", False, "it kicked itself in the head")
    assert status["passed"] is False
    assert status["required"] is True
    assert status["reason"] == "it kicked itself in the head"


def test_a_verdict_can_be_changed_by_running_the_bench_again(store, onnx_specs) -> None:
    install(store, onnx_specs)
    store.mark_bench("9f2a", False, "bad")
    assert store.bench_status("9f2a")["passed"] is False
    store.mark_bench("9f2a", True, "good, after a trim")
    assert store.bench_status("9f2a")["passed"] is True


def test_a_verdict_on_a_policy_that_is_not_installed_is_refused(store) -> None:
    with pytest.raises(policy_store.StoreError):
        store.mark_bench("9f2a", True)
    assert not (store.root / "9f2a").exists(), "a refusal created a phantom policy"


def test_a_verdict_on_the_builtin_is_accepted_and_stored_nowhere(store) -> None:
    """Benching the built-in is allowed -- Decision 10 makes it optional, not forbidden --
    and there is nothing for a verdict to unlock."""
    status = store.mark_bench(policy_store.BUILTIN_ID, True, "fine")
    assert status["required"] is False
    assert not (store.root / policy_store.BUILTIN_ID).exists()


@pytest.mark.parametrize("contents", ["", "{", "[]", '{"passed": "yes"}', '{"passed": 1}'])
def test_an_unreadable_bench_record_reads_as_never_benched(
    store, onnx_specs, contents
) -> None:
    """The direction the error has to fall. Wrongly requiring a bench costs ten seconds;
    wrongly skipping one puts an unwatched policy on a duck standing on its own feet."""
    install(store, onnx_specs)
    (store.root / "9f2a" / policy_store.BENCH_FILENAME).write_text(contents)
    assert store.bench_status("9f2a")["required"] is True


def test_a_pass_does_not_carry_over_to_different_weights(store, onnx_specs) -> None:
    """The verdict is about the bytes a person watched. Re-installing the same id with new
    content, or swapping the file on disk, means nobody has watched what is installed now."""
    install(store, onnx_specs)
    store.mark_bench("9f2a", True, "watched")
    assert store.bench_status("9f2a")["passed"] is True

    manifest_path = store.root / "9f2a" / policy_store.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    status = store.bench_status("9f2a")
    assert status["passed"] is None
    assert status["required"] is True
    assert "changed" in status["reason"]


def test_reinstalling_different_content_drops_the_verdict(store, onnx_specs) -> None:
    install(store, onnx_specs, payload=b"model-v1")
    store.mark_bench("9f2a", True, "watched v1")
    install(store, onnx_specs, payload=b"model-v2")
    assert store.bench_status("9f2a")["required"] is True


def test_an_id_that_is_not_a_directory_name_is_never_benched(store) -> None:
    assert store.bench_status("../../etc")["required"] is True


def test_the_policy_list_carries_the_gate(store, onnx_specs) -> None:
    """Studio decides whether Walk needs a bench first from the list it already polls."""
    install(store, onnx_specs)
    listed = {p["id"]: p for p in store.list()["policies"]}
    assert listed[policy_store.BUILTIN_ID]["bench"]["required"] is False
    assert listed["9f2a"]["bench"]["required"] is True

    store.mark_bench("9f2a", True, "watched")
    listed = {p["id"]: p for p in store.list()["policies"]}
    assert listed["9f2a"]["bench"]["required"] is False


def test_an_evicted_policy_loses_its_verdict(store, onnx_specs) -> None:
    """The record lives beside the model, so it goes when the model goes -- and a policy
    re-installed after eviction is one nobody has watched on this robot lately."""
    install(store, onnx_specs, policy_id="aaa", payload=b"a")
    store.mark_bench("aaa", True, "watched")
    for policy_id, payload in (("bbb", b"b"), ("ccc", b"c"), ("ddd", b"d")):
        install(store, onnx_specs, policy_id=policy_id, payload=payload)

    assert not (store.root / "aaa").exists(), "expected the LRU eviction to take aaa"
    assert store.bench_status("aaa")["required"] is True


# ── The walk loop, for real ──────────────────────────────────────────────────────
#
# The walk script cannot be imported on a machine that is not a Pi: raw_imu imports
# `board`, and Blinka raises on any platform it does not recognise. So the six hardware
# wheels are faked in sys.modules for the duration of the load, the module is executed
# under a private name, and everything the load added is taken back out -- no test that
# runs after this one sees a fake `board` or a half-imported mini_bdx_runtime submodule.

_HARDWARE_WHEELS = ("board", "busio", "digitalio", "pwmio", "adafruit_bno055", "pygame")


class _AnyAttribute(types.ModuleType):
    """A stand-in wheel. ``board.D22`` is read at import time by four of these modules, so
    the fake has to answer for any attribute rather than declare a pin map."""

    def __getattr__(self, name):  # pragma: no cover - exercised only via imports
        return f"<fake {self.__name__}.{name}>"


@pytest.fixture
def walk_module(monkeypatch, tmp_path):
    before = set(sys.modules)
    for name in _HARDWARE_WHEELS:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, _AnyAttribute(name))

    spec = importlib.util.spec_from_file_location("walk_under_test", WALK_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        spec.loader.exec_module(module)
        # The loop publishes a pose every tick; keep it out of /dev/shm.
        monkeypatch.setattr(
            walk_telemetry, "TELEMETRY_FILE", str(tmp_path / "telemetry.json")
        )
        monkeypatch.setattr(walk_telemetry, "ABORT_FILE", str(tmp_path / "abort.json"))
        # A bench of two seconds would be two seconds of suite time.
        monkeypatch.setattr(bench, "MIN_BENCH_SECONDS", 0.0)
        yield module
    finally:
        # run() installs its own SIGTERM handler and never restores it.
        signal.signal(signal.SIGTERM, original_sigterm)
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)


JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


class FakeHwi:
    def __init__(self) -> None:
        self.joints = {name: i for i, name in enumerate(JOINTS)}
        self.turned_off = 0
        self.commanded = 0

    def set_position_all(self, action_dict) -> None:
        self.commanded += 1

    def turn_off(self) -> None:
        self.turned_off += 1


class StubWalk:
    """Everything ``RLWalk.run`` touches, and nothing else.

    A stub rather than a constructed RLWalk because ``__init__`` opens a servo bus, an I2C
    IMU and two GPIO pins. ``run`` is called unbound -- ``RLWalk.run(stub, ...)`` -- so the
    code under test is the real 50 Hz loop, byte for byte.
    """

    def __init__(self, *, control_freq=500.0, abort_monitor=None, on_tick=None) -> None:
        self.hwi = FakeHwi()
        self.control_freq = control_freq
        self.num_dofs = ACT_DIM
        self.action_scale = 0.25
        self.init_pos = [0.0] * ACT_DIM
        self.motor_targets = np.zeros(ACT_DIM)
        self.prev_motor_targets = np.zeros(ACT_DIM)
        self.last_action = np.zeros(ACT_DIM)
        self.last_last_action = np.zeros(ACT_DIM)
        self.last_last_last_action = np.zeros(ACT_DIM)
        self.last_commands = [0.0] * 7
        self.imitation_phase = np.zeros(2)
        self.imitation_i = 0
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = 0.0
        self.PRM = types.SimpleNamespace(nb_steps_in_period=50)
        self.policy = types.SimpleNamespace(infer=lambda obs: np.zeros(ACT_DIM))
        self.action_filter = None
        self.commands = False
        self.paused = False
        self.save_obs = False
        self.replay_obs = None
        self.cloud_publisher = None
        self.cloud_command_receiver = None
        self.duck_config = types.SimpleNamespace(
            antennas=False, eyes=False, projector=False, speaker=False
        )
        self.feet_contacts = types.SimpleNamespace(stop=lambda: None)
        self.telemetry_joint_names = list(JOINTS)
        self.last_dof_pos = np.zeros(ACT_DIM)
        self.last_imu_data = None
        self.envelope = None
        self.envelope_prev_targets = None
        self.envelope_telemetry = None
        self._envelope_flushed_at = 0.0
        self.abort_monitor = abort_monitor
        self.bench = None
        self.ticks = 0
        self._on_tick = on_tick
        self.tilt_deg = 0.0

    def get_obs(self):
        self.ticks += 1
        pitch = math.radians(self.tilt_deg) / 2.0
        self.last_imu_data = {
            "gyro": [0.0, 0.0, 0.0],
            "accelero": [0.0, 0.0, 0.0],
            "quaternion": [math.cos(pitch), 0.0, math.sin(pitch), 0.0],
            "quaternion_t": time.monotonic(),
        }
        self.last_dof_pos = np.zeros(ACT_DIM)
        if self._on_tick is not None:
            self._on_tick(self)
        return np.zeros(OBS_DIM)


def bench_run(module, stub, seconds=0.05, **kwargs):
    return module.RLWalk.run(
        stub, mode=bench.MODE_BENCH, bench_seconds=seconds, policy_id="9f2a", **kwargs
    )


# ── the five exit paths ──────────────────────────────────────────────────────────


def test_the_timer_ends_the_bench_and_cuts_torque(walk_module) -> None:
    stub = StubWalk()

    report = bench_run(walk_module, stub, seconds=0.05)

    assert report.ended == bench.ENDED_TIMER
    assert report.ticks > 0
    assert stub.hwi.turned_off == 1
    assert bench.read_report().ended == bench.ENDED_TIMER


def test_an_operator_stop_ends_the_bench_and_cuts_torque(walk_module) -> None:
    """Stoppable instantly: a flag the loop stats every tick, so the walk ends its own run
    and still writes the report -- which is what a killed process cannot do."""
    stub = StubWalk(on_tick=lambda s: bench.request_stop() if s.ticks == 3 else None)

    report = bench_run(walk_module, stub, seconds=30.0)

    assert report.ended == bench.ENDED_OPERATOR
    assert report.ticks <= 5, "the stop was not acted on within a tick or two"
    assert stub.hwi.turned_off == 1
    assert bench.read_report().ended == bench.ENDED_OPERATOR


def test_an_abort_during_the_bench_fails_it_and_cuts_torque(walk_module) -> None:
    """Story 1.3's guards are the one judge that does not need a person."""
    monitor = AbortMonitor(tilt_limit_deg=45.0, tilt_ticks=2, budget_s=1.0, budget_ticks=5)
    stub = StubWalk(abort_monitor=monitor)
    stub.tilt_deg = 80.0

    report = bench_run(walk_module, stub, seconds=30.0)

    assert report.ended == bench.ENDED_ABORT
    assert report.aborted is True
    assert report.abort_reason == "tilt"
    assert report.as_dict()["passable"] is False
    assert stub.hwi.turned_off == 1
    written = bench.read_report()
    assert written.aborted is True and written.abort_code == "POLICY_ABORTED"


def test_sigterm_ends_the_bench_and_cuts_torque(walk_module) -> None:
    """The real signal, through the handler ``run`` installs. This is the path
    /api/walk/stop takes, and it is the mechanism every other ending reuses."""

    def kill(stub):
        if stub.ticks == 3:
            os.kill(os.getpid(), signal.SIGTERM)

    stub = StubWalk(on_tick=kill)

    report = bench_run(walk_module, stub, seconds=30.0)

    assert report.ended == bench.ENDED_SIGNAL
    assert stub.hwi.turned_off == 1
    assert bench.read_report().ended == bench.ENDED_SIGNAL


def test_an_unhandled_exception_still_cuts_torque_and_still_reports(walk_module) -> None:
    """It propagates -- the exit code and the traceback stay what they were -- but torque
    is off and the run is on record as an error rather than as a power cut."""

    def explode(stub):
        if stub.ticks == 3:
            raise RuntimeError("the servo bus fell off")

    stub = StubWalk(on_tick=explode)

    with pytest.raises(RuntimeError, match="fell off"):
        bench_run(walk_module, stub, seconds=30.0)

    assert stub.hwi.turned_off == 1
    written = bench.read_report()
    assert written.ended == bench.ENDED_ERROR
    assert written.as_dict()["passable"] is False
    assert "RuntimeError" in written.detail


def test_a_free_walk_writes_no_bench_report(walk_module) -> None:
    """Amendment A8's shape: with no bench asked for, the loop does no bench bookkeeping
    and leaves no report for a later verdict to attach itself to."""

    def stop(stub):
        if stub.ticks == 3:
            raise SystemExit(0)

    stub = StubWalk(on_tick=stop)

    report = walk_module.RLWalk.run(stub, mode=bench.MODE_FREE)

    assert report.mode == bench.MODE_FREE
    assert report.ended == bench.ENDED_SIGNAL
    assert stub.hwi.turned_off == 1
    assert bench.read_report() is None


def test_a_bench_clears_a_stale_report_before_it_starts(walk_module) -> None:
    """A verdict must never be recordable against a run that did not just happen."""
    Path(bench.REPORT_FILE).write_text(
        json.dumps({"ended": "timer", "policyId": "old", "ticks": 500})
    )
    stub = StubWalk()

    bench_run(walk_module, stub, seconds=0.05)

    assert bench.read_report().policy_id == "9f2a"


def test_a_stale_stop_flag_does_not_end_a_bench_immediately(walk_module) -> None:
    bench.request_stop()
    stub = StubWalk()

    report = bench_run(walk_module, stub, seconds=0.05)

    assert report.ended == bench.ENDED_TIMER
    assert report.ticks > 1


def test_the_bench_commands_the_servos_the_whole_time(walk_module) -> None:
    """Torque ON for the run: the operator is watching a gait, not a limp duck."""
    stub = StubWalk()

    report = bench_run(walk_module, stub, seconds=0.05)

    assert stub.hwi.commanded == report.ticks


def test_the_bench_does_not_change_what_arms_the_envelope(walk_module) -> None:
    """Arming is derived from the artifact (``envelope.is_armed``), not from the mode. A
    bench of the built-in runs the built-in's path; a bench of a custom policy is guarded
    because the policy is custom, not because it is a bench."""
    from mini_bdx_runtime.envelope import is_armed

    assert is_armed(False, "scripts/BEST_WALK_ONNX_2.onnx", env={}) is False
    assert is_armed(False, "/home/pi/.tnkr/policies/9f2a/model.onnx", env={}) is True

    calls = [
        node
        for node in ast.walk(ast.parse(WALK_SCRIPT.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_armed"
    ]
    assert len(calls) == 1, "is_armed is no longer consulted exactly once"
    arguments = [ast.unparse(a) for a in calls[0].args] + [
        ast.unparse(k.value) for k in calls[0].keywords
    ]
    assert arguments == ["custom_policy", "onnx_model_path"], arguments


# ── the loop's shape ─────────────────────────────────────────────────────────────


def _run_body() -> ast.FunctionDef:
    tree = ast.parse(WALK_SCRIPT.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLWalk")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")


def test_the_walk_has_exactly_one_control_loop() -> None:
    """Bench is a MODE of the loop, never a second loop. A second one would be a second
    safety envelope, and only one of them has been tested."""
    loops = [n for n in ast.walk(_run_body()) if isinstance(n, (ast.While, ast.For))]
    assert len(loops) == 1, f"expected one loop in RLWalk.run, found {len(loops)}"

    tree = ast.parse(WALK_SCRIPT.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLWalk")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert not (methods - {"__init__", "get_obs", "_envelope_snapshot", "start",
                          "get_phase_frequency_factor", "run"}), (
        f"a new method appeared on RLWalk: {methods}. If it is a bench loop, it must not be"
    )


def test_every_bench_check_is_gated_on_the_bench_existing() -> None:
    """With no bench asked for, the loop must run the instructions it ran before this story
    existed (amendment A8's rule, applied to the mode flag)."""
    run_body = _run_body()
    found: list[str] = []

    def walk(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                walk(child, enclosing + [ast.unparse(child.test)])
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Attribute)
                and child.func.value.attr == "bench"
            ):
                found.append(" and ".join(enclosing))
            walk(child, enclosing)

    walk(run_body, [])
    assert found, "the loop never advances the bench"
    for guard in found:
        assert "self.bench is not None" in guard, guard


def test_the_bench_ends_the_loop_with_a_break_not_an_exception() -> None:
    """The deadline arriving is not a failure. Raising would put a completed bench in the
    same bucket as an abort in every log that caught it."""
    for node in ast.walk(_run_body()):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "ended is not None" not in test:
            continue
        assert any(isinstance(s, ast.Break) for s in node.body), ast.unparse(node)
        return
    raise AssertionError("the bench's ending is never acted on")


def test_the_torque_off_and_the_report_are_both_in_the_one_finally() -> None:
    """Five exits, one teardown. The report is written after the torque is cut, because a
    duck being driven by a policy that just failed is not waiting on bookkeeping."""
    tries = [n for n in ast.walk(_run_body()) if isinstance(n, ast.Try) and n.finalbody]
    assert len(tries) == 1, "RLWalk.run should have exactly one try/finally"
    body = ast.unparse(tries[0].finalbody)
    assert "self.hwi.turn_off()" in body
    assert "write_report" in body
    assert body.index("turn_off") < body.index("write_report")

    handlers = [ast.unparse(h.type) if h.type else "bare" for h in tries[0].handlers]
    assert "PolicyAbort" in handlers[0]
    assert any("SystemExit" in h for h in handlers)
    assert any(h.endswith("Exception") for h in handlers)


def test_the_unhandled_exception_handler_re_raises() -> None:
    """Swallowing it would change the walk's exit code, and the server reads that to tell a
    crash from a clean stop."""
    tries = [n for n in ast.walk(_run_body()) if isinstance(n, ast.Try) and n.finalbody]
    for handler in tries[0].handlers:
        name = ast.unparse(handler.type) if handler.type else "bare"
        if name.endswith("Exception") and "System" not in name:
            assert any(isinstance(s, ast.Raise) for s in ast.walk(handler)), name
            return
    raise AssertionError("no handler for an unhandled exception")


# ── nothing computes a verdict ───────────────────────────────────────────────────


def test_nothing_in_the_runtime_records_a_pass_without_an_operator() -> None:
    """The one assertion this story would be worthless without.

    Decision 1 dropped the sim gate and accepted that a person watching the duck is the
    only judge of a gait. A heuristic that turned "ten seconds without an abort" into a
    pass would put that judgement back into the software while looking like a feature. So:
    every ``mark_bench`` call in the server is checked, and only the verdict endpoint --
    the one thing an operator's answer arrives at -- may pass anything other than a
    literal ``False``.
    """
    tree = ast.parse(SERVER_SCRIPT.read_text())
    calls: list[tuple[str, str]] = []

    def visit(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "mark_bench"
            ):
                calls.append((enclosing, ast.unparse(child)))
            visit(child, enclosing)

    visit(tree, "<module>")
    assert calls, "the server never records a bench verdict"
    for where, call in calls:
        if where == "bench_verdict":
            continue
        assert "False" in call, (
            f"{where}() records a bench verdict that is not a literal failure: {call}. "
            "Only an operator may pass a policy."
        )


def test_the_report_carries_no_pass_field_at_all() -> None:
    """Not a "verdict: unknown" placeholder either. The report says what happened; the
    store says what the operator decided, and there is exactly one writer of that."""
    payload = bench.BenchReport().as_dict()
    assert "passed" not in payload
    assert "verdict" not in payload


# ── the HTTP surface ─────────────────────────────────────────────────────────────


@pytest.fixture
def api(client, tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "BEST_WALK_ONNX_2.onnx").write_bytes(b"builtin")
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", scripts)
    monkeypatch.setattr(tnkr_server, "POLICY_ROOT", tmp_path / "server-policies")
    monkeypatch.setattr(tnkr_server.platform, "machine", lambda: "aarch64")
    write_walk_script(scripts, ARGV_DUMP)
    return client


ARGV_DUMP = (
    "import json, pathlib, sys\n"
    "pathlib.Path('argv.json').write_text(json.dumps(sys.argv))\n"
    "import time; time.sleep(30)\n"
)


def api_install(api, onnx_specs, monkeypatch, policy_id="9f2a"):
    import hashlib

    payload = f"model-{policy_id}".encode()
    monkeypatch.setattr(
        tnkr_server, "POLICY_FETCH", fake_fetch(onnx_specs, payload=payload)
    )
    response = api.post(
        "/api/policy/install",
        json={
            "id": policy_id,
            "url": "https://example.test/m.onnx",
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    assert response.status_code == 200, response.text
    return response


def spawned_argv(scripts_dir, timeout=10.0):
    deadline = time.monotonic() + timeout
    path = Path(scripts_dir) / "argv.json"
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                pass
        time.sleep(0.02)
    raise AssertionError("the walk script never recorded its argv")


def value_after(argv, flag):
    return argv[argv.index(flag) + 1]


def test_free_walking_a_policy_nobody_has_watched_is_refused(api, onnx_specs, monkeypatch):
    """The gate. A policy that has never completed a bench run on THIS robot cannot be
    started in free-walk mode -- checked on the robot, because the robot is the only place
    a check cannot be skipped by a caller (amendment A1)."""
    api_install(api, onnx_specs, monkeypatch)

    response = api.post("/api/walk/start", json={"policyId": "9f2a"})

    assert response.status_code == 409
    assert response.json()["code"] == bench.POLICY_BENCH_REQUIRED
    assert not tnkr_server.is_walking()
    assert spawned_argv_or_none(tnkr_server.SCRIPTS_DIR) is None


def test_the_gate_costs_no_torque_on_a_walk_already_running(api, onnx_specs, monkeypatch):
    """Same rule as the unknown-policy 404: a refusal must not stop the walk in progress,
    because stopping it cuts torque on a duck mid-stride for a request that starts nothing."""
    api_install(api, onnx_specs, monkeypatch)
    assert api.post("/api/walk/start", json={"sessionToken": "a"}).status_code == 200
    spawned_argv(tnkr_server.SCRIPTS_DIR)

    refused = api.post(
        "/api/walk/start", json={"sessionToken": "b", "policyId": "9f2a"}
    )

    assert refused.status_code == 409
    assert tnkr_server.is_walking(), "the gate killed the walk that was already running"


def test_the_builtin_is_never_gated(api):
    """Decision 11: every duck in the field keeps walking exactly as it does now."""
    response = api.post("/api/walk/start", json={})

    assert response.status_code == 200
    assert response.json()["policyId"] == policy_store.BUILTIN_ID
    assert value_after(spawned_argv(tnkr_server.SCRIPTS_DIR), "--onnx_model_path").endswith(
        "BEST_WALK_ONNX_2.onnx"
    )


def test_a_policy_dropped_into_scripts_is_not_gated_but_is_armed(api):
    """The deliberate hole, asserted so it stays a decision rather than becoming a bug.

    A file copied straight into ``scripts/`` is resolved by the built-in's glob, so it has
    no store id -- and a verdict can only be recorded against a store id, so gating it
    would break that workflow with no way out. It is still guarded: arming goes by file
    name, so anything that is not one of the two policies this repo ships runs with every
    clamp and abort on.
    """
    from mini_bdx_runtime.envelope import is_armed

    (tnkr_server.SCRIPTS_DIR / "BEST_WALK_ONNX_2.onnx").unlink()
    (tnkr_server.SCRIPTS_DIR / "somebodys-download.onnx").write_bytes(b"x")

    assert api.post("/api/walk/start", json={}).status_code == 200
    loaded = value_after(spawned_argv(tnkr_server.SCRIPTS_DIR), "--onnx_model_path")

    assert loaded.endswith("somebodys-download.onnx")
    assert is_armed(False, loaded, env={}) is True


def test_a_bench_run_is_not_gated_on_itself(api, onnx_specs, monkeypatch):
    api_install(api, onnx_specs, monkeypatch)

    response = api.post(
        "/api/walk/start", json={"policyId": "9f2a", "mode": "bench"}
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "bench"


def test_bench_mode_spawns_the_one_walk_script_with_a_flag(api, onnx_specs, monkeypatch):
    """The whole runtime half of this story, in one assertion: the same script, the same
    envelope arguments, plus a mode and a deadline."""
    api_install(api, onnx_specs, monkeypatch)

    api.post(
        "/api/walk/start",
        json={"policyId": "9f2a", "mode": "bench", "benchSeconds": 12},
    )
    argv = spawned_argv(tnkr_server.SCRIPTS_DIR)

    assert argv[0].endswith("v2_rl_walk_mujoco.py")
    assert value_after(argv, "--mode") == "bench"
    assert value_after(argv, "--bench_seconds") == "12"
    assert value_after(argv, "--policy_id") == "9f2a"
    assert "--custom_policy" in argv, "the bench must run inside the safety envelope"


def test_a_free_walk_is_spawned_exactly_as_it_was_before(api, onnx_specs, monkeypatch):
    """A duck that never benches anything runs the command line it ran before story 4.3."""
    api_install(api, onnx_specs, monkeypatch)
    tnkr_server.get_policy_store().mark_bench("9f2a", True, "watched")

    api.post("/api/walk/start", json={"policyId": "9f2a"})
    argv = spawned_argv(tnkr_server.SCRIPTS_DIR)

    assert "--mode" not in argv
    assert "--bench_seconds" not in argv
    assert "--policy_id" not in argv


def test_an_absurd_bench_duration_is_clamped_at_the_robot(api, onnx_specs, monkeypatch):
    api_install(api, onnx_specs, monkeypatch)

    api.post(
        "/api/walk/start",
        json={"policyId": "9f2a", "mode": "bench", "benchSeconds": 1e9},
    )

    seconds = float(value_after(spawned_argv(tnkr_server.SCRIPTS_DIR), "--bench_seconds"))
    assert seconds == bench.MAX_BENCH_SECONDS


def test_an_unknown_mode_is_a_400_and_starts_nothing(api):
    response = api.post("/api/walk/start", json={"mode": "benchmark"})

    assert response.status_code == 400
    assert not tnkr_server.is_walking()


def test_a_passed_bench_lets_the_policy_free_walk(api, onnx_specs, monkeypatch):
    """The end-to-end shape of the story: gated, benched, judged, walking."""
    api_install(api, onnx_specs, monkeypatch)
    assert api.post("/api/walk/start", json={"policyId": "9f2a"}).status_code == 409

    assert api.post(
        "/api/walk/start", json={"policyId": "9f2a", "mode": "bench"}
    ).status_code == 200
    spawned_argv(tnkr_server.SCRIPTS_DIR)
    with tnkr_server._walk_lock:
        tnkr_server.stop_walk_process()
    bench.write_report(
        bench.BenchReport(policy_id="9f2a", ended=bench.ENDED_TIMER, ticks=500)
    )

    verdict = api.post(
        "/api/bench/verdict",
        json={"policyId": "9f2a", "passed": True, "reason": "looked like walking"},
    )

    assert verdict.status_code == 200
    assert verdict.json()["bench"]["passed"] is True
    (tnkr_server.SCRIPTS_DIR / "argv.json").unlink()
    assert api.post("/api/walk/start", json={"policyId": "9f2a"}).status_code == 200


def test_read_bench_before_anything_ran(api):
    body = api.get("/api/bench").json()
    assert body["running"] is False
    assert body["report"] is None
    assert body["defaultSeconds"] == bench.DEFAULT_BENCH_SECONDS


def test_read_bench_while_one_is_running(api, onnx_specs, monkeypatch):
    api_install(api, onnx_specs, monkeypatch)
    api.post("/api/walk/start", json={"policyId": "9f2a", "mode": "bench"})
    spawned_argv(tnkr_server.SCRIPTS_DIR)

    body = api.get("/api/bench").json()

    assert body["running"] is True
    assert body["policyId"] == "9f2a"
    assert body["stopRequested"] is False


def test_a_free_walk_is_not_a_bench_run(api, onnx_specs, monkeypatch):
    api_install(api, onnx_specs, monkeypatch)
    tnkr_server.get_policy_store().mark_bench("9f2a", True, "watched")
    api.post("/api/walk/start", json={"policyId": "9f2a"})
    spawned_argv(tnkr_server.SCRIPTS_DIR)

    assert api.get("/api/bench").json()["running"] is False
    assert api.post("/api/bench/stop").status_code == 409


def test_stopping_a_bench_sets_the_flag_the_loop_reads(api, onnx_specs, monkeypatch):
    api_install(api, onnx_specs, monkeypatch)
    api.post("/api/walk/start", json={"policyId": "9f2a", "mode": "bench"})
    spawned_argv(tnkr_server.SCRIPTS_DIR)

    response = api.post("/api/bench/stop")

    assert response.status_code == 200
    assert response.json()["stopRequested"] is True
    assert bench.stop_requested() is True
    assert tnkr_server.is_walking(), "stopping the bench must not kill the process itself"


def test_the_ordinary_stop_still_kills_a_bench_run(api, onnx_specs, monkeypatch):
    """The E-stop path is unaffected: /api/walk/stop terminates the process, bench or not,
    and the walk's SIGTERM handler cuts torque on the way out."""
    api_install(api, onnx_specs, monkeypatch)
    api.post("/api/walk/start", json={"policyId": "9f2a", "mode": "bench"})
    spawned_argv(tnkr_server.SCRIPTS_DIR)

    assert api.post("/api/walk/stop").json()["success"] is True

    assert not tnkr_server.is_walking()
    assert api.get("/api/bench").json()["running"] is False


def test_stopping_when_nothing_is_running_is_refused(api):
    assert api.post("/api/bench/stop").status_code == 409


def test_starting_a_bench_clears_a_previous_stop_request(api, onnx_specs, monkeypatch):
    api_install(api, onnx_specs, monkeypatch)
    bench.request_stop()

    api.post("/api/walk/start", json={"policyId": "9f2a", "mode": "bench"})
    spawned_argv(tnkr_server.SCRIPTS_DIR)

    assert bench.stop_requested() is False


def test_a_verdict_with_no_bench_run_is_refused(api, onnx_specs, monkeypatch):
    """Otherwise the gate could be cleared without a duck ever having moved."""
    api_install(api, onnx_specs, monkeypatch)

    response = api.post("/api/bench/verdict", json={"policyId": "9f2a", "passed": True})

    assert response.status_code == 409
    assert tnkr_server.get_policy_store().bench_status("9f2a")["required"] is True


def test_a_verdict_for_a_different_policy_is_refused(api, onnx_specs, monkeypatch):
    api_install(api, onnx_specs, monkeypatch)
    api_install(api, onnx_specs, monkeypatch, policy_id="1c04")
    bench.write_report(bench.BenchReport(policy_id="1c04", ended=bench.ENDED_TIMER))

    response = api.post("/api/bench/verdict", json={"policyId": "9f2a", "passed": True})

    assert response.status_code == 409
    assert tnkr_server.get_policy_store().bench_status("9f2a")["required"] is True


@pytest.mark.parametrize("ending", [bench.ENDED_ABORT, bench.ENDED_SIGNAL, bench.ENDED_ERROR])
def test_a_pass_on_a_run_that_was_not_watched_to_the_end_is_refused(
    api, onnx_specs, monkeypatch, ending
):
    api_install(api, onnx_specs, monkeypatch)
    bench.write_report(bench.BenchReport(policy_id="9f2a", ended=ending))

    response = api.post("/api/bench/verdict", json={"policyId": "9f2a", "passed": True})

    assert response.status_code == 409
    assert tnkr_server.get_policy_store().bench_status("9f2a")["required"] is True


@pytest.mark.parametrize("ending", [bench.ENDED_ABORT, bench.ENDED_SIGNAL, bench.ENDED_ERROR])
def test_a_failure_is_accepted_on_any_ending(api, onnx_specs, monkeypatch, ending):
    """The operator saying "no, something is wrong" is the entire point of the feature, and
    it must be recordable however the run stopped."""
    api_install(api, onnx_specs, monkeypatch)
    bench.write_report(bench.BenchReport(policy_id="9f2a", ended=ending))

    response = api.post(
        "/api/bench/verdict",
        json={"policyId": "9f2a", "passed": False, "reason": "it kicked its own head"},
    )

    assert response.status_code == 200
    status = response.json()["bench"]
    assert status["passed"] is False and status["required"] is True


def test_the_operator_can_fail_a_run_they_stopped_early(api, onnx_specs, monkeypatch):
    api_install(api, onnx_specs, monkeypatch)
    bench.write_report(
        bench.BenchReport(policy_id="9f2a", ended=bench.ENDED_OPERATOR, ticks=90)
    )

    response = api.post(
        "/api/bench/verdict",
        json={"policyId": "9f2a", "passed": False, "reason": "one leg dragged"},
    )

    assert response.status_code == 200
    assert response.json()["bench"]["reason"] == "one leg dragged"


def test_a_verdict_on_a_policy_the_robot_does_not_have(api):
    bench.write_report(bench.BenchReport(policy_id="ghost", ended=bench.ENDED_TIMER))

    response = api.post("/api/bench/verdict", json={"policyId": "ghost", "passed": True})

    assert response.status_code == 502
    assert response.json()["code"] == policy_store.POLICY_INSTALL_FAILED


def test_a_verdict_on_the_builtin_is_accepted(api):
    """Benching the built-in is allowed but never required."""
    bench.write_report(
        bench.BenchReport(policy_id=policy_store.BUILTIN_ID, ended=bench.ENDED_TIMER)
    )

    response = api.post(
        "/api/bench/verdict", json={"policyId": policy_store.BUILTIN_ID, "passed": True}
    )

    assert response.status_code == 200
    assert response.json()["bench"]["required"] is False


def test_an_abort_during_a_bench_is_filed_as_a_failure(api, onnx_specs, monkeypatch):
    """An abort fails the bench without asking anyone: the envelope has already answered
    the question the operator was about to be asked."""
    api_install(api, onnx_specs, monkeypatch)
    session = tnkr_server.WalkSession(
        proc=types.SimpleNamespace(poll=lambda: 0, returncode=0),
        session_token=None,
        cloud_streaming=False,
        started_at=time.monotonic(),
        policy_id="9f2a",
        mode=bench.MODE_BENCH,
    )
    bench.write_report(
        bench.BenchReport(
            policy_id="9f2a",
            ended=bench.ENDED_ABORT,
            aborted=True,
            abort_reason="tilt",
            abort_code="POLICY_ABORTED",
        )
    )

    tnkr_server._finalize_bench(session)

    status = tnkr_server.get_policy_store().bench_status("9f2a")
    assert status["passed"] is False
    assert status["required"] is True
    assert status["reason"] == "tilt"


def test_a_clean_bench_run_records_nothing_on_its_own(api, onnx_specs, monkeypatch):
    """Fail closed. Ten seconds without an abort is not a pass -- if the operator never
    answers, the policy stays gated."""
    api_install(api, onnx_specs, monkeypatch)
    session = tnkr_server.WalkSession(
        proc=types.SimpleNamespace(poll=lambda: 0, returncode=0),
        session_token=None,
        cloud_streaming=False,
        started_at=time.monotonic(),
        policy_id="9f2a",
        mode=bench.MODE_BENCH,
    )
    bench.write_report(
        bench.BenchReport(policy_id="9f2a", ended=bench.ENDED_TIMER, ticks=500)
    )

    tnkr_server._finalize_bench(session)

    assert tnkr_server.get_policy_store().bench_status("9f2a")["required"] is True


def test_a_bench_that_died_before_reporting_leaves_the_policy_gated(
    api, onnx_specs, monkeypatch
):
    """SIGKILL, an OOM kill, a power cut: no report, so no verdict, so still gated."""
    api_install(api, onnx_specs, monkeypatch)
    session = tnkr_server.WalkSession(
        proc=types.SimpleNamespace(poll=lambda: -9, returncode=-9),
        session_token=None,
        cloud_streaming=False,
        started_at=time.monotonic(),
        policy_id="9f2a",
        mode=bench.MODE_BENCH,
    )

    tnkr_server._finalize_bench(session)

    assert tnkr_server.get_policy_store().bench_status("9f2a")["required"] is True
    assert api.post(
        "/api/bench/verdict", json={"policyId": "9f2a", "passed": True}
    ).status_code == 409


def test_polling_the_bench_does_not_burn_telemetry(api, captured):
    """It is polled for the whole run and again while the prompt is on screen. Same
    reasoning as /api/policy and /api/state."""
    api.get("/api/bench")
    assert not [e for e in captured if e["properties"].get("endpoint") == "/api/bench"]


def test_the_verdict_telemetry_never_names_the_policy(api, onnx_specs, monkeypatch, captured):
    api_install(api, onnx_specs, monkeypatch)
    bench.write_report(bench.BenchReport(policy_id="9f2a", ended=bench.ENDED_TIMER))

    api.post("/api/bench/verdict", json={"policyId": "9f2a", "passed": True})

    event = next(e for e in captured if e["properties"]["endpoint"].endswith("/verdict"))
    assert "9f2a" not in json.dumps(event, default=str)
    assert event["properties"]["passed"] is True
