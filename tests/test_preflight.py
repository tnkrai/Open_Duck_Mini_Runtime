"""Preflight checks against faked hardware.

Fakes are duck-typed SimpleNamespace-style objects, matching the suite's convention in
``test_hwi_adapter.py``. No robot, no bus, no I2C.
"""

from __future__ import annotations

import math

import pytest

from mini_bdx_runtime.preflight import (
    MAX_PLAUSIBLE_POSITION_RAD,
    MAX_RESTING_TILT_DEG,
    check_feet,
    check_imu,
    check_joints,
    check_offsets,
    run_preflight,
)

JOINTS = {
    "left_hip_yaw": 20,
    "left_knee": 23,
    "neck_pitch": 30,
    "right_hip_yaw": 10,
    "right_knee": 13,
}


class FakeIo:
    """Stands in for the rustypot handle. ``dead`` names servo ids that raise."""

    def __init__(self, positions: dict[int, float] | None = None, dead: set[int] | None = None):
        self._positions = positions or {}
        self._dead = dead or set()

    def read_present_position(self, ids):
        (servo_id,) = ids
        if servo_id in self._dead:
            raise OSError(f"timeout on id {servo_id}")
        return [self._positions.get(servo_id, 0.0)]


class FakeHwi:
    _IO_ATTEMPTS = 2
    _IO_RETRY_DELAY = 0

    def __init__(self, io: FakeIo, joints: dict[str, int] | None = None):
        self.io = io
        self.joints = dict(joints or JOINTS)

    def _io_retry(self, fn, joint_name, op):
        """Mirrors HWI._io_retry: retries OSError, then raises naming the joint."""
        last = None
        for attempt in range(self._IO_ATTEMPTS):
            try:
                return fn()
            except OSError as exc:
                last = exc
        raise OSError(
            f"{op} failed for '{joint_name}' (id {self.joints[joint_name]}): {last}"
        )


class FakeImu:
    def __init__(self, quaternion=(1.0, 0.0, 0.0, 0.0), raises=False, data=...):
        self._q = quaternion
        self._raises = raises
        self._data = data

    def get_data(self):
        if self._raises:
            raise RuntimeError("i2c bus error")
        if self._data is not ...:
            return self._data
        return {"quaternion": list(self._q), "gyro": [0, 0, 0], "accelero": [0, 0, 9.8]}


class FakeFeet:
    def __init__(self, contacts=(False, False), raises=False):
        self._c = contacts
        self._raises = raises

    def get(self):
        if self._raises:
            raise RuntimeError("gpio not available")
        return list(self._c) if self._c is not None else None


class FakeConfig:
    def __init__(self, offsets=..., imu_upside_down=False, default=False):
        self.joints_offset = (
            {n: 0.0 for n in JOINTS} if offsets is ... else offsets
        )
        self.imu_upside_down = imu_upside_down
        self.default = default


def _quat_for_tilt(deg: float) -> tuple[float, float, float, float]:
    """Quaternion tilted `deg` about the x-axis. _tilt_deg inverts this."""
    half = math.radians(deg) / 2.0
    return (math.cos(half), math.sin(half), 0.0, 0.0)


# ── joints ──────────────────────────────────────────────────────────────────────


def test_joints_all_responding() -> None:
    r = check_joints(FakeHwi(FakeIo()))
    assert r.ok and r.operator == ""
    assert "5/5" in r.detail


def test_one_dead_joint_is_named() -> None:
    r = check_joints(FakeHwi(FakeIo(dead={13})))
    assert not r.ok
    assert r.operator == "Right knee is not responding."
    assert "right_knee" in r.detail and "13" in r.detail


def test_several_dead_joints_are_counted_not_listed() -> None:
    r = check_joints(FakeHwi(FakeIo(dead={13, 23, 30})))
    assert not r.ok
    assert r.operator == "3 joints are not responding."
    # the log still names every one
    for name in ("right_knee", "left_knee", "neck_pitch"):
        assert name in r.detail


def test_none_from_read_counts_as_dead() -> None:
    r = check_joints(FakeHwi(FakeIo(positions={13: None})))
    assert not r.ok
    assert r.operator == "Right knee is not responding."


def test_nan_position_counts_as_dead() -> None:
    r = check_joints(FakeHwi(FakeIo(positions={23: float("nan")})))
    assert not r.ok
    assert "left_knee" in r.detail


def test_implausible_position_is_a_fault() -> None:
    r = check_joints(FakeHwi(FakeIo(positions={30: MAX_PLAUSIBLE_POSITION_RAD + 1})))
    assert not r.ok
    assert r.operator == "Neck pitch is reporting an impossible position."


def test_dead_joint_outranks_implausible_one() -> None:
    """A dead servo is the more actionable fault; report it first."""
    io = FakeIo(positions={30: 99.0}, dead={13})
    r = check_joints(FakeHwi(io))
    assert not r.ok
    assert "not responding" in r.operator


# ── offsets ─────────────────────────────────────────────────────────────────────


def test_offsets_all_present() -> None:
    r = check_offsets(FakeHwi(FakeIo()), FakeConfig())
    assert r.ok


def test_one_missing_offset_is_named() -> None:
    offsets = {n: 0.0 for n in JOINTS}
    del offsets["left_knee"]
    r = check_offsets(FakeHwi(FakeIo()), FakeConfig(offsets=offsets))
    assert not r.ok
    assert r.operator == "Left knee is not calibrated."


def test_several_missing_offsets_are_counted() -> None:
    r = check_offsets(FakeHwi(FakeIo()), FakeConfig(offsets={"left_hip_yaw": 0.0}))
    assert not r.ok
    assert r.operator == "4 joints are not calibrated."


def test_non_dict_offsets_fails() -> None:
    r = check_offsets(FakeHwi(FakeIo()), FakeConfig(offsets=None))
    assert not r.ok
    assert "NoneType" in r.detail


def test_non_numeric_offset_fails() -> None:
    offsets = {n: 0.0 for n in JOINTS}
    offsets["neck_pitch"] = "0.0"
    r = check_offsets(FakeHwi(FakeIo()), FakeConfig(offsets=offsets))
    assert not r.ok
    assert r.operator == "Neck pitch has a bad calibration value."


def test_config_that_fell_back_to_defaults_fails() -> None:
    """duck_config sets default=True when no config file loaded. Every offset is then
    0.0, which is present-but-meaningless -- the duck is uncalibrated."""
    r = check_offsets(FakeHwi(FakeIo()), FakeConfig(default=True))
    assert not r.ok
    assert r.operator == "This robot has no calibration saved."


# ── imu ─────────────────────────────────────────────────────────────────────────


def test_imu_upright_passes() -> None:
    r = check_imu(FakeImu(), FakeConfig())
    assert r.ok


def test_imu_identity_quaternion_is_upright() -> None:
    """raw_imu returns identity before its first real read; that is level, not missing."""
    r = check_imu(FakeImu(quaternion=(1.0, 0.0, 0.0, 0.0)), FakeConfig())
    assert r.ok


def test_imu_inverted_fails() -> None:
    r = check_imu(FakeImu(quaternion=_quat_for_tilt(180)), FakeConfig())
    assert not r.ok
    assert r.operator == "The robot does not think it is upright."
    assert "imu_upside_down" in r.detail


def test_imu_small_tilt_still_passes() -> None:
    r = check_imu(FakeImu(quaternion=_quat_for_tilt(MAX_RESTING_TILT_DEG - 5)), FakeConfig())
    assert r.ok


def test_imu_beyond_threshold_fails() -> None:
    r = check_imu(FakeImu(quaternion=_quat_for_tilt(MAX_RESTING_TILT_DEG + 5)), FakeConfig())
    assert not r.ok


def test_imu_raising_fails() -> None:
    r = check_imu(FakeImu(raises=True), FakeConfig())
    assert not r.ok
    assert "i2c bus error" in r.detail


def test_imu_empty_data_fails() -> None:
    r = check_imu(FakeImu(data={}), FakeConfig())
    assert not r.ok


def test_imu_zero_quaternion_fails() -> None:
    """A zero quaternion is unusable, not level. Treating it as level would be a
    silent fall -- the duck would report upright while lying down."""
    r = check_imu(FakeImu(quaternion=(0.0, 0.0, 0.0, 0.0)), FakeConfig())
    assert not r.ok


def test_imu_missing_quaternion_key_fails() -> None:
    r = check_imu(FakeImu(data={"gyro": [0, 0, 0]}), FakeConfig())
    assert not r.ok


def test_imu_unnormalised_quaternion_is_normalised_not_rejected() -> None:
    r = check_imu(FakeImu(quaternion=(2.0, 0.0, 0.0, 0.0)), FakeConfig())
    assert r.ok, "a scaled identity is still upright"


# ── feet ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("contacts", [(False, False), (True, True), (True, False)])
def test_feet_readable_passes(contacts) -> None:
    r = check_feet(FakeFeet(contacts=contacts))
    assert r.ok


def test_feet_raising_fails() -> None:
    r = check_feet(FakeFeet(raises=True))
    assert not r.ok
    assert r.operator == "The foot sensors are not responding."


def test_feet_wrong_shape_fails() -> None:
    r = check_feet(FakeFeet(contacts=(True,)))
    assert not r.ok


def test_feet_none_fails() -> None:
    r = check_feet(FakeFeet(contacts=None))
    assert not r.ok


# ── orchestration ───────────────────────────────────────────────────────────────


def test_all_pass() -> None:
    report = run_preflight(FakeHwi(FakeIo()), FakeImu(), FakeFeet(), FakeConfig())
    assert report.ok
    assert [c.name for c in report.checks] == ["joints", "offsets", "imu", "feet"]
    assert report.duration_ms >= 0


def test_every_check_runs_even_when_the_first_fails() -> None:
    """One dead servo must not hide an inverted IMU. An operator fixing faults one
    round trip at a time is worse than one list."""
    report = run_preflight(
        FakeHwi(FakeIo(dead={13})),
        FakeImu(quaternion=_quat_for_tilt(180)),
        FakeFeet(raises=True),
        FakeConfig(offsets={}),
    )
    assert not report.ok
    assert len(report.checks) == 4
    assert all(not c.ok for c in report.checks), "all four should have been evaluated"


def test_operator_strings_are_short_and_undiagnostic() -> None:
    """app/DESIGN.md#errors: one sentence, under 100 chars, no diagnostics on screen."""
    report = run_preflight(
        FakeHwi(FakeIo(dead={13})),
        FakeImu(raises=True),
        FakeFeet(raises=True),
        FakeConfig(offsets={}),
    )
    for check in report.checks:
        if check.ok:
            continue
        assert len(check.operator) <= 100, check.operator
        assert ";" not in check.operator, f"explaining, not stating: {check.operator}"
        assert check.operator.endswith("."), check.operator
        assert check.operator.count(".") == 1, f"more than one sentence: {check.operator}"


def test_report_serialises() -> None:
    report = run_preflight(FakeHwi(FakeIo()), FakeImu(), FakeFeet(), FakeConfig())
    d = report.as_dict()
    assert set(d) == {"ok", "checks", "duration_ms"}
    assert set(d["checks"][0]) == {"name", "ok", "detail", "operator"}


def test_preflight_never_writes() -> None:
    """It reads. Any set_position/turn_on/set_kps attempt is a bug -- a check that
    changes the thing it checks is not a check."""

    class ExplodingHwi(FakeHwi):
        def __getattr__(self, item):
            if item.startswith(("set_", "turn_")):
                raise AssertionError(f"preflight called {item}()")
            raise AttributeError(item)

    report = run_preflight(ExplodingHwi(FakeIo()), FakeImu(), FakeFeet(), FakeConfig())
    assert report.ok


# ── HTTP surface ────────────────────────────────────────────────────────────────


class _StubHwi(FakeHwi):
    def __init__(self, io, config):
        super().__init__(io)
        self.duck_config = config


@pytest.fixture
def stub_hardware(monkeypatch):
    """Point tnkr_server's preflight collaborators at fakes."""
    import tnkr_server

    state = {
        "io": FakeIo(),
        "imu": FakeImu(),
        "feet": FakeFeet(),
        "config": FakeConfig(),
        "feet_stopped": False,
    }

    class StoppableFeet(FakeFeet):
        def stop(self):
            state["feet_stopped"] = True

    state["feet"] = StoppableFeet()

    monkeypatch.setattr(
        tnkr_server, "get_hwi", lambda: _StubHwi(state["io"], state["config"])
    )
    monkeypatch.setattr(tnkr_server, "get_state_imu", lambda: state["imu"])
    monkeypatch.setattr(tnkr_server, "get_feet_contacts", lambda: state["feet"])
    monkeypatch.setattr(tnkr_server, "is_walking", lambda: False)
    monkeypatch.setattr(tnkr_server, "last_preflight", None)
    return state


def test_post_preflight_all_pass(client, stub_hardware):
    r = client.post("/api/preflight")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert [c["name"] for c in body["checks"]] == ["joints", "offsets", "imu", "feet"]


def test_post_preflight_reports_the_failing_joint(client, stub_hardware):
    stub_hardware["io"] = FakeIo(dead={13})
    r = client.post("/api/preflight")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    joints = next(c for c in body["checks"] if c["name"] == "joints")
    assert joints["operator"] == "Right knee is not responding."


def test_preflight_releases_the_foot_pins(client, stub_hardware):
    """A walk started right after preflight needs those pins back."""
    client.post("/api/preflight")
    assert stub_hardware["feet_stopped"] is True


def test_preflight_releases_pins_even_when_a_check_explodes(client, stub_hardware, monkeypatch):
    import tnkr_server

    monkeypatch.setattr(
        tnkr_server.preflight, "run_preflight",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    r = client.post("/api/preflight")
    assert r.status_code == 500
    assert stub_hardware["feet_stopped"] is True, "finally: must still release the pins"


def test_preflight_refuses_while_walking(client, stub_hardware, monkeypatch):
    import tnkr_server

    monkeypatch.setattr(tnkr_server, "is_walking", lambda: True)
    r = client.post("/api/preflight")
    assert r.status_code == 409
    assert "servo bus" in r.json()["detail"]


def test_preflight_503s_when_the_bus_is_unavailable(client, stub_hardware, monkeypatch):
    import tnkr_server

    def boom():
        raise RuntimeError("no adapter")

    monkeypatch.setattr(tnkr_server, "get_hwi", boom)
    r = client.post("/api/preflight")
    assert r.status_code == 503
    assert "motor controller" in r.json()["detail"]


def test_missing_gpio_is_a_reported_failure_not_a_crash(client, stub_hardware, monkeypatch):
    import tnkr_server

    monkeypatch.setattr(tnkr_server, "get_feet_contacts", lambda: None)
    r = client.post("/api/preflight")
    assert r.status_code == 200
    feet = next(c for c in r.json()["checks"] if c["name"] == "feet")
    assert feet["ok"] is False
    assert feet["operator"] == "The foot sensors are not available."


def test_get_preflight_before_any_run(client, stub_hardware):
    assert client.get("/api/preflight").json() == {"ok": None, "checks": [], "duration_ms": 0}


def test_get_preflight_returns_the_last_run(client, stub_hardware):
    client.post("/api/preflight")
    body = client.get("/api/preflight").json()
    assert body["ok"] is True
    assert len(body["checks"]) == 4


def test_preflight_telemetry_names_failures_not_values(client, stub_hardware, captured):
    stub_hardware["io"] = FakeIo(dead={13})
    client.post("/api/preflight")
    events = [e for e in captured if e["event"] == "preflight_run"]
    assert len(events) == 1
    props = events[0]["properties"]
    assert props["ok"] is False
    assert props["failed"] == ["joints"]
    # never the readings themselves
    assert not any("position" in k or "detail" in k for k in props)
