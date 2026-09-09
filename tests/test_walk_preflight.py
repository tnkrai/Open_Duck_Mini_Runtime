"""The walk pre-flight: before the walk process spawns, every servo and the IMU must
answer, or the start is refused with a coded error naming the part.

The walk's own constructor used to be the first code to ask, from a subprocess
whose traceback went to the journal. Seven walks in Sep 2026 died that way, rigid
in the crouch, with nothing above them able to say why."""

import os
import time

import tnkr_server
from conftest import write_walk_script
from fakes import FakeHWI, FakeStateImu

SLEEP = "import time; time.sleep(30)\n"


def _request_events(captured, name):
    return [
        e for e in captured
        if e["event"] == name and e["properties"].get("endpoint") == "/api/walk/start"
    ]


def _no_walk_spawned(captured):
    time.sleep(0.3)
    assert tnkr_server.walk_session is None
    assert not [e for e in captured if e["event"] == "walk_ended"]


# ── servos ───────────────────────────────────────────────────────────────────


def test_a_silent_servo_refuses_the_walk_and_names_the_joint(client, captured, fake_walk_dir, monkeypatch):
    monkeypatch.setattr(tnkr_server, "get_hwi", lambda: FakeHWI(dead={20}))  # left_hip_yaw
    write_walk_script(fake_walk_dir, SLEEP)

    r = client.post("/api/walk/start", json={"input": "keyboard"})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["code"] == "MOTORS_SILENT"
    assert detail["joint"] == "left_hip_yaw"
    assert detail["message"].startswith("1 of 14 servos did not answer: left_hip_yaw")
    _no_walk_spawned(captured)

    p = _request_events(captured, "api_request_failed")[0]["properties"]
    assert p["error_code"] == "MOTORS_SILENT"
    assert p["joint_name"] == "left_hip_yaw"
    assert p["silent_joints"] == ["left_hip_yaw"]
    assert isinstance(p["preflight_ms"], int)


def test_several_silent_servos_name_the_first_and_list_them_all(client, captured, fake_walk_dir, monkeypatch):
    monkeypatch.setattr(tnkr_server, "get_hwi", lambda: FakeHWI(dead={22, 13}))
    write_walk_script(fake_walk_dir, SLEEP)
    r = client.post("/api/walk/start", json={})
    detail = r.json()["detail"]
    # joints in HWI order: left_hip_pitch (22) comes before right_knee (13)
    assert detail["joint"] == "left_hip_pitch"
    assert "2 of 14" in detail["message"]
    p = _request_events(captured, "api_request_failed")[0]["properties"]
    assert p["silent_joints"] == ["left_hip_pitch", "right_knee"]


def test_a_bus_that_will_not_open_is_a_different_code(client, captured, fake_walk_dir, monkeypatch):
    """No joint could be asked. The operator checks the adapter, not one cable."""

    def no_bus():
        raise RuntimeError("No servo-bus USB adapter found")

    monkeypatch.setattr(tnkr_server, "get_hwi", no_bus)
    write_walk_script(fake_walk_dir, SLEEP)
    r = client.post("/api/walk/start", json={})
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "SERVO_BUS_UNAVAILABLE"
    assert "adapter" in r.json()["detail"]["message"]
    _no_walk_spawned(captured)
    assert _request_events(captured, "api_request_failed")[0]["properties"]["error_code"] == "SERVO_BUS_UNAVAILABLE"


def test_the_probe_never_writes_to_a_servo(client, fake_walk_dir, monkeypatch):
    """The duck may be standing in its stance when Walk is pressed."""
    writes = []

    class WatchedIO:
        DEAD = set()

        def set_kps(self, ids, kps):
            writes.append(("set_kps", ids))

        def disable_torque(self, ids):
            writes.append(("disable_torque", ids))

        def read_present_position(self, ids):
            return [0.0]

    hwi = FakeHWI()
    hwi.io = WatchedIO()
    monkeypatch.setattr(tnkr_server, "get_hwi", lambda: hwi)
    write_walk_script(fake_walk_dir, SLEEP)
    assert client.post("/api/walk/start", json={}).status_code == 200
    assert writes == []


# ── IMU ──────────────────────────────────────────────────────────────────────


def test_a_missing_imu_refuses_the_walk(client, captured, fake_walk_dir, monkeypatch):
    def no_chip():
        raise ValueError("No I2C device at address: 0x28")

    monkeypatch.setattr(tnkr_server, "get_state_imu", no_chip)
    write_walk_script(fake_walk_dir, SLEEP)
    r = client.post("/api/walk/start", json={})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["code"] == "IMU_NOT_FOUND"
    assert "0x28" in detail["message"]
    assert "joint" not in detail
    _no_walk_spawned(captured)
    p = _request_events(captured, "api_request_failed")[0]["properties"]
    assert p["error_code"] == "IMU_NOT_FOUND"
    assert p["imu_error_type"] == "ValueError"
    assert isinstance(p["preflight_ms"], int)


def test_servos_are_asked_before_the_imu(client, fake_walk_dir, monkeypatch):
    """Both missing: the servo is named. A bus problem changes the next move."""
    monkeypatch.setattr(tnkr_server, "get_hwi", lambda: FakeHWI(dead={23}))

    def no_chip():
        raise ValueError("No I2C device at address: 0x28")

    monkeypatch.setattr(tnkr_server, "get_state_imu", no_chip)
    write_walk_script(fake_walk_dir, SLEEP)
    r = client.post("/api/walk/start", json={})
    assert r.json()["detail"]["code"] == "MOTORS_SILENT"
    assert r.json()["detail"]["joint"] == "left_knee"


def test_the_imu_handle_is_reused_and_left_in_place_for_release(client, fake_walk_dir, monkeypatch):
    """A second BNO055_I2C would soft-reset the chip; and release_state_imu still runs
    after the probe, exactly as it did before the pre-flight existed."""
    calls = []
    released = []

    def shared():
        calls.append(1)
        return FakeStateImu()

    monkeypatch.setattr(tnkr_server, "get_state_imu", shared)
    monkeypatch.setattr(tnkr_server, "release_state_imu", lambda: released.append(1))
    write_walk_script(fake_walk_dir, SLEEP)
    assert client.post("/api/walk/start", json={}).status_code == 200
    assert calls == [1]
    assert released == [1]


# ── the happy path, the override, and the mock path ──────────────────────────


def test_a_healthy_duck_walks_and_the_pass_is_on_the_event(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, SLEEP)
    r = client.post("/api/walk/start", json={})
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert tnkr_server.walk_session is not None
    p = _request_events(captured, "api_request_completed")[0]["properties"]
    assert p["servos_responding"] == 14
    assert p["imu_present"] is True
    assert isinstance(p["preflight_ms"], int)


def test_the_only_override_is_the_environment_switch(client, captured, fake_walk_dir, monkeypatch):
    def no_bus():
        raise RuntimeError("No servo-bus USB adapter found")

    monkeypatch.setattr(tnkr_server, "get_hwi", no_bus)
    monkeypatch.setenv("TNKR_SKIP_PREFLIGHT", "1")
    write_walk_script(fake_walk_dir, SLEEP)
    r = client.post("/api/walk/start", json={})
    assert r.status_code == 200
    p = _request_events(captured, "api_request_completed")[0]["properties"]
    assert "preflight_ms" not in p
    assert "servos_responding" not in p


def test_the_mock_walk_path_never_runs_the_preflight(client, fake_walk_dir, monkeypatch):
    """Not a Pi: the mock broadcaster path, which needs cloud creds. A dead bus must
    not turn its 400 into a 503."""
    monkeypatch.setattr(tnkr_server.platform, "machine", lambda: "x86_64")

    def no_bus():
        raise RuntimeError("No servo-bus USB adapter found")

    monkeypatch.setattr(tnkr_server, "get_hwi", no_bus)
    r = client.post("/api/walk/start", json={})
    assert r.status_code == 400


def test_a_refusal_leaves_the_server_handles_where_they_were(client, fake_walk_dir, monkeypatch, tmp_path):
    """Refused before release: the HWI singleton and the stance are untouched, so a
    duck holding its pose keeps holding it."""
    from test_stance_hold import install_hwi

    fake = install_hwi(monkeypatch, tmp_path, holding=True)
    monkeypatch.setattr(tnkr_server, "get_hwi", lambda: FakeHWI(dead={24}))
    write_walk_script(fake_walk_dir, SLEEP)
    assert client.post("/api/walk/start", json={}).status_code == 502
    assert tnkr_server.hwi_instance is fake
    assert fake.turned_off is False
