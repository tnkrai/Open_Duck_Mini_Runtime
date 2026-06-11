"""tnkr_server.py telemetry: middleware capture, error causes, exclusions."""

import json

import pytest
from fastapi.testclient import TestClient

import tnkr_server
from mini_bdx_runtime import telemetry


@pytest.fixture
def client():
    return TestClient(tnkr_server.app, raise_server_exceptions=False)


class FakeIO:
    """Stub servo bus: joints 22 and 13 are unresponsive."""

    DEAD = {22, 13}

    def set_kps(self, ids, kps):
        if set(ids) & self.DEAD:
            raise OSError("timeout")

    def read_present_position(self, ids):
        if set(ids) & self.DEAD:
            raise OSError("timeout")
        return [0.0]

    def disable_torque(self, ids):
        pass


class FakeHWI:
    def __init__(self):
        self.joints = {"left_hip_pitch": 22, "right_knee": 13, "head_yaw": 32}
        self.low_torque_kps = [2]
        self.io = FakeIO()


def events_named(captured, name):
    return [e for e in captured if e["event"] == name]


# ── Generic middleware capture ───────────────────────────────────────────────

def test_success_event_with_duration(client, captured):
    r = client.get("/api/imu/calibrate/status")  # excluded — warm-up, no event
    r = client.post("/api/imu/calibrate/stop")
    assert r.status_code == 200
    done = events_named(captured, "api_request_completed")
    assert len(done) == 1
    p = done[0]["properties"]
    assert p["endpoint"] == "/api/imu/calibrate/stop"
    assert p["method"] == "POST"
    assert p["status_code"] == 200
    assert p["duration_ms"] >= 0


def test_http_exception_detail_reaches_telemetry(client, captured, monkeypatch):
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", "/nonexistent/duck_config.json")
    r = client.get("/api/config")
    assert r.status_code == 404
    failed = events_named(captured, "api_request_failed")
    assert len(failed) == 1
    p = failed[0]["properties"]
    assert p["status_code"] == 404
    assert p["error_type"] == "HTTPException"
    assert "Config file not found" in p["error_message"]


def test_unhandled_exception_generic_response_rich_telemetry(client, captured):
    async def boom():
        raise RuntimeError("kaboom: secret internals /home/duck/x.py")

    tnkr_server.app.add_api_route("/api/_test_boom", boom, methods=["POST"])
    r = client.post("/api/_test_boom")
    assert r.status_code == 500
    # Client gets a generic body — internals never leak over HTTP
    assert r.json() == {"detail": "Internal Server Error"}
    assert "kaboom" not in r.text
    failed = events_named(captured, "api_request_failed")
    assert len(failed) == 1
    p = failed[0]["properties"]
    assert p["error_type"] == "RuntimeError"
    assert "kaboom" in p["error_message"]


# ── Exclusions ───────────────────────────────────────────────────────────────

def test_excluded_and_non_api_paths_produce_no_events(client, captured):
    for _ in range(5):
        client.get("/api/health")
        client.get("/api/imu/calibrate/status")
        client.post(
            "/api/commands",
            json={"commands": [0.0], "buttons": {}},
        )
    client.options("/api/config")
    client.get("/does-not-exist")
    assert captured == []


# ── Endpoint enrichments ─────────────────────────────────────────────────────

def test_motor_check_enrichment_names_unresponsive_joints(client, captured, monkeypatch):
    monkeypatch.setattr(tnkr_server, "get_hwi", lambda: FakeHWI())
    r = client.post("/api/motors/check")
    assert r.status_code == 200
    assert r.json()["allResponsive"] is False
    p = events_named(captured, "api_request_completed")[0]["properties"]
    assert p["all_responsive"] is False
    assert p["responsive_count"] == 1
    assert sorted(p["unresponsive_joints"]) == ["left_hip_pitch", "right_knee"]


def test_motor_check_no_hardware_captures_cause(client, captured, monkeypatch):
    def no_hw():
        raise RuntimeError("No servo-bus USB adapter found")

    monkeypatch.setattr(tnkr_server, "get_hwi", no_hw)
    r = client.post("/api/motors/check")
    assert r.status_code == 503
    p = events_named(captured, "api_request_failed")[0]["properties"]
    assert p["error_type"] == "HTTPException"
    assert "Cannot connect to motor controller" in p["error_message"]


def test_add_telemetry_props_noop_outside_request():
    # Excluded/non-API requests have no contextvar dict — must not blow up.
    tnkr_server.add_telemetry_props(foo="bar")


# ── IMU worker outcome events ────────────────────────────────────────────────

def test_imu_worker_failure_event(captured, monkeypatch):
    # board/busio aren't importable off-robot → worker takes the except path.
    tnkr_server.imu_calib_status["running"] = True
    tnkr_server._imu_calibrate_worker()
    assert tnkr_server.imu_calib_status["running"] is False
    failed = events_named(captured, "imu_calibration_failed")
    assert len(failed) == 1
    p = failed[0]["properties"]
    assert p["error_type"] in ("ImportError", "ModuleNotFoundError", "NotImplementedError")
    assert p["duration_s"] >= 0


# ── Privacy: session token never appears in any event ────────────────────────

def test_walk_start_props_never_contain_session_token(client, captured, tmp_path, monkeypatch):
    token = "SECRET_TOKEN_a1b2c3"
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "v2_rl_walk_mujoco.py").write_text("import sys; sys.exit(0)\n")

    r = client.post(
        "/api/walk/start",
        json={"sessionToken": token, "supabaseUrl": "https://x.supabase.co", "supabaseKey": "k"},
    )
    assert r.status_code == 200
    client.post("/api/walk/stop")

    everything = json.dumps(captured)
    assert token not in everything
    assert "x.supabase.co" not in everything
    start = events_named(captured, "api_request_completed")[0]["properties"]
    assert start["endpoint"] == "/api/walk/start"
    assert start["cloud_streaming"] is True
    assert start["has_session"] is True


def test_walk_start_without_session_is_not_cloud_streaming(client, captured, tmp_path, monkeypatch):
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "v2_rl_walk_mujoco.py").write_text("import sys; sys.exit(0)\n")

    r = client.post("/api/walk/start", json={})
    assert r.status_code == 200
    client.post("/api/walk/stop")

    start = [
        e for e in events_named(captured, "api_request_completed")
        if e["properties"]["endpoint"] == "/api/walk/start"
    ][0]["properties"]
    assert start["cloud_streaming"] is False
    assert start["has_session"] is False


# ── Opt-out really silences the server ───────────────────────────────────────

def test_opt_out_env_var_silences_all_api_events(client, captured, monkeypatch):
    monkeypatch.setenv("TNKR_TELEMETRY", "0")
    client.post("/api/imu/calibrate/stop")
    client.get("/api/config")
    assert captured == []
