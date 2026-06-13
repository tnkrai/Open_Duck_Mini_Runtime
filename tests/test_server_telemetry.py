"""tnkr_server.py telemetry: middleware capture, error causes, exclusions."""

import json
import sys
import types

import tnkr_server
from conftest import write_walk_script


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
    client.get("/api/imu/calibrate/status")  # excluded — warm-up, no event
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
        raise RuntimeError("kaboom: secret internals /somewhere/x.py")

    tnkr_server.app.add_api_route("/api/_test_boom", boom, methods=["POST"])
    try:
        r = client.post("/api/_test_boom")
    finally:
        # Never leave the test route on the shared module-level app.
        tnkr_server.app.router.routes[:] = [
            rt for rt in tnkr_server.app.router.routes
            if getattr(rt, "path", None) != "/api/_test_boom"
        ]
    assert r.status_code == 500
    # Client gets a generic body — internals never leak over HTTP
    assert r.json() == {"detail": "Internal Server Error"}
    assert "kaboom" not in r.text
    failed = events_named(captured, "api_request_failed")
    assert len(failed) == 1
    p = failed[0]["properties"]
    assert p["error_type"] == "RuntimeError"
    assert "kaboom" in p["error_message"]


def test_validation_error_reports_field_paths_not_values(client, captured):
    r = client.post("/api/calibration/begin-joint", json={"wrong": "SECRET_VALUE"})
    assert r.status_code == 422
    failed = events_named(captured, "api_request_failed")
    assert len(failed) == 1
    p = failed[0]["properties"]
    assert p["error_type"] == "RequestValidationError"
    assert "jointName" in p["error_message"]
    assert "SECRET_VALUE" not in json.dumps(captured)


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


def test_unmatched_api_404s_are_not_captured(client, captured):
    # LAN port-scanner noise: /api/* paths that match no route must not
    # generate events (quota protection).
    for _ in range(3):
        client.get("/api/not-a-real-endpoint")
        client.post("/api/admin/login")
    assert captured == []
    # ...but a REAL endpoint returning 404 (config missing) IS captured.
    client.get("/api/config")
    assert len(events_named(captured, "api_request_failed")) == 1


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
        raise tnkr_server.HTTPException(
            status_code=503,
            detail="Cannot connect to motor controller: No servo-bus USB adapter found",
        )

    monkeypatch.setattr(tnkr_server, "get_hwi", no_hw)
    r = client.post("/api/motors/check")
    assert r.status_code == 503
    p = events_named(captured, "api_request_failed")[0]["properties"]
    assert p["error_type"] == "HTTPException"
    assert "Cannot connect to motor controller" in p["error_message"]


def test_calibration_save_reports_joint_count(client, captured, tmp_path, monkeypatch):
    cfg = tmp_path / "duck_config.json"
    cfg.write_text(json.dumps({"joints_offsets": {}}))
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", str(cfg))
    monkeypatch.setattr(tnkr_server, "calibration_offsets", {"left_knee": 0.1, "head_yaw": -0.2})
    r = client.post("/api/calibration/save")
    assert r.status_code == 200
    p = events_named(captured, "api_request_completed")[0]["properties"]
    assert p["joints_calibrated"] == 2


def test_update_config_enrichment_caps_feature_keys(client, captured, tmp_path, monkeypatch):
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", str(tmp_path / "duck_config.json"))
    features = {f"feat_{i}" * 30: True for i in range(40)}  # long keys, many of them
    r = client.post(
        "/api/config",
        json={"expression_features": features},
    )
    assert r.status_code == 200
    p = events_named(captured, "api_request_completed")[0]["properties"]
    enabled = p["expression_features_enabled"]
    assert len(enabled) <= 20
    assert all(len(k) <= 50 for k in enabled)


def test_add_telemetry_props_noop_outside_request():
    # Excluded/non-API requests have no contextvar dict — must not blow up.
    tnkr_server.add_telemetry_props(foo="bar")


# ── IMU worker outcome events ────────────────────────────────────────────────

def _stub_imu_modules(monkeypatch, fake_imu):
    monkeypatch.setitem(sys.modules, "board", types.SimpleNamespace(SCL=1, SDA=2))
    monkeypatch.setitem(
        sys.modules, "busio", types.SimpleNamespace(I2C=lambda *a, **k: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "adafruit_bno055",
        types.SimpleNamespace(BNO055_I2C=lambda i2c: fake_imu, NDOF_MODE=12),
    )


def test_imu_worker_failure_event(captured, monkeypatch):
    # Force the import to fail deterministically (off-robot there's no board,
    # but pin it so an installed adafruit-blinka can't change the path).
    monkeypatch.setitem(sys.modules, "board", None)
    status = {"running": True, "calibration_status": [0, 0, 0, 0],
              "calibrated": False, "error": None, "offsets": None}
    tnkr_server._imu_calibrate_worker(status)
    assert status["running"] is False
    failed = events_named(captured, "imu_calibration_failed")
    assert len(failed) == 1
    p = failed[0]["properties"]
    assert p["error_type"] in ("ImportError", "ModuleNotFoundError")
    assert p["duration_s"] >= 0


def test_imu_worker_completed_event(captured, monkeypatch, tmp_path):
    class FakeIMU:
        mode = None
        calibration_status = (3, 3, 3, 3)
        calibrated = True
        offsets_accelerometer = (0, 0, 0)
        offsets_gyroscope = (0, 0, 0)
        offsets_magnetometer = (0, 0, 0)

    _stub_imu_modules(monkeypatch, FakeIMU())
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", tmp_path)
    status = {"running": True, "calibration_status": [0, 0, 0, 0],
              "calibrated": False, "error": None, "offsets": None}
    tnkr_server._imu_calibrate_worker(status)
    assert (tmp_path / "imu_calib_data.pkl").exists()
    assert len(events_named(captured, "imu_calibration_completed")) == 1


def test_imu_worker_stopped_event(captured, monkeypatch):
    class FakeIMU:
        mode = None
        calibration_status = (1, 2, 0, 3)
        calibrated = False
        # never calibrates; loop exits when status["running"] flips

    fake = FakeIMU()
    _stub_imu_modules(monkeypatch, fake)
    status = {"running": True, "calibration_status": [0, 0, 0, 0],
              "calibrated": False, "error": None, "offsets": None}

    # Flip running to False after the first poll, as /stop would.
    class StopAfterFirstPoll:
        def __get__(self, obj, objtype=None):
            status["running"] = False
            return (1, 2, 0, 3)

    type(fake).calibration_status = StopAfterFirstPoll()
    tnkr_server._imu_calibrate_worker(status)
    stopped = events_named(captured, "imu_calibration_stopped")
    assert len(stopped) == 1
    assert stopped[0]["properties"]["calibration_status"] == [1, 2, 0, 3]


def test_imu_stop_then_restart_does_not_double_report(captured, monkeypatch):
    """The worker must loop on the dict it was handed, not the module global:
    /start rebinds the global, and a stopped worker latching onto the new
    dict would poll forever and double-report."""
    monkeypatch.setitem(sys.modules, "board", None)
    old_status = {"running": False, "calibration_status": [0, 0, 0, 0],
                  "calibrated": False, "error": None, "offsets": None}
    new_status = {"running": True, "calibration_status": [0, 0, 0, 0],
                  "calibrated": False, "error": None, "offsets": None}
    tnkr_server.imu_calib_status = new_status  # simulate /start rebinding
    tnkr_server._imu_calibrate_worker(old_status)  # old worker wakes up
    # The old worker must not have touched the new run's state...
    assert new_status["running"] is True
    # ...and reports exactly one outcome for itself (import fails -> failed).
    assert len(events_named(captured, "imu_calibration_failed")) == 1


# ── Privacy: session token never appears in any event ────────────────────────

def test_walk_start_props_never_contain_session_token(client, captured, fake_walk_dir):
    token = "SECRET_TOKEN_a1b2c3"
    write_walk_script(fake_walk_dir, "import sys; sys.exit(0)\n")

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


def test_walk_start_without_session_is_not_cloud_streaming(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, "import sys; sys.exit(0)\n")

    r = client.post("/api/walk/start", json={})
    assert r.status_code == 200
    client.post("/api/walk/stop")

    start = [
        e for e in events_named(captured, "api_request_completed")
        if e["properties"]["endpoint"] == "/api/walk/start"
    ][0]["properties"]
    assert start["cloud_streaming"] is False
    assert start["has_session"] is False


def test_walk_stop_reports_was_running(client, captured):
    r = client.post("/api/walk/stop")  # nothing running
    assert r.status_code == 200
    p = events_named(captured, "api_request_completed")[0]["properties"]
    assert p["was_running"] is False


# ── Opt-out really silences the server ───────────────────────────────────────

def test_opt_out_env_var_silences_all_api_events(client, captured, monkeypatch):
    monkeypatch.setenv("TNKR_TELEMETRY", "0")
    client.post("/api/imu/calibrate/stop")
    client.get("/api/config")
    assert captured == []
