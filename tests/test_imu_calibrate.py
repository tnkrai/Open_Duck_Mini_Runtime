"""IMU calibration must reuse the state IMU's chip handle.

Constructing a second BNO055_I2C soft-resets the chip (adafruit's __init__
issues SYS_TRIGGER 0x20, "reset to default settings"), which wipes the axis
remap the state reader applied for this robot's mounting — after that, every
/api/state quaternion is in the chip's factory frame and the studio's 3D duck
renders tipped over while the real duck stands upright. These tests pin the
shared-handle behavior.
"""

import time

import tnkr_server


class FakeBNO055:
    """The already-calibrated chip behind the state IMU's handle."""

    calibration_status = (3, 3, 3, 3)
    calibrated = True
    offsets_accelerometer = (1, 2, 3)
    offsets_gyroscope = (4, 5, 6)
    offsets_magnetometer = (7, 8, 9)


class FakeStateImu:
    def __init__(self):
        self.imu = FakeBNO055()


def wait_until(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_calibration_worker_uses_state_imu_handle(client, monkeypatch, tmp_path):
    # If the worker tried to construct its own BNO055_I2C, it would die on the
    # missing adafruit imports off-Pi — completing proves it went through the
    # shared handle.
    fake = FakeStateImu()
    monkeypatch.setattr(tnkr_server, "get_state_imu", lambda: fake)
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", tmp_path)

    r = client.post("/api/imu/calibrate/start")
    assert r.status_code == 200
    assert wait_until(lambda: not tnkr_server.imu_calib_status["running"])

    s = tnkr_server.imu_calib_status
    assert s["error"] is None
    assert s["calibrated"] is True
    assert s["calibration_status"] == [3, 3, 3, 3]
    assert s["offsets"]["offsets_accelerometer"] == [1, 2, 3]
    assert (tmp_path / "imu_calib_data.pkl").exists()


def test_calibration_worker_reports_error_when_imu_unavailable(client, monkeypatch):
    def no_imu():
        raise RuntimeError("no BNO055 on this machine")

    monkeypatch.setattr(tnkr_server, "get_state_imu", no_imu)

    r = client.post("/api/imu/calibrate/start")
    assert r.status_code == 200
    assert wait_until(lambda: tnkr_server.imu_calib_status["error"] is not None)
    assert tnkr_server.imu_calib_status["running"] is False
