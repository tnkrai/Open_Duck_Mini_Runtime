"""POST /api/imu/check: does the BNO055 answer? The IMU half of the walk pre-flight
on its own route, so Studio's connect-time step has something to call. It reuses
the state IMU handle (a second constructor would soft-reset the chip) and never
waits on a sample."""

import tnkr_server
from conftest import write_walk_script
from fakes import FakeStateImu


def _failed(captured):
    return [e for e in captured if e["event"] == "api_request_failed"]


def test_present_imu_answers_200(client, captured, monkeypatch):
    monkeypatch.setattr(tnkr_server, "get_state_imu", lambda: FakeStateImu())
    r = client.post("/api/imu/check")
    assert r.status_code == 200
    assert r.json() == {"present": True}
    done = [e for e in captured if e["event"] == "api_request_completed"]
    assert done[0]["properties"]["imu_present"] is True


def test_nothing_at_the_address_is_imu_not_found(client, captured, monkeypatch):
    """Blinka's probe, when no device acknowledges at 0x28. The cable is unplugged."""

    def no_chip():
        raise ValueError("No I2C device at address: 0x28")

    monkeypatch.setattr(tnkr_server, "get_state_imu", no_chip)
    r = client.post("/api/imu/check")
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["code"] == "IMU_NOT_FOUND"
    assert "0x28" in detail["message"]
    assert "joint" not in detail
    p = _failed(captured)[0]["properties"]
    assert p["error_code"] == "IMU_NOT_FOUND"
    assert p["imu_error_type"] == "ValueError"
    assert p["error_type"] == "HTTPException"  # the handler's, as on every coded refusal


def test_a_wrong_chip_is_imu_not_found_too(client, captured, monkeypatch):
    """The adafruit driver's identity check: something answered, it is not a BNO055."""

    def wrong_chip():
        raise RuntimeError("bad chip id (0x0 != 0xa0)")

    monkeypatch.setattr(tnkr_server, "get_state_imu", wrong_chip)
    r = client.post("/api/imu/check")
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "IMU_NOT_FOUND"
    assert _failed(captured)[0]["properties"]["imu_error_type"] == "RuntimeError"


def test_refused_while_a_walk_owns_the_i2c(client, monkeypatch, fake_walk_dir):
    monkeypatch.setattr(tnkr_server, "get_state_imu", lambda: FakeStateImu())
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    assert client.post("/api/walk/start", json={}).status_code == 200
    r = client.post("/api/imu/check")
    assert r.status_code == 409
    client.post("/api/walk/stop")


def test_reuses_the_shared_handle_never_a_second_constructor(client, monkeypatch):
    """A second BNO055_I2C soft-resets the chip and wipes the axis remap, so the
    check must go through get_state_imu() and only that."""
    calls = []
    fake = FakeStateImu()

    def shared():
        calls.append(1)
        return fake

    monkeypatch.setattr(tnkr_server, "get_state_imu", shared)
    client.post("/api/imu/check")
    client.post("/api/imu/check")
    assert calls == [1, 1]
