"""Torque must survive the serial-bus handoff when the duck is holding its
stance. Feetech servos hold position in firmware without bus traffic, so
freeing the port never requires going limp — cutting torque on save made the
robot collapse the moment the user saved (Sam's report).
"""

import tnkr_server
from conftest import write_walk_script


class FakeHWI:
    """Records turn_off(); real HWI's turn_off disables torque on every servo."""

    def __init__(self):
        self.joints = {"left_hip_pitch": 22, "right_knee": 13}
        self.joints_offsets = {"left_hip_pitch": 0.1, "right_knee": -0.2}
        self.init_pos = {"left_hip_pitch": -0.63, "right_knee": 1.379}
        self.turned_off = False

    def close(self):
        pass

    def turn_off(self):
        self.turned_off = True


def install_hwi(monkeypatch, tmp_path, holding):
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", str(tmp_path / "duck_config.json"))
    fake = FakeHWI()
    monkeypatch.setattr(tnkr_server, "hwi_instance", fake)
    monkeypatch.setattr(tnkr_server, "stance_holding", holding)
    return fake


def test_save_while_holding_keeps_torque_and_hwi(client, monkeypatch, tmp_path):
    fake = install_hwi(monkeypatch, tmp_path, holding=True)
    r = client.post("/api/stance/save")
    assert r.status_code == 200
    assert r.json()["offsets"] == {"left_hip_pitch": 0.1, "right_knee": -0.2}
    # Still energized, still owned: the duck keeps standing in the saved stance.
    assert fake.turned_off is False
    assert tnkr_server.hwi_instance is fake


def test_save_while_limp_still_releases(client, monkeypatch, tmp_path):
    fake = install_hwi(monkeypatch, tmp_path, holding=False)
    r = client.post("/api/stance/save")
    assert r.status_code == 200
    # Never held — nothing to keep energized; HWI released as before.
    assert fake.turned_off is True
    assert tnkr_server.hwi_instance is None


def test_walk_start_hands_off_bus_without_cutting_torque(
    client, monkeypatch, tmp_path, fake_walk_dir
):
    fake = install_hwi(monkeypatch, tmp_path, holding=True)
    write_walk_script(fake_walk_dir, "import time; time.sleep(60)")
    r = client.post("/api/walk/start", json={})
    assert r.status_code == 200
    # Port freed for the walk subprocess, but the servos keep holding their
    # pose in firmware while it boots.
    assert tnkr_server.hwi_instance is None
    assert fake.turned_off is False


def test_voltage_poll_does_not_drop_a_holding_duck(client, monkeypatch, tmp_path):
    fake = install_hwi(monkeypatch, tmp_path, holding=True)
    # Whatever the read outcome (pypot may be absent in CI -> 503), the
    # release must be bus-only: a battery poll must never collapse the robot.
    client.get("/api/voltage")
    assert fake.turned_off is False
