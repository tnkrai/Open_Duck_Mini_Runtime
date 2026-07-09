"""/api/voltage payload shape: both readings come back as joint-name-keyed maps
(the hottest servo is the story for heat, so readings carry their identity),
with pack-level and temperature rollups the UI renders directly.
"""

import sys
import types

import tnkr_server
from test_stance_hold import install_hwi


def install_fake_pypot(monkeypatch, volts_deci, temps_c):
    """A pypot stand-in: fixed register reads, no serial port (CI has neither)."""

    class FakeIO:
        def __init__(self, port, baudrate=1000000):
            pass

        def get_present_voltage(self, ids):
            return [volts_deci] * len(ids)

        def get_present_temperature(self, ids):
            return [temps_c(i) for i in range(len(ids))]

        def close(self):
            pass

    feetech = types.SimpleNamespace(FeetechSTS3215IO=FakeIO)
    monkeypatch.setitem(sys.modules, "pypot", types.SimpleNamespace(feetech=feetech))
    monkeypatch.setitem(sys.modules, "pypot.feetech", feetech)


def test_voltage_returns_named_maps_with_rollups(client, monkeypatch, tmp_path):
    install_hwi(monkeypatch, tmp_path, holding=True)
    install_fake_pypot(monkeypatch, volts_deci=76, temps_c=lambda i: 40 + i)
    r = client.get("/api/voltage")
    assert r.status_code == 200
    data = r.json()

    joint_names = set(tnkr_server.JOINTS.keys())
    assert set(data["perMotor"].keys()) == joint_names
    assert set(data["temps"].keys()) == joint_names
    assert data["volts"] == 7.6
    assert data["health"] == "ok"
    # temps ramp 40..40+N-1, so the last joint in JOINTS order is the hottest
    assert data["maxTempC"] == 40 + len(joint_names) - 1
    assert data["hottest"] == max(data["temps"], key=data["temps"].get)
    assert data["tempHealth"] == "ok"


def test_voltage_bands_low_pack_and_hot_servo(client, monkeypatch, tmp_path):
    install_hwi(monkeypatch, tmp_path, holding=True)
    install_fake_pypot(monkeypatch, volts_deci=72, temps_c=lambda i: 66)
    data = client.get("/api/voltage").json()
    assert data["health"] == "low"  # 7.2V: below 7.4 low line, above 7.0 critical
    assert data["maxTempC"] == 66
    assert data["tempHealth"] == "hot"  # at/above the 65 hot line
