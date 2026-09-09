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


# ── the reading reaches telemetry, and rides on the next walk start ──────────
# During the Sep 2026 incident the fleet had no voltage data at all: /api/voltage
# returned volts to Studio and told telemetry nothing. Now the route's own event
# carries the reading, and /api/walk/start carries the last one with its age.

from conftest import write_walk_script


def _events(captured, name, endpoint):
    return [
        e for e in captured
        if e["event"] == name and e["properties"].get("endpoint") == endpoint
    ]


def test_voltage_read_reaches_its_own_request_event(client, captured, monkeypatch, tmp_path):
    install_hwi(monkeypatch, tmp_path, holding=True)
    install_fake_pypot(monkeypatch, volts_deci=72, temps_c=lambda i: 66)
    client.get("/api/voltage")
    p = _events(captured, "api_request_completed", "/api/voltage")[0]["properties"]
    assert p["volts"] == 7.2
    assert p["health"] == "low"
    assert p["max_temp_c"] == 66
    assert p["temp_health"] == "hot"


def test_walk_start_carries_nothing_before_any_voltage_read(client, captured, fake_walk_dir, monkeypatch):
    monkeypatch.setattr(tnkr_server, "_last_voltage", None)
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    assert client.post("/api/walk/start", json={}).status_code == 200
    p = _events(captured, "api_request_completed", "/api/walk/start")[0]["properties"]
    # omitted, never null: absence and unknown must not look alike in PostHog
    assert "volts_last" not in p
    assert "volts_age_s" not in p


def test_walk_start_carries_the_last_reading_with_its_age(client, captured, fake_walk_dir, monkeypatch, tmp_path):
    install_hwi(monkeypatch, tmp_path, holding=True)
    install_fake_pypot(monkeypatch, volts_deci=76, temps_c=lambda i: 40)
    client.get("/api/voltage")
    # the read happened "12 s ago": push the cache's clock back
    tnkr_server._last_voltage["at"] -= 12.0

    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    assert client.post("/api/walk/start", json={}).status_code == 200
    p = _events(captured, "api_request_completed", "/api/walk/start")[0]["properties"]
    assert p["volts_last"] == 7.6
    assert p["volts_health_last"] == "ok"
    assert 12.0 <= p["volts_age_s"] < 13.0


def test_a_failed_voltage_read_leaves_the_cache_alone(client, captured, monkeypatch, tmp_path):
    install_hwi(monkeypatch, tmp_path, holding=True)
    install_fake_pypot(monkeypatch, volts_deci=76, temps_c=lambda i: 40)
    client.get("/api/voltage")
    before = dict(tnkr_server._last_voltage)

    class DeadIO:
        def __init__(self, port, baudrate=1000000):
            raise OSError("could not open port")

    monkeypatch.setattr(sys.modules["pypot.feetech"], "FeetechSTS3215IO", DeadIO)
    assert client.get("/api/voltage").status_code == 503
    assert tnkr_server._last_voltage == before
