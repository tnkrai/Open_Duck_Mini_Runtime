"""What a refused or rolled-back swap reports, and what it must never report.

Phase 6 of tnkr-studio's wired-physical-agents plan. A swap that was refused, or
accepted and then rolled back, is the highest-value fleet signal in the program: it
says a catalogue entry tnkr vouched for is bad on somebody else's robot, which is
exactly the responsibility decision d1 takes on.

That value buys no exemption from the opt-out, and the plan says so — reuse this
repo's existing stream rather than inventing a second one. These tests are mostly about
the second half of that sentence: what the events carry, and that silence really is
silent when somebody has asked for it.
"""

from __future__ import annotations

import json

import pytest

import tnkr_server
from conftest import install_component, write_walk_script
from mini_bdx_runtime import telemetry


@pytest.fixture
def captured_events(monkeypatch):
    """Every event the server emits, without a network anywhere."""
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        telemetry, "capture", lambda event, properties=None: events.append((event, properties or {}))
    )
    monkeypatch.setattr(tnkr_server.telemetry, "capture", lambda event, properties=None: events.append((event, properties or {})))
    return events


def _names(events):
    return [name for name, _ in events]


def _props(events, name):
    return next(props for n, props in events if n == name)


# --- a refused swap ---------------------------------------------------------

def test_a_refused_start_is_reported(client, fake_walk_dir, captured_events):
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    res = client.post("/api/walk/start", json={"componentId": "walk-v9"})
    assert res.status_code == 404

    assert "component_start_refused" in _names(captured_events)
    props = _props(captured_events, "component_start_refused")
    assert props["component_id"] == "walk-v9"
    assert props["reason"] == "COMPONENT_NOT_FOUND"


def test_a_contract_mismatch_reports_its_reason(client, fake_walk_dir, captured_events):
    """Which reason it was is the whole point — "refused" alone tells a fleet nothing."""
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    data = json.loads((fake_walk_dir / "model.manifest.json").read_text())
    blocks = data["obsSpec"]
    blocks[5], blocks[6] = blocks[6], blocks[5]
    data["obsSpec"] = blocks
    (fake_walk_dir / "model.manifest.json").write_text(json.dumps(data))

    client.post("/api/walk/start", json={})
    assert _props(captured_events, "component_start_refused")["reason"] == "COMPONENT_CONTRACT_MISMATCH"


def test_a_successful_start_reports_no_refusal(client, fake_walk_dir, captured_events):
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")
    assert "component_start_refused" not in _names(captured_events)


# --- what the events must not carry -----------------------------------------

def test_the_event_carries_no_path_no_filename_and_no_operator(client, fake_walk_dir, captured_events):
    """A component id and a reason. Nothing about the machine, the disk, or the person.

    The refusal detail is deliberately NOT sent: it contains absolute paths and hashes,
    which is a description of somebody's filesystem rather than of a catalogue entry.
    """
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    (fake_walk_dir / "model.onnx").write_bytes(b"truncated")
    client.post("/api/walk/start", json={})

    props = _props(captured_events, "component_start_refused")
    assert set(props) == {"component_id", "reason"}
    flat = json.dumps(props)
    assert "/" not in flat, f"a path leaked into telemetry: {flat}"
    assert ".onnx" not in flat


# --- the opt-out ------------------------------------------------------------

def test_the_opt_out_is_honoured(tmp_path, monkeypatch):
    """The gate asks for this to be ASSERTED, not merely implemented. A signal this
    valuable is exactly the one somebody would be tempted to except."""
    monkeypatch.setattr(telemetry.Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".tnkr-telemetry.json").write_text(json.dumps({"enabled": False}))
    telemetry._reset_state_for_tests()

    assert telemetry.is_enabled() is False


def test_an_opted_out_device_emits_nothing_at_all(tmp_path, monkeypatch):
    """The assertion that matters: not that is_enabled() returns False, but that
    capture() actually sends nothing when it does.

    An earlier version of this test checked is_enabled() and then guarded the real
    assertion behind an `if` — which would have passed silently on a machine where
    telemetry was on. A test that can decline to check anything is worse than no test.
    """
    monkeypatch.setattr(telemetry.Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".tnkr-telemetry.json").write_text(json.dumps({"enabled": False}))
    telemetry._reset_state_for_tests()

    sent: list = []
    monkeypatch.setattr(telemetry, "_get_client", lambda: sent.append("client asked for") or None)

    telemetry.capture("component_start_refused", {"component_id": "walk-v9"})
    assert sent == [], "an opted-out device reached for a client"


def test_capture_is_fail_silent_while_enabled(tmp_path, monkeypatch):
    """A robot must not fall over because an analytics endpoint did.

    Deliberately with telemetry ENABLED. Run this opted out and capture() returns
    before it ever reaches a client, so the test would pass without exercising the
    thing it names — which is how a fail-silent guarantee stops being tested.
    """
    monkeypatch.setattr(telemetry.Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".tnkr-telemetry.json").write_text(json.dumps({"enabled": True}))
    telemetry._reset_state_for_tests()
    assert telemetry.is_enabled() is True, "the test must run with telemetry ON"

    def _boom():
        raise RuntimeError("no network")

    monkeypatch.setattr(telemetry, "_get_client", _boom)
    telemetry.capture("anything", {"a": 1})  # must not raise
