"""Unit tests for mini_bdx_runtime.telemetry — consent, identity, fail-silence."""

import json
import sys

import pytest

from mini_bdx_runtime import telemetry


# ── is_enabled: env var > file > default ─────────────────────────────────────

def test_default_enabled_when_no_file():
    assert telemetry.is_enabled() is True


def test_env_zero_disables_even_when_file_says_enabled(monkeypatch):
    telemetry.TELEMETRY_FILE.write_text(json.dumps({"device_id": "x", "enabled": True}))
    monkeypatch.setenv("TNKR_TELEMETRY", "0")
    assert telemetry.is_enabled() is False


def test_env_one_enables_even_when_file_says_disabled(monkeypatch):
    telemetry.TELEMETRY_FILE.write_text(json.dumps({"device_id": "x", "enabled": False}))
    monkeypatch.setenv("TNKR_TELEMETRY", "1")
    assert telemetry.is_enabled() is True


def test_file_disabled(monkeypatch):
    telemetry.TELEMETRY_FILE.write_text(json.dumps({"device_id": "x", "enabled": False}))
    assert telemetry.is_enabled() is False


# ── device_id ────────────────────────────────────────────────────────────────

def test_device_id_lazily_creates_file_with_notice(capsys):
    did = telemetry.device_id()
    assert len(did) == 36  # uuid4
    saved = json.loads(telemetry.TELEMETRY_FILE.read_text())
    assert saved["device_id"] == did
    assert saved["enabled"] is True
    assert saved["notice_version"] == 1
    out = capsys.readouterr().out
    assert "[telemetry]" in out and "TNKR_TELEMETRY=0" in out


def test_device_id_stable_across_calls_and_processes():
    first = telemetry.device_id()
    telemetry._reset_state_for_tests()  # simulate a new process
    assert telemetry.device_id() == first


def test_device_id_reuses_existing_file_without_notice(capsys):
    telemetry.TELEMETRY_FILE.write_text(json.dumps({"device_id": "abc-123", "enabled": True}))
    assert telemetry.device_id() == "abc-123"
    assert "[telemetry]" not in capsys.readouterr().out


def test_device_id_fail_silent_on_unwritable_home(monkeypatch, tmp_path):
    monkeypatch.setattr(
        telemetry, "TELEMETRY_FILE", tmp_path / "nope" / "deep" / "t.json"
    )
    did = telemetry.device_id()  # mkdir-less path: write fails, id still returned
    assert len(did) == 36
    assert telemetry.device_id() == did  # cached within the process


# ── capture ──────────────────────────────────────────────────────────────────

def test_capture_records_event_with_base_props(captured):
    telemetry.capture("test_event", {"check": True})
    assert len(captured) == 1
    e = captured[0]
    assert e["event"] == "test_event"
    assert e["distinct_id"] == telemetry.device_id()
    props = e["properties"]
    assert props["check"] is True
    assert props["source"] == "openduck-runtime"
    assert "runtime_version" in props and "arch" in props and "python_version" in props


def test_capture_noop_when_disabled(captured, monkeypatch):
    monkeypatch.setenv("TNKR_TELEMETRY", "0")
    telemetry.capture("test_event")
    assert captured == []


def test_capture_disabled_never_creates_telemetry_file(captured, monkeypatch):
    monkeypatch.setenv("TNKR_TELEMETRY", "0")
    telemetry.capture("test_event")
    assert not telemetry.TELEMETRY_FILE.exists()


def test_capture_noop_when_posthog_missing(monkeypatch):
    # Simulate `import posthog` raising ImportError on a robot without the dep.
    monkeypatch.setitem(sys.modules, "posthog", None)
    telemetry.capture("test_event")  # must not raise


def test_set_on_first_event_only(captured):
    telemetry.capture("first")
    telemetry.capture("second")
    assert "$set" in captured[0]["properties"]
    assert "$set" not in captured[1]["properties"]


def test_set_sticky_merges_into_later_events_and_next_set(captured):
    telemetry.capture("before")
    telemetry.set_sticky(servo_adapter_chip="CH343")
    telemetry.capture("after")
    assert "servo_adapter_chip" not in captured[0]["properties"]
    assert captured[1]["properties"]["servo_adapter_chip"] == "CH343"
    assert captured[1]["properties"]["$set"]["servo_adapter_chip"] == "CH343"


def test_capture_fail_silent_when_client_raises(monkeypatch):
    class Exploding:
        def capture(self, *a, **k):
            raise ConnectionError("network down")

    monkeypatch.setattr(telemetry, "_client", Exploding())
    telemetry.capture("test_event")  # must not raise


# ── device_properties ────────────────────────────────────────────────────────

def test_device_properties_complete_and_cached():
    props = telemetry.device_properties()
    for key in ("arch", "python_version", "pi_model", "os_release", "ram_mb", "runtime_version"):
        assert key in props
    assert telemetry.device_properties() is props
