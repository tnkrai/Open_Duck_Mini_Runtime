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


@pytest.mark.parametrize(
    "val,expected",
    [("", False), ("0", False), ("FALSE", False), ("off", False), ("Off", False),
     ("1", True), ("true", True), ("2", True), ("yes", True)],
)
def test_is_enabled_env_boundary_values(monkeypatch, val, expected):
    monkeypatch.setenv("TNKR_TELEMETRY", val)
    assert telemetry.is_enabled() is expected


def test_file_disabled():
    telemetry.TELEMETRY_FILE.write_text(json.dumps({"device_id": "x", "enabled": False}))
    assert telemetry.is_enabled() is False


def test_corrupt_file_counts_as_opted_out(captured):
    # A torn/corrupt consent file must NEVER silently re-enroll a user.
    telemetry.TELEMETRY_FILE.write_text('{"enabled": false,,,')
    assert telemetry.is_enabled() is False
    telemetry.capture("test_event")
    assert captured == []


def test_corrupt_file_is_never_overwritten():
    corrupt = '{"enabled": false,,,'
    telemetry.TELEMETRY_FILE.write_text(corrupt)
    did = telemetry.device_id()  # per-process id, no clobber
    assert len(did) == 36
    assert telemetry.TELEMETRY_FILE.read_text() == corrupt


def test_handwritten_opt_out_without_device_id_is_preserved(capsys):
    # The README documents `{"enabled": false}` as a valid opt-out. Lazily
    # adding a device_id must MERGE, never flip enabled back to true.
    telemetry.TELEMETRY_FILE.write_text(json.dumps({"enabled": False}))
    did = telemetry.device_id()
    assert len(did) == 36
    saved = json.loads(telemetry.TELEMETRY_FILE.read_text())
    assert saved["enabled"] is False
    assert saved["device_id"] == did
    # No "telemetry enabled" notice for an opted-out device
    assert "[telemetry]" not in capsys.readouterr().out
    assert telemetry.is_enabled() is False


def test_enabled_decision_is_cached_off_the_hot_path():
    telemetry.TELEMETRY_FILE.write_text(json.dumps({"device_id": "x", "enabled": True}))
    assert telemetry.is_enabled() is True
    # Within the TTL the file is not re-read: flipping it has no instant effect
    telemetry.TELEMETRY_FILE.write_text(json.dumps({"device_id": "x", "enabled": False}))
    assert telemetry.is_enabled() is True
    telemetry._enabled_cache = None  # TTL expiry
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
    # Atomic write leaves no temp file behind
    assert not telemetry.TELEMETRY_FILE.with_suffix(".json.tmp").exists()


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


def test_capture_noop_when_posthog_missing(monkeypatch, real_get_client):
    # Restore the real _get_client so the import path runs, but make the
    # `import posthog` raise (robot without the dep installed).
    monkeypatch.setattr(telemetry, "_get_client", real_get_client)
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


def test_set_sticky_skips_none_values(captured):
    telemetry.set_sticky(servo_adapter_chip="FTDI")
    telemetry.set_sticky(servo_adapter_chip=None)  # must not clobber
    telemetry.capture("event")
    assert captured[0]["properties"]["servo_adapter_chip"] == "FTDI"


def test_capture_fail_silent_when_client_raises(monkeypatch):
    class Exploding:
        def capture(self, *a, **k):
            raise ConnectionError("network down")

    monkeypatch.setattr(telemetry, "_client", Exploding())
    telemetry.capture("test_event")  # must not raise


# ── privacy scrubbing ────────────────────────────────────────────────────────

def test_scrub_removes_home_paths_and_username(captured):
    import getpass
    from pathlib import Path

    home = str(Path.home())
    user = getpass.getuser()
    telemetry.capture(
        "test_event",
        {
            "error_message": f"[Errno 13] Permission denied: '{home}/duck_config.json'",
            "error_tail": ["/home/someoneelse/Open_Duck_Mini_Runtime failed"],
        },
    )
    blob = json.dumps(captured[0]["properties"])
    assert home not in blob
    assert "/home/someoneelse" not in blob
    if len(user) >= 4:
        assert user not in blob
    assert "~" in blob or "<user>" in blob


def test_scrub_survives_degenerate_home(monkeypatch):
    # HOME=/ (some init contexts) must not rewrite every path separator.
    monkeypatch.setattr(telemetry, "_HOME_STR", "/")
    assert telemetry._scrub("GET /api/walk/start on /dev/ttyACM0") == (
        "GET /api/walk/start on /dev/ttyACM0"
    )


# ── rate limiting ────────────────────────────────────────────────────────────

def test_rate_limit_caps_api_request_events_per_minute(captured):
    for i in range(telemetry.RATE_LIMIT_PER_MIN + 10):
        telemetry.capture("api_request_failed")
    assert len(captured) == telemetry.RATE_LIMIT_PER_MIN
    # Next window reports how many were dropped
    telemetry._rate_window_start -= 61
    telemetry.capture("api_request_completed")
    assert captured[-1]["properties"]["events_dropped"] == 10


def test_lifecycle_events_bypass_the_rate_limit(captured):
    # An api_request flood must never starve crash/lifecycle data.
    for i in range(telemetry.RATE_LIMIT_PER_MIN + 10):
        telemetry.capture("api_request_failed")
    telemetry.capture("walk_ended", {"crashed": True})
    telemetry.capture("imu_calibration_failed")
    names = [e["event"] for e in captured]
    assert "walk_ended" in names and "imu_calibration_failed" in names


# ── flush / shutdown ─────────────────────────────────────────────────────────

def test_flush_and_shutdown_delegate_to_client(fake_posthog):
    telemetry.flush()
    telemetry.shutdown()
    assert fake_posthog.flushed == 1
    assert fake_posthog.shutdowns == 1


def test_flush_and_shutdown_noop_without_client():
    telemetry.flush()
    telemetry.shutdown()  # _client is None: must not raise


def test_shutdown_is_time_bounded():
    import time as _time

    class Hanging:
        def shutdown(self):
            _time.sleep(30)

        def flush(self):
            _time.sleep(30)

    telemetry._client = Hanging()
    start = _time.monotonic()
    telemetry.shutdown(timeout=0.2)
    telemetry.flush(timeout=0.2)
    assert _time.monotonic() - start < 5  # never hangs a systemd stop


# ── device_properties ────────────────────────────────────────────────────────

def test_device_properties_complete_and_cached():
    props = telemetry.device_properties()
    for key in ("arch", "python_version", "pi_model", "os_release", "ram_mb", "runtime_version"):
        assert key in props
    assert telemetry.device_properties() is props


# ── bash/python contract ─────────────────────────────────────────────────────

def test_setup_sh_posthog_constants_match_python():
    import re
    from pathlib import Path

    sh = (Path(__file__).parent.parent / "scripts" / "setup.sh").read_text()
    assert re.search(r'POSTHOG_KEY="([^"]+)"', sh).group(1) == telemetry.POSTHOG_API_KEY
    assert re.search(r'POSTHOG_HOST="([^"]+)"', sh).group(1) == telemetry.POSTHOG_HOST
