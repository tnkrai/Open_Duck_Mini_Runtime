"""Steering telemetry: what /api/commands leaves behind without ever being captured.

/api/commands is excluded from request telemetry (10-50 Hz would spend the rate cap
on itself). Instead each write folds into the walk session's CommandStats, walk_ended
carries the summary, and the first nonzero command is one event. The monitor thread
also watches the walk's own snapshot so a walk that is alive but never runs a tick is
visible as first_tick_after_s: None.
"""

import json
import math
import os
import signal
import time

import pytest

import tnkr_server
from mini_bdx_runtime import walk_telemetry
from conftest import wait_for_walk_ended, write_walk_script

SLEEP_SCRIPT = "import time; time.sleep(30)\n"
ZERO = [0.0] * 7


@pytest.fixture(autouse=True)
def command_file_off_pi(tmp_path, monkeypatch):
    """The command file lives in /dev/shm, which only exists on the Pi."""
    monkeypatch.setattr(tnkr_server, "COMMAND_FILE", str(tmp_path / "tnkr_remote_commands.json"))


def events_named(captured, name):
    return [e for e in captured if e["event"] == name]


def post_command(client, commands):
    r = client.post("/api/commands", json={"commands": commands})
    assert r.status_code == 200, r.text
    return r


# ── command summary on walk_ended ────────────────────────────────────────────

def test_walk_ended_carries_the_steering_summary(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200

    post_command(client, ZERO)
    post_command(client, [0.15, 0, 0, 0, 0, 0, 0])
    post_command(client, [0, 0, -1.0, 0, 0.3, 0, 0])
    client.post("/api/walk/stop")

    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["commands_received"] == 3
    assert p["commands_nonzero"] == 2
    assert p["max_abs_vx"] == pytest.approx(0.15)
    assert p["max_abs_vy"] == 0.0
    assert p["max_abs_wz"] == pytest.approx(1.0)
    assert p["head_commands_used"] is True
    assert p["first_command_after_s"] is not None and p["first_command_after_s"] >= 0
    assert p["first_nonzero_command_after_s"] >= p["first_command_after_s"]
    assert p["longest_command_gap_s"] >= 0


def test_first_nonzero_command_is_one_event(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200

    post_command(client, ZERO)  # stopped is not steering
    post_command(client, [0.15, 0, 0, 0, 0, 0, 0])
    post_command(client, [0.15, 0, 0, 0, 0, 0, 0])  # the second one is not news
    post_command(client, [0, 0.2, 0, 0, 0, 0, 0])

    firsts = events_named(captured, "walk_first_command")
    assert len(firsts) == 1
    p = firsts[0]["properties"]
    assert p["vx"] == pytest.approx(0.15)
    assert p["vy"] == 0.0
    assert p["wz"] == 0.0
    assert p["head_used"] is False
    assert p["after_s"] >= 0
    assert p["walk_input"] == "keyboard"
    assert p["remote"] is True
    client.post("/api/walk/stop")
    wait_for_walk_ended(captured)


def test_head_only_command_counts_as_steering(client, captured, fake_walk_dir):
    """The arrow keys move the head through the same stream; a duck that only ever
    looked around still had a client reaching its loop."""
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200

    post_command(client, [0, 0, 0, 0, -0.78, 0, 0])

    firsts = events_named(captured, "walk_first_command")
    assert len(firsts) == 1
    assert firsts[0]["properties"]["head_used"] is True
    assert firsts[0]["properties"]["vx"] == 0.0
    client.post("/api/walk/stop")
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["commands_nonzero"] == 1
    assert p["head_commands_used"] is True


def test_zero_only_stream_is_received_but_never_steers(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200

    for _ in range(3):
        post_command(client, ZERO)
    client.post("/api/walk/stop")

    assert events_named(captured, "walk_first_command") == []
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["commands_received"] == 3
    assert p["commands_nonzero"] == 0
    assert p["first_command_after_s"] is not None
    assert p["first_nonzero_command_after_s"] is None


def test_no_commands_at_all_reads_as_none(client, captured, fake_walk_dir):
    """The silent case: Studio never sent a thing. Distinct from zeros."""
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")

    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["commands_received"] == 0
    assert p["first_command_after_s"] is None
    assert p["first_nonzero_command_after_s"] is None
    assert p["max_abs_vx"] == 0.0


def test_commands_outside_a_walk_are_written_but_not_counted(client, captured):
    """Pre-existing contract: the file is written whether or not a walk runs. No
    session means nothing to attribute the write to, and no event."""
    post_command(client, [0.15, 0, 0, 0, 0, 0, 0])
    assert events_named(captured, "walk_first_command") == []
    assert events_named(captured, "api_request_completed") == []  # still excluded


def test_command_stream_is_still_not_a_request_event(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200
    for _ in range(20):
        post_command(client, [0.15, 0, 0, 0, 0, 0, 0])
    requests = [
        e
        for e in captured
        if e["event"].startswith("api_request") and e["properties"]["endpoint"] == "/api/commands"
    ]
    assert requests == []
    client.post("/api/walk/stop")
    wait_for_walk_ended(captured)


def test_stats_reset_per_walk(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200
    post_command(client, [0.15, 0, 0, 0, 0, 0, 0])
    client.post("/api/walk/stop")
    wait_for_walk_ended(captured, count=1)

    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")
    second = wait_for_walk_ended(captured, count=2)[1]["properties"]
    assert second["commands_received"] == 0
    assert len(events_named(captured, "walk_first_command")) == 1


# ── who steered, and how the walk was told to listen ─────────────────────────

def test_walk_ended_reports_keyboard_walks_as_remote(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["walk_input"] == "keyboard"
    assert p["remote"] is True


def test_walk_ended_reports_pad_walks_as_not_remote(client, captured, fake_walk_dir, monkeypatch):
    monkeypatch.setattr(tnkr_server, "joystick_present", lambda: True)
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={"input": "pad"}).status_code == 200
    client.post("/api/walk/stop")
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["walk_input"] == "pad"
    assert p["remote"] is False


def test_unknown_input_string_is_recorded_as_keyboard(client, captured, fake_walk_dir):
    """walk_flags treats anything but "pad" as keyboard; telemetry must too, so a
    client cannot put its own string into the event."""
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={"input": "x" * 500}).status_code == 200
    client.post("/api/walk/stop")
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["walk_input"] == "keyboard"
    assert p["remote"] is True


# ── start_paused, read at launch ─────────────────────────────────────────────

@pytest.mark.parametrize("start_paused", [True, False])
def test_walk_ended_reads_start_paused_from_config(
    client, captured, fake_walk_dir, tmp_path, monkeypatch, start_paused
):
    cfg = tmp_path / "duck_config.json"
    cfg.write_text(json.dumps({"start_paused": start_paused, "joints_offsets": {}}))
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", str(cfg))
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")
    assert wait_for_walk_ended(captured)[0]["properties"]["start_paused"] is start_paused


def test_unreadable_config_reports_start_paused_unknown(client, captured, fake_walk_dir, monkeypatch):
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", "/nonexistent/duck_config.json")
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")
    assert wait_for_walk_ended(captured)[0]["properties"]["start_paused"] is None


# ── the walk's own snapshot: did the loop ever run a tick? ───────────────────

@pytest.fixture
def snapshot_file(tmp_path, monkeypatch):
    path = tmp_path / "tnkr_walk_telemetry.json"
    monkeypatch.setattr(walk_telemetry, "TELEMETRY_FILE", str(path))
    return path


def snapshot_writer_script(path, write_for_s):
    """A walk stand-in: writes a fresh snapshot every 0.1 s for `write_for_s`, then
    stays alive without writing (a stalled or A-button-paused loop)."""
    return (
        "import json, os, time\n"
        f"P = {str(path)!r}\n"
        f"end = time.time() + {write_for_s}\n"
        "while time.time() < end:\n"
        "    with open(P + '.tmp', 'w') as f:\n"
        "        json.dump({'timestamp': time.time(), 'joints': {}, 'imu': None}, f)\n"
        "    os.replace(P + '.tmp', P)\n"
        "    time.sleep(0.1)\n"
        "time.sleep(30)\n"
    )


def test_frozen_walk_never_writes_a_snapshot(client, captured, fake_walk_dir, snapshot_file):
    """Alive, but no policy tick ever ran (start_paused, or every observation read
    failing). This is the case that used to leave no trace at all."""
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200
    time.sleep(1.2)  # a couple of monitor polls with the process alive
    client.post("/api/walk/stop")
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["first_tick_after_s"] is None
    assert p["loop_silent_s"] == 0.0  # never ticked, so never *went* silent


def test_running_walk_reports_first_tick_and_then_silence(client, captured, fake_walk_dir, snapshot_file):
    """Writes for 0.8 s, then alive and silent. Silence starts 1 s after the last
    write (the snapshot's freshness window), so the test leaves well over a second
    of it before stopping."""
    write_walk_script(fake_walk_dir, snapshot_writer_script(snapshot_file, 0.8))
    assert client.post("/api/walk/start", json={}).status_code == 200
    time.sleep(4.0)
    client.post("/api/walk/stop")
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["first_tick_after_s"] is not None
    assert 0.0 <= p["first_tick_after_s"] < 2.0
    assert p["loop_silent_s"] >= 0.5


def test_walk_that_ticks_to_the_end_is_not_silent(client, captured, fake_walk_dir, snapshot_file):
    write_walk_script(fake_walk_dir, snapshot_writer_script(snapshot_file, 60))
    assert client.post("/api/walk/start", json={}).status_code == 200
    time.sleep(2.0)
    client.post("/api/walk/stop")
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["first_tick_after_s"] is not None
    assert p["loop_silent_s"] == 0.0


@pytest.mark.parametrize("garbage", ["null", "[]", '{"timestamp": null}', "{not json"])
def test_malformed_snapshot_file_costs_a_poll_not_the_row(
    client, captured, fake_walk_dir, snapshot_file, garbage
):
    """Anything can land in /dev/shm. A shape the reader does not expect must not
    kill the monitor thread, or the walk runs on with no walk_ended ever sent."""
    snapshot_file.write_text(garbage)
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200
    time.sleep(1.2)
    client.post("/api/walk/stop")
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["first_tick_after_s"] is None


# ── a crashed walk does not keep collecting ──────────────────────────────────

def test_commands_after_a_crash_do_not_reach_the_dead_walk(client, captured, fake_walk_dir):
    """After an OOM-style kill, walk_session stays set until the next start or stop
    while Studio keeps streaming. Those writes belong to no walk: no first-command
    event for a walk whose walk_ended has already gone out."""
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    r = client.post("/api/walk/start", json={})
    assert r.status_code == 200
    os.kill(r.json()["pid"], signal.SIGKILL)
    ended = wait_for_walk_ended(captured)[0]["properties"]
    assert ended["crashed"] is True
    assert ended["commands_received"] == 0

    post_command(client, [0.15, 0, 0, 0, 0, 0, 0])
    assert events_named(captured, "walk_first_command") == []
    assert tnkr_server.walk_session is not None  # the guard, not a cleared session
    assert tnkr_server.walk_session.commands.received == 0


# ── non-finite input ─────────────────────────────────────────────────────────

def test_nan_and_inf_are_counted_and_never_reach_an_event(client, captured, fake_walk_dir):
    """JSON's parser and pydantic's float both let NaN/Infinity through. In a max
    they would pin the column; in an event body they are a batch PostHog rejects."""
    write_walk_script(fake_walk_dir, SLEEP_SCRIPT)
    assert client.post("/api/walk/start", json={}).status_code == 200

    r = client.post(
        "/api/commands",
        content='{"commands": [NaN, Infinity, 0.5, 0, 0, 0, 0]}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 200
    post_command(client, [0.15, 0, 0, 0, 0, 0, 0])
    client.post("/api/walk/stop")

    first = events_named(captured, "walk_first_command")
    assert len(first) == 1
    fp = first[0]["properties"]
    assert fp["vx"] == 0.0 and fp["vy"] == 0.0 and fp["wz"] == pytest.approx(0.5)
    p = wait_for_walk_ended(captured)[0]["properties"]
    assert p["commands_non_finite"] == 1
    assert p["max_abs_vx"] == pytest.approx(0.15)
    assert p["max_abs_vy"] == 0.0
    for e in captured:
        for v in e["properties"].values():
            if isinstance(v, float):
                assert math.isfinite(v), (e["event"], v)


# ── the accumulator on its own ───────────────────────────────────────────────

def test_command_stats_fold_gaps_and_maxima():
    stats = tnkr_server.CommandStats()
    assert stats.record([0, 0, 0, 0, 0, 0, 0], 10.0) is False
    assert stats.record([0.15, 0, 0, 0, 0, 0, 0], 10.1) is True
    assert stats.record([0.6, -0.2, 0, 0, 0, 0, 0], 12.1) is False  # 2 s gap
    assert stats.record([0, 0, 0.5], 12.2) is False  # short vectors are fine
    assert stats.record([math.nan, math.inf, 0, 0, 0, 0, 0], 12.3) is False
    p = stats.properties(started_at=9.0)
    assert p["commands_received"] == 5
    assert p["commands_non_finite"] == 1
    assert p["commands_nonzero"] == 3
    assert p["first_command_after_s"] == pytest.approx(1.0)
    assert p["first_nonzero_command_after_s"] == pytest.approx(1.1)
    assert p["longest_command_gap_s"] == pytest.approx(2.0)
    assert p["max_abs_vx"] == pytest.approx(0.6)
    assert p["max_abs_vy"] == pytest.approx(0.2)
    assert p["max_abs_wz"] == pytest.approx(0.5)
    assert p["head_commands_used"] is False
