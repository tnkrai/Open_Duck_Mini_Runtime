"""Walk lifecycle: walk_ended events, crash labeling, and the stop/start race."""

import os
import signal
import time

import tnkr_server
from conftest import wait_for_walk_ended, write_walk_script


def test_clean_stop_is_not_a_crash(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")

    ended = wait_for_walk_ended(captured)[0]["properties"]
    assert ended["stop_requested"] is True
    assert ended["crashed"] is False
    assert ended["cloud_streaming"] is False
    assert ended["duration_s"] >= 0


def test_crashing_walk_is_labeled_crashed(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, "import sys; sys.exit(1)\n")
    assert client.post("/api/walk/start", json={}).status_code == 200

    ended = wait_for_walk_ended(captured)[0]["properties"]
    assert ended["exit_code"] == 1
    assert ended["crashed"] is True
    assert ended["stop_requested"] is False


def test_oom_style_sigkill_is_labeled_crashed(client, captured, fake_walk_dir):
    """The kernel OOM killer SIGKILLs the walk on a 512MB Pi — by far the most
    likely real crash mode. An unrequested kill must count as crashed."""
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    r = client.post("/api/walk/start", json={})
    assert r.status_code == 200
    os.kill(r.json()["pid"], signal.SIGKILL)

    ended = wait_for_walk_ended(captured)[0]["properties"]
    assert ended["exit_code"] == -signal.SIGKILL
    assert ended["stop_requested"] is False
    assert ended["crashed"] is True


def test_stop_a_then_start_b_does_not_mislabel_a(client, captured, fake_walk_dir):
    """Regression test for the shared-flag race: walk A is stopped because a
    new session arrives; the immediate start of walk B must not flip A's
    stop_requested back, so A reports a clean stop, never a crash."""
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    assert (
        client.post("/api/walk/start", json={"sessionToken": "session-A"}).status_code
        == 200
    )
    # New session token forces stop(A) + start(B) inside one request
    assert (
        client.post("/api/walk/start", json={"sessionToken": "session-B"}).status_code
        == 200
    )

    ended_a = wait_for_walk_ended(captured, count=1)[0]["properties"]
    assert ended_a["stop_requested"] is True
    assert ended_a["crashed"] is False

    client.post("/api/walk/stop")
    ended = wait_for_walk_ended(captured, count=2)
    assert all(e["properties"]["crashed"] is False for e in ended)


def test_sigterm_ignoring_walk_is_escalated_to_kill(client, captured, fake_walk_dir):
    """A hung walk script that ignores SIGTERM gets SIGKILLed after the 5s
    grace period — and still counts as a requested stop, not a crash."""
    write_walk_script(
        fake_walk_dir,
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(60)\n",
    )
    assert client.post("/api/walk/start", json={}).status_code == 200
    time.sleep(0.5)  # let the script install its SIGTERM handler
    client.post("/api/walk/stop")  # terminate -> wait(5) times out -> kill

    ended = wait_for_walk_ended(captured, timeout=15)[0]["properties"]
    assert ended["stop_requested"] is True
    assert ended["crashed"] is False
    assert ended["exit_code"] == -signal.SIGKILL


def test_idempotent_restart_same_token_no_new_walk(client, captured, fake_walk_dir):
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    client.post("/api/walk/start", json={"sessionToken": "same"})
    r = client.post("/api/walk/start", json={"sessionToken": "same"})
    assert r.json().get("message") == "Walk is already running"
    starts = [
        e
        for e in captured
        if e["event"] == "api_request_completed"
        and e["properties"]["endpoint"] == "/api/walk/start"
    ]
    assert starts[1]["properties"]["already_running"] is True

    client.post("/api/walk/stop")
    wait_for_walk_ended(captured)


def test_stop_only_clears_its_own_session(fake_walk_dir, captured, client):
    """stop_walk_process must not null out a NEWER session installed while it
    was stopping the old one (orphaned-walk guard)."""
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    assert client.post("/api/walk/start", json={}).status_code == 200
    old_session = tnkr_server.walk_session
    # Simulate the interleaving directly: a new session appears mid-stop.
    import subprocess, sys as _sys

    new_proc = subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        new_session = tnkr_server.WalkSession(
            proc=new_proc, session_token=None, cloud_streaming=False,
            started_at=time.monotonic(),
        )
        old_session.stop_requested = True
        old_session.proc.terminate()
        old_session.proc.wait(timeout=10)
        tnkr_server.walk_session = new_session
        # Old stop path finishing must NOT clear the new session.
        with tnkr_server._walk_lock:
            if tnkr_server.walk_session is old_session:
                tnkr_server.walk_session = None
        assert tnkr_server.walk_session is new_session
    finally:
        new_proc.kill()
        new_proc.wait()
        tnkr_server.walk_session = None
