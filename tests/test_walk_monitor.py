"""Walk lifecycle: walk_ended events, crash labeling, and the stop/start race."""

import time

import pytest
from fastapi.testclient import TestClient

import tnkr_server


@pytest.fixture
def client():
    return TestClient(tnkr_server.app, raise_server_exceptions=False)


@pytest.fixture
def fake_walk_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"")
    return tmp_path


def write_walk_script(d, body):
    (d / "v2_rl_walk_mujoco.py").write_text(body)


def wait_for_walk_ended(captured, count=1, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ended = [e for e in captured if e["event"] == "walk_ended"]
        if len(ended) >= count:
            return ended
        time.sleep(0.02)
    raise AssertionError(f"walk_ended x{count} not captured within {timeout}s")


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
