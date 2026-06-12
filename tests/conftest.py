"""Shared fixtures: stub rustypot, fake posthog client, isolated telemetry file.

No robot hardware and no network are needed — the telemetry client is replaced
with an in-memory fake, and the rustypot Rust extension with a stub that
raises on connect. An autouse guard makes it impossible for any test (or any
thread a test leaks) to construct a REAL posthog client and ship junk events
to production.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Stub rustypot/onnxruntime must be importable BEFORE mini_bdx_runtime.
sys.path.insert(0, str(REPO_ROOT / "tests" / "stubs"))
# tnkr_server.py lives in scripts/ (not a package).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# `python -m pytest` puts the CWD first on sys.path, where the OUTER
# mini_bdx_runtime/ directory shadows the installed inner package — make the
# inner package's parent win for both invocation styles.
sys.path.insert(0, str(REPO_ROOT / "mini_bdx_runtime"))

import pytest  # noqa: E402

from mini_bdx_runtime import telemetry  # noqa: E402
import tnkr_server  # noqa: E402

# Original _get_client, captured before the autouse guard patches it — for
# tests that need to exercise the real lazy-import path.
REAL_GET_CLIENT = telemetry._get_client


class FakePosthogClient:
    """Records capture() calls; asserts the kwargs-only calling convention."""

    def __init__(self):
        self.events = []
        self.flushed = 0
        self.shutdowns = 0

    def capture(self, *args, **kwargs):
        assert args == (), (
            "telemetry must call capture() with keyword args only "
            "(posthog-python 6.x made distinct_id keyword-only)"
        )
        self.events.append(kwargs)

    def flush(self):
        self.flushed += 1

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture(autouse=True)
def isolated_telemetry(tmp_path, monkeypatch):
    """Fresh telemetry state per test, and a hard guard against ever creating
    a real posthog client: _get_client only returns what a test installed."""
    monkeypatch.setattr(telemetry, "TELEMETRY_FILE", tmp_path / ".tnkr-telemetry.json")
    monkeypatch.delenv("TNKR_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_get_client", lambda: telemetry._client)
    telemetry._reset_state_for_tests()
    yield
    # Never leak a walk subprocess (or its monitor thread) past a test.
    with tnkr_server._walk_lock:
        tnkr_server.stop_walk_process()
    telemetry._reset_state_for_tests()


@pytest.fixture
def real_get_client():
    return REAL_GET_CLIENT


@pytest.fixture
def fake_posthog(monkeypatch):
    """Replace the real posthog client with an in-memory recorder."""
    client = FakePosthogClient()
    monkeypatch.setattr(telemetry, "_client", client)
    return client


@pytest.fixture
def captured(fake_posthog):
    """Convenience: the list of captured event dicts."""
    return fake_posthog.events


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(tnkr_server.app, raise_server_exceptions=False)


@pytest.fixture
def fake_walk_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"")
    return tmp_path


def write_walk_script(d, body):
    (d / "v2_rl_walk_mujoco.py").write_text(body)


def wait_for_walk_ended(captured, count=1, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ended = [e for e in captured if e["event"] == "walk_ended"]
        if len(ended) >= count:
            return ended
        time.sleep(0.02)
    raise AssertionError(f"walk_ended x{count} not captured within {timeout}s")
