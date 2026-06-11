"""Shared fixtures: stub rustypot, fake posthog client, isolated telemetry file.

No robot hardware and no network are needed — the telemetry client is replaced
with an in-memory fake, and the rustypot Rust extension with a stub that
raises on connect.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Stub rustypot must be importable BEFORE mini_bdx_runtime / tnkr_server.
sys.path.insert(0, str(REPO_ROOT / "tests" / "stubs"))
# tnkr_server.py lives in scripts/ (not a package).
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

from mini_bdx_runtime import telemetry  # noqa: E402


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
    """Every test gets a fresh telemetry state and its own telemetry file."""
    monkeypatch.setattr(telemetry, "TELEMETRY_FILE", tmp_path / ".tnkr-telemetry.json")
    monkeypatch.delenv("TNKR_TELEMETRY", raising=False)
    telemetry._reset_state_for_tests()
    yield
    telemetry._reset_state_for_tests()


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
