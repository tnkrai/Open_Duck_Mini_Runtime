"""Shared fixtures: stub rustypot, fake onnxruntime, fake posthog client, isolated
telemetry file.

No robot hardware and no network are needed — the telemetry client is replaced
with an in-memory fake, and the rustypot Rust extension with a stub that
raises on connect. An autouse guard makes it impossible for any test (or any
thread a test leaks) to construct a REAL posthog client and ship junk events
to production.

``onnxruntime`` is a stub too, but a configurable one: the ``onnx_specs`` fixture
below declares what graph a given .onnx path presents, which is how the contract
check that guards the servos gets tested without a 50 MB native wheel in CI.
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

import onnxruntime as onnxruntime_double  # noqa: E402  (tests/stubs, not the wheel)
import pytest  # noqa: E402

from mini_bdx_runtime import telemetry  # noqa: E402
from mini_bdx_runtime.policy_contract import OBS_INPUT_NAME  # noqa: E402
import tnkr_server  # noqa: E402

# If the real wheel ever shadows the stub, every graph a test registers would be ignored
# and the contract-check tests would quietly assert nothing.
assert hasattr(onnxruntime_double, "_register"), (
    f"onnxruntime resolved to {getattr(onnxruntime_double, '__file__', '?')}, not the "
    "double in tests/stubs — check the sys.path inserts above."
)

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


class OnnxSpecs:
    """Per-test control of the onnxruntime double.

    Registration goes through a fixture rather than the module's registry so that a
    malformed graph declared by one test cannot survive into the next — two tests writing
    a ``model.onnx`` into their own ``tmp_path`` still collide if the registry is never
    cleared, and the failure would look like a bug in the code under test.
    """

    def __init__(self, module):
        self._module = module

    def register(self, path, inputs=(), outputs=(), *, invalid=False, delay_s=0.0):
        """Declare the graph a session over ``path`` presents.

        ``inputs``/``outputs`` are ``(name, shape, type)`` tuples, e.g.
        ``("obs", [1, 101], "tensor(float)")``. ``invalid=True`` makes construction raise
        the way real onnxruntime does for a file that is not a parseable model.
        """
        self._module._register(
            path, inputs=inputs, outputs=outputs, invalid=invalid, delay_s=delay_s
        )

    def valid(self, path, *, obs_dim, act_dim, delay_s=0.0):
        """The shape a conforming duck policy has, as read off BEST_WALK_ONNX_2.onnx.

        The widths stay the caller's business — a test asserting an accepted policy should
        say what it considers accepted — but the input NAME comes from the contract, so a
        rename there cannot leave this fixture quietly building the old graph.
        """
        self.register(
            path,
            inputs=[(OBS_INPUT_NAME, [1, obs_dim], "tensor(float)")],
            outputs=[("continuous_actions", [1, act_dim], "tensor(float)")],
            delay_s=delay_s,
        )

    @property
    def constructed(self):
        """Paths a session was constructed for, in order.

        What lets a test assert the *order* of the check: an oversize or wrong-hash file
        must be refused with this list still empty, i.e. without the parser ever being
        handed the file.
        """
        return list(self._module.CONSTRUCTED)


@pytest.fixture
def onnx_specs():
    onnxruntime_double._clear()
    yield OnnxSpecs(onnxruntime_double)
    onnxruntime_double._clear()


@pytest.fixture
def fake_walk_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", tmp_path)
    # Pretend to be a Pi so walk_start takes the real walk-script path (the
    # non-Pi path spawns the mock fake_broadcaster, which needs cloud creds).
    monkeypatch.setattr(tnkr_server.platform, "machine", lambda: "aarch64")
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
