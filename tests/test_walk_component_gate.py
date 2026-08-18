"""The component checks run before the platform gate, on both platforms.

Phase 1b's structural change. Before it, `SCRIPTS_DIR.glob("*.onnx")[0]` and the 404
lived inside `if is_pi:`, and the `else` branch spawned fake_broadcaster without
reading a model at all. So on a Mac and in CI, none of this could execute — the checks
existed only where nobody could run them.

The sharpest test here is `test_a_bad_component_is_refused_on_a_mac_before_the_mock_
branch_complains`: off-Pi, walk/start demands cloud credentials and 400s without them.
If a component refusal arrives instead of that 400, the check provably ran ABOVE the
gate. If the hoist were undone, that test goes green-to-400 and says so.
"""

from __future__ import annotations

import json

import pytest

import tnkr_server
from conftest import install_component, write_walk_script

MOCK_CREDS = {
    "sessionToken": "t",
    "supabaseUrl": "https://example.supabase.co",
    "supabaseKey": "k",
}


@pytest.fixture
def not_a_pi(tmp_path, monkeypatch):
    """A Mac. The mock branch, where none of this used to be reachable."""
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(tnkr_server.platform, "machine", lambda: "x86_64")
    return tmp_path


def _break_manifest(d, **over):
    data = json.loads((d / "model.manifest.json").read_text())
    data.update(over)
    (d / "model.manifest.json").write_text(json.dumps(data))


# --- on a Pi ----------------------------------------------------------------

def test_a_published_component_starts_the_walk(client, fake_walk_dir):
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")


def test_an_unknown_component_id_is_a_404(client, fake_walk_dir):
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    res = client.post("/api/walk/start", json={"componentId": "walk-v9"})
    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "COMPONENT_NOT_FOUND"


def test_a_hash_mismatch_refuses_rather_than_running_unknown_bytes(client, fake_walk_dir):
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    # the realistic failure: a Pi Zero loses wifi and the file is present but truncated
    (fake_walk_dir / "model.onnx").write_bytes(b"half an upload")
    res = client.post("/api/walk/start", json={})
    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "COMPONENT_HASH_MISMATCH"


def test_a_contract_mismatch_refuses_even_at_the_right_width(client, fake_walk_dir):
    """A policy trained against a different observation order loads happily and walks
    the duck into the floor. Same 101 numbers, different meaning."""
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    data = json.loads((fake_walk_dir / "model.manifest.json").read_text())
    blocks = data["obsSpec"]
    blocks[5], blocks[6] = blocks[6], blocks[5]
    _break_manifest(fake_walk_dir, obsSpec=blocks)
    res = client.post("/api/walk/start", json={})
    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "COMPONENT_CONTRACT_MISMATCH"


def test_a_policy_for_another_robot_refuses(client, fake_walk_dir):
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    _break_manifest(fake_walk_dir, embodiment="trlc-dk1")
    res = client.post("/api/walk/start", json={})
    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "COMPONENT_EMBODIMENT_MISMATCH"


def test_two_components_are_addressable_rather_than_arbitrary(client, fake_walk_dir):
    """Phase 6's precondition. Under the glob, adding a second .onnx silently changed
    which policy ran, decided by sort order."""
    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    other = fake_walk_dir / "aaa-sorts-first.manifest.json"
    data = json.loads((fake_walk_dir / "model.manifest.json").read_text())
    data["id"] = "walk-v3"
    other.write_text(json.dumps(data))
    (fake_walk_dir / "aaa-sorts-first.onnx").write_bytes(b"")

    # the default still resolves to the one that was asked for, not the one that sorts first
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")
    assert client.post("/api/walk/start", json={"componentId": "walk-v3"}).status_code == 200
    client.post("/api/walk/stop")


# --- on a Mac, where none of this used to be reachable ----------------------

def test_a_bad_component_is_refused_on_a_mac_before_the_mock_branch_complains(
    client, not_a_pi
):
    """The hoist, asserted directly.

    Off-Pi, walk/start refuses without cloud credentials — a 400. Send valid creds and
    a component that does not exist: a 404 about the component proves the check ran
    ABOVE the platform gate. Undo the hoist and this test starts spawning a mock walk
    with no model at all.
    """
    res = client.post(
        "/api/walk/start", json={**MOCK_CREDS, "componentId": "walk-v9"}
    )
    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "COMPONENT_NOT_FOUND"


def test_the_contract_check_runs_on_a_mac_too(client, not_a_pi):
    install_component(not_a_pi)
    data = json.loads((not_a_pi / "model.manifest.json").read_text())
    blocks = data["obsSpec"]
    blocks[0], blocks[1] = blocks[1], blocks[0]  # gyro and accelerometer, both 3 wide
    _break_manifest(not_a_pi, obsSpec=blocks)
    res = client.post("/api/walk/start", json={**MOCK_CREDS})
    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "COMPONENT_CONTRACT_MISMATCH"


def test_the_mock_branch_still_spawns_fake_broadcaster(client, not_a_pi):
    """The else branch's job is unchanged. Only the checks moved."""
    install_component(not_a_pi)
    (not_a_pi / "fake_broadcaster.py").write_text("import time; time.sleep(30)\n")
    res = client.post("/api/walk/start", json={**MOCK_CREDS})
    assert res.status_code == 200, res.text
    client.post("/api/walk/stop")


def test_the_mock_branch_still_demands_its_credentials(client, not_a_pi):
    """A good component does not excuse missing creds — that 400 is still the mock
    branch's own rule, it just no longer hides the component checks behind itself."""
    install_component(not_a_pi)
    res = client.post("/api/walk/start", json={})
    assert res.status_code == 400


# --- the shipped policy is itself a catalogue entry -------------------------

def test_the_policy_this_repo_ships_is_published_with_a_true_hash():
    """The plan asks for BEST_WALK_ONNX_2.onnx to become a catalogue entry rather than
    a file the code finds by globbing. If the artifact is ever replaced without
    regenerating the manifest, this fails instead of the duck falling over."""
    from pathlib import Path

    from mini_bdx_runtime.components import ComponentManifest, check_contract, file_sha256
    from mini_bdx_runtime.obs_spec import WALK_OBS_SPEC

    scripts = Path(tnkr_server.__file__).parent
    manifest_path = scripts / "BEST_WALK_ONNX_2.manifest.json"
    assert manifest_path.is_file(), "the shipped policy has no manifest"

    manifest = ComponentManifest.from_dict(json.loads(manifest_path.read_text()))
    assert manifest.id == tnkr_server.DEFAULT_COMPONENT_ID
    assert manifest.embodiment == tnkr_server.EMBODIMENT
    assert manifest.hash == file_sha256(scripts / "BEST_WALK_ONNX_2.onnx")
    check_contract(manifest, WALK_OBS_SPEC)
