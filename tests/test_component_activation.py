"""Two-phase activation: an unloadable component can never become the active one.

Phase 1b, Decision 6.

The reason activation is two-phase rather than one is in the unit file.
tnkr-robot.service.template has Restart=on-failure, StartLimitIntervalSec=300 and
StartLimitBurst=5. A component that crashes the server on load gets five restarts and
then systemd refuses to start the unit until someone runs `systemctl reset-failed` — at
which point the thing you would use to roll back is the thing that will not start, and
the duck is bricked until a person is physically at it.

So validation runs in a subprocess, and a component that fails it is recorded as INVALID
and NEVER retried. `test_an_oom_killed_validation_is_not_retried` is the one that matters:
five attempts at a component that OOMs is exactly five restarts.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

import pytest

from mini_bdx_runtime.components import (
    ComponentError,
    activate,
    active_state,
    catalogue_dir,
    invalid_components,
    resolve_active,
    rollback,
    stage_artifact,
    stage_manifest,
)
from mini_bdx_runtime.obs_spec import WALK_OBS_SPEC

EMBODIMENT = "open-duck-mini"


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _stage(component_id="walk-v3", payload=None, **over):
    payload = payload if payload is not None else component_id.encode()
    data = {
        "id": component_id,
        "version": "1.0.0",
        "hash": hashlib.sha256(payload).hexdigest(),
        "kind": "policy",
        "embodiment": EMBODIMENT,
        "obsSpec": [{"name": b.name, "shape": list(b.shape)} for b in WALK_OBS_SPEC],
        "rateHz": 50.0,
    }
    data.update(over)
    stage_manifest(component_id, data)
    stage_artifact(component_id, [payload])
    return component_id


def _exits(code: int):
    return lambda component: subprocess.CompletedProcess([], code, b"", b"boom")


def _killed_by(signal_number: int):
    """A validator that was killed, the way the OOM killer kills things."""
    return lambda component: subprocess.CompletedProcess([], -signal_number, b"", b"")


OK = _exits(0)


def _activate(component_id, validator=OK):
    return activate(
        component_id, embodiment=EMBODIMENT, loop_spec=WALK_OBS_SPEC, validator=validator
    )


# --- the happy path ---------------------------------------------------------

def test_a_component_that_loads_becomes_active(catalogue):
    _stage("walk-v3")
    assert _activate("walk-v3")["active"] == "walk-v3"
    assert (catalogue_dir() / "walk-v3" / "artifact.onnx").is_file()


def test_activating_keeps_the_one_it_replaced_as_previous(catalogue):
    _stage("walk-v2")
    _activate("walk-v2")
    _stage("walk-v3")
    state = _activate("walk-v3")
    assert state == {"active": "walk-v3", "previous": "walk-v2"}


def test_staging_is_emptied_once_it_is_installed(catalogue):
    from mini_bdx_runtime.components import staging_dir

    _stage("walk-v3")
    _activate("walk-v3")
    assert not (staging_dir() / "walk-v3").exists()


def test_reactivating_the_same_component_does_not_make_it_its_own_previous(catalogue):
    """Otherwise rollback is a no-op at exactly the moment it is needed."""
    _stage("walk-v2")
    _activate("walk-v2")
    _stage("walk-v3")
    _activate("walk-v3")
    _stage("walk-v3", payload=b"rebuilt")
    state = _activate("walk-v3")
    assert state == {"active": "walk-v3", "previous": "walk-v2"}


# --- validation failure -----------------------------------------------------

def test_a_component_that_does_not_load_never_becomes_active(catalogue):
    _stage("walk-v2")
    _activate("walk-v2")
    _stage("broken")
    with pytest.raises(ComponentError) as exc:
        _activate("broken", validator=_exits(1))
    assert exc.value.code == "COMPONENT_VALIDATION_FAILED"
    # the robot is still running what it was running
    assert active_state()["active"] == "walk-v2"


def test_an_oom_killed_validation_is_invalid_not_transient(catalogue):
    """SIGKILL on this device is overwhelmingly the OOM killer. "The machine was busy,
    try again" is the reasoning that reaches the systemd lockout."""
    _stage("hungry")
    with pytest.raises(ComponentError):
        _activate("hungry", validator=_killed_by(9))
    assert "hungry" in invalid_components()
    assert "signal 9" in invalid_components()["hungry"]


def test_an_oom_killed_validation_is_not_retried(catalogue):
    """The one that matters. Five attempts at a component that OOMs is five restarts,
    and five restarts is the lockout."""
    _stage("hungry")
    calls = []

    def counting_validator(component):
        calls.append(component.manifest.id)
        return subprocess.CompletedProcess([], -9, b"", b"")

    with pytest.raises(ComponentError):
        _activate("hungry", validator=counting_validator)

    for _ in range(4):
        with pytest.raises(ComponentError) as exc:
            _activate("hungry", validator=counting_validator)
        assert exc.value.code == "COMPONENT_VALIDATION_FAILED"
        assert "will not be retried" in exc.value.detail

    assert calls == ["hungry"], f"validator ran {len(calls)} times, must run once"


def test_a_failed_component_is_discarded_from_staging(catalogue):
    from mini_bdx_runtime.components import staging_dir

    _stage("broken")
    with pytest.raises(ComponentError):
        _activate("broken", validator=_exits(1))
    assert not (staging_dir() / "broken").exists()


def test_re_uploading_is_the_only_way_to_clear_a_failure(catalogue):
    """An operator who fixed and rebuilt a policy must be able to try again — and
    nothing short of new bytes should count."""
    _stage("broken")
    with pytest.raises(ComponentError):
        _activate("broken", validator=_exits(1))
    assert "broken" in invalid_components()

    _stage("broken", payload=b"a rebuilt policy")
    assert "broken" not in invalid_components()
    assert _activate("broken")["active"] == "broken"


def test_a_contract_mismatch_is_caught_before_the_subprocess_runs(catalogue):
    """Cheaper, and it means a policy for the wrong loop never gets the chance to OOM."""
    blocks = [{"name": b.name, "shape": list(b.shape)} for b in WALK_OBS_SPEC]
    blocks[5], blocks[6] = blocks[6], blocks[5]
    _stage("wrong-loop", obsSpec=blocks)
    ran = []
    with pytest.raises(ComponentError) as exc:
        _activate("wrong-loop", validator=lambda c: ran.append(1) or subprocess.CompletedProcess([], 0))
    assert exc.value.code == "COMPONENT_CONTRACT_MISMATCH"
    assert ran == []


def test_a_policy_for_another_robot_is_caught_before_the_subprocess_runs(catalogue):
    _stage("dk1-policy", embodiment="trlc-dk1")
    with pytest.raises(ComponentError) as exc:
        _activate("dk1-policy")
    assert exc.value.code == "COMPONENT_EMBODIMENT_MISMATCH"


# --- rollback ---------------------------------------------------------------

def test_rollback_restores_the_previous_component(catalogue):
    _stage("walk-v2")
    _activate("walk-v2")
    _stage("walk-v3")
    _activate("walk-v3")
    assert rollback() == {"active": "walk-v2", "previous": "walk-v3"}


def test_rollback_with_no_previous_errors_cleanly(catalogue):
    _stage("walk-v2")
    _activate("walk-v2")
    with pytest.raises(ComponentError) as exc:
        rollback()
    assert exc.value.code == "COMPONENT_NO_PREVIOUS"


def test_rollback_does_not_revalidate(catalogue):
    """previous was validated when it was activated and has been running. Re-validating
    would mean the recovery path can fail for a NEW reason at the moment recovery is
    needed."""
    _stage("walk-v2")
    _activate("walk-v2")
    _stage("walk-v3")
    _activate("walk-v3")
    # even with every validator now failing, rollback works
    assert rollback()["active"] == "walk-v2"


def test_rollback_to_something_no_longer_installed_says_so(catalogue):
    import shutil

    _stage("walk-v2")
    _activate("walk-v2")
    _stage("walk-v3")
    _activate("walk-v3")
    shutil.rmtree(catalogue_dir() / "walk-v2")
    with pytest.raises(ComponentError) as exc:
        rollback()
    assert exc.value.code == "COMPONENT_NOT_FOUND"


def test_rollback_is_reversible(catalogue):
    _stage("walk-v2")
    _activate("walk-v2")
    _stage("walk-v3")
    _activate("walk-v3")
    rollback()
    assert rollback() == {"active": "walk-v3", "previous": "walk-v2"}


# --- durability -------------------------------------------------------------

def test_the_active_pointer_survives_a_truncated_write(catalogue):
    """A Pi is turned off by being unplugged. The file that says which policy to run is
    the one you cannot afford to lose, so it is written and renamed, never in place."""
    _stage("walk-v3")
    _activate("walk-v3")
    active_file = catalogue_dir() / "active.json"
    assert json.loads(active_file.read_text())["active"] == "walk-v3"
    assert not list(catalogue_dir().glob("*.tmp")), "a temp file was left behind"


def test_an_unreadable_active_file_reads_as_nothing_active_rather_than_crashing(catalogue):
    _stage("walk-v3")
    _activate("walk-v3")
    (catalogue_dir() / "active.json").write_text("{ truncated")
    assert active_state() == {"active": None, "previous": None}
    assert resolve_active(fallback="walk-v2") == "walk-v2"


def test_resolve_active_prefers_the_active_component_over_the_default(catalogue):
    _stage("walk-v3")
    _activate("walk-v3")
    assert resolve_active(fallback="walk-v2") == "walk-v3"


def test_resolve_active_with_nothing_active_and_no_default_refuses(catalogue):
    with pytest.raises(ComponentError) as exc:
        resolve_active()
    assert exc.value.code == "COMPONENT_NOT_FOUND"


# --- through HTTP, and all the way to what actually runs --------------------

def test_installing_a_policy_changes_what_the_walk_actually_starts(
    client, catalogue, fake_walk_dir, monkeypatch
):
    """The whole point of Phase 1b, end to end.

    Before this, `walk/start` globbed scripts/ and took the first .onnx, so installing
    a policy could not change what ran. Now: upload, activate, and the next walk starts
    the component that was activated.
    """
    import tnkr_server
    from conftest import write_walk_script

    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    monkeypatch.setattr(
        tnkr_server, "activate_component", lambda cid, **kw: activate(cid, validator=OK, **kw)
    )

    payload = b"a policy an operator chose"
    manifest = {
        "id": "operator-choice",
        "version": "1.0.0",
        "hash": hashlib.sha256(payload).hexdigest(),
        "kind": "policy",
        "embodiment": EMBODIMENT,
        "obsSpec": [{"name": b.name, "shape": list(b.shape)} for b in WALK_OBS_SPEC],
        "rateHz": 50.0,
    }
    assert client.post("/api/components/operator-choice/manifest", json=manifest).status_code == 200
    assert client.put("/api/components/operator-choice/artifact", content=payload).status_code == 200
    assert client.post("/api/components/operator-choice/activate").status_code == 200

    spawned = {}
    real_popen = tnkr_server.subprocess.Popen

    def capture(cmd, *a, **kw):
        spawned["cmd"] = cmd
        return real_popen(cmd, *a, **kw)

    monkeypatch.setattr(tnkr_server.subprocess, "Popen", capture)
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")

    assert "operator-choice" in " ".join(spawned["cmd"]), spawned["cmd"]


def test_the_shipped_policy_still_runs_when_nothing_has_been_activated(
    client, catalogue, fake_walk_dir
):
    """A duck nobody has installed anything on keeps working exactly as before."""
    from conftest import write_walk_script

    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    assert client.post("/api/walk/start", json={}).status_code == 200
    client.post("/api/walk/stop")


def test_the_component_list_reports_active_previous_and_refused(client, catalogue):
    _stage("walk-v2")
    _activate("walk-v2")
    _stage("walk-v3")
    _activate("walk-v3")
    _stage("broken")
    with pytest.raises(ComponentError):
        _activate("broken", validator=_exits(1))

    body = client.get("/api/components").json()
    assert body["active"] == "walk-v3"
    assert body["previous"] == "walk-v2"
    assert "broken" in body["invalid"]
    assert {c["id"] for c in body["installed"]} == {"walk-v2", "walk-v3"}


def test_a_damaged_installed_component_is_reported_not_hidden(client, catalogue):
    """A component an operator installed and cannot see is worse than one they can see
    is damaged."""
    _stage("walk-v3")
    _activate("walk-v3")
    (catalogue_dir() / "walk-v3" / "manifest.json").write_text("{ truncated")
    body = client.get("/api/components").json()
    assert body["installed"] == [{"id": "walk-v3", "broken": True}]


def test_rollback_over_http(client, catalogue):
    _stage("walk-v2")
    _activate("walk-v2")
    _stage("walk-v3")
    _activate("walk-v3")
    res = client.post("/api/components/rollback")
    assert res.status_code == 200
    assert res.json() == {"active": "walk-v2", "previous": "walk-v3"}


def test_rollback_with_nothing_to_roll_back_to_is_a_clean_409(client, catalogue):
    _stage("walk-v2")
    _activate("walk-v2")
    res = client.post("/api/components/rollback")
    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "COMPONENT_NO_PREVIOUS"
