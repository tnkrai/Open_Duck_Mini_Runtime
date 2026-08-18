"""Installing a policy onto a running robot: staging, streaming, and hash verification.

Phase 1b, Decisions 6 and 14.

An upload is not an installation. Bytes arrive, get hashed, and sit in STAGING until
something has proved they load. That is not tidiness: the unit has Restart=on-failure
with StartLimitBurst=5, so a component that crashes the server on load burns five
restarts and then systemd refuses to start the unit at all — at which point the thing
you would use to roll back is the thing that will not start. Staging is the property
that an unloadable component can never become the persisted active choice.

The streaming half is about a different failure. This process is capped at
MemoryMax=384M on a 512MB Pi Zero 2 W, and that device is documented to fail SILENTLY
when it runs out, so an OOM kill mid-upload looks like the robot simply stopping. 884K
is fine today and a camera-conditioned policy is not.
"""

from __future__ import annotations

import hashlib
import json

import pytest

import tnkr_server
from mini_bdx_runtime.components import (
    ArtifactWriter,
    ComponentError,
    catalogue_dir,
    stage_artifact,
    stage_manifest,
    staged_component,
    staging_dir,
)
from mini_bdx_runtime.obs_spec import WALK_OBS_SPEC


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    """A catalogue rooted in a temp dir, so no test touches a real home."""
    monkeypatch.setattr(
        "mini_bdx_runtime.components.catalogue_dir",
        lambda root=None: tmp_path / ".tnkr" / "components",
    )
    return tmp_path


def _manifest(payload: bytes, component_id="walk-v3", **over) -> dict:
    data = {
        "id": component_id,
        "version": "3.0.0",
        "hash": hashlib.sha256(payload).hexdigest(),
        "kind": "policy",
        "embodiment": "open-duck-mini",
        "obsSpec": [{"name": b.name, "shape": list(b.shape)} for b in WALK_OBS_SPEC],
        "rateHz": 50.0,
        "provenance": "test",
    }
    data.update(over)
    return data


# --- staging ----------------------------------------------------------------

def test_a_manifest_is_validated_before_any_bytes_are_spent(catalogue):
    """Refused before an operator spends a Pi Zero's wifi on an 800K upload that was
    never going to be accepted."""
    with pytest.raises(ComponentError) as exc:
        stage_manifest("walk-v3", _manifest(b"x", hash="not-a-sha"))
    assert exc.value.code == "COMPONENT_MANIFEST_INVALID"


def test_a_manifest_whose_id_disagrees_with_the_url_is_refused(catalogue):
    with pytest.raises(ComponentError) as exc:
        stage_manifest("walk-v3", _manifest(b"x", component_id="something-else"))
    assert exc.value.code == "COMPONENT_MANIFEST_INVALID"


def test_bytes_without_a_manifest_are_refused(catalogue):
    with pytest.raises(ComponentError) as exc:
        stage_artifact("walk-v3", [b"bytes with nothing declaring them"])
    assert exc.value.code == "COMPONENT_NOT_FOUND"


def test_a_new_manifest_discards_a_half_uploaded_artifact(catalogue):
    """The one combination that could pass while being wrong: a hash from manifest A
    checked against bytes uploaded for manifest B."""
    payload = b"first policy"
    stage_manifest("walk-v3", _manifest(payload))
    stage_artifact("walk-v3", [payload])
    assert (staging_dir() / "walk-v3" / "artifact.onnx").is_file()

    stage_manifest("walk-v3", _manifest(b"a different policy entirely"))
    assert not (staging_dir() / "walk-v3" / "artifact.onnx").exists()


def test_staging_never_touches_what_is_installed(catalogue):
    installed = catalogue_dir() / "walk-v2"
    installed.mkdir(parents=True)
    (installed / "artifact.onnx").write_bytes(b"the policy currently in use")

    stage_manifest("walk-v2", _manifest(b"a candidate", component_id="walk-v2"))
    stage_artifact("walk-v2", [b"a candidate"])
    assert (installed / "artifact.onnx").read_bytes() == b"the policy currently in use"


# --- the hash ---------------------------------------------------------------

def test_bytes_that_fail_their_hash_are_deleted_not_left_behind(catalogue):
    """A failed upload is not a partial success to resume from. Leaving it invites a
    later call to find a file and assume it was verified."""
    stage_manifest("walk-v3", _manifest(b"the real policy"))
    with pytest.raises(ComponentError) as exc:
        stage_artifact("walk-v3", [b"the real policy", b" plus junk"])
    assert exc.value.code == "COMPONENT_HASH_MISMATCH"
    assert not (staging_dir() / "walk-v3" / "artifact.onnx").exists()


def test_an_empty_upload_is_refused(catalogue):
    stage_manifest("walk-v3", _manifest(b"content"))
    with pytest.raises(ComponentError) as exc:
        stage_artifact("walk-v3", [])
    assert exc.value.code == "COMPONENT_HASH_MISMATCH"


def test_a_correct_upload_becomes_a_staged_component(catalogue):
    payload = b"a policy that hashes correctly"
    stage_manifest("walk-v3", _manifest(payload))
    assert stage_artifact("walk-v3", [payload[:5], payload[5:]]) == len(payload)
    component = staged_component("walk-v3")
    assert component.artifact_path.read_bytes() == payload


# --- streaming, which is the point ------------------------------------------

def test_the_writer_never_accumulates_the_artifact(catalogue):
    """The property Decision 14 actually asks for, asserted rather than asserted-about.

    The first version of the endpoint collected request.stream() into a list and handed
    it to a synchronous writer. That reads like streaming and is not — the list held
    the whole artifact, so the memory shape was identical to taking a bytes body. This
    checks the file grows as chunks arrive, which is only true if each one is written
    and dropped rather than gathered.
    """
    payload = b"chunk-" * 10_000
    stage_manifest("walk-v3", _manifest(payload))
    writer = ArtifactWriter("walk-v3")

    sizes = []
    for i in range(0, len(payload), 6000):
        writer.write(payload[i : i + 6000])
        sizes.append(writer.path.stat().st_size)

    assert sizes == sorted(sizes) and sizes[0] < sizes[-1], (
        "the artifact did not grow on disk as chunks arrived, so it was being held "
        f"somewhere else: {sizes[:3]}..."
    )
    assert writer.finish() == len(payload)


def test_an_interrupted_upload_leaves_nothing_to_mistake_for_a_verified_artifact(catalogue):
    payload = b"a long policy" * 1000
    stage_manifest("walk-v3", _manifest(payload))
    writer = ArtifactWriter("walk-v3")
    writer.write(payload[:100])
    writer.abort()  # the connection dropped
    assert not writer.path.exists()


# --- through HTTP -----------------------------------------------------------

@pytest.fixture
def http(client, catalogue):
    return client


def test_install_over_http_stages_but_does_not_install(http, catalogue):
    payload = b"a policy delivered over the wire"
    res = http.post("/api/components/walk-v3/manifest", json=_manifest(payload))
    assert res.status_code == 200, res.text

    res = http.put("/api/components/walk-v3/artifact", content=payload)
    assert res.status_code == 200, res.text
    assert res.json()["bytes"] == len(payload)

    assert http.get("/api/components/walk-v3/staged").status_code == 200
    # staged is not installed: nothing points at it yet
    assert not (catalogue_dir() / "walk-v3").exists()


def test_a_truncated_upload_is_refused_over_http(http, catalogue):
    payload = b"the whole policy"
    http.post("/api/components/walk-v3/manifest", json=_manifest(payload))
    res = http.put("/api/components/walk-v3/artifact", content=payload[:5])
    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "COMPONENT_HASH_MISMATCH"


def test_uploading_before_declaring_is_a_404_over_http(http, catalogue):
    res = http.put("/api/components/walk-v3/artifact", content=b"bytes")
    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "COMPONENT_NOT_FOUND"


def test_a_staged_upload_can_be_thrown_away(http, catalogue):
    payload = b"a candidate that turned out to be wrong"
    http.post("/api/components/walk-v3/manifest", json=_manifest(payload))
    http.put("/api/components/walk-v3/artifact", content=payload)
    assert http.delete("/api/components/walk-v3/staged").status_code == 204
    assert http.get("/api/components/walk-v3/staged").status_code == 404


def test_discarding_something_never_staged_is_not_an_error(http, catalogue):
    """Idempotent on purpose: a client retrying a cleanup should not have to care."""
    assert http.delete("/api/components/walk-v9/staged").status_code == 204
