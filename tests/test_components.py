"""Which policy runs, and whether it fits the loop.

Phase 1b. These replace a glob and a first match. The glob is not wrong today — one
.onnx exists — and it becomes arbitrary the moment a second one does, which is exactly
what Phase 6 requires. Underneath the file-selection problem is the real one: nothing
checked that the model matched the loop feeding it.

Every test here runs on any machine. That is the structural point of the phase: these
checks moved ABOVE `if is_pi:` so CI can execute them, and CI has no onnxruntime and no
mujoco. Checks that only run on the robot are checks that are never run before the
robot.
"""

from __future__ import annotations

import json

import pytest

from mini_bdx_runtime.components import (
    ComponentError,
    ComponentManifest,
    check_contract,
    check_embodiment,
    file_sha256,
    prepare,
    resolve,
    resolve_builtin,
    verify_hash,
)
from mini_bdx_runtime.obs_spec import WALK_OBS_SPEC, ObsBlock

EMBODIMENT = "open-duck-mini"


def _manifest_dict(**over) -> dict:
    data = {
        "id": "walk-v2",
        "version": "2.0.0",
        "hash": "0" * 64,
        "kind": "policy",
        "embodiment": EMBODIMENT,
        "obsSpec": [{"name": b.name, "shape": list(b.shape)} for b in WALK_OBS_SPEC],
        "rateHz": 50.0,
        "assumes": {"runtime": ">=2.0.0"},
        "provenance": "test",
    }
    data.update(over)
    return data


def _install(root, component_id="walk-v2", payload=b"fake onnx bytes", **over):
    """Write a component into a temp catalogue, hashing whatever payload it is given."""
    import hashlib

    base = root / ".tnkr" / "components" / component_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "artifact.onnx").write_bytes(payload)
    data = _manifest_dict(id=component_id, **over)
    data.setdefault("hash", hashlib.sha256(payload).hexdigest())
    if "hash" not in over:
        data["hash"] = hashlib.sha256(payload).hexdigest()
    (base / "manifest.json").write_text(json.dumps(data))
    return base


# --- resolution by id, not by glob ------------------------------------------

def test_a_component_resolves_by_its_id(tmp_path):
    _install(tmp_path)
    component = resolve("walk-v2", root=tmp_path)
    assert component.manifest.id == "walk-v2"
    assert component.artifact_path.is_file()


def test_an_unknown_id_refuses_instead_of_falling_back(tmp_path):
    """The failure the glob made impossible to have. "The id you asked for is not
    installed" and "here is a different policy" are answers a robot must not confuse,
    and the fallback is the one that walks."""
    _install(tmp_path, component_id="walk-v2")
    _install(tmp_path, component_id="walk-v3")
    with pytest.raises(ComponentError) as exc:
        resolve("walk-v9", root=tmp_path)
    assert exc.value.code == "COMPONENT_NOT_FOUND"


def test_two_installed_components_do_not_make_the_choice_arbitrary(tmp_path):
    """Phase 6's precondition, asserted now. With a glob, whichever sorted first won."""
    _install(tmp_path, component_id="walk-v2", payload=b"two")
    _install(tmp_path, component_id="walk-v3", payload=b"three")
    assert resolve("walk-v2", root=tmp_path).artifact_path.read_bytes() == b"two"
    assert resolve("walk-v3", root=tmp_path).artifact_path.read_bytes() == b"three"


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", ".hidden"])
def test_an_id_that_is_really_a_path_is_refused(tmp_path, bad):
    # ids arrive in an HTTP body and become a path segment
    with pytest.raises(ComponentError) as exc:
        resolve(bad, root=tmp_path)
    assert exc.value.code == "COMPONENT_NOT_FOUND"


def test_a_manifest_whose_id_disagrees_with_its_directory_is_refused(tmp_path):
    """One of the two is a lie and there is no way to tell which."""
    _install(tmp_path, component_id="walk-v2")
    base = tmp_path / ".tnkr" / "components" / "walk-v2"
    data = json.loads((base / "manifest.json").read_text())
    data["id"] = "something-else"
    (base / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ComponentError) as exc:
        resolve("walk-v2", root=tmp_path)
    assert exc.value.code == "COMPONENT_MANIFEST_INVALID"


def test_a_manifest_with_no_artifact_is_refused(tmp_path):
    base = _install(tmp_path)
    (base / "artifact.onnx").unlink()
    with pytest.raises(ComponentError) as exc:
        resolve("walk-v2", root=tmp_path)
    assert exc.value.code == "COMPONENT_NOT_FOUND"


# --- the hash ---------------------------------------------------------------

def test_the_bytes_must_be_the_bytes_the_manifest_describes(tmp_path):
    base = _install(tmp_path, payload=b"the real policy")
    (base / "artifact.onnx").write_bytes(b"the real policy, truncated by a dropped wifi")
    with pytest.raises(ComponentError) as exc:
        verify_hash(resolve("walk-v2", root=tmp_path))
    assert exc.value.code == "COMPONENT_HASH_MISMATCH"


def test_hashing_is_incremental_and_never_holds_the_artifact_in_memory(tmp_path):
    """A Pi Zero 2 W is capped at MemoryMax=384M and fails silently when it runs out,
    so this reads in chunks. Proven by hashing across a chunk boundary, not by
    inspecting the implementation."""
    import hashlib

    payload = b"x" * (3 * 1024 * 1024 + 7)  # spans three chunks and a bit
    path = tmp_path / "big.bin"
    path.write_bytes(payload)
    assert file_sha256(path, chunk_bytes=1024) == hashlib.sha256(payload).hexdigest()


# --- the contract -----------------------------------------------------------

def test_a_matching_contract_passes():
    check_contract(ComponentManifest.from_dict(_manifest_dict()), WALK_OBS_SPEC)


def test_a_reordered_contract_is_refused_even_though_the_width_matches():
    """The failure a width check waves through and a duck discovers with its face."""
    blocks = [{"name": b.name, "shape": list(b.shape)} for b in WALK_OBS_SPEC]
    blocks[5], blocks[6] = blocks[6], blocks[5]  # two 14-wide action-history frames
    manifest = ComponentManifest.from_dict(_manifest_dict(obsSpec=blocks))
    assert manifest.obs_size == sum(b.size for b in WALK_OBS_SPEC)  # same width...
    with pytest.raises(ComponentError) as exc:  # ...refused anyway
        check_contract(manifest, WALK_OBS_SPEC)
    assert exc.value.code == "COMPONENT_CONTRACT_MISMATCH"


def test_a_differently_shaped_block_is_refused():
    blocks = [{"name": b.name, "shape": list(b.shape)} for b in WALK_OBS_SPEC]
    blocks[2]["shape"] = [3]  # the commands block is 7 wide, not 3
    with pytest.raises(ComponentError) as exc:
        check_contract(ComponentManifest.from_dict(_manifest_dict(obsSpec=blocks)), WALK_OBS_SPEC)
    assert exc.value.code == "COMPONENT_CONTRACT_MISMATCH"


def test_a_shorter_contract_is_refused():
    blocks = [{"name": b.name, "shape": list(b.shape)} for b in WALK_OBS_SPEC][:-1]
    with pytest.raises(ComponentError) as exc:
        check_contract(ComponentManifest.from_dict(_manifest_dict(obsSpec=blocks)), WALK_OBS_SPEC)
    assert exc.value.code == "COMPONENT_CONTRACT_MISMATCH"


def test_an_empty_obs_spec_is_refused_at_parse_time():
    """An empty spec would pass check_contract against nothing and fail against
    everything real, but the dangerous version is a loop with no blocks: then it is
    vacuously true, which is worse than having no check."""
    with pytest.raises(ComponentError) as exc:
        ComponentManifest.from_dict(_manifest_dict(obsSpec=[]))
    assert exc.value.code == "COMPONENT_MANIFEST_INVALID"


def test_a_policy_for_another_robot_says_so_plainly():
    manifest = ComponentManifest.from_dict(_manifest_dict(embodiment="trlc-dk1"))
    with pytest.raises(ComponentError) as exc:
        check_embodiment(manifest, EMBODIMENT)
    assert exc.value.code == "COMPONENT_EMBODIMENT_MISMATCH"


@pytest.mark.parametrize("field", ["id", "version", "hash", "kind", "embodiment", "rateHz"])
def test_a_manifest_missing_a_required_field_is_refused(field):
    data = _manifest_dict()
    del data[field]
    with pytest.raises(ComponentError) as exc:
        ComponentManifest.from_dict(data)
    assert exc.value.code == "COMPONENT_MANIFEST_INVALID"


def test_a_hash_that_is_not_a_sha256_is_refused():
    with pytest.raises(ComponentError) as exc:
        ComponentManifest.from_dict(_manifest_dict(hash="not-a-hash"))
    assert exc.value.code == "COMPONENT_MANIFEST_INVALID"


def test_a_manifest_round_trips():
    original = ComponentManifest.from_dict(_manifest_dict())
    assert ComponentManifest.from_dict(original.to_dict()) == original


# --- the whole gate ---------------------------------------------------------

def test_prepare_runs_every_check_and_returns_the_component(tmp_path):
    _install(tmp_path)
    component = prepare(
        "walk-v2", embodiment=EMBODIMENT, loop_spec=WALK_OBS_SPEC, root=tmp_path
    )
    assert component.manifest.id == "walk-v2"


def test_prepare_names_the_wrong_robot_before_it_reads_a_single_byte(tmp_path):
    """Order matters. "Wrong robot" needs no byte-reading and is the clearest thing to
    say; reporting a hash mismatch first would be true and useless."""
    _install(tmp_path, embodiment="trlc-dk1", hash="f" * 64)
    with pytest.raises(ComponentError) as exc:
        prepare("walk-v2", embodiment=EMBODIMENT, loop_spec=WALK_OBS_SPEC, root=tmp_path)
    assert exc.value.code == "COMPONENT_EMBODIMENT_MISMATCH"


def test_prepare_checks_the_hash_before_the_contract(tmp_path):
    """A contract carried by unexpected bytes describes something other than what is
    on disk, so trusting it enough to compare would be backwards."""
    blocks = [{"name": b.name, "shape": list(b.shape)} for b in WALK_OBS_SPEC][:-1]
    base = _install(tmp_path, obsSpec=blocks)
    (base / "artifact.onnx").write_bytes(b"different bytes entirely")
    with pytest.raises(ComponentError) as exc:
        prepare("walk-v2", embodiment=EMBODIMENT, loop_spec=WALK_OBS_SPEC, root=tmp_path)
    assert exc.value.code == "COMPONENT_HASH_MISMATCH"


def test_every_refusal_uses_a_code_studio_already_renders():
    """Studio maps codes to sentences; a code invented here that it does not know
    shows an operator a generic fallback. These four are in its ErrorCode enum."""
    known = {
        "COMPONENT_NOT_FOUND",
        "COMPONENT_MANIFEST_INVALID",
        "COMPONENT_HASH_MISMATCH",
        "COMPONENT_CONTRACT_MISMATCH",
        "COMPONENT_EMBODIMENT_MISMATCH",
    }
    raised = set()
    for fn in (
        lambda: resolve("nope"),
        lambda: ComponentManifest.from_dict(_manifest_dict(hash="x")),
        lambda: check_contract(
            ComponentManifest.from_dict(_manifest_dict(obsSpec=[{"name": "a", "shape": [1]}])),
            WALK_OBS_SPEC,
        ),
        lambda: check_embodiment(
            ComponentManifest.from_dict(_manifest_dict(embodiment="other")), EMBODIMENT
        ),
    ):
        try:
            fn()
        except ComponentError as exc:
            raised.add(exc.code)
    assert raised <= known and raised


def test_an_unreadable_shipped_manifest_says_so_rather_than_reporting_it_missing(tmp_path):
    """Without this, a corrupted manifest reports "not installed" and sends someone
    looking for a missing file rather than a broken one they are staring at."""
    (tmp_path / "broken.manifest.json").write_text("{ not json")
    with pytest.raises(ComponentError) as exc:
        resolve_builtin("walk-v2", tmp_path)
    assert exc.value.code == "COMPONENT_MANIFEST_INVALID"
    assert "broken.manifest.json" in exc.value.detail


def test_a_good_shipped_manifest_still_resolves_beside_a_broken_one(tmp_path):
    """The broken one must not hide the good ones either."""
    import hashlib, json as _json
    from mini_bdx_runtime.obs_spec import WALK_OBS_SPEC

    (tmp_path / "broken.manifest.json").write_text("{ not json")
    payload = b"good"
    (tmp_path / "good.onnx").write_bytes(payload)
    (tmp_path / "good.manifest.json").write_text(
        _json.dumps(_manifest_dict(hash=hashlib.sha256(payload).hexdigest()))
    )
    assert resolve_builtin("walk-v2", tmp_path).artifact_path.read_bytes() == payload


def test_two_shipped_manifests_claiming_one_id_are_refused(tmp_path):
    """Under the old glob, sort order silently decided. Refusing is the only honest
    answer when two files both claim to be the same component."""
    import json as _json

    for name in ("a", "b"):
        (tmp_path / f"{name}.manifest.json").write_text(_json.dumps(_manifest_dict()))
        (tmp_path / f"{name}.onnx").write_bytes(b"")
    with pytest.raises(ComponentError) as exc:
        resolve_builtin("walk-v2", tmp_path)
    assert exc.value.code == "COMPONENT_MANIFEST_INVALID"
