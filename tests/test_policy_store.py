"""The policy store: installing, evicting, selecting, and getting back to the built-in.

What this file is defending
---------------------------
Three failures, in the order they would hurt:

1. **A full SD card** (amendment A3). Nothing in the original plan ever deleted a policy.
   A card that fills is a duck that will not boot, which is worse than a duck that walks
   badly and has nothing to do with walking. Hence a cap, LRU eviction, a free-space floor
   checked *before* the download, and a ceiling enforced *while* streaming.
2. **No way back** (amendment A4). ``select("builtin")`` is the E-stop of policy selection:
   it has to work with an empty store, a corrupt pointer, and a walk in progress.
3. **A half-finished install** eating the policy that was working. Install is
   transactional, and eviction happens only after the bytes are on disk and verified.

And one thing that is not a failure of this story but would be a failure of the whole
feature: a policy installed here and spawned through ``/api/walk/start`` must come out with
the safety envelope ARMED without anybody opting in. The previous version of that seam was
fail-open -- arming was read off a ``--custom_policy`` flag that no production caller passed
-- and no test caught it. ``test_a_policy_installed_from_the_store_comes_out_armed`` is that
test.

No hardware and no network: the fetch is a fake from ``conftest.fake_fetch`` and every graph
comes from the onnxruntime double.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

import tnkr_server
from conftest import fake_fetch, wait_for_walk_ended, write_walk_script
from mini_bdx_runtime import policy_store
from mini_bdx_runtime.envelope import is_armed, is_builtin_policy
from mini_bdx_runtime.policy_contract import (
    ACT_DIM,
    OBS_DIM,
    POLICY_CONTRACT_MISMATCH,
    POLICY_INSTALL_FAILED,
    POLICY_STORE_FULL,
)
from mini_bdx_runtime.policy_store import BUILTIN_ID, PolicyStore, StoreError

URL = "https://tnkr-artifacts.s3.amazonaws.com/policies/9f2a/model.onnx?X-Amz-Signature=deadbeef"


# ── harness ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def scripts_dir(tmp_path):
    """A scripts/ directory holding the bundled policy, as a real duck's does."""
    d = tmp_path / "scripts"
    d.mkdir()
    (d / "BEST_WALK_ONNX_2.onnx").write_bytes(b"the policy this repo ships")
    return d


@pytest.fixture
def store_root(tmp_path):
    return tmp_path / "policies"


@pytest.fixture
def store(store_root, scripts_dir):
    return PolicyStore(root=store_root, scripts_dir=scripts_dir)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def install(store, onnx_specs, policy_id="9f2a", *, payload=None, **kwargs):
    """Install one policy through the store, with the network faked out.

    The payload doubles as the identity of the content, so two different ids get two
    different digests and the idempotency path is not hit by accident.
    """
    body = payload if payload is not None else f"model-{policy_id}".encode()
    store.fetch = fake_fetch(onnx_specs, payload=body, **kwargs)
    return store.install(policy_id, URL, digest(body))


def register_stored(onnx_specs, store, policy_id, *, obs_dim=OBS_DIM, act_dim=ACT_DIM):
    """Declare the graph the STORED file presents.

    Needed because select() re-reads the file from its final path, and the double keys its
    registry by path -- which is the point: re-verifying reads the file that is there now,
    not the one that was checked at install time.
    """
    onnx_specs.valid(
        store.root / policy_id / policy_store.MODEL_FILENAME,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )


def age(store, policy_id, seconds):
    """Backdate a policy's last-used stamp, so LRU order is deterministic."""
    marker = store.root / policy_id / policy_store.USED_FILENAME
    when = time.time() - seconds
    os.utime(marker, (when, when))


def temp_leftovers(root: Path) -> list[str]:
    return [p.name for p in root.iterdir() if p.name.startswith(".")]


# ── install: the happy path ─────────────────────────────────────────────────────


def test_install_stores_the_model_and_a_manifest(store, onnx_specs):
    result = install(store, onnx_specs, "9f2a")

    assert result.ok, result.detail
    assert (store.root / "9f2a" / "model.onnx").read_bytes() == b"model-9f2a"
    manifest = json.loads((store.root / "9f2a" / "manifest.json").read_text())
    assert manifest["obs_dim"] == OBS_DIM
    assert manifest["act_dim"] == ACT_DIM
    assert manifest["id"] == "9f2a"
    assert manifest["sha256"] == digest(b"model-9f2a")
    assert manifest["installed_at"] > 0
    assert result.evicted is None


def test_install_does_not_change_which_policy_is_active(store, onnx_specs):
    """Installing is not selecting. A duck mid-walk on the built-in must not switch
    policies because its owner pressed Install in another tab."""
    assert store.active_id() == BUILTIN_ID
    install(store, onnx_specs, "9f2a")
    assert store.active_id() == BUILTIN_ID


def test_the_declared_size_is_not_allowed_to_overrule_the_measured_one(store, onnx_specs):
    """A sidecar manifest is a claim. The graph and the file are the evidence."""
    store.fetch = fake_fetch(onnx_specs, payload=b"twelve bytes")
    result = store.install(
        "9f2a",
        URL,
        digest(b"twelve bytes"),
        {"obs_dim": 47, "size_bytes": 999, "author": "someone on discord"},
    )
    assert result.ok
    assert result.manifest["obs_dim"] == OBS_DIM
    assert result.manifest["size_bytes"] == len(b"twelve bytes")
    # ...but a field that is genuinely the uploader's to state survives.
    assert result.manifest["author"] == "someone on discord"
    assert result.manifest["inferred"] is False


def test_reinstalling_the_same_content_does_not_download_it_again(store, onnx_specs):
    """Idempotent by content hash, so a retry after a flaky download is free."""
    install(store, onnx_specs, "9f2a")

    second = fake_fetch(onnx_specs, payload=b"model-9f2a")
    store.fetch = second
    result = store.install("9f2a", URL, digest(b"model-9f2a"))

    assert result.ok and result.already_installed
    assert second.calls == []


# ── install: every way it fails must change nothing ─────────────────────────────


def test_a_hash_mismatch_stores_nothing(store, onnx_specs):
    store.fetch = fake_fetch(onnx_specs, payload=b"not what was promised")
    result = store.install("9f2a", URL, digest(b"what was promised"))

    assert not result.ok
    assert result.code == POLICY_INSTALL_FAILED
    assert not (store.root / "9f2a").exists()
    assert temp_leftovers(store.root) == []


def test_a_contract_failure_stores_nothing(store, onnx_specs):
    result = install(store, onnx_specs, "9f2a", obs_dim=47)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert str(OBS_DIM) in result.detail  # names what it expected, for the log
    assert not (store.root / "9f2a").exists()
    assert temp_leftovers(store.root) == []


def test_an_unparseable_file_stores_nothing(store, onnx_specs):
    result = install(store, onnx_specs, "9f2a", invalid=True)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert not (store.root / "9f2a").exists()


def test_a_download_that_dies_mid_stream_leaves_the_previous_policy_intact(
    store, onnx_specs
):
    """Failure mode F5: the presigned URL expires partway through, on slow wifi.

    The whole point of the temp-then-move order. A duck that was walking on 9f2a must
    still be able to walk on 9f2a.
    """
    install(store, onnx_specs, "9f2a")
    register_stored(onnx_specs, store, "9f2a")
    store.select("9f2a")
    before = (store.root / "9f2a" / "model.onnx").read_bytes()

    result = install(
        store, onnx_specs, "1c04", partial_bytes=4, fail="connection reset after 4 bytes"
    )

    assert not result.ok
    assert result.code == POLICY_INSTALL_FAILED
    assert not (store.root / "1c04").exists()
    assert (store.root / "9f2a" / "model.onnx").read_bytes() == before
    assert store.active_id() == "9f2a"
    assert temp_leftovers(store.root) == [], "a half-downloaded file was left on the card"


def test_a_failed_reinstall_does_not_destroy_the_working_version(store, onnx_specs):
    """Same id, worse file. The install is rejected and the good one is still there."""
    install(store, onnx_specs, "9f2a")
    good = (store.root / "9f2a" / "model.onnx").read_bytes()

    result = install(store, onnx_specs, "9f2a", payload=b"a different, wrong model",
                     obs_dim=47)

    assert not result.ok
    assert (store.root / "9f2a" / "model.onnx").read_bytes() == good
    assert json.loads((store.root / "9f2a" / "manifest.json").read_text())["obs_dim"] == OBS_DIM


# ── install: the bounded store (A3) ────────────────────────────────────────────


def test_installing_one_too_many_evicts_the_least_recently_used(store, onnx_specs):
    for policy_id, used_seconds_ago in (("aaa", 300), ("bbb", 200), ("ccc", 100)):
        install(store, onnx_specs, policy_id)
        age(store, policy_id, used_seconds_ago)

    result = install(store, onnx_specs, "ddd")

    assert result.ok
    assert result.evicted is not None
    assert result.evicted["id"] == "aaa", "evicted something other than the LRU"
    assert result.evicted["reason"] == "least recently used"
    assert not (store.root / "aaa").exists()
    assert {p["id"] for p in store.list()["policies"]} == {
        BUILTIN_ID, "bbb", "ccc", "ddd",
    }


def test_the_eviction_is_named_in_the_response(store, onnx_specs):
    """"Eviction is not a surprise" -- an operator who finds out later that a policy was
    deleted has been surprised about deleted data, which is the worst kind."""
    for policy_id in ("aaa", "bbb", "ccc"):
        install(store, onnx_specs, policy_id)
        age(store, policy_id, 100)
    age(store, "aaa", 999)

    body = install(store, onnx_specs, "ddd").as_dict()

    assert body["evicted"]["id"] == "aaa"
    assert body["evicted"]["sizeBytes"] == len(b"model-aaa")


def test_the_active_policy_is_never_evicted(store, onnx_specs):
    """Even when it is the least recently used: reverting to it must stay possible, and
    deleting the file the pointer names would strand the operator on the built-in without
    saying so."""
    for policy_id in ("aaa", "bbb", "ccc"):
        install(store, onnx_specs, policy_id)
    register_stored(onnx_specs, store, "aaa")
    store.select("aaa")
    age(store, "aaa", 9999)  # by LRU, the obvious victim
    age(store, "bbb", 500)
    age(store, "ccc", 100)

    result = install(store, onnx_specs, "ddd")

    assert result.ok
    assert result.evicted["id"] == "bbb"
    assert (store.root / "aaa").exists()
    assert store.active_id() == "aaa"


def test_the_policy_a_running_walk_is_using_is_never_evicted(store, onnx_specs):
    """The store cannot see the walk process, so the server passes it in. Deleting the
    model a live 50 Hz loop has open is the one eviction that could end a walk."""
    for policy_id in ("aaa", "bbb", "ccc"):
        install(store, onnx_specs, policy_id)
    age(store, "aaa", 9999)
    age(store, "bbb", 500)
    age(store, "ccc", 100)

    result = store_install_protecting(store, onnx_specs, "ddd", protect={"aaa"})

    assert result.ok
    assert result.evicted["id"] == "bbb"
    assert (store.root / "aaa").exists()


def store_install_protecting(store, onnx_specs, policy_id, protect):
    payload = f"model-{policy_id}".encode()
    store.fetch = fake_fetch(onnx_specs, payload=payload)
    return store.install(policy_id, URL, digest(payload), None, protect=protect)


def test_the_builtin_is_never_a_candidate_and_never_counted(store, onnx_specs, scripts_dir):
    """It is a resolution, not a stored file: it cannot be evicted because it is not in
    the store, and it does not consume one of the N slots."""
    for policy_id in ("aaa", "bbb", "ccc"):
        install(store, onnx_specs, policy_id)

    assert (scripts_dir / "BEST_WALK_ONNX_2.onnx").exists()
    listed = store.list()["policies"]
    assert listed[0]["id"] == BUILTIN_ID
    assert listed[0]["evictable"] is False
    assert len([p for p in listed if p["source"] == "installed"]) == store.max_policies


def test_nothing_is_evictable_is_a_refusal_not_a_deletion(store_root, scripts_dir, onnx_specs):
    """A cap of one, with that one active. Refusing is the only safe answer left."""
    store = PolicyStore(root=store_root, scripts_dir=scripts_dir, max_policies=1)
    install(store, onnx_specs, "aaa")
    register_stored(onnx_specs, store, "aaa")
    store.select("aaa")

    result = install(store, onnx_specs, "bbb")

    assert not result.ok
    assert result.code == POLICY_STORE_FULL
    assert (store.root / "aaa").exists()
    assert not (store.root / "bbb").exists()


def test_eviction_happens_only_after_the_download_and_the_check(store, onnx_specs):
    """The ordering that makes "a failed install never disturbs the current state" true.

    If room were made first, a policy the operator had would be deleted for an install
    that then failed its hash -- paying for a new policy with an old one and getting
    neither.
    """
    for policy_id in ("aaa", "bbb", "ccc"):
        install(store, onnx_specs, policy_id)

    store.fetch = fake_fetch(onnx_specs, payload=b"junk")
    result = store.install("ddd", URL, digest(b"something else"))

    assert not result.ok
    assert {p["id"] for p in store.list()["policies"]} == {BUILTIN_ID, "aaa", "bbb", "ccc"}


# ── install: the free-space floor ──────────────────────────────────────────────


def test_install_is_refused_below_the_free_space_floor_before_downloading(
    store, onnx_specs, monkeypatch
):
    """The refusal has to come first. Downloading 16 MB onto a card with 20 MB left, to
    then decide it was a bad idea, is the failure this exists to prevent."""
    monkeypatch.setattr(store, "free_bytes", lambda: 142 * 1024**2)
    fetch = fake_fetch(onnx_specs)
    store.fetch = fetch

    result = store.install("9f2a", URL, digest(b"onnx-ish bytes"))

    assert not result.ok
    assert result.code == POLICY_STORE_FULL
    assert fetch.calls == [], "downloaded anyway on a card that cannot hold it"
    assert str(142 * 1024**2) in result.detail  # bytes go to the log, not to a person


def test_a_small_declared_size_still_respects_the_floor(store, onnx_specs, monkeypatch):
    """A sidecar manifest that says "18 MB" is checked against the floor, not trusted past
    it: 199 MB free is still below a 200 MB floor."""
    monkeypatch.setattr(store, "free_bytes", lambda: 199 * 1024**2)
    store.fetch = fake_fetch(onnx_specs, payload=b"tiny")

    result = store.install("9f2a", URL, digest(b"tiny"), {"size_bytes": 18 * 1024**2})

    assert not result.ok
    assert result.code == POLICY_STORE_FULL


@pytest.mark.parametrize("declared", [None, {"size_bytes": 1}, {"size_bytes": -5}])
def test_a_declared_size_cannot_talk_the_floor_down(
    store, onnx_specs, monkeypatch, declared
):
    """One byte above the floor is below the floor plus anything the download can write, so
    every one of these is the same refusal.

    The size in a request is a claim, not a measurement, and this endpoint has no auth --
    so a request declaring one byte must not buy the same install a request declaring
    nothing is refused. Trusting it would let the card end up ``MAX_POLICY_BYTES`` under
    the floor, because ``stream_to_file`` enforces the ceiling and knows nothing about
    free space. The test above cannot see this: 199 MB is under a 200 MB floor whatever the
    reserve is.
    """
    fetch = fake_fetch(onnx_specs, payload=b"tiny")
    store.fetch = fetch
    monkeypatch.setattr(store, "free_bytes", lambda: store.free_floor_bytes + 1)

    result = store.install("9f2a", URL, digest(b"tiny"), declared)

    assert not result.ok
    assert result.code == POLICY_STORE_FULL
    assert fetch.calls == [], "a declared size got the download started below the floor"


def test_a_declared_size_larger_than_the_ceiling_only_raises_the_reserve(
    store, onnx_specs, monkeypatch
):
    """The one sound use of the declaration: a file that says it is bigger than the ceiling
    makes the refusal stricter, never looser. It never gets downloaded either way -- the
    stream would refuse it -- but the floor should not have to find that out the hard way."""
    monkeypatch.setattr(
        store, "free_bytes", lambda: store.free_floor_bytes + store.max_policy_bytes + 1
    )
    store.fetch = fake_fetch(onnx_specs, payload=b"tiny")

    fits = store.install("9f2a", URL, digest(b"tiny"))
    claims_more = store.install(
        "b17c", URL, digest(b"tiny"), {"size_bytes": store.max_policy_bytes * 4}
    )

    assert fits.ok is True
    assert claims_more.ok is False and claims_more.code == POLICY_STORE_FULL


def test_the_floor_does_not_block_an_install_that_already_happened(
    store, onnx_specs, monkeypatch
):
    """An idempotent re-install needs no bytes, so a nearly-full card must not refuse it --
    otherwise a retry on a full card is impossible, which is when a retry is most likely."""
    install(store, onnx_specs, "9f2a")
    monkeypatch.setattr(store, "free_bytes", lambda: 0)

    store.fetch = fake_fetch(onnx_specs, payload=b"model-9f2a")
    result = store.install("9f2a", URL, digest(b"model-9f2a"))

    assert result.ok and result.already_installed


def test_free_bytes_walks_up_to_a_directory_that_exists(store_root, scripts_dir):
    """The store root does not exist until the first install, and disk_usage() of a path
    that is not there raises. A floor check that raised would be a 500 on install."""
    store = PolicyStore(root=store_root / "not" / "created" / "yet", scripts_dir=scripts_dir)
    assert store.free_bytes() > 0


# ── ids ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "policy_id",
    ["../../../etc/passwd", "a/b", "..", ".hidden", "", "   ", "with space", "sha$"],
    ids=["traversal", "slash", "dotdot", "leading-dot", "empty", "blank", "space", "dollar"],
)
def test_an_id_that_is_not_safely_a_directory_name_is_refused(store, policy_id):
    """This endpoint is unauthenticated and the id becomes a path. An id like
    "../../.ssh" is not a typo to be helpful about."""
    with pytest.raises(StoreError) as excinfo:
        store.install(policy_id, URL, digest(b"x"))
    assert excinfo.value.code == POLICY_INSTALL_FAILED


def test_installing_over_the_builtin_id_is_refused(store):
    """The built-in is resolved through scripts/*.onnx and has no directory. Allowing an
    install to claim the name would make the revert target something an uploader chose."""
    with pytest.raises(StoreError) as excinfo:
        store.install(BUILTIN_ID, URL, digest(b"x"))
    assert BUILTIN_ID in excinfo.value.detail


def test_an_absurdly_long_id_is_refused(store):
    with pytest.raises(StoreError):
        store.install("a" * 300, URL, digest(b"x"))


# ── select and revert (A4) ─────────────────────────────────────────────────────


def test_select_records_the_active_id(store, onnx_specs):
    install(store, onnx_specs, "9f2a")
    register_stored(onnx_specs, store, "9f2a")

    assert store.select("9f2a") == "9f2a"
    assert store.active_id() == "9f2a"
    assert (store.root / policy_store.ACTIVE_FILENAME).read_text().strip() == "9f2a"
    assert temp_leftovers(store.root) == [], "the pointer's temp file was left behind"


def test_select_re_verifies_the_file_on_disk(store, onnx_specs):
    """A file that passed on install can have been swapped since. Re-reading it is the
    difference between having checked a file and checking THE file."""
    install(store, onnx_specs, "9f2a")
    register_stored(onnx_specs, store, "9f2a", obs_dim=47)  # somebody swapped it

    with pytest.raises(StoreError) as excinfo:
        store.select("9f2a")

    assert excinfo.value.code == POLICY_CONTRACT_MISMATCH
    assert store.active_id() == BUILTIN_ID, "a refused selection still moved the pointer"


def test_selecting_a_policy_that_is_not_installed_is_refused(store):
    with pytest.raises(StoreError) as excinfo:
        store.select("never-heard-of-it")
    assert excinfo.value.code == POLICY_INSTALL_FAILED


def test_selecting_the_builtin_works_on_an_empty_store(store):
    """The store directory does not even exist yet. Nothing about the revert may depend on
    the store being in any particular state."""
    assert not store.root.exists()
    assert store.select(BUILTIN_ID) == BUILTIN_ID
    assert store.active_id() == BUILTIN_ID


def test_selecting_the_builtin_works_when_the_active_pointer_is_corrupt(store, onnx_specs):
    install(store, onnx_specs, "9f2a")
    (store.root / policy_store.ACTIVE_FILENAME).write_bytes(b"\x00\x01not an id at all")

    assert store.select(BUILTIN_ID) == BUILTIN_ID
    assert store.active_id() == BUILTIN_ID


def test_reverting_removes_the_pointer_rather_than_writing_one(store, onnx_specs):
    """Deliberate: an absent pointer already resolves to the built-in, so the revert needs
    no free space, no temp file and no rename. A card with nothing left on it can still
    get its duck walking."""
    install(store, onnx_specs, "9f2a")
    register_stored(onnx_specs, store, "9f2a")
    store.select("9f2a")
    assert (store.root / policy_store.ACTIVE_FILENAME).exists()

    store.select(BUILTIN_ID)

    assert not (store.root / policy_store.ACTIVE_FILENAME).exists()
    assert store.active_id() == BUILTIN_ID


def test_reverting_does_not_verify_anything(store, onnx_specs, monkeypatch):
    """No check, no hash, no graph parse. Every one of those can fail, and this is the one
    action that must not."""
    install(store, onnx_specs, "9f2a")
    register_stored(onnx_specs, store, "9f2a")
    store.select("9f2a")

    def explode(*args, **kwargs):
        raise AssertionError("the revert must not verify anything")

    monkeypatch.setattr(policy_store, "check_policy", explode)
    assert store.select(BUILTIN_ID) == BUILTIN_ID


@pytest.mark.parametrize("value", ["", "   ", BUILTIN_ID, " builtin\n"])
def test_every_spelling_of_the_builtin_reverts(store, value):
    assert store.select(value) == BUILTIN_ID


def test_the_active_pointer_is_replaced_not_written_in_place(store, onnx_specs, monkeypatch):
    """Story 2.3's "atomic activation" AC, as a behaviour rather than as prose.

    A plain ``write_text`` onto the pointer passes every other test in this file, so this is
    the one that notices: the new id goes to a temp file inside the store and arrives at
    ``active`` by rename. On a Pi, power is cut by a human pulling a battery, and the
    truncated pointer that a half-written file leaves behind is the state the rename exists
    to make unreachable.
    """
    install(store, onnx_specs, "9f2a")
    register_stored(onnx_specs, store, "9f2a")
    active = store.root / policy_store.ACTIVE_FILENAME
    renames = []
    real_replace = policy_store.os.replace
    monkeypatch.setattr(
        policy_store.os,
        "replace",
        lambda src, dst: (renames.append((Path(src), Path(dst))), real_replace(src, dst))[1],
    )

    store.select("9f2a")

    assert renames, "the active pointer was written in place instead of renamed into place"
    src, dst = renames[-1]
    assert dst == active
    assert src != active and src.parent == store.root  # same filesystem, so rename is atomic
    assert active.read_text().strip() == "9f2a"


def test_a_crash_before_the_rename_leaves_the_previous_pointer_intact(
    store, onnx_specs, monkeypatch
):
    """The failure the rename buys: the write lands, the machine dies, and the pointer is
    still the id that was working. With a plain ``write_text`` there is no rename to fail,
    the old id is already gone, and what survives a real power cut is a truncated file."""
    install(store, onnx_specs, "9f2a")
    install(store, onnx_specs, "b17c")
    register_stored(onnx_specs, store, "9f2a")
    register_stored(onnx_specs, store, "b17c")
    store.select("9f2a")

    def power_cut(src, dst):
        raise OSError("the battery came out between the write and the rename")

    monkeypatch.setattr(policy_store.os, "replace", power_cut)

    with pytest.raises(OSError):
        store.select("b17c")

    assert store.active_id() == "9f2a", "the pointer moved without the rename succeeding"
    assert temp_leftovers(store.root) == [], "a half-written pointer was left in the store"


# ── the active pointer, read back ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "contents",
    [b"", b"\n", b"\x00\x01\x02", b"../../etc/passwd\n", b"a" * 500],
    ids=["empty", "newline", "binary", "traversal", "too-long"],
)
def test_an_unreadable_pointer_resolves_to_the_builtin(store, store_root, contents):
    """A power cut mid-write, a hand-edited file, a truncated card. Every one of them
    means the duck walks on the policy it shipped with, which cannot make things worse."""
    store_root.mkdir(parents=True)
    (store_root / policy_store.ACTIVE_FILENAME).write_bytes(contents)
    assert store.active_id() == BUILTIN_ID


def test_an_active_policy_whose_directory_vanished_falls_back(store, onnx_specs):
    """Not a crash and not a refusal: the duck should still walk."""
    install(store, onnx_specs, "9f2a")
    register_stored(onnx_specs, store, "9f2a")
    store.select("9f2a")
    import shutil

    shutil.rmtree(store.root / "9f2a")

    assert store.active_id() == BUILTIN_ID
    resolved = store.resolve_active()
    assert resolved.is_builtin


def test_a_named_policy_that_is_missing_is_an_error_not_a_silent_fallback(store):
    """The active pointer falls back; an explicit request does not. A caller that asked
    for one policy and silently got another has no way to notice."""
    with pytest.raises(StoreError):
        store.resolve("9f2a")


# ── the built-in is a resolution, not a copy ───────────────────────────────────


def test_the_builtin_resolves_through_the_glob_whatever_it_is_called(
    store, scripts_dir
):
    """By discovery, not by hardcoded filename: a repo that ships a differently-named
    ONNX still works, which is what keeps Decision 11's promise across a rename."""
    (scripts_dir / "BEST_WALK_ONNX_2.onnx").unlink()
    (scripts_dir / "SOME_FUTURE_WALK.onnx").write_bytes(b"whatever ships next")

    resolved = store.resolve_active()
    assert resolved.path.name == "SOME_FUTURE_WALK.onnx"
    assert resolved.is_builtin


def test_the_builtin_is_never_copied_into_the_store(store, onnx_specs, scripts_dir):
    """Copying it would double its disk cost on the card A3 is protecting, and create a
    second thing for `git pull` to leave stale."""
    install(store, onnx_specs, "9f2a")
    stored = {p.name for p in store.root.iterdir() if p.is_dir()}
    assert stored == {"9f2a"}
    assert store.list()["policies"][0]["path"] == str(
        scripts_dir / "BEST_WALK_ONNX_2.onnx"
    )


def test_a_repo_with_no_bundled_policy_resolves_to_nothing_rather_than_crashing(
    store, scripts_dir
):
    (scripts_dir / "BEST_WALK_ONNX_2.onnx").unlink()
    assert store.resolve_active() is None
    assert store.list()["policies"][0]["available"] is False


# ── listing ────────────────────────────────────────────────────────────────────


def test_list_reports_what_studio_needs(store, onnx_specs):
    install(store, onnx_specs, "9f2a")
    register_stored(onnx_specs, store, "9f2a")
    store.select("9f2a")

    listed = store.list()
    assert listed["active"] == "9f2a"
    entry = next(p for p in listed["policies"] if p["id"] == "9f2a")
    assert entry["sizeBytes"] == len(b"model-9f2a")
    assert entry["installedAt"] > 0
    assert entry["lastUsedAt"] > 0
    assert entry["manifest"]["obs_dim"] == OBS_DIM
    assert entry["active"] is True
    assert entry["evictable"] is False  # it is the active one


def test_list_never_hashes_a_file(store, onnx_specs, monkeypatch):
    """It is polled. Hashing a 16 MB model on every poll would be a self-inflicted load
    problem on a board with one slow core."""
    install(store, onnx_specs, "9f2a")

    def explode(*args, **kwargs):
        raise AssertionError("GET /api/policy hashed a model")

    monkeypatch.setattr(policy_store, "sha256_file", explode)
    assert len(store.list()["policies"]) == 2


def test_list_ignores_a_directory_with_no_model_in_it(store, onnx_specs):
    """A staging or trash directory from an interrupted install is not a policy."""
    install(store, onnx_specs, "9f2a")
    (store.root / "half-a-thing").mkdir()
    (store.root / ".staging-leftover").mkdir()

    assert {p["id"] for p in store.list()["policies"]} == {BUILTIN_ID, "9f2a"}


def test_listing_an_empty_store_still_offers_the_builtin(store):
    listed = store.list()
    assert listed["active"] == BUILTIN_ID
    assert [p["id"] for p in listed["policies"]] == [BUILTIN_ID]


# ── the downloader ─────────────────────────────────────────────────────────────


class FakeResponse:
    """Just enough of http.client.HTTPResponse for the streaming loop."""

    def __init__(self, body: bytes):
        self._body = body
        self._offset = 0
        self.reads: list[int] = []

    def read(self, size):
        self.reads.append(size)
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "url",
    ["file:///home/pi/.ssh/id_rsa", "ftp://example.com/model.onnx", "/etc/passwd"],
    ids=["file", "ftp", "bare-path"],
)
def test_only_http_urls_are_fetched(tmp_path, url):
    """An unauthenticated endpoint that accepted file:// would be "copy any path on the
    robot into the store", which is not a policy install."""
    with pytest.raises(policy_store.DownloadFailed):
        policy_store.stream_to_file(url, tmp_path / "out")


def test_the_download_is_read_in_chunks_not_in_one_gulp(tmp_path, monkeypatch):
    """Streaming is the requirement (a 512 MB board), so the loop is asserted, not the
    outcome: a read() with no size argument would pass an outcome test and still buffer
    the whole model in RAM."""
    response = FakeResponse(b"x" * (600 * 1024))
    monkeypatch.setattr(policy_store.urllib.request, "urlopen", lambda *a, **k: response)

    written = policy_store.stream_to_file("https://example.com/m.onnx", tmp_path / "out")

    assert written == 600 * 1024
    assert response.reads and all(size == 256 * 1024 for size in response.reads)


def test_the_download_is_cut_off_at_the_ceiling(tmp_path, monkeypatch):
    """Enforced while streaming, not from Content-Length: a header is a claim, and one
    request from anyone on the LAN must not be able to fill the card."""
    response = FakeResponse(b"x" * (5 * 1024 * 1024))
    monkeypatch.setattr(policy_store.urllib.request, "urlopen", lambda *a, **k: response)

    with pytest.raises(policy_store.DownloadFailed) as excinfo:
        policy_store.stream_to_file(
            "https://example.com/m.onnx", tmp_path / "out", max_bytes=1024
        )

    assert "ceiling" in str(excinfo.value)
    # Nothing past the ceiling is ever written: the count is checked before the write.
    assert (tmp_path / "out").stat().st_size <= 1024


def test_a_failed_download_never_names_the_signature(tmp_path, monkeypatch):
    """A presigned URL's query string IS the credential, and this text reaches the robot's
    stdout and PostHog's `error_message` via the telemetry middleware."""

    def boom(*args, **kwargs):
        raise OSError("connection reset by peer")

    monkeypatch.setattr(policy_store.urllib.request, "urlopen", boom)

    with pytest.raises(policy_store.DownloadFailed) as excinfo:
        policy_store.stream_to_file(URL, tmp_path / "out")

    message = str(excinfo.value)
    assert "X-Amz-Signature" not in message and "deadbeef" not in message
    assert "tnkr-artifacts.s3.amazonaws.com" in message  # enough to diagnose the network


def test_redact_url_keeps_the_host_and_path_only():
    assert policy_store.redact_url(URL).endswith("/policies/9f2a/model.onnx")
    assert "?" not in policy_store.redact_url(URL)


# ── the HTTP surface ───────────────────────────────────────────────────────────


@pytest.fixture
def api(client, tmp_path, monkeypatch, scripts_dir):
    """The server's own store, pointed at this test's directories."""
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", scripts_dir)
    monkeypatch.setattr(tnkr_server, "POLICY_ROOT", tmp_path / "server-policies")
    return client


def api_install(api, onnx_specs, monkeypatch, policy_id="9f2a", **kwargs):
    payload = f"model-{policy_id}".encode()
    monkeypatch.setattr(
        tnkr_server, "POLICY_FETCH", fake_fetch(onnx_specs, payload=payload, **kwargs)
    )
    return api.post(
        "/api/policy/install",
        json={"id": policy_id, "url": URL, "sha256": digest(payload)},
    )


def bench_passed(policy_id="9f2a", reason="watched on the bench by this test"):
    """Clear story 4.3's first-run gate for one policy.

    Free walking a policy this robot has never watched is a refusal, so every test below
    that starts a walk on a custom policy has to say that somebody watched it -- otherwise
    it would be asserting the gate rather than what it means to assert. Called explicitly
    rather than folded into ``api_install`` so it is visible which tests depend on it.
    """
    tnkr_server.get_policy_store().mark_bench(policy_id, True, reason)


def test_get_policy_on_a_fresh_robot(api):
    body = api.get("/api/policy").json()
    assert body["active"] == BUILTIN_ID
    assert len(body["policies"]) == 1
    builtin = body["policies"][0]
    assert builtin["id"] == BUILTIN_ID
    assert builtin["source"] == "bundled"
    assert builtin["evictable"] is False
    assert builtin["available"] is True
    assert builtin["active"] is True


def test_install_endpoint_returns_the_manifest(api, onnx_specs, monkeypatch):
    response = api_install(api, onnx_specs, monkeypatch)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["manifest"]["obs_dim"] == OBS_DIM
    assert body["evicted"] is None


def test_a_refused_install_is_not_a_200(api, onnx_specs, monkeypatch):
    """A caller that only looks at the status code must not read a refusal as a success."""
    response = api_install(api, onnx_specs, monkeypatch, obs_dim=47)
    assert response.status_code == 422
    assert response.json()["code"] == POLICY_CONTRACT_MISMATCH


def test_a_full_card_answers_507(api, onnx_specs, monkeypatch):
    monkeypatch.setattr(
        policy_store.PolicyStore, "free_bytes", lambda self: 10 * 1024**2
    )
    response = api_install(api, onnx_specs, monkeypatch)
    assert response.status_code == 507
    assert response.json()["code"] == POLICY_STORE_FULL


def test_select_endpoint_reverts_to_the_builtin(api, onnx_specs, monkeypatch):
    api_install(api, onnx_specs, monkeypatch)
    onnx_specs.valid(
        tnkr_server.POLICY_ROOT / "9f2a" / "model.onnx", obs_dim=OBS_DIM, act_dim=ACT_DIM
    )
    assert api.post("/api/policy/select", json={"id": "9f2a"}).json()["active"] == "9f2a"

    body = api.post("/api/policy/select", json={"id": BUILTIN_ID}).json()

    assert body["active"] == BUILTIN_ID
    assert api.get("/api/policy").json()["active"] == BUILTIN_ID


def test_reverting_works_while_a_walk_is_running(api, onnx_specs, monkeypatch, captured):
    """Amendment A4's actual scenario: the duck is walking badly, on the floor, right now.

    Nothing on this path touches the servo bus, so it does not have to wait for the walk
    to stop -- the selection takes effect on the next start.
    """
    api_install(api, onnx_specs, monkeypatch)
    onnx_specs.valid(
        tnkr_server.POLICY_ROOT / "9f2a" / "model.onnx", obs_dim=OBS_DIM, act_dim=ACT_DIM
    )
    api.post("/api/policy/select", json={"id": "9f2a"})
    bench_passed()
    monkeypatch.setattr(tnkr_server.platform, "machine", lambda: "aarch64")
    write_walk_script(tnkr_server.SCRIPTS_DIR, "import time; time.sleep(30)\n")
    assert api.post("/api/walk/start", json={}).status_code == 200

    response = api.post("/api/policy/select", json={"id": BUILTIN_ID})

    assert response.status_code == 200
    assert response.json()["active"] == BUILTIN_ID
    assert tnkr_server.is_walking(), "reverting stopped the walk"


def test_selecting_a_policy_the_robot_does_not_have(api):
    response = api.post("/api/policy/select", json={"id": "9f2a"})
    assert response.status_code == 502
    assert response.json()["code"] == POLICY_INSTALL_FAILED


def test_install_telemetry_carries_no_url_and_no_id(api, onnx_specs, monkeypatch, captured):
    """The URL is presigned; the id says which community policy an owner is trying.
    Neither is this event's business."""
    api_install(api, onnx_specs, monkeypatch)
    event = next(e for e in captured if e["properties"]["endpoint"].endswith("/install"))
    blob = json.dumps(event, default=str)
    assert "X-Amz-Signature" not in blob and "9f2a" not in blob
    assert event["properties"]["ok"] is True


def test_the_walk_script_stub_cannot_be_written_over_the_real_one():
    """The guard that stops a forgotten monkeypatch from clobbering the walk loop.

    A round of this plan shipped a working tree where scripts/v2_rl_walk_mujoco.py had been
    replaced by a three-line argv dumper. The suite was still green -- every test that reads
    the walk script reads its own tmp_path copy -- and the duck in that checkout would have
    slept for 30 s instead of walking. This is the assertion that turns that into a failure
    at the moment the write is attempted.
    """
    with pytest.raises(AssertionError, match="real scripts"):
        write_walk_script(Path(tnkr_server.__file__).parent, "import time; time.sleep(30)\n")

    real = Path(tnkr_server.__file__).parent / "v2_rl_walk_mujoco.py"
    assert real.read_text().count("\n") > 100, "the real walk script is not a stub"


# ── the spawn path: arming, end to end ─────────────────────────────────────────
#
# The seam this section exists for was fail-open once already. Arming used to be read off
# the --custom_policy CLI flag, nothing in production passed it, and a downloaded policy
# therefore ran with every guard disarmed and nothing logged. The fix moved provenance onto
# the ARTIFACT (envelope.is_armed ends `return not is_builtin_policy(onnx_model_path)`), so
# these tests assert the property that matters -- what the spawned command actually points
# at, and what is_armed says about it -- rather than whether a flag was passed.

ARGV_DUMP = (
    "import json, pathlib, sys\n"
    "pathlib.Path('argv.json').write_text(json.dumps(sys.argv))\n"
    "import time; time.sleep(30)\n"
)


def spawned_argv(scripts_dir, timeout=10.0):
    deadline = time.monotonic() + timeout
    path = scripts_dir / "argv.json"
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                pass  # caught it mid-write
        time.sleep(0.02)
    raise AssertionError("the walk script never recorded its argv")


def value_after(argv, flag):
    return argv[argv.index(flag) + 1]


@pytest.fixture
def spawning(api, monkeypatch):
    monkeypatch.setattr(tnkr_server.platform, "machine", lambda: "aarch64")
    write_walk_script(tnkr_server.SCRIPTS_DIR, ARGV_DUMP)
    return api


def test_a_policy_installed_from_the_store_comes_out_armed(
    spawning, onnx_specs, monkeypatch
):
    """THE regression test for the fail-open arming bug.

    Nobody opts in anywhere in this test: no --custom_policy in the request, no
    TNKR_FORCE_ENVELOPE in the environment. Install, select, walk -- and the policy the
    walk was spawned on must be one the envelope guards.
    """
    api_install(spawning, onnx_specs, monkeypatch)
    onnx_specs.valid(
        tnkr_server.POLICY_ROOT / "9f2a" / "model.onnx", obs_dim=OBS_DIM, act_dim=ACT_DIM
    )
    spawning.post("/api/policy/select", json={"id": "9f2a"})
    bench_passed()

    assert spawning.post("/api/walk/start", json={}).status_code == 200
    argv = spawned_argv(tnkr_server.SCRIPTS_DIR)
    loaded = value_after(argv, "--onnx_model_path")

    assert loaded == str(tnkr_server.POLICY_ROOT / "9f2a" / "model.onnx")
    assert is_builtin_policy(loaded) is False
    assert is_armed(False, loaded, env={}) is True, (
        "a policy installed from the store spawned with the safety envelope DISARMED"
    )
    # Passed for explicitness. Arming must not depend on it -- the assertion above uses
    # custom_policy=False precisely to prove it does not.
    assert "--custom_policy" in argv


def test_the_builtin_still_comes_out_unarmed(spawning):
    """Decision 11: every duck in the field keeps walking exactly as it does now. The
    built-in is trusted by construction, so it runs the unmodified loop."""
    assert spawning.post("/api/walk/start", json={}).status_code == 200
    argv = spawned_argv(tnkr_server.SCRIPTS_DIR)
    loaded = value_after(argv, "--onnx_model_path")

    assert loaded.endswith("BEST_WALK_ONNX_2.onnx")
    assert is_armed(False, loaded, env={}) is False
    assert "--custom_policy" not in argv


def test_walk_start_can_be_told_which_policy(spawning, onnx_specs, monkeypatch):
    api_install(spawning, onnx_specs, monkeypatch, policy_id="1c04")
    bench_passed("1c04")

    assert (
        spawning.post("/api/walk/start", json={"policyId": "1c04"}).status_code == 200
    )
    argv = spawned_argv(tnkr_server.SCRIPTS_DIR)

    assert value_after(argv, "--onnx_model_path") == str(
        tnkr_server.POLICY_ROOT / "1c04" / "model.onnx"
    )
    assert is_armed(False, value_after(argv, "--onnx_model_path"), env={}) is True


def test_walk_start_reports_which_policy_it_used(spawning):
    assert spawning.post("/api/walk/start", json={}).json()["policyId"] == BUILTIN_ID


def test_walk_start_refuses_a_policy_the_robot_does_not_have(spawning):
    """404 rather than silently walking on something else, and no walk is started -- a
    duck that started walking on a different policy than the one asked for is worse than
    one that did not start."""
    response = spawning.post("/api/walk/start", json={"policyId": "9f2a"})

    assert response.status_code == 404
    assert not tnkr_server.is_walking()


def test_a_refused_policy_does_not_stop_the_walk_that_is_already_running(spawning):
    """The 404 has to cost nothing, and "nothing" means torque too.

    This is the live version of the test above, and the only one that can see the bug: a
    404 raised *after* stopping the running walk SIGTERMs the walk script, whose handler
    turns the servos off, so a duck mid-stride goes limp for a request that never starts a
    walk. Studio sending an id this robot no longer has is routine -- the store is bounded,
    so installs evict policies a cached list still shows.
    """
    assert (
        spawning.post("/api/walk/start", json={"sessionToken": "sess-1"}).status_code
        == 200
    )
    spawned_argv(tnkr_server.SCRIPTS_DIR)
    assert tnkr_server.is_walking()

    response = spawning.post(
        "/api/walk/start", json={"sessionToken": "sess-2", "policyId": "gone"}
    )

    assert response.status_code == 404
    assert tnkr_server.is_walking(), (
        "a 404 for an unknown policy killed the walk that was already running"
    )


def test_a_stale_policy_id_does_not_break_an_idempotent_retry(spawning, onnx_specs, monkeypatch):
    """Same token, same walk: the retry starts nothing, so it resolves nothing.

    Resolving first must not turn the no-op into a 404 for a policy the running walk does
    not need, which is exactly what a Studio tab retrying with its own stale list sends.
    """
    api_install(spawning, onnx_specs, monkeypatch)
    bench_passed()
    assert (
        spawning.post(
            "/api/walk/start", json={"sessionToken": "sess-1", "policyId": "9f2a"}
        ).status_code
        == 200
    )
    spawned_argv(tnkr_server.SCRIPTS_DIR)

    response = spawning.post(
        "/api/walk/start", json={"sessionToken": "sess-1", "policyId": "evicted-since"}
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Walk is already running"
    assert tnkr_server.is_walking()


def test_a_new_session_still_replaces_the_running_walk(spawning, captured):
    """The reordering must not cost the stop it was ordered around: a different token is
    still a walk bound to a channel nobody is listening to, and it still gets stopped."""
    assert (
        spawning.post("/api/walk/start", json={"sessionToken": "sess-1"}).status_code
        == 200
    )
    first = spawned_argv(tnkr_server.SCRIPTS_DIR)
    (tnkr_server.SCRIPTS_DIR / "argv.json").unlink()

    assert (
        spawning.post("/api/walk/start", json={"sessionToken": "sess-2"}).status_code
        == 200
    )

    assert spawned_argv(tnkr_server.SCRIPTS_DIR)  # a second process really did start
    ended = wait_for_walk_ended(captured)[0]["properties"]
    assert ended["stop_requested"] is True and ended["crashed"] is False
    assert value_after(first, "--cloud_channel").endswith("sess-1")


def test_walk_start_falls_back_when_the_active_policy_vanished(
    spawning, onnx_specs, monkeypatch
):
    """The card was reflashed, or an eviction raced a select. The duck still walks."""
    import shutil

    api_install(spawning, onnx_specs, monkeypatch)
    onnx_specs.valid(
        tnkr_server.POLICY_ROOT / "9f2a" / "model.onnx", obs_dim=OBS_DIM, act_dim=ACT_DIM
    )
    spawning.post("/api/policy/select", json={"id": "9f2a"})
    shutil.rmtree(tnkr_server.POLICY_ROOT / "9f2a")

    assert spawning.post("/api/walk/start", json={}).status_code == 200
    argv = spawned_argv(tnkr_server.SCRIPTS_DIR)
    assert value_after(argv, "--onnx_model_path").endswith("BEST_WALK_ONNX_2.onnx")


def test_walking_stamps_the_policy_as_used(spawning, onnx_specs, monkeypatch):
    """LRU is what decides eviction, and the policy you are walking on is the last one you
    would want deleted."""
    api_install(spawning, onnx_specs, monkeypatch)
    bench_passed()
    marker = tnkr_server.POLICY_ROOT / "9f2a" / policy_store.USED_FILENAME
    stale = time.time() - 9999
    os.utime(marker, (stale, stale))

    spawning.post("/api/walk/start", json={"policyId": "9f2a"})
    spawned_argv(tnkr_server.SCRIPTS_DIR)

    assert marker.stat().st_mtime > stale


def test_the_running_policy_is_protected_from_eviction_through_the_api(
    spawning, onnx_specs, monkeypatch
):
    """The endpoint's half of the never-evict-the-running-policy rule: the server is what
    knows which policy the walk process has open."""
    for policy_id in ("aaa", "bbb", "ccc"):
        api_install(spawning, onnx_specs, monkeypatch, policy_id=policy_id)
    bench_passed("aaa")
    marker = tnkr_server.POLICY_ROOT / "aaa" / policy_store.USED_FILENAME
    stale = time.time() - 9999

    spawning.post("/api/walk/start", json={"policyId": "aaa"})
    spawned_argv(tnkr_server.SCRIPTS_DIR)
    os.utime(marker, (stale, stale))  # LRU says evict it; the walk says do not

    body = api_install(spawning, onnx_specs, monkeypatch, policy_id="ddd").json()

    assert body["evicted"]["id"] != "aaa"
    assert (tnkr_server.POLICY_ROOT / "aaa").exists()


def test_reverting_does_not_wait_for_an_install_to_finish(api, onnx_specs, monkeypatch):
    """A4, under the load it will actually meet.

    An install holds its lock for the whole download — up to a minute on household wifi.
    If reverting queued behind that, the E-stop of policy selection would have a one-minute
    delay at exactly the moment it is needed. The watchdog releases the download after 3 s
    so a regression fails the assertion instead of hanging the suite.
    """
    import threading

    released = threading.Event()
    started = threading.Event()

    def blocking_fetch(url, dest, **kwargs):
        Path(dest).write_bytes(b"model-9f2a")
        started.set()
        released.wait(10)
        onnx_specs.valid(dest, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        return len(b"model-9f2a")

    monkeypatch.setattr(tnkr_server, "POLICY_FETCH", blocking_fetch)
    installer = threading.Thread(
        target=api.post,
        args=("/api/policy/install",),
        kwargs={"json": {"id": "9f2a", "url": URL, "sha256": digest(b"model-9f2a")}},
        daemon=True,
    )
    installer.start()
    assert started.wait(5), "the fake download never started"
    threading.Timer(3.0, released.set).start()

    began = time.monotonic()
    response = api.post("/api/policy/select", json={"id": BUILTIN_ID})
    took = time.monotonic() - began

    assert response.json()["active"] == BUILTIN_ID
    assert took < 1.0, f"reverting waited {took:.1f}s for an in-flight install"
    released.set()
    installer.join(10)


def test_a_card_that_fills_up_mid_write_is_a_refusal_not_a_crash(store, onnx_specs, monkeypatch):
    """ENOSPC while writing the temp file. The floor makes this unlikely, not impossible --
    something else on the Pi can be writing too -- and a 500 from the robot would tell the
    operator nothing."""

    def full_disk(url, dest, **kwargs):
        raise OSError(28, "No space left on device")

    store.fetch = full_disk
    result = store.install("9f2a", URL, digest(b"x"))

    assert not result.ok
    assert result.code == POLICY_INSTALL_FAILED
    assert not (store.root / "9f2a").exists()


def test_a_verified_policy_that_cannot_be_moved_into_place_changes_nothing(
    store, onnx_specs, monkeypatch
):
    """The last step can fail too. It must fail the way every earlier step does: a code, a
    detail for the log, and a store exactly as it was."""
    install(store, onnx_specs, "aaa")
    before = (store.root / "aaa" / "model.onnx").read_bytes()

    real_replace = os.replace

    def refuse(src, dst, *args, **kwargs):
        if "staging" in str(src) or "staging" in str(dst):
            raise OSError(30, "Read-only file system")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(policy_store.os, "replace", refuse)
    result = install(store, onnx_specs, "bbb")

    assert not result.ok
    assert result.code == POLICY_INSTALL_FAILED
    assert not (store.root / "bbb").exists()
    assert (store.root / "aaa" / "model.onnx").read_bytes() == before
    assert temp_leftovers(store.root) == [], "a staging directory was left behind"


def test_polling_the_policy_list_does_not_burn_telemetry(api, captured):
    """The lesson /api/state taught this file the expensive way: a polled read that worked
    is not news, and capture() is rate-capped at 60/min indiscriminately, so it crowds out
    the events the funnel is built from. Failures are still captured."""
    assert api.get("/api/policy").status_code == 200
    assert not [
        e for e in captured if e["properties"].get("endpoint") == "/api/policy"
    ]
