"""The boundary check: every way a policy can be refused, and one way it is accepted.

Why this file is long
---------------------
``check_policy`` is the only thing between an arbitrary ONNX and the servos. Amendment A1
put it here rather than in Studio because the robot's HTTP API has no authentication and its
CORS reflects any requesting origin with credentials, so any webpage the owner visits can
POST an install. A branch with no test is a hole in that.

The ten branches story 2.2 names, each with a test below:

    valid  ·  wrong input name  ·  wrong obs width  ·  wrong action width
    multiple inputs  ·  dynamic axis  ·  non-float dtype  ·  unparseable file
    oversize file  ·  sha mismatch

Plus the ones the story's "Error Scenarios" section names (zero inputs, zero outputs,
int64 output, a digest with the wrong case or stray whitespace, an unreadable file, a
missing onnxruntime) and two assertions about *order*, which is the part a reader would
otherwise have to take on trust.

No onnxruntime here
-------------------
Every graph comes from the configurable double in ``tests/stubs/onnxruntime.py`` via the
``onnx_specs`` fixture (story 2.1). CI installs no native wheel, and
``tests/test_stub_fidelity.py`` is what keeps the double honest about what real onnxruntime
presents.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import time

import numpy as np
import onnxruntime as onnx_double  # tests/stubs, not the wheel
import pytest

from mini_bdx_runtime import policy_contract
from mini_bdx_runtime.policy_contract import (
    ACT_DIM,
    MAX_POLICY_BYTES,
    OBS_DIM,
    OBS_INPUT_NAME,
    OBS_VERSION,
    POLICY_CONTRACT_MISMATCH,
    POLICY_INSTALL_FAILED,
    check_policy,
    normalise_digest,
    sha256_file,
)

FLOAT = "tensor(float)"


@pytest.fixture
def model(tmp_path):
    """A file that exists and is small. Its BYTES are irrelevant: the double decides what
    graph the path presents, exactly as ``tests/conftest.py:104`` already writes a dummy
    model.onnx for the walk tests."""
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx-ish bytes")
    return path


def digest_of(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── accepted ────────────────────────────────────────────────────────────────────


def test_accepts_the_shape_the_duck_runs(onnx_specs, model):
    """The one accepted case, at the widths BEST_WALK_ONNX_2.onnx actually has."""
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    result = check_policy(model)

    assert result.ok, result.detail
    assert result.code is None
    assert result.manifest is not None
    assert result.manifest["obs_version"] == OBS_VERSION
    assert result.manifest["obs_dim"] == OBS_DIM
    assert result.manifest["act_dim"] == ACT_DIM
    # Decision 5: a bare ONNX with no manifest is still installable, marked inferred.
    assert result.manifest["inferred"] is True
    assert result.manifest["size_bytes"] == model.stat().st_size


def test_accepts_an_unbatched_input(onnx_specs, model):
    """``[101]`` with no batch axis is still static and still 101 floats.

    Both of the only two published Open Duck policies export ``[1, 101]``, but refusing a
    batchless graph would be refusing a shape we can verify, which is not what the refusal
    is for."""
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [OBS_DIM], FLOAT)],
        outputs=[("continuous_actions", [ACT_DIM], FLOAT)],
    )

    assert check_policy(model).ok


def test_accepted_manifest_records_the_verified_digest(onnx_specs, model):
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    result = check_policy(model, expect_sha256=digest_of(model))

    assert result.ok
    assert result.manifest["sha256"] == digest_of(model)


# ── refused: the graph is not duck-obs-v1 ───────────────────────────────────────


def test_refuses_wrong_input_name(onnx_specs, model):
    """A graph whose input is called something else is a policy for something else."""
    onnx_specs.register(
        model,
        inputs=[("observations", [1, OBS_DIM], FLOAT)],
        outputs=[("continuous_actions", [1, ACT_DIM], FLOAT)],
    )

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "'observations'" in result.detail and repr(OBS_INPUT_NAME) in result.detail
    assert result.manifest is None


def test_refuses_wrong_obs_width(onnx_specs, model):
    """The story's own example: a 47-wide obs, which is what an older duck contract was."""
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [1, 47], FLOAT)],
        outputs=[("continuous_actions", [1, ACT_DIM], FLOAT)],
    )

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert str(OBS_DIM) in result.detail and "47" in result.detail


def test_refuses_wrong_action_width(onnx_specs, model):
    """Right observation, wrong number of joints -- a policy for a different robot."""
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [1, OBS_DIM], FLOAT)],
        outputs=[("continuous_actions", [1, 12], FLOAT)],
    )

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert str(ACT_DIM) in result.detail and "12" in result.detail


def test_refuses_multiple_inputs_and_names_them(onnx_specs, model):
    """A manipulation VLA aimed at the duck. Its inputs are a real shape, just not ours,
    so the refusal names what it found rather than claiming the file is broken."""
    onnx_specs.register(
        model,
        inputs=[
            ("image", [1, 3, 224, 224], FLOAT),
            ("state", [1, 32], FLOAT),
            ("prompt", [1, 77], "tensor(int64)"),
        ],
        outputs=[("actions", [1, 7], FLOAT)],
    )

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "3 inputs" in result.detail
    for name in ("image", "state", "prompt"):
        assert name in result.detail


def test_refuses_zero_inputs(onnx_specs, model):
    """A graph with no inputs parses fine and is nonsense. ``found 0 inputs: none``
    rather than an IndexError from reading ``get_inputs()[0]``."""
    onnx_specs.register(model, inputs=[], outputs=[("actions", [1, ACT_DIM], FLOAT)])

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "0 inputs" in result.detail


def test_refuses_zero_outputs(onnx_specs, model):
    onnx_specs.register(model, inputs=[(OBS_INPUT_NAME, [1, OBS_DIM], FLOAT)], outputs=[])

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "0" in result.detail


def test_refuses_multiple_outputs(onnx_specs, model):
    """Two heads (actions plus a value estimate) is a training-time export, not a
    deployable one: the loop takes ``outputs[0]`` and would silently use whichever head
    the exporter happened to put first."""
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [1, OBS_DIM], FLOAT)],
        outputs=[
            ("continuous_actions", [1, ACT_DIM], FLOAT),
            ("value", [1, 1], FLOAT),
        ],
    )

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "2" in result.detail


@pytest.mark.parametrize("dynamic_dim", [None, "batch", "N"])
def test_refuses_a_dynamic_batch_axis_and_says_why(onnx_specs, model, dynamic_dim):
    """Static shapes only, and the refusal has to explain itself.

    Both spellings of a dynamic axis appear in the wild: exporters that leave the dim
    unnamed give ``None``, ones that name it give a string. Either way we cannot assert the
    graph is 101 floats wide, and "cannot assert" on the hardware boundary means refuse.
    The detail must say that in words, because a policy that is merely dynamic-batch may
    well be legitimate and its owner needs to be able to report it rather than guess.
    """
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [dynamic_dim, OBS_DIM], FLOAT)],
        outputs=[("continuous_actions", [1, ACT_DIM], FLOAT)],
    )

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "not static" in result.detail
    assert "deliberate refusal" in result.detail


def test_refuses_a_dynamic_output_axis(onnx_specs, model):
    """The output side gets the same treatment; a passing input is not a pass."""
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [1, OBS_DIM], FLOAT)],
        outputs=[("continuous_actions", [None, ACT_DIM], FLOAT)],
    )

    result = check_policy(model)

    assert not result.ok
    assert "not static" in result.detail


def test_refuses_a_batch_larger_than_one(onnx_specs, model):
    """``[4, 101]`` is static and still wrong: the loop feeds one observation per tick."""
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [4, OBS_DIM], FLOAT)],
        outputs=[("continuous_actions", [1, ACT_DIM], FLOAT)],
    )

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH


def test_refuses_non_float_input_dtype(onnx_specs, model):
    """An int64 input is not 101 floats however wide it is -- a quantised or tokenised
    graph, fed float32 by the loop, reads garbage."""
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [1, OBS_DIM], "tensor(int64)")],
        outputs=[("continuous_actions", [1, ACT_DIM], FLOAT)],
    )

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "tensor(int64)" in result.detail and FLOAT in result.detail


def test_refuses_non_float_output_dtype(onnx_specs, model):
    """The story names this one explicitly: right width, wrong dtype."""
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [1, OBS_DIM], FLOAT)],
        outputs=[("continuous_actions", [1, ACT_DIM], "tensor(int64)")],
    )

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "tensor(int64)" in result.detail


def test_refuses_an_unparseable_file_without_raising(onnx_specs, model):
    """F7 in the architecture's failure table: onnxruntime raising on a corrupt model must
    become a refusal, not a 500 from the robot's API."""
    onnx_specs.register(model, invalid=True)

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "InvalidProtobuf" in result.detail


# ── refused: the file itself ────────────────────────────────────────────────────


def test_refuses_an_oversize_file_before_parsing_it(onnx_specs, model):
    """The size ceiling exists to keep a hostile file away from the parser, so the test
    asserts the parser was never reached -- not merely that the answer was 'no'.

    A valid graph is registered on purpose: if the order were wrong this would pass its
    shape check and the refusal would come from somewhere else, which the emptiness of
    ``constructed`` is what distinguishes."""
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    result = check_policy(model, max_bytes=4)

    assert not result.ok
    assert result.code == POLICY_CONTRACT_MISMATCH
    assert "ceiling" in result.detail
    assert onnx_specs.constructed == [], "an oversize file reached onnxruntime"


def test_refuses_a_hash_mismatch_before_parsing_it(onnx_specs, model):
    """Integrity before RAM: a file that is not the file we were promised does not get
    parsed. Same reasoning as the ceiling, same assertion on ``constructed``."""
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    result = check_policy(model, expect_sha256="0" * 64)

    assert not result.ok
    assert result.code == POLICY_INSTALL_FAILED
    assert digest_of(model) in result.detail
    assert onnx_specs.constructed == [], "a hash mismatch reached onnxruntime"


def test_accepts_a_digest_with_stray_whitespace_and_capitals(onnx_specs, model):
    """Digests arrive in JSON typed by people and emitted by other tools. Refusing a
    correct install over a capital letter would be a bug that looks like a security
    feature."""
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    result = check_policy(model, expect_sha256=f"  {digest_of(model).upper()}\n")

    assert result.ok, result.detail


def test_refuses_a_malformed_digest(onnx_specs, model):
    """``deadbeef`` is not a sha256. Comparing it would fail anyway, but the detail should
    say the request was malformed rather than implying the file was tampered with."""
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    result = check_policy(model, expect_sha256="deadbeef")

    assert not result.ok
    assert result.code == POLICY_INSTALL_FAILED
    assert "sha256" in result.detail
    assert onnx_specs.constructed == []


def test_refuses_a_file_that_is_not_there(onnx_specs, tmp_path):
    """A download that died leaves nothing behind; the check says so instead of raising."""
    result = check_policy(tmp_path / "gone.onnx")

    assert not result.ok
    assert result.code == POLICY_INSTALL_FAILED
    assert "cannot read" in result.detail


def test_refuses_when_the_read_fails_mid_check(onnx_specs, model, monkeypatch):
    """An SD card that stops answering during the hash. Refuse; never let the OSError out,
    because the caller's job at that point is to leave the previous policy intact."""
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    def boom(*args, **kwargs):
        raise OSError("Input/output error")

    # Patched on the module rather than on builtins: a broken global ``open`` also breaks
    # pytest's own reporting, which turns a failure here into an unreadable one.
    monkeypatch.setattr(policy_contract, "open", boom, raising=False)

    result = check_policy(model, expect_sha256=digest_of(model))

    assert not result.ok
    assert result.code == POLICY_INSTALL_FAILED
    assert "read failed" in result.detail


def test_refuses_when_onnxruntime_is_missing(onnx_specs, model, monkeypatch):
    """A refusal, not an ImportError.

    onnxruntime is a hard dependency on the robot (``setup.cfg``), so this should not happen
    there -- but the same module is imported by tooling off-robot, and a half-installed Pi
    is a real state (``setup.sh`` step 8 can fail on a wheel build). Refusing to verify has
    to read as "not verified", never as a crash that a caller might treat as a pass.
    """
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    class Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] == "onnxruntime":
                raise ImportError("blocked for test")
            return None

    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])

    result = check_policy(model)

    assert not result.ok
    assert result.code == POLICY_INSTALL_FAILED
    assert "onnxruntime is not installed" in result.detail


# ── the properties that keep it a boundary ──────────────────────────────────────


def test_no_caller_can_skip_the_check():
    """There is no trusted caller, so there is no parameter that turns the check off.

    This is a guard against a plausible future convenience: story 2.3 calls check_policy on
    both install and select, and someone will notice select re-hashes a file it just
    verified. The answer is not a skip flag -- re-verifying is the entire point of checking
    on select (a file can be swapped on disk between the two).
    """
    params = set(inspect.signature(check_policy).parameters)
    forbidden = {"skip", "skip_check", "trusted", "internal", "force", "unsafe", "verify"}
    assert not (params & forbidden), (
        f"check_policy grew a bypass parameter: {sorted(params & forbidden)}. The robot's "
        "API has no authentication; a bypass is a bypass for whoever finds it."
    )


def test_nothing_is_returned_to_store_on_refusal(onnx_specs, model):
    """Every refusal carries a null manifest, so a caller cannot accidentally persist a
    manifest for a policy that was rejected."""
    onnx_specs.register(
        model,
        inputs=[(OBS_INPUT_NAME, [1, 47], FLOAT)],
        outputs=[("continuous_actions", [1, ACT_DIM], FLOAT)],
    )

    result = check_policy(model)

    assert result.manifest is None
    assert result.as_dict()["manifest"] is None


def test_detail_never_leaks_to_the_operator_untyped(onnx_specs, model):
    """Every refusal has BOTH a code and a detail: the code is what Studio maps to one
    operator sentence, the detail is what goes to the log (app/DESIGN.md#errors)."""
    onnx_specs.register(model, invalid=True)

    result = check_policy(model)

    assert result.code in (POLICY_CONTRACT_MISMATCH, POLICY_INSTALL_FAILED)
    assert result.detail


def test_the_ceiling_default_is_the_contract_constant():
    """The default comes from policy_contract, not from a number typed at a callsite."""
    assert (
        inspect.signature(check_policy).parameters["max_bytes"].default == MAX_POLICY_BYTES
    )


# ── helpers the store will reuse ────────────────────────────────────────────────


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"x" * (3 * 1024 * 1024 + 7))  # spans several read chunks

    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A" * 64, "a" * 64),
        (f"  {'b' * 64}\n", "b" * 64),
        ("deadbeef", None),
        ("g" * 64, None),
        ("", None),
    ],
)
def test_normalise_digest(raw, expected):
    assert normalise_digest(raw) == expected


# ── the double itself (story 2.1) ───────────────────────────────────────────────


def test_double_run_returns_zeros_of_the_declared_shape(onnx_specs, model):
    """Story 2.6 measures latency by calling run() in a loop, so it has to return
    something of the right shape rather than None."""
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    session = onnx_double.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    (out,) = session.run(None, {OBS_INPUT_NAME: [[0.0] * OBS_DIM]})

    assert out.shape == (1, ACT_DIM)
    assert out.dtype == np.float32
    assert not out.any()


def test_double_run_honours_an_injected_delay(onnx_specs, model):
    """The over-budget branch of story 2.6 must not depend on how fast CI happens to be."""
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM, delay_s=0.02)
    session = onnx_double.InferenceSession(str(model))

    start = time.perf_counter()
    session.run(None, {OBS_INPUT_NAME: [[0.0] * OBS_DIM]})

    assert time.perf_counter() - start >= 0.02


def test_double_rejects_a_run_missing_an_input(onnx_specs, model):
    onnx_specs.valid(model, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    session = onnx_double.InferenceSession(str(model))

    with pytest.raises(onnx_double.InvalidArgument):
        session.run(None, {})


def test_double_fails_loudly_on_an_unregistered_path(tmp_path):
    """And it fails in a way ``except Exception`` cannot swallow, so a test that forgot to
    register a spec does not quietly pass as a rejection test."""
    with pytest.raises(onnx_double.UnregisteredPath) as exc:
        onnx_double.InferenceSession(str(tmp_path / "nothing.onnx"))

    assert "onnx_specs" in str(exc.value)
    assert not isinstance(exc.value, Exception), (
        "UnregisteredPath must not be an Exception subclass: check_policy's "
        "`except Exception` would absorb it and the test would pass on the wrong refusal."
    )


@pytest.fixture
def shared_model(tmp_path_factory):
    """One path shared by every test in the session, unlike ``tmp_path``.

    Two tests can only collide in the double's registry if they register the same key, and
    ``tmp_path`` makes that impossible -- which would make a leak test that uses it vacuous.
    """
    path = tmp_path_factory.getbasetemp() / "shared_model.onnx"
    path.write_bytes(b"onnx-ish bytes")
    return path


def test_specs_do_not_leak_between_tests(onnx_specs, shared_model):
    """Paired with the next test: both register the SAME path, one valid and one not.
    Whichever runs second is the one that proves the fixture tore down, because a leaked
    spec from the other would answer instead."""
    onnx_specs.valid(shared_model, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    assert check_policy(shared_model).ok


def test_specs_do_not_leak_between_tests_reverse(onnx_specs, shared_model):
    onnx_specs.register(shared_model, invalid=True)
    assert not check_policy(shared_model).ok
