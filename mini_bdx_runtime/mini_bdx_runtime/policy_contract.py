"""The observation contract a walk policy must satisfy: ``duck-obs-v1``.

Why this module exists
----------------------
The contract used to live in three places and drift silently:

  1. ``Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py``  (training env)
  2. ``Open_Duck_Playground/playground/open_duck_mini_v2/mujoco_infer.py:83-101``
  3. ``scripts/v2_rl_walk_mujoco.py:210-226``  (this repo, the on-robot loop)

They were verified term-for-term identical on 2026-08-20, which was luck rather than
enforcement: nothing in any repo would have complained if one had changed.

Naming the contract is what lets a third party target it. Until now "a policy for the
duck" was defined only by whichever ``.onnx`` happened to be in ``scripts/``.

What this module deliberately does NOT do
-----------------------------------------
It does **not** build the observation vector, and ``get_obs()`` does **not** import from
it. An earlier design had ``get_obs()`` construct its ``np.concatenate`` from ``OBS_TERMS``
so the two could not disagree. That was retracted: the same guarantee is available from a
test (``tests/test_policy_contract.py::test_get_obs_matches_contract``), and a test does
not require editing the one function that every duck in the field depends on, where a
reordering bug is invisible — the vector is still 101 floats, the duck just walks worse.

So the rule this module establishes, for anyone extending it: **prefer a test over a loop
edit whenever the test yields the same guarantee.**

The transforms stay in ``get_obs()`` on purpose
-----------------------------------------------
``dof_pos_rel`` carries a ``- init_pos`` offset and ``dof_vel_scaled`` a ``* 0.05`` scale.
This module records each term's *name and width* only. Moving the transforms here would
make it an observation *builder*, which is the manifest-interpreting design the
architecture explicitly rejected as speculative generality — ``duck-obs-v1`` is the only
contract that exists.

See ``tnkr-studio/docs/plans/custom-policy/_architecture.md``, Decision 3 and amendment A6.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Sequence

# ── The contract ────────────────────────────────────────────────────────────────
#
# Order is load-bearing, not just total width. A policy trained on
# [gyro, accelerometer, ...] reads garbage if handed [accelerometer, gyro, ...] of the
# same length, and nothing raises — it just walks badly. This tuple IS the order.
#
# Mirrors scripts/v2_rl_walk_mujoco.py:210-226 exactly:
#
#     imu_data["gyro"]              -> gyro                    3
#     imu_data["accelero"]          -> accelerometer           3
#     cmds                          -> commands                7
#     dof_pos - self.init_pos       -> dof_pos_rel            14
#     dof_vel * 0.05                -> dof_vel_scaled         14
#     self.last_action              -> last_action            14
#     self.last_last_action         -> last_last_action       14
#     self.last_last_last_action    -> last_last_last_action  14
#     self.motor_targets            -> motor_targets          14
#     feet_contacts                 -> feet_contacts           2
#     self.imitation_phase          -> imitation_phase          2
#                                                            ---
#                                                            101
OBS_TERMS: tuple[tuple[str, int], ...] = (
    ("gyro", 3),
    ("accelerometer", 3),
    ("commands", 7),
    ("dof_pos_rel", 14),
    ("dof_vel_scaled", 14),
    ("last_action", 14),
    ("last_last_action", 14),
    ("last_last_last_action", 14),
    ("motor_targets", 14),
    ("feet_contacts", 2),
    ("imitation_phase", 2),
)

# Derived, never written as a literal. If you find yourself typing 101 or 14 anywhere,
# import these instead — that is the entire point of the module.
OBS_DIM: int = sum(width for _, width in OBS_TERMS)
ACT_DIM: int = 14
CONTROL_HZ: int = 50

OBS_VERSION: str = "duck-obs-v1"

# The single ONNX input name the runtime feeds. OnnxInfer's input_name defaults to "obs"
# (onnx_infer.py:5); its awd flag defaults to False and it is the WALK SCRIPT that passes
# awd=True (v2_rl_walk_mujoco.py:78), which is what wraps the vector in a batch of one
# and reads row zero of the result back out. See _batched_width: that rank is the contract.
OBS_INPUT_NAME: str = "obs"

# Reject before parsing rather than after. A hostile or corrupt multi-hundred-MB file
# should never reach onnxruntime's parser on a memory-constrained Pi.
MAX_POLICY_BYTES: int = 256 * 1024 * 1024

# onnxruntime's dtype string for a float32 tensor, verbatim. Read off onnxruntime 1.24.4
# loading scripts/BEST_WALK_ONNX_2.onnx: input 'obs' [1, 101] tensor(float), output
# 'continuous_actions' [1, 14] tensor(float).
_FLOAT_TENSOR: str = "tensor(float)"


def term_offsets() -> dict[str, tuple[int, int]]:
    """Map each term to its ``(start, stop)`` slice in the observation vector.

    For interpreting a recorded observation after the fact — debugging a policy, or
    plotting what it actually saw. Not used by the control loop.
    """
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, width in OBS_TERMS:
        offsets[name] = (cursor, cursor + width)
        cursor += width
    return offsets


def describe() -> str:
    """One-line human summary, for logs and the ``--help`` of tooling."""
    return f"{OBS_VERSION}: {OBS_DIM} floats in, {ACT_DIM} out, at {CONTROL_HZ} Hz"


def validate_obs(obs: Any) -> None:
    """Raise ``ValueError`` unless ``obs`` is a ``duck-obs-v1`` observation vector.

    **Nothing in the control loop calls this, by design.** Story 1.1 does not edit
    ``get_obs()``, so there is no call site there — a per-tick length check would be new
    code in the 20 Hz-budget hot path to catch a bug the static guard in
    ``tests/test_policy_contract.py`` already catches at commit time.

    It ships now because later stories need exactly this, off the hot path: validating a
    replayed observation (``--replay_obs``), a recorded one from ``--save_obs``, or a
    vector handed to a policy during install-time latency measurement (story 2.6). Keeping
    it here means those stories do not each re-derive what "a valid observation" means.

    The term-level error is best-effort: a flat vector of the wrong length cannot say which
    term is wrong, so it reports the boundary the length falls inside. That is more useful
    than "expected 101, got 102" alone, because it points at the term to look at first.
    """
    if obs is None:
        raise ValueError(f"{OBS_VERSION}: observation is None, expected {OBS_DIM} floats")

    try:
        length = len(obs)
    except TypeError as exc:
        raise ValueError(
            f"{OBS_VERSION}: observation has no length ({type(obs).__name__})"
        ) from exc

    if length == OBS_DIM:
        return

    # Name the term the mismatch lands in — the first place to look.
    cursor = 0
    culprit = OBS_TERMS[-1][0]
    for name, width in OBS_TERMS:
        cursor += width
        if length < cursor:
            culprit = name
            break

    raise ValueError(
        f"{OBS_VERSION}: observation is {length} floats, expected {OBS_DIM}. "
        f"The mismatch falls at or before the {culprit!r} term "
        f"(offset {term_offsets()[culprit][0]})."
    )


# ── the boundary check ──────────────────────────────────────────────────────────
#
# Refusal codes. They are strings rather than an enum because they cross the wire to
# Studio, where they must match the ErrorCode members added in story 2.5 exactly.
POLICY_CONTRACT_MISMATCH: str = "POLICY_CONTRACT_MISMATCH"
POLICY_INSTALL_FAILED: str = "POLICY_INSTALL_FAILED"

# How much of the file to hash at a time. The point of streaming is that a 200 MB model
# never becomes a 200 MB bytes object on a Pi Zero 2W.
_HASH_CHUNK_BYTES: int = 1024 * 1024


@dataclass
class CheckResult:
    """The outcome of inspecting a candidate policy file.

    ``detail`` is developer-facing and names what was found versus what was expected --
    widths, names, dtypes. The operator sees one short sentence mapped from ``code`` in
    Studio, per ``tnkr-studio/app/DESIGN.md#errors``; none of this text reaches a screen.
    """

    ok: bool
    code: str | None
    detail: str
    manifest: dict | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "code": self.code,
            "detail": self.detail,
            "manifest": self.manifest,
        }


def normalise_digest(value: str) -> str | None:
    """Lowercase, whitespace-stripped sha256 hex, or ``None`` if it is not one.

    Callers hand us digests that came from JSON typed by a human or generated by another
    tool, so ``"  9F2A...\\n"`` is a digest and ``"deadbeef"`` (too short) is not. Comparing
    unnormalised strings would refuse a correct install over a capital letter.
    """
    candidate = value.strip().lower()
    if len(candidate) != 64:
        return None
    if any(c not in "0123456789abcdef" for c in candidate):
        return None
    return candidate


def sha256_file(
    path: "os.PathLike[str] | str", chunk_bytes: int = _HASH_CHUNK_BYTES
) -> str:
    """Streaming sha256 of a file. Raises ``OSError`` if the read fails."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def infer_manifest(session: Any) -> dict:
    """The manifest a conforming graph implies. Only valid after the shape check passed.

    ``inferred`` is always true here: this is the manifest *derived from the graph*, which
    is the Decision 5 path -- a bare ONNX off Discord is still installable, we just record
    that nobody told us what it was. Story 2.3 merges a supplied sidecar manifest over
    this and flips the flag.
    """
    obs_in = session.get_inputs()[0]
    act_out = session.get_outputs()[0]
    return {
        "obs_version": OBS_VERSION,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "inferred": True,
        "input_name": obs_in.name,
        "output_name": act_out.name,
        "input_shape": list(obs_in.shape),
        "output_shape": list(act_out.shape),
    }


def _batched_width(shape: Sequence[Any]) -> int | None:
    """The ``N`` of an exactly-``[1, N]`` shape, else ``None``.

    Rank is part of the assertion, not incidental to it. The walk loop builds its
    ``OnnxInfer`` with ``awd=True`` (``v2_rl_walk_mujoco.py:78``), which means it feeds
    ``{"obs": [obs]}`` -- rank 2, batch of one -- and reads ``outputs[0][0]``, the first
    *row* of a rank-2 result. Both halves of that are the contract.

    An earlier version of this function only checked the trailing dimension, so a graph
    exported without the batch axis (input ``[101]``, output ``[14]``) was accepted. That is
    the worst thing this check can wave through, and it is worse than a crash: real
    onnxruntime returns a ``(14,)`` array for that output, so ``outputs[0][0]`` is a numpy
    *scalar*, and ``init_pos + scalar * action_scale`` at line 450 broadcasts it into 14
    identical motor targets. Every joint gets the same offset from its init position, at
    50 Hz, with torque on -- and nothing raises, nothing logs, and the envelope's clamps see
    a perfectly plausible 14-vector. The operator has no signal at all that the policy's
    output was discarded.

    ``None`` means "cannot assert", and covers a dynamic axis, a rank that is not two, and a
    batch that is not one. Callers turn it into a refusal that names which, so this function
    deliberately does not decide.
    """
    if len(shape) != 2:
        return None
    if any(not isinstance(dim, int) for dim in shape):
        return None
    if shape[0] != 1:
        return None
    return int(shape[1])


def _describe_shape(shape: Sequence[Any]) -> str:
    return "[" + ", ".join("None" if d is None else repr(d) for d in shape) + "]"


# Why the rank is refused rather than tolerated, in the words the reporter of a legitimate
# case will need. Shared by the input and output branches because the reason is the same one.
_SHAPE_RULE: str = (
    f"{OBS_VERSION} requires exactly [1, N]: the walk loop constructs OnnxInfer with "
    "awd=True, so it feeds one observation wrapped in a batch of one and reads row zero of "
    "the result back. A shape that does not resolve to that cannot be verified before "
    "torque is enabled, and 'cannot assert' on the hardware boundary means refuse."
)


def _shape_refusal(role: str, arg: Any) -> str:
    """The refusal text for a shape that is not ``[1, N]``, naming which defect it has.

    Says *why* it is refused, not just that it was. Each of the three defects is a case that
    might be legitimate and reportable -- a dynamic-batch graph may well work, since the loop
    always calls with a batch of one -- so the detail has to distinguish them. "Bad shape"
    would leave the owner of a working policy guessing.
    """
    shape = list(arg.shape)

    if any(not isinstance(dim, int) for dim in shape):
        defect = "is not static"
        remedy = (
            "This is a deliberate refusal, not a parse failure: report it if your policy "
            "is genuinely dynamic-batch."
        )
    elif len(shape) != 2:
        defect = f"is rank {len(shape)}, not rank 2"
        remedy = (
            "Re-export with the batch axis present. A missing batch axis is the dangerous "
            "one: on the output side it makes the loop read a scalar and command every "
            "joint the same offset, so it is refused rather than run."
        )
    else:
        defect = f"is a batch of {shape[0]}, not a batch of one"
        remedy = "Re-export with a fixed batch of one."

    return (
        f"{role} {arg.name!r} has shape {_describe_shape(shape)}, which {defect}. "
        f"{_SHAPE_RULE} {remedy}"
    )


def _signature_error(session: Any) -> str | None:
    """``None`` if the graph is ``duck-obs-v1``, else why it is not (developer-facing)."""
    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())

    if len(inputs) != 1:
        found = ", ".join(a.name for a in inputs) or "none"
        return (
            f"expected 1 input named {OBS_INPUT_NAME!r}; found {len(inputs)} "
            f"inputs: {found}"
        )
    obs_in = inputs[0]

    if obs_in.name != OBS_INPUT_NAME:
        return (
            f"expected the input to be named {OBS_INPUT_NAME!r}; found {obs_in.name!r}"
        )

    if obs_in.type != _FLOAT_TENSOR:
        return (
            f"input {obs_in.name!r} is {obs_in.type}, expected {_FLOAT_TENSOR} -- "
            f"{OBS_VERSION} is {OBS_DIM} floats"
        )

    obs_width = _batched_width(obs_in.shape)
    if obs_width is None:
        return _shape_refusal("input", obs_in)
    if obs_width != OBS_DIM:
        return f"input {obs_in.name!r} is {obs_width} wide, expected {OBS_DIM}"

    if len(outputs) != 1:
        found = ", ".join(a.name for a in outputs) or "none"
        return f"expected 1 output; found {len(outputs)}: {found}"
    act_out = outputs[0]

    if act_out.type != _FLOAT_TENSOR:
        return (
            f"output {act_out.name!r} is {act_out.type}, expected {_FLOAT_TENSOR} -- "
            f"the loop writes {ACT_DIM} float motor targets"
        )

    act_width = _batched_width(act_out.shape)
    if act_width is None:
        return _shape_refusal("output", act_out)
    if act_width != ACT_DIM:
        return f"output {act_out.name!r} is {act_width} wide, expected {ACT_DIM}"

    return None


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ORDER OF CHECKS -- cheap rejections first, and that is a resource      │
# │  decision, not a style one.                                             │
# │                                                                         │
# │     size ceiling  ->  sha256  ->  parse graph  ->  input/output shape   │
# │        cheap          cheap        EXPENSIVE         cheap              │
# │      one stat()     streamed      loads the        reads the            │
# │                     in 1 MB       whole model      specs off            │
# │                     chunks        into RAM         the session          │
# │                                                                         │
# │  A hostile or corrupt multi-hundred-MB file must never reach            │
# │  onnxruntime's parser on a memory-constrained Pi, so the ceiling comes  │
# │  before anything that opens the graph. The hash comes before the parse  │
# │  because a file that is not the file we were promised does not deserve  │
# │  the RAM.                                                               │
# └─────────────────────────────────────────────────────────────────────────┘
def check_policy(
    path: "os.PathLike[str] | str",
    *,
    expect_sha256: str | None = None,
    max_bytes: int = MAX_POLICY_BYTES,
) -> CheckResult:
    """Decide whether ``path`` is a policy this robot will run. Pure; touches no hardware.

    **This is the security boundary** (amendment A1), and its authority comes from where it
    runs rather than from how thorough it is. The robot's HTTP API has no authentication on
    any endpoint, and the logged learning ``starlette-cors-wildcard-reflects-origin``
    records that its CORS reflects the requesting origin with credentials -- so any webpage
    the owner visits can drive it. A check that lives only in Studio is decoration; Studio
    keeps a copy for fast feedback (story 2.5), this one is the gate.

    Two consequences worth stating because they are easy to erode later:

    * **There is no trusted caller.** No ``skip``, no ``internal=True``, no fast path. If a
      future caller "already checked", it calls this again -- the cost is one stat, one
      hash and one graph parse, none of which happen inside the walk loop.
    * **It runs on select as well as install** (story 2.3 wires both). A file that passed on
      install can have been swapped on disk since; re-verifying is the difference between
      checking a file and checking *the* file.

    What it does not do: predict whether the policy walks *well*. Nothing in this repo does
    (architecture Decision 1 dropped the simulation gate). This is a shape check and it
    says so.
    """
    try:
        size = os.stat(path).st_size
    except OSError as exc:
        return CheckResult(
            ok=False,
            code=POLICY_INSTALL_FAILED,
            detail=f"cannot read {path}: {exc.__class__.__name__}: {exc}",
        )

    if size > max_bytes:
        return CheckResult(
            ok=False,
            code=POLICY_CONTRACT_MISMATCH,
            detail=(
                f"policy is {size} bytes, over the {max_bytes}-byte ceiling; refused "
                "before parsing so a large file cannot exhaust this robot's memory "
                "during inspection"
            ),
        )

    if expect_sha256 is not None:
        expected = normalise_digest(expect_sha256)
        if expected is None:
            return CheckResult(
                ok=False,
                code=POLICY_INSTALL_FAILED,
                detail=(
                    f"install request carried {expect_sha256!r}, which is not a sha256 "
                    "digest (64 hex characters)"
                ),
            )
        try:
            actual = sha256_file(path)
        except OSError as exc:
            return CheckResult(
                ok=False,
                code=POLICY_INSTALL_FAILED,
                detail=f"read failed while hashing {path}: {exc.__class__.__name__}: {exc}",
            )
        if actual != expected:
            return CheckResult(
                ok=False,
                code=POLICY_INSTALL_FAILED,
                detail=f"sha256 mismatch: file is {actual}, request said {expected}",
            )

    try:
        import onnxruntime
    except ImportError as exc:
        return CheckResult(
            ok=False,
            code=POLICY_INSTALL_FAILED,
            detail=f"onnxruntime is not installed, so no policy can be verified: {exc}",
        )

    # Every failure mode of the parser -- not protobuf, truncated, a valid graph this build
    # cannot run -- becomes a refusal. Letting one propagate would turn a hostile upload
    # into a 500 from the robot, which is the failure mode F7 in the architecture's table.
    try:
        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        return CheckResult(
            ok=False,
            code=POLICY_CONTRACT_MISMATCH,
            detail=f"onnxruntime could not read the file: {exc.__class__.__name__}: {exc}",
        )

    reason = _signature_error(session)
    if reason is not None:
        return CheckResult(ok=False, code=POLICY_CONTRACT_MISMATCH, detail=reason)

    manifest = infer_manifest(session)
    manifest["size_bytes"] = size
    if expect_sha256 is not None:
        manifest["sha256"] = normalise_digest(expect_sha256)

    return CheckResult(ok=True, code=None, detail=describe(), manifest=manifest)
