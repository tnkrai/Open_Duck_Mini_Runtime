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
# awd=True (v2_rl_walk_mujoco.py:67), which is what wraps the vector in a batch of one.
OBS_INPUT_NAME: str = "obs"

# Reject before parsing rather than after. A hostile or corrupt multi-hundred-MB file
# should never reach onnxruntime's parser on a memory-constrained Pi.
MAX_POLICY_BYTES: int = 256 * 1024 * 1024


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


def validate_obs(obs) -> None:
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
