"""The transcribed observation contract is not allowed to lie.

Phase 1b's "one non-obvious piece". `WALK_OBS_SPEC` is a hand-transcription of an
ordering that previously existed only as positions inside one `np.concatenate`, and a
hand-transcription is exactly the kind of work that goes subtly wrong. A wrong one is
worse than none: an authoritative-looking document asserting the wrong thing, which a
component manifest then publishes and a contract check then trusts.

So the document is pinned to the code. Build the vector both ways, assert they are
IDENTICAL elementwise. Not same-length — length equality is precisely the check a
reordering slips past, and a reordering is the failure that walks a duck into the
floor while every size assertion stays green.

ON "RECORDED FRAMES". The plan asks for this on recorded frames and this repo has
none; recording them needs a duck. Synthetic frames do the job here for a specific
reason rather than as a substitute: every element gets a DISTINCT value, so any
transposition of any two positions changes the output vector. Real frames would be
weaker at this, not stronger, since a real duck standing still produces repeated
zeros that a swap could hide inside. When recordings exist, point the same assertion
at them.
"""

from __future__ import annotations

import numpy as np
import pytest

from mini_bdx_runtime.obs_spec import (
    DOF_VEL_SCALE,
    NUM_DOFS,
    OBS_SIZE,
    WALK_OBS_SPEC,
    build_observation,
    concatenate_observation,
    sources_from_state,
)


def _frame(seed: int) -> dict:
    """One frame of loop state, every element distinct.

    A counter, not noise: `np.arange` off a per-frame offset means no two positions
    anywhere in the 101-wide vector share a value, so swapping any two blocks — or any
    two elements within a block — is guaranteed to change the result.
    """
    rng = iter(range(seed * 1000, seed * 1000 + 500))

    def take(n: int) -> np.ndarray:
        return np.array([float(next(rng)) for _ in range(n)])

    return {
        "gyro": take(3),
        "accelerometer": take(3),
        "commands": take(7),
        "dof_pos": take(NUM_DOFS),
        "init_pos": take(NUM_DOFS),
        "dof_vel": take(NUM_DOFS),
        "last_action": take(NUM_DOFS),
        "last_last_action": take(NUM_DOFS),
        "last_last_last_action": take(NUM_DOFS),
        "motor_targets": take(NUM_DOFS),
        "feet_contacts": take(2),
        "imitation_phase": take(2),
    }


@pytest.mark.parametrize("seed", [1, 2, 3, 17])
def test_the_spec_builds_exactly_what_the_policy_eats(seed):
    frame = _frame(seed)
    handwritten = concatenate_observation(**frame)
    from_spec = build_observation(sources_from_state(**frame))

    assert from_spec.shape == handwritten.shape
    # identical, not close: these are the same arithmetic on the same floats, so any
    # difference at all is a difference in ORDER, and a tolerance would hide it
    assert np.array_equal(from_spec, handwritten), (
        "the spec and the concatenate disagree; one of them was edited without the "
        "other. Positions that differ: "
        f"{np.nonzero(from_spec != handwritten)[0].tolist()}"
    )


def test_the_spec_is_the_width_the_policy_expects():
    assert OBS_SIZE == 101
    assert sum(b.size for b in WALK_OBS_SPEC) == len(concatenate_observation(**_frame(1)))


def test_a_reordered_spec_is_actually_caught():
    """The test above is only worth having if it can fail. This proves it can.

    Swap two adjacent same-width blocks — the case a size check cannot see, and the
    single most likely transcription error, since three consecutive 14-wide action
    history frames are the easiest thing in the list to get backwards.
    """
    frame = _frame(5)
    sources = sources_from_state(**frame)
    swapped = dict(sources)
    swapped["action_prev_1"], swapped["action_prev_2"] = (
        sources["action_prev_2"],
        sources["action_prev_1"],
    )
    assert len(build_observation(swapped)) == OBS_SIZE  # same length...
    assert not np.array_equal(  # ...and still caught
        build_observation(swapped), concatenate_observation(**frame)
    )


def test_the_velocity_scale_is_applied_and_named():
    """0.05 inside a concatenate is the easiest constant here to change by accident and
    the hardest to notice: the vector stays the right width and the policy just gets
    quietly worse."""
    frame = _frame(9)
    sources = sources_from_state(**frame)
    assert np.array_equal(
        sources["joint_velocities_scaled"], frame["dof_vel"] * DOF_VEL_SCALE
    )
    assert DOF_VEL_SCALE == 0.05


def test_joint_positions_are_stance_relative_not_absolute():
    frame = _frame(11)
    sources = sources_from_state(**frame)
    assert np.array_equal(
        sources["joint_positions_rel"], frame["dof_pos"] - frame["init_pos"]
    )


def test_a_missing_block_refuses_rather_than_building_a_short_vector():
    frame = _frame(13)
    sources = sources_from_state(**frame)
    del sources["imitation_phase"]
    with pytest.raises(KeyError, match="imitation_phase"):
        build_observation(sources)


def test_a_wrong_width_block_refuses_rather_than_shifting_everything_after_it():
    frame = _frame(15)
    sources = sources_from_state(**frame)
    sources["commands"] = np.zeros(3)  # the 3-vs-7 mistake this spec exists to prevent
    with pytest.raises(ValueError, match="commands"):
        build_observation(sources)
