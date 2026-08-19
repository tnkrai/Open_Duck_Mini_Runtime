"""The generated observation vector is the one the policy was always fed.

Phase 7. The hand-written `np.concatenate` in `v2_rl_walk_mujoco.py` is gone; the
vector is derived from `WALK_OBS_SPEC` by walking the list. That removes the class of
bug where a swapped policy meets a loop assembled for a different one — not by checking
that two paths agree, but by there being one.

WHAT THE SECOND PATH BOUGHT WAS EVIDENCE, AND THIS FILE IS HOW IT IS KEPT. Phase 1b's
test built the vector both ways and asserted they were identical. With one path left
that test would compare the code to itself, so the assertion is inverted:
`fixtures/walk_observation_golden.json` holds the frozen output of the deleted
concatenate, captured immediately before it was removed, and the generated path must
still reproduce it value for value.

A golden file regenerated from the code it checks is worthless. That fixture says so in
its own text, and nothing in this repo writes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mini_bdx_runtime.obs_spec import (
    DOF_VEL_SCALE,
    NUM_DOFS,
    OBS_SIZE,
    WALK_OBS_SPEC,
    build_observation,
    sources_from_state,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "walk_observation_golden.json").read_text()
)


# --- the vector the policy was always fed -----------------------------------

@pytest.mark.parametrize("case", GOLDEN["frames"], ids=lambda c: f"seed{c['seed']}")
def test_the_generated_vector_matches_what_the_old_path_produced(case):
    """The whole phase in one assertion. If this fails, a real duck's policy is being
    fed something different from what it was fed before the change."""
    sources = {k: np.array(v) for k, v in case["sources"].items()}
    built = build_observation(sources_from_state(**sources))

    assert built.shape == (len(case["expected"]),)
    assert np.array_equal(built, np.array(case["expected"])), (
        "the generated vector no longer matches what the hand-written concatenate "
        "produced. Positions that differ: "
        f"{np.nonzero(built != np.array(case['expected']))[0].tolist()}"
    )


def test_the_golden_was_not_generated_from_the_code_it_checks():
    """A golden regenerated from its subject asserts that the code equals itself.

    Pinned as a test rather than a comment because the tempting fix, when this file
    fails, is to regenerate the fixture — which makes the failure disappear and takes
    the evidence with it.
    """
    assert "concatenate" in GOLDEN["_what"]
    assert "_do_not_regenerate" in GOLDEN
    assert GOLDEN["frames"], "the golden is empty"


def test_the_golden_covers_the_whole_vector():
    for case in GOLDEN["frames"]:
        assert len(case["expected"]) == OBS_SIZE == 101


def test_no_two_blocks_in_the_golden_are_identical():
    """The property that makes a BLOCK swap detectable, stated accurately.

    An earlier version of this asserted every element in the vector was distinct, which
    is false and taught me something worth writing down: `joint_positions_rel` is
    `dof_pos - init_pos`, and the counter frames make both consecutive integers 14
    apart, so that whole block collapses to one repeated value.

    The consequence is a real limit on these frames, not a cosmetic one: a reordering
    WITHIN the stance-relative block would not change the vector and would not be
    caught here. Block-level swaps are caught, which is the failure the contract exists
    to prevent, and the narrower one is recorded rather than papered over. Recorded
    frames from a real duck would close it.
    """
    from mini_bdx_runtime.obs_spec import WALK_OBS_SPEC

    for case in GOLDEN["frames"]:
        values = case["expected"]
        blocks, at = [], 0
        for b in WALK_OBS_SPEC:
            blocks.append(tuple(values[at : at + b.size]))
            at += b.size
        same_width = {}
        for block, spec in zip(blocks, WALK_OBS_SPEC):
            same_width.setdefault(spec.size, []).append(block)
        for width, group in same_width.items():
            assert len(set(group)) == len(group), (
                f"seed {case['seed']}: two {width}-wide blocks are identical, so "
                "swapping them would not change the vector"
            )


# --- the spec still describes the thing it builds ---------------------------

def test_the_spec_and_the_vector_agree_on_width():
    assert sum(b.size for b in WALK_OBS_SPEC) == OBS_SIZE


def test_a_reordered_spec_would_break_the_golden():
    """Proof the golden can fail. Swap two adjacent same-width blocks — the case a
    width check cannot see — and the vector changes."""
    case = GOLDEN["frames"][0]
    sources = sources_from_state(**{k: np.array(v) for k, v in case["sources"].items()})
    swapped = dict(sources)
    swapped["action_prev_1"], swapped["action_prev_2"] = (
        sources["action_prev_2"],
        sources["action_prev_1"],
    )
    built = build_observation(swapped)
    assert len(built) == len(case["expected"])  # same width...
    assert not np.array_equal(built, np.array(case["expected"]))  # ...and caught


def test_the_velocity_scale_is_still_applied():
    case = GOLDEN["frames"][0]
    sources = sources_from_state(**{k: np.array(v) for k, v in case["sources"].items()})
    assert np.array_equal(
        sources["joint_velocities_scaled"], np.array(case["sources"]["dof_vel"]) * DOF_VEL_SCALE
    )
    assert DOF_VEL_SCALE == 0.05


def test_joint_positions_are_still_stance_relative():
    case = GOLDEN["frames"][0]
    sources = sources_from_state(**{k: np.array(v) for k, v in case["sources"].items()})
    assert np.array_equal(
        sources["joint_positions_rel"],
        np.array(case["sources"]["dof_pos"]) - np.array(case["sources"]["init_pos"]),
    )


def test_a_missing_block_refuses_rather_than_building_a_short_vector():
    case = GOLDEN["frames"][0]
    sources = sources_from_state(**{k: np.array(v) for k, v in case["sources"].items()})
    del sources["imitation_phase"]
    with pytest.raises(KeyError, match="imitation_phase"):
        build_observation(sources)


def test_a_wrong_width_block_refuses_rather_than_shifting_everything_after_it():
    case = GOLDEN["frames"][0]
    sources = sources_from_state(**{k: np.array(v) for k, v in case["sources"].items()})
    sources["commands"] = np.zeros(3)
    with pytest.raises(ValueError, match="commands"):
        build_observation(sources)


def test_there_is_no_second_path_left_to_drift_from():
    """Phase 7's actual deliverable. If a hand-written builder reappears, the class of
    bug this phase removed comes back with it."""
    import mini_bdx_runtime.obs_spec as mod

    assert not hasattr(mod, "concatenate_observation")
    source = Path(mod.__file__).read_text()
    assert "def concatenate_observation" not in source


def test_the_walk_script_uses_the_generated_path():
    script = (Path(__file__).parents[1] / "scripts" / "v2_rl_walk_mujoco.py").read_text()
    assert "build_observation(" in script
    assert "concatenate_observation(" not in script
    # and it no longer assembles a vector of its own
    assert "np.concatenate(" not in script
