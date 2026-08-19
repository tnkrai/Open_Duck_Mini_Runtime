"""The duck's observation contract, named and ordered.

Phase 1b of tnkr-studio's docs/designs/wired-physical-agents-plan.md.

WHAT THIS FIXES. The vector the walk policy eats is built by one `np.concatenate`
in `scripts/v2_rl_walk_mujoco.py`. Its order is carried entirely by position inside
that list, and nothing names the blocks. So the contract between a model and the loop
feeding it exists only as an implicit convention, and a model trained against a
different order loads happily and walks the duck into the floor. `WALK_OBS_SPEC` is
that order written down, in a form a component manifest can carry and a check can
compare.

WHY IT LIVES HERE AND NOT IN THE SCRIPT. That script imports the hardware stack at
module level (`rustypot_position_hwi`, `onnx_infer`), so CI cannot import it at all —
the duck's workflow installs numpy, pydantic, fastapi and pytest, with no onnxruntime
and no mujoco. A contract nothing can import is a contract nothing can test. This
module imports numpy and nothing else.

THERE IS NOW ONE PATH, AND IT IS THIS ONE (Phase 7). The hand-written `np.concatenate`
that used to live in `v2_rl_walk_mujoco.py` is deleted; the vector is derived from
`WALK_OBS_SPEC` by walking the list. That removes the class of bug where a swapped
policy meets a loop assembled for a different one — not by checking two paths agree,
but by there being one.

What the second path bought was evidence, and that is kept rather than lost:
`tests/fixtures/walk_observation_golden.json` is the frozen output of the deleted
concatenate, captured before it went. The generated path must still reproduce it value
for value. A golden file regenerated from the code it checks is worthless, which is why
that file says so in its own text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The duck's joint count. Everything joint-shaped below is this wide.
NUM_DOFS = 14

# The scale applied to joint velocities before they enter the vector. Named because a
# bare 0.05 inside a concatenate is the single easiest thing in this file to change by
# accident and the hardest to notice: the vector stays the right length and the policy
# just gets quietly worse.
DOF_VEL_SCALE = 0.05


@dataclass(frozen=True)
class ObsBlock:
    """One named, shaped slice of the observation, in its true position.

    Mirrors tnkr-studio's `ObsBlock` (server/trlc_studio/components/manifest.py) field
    for field, so a manifest built here round-trips into Studio's schema without a
    translation step that could reorder anything.
    """

    name: str
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


# The eleven blocks, in the order `np.concatenate` lays them down. Renaming an entry is
# safe; reordering one is a contract change and the equivalence test will say so.
WALK_OBS_SPEC: tuple[ObsBlock, ...] = (
    ObsBlock("gyro", (3,)),
    ObsBlock("accelerometer", (3,)),
    # 7 wide, not 3: forward, lateral, yaw, plus four head/neck channels.
    ObsBlock("commands", (7,)),
    # Stance-relative, not absolute. `init_pos` is the duck's own neutral stance, so
    # this block is "how far from standing", which is what the policy was trained on.
    ObsBlock("joint_positions_rel", (NUM_DOFS,)),
    ObsBlock("joint_velocities_scaled", (NUM_DOFS,)),
    # Three frames of action history, most recent first. The policy is not Markov in
    # the observation alone; take these away and it has no sense of its own momentum.
    ObsBlock("action_prev_1", (NUM_DOFS,)),
    ObsBlock("action_prev_2", (NUM_DOFS,)),
    ObsBlock("action_prev_3", (NUM_DOFS,)),
    ObsBlock("motor_targets", (NUM_DOFS,)),
    ObsBlock("feet_contacts", (2,)),
    # The imitation phase signal: where the duck is in its reference gait cycle.
    ObsBlock("imitation_phase", (2,)),
)

OBS_SIZE = sum(b.size for b in WALK_OBS_SPEC)


def build_observation(sources: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble the vector by walking `WALK_OBS_SPEC`, one named block at a time.

    The independent path. It never sees the concatenate above: it reads the spec, takes
    each block by NAME, and checks each one's width against the shape the spec declares.
    That independence is the entire point — two paths that shared a list would agree
    about a wrong order just as readily as a right one.
    """
    parts: list[np.ndarray] = []
    for block in WALK_OBS_SPEC:
        if block.name not in sources:
            raise KeyError(f"observation source missing block {block.name!r}")
        part = np.asarray(sources[block.name], dtype=float).reshape(-1)
        if part.size != block.size:
            raise ValueError(
                f"block {block.name!r} is {part.size} wide, spec says {block.size}"
            )
        parts.append(part)
    return np.concatenate(parts)


def sources_from_state(
    *,
    gyro,
    accelerometer,
    commands,
    dof_pos,
    init_pos,
    dof_vel,
    last_action,
    last_last_action,
    last_last_last_action,
    motor_targets,
    feet_contacts,
    imitation_phase,
) -> dict[str, np.ndarray]:
    """The loop's raw state, keyed by the spec's block names.

    This is where the two derived blocks are derived, and it is deliberately the ONLY
    place: `joint_positions_rel` subtracts the stance and `joint_velocities_scaled`
    applies DOF_VEL_SCALE. A caller that hands `build_observation` its own dict is free
    to do that arithmetic itself, which is what makes the equivalence test meaningful
    rather than circular.
    """
    return {
        "gyro": np.asarray(gyro, dtype=float),
        "accelerometer": np.asarray(accelerometer, dtype=float),
        "commands": np.asarray(commands, dtype=float),
        "joint_positions_rel": np.asarray(dof_pos, dtype=float)
        - np.asarray(init_pos, dtype=float),
        "joint_velocities_scaled": np.asarray(dof_vel, dtype=float) * DOF_VEL_SCALE,
        "action_prev_1": np.asarray(last_action, dtype=float),
        "action_prev_2": np.asarray(last_last_action, dtype=float),
        "action_prev_3": np.asarray(last_last_last_action, dtype=float),
        "motor_targets": np.asarray(motor_targets, dtype=float),
        "feet_contacts": np.asarray(feet_contacts, dtype=float),
        "imitation_phase": np.asarray(imitation_phase, dtype=float),
    }
