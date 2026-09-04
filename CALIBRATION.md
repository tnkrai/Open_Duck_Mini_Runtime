# Calibration and setup API

`scripts/tnkr_server.py` is the agent Tnkr Studio talks to. This document describes the
setup steps it serves, in the order Studio runs them, and the keys they write into
`~/duck_config.json`. Every route answers JSON and reports failures as
`{"detail": {"code": ..., "message": ...}}` with the joint name where one applies.

The servos are driven through `mini_bdx_runtime/rustypot_position_hwi.py` (the HWI).
On the way to a servo a joint's target becomes `raw = sign * position + offset`; on the
way back a reading becomes `position = sign * (raw - offset)`. The walk, the setup
steps and the live view all go through the same HWI, so what the steps below write is
what the walk uses.

## Keys in `duck_config.json`

| Key | Written by | Meaning |
| --- | --- | --- |
| `joints_offsets` | `/api/calibration/save`, the walk's live trim | Each joint's zero, radians, in raw servo space. Measured with the joint posed straight, so it survives a later sign flip. |
| `joints_signs` | `/api/directions/save` | `+1` or `-1` per joint. A servo mounted to turn the opposite way to the model gets `-1`. Absent means `+1` everywhere. |
| `servo_ids` | `/api/identify/finish`, `/api/calibration/swap-*` | The joints whose servo id differs from the runtime's table (`DEFAULT_SERVO_IDS` in the HWI), by name. Applied in place, so the joint order the walking policy indexes by never changes. |
| `swapped_pairs`, `legs_swapped` | older versions of the leg check | Whole left/right pair swaps. Still read, and applied before `servo_ids`; every writer now folds them into `servo_ids` and clears them. |

`POST /api/config` never takes `joints_signs`, `servo_ids`, `swapped_pairs` or
`legs_swapped`: each has its own route, and a screen that sends the config back whole
can only carry an older copy of them. `joints_offsets` and `expression_features` are
merged one key at a time, so a single trim or tick replaces nothing else. The servos
keep holding through a config save.

## 1. Which servo is which: `/api/identify/*`

A build can program servo ids onto the wrong servos. Two real ducks did: one had the
hip yaws crossed left to right, another had one leg's hip yaw and hip roll on each
other's servos. Rather than ask the operator to read a leg off a picture, the duck goes
loose, the operator moves one named joint on the real duck, and the bus is watched for
the servo that moved. Whichever moved is that joint.

| Route | Body | Does |
| --- | --- | --- |
| `POST /api/identify/start` | | Torque off on every joint, in place. Ends any calibration session. Returns `joints` and `servoIds`. |
| `POST /api/identify/watch` | `{"jointName"}` | Starts a sampler that reads every joint at 20 Hz and tracks each joint's travel. |
| `GET /api/identify/status` | | `done` once a joint has travelled at least 0.12 rad and everything has then been still (under 0.02 rad per sample) for 0.6 s, or after 25 s. Reports `moved` (`servoId`, `name`, `range`), a `runnerUp` if a second joint also moved, and `ranges`. |
| `POST /api/identify/assign` | `{"jointName", "servoId"}` | Records the answer. Answers are kept as a permutation: a joint that already held that servo takes the named joint's old id. |
| `POST /api/identify/finish` | `{"save": true}` | Writes `servo_ids`, rebuilds the HWI on it and powers every joint back on where it is. With `save: false` it only powers the duck back on. |

## 2. Joint zeros: `/api/calibration/*`

The duck is placed in a comfortable pose and `POST /api/calibration/start` holds it
there (goal first, then torque, kp 20, kd 0): nothing drives to servo zero. The
operator then frees one joint at a time (`begin-joint`), poses it straight by hand,
and the offset is the reading at straight. `accept` records a joint's offset,
`finish` re-powers every joint where it is, and `save` writes `joints_offsets`.

Helpers added on this branch, all of which keep the session open:

| Route | Body | Does |
| --- | --- | --- |
| `POST /api/calibration/wiggle` | `{"jointName"}` | Rocks a held joint by 0.08 rad and back, for the operator to see which joint a name reaches. |
| `POST /api/calibration/swap-ids` | `{"jointName", "otherJointName"}` | The released joint went loose somewhere else: swaps the two joints' servo ids on the duck, drops those two joints' offsets, rebuilds the HWI and holds every joint where it is. |
| `POST /api/calibration/swap-pair` | `{"jointName"}` | The left/right case of `swap-ids`: swaps the joint with its twin. |
| `POST /api/calibration/swap-legs` | `{"swapped"}` | Every leg pair crossed, or uncrossed. Ends the session; the caller arms again. |

`start` also reports `servoIds`, `swappedPairs` and `legsSwapped`, and `mode: "hold"`
so a client can refuse an older agent that drove to zero on arm.

## 3. Joint directions: `/api/directions/*`

A mirrored servo passes every position read and only shows itself when the joint
moves. The duck stands straight at low stiffness (kp 8, so a wrong-way joint meeting
its shell stalls softly), each left/right pair moves a little in the model's own
direction, and the operator says whether the real duck moved like the picture.

| Pair | Targets (left, right), radians | Expected motion |
| --- | --- | --- |
| `hip_pitch` | -0.3, +0.3 | Both thighs swing backward, feet toward the tail. |
| `knee` | +0.5, +0.5 | Both feet swing forward and the knees poke backward. |
| `ankle` | -0.4, -0.4 | Both feet tip toes-down, heels up. |

| Route | Body | Does |
| --- | --- | --- |
| `POST /api/directions/start` | | Straight at kp 8. Returns the pairs and every joint's current sign. |
| `POST /api/directions/move` / `rest` | `{"pairId"}` | Both joints of the pair to their targets, or back to straight. |
| `POST /api/directions/flip` | `{"jointName"}` | Inverts one joint's sign, live. Nothing moves until the next move. |
| `POST /api/directions/crouch` | | Every pair joint into the walking crouch, `init_pos`, with the signs as they stand. Returns the targets. This is the pose the walk sends first, shown whole so the operator confirms it before the walk does. |
| `POST /api/directions/stand` | | Every pair joint back to straight. |
| `POST /api/directions/save` | | Writes `joints_signs` for every joint, rests whatever is displaced, frees the bus with torque kept. |
| `POST /api/directions/finish` | | Ends the session without saving; unsaved flips are forgotten. |

A servo's own encoder cannot detect a wrong direction: it reports where the horn is,
not where the leg went. The crouch at the end is the one check that needs a person,
and it is one look.

## Rules every route follows

- Goal first, torque second, per joint. A servo that wakes up already at its target
  stiffens in place; enabling torque against a stale goal is a drive.
- No `turn_on()` ramps inside a session: the ramp passes through kp 2 and drives every
  joint through `init_pos` on the way.
- A save frees the serial port with torque kept; the next HWI is built from the file
  and carries what was saved.
- Nothing is written until the operator saves. Leaving a screen ends its session and
  re-powers the duck where it is.

Tests: `tests/test_joint_calibration.py` and `tests/test_joint_directions.py` run
against a fake bus and cover every route above.
