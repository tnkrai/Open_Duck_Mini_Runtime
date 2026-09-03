"""Per-joint soft-offset calibration (/api/calibration/*).

The session HOLDS the pose the duck was placed in and never commands a joint to
servo-zero; each offset is the joint's raw reading at the straight pose. The two
tests at the bottom pin that, because the failure it replaces was a real one: driving
to zero_pos to arm sent mis-seated joints — the joints calibration exists for — into
their own shells.

Behaviours this flow got wrong, each with a test here:

  1. The offset was measured against a hardcoded 0.0 while the joint was ALSO being
     held at a commanded zero under kd 0, so servo droop landed in every saved offset.
     (Moot under hold-pose: nothing is held at a commanded zero, and the joint is read
     while the operator's hands hold it, so there is no droop to cancel.)
  2. `get_present_positions()` reads all fourteen and returns None if ANY fails, so a
     silent joint 3 failed a calibration of joint 7 and nothing could name the joint
     that was actually quiet.
  3. apply-offset enabled torque BEFORE writing the goal, so the servo drove back to
     the uncalibrated zero — yanking the joint out of the operator's hand — before the
     new goal arrived.
  4. save called release_hwi(), whose disable_torque default is True, so the duck went
     limp the instant the operator saved.
  5. begin-joint zeroed a redone joint's offset and then re-asserted a hold entry
     written in the old offset space, so the re-assert drove that joint to servo-zero.
  6. finish enabled torque against stale goal registers, and left the HWI singleton
     carrying the session's zeroed offsets for the next route to drive through.
"""

import pytest

import tnkr_server


JOINTS = {"left_knee": 23, "right_knee": 13, "head_yaw": 32}


class FakeHWI:
    """Records what the endpoints ask of the bus, one joint at a time."""

    def __init__(self, readings=None, torque_fails=None, read_fails=None):
        self.joints = dict(JOINTS)
        self.joints_offsets = {n: 0.0 for n in JOINTS}
        self.joints_signs = {n: 1 for n in JOINTS}
        self.zero_pos = {n: 0.0 for n in JOINTS}
        self.init_pos = dict(self.zero_pos)
        # joint -> present position the bus will report
        self.readings = readings or {n: 0.0 for n in JOINTS}
        self.torque_fails = torque_fails or set()
        self.read_fails = read_fails or set()
        self.torque_calls = []          # (joint, enabled)
        self.position_writes = 0
        # every goal as the servo receives it, position + offset: RAW, per write
        self.raw_goals = []
        self.single_writes = []  # (joint, raw goal) from set_position, in order
        self.turned_off = False
        self.turned_on = False
        self.kds = None
        self.kps = None

    # -- the vector read, used only by /start --
    def get_present_positions(self):
        if self.read_fails:
            return None
        return [self._model_space(n) for n in self.joints]

    def _model_space(self, joint_name):
        # what the real HWI hands back: sign * (raw - offset)
        sign = self.joints_signs.get(joint_name, 1)
        return sign * (self.readings[joint_name] - self.joints_offsets.get(joint_name, 0.0))

    # -- the single-joint primitives the flow runs on --
    def get_present_position(self, joint_name):
        if joint_name in self.read_fails:
            raise OSError(f"read_present_position failed for '{joint_name}' (id ?)")
        return self._model_space(joint_name)

    def set_joint_torque(self, joint_name, enabled):
        if joint_name in self.torque_fails:
            raise OSError(f"disable_torque failed for '{joint_name}' (id ?)")
        self.torque_calls.append((joint_name, enabled))

    def set_kds(self, kds):
        self.kds = kds

    def set_kps(self, kps):
        self.kps = kps

    def turn_on(self):
        self.turned_on = True

    def turn_off(self):
        self.turned_off = True

    def set_position_all(self, positions):
        self.position_writes += 1
        self.raw_goals.append(
            {
                j: self.joints_signs.get(j, 1) * p + self.joints_offsets.get(j, 0.0)
                for j, p in positions.items()
            }
        )

    def set_position(self, joint_name, pos):
        # one joint, as the wiggle writes it; recorded in RAW terms like the rest
        raw = self.joints_signs.get(joint_name, 1) * pos + self.joints_offsets.get(joint_name, 0.0)
        self.single_writes.append((joint_name, raw))

    def close(self):
        pass


@pytest.fixture
def hwi(monkeypatch, tmp_path):
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", str(tmp_path / "duck_config.json"))
    monkeypatch.setattr(tnkr_server, "calibration_offsets", {})
    monkeypatch.setattr(tnkr_server, "calibration_baselines", {})
    monkeypatch.setattr(tnkr_server, "calibration_hold", {})

    fake = FakeHWI()

    def install(**kwargs):
        for k, v in kwargs.items():
            setattr(fake, k, v)
        monkeypatch.setattr(tnkr_server, "hwi_instance", fake)
        monkeypatch.setattr(tnkr_server, "get_hwi", lambda: fake)
        return fake

    install()
    return fake


def test_the_offset_is_the_reading_at_the_straight_pose(client, hwi):
    """The offset is measured against STRAIGHT, not against wherever the duck sits.

    The joint is placed at 0.02, released, and posed to straight where it reads 0.17.
    0.17 IS the mounting error, and the offset. Differencing against the placed pose
    would bake that pose into every saved offset — a knee calibrated in a crouch would
    carry the crouch into the walk.
    """
    hwi.readings["left_knee"] = 0.02  # wherever the operator placed it
    r = client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    assert r.status_code == 200
    assert r.json()["baseline"] == 0.0

    hwi.readings["left_knee"] = 0.17  # posed to straight
    r = client.post("/api/calibration/confirm-position", json={"jointName": "left_knee"})
    assert r.status_code == 200
    body = r.json()
    assert body["offset"] == pytest.approx(0.17)
    assert body["previousPosition"] == 0.0
    assert body["newPosition"] == pytest.approx(0.17)


def test_a_mirrored_joint_saves_its_raw_reading_as_the_offset(client, hwi):
    """joints_signs flips reads and writes at the HWI boundary, but the offset lives in
    raw servo space (the HWI adds it AFTER the sign). A mirrored joint posed straight
    reads -raw; saving that would hold it at the mirror of where the hands are."""
    hwi.joints_signs["left_knee"] = -1
    hwi.readings["left_knee"] = 0.2  # the raw servo angle at straight
    client.post("/api/calibration/start")
    client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    r = client.post("/api/calibration/confirm-position", json={"jointName": "left_knee"})
    assert r.status_code == 200
    assert r.json()["newPosition"] == pytest.approx(-0.2)  # what the model sees
    assert r.json()["offset"] == pytest.approx(0.2)  # what the servo needs

    client.post(
        "/api/calibration/apply-offset", json={"jointName": "left_knee", "offset": 0.2}
    )
    # held exactly where the hands are, not at their mirror
    assert hwi.raw_goals[-1]["left_knee"] == pytest.approx(0.2)


def test_start_holds_the_placed_pose_and_never_commands_zero(client, hwi):
    """The whole reason this flow changed.

    Arming used to drive every joint to servo-zero. On a mis-seated horn servo-zero is
    mechanically far from straight, so that drove exactly the joints calibration exists
    for into their own shells. Every goal written here is the joint's own present
    position: nothing moves.
    """
    hwi.readings = {"left_knee": 1.3, "right_knee": -0.9, "head_yaw": 0.2}
    commanded = []
    hwi.set_position_all = lambda positions: commanded.append(dict(positions))

    r = client.post("/api/calibration/start")
    assert r.status_code == 200
    assert r.json()["currentPositions"] == pytest.approx(hwi.readings)
    assert commanded == [pytest.approx(hwi.readings)]


def test_apply_holds_this_joint_straight_and_leaves_the_others_where_they_were(client, hwi):
    """The other half of the promise. Applying an offset must not be a back door to the
    drive this flow removed: the calibrated joint flips to straight, and the twelve the
    operator has not reached yet stay exactly where the duck was placed."""
    hwi.readings = {"left_knee": 1.3, "right_knee": -0.9, "head_yaw": 0.2}
    client.post("/api/calibration/start")

    commanded = []
    hwi.set_position_all = lambda positions: commanded.append(dict(positions))
    r = client.post(
        "/api/calibration/apply-offset", json={"jointName": "left_knee", "offset": 0.15}
    )
    assert r.status_code == 200

    held = commanded[-1]
    assert held["left_knee"] == 0.0  # goal 0 + offset 0.15 = the posed position
    assert held["right_knee"] == pytest.approx(-0.9)
    assert held["head_yaw"] == pytest.approx(0.2)


def test_start_writes_every_goal_before_powering_and_never_ramps(client, hwi):
    """turn_on() would have ramped every joint through kp 2 for a second — a loaded
    knee sags under the body and is hauled back when the real kp lands — and driven
    to hwi.init_pos on the way. Gains are written directly at the hold stiffness the
    operator's own script was proven with, the goals land, and only then does each
    joint get its explicit enable."""
    hwi.readings = {"left_knee": 1.3, "right_knee": -0.9, "head_yaw": 0.2}
    stance = dict(hwi.init_pos)
    order = []
    hwi.set_position_all = lambda positions: order.append("goal")
    hwi.set_joint_torque = lambda joint_name, enabled: order.append(
        f"torque:{joint_name}:{enabled}"
    )

    r = client.post("/api/calibration/start")
    assert r.status_code == 200
    assert r.json()["mode"] == "hold"
    assert order == ["goal"] + [f"torque:{j}:True" for j in JOINTS]
    assert hwi.turned_on is False
    assert hwi.kps == [tnkr_server.CALIBRATION_HOLD_KP] * len(JOINTS)
    assert hwi.kds == [0] * len(JOINTS)
    assert hwi.init_pos == stance  # the walking stance is not the session's to edit


def test_begin_joint_reads_only_the_joint_it_is_releasing(client, hwi):
    """The attribution fix. A silent joint elsewhere on the bus must not fail the
    joint being calibrated — that conflation is why 'this joint did not answer' was
    an unprovable claim."""
    hwi.read_fails = {"right_knee"}
    r = client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    assert r.status_code == 200
    assert ("left_knee", False) in hwi.torque_calls


def test_a_silent_joint_names_itself_and_maps_to_MOTORS_SILENT(client, hwi):
    hwi.read_fails = {"left_knee"}
    r = client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    assert r.status_code == 502
    detail = r.json()["detail"]
    # A real field, not a `CODE:` prefix on a log line. Studio maps status codes, and
    # one status covers several faults, so the agent has to name which — but naming it
    # by string prefix fails silently on a typo, and this is what replaced that.
    assert detail["code"] == "MOTORS_SILENT"
    assert detail["joint"] == "left_knee"
    assert "left_knee" in detail["message"]
    # nothing was released: the operator is not sent to a joint we never touched
    assert hwi.torque_calls == []


def test_a_joint_that_will_not_go_limp_says_so(client, hwi):
    hwi.torque_fails = {"left_knee"}
    r = client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "TORQUE_RELEASE_FAILED"
    assert r.json()["detail"]["joint"] == "left_knee"


def test_begin_joint_resets_this_joints_offset_so_redos_do_not_compound(client, hwi):
    """Antoine's retry branch zeroes the offset; without it, re-doing an accepted
    joint measures against its own previous correction."""
    hwi.joints_offsets["left_knee"] = 0.15
    hwi.readings["left_knee"] = 0.02
    client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    assert hwi.joints_offsets["left_knee"] == 0
    assert tnkr_server.calibration_baselines["left_knee"] == 0.0
    # and its hold entry is re-read in the new (offset 0) space: the raw position
    assert tnkr_server.calibration_hold["left_knee"] == pytest.approx(0.02)


def test_redoing_a_calibrated_joint_holds_it_where_it_is(client, hwi):
    """Found by review before it reached a robot. After apply-offset a joint's hold
    entry is 0 in a space where its offset carries the whole correction. begin-joint
    zeroes that offset for the redo, and re-asserting the entry as it stood would
    write raw 0 — servo-zero, the mis-seated horn's shell — to a powered joint at full
    stiffness. The entry has to be re-read in the new space first."""
    hwi.readings = {"left_knee": 1.3, "right_knee": -0.9, "head_yaw": 0.2}
    client.post("/api/calibration/start")
    client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    hwi.readings["left_knee"] = 0.5  # posed straight: the horn is 0.5 rad off
    m = client.post("/api/calibration/confirm-position", json={"jointName": "left_knee"}).json()
    client.post(
        "/api/calibration/apply-offset", json={"jointName": "left_knee", "offset": m["offset"]}
    )
    assert hwi.raw_goals[-1]["left_knee"] == pytest.approx(0.5)  # held straight

    r = client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    assert r.status_code == 200
    goals = hwi.raw_goals[-1]
    assert goals["left_knee"] == pytest.approx(0.5)  # not 0.0
    assert goals["right_knee"] == pytest.approx(-0.9)
    assert goals["head_yaw"] == pytest.approx(0.2)


def test_confirm_without_begin_is_a_state_error_not_a_wrong_number(client, hwi):
    r = client.post("/api/calibration/confirm-position", json={"jointName": "left_knee"})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "INVALID_STATE"


def test_apply_offset_writes_the_goal_before_re_powering(client, hwi):
    """The yank fix. At this moment the servo's goal register still holds the raw 0
    from begin-joint, so torque-first drives the joint back to the uncalibrated pose
    with a hand on it. The goal write must land first."""
    order = []
    hwi.set_position_all = lambda positions: order.append("goal")
    hwi.set_joint_torque = lambda joint_name, enabled: order.append(
        f"torque:{joint_name}:{enabled}"
    )
    r = client.post(
        "/api/calibration/apply-offset", json={"jointName": "left_knee", "offset": 0.15}
    )
    assert r.status_code == 200
    assert order == ["goal", "torque:left_knee:True"]
    assert hwi.joints_offsets["left_knee"] == pytest.approx(0.15)


def test_unknown_joint_is_rejected_before_anything_touches_the_bus(client, hwi):
    r = client.post("/api/calibration/begin-joint", json={"jointName": "elbow"})
    assert r.status_code == 400
    assert hwi.torque_calls == []
    # Its own code. A bare 400 is not in Studio's status map, so it fell through to the
    # catch-all and reported the duck as unreachable — for a duck that had just
    # answered, about a joint it does not have.
    assert r.json()["detail"]["code"] == "UNKNOWN_JOINT"


def test_failing_to_re_power_is_its_own_code(client, hwi):
    """Not MOTORS_SILENT. Re-powering hands the joint's weight back to the servo, so
    failing it leaves the joint LIMP — the opposite state from a failed release, and
    the opposite thing to tell someone deciding whether to let go."""
    hwi.torque_fails = {"left_knee"}
    r = client.post(
        "/api/calibration/apply-offset", json={"jointName": "left_knee", "offset": 0.15}
    )
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "TORQUE_ENABLE_FAILED"
    assert r.json()["detail"]["joint"] == "left_knee"


def test_start_reports_a_silent_bus_instead_of_an_empty_reading_set(client, hwi):
    """The whole-bus read belongs at arm, and its failure must be reported: the old
    code returned an empty currentPositions dict and let the session continue with no
    baselines at all."""
    hwi.read_fails = {"right_knee"}
    r = client.post("/api/calibration/start")
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "MOTORS_SILENT"
    # no joint key: the whole-bus read cannot say which of the fourteen was quiet, and
    # inventing one would be the false attribution this flow exists to avoid
    assert "joint" not in r.json()["detail"]


def test_start_zeroes_every_offset_and_the_session_state(client, hwi):
    hwi.joints_offsets["left_knee"] = 0.4
    tnkr_server.calibration_offsets["head_yaw"] = 0.1
    tnkr_server.calibration_baselines["head_yaw"] = 0.1
    r = client.post("/api/calibration/start")
    assert r.status_code == 200
    assert set(r.json()["joints"]) == set(JOINTS)
    assert hwi.joints_offsets["left_knee"] == 0
    assert tnkr_server.calibration_offsets == {}
    assert tnkr_server.calibration_baselines == {}
    # kd 0 so a released joint is not fought on the way back, matching the script
    assert hwi.kds == [0, 0, 0]


def test_save_on_a_duck_with_no_config_yet_creates_one(client, hwi, tmp_path):
    """_read_config raises FileNotFoundError on a fresh duck, and that used to become a
    500 — so fourteen joints of work failed to save on exactly the robot most likely to
    be calibrating for the first time."""
    assert not (tmp_path / "duck_config.json").exists()
    client.post("/api/calibration/accept", json={"jointName": "head_yaw", "offset": -0.03})
    r = client.post("/api/calibration/save")
    assert r.status_code == 200
    assert r.json()["offsets"] == {"head_yaw": pytest.approx(-0.03)}


def test_save_merges_offsets_and_keeps_the_duck_standing(client, hwi, tmp_path):
    """Fourteen joints of work, and the duck is standing at the tall straight-leg
    zero pose. release_hwi()'s disable_torque default is True, which dropped it."""
    import json

    (tmp_path / "duck_config.json").write_text(
        json.dumps({"start_paused": True, "joints_offsets": {"right_knee": 0.4}})
    )
    client.post("/api/calibration/accept", json={"jointName": "left_knee", "offset": 0.15})
    r = client.post("/api/calibration/save")
    assert r.status_code == 200
    offsets = r.json()["offsets"]
    # merged, not replaced: a joint skipped this session keeps the offset it had
    assert offsets["left_knee"] == pytest.approx(0.15)
    assert offsets["right_knee"] == pytest.approx(0.4)
    assert hwi.turned_off is False
    assert tnkr_server.hwi_instance is None  # bus freed, torque kept

    saved = json.loads((tmp_path / "duck_config.json").read_text())
    assert saved["joints_offsets"]["left_knee"] == pytest.approx(0.15)
    # and nothing else in the file was disturbed
    assert saved["start_paused"] is True


def test_calibration_refuses_while_a_walk_owns_the_bus(client, hwi, monkeypatch):
    """Without this the walk holds the serial port, get_hwi() fails, and the operator
    is told 'we couldn't detect your servo motors' for servos that are fine and busy."""
    monkeypatch.setattr(tnkr_server, "is_walking", lambda: True)
    for path, body in [
        ("/api/calibration/start", None),
        ("/api/calibration/begin-joint", {"jointName": "left_knee"}),
        ("/api/calibration/confirm-position", {"jointName": "left_knee"}),
        ("/api/calibration/apply-offset", {"jointName": "left_knee", "offset": 0.1}),
    ]:
        r = client.post(path, json=body) if body else client.post(path)
        assert r.status_code == 409, path


# ── telemetry ────────────────────────────────────────────────────────────────
# Fleet questions the per-request events cannot answer on their own: did the session
# finish, which joints were skipped, how many attempts did it cost, and how close does
# the result sit to the servo's +-180 command seam.


def _event(captured, name):
    hits = [e for e in captured if e["event"] == name]
    assert hits, f"{name} not captured; got {[e['event'] for e in captured]}"
    return hits[-1]["properties"]


def test_start_reports_the_pose_the_duck_was_placed_in(client, hwi, captured):
    """Was droop-at-zero, which no longer exists: nothing is commanded to zero, so
    there is no sag against a commanded zero to measure. What arming can report now is
    the pose operators actually leave the duck in, which is what says whether the
    on-screen straight reference matches the robot in front of them."""
    hwi.readings = {"left_knee": 1.3, "right_knee": -0.03, "head_yaw": 0.0}
    client.post("/api/calibration/start")
    props = _event(captured, "joint_calibration_started")
    assert props["joint_count"] == 3
    assert props["placed_pose_rad"]["right_knee"] == pytest.approx(-0.03)
    # the joint furthest from straight: how much hand-posing this session will cost
    assert props["max_abs_placed_rad"] == pytest.approx(1.3)


def test_measuring_reports_the_offset_and_its_seam_headroom(client, hwi, captured):
    # knee init_pos 1.379 rad = 79.0 deg, so a 0.5 rad (28.6 deg) offset leaves
    # 180 - 107.6 = 72.4 deg of the command window for the policy.
    hwi.init_pos = {"left_knee": 1.379, "right_knee": 1.379, "head_yaw": 0.0}
    client.post("/api/calibration/start")
    hwi.readings["left_knee"] = 0.0
    client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    hwi.readings["left_knee"] = 0.5
    client.post("/api/calibration/confirm-position", json={"jointName": "left_knee"})

    props = _event(captured, "api_request_completed")
    assert props["endpoint"] == "/api/calibration/confirm-position"
    assert props["joint_name"] == "left_knee"
    assert props["offset_rad"] == pytest.approx(0.5)
    assert props["offset_deg"] == pytest.approx(28.65, abs=0.05)
    assert props["seam_headroom_deg"] == pytest.approx(72.4, abs=0.2)


def test_saved_event_carries_the_offsets_skips_retries_and_faults(client, hwi, captured):
    hwi.init_pos = {"left_knee": 1.379, "right_knee": 1.379, "head_yaw": 0.0}
    client.post("/api/calibration/start")

    # left_knee takes two goes; right_knee one; head_yaw is skipped entirely
    for reading in (0.4, 0.5):
        client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
        hwi.readings["left_knee"] = reading
        client.post("/api/calibration/confirm-position", json={"jointName": "left_knee"})
    client.post("/api/calibration/accept", json={"jointName": "left_knee", "offset": 0.5})

    client.post("/api/calibration/begin-joint", json={"jointName": "right_knee"})
    client.post("/api/calibration/confirm-position", json={"jointName": "right_knee"})
    client.post("/api/calibration/accept", json={"jointName": "right_knee", "offset": -0.1})

    # and one fault along the way, to prove faults are grouped by code
    client.post("/api/calibration/confirm-position", json={"jointName": "head_yaw"})

    client.post("/api/calibration/save")
    props = _event(captured, "joint_calibration_saved")

    assert props["offsets_rad"] == {"left_knee": pytest.approx(0.5), "right_knee": pytest.approx(-0.1)}
    assert props["joints_calibrated"] == 2
    assert props["joints_total"] == 3
    # named, not counted: "everybody skips the head joints" and "this robot skipped a
    # knee" are different findings
    assert props["joints_skipped"] == ["head_yaw"]
    assert props["joints_needing_retry"] == ["left_knee"]
    assert props["measure_attempts"] == 3
    assert props["faults"] == {"INVALID_STATE": 1}
    assert props["fault_count"] == 1
    assert props["max_abs_offset_deg"] == pytest.approx(28.65, abs=0.05)
    # the lowest headroom is the number that says whether this robot is near the seam
    assert props["min_seam_headroom_deg"] == pytest.approx(72.4, abs=0.2)
    assert props["duration_s"] is not None


def test_a_fault_is_grouped_by_code_not_by_message(client, hwi, captured):
    """error_code as its own property: 'MOTORS_SILENT: left_knee: timeout' cannot be
    grouped in PostHog, and grouping is the entire point of collecting it."""
    client.post("/api/calibration/start")
    hwi.read_fails = {"left_knee"}
    client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})

    props = _event(captured, "api_request_failed")
    assert props["error_code"] == "MOTORS_SILENT"
    assert props["joint_name"] == "left_knee"


def test_telemetry_never_breaks_a_save(client, hwi, monkeypatch, captured):
    """The write has already landed by the time the summary is built. A bug in the
    summary must not turn a successful calibration into a 500."""
    client.post("/api/calibration/accept", json={"jointName": "left_knee", "offset": 0.1})

    def boom(*a, **k):
        raise RuntimeError("posthog exploded")

    monkeypatch.setattr(tnkr_server, "_seam_headroom", boom)
    r = client.post("/api/calibration/save")
    assert r.status_code == 200
    assert r.json()["offsets"]["left_knee"] == pytest.approx(0.1)


def test_start_does_not_alias_init_pos_to_zero_pos(client, hwi):
    """`hwi.init_pos = hwi.zero_pos` makes the two names one object, so a later
    per-joint write to either silently edits the other."""
    client.post("/api/calibration/start")
    hwi.init_pos["left_knee"] = 9.9
    assert hwi.zero_pos["left_knee"] == 0.0


def test_a_corrupt_config_refuses_rather_than_overwriting_it(client, hwi, tmp_path):
    """A READ failure that never reaches a write. Refusing is correct: the file also
    holds start_paused, imu_upside_down and expression_features, and overwriting a file
    we could not parse would destroy all of it to save the offsets."""
    (tmp_path / "duck_config.json").write_text('{"start_paused": true,,,}')
    client.post("/api/calibration/accept", json={"jointName": "left_knee", "offset": 0.15})
    r = client.post("/api/calibration/save")
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail["code"] == "CONFIG_WRITE_FAILED"
    # the log says which half failed; the operator's sentence does not need to
    assert "could not read" in detail["message"]
    # and the corrupt file is untouched
    assert (tmp_path / "duck_config.json").read_text() == '{"start_paused": true,,,}'


def test_a_full_card_gets_its_own_code(client, hwi, monkeypatch):
    """The one write failure with an action behind it: free space, or swap the card.
    Folding it into CONFIG_WRITE_FAILED buries the only cause an operator can act on
    under three they cannot."""
    import errno as errno_mod

    def full(*a, **k):
        raise OSError(errno_mod.ENOSPC, "No space left on device")

    monkeypatch.setattr(tnkr_server, "_write_config", full)
    client.post("/api/calibration/accept", json={"jointName": "left_knee", "offset": 0.15})
    r = client.post("/api/calibration/save")
    assert r.status_code == 507  # Insufficient Storage
    detail = r.json()["detail"]
    assert detail["code"] == "AGENT_DISK_FULL"
    assert "no space left" in detail["message"]


def test_a_read_only_card_gets_its_own_code(client, hwi, monkeypatch):
    """The most likely of the four, and the one that looks like nothing is wrong: the
    kernel remounts ext4 read-only when a failing card starts erroring, so the robot
    keeps running perfectly from RAM and only writes fail."""
    import errno as errno_mod

    def readonly(*a, **k):
        raise OSError(errno_mod.EROFS, "Read-only file system")

    monkeypatch.setattr(tnkr_server, "_write_config", readonly)
    client.post("/api/calibration/accept", json={"jointName": "left_knee", "offset": 0.15})
    r = client.post("/api/calibration/save")
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail["code"] == "AGENT_DISK_READONLY"
    assert "read-only" in detail["message"]


def test_other_write_errors_stay_config_write_failed(client, hwi, monkeypatch):
    """What is left after the three named causes: a root-owned file from a sudo'd setup
    step, and anything else the OS reports. No action, so no action is offered."""
    import errno as errno_mod

    def denied(*a, **k):
        raise OSError(errno_mod.EACCES, "Permission denied")

    monkeypatch.setattr(tnkr_server, "_write_config", denied)
    r = client.post("/api/calibration/save")
    assert r.status_code == 500
    assert r.json()["detail"]["code"] == "CONFIG_WRITE_FAILED"


def test_a_failed_backup_does_not_destroy_the_previous_backup(tmp_path, monkeypatch):
    """Measured on a real Pi with a full card: shutil.copyfile CREATES the destination
    and then fails on the first write, so copying straight to .bak left a zero-byte
    backup while the original was intact — the safety net destroyed at the one moment
    it is needed. The backup is staged through .bak.tmp for that reason."""
    import errno as errno_mod
    import shutil as shutil_mod

    cfg = tmp_path / "duck_config.json"
    cfg.write_text('{"joints_offsets": {"left_knee": 0.5}}')
    bak = tmp_path / "duck_config.json.bak"
    bak.write_text('{"joints_offsets": {"left_knee": 0.4}}')  # a GOOD previous backup
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", str(cfg))

    real_copy = shutil_mod.copyfile

    def copy_then_run_out(src, dst, **kw):
        # exactly what the Pi does: the destination appears, then the write fails
        open(dst, "w").close()
        raise OSError(errno_mod.ENOSPC, "No space left on device")

    monkeypatch.setattr(tnkr_server.shutil, "copyfile", copy_then_run_out)
    try:
        tnkr_server._write_config({"joints_offsets": {"left_knee": 0.6}})
    except OSError:
        pass

    # the good backup survived, because the doomed copy went to .bak.tmp
    assert bak.read_text() == '{"joints_offsets": {"left_knee": 0.4}}'
    # and the original is untouched
    assert "0.5" in cfg.read_text()
    monkeypatch.setattr(tnkr_server.shutil, "copyfile", real_copy)


def test_a_successful_write_still_refreshes_the_backup(tmp_path, monkeypatch):
    """The staging must not break the thing it protects."""
    cfg = tmp_path / "duck_config.json"
    cfg.write_text('{"joints_offsets": {"left_knee": 0.4}}')
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", str(cfg))

    tnkr_server._write_config({"joints_offsets": {"left_knee": 0.6}})

    assert "0.6" in cfg.read_text()
    assert "0.4" in (tmp_path / "duck_config.json.bak").read_text()
    # no staging litter left behind
    assert not (tmp_path / "duck_config.json.bak.tmp").exists()
    assert not (tmp_path / "duck_config.json.tmp").exists()


def test_finish_repowers_every_joint_so_none_is_left_limp(client, hwi):
    """Leaving the screen mid-session used to leave the released joint hanging — the
    operator walks away and the duck sags on one leg. No other route re-powers a joint
    without also accepting an offset for it."""
    client.post("/api/calibration/start")
    client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    hwi.torque_calls.clear()

    r = client.post("/api/calibration/finish")
    assert r.status_code == 200
    assert r.json()["repowered"] == list(JOINTS)
    assert r.json()["failed"] == []
    assert all(enabled for _, enabled in hwi.torque_calls)
    assert tnkr_server.calibration_baselines == {}


def test_finish_makes_the_rest_safe_when_one_servo_will_not_answer(client, hwi):
    """One dead servo must not stop the other thirteen being re-powered, and the caller
    is a page being navigated away from — it has nowhere to show an error."""
    client.post("/api/calibration/start")
    hwi.torque_fails = {"right_knee"}

    r = client.post("/api/calibration/finish")
    assert r.status_code == 200
    assert r.json()["failed"] == ["right_knee"]
    assert set(r.json()["repowered"]) == set(JOINTS) - {"right_knee"}


def test_finish_on_a_session_that_never_opened_is_a_no_op(client, monkeypatch):
    monkeypatch.setattr(tnkr_server, "hwi_instance", None)
    r = client.post("/api/calibration/finish")
    assert r.status_code == 200
    assert r.json() == {"success": True, "repowered": [], "failed": []}


def test_finish_re_powers_each_joint_where_it_is_and_frees_the_bus(client, hwi):
    """A released joint's goal register still says where it was before the operator's
    hands took it, so enabling torque against it is a drive. The goal is rewritten to
    the present position first. Then the singleton is dropped bus-only: it is carrying
    the session's zeroed offsets, and the next route to reuse it (head puppet turns
    the robot on through it) would command its stance through them."""
    hwi.readings = {"left_knee": 1.3, "right_knee": -0.9, "head_yaw": 0.2}
    client.post("/api/calibration/start")
    client.post("/api/calibration/begin-joint", json={"jointName": "left_knee"})
    hwi.readings["left_knee"] = 0.9  # let go of, and sagged

    order = []
    hwi.set_position_all = lambda positions: order.append(("goal", dict(positions)))
    hwi.set_joint_torque = lambda joint_name, enabled: order.append(
        ("torque", joint_name, enabled)
    )
    r = client.post("/api/calibration/finish")
    assert r.status_code == 200
    assert order[0] == ("goal", {"left_knee": pytest.approx(0.9)})
    assert order[1] == ("torque", "left_knee", True)
    assert tnkr_server.hwi_instance is None  # bus freed...
    assert hwi.turned_off is False  # ...torque kept
    assert tnkr_server.calibration_hold == {}


def test_save_clears_the_hold_so_a_later_release_cannot_replay_it(client, hwi):
    """After a save the next HWI loads the saved offsets, and the placed pose was
    recorded against zeros: re-asserting it would move every skipped joint by its
    on-disk offset. The accepted offsets stay, because a failed save retries on them."""
    hwi.readings = {"left_knee": 1.3, "right_knee": -0.9, "head_yaw": 0.2}
    client.post("/api/calibration/start")
    client.post("/api/calibration/accept", json={"jointName": "left_knee", "offset": 0.15})
    r = client.post("/api/calibration/save")
    assert r.status_code == 200
    assert tnkr_server.calibration_hold == {}
    assert tnkr_server.calibration_baselines == {}
    assert tnkr_server.calibration_offsets == {"left_knee": pytest.approx(0.15)}


def test_wiggle_rocks_the_joint_about_its_held_pose_and_puts_it_back(client, hwi, monkeypatch):
    """The identity check: which physical joint answers to this name? A few degrees
    either side of where the joint is held, then back, and nothing else moves."""
    monkeypatch.setattr(tnkr_server.time, "sleep", lambda s: None)
    hwi.readings = {"left_knee": 1.3, "right_knee": -0.9, "head_yaw": 0.2}
    client.post("/api/calibration/start")

    r = client.post("/api/calibration/wiggle", json={"jointName": "right_knee"})
    assert r.status_code == 200
    assert r.json()["jointName"] == "right_knee"
    assert all(j == "right_knee" for j, _ in hwi.single_writes)
    goals = [g for _, g in hwi.single_writes]
    assert max(goals) == pytest.approx(-0.9 + tnkr_server.CALIBRATION_WIGGLE_RAD)
    assert min(goals) == pytest.approx(-0.9 - tnkr_server.CALIBRATION_WIGGLE_RAD)
    assert goals[-1] == pytest.approx(-0.9)  # back where it was held


def test_wiggle_needs_a_held_pose_and_a_powered_joint(client, hwi, monkeypatch):
    monkeypatch.setattr(tnkr_server.time, "sleep", lambda s: None)
    r = client.post("/api/calibration/wiggle", json={"jointName": "right_knee"})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "INVALID_STATE"

    client.post("/api/calibration/start")
    client.post("/api/calibration/begin-joint", json={"jointName": "right_knee"})
    # a released joint is limp: a goal write would re-power it under a hand
    r = client.post("/api/calibration/wiggle", json={"jointName": "right_knee"})
    assert r.status_code == 409
    assert hwi.single_writes == []


def test_swap_legs_writes_the_config_and_ends_the_hold_for_a_re_arm(client, hwi, tmp_path):
    """The held pose was read per NAME under the old naming, so it cannot be patched:
    the session ends, the bus is freed (torque kept), and the screen arms again."""
    import json

    hwi.readings = {"left_knee": 1.3, "right_knee": -0.9, "head_yaw": 0.2}
    client.post("/api/calibration/start")
    r = client.post("/api/calibration/swap-legs", json={"swapped": True})
    assert r.status_code == 200
    assert r.json()["legsSwapped"] is True
    saved = json.loads((tmp_path / "duck_config.json").read_text())
    assert saved["legs_swapped"] is True
    assert tnkr_server.hwi_instance is None
    assert hwi.turned_off is False
    assert tnkr_server.calibration_hold == {}


def test_start_reports_whether_the_legs_are_swapped(client, hwi):
    import types

    assert client.post("/api/calibration/start").json()["legsSwapped"] is False
    hwi.duck_config = types.SimpleNamespace(legs_swapped=True)
    assert client.post("/api/calibration/start").json()["legsSwapped"] is True


def test_health_advertises_the_hold_so_studio_can_refuse_an_older_agent(client):
    """Studio checks this at connect and refuses to arm without it. It cannot wait for
    /start's reply: an older agent has driven every joint to servo-zero by then."""
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["calibrationMode"] == "hold"
