"""Joint direction test (/api/directions/*) and the per-joint signs behind it.

A mirrored servo passes every position read and only shows itself when the joint
moves, which on a walk means driving into the shell. The session moves the left and
right joint of a pair together at low stiffness, flips the sign of whichever joint the
operator says went the wrong way, and writes the signs to duck_config.json, where the
HWI applies them to every command and read.
"""

import json

import pytest

import tnkr_server
from mini_bdx_runtime.duck_config import DuckConfig


JOINTS = {
    "left_hip_pitch": 3,
    "left_knee": 4,
    "left_ankle": 5,
    "head_yaw": 32,
    "right_hip_pitch": 12,
    "right_knee": 13,
    "right_ankle": 14,
}


class FakeHWI:
    """Records what the endpoints ask of the bus, in RAW servo terms."""

    def __init__(self):
        self.joints = dict(JOINTS)
        self.joints_offsets = {n: 0.0 for n in JOINTS}
        self.joints_signs = {n: 1 for n in JOINTS}
        self.zero_pos = {n: 0.0 for n in JOINTS}
        self.init_pos = dict(self.zero_pos)
        self.readings = {n: 0.0 for n in JOINTS}
        self.raw_goals = {}  # joint -> the last raw goal written
        self.writes = []  # (joint, raw goal), in order
        self.torque_calls = []
        self.turned_off = False
        self.kps = None
        self.kds = None

    def get_present_positions(self):
        return [self.readings[n] for n in self.joints]

    def get_present_position(self, joint_name):
        return self.readings[joint_name]

    def _write(self, joint, position):
        raw = self.joints_signs.get(joint, 1) * position + self.joints_offsets.get(joint, 0.0)
        self.raw_goals[joint] = raw
        self.writes.append((joint, raw))

    def set_position(self, joint_name, pos):
        self._write(joint_name, pos)

    def set_position_all(self, positions):
        for joint, position in positions.items():
            self._write(joint, position)

    def set_joint_torque(self, joint_name, enabled):
        self.torque_calls.append((joint_name, enabled))

    def set_kps(self, kps):
        self.kps = kps

    def set_kds(self, kds):
        self.kds = kds

    def turn_off(self):
        self.turned_off = True

    def close(self):
        pass


@pytest.fixture
def hwi(monkeypatch, tmp_path):
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", str(tmp_path / "duck_config.json"))
    monkeypatch.setattr(tnkr_server, "directions_session", None)
    monkeypatch.setattr(tnkr_server, "calibration_session", {})
    fake = FakeHWI()
    monkeypatch.setattr(tnkr_server, "hwi_instance", fake)
    monkeypatch.setattr(tnkr_server, "get_hwi", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(tnkr_server.time, "sleep", lambda s: None)


def test_start_stands_straight_at_low_stiffness_and_names_the_pairs(client, hwi):
    """Straight is a drive, and a deliberate one: the offsets are set by now, so zero
    IS the straight pose, and kp 8 is what makes a wrong-way joint stall softly."""
    hwi.joints_offsets["right_knee"] = 0.4
    r = client.post("/api/directions/start")
    assert r.status_code == 200
    body = r.json()
    assert [p["id"] for p in body["pairs"]] == ["hip_pitch", "knee", "ankle"]
    assert body["signs"] == {n: 1 for n in JOINTS}
    assert hwi.kps == [tnkr_server.DIRECTION_HOLD_KP] * len(JOINTS)
    assert hwi.kds == [0] * len(JOINTS)
    # straight, through the offsets: raw = offset
    assert hwi.raw_goals["right_knee"] == pytest.approx(0.4)
    assert hwi.raw_goals["left_knee"] == pytest.approx(0.0)
    assert all(enabled for _, enabled in hwi.torque_calls)


def test_move_and_rest_write_the_pairs_targets_through_sign_and_offset(client, hwi):
    hwi.joints_offsets["right_hip_pitch"] = 0.1
    client.post("/api/directions/start")

    r = client.post("/api/directions/move", json={"pairId": "hip_pitch"})
    assert r.status_code == 200
    assert hwi.raw_goals["left_hip_pitch"] == pytest.approx(-0.3)
    assert hwi.raw_goals["right_hip_pitch"] == pytest.approx(0.3 + 0.1)
    assert tnkr_server.directions_session["moved"] == "hip_pitch"

    r = client.post("/api/directions/rest", json={"pairId": "hip_pitch"})
    assert r.status_code == 200
    assert hwi.raw_goals["left_hip_pitch"] == pytest.approx(0.0)
    assert hwi.raw_goals["right_hip_pitch"] == pytest.approx(0.1)
    assert tnkr_server.directions_session["moved"] is None


def test_flip_inverts_the_next_move_and_nothing_else(client, hwi):
    """The operator said the left knee went the wrong way. Flipping changes what the
    next command means; it moves nothing by itself, because straight is raw = offset
    whichever way the sign points."""
    client.post("/api/directions/start")
    writes_before = len(hwi.writes)

    r = client.post("/api/directions/flip", json={"jointName": "left_knee"})
    assert r.status_code == 200
    assert r.json()["sign"] == -1
    assert r.json()["signs"]["left_knee"] == -1
    assert r.json()["signs"]["right_knee"] == 1
    assert len(hwi.writes) == writes_before  # nothing moved

    client.post("/api/directions/move", json={"pairId": "knee"})
    assert hwi.raw_goals["left_knee"] == pytest.approx(-0.5)  # mirrored
    assert hwi.raw_goals["right_knee"] == pytest.approx(0.5)

    # flipping twice is a no-op, so a wrong answer can be taken back
    client.post("/api/directions/flip", json={"jointName": "left_knee"})
    assert tnkr_server.hwi_instance is hwi
    assert hwi.joints_signs["left_knee"] == 1


def test_save_writes_every_sign_keeps_torque_and_frees_the_bus(client, hwi, tmp_path):
    (tmp_path / "duck_config.json").write_text(
        json.dumps({"start_paused": True, "joints_offsets": {"left_knee": 0.05}})
    )
    client.post("/api/directions/start")
    client.post("/api/directions/flip", json={"jointName": "right_knee"})
    client.post("/api/directions/move", json={"pairId": "knee"})

    r = client.post("/api/directions/save")
    assert r.status_code == 200
    assert r.json()["signs"]["right_knee"] == -1
    assert r.json()["signs"]["left_knee"] == 1

    saved = json.loads((tmp_path / "duck_config.json").read_text())
    # every joint stated, not only the inverted one
    assert saved["joints_signs"] == {**{n: 1 for n in JOINTS}, "right_knee": -1}
    # and nothing else in the file was disturbed
    assert saved["start_paused"] is True
    assert saved["joints_offsets"] == {"left_knee": 0.05}
    # a displaced pair is rested before the hand-off
    assert hwi.raw_goals["right_knee"] == pytest.approx(0.0)
    assert hwi.turned_off is False
    assert tnkr_server.hwi_instance is None
    assert tnkr_server.directions_session is None


def test_finish_rests_a_displaced_pair_and_forgets_unsaved_flips(client, hwi):
    client.post("/api/directions/start")
    client.post("/api/directions/flip", json={"jointName": "left_ankle"})
    client.post("/api/directions/move", json={"pairId": "ankle"})

    r = client.post("/api/directions/finish")
    assert r.status_code == 200
    assert hwi.raw_goals["left_ankle"] == pytest.approx(0.0)
    assert hwi.raw_goals["right_ankle"] == pytest.approx(0.0)
    assert hwi.turned_off is False
    # the bus is freed: the next HWI is built from the file, which has no flip
    assert tnkr_server.hwi_instance is None
    assert tnkr_server.directions_session is None

    # idempotent: a second finish, or one without a session, is a no-op
    r = client.post("/api/directions/finish")
    assert r.status_code == 200


def test_moving_without_a_session_is_a_state_error(client, hwi):
    r = client.post("/api/directions/move", json={"pairId": "knee"})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "INVALID_STATE"
    assert hwi.writes == []


def test_an_unknown_pair_touches_nothing(client, hwi):
    client.post("/api/directions/start")
    writes_before = len(hwi.writes)
    r = client.post("/api/directions/move", json={"pairId": "elbow"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "UNKNOWN_PAIR"
    assert len(hwi.writes) == writes_before


def test_directions_refuse_while_a_walk_owns_the_bus(client, hwi, monkeypatch):
    monkeypatch.setattr(tnkr_server, "is_walking", lambda: True)
    for path, body in [
        ("/api/directions/start", None),
        ("/api/directions/move", {"pairId": "knee"}),
        ("/api/directions/flip", {"jointName": "left_knee"}),
    ]:
        r = client.post(path, json=body) if body else client.post(path)
        assert r.status_code == 409, path


# ── the signs themselves ──────────────────────────────────────────────────────


def test_duck_config_reads_the_signs_and_defaults_the_rest_to_plus_one(tmp_path):
    cfg = tmp_path / "duck_config.json"
    cfg.write_text(
        json.dumps({"joints_signs": {"right_knee": -1, "left_ankle": "-1", "head_yaw": 0.5}})
    )
    signs = DuckConfig(config_json_path=str(cfg), ignore_default=True).joints_signs
    assert signs["right_knee"] == -1
    assert signs["left_ankle"] == -1
    assert signs["head_yaw"] == 1  # not a sign: ignored rather than scaling the joint
    assert signs["left_knee"] == 1
    assert len(signs) == 14


def test_a_config_without_signs_is_all_plus_one(tmp_path):
    cfg = tmp_path / "duck_config.json"
    cfg.write_text(json.dumps({"joints_offsets": {"left_knee": 0.1}}))
    signs = DuckConfig(config_json_path=str(cfg), ignore_default=True).joints_signs
    assert set(signs.values()) == {1}


def test_hwi_swaps_the_leg_ids_by_name_when_the_config_says_so(tmp_path, monkeypatch):
    """A build that programmed the right leg's ids into the left leg's servos: every
    left_* name must reach the physical left leg. Swapped in place, so the dict order
    the policy's vectors and the gain arrays index by is untouched."""
    import mini_bdx_runtime.rustypot_position_hwi as hwi_mod

    monkeypatch.setattr(hwi_mod.rustypot, "feetech", lambda port, baud: object())
    cfg = tmp_path / "duck_config.json"

    # one crossed pair: the real build's hip yaws, and nothing else
    cfg.write_text(json.dumps({"swapped_pairs": ["hip_yaw"]}))
    hwi = hwi_mod.HWI(DuckConfig(config_json_path=str(cfg), ignore_default=True), usb_port="/dev/fake")
    assert hwi.joints["left_hip_yaw"] == 10 and hwi.joints["right_hip_yaw"] == 20
    assert hwi.joints["left_knee"] == 23 and hwi.joints["right_knee"] == 13
    assert hwi.joints["head_yaw"] == 32  # the head is not a leg
    assert list(hwi.joints)[:5] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"
    ]

    # the older whole-leg form still means all five
    cfg.write_text(json.dumps({"legs_swapped": True}))
    hwi = hwi_mod.HWI(DuckConfig(config_json_path=str(cfg), ignore_default=True), usb_port="/dev/fake")
    assert hwi.joints["left_knee"] == 13 and hwi.joints["right_knee"] == 23
    assert hwi.duck_config.legs_swapped is True

    cfg.write_text(json.dumps({"swapped_pairs": [], "legs_swapped": False}))
    hwi = hwi_mod.HWI(DuckConfig(config_json_path=str(cfg), ignore_default=True), usb_port="/dev/fake")
    assert hwi.joints["left_knee"] == 23 and hwi.joints["right_knee"] == 13


def test_hwi_applies_the_sign_on_the_way_out_and_undoes_it_on_the_way_back(tmp_path, monkeypatch):
    """raw = sign * position + offset, position = sign * (raw - offset): the offset is
    added AFTER the sign, so a calibration done before the direction test survives it."""
    import mini_bdx_runtime.rustypot_position_hwi as hwi_mod

    class FakeIO:
        def __init__(self):
            self.goals = {}

        def write_goal_position(self, ids, positions):
            self.goals[ids[0]] = positions[0]

        def read_present_position(self, ids):
            return [self.goals.get(ids[0], 0.0)]

        def read_present_velocity(self, ids):
            return [2.0]

    # the rustypot stub raises on construction; stand a recording io in for it
    monkeypatch.setattr(hwi_mod.rustypot, "feetech", lambda port, baud: FakeIO())

    cfg = tmp_path / "duck_config.json"
    offsets = {n: 0.0 for n in DuckConfig(config_json_path=None, ignore_default=True).joints_offset}
    offsets["left_knee"] = 0.1
    cfg.write_text(json.dumps({"joints_offsets": offsets, "joints_signs": {"left_knee": -1}}))
    hwi = hwi_mod.HWI(
        DuckConfig(config_json_path=str(cfg), ignore_default=True), usb_port="/dev/fake"
    )
    hwi.set_position("left_knee", 0.5)
    assert hwi.io.goals[hwi.joints["left_knee"]] == pytest.approx(-0.5 + 0.1)
    assert hwi.get_present_position("left_knee") == pytest.approx(0.5)

    hwi.set_position_all({"left_knee": 0.2, "right_knee": 0.2})
    assert hwi.io.goals[hwi.joints["left_knee"]] == pytest.approx(-0.2 + 0.1)
    assert hwi.io.goals[hwi.joints["right_knee"]] == pytest.approx(0.2)
    positions = dict(zip(hwi.joints.keys(), hwi.get_present_positions()))
    assert positions["left_knee"] == pytest.approx(0.2)
    assert positions["right_knee"] == pytest.approx(0.2)
    velocities = dict(zip(hwi.joints.keys(), hwi.get_present_velocities()))
    assert velocities["left_knee"] == pytest.approx(-2.0)
    assert velocities["right_knee"] == pytest.approx(2.0)
