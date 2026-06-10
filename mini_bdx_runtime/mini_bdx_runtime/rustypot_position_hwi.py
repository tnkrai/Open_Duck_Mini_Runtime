import time

import numpy as np
import rustypot
from mini_bdx_runtime.duck_config import DuckConfig


class HWI:
    def __init__(self, duck_config: DuckConfig, usb_port: str = "/dev/ttyACM0"):

        self.duck_config = duck_config

        # Order matters here
        self.joints = {
            "left_hip_yaw": 20,
            "left_hip_roll": 21,
            "left_hip_pitch": 22,
            "left_knee": 23,
            "left_ankle": 24,
            "neck_pitch": 30,
            "head_pitch": 31,
            "head_yaw": 32,
            "head_roll": 33,
            # "left_antenna": None,
            # "right_antenna": None,
            "right_hip_yaw": 10,
            "right_hip_roll": 11,
            "right_hip_pitch": 12,
            "right_knee": 13,
            "right_ankle": 14,
        }

        self.zero_pos = {
            "left_hip_yaw": 0,
            "left_hip_roll": 0,
            "left_hip_pitch": 0,
            "left_knee": 0,
            "left_ankle": 0,
            "neck_pitch": 0,
            "head_pitch": 0,
            "head_yaw": 0,
            "head_roll": 0,
            # "left_antenna":0,
            # "right_antenna":0,
            "right_hip_yaw": 0,
            "right_hip_roll": 0,
            "right_hip_pitch": 0,
            "right_knee": 0,
            "right_ankle": 0,
        }

        self.init_pos = {
            "left_hip_yaw": 0.002,
            "left_hip_roll": 0.053,
            "left_hip_pitch": -0.63,
            "left_knee": 1.368,
            "left_ankle": -0.784,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0,
            "head_roll": 0,
            # "left_antenna": 0,
            # "right_antenna": 0,
            "right_hip_yaw": -0.003,
            "right_hip_roll": -0.065,
            "right_hip_pitch": 0.635,
            "right_knee": 1.379,
            "right_ankle": -0.796,
        }

        self.joints_offsets = self.duck_config.joints_offset

        self.kps = np.ones(len(self.joints)) * 32  # default kp
        self.kds = np.ones(len(self.joints)) * 0  # default kd
        self.low_torque_kps = np.ones(len(self.joints)) * 2

        self.io = rustypot.feetech(usb_port, 1000000)

    def _write_kps(self, kps):
        # CH343/cdc_acm adapter can't reliably do a bulk multi-servo sync
        # transaction at 1 Mbaud (raises OSError: Parsing error). Loop one
        # servo at a time. Does NOT touch self.kps (used by turn_on).
        for id, kp in zip(self.joints.values(), kps):
            self.io.set_kps([id], [kp])

    def set_kps(self, kps):
        self.kps = kps
        self._write_kps(self.kps)

    def set_kds(self, kds):
        self.kds = kds
        # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
        for id, kd in zip(self.joints.values(), self.kds):
            self.io.set_kds([id], [kd])

    def set_kp(self, id, kp):
        self.io.set_kps([id], [kp])

    def turn_on(self):
        self._write_kps(self.low_torque_kps)
        print("turn on : low KPS set")
        time.sleep(1)

        self.set_position_all(self.init_pos)
        print("turn on : init pos set")

        time.sleep(1)

        self._write_kps(self.kps)
        print("turn on : high kps")

    def turn_off(self):
        # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
        for id in self.joints.values():
            self.io.disable_torque([id])

    def set_position(self, joint_name, pos):
        """
        pos is in radians
        """
        id = self.joints[joint_name]
        pos = pos + self.joints_offsets[joint_name]
        self.io.write_goal_position([id], [pos])

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        ids_positions = {
            self.joints[joint]: position + self.joints_offsets[joint]
            for joint, position in joints_positions.items()
        }

        # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
        for id, position in ids_positions.items():
            self.io.write_goal_position([id], [position])

    def get_present_positions(self, ignore=[]):
        """
        Returns the present positions in radians
        """

        try:
            # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
            present_positions = [
                self.io.read_present_position([id])[0]
                for id in self.joints.values()
            ]
        except Exception as e:
            print(e)
            return None

        present_positions = [
            pos - self.joints_offsets[joint]
            for joint, pos in zip(self.joints.keys(), present_positions)
            if joint not in ignore
        ]
        return np.array(np.around(present_positions, 3))

    def get_present_velocities(self, rad_s=True, ignore=[]):
        """
        Returns the present velocities in rad/s (default) or rev/min
        """
        try:
            # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
            present_velocities = [
                self.io.read_present_velocity([id])[0]
                for id in self.joints.values()
            ]
        except Exception as e:
            print(e)
            return None

        present_velocities = [
            vel
            for joint, vel in zip(self.joints.keys(), present_velocities)
            if joint not in ignore
        ]

        return np.array(np.around(present_velocities, 3))
