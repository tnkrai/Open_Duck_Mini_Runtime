import time

import numpy as np
import rustypot
import serial.tools.list_ports
from mini_bdx_runtime.duck_config import DuckConfig

# USB vendor IDs for the servo-bus adapters we ship. The bus adapter is the
# only USB-serial device on the robot, so matching the chip vendor uniquely
# identifies it — regardless of /dev/ttyACMx numbering, which USB port/cable
# it's on, or the adapter's per-unit serial number. This lets the same code
# run on any robot without a hardcoded device path.
SERVO_ADAPTER_VIDS = {
    0x1A86: "CH343",  # QinHeng (current v3 adapter, enumerates as /dev/ttyACM*)
    0x0403: "FTDI",   # FTDI (older adapter, enumerates as /dev/ttyUSB*)
}


def find_servo_adapter() -> tuple[str, str]:
    """Locate the servo-bus USB adapter by vendor ID; return (device, chip).

    ttyACMx numbers renumber on replug and a by-id path is unique per physical
    adapter, so we match the chip vendor instead — works on any robot/cable.
    Raises a clear error if no known adapter is present.
    """
    ports = list(serial.tools.list_ports.comports())
    matches = [(p.device, SERVO_ADAPTER_VIDS[p.vid]) for p in ports if p.vid in SERVO_ADAPTER_VIDS]
    if not matches:
        seen = ", ".join(
            f"{p.device} (vid={p.vid:#06x})" if p.vid else p.device for p in ports
        ) or "none"
        raise RuntimeError(
            "No servo-bus USB adapter found (looked for CH343/FTDI by vendor id). "
            f"Serial devices present: {seen}. Is the adapter plugged in?"
        )
    if len(matches) > 1:
        print(f"[HWI] Warning: multiple servo adapters found {matches}; using {matches[0][0]}")
    device, chip = matches[0]
    print(f"[HWI] Using servo adapter {device} ({chip})")
    return device, chip


def find_servo_port() -> str:
    """Back-compat wrapper around find_servo_adapter()."""
    return find_servo_adapter()[0]


class HWI:
    def __init__(self, duck_config: DuckConfig, usb_port: str | None = None):

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

        self.servo_adapter_chip: str | None = None
        if usb_port is None:
            usb_port, self.servo_adapter_chip = find_servo_adapter()
        self.io = rustypot.feetech(usb_port, 1000000)

    # CH343/cdc_acm has no latency-timer knob, so single-servo transactions
    # occasionally time out, and a brief bus voltage sag (e.g. on high-KP
    # energize) can drop one mid-write. Retry transient OSErrors a few times;
    # a truly unresponsive servo still fails every attempt and is named.
    _IO_ATTEMPTS = 3
    _IO_RETRY_DELAY = 0.02

    def _io_retry(self, fn, joint_name, op):
        """Run a single-servo io op, retrying transient OSErrors and naming
        the joint (and id) if it ultimately fails."""
        last_exc = None
        for attempt in range(self._IO_ATTEMPTS):
            try:
                return fn()
            except OSError as e:
                last_exc = e
                if attempt + 1 < self._IO_ATTEMPTS:
                    print(
                        f"[HWI] {op} timed out for '{joint_name}' "
                        f"(id {self.joints[joint_name]}), "
                        f"retry {attempt + 1}/{self._IO_ATTEMPTS - 1}: {e}"
                    )
                    time.sleep(self._IO_RETRY_DELAY)
        raise OSError(
            f"{op} failed for '{joint_name}' (id {self.joints[joint_name]}) "
            f"after {self._IO_ATTEMPTS} attempts: {last_exc}"
        )

    def _write_kps(self, kps):
        # CH343/cdc_acm adapter can't reliably do a bulk multi-servo sync
        # transaction at 1 Mbaud (raises OSError: Parsing error). Loop one
        # servo at a time. Does NOT touch self.kps (used by turn_on).
        for name, id, kp in zip(self.joints.keys(), self.joints.values(), kps):
            self._io_retry(lambda i=id, k=kp: self.io.set_kps([i], [k]), name, "set_kps")

    def set_kps(self, kps):
        self.kps = kps
        self._write_kps(self.kps)

    def set_kds(self, kds):
        self.kds = kds
        # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
        for name, id, kd in zip(self.joints.keys(), self.joints.values(), self.kds):
            self._io_retry(lambda i=id, k=kd: self.io.set_kds([i], [k]), name, "set_kds")

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
        for name, id in self.joints.items():
            self._io_retry(lambda i=id: self.io.disable_torque([i]), name, "disable_torque")

    def set_position(self, joint_name, pos):
        """
        pos is in radians
        """
        id = self.joints[joint_name]
        pos = pos + self.joints_offsets[joint_name]
        self._io_retry(
            lambda: self.io.write_goal_position([id], [pos]),
            joint_name,
            "write_goal_position",
        )

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
        for joint, position in joints_positions.items():
            id = self.joints[joint]
            target = position + self.joints_offsets[joint]
            self._io_retry(
                lambda i=id, p=target: self.io.write_goal_position([i], [p]),
                joint,
                "write_goal_position",
            )

    def get_present_positions(self, ignore=[]):
        """
        Returns the present positions in radians
        """

        try:
            # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
            present_positions = [
                self._io_retry(
                    lambda i=id: self.io.read_present_position([i])[0],
                    name,
                    "read_present_position",
                )
                for name, id in self.joints.items()
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
                self._io_retry(
                    lambda i=id: self.io.read_present_velocity([i])[0],
                    name,
                    "read_present_velocity",
                )
                for name, id in self.joints.items()
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
