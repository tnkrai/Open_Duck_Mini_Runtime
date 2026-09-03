import time
from functools import wraps
from threading import RLock

import numpy as np
import rustypot
import serial.tools.list_ports
from mini_bdx_runtime.duck_config import DuckConfig

# Exclusive owner of the servo USB adapter. rustypot wraps the serial port in
# a Rust Mutex and `unwrap()`s it; a panic while holding that mutex poisons
# every later call (PyO3 PanicException / PoisonError). The same adapter is
# also opened by pypot (voltage, rehome). One lock, one owner, or the bus
# double-opens and the rustypot object is dead until process restart.
BUS_LOCK = RLock()
_SERVO_BAUD = 1_000_000
_INTERRUPT = (KeyboardInterrupt, SystemExit, GeneratorExit)

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


def is_rust_panic(exc: BaseException) -> bool:
    """True for a PyO3 PanicException (subclass of BaseException, not Exception).

    rustypot `lock().unwrap()` after a poisoned mutex surfaces as this, so
    `except Exception` in callers never sees it — FastAPI then 500s with
    Starlette's 'No response returned'.
    """
    if isinstance(exc, _INTERRUPT):
        return False
    name = type(exc).__name__
    return name == "PanicException" or "PoisonError" in str(exc)


def _with_bus_lock(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        with BUS_LOCK:
            return fn(self, *args, **kwargs)

    return wrapped


# Which servo id each joint name drives on a duck built to the guide. A build can
# differ (see duck_config: swapped_pairs and servo_ids); the walk, the gains and the
# policy index by the ORDER of this table, never by the ids.
DEFAULT_SERVO_IDS = {
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

        # Left and right by NAME, whichever ids the build gave the servos. A build that
        # programmed a pair's ids the other way round (a real one had the hip yaws
        # crossed and nothing else) would otherwise have that left_* name drive the
        # physical right joint: the walk still walks (the gait is mirror-symmetric)
        # but the joint turns the wrong way, and no per-joint sign can undo a crossed
        # pair. The ids are swapped IN PLACE so the dict order stays as declared: the
        # policy's vectors and the gain arrays index by it.
        for pair in getattr(self.duck_config, "swapped_pairs", []) or []:
            left, right = f"left_{pair}", f"right_{pair}"
            if left in self.joints and right in self.joints:
                self.joints[left], self.joints[right] = self.joints[right], self.joints[left]

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

        # Per-joint direction, +1 or -1, from duck_config.json "joints_signs". A servo
        # mounted mirrored to the model turns the wrong way for every command and
        # reports the wrong way for every read, and a walk on it stalls that joint
        # into its shell. The sign is applied at this boundary and nowhere else, so
        # the policy, the calibration routes and the walk loop all work in model
        # space:
        #
        #     raw = sign * position + offset        position = sign * (raw - offset)
        #
        # The offset stays in raw servo space, which is why it is measured at the
        # straight pose (raw = sign * 0 + offset) and survives a later sign flip.
        self.joints_signs = {
            name: int(getattr(self.duck_config, "joints_signs", {}).get(name, 1))
            for name in self.joints
        }

        self.kps = np.ones(len(self.joints)) * 32  # default kp
        self.kds = np.ones(len(self.joints)) * 0  # default kd
        self.low_torque_kps = np.ones(len(self.joints)) * 2

        self.servo_adapter_chip: str | None = None
        if usb_port is None:
            usb_port, self.servo_adapter_chip = find_servo_adapter()
        self._usb_port = usb_port
        self.io = rustypot.feetech(usb_port, _SERVO_BAUD)

    # CH343/cdc_acm has no latency-timer knob, so single-servo transactions
    # occasionally time out, and a brief bus voltage sag (e.g. on high-KP
    # energize) can drop one mid-write. Retry transient OSErrors a few times;
    # a truly unresponsive servo still fails every attempt and is named.
    _IO_ATTEMPTS = 3
    _IO_RETRY_DELAY = 0.02

    def close(self):
        """Drop the rustypot handle so the serial port is actually freed.

        Setting the server's HWI singleton to None is not enough: another
        thread can still hold a reference, and rustypot's background controller
        keeps the fd open until the pyclass is dropped. Voltage/rehome open
        pypot on the same adapter; overlapping handles garbles the bus and
        panics rustypot's mutex.
        """
        old = getattr(self, "io", None)
        self.io = None
        try:
            del old
        except BaseException:
            pass

    def _reopen_io(self):
        """Replace a poisoned rustypot controller with a fresh serial handle."""
        self.close()
        self.io = rustypot.feetech(self._usb_port, _SERVO_BAUD)

    def _io_retry(self, fn, joint_name, op):
        """Run a single-servo io op, retrying transient OSErrors and naming
        the joint (and id) if it ultimately fails.

        A rustypot PanicException (poisoned mutex) is not an OSError and is
        not a subclass of Exception. Treat it as a dead controller: drop it,
        reopen the port, retry. Remaining panics become OSError so callers'
        `except Exception` actually runs.
        """
        last_exc = None
        for attempt in range(self._IO_ATTEMPTS):
            try:
                if self.io is None:
                    self._reopen_io()
                return fn()
            except OSError as e:
                last_exc = e
            except BaseException as e:
                if not is_rust_panic(e):
                    raise
                last_exc = OSError(f"{op} rust panic: {e}")
                print(
                    f"[HWI] {op} rustypot panic for '{joint_name}' "
                    f"(id {self.joints.get(joint_name, '?')}), reopening serial: {e}"
                )
                try:
                    self._reopen_io()
                except BaseException as reopen_err:
                    raise OSError(
                        f"{op} failed for '{joint_name}' "
                        f"(id {self.joints.get(joint_name, '?')}): rustypot panicked "
                        f"and reopen failed: {reopen_err}"
                    ) from reopen_err
            if attempt + 1 < self._IO_ATTEMPTS:
                if isinstance(last_exc, OSError) and "rust panic" not in str(last_exc):
                    print(
                        f"[HWI] {op} timed out for '{joint_name}' "
                        f"(id {self.joints.get(joint_name, '?')}), "
                        f"retry {attempt + 1}/{self._IO_ATTEMPTS - 1}: {last_exc}"
                    )
                time.sleep(self._IO_RETRY_DELAY)
        raise OSError(
            f"{op} failed for '{joint_name}' (id {self.joints.get(joint_name, '?')}) "
            f"after {self._IO_ATTEMPTS} attempts: {last_exc}"
        )

    def _write_kps(self, kps):
        # CH343/cdc_acm adapter can't reliably do a bulk multi-servo sync
        # transaction at 1 Mbaud (raises OSError: Parsing error). Loop one
        # servo at a time. Does NOT touch self.kps (used by turn_on).
        for name, id, kp in zip(self.joints.keys(), self.joints.values(), kps):
            self._io_retry(lambda i=id, k=kp: self.io.set_kps([i], [k]), name, "set_kps")

    @_with_bus_lock
    def set_kps(self, kps):
        self.kps = kps
        self._write_kps(self.kps)

    @_with_bus_lock
    def set_kds(self, kds):
        self.kds = kds
        # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
        for name, id, kd in zip(self.joints.keys(), self.joints.values(), self.kds):
            self._io_retry(lambda i=id, k=kd: self.io.set_kds([i], [k]), name, "set_kds")

    @_with_bus_lock
    def set_kp(self, id, kp):
        self._io_retry(lambda: self.io.set_kps([id], [kp]), f"id:{id}", "set_kps")

    @_with_bus_lock
    def turn_on(self):
        self._write_kps(self.low_torque_kps)
        print("turn on : low KPS set")
        time.sleep(1)

        self.set_position_all(self.init_pos)
        print("turn on : init pos set")

        time.sleep(1)

        self._write_kps(self.kps)
        print("turn on : high kps")

    @_with_bus_lock
    def turn_off(self):
        # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
        for name, id in self.joints.items():
            self._io_retry(lambda i=id: self.io.disable_torque([i]), name, "disable_torque")

    @_with_bus_lock
    def set_position(self, joint_name, pos):
        """
        pos is in radians
        """
        id = self.joints[joint_name]
        pos = self.joints_signs.get(joint_name, 1) * pos + self.joints_offsets[joint_name]
        self._io_retry(
            lambda: self.io.write_goal_position([id], [pos]),
            joint_name,
            "write_goal_position",
        )

    @_with_bus_lock
    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        # Per-servo: cdc_acm can't do a bulk sync transaction (see _write_kps).
        for joint, position in joints_positions.items():
            id = self.joints[joint]
            target = self.joints_signs.get(joint, 1) * position + self.joints_offsets[joint]
            self._io_retry(
                lambda i=id, p=target: self.io.write_goal_position([i], [p]),
                joint,
                "write_goal_position",
            )

    @_with_bus_lock
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
            self.joints_signs.get(joint, 1) * (pos - self.joints_offsets[joint])
            for joint, pos in zip(self.joints.keys(), present_positions)
            if joint not in ignore
        ]
        return np.array(np.around(present_positions, 3))

    @_with_bus_lock
    def get_present_position(self, joint_name):
        """Present position of ONE joint, in radians, offset-corrected.

        `get_present_positions()` reads all fourteen and returns None if ANY of them
        fails, which is right for the walk loop (it needs the whole vector or nothing)
        and wrong for anything working on a single joint: a silent joint 3 makes a read
        for joint 7 fail, with no way left to say which joint was actually quiet.
        Per-joint calibration needs the failure attributable, so this reads one servo
        and lets `_io_retry`'s OSError through — that message names the joint and its
        id. Callers turn it into a message about that joint.

        Offset- and sign-corrected like the plural version, so the two cannot disagree
        about what "position" means. `.get(..., 0.0)` rather than `[...]` because a
        hand-edited duck_config.json can carry a partial joints_offsets dict, and a
        KeyError here would read as a dead servo.
        """
        joint_id = self.joints[joint_name]
        raw = self._io_retry(
            lambda: self.io.read_present_position([joint_id])[0],
            joint_name,
            "read_present_position",
        )
        sign = self.joints_signs.get(joint_name, 1)
        return round(sign * (float(raw) - self.joints_offsets.get(joint_name, 0.0)), 3)

    @_with_bus_lock
    def set_joint_torque(self, joint_name, enabled):
        """Torque on or off for ONE joint, retried, locked, and attributable.

        The HWI had no single-joint torque call, so callers reached through to
        `hwi.io.disable_torque([id])` directly. That bypasses three things at once: the
        bus lock (so a concurrent /api/state read can garble the write), `_io_retry`
        (so a transient timeout is a hard failure instead of one of three attempts),
        and the panic normalisation inside it (so a rustypot PanicException — a
        BaseException — escapes past every `except Exception` and kills the ASGI task).
        """
        joint_id = self.joints[joint_name]
        op = "enable_torque" if enabled else "disable_torque"
        self._io_retry(
            lambda: (self.io.enable_torque if enabled else self.io.disable_torque)([joint_id]),
            joint_name,
            op,
        )

    @_with_bus_lock
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
            self.joints_signs.get(joint, 1) * vel
            for joint, vel in zip(self.joints.keys(), present_velocities)
            if joint not in ignore
        ]

        return np.array(np.around(present_velocities, 3))
