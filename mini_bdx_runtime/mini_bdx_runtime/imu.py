"""
Fused orientation IMU driver for the BNO055 sensor.

Unlike raw_imu.py (which provides raw gyro/accel), this module uses the
BNO055's onboard sensor fusion to produce quaternion orientation.  A
background thread polls the sensor and converts the quaternion into Euler
angles, applying pitch bias correction before re-encoding as a quaternion
in the scalar-last convention expected by downstream consumers (e.g. Isaac).

Calibration and offset persistence work identically to raw_imu.py — see
that module's docstring for the full calibration flow description.
"""

import adafruit_bno055
import board
import busio
import numpy as np
import pickle
import os

# import serial

from queue import Queue
from threading import Thread
import time
from scipy.spatial.transform import Rotation as R


# TODO filter spikes
class Imu:
    def __init__(
        self, sampling_freq, user_pitch_bias=0, calibrate=False, upside_down=True
    ):
        self.sampling_freq = sampling_freq
        self.user_pitch_bias = user_pitch_bias
        self.nominal_pitch_bias = 0
        self.calibrate = calibrate

        # self.uart = serial.Serial("/dev/ttyS0", baudrate=9600)
        # self.imu = adafruit_bno055.BNO055_UART(self.uart)

        # Initialize I2C bus and BNO055 sensor
        i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = adafruit_bno055.BNO055_I2C(i2c)

        # IMUPLUS_MODE fuses accel + gyro (no magnetometer) for relative orientation.
        # This avoids magnetometer interference but has no absolute heading reference.
        self.imu.mode = adafruit_bno055.IMUPLUS_MODE
        print("[IMU MODE]: IMUPLUS_MODE (startup)")

        # Remap physical axes to match the robot's coordinate frame.
        # Sign bits differ based on whether the sensor is mounted upside-down.
        if upside_down:
            self.imu.axis_remap = (
                adafruit_bno055.AXIS_REMAP_Y,
                adafruit_bno055.AXIS_REMAP_X,
                adafruit_bno055.AXIS_REMAP_Z,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
            )
        else:
            self.imu.axis_remap = (
                adafruit_bno055.AXIS_REMAP_Y,
                adafruit_bno055.AXIS_REMAP_X,
                adafruit_bno055.AXIS_REMAP_Z,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
                adafruit_bno055.AXIS_REMAP_POSITIVE,
                adafruit_bno055.AXIS_REMAP_POSITIVE,
            )

        # Combined pitch bias = hardware nominal + user-specified adjustment
        self.pitch_bias = self.nominal_pitch_bias + self.user_pitch_bias

        # --- Calibration mode: block until fully calibrated, save offsets, exit ---
        if self.calibrate:
            # NDOF_MODE is required for full calibration (accel + gyro + mag)
            self.imu.mode = adafruit_bno055.NDOF_MODE
            calibrated = self.imu.calibrated
            while not calibrated:
                # calibration_status returns (sys, gyro, accel, mag), each 0-3
                print("Calibration status: ", self.imu.calibration_status)
                print("Calibrated : ", self.imu.calibrated)
                calibrated = self.imu.calibrated
                time.sleep(0.1)
            print("CALIBRATION DONE")

            # Read the computed sensor offsets, retrying until all are valid.
            # The sensor registers may not be ready immediately after
            # calibration completes, so we poll until we get real values.
            # Keep reading until all three return real values.
            while True:
                offsets_accelerometer = self.imu.offsets_accelerometer
                offsets_gyroscope = self.imu.offsets_gyroscope
                offsets_magnetometer = self.imu.offsets_magnetometer
                if (
                    offsets_accelerometer is not None
                    and offsets_gyroscope is not None
                    and offsets_magnetometer is not None
                ):
                    break
                print("Waiting for offset registers to be ready...")
                time.sleep(0.1)

            imu_calib_data = {
                "offsets_accelerometer": offsets_accelerometer,
                "offsets_gyroscope": offsets_gyroscope,
                "offsets_magnetometer": offsets_magnetometer,
            }
            for k, v in imu_calib_data.items():
                print(k, v)

            pickle.dump(imu_calib_data, open("imu_calib_data.pkl", "wb"))

            print("Saved", "imu_calib_data.pkl")
            exit()

        # --- Normal mode: restore saved calibration offsets if available ---
        if os.path.exists("imu_calib_data.pkl"):
            imu_calib_data = pickle.load(open("imu_calib_data.pkl", "rb"))
            # Must enter CONFIG_MODE to write offset registers.
            # Wait 600ms after each mode switch — the BNO055 datasheet
            # specifies this as the time needed for the sensor to be ready.
            self.imu.mode = adafruit_bno055.CONFIG_MODE
            print("[IMU MODE]: CONFIG_MODE (writing saved offsets)")
            time.sleep(0.05)
            self.imu.offsets_accelerometer = imu_calib_data["offsets_accelerometer"]
            self.imu.offsets_gyroscope = imu_calib_data["offsets_gyroscope"]
            self.imu.offsets_magnetometer = imu_calib_data["offsets_magnetometer"]
            # Switch back to operating mode after writing offsets
            self.imu.mode = adafruit_bno055.IMUPLUS_MODE
            print("[IMU MODE]: IMUPLUS_MODE (offsets loaded, ready)")
            time.sleep(0.6)
        else:
            print("imu_calib_data.pkl not found")
            print("Imu is running uncalibrated")

        self.last_imu_data = [0, 0, 0, 0]
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def convert_axes(self, euler):
        euler = [np.pi + euler[1], euler[0], euler[2]]
        return euler

    def imu_worker(self):
        """Background thread: polls quaternion orientation at sampling_freq.

        Reads the fused quaternion from the BNO055 (scalar-first convention),
        converts to Euler angles, applies pitch bias, then re-encodes as a
        quaternion in scalar-last convention for downstream use.
        """
        while True:
            s = time.time()
            try:
                # Read raw quaternion tuple first to check for None values
                # before wrapping in np.array.
                # BNO055 returns quaternion in scalar-first order (w, x, y, z).
                raw_quat = self.imu.quaternion
                if raw_quat is None or None in raw_quat:
                    print(f"[IMU SKIP]: quaternion has None — raw={raw_quat}")
                    continue
                raw_orientation = np.array(raw_quat).copy()
                euler = (
                    R.from_quat(raw_orientation, scalar_first=True)
                    .as_euler("xyz")
                    .copy()
                )
            except Exception as e:
                print("[IMU]:", e)
                continue

            # Apply pitch bias correction (in radians)
            euler[1] -= np.deg2rad(self.pitch_bias)

            # Re-encode as quaternion in scalar-last order (x, y, z, w)
            # which is what Isaac Sim / downstream consumers expect
            final_orientation_quat = R.from_euler("xyz", euler).as_quat()

            self.imu_queue.put(final_orientation_quat.copy())
            took = time.time() - s
            time.sleep(max(0, 1 / self.sampling_freq - took))

    def get_data(self, euler=False, mat=False):
        """Return the most recent orientation (non-blocking).

        Args:
            euler: If True, return Euler angles (xyz) instead of quaternion.
            mat:   If True, return a 3x3 rotation matrix instead of quaternion.

        Returns the previous reading if no new data is available.
        """
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        try:
            if not euler and not mat:
                return self.last_imu_data
            elif euler:
                return R.from_quat(self.last_imu_data).as_euler("xyz")
            elif mat:
                return R.from_quat(self.last_imu_data).as_matrix()

        except Exception as e:
            print("[IMU]: ", e)
            return None


if __name__ == "__main__":
    imu = Imu(50, calibrate=True, upside_down=False)
    # imu = Imu(50, upside_down=False)
    while True:
        data = imu.get_data()
        # print(data)
        print("gyro", np.around(data["gyro"], 3))
        print("accelero", np.around(data["accelero"], 3))
        print("---")
        time.sleep(1 / 25)
