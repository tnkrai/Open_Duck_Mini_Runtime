"""
Raw IMU driver for the BNO055 sensor.

Provides raw gyroscope and accelerometer readings (as opposed to imu.py which
provides fused quaternion orientation).  Runs a background thread that polls
the sensor at a configurable frequency and exposes the latest reading via
get_data().

Calibration flow (calibrate=True):
  1. Switch to NDOF_MODE (uses all sensors for full calibration).
  2. Block until the BNO055 reports fully calibrated on all subsystems
     (calibration_status returns (3, 3, 3, 3) for sys, gyro, accel, mag).
  3. Read the computed offsets for accelerometer, gyroscope, and magnetometer.
  4. Save offsets to 'imu_calib_data.pkl' and exit.

Normal operation (calibrate=False):
  - If 'imu_calib_data.pkl' exists, load saved offsets, briefly enter
    CONFIG_MODE to write them into the sensor, then switch back to NDOF_MODE.
  - If no calibration file exists, run uncalibrated (offsets default to zero).
"""

import adafruit_bno055
import board
import busio
import numpy as np
import os
import pickle

from queue import Queue
from threading import Thread
import time


# TODO filter spikes
class Imu:
    def __init__(
        self, sampling_freq, user_pitch_bias=0, calibrate=False, upside_down=True
    ):
        self.sampling_freq = sampling_freq
        self.calibrate = calibrate

        # Initialize I2C bus and BNO055 sensor
        i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = adafruit_bno055.BNO055_I2C(i2c)

        # NDOF_MODE fuses accel + gyro + magnetometer for absolute orientation.
        # Other modes are available but commented out for reference.
        self.imu.mode = adafruit_bno055.NDOF_MODE
        print("[IMU MODE]: NDOF_MODE (startup)")

        # Remap physical axes to match the robot's coordinate frame.
        # The BNO055 has a default axis orientation that may not match how
        # the sensor is physically mounted on the robot.  The remap swaps
        # X<->Y and flips sign bits depending on whether the board is
        # mounted upside-down.
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

        # --- Calibration mode: block until fully calibrated, save offsets, exit ---
        if self.calibrate:
            self.imu.mode = adafruit_bno055.NDOF_MODE
            calibrated = self.imu.calibrated
            while not calibrated:
                # calibration_status returns a tuple (sys, gyro, accel, mag)
                # each ranging from 0 (uncalibrated) to 3 (fully calibrated).
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
            self.imu.mode = adafruit_bno055.NDOF_MODE
            print("[IMU MODE]: NDOF_MODE (offsets loaded, ready)")
            time.sleep(0.6)
        else:
            print("imu_calib_data.pkl not found")
            print("Imu is running uncalibrated")

        self.x_offset = 0

        # self.tare_x()

        self.last_imu_data = {
            "gyro": [0, 0, 0],
            "accelero": [0, 0, 0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],  # w, x, y, z (identity)
            # None until a fused read has actually succeeded — see _read_quaternion
            "quaternion_t": None,
        }
        # fused orientation for viewers/telemetry; the policy only uses gyro+accel
        self._last_quat = [1.0, 0.0, 0.0, 0.0]
        self._last_quat_t = None
        self._stop = False
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def stop(self):
        """Stop the worker so another owner (the walk subprocess) can have the
        BNO055's I2C to itself — same exclusivity rule as the servo bus."""
        self._stop = True

    def tare_x(self):
        """Compute a static X-axis accelerometer offset (tare).

        Collects 100 samples and waits until the standard deviation is
        below 0.05 (i.e. the robot is stationary), then stores the mean
        as self.x_offset to subtract from future readings.
        """
        print("Taring x ...")
        x_values = []
        num_values = 100
        ok = False
        while not ok:
            x_values.append(np.array(self.imu.acceleration)[0])

            x_values = x_values[-num_values:]

            if len(x_values) == num_values:
                mean = np.mean(x_values)
                std = np.std(x_values)
                if std < 0.05:
                    ok = True
                    self.x_offset = mean
                    print("Tare x done")
                else:
                    print(std)

            time.sleep(0.01)

    def imu_worker(self):
        """Background thread: polls gyro and accelerometer at sampling_freq.

        Reads raw gyroscope and accelerometer values from the BNO055,
        applies the X-axis tare offset, and pushes the result into a
        single-slot queue for consumption by get_data().
        """
        while not self._stop:
            s = time.time()
            try:
                # Read raw tuples first so we can check for None before
                # wrapping in np.array (which would hide the None values).
                raw_gyro = self.imu.gyro
                raw_accelero = self.imu.acceleration
            except Exception as e:
                print("[IMU]:", e)
                time.sleep(0.1)
                continue

            # Skip this reading if the sensor returned None for any axis
            if raw_gyro is None or raw_accelero is None:
                print("[IMU SKIP]: entire reading was None (sensor not ready)")
                continue
            if None in raw_gyro or None in raw_accelero:
                print(f"[IMU SKIP]: partial None — gyro={raw_gyro}, accel={raw_accelero}")
                continue

            gyro = np.array(raw_gyro).copy()
            accelero = np.array(raw_accelero).copy()

            accelero[0] -= self.x_offset

            self._read_quaternion()

            data = {
                "gyro": gyro,
                "accelero": accelero,
                "quaternion": list(self._last_quat),
                # When that quaternion was last actually read from the chip. A
                # consumer that only checks for None cannot tell a level duck from
                # a fused read that stopped answering: the quaternion above is the
                # previous value either way (see _read_quaternion).
                "quaternion_t": self._last_quat_t,
            }

            self.imu_queue.put(data)
            took = time.time() - s
            time.sleep(max(0, 1 / self.sampling_freq - took))

    def _read_quaternion(self):
        """Refresh the fused orientation, and record WHEN it was refreshed.

        A failed read keeps the previous value — never skip the whole sample over
        it, the policy's gyro/accel must keep flowing.  But the previous value
        must not pass for a fresh one: the walk's tilt guard treats an unknown
        orientation as unsafe, and without the timestamp there is no unknown to
        report.  The BNO055 can lose its fused output (a mode or fusion fault)
        while gyro and accel keep streaming, and _last_quat starts at identity —
        so a duck that has fallen would keep reporting itself perfectly level.
        """
        try:
            raw_quat = self.imu.quaternion
        except Exception:
            return
        if raw_quat is None or None in raw_quat:
            return
        self._last_quat = [float(q) for q in raw_quat]
        self._last_quat_t = time.monotonic()

    def get_data(self):
        """Return the most recent IMU reading (non-blocking).

        Returns a dict with 'gyro' and 'accelero' keys.  If no new data
        is available from the worker thread, returns the previous reading.
        """
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_imu_data


if __name__ == "__main__":
    imu = Imu(50, upside_down=False)
    while True:
        data = imu.get_data()
        # print(data)
        print("gyro", np.around(data["gyro"], 3))
        print("accelero", np.around(data["accelero"], 3))
        print("---")
        time.sleep(1 / 25)
