"""
IMU Calibration Script

Runs the BNO055 IMU through its full NDOF calibration procedure.
The user must physically move the robot through specific orientations
until the sensor reports fully calibrated (sys, gyro, accel, mag all = 3).

Once calibrated, the accelerometer, gyroscope, and magnetometer offsets
are saved to 'imu_calib_data.pkl' so subsequent runs can restore them
without re-calibrating.

Usage:
    python calibrate_imu.py

The script exits automatically after saving calibration data.
"""

from mini_bdx_runtime.raw_imu import Imu

if __name__ == "__main__":
    # calibrate=True enters the calibration loop, then saves offsets and exits.
    # upside_down=False assumes the IMU is mounted right-side up during calibration.
    imu = Imu(50, calibrate=True, upside_down=False)