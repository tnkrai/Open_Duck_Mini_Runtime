"""
Fake telemetry broadcaster — simulates a walking robot sending data
to a Supabase Realtime channel so the dashboard live viewer can be tested
without a physical robot.

Usage:
    python fake_broadcaster.py <session_token>

Example:
    python fake_broadcaster.py 84c412c3-5260-492e-a8da-c6d4c9b31cec
"""

import sys
import math
import time

# Reuse the existing CloudPublisher
sys.path.insert(0, "../mini_bdx_runtime")
from mini_bdx_runtime.cloud_publisher import CloudPublisher

# Joint names matching the real robot (14 joints, no antennas)
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "head_yaw", "head_pitch",
]


def generate_walking_state(t: float) -> dict:
    """Generate a fake walking state with sinusoidal joint motion."""
    freq = 1.5  # walking frequency Hz
    phase = 2 * math.pi * freq * t

    joint_positions = {}
    joint_velocities = {}
    motor_targets = {}

    for i, name in enumerate(JOINT_NAMES):
        # Legs get walking motion, head gets gentle sway
        if "head" in name:
            amp = 0.1
            offset = 0.0
        elif "hip_pitch" in name:
            amp = 0.4
            offset = -0.3
        elif "knee" in name:
            amp = 0.5
            offset = 0.6
        elif "ankle_pitch" in name:
            amp = 0.3
            offset = -0.2
        else:
            amp = 0.1
            offset = 0.0

        # Left and right legs are 180° out of phase
        leg_phase = phase if "left" in name else phase + math.pi
        pos = offset + amp * math.sin(leg_phase + i * 0.1)
        vel = amp * freq * 2 * math.pi * math.cos(leg_phase + i * 0.1)

        joint_positions[name] = pos
        joint_velocities[name] = vel
        motor_targets[name] = pos + 0.01 * math.sin(phase * 3)

    # Simulated IMU — gentle body sway
    roll = 0.05 * math.sin(phase)
    pitch = 0.03 * math.sin(phase * 2)
    # Simple quaternion from small roll/pitch
    qx = math.sin(roll / 2)
    qy = math.sin(pitch / 2)
    qz = 0.0
    qw = math.cos(roll / 2) * math.cos(pitch / 2)

    return {
        "timestamp": time.time(),
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "motor_targets": motor_targets,
        "imu": {
            "quaternion": [qx, qy, qz, qw],
            "gyro": [
                0.05 * math.cos(phase),
                0.03 * math.cos(phase * 2),
                0.01 * math.sin(phase),
            ],
            "accelero": [
                0.1 * math.sin(phase),
                0.0,
                9.81 + 0.05 * math.sin(phase * 2),
            ],
        },
        "feet_contacts": {
            "left": math.sin(phase) > 0,
            "right": math.sin(phase + math.pi) > 0,
        },
        "paused": False,
        "fps": 50,
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: python fake_broadcaster.py <session_token> <supabase_url> <supabase_key>")
        sys.exit(1)

    session_token = sys.argv[1]
    supabase_url = sys.argv[2]
    supabase_key = sys.argv[3]
    channel_name = f"robot-telemetry-{session_token}"

    print(f"[FakeBroadcaster] Starting on channel: {channel_name}")
    publisher = CloudPublisher(channel_name, supabase_url, supabase_key, target_hz=10)
    publisher.start()

    print("[FakeBroadcaster] Broadcasting fake walking data at ~50 Hz (throttled to 10 Hz)...")
    print("[FakeBroadcaster] Press Ctrl+C to stop")

    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            state = generate_walking_state(t)
            publisher.publish(state)
            time.sleep(0.02)  # 50 Hz loop
    except KeyboardInterrupt:
        print("\n[FakeBroadcaster] Stopping...")
        publisher.stop()
        print("[FakeBroadcaster] Done.")


if __name__ == "__main__":
    main()
