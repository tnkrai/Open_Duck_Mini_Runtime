#!/usr/bin/env python3
"""
calibrate_servo_zero.py — re-home each Feetech STS3215 servo's ZERO *in the
servo's own firmware* (the Position Correction register, addr 31), instead of
carrying the correction as a software offset in duck_config.json.

Why you'd do this
-----------------
A software offset (duck_config.json -> joints_offsets) is added to every goal
position. If init_pos + offset ever exceeds the servo's representable command
range (±180° / ±π rad, the seam at the 0/4096-count boundary), the value wraps
to a different physical position. Pushing the *servo's* zero instead keeps the
correction inside the servo, applied in count-space, so it can't wrap and the
config offsets stay ~0.

This is the firmware equivalent of find_soft_offsets.py.

How it works (per joint)
------------------------
  1. Torque is disabled so you can hand-pose the joint.
  2. You move the joint to its MECHANICAL ZERO (the URDF zero pose — the same
     pose find_soft_offsets.py calls zero_pos) and confirm.
  3. The script measures how the Position Correction register moves the
     reported position (it probes a small step and checks the slope), then
     solves for the correction that makes the joint read 0° at the pose you're
     holding, writes it to EEPROM (unlock -> write -> lock), and VERIFIES by
     re-reading. The empirical slope means it works regardless of the
     register's documented sign convention, and it aborts safely if the servo
     doesn't behave as expected.
  4. The joint's software offset in duck_config.json is then zeroed (a backup
     is written first), because the servo now self-corrects and rustypot reads
     the corrected position at runtime.

Caveats
-------
  * Writes servo EEPROM (limited write cycles) — fine occasionally, don't loop.
  * The correction register spans only ±180°. If a horn is seated more than a
    half-turn off, you must physically re-seat it; the script will say so.
  * Stop walk_server.py / any other process using the bus before running this
    (the serial port can only be opened once).

Usage:
    python scripts/calibrate_servo_zero.py
    python scripts/calibrate_servo_zero.py --serial_port /dev/ttyACM0
    python scripts/calibrate_servo_zero.py --no_config_update   # don't touch json
"""
import argparse
import json
import os
import shutil
import time

from pypot.feetech import FeetechSTS3215IO

HOME_DIR = os.path.expanduser("~")

# Joint name -> servo id, same order as HWI.joints in
# mini_bdx_runtime/rustypot_position_hwi.py (kept local so this script can run
# without opening the rustypot HWI, which would claim the serial port).
JOINTS = {
    "left_hip_yaw": 20,
    "left_hip_roll": 21,
    "left_hip_pitch": 22,
    "left_knee": 23,
    "left_ankle": 24,
    "neck_pitch": 30,
    "head_pitch": 31,
    "head_yaw": 32,
    "head_roll": 33,
    "right_hip_yaw": 10,
    "right_hip_roll": 11,
    "right_hip_pitch": 12,
    "right_knee": 13,
    "right_ankle": 14,
}

VERIFY_TOL_DEG = 2.0   # acceptable |present| after zeroing (hand-held wobble)
PROBE_DEG = 10.0       # step used to measure correction-register -> feedback gain
LIMIT_DEG = 180.0      # correction register range (pypot's full-scale for addr 31)
MAX_ITERS = 6          # Newton steps to drive present -> 0


def _r(x):
    return round(float(x), 2)


def calibrate_joint(io, name, sid):
    """Zero this servo at its current physical pose. Returns (ok, residual_deg).

    The Position Correction register (addr 31) shifts the reported position
    LINEARLY, but pypot's degree conversion for that register doesn't match the
    firmware's encoding, so the gain is not 1:1 (observed ~9x). We therefore
    measure the gain empirically (probe one small step) and then Newton-iterate
    to drive the reported position to 0, re-reading after every write. This is
    encoding-agnostic and self-correcting; it only gives up if the register
    doesn't move the feedback at all, or if a step diverges (bad gain estimate).
    """
    io.set_lock({sid: 0})            # unlock EEPROM writes
    time.sleep(0.1)

    o0 = io.get_offset([sid])[0]
    p0 = io.get_present_position([sid])[0]
    print(f"    start: present={_r(p0)}°  correction_reg={_r(o0)}°")

    # Probe toward center (away from the ±180° register seam) to measure gain.
    probe = -PROBE_DEG if o0 > 0 else PROBE_DEG
    io.set_offset({sid: _r(o0 + probe)})
    time.sleep(0.15)
    p1 = io.get_present_position([sid])[0]
    slope = (p1 - p0) / probe        # reg-degrees -> feedback-degrees gain

    if not (0.2 <= abs(slope) <= 200.0):
        io.set_offset({sid: _r(o0)})  # restore
        io.set_lock({sid: 1})
        print(f"    !! correction register barely moves the feedback "
              f"(slope={slope:.2f}). Skipping {name} — left untouched.")
        return False, None

    # Newton: present(o) ~= pv + slope*(o - o_cur); step o by -pv/slope to hit 0.
    o, pv = o0 + probe, p1
    for _ in range(MAX_ITERS):
        if abs(pv) <= VERIFY_TOL_DEG:
            break
        o_new = max(-LIMIT_DEG, min(LIMIT_DEG, o - pv / slope))
        if o_new == o:               # clamped at register limit; can't improve
            break
        io.set_offset({sid: _r(o_new)})
        time.sleep(0.15)
        pv_new = io.get_present_position([sid])[0]
        if abs(pv_new) > abs(pv) + 1.0:   # diverging -> gain estimate is bad
            io.set_offset({sid: _r(o)})    # revert to best
            time.sleep(0.1)
            pv = io.get_present_position([sid])[0]
            break
        o, pv = o_new, pv_new

    io.set_lock({sid: 1})            # re-lock EEPROM
    time.sleep(0.1)

    ok = abs(pv) <= VERIFY_TOL_DEG
    print(f"    -> correction_reg={_r(io.get_offset([sid])[0])}°  "
          f"present now {_r(pv)}°  [{'OK' if ok else 'RESIDUAL'}]")
    if abs(o) >= LIMIT_DEG - 0.1:
        print("    !! correction hit the register limit — the horn is seated too "
              "far off. Re-seat it physically closer to zero, then re-run.")
    return ok, _r(pv)


def update_config(path, calibrated):
    """Zero the software offsets for joints we just re-homed in firmware."""
    try:
        with open(path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"  config {path} not found; skipping config update.")
        return
    backup = path + ".bak"
    shutil.copyfile(path, backup)
    offs = cfg.get("joints_offsets", {})
    for name in calibrated:
        offs[name] = 0.0
    cfg["joints_offsets"] = offs
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  Zeroed software offsets for {len(calibrated)} joint(s) in {path}")
    print(f"  (backup saved to {backup})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duck_config_path", default=f"{HOME_DIR}/duck_config.json")
    ap.add_argument("--serial_port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--no_config_update", action="store_true",
                    help="don't zero the software offsets in duck_config.json")
    args = ap.parse_args()

    print("==========================================================")
    print(" Firmware servo-zero calibration")
    print("==========================================================")
    print("Make sure no other process is using the servo bus (stop")
    print("walk / tnkr_server.py first), and that you can support")
    print("the robot — joints go LIMP one at a time.\n")
    print(f"Opening {args.serial_port} @ {args.baud} ...")
    io = FeetechSTS3215IO(args.serial_port, baudrate=args.baud)

    calibrated = []
    try:
        for name, sid in JOINTS.items():
            print(f"\n=== {name}  (id {sid}) ===")
            res = input("  Calibrate this joint? [Enter=yes / s=skip / q=quit]: ").strip().lower()
            if res in ("q", "quit"):
                break
            if res in ("s", "skip"):
                continue

            io.disable_torque([sid])
            input(f"  {name} is now LIMP. Move it to its MECHANICAL ZERO and "
                  f"hold it steady, then press Enter...")
            ok, _ = calibrate_joint(io, name, sid)
            if ok:
                calibrated.append(name)
                # hold at the freshly-set zero so the pose builds up
                io.set_goal_position({sid: 0.0})
                io.enable_torque([sid])

        print("\n==========================================================")
        print(f" Done. Re-homed {len(calibrated)}/{len(JOINTS)} joints:")
        for n in calibrated:
            print(f"   - {n}")
        if not args.no_config_update and calibrated:
            update_config(args.duck_config_path, calibrated)
        elif args.no_config_update:
            print("  (--no_config_update: left duck_config.json untouched. Set the")
            print("   re-homed joints' offsets to 0 yourself, or they'll double-correct.)")

        input("\nPress Enter to RELEASE all torque (robot will go limp)...")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for sid in JOINTS.values():
            try:
                io.disable_torque([sid])
            except Exception:
                pass
        print("All torque off.")


if __name__ == "__main__":
    main()
