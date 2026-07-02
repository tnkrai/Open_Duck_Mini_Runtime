"""
TNKR Robot Server

HTTP API server for the Open Duck Mini robot. Exposes motor check,
servo rehoming (firmware zero), stance calibration, config management,
and walk control endpoints.
Telemetry is streamed via Supabase Realtime broadcast channels.
"""

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread, Lock

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# ── Runtime imports (available after pip install -e .) ────────────────────────

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig

# ── Constants ─────────────────────────────────────────────────────────────────

HOME_DIR = os.path.expanduser("~")
CONFIG_PATH = os.path.join(HOME_DIR, "duck_config.json")
SCRIPTS_DIR = Path(__file__).parent
SERVER_PORT = 8000
USB_PORT = "/dev/ttyACM0"

# Joint name -> servo id, same mapping as HWI.joints. Kept as a module
# constant so the rehoming flow can address servos without opening the
# rustypot HWI (the serial port only supports one owner at a time).
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

# ── Shared state ──────────────────────────────────────────────────────────────

hwi_instance: HWI | None = None
walk_process: subprocess.Popen | None = None
current_session_token: str | None = None
rehome_io = None  # pypot FeetechSTS3215IO while a rehoming session is open


def get_hwi() -> HWI:
    """Get or create the HWI singleton. Reuses existing connection."""
    global hwi_instance
    if rehome_io is not None:
        raise RuntimeError(
            "Servo rehoming session in progress — finish it before other motor operations"
        )
    if hwi_instance is None:
        config = DuckConfig(config_json_path=CONFIG_PATH, ignore_default=True)
        hwi_instance = HWI(duck_config=config, usb_port=USB_PORT)
    return hwi_instance


def release_hwi():
    """Release the HWI connection so the walk script can use it."""
    global hwi_instance
    if hwi_instance is not None:
        try:
            hwi_instance.turn_off()
        except Exception:
            pass
        hwi_instance = None


# ── Request/Response models ───────────────────────────────────────────────────

class JointRequest(BaseModel):
    jointName: str


class ApplyOffsetRequest(BaseModel):
    jointName: str
    offset: float


class AcceptJointRequest(BaseModel):
    jointName: str
    offset: float


class DuckConfigModel(BaseModel):
    start_paused: bool = False
    imu_upside_down: bool = False
    phase_frequency_factor_offset: float = 0.0
    expression_features: dict = {}
    joints_offsets: dict = {}


class CommandRequest(BaseModel):
    commands: list[float]
    buttons: dict = {}
    left_trigger: float = 0.0
    right_trigger: float = 0.0


COMMAND_FILE = "/dev/shm/tnkr_remote_commands.json"
_command_file_lock = Lock()


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _close_rehome_io()
    release_hwi()
    stop_walk_process()


def _close_rehome_io():
    """Drop torque and free the bus if a rehoming session is still open."""
    global rehome_io
    if rehome_io is None:
        return
    io = rehome_io
    rehome_io = None
    for sid in JOINTS.values():
        try:
            io.disable_torque([sid])
        except Exception:
            pass
    try:
        io.close()
    except Exception:
        pass


app = FastAPI(title="TNKR Robot Server", lifespan=lifespan)


class PrivateNetworkMiddleware(BaseHTTPMiddleware):
    """Handle Chrome's Private Network Access preflight requests.

    HTTPS pages accessing local network devices trigger a preflight with
    Access-Control-Request-Private-Network: true. The server must respond
    with Access-Control-Allow-Private-Network: true or Chrome blocks it.
    """

    async def dispatch(self, request: Request, call_next):
        if (
            request.method == "OPTIONS"
            and request.headers.get("access-control-request-private-network") == "true"
        ):
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Private-Network": "true",
                },
            )
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response


# PrivateNetworkMiddleware must be added first (outermost) so it handles
# preflight OPTIONS before CORSMiddleware can reject them.
app.add_middleware(PrivateNetworkMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    machine = platform.machine()  # 'aarch64' / 'armv7l' on Pi, 'x86_64' / 'arm64' on Mac
    is_pi = machine in ("aarch64", "armv7l")
    return {"status": "ok", "is_pi": is_pi, "platform": machine}


# ── Motor Check ───────────────────────────────────────────────────────────────

@app.post("/api/motors/check")
def check_motors():
    """Check all 14 motors for responsiveness."""
    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to motor controller: {e}",
        )

    motors = []
    for joint_name, joint_id in hwi.joints.items():
        motor_info = {
            "jointName": joint_name,
            "servoId": joint_id,
            "responsive": False,
            "position": None,
            "error": None,
        }
        try:
            hwi.io.set_kps([joint_id], [hwi.low_torque_kps[0]])
            position = hwi.io.read_present_position([joint_id])
            motor_info["responsive"] = True
            motor_info["position"] = round(float(position[0]), 3)
        except Exception as e:
            motor_info["error"] = str(e)
        motors.append(motor_info)

    # Disable torque after check
    for joint_name, joint_id in hwi.joints.items():
        try:
            hwi.io.disable_torque([joint_id])
        except Exception:
            pass

    all_responsive = all(m["responsive"] for m in motors)
    return {"motors": motors, "allResponsive": all_responsive}


# ── Calibration (DEPRECATED) ──────────────────────────────────────────────────
# The per-joint "capture zero as a software offset" flow below is superseded by
# the rehome (/api/rehome/*) + stance (/api/stance/*) endpoints. Big software
# offsets can push init_pos + offset past the servo's ±π command seam, where
# the value wraps to a different physical position (joints jam against the
# shell and the firmware cuts torque). Kept for older dashboard clients.

# Temporary calibration state
calibration_offsets: dict[str, float] = {}


@app.post("/api/calibration/start")
def calibration_start():
    """Move all joints to zero position and prepare for calibration."""
    global calibration_offsets

    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Reset offsets to zero for calibration
    for joint_name in hwi.joints:
        hwi.joints_offsets[joint_name] = 0
    calibration_offsets = {}

    # Move to zero with low damping
    hwi.set_kds([0] * len(hwi.joints))
    hwi.init_pos = hwi.zero_pos
    hwi.turn_on()
    hwi.set_position_all(hwi.zero_pos)
    time.sleep(1)

    # Read current positions
    positions = hwi.get_present_positions()
    joint_names = list(hwi.joints.keys())
    current_positions = {}
    if positions is not None:
        for i, name in enumerate(joint_names):
            current_positions[name] = round(float(positions[i]), 3)

    return {
        "joints": joint_names,
        "currentPositions": current_positions,
    }


@app.post("/api/calibration/begin-joint")
def calibration_begin_joint(req: JointRequest):
    """Disable torque on a joint so the user can move it by hand."""
    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    if req.jointName not in hwi.joints:
        raise HTTPException(status_code=400, detail=f"Unknown joint: {req.jointName}")

    joint_id = hwi.joints[req.jointName]

    # Move all to zero first (in case previous joint moved things)
    hwi.set_position_all(hwi.zero_pos)
    time.sleep(0.5)

    # Disable torque on this joint
    hwi.io.disable_torque([joint_id])

    return {"success": True}


@app.post("/api/calibration/confirm-position")
def calibration_confirm_position(req: JointRequest):
    """Read the joint's current position and calculate the offset."""
    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    if req.jointName not in hwi.joints:
        raise HTTPException(status_code=400, detail=f"Unknown joint: {req.jointName}")

    joint_names = list(hwi.joints.keys())
    joint_index = joint_names.index(req.jointName)

    # The zero position command was 0.0 (since offsets are zeroed)
    current_command = 0.0

    positions = hwi.get_present_positions()
    if positions is None:
        raise HTTPException(status_code=503, detail="Could not read motor position")

    new_pos = float(positions[joint_index])
    offset = new_pos - current_command

    return {
        "jointName": req.jointName,
        "offset": round(offset, 4),
        "previousPosition": round(current_command, 4),
        "newPosition": round(new_pos, 4),
    }


@app.post("/api/calibration/apply-offset")
def calibration_apply_offset(req: ApplyOffsetRequest):
    """Apply the offset and move joint to zero to verify."""
    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    if req.jointName not in hwi.joints:
        raise HTTPException(status_code=400, detail=f"Unknown joint: {req.jointName}")

    joint_id = hwi.joints[req.jointName]

    # Apply offset
    hwi.joints_offsets[req.jointName] = req.offset

    # Re-enable torque and move to zero (with offset applied)
    hwi.io.enable_torque([joint_id])
    hwi.set_position_all(hwi.zero_pos)
    time.sleep(0.5)

    return {"success": True}


@app.post("/api/calibration/accept")
def calibration_accept(req: AcceptJointRequest):
    """Accept the offset for a joint."""
    global calibration_offsets

    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    if req.jointName not in hwi.joints:
        raise HTTPException(status_code=400, detail=f"Unknown joint: {req.jointName}")

    hwi.joints_offsets[req.jointName] = req.offset
    calibration_offsets[req.jointName] = req.offset

    return {"success": True, "offsets": calibration_offsets}


@app.post("/api/calibration/save")
def calibration_save():
    """Save all calibration offsets to duck_config.json."""
    global calibration_offsets

    try:
        config = _read_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Merge calibration offsets into config
    if "joints_offsets" not in config:
        config["joints_offsets"] = {}
    config["joints_offsets"].update(calibration_offsets)

    try:
        _write_config(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Reload HWI with new config
    release_hwi()

    return {"success": True, "offsets": config["joints_offsets"]}


# ── Servo zero rehoming (firmware) ────────────────────────────────────────────
# Ported from scripts/calibrate_servo_zero.py: write the STS3215 Position
# Correction register (addr 31, EEPROM) so each servo reads 0 at the robot's
# mechanical zero. The correction then lives inside the servo in count-space
# where it can't wrap, and the software offsets in duck_config.json stay ~0.
#
# Uses pypot (not the rustypot HWI) because it exposes the correction/lock
# registers; the HWI is released for the duration of the session since the
# serial port only supports one owner.

REHOME_VERIFY_TOL_DEG = 2.0   # acceptable |present| after zeroing (hand wobble)
REHOME_PROBE_DEG = 10.0       # step used to measure register -> feedback gain
REHOME_LIMIT_DEG = 180.0      # correction register full-scale (pypot degrees)
REHOME_MAX_ITERS = 6          # Newton steps to drive present -> 0

rehome_config_backed_up = False


def _require_rehome_io():
    if rehome_io is None:
        raise HTTPException(
            status_code=409,
            detail="No rehoming session — call /api/rehome/start first",
        )
    return rehome_io


def _rehome_calibrate_joint(io, sid: int) -> dict:
    """Zero this servo at its current physical pose.

    The Position Correction register shifts the reported position LINEARLY,
    but pypot's degree conversion for that register doesn't match the
    firmware's encoding (observed ~9x). So the gain is measured empirically
    (probe one small step) and then Newton-iterated to drive the reported
    position to 0, re-reading after every write. Encoding-agnostic and
    self-correcting; gives up only if the register doesn't move the feedback
    at all, or a step diverges (bad gain estimate).
    """
    io.set_lock({sid: 0})            # unlock EEPROM writes
    time.sleep(0.1)

    o0 = float(io.get_offset([sid])[0])
    p0 = float(io.get_present_position([sid])[0])

    # Probe toward center (away from the ±180° register seam) to measure gain.
    probe = -REHOME_PROBE_DEG if o0 > 0 else REHOME_PROBE_DEG
    io.set_offset({sid: round(o0 + probe, 2)})
    time.sleep(0.15)
    p1 = float(io.get_present_position([sid])[0])
    slope = (p1 - p0) / probe        # reg-degrees -> feedback-degrees gain

    if not (0.2 <= abs(slope) <= 200.0):
        io.set_offset({sid: round(o0, 2)})  # restore
        io.set_lock({sid: 1})
        return {
            "ok": False,
            "error": "The correction register barely moves the feedback — servo left untouched",
            "residualDeg": None,
            "correctionDeg": round(o0, 2),
            "hitLimit": False,
        }

    # Newton: present(o) ~= pv + slope*(o - o_cur); step o by -pv/slope to hit 0.
    o, pv = o0 + probe, p1
    for _ in range(REHOME_MAX_ITERS):
        if abs(pv) <= REHOME_VERIFY_TOL_DEG:
            break
        o_new = max(-REHOME_LIMIT_DEG, min(REHOME_LIMIT_DEG, o - pv / slope))
        if o_new == o:               # clamped at register limit; can't improve
            break
        io.set_offset({sid: round(o_new, 2)})
        time.sleep(0.15)
        pv_new = float(io.get_present_position([sid])[0])
        if abs(pv_new) > abs(pv) + 1.0:   # diverging -> gain estimate is bad
            io.set_offset({sid: round(o, 2)})  # revert to best
            time.sleep(0.1)
            pv = float(io.get_present_position([sid])[0])
            break
        o, pv = o_new, pv_new

    io.set_lock({sid: 1})            # re-lock EEPROM
    time.sleep(0.1)

    return {
        "ok": abs(pv) <= REHOME_VERIFY_TOL_DEG,
        "error": None,
        "residualDeg": round(pv, 2),
        "correctionDeg": round(float(io.get_offset([sid])[0]), 2),
        # Correction hit ±180°: the horn is seated more than a half-turn off
        # and must be physically re-seated closer to zero.
        "hitLimit": abs(o) >= REHOME_LIMIT_DEG - 0.1,
    }


def _zero_config_offset(joint_name: str):
    """Zero the joint's software offset — the servo now self-corrects."""
    global rehome_config_backed_up
    try:
        config = _read_config()
    except FileNotFoundError:
        config = DuckConfigModel().model_dump()
    if not rehome_config_backed_up and os.path.exists(CONFIG_PATH):
        shutil.copyfile(CONFIG_PATH, CONFIG_PATH + ".bak")
        rehome_config_backed_up = True
    config.setdefault("joints_offsets", {})[joint_name] = 0.0
    _write_config(config)


@app.post("/api/rehome/start")
def rehome_start():
    """Open a rehoming session: release the HWI, claim the bus via pypot."""
    global rehome_io, rehome_config_backed_up

    if walk_process is not None and walk_process.poll() is None:
        raise HTTPException(
            status_code=409, detail="Cannot rehome while a walk is running"
        )
    if rehome_io is not None:
        return {"joints": list(JOINTS.keys()), "alreadyStarted": True}

    release_hwi()
    try:
        from pypot.feetech import FeetechSTS3215IO

        io = FeetechSTS3215IO(USB_PORT, baudrate=1000000)
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Cannot open servo bus for rehoming: {e}"
        )
    rehome_io = io
    rehome_config_backed_up = False
    return {"joints": list(JOINTS.keys()), "alreadyStarted": False}


@app.post("/api/rehome/begin-joint")
def rehome_begin_joint(req: JointRequest):
    """Release torque on one joint so the user can hand-pose it to mechanical zero."""
    io = _require_rehome_io()
    if req.jointName not in JOINTS:
        raise HTTPException(status_code=400, detail=f"Unknown joint: {req.jointName}")
    try:
        io.disable_torque([JOINTS[req.jointName]])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not release torque: {e}")
    return {"success": True}


@app.post("/api/rehome/set-zero")
def rehome_set_zero(req: JointRequest):
    """Adopt the joint's current hand-held pose as the servo's firmware zero."""
    io = _require_rehome_io()
    if req.jointName not in JOINTS:
        raise HTTPException(status_code=400, detail=f"Unknown joint: {req.jointName}")
    sid = JOINTS[req.jointName]

    try:
        result = _rehome_calibrate_joint(io, sid)
        if result["ok"]:
            # Hold at the fresh zero so the calibrated pose builds up joint by
            # joint, then drop the now-redundant software offset.
            io.set_goal_position({sid: 0.0})
            io.enable_torque([sid])
            _zero_config_offset(req.jointName)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Rehoming failed: {e}")

    return {"jointName": req.jointName, **result}


@app.post("/api/rehome/finish")
def rehome_finish():
    """Close the rehoming session: release all torque, free the bus."""
    if rehome_io is None:
        return {"success": True, "message": "No rehoming session was running"}
    _close_rehome_io()
    return {"success": True}


# ── Stance (initial pose) calibration ─────────────────────────────────────────
# Ported from Sam's walk_server.py offset flow: release all torque, hand-pose
# the whole duck into its standing stance, capture every offset at once
# (offset = raw - init_pos, self-correcting no matter how far the old offsets
# drifted), then hold the pose and fine-tune per-joint offsets live before
# saving to duck_config.json.

# Position-mode STS3215 servos are drivable over ~±π rad. A commanded target
# (init_pos + offset) beyond this can't be reached: the servo clamps/wraps, so
# the held pose won't match what was captured. A hair inside π for margin.
SERVO_RANGE_RAD = 3.05

stance_holding = False


def _stance_unreachable(hwi) -> list[str]:
    """Joints whose commanded target falls outside the servo's drivable window."""
    return [
        name
        for name in hwi.joints
        if abs(float(hwi.init_pos[name]) + float(hwi.joints_offsets.get(name, 0.0)))
        > SERVO_RANGE_RAD
    ]


def _stance_offsets(hwi) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in hwi.joints_offsets.items()}


@app.post("/api/stance/start")
def stance_start():
    """Begin a stance session with a pristine HWI (offsets reloaded from disk)."""
    global stance_holding
    # Reload so leftovers from other flows (e.g. the deprecated calibration
    # endpoints mutate init_pos) can't leak into the stance session.
    try:
        release_hwi()
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    stance_holding = False
    return {
        "joints": list(hwi.joints.keys()),
        "initPos": {k: round(float(v), 4) for k, v in hwi.init_pos.items()},
        "offsets": _stance_offsets(hwi),
        "servoRange": SERVO_RANGE_RAD,
    }


@app.post("/api/stance/release")
def stance_release():
    """Release all torque so the user can hand-pose the whole robot."""
    global stance_holding
    try:
        hwi = get_hwi()
        hwi.turn_off()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    stance_holding = False
    return {"success": True}


@app.post("/api/stance/capture")
def stance_capture():
    """Capture all offsets from the current physical pose.

    get_present_positions() already returns raw - offset_current, so:
        offset_new = offset_current + (present - init_pos) = raw - init_pos
    which makes the robot's CURRENT pose read back as init_pos, no matter how
    far the old offsets had drifted.
    """
    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    present = hwi.get_present_positions()
    if present is None or len(present) != len(hwi.joints):
        raise HTTPException(status_code=503, detail="Could not read servo positions")

    for name, p in zip(hwi.joints.keys(), present):
        cur = float(hwi.joints_offsets.get(name, 0.0))
        hwi.joints_offsets[name] = round(
            cur + float(p) - float(hwi.init_pos[name]), 4
        )

    return {
        "offsets": _stance_offsets(hwi),
        "unreachable": _stance_unreachable(hwi),
    }


@app.post("/api/stance/hold")
def stance_hold():
    """Power on and actively hold init_pos + offsets so the stance is visible."""
    global stance_holding
    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    unreachable = _stance_unreachable(hwi)
    if unreachable:
        # Commanding past the drivable window stalls the servo -> sustained
        # over-current -> firmware cuts torque. Refuse instead.
        raise HTTPException(
            status_code=409,
            detail="Targets beyond servo range for: "
            + ", ".join(unreachable)
            + ". Re-seat the horn or rehome these joints first.",
        )
    try:
        hwi.turn_on()  # low kps -> init_pos (+ captured offsets) -> normal kps
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    stance_holding = True
    return {"success": True}


class StanceOffsetRequest(BaseModel):
    jointName: str
    offset: float


@app.post("/api/stance/offset")
def stance_offset(req: StanceOffsetRequest):
    """Set one joint's offset live; while holding, the joint moves immediately."""
    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    if req.jointName not in hwi.joints:
        raise HTTPException(status_code=400, detail=f"Unknown joint: {req.jointName}")

    init = float(hwi.init_pos[req.jointName])
    target = init + float(req.offset)
    clamped_target = max(-SERVO_RANGE_RAD, min(SERVO_RANGE_RAD, target))
    hwi.joints_offsets[req.jointName] = round(clamped_target - init, 4)

    if stance_holding:
        try:
            hwi.set_position_all(hwi.init_pos)
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    return {
        "success": True,
        "offset": hwi.joints_offsets[req.jointName],
        "target": round(clamped_target, 4),
        "clamped": clamped_target != target,
    }


@app.get("/api/stance/positions")
def stance_positions():
    """Raw servo angles in radians (no offset subtraction), for pose display."""
    try:
        hwi = get_hwi()
        raw = hwi.io.read_present_position(list(hwi.joints.values()))
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "positions": {
            name: round(float(p), 4) for name, p in zip(hwi.joints.keys(), raw)
        }
    }


@app.post("/api/stance/save")
def stance_save():
    """Persist the session's offsets to duck_config.json (with a .bak backup)."""
    global stance_holding
    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    offsets = _stance_offsets(hwi)
    try:
        config = _read_config()
    except FileNotFoundError:
        config = DuckConfigModel().model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if os.path.exists(CONFIG_PATH):
        shutil.copyfile(CONFIG_PATH, CONFIG_PATH + ".bak")
    config["joints_offsets"] = offsets
    try:
        _write_config(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    stance_holding = False
    release_hwi()  # robot goes limp; next use reloads the saved config

    return {"success": True, "offsets": offsets}


# ── IMU Calibration ───────────────────────────────────────────────────────

imu_calib_thread: Thread | None = None
imu_calib_status: dict = {
    "running": False,
    "calibration_status": [0, 0, 0, 0],
    "calibrated": False,
    "error": None,
    "offsets": None,
}


def _imu_calibrate_worker():
    """Run IMU calibration in a background thread."""
    global imu_calib_status

    try:
        import adafruit_bno055
        import board
        import busio
        import pickle

        i2c = busio.I2C(board.SCL, board.SDA)
        imu = adafruit_bno055.BNO055_I2C(i2c)
        imu.mode = adafruit_bno055.NDOF_MODE

        # Poll until calibrated or stopped
        while imu_calib_status["running"]:
            status = imu.calibration_status  # (sys, gyro, accel, mag)
            calibrated = imu.calibrated
            imu_calib_status["calibration_status"] = list(status)
            imu_calib_status["calibrated"] = calibrated

            if calibrated:
                offsets = {
                    "offsets_accelerometer": imu.offsets_accelerometer,
                    "offsets_gyroscope": imu.offsets_gyroscope,
                    "offsets_magnetometer": imu.offsets_magnetometer,
                }
                imu_calib_status["offsets"] = {
                    k: list(v) for k, v in offsets.items()
                }

                # Save PKL to scripts dir so walk script can find it
                pkl_path = str(SCRIPTS_DIR / "imu_calib_data.pkl")
                pickle.dump(offsets, open(pkl_path, "wb"))

                imu_calib_status["running"] = False
                return

            time.sleep(0.1)

    except Exception as e:
        imu_calib_status["error"] = str(e)
        imu_calib_status["running"] = False


@app.post("/api/imu/calibrate/start")
def imu_calibrate_start():
    """Start IMU calibration in a background thread."""
    global imu_calib_thread, imu_calib_status

    if imu_calib_status["running"]:
        return {"success": True, "message": "Calibration already running"}

    imu_calib_status = {
        "running": True,
        "calibration_status": [0, 0, 0, 0],
        "calibrated": False,
        "error": None,
        "offsets": None,
    }

    imu_calib_thread = Thread(target=_imu_calibrate_worker, daemon=True)
    imu_calib_thread.start()

    return {"success": True}


@app.get("/api/imu/calibrate/status")
def imu_calibrate_status():
    """Get current IMU calibration status."""
    return imu_calib_status


@app.post("/api/imu/calibrate/stop")
def imu_calibrate_stop():
    """Stop IMU calibration."""
    global imu_calib_status
    imu_calib_status["running"] = False
    return {"success": True}


# ── Config ────────────────────────────────────────────────────────────────────

def _read_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def _write_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


@app.get("/api/config")
def get_config():
    try:
        return _read_config()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Config file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config")
def update_config(config: DuckConfigModel):
    try:
        _write_config(config.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Reload HWI with new config
    release_hwi()

    return {"success": True}


# ── Walk Control ──────────────────────────────────────────────────────────────

def stop_walk_process():
    global walk_process, current_session_token
    if walk_process is not None and walk_process.poll() is None:
        walk_process.terminate()
        try:
            walk_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            walk_process.kill()
    walk_process = None
    current_session_token = None

    # Clean up remote command file
    try:
        os.remove(COMMAND_FILE)
    except FileNotFoundError:
        pass


class WalkStartRequest(BaseModel):
    sessionToken: str | None = None
    supabaseUrl: str | None = None
    supabaseKey: str | None = None


@app.post("/api/walk/start")
def walk_start(body: WalkStartRequest = WalkStartRequest()):
    """Start the walk script with streaming enabled.

    If a walk is already running and the incoming sessionToken matches the
    current one, this is a no-op (idempotent retry). If the token differs,
    the running walk is stopped and a fresh one is started for the new
    session, so a new browser tab / wizard run isn't silently bound to a
    dead channel from a previous attempt.
    """
    global walk_process, current_session_token

    if rehome_io is not None:
        raise HTTPException(
            status_code=409,
            detail="Servo rehoming session in progress — finish it before walking",
        )

    if walk_process is not None and walk_process.poll() is None:
        if body.sessionToken and body.sessionToken == current_session_token:
            return {"success": True, "message": "Walk is already running"}
        stop_walk_process()

    # Release HWI so the walk script can use the USB port
    release_hwi()

    venv_python = sys.executable
    is_pi = platform.machine() in ("aarch64", "armv7l")

    if is_pi:
        # Find the ONNX model — look for any .onnx file in scripts/
        onnx_files = list(SCRIPTS_DIR.glob("*.onnx"))
        if not onnx_files:
            raise HTTPException(
                status_code=404,
                detail="No ONNX model found in scripts/ directory",
            )
        onnx_path = str(onnx_files[0])
        walk_script = str(SCRIPTS_DIR / "v2_rl_walk_mujoco.py")

        cmd = [
            venv_python,
            walk_script,
            "--onnx_model_path", onnx_path,
            "--remote",
            "--commands",
        ]
        if body.sessionToken:
            cmd.extend(["--cloud_channel", f"robot-telemetry-{body.sessionToken}"])
            cmd.extend(["--cloud_commands_channel", f"robot-commands-{body.sessionToken}"])
            if body.supabaseUrl:
                cmd.extend(["--supabase_url", body.supabaseUrl])
            if body.supabaseKey:
                cmd.extend(["--supabase_key", body.supabaseKey])
    else:
        # Mock walk on non-Pi (Mac dev): spawn fake broadcaster using the
        # creds the dashboard sent in the POST body.
        if not (body.sessionToken and body.supabaseUrl and body.supabaseKey):
            raise HTTPException(
                status_code=400,
                detail="Mock walk requires sessionToken, supabaseUrl, and supabaseKey in request body",
            )
        cmd = [
            venv_python, "-u",
            str(SCRIPTS_DIR / "fake_broadcaster.py"),
            body.sessionToken,
            body.supabaseUrl,
            body.supabaseKey,
        ]

    print(f"[walk_start] spawning: {' '.join(cmd[:3])} ... ({'real' if is_pi else 'mock'} mode)")
    walk_process = subprocess.Popen(cmd, cwd=str(SCRIPTS_DIR))
    current_session_token = body.sessionToken

    return {"success": True, "pid": walk_process.pid}


@app.post("/api/walk/stop")
def walk_stop():
    """Stop the walk script."""
    if walk_process is None or walk_process.poll() is not None:
        stop_walk_process()
        return {"success": True, "message": "Walk was not running"}

    stop_walk_process()
    return {"success": True}


# ── Remote Commands ──────────────────────────────────────────────────────────

@app.post("/api/commands")
def send_commands(req: CommandRequest):
    """Write remote commands for the walk script to consume."""
    data = {
        "commands": req.commands[:7],
        "buttons": req.buttons,
        "left_trigger": req.left_trigger,
        "right_trigger": req.right_trigger,
        "timestamp": time.time(),
    }
    # Atomic write: write to temp, rename (locked to prevent race condition)
    with _command_file_lock:
        tmp_path = COMMAND_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, COMMAND_FILE)
    return {"success": True}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
