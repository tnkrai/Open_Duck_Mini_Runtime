"""
TNKR Robot Server

HTTP API server for the Open Duck Mini robot. Exposes motor check,
calibration, config management, and walk control endpoints.
Telemetry is streamed via Supabase Realtime broadcast channels.
"""

import json
import os
import platform
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

# ── Shared state ──────────────────────────────────────────────────────────────

hwi_instance: HWI | None = None
walk_process: subprocess.Popen | None = None


def get_hwi() -> HWI:
    """Get or create the HWI singleton. Reuses existing connection."""
    global hwi_instance
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
    release_hwi()
    stop_walk_process()


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


# ── Calibration ───────────────────────────────────────────────────────────────

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
    global walk_process
    if walk_process is not None and walk_process.poll() is None:
        walk_process.terminate()
        try:
            walk_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            walk_process.kill()
        walk_process = None

    # Clean up remote command file
    try:
        os.remove(COMMAND_FILE)
    except FileNotFoundError:
        pass


class WalkStartRequest(BaseModel):
    sessionToken: str | None = None


@app.post("/api/walk/start")
def walk_start(body: WalkStartRequest = WalkStartRequest()):
    """Start the walk script with streaming enabled."""
    global walk_process

    if walk_process is not None and walk_process.poll() is None:
        return {"success": True, "message": "Walk is already running"}

    # Release HWI so the walk script can use the USB port
    release_hwi()

    # Find the ONNX model — look for any .onnx file in scripts/
    onnx_files = list(SCRIPTS_DIR.glob("*.onnx"))
    if not onnx_files:
        raise HTTPException(
            status_code=404,
            detail="No ONNX model found in scripts/ directory",
        )
    onnx_path = str(onnx_files[0])

    venv_python = sys.executable
    walk_script = str(SCRIPTS_DIR / "v2_rl_walk_mujoco.py")

    cmd = [
        venv_python,
        walk_script,
        "--onnx_model_path", onnx_path,
        "--enable_streaming",
        "--streaming_port", "8765",
        "--remote",
        "--commands",
    ]

    # If a session token is provided, enable cloud telemetry + command relay
    if body.sessionToken:
        cmd.extend(["--cloud_channel", f"robot-telemetry-{body.sessionToken}"])
        cmd.extend(["--cloud_commands_channel", f"robot-commands-{body.sessionToken}"])

    walk_process = subprocess.Popen(cmd, cwd=str(SCRIPTS_DIR))

    return {"success": True, "pid": walk_process.pid}


@app.post("/api/walk/stop")
def walk_stop():
    """Stop the walk script."""
    global walk_process

    if walk_process is None or walk_process.poll() is not None:
        walk_process = None
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
