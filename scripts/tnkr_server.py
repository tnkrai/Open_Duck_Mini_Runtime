"""
TNKR Robot Server

HTTP API server for the Open Duck Mini robot. Exposes motor check,
calibration, config management, and walk control endpoints.
Telemetry is streamed via Supabase Realtime broadcast channels.
"""

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Thread, Lock

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import anyio
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# ── Runtime imports (available after pip install -e .) ────────────────────────

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime import telemetry

# ── Constants ─────────────────────────────────────────────────────────────────

HOME_DIR = os.path.expanduser("~")
CONFIG_PATH = os.path.join(HOME_DIR, "duck_config.json")
SCRIPTS_DIR = Path(__file__).parent
SERVER_PORT = 8000
# None -> HWI auto-detects the servo adapter by USB vendor id via
# find_servo_adapter() (CH343/FTDI), so the same code runs on any robot
# regardless of which /dev/ttyACMx it enumerates as, the cable, or the
# adapter's serial number.
USB_PORT = None

# ── Shared state ──────────────────────────────────────────────────────────────

hwi_instance: HWI | None = None


@dataclass
class WalkSession:
    """One walk-script launch. Each launch gets its own session object so a
    stop targeted at THIS process can never be misread by the monitor thread
    of another launch (stop A / start B race)."""

    proc: subprocess.Popen
    session_token: str | None
    cloud_streaming: bool
    started_at: float
    stop_requested: bool = False


walk_session: WalkSession | None = None
# Serializes walk start/stop. Without it, a stop from one browser tab can
# block in proc.wait() while a start from another tab installs a NEW session,
# which the stop then clobbers — orphaning a walk that holds the servo port.
_walk_lock = Lock()


def get_hwi() -> HWI:
    """Get or create the HWI singleton. Reuses existing connection."""
    global hwi_instance
    if hwi_instance is None:
        config = DuckConfig(config_json_path=CONFIG_PATH, ignore_default=True)
        hwi_instance = HWI(duck_config=config, usb_port=USB_PORT)
        # Which adapter chip (CH343/FTDI) this robot uses — attached to all
        # subsequent telemetry events and the device's person profile.
        telemetry.set_sticky(servo_adapter_chip=hwi_instance.servo_adapter_chip)
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

def _locked_stop_walk():
    with _walk_lock:
        stop_walk_process()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    release_hwi()
    # Off the event loop: stopping a SIGTERM-ignoring walk can block ~5s in
    # proc.wait(), which must not freeze in-flight responses during shutdown.
    await anyio.to_thread.run_sync(_locked_stop_walk)


app = FastAPI(title="TNKR Robot Server", lifespan=lifespan)


# ── Telemetry capture ─────────────────────────────────────────────────────────
#
#   request ──▶ TelemetryMiddleware            (sets _request_props = {})
#                  │  call_next
#                  ▼
#               endpoint ── add_telemetry_props(...)   (enrichments)
#                  │
#                  ├─ ok ────────────────────────────┐
#                  └─ raises ──▶ exception handler   │ (stashes error_type/
#                                 returns response   │  error_message via
#                  ┌──────────────────────────────────  add_telemetry_props)
#                  ▼
#               TelemetryMiddleware (finally:)
#                  └─▶ telemetry.capture("api_request_completed"/"_failed",
#                        {endpoint, status_code, duration_ms, **props})
#
# The handler/middleware split exists because Starlette converts HTTPException
# to a response *before* the outer middleware sees it — the middleware only
# observes the status code, so the error detail travels via the contextvar.

_request_props: ContextVar[dict | None] = ContextVar(
    "telemetry_request_props", default=None
)

# Endpoints that are polled or high-frequency — never captured.
TELEMETRY_EXCLUDED_PATHS = {
    "/api/commands",              # 50 Hz remote-control stream
    "/api/health",                # dashboard polling
    "/api/imu/calibrate/status",  # calibration UI polling
}


ERROR_MESSAGE_MAX_LEN = 500


def add_telemetry_props(**props):
    """Attach properties to the telemetry event for the current request.

    MUTATES the dict — never replaces it via _request_props.set(). Sync
    endpoints run in Starlette's threadpool, which copies the contextvars
    Context; the copy shares the same dict OBJECT, so mutations are visible
    to the middleware but a .set() in the copy would be lost.
    """
    d = _request_props.get()
    if d is not None:
        d.update(props)


def _stash_error(exc: BaseException, error_type: str | None = None):
    add_telemetry_props(
        error_type=error_type or type(exc).__name__,
        error_message=str(getattr(exc, "detail", None) or exc)[:ERROR_MESSAGE_MAX_LEN],
    )


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Capture outcome + duration + failure cause for every /api/ request."""

    async def dispatch(self, request: Request, call_next):
        if (
            request.method == "OPTIONS"
            or not request.url.path.startswith("/api/")
            or request.url.path in TELEMETRY_EXCLUDED_PATHS
        ):
            return await call_next(request)

        token = _request_props.set({})
        start = time.monotonic()
        status = 500
        skip_capture = False
        try:
            response = await call_next(request)
            status = response.status_code
            # 404/405 on a path that matched no route = LAN noise (port
            # scanners, stray apps), not a robot failure. Don't burn events.
            if status in (404, 405) and "route" not in request.scope:
                skip_capture = True
            return response
        except Exception as e:
            # Safety net — exception handlers normally convert before this.
            _stash_error(e)
            raise
        except BaseException:
            # Cancellation (client disconnected, server shutting down) is not
            # an API failure — capturing it would fabricate phantom 500s.
            skip_capture = True
            raise
        finally:
            if not skip_capture:
                props = {
                    "endpoint": request.url.path,
                    "method": request.method,
                    "status_code": status,
                    "duration_ms": round((time.monotonic() - start) * 1000, 1),
                    **(_request_props.get() or {}),
                }
                telemetry.capture(
                    "api_request_completed" if status < 400 else "api_request_failed",
                    props,
                )
            _request_props.reset(token)


@app.exception_handler(StarletteHTTPException)
async def _telemetry_http_exception_handler(request: Request, exc: StarletteHTTPException):
    _stash_error(exc, error_type="HTTPException")
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _telemetry_validation_exception_handler(request: Request, exc: RequestValidationError):
    # Field paths only — never the submitted values (they could hold tokens).
    locs = "; ".join(
        ".".join(str(p) for p in err.get("loc", [])) for err in exc.errors()[:10]
    )
    add_telemetry_props(
        error_type="RequestValidationError",
        error_message=locs[:ERROR_MESSAGE_MAX_LEN],
    )
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def _telemetry_unhandled_exception_handler(request: Request, exc: Exception):
    _stash_error(exc)
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    # Generic body on purpose: the rich error goes to telemetry, not to every
    # browser on the LAN (CORS here is wide open).
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


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


# Middleware add order = innermost first: TelemetryMiddleware sits closest to
# the app so duration_ms times only the endpoint, not the other middleware.
# PrivateNetworkMiddleware must be added before CORSMiddleware (i.e. inside it)
# so it handles preflight OPTIONS before CORSMiddleware can reject them.
app.add_middleware(TelemetryMiddleware)
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
    add_telemetry_props(
        all_responsive=all_responsive,
        responsive_count=sum(1 for m in motors if m["responsive"]),
        unresponsive_joints=[m["jointName"] for m in motors if not m["responsive"]],
    )
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

    add_telemetry_props(joints_calibrated=len(calibration_offsets))
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
# Serializes the running-check + rebind + spawn in /start (two concurrent
# starts would otherwise race past the check and spawn two I2C workers).
_imu_lock = Lock()


def _imu_calibrate_worker(status: dict):
    """Run IMU calibration in a background thread.

    Telemetry is captured here (not in the start/stop endpoints) because the
    worker runs exactly once per calibration, so each outcome — completed,
    failed, or user-stopped — produces exactly one event. The worker loops on
    the `status` dict it was HANDED, not the module global: /start rebinds the
    global to a fresh dict, and reading the global here would let a stopped
    worker latch onto the next run's dict (two workers, duplicate events).
    """
    started = time.monotonic()
    try:
        import adafruit_bno055
        import board
        import busio
        import pickle

        i2c = busio.I2C(board.SCL, board.SDA)
        imu = adafruit_bno055.BNO055_I2C(i2c)
        imu.mode = adafruit_bno055.NDOF_MODE

        # Poll until calibrated or stopped
        while status["running"]:
            cal_status = imu.calibration_status  # (sys, gyro, accel, mag)
            calibrated = imu.calibrated
            status["calibration_status"] = list(cal_status)
            status["calibrated"] = calibrated

            if calibrated:
                offsets = {
                    "offsets_accelerometer": imu.offsets_accelerometer,
                    "offsets_gyroscope": imu.offsets_gyroscope,
                    "offsets_magnetometer": imu.offsets_magnetometer,
                }
                status["offsets"] = {
                    k: list(v) for k, v in offsets.items()
                }

                # Save PKL to scripts dir so walk script can find it
                pkl_path = str(SCRIPTS_DIR / "imu_calib_data.pkl")
                pickle.dump(offsets, open(pkl_path, "wb"))

                status["running"] = False
                telemetry.capture(
                    "imu_calibration_completed",
                    {"duration_s": round(time.monotonic() - started, 1)},
                )
                return

            time.sleep(0.1)

        # Loop exited without calibrating: /api/imu/calibrate/stop flipped
        # `running` to False.
        telemetry.capture(
            "imu_calibration_stopped",
            {
                "duration_s": round(time.monotonic() - started, 1),
                "calibration_status": status["calibration_status"],
            },
        )

    except Exception as e:
        status["error"] = str(e)
        status["running"] = False
        telemetry.capture(
            "imu_calibration_failed",
            {
                "error_type": type(e).__name__,
                "error_message": str(e)[:500],
                "duration_s": round(time.monotonic() - started, 1),
            },
        )


@app.post("/api/imu/calibrate/start")
def imu_calibrate_start():
    """Start IMU calibration in a background thread."""
    global imu_calib_thread, imu_calib_status

    with _imu_lock:
        if imu_calib_status["running"]:
            return {"success": True, "message": "Calibration already running"}

        imu_calib_status = {
            "running": True,
            "calibration_status": [0, 0, 0, 0],
            "calibrated": False,
            "error": None,
            "offsets": None,
        }

        imu_calib_thread = Thread(
            target=_imu_calibrate_worker, args=(imu_calib_status,), daemon=True
        )
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

    # Keys only (capped) — the dict is open-typed, so never forward values.
    add_telemetry_props(
        expression_features_enabled=[
            str(k)[:50] for k, v in config.expression_features.items() if v
        ][:20]
    )
    return {"success": True}


# ── Walk Control ──────────────────────────────────────────────────────────────

def stop_walk_process():
    """Stop the current walk. Callers must hold _walk_lock."""
    global walk_session
    session = walk_session
    if session is not None and session.proc.poll() is None:
        # Mark THIS launch as deliberately stopped before terminating, so its
        # monitor thread reports stop_requested=True / crashed=False.
        session.stop_requested = True
        session.proc.terminate()
        try:
            session.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            session.proc.kill()
    # Only clear if no newer session was installed meanwhile.
    if walk_session is session:
        walk_session = None

    # Clean up remote command file
    try:
        os.remove(COMMAND_FILE)
    except FileNotFoundError:
        pass


def _monitor_walk(session: WalkSession):
    """Wait for one walk launch to exit and report how it ended."""
    rc = session.proc.wait()
    # Any nonzero exit we didn't ask for is a crash — including -SIGKILL,
    # which is how the kernel OOM killer ends walks on a 512MB Pi. Exempting
    # signals wholesale would blind the crash-rate dashboard to OOM.
    crashed = rc != 0 and not session.stop_requested
    telemetry.capture(
        "walk_ended",
        {
            "duration_s": round(time.monotonic() - session.started_at, 1),
            "exit_code": rc,
            "crashed": crashed,
            "stop_requested": session.stop_requested,
            "cloud_streaming": session.cloud_streaming,
        },
    )


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
    with _walk_lock:
        return _walk_start_locked(body)


def _walk_start_locked(body: WalkStartRequest):
    global walk_session

    if walk_session is not None and walk_session.proc.poll() is None:
        if body.sessionToken and body.sessionToken == walk_session.session_token:
            add_telemetry_props(already_running=True)
            return {"success": True, "message": "Walk is already running"}
        stop_walk_process()

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
        "--remote",
        "--commands",
    ]

    # If a session token is provided, enable cloud telemetry + command relay
    if body.sessionToken:
        cmd.extend(["--cloud_channel", f"robot-telemetry-{body.sessionToken}"])
        cmd.extend(["--cloud_commands_channel", f"robot-commands-{body.sessionToken}"])
        if body.supabaseUrl:
            cmd.extend(["--supabase_url", body.supabaseUrl])
        if body.supabaseKey:
            cmd.extend(["--supabase_key", body.supabaseKey])

    # Whether joint data streams to the cloud (boolean only — never the
    # token value or the joint stream itself).
    cloud_streaming = bool(body.sessionToken and body.supabaseUrl and body.supabaseKey)
    add_telemetry_props(
        cloud_streaming=cloud_streaming,
        has_session=bool(body.sessionToken),
        already_running=False,
    )

    proc = subprocess.Popen(cmd, cwd=str(SCRIPTS_DIR))
    walk_session = WalkSession(
        proc=proc,
        session_token=body.sessionToken,
        cloud_streaming=cloud_streaming,
        started_at=time.monotonic(),
    )
    Thread(target=_monitor_walk, args=(walk_session,), daemon=True).start()

    return {"success": True, "pid": proc.pid}


@app.post("/api/walk/stop")
def walk_stop():
    """Stop the walk script."""
    with _walk_lock:
        was_running = walk_session is not None and walk_session.proc.poll() is None
        add_telemetry_props(was_running=was_running)
        stop_walk_process()
    if not was_running:
        return {"success": True, "message": "Walk was not running"}
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
    telemetry.capture("server_started", {"server_port": SERVER_PORT})
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
