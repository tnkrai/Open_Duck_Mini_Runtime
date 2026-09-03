"""
TNKR Robot Server

HTTP API server for the Open Duck Mini robot. Exposes motor check,
servo rehoming (firmware zero), stance calibration, config management,
and walk control endpoints.
Telemetry is streamed via Supabase Realtime broadcast channels.
"""

import errno
import json
import math
import os
import platform
import shutil
import signal
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

from mini_bdx_runtime.rustypot_position_hwi import (
    BUS_LOCK,
    HWI,
    find_servo_adapter,
    is_rust_panic,
)
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime import telemetry
from mini_bdx_runtime import walk_telemetry
from mini_bdx_runtime import walk_pause
from mini_bdx_runtime import walk_offsets
from mini_bdx_runtime.pad import (
    disconnect_pad,
    forget_pad,
    joystick_present,
    pair_pad,
    pad_status,
    scan_pad,
    wake_adapter,
    walk_flags,
)
# ── Constants ─────────────────────────────────────────────────────────────────

HOME_DIR = os.path.expanduser("~")
CONFIG_PATH = os.path.join(HOME_DIR, "duck_config.json")
SCRIPTS_DIR = Path(__file__).parent
SERVER_PORT = 8000


def _resolve_usb_port(default: str = "/dev/ttyACM0") -> str:
    """Resolve the servo-bus serial port for direct pypot IO (rehoming, voltage).

    HWI auto-detects its own port via find_servo_adapter(); this is for the
    endpoints that open the bus without HWI. ttyACMx numbers are NOT stable
    across reboots/replugs, so match the adapter by USB vendor id
    (CH343/FTDI) instead. An explicit TNKR_USB_PORT env var overrides
    everything; if no adapter is found, fall back to `default` so the module
    stays importable on dev machines.
    """
    env = os.environ.get("TNKR_USB_PORT")
    if env:
        return env
    try:
        return find_servo_adapter()[0]
    except Exception:
        return default


USB_PORT = _resolve_usb_port()

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
last_walk_exit_code: int | None = None  # non-zero surfaces a crashed walk to clients
rehome_io = None  # pypot FeetechSTS3215IO while a rehoming session is open
state_imu = None  # raw_imu.Imu singleton for idle /api/state reads (BNO055, I2C)
last_state_joints: dict[str, float] = {}  # last-read pose, served while hardware is owned elsewhere


def get_hwi() -> HWI:
    """Get or create the HWI singleton. Reuses existing connection."""
    global hwi_instance
    if rehome_io is not None:
        raise RuntimeError(
            "Servo rehoming session in progress — finish it before other motor operations"
        )
    with BUS_LOCK:
        if hwi_instance is None:
            config = DuckConfig(config_json_path=CONFIG_PATH, ignore_default=True)
            # None unless TNKR_USB_PORT overrides — HWI then auto-detects the
            # adapter itself and records which chip it found for telemetry.
            hwi_instance = HWI(
                duck_config=config, usb_port=os.environ.get("TNKR_USB_PORT")
            )
            # Which adapter chip (CH343/FTDI) this robot uses — attached to all
            # subsequent telemetry events and the device's person profile.
            telemetry.set_sticky(servo_adapter_chip=hwi_instance.servo_adapter_chip)
        return hwi_instance


def release_hwi(disable_torque: bool = True):
    """Release the HWI connection so another flow can take the serial bus.

    Feetech servos hold their last commanded position in firmware without any
    bus traffic, so freeing the port does NOT require cutting torque. Pass
    disable_torque=False to hand off the bus while the robot keeps holding its
    pose (e.g. the walking stance) — going limp mid-stand makes the duck fall.

    Drops the rustypot handle (close()) under BUS_LOCK so pypot (voltage,
    rehome) cannot open the same adapter while rustypot still holds the fd.
    """
    global hwi_instance, head_puppet_active
    with BUS_LOCK:
        if hwi_instance is not None:
            if disable_torque:
                try:
                    hwi_instance.turn_off()
                except Exception:
                    pass
            try:
                hwi_instance.close()
            except BaseException:
                pass
            hwi_instance = None
    # A puppet session is only as live as the HWI it drives: whoever takes the
    # bus next (walk, rehome, stance) leaves the head somewhere we no longer
    # know, so the next request must re-enter rather than skip the setup.
    head_puppet_active = False


def is_walking() -> bool:
    return walk_session is not None and walk_session.proc.poll() is None


def is_paused() -> bool:
    """Whether the running walk is frozen (torque held) — see /api/walk/pause.
    Only meaningful while a walk is running; idle, there is no gait to pause."""
    return is_walking() and bool(walk_pause.read())


def walk_exit_code() -> int | None:
    """The last walk's exit code — non-zero/None surfaces a crash to clients
    (never a zombie or a duck stuck mid-gait without anyone knowing)."""
    global last_walk_exit_code
    if walk_session is not None and walk_session.proc.poll() is not None:
        last_walk_exit_code = walk_session.proc.returncode
    return last_walk_exit_code


def refuse_while_walking():
    if is_walking():
        raise HTTPException(
            status_code=409, detail="The walk owns the servo bus — stop it first"
        )


def get_state_imu():
    """Lazy BNO055 reader for idle /api/state (live joints + orientation).

    Same exclusivity rule as the servo bus: released before the walk spawns
    (the walk subprocess owns the I2C then; orientation comes from its
    telemetry snapshot instead). The IMU-calibration worker shares this
    handle (see _imu_calibrate_worker) — constructing a second BNO055_I2C
    would soft-reset the chip and wipe the axis remap."""
    global state_imu
    if state_imu is None:
        from mini_bdx_runtime.raw_imu import Imu

        upside_down = False
        try:
            upside_down = bool(_read_config().get("imu_upside_down", False))
        except Exception:
            pass
        state_imu = Imu(sampling_freq=15, user_pitch_bias=0, upside_down=upside_down)
    return state_imu


def release_state_imu():
    global state_imu
    if state_imu is not None:
        try:
            state_imu.stop()
        except Exception:
            pass
        state_imu = None


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
    # per-joint direction, +1 or -1 (see HWI.joints_signs); written by /api/directions/save
    joints_signs: dict = {}
    # the build gave the right leg's servo ids to the left leg; written by
    # /api/calibration/swap-legs (see HWI.__init__)
    legs_swapped: bool = False


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
    _close_rehome_io()
    release_hwi()
    # Off the event loop: stopping a SIGTERM-ignoring walk can block ~5s in
    # proc.wait(), which must not freeze in-flight responses during shutdown.
    await anyio.to_thread.run_sync(_locked_stop_walk)


def _close_rehome_io(disable_torque: bool = True):
    """Free the bus if a rehoming session is still open, optionally dropping torque."""
    global rehome_io
    if rehome_io is None:
        return
    io = rehome_io
    rehome_io = None
    if disable_torque:
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
    "/api/head/puppet",           # on-screen joystick stream
    "/api/health",                # dashboard polling
    "/api/imu/calibrate/status",  # calibration UI polling
    "/api/telemetry/identity",    # asking who we are is not an event about us
}

# Polled READ endpoints: capture failures, never successes.
#
# These were the hole in the list above. It excluded the 50 Hz *write* stream
# (/api/commands) and missed the read side, so /api/state — the live joints and
# IMU poll behind Studio's 3D viewer — became 96% of all telemetry: 68,179 of
# 70,730 api_request_completed across the fleet's first month.
#
# That is not merely a volume bill. capture() is rate-capped at
# RATE_LIMIT_PER_MIN=60 and the cap is indiscriminate, so viewer polling
# crowded out the events the funnel is built from. Measured on the real fleet
# before this change: 6 of 11 devices hitting the cap, discarding 710 events on
# average and 1,870 at worst. Every setup_step_failed and walk_ended lost in
# those windows is gone.
#
# Full exclusion would have been the wrong fix. A failing /api/state means the
# viewer has no data — a real product failure worth knowing about — so only the
# success case is dropped. A 200 here says "the viewer polled", which nothing
# reads; a 500 says "the robot stopped answering", which somebody should.
TELEMETRY_FAILURES_ONLY_PATHS = {
    "/api/state",              # live joints + IMU, polled continuously by the viewer
    "/api/stance/positions",   # polled while the stance editor is open
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
            # A polled read that worked is not news. Checked on the way out
            # rather than at the top of dispatch, because the failure still is.
            elif status < 400 and request.url.path in TELEMETRY_FAILURES_ONLY_PATHS:
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

@app.get("/api/telemetry/identity")
def telemetry_identity():
    """This robot's anonymous telemetry id, so Studio knows which duck it reached.

    Read-only and deliberately dull. It returns a random UUID and a boolean, and
    nothing else — no owner, no account, no claim state. Ownership is decided in
    tnkr-core behind a verified Supabase token, precisely because this server
    authenticates nobody and is reachable from any page the operator visits.

    There is no write counterpart, and adding one is not a small change: it would
    let anything on the network (or any webpage) assert who owns this robot.

    Telemetry off => {"enabled": false} with no id at all, so opting out on the
    robot also prevents a signed-in Studio session from claiming it.
    """
    snapshot = telemetry.identity_snapshot()
    body = {"enabled": bool(snapshot.get("enabled"))}
    device = snapshot.get("device_id")
    if device:
        body["deviceId"] = device
    return body


@app.get("/api/health")
def health():
    machine = platform.machine()  # 'aarch64' / 'armv7l' on Pi, 'x86_64' / 'arm64' on Mac
    is_pi = machine in ("aarch64", "armv7l")
    return {
        "status": "ok",
        "is_pi": is_pi,
        "platform": machine,
        "walking": is_walking(),
        "paused": is_paused(),
        "walkExitCode": walk_exit_code(),
        # What /api/calibration/start does to the robot. Studio reads this at connect
        # and refuses to arm a joint calibration against an agent that does not say
        # "hold": the older flow drove every joint to servo-zero the moment it was
        # armed, and the screen that arms it now promises the opposite. It has to be
        # here rather than in /start's reply, because by the time an old agent has
        # replied it has already moved the duck.
        "calibrationMode": "hold",
    }


@app.get("/api/state")
def read_state():
    """Live joints (radians) + IMU orientation for the studio's viewer/recorder.

    At idle, read joints off the bus and orientation off the BNO055. While the
    walk owns the hardware, serve the walk loop's shared-memory snapshot (it
    reads every joint at 50 Hz for its policy anyway); fall back to the
    last-known pose if the snapshot is missing or stale — never fight for the
    port, never serve a dead snapshot as live."""
    global last_state_joints, hwi_instance
    imu_payload = None
    if is_walking():
        snap = walk_telemetry.read_snapshot()
        if snap and snap.get("joints"):
            last_state_joints = {
                n: round(float(p), 4) for n, p in snap["joints"].items()
            }
            snap_imu = snap.get("imu") or {}
            if snap_imu.get("quaternion"):
                imu_payload = {
                    "quaternion": snap_imu["quaternion"],
                    "gyro": snap_imu.get("gyro", []),
                    "accel": snap_imu.get("accelero", []),
                }
    elif rehome_io is None:
        try:
            # Hold the bus for get_hwi + the 14-joint read so /api/voltage
            # cannot open pypot on the same adapter mid-poll (that double-open
            # is what poisons rustypot's mutex and 500s every later /api/state).
            with BUS_LOCK:
                hwi = get_hwi()
                positions = hwi.get_present_positions()
            if positions is not None and len(positions) == len(hwi.joints):
                last_state_joints = {
                    n: round(float(p), 4)
                    for n, p in zip(hwi.joints.keys(), positions)
                }
        except BaseException as e:
            # PanicException subclasses BaseException, not Exception — the
            # previous `except Exception` let it kill the ASGI task.
            if is_rust_panic(e):
                try:
                    release_hwi(disable_torque=False)
                except BaseException:
                    hwi_instance = None
            elif not isinstance(e, Exception):
                raise
            # transient read blip / recovered panic → serve the cached pose
        try:
            imu_data = get_state_imu().get_data()
            imu_payload = {
                "quaternion": [
                    float(q) for q in imu_data.get("quaternion", [1.0, 0.0, 0.0, 0.0])
                ],
                "gyro": [float(g) for g in imu_data["gyro"]],
                "accel": [float(a) for a in imu_data["accelero"]],
            }
        except Exception:
            pass  # IMU absent/unready → joints still served
    return {"joints": last_state_joints, "imu": imu_payload, "fps": 0.0}


# ── Motor Check ───────────────────────────────────────────────────────────────

@app.post("/api/motors/check")
def check_motors():
    """Check all 14 motors for responsiveness."""
    refuse_while_walking()
    with BUS_LOCK:
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
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                if is_rust_panic(e):
                    try:
                        hwi._reopen_io()
                    except BaseException:
                        pass
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


# ── Joint calibration (soft offsets) ─────────────────────────────────────────
# The duck holds the pose it was PLACED in, one joint is released at a time, the
# operator hand-poses that joint to straight, and the servo's reading there becomes
# the joint's software offset in duck_config.json.
#
# NOTHING IS EVER COMMANDED TO SERVO-ZERO, and that is the point. The earlier flow (a
# port of scripts/find_soft_offsets.py) drove every joint to zero_pos to arm, and took
# each offset as a delta from the reading there. That is self-defeating on exactly the
# joints calibration exists for: when a horn is mis-seated, servo-zero is mechanically
# far from straight, so arming drove those joints toward their own shells — reported
# from a real build as joints "hitting their limits and colliding with their own parts"
# the moment calibration started. Holding the placed pose moves nothing.
#
# The offset is then read directly rather than differenced:
#
#     offset = the raw servo reading with the joint held at straight
#
# which IS the mounting error (a perfectly seated joint reads 0 there). At walk time
# the HWI writes target + offset, so commanding straight puts the servo exactly where
# the operator taught it. No droop baseline is needed, because no joint is ever held at
# a commanded position under kd 0 for one to be measured against: the operator's hands
# hold the joint while it is read, and a Feetech encoder reports true position with
# torque off.
#
# Known limit, and the reason /api/rehome/* exists alongside this: a walk commands
# init_pos + offset + policy, and init_pos already spends 79 deg of the servo's 180
# deg window at the knees. A large offset pushes past the command seam, where the
# value wraps to a different physical position and the joint is driven into the
# shell. NOTHING GUARDS THIS TODAY. The margin is measured and sent to telemetry as
# seam_headroom_deg and deliberately not enforced, so fleet data decides whether a
# threshold is worth having; there is no OFFSET_OUT_OF_RANGE code on either side
# (jointCalErrors.test.ts asserts its absence). Rehoming avoids the seam entirely by
# correcting inside the servo, in count-space.
#
# Everything here reads and writes ONE joint at a time. get_present_positions()
# returns None if any of the fourteen fails, which made a silent joint 3 fail a
# calibration of joint 7 with no way to name the joint that was actually quiet.

# Offsets accepted this session, and the baseline each was measured against. Both
# keyed by joint name, both reset by /start.
calibration_offsets: dict[str, float] = {}
calibration_baselines: dict[str, float] = {}

# The pose the session HOLDS while joints are released, in policy space (what
# set_position_all is given; the servo goal is this plus the joint's offset).
#
# Starts as the pose the duck was PLACED in, so arming moves nothing. A joint's entry
# flips to 0 — straight — once its offset is applied, so every later re-assert keeps
# calibrated joints straight and leaves the uncalibrated ones where they were placed.
# An entry is only meaningful in the offset space it was written in, which is why
# begin-joint re-reads the joint it is about to release and rewrites its entry in the
# same breath as zeroing its offset (see there). Cleared by /finish and /save, so a
# stray re-assert after either is a no-op rather than a drive.
calibration_hold: dict[str, float] = {}

# Stiffness for the held pose. The walk's kp of 32 is more than keeping a placed duck
# still needs; 20 is what the operator's own hold-and-release script was proven with
# on the robot. kd is 0 for the session so a released joint is not fought.
CALIBRATION_HOLD_KP = 20

# What the fleet analytics needs that a per-request event cannot carry: how long the
# whole session took, how many attempts each joint needed, which faults came up, and
# the init_pos the offsets will actually be added to at walk time.
#
# init_pos is copied here rather than read at /save because /save runs after the HWI
# may have been dropped. hwi.init_pos itself is never touched by the session: the
# placed pose is written as goals, not installed as the stance, so nothing that turns
# the robot on later walks to it.
calibration_session: dict = {}


def _reset_calibration_session(hwi) -> None:
    global calibration_session
    calibration_session = {
        "started_at": time.time(),
        # joint -> how many times it was released / measured / accepted. Attempts are
        # the signal that matters most: a joint that takes four goes is either badly
        # assembled or badly explained, and we cannot tell which from one robot.
        "begins": {},
        "confirms": {},
        "accepts": {},
        # error code -> count, so a fault that only shows up on real hardware is
        # visible without reading a log off an SD card.
        "faults": {},
        # the walking pose the offsets get added to, before /start clobbers it
        "init_pos": {k: round(float(v), 4) for k, v in hwi.init_pos.items()},
    }


def _bump(bucket: str, key: str) -> None:
    counts = calibration_session.get(bucket)
    if isinstance(counts, dict):
        counts[key] = counts.get(key, 0) + 1


def _note_calibration_fault(code: str, joint_name: str | None = None) -> None:
    """Record a fault for the session summary AND on this request's own event.

    `error_code` as its own property rather than only inside error_message: a string
    like "MOTORS_SILENT: left_knee: timeout" cannot be grouped in PostHog, and the
    grouping is the entire point of collecting it.
    """
    _bump("faults", code)
    props = {"error_code": code}
    if joint_name:
        props["joint_name"] = joint_name
    add_telemetry_props(**props)


def _seam_headroom(offsets: dict, init_pos: dict) -> dict:
    """Degrees left in each servo's +-180 command window once the offset is added.

    This is the number that says whether the soft-offset approach is safe on real
    robots. A walk commands init_pos + offset + policy, and init_pos already spends 79
    of the 180 degrees at the knees, so a large offset leaves the policy no room and
    the value wraps. Measuring the margin across the fleet is how we learn whether the
    guard is worth building, instead of guessing.
    """
    out = {}
    for joint, offset in offsets.items():
        target = math.degrees(init_pos.get(joint, 0.0) + offset)
        out[joint] = round(180.0 - abs(target), 1)
    return out


def _agent_error(status: int, code: str, message: str, joint: str | None = None):
    """Raise a failure Studio can map without parsing prose.

    `detail` is a DICT here, not a string. Studio's agent_client used to recover the
    code by string-sniffing this field for a `CODE:` prefix — a protocol pretending to
    be a log message, O(n) over the ErrorCode enum on every failure, and silent on a
    typo: write `MOTOR_SILENT:` and nothing errors, it just falls back to the status
    map, which is how a loose cable became "we couldn't reach your duck". A real field
    is read once and cannot be misspelled without something visible going wrong.

    `joint` travels as its own key for the same reason: the copy names the joint, and
    scraping a joint name back out of a sentence is not a thing to build on.

    Scoped to the calibration routes deliberately. The older prefix convention is still
    how the pad/rehome/stance routes talk, those are deployed, and an older Studio only
    understands the prefix — converting them would regress anyone who updates their
    robot before their Studio. New routes use this; the legacy path stays until they
    are migrated together.
    """
    detail: dict = {"code": code, "message": message}
    if joint:
        detail["joint"] = joint
    raise HTTPException(status_code=status, detail=detail)


def _calibration_hwi():
    """The HWI, or a 503 naming why not. Refuses while a walk owns the bus."""
    refuse_while_walking()
    try:
        return get_hwi()
    except Exception as e:
        _agent_error(503, "SERVO_BUS_UNAVAILABLE", str(e))


def _calibration_joint(hwi, joint_name: str) -> str:
    if joint_name not in hwi.joints:
        # Its own code. This used to be a bare 400, and 400 is not in Studio's status
        # map, so it fell through to the catch-all and told the operator their duck was
        # unreachable — for a duck that had just answered, about a joint it does not
        # have. A frontend bug wearing a hardware fault's message.
        _note_calibration_fault("UNKNOWN_JOINT", joint_name)
        _agent_error(400, "UNKNOWN_JOINT", f"no such joint: {joint_name}", joint_name)
    return joint_name


def _joint_sign(hwi, joint_name: str) -> int:
    """+1 or -1: the joint's direction relative to the model (HWI.joints_signs)."""
    return int(getattr(hwi, "joints_signs", {}).get(joint_name, 1))


def _legs_swapped(hwi) -> bool:
    """Whether this build's left and right leg ids are the other way round (config)."""
    return bool(getattr(getattr(hwi, "duck_config", None), "legs_swapped", False))


def _read_one(hwi, joint_name: str) -> float:
    """One joint's present position, or a 502 that names the joint.

    The code travels in the detail because agent_client maps status codes, not
    messages, and its escape hatch honours a leading `CODE:` — without it a silent
    servo arrives at Studio as SERVO_BUS_UNAVAILABLE ("check they're connected and
    powered") when the bus is open and fourteen other servos are answering on it.
    """
    try:
        return float(hwi.get_present_position(joint_name))
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        _note_calibration_fault("MOTORS_SILENT", joint_name)
        _agent_error(502, "MOTORS_SILENT", str(e), joint_name)


@app.post("/api/calibration/start")
def calibration_start():
    """Hold the pose the duck was placed in, so joints can be released one at a time.

    Moves nothing: every joint is commanded to the position it is already at.
    """
    global calibration_offsets, calibration_baselines, calibration_hold

    hwi = _calibration_hwi()

    # Offsets zeroed first, so for the rest of the session a present read is the RAW
    # servo angle — the space the offset itself is measured in.
    for joint_name in hwi.joints:
        hwi.joints_offsets[joint_name] = 0
    calibration_offsets = {}
    calibration_baselines = {}

    _reset_calibration_session(hwi)

    # Read BEFORE commanding anything. This is the placed pose — the thing the session
    # will hold — and it doubles as the whole-bus silent-joint check, which belongs
    # here and only here: arming is the one step that is about every joint, so a servo
    # that is already quiet is caught before the operator's hands are on the robot.
    # Reported rather than swallowed: the old code returned an empty currentPositions
    # dict and let the session continue with no pose at all.
    positions = hwi.get_present_positions()
    if positions is None:
        _note_calibration_fault("MOTORS_SILENT")
        _agent_error(502, "MOTORS_SILENT", "at least one joint did not answer")

    joint_names = list(hwi.joints.keys())
    current_positions = {
        name: round(float(positions[i]), 3) for i, name in enumerate(joint_names)
    }

    # Gains written directly, once. NOT turn_on(): that ramps every joint through kp 2
    # for a second before its goal lands, and with kd at 0 a loaded knee sags under
    # the body for those seconds and is hauled back when the real kp arrives — a
    # visible dip on a duck that was promised nothing would move. It would also have
    # driven every joint to hwi.init_pos on the way, which is not where the duck is.
    try:
        hwi.set_kds([0] * len(joint_names))
        hwi.set_kps([CALIBRATION_HOLD_KP] * len(joint_names))
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        _note_calibration_fault("MOTORS_SILENT")
        _agent_error(502, "MOTORS_SILENT", str(e))

    # Hold exactly where each joint already is. The offsets were zeroed above, so
    # set_position_all writes each goal unchanged: every servo's goal IS its present
    # position, and nothing moves. Goal first, torque second, per joint — the order
    # apply-offset uses, for the same reason: a servo that wakes up already at its
    # target stiffens in place, and an explicit enable after the goal does not depend
    # on the firmware doing it for us. dict(), not a bare assignment — sharing the
    # object with anything else would make a later per-joint write edit both.
    calibration_hold = dict(current_positions)
    hwi.set_position_all(calibration_hold)
    for joint_name in joint_names:
        try:
            hwi.set_joint_torque(joint_name, True)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            _note_calibration_fault("TORQUE_ENABLE_FAILED", joint_name)
            _agent_error(502, "TORQUE_ENABLE_FAILED", str(e), joint_name)
    time.sleep(0.5)

    # The placed pose, fleet-wide: how far from straight operators actually leave the
    # duck when they calibrate. That is what says whether the on-screen reference
    # matches what people really do with the robot in front of them.
    placed = [abs(v) for v in current_positions.values()]
    telemetry.capture(
        "joint_calibration_started",
        {
            "joint_count": len(joint_names),
            "placed_pose_rad": current_positions,
            "max_abs_placed_rad": round(max(placed), 4) if placed else 0.0,
        },
    )
    # `mode` is the belt to /api/health's braces: Studio refuses to arm against an
    # agent whose health does not say "hold", and checks this on the way back.
    return {
        "joints": joint_names,
        "currentPositions": current_positions,
        "mode": "hold",
        "legsSwapped": _legs_swapped(hwi),
    }


@app.post("/api/calibration/begin-joint")
def calibration_begin_joint(req: JointRequest):
    """Take this joint's baseline reading, then drop its torque so it can be posed."""
    hwi = _calibration_hwi()
    joint_name = _calibration_joint(hwi, req.jointName)

    # Start this joint from scratch. Antoine's retry branch does the same
    # (joints_offsets[joint] = 0): without it, re-doing an accepted joint measures
    # against its own previous correction and the offsets compound.
    hwi.joints_offsets[joint_name] = 0

    # Read the joint where it is — raw, now that its offset is 0 — and make THAT its
    # hold entry before anything is re-asserted. The entry it had cannot survive the
    # offset change: after apply-offset it is 0 in a space where the offset carried
    # the whole correction, and re-asserting that 0 with the offset just zeroed would
    # send a redone joint from straight to servo-zero at full stiffness, which is the
    # one drive this flow exists never to make. Reading fresh also means a joint that
    # is limp is held where it hangs rather than hauled back to where it was released.
    #
    # NOT a baseline: the offset is measured against straight, so the baseline is 0
    # by definition and there is nothing to subtract. The read earns its place as the
    # liveness check — this is the last moment before the operator's hands go on the
    # joint, and "this joint did not answer" is worth saying now rather than after
    # they have posed it.
    pre_release = _read_one(hwi, joint_name)
    calibration_hold[joint_name] = pre_release

    # Re-assert the HELD pose before releasing: hand-posing one joint drags its
    # neighbours off their goals. Calibrated joints are held straight (their entry
    # flipped to 0 in apply-offset), the rest stay where the duck was placed, and this
    # joint's goal is the position it is already at. NOT zero_pos — driving the whole
    # robot to servo-zero is what this flow exists to avoid.
    hwi.set_position_all(calibration_hold)
    time.sleep(0.5)

    # Recorded even though it is a constant: confirm-position stays a subtraction, the
    # response shape is unchanged, and its presence is what marks this joint released
    # (see the INVALID_STATE guard there).
    baseline = 0.0
    calibration_baselines[joint_name] = baseline

    try:
        hwi.set_joint_torque(joint_name, False)
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        # The joint may or may not have gone limp: a serial timeout means no reply
        # came back, not that the write failed to land. Studio's copy says only what
        # is safe under either state.
        _note_calibration_fault("TORQUE_RELEASE_FAILED", joint_name)
        _agent_error(502, "TORQUE_RELEASE_FAILED", str(e), joint_name)

    _bump("begins", joint_name)
    add_telemetry_props(
        joint_name=joint_name,
        baseline_rad=round(baseline, 4),
        # Where the joint sat before the operator touched it. Not a baseline any more,
        # but the distance from here to the straight pose is how far a mis-seated horn
        # actually had to be moved — which the old flow could never report, because it
        # had already dragged every joint to servo-zero before anyone looked.
        pre_release_rad=round(pre_release, 4),
        # 2 means the operator came back to this joint: worth knowing per joint, and
        # worth knowing whether it is always the same joints across the fleet.
        attempt=calibration_session.get("begins", {}).get(joint_name, 1),
    )
    return {"success": True, "jointName": joint_name, "baseline": round(baseline, 4)}


@app.post("/api/calibration/confirm-position")
def calibration_confirm_position(req: JointRequest):
    """Read the joint at the straight pose. That raw reading IS the offset.

    Kept as a subtraction against the baseline (0) rather than collapsed to the read:
    the arithmetic is the thing being asserted, and a future baseline that is not zero
    would otherwise have to reintroduce it.
    """
    hwi = _calibration_hwi()
    joint_name = _calibration_joint(hwi, req.jointName)

    if joint_name not in calibration_baselines:
        _note_calibration_fault("INVALID_STATE", joint_name)
        _agent_error(
            409, "INVALID_STATE", f"no baseline for {joint_name}: call begin-joint first", joint_name
        )

    baseline = calibration_baselines[joint_name]
    new_pos = _read_one(hwi, joint_name)
    # The read is in model space: sign * raw, with the offset at 0 for the session.
    # The offset lives in raw servo space (the HWI adds it AFTER the sign), so the
    # sign is undone here: a mirrored joint posed straight reads -raw and must save
    # +raw, or applying it would hold the joint at the mirror of where the hands are.
    offset = _joint_sign(hwi, joint_name) * (new_pos - baseline)

    _bump("confirms", joint_name)
    add_telemetry_props(
        joint_name=joint_name,
        baseline_rad=round(baseline, 4),
        measured_rad=round(new_pos, 4),
        offset_rad=round(offset, 4),
        offset_deg=round(math.degrees(offset), 2),
        # how close this offset would put the joint to the servo's command seam once
        # the walk adds init_pos on top. Reported, not enforced.
        seam_headroom_deg=_seam_headroom(
            {joint_name: offset}, calibration_session.get("init_pos", {})
        ).get(joint_name),
        attempt=calibration_session.get("confirms", {}).get(joint_name, 1),
    )
    return {
        "jointName": joint_name,
        "offset": round(offset, 4),
        "previousPosition": round(baseline, 4),
        "newPosition": round(new_pos, 4),
    }


@app.post("/api/calibration/apply-offset")
def calibration_apply_offset(req: ApplyOffsetRequest):
    """Hold the posed position: write the goal WITH the offset, then re-power."""
    hwi = _calibration_hwi()
    joint_name = _calibration_joint(hwi, req.jointName)

    hwi.joints_offsets[joint_name] = req.offset

    # This joint now holds STRAIGHT. With its offset set, a policy-zero goal puts the
    # servo at raw = offset, which is exactly where the operator's hands are — so the
    # entry flips to 0 and every later re-assert keeps it there.
    calibration_hold[joint_name] = 0.0

    # Goal FIRST, torque second. set_position_all writes pos + offset, so the goal is
    # where the operator's hands are and the servo wakes already at its target: the
    # joint stiffens in place, which is the null test the operator is judging.
    #
    # The reverse order looks equivalent and is not. At this moment the servo's goal
    # register still holds what begin-joint wrote, so enabling torque first drives the
    # joint back there — yanking it out of the operator's hand — before the new goal
    # arrives and it comes back.
    #
    # Commanding calibration_hold rather than zero_pos is what keeps the promise for
    # the OTHER thirteen: an uncalibrated joint stays where the duck was placed instead
    # of being dragged to servo-zero behind the operator's back.
    hwi.set_position_all(calibration_hold)
    time.sleep(0.5)
    try:
        hwi.set_joint_torque(joint_name, True)
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        # Its own code, not MOTORS_SILENT. Re-powering is the step that hands the
        # joint's weight back to the servo, so failing it leaves the joint LIMP — the
        # opposite state from a failed release, and the opposite thing to tell someone
        # who is deciding whether to let go of it.
        _note_calibration_fault("TORQUE_ENABLE_FAILED", joint_name)
        _agent_error(502, "TORQUE_ENABLE_FAILED", str(e), joint_name)

    add_telemetry_props(
        joint_name=joint_name,
        offset_rad=round(req.offset, 4),
        offset_deg=round(math.degrees(req.offset), 2),
    )
    return {"success": True}


@app.post("/api/calibration/accept")
def calibration_accept(req: AcceptJointRequest):
    """Keep this joint's offset for the session. Nothing is on disk until /save."""
    global calibration_offsets

    hwi = _calibration_hwi()
    joint_name = _calibration_joint(hwi, req.jointName)

    hwi.joints_offsets[joint_name] = req.offset
    calibration_offsets[joint_name] = req.offset

    _bump("accepts", joint_name)
    add_telemetry_props(
        joint_name=joint_name,
        offset_rad=round(req.offset, 4),
        offset_deg=round(math.degrees(req.offset), 2),
        accepted_count=len(calibration_offsets),
        # attempts BEFORE this one stuck: how much work this joint cost
        attempts=calibration_session.get("confirms", {}).get(joint_name, 1),
    )
    return {"success": True, "offsets": calibration_offsets}


# How far the identity check rocks a joint about its held pose, in radians: a few
# degrees, enough to see which leg answers to a name and not enough to matter.
CALIBRATION_WIGGLE_RAD = 0.08


class SwapLegsRequest(BaseModel):
    swapped: bool


@app.post("/api/calibration/wiggle")
def calibration_wiggle(req: JointRequest):
    """Rock ONE held joint a few degrees and put it back: which physical joint answers
    to this name?

    The identity check before the offsets. A build that programmed the right leg's
    servo ids into the left leg's servos passes every other check — each joint reads,
    holds and measures fine — and only shows itself when a name moves the wrong leg.
    So the screen wiggles a right-leg joint, asks which leg moved, and swaps the leg
    names (see /swap-legs) if the answer is the left one.
    """
    hwi = _calibration_hwi()
    joint_name = _calibration_joint(hwi, req.jointName)
    if joint_name not in calibration_hold:
        _note_calibration_fault("INVALID_STATE", joint_name)
        _agent_error(
            409, "INVALID_STATE", "no held pose: call /api/calibration/start first", joint_name
        )
    if joint_name in calibration_baselines:
        _note_calibration_fault("INVALID_STATE", joint_name)
        _agent_error(
            409, "INVALID_STATE", f"{joint_name} is released; a limp joint cannot wiggle", joint_name
        )
    centre = float(calibration_hold[joint_name])
    try:
        for _ in range(2):
            hwi.set_position(joint_name, centre + CALIBRATION_WIGGLE_RAD)
            time.sleep(0.25)
            hwi.set_position(joint_name, centre - CALIBRATION_WIGGLE_RAD)
            time.sleep(0.25)
        hwi.set_position(joint_name, centre)
        time.sleep(0.25)
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        _note_calibration_fault("MOTORS_SILENT", joint_name)
        _agent_error(502, "MOTORS_SILENT", str(e), joint_name)
    add_telemetry_props(joint_name=joint_name)
    return {"success": True, "jointName": joint_name}


@app.post("/api/calibration/swap-legs")
def calibration_swap_legs(req: SwapLegsRequest):
    """Record that this build's left and right leg servo ids are the other way round,
    and rebuild the HWI on it.

    Ends the hold session rather than patching it: the held pose was read per NAME
    under the old naming, so under the new one every value would sit on the other leg.
    The caller arms again with /start, which reads the pose afresh. Torque stays on
    throughout and the servos keep their goals in firmware, so nothing moves.
    """
    global calibration_baselines, calibration_hold, calibration_offsets
    refuse_while_walking()
    config = _config_for_update()
    config["legs_swapped"] = bool(req.swapped)
    _save_config(config)
    calibration_baselines = {}
    calibration_hold = {}
    calibration_offsets = {}
    release_hwi(disable_torque=False)
    telemetry.capture("legs_swapped", {"swapped": bool(req.swapped)})
    return {"success": True, "legsSwapped": bool(req.swapped)}


@app.post("/api/calibration/finish")
def calibration_finish():
    """End the session, leaving no joint limp and moving none. Torque stays ON.

    Without this, leaving the screen mid-session leaves whichever joint was released
    hanging — the operator walks away and the duck sags on one leg. There is no other
    route that re-powers a single joint without also accepting an offset for it.

    Each joint is re-powered at the position it is in NOW, not at its goal register: a
    released joint's register still says where it was before the operator's hands took
    it, and enabling torque against that is a drive, not a hold. Goal first, torque
    second — apply-offset's rule, for apply-offset's reason.

    Idempotent and best-effort per joint: a session that never opened has nothing to
    re-power, and one servo failing to answer must not stop the other thirteen being
    made safe. Reports which joints could not be re-powered rather than raising, because
    the caller is a page being navigated away from and has nowhere to show an error.

    Ends by dropping the HWI singleton bus-only, as /save does. The session zeroed every
    offset in memory, so the singleton carries a calibration that is not the one on
    disk, and the next route to reuse it — /api/head/puppet turns the robot on through
    it — would command its stance through those offsets. A fresh HWI reloads the file.
    """
    global calibration_baselines, calibration_hold

    if hwi_instance is None:
        calibration_baselines = {}
        calibration_hold = {}
        return {"success": True, "repowered": [], "failed": []}

    hwi = hwi_instance
    repowered, failed = [], []
    for joint_name in list(hwi.joints.keys()):
        try:
            hwi.set_position_all({joint_name: hwi.get_present_position(joint_name)})
            hwi.set_joint_torque(joint_name, True)
            repowered.append(joint_name)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            failed.append(joint_name)

    calibration_baselines = {}
    # Cleared with the rest of the session: a stale hold pose would let a later
    # re-assert command a pose the operator never placed the duck in.
    calibration_hold = {}
    release_hwi(disable_torque=False)
    add_telemetry_props(repowered=len(repowered), repower_failed=failed)
    return {"success": True, "repowered": repowered, "failed": failed}


def _config_for_update() -> dict:
    """duck_config.json as a dict to merge into, or the agent error that names why not.

    A duck that has never been configured is not an error: this is its first
    calibration, and _write_config creates the file. The old code let FileNotFoundError
    become a 500, so fourteen joints of work failed to save on exactly the robot most
    likely to be doing this for the first time.

    A READ failure never reaches a write. It is almost always a corrupt
    duck_config.json — a half-written file from a previous power cut, or a hand-edit
    with a trailing comma. Refusing is correct: the file also holds start_paused,
    imu_upside_down and expression_features, and overwriting a file we could not
    parse would destroy all of it to save one section.
    """
    try:
        return _read_config()
    except FileNotFoundError:
        return {}
    except Exception as e:
        _note_calibration_fault("CONFIG_WRITE_FAILED")
        _agent_error(500, "CONFIG_WRITE_FAILED", f"could not read the existing config: {e}")


def _save_config(config: dict) -> None:
    """Write duck_config.json, or raise the agent error that names why not.

    A full card gets its own code, because it is the one cause here an operator can
    actually act on — free space, or swap the card. Every other write failure ends in
    "nothing was saved" with nothing to do about it, and folding them together would
    bury the one that has an action.

    ENOSPC comes from the .bak copy, not the config write. Measured on a real Pi
    (kernel 6.18.34-rpi-v8, ext4): with the filesystem at literally 0 bytes free, the
    failure lands at the backup step. It also needs to be at 0 — 3 KB free was enough
    for a 569-byte config to save fine — so this is a genuinely full card, not a
    nearly-full one.

    A read-only card is the classic Pi death, and the most likely of the four. The
    kernel remounts ext4 read-only when a failing card starts erroring, so the robot
    keeps running perfectly from RAM and only writes fail — which is exactly why
    "just a file write" is worth its own code. Naming the cause IS the action: an
    operator told their card is read-only knows to reflash or replace it, and nothing
    they do on this screen will help. errno 30 verified against a real read-only ext4
    on the robot, not assumed.

    What is left: the file owned by root from a sudo'd setup step while this server
    runs as pi, and anything else the OS reports.
    """
    try:
        _write_config(config)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            _note_calibration_fault("AGENT_DISK_FULL")
            _agent_error(507, "AGENT_DISK_FULL", f"no space left writing the config: {e}")
        if e.errno == errno.EROFS:
            _note_calibration_fault("AGENT_DISK_READONLY")
            _agent_error(500, "AGENT_DISK_READONLY", f"filesystem is read-only: {e}")
        _note_calibration_fault("CONFIG_WRITE_FAILED")
        _agent_error(500, "CONFIG_WRITE_FAILED", f"could not write the config: {e}")
    except Exception as e:
        _note_calibration_fault("CONFIG_WRITE_FAILED")
        _agent_error(500, "CONFIG_WRITE_FAILED", f"could not write the config: {e}")


@app.post("/api/calibration/save")
def calibration_save():
    """Merge the session's offsets into duck_config.json. Torque stays ON."""
    global calibration_offsets, calibration_baselines, calibration_hold

    # Read before release_hwi() drops the singleton — the summary event needs the full
    # joint list to say which joints were SKIPPED, and skipped joints are by definition
    # the ones missing from calibration_offsets.
    hwi = hwi_instance
    all_joints = list(hwi.joints.keys()) if hwi is not None else []

    config = _config_for_update()
    if "joints_offsets" not in config:
        config["joints_offsets"] = {}
    config["joints_offsets"].update(calibration_offsets)
    _save_config(config)

    # Free the bus WITHOUT dropping torque. This used to be a bare release_hwi(),
    # whose disable_torque default is True - so the duck went limp the instant the
    # operator saved, standing at the tall straight-leg zero pose after fourteen
    # joints of work. Feetech servos hold position in firmware with no bus traffic,
    # so freeing the port never requires going limp. Same fix, same reason, as
    # stance_save.
    release_hwi(disable_torque=False)

    _capture_calibration_saved(all_joints)
    add_telemetry_props(joints_calibrated=len(calibration_offsets))

    # The hold is over once the offsets are on disk. Left behind, a Redo after saving
    # would re-assert the placed pose through a freshly built HWI whose offsets are
    # now the saved ones rather than the zeros the pose was recorded against, and
    # every skipped joint would move by its on-disk offset. calibration_offsets stays:
    # a save that fails is retried against it.
    calibration_baselines = {}
    calibration_hold = {}
    return {"success": True, "offsets": config["joints_offsets"]}


def _capture_calibration_saved(all_joints: list) -> None:
    """The one event that carries a whole calibration.

    The per-request events already hold each joint's numbers, but they cannot answer
    the questions that matter across a fleet: did this session finish, which joints
    were skipped, how many attempts did it cost, and how close does the result sit to
    the servo's command seam. Those are session-shaped, so this is a session-shaped
    event.

    Fail-silent, and AFTER the write: telemetry must never be the reason a calibration
    fails to save.
    """
    try:
        session = calibration_session or {}
        offsets = dict(calibration_offsets)
        init_pos = session.get("init_pos", {})
        magnitudes = [abs(math.degrees(v)) for v in offsets.values()]
        headroom = _seam_headroom(offsets, init_pos)
        started = session.get("started_at")
        confirms = session.get("confirms", {})
        faults = dict(session.get("faults", {}))
        telemetry.capture(
            "joint_calibration_saved",
            {
                # The values themselves, in both units: radians is what the config
                # holds, degrees is what a human reads on a chart.
                "offsets_rad": {k: round(v, 4) for k, v in offsets.items()},
                "offsets_deg": {k: round(math.degrees(v), 2) for k, v in offsets.items()},
                "joints_calibrated": len(offsets),
                "joints_total": len(all_joints),
                # Named, not just counted. "Everybody skips the head joints" and "this
                # one robot skipped a knee" are different findings, and a count cannot
                # tell them apart.
                "joints_skipped": [j for j in all_joints if j not in offsets],
                "max_abs_offset_deg": round(max(magnitudes), 2) if magnitudes else 0.0,
                "mean_abs_offset_deg": (
                    round(sum(magnitudes) / len(magnitudes), 2) if magnitudes else 0.0
                ),
                # How close the result puts each joint to the +-180 seam once the walk
                # adds init_pos on top. The lowest number is the one that matters, and
                # it is the evidence for or against building the guard.
                "seam_headroom_deg": headroom,
                "min_seam_headroom_deg": round(min(headroom.values()), 1) if headroom else None,
                # Effort. Total measurements against joints kept says how often a joint
                # had to be redone, which is the assembly-instructions signal.
                "measure_attempts": sum(confirms.values()),
                "joints_needing_retry": [j for j, n in confirms.items() if n > 1],
                "faults": faults,
                "fault_count": sum(faults.values()),
                "duration_s": round(time.time() - started, 1) if started else None,
            },
        )
    except Exception:
        pass


# ── Joint directions (per-joint signs) ───────────────────────────────────────
# The last thing between a fresh build and a walk: does each servo turn the way the
# model expects? A mirrored horn passes every position read — the servo tracks its
# numeric goal either way — and only shows itself when the joint MOVES, which on a
# walk means driving that joint into its own shell at full stiffness. So this moves
# the left and right joint of a pair together at low stiffness and asks the operator
# two questions: did both sides move the same way (else which side is wrong — flip
# that joint), and was it the way the model shows (else flip both). The signs flip
# LIVE in the HWI, so the retest uses them, and persist to duck_config.json as
# "joints_signs", which the HWI applies to every command and read from then on.
#
# Pairs, not single joints, because a mirrored joint is easiest to see beside its
# twin, and the model's mirrored axes (left hip pitch -0.63, right +0.635 in the
# stance) are exactly the trap nobody can judge from a number.
#
# The targets are a fraction of the walking stance in each joint's own direction, so
# the expected motion is "a bit of the crouch": the safest way to move a straight leg
# on a stand, and what the picture on screen is drawn from.

DIRECTION_HOLD_KP = 8  # low stiffness: a wrong-way joint meeting its shell stalls softly

DIRECTION_PAIRS = [
    {
        "id": "hip_pitch",
        "label": "Hip pitch",
        "left": "left_hip_pitch",
        "leftTarget": -0.3,
        "right": "right_hip_pitch",
        "rightTarget": 0.3,
        "expect": "Both thighs swing backward, feet toward the tail.",
    },
    {
        "id": "knee",
        "label": "Knee",
        "left": "left_knee",
        "leftTarget": 0.5,
        "right": "right_knee",
        "rightTarget": 0.5,
        "expect": "Both feet swing forward and the knees poke backward, like a bird's.",
    },
    {
        "id": "ankle",
        "label": "Ankle",
        "left": "left_ankle",
        "leftTarget": -0.4,
        "right": "right_ankle",
        "rightTarget": -0.4,
        "expect": "Both feet tip toes-down, heels up.",
    },
]

# The open session: which pair, if any, is displaced right now. None outside one.
directions_session: dict | None = None


class PairRequest(BaseModel):
    pairId: str


def _direction_pair(pair_id: str) -> dict:
    for pair in DIRECTION_PAIRS:
        if pair["id"] == pair_id:
            return pair
    _agent_error(400, "UNKNOWN_PAIR", f"no such joint pair: {pair_id}")


def _directions_open() -> dict:
    if directions_session is None:
        _agent_error(
            409, "INVALID_STATE", "no direction session: call /api/directions/start first"
        )
    return directions_session


def _write_pair(hwi, pair: dict, extended: bool) -> None:
    """Both joints of the pair to their targets, or both back to straight."""
    for side in ("left", "right"):
        joint = pair[side]
        target = pair[f"{side}Target"] if extended else 0.0
        try:
            hwi.set_position(joint, target)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            _note_calibration_fault("MOTORS_SILENT", joint)
            _agent_error(502, "MOTORS_SILENT", str(e), joint)


@app.post("/api/directions/start")
def directions_start():
    """Stand the duck straight at low stiffness, ready to move pairs.

    Straight IS a drive here, and a deliberate one: this runs after the offsets are
    set, so zero is the straight pose by construction, and kp 8 is what the operator's
    own script used so that a wrong-way joint stalls softly instead of pushing.
    """
    global directions_session
    hwi = _calibration_hwi()

    # the whole-bus liveness check, before anything is commanded
    if hwi.get_present_positions() is None:
        _note_calibration_fault("MOTORS_SILENT")
        _agent_error(502, "MOTORS_SILENT", "at least one joint did not answer")

    joint_names = list(hwi.joints.keys())
    try:
        hwi.set_kds([0] * len(joint_names))
        hwi.set_kps([DIRECTION_HOLD_KP] * len(joint_names))
        hwi.set_position_all(hwi.zero_pos)
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        _note_calibration_fault("MOTORS_SILENT")
        _agent_error(502, "MOTORS_SILENT", str(e))
    # goal first, torque second: the calibration rule, for the calibration reason
    for joint_name in joint_names:
        try:
            hwi.set_joint_torque(joint_name, True)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            _note_calibration_fault("TORQUE_ENABLE_FAILED", joint_name)
            _agent_error(502, "TORQUE_ENABLE_FAILED", str(e), joint_name)
    time.sleep(2.0)

    directions_session = {"moved": None}
    signs = {name: _joint_sign(hwi, name) for name in joint_names}
    add_telemetry_props(inverted_before=[j for j, s in signs.items() if s == -1])
    return {"pairs": DIRECTION_PAIRS, "signs": signs}


@app.post("/api/directions/move")
def directions_move(req: PairRequest):
    """Both joints of the pair to their targets. Answers once they have had time to
    arrive, so the operator is asked about a motion that has finished."""
    hwi = _calibration_hwi()
    session = _directions_open()
    pair = _direction_pair(req.pairId)
    _write_pair(hwi, pair, extended=True)
    session["moved"] = pair["id"]
    time.sleep(1.5)
    add_telemetry_props(pair=pair["id"])
    return {"success": True, "pairId": pair["id"]}


@app.post("/api/directions/rest")
def directions_rest(req: PairRequest):
    """Both joints of the pair back to straight."""
    hwi = _calibration_hwi()
    session = _directions_open()
    pair = _direction_pair(req.pairId)
    _write_pair(hwi, pair, extended=False)
    session["moved"] = None
    time.sleep(1.0)
    return {"success": True, "pairId": pair["id"]}


@app.post("/api/directions/flip")
def directions_flip(req: JointRequest):
    """Invert this joint's direction, live.

    Nothing moves on its own: straight is raw = offset whichever way the sign points,
    and the next move or rest is what uses the new sign. Live only — /save is what
    writes it to the config, and leaving without saving forgets it, because the next
    HWI is built from the file.
    """
    hwi = _calibration_hwi()
    _directions_open()
    joint_name = _calibration_joint(hwi, req.jointName)
    hwi.joints_signs[joint_name] = -_joint_sign(hwi, joint_name)
    signs = {name: _joint_sign(hwi, name) for name in hwi.joints}
    add_telemetry_props(joint_name=joint_name, sign=signs[joint_name])
    return {"jointName": joint_name, "sign": signs[joint_name], "signs": signs}


@app.post("/api/directions/save")
def directions_save():
    """Write the signs into duck_config.json. Torque stays ON, the bus is freed.

    Every joint is written, not only the inverted ones, so a person reading the file
    sees each joint's direction stated rather than inferred from an absence.
    """
    global directions_session
    hwi = _calibration_hwi()
    session = _directions_open()

    signs = {name: _joint_sign(hwi, name) for name in hwi.joints}
    config = _config_for_update()
    config["joints_signs"] = signs
    _save_config(config)

    if session.get("moved"):
        _write_pair(hwi, _direction_pair(session["moved"]), extended=False)
    directions_session = None
    # Same hand-off as the offsets: the servos hold in firmware, the port is freed,
    # and the next HWI is built from the file that now carries the signs.
    release_hwi(disable_torque=False)

    inverted = [j for j, s in signs.items() if s == -1]
    telemetry.capture(
        "joint_directions_saved",
        {"signs": signs, "inverted": inverted, "inverted_count": len(inverted)},
    )
    return {"success": True, "signs": signs}


@app.post("/api/directions/finish")
def directions_finish():
    """End the session: rest a displaced pair, keep torque, free the bus.

    Idempotent and best-effort, like calibration/finish: the caller is a page being
    navigated away from. Unsaved flips are forgotten with the HWI, which is the point
    of a save step.
    """
    global directions_session
    if directions_session is None or hwi_instance is None:
        directions_session = None
        return {"success": True}

    moved = directions_session.get("moved")
    if moved:
        try:
            _write_pair(hwi_instance, _direction_pair(moved), extended=False)
            time.sleep(0.5)
        except HTTPException:
            pass
    directions_session = None
    release_hwi(disable_torque=False)
    return {"success": True}


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

    if is_walking():
        raise HTTPException(
            status_code=409, detail="Cannot rehome while a walk is running"
        )
    with BUS_LOCK:
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
    """Close the rehoming session: free the bus but KEEP torque on, so the duck
    stays rigid holding the fresh zero pose it built up joint by joint. IMU
    calibration runs next on that rigid, upright duck; stance calibration goes
    limp explicitly via /api/stance/release. Server shutdown (lifespan) still
    drops torque so an unplugged duck never fights being handled."""
    if rehome_io is None:
        return {"success": True, "message": "No rehoming session was running"}
    _close_rehome_io(disable_torque=False)
    return {"success": True}


# ── Stance (initial pose) calibration ─────────────────────────────────────────
# Ported from Sam's walk_server.py offset flow: release all torque, hand-pose
# the whole duck into its standing stance, capture every offset at once
# (offset = raw - init_pos, self-correcting no matter how far the old offsets
# drifted), then hold the pose and fine-tune per-joint offsets live before
# saving to duck_config.json. After a save from the holding state the duck
# KEEPS holding the stance (torque stays on) — it doesn't go limp and fall.

# Position-mode STS3215 servos are drivable over ~±π rad. A commanded target
# (init_pos + offset) beyond this can't be reached: the servo clamps/wraps, so
# the held pose won't match what was captured. A hair inside π for margin.
SERVO_RANGE_RAD = 3.05

stance_holding = False


def _stance_unreachable(hwi) -> list[str]:
    """Joints whose commanded target falls outside the servo's drivable window."""
    # the raw servo target is sign * init_pos + offset (see HWI.joints_signs)
    return [
        name
        for name in hwi.joints
        if abs(
            _joint_sign(hwi, name) * float(hwi.init_pos[name])
            + float(hwi.joints_offsets.get(name, 0.0))
        )
        > SERVO_RANGE_RAD
    ]


def _stance_offsets(hwi) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in hwi.joints_offsets.items()}


@app.post("/api/stance/start")
def stance_start():
    """Begin a stance session with a pristine HWI (offsets reloaded from disk)."""
    global stance_holding
    refuse_while_walking()
    # Reload so leftovers from other flows (e.g. the deprecated calibration
    # endpoints mutate init_pos) can't leak into the stance session. Bus-only:
    # re-entering the wizard must not drop a duck still holding its stance —
    # the user goes limp explicitly via /api/stance/release.
    try:
        release_hwi(disable_torque=False)
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

    get_present_positions() returns sign * (raw - offset_current), so:
        offset_new = offset_current + sign * (present - init_pos) = raw - sign * init_pos
    which makes the robot's CURRENT pose read back as init_pos, no matter how
    far the old offsets had drifted, and whichever way the joint is mounted.
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
        sign = _joint_sign(hwi, name)
        hwi.joints_offsets[name] = round(
            cur + sign * (float(p) - float(hwi.init_pos[name])), 4
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
    sign = _joint_sign(hwi, req.jointName)
    # the raw servo target is sign * init + offset; the offset is what is stored
    target = sign * init + float(req.offset)
    clamped_target = max(-SERVO_RANGE_RAD, min(SERVO_RANGE_RAD, target))
    hwi.joints_offsets[req.jointName] = round(clamped_target - sign * init, 4)

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
        with BUS_LOCK:
            hwi = get_hwi()
            raw = hwi.io.read_present_position(list(hwi.joints.values()))
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "positions": {
            name: round(float(p), 4) for name, p in zip(hwi.joints.keys(), raw)
        }
    }


@app.post("/api/stance/save")
def stance_save():
    """Persist the session's offsets to duck_config.json (with a .bak backup)."""
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

    if not stance_holding:
        # Saved while limp (user never pressed hold) — nothing to keep
        # energized; next use reloads the saved config from disk.
        release_hwi()
    # else: keep torque on so the duck stays in the saved stance — cutting it
    # here made the robot collapse the moment the user saved. The in-memory
    # offsets equal what was just written to disk, and every other bus
    # consumer (walk start, rehome, voltage) releases the HWI itself first.

    return {"success": True, "offsets": offsets}


# ── Head puppet (on-screen joystick drives the head) ─────────────────────────
#
# The HTTP half of scripts/head_puppet.py, which needs a physical Xbox pad.
# Idle only: while a walk runs it owns the bus, and head targets already ride
# along in the /api/commands stream (indices 3..6), so there is nothing to add
# there — a second writer would just fight the joystick stream for the file.
#
# Dashboard flow: POST axes as the user drags (the first one enters the mode),
# /stop when the panel closes.

# Mechanical travel per head joint, degrees — the values head_puppet.py uses.
HEAD_LIMITS_DEG = {
    "neck_pitch": (-20.0, 60.0),
    "head_pitch": (-60.0, 45.0),
    "head_yaw": (-60.0, 60.0),
    "head_roll": (-20.0, 20.0),
}

# Axis name in the request -> joint it drives. Named for what the user is
# doing to the head, so the frontend never has to know the joint names.
HEAD_AXES = {
    "yaw": "head_yaw",
    "pitch": "head_pitch",
    "roll": "head_roll",
    "neckPitch": "neck_pitch",
}

# A compliant head is smooth to puppet and safe to catch hold of. Only the
# head servos are softened — dropping the legs to this would fold a duck that
# is standing, which is exactly the state we expect to be puppeted in.
HEAD_PUPPET_KP = 8

head_puppet_active = False
# Last axis positions, so a request carrying only the axes that moved leaves
# the rest of the head where it is (a 2-axis on-screen stick shouldn't yank
# roll back to centre every frame).
head_puppet_axes = {name: 0.0 for name in HEAD_AXES}


def _head_axis_to_rad(joint: str, axis: float) -> float:
    """Stick axis (-1..1) -> joint angle, radians. 0 is the middle of the
    joint's travel, matching head_puppet.py — note that for the asymmetric
    joints (head_pitch, neck_pitch) that middle is not the init pose."""
    lo, hi = HEAD_LIMITS_DEG[joint]
    return math.radians(lo + (axis + 1.0) / 2.0 * (hi - lo))


def _head_puppet_payload(**extra) -> dict:
    return {
        "active": head_puppet_active,
        "axes": dict(head_puppet_axes),
        "applied": {
            joint: round(math.degrees(_head_axis_to_rad(joint, head_puppet_axes[axis])), 2)
            for axis, joint in HEAD_AXES.items()
        },
        "limits": {j: list(v) for j, v in HEAD_LIMITS_DEG.items()},
        "axisNames": list(HEAD_AXES),
        **extra,
    }


class HeadPuppetRequest(BaseModel):
    # All optional: a request moves only the axes it names. -1..1, clamped.
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    neckPitch: float | None = None


@app.post("/api/head/puppet")
def head_puppet(req: HeadPuppetRequest):
    """Point the head from on-screen joystick axes, in -1..1.

    The first call enters puppet mode — torque on, head softened to
    HEAD_PUPPET_KP — so the frontend can just start streaming axes as the user
    drags. Every call after that is only the four servo writes, which is what
    keeps this cheap enough to post at joystick rates.
    """
    global head_puppet_active

    requested = {
        axis: getattr(req, axis) for axis in HEAD_AXES if getattr(req, axis) is not None
    }
    # inf/NaN survive pydantic's float coercion and would land in a goal
    # position; reject before anything moves.
    for axis, value in requested.items():
        if not math.isfinite(value):
            raise HTTPException(
                status_code=400, detail=f"Axis {axis} must be a finite number"
            )

    refuse_while_walking()
    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    entering = not head_puppet_active
    if entering:
        try:
            # Stand the duck up in its init pose, as head_puppet.py does, then
            # soften the head alone. turn_on() leaves legs at their normal kps,
            # so a duck already holding a stance stays rigid through this.
            hwi.turn_on()
            for joint in HEAD_LIMITS_DEG:
                hwi.set_kp(hwi.joints[joint], HEAD_PUPPET_KP)
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
        head_puppet_active = True

    for axis, value in requested.items():
        head_puppet_axes[axis] = max(-1.0, min(1.0, float(value)))

    try:
        for axis, joint in HEAD_AXES.items():
            hwi.set_position(joint, _head_axis_to_rad(joint, head_puppet_axes[axis]))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not move head: {e}")

    if entering:
        add_telemetry_props(entered=True)
    return _head_puppet_payload(success=True)


@app.post("/api/head/puppet/stop")
def head_puppet_stop():
    """Leave puppet mode: head back to centre, normal kps restored.

    Torque stays on — same rule as the stance endpoints, since cutting it
    would drop a duck that is standing. Use /api/stance/release to go limp.
    """
    global head_puppet_active
    if not head_puppet_active:
        return _head_puppet_payload(success=True, message="Head puppet was not active")

    # Drop the mode first: if the bus write below fails, the next request must
    # re-enter cleanly rather than believe a head it never re-centred is live.
    head_puppet_active = False
    for axis in head_puppet_axes:
        head_puppet_axes[axis] = 0.0

    # No is_walking() guard needed: walk start releases the HWI, which clears
    # head_puppet_active, so a session never survives into a walk to be
    # stopped here.
    try:
        hwi = get_hwi()
        for axis, joint in HEAD_AXES.items():
            hwi.set_position(joint, _head_axis_to_rad(joint, 0.0))
        # hwi.kps is indexed by joint order — restore each head joint to the
        # stiffness the rest of the robot is running at.
        joint_order = list(hwi.joints)
        for joint in HEAD_LIMITS_DEG:
            hwi.set_kp(hwi.joints[joint], float(hwi.kps[joint_order.index(joint)]))
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return _head_puppet_payload(success=True)


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
        import pickle

        # Reuse the state IMU's chip handle rather than opening a second one:
        # adafruit_bno055's constructor soft-resets the BNO055 (SYS_TRIGGER
        # 0x20, "reset to default settings"), which wipes the axis remap
        # raw_imu.Imu applied for this robot's mounting. Every /api/state
        # quaternion after that reset is in the chip's factory frame — the
        # studio's 3D duck renders tipped over while the real duck stands
        # upright. Sharing the handle keeps the remap (and the live
        # orientation view) intact; the Imu constructor already put the chip
        # in NDOF_MODE.
        imu = get_state_imu().imu

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
    """Atomic write (temp + rename) with a .bak of the previous version — a
    power cut mid-write must never cost the calibration in duck_config.json.

    The BACKUP is staged through a temp file too, and that is not symmetry for its own
    sake. Measured on a real Pi with a full card: `shutil.copyfile` CREATES the
    destination and then fails on the first write, so copying straight to `.bak` left a
    zero-byte backup while the original was still intact — the one moment the safety
    net is needed is the one moment it had just been destroyed. Staging means a failed
    copy leaves `.bak.tmp` behind and the real `.bak` untouched.

    Same reason the config itself goes through `.tmp`; the backup had simply been
    forgotten.
    """
    if os.path.exists(CONFIG_PATH):
        bak_tmp = CONFIG_PATH + ".bak.tmp"
        shutil.copyfile(CONFIG_PATH, bak_tmp)
        os.replace(bak_tmp, CONFIG_PATH + ".bak")
    tmp_path = CONFIG_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=4)
    os.replace(tmp_path, CONFIG_PATH)


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
    # Merged over the file, not written wholesale. Studio sends the fields it knows,
    # and a rewrite from those alone dropped everything else: the joint directions a
    # calibration had just saved, and any key a newer runtime adds before Studio
    # learns it. Only the fields the client actually sent replace what is there.
    try:
        existing = _read_config()
    except Exception:
        existing = {}
    merged = dict(existing)
    merged.update(config.model_dump(exclude_unset=True))
    try:
        _write_config(merged)
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
    global walk_session, last_walk_exit_code
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
    if session is not None:
        last_walk_exit_code = session.proc.returncode
    # Only clear if no newer session was installed meanwhile.
    if walk_session is session:
        walk_session = None

    # Clean up remote command file
    try:
        os.remove(COMMAND_FILE)
    except FileNotFoundError:
        pass
    # A pause belongs to the walk that was paused — never inherit it
    walk_pause.clear()
    # Same for an unsaved live trim: the next walk starts from duck_config.json,
    # not from whatever the last session was mid-experiment with.
    walk_offsets.clear()
    # Same rule as the walk's own shutdown: no dead telemetry snapshots
    walk_telemetry.clear()


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
    # Who steers this walk. Omitted (old Studio / dashboard) means keyboard.
    input: str | None = None


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

    if rehome_io is not None:
        raise HTTPException(
            status_code=409,
            detail="Servo rehoming session in progress — finish it before walking",
        )

    if walk_session is not None and walk_session.proc.poll() is None:
        if body.sessionToken and body.sessionToken == walk_session.session_token:
            add_telemetry_props(already_running=True)
            return {"success": True, "message": "Walk is already running"}
        stop_walk_process()

    # Release HWI + BNO055 so the walk script owns both buses (serial + I2C);
    # /api/state serves the walk's telemetry snapshot while it runs.
    # Bus-only release: the servos keep holding their pose in firmware while
    # the walk process boots (several seconds of python/onnxruntime imports),
    # so a duck standing in its stance doesn't collapse; the walk script's
    # own turn_on() then takes over from that pose.
    release_hwi(disable_torque=False)
    # The walk subprocess constructs its own Imu, whose adafruit constructor
    # soft-resets the BNO055 — stop any running calibration first so its
    # worker doesn't keep polling (and fighting over I2C with) a chip it no
    # longer owns.
    imu_calib_status["running"] = False
    release_state_imu()

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
        walk_input = body.input or "keyboard"
        # Which device is driving. Absent until now, which meant the one
        # question worth asking about the controller -- did anyone ever manage
        # to drive with it -- could not be answered from the data at all.
        add_telemetry_props(walk_input=walk_input)
        if walk_input == "pad" and not joystick_present():
            # The pad was chosen and the stick is not there. Say whether the
            # radio is the reason, so this does not read as a flat battery.
            # From the cache, not a fresh probe: Studio polls /api/pad the whole
            # time its pair step is open, so this is current, and a walk start
            # is the wrong place to spend two subprocess spawns.
            add_telemetry_props(adapter_reason=_last_adapter_reason())
            raise HTTPException(
                status_code=409,
                detail="PAD_NOT_FOUND: no joystick",
            )

        cmd = [
            venv_python,
            walk_script,
            "--onnx_model_path", onnx_path,
            *walk_flags(walk_input),
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

    # Whether joint data streams to the cloud (boolean only — never the
    # token value or the joint stream itself).
    cloud_streaming = bool(body.sessionToken and body.supabaseUrl and body.supabaseKey)
    add_telemetry_props(
        cloud_streaming=cloud_streaming,
        has_session=bool(body.sessionToken),
        already_running=False,
    )

    # Start from a clean slate. stop_walk_process() clears these, but a walk
    # that died hard (SIGKILL, power cut) never ran its own cleanup — and
    # is_walking() goes true the moment we spawn, seconds before the walk
    # publishes its own state, so a leftover file would read as *this* walk's
    # pause/trim for the whole boot window.
    walk_pause.clear()
    walk_offsets.clear()

    proc = subprocess.Popen(cmd, cwd=str(SCRIPTS_DIR))
    walk_session = WalkSession(
        proc=proc,
        session_token=body.sessionToken,
        cloud_streaming=cloud_streaming,
        started_at=time.monotonic(),
    )
    Thread(target=_monitor_walk, args=(walk_session,), daemon=True).start()

    return {"success": True, "pid": proc.pid}


class PadPairRequest(BaseModel):
    address: str | None = None


#: "we have not looked yet". A distinct sentinel because None is a real reason
#: value here -- it means the adapter is fine -- and the first observation of a
#: healthy radio is not a change worth an event.
_PAD_ADAPTER_UNSEEN = object()
_pad_adapter_reason = _PAD_ADAPTER_UNSEEN


def _last_adapter_reason() -> str:
    """The radio's state as of the last pad call, without touching the radio."""
    if _pad_adapter_reason is _PAD_ADAPTER_UNSEEN:
        return "unknown"
    return _pad_adapter_reason or "ok"


def _note_pad(status: dict) -> dict:
    """Attach the radio's state to this request, and announce a change once.

    Every pad route returns through here, so `api_request_completed` carries
    `adapter_reason` for the whole pad surface. That property is the one that
    was missing: an empty `Nearby` list is ordinary, an empty list on a blocked
    radio is a bug report, and until now both arrived as the same silence.

    The dedicated event is transition-gated on purpose. Studio re-scans every
    couple of seconds while its pair step is open, and a per-poll event would
    turn one stuck operator into thousands of rows saying the same thing.
    """
    global _pad_adapter_reason
    adapter = status.get("adapter") or {}
    reason = adapter.get("reason")
    add_telemetry_props(
        adapter_present=bool(adapter.get("present")),
        adapter_powered=bool(adapter.get("powered")),
        adapter_blocked=bool(adapter.get("blocked")),
        adapter_hard_blocked=bool(adapter.get("hardBlocked")),
        adapter_reason=reason or "ok",
        # Which escalation step woke the radio, and BlueZ's own sentence when
        # none of them did. `woke_via` is the property that answers "why did the
        # shipped power-on not work" without anyone having to reproduce it.
        adapter_woke_via=adapter.get("wokeVia") or "none",
        adapter_wake_error=adapter.get("wakeError"),
        pad_devices=len(status.get("devices") or []),
        pad_connected=bool(status.get("connected")),
    )
    if reason != _pad_adapter_reason:
        previous = _pad_adapter_reason
        _pad_adapter_reason = reason
        first_look = previous is _PAD_ADAPTER_UNSEEN
        if not (first_look and reason is None):
            telemetry.capture(
                "pad_adapter_changed",
                {
                    "reason": reason or "ok",
                    "previous": None if first_look else (previous or "ok"),
                    "woke_via": adapter.get("wokeVia") or "none",
                    "wake_error": adapter.get("wakeError"),
                },
            )
    return status


@app.get("/api/pad")
def get_pad():
    return _note_pad(pad_status())


@app.post("/api/pad/scan")
def post_pad_scan():
    return _note_pad(scan_pad())


@app.post("/api/pad/pair")
def post_pad_pair(body: PadPairRequest = PadPairRequest()):
    try:
        result = _note_pad(pair_pad(body.address))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result.get("connected"):
        # Name the radio when the radio is why. PAD_PAIR_FAILED sends the
        # operator back to the sync button, which is the wrong button when the
        # adapter is blocked -- that was the whole of the 30 minutes this
        # endpoint used to cost.
        adapter = result.get("adapter") or {}
        if adapter.get("reason"):
            raise HTTPException(
                status_code=409,
                detail="PAD_RADIO_OFF: bluetooth adapter %s" % adapter["reason"],
            )
        raise HTTPException(status_code=409, detail="PAD_PAIR_FAILED: did not bond")
    return result


@app.post("/api/pad/disconnect")
def post_pad_disconnect(body: PadPairRequest = PadPairRequest()):
    try:
        return _note_pad(disconnect_pad(body.address))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pad/forget")
def post_pad_forget(body: PadPairRequest = PadPairRequest()):
    try:
        return _note_pad(forget_pad(body.address))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pad/adapter/on")
def post_pad_adapter_on():
    """Turn the radio on, because someone asked for it.

    The only caller of `wake_adapter` in this process. Every other pad route
    reads the radio and refuses when it is down, so this endpoint is the whole
    of the escalation ladder\'s reach: an operator pressing Turn on.

    It returns a full pad status rather than the adapter alone, so the caller
    that just powered the radio up gets the device list of the room it can now
    see, in the same round trip. The previous radio-only route returned a body
    with no `present` key, which is not a PadStatus and which nothing ever
    called; `GET /api/pad` already carries the adapter for anyone who wants it.
    """
    return _note_pad(pad_status(wake_adapter()))


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


def _set_walk_pause(paused: bool):
    if not is_walking():
        raise HTTPException(
            status_code=409,
            detail="No walk is running — nothing to pause",
        )
    walk_pause.write(paused)
    add_telemetry_props(paused=paused)
    return {"success": True, "paused": paused}


@app.post("/api/walk/pause")
def walk_pause_endpoint():
    """Freeze the gait while keeping the servos powered.

    The difference from /api/walk/stop: stopping kills the walk process, whose
    cleanup disables torque, so the duck goes limp and falls. Pausing leaves the
    walk running — it just stops issuing new motor targets, and Feetech servos
    hold their last commanded target in firmware, so the duck stays standing
    exactly where it was. The walk picks the flag up within one control tick
    (~20ms). Same state the controller's A button toggles, so the two agree.
    """
    return _set_walk_pause(True)


@app.post("/api/walk/resume")
def walk_resume():
    """Resume a paused walk — the policy takes over from the held pose."""
    return _set_walk_pause(False)


# ── Live joint trim (offsets, adjustable mid-gait) ────────────────────────────
#
# The stance wizard trims a standing duck; a sagging hip or a toed-in ankle
# only shows up in the gait, so tuning it there means a stop/adjust/start cycle
# per guess. These endpoints move joints_offsets while the walk runs — the walk
# reads the shared file each tick and slews to the new values.
#
# Dashboard flow: GET to populate the sliders, POST deltas as the user nudges,
# /save when the duck looks right, /reset to get back to the saved values.

class WalkOffsetsRequest(BaseModel):
    offsets: dict[str, float]
    # "absolute" sets the offset outright (sliders); "delta" adds to the
    # current live value (nudge buttons, which shouldn't have to know it).
    mode: str = "absolute"


def _saved_offsets() -> dict[str, float]:
    """The trim in duck_config.json — what the walk starts from, and the
    anchor a live trim may only stray TRIM_LIMIT_RAD from."""
    try:
        saved = _read_config().get("joints_offsets", {})
    except (FileNotFoundError, ValueError, OSError):
        saved = {}
    return {name: float(saved.get(name, 0.0)) for name in JOINTS}


def _live_offsets() -> dict[str, float]:
    """The trim in effect right now: the walk's published values while it
    runs, otherwise what the next walk will start with."""
    live = walk_offsets.read() if is_walking() else None
    saved = _saved_offsets()
    if live is None:
        return saved
    # Only joints we know: a stray key in the file must not invent a joint.
    return {name: round(live.get(name, saved[name]), 4) for name in JOINTS}


def _offsets_payload(**extra) -> dict:
    saved = _saved_offsets()
    offsets = _live_offsets()
    return {
        "walking": is_walking(),
        "offsets": offsets,
        "saved": saved,
        # Trims the user would lose by stopping the walk without saving.
        "unsaved": sorted(
            name for name in JOINTS if abs(offsets[name] - saved[name]) > 1e-6
        ),
        "joints": list(JOINTS),
        "limit": walk_offsets.TRIM_LIMIT_RAD,
        "rampRate": walk_offsets.RAMP_RATE_RAD_S,
        **extra,
    }


@app.get("/api/walk/offsets")
def walk_offsets_get():
    """Current trim, the saved baseline, and the limits — everything the
    dashboard needs to render its sliders. Works whether or not a walk is
    running; idle, `offsets` is what the next walk will start from."""
    return _offsets_payload()


@app.post("/api/walk/offsets")
def walk_offsets_set(req: WalkOffsetsRequest):
    """Adjust the trim of a live gait, one joint or many.

    Absolute mode replaces the offset, delta mode adds to it, and a partial
    dict leaves every other joint alone — so a slider and a +/- button can
    both post here. Values are clamped to the saved value ± limit, and the
    walk ramps to them at rampRate rad/s rather than snapping, so a whole
    session of trimming never asks the duck to take a step it can't.
    """
    if req.mode not in ("absolute", "delta"):
        raise HTTPException(
            status_code=400, detail=f"mode must be 'absolute' or 'delta', got {req.mode!r}"
        )
    if not is_walking():
        raise HTTPException(
            status_code=409,
            detail="No walk is running — trim a standing duck with /api/stance/offset",
        )
    unknown = [name for name in req.offsets if name not in JOINTS]
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"Unknown joint(s): {', '.join(sorted(unknown))}"
        )
    for name, value in req.offsets.items():
        # inf/NaN survive pydantic's float coercion and would be written
        # straight into a goal position.
        if not math.isfinite(value):
            raise HTTPException(
                status_code=400, detail=f"Offset for {name} must be a finite number"
            )

    saved = _saved_offsets()
    current = _live_offsets()
    limit = walk_offsets.TRIM_LIMIT_RAD

    updated = dict(current)
    clamped = []
    for name, value in req.offsets.items():
        target = value if req.mode == "absolute" else current[name] + value
        bounded = max(saved[name] - limit, min(saved[name] + limit, target))
        if abs(bounded - target) > 1e-9:
            clamped.append(name)
        updated[name] = round(bounded, 4)

    try:
        walk_offsets.write(updated)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not publish offsets: {e}")

    add_telemetry_props(
        mode=req.mode,
        joints_adjusted=len(req.offsets),
        clamped=len(clamped),
        max_trim=round(max(abs(updated[n] - saved[n]) for n in JOINTS), 4),
    )
    return _offsets_payload(success=True, clamped=sorted(clamped))


@app.post("/api/walk/offsets/reset")
def walk_offsets_reset():
    """Drop the live trim back to the saved values — the undo for a session
    of nudging that made things worse. Ramps back, same as any other change."""
    if not is_walking():
        raise HTTPException(status_code=409, detail="No walk is running")
    try:
        walk_offsets.write(_saved_offsets())
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not publish offsets: {e}")
    return _offsets_payload(success=True)


@app.post("/api/walk/offsets/save")
def walk_offsets_save():
    """Persist the live trim to duck_config.json (with a .bak), without
    interrupting the walk. Until this is called the trim lives only in shared
    memory and dies with the walk — this is the button that keeps the tuning."""
    offsets = _live_offsets()
    try:
        config = _read_config()
    except FileNotFoundError:
        config = DuckConfigModel().model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    config["joints_offsets"] = offsets
    try:
        _write_config(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # The walk owns the bus and its own copy of the offsets; the idle HWI (if
    # any) would keep serving the pre-save config, so drop it. Bus-only —
    # cutting torque here would drop a duck that's standing.
    if not is_walking():
        release_hwi(disable_torque=False)

    add_telemetry_props(joints_saved=len(offsets))
    return _offsets_payload(success=True)


# ── Remote Commands ──────────────────────────────────────────────────────────

# ── Voltage + temperature (idle-only servo vitals read) ───────────────────────

# Battery health bands for the duck's 2S pack.
VOLTAGE_LOW = 7.4
VOLTAGE_CRITICAL = 7.0

# Servo temperature bands (°C). The STS3215 firmware cuts torque at its
# over-temperature limit (70 by default) — warn well before that.
TEMP_WARM = 55
TEMP_HOT = 65


@app.get("/api/voltage")
def voltage():
    """Battery voltage + servo temperatures read via pypot (check_voltage.py's
    approach — the voltage register reads back decivolts). Both maps are keyed
    by joint name — the hottest servo is the story for heat, so readings carry
    their identity. Idle-only: refused while the walk or a rehoming session
    owns the bus."""
    refuse_while_walking()
    if rehome_io is not None:
        raise HTTPException(
            status_code=503,
            detail="Servo rehoming session in progress — finish it first",
        )
    # pypot needs the port; the next idle op lazily reopens HWI. Bus-only:
    # a battery poll must never drop a duck that is holding its stance.
    # Hold BUS_LOCK across the rustypot close AND the pypot session so a
    # concurrent /api/state cannot keep rustypot's fd open on this adapter.
    with BUS_LOCK:
        release_hwi(disable_torque=False)
        try:
            from pypot.feetech import FeetechSTS3215IO

            io = FeetechSTS3215IO(USB_PORT, baudrate=1000000)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Could not open servo bus: {e}")
        names = list(JOINTS.keys())
        try:
            raw = io.get_present_voltage(list(JOINTS.values()))
            per_motor = {n: round(float(v) * 0.1, 2) for n, v in zip(names, raw)}
            raw_temp = io.get_present_temperature(list(JOINTS.values()))
            temps = {n: int(t) for n, t in zip(names, raw_temp)}  # already °C, 1 byte
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Voltage read failed: {e}")
        finally:
            try:
                io.close()
            except Exception:
                pass
    volts = round(sum(per_motor.values()) / len(per_motor), 2) if per_motor else 0.0
    health_band = (
        "ok" if volts >= VOLTAGE_LOW else ("low" if volts >= VOLTAGE_CRITICAL else "critical")
    )
    hottest = max(temps, key=temps.get) if temps else None
    max_temp = temps[hottest] if hottest is not None else 0
    temp_band = "ok" if max_temp < TEMP_WARM else ("warm" if max_temp < TEMP_HOT else "hot")
    return {
        "volts": volts,
        "perMotor": per_motor,
        "health": health_band,
        "temps": temps,
        "maxTempC": max_temp,
        "hottest": hottest,
        "tempHealth": temp_band,
    }


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
