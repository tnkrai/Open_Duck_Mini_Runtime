"""
TNKR Robot Server

HTTP API server for the Open Duck Mini robot. Exposes motor check,
servo rehoming (firmware zero), stance calibration, config management,
and walk control endpoints.
Telemetry is streamed via Supabase Realtime broadcast channels.
"""

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

from mini_bdx_runtime.rustypot_position_hwi import HWI, find_servo_adapter
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime import telemetry
from mini_bdx_runtime import walk_telemetry
from mini_bdx_runtime import walk_pause
from mini_bdx_runtime import walk_offsets
from mini_bdx_runtime import preflight
from mini_bdx_runtime import policy_contract
from mini_bdx_runtime import policy_store
from mini_bdx_runtime import bench

# ── Constants ─────────────────────────────────────────────────────────────────

HOME_DIR = os.path.expanduser("~")
CONFIG_PATH = os.path.join(HOME_DIR, "duck_config.json")
SCRIPTS_DIR = Path(__file__).parent
SERVER_PORT = 8000

# Where installed policies live. A module global (not a constant baked into a store
# instance) so a test -- or a future config option -- can move it without this file
# needing to know.
POLICY_ROOT = Path(HOME_DIR) / ".tnkr" / "policies"

# How the robot fetches a policy. Swapped in tests; there is no network in the suite.
POLICY_FETCH = policy_store.stream_to_file


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
    # Which policy this launch loaded. Recorded so eviction can refuse to delete the
    # model a running walk has open, which the store cannot know by itself.
    policy_id: str | None = None
    # "free" or "bench" (story 4.3). The monitor thread reads it to decide whether this
    # launch's exit needs a bench report filed against the policy it ran.
    mode: str = bench.MODE_FREE


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
    """
    global hwi_instance, head_puppet_active
    if hwi_instance is not None:
        if disable_torque:
            try:
                hwi_instance.turn_off()
            except Exception:
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
    # The policy list is built to be polled (story 2.3), so it belongs here the day it
    # ships rather than after it has crowded out a month of setup_step_failed events the
    # way /api/state did. A 200 says "the policy UI is open", which nothing reads; a 500
    # says the robot stopped answering, which somebody should.
    "/api/policy",
    # Polled for the ten seconds a bench run lasts, then again while the operator is
    # looking at the verdict prompt. Same reasoning as /api/policy.
    "/api/bench",
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


# ── Capabilities ──────────────────────────────────────────────────────────────
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  CAPABILITY NAMES ARE A PERMANENT CONTRACT.                              │
# │                                                                          │
# │  A Pi updates only when its owner re-runs scripts/setup.sh, so the fleet  │
# │  runs mixed versions forever and an OLD Studio will be matching on these │
# │  exact strings for as long as any duck is switched on. Therefore:        │
# │                                                                          │
# │    * namespaced, dotted, lowercase   ("policy.install")                  │
# │    * ADD names, never rename or remove one                               │
# │    * a rename is a new name plus the old one kept forever                │
# │                                                                          │
# │  Tidying these up is not a refactor, it is a fleet-wide capability       │
# │  regression that no test in Studio's repo can see.                       │
# └──────────────────────────────────────────────────────────────────────────┘
#
# DERIVED, NOT LISTED. Each name maps to the routes that implement it, and a name is
# only claimed when every one of its routes is actually registered on this process's
# app. A hardcoded list would let a half-applied `git pull` -- setup.sh's update path is
# a git pull, and it can fail partway -- advertise a capability whose handler is missing,
# which is worse than not advertising it: Studio would show the UI and then 404.
CAPABILITY_ROUTES: dict[str, tuple[tuple[str, str], ...]] = {
    "preflight": (("POST", "/api/preflight"),),
    "policy.list": (("GET", "/api/policy"),),
    "policy.install": (("POST", "/api/policy/install"),),
    "policy.select": (("POST", "/api/policy/select"),),
    # The supervised bench run (story 4.3). One name for the whole flow -- start it,
    # stop it, read it, judge it -- because Studio has no use for half of it: without
    # the verdict endpoint the gate can never be cleared, and without the gate there is
    # nothing to clear.
    "bench": (
        ("GET", "/api/bench"),
        ("POST", "/api/bench/stop"),
        ("POST", "/api/bench/verdict"),
    ),
}

_capabilities_cache: list[str] | None = None


def capabilities_for(routes) -> list[str]:
    """The capability names satisfied by ``routes``. Pure, so every case is testable.

    Sorted for a stable payload — a set's iteration order would make /api/health's body
    differ between boots for no reason.
    """
    registered = set()
    for route in routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or ()
        if path is None:
            continue
        for method in methods:
            registered.add((method.upper(), path))
    return sorted(
        name
        for name, needed in CAPABILITY_ROUTES.items()
        if all(pair in registered for pair in needed)
    )


def capabilities() -> list[str]:
    """This robot's capability list, computed once.

    /api/health is polled by Studio, so this walks the route table on the first call and
    then never again. The route table cannot change after startup — FastAPI registers at
    import — so a cache is not a staleness risk.
    """
    global _capabilities_cache
    if _capabilities_cache is None:
        _capabilities_cache = capabilities_for(app.routes)
    return list(_capabilities_cache)


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
        # Absent on an older robot, which deserialises to [] in Studio — the correct
        # fallback for free, and the reason this is a list and not a version string.
        "capabilities": capabilities(),
    }


@app.get("/api/state")
def read_state():
    """Live joints (radians) + IMU orientation for the studio's viewer/recorder.

    At idle, read joints off the bus and orientation off the BNO055. While the
    walk owns the hardware, serve the walk loop's shared-memory snapshot (it
    reads every joint at 50 Hz for its policy anyway); fall back to the
    last-known pose if the snapshot is missing or stale — never fight for the
    port, never serve a dead snapshot as live."""
    global last_state_joints
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
            hwi = get_hwi()
            positions = hwi.get_present_positions()
            if positions is not None and len(positions) == len(hwi.joints):
                last_state_joints = {
                    n: round(float(p), 4)
                    for n, p in zip(hwi.joints.keys(), positions)
                }
        except Exception:
            pass  # transient read blip → serve the cached pose
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

    add_telemetry_props(joints_calibrated=len(calibration_offsets))
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

    if is_walking():
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
    power cut mid-write must never cost the calibration in duck_config.json."""
    if os.path.exists(CONFIG_PATH):
        shutil.copyfile(CONFIG_PATH, CONFIG_PATH + ".bak")
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


# ── Policies ──────────────────────────────────────────────────────────────────
#
# The store is the file half of this (mini_bdx_runtime/policy_store.py); these three
# endpoints are the HTTP half and hold no state of their own.
#
# NONE of them refuse while a walk is running, and that is deliberate rather than an
# oversight. They touch no servo bus, no I2C and no GPIO — install is a download plus a
# graph parse, select writes one pointer file — and amendment A4 requires reverting to the
# built-in to work exactly when a bad policy is running, because that is the moment the
# operator needs it. A selection takes effect on the next walk start.
#
# A NOTE ON WHAT MAY BE LOGGED HERE: the install URL is presigned, so its query string IS
# a credential. The telemetry middleware ships an endpoint's error detail to PostHog as
# `error_message`, so no failure text on this path may contain the URL. policy_store
# redacts it down to scheme/host/path; do not undo that, and do not add the URL to a
# telemetry prop.

_policy_lock = Lock()

# Which HTTP status a refusal gets. Studio renders the operator sentence off `code`, so
# these exist for HTTP's sake — but a failure must not be a 200 with ok:false, or a
# caller that only checks the status would treat a refused install as a success.
POLICY_STATUS_CODES: dict[str, int] = {
    policy_contract.POLICY_CONTRACT_MISMATCH: 422,  # the file is not a duck policy
    policy_contract.POLICY_STORE_FULL: 507,         # Insufficient Storage, literally
    policy_contract.POLICY_INSTALL_FAILED: 502,     # could not obtain what was promised
    # 409: the request is well-formed and the policy is fine -- the robot's state is what
    # refuses it. This policy has never been watched moving a real duck (story 4.3).
    bench.POLICY_BENCH_REQUIRED: 409,
}


def get_policy_store() -> policy_store.PolicyStore:
    """A store bound to the CURRENT module globals.

    Rebuilt per request on purpose: construction reads no files and creates no
    directories, and capturing SCRIPTS_DIR/POLICY_ROOT once at import would make the
    built-in's glob resolve against a path a test (or a future config reload) has since
    moved.
    """
    return policy_store.PolicyStore(
        root=POLICY_ROOT,
        scripts_dir=SCRIPTS_DIR,
        fetch=POLICY_FETCH,
    )


def _policy_failure(code: str, detail: str) -> JSONResponse:
    add_telemetry_props(policy_code=code)
    return JSONResponse(
        status_code=POLICY_STATUS_CODES.get(code, 400),
        content={"ok": False, "code": code, "detail": detail},
    )


def _running_policy_ids() -> set[str]:
    """The policy a live walk has open, so eviction leaves it alone."""
    session = walk_session
    if session is None or session.proc.poll() is not None:
        return set()
    return {session.policy_id} if session.policy_id else set()


class PolicyInstallRequest(BaseModel):
    id: str
    url: str
    sha256: str
    manifest: dict | None = None


class PolicySelectRequest(BaseModel):
    id: str


@app.get("/api/policy")
def list_policies():
    """Installed policies, which is active, and the built-in that always exists.

    Cheap enough to poll: one stat per policy and one small JSON read. It never hashes a
    file — that is install and select's job, and hashing a 16 MB model on every poll would
    be a self-inflicted load problem on a Pi Zero 2W.
    """
    return get_policy_store().list()


@app.post("/api/policy/install")
def install_policy(body: PolicyInstallRequest):
    """Fetch, verify and store one policy. Transactional: on any failure, nothing changes.

    The verification is `policy_contract.check_policy`, which is the security boundary
    (amendment A1) — this API authenticates nobody and its CORS reflects any origin, so
    the check lives here rather than in Studio, and there is no way for a caller to skip it.
    """
    store = get_policy_store()
    with _policy_lock:
        try:
            result = store.install(
                body.id,
                body.url,
                body.sha256,
                body.manifest,
                protect=_running_policy_ids(),
            )
        except policy_store.StoreError as exc:
            return _policy_failure(exc.code, exc.detail)

    stored = result.manifest or {}
    add_telemetry_props(
        ok=result.ok,
        already_installed=result.already_installed,
        evicted=result.evicted is not None,
        # Read off the manifest, not off `warning`: a warning also rides an install whose
        # latency could not be measured at all, and counting those as slow policies would
        # make the one number this feature exists to gather wrong.
        over_budget=bool(stored.get("latency_over_budget")),
        latency_measured=bool(stored.get("latency_measured")),
        latency_p99_ms=stored.get("latency_p99_ms"),
    )
    if not result.ok:
        return _policy_failure(result.code or policy_contract.POLICY_INSTALL_FAILED,
                               result.detail)
    return result.as_dict()


@app.post("/api/policy/select")
def select_policy(body: PolicySelectRequest):
    """Record which policy the next walk starts on.

    `id="builtin"` is the one-action revert (amendment A4) and always succeeds: it removes
    the pointer file rather than writing one, so it works with an empty store, a corrupt
    active pointer, and a walk in progress.
    """
    store = get_policy_store()
    # DELIBERATELY OUTSIDE _policy_lock. An install holds that lock for as long as the
    # download takes -- up to a minute on household wifi -- and amendment A4 says the way
    # back to the built-in must work at the moment a bad policy is running, which is
    # exactly when someone might also be installing the next one. Selecting is one atomic
    # pointer write (or one unlink), so it needs no mutual exclusion of its own.
    #
    # The race it admits: an install's eviction reads the active id, then a select lands
    # on a policy that eviction is about to delete. The result is an active pointer naming
    # a directory that no longer exists, which resolves to the built-in and logs -- the
    # same fallback a reflashed card produces, and tested as such.
    try:
        active = store.select(body.id)
    except policy_store.StoreError as exc:
        return _policy_failure(exc.code, exc.detail)
    add_telemetry_props(builtin=active == policy_store.BUILTIN_ID)
    return {"ok": True, "active": active}


# ── The supervised bench run (story 4.3, Decision 10) ─────────────────────────
#
# Decision 1 removed the simulation gate and recorded the consequence: nothing in this
# design predicts whether a policy WALKS WELL. So the first hardware run of a policy this
# robot has never run is ten bounded seconds with the operator holding the duck, and the
# operator's answer is the only judge there is.
#
# These endpoints record a verdict. They never compute one. There is deliberately no
# heuristic anywhere below that turns "did not abort" into a pass -- a policy can hold
# every joint inside every limit, keep every deadline, and produce a gait that faceplants
# the moment it bears weight, which is exactly the case a rubber stamp would wave through.
#
# The one exception runs the other way: an abort from the safety envelope FAILS the bench
# without asking, because the envelope has already answered the question.


class BenchVerdictRequest(BaseModel):
    policyId: str
    passed: bool
    reason: str | None = None


def _bench_running() -> bool:
    session = walk_session
    return (
        session is not None
        and session.mode == bench.MODE_BENCH
        and session.proc.poll() is None
    )


def _bench_payload() -> dict:
    """What /api/bench answers, and what /api/bench/stop echoes back.

    ``policyId`` is the policy of the bench in progress, or of the last one that reported
    -- never of the free walk that happens to be running, which is not a bench and whose
    id would read as one on the screen the operator is watching.
    """
    report = bench.read_report()
    running = _bench_running()
    session = walk_session
    return {
        "running": running,
        "policyId": (
            session.policy_id
            if running and session is not None
            else (report.policy_id if report is not None else None)
        ),
        "stopRequested": bench.stop_requested(),
        "defaultSeconds": bench.DEFAULT_BENCH_SECONDS,
        "report": report.as_dict() if report is not None else None,
    }


def _finalize_bench(session: WalkSession) -> None:
    """After a bench walk exits: fail it here if a guard tripped. Never pass it here.

    Called from the monitor thread, so it must not raise -- and it is the only place a
    verdict is written without a person, which is why it can only ever write a *failure*.

    Everything else -- a clean ten seconds, an operator stop, a crash, a power cut --
    leaves no record at all, and no record means still gated. That is the fail-closed
    direction: the cost of forgetting a pass is another ten seconds, and the cost of
    inventing one is an unwatched policy on a duck standing on its own feet.
    """
    if session.mode != bench.MODE_BENCH or not session.policy_id:
        return
    report = bench.read_report()
    if report is None or not report.aborted:
        return
    reason = report.abort_reason or "the safety envelope stopped the run"
    try:
        get_policy_store().mark_bench(session.policy_id, False, reason)
    except Exception as exc:  # a bookkeeping failure must not kill the monitor thread
        print(f"[bench] could not record the failed bench run: {exc}")


@app.get("/api/bench")
def read_bench():
    """Whether a bench run is in progress, and the last one's report.

    Polled while the run is on screen, so it is on the failures-only telemetry list.
    """
    return _bench_payload()


@app.post("/api/bench/stop")
def stop_bench():
    """End the bench run now.

    A file the loop stats each tick, not a signal: the walk then ends its own bench, cuts
    torque through the teardown it already has, and writes the report saying the operator
    stopped it. Killing the process would cut torque just as fast and lose the report,
    which is the difference between a run the operator can fail on purpose and a run that
    merely vanished.

    /api/walk/stop still works during a bench and still cuts torque immediately. This is
    the gentler of the two, not a replacement for it.
    """
    if not _bench_running():
        raise HTTPException(
            status_code=409, detail="No bench run is in progress -- nothing to stop"
        )
    bench.request_stop()
    return _bench_payload()


@app.post("/api/bench/verdict")
def bench_verdict(body: BenchVerdictRequest):
    """Record the operator's answer to "did that look like walking?".

    Refusals, and why each one is a refusal rather than a shrug:

    * no report -- nothing ran, or the run died before it could say what it did. Accepting
      a pass here would let the gate be cleared without a duck ever moving.
    * a report for a different policy -- the verdict would be filed against weights nobody
      watched.
    * a pass on a run that aborted, crashed or was signalled away -- the envelope already
      failed it, or nobody saw it finish. A *fail* on any of those is always accepted.
    """
    store = get_policy_store()
    report = bench.read_report()
    if report is None:
        raise HTTPException(
            status_code=409,
            detail="No bench run to judge -- run one before recording a verdict",
        )
    if (report.policy_id or policy_store.BUILTIN_ID) != body.policyId:
        raise HTTPException(
            status_code=409,
            detail=(
                f"the last bench run was of {report.policy_id!r}, "
                f"not {body.policyId!r}"
            ),
        )
    if body.passed and report.ended not in bench.PASSABLE_ENDINGS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"that bench run ended as {report.ended!r}, which cannot be passed. "
                "Run the bench again."
            ),
        )
    try:
        status = store.mark_bench(body.policyId, body.passed, body.reason or "")
    except policy_store.StoreError as exc:
        return _policy_failure(exc.code, exc.detail)
    add_telemetry_props(
        passed=body.passed,
        ended=report.ended,
        # A boolean, never the id: which community policy an owner is trying is not this
        # event's business (same rule as the install event).
        builtin=body.policyId == policy_store.BUILTIN_ID,
        ticks=report.ticks,
        clamp_events=report.clamp_events,
    )
    return {"ok": True, "bench": status}


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
    # Before the telemetry, because this is the only chance to see the report the walk
    # just wrote: a bench that aborted is a bench that failed, and story 1.3's guards are
    # the one judge that does not need a person (story 4.3).
    try:
        _finalize_bench(session)
    except Exception as exc:
        print(f"[bench] finalize failed: {exc}")
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
            "mode": session.mode,
        },
    )


class WalkStartRequest(BaseModel):
    sessionToken: str | None = None
    supabaseUrl: str | None = None
    supabaseKey: str | None = None
    # Which policy to walk on. Omitted (the only thing Studio sends today) means the
    # active one, which on a duck that has never installed anything is the built-in.
    policyId: str | None = None
    # "free" (the default, and everything before story 4.3) or "bench": a bounded
    # supervised run of a policy this robot has never run. Omitted means free, so a Studio
    # that predates the bench sends exactly what it sent before.
    mode: str | None = None
    benchSeconds: float | None = None


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

    try:
        mode = bench.parse_mode(body.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if rehome_io is not None:
        raise HTTPException(
            status_code=409,
            detail="Servo rehoming session in progress — finish it before walking",
        )

    running = walk_session is not None and walk_session.proc.poll() is None
    if running and body.sessionToken and body.sessionToken == walk_session.session_token:
        # An idempotent retry of the walk that is already running. It starts nothing, so
        # it resolves nothing: a stale policyId must not turn a no-op into a 404.
        add_telemetry_props(already_running=True)
        return {"success": True, "message": "Walk is already running"}

    venv_python = sys.executable
    is_pi = platform.machine() in ("aarch64", "armv7l")

    # Resolve the policy BEFORE stopping the running walk and before releasing the buses.
    # A request naming a policy this robot does not have is a 404 that should cost
    # nothing, and both of those cost torque: stopping first cuts it on a duck mid-stride
    # for a request that never starts a walk, and Studio sending an id this robot no
    # longer has is routine, because a bounded store evicts policies its cached list
    # still shows.
    resolved = None
    if is_pi:
        store = get_policy_store()
        try:
            resolved = (
                store.resolve(body.policyId)
                if body.policyId
                else store.resolve_active()
            )
        except policy_store.StoreError as exc:
            raise HTTPException(status_code=404, detail=exc.detail)
        if resolved is None:
            # Same message this endpoint has always given when scripts/ holds no .onnx.
            raise HTTPException(
                status_code=404,
                detail="No ONNX model found in scripts/ directory",
            )

        # The gate (story 4.3). Checked here, in the same window as the 404 above and for
        # the same reason: a refusal must cost nothing, and stopping the running walk
        # first would cut torque on a duck mid-stride for a request that starts nothing.
        #
        # The built-in is exempt -- it is what every duck sold walks on. A bench run is
        # never gated on itself, which is the whole point of it.
        #
        # DELIBERATE HOLE, worth stating: a file an owner copies straight into scripts/ is
        # resolved by the built-in's glob, so it gets id "builtin" here and is NOT gated.
        # Gating it would be unclearable -- a verdict is recorded against a store id, and
        # that file has none -- so the gate would permanently break the workflow the
        # envelope stories describe as how people run downloaded policies today. It is
        # still guarded: envelope.is_armed goes by FILE NAME, not by this id, so anything
        # that is not one of the two policies this repo ships runs with every clamp and
        # abort armed. Fail-safe there, functional here.
        if mode == bench.MODE_FREE:
            status = store.bench_status(resolved.id)
            if status["required"]:
                return _policy_failure(
                    bench.POLICY_BENCH_REQUIRED,
                    f"{resolved.id} has not passed a supervised bench run on this robot"
                    + (f": {status['reason']}" if status["reason"] else ""),
                )

    if running:
        # A different session token: the old walk is bound to a channel nobody is
        # listening to any more. Only now, with a policy in hand and a walk certain to
        # start, is it worth stopping.
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

    if is_pi:
        assert resolved is not None  # resolved above, before the buses were released
        walk_script = str(SCRIPTS_DIR / "v2_rl_walk_mujoco.py")

        cmd = [
            venv_python,
            walk_script,
            "--onnx_model_path", str(resolved.path),
            "--remote",
            "--commands",
        ]
        if not resolved.is_builtin:
            # Explicitness, NOT the arming mechanism. envelope.is_armed() decides from the
            # ARTIFACT — `not is_builtin_policy(onnx_model_path)` — precisely because
            # nothing used to pass this flag, so a downloaded policy dropped in scripts/
            # and started from here ran with every guard disarmed and nothing logged. That
            # was fail-open. If this line is ever deleted the envelope must still arm; if
            # it stops arming, the bug is in is_armed, not here.
            cmd.append("--custom_policy")
        if mode == bench.MODE_BENCH:
            # Bench mode adds three arguments and no second code path: --mode is read by
            # the same loop, and --policy_id is carried only so the report the walk writes
            # names the policy the operator's verdict will be filed against.
            cmd.extend(
                [
                    "--mode",
                    bench.MODE_BENCH,
                    "--bench_seconds",
                    f"{bench.clamp_seconds(body.benchSeconds):g}",
                    "--policy_id",
                    resolved.id,
                ]
            )
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
        # Whether this walk is the policy the robot shipped with. A boolean, never the id:
        # which community policy an owner is trying is not this event's business.
        builtin_policy=bool(resolved.is_builtin) if resolved else None,
        mode=mode,
    )

    # Start from a clean slate. stop_walk_process() clears these, but a walk
    # that died hard (SIGKILL, power cut) never ran its own cleanup — and
    # is_walking() goes true the moment we spawn, seconds before the walk
    # publishes its own state, so a leftover file would read as *this* walk's
    # pause/trim for the whole boot window.
    walk_pause.clear()
    walk_offsets.clear()
    # Same rule, for the same reason: a stop request belongs to the bench that was
    # stopped, and a leftover one would end the next bench on its first tick. Cleared on
    # every start, because it is a REQUEST -- it expires with the run it was aimed at, and
    # nothing is lost by dropping one nobody is waiting on.
    bench.clear_stop()
    # The report is the opposite: an ANSWER, and one the operator still owes a verdict on.
    # POST /api/bench/verdict refuses with 409 when there is no report, so deleting it
    # here costs somebody another ten seconds holding a duck. Clearing it on every start
    # (which is what this did) threw that away whenever anyone pressed Walk in between --
    # a bench finishes, the "did that look like walking?" prompt is on screen, the
    # operator starts the built-in to compare the two, and the answer can no longer be
    # recorded. Nothing unsafe: the policy stays gated, which is the fail-closed
    # direction. It is ten seconds of the operator's time, lost for no reason.
    #
    # So only a bench start clears it, which is the case the clear exists for: /api/bench
    # must not serve a previous run's outcome as THIS run's while the walk is still
    # booting. A free walk cannot be mistaken for a bench -- `running` comes off the
    # session's mode, never off the report lying beside it -- and the loop writes a report
    # only when self.bench is not None (v2_rl_walk_mujoco.py, mode == bench).
    #
    # The walk clears both again itself when it starts a bench, which is what keeps this
    # correct when a bench is started from a terminal with no server involved.
    if mode == bench.MODE_BENCH:
        bench.clear_report()

    proc = subprocess.Popen(cmd, cwd=str(SCRIPTS_DIR))
    walk_session = WalkSession(
        proc=proc,
        session_token=body.sessionToken,
        cloud_streaming=cloud_streaming,
        started_at=time.monotonic(),
        policy_id=resolved.id if resolved else None,
        mode=mode,
    )
    Thread(target=_monitor_walk, args=(walk_session,), daemon=True).start()
    if resolved is not None:
        # LRU stamp: the policy you are walking on is the last one you would want evicted.
        get_policy_store().mark_used(resolved.id)

    return {
        "success": True,
        "pid": proc.pid,
        "policyId": resolved.id if resolved else None,
        "mode": mode,
    }


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


# ── Preflight ─────────────────────────────────────────────────────────────────
#
# Implements the open TODO at the bottom of checklist.md ("Make a script that goes
# through all this automatically"). It reports; it never decides whether a walk may
# start. That call belongs to whoever is starting the walk.

last_preflight: dict | None = None  # last result, so a client can re-read without re-running


def get_feet_contacts():
    """Lazy foot-switch reader, or None when this machine has no GPIO.

    Not a singleton: FeetContacts holds two DigitalInOut pins and the walk subprocess
    constructs its own, so preflight opens and closes its own short-lived reader rather
    than holding pins a walk will want.
    """
    try:
        from mini_bdx_runtime.feet_contacts import FeetContacts

        return FeetContacts()
    except Exception as e:
        print(f"[preflight] foot switches unavailable: {e}")
        return None


@app.post("/api/preflight")
def run_preflight_endpoint():
    """Check joints, calibration offsets, IMU orientation and foot switches."""
    global last_preflight

    # The walk subprocess owns both the serial bus and the I2C while it runs.
    refuse_while_walking()

    try:
        hwi = get_hwi()
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Cannot connect to motor controller: {e}"
        )

    try:
        imu = get_state_imu()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot open the IMU: {e}")

    feet = get_feet_contacts()
    try:
        report = preflight.run_preflight(hwi, imu, feet, hwi.duck_config)
    finally:
        # Release the pins immediately — a walk started right after preflight needs them.
        if feet is not None:
            try:
                feet.stop()
            except Exception:
                pass

    last_preflight = report.as_dict()
    telemetry.capture(
        "preflight_run",
        {
            "ok": report.ok,
            # Which checks failed, never the values they read.
            "failed": [c.name for c in report.checks if not c.ok],
            "duration_ms": report.duration_ms,
        },
    )
    return last_preflight


@app.get("/api/preflight")
def read_preflight():
    """The last preflight result, or nulls if none has run this boot."""
    if last_preflight is None:
        return {"ok": None, "checks": [], "duration_ms": 0}
    return last_preflight


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    telemetry.capture("server_started", {"server_port": SERVER_PORT})
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
