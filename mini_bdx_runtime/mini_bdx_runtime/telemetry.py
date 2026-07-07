"""
Anonymous usage telemetry for the Open Duck Mini runtime.

Privacy contract (also documented in the README "Telemetry" section):
  - distinct_id is a random UUID generated on this device. No hostnames,
    usernames, IPs (GeoIP disabled), session tokens, or joint/motion data
    are ever sent. String properties are scrubbed of home-directory paths
    and the local username before sending.
  - Disable any time with TNKR_TELEMETRY=0 (env var) or by setting
    "enabled": false in ~/.tnkr-telemetry.json.
  - A consent file that exists but cannot be parsed counts as OPTED OUT:
    we never re-enroll a user because their SD card corrupted the file.

Every public function is fail-silent: telemetry must never crash or stall
the robot (same ethos as cloud_publisher.py). The posthog SDK delivers
events from its own background thread; we never block on network, and
shutdown is time-bounded so a powered-off network can't hang systemd stops.

Events are rate-capped client-side (RATE_LIMIT_PER_MIN) so a misbehaving
LAN client hammering the API can't burn the PostHog quota.

The PostHog key/host constants and the device property names (pi_model,
arch, ram_mb, os_release) must match scripts/setup.sh, which sends
setup-time events via curl before the venv exists.
"""

import atexit
import json
import os
import platform
import re
import time
import uuid
from pathlib import Path
from threading import Lock, Thread

# Write-only ingestion key (can send events, cannot read data).
# Must match POSTHOG_KEY / POSTHOG_HOST in scripts/setup.sh.
POSTHOG_API_KEY = "phc_FarYZWwIbyZFV2iUKyl8WyRRdFFuw2MH3NZat4zPmEK"
POSTHOG_HOST = "https://us.i.posthog.com"
TELEMETRY_FILE = Path.home() / ".tnkr-telemetry.json"
SOURCE = "openduck-runtime"
RATE_LIMIT_PER_MIN = 60
ENABLED_CACHE_TTL_S = 60.0

_MISSING = object()  # consent file does not exist (distinct from "corrupt")

_client = None
_device_id: str | None = None
_device_props: dict | None = None
_sticky: dict = {}
_set_pending = True  # send $set person props on the next captured event
_enabled_cache: tuple[float, bool] | None = None  # (expires_at, value)
_rate_window_start = 0.0
_rate_count = 0
_rate_dropped = 0
# capture() runs concurrently on the event loop, walk-monitor threads, and
# the IMU worker — the rate counters, $set flag, and device-id lazy init are
# read-modify-write and need this lock.
_state_lock = Lock()

_HOME_STR = str(Path.home())
try:
    import getpass

    _USERNAME = getpass.getuser()
except Exception:
    _USERNAME = None


def _scrub(value):
    """Remove home-directory paths and the local username from outgoing
    properties — error messages and log tails routinely embed
    /home/<username>/... paths, which would break the privacy contract."""
    if isinstance(value, str):
        v = value
        # Guard against degenerate HOME (e.g. "/" under some init contexts),
        # which would rewrite every path separator.
        if len(_HOME_STR) > 5 and _HOME_STR.startswith(("/home/", "/root", "/Users/")):
            v = v.replace(_HOME_STR, "~")
        v = re.sub(r"/home/[A-Za-z0-9._-]+", "/home/<user>", v)
        v = re.sub(r"/Users/[A-Za-z0-9._-]+", "/Users/<user>", v)
        if _USERNAME and len(_USERNAME) >= 4:
            v = v.replace(_USERNAME, "<user>")
        return v
    if isinstance(value, list):
        return [_scrub(x) for x in value]
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


def _read_file():
    """Parsed consent file dict, _MISSING if absent, None if corrupt."""
    try:
        if not TELEMETRY_FILE.exists():
            return _MISSING
        return json.loads(TELEMETRY_FILE.read_text())
    except Exception:
        return None


def is_enabled() -> bool:
    """TNKR_TELEMETRY env var (0/1 hard override) > file `enabled` > default True.

    The file decision is cached for ENABLED_CACHE_TTL_S so per-event capture
    never does file I/O on the server's event loop (SD cards stall).
    """
    global _enabled_cache
    try:
        env = os.environ.get("TNKR_TELEMETRY")
        if env is not None:
            return env.strip().lower() not in ("0", "false", "off", "")
        now = time.monotonic()
        if _enabled_cache is not None and now < _enabled_cache[0]:
            return _enabled_cache[1]
        cfg = _read_file()
        if cfg is _MISSING:
            value = True
        elif cfg is None:
            value = False  # exists but unreadable: never assume consent
        else:
            value = bool(cfg.get("enabled", True))
        _enabled_cache = (now + ENABLED_CACHE_TTL_S, value)
        return value
    except Exception:
        return False


def device_id() -> str:
    """Read (or lazily create) the anonymous device id in ~/.tnkr-telemetry.json."""
    global _device_id
    if _device_id is not None:
        return _device_id
    try:
        with _state_lock:
            if _device_id is not None:
                return _device_id
            cfg = _read_file()
            if isinstance(cfg, dict) and cfg.get("device_id"):
                _device_id = str(cfg["device_id"])
                return _device_id

            _device_id = str(uuid.uuid4())
            if cfg is None:
                # File exists but is corrupt: never overwrite it — it may
                # hold an opt-out we can no longer read (is_enabled treats
                # this as disabled anyway). Keep a per-process id only.
                return _device_id

            # MERGE into any existing parseable file — a hand-written
            # {"enabled": false} opt-out must keep its `enabled` value; we
            # only ever add the missing fields.
            existing = cfg if isinstance(cfg, dict) else {}
            from datetime import datetime, timezone

            merged = {
                "enabled": True,
                "notice_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **existing,
                "device_id": _device_id,
            }
            if merged.get("enabled", True):
                # Lazy creation happens on robots that upgraded via git pull
                # and never ran the new setup.sh. Same notice as setup.sh,
                # journalctl-visible, printed BEFORE the write so it appears
                # even if the filesystem is read-only.
                print(
                    f"[telemetry] Anonymous usage telemetry enabled (device {_device_id[:8]}…). "
                    f"Disable: TNKR_TELEMETRY=0 or {TELEMETRY_FILE}"
                )
            try:
                # Atomic write: these robots lose power mid-write all the
                # time, and a torn consent file must never appear.
                tmp = TELEMETRY_FILE.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(merged, indent=2))
                os.replace(tmp, TELEMETRY_FILE)
            except Exception:
                pass  # unwritable HOME: keep the per-process id, stay silent
            return _device_id
    except Exception:
        _device_id = _device_id or str(uuid.uuid4())
        return _device_id


def device_properties() -> dict:
    """Hardware/runtime specs attached to every event (cached). Names are a
    contract shared with scripts/setup.sh — keep them identical."""
    global _device_props
    if _device_props is not None:
        return _device_props
    props: dict = {}
    try:
        props["arch"] = platform.machine()
        props["python_version"] = platform.python_version()
    except Exception:
        pass
    try:
        model = Path("/proc/device-tree/model").read_text()
        props["pi_model"] = model.replace("\x00", "").strip()
    except Exception:
        props["pi_model"] = None
    try:
        os_release = None
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                os_release = line.split("=", 1)[1].strip().strip('"')
                break
        props["os_release"] = os_release or platform.platform()
    except Exception:
        try:
            props["os_release"] = platform.platform()
        except Exception:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                props["ram_mb"] = int(line.split()[1]) // 1024
                break
    except Exception:
        props["ram_mb"] = None
    try:
        from importlib.metadata import version

        props["runtime_version"] = version("mini-bdx-runtime")
    except Exception:
        props["runtime_version"] = "unknown"
    _device_props = props
    return props


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from posthog import Posthog

        _client = Posthog(POSTHOG_API_KEY, host=POSTHOG_HOST, disable_geoip=True)
        atexit.register(shutdown)
        return _client
    except Exception:
        return None


def _rate_limit_check() -> tuple[bool, int]:
    """Fixed window: (allowed, dropped_in_previous_window). Caller must only
    apply this to flood-prone api_request_* events — low-frequency lifecycle
    events (walk_ended, imu_calibration_*, server_started) bypass the limiter
    so a request flood can never starve the crash-rate data."""
    global _rate_window_start, _rate_count, _rate_dropped
    with _state_lock:
        now = time.monotonic()
        dropped = 0
        if now - _rate_window_start >= 60.0:
            _rate_window_start = now
            _rate_count = 0
            dropped = _rate_dropped
            _rate_dropped = 0
        if _rate_count >= RATE_LIMIT_PER_MIN:
            _rate_dropped += 1
            return False, dropped
        _rate_count += 1
        return True, dropped


def capture(event: str, properties: dict | None = None) -> None:
    """Send one event. No-op when disabled, rate-capped, or posthog missing."""
    global _set_pending
    try:
        if not is_enabled():
            return
        dropped = 0
        if event.startswith("api_request"):
            allowed, dropped = _rate_limit_check()
            if not allowed:
                return
        client = _get_client()
        if client is None:
            return
        sticky = dict(_sticky)  # snapshot: set_sticky may run on another thread
        props = {
            **device_properties(),
            **sticky,
            **(properties or {}),
            "source": SOURCE,
        }
        if dropped:
            props["events_dropped"] = dropped
        with _state_lock:
            send_set = _set_pending
            _set_pending = False
        if send_set:
            props["$set"] = {**device_properties(), **sticky}
        props = _scrub(props)
        # Keyword args only: survives the posthog-python 3.x -> 6.x signature
        # change (distinct_id became keyword-only).
        client.capture(distinct_id=device_id(), event=event, properties=props)
    except Exception:
        pass


def set_sticky(**props) -> None:
    """Attach props (e.g. servo_adapter_chip) to all later events and to the
    next event's $set person properties. None values are skipped so an
    unknown chip never clobbers a previously-reported one."""
    global _set_pending
    try:
        cleaned = {k: v for k, v in props.items() if v is not None}
        if cleaned:
            _sticky.update(cleaned)
            _set_pending = True
    except Exception:
        pass


def _call_bounded(fn, timeout: float) -> None:
    """Run fn in a daemon thread, waiting at most `timeout` seconds — an
    offline robot must never hang a systemd stop on network retries."""
    t = Thread(target=fn, daemon=True)
    t.start()
    t.join(timeout)


def flush(timeout: float = 3.0) -> None:
    try:
        if _client is not None:
            _call_bounded(_client.flush, timeout)
    except Exception:
        pass


def shutdown(timeout: float = 3.0) -> None:
    try:
        if _client is not None:
            _call_bounded(_client.shutdown, timeout)
    except Exception:
        pass


def _reset_state_for_tests() -> None:
    """Test hook: clear module-level caches. Not for production use."""
    global _client, _device_id, _device_props, _sticky, _set_pending
    global _enabled_cache, _rate_window_start, _rate_count, _rate_dropped
    _client = None
    _device_id = None
    _device_props = None
    _sticky = {}
    _set_pending = True
    _enabled_cache = None
    _rate_window_start = 0.0
    _rate_count = 0
    _rate_dropped = 0
