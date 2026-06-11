"""
Anonymous usage telemetry for the Open Duck Mini runtime.

Privacy contract (also documented in the README "Telemetry" section):
  - distinct_id is a random UUID generated on this device. No hostnames,
    usernames, IPs (GeoIP disabled), session tokens, or joint/motion data
    are ever sent.
  - Disable any time with TNKR_TELEMETRY=0 (env var) or by setting
    "enabled": false in ~/.tnkr-telemetry.json.

Every public function is fail-silent: telemetry must never crash or stall
the robot (same ethos as cloud_publisher.py). The posthog SDK delivers
events from its own background thread; we never block on network.

The PostHog key/host constants and the device property names (pi_model,
arch, ram_mb, os_release) must match scripts/setup.sh, which sends
setup-time events via curl before the venv exists.
"""

import atexit
import json
import os
import platform
import uuid
from pathlib import Path

# Write-only ingestion key (can send events, cannot read data).
# Must match POSTHOG_KEY / POSTHOG_HOST in scripts/setup.sh.
POSTHOG_API_KEY = "phc_FarYZWwIbyZFV2iUKyl8WyRRdFFuw2MH3NZat4zPmEK"
POSTHOG_HOST = "https://us.i.posthog.com"
TELEMETRY_FILE = Path.home() / ".tnkr-telemetry.json"
SOURCE = "openduck-runtime"

_client = None
_device_id: str | None = None
_device_props: dict | None = None
_sticky: dict = {}
_set_pending = True  # send $set person props on the next captured event


def _read_file() -> dict | None:
    try:
        with open(TELEMETRY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def is_enabled() -> bool:
    """TNKR_TELEMETRY env var (0/1 hard override) > file `enabled` > default True."""
    try:
        env = os.environ.get("TNKR_TELEMETRY")
        if env is not None:
            return env.strip().lower() not in ("0", "false", "off", "")
        cfg = _read_file()
        if cfg is not None:
            return bool(cfg.get("enabled", True))
        return True
    except Exception:
        return False


def device_id() -> str:
    """Read (or lazily create) the anonymous device id in ~/.tnkr-telemetry.json."""
    global _device_id
    if _device_id is not None:
        return _device_id
    try:
        cfg = _read_file()
        if cfg and cfg.get("device_id"):
            _device_id = str(cfg["device_id"])
            return _device_id

        _device_id = str(uuid.uuid4())
        # Lazy creation happens on robots that upgraded via git pull and never
        # ran the new setup.sh (which normally creates this file with a consent
        # notice). Give them the same notice, journalctl-visible.
        try:
            from datetime import datetime, timezone

            TELEMETRY_FILE.write_text(
                json.dumps(
                    {
                        "device_id": _device_id,
                        "enabled": True,
                        "notice_version": 1,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
            print(
                f"[telemetry] Anonymous usage telemetry enabled (device {_device_id[:8]}…). "
                f"Disable: TNKR_TELEMETRY=0 or {TELEMETRY_FILE}"
            )
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


def capture(event: str, properties: dict | None = None) -> None:
    """Send one event. No-op when disabled or when posthog is unavailable."""
    global _set_pending
    try:
        if not is_enabled():
            return
        client = _get_client()
        if client is None:
            return
        props = {
            **device_properties(),
            **_sticky,
            **(properties or {}),
            "source": SOURCE,
        }
        if _set_pending:
            props["$set"] = {**device_properties(), **_sticky}
            _set_pending = False
        # Keyword args only: survives the posthog-python 3.x -> 6.x signature
        # change (distinct_id became keyword-only).
        client.capture(distinct_id=device_id(), event=event, properties=props)
    except Exception:
        pass


def set_sticky(**props) -> None:
    """Attach props (e.g. servo_adapter_chip) to all later events and to the
    next event's $set person properties."""
    global _set_pending
    try:
        _sticky.update(props)
        _set_pending = True
    except Exception:
        pass


def flush(timeout: float = 3.0) -> None:
    try:
        if _client is not None:
            _client.flush()
    except Exception:
        pass


def shutdown() -> None:
    try:
        if _client is not None:
            _client.shutdown()
    except Exception:
        pass


def _reset_state_for_tests() -> None:
    """Test hook: clear module-level caches. Not for production use."""
    global _client, _device_id, _device_props, _sticky, _set_pending
    _client = None
    _device_id = None
    _device_props = None
    _sticky = {}
    _set_pending = True
