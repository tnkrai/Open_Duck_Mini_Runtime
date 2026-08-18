"""GET /api/telemetry/identity: reports the device id, never provisions one.

The endpoint exists so Studio can learn which robot it just connected to, and
hand that id to tnkr-core, which decides ownership behind a verified token. Two
properties are load-bearing and every test here defends one of them:

  1. It NEVER returns user data. This server has no auth on any endpoint and
     PrivateNetworkMiddleware grants browser access, so the response should be
     assumed readable by any page the operator visits.
  2. It NEVER creates state. Reading must not enrol a robot that never ran
     setup.sh, and an opted-out robot must not be claimable.
"""

import json

from mini_bdx_runtime import telemetry


def _write_consent(payload):
    telemetry.TELEMETRY_FILE.write_text(json.dumps(payload))
    telemetry._reset_state_for_tests()


# ── enabled ──────────────────────────────────────────────────────────────────

def test_reports_device_id_when_enabled(client):
    _write_consent({"device_id": "abc-123", "enabled": True})

    body = client.get("/api/telemetry/identity").json()

    assert body == {"enabled": True, "deviceId": "abc-123"}


def test_enabled_but_no_id_yet_reports_enabled_without_an_id(client):
    """Fresh robot, or an upgrade-path robot that has never captured an event.

    Enabled is true, but there is nothing to report and we must not invent one.
    """
    _write_consent({"enabled": True})

    body = client.get("/api/telemetry/identity").json()

    assert body == {"enabled": True}
    assert "deviceId" not in body


# ── disabled: every route to "off" withholds the id ──────────────────────────

def test_env_override_disables_and_withholds_the_id(client, monkeypatch):
    _write_consent({"device_id": "abc-123", "enabled": True})
    monkeypatch.setenv("TNKR_TELEMETRY", "0")
    telemetry._reset_state_for_tests()

    body = client.get("/api/telemetry/identity").json()

    assert body == {"enabled": False}
    assert "deviceId" not in body


def test_file_opt_out_disables_and_withholds_the_id(client):
    _write_consent({"device_id": "abc-123", "enabled": False})

    body = client.get("/api/telemetry/identity").json()

    assert body == {"enabled": False}
    assert "deviceId" not in body


def test_corrupt_consent_file_counts_as_opted_out(client):
    """A corrupt file may hold an opt-out we can no longer read.

    telemetry.is_enabled() already treats unreadable as disabled; the endpoint
    must not quietly disagree with it and hand out an id anyway.
    """
    telemetry.TELEMETRY_FILE.write_text("{not json at all")
    telemetry._reset_state_for_tests()

    body = client.get("/api/telemetry/identity").json()

    assert body == {"enabled": False}
    assert "deviceId" not in body


# ── never provisions ─────────────────────────────────────────────────────────

def test_reading_identity_never_creates_the_consent_file(client):
    """Reporting is not provisioning.

    device_id() lazily mints and writes when the file is missing. Doing that
    here would mean a laptop on the network could enrol a robot that never ran
    setup.sh, which is a consent decision the robot's owner never made.
    """
    assert not telemetry.TELEMETRY_FILE.exists()

    body = client.get("/api/telemetry/identity").json()

    assert body == {"enabled": True}
    assert "deviceId" not in body
    assert not telemetry.TELEMETRY_FILE.exists()


def test_repeated_reads_never_mint_an_id(client):
    for _ in range(5):
        client.get("/api/telemetry/identity")

    assert not telemetry.TELEMETRY_FILE.exists()


def test_snapshot_does_not_clobber_an_existing_opt_out(client):
    _write_consent({"enabled": False})

    client.get("/api/telemetry/identity")

    assert json.loads(telemetry.TELEMETRY_FILE.read_text()) == {"enabled": False}


# ── never leaks identity ─────────────────────────────────────────────────────

FORBIDDEN_KEYS = (
    "user_id", "userId", "owner", "owner_user_id", "ownerUserId",
    "aliased_to", "aliasedTo", "email", "account", "claim", "claimed",
)


def test_response_carries_no_user_or_owner_data(client):
    """The regression guard for the design decision this endpoint exists under.

    An earlier draft returned `aliased_to`, i.e. a Supabase user id, from an
    unauthenticated endpoint reachable by any webpage. If someone adds an owner
    field back here, this fails.
    """
    _write_consent({"device_id": "abc-123", "enabled": True, "aliased_to": "user_9f3b"})

    raw = client.get("/api/telemetry/identity").text
    body = json.loads(raw)

    assert set(body) <= {"enabled", "deviceId"}
    for key in FORBIDDEN_KEYS:
        assert key not in body
        assert key not in raw
    assert "user_9f3b" not in raw


def test_no_write_counterpart_exists(client):
    """Ownership is core's decision. There must be no way to set it here."""
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)("/api/telemetry/owner")
        assert response.status_code in (404, 405)


# ── stays out of its own event stream ────────────────────────────────────────

def test_identity_reads_emit_no_telemetry_events(client, captured):
    """Studio calls this on every connect. It must not become a metric itself."""
    _write_consent({"device_id": "abc-123", "enabled": True})

    for _ in range(10):
        client.get("/api/telemetry/identity")

    endpoints = [e.get("properties", {}).get("endpoint") for e in captured]
    assert "/api/telemetry/identity" not in endpoints
