"""What the robot says it can do, and why the list is derived rather than written down.

Amendment A2. A Pi updates only when its owner re-runs ``scripts/setup.sh``, which git-pulls
``v2``, so the fleet runs mixed versions permanently and Studio has to be able to ask. It
asks for capabilities rather than a version because its real question is "can you do this",
and a capability list stays correct for a fork and for a Pi whose update applied halfway.

Two properties are load-bearing and both are tested here:

* **Derived from the route table.** A hardcoded list lets a half-applied ``git pull``
  advertise a capability whose handler is missing, which is worse than silence: Studio would
  show the UI and then 404.
* **The names are permanent.** An old Studio matches on the exact strings for as long as any
  duck is switched on, so a rename is a fleet-wide regression that no test in Studio's repo
  can see. ``test_the_capability_names_are_frozen`` is that test, on this side.
"""

from __future__ import annotations

import pytest

import tnkr_server
from tnkr_server import CAPABILITY_ROUTES, capabilities, capabilities_for


class FakeRoute:
    def __init__(self, path, methods):
        self.path = path
        self.methods = set(methods)


# Every name this repo has ever shipped. ADD to this list; never edit or remove an entry.
# A name that leaves this list has been broken for every Studio already installed.
SHIPPED_CAPABILITIES = frozenset(
    {"preflight", "policy.list", "policy.install", "policy.select", "bench"}
)


def test_health_reports_capabilities(client):
    body = client.get("/api/health").json()
    assert set(body["capabilities"]) >= {"policy.install", "policy.select", "preflight"}


def test_health_keeps_every_field_an_older_studio_reads(client):
    """The capability list is additive. A duck that gains it must not lose anything."""
    body = client.get("/api/health").json()
    assert set(body) >= {
        "status",
        "is_pi",
        "platform",
        "walking",
        "paused",
        "walkExitCode",
        "capabilities",
    }
    assert body["status"] == "ok"


def test_every_declared_capability_is_actually_registered():
    """The drift guard. Renaming an endpoint without updating CAPABILITY_ROUTES would
    silently drop a capability, and the symptom would appear in Studio, one repo away."""
    assert capabilities() == sorted(CAPABILITY_ROUTES)


def test_the_capability_names_are_frozen():
    assert set(CAPABILITY_ROUTES) == SHIPPED_CAPABILITIES, (
        "a capability name changed. Old Studio installs match on the exact string, so a "
        "rename means adding the new name AND keeping the old one forever."
    )


@pytest.mark.parametrize("name", sorted(SHIPPED_CAPABILITIES))
def test_names_are_namespaced_dotted_and_lowercase(name):
    assert name == name.lower()
    assert " " not in name and "_" not in name and "-" not in name
    assert not name.startswith(".") and not name.endswith(".")


def test_a_capability_whose_handler_is_missing_is_not_claimed():
    """The half-applied update case: the route table has preflight but not the policy
    endpoints, so the robot admits it cannot install a policy."""
    routes = [FakeRoute("/api/preflight", {"POST"}), FakeRoute("/api/health", {"GET"})]
    assert capabilities_for(routes) == ["preflight"]


def test_a_route_registered_with_the_wrong_method_does_not_count():
    """GET /api/policy/install is not POST /api/policy/install. Matching on the path alone
    would claim a capability that 405s."""
    routes = [FakeRoute("/api/policy/install", {"GET"})]
    assert capabilities_for(routes) == []


def test_an_empty_route_table_claims_nothing():
    assert capabilities_for([]) == []


def test_routes_without_methods_are_ignored():
    """Mounts and websocket routes have no ``methods``; walking them must not raise, since
    /api/health is the one endpoint that has to answer on any robot."""
    class Mount:
        path = "/static"
        methods = None

    assert capabilities_for([Mount()]) == []


def test_the_list_is_sorted_so_the_payload_is_stable():
    routes = [
        FakeRoute("/api/policy/select", {"POST"}),
        FakeRoute("/api/policy/install", {"POST"}),
        FakeRoute("/api/preflight", {"POST"}),
    ]
    assert capabilities_for(routes) == ["policy.install", "policy.select", "preflight"]


def test_the_list_is_computed_once(monkeypatch):
    """/api/health is polled. Walking the route table per request would be work done
    forever to answer a question whose answer cannot change after startup."""
    calls = []
    real = capabilities_for

    def counting(routes):
        calls.append(1)
        return real(routes)

    monkeypatch.setattr(tnkr_server, "capabilities_for", counting)
    monkeypatch.setattr(tnkr_server, "_capabilities_cache", None)

    first = capabilities()
    second = capabilities()

    assert first == second
    assert len(calls) == 1


def test_the_returned_list_cannot_be_mutated_by_a_caller(monkeypatch):
    """It is cached. A caller that appended to it would corrupt every later /api/health."""
    monkeypatch.setattr(tnkr_server, "_capabilities_cache", None)
    capabilities().append("policy.everything")
    assert "policy.everything" not in capabilities()


def test_health_is_not_captured_as_telemetry(client, captured):
    """Studio polls it; it is on the excluded list. Asserted here because adding a field to
    a polled endpoint is exactly when someone reconsiders that."""
    client.get("/api/health")
    assert not [e for e in captured if e["properties"].get("endpoint") == "/api/health"]
