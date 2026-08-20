"""How long a policy takes to think, measured at install (story 2.6, amendment A7).

The rule this file enforces
---------------------------
**Measure and report. Never block.** No number produced here may refuse an install. That is
not timidity: ``kinfer/examples/timing.py`` runs the same 20 ms / 50 Hz loop and reports
deviation from the expected tick with no pass/fail threshold anywhere, and nobody has
published what a community-trained Open Duck policy costs per step. A threshold invented
today would reject working policies. So an over-budget policy installs, flagged
``POLICY_SLOW``, and story 1.3's reactive abort stays the enforcement.

How the timings are made deterministic
--------------------------------------
Two mechanisms, on purpose:

* Most tests drive a **scripted clock**: the fake session advances a fake ``perf_counter``
  by exactly the cost it is meant to have. That gives exact p50/p99 assertions with no
  sleeping, so "warm-up was excluded" is provable rather than probable.
* One test goes through the **real store and the real onnxruntime double** with an injected
  ``delay_s``, because a chain of unit tests can all pass while nothing is wired together.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest

import tnkr_server
from conftest import fake_fetch
from mini_bdx_runtime import policy_contract, policy_store
from mini_bdx_runtime.policy_contract import (
    ACT_DIM,
    BUDGET_MS,
    CONTROL_HZ,
    OBS_DIM,
    OBS_INPUT_NAME,
    POLICY_SLOW,
    LatencyReport,
    measure_latency,
    measure_latency_at,
    percentile,
)
from mini_bdx_runtime.policy_store import PolicyStore

URL = "https://example.com/policies/9f2a/model.onnx?X-Amz-Signature=deadbeef"


# ── harness ─────────────────────────────────────────────────────────────────────


class Clock:
    """A fake ``perf_counter``. Only the scripted session moves it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedSession:
    """A session whose every inference costs exactly what the test says it costs.

    Sleeping for real would make a p99 assertion a bet on the CI machine's scheduler. This
    advances the clock instead, so 50 inferences of 5 ms each take no wall-clock time and
    still measure as 5 ms.
    """

    def __init__(self, clock: Clock, costs_ms, *, raises: type[Exception] | None = None):
        self.clock = clock
        self.costs_ms = list(costs_ms)
        self.raises = raises
        self.runs = 0

    def get_inputs(self):
        from onnxruntime import NodeArg

        return [NodeArg(OBS_INPUT_NAME, [1, OBS_DIM], "tensor(float)")]

    def get_outputs(self):
        from onnxruntime import NodeArg

        return [NodeArg("continuous_actions", [1, ACT_DIM], "tensor(float)")]

    def run(self, output_names, feed):
        if self.raises is not None:
            raise self.raises("this graph will not run on zeros")
        cost = self.costs_ms[min(self.runs, len(self.costs_ms) - 1)]
        self.runs += 1
        self.clock.advance(cost / 1000.0)
        return []


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(policy_contract.time, "perf_counter", c)
    return c


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ── percentiles ─────────────────────────────────────────────────────────────────


def test_percentile_on_a_known_distribution():
    samples = list(range(1, 101))
    assert percentile(samples, 0.50) == 50
    assert percentile(samples, 0.99) == 99
    assert percentile(samples, 1.0) == 100


def test_percentile_does_not_need_sorted_input():
    assert percentile([9.0, 1.0, 5.0], 0.5) == 5.0


def test_percentile_of_one_sample_is_that_sample():
    assert percentile([7.5], 0.99) == 7.5


def test_percentile_returns_a_measurement_that_happened():
    """Nearest-rank, not interpolated: an interpolated p99 of 45 samples invents a value
    between the two slowest observations, and what is being reported is how slow it
    actually got."""
    assert percentile([1.0, 2.0], 0.99) in (1.0, 2.0)


def test_percentile_of_nothing_is_an_error_not_a_zero():
    """Zero would read as "instant" on a screen."""
    with pytest.raises(ValueError):
        percentile([], 0.5)


# ── measuring ───────────────────────────────────────────────────────────────────


def test_warmup_inferences_are_excluded(clock):
    """The first calls pay allocation and page-fault costs no steady-state tick pays.

    Five 100 ms warm-ups followed by fifty 5 ms inferences: if warm-up leaked in, p99 would
    be 100.
    """
    session = ScriptedSession(clock, [100.0] * 5 + [5.0] * 50)

    report = measure_latency(session, OBS_DIM, iterations=50, warmup=5)

    assert report.measured is True
    assert report.samples == 50
    assert report.p50_ms == 5.0
    assert report.p99_ms == 5.0


def test_the_tail_is_what_gets_reported(clock):
    """A policy averaging 8 ms that spikes to 25 ms every tenth step falls over, so the
    tail is the number that matters -- it is what trips story 1.3's abort."""
    costs = [5.0] * 49 + [40.0]
    session = ScriptedSession(clock, [0.0] * 5 + costs)

    report = measure_latency(session, OBS_DIM, iterations=50, warmup=5)

    assert report.p50_ms == 5.0
    assert report.p99_ms == 40.0


def test_the_budget_comes_from_the_control_rate(clock):
    session = ScriptedSession(clock, [1.0])
    assert measure_latency(session, OBS_DIM).budget_ms == BUDGET_MS == 1000.0 / CONTROL_HZ
    assert measure_latency(session, OBS_DIM, control_hz=25).budget_ms == 40.0


def test_over_budget_is_flagged_and_named(clock):
    session = ScriptedSession(clock, [31.4] * 60)

    report = measure_latency(session, OBS_DIM, iterations=50, warmup=5)

    assert report.over_budget is True
    assert report.warning_code == POLICY_SLOW
    # The number is meaningless alone, so the detail always carries the comparison.
    assert "31.4" in report.detail and "20.0" in report.detail


def test_inside_budget_is_not_flagged(clock):
    report = measure_latency(ScriptedSession(clock, [6.8] * 60), OBS_DIM)
    assert report.over_budget is False
    assert report.p50_ms == 6.8


def test_exactly_at_the_budget_is_not_over_it(clock):
    report = measure_latency(ScriptedSession(clock, [BUDGET_MS] * 60), OBS_DIM)
    assert report.over_budget is False


def test_the_machine_is_recorded(clock):
    """Worthless without it: an SD card moved into a different Pi carries its old numbers
    along, and the machine string is what makes that visible instead of misleading."""
    report = measure_latency(ScriptedSession(clock, [1.0]), OBS_DIM)
    assert report.machine == platform.machine()


def test_measurement_stops_at_the_wall_clock_cap(clock):
    """50 iterations of a 1 s policy would be 55 s of install. The cap keeps install
    responsive, and costs precision only on policies whose slowness is already obvious."""
    session = ScriptedSession(clock, [1000.0] * 60)

    report = measure_latency(session, OBS_DIM, iterations=50, warmup=1, max_seconds=2.0)

    assert report.measured is True
    assert 0 < report.samples < 50
    assert report.over_budget is True


def test_the_cap_bounds_the_warm_up_too_at_production_defaults(clock):
    """The cap has to cover the untimed inferences, and this is the only test that would
    notice if it did not: it passes NO kwargs, so `warmup=5` is the real default.

    A 1 s policy paying five untimed warm-ups before the first timed one spends 6 s inside
    an install request that holds the policy lock -- three times the ~2 s the story budgets,
    for a robot whose owner was told nothing. `warmup * cost` is unbounded in cost.
    """
    session = ScriptedSession(clock, [1000.0] * 60)
    started = clock.now

    report = measure_latency(session, OBS_DIM)

    consumed = clock.now - started
    # 1 s inferences and a 2 s cap: two warm-ups fit, then one timed one. The general
    # ceiling is the cap plus two inferences -- the warm-up in flight when the deadline
    # passes, and the timed one that must follow it -- and both are inherent.
    assert consumed == 3.0, (
        f"measurement spent {consumed}s against a {policy_contract.MAX_MEASURE_SECONDS}s cap"
    )
    assert report.measured is True and report.samples == 1
    assert report.over_budget is True


def test_a_slow_policy_still_gets_measured_rather_than_skipped(clock):
    """The warm-up cap must not turn a slow policy into an unknown one. Slow policies are
    the whole reason story 2.6 exists, so "ran out of time warming up" means start timing,
    never give up: `measured=False` here would hide the exact case worth reporting."""
    report = measure_latency(
        ScriptedSession(clock, [5000.0] * 10), OBS_DIM, max_seconds=0.5
    )

    assert report.measured is True
    assert report.samples == 1
    assert report.p99_ms == 5000.0


def test_the_cap_never_produces_a_report_with_no_samples(clock):
    """One inference slower than the whole cap still yields a number, because the deadline
    is checked after the sample is recorded rather than before it is taken."""
    report = measure_latency(
        ScriptedSession(clock, [9999.0]), OBS_DIM, iterations=50, warmup=0, max_seconds=0.0
    )
    assert report.measured is True
    assert report.samples == 1


# ── when measurement cannot happen ─────────────────────────────────────────────


def test_a_graph_that_refuses_zeros_reports_unknown_rather_than_raising(clock):
    """Story 2.6's named error scenario. A graph asserting an input range may reject a
    zeros observation; that is a reportable unknown, never a failed install."""
    session = ScriptedSession(clock, [1.0], raises=ValueError)

    report = measure_latency(session, OBS_DIM)

    assert report.measured is False
    assert report.p50_ms is None and report.p99_ms is None
    assert report.over_budget is False, "unknown must not read as over budget"
    assert "unknown" in report.detail


def test_a_session_whose_specs_cannot_be_read_reports_unknown():
    class Hostile:
        def get_inputs(self):
            raise RuntimeError("no graph here")

    report = measure_latency(Hostile(), OBS_DIM)
    assert report.measured is False
    assert report.p50_ms is None


def test_a_graph_with_no_input_reports_unknown():
    class Empty:
        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

    assert measure_latency(Empty(), OBS_DIM).measured is False


def test_measuring_a_file_that_will_not_open_reports_unknown(tmp_path, onnx_specs):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not a model")
    onnx_specs.register(path, invalid=True)

    report = measure_latency_at(path)

    assert report.measured is False
    assert report.p50_ms is None


# ── the manifest ───────────────────────────────────────────────────────────────


def test_the_manifest_fields_are_the_ones_studio_reads(clock):
    fields = measure_latency(ScriptedSession(clock, [6.8] * 60), OBS_DIM).as_manifest_fields()

    assert set(fields) == {
        "latency_p50_ms",
        "latency_p99_ms",
        "latency_budget_ms",
        "latency_samples",
        "latency_measured",
        "latency_over_budget",
        "latency_detail",
        "machine",
    }
    assert fields["latency_budget_ms"] == 20.0


def test_the_manifest_docstring_says_it_is_not_headroom():
    """Documented in the code the number comes from, not only in a plan file. A UI that
    presents 18-of-20 as "2 ms spare" is wrong, and the docstring is where the next person
    to touch this will look."""
    doc = LatencyReport.as_manifest_fields.__doc__ or ""
    assert "NOT tick time" in doc
    assert "headroom" in doc


# ── wired into install ─────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "BEST_WALK_ONNX_2.onnx").write_bytes(b"builtin")
    return PolicyStore(root=tmp_path / "policies", scripts_dir=scripts)


def install(store, onnx_specs, policy_id="9f2a", **kwargs):
    payload = f"model-{policy_id}".encode()
    store.fetch = fake_fetch(onnx_specs, payload=payload, **kwargs)
    return store.install(policy_id, URL, digest(payload))


def test_install_records_the_measurement_on_the_manifest(store, onnx_specs):
    result = install(store, onnx_specs)

    stored = json.loads((store.root / "9f2a" / "manifest.json").read_text())
    assert stored["latency_measured"] is True
    assert stored["latency_p50_ms"] >= 0
    assert stored["latency_p99_ms"] >= stored["latency_p50_ms"]
    assert stored["latency_budget_ms"] == 20.0
    assert stored["machine"] == platform.machine()
    assert result.warning is None


def test_a_slow_policy_installs_anyway_and_is_flagged(store, onnx_specs):
    """End to end through the real store and the real double, with a 21 ms inference
    injected: over budget, flagged POLICY_SLOW, and INSTALLED.

    The slowest test in this file by design -- it is the one that proves the wiring, and a
    chain of unit tests can all pass while nothing is connected.
    """
    result = install(store, onnx_specs, delay_s=0.021)

    assert result.ok is True
    assert (store.root / "9f2a" / "model.onnx").exists()
    assert result.warning["code"] == POLICY_SLOW
    assert "budget" in result.warning["detail"]
    assert result.manifest["latency_over_budget"] is True


def test_an_unmeasurable_policy_still_installs(store, onnx_specs):
    """Never fail an install because measurement failed. The shape check is the gate."""
    result = install(store, onnx_specs, run_error=ValueError)

    assert result.ok is True
    assert (store.root / "9f2a" / "model.onnx").exists()
    assert result.manifest["latency_measured"] is False
    assert result.manifest["latency_p99_ms"] is None


def test_an_unmeasurable_policy_says_latency_is_unknown(store, onnx_specs):
    """Story 2.6's error scenario ends "warning says latency is unknown", and this is the
    half that is easy to lose: `warning: None` is what a healthy 7 ms policy sends, so a
    caller that checks only `warning` would read "the stopwatch failed" as "fast enough".
    It carries no code -- unknown is not POLICY_SLOW, which would assert a slowness nobody
    measured.
    """
    healthy = install(store, onnx_specs, "fast")

    result = install(store, onnx_specs, run_error=ValueError)

    assert healthy.warning is None
    assert result.warning is not None, (
        "an unknown latency is indistinguishable from a fast policy"
    )
    assert result.warning["code"] is None
    assert "unknown" in result.warning["detail"]
    assert result.warning["code"] != POLICY_SLOW


def test_the_reason_measurement_failed_outlives_the_response(store, onnx_specs, capsys):
    """The response is gone once Studio has read it, so the *why* is recorded twice: in the
    policy's own manifest.json and on the robot's console. `latency_measured: false` three
    fields deep does not say the graph refused a zeros observation."""
    result = install(store, onnx_specs, run_error=ValueError)

    on_disk = json.loads((store.root / "9f2a" / "manifest.json").read_text())
    assert "zeros observation" in on_disk["latency_detail"]
    assert "ValueError" in on_disk["latency_detail"]
    assert "9f2a" in capsys.readouterr().out


def test_a_healthy_policy_records_its_numbers_in_the_manifest_detail(store, onnx_specs):
    """The same field on a policy that measured: recorded either way, so a manifest read
    months later never has to guess whether the field is missing or the measurement was."""
    result = install(store, onnx_specs)

    assert result.manifest["latency_measured"] is True
    assert "budget" in result.manifest["latency_detail"]
    assert result.warning is None


def test_slowness_is_never_an_http_failure():
    """Structural, because this is the whole point of A7: POLICY_SLOW rides an ok:true
    response. If it ever appears in the refusal table it has become a threshold, and
    nobody has the data to set one."""
    assert POLICY_SLOW not in tnkr_server.POLICY_STATUS_CODES
    store_source = Path(policy_store.__file__).read_text()
    assert "POLICY_SLOW" not in store_source, (
        "the store references POLICY_SLOW by name; it should only ever arrive as a "
        "warning code carried on a LatencyReport"
    )


def test_the_slow_install_still_reports_over_budget_over_the_api(
    client, tmp_path, monkeypatch, onnx_specs
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "BEST_WALK_ONNX_2.onnx").write_bytes(b"builtin")
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", scripts)
    monkeypatch.setattr(tnkr_server, "POLICY_ROOT", tmp_path / "policies")
    monkeypatch.setattr(
        tnkr_server,
        "POLICY_FETCH",
        fake_fetch(onnx_specs, payload=b"heavy model", delay_s=0.021),
    )

    response = client.post(
        "/api/policy/install",
        json={"id": "heavy", "url": URL, "sha256": digest(b"heavy model")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["warning"]["code"] == POLICY_SLOW
    assert body["manifest"]["latency_p99_ms"] > 20.0


def test_an_unmeasurable_install_says_so_over_the_api(
    client, tmp_path, monkeypatch, onnx_specs, captured
):
    """The wire, not just the dataclass: what Studio actually receives for a policy whose
    graph will not run on zeros. `ok: true` with `warning: null` would be the same body a
    7 ms policy sends, and the reason would exist nowhere in the response."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "BEST_WALK_ONNX_2.onnx").write_bytes(b"builtin")
    monkeypatch.setattr(tnkr_server, "SCRIPTS_DIR", scripts)
    monkeypatch.setattr(tnkr_server, "POLICY_ROOT", tmp_path / "policies")
    monkeypatch.setattr(
        tnkr_server,
        "POLICY_FETCH",
        fake_fetch(onnx_specs, payload=b"picky model", run_error=ValueError),
    )

    response = client.post(
        "/api/policy/install",
        json={"id": "picky", "url": URL, "sha256": digest(b"picky model")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["warning"] is not None and body["warning"]["code"] is None
    assert "unknown" in body["warning"]["detail"]
    assert body["manifest"]["latency_measured"] is False
    # A policy nobody could time is not a slow policy. `over_budget` is the number this
    # feature exists to gather -- A7 says measure now and set a threshold once there is
    # data -- so counting unmeasurable installs into it would poison exactly that data.
    props = next(
        e["properties"] for e in captured if e["properties"]["endpoint"].endswith("/install")
    )
    assert props["over_budget"] is False
    assert props["latency_measured"] is False
