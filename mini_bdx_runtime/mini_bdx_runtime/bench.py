"""The supervised bench run: a bounded first run of an unproven policy, watched.

Why this exists (architecture Decision 10)
------------------------------------------
Decision 1 dropped the MuJoCo gate, and recorded the consequence honestly: nothing in the
custom-policy design predicts whether a policy *walks well*. A policy can hold every joint
inside its limits, keep every deadline, and still produce a gait that faceplants the moment
it bears weight. The only judge available is a person looking at the duck, so the first
hardware run of a policy this robot has never run is ten seconds with the operator holding
it off the ground.

What this module is NOT
----------------------
It is not a second control loop. ``mode`` is a flag on ``RLWalk.run``, and everything here
is state a single tick of the existing 50 Hz loop can advance in a few microseconds:

    while True:                      # the one loop, unchanged
        ...
        if self.bench is not None:   # one attribute test per tick
            ended = self.bench.tick(took)
            if ended is not None:
                break                # -> the loop's own finally -> torque off

A separate bench loop would be a second, untested safety envelope; the whole argument for
running the bench inside this loop is that the clamps and aborts guarding it are the ones
that have been tested.

It also computes no verdict. ``did not abort`` is not ``walked well``, and a heuristic that
inferred a pass would be a rubber stamp on the one check the design has left. This module
records what happened -- ticks, timing, clamp counts, how the run ended -- and the operator
answers the question.

The two files, and why they are files
------------------------------------
The walk is a separate process that owns the servo bus, so the server cannot reach into its
loop. Same shape as ``walk_pause`` and ``walk_telemetry``:

``STOP_FILE``
    Existence *is* the request: the server creates it on ``POST /api/bench/stop`` and the
    loop ends the bench on its next tick. A marker rather than JSON because this is read
    50 times a second inside a 20 ms budget, and because there is nothing to say -- the
    only content would be "yes". Cleared when a bench starts, so a stale request from a
    previous run cannot end the next one before it begins.

``REPORT_FILE``
    Latched, like the abort file next door: the server asks what happened *after* the
    process is gone, so the report has to outlive the writer. Written in ``RLWalk.run``'s
    ``finally``, which is the one teardown -- the same path SIGTERM already takes -- so
    every way out of the loop leaves a report and none of them can forget one.

Fail closed
-----------
A pass is recorded by the *server*, into the policy store, when the operator answers. If
the walk process dies before its report is written -- SIGKILL, a power cut, an OOM kill --
there is no report, so there is no verdict, so the policy stays gated and the bench must be
re-run. Every unreadable, truncated or unparseable state in here resolves the same way: not
benched.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

# ── Modes ───────────────────────────────────────────────────────────────────────

#: Ordinary walking. The default, and what every duck in the field does today.
MODE_FREE: str = "free"
#: A bounded, supervised run. Same loop, same envelope, a deadline and a stop flag.
MODE_BENCH: str = "bench"
MODES: tuple[str, ...] = (MODE_FREE, MODE_BENCH)

# Ten seconds is long enough to see a gait cycle several times over and short enough that
# an operator can hold a duck at arm's length for the whole run without their arm shaking,
# which would be a fake tilt abort.
DEFAULT_BENCH_SECONDS: float = 10.0
# A bench shorter than this shows nothing (the walk's own start() sleeps 2 s before the
# loop even begins); longer than this is not a bench, it is walking while holding a duck.
MIN_BENCH_SECONDS: float = 2.0
MAX_BENCH_SECONDS: float = 60.0

# ── How a bench ended ───────────────────────────────────────────────────────────
#
# Five, because there are five ways out of the loop and every one of them must cut torque.
# The names cross the wire to Studio, so they are added to, never renamed.

#: The deadline arrived. The only ending that means the run did what it was asked to.
ENDED_TIMER: str = "timer"
#: The operator ended it early, through POST /api/bench/stop.
ENDED_OPERATOR: str = "operator"
#: A guard from story 1.3 tripped. This one is a failure on its own, without a verdict.
ENDED_ABORT: str = "abort"
#: SIGTERM (POST /api/walk/stop) or Ctrl-C.
ENDED_SIGNAL: str = "signal"
#: An unhandled exception. The report is written, then the exception carries on out.
ENDED_ERROR: str = "error"

# The refusal a free walk gets when the policy it names has never passed a bench run.
# A string, not an enum, because it crosses the wire to Studio, where it needs an
# ErrorCode member and one sentence of operator copy of its own -- the same treatment the
# six codes in policy_contract got in story 2.5. Until Studio has it, this arrives there
# as an unknown code and renders as the generic refusal, which is a worse message but the
# right behaviour: the walk still does not start.
POLICY_BENCH_REQUIRED: str = "POLICY_BENCH_REQUIRED"

#: Endings after which the operator may still record a pass. An aborted run has already
#: been judged by the envelope, and a crashed or signalled one was not watched to its end,
#: so neither can be passed -- while a *fail* verdict is accepted on any of them.
PASSABLE_ENDINGS: frozenset[str] = frozenset({ENDED_TIMER, ENDED_OPERATOR})

# tmpfs on the Pi (RAM -- no SD-card wear); tempdir off-Pi so this stays importable and
# testable on a dev machine with no /dev/shm. Mirrors walk_pause and walk_telemetry.
_SHM = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()

STOP_FILE: str = os.path.join(_SHM, "tnkr_bench_stop")
REPORT_FILE: str = os.path.join(_SHM, "tnkr_bench_report.json")


def clamp_seconds(seconds: float | None) -> float:
    """A usable bench duration from whatever arrived over an unauthenticated HTTP API.

    Clamped rather than refused: this endpoint has no auth (amendment A1's premise), so
    ``benchSeconds: 1e9`` is a value someone can send, and the answer to it is ten seconds
    of torque, not an unbounded run. ``None`` and unparseable both mean the default.
    """
    if seconds is None:
        return DEFAULT_BENCH_SECONDS
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return DEFAULT_BENCH_SECONDS
    if value != value:  # NaN, which every comparison below would silently pass
        return DEFAULT_BENCH_SECONDS
    return min(MAX_BENCH_SECONDS, max(MIN_BENCH_SECONDS, value))


def parse_mode(mode: str | None) -> str:
    """``"free"`` or ``"bench"``. Anything else raises ``ValueError``.

    Deliberately not lenient. A caller that sent ``"Bench"`` and silently got free walking
    would have skipped the gate this whole story exists to impose.
    """
    if mode is None:
        return MODE_FREE
    if mode in MODES:
        return mode
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


# ── The stop request ────────────────────────────────────────────────────────────


def request_stop() -> None:
    """Ask the running bench to end now. Idempotent."""
    with open(STOP_FILE, "wb"):
        pass


def stop_requested() -> bool:
    """Whether a stop has been asked for. One stat, on tmpfs, per tick."""
    return os.path.exists(STOP_FILE)


def clear_stop() -> None:
    try:
        os.remove(STOP_FILE)
    except FileNotFoundError:
        pass


# ── The report ──────────────────────────────────────────────────────────────────


@dataclass
class BenchReport:
    """What one bench run did. Facts only -- no verdict, by design.

    ``ended`` is the load-bearing field: it says which of the five exits ran, and the
    server refuses to record a *pass* for an ending that was not watched to a clean finish.
    """

    mode: str = MODE_BENCH
    policy_id: str | None = None
    seconds: float = DEFAULT_BENCH_SECONDS
    elapsed_s: float = 0.0
    ticks: int = 0
    ended: str = ENDED_ERROR
    detail: str = ""
    aborted: bool = False
    abort_code: str | None = None
    abort_reason: str | None = None
    abort_operator: str | None = None
    mean_tick_ms: float | None = None
    clamped: dict[str, int] = field(default_factory=dict)

    @property
    def clamp_events(self) -> int:
        return sum(self.clamped.values())

    def as_dict(self) -> dict:
        """camelCase, because Studio reads it. Same convention as the policy list."""
        return {
            "mode": self.mode,
            "policyId": self.policy_id,
            "seconds": round(self.seconds, 3),
            "elapsedS": round(self.elapsed_s, 3),
            "ticks": self.ticks,
            "ended": self.ended,
            "detail": self.detail,
            "aborted": self.aborted,
            "abortCode": self.abort_code,
            "abortReason": self.abort_reason,
            "abortOperator": self.abort_operator,
            "meanTickMs": (
                None if self.mean_tick_ms is None else round(self.mean_tick_ms, 2)
            ),
            "clamped": dict(self.clamped),
            "clampEvents": self.clamp_events,
            # Whether a pass is even offerable. Studio needs this to decide between "did
            # that look like walking?" and "the run stopped itself" -- and the verdict
            # endpoint enforces the same rule, so the UI cannot talk past it.
            "passable": self.ended in PASSABLE_ENDINGS,
        }

    def summary(self) -> str:
        """The one line the walk prints on its way out, for the robot's own console."""
        timing = (
            "" if self.mean_tick_ms is None else f", mean {self.mean_tick_ms:.1f}ms/tick"
        )
        return (
            f"{self.ended}: {self.ticks} ticks in {self.elapsed_s:.1f}s"
            f"{timing}, {self.clamp_events} clamp events"
            f"{', aborted' if self.aborted else ''}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "BenchReport":
        """Rebuild a report from the file. Anything unexpected reads as an error ending.

        Fail closed: a report whose ``ended`` field was truncated must not become a
        passable run, so the default for every field is the pessimistic one.
        """
        ended = data.get("ended")
        clamped = data.get("clamped")
        if not isinstance(clamped, dict):
            clamped = {}
        return cls(
            mode=str(data.get("mode") or MODE_BENCH),
            policy_id=data.get("policyId"),
            seconds=_as_float(data.get("seconds"), DEFAULT_BENCH_SECONDS),
            elapsed_s=_as_float(data.get("elapsedS"), 0.0),
            ticks=int(_as_float(data.get("ticks"), 0.0)),
            ended=ended if ended in _ENDINGS else ENDED_ERROR,
            detail=str(data.get("detail") or ""),
            aborted=bool(data.get("aborted")),
            abort_code=data.get("abortCode"),
            abort_reason=data.get("abortReason"),
            abort_operator=data.get("abortOperator"),
            mean_tick_ms=(
                None
                if data.get("meanTickMs") is None
                else _as_float(data.get("meanTickMs"), 0.0)
            ),
            clamped={str(k): int(_as_float(v, 0.0)) for k, v in clamped.items()},
        )


_ENDINGS: frozenset[str] = frozenset(
    {ENDED_TIMER, ENDED_OPERATOR, ENDED_ABORT, ENDED_SIGNAL, ENDED_ERROR}
)


def _as_float(value: object, fallback: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def write_report(report: BenchReport) -> None:
    """Publish the report. Temp + rename, so a reader never sees half of one."""
    tmp = REPORT_FILE + ".tmp"
    payload = dict(report.as_dict())
    payload["timestamp"] = time.time()
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, REPORT_FILE)


def read_report() -> BenchReport | None:
    """The last bench run's report, or ``None``. No freshness window: the reader is asking
    about a process that has already exited."""
    try:
        with open(REPORT_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return BenchReport.from_dict(data)


def clear_report() -> None:
    try:
        os.remove(REPORT_FILE)
    except FileNotFoundError:
        pass


# ── The run ─────────────────────────────────────────────────────────────────────


class BenchRun:
    """The bench's whole state: a deadline, a tick count, and how it ended.

    Pure except for the stop-flag stat, so every ending can be exercised without a robot
    -- which matters more here than anywhere else in this feature, because the thing under
    test is that torque comes off on all five of them.
    """

    def __init__(
        self,
        seconds: float | None = None,
        policy_id: str | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        stop_check: Callable[[], bool] = stop_requested,
    ) -> None:
        self.seconds: float = clamp_seconds(seconds)
        self.policy_id = policy_id
        self._clock = clock
        self._stop_check = stop_check
        self.started_at: float = self._clock()
        self.ticks: int = 0
        self._took_sum_s: float = 0.0
        self.clamped: dict[str, int] = {}
        self.ended: str | None = None
        self.detail: str = ""
        self._abort: object | None = None

    # ── during the run ──────────────────────────────────────────────────────────

    def elapsed(self) -> float:
        return self._clock() - self.started_at

    def tick(self, took_s: float = 0.0) -> str | None:
        """Advance one control tick. Returns the ending, or ``None`` to keep walking.

        Called from the one place in the loop that already has ``took``, so the bench
        costs the hot path a counter, an addition and one stat. Returning a string rather
        than raising keeps the ordinary ending an ordinary ``break``: the loop's ``finally``
        is the teardown either way, and an exception for "the ten seconds are up" would
        read as a failure in every log that caught it.
        """
        self.ticks += 1
        self._took_sum_s += max(0.0, float(took_s))
        if self.ended is not None:
            return self.ended
        # The stop is checked before the deadline so an operator who pressed stop on the
        # last tick gets the ending they asked for, rather than a timer report that says
        # the run completed.
        if self._stop_check():
            return self.end(ENDED_OPERATOR, "the operator ended the run")
        if self.elapsed() >= self.seconds:
            return self.end(ENDED_TIMER, f"the {self.seconds:g}s bench completed")
        return None

    def add_clamps(self, counts: dict[str, int]) -> None:
        """Accumulate one window of the envelope's clamp counts.

        The loop resets the envelope's counters every second when it publishes telemetry,
        so a bench total has to be summed as the windows go by. Clamp counts are not a
        verdict -- a policy can clamp constantly and still walk -- but they are the number
        that tells an operator whose duck looked "nearly right" that it was saturating.
        """
        for name, count in counts.items():
            self.clamped[name] = self.clamped.get(name, 0) + int(count)

    # ── endings ─────────────────────────────────────────────────────────────────

    def end(self, reason: str, detail: str = "") -> str:
        """Record how the run ended. First ending wins.

        First-wins because the loop can end for one reason and the teardown then observe
        another: a timer end unwinds through the same ``except (KeyboardInterrupt,
        SystemExit)`` clause a SIGTERM does when the two race, and "the bench completed"
        must not be overwritten by "a signal arrived while it was completing".
        """
        if self.ended is None:
            self.ended = reason
            self.detail = detail
        return self.ended

    def aborted(self, abort: object) -> str:
        """Record a tripped guard (story 1.3) as this bench's ending.

        An abort is the one ending that fails the bench without asking the operator: the
        envelope has already answered the question the operator was going to be asked.
        """
        self._abort = abort
        self.ended = ENDED_ABORT  # overrides, because an abort outranks any other ending
        self.detail = str(getattr(abort, "detail", "") or abort)
        return self.ended

    def report(self) -> BenchReport:
        """The report as it stands. Safe to call from a ``finally``."""
        abort = self._abort
        return BenchReport(
            mode=MODE_BENCH,
            policy_id=self.policy_id,
            seconds=self.seconds,
            elapsed_s=self.elapsed(),
            ticks=self.ticks,
            # An ending is only unset if the loop left without going through any of the
            # handlers, which is not a thing Python does -- but the field decides whether
            # a pass can be recorded, so its default is the ending that cannot be passed.
            ended=self.ended or ENDED_ERROR,
            detail=self.detail,
            aborted=abort is not None,
            abort_code=getattr(abort, "code", None) if abort is not None else None,
            abort_reason=getattr(abort, "reason", None) if abort is not None else None,
            abort_operator=(
                getattr(abort, "operator", None) if abort is not None else None
            ),
            mean_tick_ms=(
                None if self.ticks == 0 else (self._took_sum_s / self.ticks) * 1000.0
            ),
            clamped=dict(self.clamped),
        )
