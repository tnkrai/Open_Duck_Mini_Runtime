"""Configurable test double for onnxruntime (heavy native wheel; only on-robot).

Why this is a double and not the real wheel
-------------------------------------------
``onnxruntime`` is a ~50 MB native wheel that CI does not install
(``.github/workflows/test.yml`` installs fastapi/numpy/pytest and then
``pip install -e . --no-deps``). ``mini_bdx_runtime/__init__.py`` imports OnnxInfer, which
imports onnxruntime at module level, so *something* has to be importable under that name.

Why it is no longer a raising placeholder
-----------------------------------------
It used to be::

    class InferenceSession:  # pragma: no cover - never instantiated in tests
        def __init__(self, *args, **kwargs):
            raise RuntimeError("onnxruntime stub: not available in tests")

That was true right up until the Pi's contract check became the security boundary
(``_architecture.md`` amendment A1: the robot's HTTP API has no auth and its CORS reflects
any requesting origin, so a check that only lives in Studio is decoration). The boundary
check must construct a session to read a graph's input and output specs -- under the old
stub every test of it died on setup, which is the same as not testing the thing standing
between an arbitrary ONNX and the servos.

So this is now a double that presents whatever graph a test asks for, including graphs no
exporter would ever emit -- three inputs, a dynamic batch axis, an int64 action head.

How a test uses it
------------------
Through the ``onnx_specs`` fixture (``tests/conftest.py``), never by touching this module's
registry directly -- the fixture is what guarantees teardown, so one test's malformed graph
cannot leak into the next::

    onnx_specs.register(path, inputs=[("obs", [1, 101], "tensor(float)")],
                              outputs=[("continuous_actions", [1, 14], "tensor(float)")])

Fidelity
--------
A hand-written fake of a third-party API drifts. ``tests/test_stub_fidelity.py`` pins it:
when the real wheel *is* importable it asserts, against real onnxruntime, that ``NodeArg``
exposes exactly ``name``/``shape``/``type``, that dtype strings read ``tensor(float)``, and
that a corrupt file raises ``InvalidProtobuf``. Every value hardcoded below was read off
onnxruntime 1.24.4 loading ``scripts/BEST_WALK_ONNX_2.onnx``:

    NodeArg 'obs'                [1, 101]  tensor(float)
    NodeArg 'continuous_actions'  [1, 14]  tensor(float)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# Shape entries are ints for a static axis; ``None`` or a string name for a dynamic one --
# both spellings occur in the wild and both must be representable, because the check refuses
# them and that refusal needs a test.
Dim = int | str | None
SpecTuple = tuple[str, Sequence[Dim], str]

FLOAT_TYPE = "tensor(float)"


# ── the exception surface, mirroring onnxruntime.capi.onnxruntime_pybind11_state ──
#
# All three are direct subclasses of Exception in the real wheel -- Fail is NOT their base
# class, which matters because code that catches Fail alone would miss InvalidProtobuf.
class Fail(Exception):
    """Real ort raises this for a structurally invalid model (e.g. an empty file)."""


class InvalidProtobuf(Exception):
    """Real ort raises this for a file that is not protobuf at all."""


class NoSuchFile(Exception):
    """Real ort raises this when the path does not exist."""


class InvalidArgument(Exception):
    """Real ort raises this when ``run()`` is missing a required input."""


class UnregisteredPath(BaseException):
    """A session was constructed for a path no fixture registered.

    Deliberately derived from ``BaseException``, not ``Exception``: the code under test
    wraps session construction in ``except Exception`` and maps it to a refusal, so an
    Exception here would be silently absorbed and a test that forgot to register a spec
    would still pass -- as a rejection test, on a rejection it never actually caused.
    Escaping that ``except`` is the whole point.
    """


@dataclass
class NodeArg:
    """Mirrors onnxruntime's NodeArg: exactly these three attributes, nothing else."""

    name: str
    shape: list[Dim]
    type: str


@dataclass
class _Spec:
    inputs: list[NodeArg] = field(default_factory=list)
    outputs: list[NodeArg] = field(default_factory=list)
    invalid: bool = False
    error: type[BaseException] = InvalidProtobuf
    delay_s: float = 0.0


# Registry and construction log. Module-level because the import system gives us exactly one
# module instance; the fixture owns the lifecycle and clears both after every test.
_SPECS: dict[str, _Spec] = {}
CONSTRUCTED: list[str] = []


def _key(path_or_bytes: Any) -> str:
    return str(path_or_bytes)


def _node_args(specs: Iterable[SpecTuple]) -> list[NodeArg]:
    return [NodeArg(name, list(shape), dtype) for name, shape, dtype in specs]


def _register(
    path: Any,
    inputs: Iterable[SpecTuple] = (),
    outputs: Iterable[SpecTuple] = (),
    *,
    invalid: bool = False,
    error: type[BaseException] = InvalidProtobuf,
    delay_s: float = 0.0,
) -> None:
    """Declare the graph a session over ``path`` presents.

    Called by the ``onnx_specs`` fixture, which owns clearing it again.
    """
    _SPECS[_key(path)] = _Spec(
        inputs=_node_args(inputs),
        outputs=_node_args(outputs),
        invalid=invalid,
        error=error,
        delay_s=delay_s,
    )


def _clear() -> None:
    _SPECS.clear()
    CONSTRUCTED.clear()


class InferenceSession:
    """Presents the graph a test registered for this path.

    Signature follows the real one loosely: the runtime only ever calls it as
    ``InferenceSession(path, providers=[...])`` (``onnx_infer.py:6``), so the rest is
    accepted and ignored.
    """

    def __init__(
        self,
        path_or_bytes: Any,
        sess_options: Any = None,
        providers: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        key = _key(path_or_bytes)
        CONSTRUCTED.append(key)

        spec = _SPECS.get(key)
        if spec is None:
            raise UnregisteredPath(
                f"onnxruntime double: no graph registered for {key!r}.\n"
                "Register one in the test via the onnx_specs fixture, e.g.\n"
                "    onnx_specs.register(\n"
                "        path,\n"
                "        inputs=[('obs', [1, 101], 'tensor(float)')],\n"
                "        outputs=[('action', [1, 14], 'tensor(float)')],\n"
                "    )\n"
                "or onnx_specs.register(path, invalid=True) for a file that will not parse."
            )
        if spec.invalid:
            raise spec.error(
                f"[ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model from {key} failed:"
                "Protobuf parsing failed."
            )

        self._spec = spec
        self._path = key

    # Real ort returns fresh NodeArg objects; copying keeps a test that mutates a returned
    # shape from editing the registry another assertion in the same test then reads.
    def get_inputs(self) -> list[NodeArg]:
        return [NodeArg(a.name, list(a.shape), a.type) for a in self._spec.inputs]

    def get_outputs(self) -> list[NodeArg]:
        return [NodeArg(a.name, list(a.shape), a.type) for a in self._spec.outputs]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: dict[str, Any],
        run_options: Any = None,
    ) -> list[Any]:
        """Zeros of the declared output shape, after an optional injected delay.

        Story 2.6 measures inference latency by calling this ~50 times and taking p50/p99,
        so ``delay_s`` is what makes its over-budget branch deterministic instead of a
        function of how fast the machine running CI happens to be.
        """
        missing = [a.name for a in self._spec.inputs if a.name not in input_feed]
        if missing:
            raise InvalidArgument(
                f"[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Missing Input: {missing[0]}"
            )

        if self._spec.delay_s:
            time.sleep(self._spec.delay_s)

        wanted = self._spec.outputs
        if output_names:
            by_name = {a.name: a for a in self._spec.outputs}
            wanted = [by_name[n] for n in output_names if n in by_name]

        # A dynamic axis has no concrete size; a batch of one is what the walk loop asks
        # for (awd=True wraps the observation in a list), so that is what a zero fill uses.
        return [
            np.zeros(
                tuple(d if isinstance(d, int) else 1 for d in arg.shape),
                dtype=np.float32,
            )
            for arg in wanted
        ]


def get_available_providers() -> list[str]:
    return ["CPUExecutionProvider"]


__all__ = [
    "CONSTRUCTED",
    "Fail",
    "InferenceSession",
    "InvalidArgument",
    "InvalidProtobuf",
    "NoSuchFile",
    "NodeArg",
    "UnregisteredPath",
    "get_available_providers",
]