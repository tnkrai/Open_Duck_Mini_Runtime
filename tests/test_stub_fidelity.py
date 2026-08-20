"""Pins the onnxruntime double to real onnxruntime. Skipped where the wheel is absent.

Why this file exists
--------------------
``tests/stubs/onnxruntime.py`` is a hand-written fake of a third-party API, and hand-written
fakes drift. Drift here is not a broken test -- it is a contract check that passes against a
graph shape the real robot would never see, which is worse than no check because it reads
like one.

So: when the real wheel *is* importable (a developer's machine, a Pi, or an opt-in CI job)
this file loads a genuine ONNX with genuine onnxruntime and asserts the double tells the
same story -- the NodeArg attribute set, the dtype spelling, the exception raised by a
corrupt file, and the shape ``run()`` returns.

Skipped by default, deliberately: CI installs no native wheel
(``.github/workflows/test.yml``), which is the entire reason the double exists. A skip here
means "unverified this run", not "fine".

    $ pytest tests/test_stub_fidelity.py -v
    SKIPPED [9] real onnxruntime is not installed

Importing the real wheel takes a detour
---------------------------------------
``tests/conftest.py`` puts ``tests/stubs`` first on ``sys.path``, so a plain
``import onnxruntime`` anywhere in this suite gets the double -- which is what every other
test wants. Here we need the opposite, so the real module is located with a PathFinder over
``sys.path`` minus the stub directory.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import onnxruntime as onnx_double  # tests/stubs
from mini_bdx_runtime import policy_contract
from mini_bdx_runtime.policy_contract import ACT_DIM, OBS_DIM, OBS_INPUT_NAME, check_policy

REPO = Path(__file__).parent.parent
STUB_DIR = str(Path(__file__).parent / "stubs")

# A real graph to read. The repo ships one of the only two published Open Duck policies,
# which is a better reference than anything this test could synthesise -- and synthesising
# one would need the `onnx` package, which is not a dependency of anything here.
REAL_MODELS = sorted((REPO / "scripts").glob("*.onnx"))


def _load_real_onnxruntime():
    """The wheel, not the double, or ``None`` if it is not installed."""
    search = [p for p in sys.path if p != STUB_DIR]
    spec = importlib.machinery.PathFinder().find_spec("onnxruntime", search)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # onnxruntime's own submodule imports resolve through sys.modules["onnxruntime"], so it
    # has to be registered under its real name while it executes. Restored by the fixture.
    saved = sys.modules.get("onnxruntime")
    sys.modules["onnxruntime"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - a broken wheel is a skip, not a failure
        sys.modules["onnxruntime"] = saved
        return None
    sys.modules["onnxruntime"] = saved
    return module


REAL_ORT = _load_real_onnxruntime()

pytestmark = [
    pytest.mark.skipif(REAL_ORT is None, reason="real onnxruntime is not installed"),
    pytest.mark.skipif(not REAL_MODELS, reason="no .onnx in scripts/ to read"),
]


@pytest.fixture
def real_ort(monkeypatch):
    """The real module, also installed as ``sys.modules['onnxruntime']`` for the test.

    That second part is what lets ``check_policy`` -- which imports onnxruntime lazily,
    inside the function -- run against the real wheel here while every other test in the
    suite keeps getting the double.
    """
    monkeypatch.setitem(sys.modules, "onnxruntime", REAL_ORT)
    return REAL_ORT


@pytest.fixture
def real_session(real_ort):
    return real_ort.InferenceSession(
        str(REAL_MODELS[0]), providers=["CPUExecutionProvider"]
    )


@pytest.fixture
def double_session(onnx_specs, tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx-ish bytes")
    onnx_specs.valid(path, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    return onnx_double.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _public_attrs(obj) -> set[str]:
    return {name for name in dir(obj) if not name.startswith("_")}


# ── NodeArg ─────────────────────────────────────────────────────────────────────


def test_node_arg_exposes_the_same_attributes(real_session, double_session):
    """Real NodeArg has exactly name/shape/type. If the double grew a fourth attribute the
    check could start relying on something that does not exist on hardware."""
    real = real_session.get_inputs()[0]
    fake = double_session.get_inputs()[0]

    assert _public_attrs(real) == {"name", "shape", "type"}
    assert _public_attrs(real) <= _public_attrs(fake)


def test_float_dtype_is_spelled_the_same(real_session, double_session):
    """``check_policy`` compares this string literally, so its spelling is load-bearing."""
    real_dtype = real_session.get_inputs()[0].type
    assert real_dtype == onnx_double.FLOAT_TYPE
    assert double_session.get_inputs()[0].type == real_dtype
    # The private name is the point: it is the literal check_policy compares against, so a
    # drift in onnxruntime's spelling would refuse every policy ever exported.
    assert policy_contract._FLOAT_TENSOR == real_dtype


def test_static_shape_entries_are_plain_ints(real_session, double_session):
    """The check treats a non-int dim as dynamic and refuses it. If real ort returned, say,
    a numpy int for a static axis, every real policy would be refused as dynamic."""
    real_shape = real_session.get_inputs()[0].shape
    assert all(type(d) is int for d in real_shape), real_shape
    assert all(
        type(d) is int for d in double_session.get_inputs()[0].shape
    ), "the double's static shape must be ints too"


def test_the_shipped_policy_has_the_shape_the_double_pretends(real_session):
    """The double's ``valid()`` helper claims ``[1, OBS_DIM]`` in and ``[1, ACT_DIM]`` out.
    This is the assertion that the claim is not made up."""
    obs_in = real_session.get_inputs()[0]
    act_out = real_session.get_outputs()[0]

    assert obs_in.name == OBS_INPUT_NAME
    assert list(obs_in.shape) == [1, OBS_DIM]
    assert list(act_out.shape) == [1, ACT_DIM]


# ── run() ───────────────────────────────────────────────────────────────────────


def test_run_returns_the_same_kind_of_thing(real_session, double_session):
    """Story 2.6 times ``run()`` in a loop; it must get back the same container and shape
    from both, or the latency path is written against the double's fiction."""
    obs = np.zeros((1, OBS_DIM), dtype=np.float32)

    (real_out,) = real_session.run(None, {OBS_INPUT_NAME: obs})
    (fake_out,) = double_session.run(None, {OBS_INPUT_NAME: obs})

    assert isinstance(real_out, np.ndarray)
    assert isinstance(fake_out, np.ndarray)
    assert real_out.shape == fake_out.shape == (1, ACT_DIM)
    assert real_out.dtype == fake_out.dtype == np.float32


# ── the exception surface ───────────────────────────────────────────────────────


def test_a_corrupt_file_raises_invalid_protobuf(real_ort, tmp_path):
    """The double raises ``InvalidProtobuf`` for ``invalid=True``. This is where that name
    comes from, and it is not a subclass of ``Fail`` -- code catching Fail alone would
    miss it, which is why ``check_policy`` catches ``Exception``."""
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file at all")

    with pytest.raises(Exception) as exc:
        real_ort.InferenceSession(str(bad), providers=["CPUExecutionProvider"])

    assert type(exc.value).__name__ == onnx_double.InvalidProtobuf.__name__
    bases = type(exc.value).__bases__
    assert bases == onnx_double.InvalidProtobuf.__bases__ == (Exception,)


def test_the_named_error_types_all_exist_upstream(real_ort):
    """Every exception the double defines must be a real one, or the double is inventing a
    failure mode nobody will ever see."""
    state = real_ort.capi.onnxruntime_pybind11_state
    for name in ("Fail", "InvalidProtobuf", "NoSuchFile", "InvalidArgument"):
        assert hasattr(state, name), f"real onnxruntime has no {name}"


# ── the check itself, against the real thing ────────────────────────────────────


def test_check_policy_accepts_the_shipped_policy(real_ort):
    """The end-to-end fidelity assertion: the boundary check, real onnxruntime, and the
    policy every duck in the field is running right now.

    A version of this on real hardware is the story's Definition of Done. This is the same
    code path on a workstation -- it cannot prove the Pi's timing, but it does prove the
    check does not refuse the policy the robot already runs, which is the failure that
    would brick the feature on day one.
    """
    result = check_policy(REAL_MODELS[0])

    assert result.ok, result.detail
    assert result.manifest["obs_dim"] == OBS_DIM
    assert result.manifest["act_dim"] == ACT_DIM


def test_check_policy_refuses_a_real_corrupt_file(real_ort, tmp_path):
    """The same mapping as the double's ``invalid=True`` branch, on a genuinely corrupt
    file: a refusal, not a raise."""
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file at all")

    result = check_policy(bad)

    assert not result.ok
    assert result.code == "POLICY_CONTRACT_MISMATCH"
