"""The observation contract, and the guard that keeps the walk loop matching it.

Story 1.1 deliberately does NOT refactor ``get_obs()`` to build from ``OBS_TERMS`` --
editing the function every duck in the field depends on, to buy a guarantee a test can
provide, is a bad trade. This file is that test.

It reads the source with ``ast`` rather than importing it. Importing pulls adafruit /
rustypot / onnxruntime in at module scope, and constructing ``RLWalk`` needs a real bus, an
IMU and GPIO -- which is why no test in this suite instantiates it.

What this guards, and what it does not
--------------------------------------
Two independent halves, because the two failure modes are different:

* **Order** (``test_obs_term_order``) -- the sequence of expressions the loop concatenates.
  A policy trained on ``[gyro, accelerometer, ...]`` handed ``[accelerometer, gyro, ...]``
  of the same width reads garbage and nothing raises.

* **Widths** (``test_obs_term_widths``) -- how wide each term actually is, recovered
  statically from the source that produces it. An earlier version of this file checked
  only order, and a review demonstrated the hole: adding an eighth element to
  ``last_commands`` made the real observation 102 floats while ``OBS_DIM`` stayed 101, and
  every test still passed.

Both are static-text checks, so neither can see a change in what an expression *evaluates*
to. Known blind spots, stated rather than implied:

* the action-history rotation order at ``v2_rl_walk_mujoco.py:343-345`` -- covered
  separately by ``test_action_history_rotates_oldest_first``, because it is load-bearing;
* a key swap inside ``raw_imu``'s returned dict (putting accelerometer data under
  ``"gyro"``), which no text comparison here can detect.

See ``tnkr-studio/docs/plans/custom-policy/_architecture.md``, Decision 3 and amendment A6.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from mini_bdx_runtime import policy_contract
from mini_bdx_runtime.policy_contract import (
    ACT_DIM,
    CONTROL_HZ,
    MAX_POLICY_BYTES,
    OBS_DIM,
    OBS_INPUT_NAME,
    OBS_TERMS,
    OBS_VERSION,
    describe,
    term_offsets,
    validate_obs,
)

REPO = Path(__file__).parent.parent
WALK_SCRIPT = REPO / "scripts" / "v2_rl_walk_mujoco.py"
RAW_IMU = REPO / "mini_bdx_runtime" / "mini_bdx_runtime" / "raw_imu.py"
FEET = REPO / "mini_bdx_runtime" / "mini_bdx_runtime" / "feet_contacts.py"

# The expression each declared term corresponds to, verbatim from the loop's
# np.concatenate, normalised through ast.unparse.
#
# A mismatch here is EITHER a real reordering (serious) OR a behaviour-identical rewording
# such as renaming a local (harmless). test_obs_term_order distinguishes the two, because
# telling someone to bump OBS_VERSION over a variable rename would gratuitously invalidate
# every trained policy.
TERM_TO_SOURCE: dict[str, str] = {
    "gyro": "imu_data['gyro']",
    "accelerometer": "imu_data['accelero']",
    "commands": "cmds",
    "dof_pos_rel": "dof_pos - self.init_pos",
    "dof_vel_scaled": "dof_vel * 0.05",
    "last_action": "self.last_action",
    "last_last_action": "self.last_last_action",
    "last_last_last_action": "self.last_last_last_action",
    "motor_targets": "self.motor_targets",
    "feet_contacts": "feet_contacts",
    "imitation_phase": "self.imitation_phase",
}

# Terms whose width is `num_dofs` in the walk script rather than a literal.
DOF_WIDTH_TERMS = frozenset(
    {
        "dof_pos_rel",
        "dof_vel_scaled",
        "last_action",
        "last_last_action",
        "last_last_last_action",
        "motor_targets",
    }
)


# ── static source helpers ───────────────────────────────────────────────────────


def _rlwalk_class() -> ast.ClassDef:
    tree = ast.parse(WALK_SCRIPT.read_text())
    found = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLWalk"
    ]
    assert len(found) == 1, f"expected exactly one RLWalk in {WALK_SCRIPT}, got {len(found)}"
    return found[0]


def _get_obs_fn() -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The RLWalk.get_obs definition. Scoped to the class and required to be unique, so a
    second get_obs elsewhere in the file cannot silently be validated instead."""
    cls = _rlwalk_class()
    found = [
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "get_obs"
    ]
    assert len(found) == 1, f"expected exactly one RLWalk.get_obs, got {len(found)}"
    return found[0]


def _obs_concat_elements() -> list[str]:
    """Unparsed elements of get_obs()'s np.concatenate list, in source order."""
    concats = [
        node
        for node in ast.walk(_get_obs_fn())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "concatenate"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.args
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ]
    assert len(concats) == 1, (
        f"expected exactly one np.concatenate([...]) in get_obs, found {len(concats)}. "
        "If get_obs was split, this guard needs updating deliberately."
    )
    return [ast.unparse(el) for el in concats[0].args[0].elts]


def _self_assignments(node: ast.AST) -> dict[str, ast.expr]:
    """Map ``self.x = <expr>`` to <expr>, last assignment winning."""
    out: dict[str, ast.expr] = {}
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        for target in sub.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                out[target.attr] = sub.value
    return out


def _seq_len(expr: ast.expr) -> int | None:
    """Length of a literal list/tuple, unwrapping a single-arg np.array(...) call."""
    if isinstance(expr, ast.Call) and expr.args:
        expr = expr.args[0]
    if isinstance(expr, (ast.List, ast.Tuple)):
        return len(expr.elts)
    return None


def _static_widths() -> dict[str, int]:
    """Recover every term's real width from the source that produces it.

    Every one of the eleven is statically determinable, which is what makes the width
    guard possible without a robot:
      num_dofs / last_commands / imitation_phase  -> v2_rl_walk_mujoco.py
      gyro / accelero                             -> raw_imu.py's default dict
      feet_contacts                               -> feet_contacts.py's return
    """
    cls = _rlwalk_class()
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assigns = _self_assignments(init)

    num_dofs_node = assigns.get("num_dofs")
    assert isinstance(num_dofs_node, ast.Constant), "RLWalk.num_dofs is not a literal"
    num_dofs = int(num_dofs_node.value)

    commands_len = _seq_len(assigns["last_commands"])
    assert commands_len is not None, "self.last_commands is not a literal sequence"

    phase_len = _seq_len(assigns["imitation_phase"])
    assert phase_len is not None, "self.imitation_phase is not a literal sequence"

    # raw_imu's fallback dict is the declared shape of a reading.
    imu_tree = ast.parse(RAW_IMU.read_text())
    imu_widths: dict[str, int] = {}
    for node in ast.walk(imu_tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value in ("gyro", "accelero"):
                n = _seq_len(value)
                if n is not None:
                    imu_widths.setdefault(key.value, n)
    for k in ("gyro", "accelero"):
        assert k in imu_widths, f"could not recover {k!r} width from {RAW_IMU}"

    # feet_contacts.get() -> [left, right]
    feet_tree = ast.parse(FEET.read_text())
    feet_get = next(
        n
        for n in ast.walk(feet_tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "get"
    )
    feet_len = next(
        (
            _seq_len(r.value)
            for r in ast.walk(feet_get)
            if isinstance(r, ast.Return) and r.value is not None
        ),
        None,
    )
    assert feet_len is not None, f"could not recover feet width from {FEET}"

    widths = {name: num_dofs for name in DOF_WIDTH_TERMS}
    widths["commands"] = commands_len
    widths["imitation_phase"] = phase_len
    widths["gyro"] = imu_widths["gyro"]
    widths["accelerometer"] = imu_widths["accelero"]
    widths["feet_contacts"] = feet_len
    return widths


# ── the drift guards ────────────────────────────────────────────────────────────


def test_obs_term_order() -> None:
    """CRITICAL. The loop's observation ORDER must equal the declared order.

    Read the failure carefully before changing anything: a *reordering* is serious, a
    *rewording* is not, and the message below tells you which you are looking at.
    """
    actual = _obs_concat_elements()
    expected = [TERM_TO_SOURCE[name] for name, _ in OBS_TERMS]
    if actual == expected:
        return

    if sorted(actual) == sorted(expected):
        pytest.fail(
            "the walk loop's observation TERMS WERE REORDERED.\n"
            f"  loop     : {actual}\n"
            f"  contract : {expected}\n"
            "Every policy ever trained for this robot expects the contract order, so this "
            "is a breaking change. If it is intentional, bump OBS_VERSION and update "
            "OBS_TERMS, and accept that existing policies are now incompatible."
        )

    pytest.fail(
        "the walk loop's observation expressions no longer MATCH THE TEXT this guard "
        "compares against.\n"
        f"  loop     : {actual}\n"
        f"  contract : {expected}\n"
        "If the terms are the same quantities written differently (a local renamed, "
        "operands swapped in a product), this is harmless: update TERM_TO_SOURCE in this "
        "file only. Do NOT bump OBS_VERSION -- that would invalidate every trained policy "
        "over a rename. If a term was genuinely added, removed or replaced, update "
        "OBS_TERMS and bump OBS_VERSION."
    )


def test_obs_term_widths() -> None:
    """CRITICAL. Each term's real width must equal its declared width.

    This is the half an earlier version of this file missed. A review showed that making
    ``last_commands`` eight long left the vector at 102 floats while OBS_DIM stayed 101,
    with every test green -- surfacing on hardware as an onnxruntime shape error at the
    first tick, and in later stories as a manifest that is silently wrong.
    """
    static = _static_widths()
    declared = dict(OBS_TERMS)

    mismatches = {
        name: (declared[name], static[name])
        for name in declared
        if name in static and declared[name] != static[name]
    }
    assert not mismatches, (
        "declared term widths no longer match the source:\n"
        + "\n".join(
            f"  {name}: contract says {d}, source says {s}"
            for name, (d, s) in sorted(mismatches.items())
        )
        + "\nThe real observation vector is now "
        + str(sum(static.get(n, w) for n, w in OBS_TERMS))
        + f" floats but OBS_DIM is {OBS_DIM}. Update OBS_TERMS and bump OBS_VERSION: "
        "every existing policy expects the old width."
    )

    uncovered = set(declared) - set(static)
    assert not uncovered, f"no static width recovered for: {sorted(uncovered)}"


def test_real_obs_dim_matches_declared() -> None:
    """The sum of the source's own widths equals OBS_DIM."""
    static = _static_widths()
    assert sum(static[name] for name, _ in OBS_TERMS) == OBS_DIM


def test_act_dim_matches_num_dofs() -> None:
    """The policy's output width is the loop's num_dofs -- it writes one target per dof."""
    assert _static_widths()["motor_targets"] == ACT_DIM


def test_action_history_rotates_oldest_first() -> None:
    """The three action-history slots must be assigned oldest-first.

    ``last_last_last = last_last`` then ``last_last = last`` then ``last = action``.
    Swapping any two makes two slots hold the same tick and the policy gets a corrupted
    history -- with the concatenate text and every width unchanged, so neither guard above
    can see it. Called out by the review as load-bearing enough for its own assertion.
    """
    cls = _rlwalk_class()
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")

    order: list[tuple[str, str]] = []
    for node in ast.walk(run):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr
            in ("last_action", "last_last_action", "last_last_last_action")
        ):
            continue
        order.append((target.attr, ast.unparse(node.value)))

    assert order == [
        ("last_last_last_action", "self.last_last_action.copy()"),
        ("last_last_action", "self.last_action.copy()"),
        ("last_action", "action.copy()"),
    ], (
        "the action-history rotation in RLWalk.run changed:\n"
        f"  {order}\n"
        "It must shift oldest-first, or two history slots hold the same tick and every "
        "policy sees a corrupted action history -- silently, with the right widths."
    )


def test_every_declared_term_has_a_source_expression() -> None:
    """Guards the guard: a term added to OBS_TERMS with no mapping would otherwise raise
    KeyError instead of failing usefully."""
    missing = [name for name, _ in OBS_TERMS if name not in TERM_TO_SOURCE]
    assert not missing, f"OBS_TERMS entries with no source expression: {missing}"
    extra = set(TERM_TO_SOURCE) - {name for name, _ in OBS_TERMS}
    assert not extra, f"source expressions for terms not in OBS_TERMS: {sorted(extra)}"


def test_loop_term_count_matches() -> None:
    assert len(_obs_concat_elements()) == len(OBS_TERMS)


# ── the contract's own invariants ───────────────────────────────────────────────


def test_obs_dim_is_derived_not_asserted() -> None:
    assert OBS_DIM == sum(width for _, width in OBS_TERMS)
    assert OBS_DIM == 101


def test_action_and_rate() -> None:
    assert ACT_DIM == 14
    assert CONTROL_HZ == 50


def test_version_and_input_name() -> None:
    assert OBS_VERSION == "duck-obs-v1"
    assert OBS_INPUT_NAME == "obs"


def test_term_offsets_tile_the_vector_exactly() -> None:
    offsets = term_offsets()
    assert len(offsets) == len(OBS_TERMS)
    cursor = 0
    for name, width in OBS_TERMS:
        start, stop = offsets[name]
        assert start == cursor, f"{name} starts at {start}, expected {cursor}"
        assert stop - start == width
        cursor = stop
    assert cursor == OBS_DIM, "offsets leave a gap or overrun the vector"


def test_term_names_are_unique() -> None:
    names = [name for name, _ in OBS_TERMS]
    assert len(names) == len(set(names))


def test_widths_are_positive() -> None:
    assert all(width > 0 for _, width in OBS_TERMS)


# The board in the duck: Raspberry Pi Zero 2W, 512 MB total, already running the OS and
# tnkr_server. Named here because the ceiling below is only meaningful relative to it.
PI_RAM_BYTES = 512 * 1024**2
# scripts/BEST_WALK_ONNX_2.onnx, the policy every duck ships with, to the byte.
BUILTIN_POLICY_BYTES = 884177
# onnxruntime holds the protobuf plus its initializers while parsing, so peak RSS during
# inspection is a small multiple of the file. Three is the pessimistic end of that.
PARSE_MEMORY_FACTOR = 3


def test_the_size_ceiling_is_small_enough_to_be_a_ceiling() -> None:
    """An earlier version of this test asserted only ``32 MB <= x <= 512 MB``, and 512 MB
    is the whole board -- so a 256 MB ceiling passed it while letting a 200 MB file reach
    onnxruntime's parser, which is the exact memory exhaustion the ceiling exists to stop
    (story 2.2: "a large upload cannot exhaust a Pi's memory during inspection"). The
    bound has to be stated against the Pi's RAM or it states nothing.
    """
    assert MAX_POLICY_BYTES * PARSE_MEMORY_FACTOR <= PI_RAM_BYTES // 4, (
        f"a file at the {MAX_POLICY_BYTES / 1024**2:.0f} MB ceiling could cost "
        f"{MAX_POLICY_BYTES * PARSE_MEMORY_FACTOR / 1024**2:.0f} MB to parse on a "
        f"{PI_RAM_BYTES / 1024**2:.0f} MB board that is also running tnkr_server, so the "
        "install would be refused by the OOM killer rather than by a typed refusal"
    )


def test_the_size_ceiling_accepts_every_policy_that_exists() -> None:
    """The other half: a ceiling tight enough to refuse a real policy is a bug report, not
    a safety feature. Ten times the built-in is the margin, and the built-in is the largest
    Open Duck ONNX anyone has published."""
    assert MAX_POLICY_BYTES >= 10 * BUILTIN_POLICY_BYTES


def test_describe_is_one_line() -> None:
    line = describe()
    assert "\n" not in line
    assert OBS_VERSION in line and str(OBS_DIM) in line


# ── validate_obs ────────────────────────────────────────────────────────────────


def test_validate_obs_accepts_the_right_length() -> None:
    validate_obs([0.0] * OBS_DIM)


def test_validate_obs_names_the_term_that_overflows() -> None:
    with pytest.raises(ValueError) as exc:
        validate_obs([0.0] * (OBS_DIM + 1))
    assert str(OBS_DIM) in str(exc.value) and str(OBS_DIM + 1) in str(exc.value)


def test_validate_obs_rejects_short_vectors() -> None:
    with pytest.raises(ValueError):
        validate_obs([0.0] * (OBS_DIM - 1))


def test_validate_obs_rejects_none() -> None:
    with pytest.raises(ValueError):
        validate_obs(None)


# ── dependency boundary ─────────────────────────────────────────────────────────


def _module_level_imports() -> list[str]:
    """Names imported at module scope (including inside a top-level ``if``)."""
    source_path = inspect.getsourcefile(policy_contract)
    assert source_path, "could not locate policy_contract's source"
    tree = ast.parse(Path(source_path).read_text())

    imported: list[str] = []
    for node in tree.body:
        nodes = [node]
        if isinstance(node, (ast.If, ast.Try)):
            nodes = list(ast.walk(node))
        for sub in nodes:
            if isinstance(sub, ast.Import):
                imported += [a.name for a in sub.names]
            elif isinstance(sub, ast.ImportFrom) and sub.module:
                imported.append(sub.module)
    return imported


FORBIDDEN_AT_MODULE_SCOPE = (
    "mini_bdx_runtime",
    "numpy",
    "onnxruntime",
    "rustypot",
    "adafruit",
    "board",
    "busio",
    "digitalio",
    "serial",
)


def test_module_pulls_in_no_hardware_or_runtime_deps() -> None:
    """policy_contract must be IMPORTABLE by tooling on a workstation.

    Studio-side tooling and CI need the constants with no robot attached, so nothing here
    may be needed *to import the module*. A denylist, not an allowlist: adding a stdlib
    ``hashlib`` import is fine and should not turn this red with a message that reads like
    a hardware violation.

    Narrowed in story 2.2 from "no import of these anywhere in the file" to "no import of
    these at module scope". ``check_policy`` has to construct an onnxruntime session -- it
    is the boundary that reads a candidate graph's shape -- and it imports the wheel inside
    the function, which is what keeps the import-time guarantee above intact. The next test
    proves that rather than trusting it.
    """
    offenders = [
        m
        for m in _module_level_imports()
        if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_AT_MODULE_SCOPE)
    ]
    assert not offenders, (
        f"policy_contract imports {offenders} at module scope. It declares constants and "
        "one file check; it must stay importable on a machine with no robot and no native "
        "wheels. If a function needs onnxruntime, import it inside that function."
    )


def test_importable_with_onnxruntime_absent(monkeypatch) -> None:
    """Load the module from source with the native wheels unimportable.

    The AST test above can only see import *statements*; this executes the module for real
    with ``onnxruntime`` and ``numpy`` made to fail, which is the situation on a
    workstation or in a CI job that only wants the constants.
    """
    blocked = ("onnxruntime", "numpy")

    class Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked:
                raise ImportError(f"blocked for test: {fullname}")
            return None

    for name in list(sys.modules):
        if name.split(".")[0] in blocked:
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])

    source_path = inspect.getsourcefile(policy_contract)
    assert source_path
    spec = importlib.util.spec_from_file_location("policy_contract_isolated", source_path)
    assert spec and spec.loader
    isolated = importlib.util.module_from_spec(spec)
    # dataclasses resolves a string annotation through sys.modules[cls.__module__], so the
    # module has to be registered before it executes. monkeypatch removes it after.
    monkeypatch.setitem(sys.modules, spec.name, isolated)
    spec.loader.exec_module(isolated)  # raises ImportError if a wheel is needed to import

    assert isolated.OBS_DIM == OBS_DIM
    assert isolated.MAX_POLICY_BYTES == MAX_POLICY_BYTES
