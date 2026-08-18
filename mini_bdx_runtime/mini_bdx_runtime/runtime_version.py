"""What version of the runtime this is, and what shape its control loop has.

Phase 1b of tnkr-studio's docs/designs/wired-physical-agents-plan.md.

WHY THIS EXISTS. There are two version axes and they interact. `tnkr upgrade
openduck-mini` versions the RUNTIME; Phase 1 versions the COMPONENTS inside it. The
loop that feeds a policy lives in the runtime, so a runtime upgrade can silently change
the contract under an already-installed component — the component is untouched, still
hashes correctly, still says the same thing about itself, and is now wrong.

WHY NOT setup.cfg's VERSION. That file says 0.1.0 and belongs to the upstream
mini_BDX_runtime package this repo derives from. It does not move when tnkr changes the
observation loop, which is the only thing the contract cares about, so using it as the
floor would be a number that looks meaningful and is not.

WHY A FINGERPRINT AS WELL AS A VERSION. The plan asks for a runtime version floor, and
that is here. But a version is a PROXY for "did the loop change", and it is wrong in
both directions: someone can reorder a block without bumping the version, and a version
can bump for a change that never touches the loop. LOOP_FINGERPRINT is derived from the
ordered block names and shapes themselves, so it changes exactly when the contract
changes and never otherwise. The version is what a component DECLARES against; the
fingerprint is what actually gets compared.
"""

from __future__ import annotations

import hashlib

from mini_bdx_runtime.obs_spec import WALK_OBS_SPEC

# tnkr's runtime version, tracking the v2 line. Bump it when the runtime changes in a
# way an installed component could care about. It is deliberately not read from
# setup.cfg, which versions the upstream package rather than this loop.
RUNTIME_VERSION = "2.0.0"


def loop_fingerprint() -> str:
    """A short digest of the observation contract this runtime provides.

    Names and shapes, in order, which is exactly what check_contract compares. Two
    runtimes with the same fingerprint feed a policy identically; two with different
    fingerprints do not, whatever their version numbers say.
    """
    material = "|".join(f"{b.name}:{','.join(map(str, b.shape))}" for b in WALK_OBS_SPEC)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def parse_version(text: str) -> tuple[int, ...]:
    parts = text.strip().split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"not a version: {text!r}") from exc


def satisfies(requirement: str, version: str = RUNTIME_VERSION) -> bool:
    """Does `version` satisfy a requirement like ">=2.0.0"?

    Deliberately tiny: `>=`, `>`, `==` and a bare version meaning `>=`. Anything else
    raises rather than being interpreted generously — a floor nobody can parse is a
    floor that is not being enforced, and silently passing it is the worst outcome.
    """
    text = requirement.strip()
    for op in (">=", "==", ">"):
        if text.startswith(op):
            bound = parse_version(text[len(op):])
            actual = parse_version(version)
            if op == ">=":
                return actual >= bound
            if op == ">":
                return actual > bound
            return actual == bound
    return parse_version(version) >= parse_version(text)
