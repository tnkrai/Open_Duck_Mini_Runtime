"""The duck's component catalogue: which policy runs, and whether it fits the loop.

Phase 1b of tnkr-studio's docs/designs/wired-physical-agents-plan.md.

WHAT THIS REPLACES. `SCRIPTS_DIR.glob("*.onnx")` and take `[0]`. That is not wrong
today — one `.onnx` exists, so first-match is the right match — and it becomes
arbitrary the moment a second one does, which is exactly what Phase 6 requires.
Underneath the file-selection problem is the real one: nothing checked that the model
matched the loop feeding it. A policy trained against a different observation order
loads happily and walks the duck into the floor.

THE CHECKS RUN EVERYWHERE; ONLY THE SPAWN IS PLATFORM-SPECIFIC. Everything here is
deliberately importable with numpy and pydantic alone. The duck's CI has no
onnxruntime and no mujoco and never will — it cannot test inference, and does not need
to. It CAN test resolution, hashing and the contract check, which is the whole reason
those three moved above `if is_pi:` in tnkr_server.py. Checks that only run on the
robot are checks that are never run before the robot.

WHY A HASH AT ALL, WHEN THE PATH IS RIGHT THERE. Two reasons, and neither is
tamper-proofing: this is a robot on a home network, not a threat model. First, a
truncated upload is the realistic failure — a Pi Zero 2 W losing wifi mid-transfer
leaves a file that is present, readable, and wrong. Second, the manifest asserts a
contract about specific bytes; if the bytes are not those bytes, the contract it
carries describes something else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from mini_bdx_runtime.obs_spec import ObsBlock

# Read in chunks rather than whole. 884K is nothing, and a camera-conditioned policy
# is not: tnkr-robot.service.template caps this process at MemoryMax=384M on a 512MB
# Pi Zero 2 W, and that device is documented to fail SILENTLY when it runs out.
HASH_CHUNK_BYTES = 1024 * 1024


class ComponentError(Exception):
    """A component cannot be used, with a stable machine code.

    The codes match tnkr-studio's ErrorCode names exactly, because Studio is what
    renders these to an operator and it maps codes to sentences. A code invented here
    that Studio does not know produces a generic fallback on screen.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ComponentManifest:
    """The identity, contract and provenance of one swappable component.

    Mirrors tnkr-studio's ComponentManifest (server/trlc_studio/components/manifest.py).
    Kept as a plain dataclass rather than a pydantic model so this module stays
    importable in the narrowest environment the CI installs.
    """

    id: str
    version: str
    hash: str
    kind: str
    embodiment: str
    obs_spec: tuple[ObsBlock, ...]
    rate_hz: float
    assumes: dict[str, str] = field(default_factory=dict)
    provenance: str = ""

    @property
    def obs_size(self) -> int:
        return sum(b.size for b in self.obs_spec)

    @classmethod
    def from_dict(cls, data: dict) -> "ComponentManifest":
        try:
            spec = tuple(
                ObsBlock(name=b["name"], shape=tuple(b["shape"])) for b in data["obsSpec"]
            )
            manifest = cls(
                id=data["id"],
                version=data["version"],
                hash=str(data["hash"]).strip().lower(),
                kind=data["kind"],
                embodiment=data["embodiment"],
                obs_spec=spec,
                rate_hz=float(data["rateHz"]),
                assumes=dict(data.get("assumes") or {}),
                provenance=str(data.get("provenance") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ComponentError(
                "COMPONENT_MANIFEST_INVALID", f"{type(exc).__name__}: {exc}"
            ) from exc
        if len(manifest.hash) != 64 or any(c not in "0123456789abcdef" for c in manifest.hash):
            raise ComponentError(
                "COMPONENT_MANIFEST_INVALID",
                "hash must be 64 lowercase hex characters (sha256)",
            )
        if not manifest.obs_spec:
            # An empty obs_spec would make the contract check below vacuously pass
            # against any loop at all, which is worse than having no check.
            raise ComponentError("COMPONENT_MANIFEST_INVALID", "obsSpec is empty")
        return manifest

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "hash": self.hash,
            "kind": self.kind,
            "embodiment": self.embodiment,
            "obsSpec": [{"name": b.name, "shape": list(b.shape)} for b in self.obs_spec],
            "rateHz": self.rate_hz,
            "assumes": dict(self.assumes),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Component:
    manifest: ComponentManifest
    artifact_path: Path


def catalogue_dir(root: Path | None = None) -> Path:
    """Where installed components live. Overridable so tests never touch a real home."""
    return (root or Path.home()) / ".tnkr" / "components"


def file_sha256(path: Path, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    """Incremental sha256. Never holds the artifact in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _builtin_entries(builtin_dir: Path) -> dict[str, Path]:
    """The policies shipped with the runtime, keyed by id.

    A flat directory of `<name>.onnx` beside `<name>.manifest.json`. It is scanned
    rather than indexed, but the scan is over MANIFESTS and the match is on the id
    inside one — which is the difference that matters. The old code globbed artifacts
    and took `[0]`, so adding a second file silently changed which policy ran; here a
    second file is simply a second addressable id, and an id that appears twice is
    refused rather than resolved to whichever sorted first.
    """
    found: dict[str, Path] = {}
    unreadable: list[str] = []
    for manifest_path in sorted(builtin_dir.glob("*.manifest.json")):
        try:
            data = json.loads(manifest_path.read_text())
            component_id = str(data["id"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            # A malformed shipped manifest must not hide the good ones, but it must not
            # vanish either: without this an unreadable manifest reports "not
            # installed", which sends someone looking for a missing file rather than a
            # broken one they are staring at.
            unreadable.append(manifest_path.name)
            continue
        if component_id in found:
            raise ComponentError(
                "COMPONENT_MANIFEST_INVALID",
                f"two shipped manifests both claim id {component_id!r}",
            )
        found[component_id] = manifest_path
    if unreadable:
        found["__unreadable__"] = Path("|".join(unreadable))
    return found


def resolve_builtin(component_id: str, builtin_dir: Path) -> Component:
    """One of the policies shipped in the repo, by id."""
    entries = _builtin_entries(builtin_dir)
    unreadable = entries.pop("__unreadable__", None)
    manifest_path = entries.get(component_id)
    if manifest_path is None:
        if unreadable is not None:
            raise ComponentError(
                "COMPONENT_MANIFEST_INVALID",
                f"no component {component_id!r}, and these shipped manifests could not "
                f"be read: {unreadable.name}",
            )
        raise ComponentError(
            "COMPONENT_NOT_FOUND", f"no component {component_id!r} installed or shipped"
        )
    try:
        manifest = ComponentManifest.from_dict(json.loads(manifest_path.read_text()))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComponentError("COMPONENT_MANIFEST_INVALID", f"{manifest_path}: {exc}") from exc

    artifact = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".onnx"))
    if not artifact.is_file():
        raise ComponentError(
            "COMPONENT_NOT_FOUND", f"{component_id!r} has a shipped manifest but no artifact"
        )
    return Component(manifest=manifest, artifact_path=artifact)


def resolve(component_id: str, *, root: Path | None = None) -> Component:
    """Find one component by id. Never by glob, never by first match.

    Raises rather than falling back to whatever else is on disk. "The id you asked for
    is not installed" and "here is a different policy" are answers a robot must not
    confuse, and the fallback is the one that walks.
    """
    if not component_id or "/" in component_id or component_id.startswith("."):
        # Ids reach this from an HTTP body and end up as a path segment. Rejecting the
        # shape is cheaper than sanitising it, and there is no legitimate id with a
        # slash in it.
        raise ComponentError("COMPONENT_NOT_FOUND", f"invalid component id {component_id!r}")

    base = catalogue_dir(root) / component_id
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise ComponentError("COMPONENT_NOT_FOUND", f"no component {component_id!r} installed")

    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ComponentError(
            "COMPONENT_MANIFEST_INVALID", f"{manifest_path}: {exc}"
        ) from exc

    manifest = ComponentManifest.from_dict(data)
    if manifest.id != component_id:
        # The directory name and the manifest disagree, so one of them is a lie and
        # there is no way to tell which. Refuse rather than pick.
        raise ComponentError(
            "COMPONENT_MANIFEST_INVALID",
            f"manifest says id={manifest.id!r} but it is installed as {component_id!r}",
        )

    artifact = base / "artifact.onnx"
    if not artifact.is_file():
        raise ComponentError(
            "COMPONENT_NOT_FOUND", f"{component_id!r} has a manifest but no artifact"
        )
    return Component(manifest=manifest, artifact_path=artifact)


def verify_hash(component: Component) -> None:
    """The bytes on disk are the bytes the manifest describes, or refuse."""
    actual = file_sha256(component.artifact_path)
    if actual != component.manifest.hash:
        raise ComponentError(
            "COMPONENT_HASH_MISMATCH",
            f"{component.manifest.id}: artifact is {actual}, manifest says "
            f"{component.manifest.hash}",
        )


def check_contract(manifest: ComponentManifest, loop_spec: tuple[ObsBlock, ...]) -> None:
    """The component's observation contract matches the loop about to feed it.

    Block by block, name AND shape, IN ORDER. Not by total width: two specs can agree
    on 101 and disagree on everything that matters, and a reordering is the failure
    that a width check waves through and a duck discovers with its face.
    """
    if len(manifest.obs_spec) != len(loop_spec):
        raise ComponentError(
            "COMPONENT_CONTRACT_MISMATCH",
            f"{manifest.id}: component declares {len(manifest.obs_spec)} observation "
            f"blocks, the loop provides {len(loop_spec)}",
        )
    for i, (want, have) in enumerate(zip(manifest.obs_spec, loop_spec)):
        if want.name != have.name or tuple(want.shape) != tuple(have.shape):
            raise ComponentError(
                "COMPONENT_CONTRACT_MISMATCH",
                f"{manifest.id}: block {i} is {want.name}{tuple(want.shape)} in the "
                f"component and {have.name}{tuple(have.shape)} in the loop",
            )


def check_assumes(manifest: ComponentManifest, *, runtime_version: str) -> None:
    """Every `assumes` entry this runtime knows how to check, checked.

    Today that is the runtime version floor. An entry with a key we do not recognise is
    IGNORED rather than refused: a component built for a newer runtime may declare
    things this one has never heard of, and refusing on that would make every forward-
    looking manifest unusable. An entry we DO recognise and cannot parse is a different
    matter and refuses, because a floor nobody can read is a floor not being enforced.
    """
    from mini_bdx_runtime.runtime_version import satisfies

    floor = manifest.assumes.get("runtime")
    if not floor:
        return
    try:
        ok = satisfies(str(floor), runtime_version)
    except ValueError as exc:
        raise ComponentError(
            "COMPONENT_ASSUMPTION_UNMET",
            f"{manifest.id}: cannot read its runtime requirement {floor!r}: {exc}",
        ) from exc
    if not ok:
        raise ComponentError(
            "COMPONENT_ASSUMPTION_UNMET",
            f"{manifest.id} needs runtime {floor}, this runtime is {runtime_version}",
        )


def check_embodiment(manifest: ComponentManifest, embodiment: str) -> None:
    """This policy was built for this robot. A DK1 policy on a duck is not a contract
    mismatch to be explained by block names — it is the wrong robot entirely."""
    if manifest.embodiment != embodiment:
        raise ComponentError(
            "COMPONENT_EMBODIMENT_MISMATCH",
            f"{manifest.id} is for {manifest.embodiment!r}, this robot is {embodiment!r}",
        )


def prepare(
    component_id: str,
    *,
    embodiment: str,
    loop_spec: tuple[ObsBlock, ...],
    root: Path | None = None,
    builtin_dir: Path | None = None,
) -> Component:
    """Resolve, verify and contract-check, in that order. The whole gate, one call.

    Order is not arbitrary. Embodiment first because "wrong robot" is the clearest
    thing to say and needs no byte-reading; then the hash, because a contract carried
    by unexpected bytes describes something other than what is on disk; then the
    contract itself. Any failure raises before the caller reaches its platform gate,
    so the refusal is identical on a Pi and on a laptop.
    """
    try:
        component = resolve(component_id, root=root)
    except ComponentError as exc:
        # Installed first, shipped second, and only for NOT_FOUND. An installed
        # component that is present but broken must surface its own failure rather than
        # being papered over by the shipped one — silently running a different policy
        # than the operator installed is the exact behaviour this phase removes.
        if exc.code != "COMPONENT_NOT_FOUND" or builtin_dir is None:
            raise
        component = resolve_builtin(component_id, builtin_dir)
    check_embodiment(component.manifest, embodiment)
    verify_hash(component)
    check_contract(component.manifest, loop_spec)
    return component


# ---------------------------------------------------------------------------
# Staging: where a component lives between arriving and being trusted
# ---------------------------------------------------------------------------
# Decision 6 and Decision 14. An upload is not an installation. Bytes arrive, get
# hashed, and sit in staging until something has proved they LOAD — because the unit
# has Restart=on-failure with StartLimitBurst=5, so a component that crashes the server
# on load burns five restarts and then systemd refuses to start the unit at all. At
# that point the thing you would use to roll back is the thing that will not start.
#
# So staging is not tidiness. It is the property that an unloadable component can never
# become the persisted active choice.


def staging_dir(root: Path | None = None) -> Path:
    return catalogue_dir(root) / ".staging"


def stage_manifest(component_id: str, data: dict, *, root: Path | None = None) -> ComponentManifest:
    """Park a manifest for a component whose bytes have not arrived yet.

    Parsed and validated NOW rather than at activation. A malformed manifest should be
    refused before an operator spends a Pi Zero's wifi on an 800K upload that was never
    going to be accepted.
    """
    if not component_id or "/" in component_id or component_id.startswith("."):
        raise ComponentError("COMPONENT_NOT_FOUND", f"invalid component id {component_id!r}")
    manifest = ComponentManifest.from_dict(data)
    if manifest.id != component_id:
        raise ComponentError(
            "COMPONENT_MANIFEST_INVALID",
            f"manifest says id={manifest.id!r} but it is being installed as {component_id!r}",
        )
    base = staging_dir(root) / component_id
    base.mkdir(parents=True, exist_ok=True)
    # Re-uploading is the ONLY way to clear a failed validation. Activation refuses a
    # known-bad component outright rather than retrying it, so without this an operator
    # who fixed and re-uploaded a policy could never activate it.
    clear_invalid(component_id, root=root)
    # A new manifest invalidates any half-uploaded artifact staged against the old one.
    # Leaving it would let a hash from manifest A be checked against bytes uploaded for
    # manifest B, which is the one combination that could pass while being wrong.
    artifact = base / "artifact.onnx"
    if artifact.exists():
        artifact.unlink()
    (base / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    return manifest


def staged_manifest(component_id: str, *, root: Path | None = None) -> ComponentManifest:
    path = staging_dir(root) / component_id / "manifest.json"
    if not path.is_file():
        raise ComponentError(
            "COMPONENT_NOT_FOUND", f"nothing staged for {component_id!r}; send its manifest first"
        )
    try:
        return ComponentManifest.from_dict(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComponentError("COMPONENT_MANIFEST_INVALID", f"{path}: {exc}") from exc


class ArtifactWriter:
    """Writes an artifact to staging one chunk at a time, hashing as it goes.

    Exists so the async HTTP path and the sync helper share ONE implementation of
    "write, hash, verify, delete on mismatch". The first version of the endpoint
    collected `request.stream()` into a list and handed it to a synchronous writer,
    which reads like streaming and is not: the list held the entire artifact, so the
    memory shape was identical to taking a `bytes` body. This class is what makes the
    async path genuinely incremental — a chunk is written and forgotten before the next
    one arrives.
    """

    def __init__(self, component_id: str, *, root: Path | None = None) -> None:
        self.manifest = staged_manifest(component_id, root=root)
        self.component_id = component_id
        base = staging_dir(root) / component_id
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / "artifact.onnx"
        self._digest = hashlib.sha256()
        self._written = 0
        self._fh = self.path.open("wb")

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        try:
            self._digest.update(chunk)
            self._fh.write(chunk)
            self._written += len(chunk)
        except OSError as exc:
            self.abort()
            raise ComponentError("COMPONENT_WRITE_FAILED", f"{self.path}: {exc}") from exc

    def abort(self) -> None:
        """Close and remove. Bytes that failed their check are not a partial success to
        resume from; leaving them invites a later call to find a file and assume it was
        verified."""
        try:
            self._fh.close()
        except OSError:
            pass
        self.path.unlink(missing_ok=True)

    def finish(self) -> int:
        try:
            self._fh.close()
        except OSError as exc:
            self.path.unlink(missing_ok=True)
            raise ComponentError("COMPONENT_WRITE_FAILED", f"{self.path}: {exc}") from exc

        if self._written == 0:
            self.path.unlink(missing_ok=True)
            raise ComponentError("COMPONENT_HASH_MISMATCH", "no bytes were uploaded")

        actual = self._digest.hexdigest()
        if actual != self.manifest.hash:
            self.path.unlink(missing_ok=True)
            raise ComponentError(
                "COMPONENT_HASH_MISMATCH",
                f"{self.component_id}: uploaded {self._written} bytes hashing to "
                f"{actual}, manifest says {self.manifest.hash}",
            )
        return self._written


def stage_artifact(component_id: str, chunks, *, root: Path | None = None) -> int:
    """Write an artifact to staging from an iterable of chunks. Returns bytes written.

    NEVER holds the artifact in memory, and never holds it in memory "briefly" either:
    each chunk is written and hashed and dropped. tnkr-robot.service.template caps this
    process at MemoryMax=384M on a 512MB Pi Zero 2 W, and that device is documented to
    fail SILENTLY when it runs out — an OOM kill mid-upload looks like the robot simply
    stopping. 884K is fine today and a camera-conditioned policy is not, which is the
    whole reason this is a stream rather than a `body: bytes`.

    A hash mismatch DELETES the staged artifact rather than leaving it. Bytes that
    failed their check are not a partial success to resume from; leaving them invites a
    later call to find a file and assume it was verified.
    """
    writer = ArtifactWriter(component_id, root=root)
    try:
        for chunk in chunks:
            writer.write(chunk)
    except ComponentError:
        raise
    except Exception:
        writer.abort()
        raise
    return writer.finish()


def staged_component(component_id: str, *, root: Path | None = None) -> Component:
    """What is staged, if both halves arrived."""
    manifest = staged_manifest(component_id, root=root)
    artifact = staging_dir(root) / component_id / "artifact.onnx"
    if not artifact.is_file():
        raise ComponentError(
            "COMPONENT_NOT_FOUND",
            f"{component_id!r} has a staged manifest but no artifact yet",
        )
    return Component(manifest=manifest, artifact_path=artifact)


def discard_staged(component_id: str, *, root: Path | None = None) -> None:
    """Throw away a staged component. Never touches what is installed."""
    import shutil as _shutil

    base = staging_dir(root) / component_id
    if base.is_dir():
        _shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------------------
# Activation: two-phase, crash-safe, and never retried
# ---------------------------------------------------------------------------
# Decision 6, and the reason it is two-phase rather than one.
#
# tnkr-robot.service.template has Restart=on-failure, StartLimitIntervalSec=300 and
# StartLimitBurst=5. A component that crashes the server on load gets five restarts and
# then systemd refuses to start the unit until someone runs `systemctl reset-failed`.
# At that point the thing you would use to roll back is the thing that will not start,
# and the duck is bricked until a person is physically at it.
#
# So an unloadable component must never become the persisted active choice. Validation
# happens in a SUBPROCESS — the cgroup applies there, so an OOM is contained — and only
# a clean exit promotes staging to active.
#
# AN OOM-KILLED VALIDATION IS `INVALID`, NEVER "transient, retry". This is the single
# most important line in this file. Retrying is how you reach the lockout: five
# attempts at a component that OOMs is exactly five restarts. A component that failed
# validation is recorded as failed and is never validated again without being
# re-uploaded.

ACTIVE_FILE = "active.json"
INVALID_FILE = "invalid.json"


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a power cut mid-write must not leave a truncated active.json,
    # because the file that says which policy to run is the file you cannot afford to
    # lose. A Pi loses power by being unplugged, which is the normal way it is turned off.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def active_state(root: Path | None = None) -> dict:
    """`{"active": id|None, "previous": id|None}`."""
    state = _read_json(catalogue_dir(root) / ACTIVE_FILE, {})
    return {
        "active": state.get("active"),
        "previous": state.get("previous"),
        "runtimeVersion": state.get("runtimeVersion"),
        "loopFingerprint": state.get("loopFingerprint"),
    }


def invalid_components(root: Path | None = None) -> dict:
    """Components that failed validation, and why. Never retried."""
    return _read_json(catalogue_dir(root) / INVALID_FILE, {})


def mark_invalid(component_id: str, reason: str, *, root: Path | None = None) -> None:
    record = invalid_components(root)
    record[component_id] = reason
    _write_json(catalogue_dir(root) / INVALID_FILE, record)


def clear_invalid(component_id: str, *, root: Path | None = None) -> None:
    """Only a fresh upload clears the mark — see stage_manifest."""
    record = invalid_components(root)
    if record.pop(component_id, None) is not None:
        _write_json(catalogue_dir(root) / INVALID_FILE, record)


def default_validator(component: Component) -> "subprocess.CompletedProcess":
    """Load the component in a child process and see whether it survives.

    A child, so an OOM kill lands on something disposable. The cgroup that caps this
    service applies to its children, so a policy too large to load fails HERE — bounded,
    reported, and with the running server untouched — rather than by taking the server
    down on the next start and spending the restart budget doing it.
    """
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", _VALIDATE_SNIPPET, str(component.artifact_path)],
        capture_output=True,
        timeout=120,
    )


# Runs in the child. Imports onnxruntime only here, which is why none of this module
# needs it: CI has no onnxruntime and tests substitute their own validator.
_VALIDATE_SNIPPET = (
    "import sys\n"
    "import onnxruntime\n"
    "onnxruntime.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])\n"
)


def activate(
    component_id: str,
    *,
    root: Path | None = None,
    embodiment: str,
    loop_spec: tuple[ObsBlock, ...],
    validator=None,
) -> dict:
    """Promote a staged component to active, if it loads. Returns the new state.

    Order: refuse anything already known bad, re-check the contract against the loop it
    is about to feed, validate in a subprocess, and only then move files and repoint
    active. Nothing before the final step changes what the robot runs, so every failure
    leaves the previous policy in place and running.
    """
    if component_id in invalid_components(root):
        # Never revalidated. Five attempts at a component that OOMs is five restarts,
        # and five restarts is the lockout.
        raise ComponentError(
            "COMPONENT_VALIDATION_FAILED",
            f"{component_id} already failed validation and will not be retried; "
            "upload it again to try a different build",
        )

    from mini_bdx_runtime.runtime_version import RUNTIME_VERSION

    component = staged_component(component_id, root=root)
    check_embodiment(component.manifest, embodiment)
    check_assumes(component.manifest, runtime_version=RUNTIME_VERSION)
    check_contract(component.manifest, loop_spec)
    verify_hash(component)

    result = (validator or default_validator)(component)
    returncode = getattr(result, "returncode", 1)
    if returncode != 0:
        # Negative means a signal. -9 is SIGKILL, which on this device is
        # overwhelmingly the OOM killer, and it is INVALID like any other failure —
        # "the machine was busy, try again" is the reasoning that reaches the lockout.
        how = f"killed by signal {-returncode}" if returncode < 0 else f"exit {returncode}"
        stderr = getattr(result, "stderr", b"") or b""
        detail = f"{component_id} failed to load ({how})"
        if stderr:
            detail += f": {stderr.decode('utf-8', 'replace')[:400]}"
        mark_invalid(component_id, detail, root=root)
        discard_staged(component_id, root=root)
        raise ComponentError("COMPONENT_VALIDATION_FAILED", detail)

    # It loads. Move it in, then repoint — in that order, so a crash between the two
    # leaves the old policy active and the new one merely present.
    import shutil as _shutil

    installed = catalogue_dir(root) / component_id
    if installed.exists():
        _shutil.rmtree(installed, ignore_errors=True)
    installed.parent.mkdir(parents=True, exist_ok=True)
    _shutil.move(str(staging_dir(root) / component_id), str(installed))

    state = active_state(root)
    previous = state["active"]
    from mini_bdx_runtime.runtime_version import RUNTIME_VERSION, loop_fingerprint

    _write_json(
        catalogue_dir(root) / ACTIVE_FILE,
        # The one being replaced becomes previous. A component that is re-activated
        # must not become its own previous, or rollback is a no-op exactly when it is
        # needed most.
        {
            "active": component_id,
            "previous": previous if previous != component_id else state["previous"],
            # What the loop looked like when this was proved to fit it. A runtime
            # upgrade changes these under an untouched component, which is the gap
            # check_still_fits closes.
            "runtimeVersion": RUNTIME_VERSION,
            "loopFingerprint": loop_fingerprint(),
        },
    )
    return active_state(root)


def rollback(*, root: Path | None = None) -> dict:
    """Make the previous component active again. The escape hatch, robot-side.

    Does NOT validate: previous was validated when it was activated and has been
    running. Re-validating here would mean the recovery path can fail for a new reason
    at the moment recovery is needed.
    """
    state = active_state(root)
    previous = state["previous"]
    if not previous:
        raise ComponentError(
            "COMPONENT_NO_PREVIOUS",
            "nothing to roll back to; only one component has ever been active",
        )
    if not (catalogue_dir(root) / previous / "manifest.json").is_file():
        raise ComponentError(
            "COMPONENT_NOT_FOUND",
            f"the previous component {previous!r} is no longer installed",
        )
    _write_json(
        catalogue_dir(root) / ACTIVE_FILE,
        {
            "active": previous,
            "previous": state["active"],
            # Carried, not re-stamped. Rolling back does not re-prove anything, so
            # claiming this runtime validated it would be a lie the next check trusts.
            "runtimeVersion": state.get("runtimeVersion"),
            "loopFingerprint": state.get("loopFingerprint"),
        },
    )
    return active_state(root)


def resolve_active(
    *, root: Path | None = None, fallback: str | None = None
) -> str:
    """Which component the robot should run. The active one, or a named fallback."""
    active = active_state(root)["active"]
    if active:
        return active
    if fallback:
        return fallback
    raise ComponentError("COMPONENT_NOT_FOUND", "no component is active and no default is set")


def check_still_fits(
    component: Component, *, root: Path | None = None, loop_spec: tuple[ObsBlock, ...]
) -> None:
    """Re-check a component against the loop it is about to feed, after an upgrade.

    Install-time checking is not enough, and this is the gap that only appears once you
    read the two repos together. `tnkr upgrade openduck-mini` versions the RUNTIME;
    Phase 1 versions the COMPONENTS inside it. The loop lives in the runtime, so an
    upgrade can change the observation contract under a component that was never
    touched — it still hashes correctly, still says the same thing about itself, and is
    now wrong. Nothing at install time can see that, because at install time it was
    right.

    Cheap by design: compare the recorded fingerprint against this runtime's. Equal
    means the loop is the one the component was proved against and there is nothing to
    do. Only a difference pays for the full re-check.
    """
    from mini_bdx_runtime.runtime_version import RUNTIME_VERSION, loop_fingerprint

    state = active_state(root)
    recorded = state.get("loopFingerprint")
    current = loop_fingerprint()
    if recorded == current:
        return

    # The loop moved. Everything below re-runs what activation ran, and a failure here
    # names the upgrade rather than the component, because the component did not change.
    try:
        check_assumes(component.manifest, runtime_version=RUNTIME_VERSION)
        check_contract(component.manifest, loop_spec)
    except ComponentError as exc:
        raise ComponentError(
            exc.code,
            f"{exc.detail} (the runtime changed under it: was "
            f"{state.get('runtimeVersion') or 'unrecorded'}, is now {RUNTIME_VERSION})",
        ) from exc

    # It still fits. Record the new loop so the next start is the cheap path again.
    _write_json(
        catalogue_dir(root) / ACTIVE_FILE,
        {
            "active": state["active"],
            "previous": state["previous"],
            "runtimeVersion": RUNTIME_VERSION,
            "loopFingerprint": current,
        },
    )
