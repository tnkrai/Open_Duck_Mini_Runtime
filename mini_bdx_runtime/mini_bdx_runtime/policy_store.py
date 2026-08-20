"""The robot's policy store: install, list, select, evict, and always get back.

Layout on disk
--------------
::

    ~/.tnkr/policies/
    |-- active                     # pointer file: one id, written temp-then-renamed
    |-- builtin                    # NOT a directory. Resolved through scripts/*.onnx
    |                              # at read time; never copied, never evictable.
    `-- 9f2a.../
        |-- model.onnx             # the verified artifact
        |-- manifest.json          # what it claims + what we measured
        `-- last_used              # empty; its mtime is the LRU key

Why the built-in is a resolution and not a copy
-----------------------------------------------
Copying the bundled ONNX in would double its disk cost on a card that A3 exists to keep
from filling, and create a second thing to keep in sync with ``git pull``. Resolving it
through the same ``scripts/*.onnx`` glob ``/api/walk/start`` has always used means the
built-in tracks whatever the repo ships and cannot be evicted, corrupted by eviction
logic, or left stale by an update. Architecture Decision 11 promises every duck in the
field keeps walking exactly as it does now; the glob is how that promise is kept, so it
is deliberately preserved rather than replaced with a hardcoded filename.

Why the store is bounded (amendment A3)
---------------------------------------
Nothing in the original plan ever deleted a policy. An unbounded directory on a Pi's SD
card ends in a full card, and a full card is a duck that will not boot -- a worse failure
than walking badly, and unrelated to it. So: at most ``max_policies`` installed models,
least-recently-used evicted to make room, and a refusal before the download starts if the
free space would drop below a floor. The built-in and the policy a walk is currently
running are never eviction candidates.

Why reverting cannot be allowed to fail (amendment A4)
------------------------------------------------------
``select("builtin")`` is the E-stop of policy selection: the operator reaching for it is
holding a duck that just fell over. So it does not verify anything, does not read the
store index, and does not need the store to be in a good state. It is implemented as
*removing* the pointer file rather than writing one, because an absent or unparseable
pointer already resolves to the built-in -- the revert therefore needs no free space, no
temp file and no rename to succeed.

Install is transactional
------------------------
::

    disk floor -> stream to temp -> sha256 + check_policy -> measure latency
                                                                  |
                        evict LRU (only now) -> atomic move into place
                                    |
        on ANY failure: delete the temp, change nothing at all

Eviction happens *after* the download and the check, never before: a policy deleted to
make room for an install that then failed would be the worst possible reading of "bounded
store". The floor check is what makes room for the extra copy in the meantime.

Nothing here touches the servo bus, the IMU or the GPIO, so it is safe to run while a walk
is running -- which A4 requires, since the walk that needs reverting is the one that is
running.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from mini_bdx_runtime.policy_contract import (
    MAX_POLICY_BYTES,
    POLICY_CONTRACT_MISMATCH,
    POLICY_INSTALL_FAILED,
    POLICY_STORE_FULL,
    check_policy,
    measure_latency_at,
    normalise_digest,
    sha256_file,
)

BUILTIN_ID: str = "builtin"

MODEL_FILENAME: str = "model.onnx"
MANIFEST_FILENAME: str = "manifest.json"
ACTIVE_FILENAME: str = "active"
USED_FILENAME: str = "last_used"

# Three installed policies plus the built-in. Chosen from the failure it prevents rather
# than from taste: the ceiling in policy_contract is 16 MB, so a full store is at most
# ~48 MB of a card whose smallest supported size is 16 GB, and three is enough to hold the
# one you are trying, the one you were using, and the one you want to compare against.
DEFAULT_MAX_POLICIES: int = 3

# Never let an install take free space below this. 200 MB is not the size of anything --
# it is headroom for the OS: journald, /dev/shm, apt, and the walk's own logs all need
# room, and a Pi that cannot write is a Pi that will not boot.
DEFAULT_FREE_FLOOR_BYTES: int = 200 * 1024**2

# Read in chunks so a 16 MB model never becomes a 16 MB bytes object in RAM on a board
# with 512 MB of it.
_DOWNLOAD_CHUNK_BYTES: int = 256 * 1024

# Generous, because household wifi to S3 is the normal case and a slow one still works.
# Bounded, because the alternative is an install request that hangs a worker forever.
DEFAULT_TIMEOUT_S: float = 60.0

# An id becomes a directory name, and it arrives over an HTTP API with no authentication.
# Anything outside this set is refused rather than sanitised: a request naming
# "../../../etc" is not a typo to be helpful about.
_ID_ALLOWED: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)
_ID_MAX_LEN: int = 128


class StoreError(Exception):
    """A refusal with a wire code. ``detail`` is developer-facing, never operator copy."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code: str = code
        self.detail: str = detail


class DownloadFailed(Exception):
    """The bytes did not arrive. Message never contains the URL's query string."""


def redact_url(url: str) -> str:
    """``scheme://host/path`` -- the query string dropped.

    A presigned URL's query string *is* the credential, and this text goes into refusal
    details, print() on the robot's stdout, and PostHog via the telemetry middleware's
    ``error_message``. Naming the host and path is enough to tell "the robot has no route
    to the internet" from "that object is gone"; the signature is never anyone's business.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    if not parts.scheme and not parts.netloc:
        return "<no host>"
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def stream_to_file(
    url: str,
    dest: "os.PathLike[str] | str",
    *,
    max_bytes: int = MAX_POLICY_BYTES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    chunk_bytes: int = _DOWNLOAD_CHUNK_BYTES,
) -> int:
    """Stream ``url`` into ``dest``, refusing past ``max_bytes``. Returns bytes written.

    ``urllib`` rather than ``httpx`` or ``requests``: the runtime ships neither as a direct
    dependency, and a new dependency on the Pi for one GET is a worse trade than the
    stdlib's clumsier API.

    The ceiling is enforced *while streaming*, not from ``Content-Length``, because a
    header is a claim and this endpoint takes a URL from anyone who can reach the robot.
    Without it, one request could fill the SD card -- the exact failure A3 exists to
    prevent -- before any of the checks downstream ever ran.

    Only http(s) is accepted. ``file://`` would turn an unauthenticated install endpoint
    into "copy an arbitrary path on the robot into the store", which is not a policy
    install.
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise DownloadFailed(
            f"refusing a {scheme or 'schemeless'} URL; a policy is fetched over http(s)"
        )

    total = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "tnkr-duck"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            with open(dest, "wb") as out:
                while True:
                    chunk = response.read(chunk_bytes)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise DownloadFailed(
                            f"download from {redact_url(url)} passed the "
                            f"{max_bytes}-byte ceiling and was cut off"
                        )
                    out.write(chunk)
    except DownloadFailed:
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # URLError covers "no route to the internet", which some owners have permanently
        # (an isolated AP), and an expired presigned URL mid-download (failure mode F5).
        raise DownloadFailed(
            f"fetch from {redact_url(url)} failed after {total} bytes: "
            f"{exc.__class__.__name__}: {exc}"
        ) from exc
    return total


@dataclass
class InstallResult:
    """What an install did, or why it did nothing."""

    ok: bool
    code: str | None = None
    detail: str = ""
    id: str | None = None
    manifest: dict | None = None
    evicted: dict | None = None
    warning: dict | None = None
    already_installed: bool = False

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "code": self.code,
            "detail": self.detail,
            "id": self.id,
            "manifest": self.manifest,
            "evicted": self.evicted,
            "warning": self.warning,
            "alreadyInstalled": self.already_installed,
        }


@dataclass
class Resolved:
    """A policy the walk can be spawned on."""

    id: str
    path: Path
    is_builtin: bool


@dataclass
class _Entry:
    """One installed policy, as read off disk. Stats only -- never a hash."""

    id: str
    path: Path
    manifest: dict = field(default_factory=dict)
    size_bytes: int = 0
    installed_at: float = 0.0
    last_used_at: float = 0.0


def validate_id(policy_id: str) -> str:
    """The id, or ``StoreError`` if it cannot safely be a directory name."""
    candidate = (policy_id or "").strip()
    if not candidate:
        raise StoreError(POLICY_INSTALL_FAILED, "install request carried no policy id")
    if candidate == BUILTIN_ID:
        raise StoreError(
            POLICY_INSTALL_FAILED,
            f"{BUILTIN_ID!r} is the policy this repo ships and is resolved through "
            "scripts/*.onnx; it cannot be installed over",
        )
    if len(candidate) > _ID_MAX_LEN:
        raise StoreError(
            POLICY_INSTALL_FAILED,
            f"policy id is {len(candidate)} characters, over the {_ID_MAX_LEN} ceiling",
        )
    bad = sorted({c for c in candidate if c not in _ID_ALLOWED})
    if bad or candidate.startswith(".") or candidate in ("..",):
        raise StoreError(
            POLICY_INSTALL_FAILED,
            f"policy id {candidate!r} is not usable as a directory name "
            f"(rejected characters: {bad or ['leading dot']})",
        )
    return candidate


class PolicyStore:
    """The installed policies, the active one, and the way back to the built-in.

    Constructing this is free: no directory is created and no file is read until a method
    that needs one runs. That is deliberate -- the server builds one per request so a test
    (or a future config reload) can move ``SCRIPTS_DIR`` without a stale capture.
    """

    def __init__(
        self,
        root: "os.PathLike[str] | str",
        scripts_dir: "os.PathLike[str] | str",
        max_policies: int = DEFAULT_MAX_POLICIES,
        free_floor_bytes: int = DEFAULT_FREE_FLOOR_BYTES,
        *,
        max_policy_bytes: int = MAX_POLICY_BYTES,
        fetch: Callable[..., int] = stream_to_file,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.scripts_dir = Path(scripts_dir)
        self.max_policies = max_policies
        self.free_floor_bytes = free_floor_bytes
        self.max_policy_bytes = max_policy_bytes
        self.fetch = fetch
        self._now = now

    # ── the built-in ────────────────────────────────────────────────────────────

    def builtin_path(self) -> Path | None:
        """The bundled ONNX, resolved the way ``/api/walk/start`` has always resolved it.

        ``list(glob)[0]`` is preserved verbatim, including its unsorted, filesystem-order
        arbitrariness. Sorting would be tidier and would be a behaviour change: a duck with
        two .onnx files in ``scripts/`` could start walking on the other one after an
        update that touched nothing but this line.
        """
        try:
            candidates = list(self.scripts_dir.glob("*.onnx"))
        except OSError:
            return None
        if not candidates:
            return None
        return candidates[0]

    # ── the active pointer ──────────────────────────────────────────────────────

    @property
    def _active_file(self) -> Path:
        return self.root / ACTIVE_FILENAME

    def active_id(self) -> str:
        """The selected policy's id, or ``builtin``.

        Every way of failing to read the pointer -- absent, empty, unparseable, a name no
        directory answers to -- resolves to the built-in. A duck whose pointer file got
        half-written by a power cut is a duck that walks on the policy it shipped with,
        which is the only answer that cannot make things worse.
        """
        try:
            raw = self._active_file.read_text()
        except (FileNotFoundError, NotADirectoryError):
            return BUILTIN_ID
        except OSError as exc:
            print(f"[policy_store] cannot read the active pointer: {exc}")
            return BUILTIN_ID

        candidate = raw.strip()
        if not candidate or candidate == BUILTIN_ID:
            return BUILTIN_ID
        try:
            candidate = validate_id(candidate)
        except StoreError:
            print("[policy_store] active pointer holds an unusable id; using the built-in")
            return BUILTIN_ID
        if not (self.root / candidate / MODEL_FILENAME).is_file():
            print(
                f"[policy_store] active policy {candidate!r} is no longer installed; "
                "falling back to the built-in"
            )
            return BUILTIN_ID
        return candidate

    def _write_active(self, policy_id: str) -> None:
        """Temp-then-rename, so a power cut cannot leave an unparseable pointer."""
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.root / f".{ACTIVE_FILENAME}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            temp.write_text(f"{policy_id}\n")
            os.replace(temp, self._active_file)
        except OSError:
            temp.unlink(missing_ok=True)
            raise

    # ── reading the store ───────────────────────────────────────────────────────

    def _entries(self) -> list[_Entry]:
        """Installed policies, newest-used last. Stats and one small JSON read each."""
        entries: list[_Entry] = []
        try:
            children = sorted(p for p in self.root.iterdir() if p.is_dir())
        except (FileNotFoundError, NotADirectoryError):
            return entries
        except OSError as exc:
            print(f"[policy_store] cannot list the store: {exc}")
            return entries

        for child in children:
            if child.name.startswith("."):
                continue  # a staging or trash directory mid-install
            model = child / MODEL_FILENAME
            try:
                size = model.stat().st_size
            except OSError:
                continue  # a directory with no model in it is not an installed policy
            manifest = self._read_manifest(child)
            # A hand-edited or truncated manifest must not be able to break the listing --
            # /api/policy is what Studio polls, and a 500 there hides the whole store.
            try:
                installed_at = float(manifest.get("installed_at") or 0.0)
            except (TypeError, ValueError):
                installed_at = 0.0
            if not installed_at:
                try:
                    installed_at = model.stat().st_mtime
                except OSError:
                    installed_at = 0.0
            entries.append(
                _Entry(
                    id=child.name,
                    path=model,
                    manifest=manifest,
                    size_bytes=size,
                    installed_at=installed_at,
                    last_used_at=self._last_used_at(child, installed_at),
                )
            )
        entries.sort(key=lambda e: e.last_used_at)
        return entries

    def _read_manifest(self, policy_dir: Path) -> dict:
        try:
            loaded = json.loads((policy_dir / MANIFEST_FILENAME).read_text())
        except (OSError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _last_used_at(self, policy_dir: Path, fallback: float) -> float:
        """LRU key: the mtime of an empty marker file.

        A stamp inside ``manifest.json`` would mean rewriting the manifest on every select,
        which is a bigger write for a smaller guarantee. A marker's mtime survives a
        half-finished write, and ``GET /api/policy`` stays one stat per policy.
        """
        try:
            return (policy_dir / USED_FILENAME).stat().st_mtime
        except OSError:
            return fallback

    def mark_used(self, policy_id: str) -> None:
        """Record that a policy was selected or spawned. Best effort, never raises.

        Failing to stamp costs LRU accuracy. Failing a *walk start* because a stamp could
        not be written would be trading the robot's whole job for bookkeeping.
        """
        if policy_id == BUILTIN_ID:
            return
        try:
            (self.root / policy_id / USED_FILENAME).write_bytes(b"")
        except OSError as exc:
            print(f"[policy_store] could not stamp {policy_id} as used: {exc}")

    def list(self) -> dict:
        """``{"active": id, "policies": [...]}`` -- cheap enough to poll.

        The built-in is always first and always present, because it is a resolution rather
        than a stored file: there is no state in which the robot has no policy to fall back
        to (unless the repo itself ships no .onnx, which ``available`` reports).
        """
        active = self.active_id()
        builtin = self.builtin_path()
        policies: list[dict] = [
            {
                "id": BUILTIN_ID,
                "source": "bundled",
                "evictable": False,
                "available": builtin is not None,
                "path": str(builtin) if builtin else None,
                "sizeBytes": _size_of(builtin),
                "installedAt": None,
                "lastUsedAt": None,
                "manifest": None,
                "active": active == BUILTIN_ID,
            }
        ]
        for entry in reversed(self._entries()):  # most recently used first
            policies.append(
                {
                    "id": entry.id,
                    "source": "installed",
                    "evictable": entry.id != active,
                    "available": True,
                    "path": str(entry.path),
                    "sizeBytes": entry.size_bytes,
                    "installedAt": round(entry.installed_at, 3),
                    "lastUsedAt": round(entry.last_used_at, 3),
                    "manifest": entry.manifest or None,
                    "active": entry.id == active,
                }
            )
        return {
            "active": active,
            "maxPolicies": self.max_policies,
            "policies": policies,
        }

    # ── selecting ───────────────────────────────────────────────────────────────

    def select(self, policy_id: str, *, verify: bool = True) -> str:
        """Make ``policy_id`` the policy the next walk starts on. Returns the active id.

        ``builtin`` takes the first branch and touches nothing that can fail (A4). Any
        other id is re-verified against the same check that gated its install, because a
        file that passed on install can have been swapped on disk since -- re-reading it is
        the difference between having checked a file and checking *the* file.
        """
        if (policy_id or "").strip() in ("", BUILTIN_ID):
            return self.revert_to_builtin()

        candidate = validate_id(policy_id)
        model = self.root / candidate / MODEL_FILENAME
        if not model.is_file():
            raise StoreError(
                POLICY_INSTALL_FAILED,
                f"policy {candidate!r} is not installed on this robot",
            )

        if verify:
            manifest = self._read_manifest(self.root / candidate)
            expect = manifest.get("sha256")
            result = check_policy(
                model,
                expect_sha256=expect if isinstance(expect, str) else None,
                max_bytes=self.max_policy_bytes,
            )
            if not result.ok:
                raise StoreError(
                    result.code or POLICY_CONTRACT_MISMATCH,
                    f"{candidate} failed re-verification: {result.detail}",
                )

        self._write_active(candidate)
        self.mark_used(candidate)
        return candidate

    def revert_to_builtin(self) -> str:
        """Back to the policy the robot shipped with. The one action that must not fail.

        Implemented as a *removal*: an absent pointer already resolves to the built-in
        (see ``active_id``), so this needs no free space, no temp file and no rename, and
        works with the store empty, the active file corrupt, and a walk running. The write
        is only a fallback for a filesystem that will not unlink but will replace.
        """
        try:
            self._active_file.unlink()
            return BUILTIN_ID
        except FileNotFoundError:
            return BUILTIN_ID
        except OSError as exc:
            print(f"[policy_store] could not remove the active pointer: {exc}")

        try:
            self._write_active(BUILTIN_ID)
        except OSError as exc:
            raise StoreError(
                POLICY_INSTALL_FAILED,
                f"cannot revert to the built-in policy: {self.root} is not writable "
                f"({exc.__class__.__name__}: {exc})",
            ) from exc
        return BUILTIN_ID

    # ── resolving, for the walk ─────────────────────────────────────────────────

    def resolve_active(self) -> Resolved | None:
        """What the next walk runs with nobody naming a policy. ``None`` if there is none.

        A stale active pointer falls back to the built-in rather than refusing: the duck
        that lost its policy directory should still walk.
        """
        return self.resolve(self.active_id(), fall_back=True)

    def resolve(self, policy_id: str | None, *, fall_back: bool = False) -> Resolved | None:
        """Resolve one id to a path. Raises for a *named* policy that is not installed.

        ``fall_back`` is for the active pointer only. A caller that asked for a specific
        policy and silently got a different one has no way to notice, so an explicit
        request is refused instead.
        """
        wanted = (policy_id or BUILTIN_ID).strip() or BUILTIN_ID
        if wanted != BUILTIN_ID:
            candidate = validate_id(wanted)
            model = self.root / candidate / MODEL_FILENAME
            if model.is_file():
                return Resolved(id=candidate, path=model, is_builtin=False)
            if not fall_back:
                raise StoreError(
                    POLICY_INSTALL_FAILED,
                    f"policy {candidate!r} is not installed on this robot",
                )

        builtin = self.builtin_path()
        if builtin is None:
            return None
        return Resolved(id=BUILTIN_ID, path=builtin, is_builtin=True)

    # ── installing ──────────────────────────────────────────────────────────────

    def free_bytes(self) -> int:
        """Free space on the filesystem the store lives on, walking up to one that exists."""
        probe = self.root
        while True:
            try:
                return shutil.disk_usage(probe).free
            except OSError:
                if probe.parent == probe:
                    return 0
                probe = probe.parent

    def install(
        self,
        policy_id: str,
        url: str,
        sha256: str,
        manifest: dict | None = None,
        *,
        protect: Iterable[str] = (),
    ) -> InstallResult:
        """Fetch, verify, measure and store one policy. See the module docstring's diagram.

        Returns an ``InstallResult`` rather than raising for the expected refusals -- a
        wrong-shaped file and a full card are outcomes this endpoint reports, not bugs.
        ``protect`` names ids eviction must not consider: the server passes the policy a
        running walk is using, which the store cannot see for itself.
        """
        candidate = validate_id(policy_id)
        self.root.mkdir(parents=True, exist_ok=True)

        # Idempotency first, because it needs no bytes: a retry after a wifi drop, or two
        # tabs installing the same policy, must not be refused by a disk floor it never
        # would have crossed.
        existing = self.root / candidate / MODEL_FILENAME
        if existing.is_file() and self._matches_digest(existing, sha256):
            self.mark_used(candidate)
            return InstallResult(
                ok=True,
                id=candidate,
                manifest=self._read_manifest(self.root / candidate) or None,
                detail="already installed with this content",
                already_installed=True,
            )

        # The floor is checked BEFORE the download, so a card that is nearly full is a
        # refusal rather than a card that is full. The size comes from the sidecar manifest
        # when there is one and from the contract's ceiling when there is not -- assuming
        # the worst case is the only honest guess about a file we have not fetched.
        declared = manifest.get("size_bytes") if isinstance(manifest, dict) else None
        reserve = (
            int(declared)
            if isinstance(declared, int) and declared > 0
            else self.max_policy_bytes
        )
        free = self.free_bytes()
        if free < self.free_floor_bytes + reserve:
            return InstallResult(
                ok=False,
                code=POLICY_STORE_FULL,
                detail=(
                    f"{free} bytes free; this install needs {reserve} bytes plus the "
                    f"{self.free_floor_bytes}-byte floor the OS keeps to stay bootable"
                ),
                id=candidate,
            )

        temp = self.root / f".tmp-{candidate}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            return self._install_from(temp, candidate, url, sha256, manifest, protect)
        finally:
            temp.unlink(missing_ok=True)

    def _install_from(
        self,
        temp: Path,
        candidate: str,
        url: str,
        sha256: str,
        manifest: dict | None,
        protect: Iterable[str],
    ) -> InstallResult:
        try:
            self.fetch(url, temp, max_bytes=self.max_policy_bytes)
        except (DownloadFailed, OSError) as exc:
            # OSError as well as the store's own type: the card can fill up *during* the
            # write (ENOSPC), and a failed install must be a refusal with a code, never a
            # 500 from the robot.
            return InstallResult(
                ok=False,
                code=POLICY_INSTALL_FAILED,
                detail=f"{exc.__class__.__name__}: {exc}",
                id=candidate,
            )

        result = check_policy(
            temp, expect_sha256=sha256, max_bytes=self.max_policy_bytes
        )
        if not result.ok or result.manifest is None:
            return InstallResult(
                ok=False,
                code=result.code or POLICY_CONTRACT_MISMATCH,
                detail=result.detail,
                id=candidate,
            )

        # Off the hot path, on the robot that will run it (story 2.6). Never blocks: an
        # over-budget policy installs and is flagged, and a measurement that could not run
        # at all still installs.
        latency = measure_latency_at(temp)

        merged: dict = dict(result.manifest)
        if isinstance(manifest, dict):
            # A sidecar manifest describes the policy; it does not get to overrule what the
            # graph and the stopwatch actually said.
            reserved = set(latency.as_manifest_fields()) | {
                "obs_dim",
                "act_dim",
                "input_shape",
                "output_shape",
                "input_name",
                "output_name",
                "size_bytes",
                "sha256",
            }
            merged.update({k: v for k, v in manifest.items() if k not in reserved})
            merged["inferred"] = False
        merged.update(latency.as_manifest_fields())
        merged["id"] = candidate
        merged["installed_at"] = round(self._now(), 3)

        evicted = self._make_room(candidate, protect)
        if self._would_exceed(candidate):
            return InstallResult(
                ok=False,
                code=POLICY_STORE_FULL,
                detail=(
                    f"the store holds {self.max_policies} policies and none of them is "
                    "evictable: the built-in is never evicted, and neither is the active "
                    "policy or the one a running walk is using"
                ),
                id=candidate,
                evicted=evicted,
            )

        try:
            self._commit(temp, candidate, merged)
        except OSError as exc:
            return InstallResult(
                ok=False,
                code=POLICY_INSTALL_FAILED,
                detail=(
                    f"verified {candidate} but could not store it: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
                id=candidate,
                evicted=evicted,
            )

        self.mark_used(candidate)
        warning = None
        if latency.over_budget:
            warning = {"code": latency.warning_code, "detail": latency.detail}
        return InstallResult(
            ok=True,
            id=candidate,
            manifest=merged,
            detail=result.detail,
            evicted=evicted,
            warning=warning,
        )

    def _matches_digest(self, path: Path, sha256: str) -> bool:
        """Whether the file already on disk is the content being asked for.

        Concurrent installs of the same id are idempotent by content, which is what makes
        a retry after a flaky wifi drop free. The file is hashed rather than the manifest
        trusted: the manifest records what we were *told*.
        """
        expected = normalise_digest(sha256) if isinstance(sha256, str) else None
        if expected is None:
            return False
        try:
            return sha256_file(path) == expected
        except OSError:
            return False

    def _would_exceed(self, candidate: str) -> bool:
        """Whether storing ``candidate`` would leave more policies than the cap allows."""
        return len({e.id for e in self._entries()} | {candidate}) > self.max_policies

    def _make_room(self, candidate: str, protect: Iterable[str]) -> dict | None:
        """Evict least-recently-used policies until ``candidate`` fits. Names what went.

        Runs after the download and the check, so a failed install never costs the operator
        a policy they had. An eviction the operator only finds out about later is a
        surprise, and a surprise about deleted data is the worst kind, so the caller puts
        this in the response.

        Never evicts the built-in (it is not in the store to begin with), the active
        policy, or an id in ``protect`` -- which is how the server keeps a running walk's
        policy from being deleted out from under it.
        """
        keep = {candidate, BUILTIN_ID, self.active_id(), *protect}
        gone: list[dict] = []
        while self._would_exceed(candidate):
            victim = next(
                (e for e in self._entries() if e.id not in keep), None
            )  # _entries() is least-recently-used first
            if victim is None:
                break
            try:
                shutil.rmtree(self.root / victim.id)
            except OSError as exc:
                print(f"[policy_store] could not evict {victim.id}: {exc}")
                keep.add(victim.id)  # do not spin on a directory that will not go
                continue
            gone.append(
                {
                    "id": victim.id,
                    "reason": "least recently used",
                    "sizeBytes": victim.size_bytes,
                    "lastUsedAt": round(victim.last_used_at, 3),
                }
            )

        if not gone:
            return None
        evicted = dict(gone[0])
        if len(gone) > 1:
            # Only reachable if the cap was lowered under a store that was already full.
            evicted["also"] = [e["id"] for e in gone[1:]]
        return evicted

    def _commit(self, temp: Path, candidate: str, manifest: dict) -> None:
        """Move a verified temp file into place, atomically enough for a power cut.

        The model is renamed (never copied) into a staging directory on the same
        filesystem, the manifest is written beside it, and the staging directory replaces
        the destination in one rename. A reader therefore sees either the old policy or the
        new one, never a directory holding a new model and an old manifest.
        """
        final = self.root / candidate
        staging = self.root / f".staging-{candidate}-{uuid.uuid4().hex[:8]}"
        trash = self.root / f".trash-{candidate}-{uuid.uuid4().hex[:8]}"

        staging.mkdir(parents=True, exist_ok=False)
        try:
            os.replace(temp, staging / MODEL_FILENAME)
            (staging / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
            (staging / USED_FILENAME).write_bytes(b"")
            rotated = False
            if final.exists():
                os.replace(final, trash)
                rotated = True
            try:
                os.replace(staging, final)
            except OSError:
                if rotated:
                    os.replace(trash, final)  # put the old one back
                raise
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(trash, ignore_errors=True)


def _size_of(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_size
    except OSError:
        return None
