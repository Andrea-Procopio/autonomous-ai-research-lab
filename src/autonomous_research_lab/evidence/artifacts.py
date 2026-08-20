"""Content-addressed storage for what an experiment left behind.

An ``ExperimentResult`` names its outputs by absolute path into a run
directory, and the only record of their contents is a ``manifest.json``
sitting in that same directory. Both disappear together. This module
keeps the bytes::

    <root>/
    ├── blobs/<aa>/<sha256>          the bytes, once
    └── artifacts/<result_id>.json   one manifest record per result

Blobs are content-addressed, so identical bytes are stored once however
many results produced them, and re-ingesting a result is a no-op rather
than a duplicate. A blob is published by hard-linking a scratch file
into place — the same trick the budget ledger uses — so a crash
mid-write leaves an ignorable scratch file instead of a truncated blob
under a name that promises its own digest.

Ingest refuses more than it accepts, on purpose:

* a path that resolves outside the run directory is not this run's
  output, and the executor already declines to collect one;
* a file that no longer hashes to what the run's own ``manifest.json``
  recorded is a post-hoc edit, and storing the newer bytes would launder
  it;
* a file past :data:`MAX_BLOB_BYTES` fails loudly rather than quietly
  filling a disk.

What this module never does is interpret. A manifest says which bytes
existed and what they hashed to. What they mean is evidence, and what
that means is a claim.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

from ..core.experiment import ExperimentResult
from ..core.ids import content_id, occurrence_id

MANIFEST_FILENAME: Final = "manifest.json"
"""Part of the executor's run-directory contract (mirrored here rather
than imported, exactly as ``runtime.validation`` mirrors it, so that the
evidence layer keeps depending on ``core`` alone): relative artifact
path -> sha256."""

MAX_BLOB_BYTES: Final = 64 * 1024 * 1024
"""The largest single file this store will hold. Today's artifacts are
kilobytes of metrics, logs, and plots. Model checkpoints are a different
problem and deserve an explicit policy rather than a silent default."""

_BLOBS: Final = "blobs"
_MANIFESTS: Final = "artifacts"
_RECORD_SUFFIX: Final = ".json"

_MEDIA_TYPES: Final = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".py": "text/x-python",
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
    ".yaml": "application/yaml",
}
"""A small fixed table. Anything else is bytes, and says so."""

_DEFAULT_MEDIA_TYPE: Final = "application/octet-stream"


class ArtifactRefusedError(RuntimeError):
    """Ingest refused a file: it escapes the run directory, it no longer
    matches the digest the run recorded, it exceeds the size ceiling, or
    it is gone. Raised before anything is written."""


class ArtifactConflictError(RuntimeError):
    """A write-once artifact record would be overwritten with different
    content."""


class ArtifactIntegrityError(RuntimeError):
    """A stored manifest no longer matches its own identity, or a blob it
    names is missing or no longer hashes to its digest."""


class ArtifactKind(StrEnum):
    ARTIFACT = "artifact"
    """A file the experiment process wrote into its run directory."""

    LOG = "log"
    """The run's captured stdout or stderr. Kept because a failed run's
    output is the whole diagnosis."""


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """One stored file: where the run put it, and what it was."""

    path: str
    """Run-directory-relative, so the record survives the absolute path."""

    digest: str
    size_bytes: int
    media_type: str
    kind: ArtifactKind

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("an artifact entry must name its path")
        if len(self.digest) != 64:
            raise ValueError(
                f"an artifact digest is a sha256 hex string, got "
                f"{self.digest!r}"
            )
        if self.size_bytes < 0:
            raise ValueError("an artifact cannot have a negative size")


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Everything one result left behind, as stored.

    Content-addressed over every field, so unlike the result it belongs
    to — whose id derives from its job alone — a manifest re-derives a
    different id the moment anything in it is edited.
    """

    result_id: str
    spec_id: str
    job_id: str
    entries: tuple[ArtifactEntry, ...] = ()
    id: str = field(default="")

    def __post_init__(self) -> None:
        for label, value in (
            ("result_id", self.result_id),
            ("spec_id", self.spec_id),
            ("job_id", self.job_id),
        ):
            if not value.strip():
                raise ValueError(f"an artifact manifest must name its {label}")
        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError(
                "a manifest holds one entry per path; duplicates would make "
                "the record ambiguous about which bytes were stored"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "amf",
                    self.result_id,
                    self.spec_id,
                    self.job_id,
                    tuple(
                        (
                            entry.path,
                            entry.digest,
                            entry.size_bytes,
                            entry.media_type,
                            str(entry.kind),
                        )
                        for entry in self.entries
                    ),
                ),
            )

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)


class ArtifactStore(Protocol):
    """What an evidence store needs from artifact storage. A protocol so
    the policy is injected rather than assumed: a stricter ceiling, a
    stricter refusal, or one day a remote object store."""

    def ingest(self, result: ExperimentResult) -> ArtifactManifest: ...

    def get(self, result_id: str) -> ArtifactManifest | None: ...

    def manifests(self) -> tuple[ArtifactManifest, ...]: ...

    def blob_path(self, digest: str) -> Path: ...


class FileArtifactStore:
    """Blobs and manifests under one root."""

    def __init__(
        self, root: Path | str, *, max_blob_bytes: int = MAX_BLOB_BYTES
    ) -> None:
        if max_blob_bytes <= 0:
            raise ValueError("the blob ceiling must be a positive size")
        self._root = Path(root)
        self._blobs = self._root / _BLOBS
        self._manifests = self._root / _MANIFESTS
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._manifests.mkdir(parents=True, exist_ok=True)
        self._max_blob_bytes = max_blob_bytes

    @property
    def root(self) -> Path:
        return self._root

    def blob_path(self, digest: str) -> Path:
        return self._blobs / digest[:2] / digest

    # -- ingest ----------------------------------------------------------------

    def ingest(self, result: ExperimentResult) -> ArtifactManifest:
        """Store every file ``result`` names, then record what was stored.

        Idempotent: re-ingesting one result re-hashes the same bytes into
        the blobs already present and returns the manifest already
        recorded. Nothing is written if any file is refused.
        """
        existing = self.get(result.id)
        run_dir = _run_dir_of(result)
        planned = self._plan(result, run_dir)
        manifest = ArtifactManifest(
            result_id=result.id,
            spec_id=result.spec_id,
            job_id=result.job_id,
            entries=tuple(entry for entry, _ in planned),
        )
        if existing is not None:
            if existing != manifest:
                raise ArtifactConflictError(
                    f"result {result.id} already has a different artifact "
                    f"manifest; manifests are never rewritten"
                )
            # The bytes may still be missing if a previous ingest was
            # interrupted between the blobs and the record, so store them
            # again — content addressing makes that a no-op when present.
            for _, source in planned:
                self._store_blob(source)
            return existing
        for _, source in planned:
            self._store_blob(source)
        self._write_manifest(manifest)
        return manifest

    def _plan(
        self, result: ExperimentResult, run_dir: Path | None
    ) -> list[tuple[ArtifactEntry, Path]]:
        """Every file to store, checked before anything is written."""
        named = [(path, ArtifactKind.ARTIFACT) for path in result.artifacts]
        named += [(path, ArtifactKind.LOG) for path in result.logs]
        if not named:
            return []
        if run_dir is None:
            raise ArtifactRefusedError(
                f"result {result.id} names {len(named)} file(s) but no logs; "
                f"its run directory cannot be located, so nothing can be "
                f"confined to it"
            )
        declared = _declared_digests(run_dir)
        planned: list[tuple[ArtifactEntry, Path]] = []
        seen: set[str] = set()
        for raw, kind in named:
            source = Path(raw).resolve()
            relative = _confine(source, run_dir, result.id)
            if relative in seen:
                continue
            seen.add(relative)
            if not source.is_file():
                raise ArtifactRefusedError(
                    f"result {result.id} names {relative}, which is not a "
                    f"file; a result cannot store an output it did not leave"
                )
            size = source.stat().st_size
            if size > self._max_blob_bytes:
                raise ArtifactRefusedError(
                    f"{relative} is {size} bytes, over the {self._max_blob_bytes}"
                    f"-byte ceiling; storing it needs an explicit policy"
                )
            digest = _sha256_of(source)
            recorded = declared.get(relative)
            if recorded is not None and recorded != digest:
                raise ArtifactRefusedError(
                    f"{relative} no longer hashes to what the run recorded "
                    f"in {MANIFEST_FILENAME}; storing the newer bytes would "
                    f"launder a post-hoc edit"
                )
            planned.append(
                (
                    ArtifactEntry(
                        path=relative,
                        digest=digest,
                        size_bytes=size,
                        media_type=_media_type(source),
                        kind=kind,
                    ),
                    source,
                )
            )
        planned.sort(key=lambda item: item[0].path)
        return planned

    def _store_blob(self, source: Path) -> None:
        digest = _sha256_of(source)
        target = self.blob_path(digest)
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch = target.parent / f"{occurrence_id('blob')}.tmp"
        scratch.write_bytes(source.read_bytes())
        try:
            os.link(scratch, target)
        except FileExistsError:
            # Another writer stored the same bytes first. Content
            # addressing makes that indistinguishable from success.
            pass
        finally:
            scratch.unlink(missing_ok=True)

    # -- records ---------------------------------------------------------------

    def get(self, result_id: str) -> ArtifactManifest | None:
        path = self._manifest_path(result_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ArtifactIntegrityError(
                f"artifact manifest {path.name} is not an object"
            )
        manifest = _manifest_from(payload)
        filed_as = payload.get("id")
        if filed_as != manifest.id:
            raise ArtifactIntegrityError(
                f"artifact manifest {path.name} claims id {filed_as!r} but "
                f"re-derives {manifest.id}; the file was edited"
            )
        return manifest

    def manifests(self) -> tuple[ArtifactManifest, ...]:
        loaded = []
        for path in sorted(self._manifests.glob(f"*{_RECORD_SUFFIX}")):
            manifest = self.get(path.stem)
            assert manifest is not None
            loaded.append(manifest)
        return tuple(loaded)

    def _manifest_path(self, result_id: str) -> Path:
        return self._manifests / f"{result_id}{_RECORD_SUFFIX}"

    def _write_manifest(self, manifest: ArtifactManifest) -> None:
        self._manifest_path(manifest.result_id).write_text(
            json.dumps(_manifest_payload(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )


# -- helpers -------------------------------------------------------------------


def _run_dir_of(result: ExperimentResult) -> Path | None:
    """The executor writes both logs into the run directory, so its parent
    is the directory this result is authorized to have written in."""
    if not result.logs:
        return None
    return Path(result.logs[0]).resolve().parent


def _confine(source: Path, run_dir: Path, result_id: str) -> str:
    try:
        return str(source.relative_to(run_dir))
    except ValueError as exc:
        raise ArtifactRefusedError(
            f"result {result_id} names {source}, which resolves outside its "
            f"run directory {run_dir}; what lies there was not produced by "
            f"this run"
        ) from exc


def _declared_digests(run_dir: Path) -> dict[str, str]:
    """What the run itself recorded, when it recorded anything."""
    path = run_dir / MANIFEST_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in payload.items()
        if isinstance(value, str)
    }


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), _DEFAULT_MEDIA_TYPE)


def _manifest_payload(manifest: ArtifactManifest) -> dict[str, object]:
    return {
        "id": manifest.id,
        "result_id": manifest.result_id,
        "spec_id": manifest.spec_id,
        "job_id": manifest.job_id,
        "entries": [
            {
                "path": entry.path,
                "digest": entry.digest,
                "size_bytes": entry.size_bytes,
                "media_type": entry.media_type,
                "kind": str(entry.kind),
            }
            for entry in manifest.entries
        ],
    }


def _manifest_from(payload: dict[str, object]) -> ArtifactManifest:
    raw = payload.get("entries")
    if not isinstance(raw, list):
        raise ArtifactIntegrityError("entries must be a list")
    entries = []
    for item in raw:
        if not isinstance(item, dict):
            raise ArtifactIntegrityError("each entry must be an object")
        try:
            entries.append(
                ArtifactEntry(
                    path=_text(item, "path"),
                    digest=_text(item, "digest"),
                    size_bytes=_integer(item, "size_bytes"),
                    media_type=_text(item, "media_type"),
                    kind=ArtifactKind(_text(item, "kind")),
                )
            )
        except ValueError as exc:
            raise ArtifactIntegrityError(f"unreadable entry: {exc}") from exc
    return ArtifactManifest(
        result_id=_text(payload, "result_id"),
        spec_id=_text(payload, "spec_id"),
        job_id=_text(payload, "job_id"),
        entries=tuple(entries),
    )


def _text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ArtifactIntegrityError(f"{key} must be a string")
    return value


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactIntegrityError(f"{key} must be an integer")
    return value
