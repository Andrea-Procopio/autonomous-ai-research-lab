"""Datasets a job may read: staged by trusted code, named by content.

A training job runs with no network — in a container, structurally so —
which means its dataset must already be on disk, and everything about
scientific provenance then hangs on one question: *which bytes were
those?* The answer here is the ledger's answer everywhere else in this
repository: a content-derived identity over the parts, written once,
verified by re-reading.

A :class:`DatasetManifest` records every file's digest and size; its id
is derived from those digests, so the same bytes staged on any machine
get the same id — which is what makes ``dataset_id`` safe to put in job
config without smuggling a backend into the scientific record. The
:class:`DatasetStore` keeps manifests write-once beside the staged
files, and :class:`DatasetStaged` is the preflight check that refuses a
job whose declared dataset is absent, altered, or not the one its
manifest claims.

Staging is an operator act (see ``stage_cifar10``): the one place a
network fetch happens, in trusted code, against a pinned archive digest,
before any run exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.ids import content_id
from autonomous_research_lab.runtime.preflight import JobLike
from autonomous_research_lab.runtime.verification import (
    CheckState,
    ValidityDimension,
    VerificationCheck,
)

MANIFESTS_DIRNAME = "manifests"

DATASET_ID_KEY = "dataset_id"
"""The job-config key naming which bytes the job may read — a manifest
id, machine-independent by construction."""

DATASET_ROOT_KEY = "dataset_root"
"""The job-config key naming where those bytes are, *as the job's process
sees them* — a host path for a host backend, ``/arl/data/<name>`` inside
a container. Deliberately not ``*_dir``/``*_path``: the generic
path-exists preflight must not host-check a container path, and
:class:`DatasetStaged` is the stronger, backend-aware check instead."""


class DatasetConflictError(RuntimeError):
    """A manifest already exists with different content. Manifests are
    write-once; restaging different bytes under the same name is a new
    dataset pretending to be an old one."""


class UnknownDatasetError(KeyError):
    """No manifest under that name."""


class DatasetIntegrityError(RuntimeError):
    """A manifest that does not survive its own digests."""


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """What one staged dataset is, file by file."""

    name: str
    source_url: str
    archive_sha256: str
    """Digest of the archive as fetched; empty when staged from local
    files rather than an archive."""

    retrieved_at: str
    """ISO 8601, staging time. Provenance, deliberately outside the id:
    the same bytes staged twice are the same dataset."""

    files: tuple[tuple[str, str, int], ...]
    """``(relative_path, sha256, size_bytes)``, sorted by path."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name.strip() or "/" in self.name or ".." in self.name:
            raise ValueError(
                f"dataset name {self.name!r} must be a plain directory name"
            )
        if not self.files:
            raise ValueError("a dataset manifest must name at least one file")
        paths = [path for path, _, _ in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("manifest files must be sorted and unique by path")
        for path, digest, size in self.files:
            if not path.strip() or Path(path).is_absolute() or ".." in path:
                raise ValueError(f"file path {path!r} must stay relative")
            if len(digest) != 64 or any(
                c not in "0123456789abcdef" for c in digest
            ):
                raise ValueError(f"file {path!r} needs a sha256 hex digest")
            if size < 0:
                raise ValueError(f"file {path!r} cannot have negative size")
        derived = content_id("dset", self.name, self.files)
        if not self.id:
            object.__setattr__(self, "id", derived)
        elif self.id != derived:
            raise DatasetIntegrityError(
                f"manifest {self.name} carries id {self.id}, but its files "
                f"derive {derived}; the record does not survive its digests"
            )

    @property
    def total_bytes(self) -> int:
        return sum(size for _, _, size in self.files)


class DatasetStore:
    """Write-once manifests beside the staged files they describe.

    Layout, under an operator-chosen root::

        <root>/
        ├── <name>/            the files
        └── manifests/
            └── <name>.json    the manifest, id-carrying, write-once
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, name: str) -> Path:
        return self._root / name

    def stage(
        self,
        name: str,
        *,
        fetch: Callable[[Path], None],
        source_url: str,
        archive_sha256: str = "",
        retrieved_at: str = "",
    ) -> DatasetManifest:
        """Make ``name`` durable and manifested, once.

        Already staged means verified and returned — ``fetch`` is never
        called twice for one name, so staging is idempotent and cheap to
        re-run. Otherwise ``fetch`` populates a scratch directory, every
        file is hashed, the manifest is written, and the files move into
        place — manifest last, so a crash mid-staging leaves an obvious
        partial rather than a dataset that looks finished.
        """
        existing = self._read(name)
        if existing is not None:
            self.verify(name)
            return existing
        scratch = self._root / f".staging-{name}"
        if scratch.exists():
            raise DatasetConflictError(
                f"a previous staging of {name!r} left {scratch}; inspect "
                f"and remove it before staging again"
            )
        scratch.mkdir(parents=True)
        fetch(scratch)
        manifest = DatasetManifest(
            name=name,
            source_url=source_url,
            archive_sha256=archive_sha256,
            retrieved_at=retrieved_at,
            files=_hashed(scratch),
        )
        scratch.replace(self.path_for(name))
        self._write(manifest)
        return manifest

    def manifest(self, name: str) -> DatasetManifest:
        found = self._read(name)
        if found is None:
            raise UnknownDatasetError(name)
        return found

    def manifests(self) -> tuple[DatasetManifest, ...]:
        directory = self._root / MANIFESTS_DIRNAME
        if not directory.is_dir():
            return ()
        return tuple(
            self.manifest(path.stem)
            for path in sorted(directory.glob("*.json"))
        )

    def by_id(self, dataset_id: str) -> DatasetManifest | None:
        return next(
            (m for m in self.manifests() if m.id == dataset_id), None
        )

    def verify(self, name: str, *, deep: bool = True) -> tuple[str, ...]:
        """Every way the staged files disagree with their manifest.

        Deep re-hashes every byte; shallow settles for existence and
        size. Returns problems rather than raising, because the caller
        deciding what a problem means is the point of having one."""
        manifest = self.manifest(name)
        base = self.path_for(name)
        problems: list[str] = []
        for relative, digest, size in manifest.files:
            path = base / relative
            if not path.is_file():
                problems.append(f"{relative}: missing")
                continue
            actual_size = path.stat().st_size
            if actual_size != size:
                problems.append(
                    f"{relative}: {actual_size} bytes where the manifest "
                    f"says {size}"
                )
                continue
            if deep and _sha256(path) != digest:
                problems.append(f"{relative}: contents no longer match")
        return tuple(problems)

    def _manifest_path(self, name: str) -> Path:
        return self._root / MANIFESTS_DIRNAME / f"{name}.json"

    def _write(self, manifest: DatasetManifest) -> None:
        path = self._manifest_path(manifest.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": manifest.id,
            "name": manifest.name,
            "source_url": manifest.source_url,
            "archive_sha256": manifest.archive_sha256,
            "retrieved_at": manifest.retrieved_at,
            "files": [list(entry) for entry in manifest.files],
        }
        scratch = path.with_suffix(".tmp")
        scratch.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        scratch.replace(path)

    def _read(self, name: str) -> DatasetManifest | None:
        path = self._manifest_path(name)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = DatasetManifest(
            name=str(payload["name"]),
            source_url=str(payload["source_url"]),
            archive_sha256=str(payload["archive_sha256"]),
            retrieved_at=str(payload["retrieved_at"]),
            files=tuple(
                (str(p), str(d), int(s)) for p, d, s in payload["files"]
            ),
            id=str(payload["id"]),
        )
        return manifest


@dataclass(frozen=True, slots=True)
class DatasetStaged:
    """Preflight: the job's declared dataset exists and still verifies.

    Checked against the *host-side* staging directory, deliberately —
    the job may see the bytes at a container path, but the bytes being
    mounted are these, and this is where they can be read before any
    container exists. ``NOT_APPLICABLE`` for jobs that declare no
    dataset, so the check composes into any preflight tuple.
    """

    store: DatasetStore
    deep: bool = True

    def check(
        self,
        job: JobLike,
        spec: ExperimentSpec | None,  # noqa: ARG002 - job-only check
    ) -> VerificationCheck:
        declared = job.config.get(DATASET_ID_KEY)
        if declared is None:
            return _check(
                CheckState.NOT_APPLICABLE, "no dataset declared"
            )
        if not isinstance(declared, str) or not declared:
            return _check(
                CheckState.FAIL, f"{DATASET_ID_KEY} must be a manifest id"
            )
        manifest = self.store.by_id(declared)
        if manifest is None:
            return _check(
                CheckState.FAIL,
                f"no staged dataset carries manifest id {declared}",
            )
        problems = self.store.verify(manifest.name, deep=self.deep)
        if problems:
            return _check(
                CheckState.FAIL,
                f"dataset {manifest.name} no longer verifies: "
                + "; ".join(problems[:3]),
            )
        return _check(
            CheckState.PASS,
            f"dataset {manifest.name} ({manifest.id}) verifies",
        )


def _check(state: CheckState, detail: str) -> VerificationCheck:
    return VerificationCheck(
        dimension=ValidityDimension.EXECUTION,
        name="preflight:dataset_staged",
        state=state,
        detail=detail,
    )


def _hashed(base: Path) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        sorted(
            (
                str(path.relative_to(base)),
                _sha256(path),
                path.stat().st_size,
            )
            for path in base.rglob("*")
            if path.is_file()
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
