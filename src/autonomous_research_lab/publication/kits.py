"""Venue kits: the conference's own style files, staged and pinned.

The dataset store's discipline, transposed: an operator stages the
venue's official kit into a kits directory; every file is hashed into a
write-once, content-id-carrying manifest; rendering verifies the staged
files against it and refuses a kit that is missing, tampered, or was
never staged. Machine paths stay in deployment arguments — nothing here
names where a kit lives on this host inside any record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..core.ids import content_id

MANIFESTS_DIRNAME: Final = "manifests"


class KitConflictError(RuntimeError):
    """A kit manifest already exists with different content. Manifests
    are write-once; restaging different bytes under the same name is a
    new kit pretending to be an old one."""


class UnknownKitError(KeyError):
    """No kit manifest under that name."""


class KitIntegrityError(RuntimeError):
    """A kit manifest that does not survive its own digests."""


@dataclass(frozen=True, slots=True)
class KitManifest:
    """What one staged venue kit is, file by file."""

    name: str
    source_url: str
    archive_sha256: str
    """Digest of the archive as fetched; empty when staged from a local
    directory rather than an archive."""

    retrieved_at: str
    """ISO 8601, staging time. Provenance, deliberately outside the id:
    the same bytes staged twice are the same kit."""

    files: tuple[tuple[str, str, int], ...]
    """``(relative_path, sha256, size_bytes)``, sorted by path."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name.strip() or "/" in self.name or ".." in self.name:
            raise ValueError(
                f"kit name {self.name!r} must be a plain directory name"
            )
        if not self.files:
            raise ValueError("a kit manifest must name at least one file")
        paths = [path for path, _, _ in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError(
                "manifest files must be sorted and unique by path"
            )
        for path, digest, size in self.files:
            if not path.strip() or Path(path).is_absolute() or ".." in path:
                raise ValueError(f"file path {path!r} must stay relative")
            if len(digest) != 64 or any(
                c not in "0123456789abcdef" for c in digest
            ):
                raise ValueError(f"file {path!r} needs a sha256 hex digest")
            if size < 0:
                raise ValueError(f"file {path!r} cannot have negative size")
        derived = content_id("vkit", self.name, self.files)
        if not self.id:
            object.__setattr__(self, "id", derived)
        elif self.id != derived:
            raise KitIntegrityError(
                f"manifest {self.name} carries id {self.id}, but its "
                f"files derive {derived}; the record does not survive "
                f"its digests"
            )

    @property
    def total_bytes(self) -> int:
        return sum(size for _, _, size in self.files)


class KitStore:
    """Write-once kit manifests beside the staged files they describe.

    Layout, under an operator-chosen root::

        <root>/
        ├── <name>/            the kit's files
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
    ) -> KitManifest:
        """Make ``name`` durable and manifested, once.

        Already staged means verified and returned — ``fetch`` is never
        called twice for one name. Otherwise ``fetch`` populates a
        scratch directory, every file is hashed, the files move into
        place, and the manifest is written last, so a crash mid-staging
        leaves an obvious partial rather than a kit that looks finished.
        """
        existing = self._read(name)
        if existing is not None:
            self.verify(name)
            return existing
        scratch = self._root / f".staging-{name}"
        if scratch.exists():
            raise KitConflictError(
                f"a previous staging of {name!r} left {scratch}; inspect "
                f"and remove it before staging again"
            )
        scratch.mkdir(parents=True)
        fetch(scratch)
        manifest = KitManifest(
            name=name,
            source_url=source_url,
            archive_sha256=archive_sha256,
            retrieved_at=retrieved_at,
            files=_hashed(scratch),
        )
        scratch.replace(self.path_for(name))
        self._write(manifest)
        return manifest

    def manifest(self, name: str) -> KitManifest:
        found = self._read(name)
        if found is None:
            raise UnknownKitError(name)
        return found

    def manifests(self) -> tuple[KitManifest, ...]:
        directory = self._root / MANIFESTS_DIRNAME
        if not directory.is_dir():
            return ()
        return tuple(
            self.manifest(path.stem)
            for path in sorted(directory.glob("*.json"))
        )

    def verify(self, name: str, *, deep: bool = True) -> tuple[str, ...]:
        """Every way the staged files disagree with their manifest.
        Returns problems rather than raising, because the caller
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

    def _write(self, manifest: KitManifest) -> None:
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

    def _read(self, name: str) -> KitManifest | None:
        path = self._manifest_path(name)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return KitManifest(
            name=str(payload["name"]),
            source_url=str(payload["source_url"]),
            archive_sha256=str(payload["archive_sha256"]),
            retrieved_at=str(payload["retrieved_at"]),
            files=tuple(
                (str(p), str(d), int(s)) for p, d, s in payload["files"]
            ),
            id=str(payload["id"]),
        )


def _hashed(base: Path) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (
            str(path.relative_to(base)),
            _sha256(path),
            path.stat().st_size,
        )
        for path in sorted(base.rglob("*"))
        if path.is_file()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()
