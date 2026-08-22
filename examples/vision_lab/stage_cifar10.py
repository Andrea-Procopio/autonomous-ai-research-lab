"""Stage CIFAR-10: the one place a dataset touches the network.

An operator act, run once per machine, in trusted code, against a pinned
archive digest — before any run exists and outside every budget::

    python -m examples.vision_lab.stage_cifar10 --datasets-root ~/arl-data

Downloads the canonical ``cifar-10-python.tar.gz``, verifies the archive
digest *before* extracting a byte, extracts the batches, and writes the
write-once manifest the lab's preflight verifies jobs against. Stdlib
only: staging must not need torch — a container-backend operator never
installs the ``[vision]`` extra on the host.

Idempotent: a dataset already staged is verified and left alone.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .datasets import DatasetStore

DATASET_NAME = "cifar10"

SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

ARCHIVE_SHA256 = (
    "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
)
"""The canonical archive's digest, pinned here the way container images
are pinned by digest: what gets extracted is what this line names, or
nothing does."""


def fetch(destination: Path) -> None:
    """Download, verify, extract — refusing before extraction on any
    digest mismatch, so unverified bytes never reach the staging dir."""
    archive = destination.parent / f".{DATASET_NAME}.tar.gz"
    print(f"downloading {SOURCE_URL} ...", flush=True)
    urllib.request.urlretrieve(SOURCE_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        archive.unlink()
        raise SystemExit(
            f"archive digest {digest} does not match the pinned "
            f"{ARCHIVE_SHA256}; refusing to extract"
        )
    print("digest verified; extracting ...", flush=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    archive.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        required=True,
        help="where staged datasets and their manifests live",
    )
    arguments = parser.parse_args(argv)
    store = DatasetStore(arguments.datasets_root)
    manifest = store.stage(
        DATASET_NAME,
        fetch=fetch,
        source_url=SOURCE_URL,
        archive_sha256=ARCHIVE_SHA256,
        retrieved_at=datetime.now(UTC).isoformat(),
    )
    problems = store.verify(DATASET_NAME)
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(
        f"staged {manifest.name}: {len(manifest.files)} file(s), "
        f"{manifest.total_bytes:,} bytes, manifest {manifest.id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
