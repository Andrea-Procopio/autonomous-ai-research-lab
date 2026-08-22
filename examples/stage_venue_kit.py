"""Stage one conference's LaTeX kit for `arl render`.

Operators run this once per venue, pointing it at the official style
archive (or an unpacked local directory)::

    python -m examples.stage_venue_kit --kits-root ~/arl-kits \\
        --venue neurips --from-zip <official styles .zip or URL> \\
        [--sha256 <pinned hex>]

The archive is hashed BEFORE extraction: with ``--sha256`` given, a
mismatch deletes the download and refuses — what gets staged is what
that line names, or nothing does. Without it, the computed hash is
printed so the operator can pin it for next time. Extraction guards
member paths (no absolute paths, no ``..``). Staging is idempotent:
an already-staged kit is verified and left alone.

Standard library only, deliberately: staging must not need TeX.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from autonomous_research_lab.publication.kits import KitStore


def _fetch_zip(
    archive_source: str, pinned_sha256: str
) -> tuple[Path, str, bool]:
    """The archive as a local file, its digest checked against the pin.
    Returns (path, sha256, downloaded)."""
    if archive_source.startswith(("http://", "https://")):
        local = Path(f".kit-download-{Path(archive_source).name or 'kit'}")
        with urllib.request.urlopen(archive_source) as response:
            local.write_bytes(response.read())
        downloaded = True
    else:
        local = Path(archive_source)
        downloaded = False
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    if pinned_sha256 and digest != pinned_sha256:
        if downloaded:
            local.unlink()
        raise SystemExit(
            f"refusing to stage: archive sha256 is {digest}, "
            f"--sha256 pinned {pinned_sha256}"
        )
    return local, digest, downloaded


def _extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SystemExit(
                    f"refusing to stage: archive member {member!r} "
                    f"escapes the kit directory"
                )
        bundle.extractall(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kits-root", type=Path, required=True)
    parser.add_argument("--venue", required=True, help="kit name to stage")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-zip", help="path or URL of the official style archive"
    )
    source.add_argument(
        "--from-dir", type=Path, help="an already-unpacked kit directory"
    )
    parser.add_argument(
        "--sha256",
        default="",
        help="pinned archive digest; a mismatch refuses before extraction",
    )
    arguments = parser.parse_args(argv)

    store = KitStore(arguments.kits_root)
    if arguments.from_zip:
        archive, digest, downloaded = _fetch_zip(
            arguments.from_zip, arguments.sha256
        )
        print(f"archive sha256 {digest}")

        def fetch(destination: Path) -> None:
            _extract(archive, destination)
            if downloaded:
                archive.unlink()

        manifest = store.stage(
            arguments.venue,
            fetch=fetch,
            source_url=arguments.from_zip,
            archive_sha256=digest,
            retrieved_at=datetime.now(UTC).isoformat(),
        )
    else:
        if not arguments.from_dir.is_dir():
            print(
                f"FATAL: {arguments.from_dir} is not a directory",
                file=sys.stderr,
            )
            return 1

        def fetch(destination: Path) -> None:
            for path in sorted(arguments.from_dir.rglob("*")):
                if path.is_file():
                    target = destination / path.relative_to(
                        arguments.from_dir
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)

        manifest = store.stage(
            arguments.venue,
            fetch=fetch,
            source_url=str(arguments.from_dir),
            retrieved_at=datetime.now(UTC).isoformat(),
        )

    problems = store.verify(arguments.venue)
    if problems:
        for problem in problems:
            print(f"FATAL: {problem}", file=sys.stderr)
        return 1
    print(
        f"staged kit {manifest.name}: {len(manifest.files)} file(s), "
        f"{manifest.total_bytes} bytes, manifest {manifest.id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
