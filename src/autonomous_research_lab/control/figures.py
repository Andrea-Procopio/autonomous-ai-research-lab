"""Rendering the figures: the composition root's half of Task 8F.

Order of operations, and what each step guarantees:

1. **The record first.** :func:`~.packet.read_run_checked` resolves the
   investigation, verifies the run from cold, and hands back the same
   reading the packet build uses — the figures verb never draws from a
   record the packet would refuse.
2. **One projection.** :func:`~..publication.figures.planned_figures`
   is the single family-to-figure projection, shared with the packet
   build; what this verb draws is exactly what the packet will mirror.
3. **Replay before matplotlib.** An existing manifest is verified and
   returned with zero rendering work — replaying a finished store needs
   no matplotlib at all. Only a figure that does not exist yet touches
   the renderer, and a missing matplotlib is a refusal naming the
   extra, not a crash.
4. **Hashed at creation.** The first rendering is the record: bytes and
   manifest are written write-once, and every later reader verifies
   digests rather than re-deriving bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..publication.figures import (
    FigureData,
    FigureIntegrityError,
    FigureManifest,
    FigureStore,
    NothingToDrawError,
    figure_id_for,
    planned_figures,
    render_and_manifest,
)
from .packet import read_run_checked

_FIGURES = "figures"


@dataclass(frozen=True, slots=True)
class FiguresRunResult:
    manifests: tuple[FigureManifest, ...]
    rendered: tuple[str, ...]
    replayed: tuple[str, ...]
    store_root: Path


def render_figures(
    root: Path, investigation_id: str | None = None
) -> FiguresRunResult:
    """Every figure the record supports, drawn once, write-once.

    Raises :class:`MissingFactError` (no research state) and
    :class:`NothingToDrawError` (no statistician family with
    observations) and :class:`FiguresUnavailableError` (matplotlib
    missing) as refusals; :class:`PacketError` and
    :class:`FigureError` as failures.
    """
    reading = read_run_checked(root, investigation_id)
    plan = planned_figures(
        reading.head,
        reading.stores.evidence,
        admissible=reading.admissible,
        seed_of=reading.seed_of,
    )
    if not plan:
        raise NothingToDrawError(
            "no claim carries a statistician assessment with "
            "observations; there is nothing trusted code may draw"
        )
    store = FigureStore(root / _FIGURES)
    manifests: list[FigureManifest] = []
    rendered: list[str] = []
    replayed: list[str] = []
    for data in plan:
        manifest, files = _existing_or_drawn(store, data)
        if files is None:
            replayed.append(manifest.figure_id)
        else:
            manifest = store.record(manifest, files)
            rendered.append(manifest.figure_id)
        manifests.append(manifest)
    return FiguresRunResult(
        manifests=tuple(manifests),
        rendered=tuple(rendered),
        replayed=tuple(replayed),
        store_root=store.root,
    )


def _existing_or_drawn(
    store: FigureStore, data: FigureData
) -> tuple[FigureManifest, dict[str, bytes] | None]:
    existing = store.get(figure_id_for(data))
    if existing is not None:
        problems = store.verify(existing.figure_id)
        if problems:
            raise FigureIntegrityError(
                f"figure {existing.figure_id} no longer survives its "
                f"manifest: " + "; ".join(problems)
            )
        return existing, None
    manifest, files = render_and_manifest(
        data,
        rendered_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return manifest, files
