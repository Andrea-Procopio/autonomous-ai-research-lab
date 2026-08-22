"""Rendered figures: trusted code draws what the record proves.

The venue kits' discipline, turned inward: figures are drawn once from
the re-derived replication families — the same families the packet's
figures check re-derives — and the bytes are hashed at creation into a
write-once, content-id-carrying manifest. The identity is the DATA
alone: the family's numbers are the scientific object, and the rendered
bytes are one pinned occurrence, version-dependent the way a staged
kit's ``retrieved_at`` is time-dependent. The bytes stay fully pinned
anyway — their digests live in the manifest and in the packet mirror,
so the packet id covers them.

What is deliberately absent: any model involvement (the model never
sees, names, or requests a figure — captions are trusted text), any
byte re-derivation (matplotlib output varies across versions, so the
first rendering is the record and everything after verifies digests),
and any second write path (a manifest that exists refuses different
bytes rather than absorbing them).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..core.ids import content_id
from ..core.serialize import to_jsonable
from ..core.state import ResearchState
from ..evidence.store import EvidenceStore
from .packet import (
    STATISTICIAN_METHOD,
    PacketError,
    check_statistician_assessment,
    own_predictions,
    replication_family,
)

MANIFESTS_DIRNAME: Final = "manifests"


class FigureError(RuntimeError):
    """A figure cannot honestly be rendered or read."""


class FigureConflictError(FigureError):
    """A figure manifest already exists with different content.
    Manifests are write-once; the first rendering of a family is the
    record, and different bytes under the same id are a new figure
    pretending to be an old one."""


class FigureIntegrityError(FigureError):
    """A figure record that does not survive its own digests."""


class FiguresUnavailableError(FigureError):
    """matplotlib is not installed. Rendering a new figure is the one
    act that needs it; install the ``figures`` extra:
    ``pip install "autonomous-research-lab[figures]"``."""


class NothingToDrawError(FigureError):
    """No claim carries a statistician assessment with observations;
    there is nothing trusted code may draw."""


class UnknownFigureError(KeyError):
    """No figure manifest under that id."""


class StaleFigureError(PacketError):
    """The figure store holds a figure the record no longer derives.
    A missing figure is honest absence; an unexpected one is drift or
    tampering, and the packet refuses it loudly."""


@dataclass(frozen=True, slots=True)
class FigureData:
    """The numbers one figure plots — nothing else is identity."""

    claim_id: str
    prediction_id: str
    metric: str
    comparator: str
    threshold: float
    points: tuple[tuple[int | None, float], ...]
    """``(seed, observed)`` per conclusive admissible test, seed-ordered
    exactly as the replication family orders them."""

    n: int
    """The family size. A conclusive test may observe nothing and still
    count, so ``n`` can exceed the number of points."""

    mean: float | None
    stdev: float | None
    caption: str
    """Trusted-code prose; the model never writes a caption."""

    def __post_init__(self) -> None:
        for name in ("claim_id", "prediction_id", "metric", "comparator"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"a figure needs a non-empty {name}")
        if not self.points:
            raise ValueError("a figure with no points draws nothing")
        if self.n < len(self.points):
            raise ValueError(
                f"a family of n={self.n} cannot yield "
                f"{len(self.points)} observations"
            )
        if not self.caption.strip():
            raise ValueError("a figure needs a caption")


def figure_id_for(data: FigureData) -> str:
    return content_id("fig", json.dumps(to_jsonable(data), sort_keys=True))


def compose_caption(
    *,
    claim_id: str,
    prediction_id: str,
    metric: str,
    comparator: str,
    threshold: float,
    n: int,
    mean: float | None,
    stdev: float | None,
) -> str:
    """The one caption spelling, trusted text with the packet's own
    number formats."""
    if mean is None:
        stats = "mean n/a"
    elif stdev is None:
        stats = f"mean {mean:.6g}"
    else:
        stats = f"mean {mean:.6g}, stdev {stdev:.6g}"
    return (
        f"Replication family for {metric} {comparator} {threshold:g} "
        f"(claim {claim_id}, prediction {prediction_id}): n={n}, "
        f"{stats}. Seed-labeled observations; the dashed line marks "
        f"the pre-registered threshold."
    )


@dataclass(frozen=True, slots=True)
class FigureManifest:
    """One rendered figure: the data that is its identity, and the
    pinned bytes that are its occurrence."""

    data: FigureData
    files: tuple[tuple[str, str, int], ...]
    """``(name, sha256, size_bytes)``, sorted by name — exactly the
    figure's ``.pdf`` and ``.png``."""

    renderer: str
    """What drew the bytes, e.g. ``matplotlib 3.9.2 (agg)``.
    Provenance, deliberately outside the id: the numbers are the
    figure; the renderer explains the pinned occurrence."""

    rendered_at: str
    """ISO 8601, rendering time. Provenance, outside the id — the kit
    manifest's ``retrieved_at`` precedent."""

    figure_id: str = field(default="")

    def __post_init__(self) -> None:
        derived = figure_id_for(self.data)
        expected = (f"{derived}.pdf", f"{derived}.png")
        names = tuple(name for name, _, _ in self.files)
        if names != expected:
            raise ValueError(
                f"a figure manifest names exactly {expected}, sorted; "
                f"got {names}"
            )
        for name, digest, size in self.files:
            if len(digest) != 64 or any(
                c not in "0123456789abcdef" for c in digest
            ):
                raise ValueError(f"file {name!r} needs a sha256 hex digest")
            if size < 0:
                raise ValueError(f"file {name!r} cannot have negative size")
        if not self.renderer.strip():
            raise ValueError("a figure manifest names its renderer")
        if not self.figure_id:
            object.__setattr__(self, "figure_id", derived)
        elif self.figure_id != derived:
            raise FigureIntegrityError(
                f"manifest carries id {self.figure_id}, but its data "
                f"derives {derived}; the record does not survive its "
                f"digests"
            )


def planned_figures(
    state: ResearchState,
    store: EvidenceStore,
    *,
    admissible: Callable[[str], bool],
    seed_of: Mapping[str, int | None],
) -> tuple[FigureData, ...]:
    """Every figure this record supports, in claim order — the single
    family-to-figure projection, shared by the figures verb and the
    packet build so the two cannot drift.

    Claims without a statistician assessment are skipped (their figures
    were never re-derivable); an invalid statistician record propagates
    the figures check's own refusal.
    """
    plans: list[FigureData] = []
    for claim in state.claims:
        assessment = state.current_assessment(claim.id)
        if assessment is None or assessment.method != STATISTICIAN_METHOD:
            continue
        assessed = check_statistician_assessment(
            assessment, state, store, admissible=admissible, seed_of=seed_of
        )
        if not assessed:
            continue
        own = own_predictions(state, claim, store)
        for prediction, (_, stats) in zip(own, assessed, strict=True):
            family = replication_family(
                state, prediction, admissible=admissible, seed_of=seed_of
            )
            points = tuple(
                (seed_of.get(test.result_id), test.observed)
                for test in family
                if test.observed is not None
            )
            if not points:
                continue
            comparator = str(stats.comparator)
            plans.append(
                FigureData(
                    claim_id=claim.id,
                    prediction_id=prediction.id,
                    metric=stats.metric,
                    comparator=comparator,
                    threshold=stats.threshold,
                    points=points,
                    n=stats.n,
                    mean=stats.mean,
                    stdev=stats.stdev,
                    caption=compose_caption(
                        claim_id=claim.id,
                        prediction_id=prediction.id,
                        metric=stats.metric,
                        comparator=comparator,
                        threshold=stats.threshold,
                        n=stats.n,
                        mean=stats.mean,
                        stdev=stats.stdev,
                    ),
                )
            )
    return tuple(plans)


def render_figure(data: FigureData) -> tuple[bytes, bytes]:
    """One strip plot of the replication family, as ``(png, pdf)``.

    Agg, fixed geometry, and metadata suppression are best-effort
    stability only: the discipline is hashed-at-creation, so byte drift
    across matplotlib versions is expected and harmless — it is caught
    only if someone swaps bytes under a recorded manifest.
    """
    try:
        import matplotlib
    except ImportError as error:
        raise FiguresUnavailableError(
            "matplotlib is not installed; rendering needs the figures "
            'extra: pip install "autonomous-research-lab[figures]"'
        ) from error
    matplotlib.use("Agg", force=True)
    import io
    import textwrap

    from matplotlib import pyplot

    values = [value for _, value in data.points]
    labels = [
        "no seed" if seed is None else f"seed {seed}"
        for seed, _ in data.points
    ]
    figure, axes = pyplot.subplots(figsize=(4.8, 3.2))
    try:
        axes.scatter(
            range(len(values)), values, marker="o", color="black", zorder=3
        )
        axes.axhline(data.threshold, linestyle="--", color="gray")
        if data.mean is not None:
            axes.axhline(
                data.mean, linestyle="-", color="black", linewidth=0.8
            )
        axes.set_xticks(range(len(labels)), labels)
        axes.set_ylabel(
            "\n".join(textwrap.wrap(data.metric, width=38)), fontsize=9
        )
        low = min([*values, data.threshold])
        high = max([*values, data.threshold])
        pad = 0.15 * (high - low) or max(1.0, 0.1 * abs(high))
        axes.set_ylim(low - pad, high + pad)
        figure.tight_layout()
        png = io.BytesIO()
        figure.savefig(
            png, format="png", dpi=200, metadata={"Software": None}
        )
        pdf = io.BytesIO()
        figure.savefig(
            pdf,
            format="pdf",
            metadata={
                "Creator": None,
                "Producer": None,
                "CreationDate": None,
            },
        )
    finally:
        pyplot.close(figure)
    return png.getvalue(), pdf.getvalue()


def render_and_manifest(
    data: FigureData, *, rendered_at: str = ""
) -> tuple[FigureManifest, dict[str, bytes]]:
    """Draw one figure and pin its bytes into a manifest."""
    png, pdf = render_figure(data)
    import matplotlib

    figure_id = figure_id_for(data)
    files = {f"{figure_id}.pdf": pdf, f"{figure_id}.png": png}
    manifest = FigureManifest(
        data=data,
        files=tuple(
            (name, _sha256_bytes(content), len(content))
            for name, content in sorted(files.items())
        ),
        renderer=f"matplotlib {matplotlib.__version__} (agg)",
        rendered_at=rendered_at,
    )
    return manifest, files


class FigureStore:
    """Write-once figure manifests beside the rendered files they pin.

    Layout, mounted by the composition root at ``<run root>/figures``::

        <root>/
        ├── fig_<hex>.png
        ├── fig_<hex>.pdf
        └── manifests/
            └── fig_<hex>.json   the manifest, id-carrying, write-once
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def file_path(self, name: str) -> Path:
        return self._root / name

    def record(
        self, manifest: FigureManifest, files: Mapping[str, bytes]
    ) -> FigureManifest:
        """Make one figure durable, once.

        The bytes must re-hash to the manifest they arrive with — a
        manifest that does not describe the bytes it is handed refuses.
        An identical existing manifest is a replay: the stored files
        are verified and the record returned, no bytes written.
        """
        expected = {name: (digest, size) for name, digest, size in manifest.files}
        if set(files) != set(expected):
            raise FigureIntegrityError(
                f"figure {manifest.figure_id}: the bytes name "
                f"{sorted(files)} where the manifest names "
                f"{sorted(expected)}"
            )
        for name, content in files.items():
            digest, size = expected[name]
            if len(content) != size or _sha256_bytes(content) != digest:
                raise FigureIntegrityError(
                    f"figure {manifest.figure_id}: {name} does not hash "
                    f"to the digest its manifest records"
                )
        existing = self.get(manifest.figure_id)
        if existing is not None:
            if existing != manifest:
                raise FigureConflictError(
                    f"figure {manifest.figure_id} is already recorded "
                    f"with different content; figure manifests are "
                    f"never rewritten"
                )
            problems = self.verify(manifest.figure_id)
            if problems:
                raise FigureIntegrityError(
                    f"figure {manifest.figure_id} no longer survives "
                    f"its manifest: " + "; ".join(problems)
                )
            return existing
        self._root.mkdir(parents=True, exist_ok=True)
        for name, content in sorted(files.items()):
            target = self._root / name
            if target.exists():
                if target.read_bytes() != content:
                    raise FigureConflictError(
                        f"{target} exists with different content; "
                        f"figure files are never rewritten"
                    )
                continue
            scratch = self._root / f".staging-{name}"
            scratch.write_bytes(content)
            scratch.replace(target)
        self._write(manifest)
        return manifest

    def get(self, figure_id: str) -> FigureManifest | None:
        path = self._manifest_path(figure_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded = _manifest_from(payload)
        if loaded.figure_id != figure_id:
            raise FigureIntegrityError(
                f"{path} holds manifest {loaded.figure_id}; a record "
                f"filed under the wrong name is not trusted"
            )
        return loaded

    def manifest(self, figure_id: str) -> FigureManifest:
        found = self.get(figure_id)
        if found is None:
            raise UnknownFigureError(figure_id)
        return found

    def manifests(self) -> tuple[FigureManifest, ...]:
        directory = self._root / MANIFESTS_DIRNAME
        if not directory.is_dir():
            return ()
        return tuple(
            self.manifest(path.stem)
            for path in sorted(directory.glob("*.json"))
        )

    def verify(self, figure_id: str, *, deep: bool = True) -> tuple[str, ...]:
        """Every way the stored files disagree with their manifest.
        Returns problems rather than raising, because the caller
        deciding what a problem means is the point of having one."""
        manifest = self.manifest(figure_id)
        problems: list[str] = []
        for name, digest, size in manifest.files:
            path = self._root / name
            if not path.is_file():
                problems.append(f"{name}: missing")
                continue
            actual_size = path.stat().st_size
            if actual_size != size:
                problems.append(
                    f"{name}: {actual_size} bytes where the manifest "
                    f"says {size}"
                )
                continue
            if deep and _sha256_bytes(path.read_bytes()) != digest:
                problems.append(f"{name}: contents no longer match")
        return tuple(problems)

    def _manifest_path(self, figure_id: str) -> Path:
        return self._root / MANIFESTS_DIRNAME / f"{figure_id}.json"

    def _write(self, manifest: FigureManifest) -> None:
        path = self._manifest_path(manifest.figure_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "figure_id": manifest.figure_id,
            "data": to_jsonable(manifest.data),
            "files": [list(entry) for entry in manifest.files],
            "renderer": manifest.renderer,
            "rendered_at": manifest.rendered_at,
        }
        scratch = path.with_suffix(".tmp")
        scratch.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        scratch.replace(path)


def _manifest_from(payload: Mapping[str, object]) -> FigureManifest:
    data = payload["data"]
    if not isinstance(data, Mapping):
        raise FigureIntegrityError("a figure manifest holds a data mapping")
    points = data["points"]
    if not isinstance(points, list):
        raise FigureIntegrityError("figure data holds a points list")
    files = payload["files"]
    if not isinstance(files, list):
        raise FigureIntegrityError("a figure manifest holds a files list")
    mean = data["mean"]
    stdev = data["stdev"]
    return FigureManifest(
        data=FigureData(
            claim_id=str(data["claim_id"]),
            prediction_id=str(data["prediction_id"]),
            metric=str(data["metric"]),
            comparator=str(data["comparator"]),
            threshold=float(str(data["threshold"])),
            points=tuple(
                (None if seed is None else int(str(seed)), float(str(value)))
                for seed, value in points
            ),
            n=int(str(data["n"])),
            mean=None if mean is None else float(str(mean)),
            stdev=None if stdev is None else float(str(stdev)),
            caption=str(data["caption"]),
        ),
        files=tuple(
            (str(name), str(digest), int(str(size)))
            for name, digest, size in files
        ),
        renderer=str(payload["renderer"]),
        rendered_at=str(payload["rendered_at"]),
        figure_id=str(payload["figure_id"]),
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
