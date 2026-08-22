"""Rendering the approved manuscript for a venue: Task 8D's composition.

The gate is the review: a submission is rendered only from the standing
draft whose faithfulness review is APPROVED — no review, or a standing
REVISE, is a refusal, because an unapproved draft is not the lab's word.

The venue is deployment configuration and never enters a record: a
builtin name or a venue JSON file picks the LaTeX kit, and machine
paths (the kits directory) arrive as explicit arguments, never guessed
from the environment. The submission tree — ``main.tex``,
``references.bib``, and the verified kit files beside them — is written
once, byte-compared forever; re-rendering an unchanged record is a
no-op, and different bytes refuse. The PDF a toolchain makes from that
tree is a **derived artifact**, exempt from write-once (toolchains
embed timestamps), which is exactly why the ``.tex`` is the record and
the PDF is not.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..publication.kits import KitStore, UnknownKitError
from ..publication.latex import VENUES, VenueError, VenueSpec, render_latex, venue_from
from ..publication.manuscript import Manuscript, require_reportable
from ..publication.packet import EvidencePacket
from ..publication.review import (
    NothingToReviewError,
    ReviewRecord,
    ReviewVerdict,
)
from ..publication.store import ManuscriptStore, ReviewStore, head_for
from .packet import build_packet, write_packet

_PACKET = "packet"
_MANUSCRIPT = "manuscript"
_REVIEW = "review"
_SUBMISSION = "submission"

_LOG_TAIL_LINES = 30


class RenderError(RuntimeError):
    """The submission cannot be produced as asked."""


class NotApprovedError(RuntimeError):
    """The standing draft's review does not approve it: a precondition,
    not a fault. Deliberately not a ReviewError subclass, so the CLI's
    refusal tuple stays independent of the failure tuple."""


@dataclass(frozen=True, slots=True)
class RenderRunResult:
    submission_dir: Path
    tex_path: Path
    bib_path: Path
    pdf_path: Path | None
    manuscript: Manuscript
    review: ReviewRecord
    packet: EvidencePacket
    venue: VenueSpec
    replayed: bool


def render_submission(
    root: Path,
    investigation_id: str | None = None,
    *,
    venue: str | None,
    venue_config: Path | None,
    kits_root: Path | None,
    out_dir: Path | None = None,
    pdf: bool = False,
) -> RenderRunResult:
    """One venue submission from the approved record, write-once."""
    packet = build_packet(root, investigation_id)
    require_reportable(packet)
    write_packet(packet, root / _PACKET)

    manuscripts = ManuscriptStore(root / _MANUSCRIPT)
    reviews = ReviewStore(root / _REVIEW)
    heads = head_for(manuscripts, reviews, packet.packet_id)
    if not heads:
        raise NothingToReviewError(
            f"no manuscript exists for packet {packet.packet_id}; "
            f"author first (arl manuscript)"
        )
    head = heads[0]  # multi-head raises AmbiguousHeadError in head_for's callers;
    # here len>1 is the interrupted-cycle state the review verb owns:
    if len(heads) > 1:
        raise RenderError(
            f"{len(heads)} drafts stand for packet {packet.packet_id}; "
            f"run arl review to complete the interrupted cycle first"
        )
    standing = reviews.for_manuscript(head.manuscript_id)
    if not standing:
        raise NotApprovedError(
            f"no review stands for manuscript {head.manuscript_id}; "
            f"review first (arl review)"
        )
    review = standing[0]
    if review.verdict is not ReviewVerdict.APPROVED:
        raise NotApprovedError(
            f"the standing review {review.review_id} is "
            f"{review.verdict} ({len(review.findings)} finding(s)); an "
            f"unapproved draft is not rendered for submission"
        )

    spec = _venue_spec(venue, venue_config)
    kit_files = _kit_files(spec, kits_root)
    main_tex, references_bib = render_latex(packet, head, spec)

    submission_dir = (
        (out_dir if out_dir is not None else root / _SUBMISSION)
        / spec.name
        / head.manuscript_id
    )
    submission_dir.mkdir(parents=True, exist_ok=True)
    written = False
    for relative, content in (
        ("main.tex", main_tex.encode("utf-8")),
        ("references.bib", references_bib.encode("utf-8")),
        *kit_files,
    ):
        target = submission_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != content:
                raise RenderError(
                    f"{target} exists with different content; submission "
                    f"files are never rewritten"
                )
            continue
        target.write_bytes(content)
        written = True

    pdf_path = _compiled(submission_dir) if pdf else None
    return RenderRunResult(
        submission_dir=submission_dir,
        tex_path=submission_dir / "main.tex",
        bib_path=submission_dir / "references.bib",
        pdf_path=pdf_path,
        manuscript=head,
        review=review,
        packet=packet,
        venue=spec,
        replayed=not written,
    )


def _venue_spec(venue: str | None, venue_config: Path | None) -> VenueSpec:
    if (venue is None) == (venue_config is None):
        raise VenueError(
            "name exactly one venue: --venue NAME or --venue-config FILE"
        )
    if venue is not None:
        found = VENUES.get(venue)
        if found is None:
            raise VenueError(
                f"unknown venue {venue!r} (one of: "
                f"{', '.join(sorted(VENUES))}); or pass --venue-config"
            )
        return found
    assert venue_config is not None
    if not venue_config.is_file():
        raise VenueError(f"no venue config at {venue_config}")
    payload = json.loads(venue_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VenueError("a venue config is a JSON object")
    return venue_from(payload)


def _kit_files(
    spec: VenueSpec, kits_root: Path | None
) -> tuple[tuple[str, bytes], ...]:
    """The verified staged kit's files, as (relative_path, bytes) — or
    nothing for a kit-less venue. Machine paths arrive as arguments,
    never guessed: a kit venue without --kits refuses, naming the
    staging command."""
    if not spec.kit:
        return ()
    instruction = (
        f"stage it and pass --kits: python -m examples.stage_venue_kit "
        f"--kits-root DIR --venue {spec.kit} --from-zip <official "
        f"archive> [--sha256 <pinned>]"
    )
    if kits_root is None:
        raise RenderError(
            f"venue {spec.name} needs staged kit {spec.kit!r}; {instruction}"
        )
    store = KitStore(kits_root)
    try:
        manifest = store.manifest(spec.kit)
    except UnknownKitError as error:
        raise RenderError(
            f"kit {spec.kit!r} is not staged under {kits_root}; "
            f"{instruction}"
        ) from error
    problems = store.verify(spec.kit)
    if problems:
        listed = "; ".join(problems[:3])
        raise RenderError(
            f"staged kit {spec.kit!r} does not match its manifest "
            f"({listed}); restage it"
        )
    base = store.path_for(spec.kit)
    return tuple(
        (relative, (base / relative).read_bytes())
        for relative, _, _ in manifest.files
    )


def latex_commands(
    *, latexmk: bool, pdflatex: bool, bibtex: bool
) -> tuple[tuple[str, ...], ...]:
    """The compile plan as pure data — testable without any toolchain."""
    if latexmk:
        return (("latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"),)
    if not pdflatex:
        return ()
    run = ("pdflatex", "-interaction=nonstopmode", "main.tex")
    if bibtex:
        return (run, ("bibtex", "main"), run, run)
    return (run, run)


def _compiled(submission_dir: Path) -> Path:
    """The derived PDF. The operator asked for one, so a missing
    toolchain or a failed compile is a failure, not a shrug."""
    plan = latex_commands(
        latexmk=shutil.which("latexmk") is not None,
        pdflatex=shutil.which("pdflatex") is not None,
        bibtex=shutil.which("bibtex") is not None,
    )
    if not plan:
        raise RenderError(
            "no LaTeX toolchain (latexmk or pdflatex) on PATH; install "
            "MacTeX/TeX Live or drop --pdf — main.tex and references.bib "
            "are written"
        )
    for command in plan:
        completed = subprocess.run(
            command,
            cwd=submission_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join(
                (completed.stdout + completed.stderr).splitlines()[
                    -_LOG_TAIL_LINES:
                ]
            )
            raise RenderError(
                f"{' '.join(command)} exited {completed.returncode}:\n"
                f"{tail}"
            )
    produced = submission_dir / "main.pdf"
    if not produced.is_file():
        raise RenderError(
            "the toolchain exited cleanly but produced no main.pdf"
        )
    return produced
