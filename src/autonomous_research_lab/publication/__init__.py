"""Reporting findings.

No longer empty, and the order mattered: the lab learned to measure
before it learned to report. What lives here now:

* the **evidence packet** (:mod:`.packet`) — a deterministic, checked
  projection of one completed run into a single durable document, where
  every claim carries its verdict, its re-derived figures, and the ids
  of the records and artifact digests behind it. Nothing model-authored
  enters it;
* the **manuscript** (:mod:`.manuscript`, :mod:`.author`, :mod:`.store`)
  — the first model-authored document, held to the packet by
  deterministic gates: a number is admissible only if trusted code
  already printed it for this packet, a citation only if it names a
  bibliography entry, and the document's structure is not the model's
  to write. Trusted code assembles every load-bearing section from the
  packet's own renderers, byte for byte;
* the **faithfulness review** (:mod:`.review`, :mod:`.reviewer`) — the
  seat that judges whether the prose claims anything the packet does
  not record. Trusted code takes its own reading (forbidden strength,
  unlicensed verdict words); the model's findings must ground
  themselves — verbatim quote, printed record id — or earn a
  corrective call; and the verdict is derived from the findings, never
  asked for. A REVISE demands one revision, recorded as its own
  succession fact.

This package deliberately imports nothing from the analysis chain — the
layering tests hold every stage store to its named consumers — so the
schema here is flat data, and the composition root does the reading.

Still deliberately absent: venue LaTeX kits (8D), the venue simulator
that scores conference-readiness (8E — an impression instrument, kept
separate from this reviewer by design), rendered figures, and signing.
"""

from .author import AUTHOR_INSTRUCTION, PROSE_SCHEMA, ManuscriptAuthor
from .manuscript import (
    PROSE_SECTIONS,
    Manuscript,
    ManuscriptError,
    ManuscriptRejectedError,
    NothingToReportError,
    assemble,
    gate_prose,
    known_renderings,
    require_reportable,
)
from .packet import (
    EvidencePacket,
    FiguresMismatchError,
    PacketError,
    render_markdown,
    to_json,
)
from .review import (
    FORBIDDEN_STRENGTH,
    MODEL_ISSUES,
    REVIEWER_INSTRUCTION,
    NothingToReviewError,
    ReviewError,
    ReviewFinding,
    ReviewRecord,
    ReviewRejectedError,
    ReviewVerdict,
    RevisionRecord,
    derive_verdict,
    deterministic_findings,
    ground_findings,
    review_schema,
)
from .reviewer import FaithfulnessReviewer
from .store import (
    AmbiguousHeadError,
    ManuscriptConflictError,
    ManuscriptIntegrityError,
    ManuscriptStore,
    ReviewConflictError,
    ReviewIntegrityError,
    ReviewStore,
    head_for,
)

__all__ = [
    "AUTHOR_INSTRUCTION",
    "FORBIDDEN_STRENGTH",
    "MODEL_ISSUES",
    "PROSE_SCHEMA",
    "PROSE_SECTIONS",
    "REVIEWER_INSTRUCTION",
    "AmbiguousHeadError",
    "EvidencePacket",
    "FaithfulnessReviewer",
    "FiguresMismatchError",
    "Manuscript",
    "ManuscriptAuthor",
    "ManuscriptConflictError",
    "ManuscriptError",
    "ManuscriptIntegrityError",
    "ManuscriptRejectedError",
    "ManuscriptStore",
    "NothingToReportError",
    "NothingToReviewError",
    "PacketError",
    "ReviewConflictError",
    "ReviewError",
    "ReviewFinding",
    "ReviewIntegrityError",
    "ReviewRecord",
    "ReviewRejectedError",
    "ReviewStore",
    "ReviewVerdict",
    "RevisionRecord",
    "assemble",
    "derive_verdict",
    "deterministic_findings",
    "gate_prose",
    "ground_findings",
    "head_for",
    "known_renderings",
    "render_markdown",
    "require_reportable",
    "review_schema",
    "to_json",
]
