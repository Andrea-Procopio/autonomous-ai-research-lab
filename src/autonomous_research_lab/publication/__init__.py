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

* the **venue rendering** (:mod:`.latex`, :mod:`.kits`) — the approved
  draft typeset for a conference. The venue is deployment
  configuration: a builtin spec or a venue JSON picks the LaTeX kit,
  operator-staged and hash-pinned; trusted code renders ``main.tex``
  and ``references.bib`` printing exactly the number strings the
  packet prints; and nothing venue-shaped enters any record.

* the **venue simulator** (:mod:`.simulator`) — the impression
  instrument, kept separate from the faithfulness reviewer by design:
  an ensemble of NeurIPS-form reviews over exactly the rendered
  submission (blind to the record), lens-diverse at temperature zero,
  aggregated by trusted-code medians against a configured bar. The
  model never outputs a verdict; the score is an instrument reading,
  never the objective.

* the **rendered figures** (:mod:`.figures`) — trusted code draws the
  replication families the packet re-derives; the model never sees,
  names, or captions a figure. The numbers are the identity; the bytes
  are hashed at creation into a write-once manifest and verified ever
  after, because matplotlib output varies across versions and the
  first rendering is the record.

Still deliberately absent: full venue-macro fidelity, and signing.
"""

from .author import AUTHOR_INSTRUCTION, PROSE_SCHEMA, ManuscriptAuthor
from .figures import (
    FigureConflictError,
    FigureData,
    FigureError,
    FigureIntegrityError,
    FigureManifest,
    FigureStore,
    FiguresUnavailableError,
    NothingToDrawError,
    StaleFigureError,
    UnknownFigureError,
    compose_caption,
    figure_id_for,
    planned_figures,
    render_and_manifest,
    render_figure,
)
from .kits import (
    KitConflictError,
    KitIntegrityError,
    KitManifest,
    KitStore,
    UnknownKitError,
)
from .latex import (
    VENUES,
    VenueError,
    VenueSpec,
    bibtex_entries,
    escape,
    prose_to_latex,
    render_latex,
    venue_from,
)
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
    RenderedFigure,
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
from .simulator import (
    DIMENSIONS,
    LENSES,
    SIMULATOR_INSTRUCTION,
    PolishRecord,
    SimulationError,
    SimulationRecord,
    SimulationRejectedError,
    VenueReview,
    VenueSimulator,
    aggregate,
    meets,
    review_form_schema,
)
from .store import (
    AmbiguousHeadError,
    ManuscriptConflictError,
    ManuscriptIntegrityError,
    ManuscriptStore,
    ReviewConflictError,
    ReviewIntegrityError,
    ReviewStore,
    SimulationConflictError,
    SimulationIntegrityError,
    SimulationStore,
    head_for,
)

__all__ = [
    "AUTHOR_INSTRUCTION",
    "DIMENSIONS",
    "FORBIDDEN_STRENGTH",
    "LENSES",
    "MODEL_ISSUES",
    "PROSE_SCHEMA",
    "PROSE_SECTIONS",
    "REVIEWER_INSTRUCTION",
    "SIMULATOR_INSTRUCTION",
    "VENUES",
    "AmbiguousHeadError",
    "EvidencePacket",
    "FaithfulnessReviewer",
    "FigureConflictError",
    "FigureData",
    "FigureError",
    "FigureIntegrityError",
    "FigureManifest",
    "FigureStore",
    "FiguresMismatchError",
    "FiguresUnavailableError",
    "KitConflictError",
    "KitIntegrityError",
    "KitManifest",
    "KitStore",
    "Manuscript",
    "ManuscriptAuthor",
    "ManuscriptConflictError",
    "ManuscriptError",
    "ManuscriptIntegrityError",
    "ManuscriptRejectedError",
    "ManuscriptStore",
    "NothingToDrawError",
    "NothingToReportError",
    "NothingToReviewError",
    "PacketError",
    "PolishRecord",
    "RenderedFigure",
    "ReviewConflictError",
    "ReviewError",
    "ReviewFinding",
    "ReviewIntegrityError",
    "ReviewRecord",
    "ReviewRejectedError",
    "ReviewStore",
    "ReviewVerdict",
    "RevisionRecord",
    "SimulationConflictError",
    "SimulationError",
    "SimulationIntegrityError",
    "SimulationRecord",
    "SimulationRejectedError",
    "SimulationStore",
    "StaleFigureError",
    "UnknownFigureError",
    "UnknownKitError",
    "VenueError",
    "VenueReview",
    "VenueSimulator",
    "VenueSpec",
    "aggregate",
    "assemble",
    "bibtex_entries",
    "compose_caption",
    "derive_verdict",
    "deterministic_findings",
    "escape",
    "figure_id_for",
    "gate_prose",
    "ground_findings",
    "head_for",
    "known_renderings",
    "meets",
    "planned_figures",
    "prose_to_latex",
    "render_and_manifest",
    "render_figure",
    "render_latex",
    "render_markdown",
    "require_reportable",
    "review_form_schema",
    "review_schema",
    "to_json",
    "venue_from",
]
