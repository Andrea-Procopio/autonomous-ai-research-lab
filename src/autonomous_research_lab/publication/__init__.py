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
  packet's own renderers, byte for byte.

This package deliberately imports nothing from the analysis chain — the
layering tests hold every stage store to its named consumers — so the
schema here is flat data, and the composition root does the reading.

Still deliberately absent: the reviewer role that checks a manuscript
against the claim-evidence graph rather than against its own impression
of the prose (8C), rendered figures, and any submission format beyond
markdown.
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
from .store import (
    ManuscriptConflictError,
    ManuscriptIntegrityError,
    ManuscriptStore,
)

__all__ = [
    "AUTHOR_INSTRUCTION",
    "PROSE_SCHEMA",
    "PROSE_SECTIONS",
    "EvidencePacket",
    "FiguresMismatchError",
    "Manuscript",
    "ManuscriptAuthor",
    "ManuscriptConflictError",
    "ManuscriptError",
    "ManuscriptIntegrityError",
    "ManuscriptRejectedError",
    "ManuscriptStore",
    "NothingToReportError",
    "PacketError",
    "assemble",
    "gate_prose",
    "known_renderings",
    "render_markdown",
    "require_reportable",
    "to_json",
]
