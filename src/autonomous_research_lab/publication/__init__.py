"""Reporting findings.

No longer empty, and the order mattered: the lab learned to measure
before it learned to report. What lives here now is the **evidence
packet** (:mod:`.packet`) — a deterministic, checked projection of one
completed run into a single durable document, where every claim carries
its verdict, its re-derived figures, and the ids of the records and
artifact digests behind it. The packet is what any later writing must
cite; nothing model-authored enters it.

This package deliberately imports nothing from the analysis chain — the
layering tests hold every stage store to its named consumers — so the
schema here is flat data, and the composition root does the reading.

Still deliberately absent: manuscript generation (a model writing prose
from the packet, behind deterministic gates that refuse unknown numbers
and unresolved citations), and the reviewer role that checks a
manuscript against the claim-evidence graph rather than against its own
impression of the prose.
"""

from .packet import (
    EvidencePacket,
    FiguresMismatchError,
    PacketError,
    render_markdown,
    to_json,
)

__all__ = [
    "EvidencePacket",
    "FiguresMismatchError",
    "PacketError",
    "render_markdown",
    "to_json",
]
