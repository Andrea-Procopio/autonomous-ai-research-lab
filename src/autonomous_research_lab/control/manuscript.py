"""Authoring the manuscript: the composition root's half of Task 8B.

Order of operations, and what each step guarantees:

1. **The packet first.** :func:`~.packet.build_packet` runs the whole
   export discipline — cold verification, citation chain, figure
   re-derivation — so the author never sees a record that does not
   survive itself, and the packet the manuscript cites is written
   durably beside it before any model call.
2. **Replay before spend.** A manuscript's content id includes its
   call's provenance, so identity cannot make re-runs idempotent; the
   store's packet-id lookup does. An existing manuscript for this
   packet is returned with zero provider calls.
3. **The recorded configuration, not the current file.** The model and
   timeout come from the config the investigation recorded, exactly as
   a resumed walk reads them — an operator who edits the file gets the
   run they started. An explicit ``model`` override is allowed because
   the recorded config predates the writing seat; the override is not
   silent — the manuscript's provenance records the requested and the
   served model either way.
4. **The call is the author's own.** No charge reaches the run's grant:
   the run is settled and its packet already states the balance. The
   accepted call's spend is durable in the manuscript record; refused
   drafts and their spend are durable in the store's rejected payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..publication.author import ManuscriptAuthor
from ..publication.manuscript import (
    Manuscript,
    ManuscriptError,
    assemble,
    require_reportable,
)
from ..publication.packet import EvidencePacket
from ..publication.store import (
    AmbiguousHeadError,
    ManuscriptStore,
    ReviewStore,
    head_for,
)
from ..runtime.providers import ModelProvider, UsageLedger
from .config import parse_config
from .investigation import InvestigationStore
from .lab import DefaultLab, Lab
from .packet import build_packet, write_packet
from .stage import StageName

_CONTROL = "control"
_PACKET = "packet"
_MANUSCRIPT = "manuscript"
_REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ManuscriptRunResult:
    manuscript: Manuscript
    packet: EvidencePacket
    json_path: Path
    markdown_path: Path
    replayed: bool


def author_manuscript(
    root: Path,
    investigation_id: str | None = None,
    *,
    lab: Lab | None = None,
    out_dir: Path | None = None,
    model: str | None = None,
) -> ManuscriptRunResult:
    """One manuscript for the investigation's packet, write-once.

    Raises :class:`MissingFactError` (no research state) and
    :class:`NothingToReportError` (no claims) as refusals;
    :class:`PacketError`, :class:`ManuscriptError`, and provider errors
    as failures.
    """
    packet = build_packet(root, investigation_id)
    require_reportable(packet)
    write_packet(packet, root / _PACKET)

    store = ManuscriptStore(out_dir if out_dir is not None else root / _MANUSCRIPT)
    heads = head_for(
        store, ReviewStore(root / _REVIEW), packet.packet_id
    )
    if len(heads) > 1:
        raise AmbiguousHeadError(
            f"an interrupted review cycle left {len(heads)} drafts "
            f"standing for packet {packet.packet_id}; run arl review to "
            f"complete it"
        )
    if heads:
        manuscript = heads[0]
        markdown_path = _written_markdown(store, packet, manuscript)
        return ManuscriptRunResult(
            manuscript=manuscript,
            packet=packet,
            json_path=store.root / f"{manuscript.manuscript_id}.json",
            markdown_path=markdown_path,
            replayed=True,
        )

    provider, seat_model, timeout = publication_seat(
        root, packet, lab=lab, model=model
    )
    author = ManuscriptAuthor(
        provider=provider,
        model=seat_model,
        ledger=UsageLedger(),
        store=store,
        request_timeout_seconds=timeout,
    )
    manuscript = store.record(author.author(packet))
    markdown_path = _written_markdown(store, packet, manuscript)
    return ManuscriptRunResult(
        manuscript=manuscript,
        packet=packet,
        json_path=store.root / f"{manuscript.manuscript_id}.json",
        markdown_path=markdown_path,
        replayed=False,
    )


def publication_seat(
    root: Path,
    packet: EvidencePacket,
    *,
    lab: Lab | None,
    model: str | None,
) -> tuple[ModelProvider, str, float]:
    """The writing seat's provider, model, and timeout: the recorded
    configuration, not the current file, with an explicit model
    override allowed because the recorded config predates the seat —
    never silent, since every record carries requested and served."""
    payload = InvestigationStore(root / _CONTROL).get_config(
        packet.provenance.config_id
    )
    if payload is None:
        raise ManuscriptError(
            f"the recorded config {packet.provenance.config_id} is not "
            f"under this root; the seat takes its model from the record, "
            f"not from a guess"
        )
    config = parse_config(payload)
    provider = (lab if lab is not None else DefaultLab()).model_provider(
        StageName.MANUSCRIPT
    )
    return (
        provider,
        model if model is not None else config.model,
        config.request_timeout_seconds,
    )


def written_markdown(
    store: ManuscriptStore, packet: EvidencePacket, manuscript: Manuscript
) -> Path:
    """The assembled reading copy, write-once by content id."""
    return _written_markdown(store, packet, manuscript)


def _written_markdown(
    store: ManuscriptStore, packet: EvidencePacket, manuscript: Manuscript
) -> Path:
    """The assembled reading copy, write-once by content id."""
    path = store.root / f"{manuscript.manuscript_id}.md"
    document = assemble(packet, manuscript)
    if path.exists():
        if path.read_text(encoding="utf-8") != document:
            raise ManuscriptError(
                f"{path} exists with different content; manuscript files "
                f"are never rewritten"
            )
        return path
    path.write_text(document, encoding="utf-8")
    return path
