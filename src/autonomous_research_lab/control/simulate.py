"""Simulating a venue's reading of the submission: Task 8E's composition.

One bounded polish cycle, every state durable before the next, and
recovery as dispatch on the record — re-running ``arl simulate``
completes an interrupted cycle without repeating a paid call:

1. **Each lens review is durable the moment it returns**, keyed by
   (manuscript, tex digest, lens), so a crash mid-ensemble re-runs only
   the lenses it still owes.
2. **The simulation record is trusted derivation** over recorded
   reviews: re-running with a different bar derives a new record from
   the same reviews with zero model calls.
3. **The polish succession is its own record** — recorded after both
   its ends exist, adopted by the pre-pass when a crash orphaned the
   revision draft, and bounded absolutely: one polish per packet, ever.
   The polish revision must then re-pass the faithfulness review (a
   REVISE there is a typed stop — presentation polish does not outrank
   the record) before it is re-rendered and re-scored once.

The score is an instrument reading, never the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..publication.author import ManuscriptAuthor
from ..publication.manuscript import require_reportable
from ..publication.packet import EvidencePacket
from ..publication.review import ReviewRecord, ReviewVerdict
from ..publication.simulator import (
    LENSES,
    PolishRecord,
    SimulationError,
    SimulationRecord,
    VenueReview,
    VenueSimulator,
    aggregate,
    meets,
)
from ..publication.store import (
    AmbiguousHeadError,
    ManuscriptStore,
    ReviewStore,
    SimulationStore,
    head_for,
)
from ..runtime.implementation_store import sha256_text
from ..runtime.providers import UsageLedger
from .lab import Lab
from .manuscript import publication_seat, written_markdown
from .packet import build_packet, write_packet
from .render import RenderRunResult, render_submission
from .review import review_manuscript

_PACKET = "packet"
_MANUSCRIPT = "manuscript"
_REVIEW = "review"
_SIMULATION = "simulation"


class PolishRejectedError(SimulationError):
    """The polish revision failed the faithfulness review; its findings
    and the standing venue verdicts are recorded."""


@dataclass(frozen=True, slots=True)
class SimulateRunResult:
    simulation: SimulationRecord
    reviews: tuple[VenueReview, ...]
    render: RenderRunResult
    opening_simulation: SimulationRecord | None
    polish: PolishRecord | None
    faithfulness: ReviewRecord | None
    replayed: bool


def simulate_submission(
    root: Path,
    investigation_id: str | None = None,
    *,
    venue: str | None,
    venue_config: Path | None,
    kits_root: Path | None,
    lab: Lab | None = None,
    model: str | None = None,
    reviews: int = 3,
    bar: int = 6,
    polish: bool = True,
    out_dir: Path | None = None,
) -> SimulateRunResult:
    """One venue reading of the standing approved submission, with at
    most one polish cycle."""
    if not 1 <= reviews <= len(LENSES):
        raise SimulationError(
            f"reviews must be in 1..{len(LENSES)}: each lens is a "
            f"distinct deterministic perspective, and the same lens "
            f"twice at temperature zero buys nothing"
        )
    if not 1 <= bar <= 10:
        raise SimulationError(f"bar must be in 1..10, got {bar}")

    packet = build_packet(root, investigation_id)
    require_reportable(packet)
    write_packet(packet, root / _PACKET)
    manuscripts = ManuscriptStore(root / _MANUSCRIPT)
    review_store = ReviewStore(root / _REVIEW)
    simulations = SimulationStore(root / _SIMULATION)

    cycle = _Cycle(
        root=root,
        investigation_id=investigation_id,
        packet_id=packet.packet_id,
        manuscripts=manuscripts,
        review_store=review_store,
        simulations=simulations,
        lab=lab,
        model=model,
        venue=venue,
        venue_config=venue_config,
        kits_root=kits_root,
        out_dir=out_dir,
        reviews=reviews,
        bar=bar,
    )
    _adopt_orphan_polish(cycle, packet)

    first = _pass(cycle, packet)
    if (
        first.simulation.meets_bar
        or not polish
        or simulations.polishes_for(packet.packet_id)
    ):
        return first
    second = _polish(cycle, packet, first)
    return second


class _Cycle:
    """One run's context: stores, the seat (built lazily so pure
    replays construct zero providers), and whether any call happened."""

    def __init__(
        self,
        *,
        root: Path,
        investigation_id: str | None,
        packet_id: str,
        manuscripts: ManuscriptStore,
        review_store: ReviewStore,
        simulations: SimulationStore,
        lab: Lab | None,
        model: str | None,
        venue: str | None,
        venue_config: Path | None,
        kits_root: Path | None,
        out_dir: Path | None,
        reviews: int,
        bar: int,
    ) -> None:
        self.root = root
        self.investigation_id = investigation_id
        self.packet_id = packet_id
        self.manuscripts = manuscripts
        self.review_store = review_store
        self.simulations = simulations
        self.lab = lab
        self.model = model
        self.venue = venue
        self.venue_config = venue_config
        self.kits_root = kits_root
        self.out_dir = out_dir
        self.reviews = reviews
        self.bar = bar
        self.called = False
        self._simulator: VenueSimulator | None = None
        self._author: ManuscriptAuthor | None = None

    def simulator(self, packet: EvidencePacket) -> VenueSimulator:
        if self._simulator is None:
            provider, model, timeout = publication_seat(
                self.root, packet, lab=self.lab, model=self.model
            )
            self._simulator = VenueSimulator(
                provider=provider,
                model=model,
                ledger=UsageLedger(),
                store=self.simulations,
                request_timeout_seconds=timeout,
            )
        return self._simulator

    def author(self, packet: EvidencePacket) -> ManuscriptAuthor:
        if self._author is None:
            provider, model, timeout = publication_seat(
                self.root, packet, lab=self.lab, model=self.model
            )
            self._author = ManuscriptAuthor(
                provider=provider,
                model=model,
                ledger=UsageLedger(),
                store=self.manuscripts,
                request_timeout_seconds=timeout,
            )
        return self._author


def _adopt_orphan_polish(cycle: _Cycle, packet: EvidencePacket) -> None:
    """Crash window W4: a polish draft was recorded but its succession
    was not. Recognizable exactly: two heads, no polish record, one
    head APPROVED with a standing below-bar simulation, the other
    reviewed by nothing at all. Anything else is not a state this lab
    writes, and the ordinary multi-head refusals speak."""
    heads = head_for(
        cycle.manuscripts,
        cycle.review_store,
        cycle.packet_id,
        cycle.simulations,
    )
    if len(heads) != 2 or cycle.simulations.polishes_for(cycle.packet_id):
        return
    approved_below_bar = [
        found
        for found in heads
        if (standing := cycle.review_store.for_manuscript(found.manuscript_id))
        and standing[0].verdict is ReviewVerdict.APPROVED
        and any(
            not simulation.meets_bar
            for simulation in cycle.simulations.for_manuscript(
                found.manuscript_id
            )
        )
    ]
    unreviewed = [
        found
        for found in heads
        if not cycle.review_store.for_manuscript(found.manuscript_id)
        and not cycle.simulations.for_manuscript(found.manuscript_id)
    ]
    if (
        len(approved_below_bar) == 1
        and len(unreviewed) == 1
        and approved_below_bar[0].manuscript_id
        != unreviewed[0].manuscript_id
    ):
        opening = next(
            simulation
            for simulation in cycle.simulations.for_manuscript(
                approved_below_bar[0].manuscript_id
            )
            if not simulation.meets_bar
        )
        cycle.simulations.record_polish(
            PolishRecord(
                packet_id=cycle.packet_id,
                simulation_id=opening.simulation_id,
                superseded_manuscript_id=approved_below_bar[0].manuscript_id,
                revision_manuscript_id=unreviewed[0].manuscript_id,
            )
        )
        written_markdown(cycle.manuscripts, packet, unreviewed[0])
        return
    raise AmbiguousHeadError(
        f"{len(heads)} drafts stand for packet {cycle.packet_id} and "
        f"the records do not say which succeeded which; this is not a "
        f"state the lab writes"
    )


def _pass(cycle: _Cycle, packet: EvidencePacket) -> SimulateRunResult:
    """Score the standing head once: faithfulness first when the head
    is an unreviewed polish revision, then render, then the ensemble,
    then the trusted aggregate."""
    faithfulness: ReviewRecord | None = None
    heads = head_for(
        cycle.manuscripts,
        cycle.review_store,
        cycle.packet_id,
        cycle.simulations,
    )
    polishes = cycle.simulations.polishes_for(cycle.packet_id)
    if (
        len(heads) == 1
        and polishes
        and heads[0].manuscript_id == polishes[0].revision_manuscript_id
    ):
        standing = cycle.review_store.for_manuscript(
            heads[0].manuscript_id
        )
        if standing:
            faithfulness = standing[0]
        else:
            outcome = review_manuscript(
                cycle.root,
                cycle.investigation_id,
                lab=cycle.lab,
                model=cycle.model,
                revise=False,
            )
            if not outcome.replayed:
                cycle.called = True
            faithfulness = outcome.review
        if faithfulness.verdict is not ReviewVerdict.APPROVED:
            raise PolishRejectedError(
                f"the polish revision failed the faithfulness review "
                f"({faithfulness.review_id}, "
                f"{len(faithfulness.findings)} finding(s)); "
                f"presentation polish does not outrank the record, and "
                f"the standing venue verdicts remain"
            )

    render = render_submission(
        cycle.root,
        cycle.investigation_id,
        venue=cycle.venue,
        venue_config=cycle.venue_config,
        kits_root=cycle.kits_root,
        out_dir=cycle.out_dir,
    )
    main_tex = render.tex_path.read_text(encoding="utf-8")
    references_bib = render.bib_path.read_text(encoding="utf-8")
    digest = sha256_text(main_tex)

    ensemble: list[VenueReview] = []
    recorded = cycle.simulations.reviews_for(
        render.manuscript.manuscript_id, digest
    )
    by_lens = {found.lens: found for found in recorded}
    for lens, focus in LENSES[: cycle.reviews]:
        existing = by_lens.get(lens)
        if existing is not None:
            ensemble.append(existing)
            continue
        cycle.called = True
        ensemble.append(
            cycle.simulations.record_review(
                cycle.simulator(packet).review(
                    main_tex=main_tex,
                    references_bib=references_bib,
                    venue_name=render.venue.name,
                    anonymous=render.venue.anonymous,
                    lens=lens,
                    focus=focus,
                    manuscript_id=render.manuscript.manuscript_id,
                    packet_id=cycle.packet_id,
                    tex_sha256=digest,
                )
            )
        )
    medians = aggregate(ensemble)
    overall = next(value for name, value in medians if name == "overall")
    simulation = cycle.simulations.record_simulation(
        SimulationRecord(
            manuscript_id=render.manuscript.manuscript_id,
            packet_id=cycle.packet_id,
            venue_name=render.venue.name,
            tex_sha256=digest,
            bar=cycle.bar,
            review_ids=tuple(found.review_id for found in ensemble),
            medians=medians,
            meets_bar=meets(cycle.bar, overall),
        )
    )
    return SimulateRunResult(
        simulation=simulation,
        reviews=tuple(ensemble),
        render=render,
        opening_simulation=None,
        polish=None,
        faithfulness=faithfulness,
        replayed=not cycle.called,
    )


def _polish(
    cycle: _Cycle, packet: EvidencePacket, first: SimulateRunResult
) -> SimulateRunResult:
    """The one polish cycle: revise from the recorded weaknesses,
    record the succession, then score the revision once."""
    notes = tuple(
        f"{review.lens}: {weakness}"
        for review in first.reviews
        for weakness in review.weaknesses
    ) or tuple(
        f"{review.lens}: overall {review.overall}/10 — {review.summary}"
        for review in first.reviews
    )
    cycle.called = True
    revision = cycle.manuscripts.record(
        cycle.author(packet).author(
            packet,
            revision_of=first.render.manuscript,
            polish_notes=notes,
        )
    )
    written_markdown(cycle.manuscripts, packet, revision)
    polish_record = cycle.simulations.record_polish(
        PolishRecord(
            packet_id=cycle.packet_id,
            simulation_id=first.simulation.simulation_id,
            superseded_manuscript_id=first.render.manuscript.manuscript_id,
            revision_manuscript_id=revision.manuscript_id,
        )
    )
    second = _pass(cycle, packet)
    return SimulateRunResult(
        simulation=second.simulation,
        reviews=second.reviews,
        render=second.render,
        opening_simulation=first.simulation,
        polish=polish_record,
        faithfulness=second.faithfulness,
        replayed=False,
    )
