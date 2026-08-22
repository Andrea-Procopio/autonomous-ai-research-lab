"""Reviewing the manuscript: the composition root's half of Task 8C.

One bounded cycle, every intermediate state durable, and recovery as
deterministic dispatch on the record — re-running ``arl review``
completes an interrupted cycle without repeating a paid call:

1. **The first review is durable before anything else.** Its grounded
   findings and its spend exist on disk before any revision is
   attempted, so a crash while authoring the revision loses nothing.
2. **Succession is recorded after both of its ends exist.** The
   revision manuscript is recorded, then the :class:`RevisionRecord`
   naming review, superseded draft, and successor. A crash between the
   two leaves a recognizable state — one packet, two drafts, exactly
   one REVISE review without a successor — which the adoption rule
   below repairs; any other multi-head pattern is a hand-edited store
   and a refusal, not a guess.
3. **The cycle is bounded by the record, not by memory.** If any
   revision record exists for the packet, no further draft is ever
   authored: a standing REVISE on the revision replays idempotently,
   and the operator reads its findings rather than paying for a third
   draft nothing in the record demanded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..publication.author import ManuscriptAuthor
from ..publication.manuscript import Manuscript, require_reportable
from ..publication.packet import EvidencePacket
from ..publication.review import (
    NothingToReviewError,
    ReviewRecord,
    ReviewVerdict,
    RevisionRecord,
)
from ..publication.reviewer import FaithfulnessReviewer
from ..publication.store import (
    AmbiguousHeadError,
    ManuscriptStore,
    ReviewStore,
    head_for,
)
from ..runtime.providers import UsageLedger
from .lab import Lab
from .manuscript import publication_seat, written_markdown
from .packet import build_packet, write_packet

_PACKET = "packet"
_MANUSCRIPT = "manuscript"
_REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ReviewRunResult:
    review: ReviewRecord
    manuscript: Manuscript
    packet: EvidencePacket
    opening_review: ReviewRecord | None
    superseded_manuscript: Manuscript | None
    review_path: Path
    manuscript_markdown_path: Path
    replayed: bool


class _Cycle:
    """One run's mutable context: the stores, the seat (built lazily so
    a pure replay makes zero provider constructions), and whether any
    model call happened."""

    def __init__(
        self,
        root: Path,
        packet: EvidencePacket,
        manuscripts: ManuscriptStore,
        reviews: ReviewStore,
        lab: Lab | None,
        model: str | None,
    ) -> None:
        self.root = root
        self.packet = packet
        self.manuscripts = manuscripts
        self.reviews = reviews
        self._lab = lab
        self._model = model
        self.called = False
        self._reviewer: FaithfulnessReviewer | None = None
        self._author: ManuscriptAuthor | None = None

    def reviewer(self) -> FaithfulnessReviewer:
        if self._reviewer is None:
            provider, model, timeout = publication_seat(
                self.root, self.packet, lab=self._lab, model=self._model
            )
            self._reviewer = FaithfulnessReviewer(
                provider=provider,
                model=model,
                ledger=UsageLedger(),
                store=self.reviews,
                request_timeout_seconds=timeout,
            )
        return self._reviewer

    def author(self) -> ManuscriptAuthor:
        if self._author is None:
            provider, model, timeout = publication_seat(
                self.root, self.packet, lab=self._lab, model=self._model
            )
            self._author = ManuscriptAuthor(
                provider=provider,
                model=model,
                ledger=UsageLedger(),
                store=self.manuscripts,
                request_timeout_seconds=timeout,
            )
        return self._author

    def review_of(self, manuscript: Manuscript) -> ReviewRecord:
        existing = self.reviews.for_manuscript(manuscript.manuscript_id)
        if existing:
            return existing[0]
        self.called = True
        return self.reviews.record_review(
            self.reviewer().review(self.packet, manuscript)
        )


def review_manuscript(
    root: Path,
    investigation_id: str | None = None,
    *,
    lab: Lab | None = None,
    out_dir: Path | None = None,
    model: str | None = None,
    revise: bool = True,
) -> ReviewRunResult:
    """One faithfulness review of the packet's standing draft, with at
    most one revise cycle. Raises the packet path's refusals, plus
    :class:`NothingToReviewError` when no draft exists to review.
    """
    packet = build_packet(root, investigation_id)
    require_reportable(packet)
    write_packet(packet, root / _PACKET)
    manuscripts = ManuscriptStore(root / _MANUSCRIPT)
    reviews = ReviewStore(
        out_dir if out_dir is not None else root / _REVIEW
    )
    if not manuscripts.for_packet(packet.packet_id):
        raise NothingToReviewError(
            f"no manuscript exists for packet {packet.packet_id}; "
            f"author first (arl manuscript)"
        )
    cycle = _Cycle(root, packet, manuscripts, reviews, lab, model)

    head = _resolved_head(cycle)
    review = cycle.review_of(head)
    if review.verdict is ReviewVerdict.REVISE and revise:
        already_revised = bool(
            cycle.reviews.revisions_for(packet.packet_id)
        )
        if not already_revised:
            head, review = _revise(cycle, head, review)
    return _result(cycle, head, review)


def _resolved_head(cycle: _Cycle) -> Manuscript:
    """The one standing draft — after repairing the single crash window
    the cycle can leave: a revision authored but its succession not yet
    recorded (the adoption rule)."""
    heads = head_for(
        cycle.manuscripts, cycle.reviews, cycle.packet.packet_id
    )
    if len(heads) == 1:
        return heads[0]
    if len(heads) == 2 and not cycle.reviews.revisions_for(
        cycle.packet.packet_id
    ):
        opening = [
            found
            for found in cycle.reviews.for_packet(cycle.packet.packet_id)
            if found.verdict is ReviewVerdict.REVISE
        ]
        unreviewed = [
            found
            for found in heads
            if not cycle.reviews.for_manuscript(found.manuscript_id)
        ]
        if (
            len(opening) == 1
            and len(unreviewed) == 1
            and opening[0].manuscript_id != unreviewed[0].manuscript_id
            and opening[0].manuscript_id
            in {found.manuscript_id for found in heads}
        ):
            cycle.reviews.record_revision(
                RevisionRecord(
                    packet_id=cycle.packet.packet_id,
                    review_id=opening[0].review_id,
                    superseded_manuscript_id=opening[0].manuscript_id,
                    revision_manuscript_id=unreviewed[0].manuscript_id,
                )
            )
            return unreviewed[0]
    raise AmbiguousHeadError(
        f"{len(heads)} drafts stand for packet "
        f"{cycle.packet.packet_id} and the records do not say which "
        f"succeeded which; this is not a state the lab writes"
    )


def _revise(
    cycle: _Cycle, superseded: Manuscript, opening: ReviewRecord
) -> tuple[Manuscript, ReviewRecord]:
    """The one revise cycle: author from the recorded findings, record
    the succession, review the revision."""
    cycle.called = True
    revision = cycle.manuscripts.record(
        cycle.author().author(
            cycle.packet,
            revision_of=superseded,
            findings=opening.findings,
        )
    )
    written_markdown(cycle.manuscripts, cycle.packet, revision)
    cycle.reviews.record_revision(
        RevisionRecord(
            packet_id=cycle.packet.packet_id,
            review_id=opening.review_id,
            superseded_manuscript_id=superseded.manuscript_id,
            revision_manuscript_id=revision.manuscript_id,
        )
    )
    return revision, cycle.review_of(revision)


def _result(
    cycle: _Cycle, head: Manuscript, review: ReviewRecord
) -> ReviewRunResult:
    opening_review: ReviewRecord | None = None
    superseded: Manuscript | None = None
    revisions = cycle.reviews.revisions_for(cycle.packet.packet_id)
    if revisions and revisions[0].revision_manuscript_id == head.manuscript_id:
        opening_review = cycle.reviews.get_review(revisions[0].review_id)
        superseded = cycle.manuscripts.get(
            revisions[0].superseded_manuscript_id
        )
    return ReviewRunResult(
        review=review,
        manuscript=head,
        packet=cycle.packet,
        opening_review=opening_review,
        superseded_manuscript=superseded,
        review_path=cycle.reviews.root / f"{review.review_id}.json",
        manuscript_markdown_path=(
            cycle.manuscripts.root / f"{head.manuscript_id}.md"
        ),
        replayed=not cycle.called,
    )
