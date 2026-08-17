"""The claim-evidence graph: a factual read model, not an epistemology.

This module joins claims and links (held in
:class:`~autonomous_research_lab.core.state.ResearchState`) to evidence (held
in the store) and answers *structural* questions:

* what evidence bears on this claim, and how does each piece relate to it?
* which claims does some evidence contradict?
* which claims have never been epistemically assessed?

It deliberately answers no epistemic question. There is no method here that
turns edge counts into a verdict — an earlier draft had an advisory one, and
it was removed because anything that maps edges to a status will end up being
treated as authoritative epistemology no matter what its docstring says.
Whether a claim should be believed is the business of an
:class:`~autonomous_research_lab.core.assessment.EpistemicAssessment`,
which names its method, its evidence, and its confidence, and can be
challenged.

There is no graph database. At this size, traversal over tuples is adequate,
and committing to a storage engine would fix the schema before the schema has
earned it. This interface is what a backing store would eventually implement.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.claim import Claim, EvidenceRelation
from ..core.evidence import Evidence
from ..core.state import ResearchState
from ..evidence.store import EvidenceStore


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    """Everything factual the graph knows about one claim: the evidence
    bearing on it, split by stated relation. No verdict — see the module
    docstring for why none is offered."""

    claim: Claim
    supporting: tuple[Evidence, ...] = ()
    contradicting: tuple[Evidence, ...] = ()
    inconclusive: tuple[Evidence, ...] = ()

    @property
    def has_evidence(self) -> bool:
        return bool(self.supporting or self.contradicting or self.inconclusive)


class ClaimEvidenceGraph:
    def __init__(self, state: ResearchState, store: EvidenceStore) -> None:
        self._state = state
        self._store = store

    @classmethod
    def from_state(
        cls, state: ResearchState, store: EvidenceStore
    ) -> ClaimEvidenceGraph:
        return cls(state, store)

    def evidence_for(self, claim_id: str) -> ClaimEvidence:
        claim = self._state.claim(claim_id)
        if claim is None:
            raise KeyError(claim_id)
        by_relation: dict[EvidenceRelation, list[Evidence]] = {
            relation: [] for relation in EvidenceRelation
        }
        for link in self._state.evidence_links:
            if link.claim_id == claim_id:
                by_relation[link.relation].append(
                    self._store.get_evidence(link.evidence_id)
                )
        return ClaimEvidence(
            claim=claim,
            supporting=tuple(by_relation[EvidenceRelation.SUPPORTS]),
            contradicting=tuple(by_relation[EvidenceRelation.CONTRADICTS]),
            inconclusive=tuple(by_relation[EvidenceRelation.INCONCLUSIVE]),
        )

    def all_claims(self) -> tuple[ClaimEvidence, ...]:
        return tuple(self.evidence_for(claim.id) for claim in self._state.claims)

    def without_evidence(self) -> tuple[Claim, ...]:
        """Claims nothing measured has ever touched — the cheapest kind for a
        fluent system to produce, hence the first place to look."""
        return tuple(c.claim for c in self.all_claims() if not c.has_evidence)

    def contradicted(self) -> tuple[ClaimEvidence, ...]:
        return tuple(c for c in self.all_claims() if c.contradicting)

    def unassessed(self) -> tuple[Claim, ...]:
        """Claims with no current epistemic assessment: belief running ahead
        of judgment."""
        return tuple(
            claim
            for claim in self._state.claims
            if self._state.current_assessment(claim.id) is None
        )
