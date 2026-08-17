"""The claim-evidence graph.

This is the domain abstraction, not a storage engine. Claims and links live in
:class:`~autonomous_research_lab.core.state.ResearchState`; evidence lives in
the store; this module is the read model that joins them and answers the
questions a research director actually needs to ask:

* which claims are weakly supported?
* which claims does the evidence contradict?
* where is belief running ahead of measurement?

A graph database is not warranted at this size and would fix the schema before
the schema has earned it. The traversal is linear over tuples; when that stops
being adequate, the interface here is what a backing store would implement.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..core.claim import Claim, ClaimStatus, EvidenceLink, EvidenceRelation
from ..core.evidence import Evidence
from ..core.state import ResearchState
from ..evidence.store import EvidenceStore


@dataclass(frozen=True, slots=True)
class ClaimSupport:
    """The evidential standing of a single claim."""

    claim: Claim
    supporting: tuple[Evidence, ...] = ()
    contradicting: tuple[Evidence, ...] = ()
    support_weight: float = 0.0
    contradiction_weight: float = 0.0

    @property
    def net_weight(self) -> float:
        return self.support_weight - self.contradiction_weight

    @property
    def is_unsupported(self) -> bool:
        return not self.supporting and not self.contradicting

    def suggested_status(self, *, support_threshold: float = 1.0) -> ClaimStatus:
        """A deliberately crude reading of the edges.

        It is advisory only -- nothing applies it automatically. Deciding that
        evidence settles a claim is the statistician's and verifier's job, and
        those roles do not exist yet. Encoding a confident rule here now would
        quietly become the system's epistemology.
        """
        if self.is_unsupported:
            return ClaimStatus.PROPOSED
        if self.supporting and self.contradicting:
            return ClaimStatus.CONTESTED
        if self.contradicting:
            return ClaimStatus.REFUTED
        if self.support_weight >= support_threshold:
            return ClaimStatus.SUPPORTED
        return ClaimStatus.PROPOSED


class ClaimEvidenceGraph:
    def __init__(
        self,
        claims: Iterable[Claim],
        links: Iterable[EvidenceLink],
        store: EvidenceStore,
    ) -> None:
        self._claims = {claim.id: claim for claim in claims}
        self._links = tuple(links)
        self._store = store

    @classmethod
    def from_state(
        cls, state: ResearchState, store: EvidenceStore
    ) -> ClaimEvidenceGraph:
        return cls(state.claims, state.evidence_links, store)

    def support_for(self, claim_id: str) -> ClaimSupport:
        claim = self._claims[claim_id]
        supporting: list[Evidence] = []
        contradicting: list[Evidence] = []
        support_weight = 0.0
        contradiction_weight = 0.0
        for link in self._links:
            if link.claim_id != claim_id:
                continue
            if link.relation is EvidenceRelation.SUPPORTS:
                supporting.append(self._store.get_evidence(link.evidence_id))
                support_weight += link.weight
            elif link.relation is EvidenceRelation.CONTRADICTS:
                contradicting.append(self._store.get_evidence(link.evidence_id))
                contradiction_weight += link.weight
        return ClaimSupport(
            claim=claim,
            supporting=tuple(supporting),
            contradicting=tuple(contradicting),
            support_weight=support_weight,
            contradiction_weight=contradiction_weight,
        )

    def all_support(self) -> tuple[ClaimSupport, ...]:
        return tuple(self.support_for(claim_id) for claim_id in self._claims)

    def weakly_supported(self, *, threshold: float = 1.0) -> tuple[ClaimSupport, ...]:
        """Claims whose net evidential weight falls below ``threshold``.

        Includes claims with no evidence at all: an unsupported claim is the
        weakest kind, and it is exactly the kind a fluent system produces most
        easily.
        """
        return tuple(s for s in self.all_support() if s.net_weight < threshold)

    def contradicted(self) -> tuple[ClaimSupport, ...]:
        return tuple(s for s in self.all_support() if s.contradicting)
