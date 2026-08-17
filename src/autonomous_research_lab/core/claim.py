"""Claims and their links to evidence.

A claim never contains its own supporting numbers. Support is expressed as
:class:`EvidenceLink` edges into the evidence store, so that "how well
supported is this?" is a question answered by traversing the graph rather than
by trusting a summary written alongside the claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from .ids import content_id


class ClaimStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    """Supporting and contradicting evidence both exist and neither dominates."""

    REFUTED = "refuted"
    WITHDRAWN = "withdrawn"


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class Claim:
    statement: str
    scope: str = ""
    """The conditions under which the claim is asserted. An unscoped claim
    tends to be over-general, so the field is prominent rather than optional
    in spirit."""

    hypothesis_id: str | None = None
    status: ClaimStatus = ClaimStatus.PROPOSED
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("claim statement must be non-empty")
        if not self.id:
            object.__setattr__(
                self, "id", content_id("clm", self.statement, self.scope)
            )

    def with_status(self, status: ClaimStatus) -> Claim:
        return replace(self, status=status)


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    claim_id: str
    evidence_id: str
    relation: EvidenceRelation
    weight: float = 1.0
    """Strength of the relation. Left as an opaque scalar for now: a principled
    weighting needs the statistician role, which does not exist yet."""

    rationale: str = ""
    id: str = field(default="")

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("evidence link weight must be non-negative")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id("lnk", self.claim_id, self.evidence_id, self.relation),
            )
