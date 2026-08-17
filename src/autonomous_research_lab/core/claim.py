"""Claims and their links to evidence.

A claim is a statement someone is prepared to assert, scoped to the conditions
under which it is asserted. It carries **no status and no numbers**:

* its factual support is the set of :class:`EvidenceLink` edges into the
  evidence store;
* its epistemic standing is the latest
  :class:`~.assessment.EpistemicAssessment` targeting it.

The omission of a status field is deliberate. A status stored on the claim
invites exactly the shortcut this architecture forbids — deciding truth by
counting edges. How an observation *relates* to a claim (below) is a factual
annotation; whether the claim should be believed is a separate judgment with
its own object, author, and rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .ids import content_id


class EvidenceRelation(StrEnum):
    """How one piece of evidence bears on one claim. A description of
    relevance, not a verdict."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"
    """Bears on the claim but neither supports nor contradicts it — an
    underpowered run, a confounded comparison, an ambiguous measurement."""


@dataclass(frozen=True, slots=True)
class Claim:
    statement: str
    scope: str = ""
    """The conditions under which the claim is asserted. An unscoped claim
    tends to be over-general, so the field is prominent rather than optional
    in spirit."""

    hypothesis_id: str | None = None
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("claim statement must be non-empty")
        if not self.id:
            object.__setattr__(
                self, "id", content_id("clm", self.statement, self.scope)
            )


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    claim_id: str
    evidence_id: str
    relation: EvidenceRelation
    rationale: str = ""
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id("lnk", self.claim_id, self.evidence_id, self.relation),
            )
