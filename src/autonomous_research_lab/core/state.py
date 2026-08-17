"""The authoritative scientific state of a research program.

Two properties drive the design.

**The state is the record, not the transcript.** Everything a decision depends
on is here as structured data. Nothing important should exist only inside a
model's conversation history.

**The state is immutable and lineage-carrying.** Every mutation returns a new
state whose ``parent_id`` points at its predecessor. That makes the research
trajectory inspectable after the fact, and lets a search policy branch over
states without any component having to defensively copy.

States hold *beliefs* -- questions, hypotheses, claims, links. They hold
*references* to facts -- results and evidence live in the append-only evidence
store, shared across every branch, because a fact does not become a different
fact on a different branch of the search.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, TypeVar

from .actions import ResearchAction
from .budget import ResearchBudget, ResourceCost
from .claim import Claim, EvidenceLink
from .experiment import ExperimentSpec, ResultRef
from .hypothesis import Hypothesis
from .ids import content_id
from .question import ResearchQuestion


@dataclass(frozen=True, slots=True)
class ResearchState:
    objective: str
    questions: tuple[ResearchQuestion, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    experiments: tuple[ExperimentSpec, ...] = ()
    results: tuple[ResultRef, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = ()
    evidence_links: tuple[EvidenceLink, ...] = ()
    budget: ResearchBudget = field(default_factory=ResearchBudget.zero)
    history: tuple[ResearchAction, ...] = ()
    parent_id: str | None = None
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "st",
                    self.objective,
                    tuple(q.id for q in self.questions),
                    tuple(h.id for h in self.hypotheses),
                    tuple(e.id for e in self.experiments),
                    tuple(r.result_id for r in self.results),
                    self.evidence_ids,
                    tuple(c.id for c in self.claims),
                    tuple(link.id for link in self.evidence_links),
                    tuple(a.id for a in self.history),
                    self.parent_id,
                ),
            )

    # -- evolution ---------------------------------------------------------

    def _evolve(self, **changes: Any) -> ResearchState:
        """Derive a successor state. The one loosely typed seam in the domain:
        every public method below pins its own field types, so nothing untyped
        reaches a caller."""
        return replace(self, parent_id=self.id, id="", **changes)

    def upsert_question(self, question: ResearchQuestion) -> ResearchState:
        return self._evolve(questions=_upsert(self.questions, question))

    def upsert_hypothesis(self, hypothesis: Hypothesis) -> ResearchState:
        return self._evolve(hypotheses=_upsert(self.hypotheses, hypothesis))

    def add_experiment(self, spec: ExperimentSpec) -> ResearchState:
        return self._evolve(experiments=_upsert(self.experiments, spec))

    def record_result(self, ref: ResultRef) -> ResearchState:
        if any(r.result_id == ref.result_id for r in self.results):
            return self
        return self._evolve(results=(*self.results, ref))

    def record_evidence(self, evidence_id: str) -> ResearchState:
        if evidence_id in self.evidence_ids:
            return self
        return self._evolve(evidence_ids=(*self.evidence_ids, evidence_id))

    def upsert_claim(self, claim: Claim) -> ResearchState:
        return self._evolve(claims=_upsert(self.claims, claim))

    def link_evidence(self, link: EvidenceLink) -> ResearchState:
        return self._evolve(evidence_links=_upsert(self.evidence_links, link))

    def apply(self, action: ResearchAction) -> ResearchState:
        """Record that ``action`` was taken. Does not charge the budget: the
        estimate that justified the action is not the cost that was paid."""
        return self._evolve(history=(*self.history, action))

    def charge(self, cost: ResourceCost) -> ResearchState:
        return self._evolve(budget=self.budget.spend(cost))

    # -- queries -----------------------------------------------------------

    def hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return next((h for h in self.hypotheses if h.id == hypothesis_id), None)

    def claim(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.id == claim_id), None)

    def experiments_for(self, hypothesis_id: str) -> tuple[ExperimentSpec, ...]:
        return tuple(e for e in self.experiments if e.hypothesis_id == hypothesis_id)

    def results_for(self, spec_id: str) -> tuple[ResultRef, ...]:
        return tuple(r for r in self.results if r.spec_id == spec_id)


class _HasId(Protocol):
    @property
    def id(self) -> str: ...


_T = TypeVar("_T", bound=_HasId)


def _upsert(items: tuple[_T, ...], item: _T) -> tuple[_T, ...]:
    """Replace an item carrying the same id, or append it."""
    for index, existing in enumerate(items):
        if existing.id == item.id:
            return (*items[:index], item, *items[index + 1 :])
    return (*items, item)
