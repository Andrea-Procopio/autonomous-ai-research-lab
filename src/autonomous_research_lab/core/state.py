"""The authoritative scientific state of a research program.

Two properties drive the design.

**The state is the record, not the transcript.** Everything a decision depends
on is here as structured data. Nothing important should exist only inside a
model's conversation history.

**The state is immutable and lineage-carrying.** Every mutation returns a new
state whose ``parent_id`` points at its predecessor. That makes the research
trajectory inspectable after the fact, and lets a search policy branch over
states without any component having to defensively copy.

States hold *propositions* — questions, hypotheses, predictions, claims —
and *judgments about them* — assessments. Propositions never carry their own
standing: what is currently believed about a hypothesis or claim is the
latest assessment targeting it (:meth:`ResearchState.current_assessment`),
and what an execution observed about a prediction is the set of
:class:`~.prediction.PredictionTest` records naming it. Neither is a field on
the proposition, so a change of belief can never rewrite what was proposed.

States hold *references* to facts — results and evidence live in the
append-only evidence store, shared across every branch, because a fact does
not become a different fact on a different branch of the search.

Two bookkeeping fields deserve explicit contracts:

``attempts``
    The operational record of action execution. Whether work is done is a
    question about attempts with **succeeded** outcomes
    (:meth:`ResearchState.has_succeeded`) — a failed attempt leaves the work
    open, and a retry is a new attempt.

``history``
    An audit trail of the actions selected, in order. **Never** a source of
    operational truth: nothing may infer completion, or anything else, from an
    action's presence here.

The mutator methods on this class are the commit layer's API. Roles do not
call them — roles produce proposals, and the transition layer in
``orchestration`` validates and commits (enforced structurally by
``tests/test_layering.py``). One mutator sits outside that layer:
:meth:`ResearchState.fund` derives a funded successor of a genesis state,
and belongs to the ``program`` package that authorizes and records the
grant. It is the only mutator that package may call, which the same
structural test pins.

On decomposition: this state is deliberately still one object. The split into
sub-states (scientific / execution / epistemic) is documented as a threshold
decision in ``docs/ARCHITECTURE.md`` and should happen when one of the listed
conditions is met, not before.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, TypeVar

from .actions import ResearchAction, ResearchActionType
from .assessment import EpistemicAssessment
from .attempt import ActionAttempt
from .budget import ResearchBudget, ResourceCost
from .claim import Claim, EvidenceLink
from .experiment import ExperimentSpec, ResultRef
from .hypothesis import Hypothesis
from .ids import content_id
from .prediction import Prediction, PredictionTest
from .question import ResearchQuestion


@dataclass(frozen=True, slots=True)
class ResearchState:
    objective: str
    questions: tuple[ResearchQuestion, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    predictions: tuple[Prediction, ...] = ()
    experiments: tuple[ExperimentSpec, ...] = ()
    results: tuple[ResultRef, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    prediction_tests: tuple[PredictionTest, ...] = ()
    claims: tuple[Claim, ...] = ()
    evidence_links: tuple[EvidenceLink, ...] = ()
    assessments: tuple[EpistemicAssessment, ...] = ()
    attempts: tuple[ActionAttempt, ...] = ()
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
                    tuple((q.id, q.status) for q in self.questions),
                    tuple(h.id for h in self.hypotheses),
                    tuple(p.id for p in self.predictions),
                    tuple(e.id for e in self.experiments),
                    tuple(r.result_id for r in self.results),
                    self.evidence_ids,
                    tuple(t.id for t in self.prediction_tests),
                    tuple(c.id for c in self.claims),
                    tuple(link.id for link in self.evidence_links),
                    tuple(a.id for a in self.assessments),
                    tuple((a.id, a.status) for a in self.attempts),
                    tuple(a.id for a in self.history),
                    self.parent_id,
                ),
            )

    # -- evolution (commit-layer API; roles produce proposals instead) ------

    def _evolve(self, **changes: Any) -> ResearchState:
        """Derive a successor state. The one loosely typed seam in the domain:
        every public method below pins its own field types, so nothing untyped
        reaches a caller."""
        return replace(self, parent_id=self.id, id="", **changes)

    def upsert_question(self, question: ResearchQuestion) -> ResearchState:
        return self._evolve(questions=_upsert(self.questions, question))

    def upsert_hypothesis(self, hypothesis: Hypothesis) -> ResearchState:
        return self._evolve(hypotheses=_upsert(self.hypotheses, hypothesis))

    def upsert_prediction(self, prediction: Prediction) -> ResearchState:
        return self._evolve(predictions=_upsert(self.predictions, prediction))

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

    def record_prediction_test(self, test: PredictionTest) -> ResearchState:
        """Append one mechanical observation-vs-prediction comparison. Tests
        accumulate; a new test never replaces or reinterprets an old one."""
        if any(t.id == test.id for t in self.prediction_tests):
            return self
        return self._evolve(prediction_tests=(*self.prediction_tests, test))

    def upsert_claim(self, claim: Claim) -> ResearchState:
        return self._evolve(claims=_upsert(self.claims, claim))

    def link_evidence(self, link: EvidenceLink) -> ResearchState:
        return self._evolve(evidence_links=_upsert(self.evidence_links, link))

    def record_assessment(self, assessment: EpistemicAssessment) -> ResearchState:
        return self._evolve(assessments=_upsert(self.assessments, assessment))

    def begin_attempt(self, attempt: ActionAttempt) -> ResearchState:
        if attempt.status.is_terminal:
            raise ValueError(f"attempt {attempt.id} began in a terminal status")
        return self._evolve(attempts=_upsert(self.attempts, attempt))

    def resolve_attempt(self, attempt: ActionAttempt) -> ResearchState:
        """Record the terminal form of an attempt begun earlier. The attempt
        keeps its id; the state keeps every attempt, failed ones included."""
        if not attempt.status.is_terminal:
            raise ValueError(f"attempt {attempt.id} is not terminal")
        if not any(a.id == attempt.id for a in self.attempts):
            raise ValueError(f"attempt {attempt.id} was never begun on this state")
        return self._evolve(attempts=_upsert(self.attempts, attempt))

    def apply(self, action: ResearchAction) -> ResearchState:
        """Append to the audit trail. Nothing operational may read ``history``."""
        return self._evolve(history=(*self.history, action))

    def charge(self, cost: ResourceCost) -> ResearchState:
        return self._evolve(budget=self.budget.spend(cost))

    def fund(self, grant: ResearchBudget) -> ResearchState:
        """Derive a successor holding an operator grant -- the counterpart
        to :meth:`charge`.

        Funding is *succession*, never replacement. A state's content id
        deliberately excludes its budget, so overwriting the budget of an
        existing state would leave two different snapshots claiming one id;
        a successor carries a fresh ``parent_id`` and therefore a fresh
        identity. The genesis state a program is admitted with keeps its
        zero budget forever, and the funded state names it as its parent.

        The grant is added, so a first grant and a later top-up are the
        same operation. This method judges nothing: *which* states may be
        funded, by whose authority, and how the grant and every debit
        against it are recorded durably, all belong to the caller.
        """
        return self._evolve(budget=self.budget.plus(grant))

    # -- queries ------------------------------------------------------------

    def question(self, question_id: str) -> ResearchQuestion | None:
        return next((q for q in self.questions if q.id == question_id), None)

    def hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return next((h for h in self.hypotheses if h.id == hypothesis_id), None)

    def prediction(self, prediction_id: str) -> Prediction | None:
        return next((p for p in self.predictions if p.id == prediction_id), None)

    def claim(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.id == claim_id), None)

    def experiment(self, spec_id: str) -> ExperimentSpec | None:
        return next((e for e in self.experiments if e.id == spec_id), None)

    def hypotheses_for(self, question_id: str) -> tuple[Hypothesis, ...]:
        return tuple(h for h in self.hypotheses if h.question_id == question_id)

    def predictions_for(self, hypothesis_id: str) -> tuple[Prediction, ...]:
        return tuple(p for p in self.predictions if p.hypothesis_id == hypothesis_id)

    def experiments_for(self, prediction_id: str) -> tuple[ExperimentSpec, ...]:
        return tuple(e for e in self.experiments if e.prediction_id == prediction_id)

    def results_for(self, spec_id: str) -> tuple[ResultRef, ...]:
        return tuple(r for r in self.results if r.spec_id == spec_id)

    def tests_for(self, prediction_id: str) -> tuple[PredictionTest, ...]:
        """Every mechanical test of this prediction, one per bearing result.
        The caller receives all of them — consistent, inconsistent and
        inconclusive together — because summarizing them is an epistemic act
        this query refuses to perform."""
        return tuple(
            t for t in self.prediction_tests if t.prediction_id == prediction_id
        )

    def test_for_result(
        self, prediction_id: str, result_id: str
    ) -> PredictionTest | None:
        return next(
            (
                t
                for t in self.prediction_tests
                if t.prediction_id == prediction_id and t.result_id == result_id
            ),
            None,
        )

    def attempts_for(self, action_id: str) -> tuple[ActionAttempt, ...]:
        return tuple(a for a in self.attempts if a.action.id == action_id)

    def has_succeeded(
        self, action_type: ResearchActionType, target: str | None = None
    ) -> bool:
        """Whether the work is *done*: some attempt of this action type (on
        ``target``, if given) resolved with a succeeded outcome. Failed
        attempts do not count — that is the point."""
        return any(
            a.succeeded
            and a.action.action_type is action_type
            and (target is None or target in a.action.targets)
            for a in self.attempts
        )

    def in_flight(
        self, action_type: ResearchActionType, target: str | None = None
    ) -> bool:
        """Whether such work is currently queued or running."""
        return any(
            not a.status.is_terminal
            and a.action.action_type is action_type
            and (target is None or target in a.action.targets)
            for a in self.attempts
        )

    def current_assessment(self, subject_id: str) -> EpistemicAssessment | None:
        """The latest assessment targeting ``subject_id``, honouring
        supersession: an assessment that another one names in ``supersedes``
        is no longer current. This query — not any field on the subject — is
        how current epistemic standing is read."""
        superseded = {a.supersedes for a in self.assessments if a.supersedes}
        for assessment in reversed(self.assessments):
            if assessment.subject_id == subject_id and assessment.id not in superseded:
                return assessment
        return None


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
