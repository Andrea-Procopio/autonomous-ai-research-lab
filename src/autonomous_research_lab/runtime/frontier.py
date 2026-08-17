"""The research frontier: what the director actually needs to reason over.

A :class:`ResearchFrontier` is a **derived view** of one
:class:`~autonomous_research_lab.core.state.ResearchState` — never a second
authority. It exists because the director should receive a compact, stable
projection rather than the full raw state: smaller prompts, less repeated
irrelevant information, and one clean seam where context selection can later
become smarter without touching the state.

Three properties are load-bearing:

* **Derived, always.** :func:`build_frontier` is a pure function of the
  state (plus an optional admissibility policy). The frontier has no
  mutators, is never persisted as authority, and carries the id of the
  state it projects so any consumer can go back to the source.
* **Work queues are fact-based.** "This experiment has not been run" is read
  from results and succeeded attempts, exactly as candidate generation
  already does — never from history. In-flight work is excluded so nothing
  is offered twice.
* **Standing is consumed, not decided.** A hypothesis is *settled* here iff
  its current epistemic assessment says SUPPORTED or REFUTED. The frontier
  reads judgments made elsewhere; it makes none.

Scientific standing is governed the same way: the caller may inject an
``admissible`` callback (canonically
:class:`~autonomous_research_lab.runtime.verification_store.
ScientificAdmissibility`), and only scientifically admissible conclusive
tests then resolve predictions or form contradictions. Inadmissible tests
stay fully recorded in the state — the frontier does not hide them, it
declines to *count* them, and it stores no verdict of its own: the
permanent truths remain ``ResearchState`` and the verification store, the
frontier is only a projection of the two. ``None`` (the default) is the
legacy projection in which everything recorded counts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.actions import ResearchActionType
from ..core.assessment import AssessmentVerdict, EpistemicAssessment
from ..core.attempt import ActionAttempt, AttemptStatus
from ..core.budget import ResearchBudget
from ..core.claim import Claim, EvidenceRelation
from ..core.experiment import ExperimentSpec, ResultRef
from ..core.hypothesis import Hypothesis
from ..core.prediction import Consistency, Prediction
from ..core.question import QuestionStatus, ResearchQuestion
from ..core.state import ResearchState

_SETTLED = frozenset({AssessmentVerdict.SUPPORTED, AssessmentVerdict.REFUTED})
_REVISIT = frozenset({AttemptStatus.FAILED, AttemptStatus.TIMED_OUT})

AdmissibilityCheck = Callable[[str], bool]
"""``result_id -> may this result participate in scientific inference?``"""


def _admits(admissible: AdmissibilityCheck | None, result_id: str) -> bool:
    return admissible is None or admissible(result_id)


@dataclass(frozen=True, slots=True)
class Contradiction:
    """A derived flag: something on the record points both ways."""

    subject_kind: str
    """``"prediction"`` (mixed conclusive tests) or ``"claim"`` (evidence
    both supporting and contradicting)."""

    subject_id: str
    detail: str


@dataclass(frozen=True, slots=True)
class ResearchFrontier:
    """A compact projection of one research state, for one deliberation.

    Every field is derived; nothing here may be written back. ``state_id``
    names the state this view was built from, which is the only sense in
    which a frontier is ever "current".
    """

    state_id: str
    objective: str
    open_questions: tuple[ResearchQuestion, ...]
    active_hypotheses: tuple[Hypothesis, ...]
    settled_hypotheses: tuple[Hypothesis, ...]
    hypotheses_without_predictions: tuple[Hypothesis, ...]
    untested_predictions: tuple[Prediction, ...]
    """Predictions with no experiment designed to test them."""

    unresolved_predictions: tuple[Prediction, ...]
    """Predictions with an experiment but no scientifically admissible
    conclusive test yet. A prediction whose only conclusive tests come from
    unverified or invalid results remains scientifically unresolved, even
    though those tests stay on the mechanical record."""

    pending_experiments: tuple[ExperimentSpec, ...]
    """Designed but never run."""

    replication_gaps: tuple[ExperimentSpec, ...]
    """Run at least once, with declared seeds still unused — replication is
    available without new design work."""

    recent_results: tuple[ResultRef, ...]
    unsynthesized_evidence: tuple[str, ...]
    """Evidence ids not yet linked to any claim."""

    unassessed_claims: tuple[Claim, ...]
    contradictions: tuple[Contradiction, ...]
    failed_attempts: tuple[ActionAttempt, ...]
    """Failed or timed-out attempts whose work never subsequently succeeded —
    worth revisiting, or worth explaining."""

    best_findings: tuple[EpistemicAssessment, ...]
    """Current conclusive assessments — what the program currently takes
    itself to have established or ruled out."""

    open_decisions: tuple[str, ...]
    """Orchestrator notes awaiting the director's attention — e.g. the last
    synthesis review's recommendation."""

    remaining_budget: ResearchBudget


def build_frontier(
    state: ResearchState,
    *,
    recent_results: int = 5,
    open_decisions: tuple[str, ...] = (),
    admissible: AdmissibilityCheck | None = None,
) -> ResearchFrontier:
    """Project ``state`` into the view one deliberation needs.

    ``admissible`` governs which recorded results carry scientific weight
    in the projection (prediction resolution, contradictions); ``None``
    projects the ungoverned legacy view in which everything counts.
    """

    def settled(hypothesis_id: str) -> bool:
        assessment = state.current_assessment(hypothesis_id)
        return assessment is not None and assessment.verdict in _SETTLED

    active = tuple(h for h in state.hypotheses if not settled(h.id))
    active_ids = {h.id for h in active}

    without_predictions = tuple(
        h
        for h in active
        if not state.predictions_for(h.id)
        and not state.in_flight(ResearchActionType.DERIVE_PREDICTION, h.id)
    )

    live_predictions = tuple(
        p for p in state.predictions if p.hypothesis_id in active_ids
    )
    untested = tuple(
        p
        for p in live_predictions
        if not state.experiments_for(p.id)
        and not state.in_flight(ResearchActionType.DESIGN_EXPERIMENT, p.id)
    )
    unresolved = tuple(
        p
        for p in live_predictions
        if state.experiments_for(p.id)
        and not any(
            t.consistency is not Consistency.INCONCLUSIVE
            and _admits(admissible, t.result_id)
            for t in state.tests_for(p.id)
        )
    )

    pending = tuple(
        e
        for e in state.experiments
        if not state.results_for(e.id)
        and not state.in_flight(ResearchActionType.RUN_EXPERIMENT, e.id)
    )
    replication_gaps = tuple(
        e
        for e in state.experiments
        if 0 < len(state.results_for(e.id)) < len(e.seeds)
        and not state.in_flight(ResearchActionType.REPLICATE, e.id)
        and not state.in_flight(ResearchActionType.RUN_EXPERIMENT, e.id)
    )

    linked_evidence = {link.evidence_id for link in state.evidence_links}
    unsynthesized = tuple(
        evidence_id
        for evidence_id in state.evidence_ids
        if evidence_id not in linked_evidence
        and not state.in_flight(
            ResearchActionType.SYNTHESIZE_FINDING, evidence_id
        )
    )

    unassessed = tuple(
        c
        for c in state.claims
        if state.current_assessment(c.id) is None
        and not state.in_flight(ResearchActionType.ASSESS_CLAIM, c.id)
    )

    failed = tuple(
        a
        for a in state.attempts
        if a.status in _REVISIT
        and not state.has_succeeded(
            a.action.action_type,
            a.action.targets[0] if a.action.targets else None,
        )
    )

    superseded = {a.supersedes for a in state.assessments if a.supersedes}
    findings = tuple(
        a
        for a in state.assessments
        if a.verdict in _SETTLED and a.id not in superseded
    )

    return ResearchFrontier(
        state_id=state.id,
        objective=state.objective,
        open_questions=tuple(
            q for q in state.questions if q.status is QuestionStatus.OPEN
        ),
        active_hypotheses=active,
        settled_hypotheses=tuple(h for h in state.hypotheses if settled(h.id)),
        hypotheses_without_predictions=without_predictions,
        untested_predictions=untested,
        unresolved_predictions=unresolved,
        pending_experiments=pending,
        replication_gaps=replication_gaps,
        recent_results=state.results[-recent_results:] if recent_results else (),
        unsynthesized_evidence=unsynthesized,
        unassessed_claims=unassessed,
        contradictions=find_contradictions(state, admissible=admissible),
        failed_attempts=failed,
        best_findings=findings,
        open_decisions=open_decisions,
        remaining_budget=state.budget,
    )


def find_contradictions(
    state: ResearchState,
    *,
    admissible: AdmissibilityCheck | None = None,
) -> tuple[Contradiction, ...]:
    """Everything on the record that currently points both ways.

    Purely structural: mixed conclusive prediction tests, and claims with
    evidence linked on both sides. What a contradiction *means* is critic
    business; that one exists is a fact. Under an admissibility policy,
    only scientifically admissible tests can oppose each other — an
    invalid negative next to a verified positive is *not* a scientific
    contradiction, though both stay on the record. (Claim-side links are
    already governed at commit time by the promotion gate.)
    """
    found: list[Contradiction] = []
    for prediction in state.predictions:
        tests = tuple(
            t
            for t in state.tests_for(prediction.id)
            if _admits(admissible, t.result_id)
        )
        consistent = sum(
            1 for t in tests if t.consistency is Consistency.CONSISTENT
        )
        inconsistent = sum(
            1 for t in tests if t.consistency is Consistency.INCONSISTENT
        )
        if consistent and inconsistent:
            found.append(
                Contradiction(
                    subject_kind="prediction",
                    subject_id=prediction.id,
                    detail=(
                        f"{consistent} consistent and {inconsistent} "
                        f"inconsistent test(s) of the same prediction"
                    ),
                )
            )
    for claim in state.claims:
        supports = sum(
            1
            for link in state.evidence_links
            if link.claim_id == claim.id
            and link.relation is EvidenceRelation.SUPPORTS
        )
        contradicts = sum(
            1
            for link in state.evidence_links
            if link.claim_id == claim.id
            and link.relation is EvidenceRelation.CONTRADICTS
        )
        if supports and contradicts:
            found.append(
                Contradiction(
                    subject_kind="claim",
                    subject_id=claim.id,
                    detail=(
                        f"evidence links both support ({supports}) and "
                        f"contradict ({contradicts}) the claim"
                    ),
                )
            )
    return tuple(found)
