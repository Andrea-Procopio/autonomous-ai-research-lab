"""The transition layer: the only path from a proposal to a state change.

Roles read state and produce proposals; this module validates a proposal
against the current state and evidence store, commits it, and returns the
successor state. Nothing else calls the state's mutator methods (enforced
structurally for ``roles/`` by ``tests/test_layering.py``).

Validation here is referential and mechanical, not epistemic:

* referential — a hypothesis must name a known question (when it names one),
  a prediction a known hypothesis, an experiment a known prediction, an
  assessment a known subject, evidence a recorded result;
* mechanical — committing a result triggers the pre-registered prediction
  check: the observed value is compared against the comparator and threshold
  fixed before the run, and the comparison is recorded as a
  :class:`~autonomous_research_lab.core.prediction.PredictionTest`. Every
  execution yields its own test; nothing is ever marked on the prediction.

What this layer never does is judge: whether a set of prediction tests
*means* anything for a hypothesis is the business of an
:class:`~autonomous_research_lab.core.assessment.EpistemicAssessment`
proposed by whoever is prepared to sign the judgment.

Atomic commits
--------------

:func:`commit` applies one proposal. :func:`commit_bundle` applies the entire
effect of one attempt — proposals plus outcome — as a single transaction:

1. the attempt must exist on the state and be unresolved;
2. every proposal must validate and commit, in order;
3. a succeeded outcome may only claim ``produced`` ids that this bundle
   actually committed (mechanically created prediction tests included);
4. the attempt resolves with the outcome;
5. any violation rejects the whole transition — the caller's state is
   untouched, because states are immutable and the failed transition's
   intermediate states are simply discarded.

The evidence store is append-only and idempotent, so a result recorded during
a transition that later fails is a recorded fact without a referencing state —
harmless, and honest: the process really ran. State membership, not store
membership, is what transitions guarantee atomically.
"""

from __future__ import annotations

from ..core.budget import ResourceCost
from ..core.commit import CommitBundle
from ..core.experiment import ExperimentResult, ExperimentStatus, ResultRef
from ..core.prediction import Consistency, Prediction, PredictionTest
from ..core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    EvidenceProposal,
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    Proposal,
    QuestionProposal,
    ResultProposal,
    payload_ids,
)
from ..core.state import ResearchState
from ..evidence.store import EvidenceStore, UnknownRecordError


class TransitionError(ValueError):
    """Raised when a proposal or bundle cannot be committed onto the state."""


def commit(
    state: ResearchState, proposal: Proposal, store: EvidenceStore
) -> ResearchState:
    """Validate ``proposal`` against ``state`` and ``store`` and apply it."""
    match proposal:
        case QuestionProposal():
            return _commit_question(state, proposal)
        case HypothesisProposal():
            return _commit_hypothesis(state, proposal)
        case PredictionProposal():
            return _commit_prediction(state, proposal)
        case ExperimentProposal():
            return _commit_experiment(state, proposal)
        case ResultProposal():
            return _commit_result(state, proposal, store)
        case EvidenceProposal():
            return _commit_evidence(state, proposal, store)
        case ClaimProposal():
            return _commit_claim(state, proposal, store)
        case AssessmentProposal():
            return _commit_assessment(state, proposal, store)
    raise TransitionError(f"unknown proposal type {type(proposal).__name__}")


def commit_bundle(
    state: ResearchState, bundle: CommitBundle, store: EvidenceStore
) -> ResearchState:
    """Commit the complete effect of one attempt atomically.

    Returns the successor state with every proposal applied and the attempt
    resolved, or raises :class:`TransitionError` leaving ``state`` unchanged.
    The invariant enforced here: **a successful action cannot claim outputs
    that do not exist in the resulting state/store.**
    """
    attempt = next((a for a in state.attempts if a.id == bundle.attempt_id), None)
    if attempt is None:
        raise TransitionError(
            f"bundle names attempt {bundle.attempt_id}, which was never begun "
            f"on this state"
        )
    if attempt.status.is_terminal:
        raise TransitionError(
            f"attempt {attempt.id} is already terminal ({attempt.status})"
        )

    working = state
    committed: set[str] = set()
    for proposal in bundle.proposals:
        working = commit(working, proposal, store)
        committed.update(payload_ids(proposal))
    # Mechanically created objects (prediction tests) count as committed.
    prior_tests = {t.id for t in state.prediction_tests}
    committed.update(
        t.id for t in working.prediction_tests if t.id not in prior_tests
    )

    missing = set(bundle.outcome.produced) - committed
    if missing:
        raise TransitionError(
            f"attempt {attempt.id} resolved {bundle.outcome.status} claiming "
            f"outputs that were not committed: {sorted(missing)}"
        )

    return working.resolve_attempt(attempt.resolved(bundle.outcome))


def reconcile_charge(
    state: ResearchState, cost: ResourceCost
) -> ResearchState:
    """A successor whose budget is ``cost`` smaller, and nothing else.

    The seam recovery charges through. The state and the ledger are two
    records of one number, and a process that died between moving them
    left them disagreeing; this is how the disagreement ends — by moving
    the one that lagged, never by adjusting the one that led.

    Overdrawing is allowed because this is a report, not a request. The
    money is gone whether or not the budget covered it, and a remainder
    clamped at zero would say otherwise.
    """
    return state.charge(cost, allow_overdraw=True)


def _commit_question(
    state: ResearchState, proposal: QuestionProposal
) -> ResearchState:
    question = proposal.question
    if question.parent_id is not None and state.question(question.parent_id) is None:
        raise TransitionError(
            f"question {question.id} references unknown parent {question.parent_id}"
        )
    return state.upsert_question(question)


def _commit_hypothesis(
    state: ResearchState, proposal: HypothesisProposal
) -> ResearchState:
    hypothesis = proposal.hypothesis
    if (
        hypothesis.question_id is not None
        and state.question(hypothesis.question_id) is None
    ):
        raise TransitionError(
            f"hypothesis {hypothesis.id} references unknown question "
            f"{hypothesis.question_id}"
        )
    if (
        hypothesis.parent_id is not None
        and state.hypothesis(hypothesis.parent_id) is None
    ):
        raise TransitionError(
            f"hypothesis {hypothesis.id} refines unknown hypothesis "
            f"{hypothesis.parent_id}"
        )
    return state.upsert_hypothesis(hypothesis)


def _commit_prediction(
    state: ResearchState, proposal: PredictionProposal
) -> ResearchState:
    prediction = proposal.prediction
    if state.hypothesis(prediction.hypothesis_id) is None:
        raise TransitionError(
            f"prediction {prediction.id} references unknown hypothesis "
            f"{prediction.hypothesis_id}"
        )
    return state.upsert_prediction(prediction)


def _commit_experiment(
    state: ResearchState, proposal: ExperimentProposal
) -> ResearchState:
    spec = proposal.spec
    prediction = state.prediction(spec.prediction_id)
    if prediction is None:
        raise TransitionError(
            f"experiment {spec.id} references unknown prediction {spec.prediction_id}"
        )
    if prediction.metric not in spec.metrics:
        raise TransitionError(
            f"experiment {spec.id} does not measure {prediction.metric!r}, "
            f"the metric its prediction is stated in"
        )
    return state.add_experiment(spec)


def _commit_result(
    state: ResearchState, proposal: ResultProposal, store: EvidenceStore
) -> ResearchState:
    result = proposal.result
    spec = state.experiment(result.spec_id)
    if spec is None:
        raise TransitionError(
            f"result {result.id} references unknown experiment {result.spec_id}"
        )
    recorded = store.record_result(result)
    if any(r.result_id == recorded.id for r in state.results):
        return state
    updated = state.record_result(
        ResultRef(
            result_id=recorded.id, spec_id=recorded.spec_id, status=recorded.status
        )
    )

    # Mechanical prediction check: fixed before the run, applied on commit.
    # One test per (prediction, result) — a replication yields its own test,
    # and nothing is ever written onto the prediction.
    prediction = updated.prediction(spec.prediction_id)
    if prediction is not None:
        updated = updated.record_prediction_test(_tested(prediction, recorded))
    return updated


def _tested(prediction: Prediction, result: ExperimentResult) -> PredictionTest:
    observed = result.metrics.get(prediction.metric)
    if result.status is not ExperimentStatus.COMPLETED:
        consistency = Consistency.INCONCLUSIVE
        detail = f"run did not complete: {result.failure_reason or result.status}"
        observed = None
    elif observed is None:
        consistency = Consistency.INCONCLUSIVE
        detail = f"result reported no value for {prediction.metric!r}"
    else:
        held = prediction.check(observed)
        consistency = Consistency.CONSISTENT if held else Consistency.INCONSISTENT
        detail = (
            f"observed {observed} vs {prediction.comparator} "
            f"{prediction.threshold}"
            + (
                f" (tolerance {prediction.tolerance})"
                if prediction.tolerance
                else ""
            )
        )
    return PredictionTest(
        prediction_id=prediction.id,
        result_id=result.id,
        metric=prediction.metric,
        observed=observed,
        consistency=consistency,
        detail=detail,
    )


def _commit_evidence(
    state: ResearchState, proposal: EvidenceProposal, store: EvidenceStore
) -> ResearchState:
    evidence = proposal.evidence
    try:
        recorded = store.record_evidence(evidence)
    except UnknownRecordError as exc:
        raise TransitionError(
            f"evidence {evidence.id} references an unrecorded result"
        ) from exc
    return state.record_evidence(recorded.id)


def _commit_claim(
    state: ResearchState, proposal: ClaimProposal, store: EvidenceStore
) -> ResearchState:
    claim = proposal.claim
    if claim.hypothesis_id is not None and (
        state.hypothesis(claim.hypothesis_id) is None
    ):
        raise TransitionError(
            f"claim {claim.id} references unknown hypothesis {claim.hypothesis_id}"
        )
    for link in proposal.links:
        if link.claim_id != claim.id:
            raise TransitionError(
                f"link {link.id} belongs to claim {link.claim_id}, "
                f"not the proposed claim {claim.id}"
            )
        try:
            store.get_evidence(link.evidence_id)
        except UnknownRecordError as exc:
            raise TransitionError(
                f"link {link.id} references unrecorded evidence {link.evidence_id}"
            ) from exc
    updated = state.upsert_claim(claim)
    for link in proposal.links:
        updated = updated.link_evidence(link)
    return updated


def _commit_assessment(
    state: ResearchState, proposal: AssessmentProposal, store: EvidenceStore
) -> ResearchState:
    assessment = proposal.assessment
    subject_known = (
        state.claim(assessment.subject_id) is not None
        or state.hypothesis(assessment.subject_id) is not None
    )
    if not subject_known:
        raise TransitionError(
            f"assessment {assessment.id} targets unknown subject "
            f"{assessment.subject_id}"
        )
    for evidence_id in assessment.evidence_ids:
        try:
            store.get_evidence(evidence_id)
        except UnknownRecordError as exc:
            raise TransitionError(
                f"assessment {assessment.id} cites unrecorded evidence {evidence_id}"
            ) from exc
    if assessment.supersedes is not None and not any(
        a.id == assessment.supersedes for a in state.assessments
    ):
        raise TransitionError(
            f"assessment {assessment.id} supersedes unknown assessment "
            f"{assessment.supersedes}"
        )
    # Recording the judgment is the whole effect. The subject is a
    # proposition; nothing on it changes.
    return state.record_assessment(assessment)
