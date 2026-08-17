"""The transition layer: the only path from a proposal to a state change.

Roles read state and produce proposals; this module validates a proposal
against the current state and evidence store, commits it, and returns the
successor state. Nothing else calls the state's mutator methods (enforced
structurally for ``roles/`` by ``tests/test_layering.py``).

Validation here is referential and mechanical, not epistemic:

* referential — a prediction must name a known hypothesis, an experiment a
  known prediction, an assessment a known subject, evidence a recorded result;
* mechanical — committing evidence triggers the pre-registered prediction
  check (a comparison fixed before the run), and committing an assessment
  updates the subject hypothesis's lifecycle status.

What it never does is judge: whether evidence *means* anything is the business
of an :class:`~autonomous_research_lab.core.assessment.EpistemicAssessment`
proposed by whoever is prepared to sign the judgment.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ..core.assessment import AssessmentVerdict
from ..core.experiment import ResultRef
from ..core.hypothesis import HypothesisStatus
from ..core.prediction import Prediction, PredictionStatus
from ..core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    EvidenceProposal,
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    Proposal,
    ResultProposal,
)
from ..core.state import ResearchState
from ..evidence.store import EvidenceStore, UnknownRecordError


class TransitionError(ValueError):
    """Raised when a proposal cannot be committed onto the given state."""


#: How an assessment verdict moves the subject hypothesis's lifecycle status.
#: A cache-maintenance rule, not epistemology: the judgment is the assessment.
_VERDICT_TO_STATUS: Final[dict[AssessmentVerdict, HypothesisStatus | None]] = {
    AssessmentVerdict.SUPPORTED: HypothesisStatus.SUPPORTED,
    AssessmentVerdict.REFUTED: HypothesisStatus.FALSIFIED,
    AssessmentVerdict.CONTESTED: HypothesisStatus.INCONCLUSIVE,
    AssessmentVerdict.UNDETERMINED: None,
    AssessmentVerdict.PLAUSIBLE: None,
}


def commit(
    state: ResearchState, proposal: Proposal, store: EvidenceStore
) -> ResearchState:
    """Validate ``proposal`` against ``state`` and ``store`` and apply it."""
    match proposal:
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


def _commit_hypothesis(
    state: ResearchState, proposal: HypothesisProposal
) -> ResearchState:
    hypothesis = proposal.hypothesis
    if hypothesis.question_id is not None and not any(
        q.id == hypothesis.question_id for q in state.questions
    ):
        raise TransitionError(
            f"hypothesis {hypothesis.id} references unknown question "
            f"{hypothesis.question_id}"
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
    updated = state.add_experiment(spec)
    hypothesis = updated.hypothesis(prediction.hypothesis_id)
    if hypothesis is not None and hypothesis.status is HypothesisStatus.PROPOSED:
        updated = updated.upsert_hypothesis(
            hypothesis.with_status(HypothesisStatus.UNDER_TEST)
        )
    return updated


def _commit_result(
    state: ResearchState, proposal: ResultProposal, store: EvidenceStore
) -> ResearchState:
    result = proposal.result
    if state.experiment(result.spec_id) is None:
        raise TransitionError(
            f"result {result.id} references unknown experiment {result.spec_id}"
        )
    recorded = store.record_result(result)
    return state.record_result(
        ResultRef(
            result_id=recorded.id, spec_id=recorded.spec_id, status=recorded.status
        )
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
    updated = state.record_evidence(recorded.id)

    # Mechanical prediction check: fixed before the run, applied on commit.
    spec = updated.experiment(recorded.spec_id)
    prediction = updated.prediction(spec.prediction_id) if spec else None
    if prediction is not None and prediction.status is PredictionStatus.UNTESTED:
        updated = updated.upsert_prediction(_checked(prediction, recorded.metrics))
    return updated


def _checked(prediction: Prediction, metrics: Mapping[str, float]) -> Prediction:
    observed = metrics.get(prediction.metric)
    if observed is None:
        return prediction.with_status(PredictionStatus.INDETERMINATE)
    held = prediction.check(observed)
    return prediction.with_status(
        PredictionStatus.HELD if held else PredictionStatus.FAILED
    )


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
    subject_is_claim = state.claim(assessment.subject_id) is not None
    subject_hypothesis = state.hypothesis(assessment.subject_id)
    if not subject_is_claim and subject_hypothesis is None:
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

    updated = state.record_assessment(assessment)
    if subject_hypothesis is not None:
        status = _VERDICT_TO_STATUS[assessment.verdict]
        if status is not None and subject_hypothesis.status is not status:
            updated = updated.upsert_hypothesis(
                subject_hypothesis.with_status(status)
            )
    return updated
