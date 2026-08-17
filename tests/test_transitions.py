"""The transition layer: proposals are validated, then committed.

Also the enforcement point for the roles-never-mutate-state invariant: state
changes flow through ``commit`` alone (see ``test_layering.py`` for the
structural half of that guarantee).
"""

from __future__ import annotations

import pytest

from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.claim import Claim, EvidenceLink, EvidenceRelation
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.hypothesis import Hypothesis, HypothesisStatus
from autonomous_research_lab.core.prediction import (
    Comparator,
    Prediction,
    PredictionStatus,
)
from autonomous_research_lab.core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    EvidenceProposal,
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    ResultProposal,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.orchestration.transitions import TransitionError, commit

HYPOTHESIS = Hypothesis(statement="The stream is biased toward heads.")
PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="4000 seeded draws",
    metric="heads_rate",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.55,
)
SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="o",
    procedure="p",
    metrics=("heads_rate", "n_draws"),
)


def base_state() -> ResearchState:
    return ResearchState(objective="o")


def with_experiment() -> tuple[ResearchState, InMemoryEvidenceStore]:
    store = InMemoryEvidenceStore()
    state = base_state()
    state = commit(state, HypothesisProposal(HYPOTHESIS, proposer="t"), store)
    state = commit(state, PredictionProposal(PREDICTION, proposer="t"), store)
    state = commit(state, ExperimentProposal(SPEC, proposer="t"), store)
    return state, store


def result(metrics: dict[str, float] | None = None) -> ExperimentResult:
    return ExperimentResult(
        spec_id=SPEC.id,
        job_id="job_1",
        status=ExperimentStatus.COMPLETED,
        command=("python", "run.py"),
        environment=Environment(python_version="3.11.0", platform="test"),
        metrics=metrics
        if metrics is not None
        else {"heads_rate": 0.51, "n_draws": 4000},
    )


def evidence_for(res: ExperimentResult) -> Evidence:
    return Evidence(
        result_id=res.id,
        spec_id=res.spec_id,
        kind=EvidenceKind.MEASUREMENT,
        observation="heads_rate 0.51 over 4000 draws",
        metrics=dict(res.metrics),
    )


class TestReferentialValidation:
    def test_prediction_requires_known_hypothesis(self) -> None:
        with pytest.raises(TransitionError, match="unknown hypothesis"):
            commit(
                base_state(),
                PredictionProposal(PREDICTION, proposer="t"),
                InMemoryEvidenceStore(),
            )

    def test_experiment_requires_known_prediction(self) -> None:
        with pytest.raises(TransitionError, match="unknown prediction"):
            commit(
                base_state(),
                ExperimentProposal(SPEC, proposer="t"),
                InMemoryEvidenceStore(),
            )

    def test_experiment_must_measure_the_predictions_metric(self) -> None:
        state, store = with_experiment()
        blind_spec = ExperimentSpec(
            prediction_id=PREDICTION.id,
            objective="o2",
            procedure="p2",
            metrics=("runtime_seconds",),
        )
        with pytest.raises(TransitionError, match="does not measure"):
            commit(state, ExperimentProposal(blind_spec, proposer="t"), store)

    def test_result_requires_known_experiment(self) -> None:
        with pytest.raises(TransitionError, match="unknown experiment"):
            commit(
                base_state(),
                ResultProposal(result(), proposer="executor"),
                InMemoryEvidenceStore(),
            )

    def test_evidence_requires_recorded_result(self) -> None:
        state, store = with_experiment()
        with pytest.raises(TransitionError, match="unrecorded result"):
            commit(
                state, EvidenceProposal(evidence_for(result()), proposer="t"), store
            )

    def test_assessment_requires_known_subject(self) -> None:
        orphan = EpistemicAssessment(
            subject_id="clm_missing", verdict=AssessmentVerdict.SUPPORTED, method="m"
        )
        with pytest.raises(TransitionError, match="unknown subject"):
            commit(
                base_state(),
                AssessmentProposal(orphan, proposer="t"),
                InMemoryEvidenceStore(),
            )

    def test_claim_links_must_reference_recorded_evidence(self) -> None:
        state, store = with_experiment()
        claim = Claim(statement="s", hypothesis_id=HYPOTHESIS.id)
        link = EvidenceLink(
            claim_id=claim.id,
            evidence_id="ev_missing",
            relation=EvidenceRelation.SUPPORTS,
        )
        with pytest.raises(TransitionError, match="unrecorded evidence"):
            commit(state, ClaimProposal(claim, links=(link,), proposer="t"), store)


class TestMechanicalConsequences:
    def test_committing_an_experiment_puts_the_hypothesis_under_test(self) -> None:
        state, _ = with_experiment()
        hypothesis = state.hypothesis(HYPOTHESIS.id)
        assert hypothesis is not None
        assert hypothesis.status is HypothesisStatus.UNDER_TEST

    def test_evidence_commit_checks_the_prediction_mechanically(self) -> None:
        """0.51 < 0.55: the pre-registered check fails the prediction, with no
        role expressing an opinion anywhere."""
        state, store = with_experiment()
        res = result()
        state = commit(state, ResultProposal(res, proposer="executor"), store)
        state = commit(state, EvidenceProposal(evidence_for(res), proposer="t"), store)

        prediction = state.prediction(PREDICTION.id)
        assert prediction is not None
        assert prediction.status is PredictionStatus.FAILED
        assert prediction.id == PREDICTION.id  # identity preserved

    def test_prediction_holds_when_the_number_clears_the_threshold(self) -> None:
        state, store = with_experiment()
        res = result({"heads_rate": 0.61, "n_draws": 4000})
        state = commit(state, ResultProposal(res, proposer="executor"), store)
        state = commit(state, EvidenceProposal(evidence_for(res), proposer="t"), store)
        prediction = state.prediction(PREDICTION.id)
        assert prediction is not None
        assert prediction.status is PredictionStatus.HELD

    def test_unmeasurable_prediction_is_indeterminate(self) -> None:
        state, store = with_experiment()
        res = result({"n_draws": 4000.0})
        state = commit(state, ResultProposal(res, proposer="executor"), store)
        state = commit(state, EvidenceProposal(evidence_for(res), proposer="t"), store)
        prediction = state.prediction(PREDICTION.id)
        assert prediction is not None
        assert prediction.status is PredictionStatus.INDETERMINATE

    def test_assessment_on_hypothesis_updates_its_lifecycle_status(self) -> None:
        state, store = with_experiment()
        assessment = EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.REFUTED,
            method="test:v0",
        )
        state = commit(state, AssessmentProposal(assessment, proposer="t"), store)
        hypothesis = state.hypothesis(HYPOTHESIS.id)
        assert hypothesis is not None
        assert hypothesis.status is HypothesisStatus.FALSIFIED
        assert state.current_assessment(HYPOTHESIS.id) == assessment

    def test_superseding_assessment_becomes_current(self) -> None:
        state, store = with_experiment()
        first = EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.PLAUSIBLE,
            method="test:v0",
        )
        state = commit(state, AssessmentProposal(first, proposer="t"), store)
        second = EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.REFUTED,
            method="test:v1",
            supersedes=first.id,
        )
        state = commit(state, AssessmentProposal(second, proposer="t"), store)
        current = state.current_assessment(HYPOTHESIS.id)
        assert current is not None
        assert current.id == second.id
