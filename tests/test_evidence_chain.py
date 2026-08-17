"""The evidence-chain validator walks beliefs back to processes — and says
where the walk breaks."""

from __future__ import annotations

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
    ResultRef,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.evidence.validation import (
    ChainIssueKind,
    validate_evidence_chain,
)

QUESTION = ResearchQuestion(text="Is the stream fair?")
HYPOTHESIS = Hypothesis(statement="The stream is biased.", question_id=QUESTION.id)
PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="4000 draws",
    metric="heads_rate",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.5,
)
SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="measure",
    procedure="draw and count",
    metrics=("heads_rate",),
)


def _result(observed: float) -> ExperimentResult:
    return ExperimentResult(
        spec_id=SPEC.id,
        job_id="job_chain",
        status=ExperimentStatus.COMPLETED,
        command=("run",),
        environment=Environment(python_version="3.11", platform="test"),
        metrics={"heads_rate": observed},
        exit_code=0,
    )


def _grounded() -> tuple[ResearchState, InMemoryEvidenceStore]:
    """A complete, honest chain: question -> ... -> assessed claim."""
    store = InMemoryEvidenceStore()
    result = store.record_result(_result(0.503))
    evidence = store.record_evidence(
        Evidence(
            result_id=result.id,
            spec_id=SPEC.id,
            kind=EvidenceKind.MEASUREMENT,
            observation="heads_rate 0.503",
        )
    )
    claim = Claim(statement=HYPOTHESIS.statement, scope=SPEC.procedure)
    state = (
        ResearchState(objective="fairness")
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(PREDICTION)
        .add_experiment(SPEC)
        .record_result(
            ResultRef(
                result_id=result.id,
                spec_id=SPEC.id,
                status=ExperimentStatus.COMPLETED,
            )
        )
        .record_evidence(evidence.id)
        .record_prediction_test(
            PredictionTest(
                prediction_id=PREDICTION.id,
                result_id=result.id,
                metric="heads_rate",
                observed=0.503,
                consistency=Consistency.CONSISTENT,
            )
        )
        .upsert_claim(claim)
        .link_evidence(
            EvidenceLink(
                claim_id=claim.id,
                evidence_id=evidence.id,
                relation=EvidenceRelation.SUPPORTS,
            )
        )
        .record_assessment(
            EpistemicAssessment(
                subject_id=claim.id,
                verdict=AssessmentVerdict.SUPPORTED,
                method="test",
                evidence_ids=(evidence.id,),
            )
        )
    )
    return state, store


def test_an_honest_chain_raises_no_issues() -> None:
    state, store = _grounded()
    assert validate_evidence_chain(state, store) == ()


def test_dangling_references_are_detected() -> None:
    orphan = Prediction(
        hypothesis_id="hyp_missing",
        condition="c",
        metric="m",
        comparator=Comparator.LESS_THAN,
        threshold=1.0,
    )
    state = ResearchState(objective="x", predictions=(orphan,))
    issues = validate_evidence_chain(state, InMemoryEvidenceStore())
    assert [i.kind for i in issues] == [ChainIssueKind.DANGLING_REFERENCE]


def test_missing_facts_are_detected() -> None:
    state, _ = _grounded()
    empty_store = InMemoryEvidenceStore()  # the facts are gone
    kinds = {i.kind for i in validate_evidence_chain(state, empty_store)}
    assert ChainIssueKind.MISSING_FACT in kinds


def test_tampered_prediction_test_is_detected() -> None:
    """A test whose recorded observation disagrees with the stored result is
    broken provenance — the exact thing this ontology exists to prevent."""
    state, store = _grounded()
    doctored = PredictionTest(
        prediction_id=PREDICTION.id,
        result_id=state.results[0].result_id,
        metric="heads_rate",
        observed=0.99,  # the stored result says 0.503
        consistency=Consistency.CONSISTENT,
    )
    state = state.record_prediction_test(doctored)
    issues = validate_evidence_chain(state, store)
    assert any(i.kind is ChainIssueKind.TAMPERED_TEST for i in issues)


def test_miscomputed_consistency_is_detected() -> None:
    state, store = _grounded()
    flipped = PredictionTest(
        prediction_id=PREDICTION.id,
        result_id=state.results[0].result_id,
        metric="heads_rate",
        observed=0.503,  # matches the store, but 0.503 >= 0.5 is consistent
        consistency=Consistency.INCONSISTENT,
    )
    state = state.record_prediction_test(flipped)
    issues = validate_evidence_chain(state, store)
    assert any(i.kind is ChainIssueKind.TAMPERED_TEST for i in issues)


def test_claims_without_evidence_are_flagged() -> None:
    state, store = _grounded()
    state = state.upsert_claim(
        Claim(statement="An unmeasured assertion.", scope="anywhere")
    )
    issues = validate_evidence_chain(state, store)
    assert any(i.kind is ChainIssueKind.UNSUPPORTED_CLAIM for i in issues)


def test_conclusive_assessments_without_evidence_are_flagged() -> None:
    state, store = _grounded()
    state = state.record_assessment(
        EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.REFUTED,
            method="vibes",
            evidence_ids=(),
        )
    )
    issues = validate_evidence_chain(state, store)
    assert any(i.kind is ChainIssueKind.UNGROUNDED_ASSESSMENT for i in issues)


def test_contradictions_are_surfaced_as_facts() -> None:
    state, store = _grounded()
    second = store.record_result(
        ExperimentResult(
            spec_id=SPEC.id,
            job_id="job_chain_2",
            status=ExperimentStatus.COMPLETED,
            command=("run",),
            environment=Environment(python_version="3.11", platform="test"),
            metrics={"heads_rate": 0.492},
            exit_code=0,
        )
    )
    state = state.record_result(
        ResultRef(
            result_id=second.id,
            spec_id=SPEC.id,
            status=ExperimentStatus.COMPLETED,
        )
    ).record_prediction_test(
        PredictionTest(
            prediction_id=PREDICTION.id,
            result_id=second.id,
            metric="heads_rate",
            observed=0.492,
            consistency=Consistency.INCONSISTENT,
        )
    )
    issues = validate_evidence_chain(state, store)
    assert any(i.kind is ChainIssueKind.CONTRADICTION for i in issues)
