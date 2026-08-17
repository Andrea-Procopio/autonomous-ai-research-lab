"""The critic is event-triggered: ordinary results never pay for critique."""

from __future__ import annotations

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.attempt import (
    ActionAttempt,
    ActionOutcome,
    AttemptStatus,
)
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.orchestration.critic_trigger import CriticTrigger
from autonomous_research_lab.runtime.validation import (
    ValidationCheck,
    ValidationReport,
)

TRIGGER = CriticTrigger()
PASSING = ValidationReport(checks=())
HYPOTHESIS = Hypothesis(statement="The stream is biased.")
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


def _result() -> ExperimentResult:
    return ExperimentResult(
        spec_id=SPEC.id,
        job_id="job_t",
        status=ExperimentStatus.COMPLETED,
        command=("run",),
        environment=Environment(python_version="3.11", platform="test"),
        metrics={"heads_rate": 0.503},
        exit_code=0,
    )


def _test(result_id: str, observed: float) -> PredictionTest:
    return PredictionTest(
        prediction_id=PREDICTION.id,
        result_id=result_id,
        metric="heads_rate",
        observed=observed,
        consistency=Consistency.CONSISTENT
        if observed >= 0.5
        else Consistency.INCONSISTENT,
    )


def _state(*tests: PredictionTest) -> ResearchState:
    state = (
        ResearchState(objective="fairness")
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(PREDICTION)
        .add_experiment(SPEC)
    )
    for test in tests:
        state = state.record_prediction_test(test)
    return state


def test_ordinary_result_does_not_trigger_the_critic() -> None:
    test = _test("res_1", 0.503)
    reasons = TRIGGER.reasons(
        _state(test), result=_result(), validation=PASSING, test=test
    )
    assert reasons == ()


def test_contradictory_replications_trigger() -> None:
    first, second = _test("res_1", 0.503), _test("res_2", 0.492)
    reasons = TRIGGER.reasons(
        _state(first, second),
        result=_result(),
        validation=PASSING,
        test=second,
    )
    assert any("contradictory replications" in r for r in reasons)


def test_unexpectedly_large_effect_triggers() -> None:
    test = _test("res_1", 5.0)
    reasons = TRIGGER.reasons(
        _state(test), result=_result(), validation=PASSING, test=test
    )
    assert any("unexpectedly large effect" in r for r in reasons)


def test_challenge_to_settled_standing_triggers() -> None:
    test = _test("res_2", 0.492)  # inconsistent
    state = _state(test).record_assessment(
        EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.SUPPORTED,
            method="prior",
        )
    )
    reasons = TRIGGER.reasons(
        state, result=_result(), validation=PASSING, test=test
    )
    assert any("standing challenged" in r for r in reasons)


def test_repeated_failures_trigger() -> None:
    state = _state()
    for _ in range(2):
        attempt = ActionAttempt(
            action=ResearchAction(
                action_type=ResearchActionType.RUN_EXPERIMENT,
                rationale="try",
                targets=(SPEC.id,),
            )
        ).started()
        state = state.begin_attempt(attempt).resolve_attempt(
            attempt.resolved(
                ActionOutcome(status=AttemptStatus.FAILED, error="boom")
            )
        )
    reasons = TRIGGER.reasons(
        state, result=_result(), validation=PASSING, test=None
    )
    assert any("repeated failures" in r for r in reasons)


def test_validation_problems_on_a_completed_run_trigger() -> None:
    failing = ValidationReport(
        checks=(
            ValidationCheck(name="metrics_finite", passed=False, detail="nan"),
        )
    )
    reasons = TRIGGER.reasons(
        _state(), result=_result(), validation=failing, test=None
    )
    assert any("implementation uncertainty" in r for r in reasons)


def test_the_director_may_always_request_review() -> None:
    reasons = TRIGGER.reasons(
        _state(),
        result=_result(),
        validation=PASSING,
        test=None,
        director_request="this will decide the branch",
    )
    assert any("director request" in r for r in reasons)
