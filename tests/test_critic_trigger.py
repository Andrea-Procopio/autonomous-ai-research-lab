"""The critic is event-triggered, and only by scientific reasons: ordinary
results never pay for critique, and engineering trouble never reaches it."""

from __future__ import annotations

from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.orchestration.critic_trigger import CriticTrigger

TRIGGER = CriticTrigger()
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
    assert TRIGGER.reasons(_state(test), test=test) == ()


def test_no_test_means_no_scientific_question_to_review() -> None:
    assert TRIGGER.reasons(_state(), test=None) == ()


def test_contradictory_replications_trigger() -> None:
    first, second = _test("res_1", 0.503), _test("res_2", 0.492)
    reasons = TRIGGER.reasons(_state(first, second), test=second)
    assert any("contradictory replications" in r for r in reasons)


def test_unexpectedly_large_effect_triggers() -> None:
    test = _test("res_1", 5.0)
    reasons = TRIGGER.reasons(_state(test), test=test)
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
    reasons = TRIGGER.reasons(state, test=test)
    assert any("standing challenged" in r for r in reasons)


def test_the_director_may_always_request_review() -> None:
    reasons = TRIGGER.reasons(
        _state(), test=None, director_request="this will decide the branch"
    )
    assert any("director request" in r for r in reasons)


def test_the_trigger_offers_no_engineering_inputs() -> None:
    """Validation failures and repeated execution failures are engineering
    signals handled deterministically by the runtime — the trigger's
    interface cannot even receive them."""
    import inspect

    parameters = inspect.signature(CriticTrigger.reasons).parameters
    assert "validation" not in parameters
    assert "result" not in parameters
    assert set(parameters) == {"self", "state", "test", "director_request"}
