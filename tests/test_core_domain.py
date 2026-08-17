from __future__ import annotations

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.budget import (
    InsufficientBudgetError,
    ResearchBudget,
    ResourceCost,
)
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.hypothesis import Hypothesis, HypothesisStatus
from autonomous_research_lab.core.prediction import (
    Comparator,
    Prediction,
    PredictionStatus,
)
from autonomous_research_lab.core.state import ResearchState


def make_hypothesis(statement: str = "X causes Y.") -> Hypothesis:
    return Hypothesis(statement=statement)


def make_prediction(threshold: float = 0.5) -> Prediction:
    return Prediction(
        hypothesis_id="hyp_1",
        condition="under the standard setup",
        metric="effect_size",
        comparator=Comparator.GREATER_THAN,
        threshold=threshold,
    )


class TestPrediction:
    def test_requires_a_metric(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            Prediction(
                hypothesis_id="hyp_1",
                condition="c",
                metric="  ",
                comparator=Comparator.GREATER_THAN,
                threshold=0.5,
            )

    def test_check_is_mechanical(self) -> None:
        prediction = make_prediction(threshold=0.5)
        assert prediction.check(0.6)
        assert not prediction.check(0.4)

    def test_approximately_uses_tolerance(self) -> None:
        prediction = Prediction(
            hypothesis_id="hyp_1",
            condition="c",
            metric="m",
            comparator=Comparator.APPROXIMATELY,
            threshold=1.0,
            tolerance=0.1,
        )
        assert prediction.check(1.05)
        assert not prediction.check(1.2)

    def test_status_change_preserves_identity(self) -> None:
        prediction = make_prediction()
        assert prediction.with_status(PredictionStatus.FAILED).id == prediction.id


class TestHypothesis:
    def test_statement_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Hypothesis(statement="   ")

    def test_status_change_preserves_identity(self) -> None:
        """A falsified hypothesis is the same hypothesis. If its id changed,
        every reference to it — from predictions, claims, assessments —
        would dangle."""
        hypothesis = make_hypothesis()
        assert hypothesis.with_status(HypothesisStatus.FALSIFIED).id == hypothesis.id


class TestExperimentSpec:
    def test_must_declare_metrics(self) -> None:
        with pytest.raises(ValueError, match="at least one metric"):
            ExperimentSpec(
                prediction_id="pred_1", objective="o", procedure="p", metrics=()
            )


class TestResearchState:
    def test_mutation_returns_a_new_state_and_leaves_the_old_one_alone(self) -> None:
        state = ResearchState(objective="Understand Z.")
        evolved = state.upsert_hypothesis(make_hypothesis())

        assert state.hypotheses == ()
        assert len(evolved.hypotheses) == 1
        assert evolved.parent_id == state.id
        assert evolved.id != state.id

    def test_upsert_replaces_in_place_rather_than_appending(self) -> None:
        hypothesis = make_hypothesis()
        state = ResearchState(objective="Understand Z.").upsert_hypothesis(hypothesis)
        updated = state.upsert_hypothesis(
            hypothesis.with_status(HypothesisStatus.FALSIFIED)
        )

        assert len(updated.hypotheses) == 1
        assert updated.hypotheses[0].status is HypothesisStatus.FALSIFIED

    def test_sibling_states_with_different_statuses_are_different_states(self) -> None:
        hypothesis = make_hypothesis()
        base = ResearchState(objective="o").upsert_hypothesis(hypothesis)
        falsified = base.upsert_hypothesis(
            hypothesis.with_status(HypothesisStatus.FALSIFIED)
        )
        supported = base.upsert_hypothesis(
            hypothesis.with_status(HypothesisStatus.SUPPORTED)
        )
        assert falsified.id != supported.id

    def test_history_records_the_actions_taken(self) -> None:
        action = ResearchAction(
            action_type=ResearchActionType.ANALYZE, rationale="read the result"
        )
        state = ResearchState(objective="Understand Z.").apply(action)
        assert state.history == (action,)

    def test_predictions_join_hypotheses_to_experiments(self) -> None:
        hypothesis = make_hypothesis()
        prediction = Prediction(
            hypothesis_id=hypothesis.id,
            condition="c",
            metric="m",
            comparator=Comparator.LESS_THAN,
            threshold=1.0,
        )
        spec = ExperimentSpec(
            prediction_id=prediction.id, objective="o", procedure="p", metrics=("m",)
        )
        state = (
            ResearchState(objective="o")
            .upsert_hypothesis(hypothesis)
            .upsert_prediction(prediction)
            .add_experiment(spec)
        )
        assert state.predictions_for(hypothesis.id) == (prediction,)
        assert state.experiments_for(prediction.id) == (spec,)


class TestBudget:
    def test_spend_is_checked_not_clamped(self) -> None:
        budget = ResearchBudget(wall_clock_seconds=10.0, usd=1.0, model_tokens=100)

        remaining = budget.spend(ResourceCost(wall_clock_seconds=4.0, model_tokens=40))
        assert remaining.wall_clock_seconds == 6.0
        assert remaining.model_tokens == 60

        with pytest.raises(InsufficientBudgetError):
            budget.spend(ResourceCost(usd=5.0))
