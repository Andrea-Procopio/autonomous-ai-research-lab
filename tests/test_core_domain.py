from __future__ import annotations

from dataclasses import replace

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.budget import (
    InsufficientBudgetError,
    ResearchBudget,
    ResourceCost,
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

    def test_a_prediction_is_a_proposition_and_carries_no_status(self) -> None:
        """The empirical record about a prediction lives in PredictionTest
        objects, one per bearing execution — never on the proposition."""
        prediction = make_prediction()
        assert not hasattr(prediction, "status")
        assert not hasattr(prediction, "with_status")


class TestPredictionTest:
    def test_conclusive_test_requires_an_observation(self) -> None:
        with pytest.raises(ValueError, match="observed value"):
            PredictionTest(
                prediction_id="pred_1",
                result_id="res_1",
                metric="m",
                observed=None,
                consistency=Consistency.CONSISTENT,
            )

    def test_contradictory_tests_of_one_prediction_coexist(self) -> None:
        """Run 1 consistent, run 2 inconsistent, run 3 inconclusive: three
        facts, three records, no verdict anywhere."""

        def test_for(result_id: str, consistency: Consistency) -> PredictionTest:
            observed = None if consistency is Consistency.INCONCLUSIVE else 0.6
            return PredictionTest(
                prediction_id="pred_1",
                result_id=result_id,
                metric="m",
                observed=observed,
                consistency=consistency,
            )

        state = ResearchState(objective="o")
        state = state.record_prediction_test(test_for("res_1", Consistency.CONSISTENT))
        state = state.record_prediction_test(
            test_for("res_2", Consistency.INCONSISTENT)
        )
        state = state.record_prediction_test(
            test_for("res_3", Consistency.INCONCLUSIVE)
        )

        tests = state.tests_for("pred_1")
        assert [t.consistency for t in tests] == [
            Consistency.CONSISTENT,
            Consistency.INCONSISTENT,
            Consistency.INCONCLUSIVE,
        ]

    def test_recording_the_same_test_twice_is_idempotent(self) -> None:
        test = PredictionTest(
            prediction_id="pred_1",
            result_id="res_1",
            metric="m",
            observed=0.6,
            consistency=Consistency.CONSISTENT,
        )
        state = ResearchState(objective="o").record_prediction_test(test)
        assert state.record_prediction_test(test) is state


class TestHypothesis:
    def test_statement_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Hypothesis(statement="   ")

    def test_a_hypothesis_is_a_proposition_and_carries_no_status(self) -> None:
        """Current standing is the latest assessment targeting the hypothesis
        (ResearchState.current_assessment), never a field on it."""
        hypothesis = make_hypothesis()
        assert not hasattr(hypothesis, "status")
        assert not hasattr(hypothesis, "with_status")


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
        """Content id ignores the rationale, so a re-proposed hypothesis with
        a better rationale replaces the original rather than duplicating it."""
        hypothesis = make_hypothesis()
        refined = Hypothesis(statement=hypothesis.statement, rationale="sharper")
        assert refined.id == hypothesis.id

        state = ResearchState(objective="Understand Z.").upsert_hypothesis(hypothesis)
        updated = state.upsert_hypothesis(refined)

        assert len(updated.hypotheses) == 1
        assert updated.hypotheses[0].rationale == "sharper"

    def test_states_with_different_tests_are_different_states(self) -> None:
        base = ResearchState(objective="o")
        consistent = base.record_prediction_test(
            PredictionTest(
                prediction_id="pred_1",
                result_id="res_1",
                metric="m",
                observed=0.6,
                consistency=Consistency.CONSISTENT,
            )
        )
        inconsistent = base.record_prediction_test(
            PredictionTest(
                prediction_id="pred_1",
                result_id="res_1",
                metric="m",
                observed=0.4,
                consistency=Consistency.INCONSISTENT,
            )
        )
        assert consistent.id != inconsistent.id

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

    def test_a_grant_is_added_to_what_remains(self) -> None:
        budget = ResearchBudget(wall_clock_seconds=10.0, usd=1.0, model_tokens=100)

        topped_up = budget.plus(ResearchBudget(usd=4.0, model_tokens=50))

        assert topped_up.usd == 5.0
        assert topped_up.model_tokens == 150
        assert topped_up.wall_clock_seconds == 10.0
        assert budget.usd == 1.0  # the original is untouched


class TestFunding:
    """The genesis state an admission produces carries no budget, and a
    state's content id deliberately excludes the budget. Funding must
    therefore be succession, never replacement -- pinned here from both
    sides: the tempting shortcut, and the supported operation."""

    def test_replacing_the_budget_keeps_the_id_and_changes_the_bytes(self) -> None:
        genesis = ResearchState(objective="o", budget=ResearchBudget.zero())

        shortcut = replace(
            genesis, budget=ResearchBudget(usd=100.0, model_tokens=10_000)
        )

        # The identity is unchanged because the budget is not part of it,
        # and `replace` carries the populated id straight through. Two
        # different states now claim one id -- which is exactly what an
        # append-only, content-addressed snapshot store must refuse.
        assert shortcut.id == genesis.id
        assert shortcut.budget != genesis.budget

    def test_funding_derives_a_successor_with_its_own_identity(self) -> None:
        genesis = ResearchState(objective="o", budget=ResearchBudget.zero())

        funded = genesis.fund(ResearchBudget(usd=100.0, model_tokens=10_000))

        assert funded.id != genesis.id
        assert funded.parent_id == genesis.id
        assert funded.budget == ResearchBudget(usd=100.0, model_tokens=10_000)
        assert genesis.budget.is_exhausted  # the genesis state is untouched

    def test_a_grant_adds_to_a_budget_already_held(self) -> None:
        funded = ResearchState(objective="o").fund(ResearchBudget(usd=10.0))

        topped_up = funded.fund(ResearchBudget(usd=5.0))

        assert topped_up.budget.usd == 15.0
        assert topped_up.parent_id == funded.id

    def test_the_scientific_content_survives_funding_unchanged(self) -> None:
        hypothesis = make_hypothesis()
        genesis = ResearchState(objective="o").upsert_hypothesis(hypothesis)

        funded = genesis.fund(ResearchBudget(usd=10.0))

        assert funded.hypotheses == genesis.hypotheses
        assert funded.questions == genesis.questions
        assert funded.predictions == genesis.predictions
