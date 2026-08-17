"""The identity invariant:

    identical content        -> identical content id      (semantic objects)
    identical construction   -> distinct occurrence ids   (execution events)

Semantic objects — hypotheses, predictions, specs, claims — are *what is being
said*; saying it twice is saying the same thing, and cross-run comparison of
trajectories depends on that. Events — attempts, jobs, decisions — are *things
that happened*; happening twice is two happenings, and collapsing them would
merge a retry with the run it retries.
"""

from __future__ import annotations

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.attempt import ActionAttempt
from autonomous_research_lab.core.claim import Claim
from autonomous_research_lab.core.decision import DecisionRecord
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.execution.executor import ExperimentJob

ACTION = ResearchAction(action_type=ResearchActionType.ANALYZE, rationale="r")


def make_prediction() -> Prediction:
    return Prediction(
        hypothesis_id="hyp_1",
        condition="c",
        metric="m",
        comparator=Comparator.GREATER_THAN,
        threshold=0.5,
    )


def make_job() -> ExperimentJob:
    return ExperimentJob(
        spec_id="exp_1", command=("python", "run.py"), working_dir="/tmp", seed=3
    )


class TestSemanticIdentity:
    def test_identical_content_shares_an_id(self) -> None:
        assert Hypothesis(statement="X causes Y.").id == Hypothesis(
            statement="X causes Y."
        ).id
        assert make_prediction().id == make_prediction().id
        assert (
            Claim(statement="s", scope="sc").id
            == Claim(statement="s", scope="sc").id
        )

    def test_different_content_differs(self) -> None:
        assert (
            Hypothesis(statement="X causes Y.").id
            != Hypothesis(statement="X causes Z.").id
        )

    def test_spec_identity_is_its_design(self) -> None:
        def spec() -> ExperimentSpec:
            return ExperimentSpec(
                prediction_id="pred_1",
                objective="o",
                procedure="p",
                metrics=("m",),
            )

        assert spec().id == spec().id

    def test_a_prediction_test_inherits_its_results_occurrence(self) -> None:
        """A test is content-addressed, but its content includes the result id
        — so the same comparison on two different executions is two tests,
        while replaying one comparison is idempotent."""

        def test_of(result_id: str) -> PredictionTest:
            return PredictionTest(
                prediction_id="pred_1",
                result_id=result_id,
                metric="m",
                observed=0.6,
                consistency=Consistency.CONSISTENT,
            )

        assert test_of("res_1").id == test_of("res_1").id
        assert test_of("res_1").id != test_of("res_2").id


class TestOccurrenceIdentity:
    def test_identical_attempts_are_distinct_events(self) -> None:
        assert ActionAttempt(action=ACTION).id != ActionAttempt(action=ACTION).id

    def test_identical_jobs_are_distinct_events(self) -> None:
        """Two identically configured executions of one spec are different
        events — a replication is not its original."""
        assert make_job().id != make_job().id

    def test_identical_decisions_are_distinct_events(self) -> None:
        def record() -> DecisionRecord:
            return DecisionRecord(
                state_before_id="st_1",
                evaluated=(),
                selected_action_id=None,
                generator="g",
                evaluator="e",
                policy="p",
            )

        assert record().id != record().id
