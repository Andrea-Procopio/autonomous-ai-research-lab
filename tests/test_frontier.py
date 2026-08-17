"""The frontier is a derived view: relevant, compact, and never authoritative."""

from __future__ import annotations

import dataclasses

import pytest

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
from autonomous_research_lab.core.claim import Claim, EvidenceLink, EvidenceRelation
from autonomous_research_lab.core.experiment import (
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
from autonomous_research_lab.runtime.frontier import build_frontier

QUESTION = ResearchQuestion(text="Is the stream fair?")
HYPOTHESIS = Hypothesis(statement="The stream is biased.", question_id=QUESTION.id)
PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="4000 draws",
    metric="heads_rate",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.55,
)
SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="measure",
    procedure="draw and count",
    metrics=("heads_rate",),
    seeds=(1, 2),
)


def _base_state() -> ResearchState:
    return (
        ResearchState(objective="fairness")
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(PREDICTION)
        .add_experiment(SPEC)
    )


def test_frontier_projects_the_relevant_state() -> None:
    state = _base_state()
    frontier = build_frontier(state)

    assert frontier.state_id == state.id
    assert frontier.objective == state.objective
    assert frontier.open_questions == (QUESTION,)
    assert frontier.active_hypotheses == (HYPOTHESIS,)
    # The prediction has an experiment, so it is not untested; the experiment
    # has no result, so it is pending.
    assert frontier.untested_predictions == ()
    assert frontier.pending_experiments == (SPEC,)
    assert frontier.contradictions == ()
    assert frontier.remaining_budget == state.budget


def test_settled_hypotheses_leave_the_active_frontier() -> None:
    state = _base_state().record_assessment(
        EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.REFUTED,
            method="test",
        )
    )
    frontier = build_frontier(state)

    assert frontier.active_hypotheses == ()
    assert frontier.settled_hypotheses == (HYPOTHESIS,)
    assert frontier.best_findings != ()


def test_replication_gap_appears_when_seeds_remain() -> None:
    state = _base_state().record_result(
        ResultRef(
            result_id="res_1", spec_id=SPEC.id, status=ExperimentStatus.COMPLETED
        )
    )
    frontier = build_frontier(state)

    assert frontier.pending_experiments == ()
    assert frontier.replication_gaps == (SPEC,)


def test_contradictory_tests_surface_as_contradictions() -> None:
    state = _base_state()
    for result_id, observed in (("res_a", 0.60), ("res_b", 0.40)):
        state = state.record_prediction_test(
            PredictionTest(
                prediction_id=PREDICTION.id,
                result_id=result_id,
                metric="heads_rate",
                observed=observed,
                consistency=Consistency.CONSISTENT
                if observed >= 0.55
                else Consistency.INCONSISTENT,
            )
        )
    frontier = build_frontier(state)

    (contradiction,) = frontier.contradictions
    assert contradiction.subject_kind == "prediction"
    assert contradiction.subject_id == PREDICTION.id


def test_claim_contradictions_are_derived_from_links() -> None:
    claim = Claim(statement="The stream is biased.", scope="4000 draws")
    state = _base_state().upsert_claim(claim)
    for evidence_id, relation in (
        ("ev_1", EvidenceRelation.SUPPORTS),
        ("ev_2", EvidenceRelation.CONTRADICTS),
    ):
        state = state.link_evidence(
            EvidenceLink(
                claim_id=claim.id, evidence_id=evidence_id, relation=relation
            )
        )
    frontier = build_frontier(state)

    assert any(c.subject_kind == "claim" for c in frontier.contradictions)


def test_failed_attempts_worth_revisiting_are_carried() -> None:
    action = ResearchAction(
        action_type=ResearchActionType.RUN_EXPERIMENT,
        rationale="run it",
        targets=(SPEC.id,),
    )
    attempt = ActionAttempt(action=action).started()
    state = _base_state().begin_attempt(attempt)
    state = state.resolve_attempt(
        attempt.resolved(
            ActionOutcome(status=AttemptStatus.FAILED, error="boom")
        )
    )
    frontier = build_frontier(state)

    assert [a.id for a in frontier.failed_attempts] == [attempt.id]


def test_in_flight_work_is_not_offered_again() -> None:
    action = ResearchAction(
        action_type=ResearchActionType.RUN_EXPERIMENT,
        rationale="run it",
        targets=(SPEC.id,),
    )
    state = _base_state().begin_attempt(ActionAttempt(action=action))
    frontier = build_frontier(state)

    assert frontier.pending_experiments == ()


def test_frontier_is_a_view_not_an_authority() -> None:
    state = _base_state()
    frontier = build_frontier(state)

    # Immutable, with no mutation surface: nothing resembling the state's
    # commit-layer API exists on the projection.
    with pytest.raises(dataclasses.FrozenInstanceError):
        frontier.objective = "rewritten"  # type: ignore[misc]
    for mutator in ("upsert_hypothesis", "record_result", "apply", "_evolve"):
        assert not hasattr(frontier, mutator)

    # Derived, deterministically: same state, same projection; changed state,
    # changed projection — and the state itself is untouched by projection.
    assert build_frontier(state) == frontier
    evolved = state.upsert_hypothesis(
        Hypothesis(statement="Another idea.", question_id=QUESTION.id)
    )
    assert build_frontier(evolved) != frontier
    assert build_frontier(state) == frontier
