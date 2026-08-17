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
from autonomous_research_lab.core.state import ResearchState


def make_hypothesis(statement: str = "X causes Y.") -> Hypothesis:
    return Hypothesis(
        statement=statement,
        falsification_criterion="Observing Y without X falsifies it.",
    )


def test_hypothesis_requires_a_falsification_criterion() -> None:
    with pytest.raises(ValueError, match="falsification criterion"):
        Hypothesis(statement="X causes Y.", falsification_criterion="  ")


def test_identifiers_are_content_addressed() -> None:
    assert make_hypothesis().id == make_hypothesis().id
    assert make_hypothesis("X causes Z.").id != make_hypothesis().id


def test_status_change_preserves_identity() -> None:
    """A hypothesis under test is the same hypothesis. If its id changed, every
    reference to it -- from experiments, evidence, claims -- would dangle."""
    hypothesis = make_hypothesis()
    assert hypothesis.with_status(HypothesisStatus.FALSIFIED).id == hypothesis.id


def test_experiment_spec_must_declare_metrics() -> None:
    with pytest.raises(ValueError, match="at least one metric"):
        ExperimentSpec(
            hypothesis_id="hyp_1",
            objective="o",
            procedure="p",
            metrics=(),
            falsification_criterion="c",
        )


def test_state_mutation_returns_a_new_state_and_leaves_the_old_one_alone() -> None:
    state = ResearchState(objective="Understand Z.")
    evolved = state.upsert_hypothesis(make_hypothesis())

    assert state.hypotheses == ()
    assert len(evolved.hypotheses) == 1
    assert evolved.parent_id == state.id
    assert evolved.id != state.id


def test_upsert_replaces_in_place_rather_than_appending() -> None:
    hypothesis = make_hypothesis()
    state = ResearchState(objective="Understand Z.").upsert_hypothesis(hypothesis)
    updated = state.upsert_hypothesis(
        hypothesis.with_status(HypothesisStatus.FALSIFIED)
    )

    assert len(updated.hypotheses) == 1
    assert updated.hypotheses[0].status is HypothesisStatus.FALSIFIED


def test_history_records_the_actions_taken() -> None:
    action = ResearchAction(
        action_type=ResearchActionType.ANALYZE, rationale="read the result"
    )
    state = ResearchState(objective="Understand Z.").apply(action)
    assert state.history == (action,)


def test_budget_spend_is_checked_not_clamped() -> None:
    budget = ResearchBudget(wall_clock_seconds=10.0, usd=1.0, model_tokens=100)

    remaining = budget.spend(ResourceCost(wall_clock_seconds=4.0, model_tokens=40))
    assert remaining.wall_clock_seconds == 6.0
    assert remaining.model_tokens == 60

    with pytest.raises(InsufficientBudgetError):
        budget.spend(ResourceCost(usd=5.0))


def test_unestimated_information_gain_is_not_zero() -> None:
    """``None`` means "not estimated". Policies must be able to tell that apart
    from a genuine estimate of no value."""
    action = ResearchAction(
        action_type=ResearchActionType.EXPLORE_ALTERNATIVE,
        rationale="try another route",
    )
    assert action.expected_information_gain is None
