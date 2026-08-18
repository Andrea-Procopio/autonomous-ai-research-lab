"""The deterministic planning director: fixed priority, durable dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autonomous_research_lab.core.actions import ResearchActionType
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.orchestration.planning import (
    DEFAULT_PLAN_COST,
    PlanningDirector,
)
from autonomous_research_lab.runtime.frontier import ResearchFrontier
from autonomous_research_lab.runtime.planning_store import (
    PlanningAction,
    PlanningRecord,
    PlanningStore,
    StopReason,
)

SPEC = ExperimentSpec(
    prediction_id="pred_1",
    objective="measure",
    procedure="run the design",
    metrics=("m",),
    seeds=(7, 11),
    estimated_cost=ResourceCost(wall_clock_seconds=120.0),
)


def _frontier(**overrides: Any) -> ResearchFrontier:
    values: dict[str, Any] = {
        "state_id": "st_x",
        "objective": "objective",
        "open_questions": (),
        "active_hypotheses": (),
        "settled_hypotheses": (),
        "hypotheses_without_predictions": (),
        "untested_predictions": (),
        "unresolved_predictions": (),
        "pending_experiments": (),
        "replication_gaps": (),
        "recent_results": (),
        "unsynthesized_evidence": (),
        "unassessed_claims": (),
        "contradictions": (),
        "failed_attempts": (),
        "best_findings": (),
        "open_decisions": (),
        "remaining_budget": ResearchBudget(
            wall_clock_seconds=1000.0, usd=5.0, model_tokens=100_000
        ),
    }
    values.update(overrides)
    return ResearchFrontier(**values)


def _record(**overrides: Any) -> PlanningRecord:
    values: dict[str, Any] = {
        "invocation_id": "inv_1",
        "action": PlanningAction.NEW_EXPERIMENT,
        "question_id": "q_1",
        "rationale": "probe robustness",
        "evidence_ids": ("ev_1",),
        "spec_id": SPEC.id,
        "response_id": "mcall_1",
    }
    values.update(overrides)
    return PlanningRecord(**values)


def _director(tmp_path: Path) -> tuple[PlanningDirector, PlanningStore]:
    plans = PlanningStore(tmp_path / "planning")
    return PlanningDirector(plans=plans), plans


def test_an_open_stop_decision_becomes_the_halt_action(tmp_path: Path) -> None:
    director, plans = _director(tmp_path)
    stop = plans.record(
        _record(
            action=PlanningAction.STOP,
            spec_id="",
            stop_reason=StopReason.QUESTION_RESOLVED,
        )
    )

    deliberation = director.deliberate(_frontier())

    selected = deliberation.selected
    assert selected is not None
    action = selected.action
    assert action.action_type is ResearchActionType.STOP_INVESTIGATION
    assert "planner stop: question_resolved" in action.rationale
    assert stop.id in action.rationale  # the typed decision is named
    assert plans.is_dispatched(stop.id)


def test_a_pending_experiment_dispatches_its_planning_decision(
    tmp_path: Path,
) -> None:
    director, plans = _director(tmp_path)
    decision = plans.record(_record())

    deliberation = director.deliberate(
        _frontier(pending_experiments=(SPEC,))
    )

    selected = deliberation.selected
    assert selected is not None
    assert selected.action.action_type is ResearchActionType.RUN_EXPERIMENT
    assert selected.action.targets == (SPEC.id,)
    assert decision.id in selected.action.rationale
    assert selected.valuation.expected_cost == SPEC.estimated_cost
    assert plans.is_dispatched(decision.id)


def test_a_pending_experiment_reruns_without_a_decision(
    tmp_path: Path,
) -> None:
    """After an engineering failure the spec is still pending and its
    decision already dispatched: the run is re-emitted on the loop's own
    economics, never as a new planner consultation."""
    director, plans = _director(tmp_path)
    decision = plans.record(_record())
    frontier = _frontier(pending_experiments=(SPEC,))

    director.deliberate(frontier)  # dispatches the decision
    again = director.deliberate(frontier)

    selected = again.selected
    assert selected is not None
    assert selected.action.action_type is ResearchActionType.RUN_EXPERIMENT
    assert decision.id not in selected.action.rationale


def test_an_open_replicate_decision_is_dispatched_exactly_once(
    tmp_path: Path,
) -> None:
    director, plans = _director(tmp_path)
    decision = plans.record(
        _record(action=PlanningAction.REPLICATE, replication_seed=11)
    )
    frontier = _frontier(replication_gaps=(SPEC,))

    first = director.deliberate(frontier)
    selected = first.selected
    assert selected is not None
    assert selected.action.action_type is ResearchActionType.REPLICATE
    assert selected.action.targets == (SPEC.id,)
    assert plans.is_dispatched(decision.id)

    second = director.deliberate(frontier)
    follow_up = second.selected
    assert follow_up is not None
    assert follow_up.action.action_type is (
        ResearchActionType.PLAN_NEXT_ACTION
    )


def test_stale_decisions_are_dispatched_as_stale_not_forgotten(
    tmp_path: Path,
) -> None:
    director, plans = _director(tmp_path)
    stale_replicate = plans.record(
        _record(action=PlanningAction.REPLICATE, replication_seed=11)
    )
    stale_chain = plans.record(
        _record(spec_id="exp_never_committed", response_id="mcall_2")
    )

    deliberation = director.deliberate(_frontier())  # no gap, nothing pending

    selected = deliberation.selected
    assert selected is not None
    assert selected.action.action_type is (
        ResearchActionType.PLAN_NEXT_ACTION
    )
    assert selected.valuation.expected_cost == DEFAULT_PLAN_COST
    assert plans.is_dispatched(stale_replicate.id)
    assert plans.is_dispatched(stale_chain.id)


def test_with_nothing_open_the_planner_itself_is_selected(
    tmp_path: Path,
) -> None:
    director, _ = _director(tmp_path)
    deliberation = director.deliberate(_frontier())
    selected = deliberation.selected
    assert selected is not None
    assert selected.action.action_type is (
        ResearchActionType.PLAN_NEXT_ACTION
    )
    assert selected.action.targets == ()
