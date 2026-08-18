"""Role routing is deterministic, total, and costs zero model calls."""

from __future__ import annotations

from autonomous_research_lab.core.actions import ResearchActionType
from autonomous_research_lab.core.proposals import ProposalKind
from autonomous_research_lab.orchestration.routing import (
    expected_proposals,
    route,
)
from autonomous_research_lab.roles.base import RoleName


def test_routing_is_total_over_the_action_types() -> None:
    for action_type in ResearchActionType:
        assert isinstance(route(action_type), RoleName)
        assert isinstance(expected_proposals(action_type), frozenset)


def test_the_three_seats_get_their_obvious_work() -> None:
    assert route(ResearchActionType.DESIGN_EXPERIMENT) is RoleName.RESEARCH_DIRECTOR
    assert route(ResearchActionType.GENERATE_HYPOTHESIS) is (
        RoleName.RESEARCH_DIRECTOR
    )
    assert route(ResearchActionType.IMPLEMENT) is RoleName.RESEARCH_ENGINEER
    assert route(ResearchActionType.RUN_EXPERIMENT) is RoleName.RESEARCH_ENGINEER
    assert route(ResearchActionType.REPLICATE) is RoleName.RESEARCH_ENGINEER
    assert route(ResearchActionType.ANALYZE) is RoleName.RESULT_ANALYST
    assert route(ResearchActionType.ASSESS_CLAIM) is RoleName.RESULT_ANALYST


def test_output_contracts_match_the_scientific_chain() -> None:
    assert expected_proposals(ResearchActionType.RUN_EXPERIMENT) == frozenset(
        {ProposalKind.RESULT}
    )
    assert expected_proposals(ResearchActionType.GENERATE_HYPOTHESIS) == (
        frozenset({ProposalKind.HYPOTHESIS})
    )
    assert expected_proposals(ResearchActionType.STOP_INVESTIGATION) == frozenset()
    assert ProposalKind.ASSESSMENT in expected_proposals(
        ResearchActionType.ASSESS_CLAIM
    )


def test_routing_only_uses_the_three_runtime_seats() -> None:
    seats = {route(action_type) for action_type in ResearchActionType}
    assert seats == {
        RoleName.RESEARCH_DIRECTOR,
        RoleName.RESEARCH_ENGINEER,
        RoleName.RESULT_ANALYST,
    }


def test_planning_routes_to_the_scientist_seat() -> None:
    assert route(ResearchActionType.PLAN_NEXT_ACTION) is (
        RoleName.RESEARCH_DIRECTOR
    )
    assert expected_proposals(ResearchActionType.PLAN_NEXT_ACTION) == frozenset(
        {ProposalKind.HYPOTHESIS, ProposalKind.PREDICTION, ProposalKind.EXPERIMENT}
    )
