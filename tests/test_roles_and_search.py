from __future__ import annotations

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.orchestration.assignment import RegistryAssigner
from autonomous_research_lab.orchestration.rule_based import RuleBasedDirector
from autonomous_research_lab.roles.base import ResearchRole, RoleName, UtilityScore
from autonomous_research_lab.search.policy import GreedySearchPolicy

STATE = ResearchState(objective="o")


class StubRole(ResearchRole):
    """A role is a (name, authority, objective) triple; that is all the base
    contract asks for, and all this stub supplies."""

    def __init__(
        self,
        name: RoleName,
        actions: frozenset[ResearchActionType],
        value: float,
    ) -> None:
        self._name = name
        self._actions = actions
        self._value = value

    @property
    def name(self) -> RoleName:
        return self._name

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return self._actions

    def utility(self, state: ResearchState, action: ResearchAction) -> UtilityScore:
        return UtilityScore(value=self._value)


def action(
    action_type: ResearchActionType, *, gain: float | None = None, usd: float = 0.0
) -> ResearchAction:
    return ResearchAction(
        action_type=action_type,
        rationale="r",
        estimated_cost=ResourceCost(usd=usd),
        expected_information_gain=gain,
    )


def test_assignment_respects_authority() -> None:
    assigner = RegistryAssigner(
        [
            StubRole(
                RoleName.PAPER_WRITER, frozenset({ResearchActionType.ANALYZE}), 10.0
            )
        ]
    )
    assert assigner.assign(STATE, action(ResearchActionType.RUN_EXPERIMENT)) is None


def test_roles_are_not_ranked_by_a_shared_reward() -> None:
    """Each capable role scores the work by its own objective; assignment picks
    the one that values it most, rather than a single global score."""
    keen = StubRole(RoleName.SKEPTIC, frozenset({ResearchActionType.FALSIFY}), 9.0)
    indifferent = StubRole(
        RoleName.PAPER_WRITER, frozenset({ResearchActionType.FALSIFY}), 1.0
    )
    assigner = RegistryAssigner([indifferent, keen])

    assigned = assigner.assign(STATE, action(ResearchActionType.FALSIFY))
    assert assigned is keen


def test_greedy_policy_prefers_information_per_unit_cost() -> None:
    policy = GreedySearchPolicy()
    cheap = action(ResearchActionType.ANALYZE, gain=1.0, usd=1.0)
    expensive = action(ResearchActionType.SCALE_EXPERIMENT, gain=2.0, usd=100.0)

    assert policy.select(STATE, [expensive, cheap]) is cheap


def test_director_will_not_propose_what_the_budget_cannot_cover() -> None:
    broke = ResearchState(objective="o", budget=ResearchBudget.zero())
    assert RuleBasedDirector().propose(broke) is None


def test_director_stops_when_no_work_is_open() -> None:
    """Halting is a real outcome. A director that always finds something else
    to try is spending budget, not doing research."""
    funded = ResearchState(
        objective="o",
        budget=ResearchBudget(wall_clock_seconds=1e6, usd=1e3, model_tokens=10**7),
    )
    first = RuleBasedDirector().propose(funded)
    assert first is not None
    assert first.action_type is ResearchActionType.GENERATE_HYPOTHESIS
