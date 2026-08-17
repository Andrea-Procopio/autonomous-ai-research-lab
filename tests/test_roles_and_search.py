from __future__ import annotations

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.decision import (
    ActionCandidate,
    ActionUtility,
    EvaluatedCandidate,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.orchestration.assignment import RegistryAssigner
from autonomous_research_lab.orchestration.candidates import RuleBasedCandidateGenerator
from autonomous_research_lab.orchestration.director import ResearchDirector
from autonomous_research_lab.orchestration.evaluation import HeuristicUtilityEvaluator
from autonomous_research_lab.roles.base import ResearchRole, RoleName, UtilityScore
from autonomous_research_lab.search.policy import GreedySearchPolicy

STATE = ResearchState(objective="o")
FUNDED = ResearchState(
    objective="o",
    budget=ResearchBudget(wall_clock_seconds=1e6, usd=1e3, model_tokens=10**7),
)


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


def action(action_type: ResearchActionType) -> ResearchAction:
    return ResearchAction(action_type=action_type, rationale="r")


def evaluated(
    action_type: ResearchActionType,
    *,
    gain: float | None = None,
    usd: float = 0.0,
) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        candidate=ActionCandidate(action=action(action_type), generated_by="test"),
        utility=ActionUtility(
            expected_information_gain=gain,
            expected_cost=ResourceCost(usd=usd),
            method="test",
        ),
    )


class TestAssignment:
    def test_assignment_respects_authority(self) -> None:
        assigner = RegistryAssigner(
            [
                StubRole(
                    RoleName.PAPER_WRITER, frozenset({ResearchActionType.ANALYZE}), 10.0
                )
            ]
        )
        assert assigner.assign(STATE, action(ResearchActionType.RUN_EXPERIMENT)) is None

    def test_roles_are_not_ranked_by_a_shared_reward(self) -> None:
        """Each capable role scores the work by its own objective; assignment
        picks the one that values it most, rather than a single global
        score."""
        keen = StubRole(RoleName.SKEPTIC, frozenset({ResearchActionType.FALSIFY}), 9.0)
        indifferent = StubRole(
            RoleName.PAPER_WRITER, frozenset({ResearchActionType.FALSIFY}), 1.0
        )
        assigner = RegistryAssigner([indifferent, keen])

        assert assigner.assign(STATE, action(ResearchActionType.FALSIFY)) is keen


class TestGreedyPolicy:
    def test_prefers_information_per_unit_cost(self) -> None:
        policy = GreedySearchPolicy()
        cheap = evaluated(ResearchActionType.ANALYZE, gain=1.0, usd=1.0)
        expensive = evaluated(
            ResearchActionType.SCALE_EXPERIMENT, gain=2.0, usd=100.0
        )
        assert policy.select(FUNDED, [expensive, cheap]) is cheap

    def test_unaffordable_candidates_are_never_selected(self) -> None:
        policy = GreedySearchPolicy()
        broke = ResearchState(objective="o", budget=ResearchBudget.zero())
        priced = evaluated(ResearchActionType.ANALYZE, gain=10.0, usd=1.0)
        assert policy.select(broke, [priced]) is None

    def test_unestimated_gain_is_a_default_not_zero(self) -> None:
        """``None`` means "not estimated"; the policy must not starve
        unestimated actions of all chance."""
        policy = GreedySearchPolicy(default_gain=1.0)
        unestimated = evaluated(ResearchActionType.EXPLORE_ALTERNATIVE, usd=1.0)
        worthless = evaluated(ResearchActionType.ANALYZE, gain=0.0, usd=1.0)
        assert policy.select(FUNDED, [worthless, unestimated]) is unestimated

    def test_ranking_is_deterministic_under_ties(self) -> None:
        policy = GreedySearchPolicy()
        first = evaluated(ResearchActionType.ANALYZE, gain=1.0, usd=1.0)
        second = evaluated(ResearchActionType.REPLICATE, gain=1.0, usd=1.0)
        forward = policy.rank(FUNDED, [first, second])
        backward = policy.rank(FUNDED, [second, first])
        assert [e.action.id for e in forward] == [e.action.id for e in backward]


class TestDirector:
    def make(self) -> ResearchDirector:
        return ResearchDirector(
            generator=RuleBasedCandidateGenerator(),
            evaluator=HeuristicUtilityEvaluator(),
            policy=GreedySearchPolicy(),
        )

    def test_starts_with_hypothesis_generation(self) -> None:
        decision = self.make().decide(FUNDED)
        assert decision.action is not None
        assert decision.action.action_type is ResearchActionType.GENERATE_HYPOTHESIS

    def test_declines_when_nothing_is_affordable(self) -> None:
        """Halting is a real outcome: with a zero budget every priced
        candidate is out of reach and the policy declines."""
        broke = ResearchState(objective="o", budget=ResearchBudget.zero())
        decision = self.make().decide(broke)
        assert decision.action is None
        assert decision.record.selected_action_id is None
        assert decision.record.evaluated  # candidates were still recorded
