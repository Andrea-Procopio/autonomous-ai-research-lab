"""Search over scientific states and actions.

The research loop is a search problem: states are scientific situations,
actions are moves, and the policy decides which move to make. Writing it this
way -- rather than as a fixed sequence of stages -- is what makes greedy,
best-first, beam, bandit, MCTS and learned policies interchangeable later.

A policy sees only :class:`~autonomous_research_lab.core.state.ResearchState`
and :class:`~autonomous_research_lab.core.actions.ResearchAction`. It has no
handle on roles, models or executors, so a search algorithm can be replaced
without touching any agent, and vice versa.

Only a greedy policy is implemented. Anything more sophisticated needs a
calibrated value estimate, and there is nothing to calibrate against until real
trajectories exist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.actions import ResearchAction
from ..core.state import ResearchState


@dataclass(frozen=True, slots=True)
class ScoredAction:
    action: ResearchAction
    score: float
    rationale: str = ""


class SearchPolicy(ABC):
    @abstractmethod
    def score(self, state: ResearchState, action: ResearchAction) -> ScoredAction: ...

    def rank(
        self, state: ResearchState, candidates: Sequence[ResearchAction]
    ) -> tuple[ScoredAction, ...]:
        scored = [self.score(state, action) for action in candidates]
        # Ties break on action id rather than enumeration order, so a policy's
        # choice does not depend on how candidates happened to be generated.
        return tuple(sorted(scored, key=lambda s: (-s.score, s.action.id)))

    def select(
        self, state: ResearchState, candidates: Sequence[ResearchAction]
    ) -> ResearchAction | None:
        ranked = self.rank(state, candidates)
        return ranked[0].action if ranked else None


class GreedySearchPolicy(SearchPolicy):
    """Maximise expected information gain per unit cost, one step at a time.

    Actions with no information-gain estimate score ``default_gain`` rather than
    zero: "not yet estimated" is not the same as "worthless", and treating it as
    zero would make the system structurally unable to try anything it has not
    already learned to value.
    """

    def __init__(self, *, default_gain: float = 1.0, cost_floor: float = 1e-6) -> None:
        self._default_gain = default_gain
        self._cost_floor = cost_floor

    def score(
        self,
        state: ResearchState,  # noqa: ARG002 - myopic by design; part of the contract
        action: ResearchAction,
    ) -> ScoredAction:
        gain = (
            action.expected_information_gain
            if action.expected_information_gain is not None
            else self._default_gain
        )
        cost = action.estimated_cost
        # A single scalar cost is a simplification: wall-clock, money and GPU
        # time are not fungible in general. Replacing this with a proper
        # exchange rate is a Horizon 2 concern.
        scalar_cost = max(
            cost.usd + cost.gpu_hours + cost.wall_clock_seconds / 3600.0,
            self._cost_floor,
        )
        return ScoredAction(
            action=action,
            score=gain / scalar_cost,
            rationale=f"gain {gain:.3g} / cost {scalar_cost:.3g}",
        )
