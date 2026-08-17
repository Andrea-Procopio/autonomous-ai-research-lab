"""Search policies: selection over evaluated candidates.

A policy answers exactly one question: *given these candidates and these
utility estimates, under this state's resources, what do we do next?* It does
not generate candidates and it does not estimate value — those are the
candidate generator's and utility evaluator's jobs. The separation is the
invariant that lets greedy, best-first, beam, bandit, MCTS and learned
policies swap in without touching either of the other two.

A policy sees only :class:`~autonomous_research_lab.core.state.ResearchState`
and :class:`~autonomous_research_lab.core.decision.EvaluatedCandidate` — no
roles, no models, no executors.

Utilities are multi-dimensional; policies that need a scalar do their own
scalarization, explicitly and by name, so "how we collapse value" is a
property of the explorer, not a hidden property of the value.

Only a greedy policy is implemented. Anything more sophisticated needs a
calibrated value estimate, and there is nothing to calibrate against until
real trajectories exist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..core.budget import ResourceCost
from ..core.decision import ActionUtility, EvaluatedCandidate
from ..core.state import ResearchState


class SearchPolicy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Recorded in every DecisionRecord this policy contributes to."""

    @abstractmethod
    def rank(
        self, state: ResearchState, evaluated: Sequence[EvaluatedCandidate]
    ) -> tuple[EvaluatedCandidate, ...]:
        """Order candidates by preference, best first. May drop candidates the
        policy would never take (e.g. unaffordable ones)."""

    def select(
        self, state: ResearchState, evaluated: Sequence[EvaluatedCandidate]
    ) -> EvaluatedCandidate | None:
        """The policy's choice, or ``None`` when it declines every candidate.

        ``None`` is a legitimate outcome: a policy that always finds something
        to take is spending budget, not doing research."""
        ranked = self.rank(state, evaluated)
        return ranked[0] if ranked else None


class GreedySearchPolicy(SearchPolicy):
    """Maximise estimated value per unit cost, one step at a time.

    Scalarization is this policy's own, and deliberately crude: value
    dimensions are summed, weighted by success probability when present, and
    divided by a single scalar cost. Both collapses are modelling choices a
    better policy would replace — they live here, named, rather than on
    ``ActionUtility`` where they would masquerade as properties of the value
    itself.

    Dimensions that are ``None`` are *not estimated*: information gain falls
    back to ``default_gain`` rather than zero, because a system that reads
    "unknown" as "worthless" can never try anything it has not already learned
    to value.
    """

    def __init__(self, *, default_gain: float = 1.0, cost_floor: float = 1e-6) -> None:
        self._default_gain = default_gain
        self._cost_floor = cost_floor

    @property
    def name(self) -> str:
        return "greedy:v0"

    def rank(
        self, state: ResearchState, evaluated: Sequence[EvaluatedCandidate]
    ) -> tuple[EvaluatedCandidate, ...]:
        affordable = [
            e for e in evaluated if state.budget.can_afford(e.utility.expected_cost)
        ]
        return tuple(
            sorted(
                affordable,
                key=lambda e: (-self._score(e.utility), e.action.id),
            )
        )

    def _score(self, utility: ActionUtility) -> float:
        gain = (
            utility.expected_information_gain
            if utility.expected_information_gain is not None
            else self._default_gain
        )
        value = gain + sum(
            dim
            for dim in (
                utility.discrimination_value,
                utility.importance,
                utility.novelty,
                utility.replication_value,
            )
            if dim is not None
        )
        if utility.expected_success_probability is not None:
            value *= utility.expected_success_probability
        return value / self._scalar_cost(utility.expected_cost)

    def _scalar_cost(self, cost: ResourceCost) -> float:
        # An arbitrary exchange rate between non-fungible resources — this
        # policy's simplification, not a domain truth. Horizon 2 work.
        scalar = (
            cost.usd
            + cost.gpu_hours
            + cost.wall_clock_seconds / 3600.0
            + cost.model_tokens / 1_000_000.0
        )
        return max(scalar, self._cost_floor)
