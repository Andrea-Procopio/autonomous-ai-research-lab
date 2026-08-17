"""Deciding what to do next.

The director answers exactly one question: *given this scientific state, what
action should the program take next?* It does not decide who performs the
action (that is assignment), it does not perform it, and it does not judge the
evidence that comes back.

The decision is split in two on purpose:

``candidate_actions``
    enumerates what is scientifically available -- the part that needs domain
    understanding;

``propose``
    filters by budget and defers the choice to a
    :class:`~autonomous_research_lab.search.policy.SearchPolicy`.

So a subclass supplies scientific judgement about *what is possible*, and the
search policy supplies the strategy for *what is worth doing*. Swapping greedy
for a bandit later changes no director.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..core.actions import ResearchAction
from ..core.state import ResearchState
from ..search.policy import GreedySearchPolicy, SearchPolicy


class ResearchDirector(ABC):
    def __init__(self, policy: SearchPolicy | None = None) -> None:
        self._policy = policy if policy is not None else GreedySearchPolicy()

    @property
    def policy(self) -> SearchPolicy:
        return self._policy

    @abstractmethod
    def candidate_actions(self, state: ResearchState) -> Sequence[ResearchAction]:
        """Actions that are scientifically available in ``state``."""

    def propose(self, state: ResearchState) -> ResearchAction | None:
        """Return the next action, or ``None`` when the program should halt.

        Returning ``None`` is a legitimate outcome. A system that always finds
        something else to try is not doing research; it is spending budget.
        """
        affordable = [
            action
            for action in self.candidate_actions(state)
            if state.budget.can_afford(action.estimated_cost)
        ]
        return self._policy.select(state, affordable)
