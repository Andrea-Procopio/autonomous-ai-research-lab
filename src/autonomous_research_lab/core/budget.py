"""Resource accounting.

Every research decision is made under a budget. The point of making this a
first-class domain type -- rather than a runtime concern handled somewhere in
the executor -- is that a research policy should be able to reason about the
scientific value of an action *per unit of resource*, before spending anything.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final


class InsufficientBudgetError(RuntimeError):
    """Raised when a spend would overdraw a :class:`ResearchBudget`."""


@dataclass(frozen=True, slots=True)
class ResourceCost:
    """Resources consumed, or expected to be consumed, by an action."""

    wall_clock_seconds: float = 0.0
    gpu_hours: float = 0.0
    usd: float = 0.0
    model_tokens: int = 0

    def __add__(self, other: ResourceCost) -> ResourceCost:
        return ResourceCost(
            wall_clock_seconds=self.wall_clock_seconds + other.wall_clock_seconds,
            gpu_hours=self.gpu_hours + other.gpu_hours,
            usd=self.usd + other.usd,
            model_tokens=self.model_tokens + other.model_tokens,
        )

    @property
    def is_zero(self) -> bool:
        return (
            self.wall_clock_seconds == 0.0
            and self.gpu_hours == 0.0
            and self.usd == 0.0
            and self.model_tokens == 0
        )


NO_COST: Final = ResourceCost()
"""Shared zero cost, used as a dataclass default so that "free" is a named
value rather than an unpriced blank."""


@dataclass(frozen=True, slots=True)
class ResearchBudget:
    """Remaining resources available to a research program."""

    wall_clock_seconds: float = 0.0
    gpu_hours: float = 0.0
    usd: float = 0.0
    model_tokens: int = 0

    @classmethod
    def zero(cls) -> ResearchBudget:
        return cls()

    def can_afford(self, cost: ResourceCost) -> bool:
        return (
            cost.wall_clock_seconds <= self.wall_clock_seconds
            and cost.gpu_hours <= self.gpu_hours
            and cost.usd <= self.usd
            and cost.model_tokens <= self.model_tokens
        )

    def spend(self, cost: ResourceCost) -> ResearchBudget:
        """Return the budget remaining after ``cost``.

        Raises :class:`InsufficientBudgetError` rather than clamping: an overdrawn
        budget is a decision error worth surfacing, not a value to round off.
        """
        if not self.can_afford(cost):
            raise InsufficientBudgetError(f"cannot afford {cost} from {self}")
        return replace(
            self,
            wall_clock_seconds=self.wall_clock_seconds - cost.wall_clock_seconds,
            gpu_hours=self.gpu_hours - cost.gpu_hours,
            usd=self.usd - cost.usd,
            model_tokens=self.model_tokens - cost.model_tokens,
        )

    @property
    def is_exhausted(self) -> bool:
        return (
            self.wall_clock_seconds <= 0.0
            and self.gpu_hours <= 0.0
            and self.usd <= 0.0
            and self.model_tokens <= 0
        )
