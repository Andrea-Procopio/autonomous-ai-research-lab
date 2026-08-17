"""Held-out evaluation hooks: the interface, not the benchmarks.

A research system that both optimizes against an evaluator and reports that
evaluator's number has overfit by construction. The guard is the same one
human ML practice uses: a **development** evaluator the loop may consult
freely, and a **held-out** evaluator it may not — every held-out access
requires an explicit release credential and is recorded.

This module is deliberately only the seam. No domain benchmark exists yet;
what exists is the place one will plug in, and the property that autonomous
code cannot quietly score against the held-out side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.experiment import ExperimentResult


class ObjectiveEvaluator(Protocol):
    """An external scoring function over results — a benchmark, a judge, a
    proxy metric. Named, so scores are attributable."""

    @property
    def name(self) -> str: ...

    def score(self, result: ExperimentResult) -> float: ...


class HeldOutAccessError(RuntimeError):
    """Raised when the held-out evaluator is consulted without a release."""


@dataclass(frozen=True, slots=True)
class HeldOutAccess:
    """One recorded consultation of the held-out evaluator."""

    result_id: str
    score: float
    released_by: str


class EvaluationHooks:
    """The development / held-out split, with the split enforced.

    ``score_development`` is free. ``score_held_out`` demands a non-empty
    ``released_by`` — the human or procedure taking responsibility for the
    look — and appends every access to an audit trail. The autonomous loop
    has no credential to pass, which is the point.
    """

    def __init__(
        self,
        *,
        development: ObjectiveEvaluator | None = None,
        held_out: ObjectiveEvaluator | None = None,
    ) -> None:
        self._development = development
        self._held_out = held_out
        self._accesses: list[HeldOutAccess] = []

    def score_development(self, result: ExperimentResult) -> float | None:
        if self._development is None:
            return None
        return self._development.score(result)

    def score_held_out(
        self, result: ExperimentResult, *, released_by: str
    ) -> float:
        if self._held_out is None:
            raise HeldOutAccessError("no held-out evaluator is configured")
        if not released_by.strip():
            raise HeldOutAccessError(
                "held-out evaluation requires an explicit, attributable release"
            )
        score = self._held_out.score(result)
        self._accesses.append(
            HeldOutAccess(
                result_id=result.id, score=score, released_by=released_by
            )
        )
        return score

    @property
    def held_out_accesses(self) -> tuple[HeldOutAccess, ...]:
        return tuple(self._accesses)
