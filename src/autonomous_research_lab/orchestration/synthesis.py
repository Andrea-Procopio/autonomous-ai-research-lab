"""The slow loop: periodic synthesis, without a second permanent agent.

Two timescales, one director. The fast loop optimizes throughput::

    director -> experiment -> execute -> validate -> commit

The slow loop steps back and asks what has actually been learned, whether
the program is drifting from its question, and whether to continue, pivot,
replicate, branch, or stop. It is *not* a new agent: it is the same director
invoked in a stronger reasoning mode
(:meth:`~autonomous_research_lab.orchestration.director.FrontierDirector.
synthesize`), on the same frontier projection.

When the slow loop runs is decided deterministically by
:class:`SynthesisTrigger`: every N committed experiment results, when a
contradiction first appears, and before the program stops. "Before a major
pivot" is the stop/contradiction cases in practice; anything subtler is the
director's to request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..runtime.escalation import ReasoningTier


class SynthesisRecommendation(StrEnum):
    CONTINUE = "continue"
    REPLICATE = "replicate"
    PIVOT = "pivot"
    BRANCH = "branch"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class SynthesisReview:
    """The outcome of one slow-loop invocation: prose that will be preserved,
    and a coarse recommendation the next fast-loop deliberation sees."""

    summary: str
    recommendation: SynthesisRecommendation
    tier: ReasoningTier = ReasoningTier.STRONG


@dataclass(frozen=True, slots=True)
class SynthesisTrigger:
    """Deterministic cadence for the slow loop."""

    every: int = 5
    """Committed experiment results between scheduled reviews."""

    def reasons(
        self,
        *,
        results_since_synthesis: int,
        new_contradiction: bool,
        stopping: bool,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if new_contradiction:
            reasons.append("a contradiction has appeared on the record")
        if stopping:
            reasons.append("the program is about to stop")
        if results_since_synthesis >= self.every:
            reasons.append(
                f"{results_since_synthesis} results committed since the "
                f"last synthesis (cadence {self.every})"
            )
        return tuple(reasons)
