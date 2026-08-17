"""Cost-aware reasoning: tiers, coarse valuations, and when to escalate.

The runtime's spending principle::

    Tier 0 — deterministic code
    Tier 1 — cheap model / routine reasoning
    Tier 2 — strongest model for difficult decisions
    Tier 3 — multi-sample / debate, only when justified

An LLM is never asked to infer what code can determine reliably, and the
strongest (most expensive) reasoning is reserved for decisions where being
wrong is expensive: large downstream compute commitments, conflicting
evidence around a central result, a pivot or stop decision.

Two deliberately coarse vocabularies live here alongside the tiers:

:class:`Level`
    A three-value ordinal scale. The runtime has no calibration data, so it
    does not pretend to distinguish ``novelty = 0.73`` from ``0.68``. HIGH /
    MEDIUM / LOW is the honest resolution until real trajectories show finer
    estimates are calibrated.

:class:`CompactValuation`
    The runtime's working estimate of an action's worth: coarse scientific
    value, an expected resource cost, coarse uncertainty, and a rationale.
    It maps into the richer :class:`~autonomous_research_lab.core.decision.
    ActionUtility` (via :func:`as_action_utility`) so decision records keep
    their existing shape — the mapping is an *ordinal embedding*, named as
    such in ``method``, never a claim of calibrated precision.

:class:`EscalationPolicy` is deliberately a small rule table, not a model
router. It chooses the cheapest tier the signals permit; anything cleverer
needs evidence that cleverness pays.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final

from ..core.budget import NO_COST, ResourceCost
from ..core.decision import ActionUtility


class Level(StrEnum):
    """Three-value ordinal scale for value, importance and uncertainty."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def ordinal(self) -> int:
        return _ORDINAL[self]


_ORDINAL: Final = {"low": 0, "medium": 1, "high": 2}


class ReasoningTier(IntEnum):
    """How much reasoning a decision is worth. Ordered by cost, so the
    cheapest sufficient tier is the smallest one."""

    DETERMINISTIC = 0
    ROUTINE = 1
    STRONG = 2
    DELIBERATE = 3
    """Multi-sample / debate. Reserved for central results with conflicting
    evidence; never a default."""


@dataclass(frozen=True, slots=True)
class CompactValuation:
    """A coarse, honest estimate of one candidate action's worth."""

    scientific_value: Level
    expected_cost: ResourceCost = NO_COST
    uncertainty: Level = Level.MEDIUM
    rationale: str = ""


#: The ordinal embedding used by :func:`as_action_utility`. Three points, not
#: a scale: the runtime refuses to manufacture precision it cannot defend.
_EMBEDDING: Final = {Level.LOW: 0.0, Level.MEDIUM: 0.5, Level.HIGH: 1.0}
_UNCERTAINTY_EMBEDDING: Final = {Level.LOW: 0.25, Level.MEDIUM: 0.5, Level.HIGH: 1.0}

COMPACT_METHOD: Final = "compact-ordinal:v1"


def as_action_utility(valuation: CompactValuation) -> ActionUtility:
    """Embed a compact valuation into the rich utility type.

    Only ``expected_information_gain`` (from the coarse value),
    ``estimate_uncertainty`` and ``expected_cost`` are populated; every other
    dimension stays ``None`` — *not estimated*, which downstream consumers may
    not read as zero. ``method`` names the embedding so trajectory analysis
    can never mistake these for calibrated estimates.
    """
    return ActionUtility(
        expected_information_gain=_EMBEDDING[valuation.scientific_value],
        expected_cost=valuation.expected_cost,
        estimate_uncertainty=_UNCERTAINTY_EMBEDDING[valuation.uncertainty],
        method=COMPACT_METHOD,
        rationale=valuation.rationale,
    )


@dataclass(frozen=True, slots=True)
class EscalationSignals:
    """What is known about a decision before choosing how hard to think.

    The four inputs escalation depends on: how much the decision matters, how
    unsure the system is, what acting on it would commit downstream, and
    whether code could settle it outright.
    """

    importance: Level = Level.MEDIUM
    uncertainty: Level = Level.MEDIUM
    downstream_cost: ResourceCost = NO_COST
    """The resources the decision would commit if taken — the cost of being
    wrong, not the cost of deciding."""

    conflicting_evidence: bool = False
    mechanically_checkable: bool = False
    """True when deterministic code can settle the question — an existence
    check, a threshold comparison, an obvious syntax failure. No model call
    is justified for these."""


@dataclass(frozen=True, slots=True)
class EscalationPolicy:
    """Choose the cheapest sufficient reasoning tier. A rule table on
    purpose: it is inspectable, testable, and removable."""

    strong_usd_threshold: float = 100.0
    strong_gpu_hours_threshold: float = 4.0
    """Downstream commitments above these make a wrong decision expensive
    enough to justify the strongest model."""

    def tier_for(self, signals: EscalationSignals) -> ReasoningTier:
        if signals.mechanically_checkable:
            return ReasoningTier.DETERMINISTIC
        if signals.conflicting_evidence and signals.importance is Level.HIGH:
            # A central result with evidence pointing both ways is the one
            # place multi-sample review earns its cost.
            return ReasoningTier.DELIBERATE
        if (
            signals.importance is Level.HIGH
            or signals.uncertainty is Level.HIGH
            or signals.downstream_cost.usd > self.strong_usd_threshold
            or signals.downstream_cost.gpu_hours > self.strong_gpu_hours_threshold
        ):
            return ReasoningTier.STRONG
        return ReasoningTier.ROUTINE
