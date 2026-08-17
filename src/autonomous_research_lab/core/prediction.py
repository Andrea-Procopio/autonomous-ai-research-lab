"""Predictions: the falsifiable content of a hypothesis.

The scientific chain this system is built around::

    Hypothesis -> Prediction(s) -> Experiment -> Observation

A hypothesis is a general statement; a prediction is what that statement
commits to *observably*, under stated conditions, in terms of a named metric.
Experiments test predictions, not hypotheses directly — one hypothesis may
issue several predictions, and which of them hold constrains the hypothesis
far more informatively than a single pass/fail bit.

Checking a prediction against an observed value is *mechanical* — a comparison
fixed before the run, applied by the transition layer when evidence is
committed. What a failed prediction means for the hypothesis (wrong theory?
broken auxiliary assumption? bad instrument?) is an epistemic judgment and
lives in :class:`~.assessment.EpistemicAssessment`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from .ids import content_id


class Comparator(StrEnum):
    LESS_THAN = "lt"
    LESS_OR_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_OR_EQUAL = "ge"
    APPROXIMATELY = "approx"
    """Within ``tolerance`` of the threshold."""


class PredictionStatus(StrEnum):
    UNTESTED = "untested"
    HELD = "held"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
    """The relevant experiment ran but did not yield a checkable value."""


@dataclass(frozen=True, slots=True)
class Prediction:
    hypothesis_id: str
    condition: str
    """The context under which the prediction is asserted — data, regime,
    procedure. An unconditioned prediction is almost always over-general."""

    metric: str
    comparator: Comparator
    threshold: float
    tolerance: float = 0.0
    expectation: str = ""
    """Prose statement of the expected observation, for human and model
    consumption. The machine-checkable content is metric/comparator/threshold."""

    scope: str = ""
    status: PredictionStatus = PredictionStatus.UNTESTED
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("prediction requires a metric name")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "pred",
                    self.hypothesis_id,
                    self.condition,
                    self.metric,
                    self.comparator,
                    self.threshold,
                    self.tolerance,
                    self.scope,
                ),
            )

    def check(self, value: float) -> bool:
        """Mechanically evaluate the prediction against an observed value."""
        match self.comparator:
            case Comparator.LESS_THAN:
                return value < self.threshold
            case Comparator.LESS_OR_EQUAL:
                return value <= self.threshold
            case Comparator.GREATER_THAN:
                return value > self.threshold
            case Comparator.GREATER_OR_EQUAL:
                return value >= self.threshold
            case Comparator.APPROXIMATELY:
                return abs(value - self.threshold) <= self.tolerance
        raise AssertionError(f"unhandled comparator {self.comparator}")

    def with_status(self, status: PredictionStatus) -> Prediction:
        """Status changes preserve identity: a tested prediction is the same
        prediction."""
        return replace(self, status=status)
