"""Predictions: the falsifiable content of a hypothesis — and their tests.

The scientific chain this system is built around::

    Hypothesis -> Prediction(s) -> Experiment -> ExperimentResult
                                                     |
                                                     v
                                              PredictionTest(s)

A hypothesis is a general statement; a prediction is what that statement
commits to *observably*, under stated conditions, in terms of a named metric.
Experiments test predictions, not hypotheses directly — one hypothesis may
issue several predictions, and which of them hold constrains the hypothesis
far more informatively than a single pass/fail bit.

A :class:`Prediction` is a scientific proposition. It carries **no status**:
what the world turned out to look like is not part of what was predicted.
Each execution that bears on a prediction yields its own
:class:`PredictionTest` — one observation compared, mechanically, against the
pre-registered commitment. Four runs produce four tests, and a mixed record
(consistent, consistent, inconsistent, inconclusive) is preserved as exactly
that: four coexisting facts, never collapsed into a verdict on the prediction.

What the tests *mean* for the hypothesis (wrong theory? broken auxiliary
assumption? bad instrument?) is an epistemic judgment and lives in
:class:`~.assessment.EpistemicAssessment`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .ids import content_id


class Comparator(StrEnum):
    LESS_THAN = "lt"
    LESS_OR_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_OR_EQUAL = "ge"
    APPROXIMATELY = "approx"
    """Within ``tolerance`` of the threshold."""


@dataclass(frozen=True, slots=True)
class Prediction:
    """A pre-registered, machine-checkable commitment. Immutable: a tested
    prediction is the same proposition it was before the test ran."""

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


class Consistency(StrEnum):
    """How one observation relates to one pre-registered prediction. A
    mechanical comparison outcome, never a statement about the hypothesis."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    INCONCLUSIVE = "inconclusive"
    """The execution produced nothing checkable — the run failed, or the
    prediction's metric was not among the reported values."""


@dataclass(frozen=True, slots=True)
class PredictionTest:
    """What one experiment execution observed with respect to one prediction.

    Created only by the transition layer, mechanically, when a result is
    committed: the observed value is compared against the pre-registered
    comparator and threshold, fixed before the run. Occurrence-specific
    through ``result_id`` — a replication yields a new test, and contradictory
    tests of the same prediction coexist. This object never decides whether
    the hypothesis is true; it records a single comparison.
    """

    prediction_id: str
    result_id: str
    metric: str
    """The prediction's metric, pinned at check time."""

    observed: float | None
    """The value the result reported for ``metric``; ``None`` when the run
    reported nothing checkable."""

    consistency: Consistency
    detail: str = ""
    """Mechanical context — the comparison applied, or why the test was
    inconclusive. Never an interpretation."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if self.consistency is not Consistency.INCONCLUSIVE and self.observed is None:
            raise ValueError(
                "a conclusive prediction test requires an observed value"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "ptst",
                    self.prediction_id,
                    self.result_id,
                    self.metric,
                    self.observed,
                    self.consistency,
                    self.detail,
                ),
            )
