"""Hypotheses.

A hypothesis is a general, revisable statement about the world. Its
*falsifiable content* lives in the :class:`~.prediction.Prediction` objects
derived from it — each one a concrete, pre-registered commitment about a
measurable quantity. A hypothesis with no predictions is not yet testable,
and candidate generation treats deriving one as open work.

``status`` is an operational lifecycle marker, updated only through the
transition layer — typically when an :class:`~.assessment.EpistemicAssessment`
targeting the hypothesis is committed. It is a cache of the current standing,
never a substitute for the assessment that justified it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from .ids import content_id


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    UNDER_TEST = "under_test"
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"
    ABANDONED = "abandoned"

    @property
    def is_terminal(self) -> bool:
        return self in {
            HypothesisStatus.SUPPORTED,
            HypothesisStatus.FALSIFIED,
            HypothesisStatus.ABANDONED,
        }


@dataclass(frozen=True, slots=True)
class Hypothesis:
    statement: str
    rationale: str = ""
    assumptions: tuple[str, ...] = ()
    question_id: str | None = None
    parent_id: str | None = None
    """Set when this hypothesis is a refinement of another, so refinement
    lineage survives in the state rather than only in a conversation log."""

    status: HypothesisStatus = HypothesisStatus.PROPOSED
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("hypothesis statement must be non-empty")
        if not self.id:
            object.__setattr__(
                self, "id", content_id("hyp", self.statement, self.parent_id)
            )

    def with_status(self, status: HypothesisStatus) -> Hypothesis:
        """Status changes preserve identity: a falsified hypothesis is the
        same hypothesis, or every reference to it would dangle."""
        return replace(self, status=status)
