"""Hypotheses.

``falsification_criterion`` is a required field. A statement that cannot be
stated together with the observation that would refute it is not admissible as
a hypothesis in this system -- it is a research question, an assumption, or a
slogan. Requiring it at construction time makes that rule structural rather
than aspirational.
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
    falsification_criterion: str
    rationale: str = ""
    assumptions: tuple[str, ...] = ()
    question_id: str | None = None
    parent_id: str | None = None
    """Set when this hypothesis is a refinement of another, so that refinement
    lineage survives in the state rather than only in a conversation log."""

    status: HypothesisStatus = HypothesisStatus.PROPOSED
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("hypothesis statement must be non-empty")
        if not self.falsification_criterion.strip():
            raise ValueError(
                "hypothesis requires a falsification criterion; "
                f"none given for {self.statement!r}"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "hyp", self.statement, self.falsification_criterion, self.parent_id
                ),
            )

    def with_status(self, status: HypothesisStatus) -> Hypothesis:
        return replace(self, status=status)
