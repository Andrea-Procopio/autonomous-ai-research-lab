"""Hypotheses.

A hypothesis is a general, revisable statement about the world. Its
*falsifiable content* lives in the :class:`~.prediction.Prediction` objects
derived from it — each one a concrete, pre-registered commitment about a
measurable quantity. A hypothesis with no predictions is not yet testable,
and candidate generation treats deriving one as open work.

A hypothesis is a scientific proposition, and propositions do not carry
truth status: what is currently believed about a hypothesis is the latest
:class:`~.assessment.EpistemicAssessment` targeting it, queryable via
``ResearchState.current_assessment``. An earlier design cached a lifecycle
status here; it was removed because a cached standing on the proposition is
exactly the shortcut — belief mutating the fact it is about — that this
ontology exists to forbid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ids import content_id


@dataclass(frozen=True, slots=True)
class Hypothesis:
    statement: str
    rationale: str = ""
    assumptions: tuple[str, ...] = ()
    question_id: str | None = None
    """The research question this hypothesis attempts to answer. Nearly always
    set: a hypothesis that answers no question has no scientific relevance to
    anchor its utility to."""

    parent_id: str | None = None
    """Set when this hypothesis is a refinement of another, so refinement
    lineage survives in the state rather than only in a conversation log."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("hypothesis statement must be non-empty")
        if not self.id:
            object.__setattr__(
                self, "id", content_id("hyp", self.statement, self.parent_id)
            )
