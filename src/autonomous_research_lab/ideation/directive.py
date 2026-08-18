"""The ideation directive: the one bounded input of a candidate-generation
run.

A directive names the adequacy assessment whose door the run must walk
through (:func:`~..mapping.adequacy.require_adequate_for_idea_generation`
is the only entrance) and the CFP snapshot that directs relevance, and
carries the explicit budgets that keep the run finite: how many candidates
the portfolio may hold and how many model calls the whole run may spend.

Everything is validated at construction against hard ceilings, exactly as
:class:`~..mapping.brief.ResearchBrief` does one level down: a run that
could grow without bound cannot be expressed at all. Sampling parameters
(temperature, token limits, timeouts) are deliberately absent — they are
generator wiring, and the request fingerprint embedded in every record's
provenance preserves them for reproducibility. A directive is a scientific
assignment; two assignments differing only in sampling knobs must not
become two objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ..core.ids import content_id

MAX_CANDIDATES_CEILING: Final = 10
MODEL_CALLS_CEILING: Final = 12


@dataclass(frozen=True, slots=True)
class IdeationDirective:
    """One bounded candidate-generation assignment. Content identity: the
    same assignment over the same assessed map and the same call text is
    the same directive wherever it is constructed; two *runs* of it are
    two occurrences."""

    assessment_id: str
    snapshot_id: str
    max_candidates: int = 5
    max_model_calls: int = 4
    """Two gated stages (direction extraction, candidate generation) at
    one call plus at most one corrective call each."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.assessment_id.strip():
            raise ValueError(
                "a directive must name the adequacy assessment it enters "
                "through"
            )
        if not self.snapshot_id.strip():
            raise ValueError(
                "a directive must name the CFP snapshot that directs it"
            )
        _bounded("max_candidates", self.max_candidates, MAX_CANDIDATES_CEILING)
        _bounded("max_model_calls", self.max_model_calls, MODEL_CALLS_CEILING)
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "idir",
                    self.assessment_id,
                    self.snapshot_id,
                    self.max_candidates,
                    self.max_model_calls,
                ),
            )


def _bounded(label: str, value: int, ceiling: int) -> None:
    if not 1 <= value <= ceiling:
        raise ValueError(f"{label} must be in 1..{ceiling}, got {value}")
