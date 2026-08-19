"""The prior-art directive: the one bounded input of a prior-art
challenge run.

A directive names the ideation run record whose candidate portfolio it
challenges (:func:`~.assessment.require_candidates_for_prior_art` is the
only entrance) and carries the explicit dates and budgets that keep the
run finite and honest: the cutoff that fixes what "prior art as of" even
means, and the caps on retrieval, screening, comparison, and model calls.

Everything is validated at construction against hard ceilings, exactly as
:class:`~..ideation.directive.IdeationDirective` does one level down: a
run that could grow without bound cannot be expressed at all. The cutoff
is a recorded scientific fact, not a wall-clock accident — trusted code
derives every query's date range from it, and a comparison citing a
source dated after it is a gate violation. Sampling parameters
(temperature, token limits, timeouts) are deliberately absent — they are
challenger wiring, and the request fingerprint embedded in every record's
provenance preserves them for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Final

from ..core.ids import content_id

RESULTS_PER_QUERY_CEILING: Final = 25
SCREENED_PER_CANDIDATE_CEILING: Final = 40
COMPARED_WORKS_CEILING: Final = 8
MODEL_CALLS_CEILING: Final = 36


@dataclass(frozen=True, slots=True)
class PriorArtDirective:
    """One bounded prior-art challenge assignment. Content identity: the
    same challenge of the same immutable portfolio under the same cutoff
    and budgets is the same directive wherever it is constructed; two
    *runs* of it are two occurrences."""

    ideation_run_record_id: str
    cutoff_date: str
    """Inclusive ISO upper bound on every search: prior art is assessed
    as of this date, recorded rather than implied."""

    recent_window_start: str
    """Inclusive ISO lower bound of the RECENT family's window only;
    every other family searches arbitrarily far back."""

    results_per_query: int = 5
    max_screened_per_candidate: int = 35
    """Must cover the worst-case pool the directive itself can
    retrieve: six families at ``results_per_query`` plus the
    candidate's cited injection. The preflight refuses a directive
    whose own successful retrieval would mechanically truncate — the
    Task 5D.1 live shape (23 pooled, 20 screenable, 3 truncated, the
    truncation then reported as a scientific deficiency)."""

    max_compared_works: int = 4
    """Comparison is the largest structured reply the challenger asks
    for; the cap keeps one candidate's comparison call inside the proven
    output envelope."""

    max_model_calls: int = 36
    """The worst case the preflight reserves: per candidate, one query
    proposal, ``ceil(35/12) + 1`` screening batches (abstract-level and
    metadata-only apart), and one comparison — six gated stages — each
    with at most one corrective call, across a three-candidate
    portfolio: 3 x 6 x 2 = 36."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.ideation_run_record_id.strip():
            raise ValueError(
                "a directive must name the ideation run record whose "
                "portfolio it challenges"
            )
        for label, value in (
            ("cutoff_date", self.cutoff_date),
            ("recent_window_start", self.recent_window_start),
        ):
            try:
                date.fromisoformat(value)
            except ValueError as error:
                raise ValueError(
                    f"{label} must be an ISO date (YYYY-MM-DD), got "
                    f"{value!r}"
                ) from error
        if self.recent_window_start > self.cutoff_date:
            raise ValueError(
                "the recent window cannot start after the cutoff"
            )
        _bounded(
            "results_per_query",
            self.results_per_query,
            RESULTS_PER_QUERY_CEILING,
        )
        _bounded(
            "max_screened_per_candidate",
            self.max_screened_per_candidate,
            SCREENED_PER_CANDIDATE_CEILING,
        )
        _bounded(
            "max_compared_works",
            self.max_compared_works,
            COMPARED_WORKS_CEILING,
        )
        _bounded("max_model_calls", self.max_model_calls, MODEL_CALLS_CEILING)
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "pdir",
                    self.ideation_run_record_id,
                    self.cutoff_date,
                    self.recent_window_start,
                    self.results_per_query,
                    self.max_screened_per_candidate,
                    self.max_compared_works,
                    self.max_model_calls,
                ),
            )


def _bounded(label: str, value: int, ceiling: int) -> None:
    if not 1 <= value <= ceiling:
        raise ValueError(f"{label} must be in 1..{ceiling}, got {value}")
