"""The research brief: the one bounded input of a field-mapping run.

A brief names a broad research direction, fixes the time axis — a hard
cutoff date and the start of the "recent work" window — and carries the
explicit budgets that keep every downstream stage finite: how many query
families, how many queries each, how many results per query, how many
sources may be screened and extracted, and how many model calls the whole
run may spend. Optional workshop/CFP hints are carried verbatim as
context for query proposal; they grant no authority.

Everything is validated at construction against hard ceilings, exactly as
:class:`~autonomous_research_lab.literature.retrieval.LiteratureQuery`
does one level down: a mapping run that could grow without bound cannot
be expressed at all.

The recent/foundational split is a property of the brief, not a model
opinion: a source is *recent* when its publication date falls inside
``[recent_window_start, cutoff_date]`` and *foundational* when it falls
before the window. Trusted code computes the classification; the gate
holds every model-authored era claim to it.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from ..core.ids import content_id

MAX_QUERIES_PER_FAMILY_CEILING: Final = 3
RESULTS_PER_QUERY_CEILING: Final = 100
SCREENED_SOURCES_CEILING: Final = 500
EXTRACTED_SOURCES_CEILING: Final = 100
MODEL_CALLS_CEILING: Final = 200
HINTS_CEILING: Final = 20


class QueryFamily(StrEnum):
    """The fixed vocabulary of focused query families. The model chooses
    query *text* within these; it cannot invent a family, and trusted code
    alone derives each family's date range from the brief."""

    RECENT = "recent"
    FOUNDATIONAL = "foundational"
    METHODS = "methods"
    DATASETS_BENCHMARKS = "datasets_benchmarks"
    METRICS_EVALUATION = "metrics_evaluation"
    BASELINES = "baselines"
    LIMITATIONS_OPEN_PROBLEMS = "limitations_open_problems"


#: Families a valid query proposal must cover: the era split needs both
#: sides retrieved, and a problem inventory without limitation-focused
#: retrieval would lean entirely on inference.
REQUIRED_FAMILIES: Final = frozenset(
    {
        QueryFamily.RECENT,
        QueryFamily.FOUNDATIONAL,
        QueryFamily.LIMITATIONS_OPEN_PROBLEMS,
    }
)


@dataclass(frozen=True, slots=True)
class ResearchBrief:
    """One bounded field-mapping assignment. Content identity: the same
    brief is the same brief wherever it is constructed."""

    topic: str
    cutoff_date: str
    recent_window_start: str
    workshop_hints: tuple[str, ...] = ()
    max_queries_per_family: int = 2
    results_per_query: int = 25
    max_screened_sources: int = 120
    max_extracted_sources: int = 40
    max_model_calls: int = 60
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("a brief must name a topic")
        for label, value in (
            ("cutoff_date", self.cutoff_date),
            ("recent_window_start", self.recent_window_start),
        ):
            try:
                _dt.date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"{label} must be an ISO date (YYYY-MM-DD): {value!r}"
                ) from exc
        if self.recent_window_start > self.cutoff_date:
            raise ValueError(
                f"the recent window starts ({self.recent_window_start}) "
                f"after the cutoff ({self.cutoff_date})"
            )
        if len(self.workshop_hints) > HINTS_CEILING:
            raise ValueError(
                f"at most {HINTS_CEILING} workshop hints are supported"
            )
        if any(not hint.strip() for hint in self.workshop_hints):
            raise ValueError("workshop hints must be non-empty")
        _bounded(
            "max_queries_per_family",
            self.max_queries_per_family,
            MAX_QUERIES_PER_FAMILY_CEILING,
        )
        _bounded(
            "results_per_query", self.results_per_query, RESULTS_PER_QUERY_CEILING
        )
        _bounded(
            "max_screened_sources",
            self.max_screened_sources,
            SCREENED_SOURCES_CEILING,
        )
        _bounded(
            "max_extracted_sources",
            self.max_extracted_sources,
            EXTRACTED_SOURCES_CEILING,
        )
        _bounded("max_model_calls", self.max_model_calls, MODEL_CALLS_CEILING)
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "brief",
                    self.topic,
                    self.cutoff_date,
                    self.recent_window_start,
                    self.workshop_hints,
                    self.max_queries_per_family,
                    self.results_per_query,
                    self.max_screened_sources,
                    self.max_extracted_sources,
                    self.max_model_calls,
                ),
            )

    def date_range(self, family: QueryFamily) -> tuple[str, str]:
        """The trusted date range for one family's retrieval: recent work
        inside the window, foundational work strictly before it, and every
        thematic family across both eras up to the cutoff. The model never
        chooses dates."""
        if family is QueryFamily.RECENT:
            return self.recent_window_start, self.cutoff_date
        if family is QueryFamily.FOUNDATIONAL:
            return "", _day_before(self.recent_window_start)
        return "", self.cutoff_date


class SourceEra(StrEnum):
    """Where one source sits on the brief's time axis — computed by
    trusted code from the reported publication date, never asserted by a
    model."""

    RECENT = "recent"
    FOUNDATIONAL = "foundational"
    UNDATED = "undated"


def classify_era(
    publication_date: str | None, brief: ResearchBrief
) -> SourceEra:
    """Deterministic era classification. A source without a reported
    publication date is honestly ``undated``, not silently assigned."""
    if publication_date is None:
        return SourceEra.UNDATED
    if publication_date >= brief.recent_window_start:
        return SourceEra.RECENT
    return SourceEra.FOUNDATIONAL


def _day_before(iso_date: str) -> str:
    return (_dt.date.fromisoformat(iso_date) - _dt.timedelta(days=1)).isoformat()


def _bounded(label: str, value: int, ceiling: int) -> None:
    if not 1 <= value <= ceiling:
        raise ValueError(f"{label} must be in 1..{ceiling}, got {value}")
