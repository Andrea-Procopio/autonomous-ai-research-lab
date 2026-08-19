"""The selection directive: the one bounded input of a candidate
selection run.

A directive names the prior-art run record whose challenged portfolio it
selects from (:func:`~.eligibility.require_challenged_portfolio_for_selection`
is the only entrance) and carries the operator's resource constraints as
four short free-text statements: what compute, data, time, and
experimental capability are actually available. The statements are facts
about the operator's situation, quoted verbatim by any attested hard
disqualifier — never machine-comparable numbers, and never tuned after a
run. There is no "latest assessment" anywhere: assessments outside the
named run do not exist for this run, which is what makes staleness
well-defined.

Everything is validated at construction against hard ceilings, exactly
as :class:`~..priorart.directive.PriorArtDirective` does one stage down:
a run that could grow without bound cannot be expressed at all. Sampling
parameters (temperature, token limits, timeouts) are deliberately
absent — they are selector wiring, and the request fingerprint embedded
in every record's provenance preserves them for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ..core.ids import content_id

MAX_CONSTRAINT_CHARS: Final = 400
"""A constraint that cannot be quoted whole in a disqualifier is not a
short statement; the bound keeps every attestation haystack small enough
to hold in one prompt beside the portfolio."""

ELIGIBLE_CANDIDATES_CEILING: Final = 6
"""The largest eligible set whose worst-case comparative-review reply
provably fits the proven 16384-token output envelope: the preflight's
arithmetic gives 400 + 6*1800 + 15*250 = 14950 tokens at six candidates
and 18250 at seven."""

MODEL_CALLS_CEILING: Final = 4
"""Two gated stages, at most one corrective call each: 2 x 2. A
selection run has no other call to make."""


@dataclass(frozen=True, slots=True)
class SelectionDirective:
    """One bounded selection assignment. Content identity: the same
    selection over the same challenged portfolio under the same
    constraints and budgets is the same directive wherever it is
    constructed; two *runs* of it are two occurrences."""

    prior_art_run_record_id: str
    """The one door in: eligibility is computed from this run record's
    assessments only. A candidate's verdict in any other run does not
    exist for this selection."""

    compute_constraint: str
    data_constraint: str
    time_constraint: str
    experimental_constraint: str

    max_eligible_candidates: int = 5
    """More eligible candidates than this refuse at preflight: the
    comparative review judges every pair in one reply, and the reply
    must fit the output envelope."""

    max_model_calls: int = 4
    """The worst case the preflight reserves: two gated stages with at
    most one corrective call each."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.prior_art_run_record_id.strip():
            raise ValueError(
                "a directive must name the prior-art run record whose "
                "portfolio it selects from"
            )
        for label, value in (
            ("compute_constraint", self.compute_constraint),
            ("data_constraint", self.data_constraint),
            ("time_constraint", self.time_constraint),
            ("experimental_constraint", self.experimental_constraint),
        ):
            if not value.strip():
                raise ValueError(
                    f"{label} must state what is actually available"
                )
            if len(value) > MAX_CONSTRAINT_CHARS:
                raise ValueError(
                    f"{label} must be a short statement of at most "
                    f"{MAX_CONSTRAINT_CHARS} characters, got {len(value)}"
                )
        _bounded(
            "max_eligible_candidates",
            self.max_eligible_candidates,
            ELIGIBLE_CANDIDATES_CEILING,
        )
        _bounded("max_model_calls", self.max_model_calls, MODEL_CALLS_CEILING)
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "sdir",
                    self.prior_art_run_record_id,
                    self.compute_constraint,
                    self.data_constraint,
                    self.time_constraint,
                    self.experimental_constraint,
                    self.max_eligible_candidates,
                    self.max_model_calls,
                ),
            )


def _bounded(label: str, value: int, ceiling: int) -> None:
    if not 1 <= value <= ceiling:
        raise ValueError(f"{label} must be in 1..{ceiling}, got {value}")
