"""The admission directive: the one bounded input of an admission run.

A directive names the selection run record whose SELECTED candidate it
admits (:func:`~.door.require_selected_candidate_for_admission` is the
only entrance) and carries three operator statements about the execution
environment the admitted work will eventually run in: how execution is
scheduled, how long one job may run, and what interruption survival is
required. The statements are operator facts, quoted verbatim onto the
admission record as operator-stated requirements — never inherited from
any upstream record, never presented as inherited, and never
machine-compared. There is no "latest selection" anywhere: a selection
run this directive does not name does not exist for this admission.

Everything is validated at construction against hard ceilings, exactly
as :class:`~..selection.directive.SelectionDirective` does one stage
down. Sampling parameters (temperature, token limits, timeouts) are
deliberately absent — they are admitter wiring, and the request
fingerprint embedded in the record's provenance preserves them for
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ..core.ids import content_id

MAX_REQUIREMENT_CHARS: Final = 400
"""A requirement that cannot be quoted whole on the admission record is
not a short statement; the bound keeps the operator requirements small
enough to sit beside the inherited ones without dominating the record."""

MODEL_CALLS_CEILING: Final = 4
"""One gated stage with at most one corrective call needs two; the
ceiling leaves room for an operator who wires a larger corrective
budget, and nothing else. An admission has no other call to make."""


@dataclass(frozen=True, slots=True)
class AdmissionDirective:
    """One bounded admission assignment. Content identity: admitting the
    same selection under the same operator statements and budget is the
    same directive wherever it is constructed. Because the identity
    includes the selection run record id, and at most one admission may
    ever exist per selection run, re-running a completed directive
    replays the stored result instead of spending calls."""

    selection_run_record_id: str
    """The one door in: the admitted candidate is computed from this
    selection run record only. A selection in any other run does not
    exist for this admission."""

    scheduling_requirement: str
    job_duration_requirement: str
    checkpoint_requirement: str

    max_model_calls: int = 2
    """The worst case the preflight reserves: one gated stage with at
    most one corrective call."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.selection_run_record_id.strip():
            raise ValueError(
                "a directive must name the selection run record whose "
                "selected candidate it admits"
            )
        for label, value in (
            ("scheduling_requirement", self.scheduling_requirement),
            ("job_duration_requirement", self.job_duration_requirement),
            ("checkpoint_requirement", self.checkpoint_requirement),
        ):
            if not value.strip():
                raise ValueError(
                    f"{label} must state an operator fact about the "
                    f"execution environment"
                )
            if len(value) > MAX_REQUIREMENT_CHARS:
                raise ValueError(
                    f"{label} must be a short statement of at most "
                    f"{MAX_REQUIREMENT_CHARS} characters, got {len(value)}"
                )
        _bounded("max_model_calls", self.max_model_calls, MODEL_CALLS_CEILING)
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "adir",
                    self.selection_run_record_id,
                    self.scheduling_requirement,
                    self.job_duration_requirement,
                    self.checkpoint_requirement,
                    self.max_model_calls,
                ),
            )


def _bounded(label: str, value: int, ceiling: int) -> None:
    if not 1 <= value <= ceiling:
        raise ValueError(f"{label} must be in 1..{ceiling}, got {value}")
