"""The funding authorization: an operator's grant, made durable.

An admitted state carries no budget. It cannot: a state's content id
excludes its budget, so a state is not the place where "what may be
spent" can safely live. Spending authority is an operator act with its
own record, and this is that record.

Everything is validated at construction against hard ceilings, exactly
as :class:`~..admission.directive.AdmissionDirective` does one stage up:
an unbounded grant cannot be expressed. The ceilings are deliberately
generous and deliberately finite — they exist so that a typo cannot
authorize a fortune, not to express a spending policy.

An authorization is content-addressed: the same grant, for the same
admission, on the same stated authority, is the same authorization
wherever it is constructed. That is what makes granting idempotent —
re-recording one authorization credits a run once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ..core.budget import ResearchBudget
from ..core.ids import content_id

MAX_AUTHORITY_CHARS: Final = 400
"""An authority that cannot be quoted whole on the run envelope is not a
short statement of who authorized what."""

MAX_WALL_CLOCK_SECONDS: Final = 30 * 24 * 60 * 60.0
MAX_GPU_HOURS: Final = 1_000.0
MAX_USD: Final = 10_000.0
MAX_MODEL_TOKENS: Final = 100_000_000
"""Hard ceilings, one per resource dimension. A grant beyond any of them
is refused at construction rather than at the moment it is spent."""


class UnauthorizedGrantError(ValueError):
    """A grant that cannot be authorized as written: negative, or past a
    ceiling. Raised at construction, so no such authorization exists."""


@dataclass(frozen=True, slots=True)
class FundingAuthorization:
    """One operator grant against one admission.

    The grant is a :class:`~..core.budget.ResearchBudget` because that is
    what it becomes: the remainder a funded state starts with. What it is
    *not* is scientific standing. Authorizing spend on an admitted idea
    says nothing about whether the idea is true, novel, or supported.
    """

    admission_record_id: str
    """The admission this grant is for. The run door refuses a directive
    whose authorization names a different admission."""

    granted: ResearchBudget
    authority: str
    """Who authorized this grant, and under what standing. An operator
    statement, quoted verbatim onto the run envelope, never inferred and
    never machine-compared."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.admission_record_id.strip():
            raise ValueError(
                "an authorization must name the admission record it funds"
            )
        if not self.authority.strip():
            raise ValueError(
                "an authorization must state who authorized the grant"
            )
        if len(self.authority) > MAX_AUTHORITY_CHARS:
            raise ValueError(
                f"authority must be a short statement of at most "
                f"{MAX_AUTHORITY_CHARS} characters, got {len(self.authority)}"
            )
        for label, value, ceiling in (
            (
                "wall_clock_seconds",
                self.granted.wall_clock_seconds,
                MAX_WALL_CLOCK_SECONDS,
            ),
            ("gpu_hours", self.granted.gpu_hours, MAX_GPU_HOURS),
            ("usd", self.granted.usd, MAX_USD),
            ("model_tokens", float(self.granted.model_tokens), MAX_MODEL_TOKENS),
        ):
            if value < 0:
                raise UnauthorizedGrantError(
                    f"a grant of {label} cannot be negative, got {value}"
                )
            if value > ceiling:
                raise UnauthorizedGrantError(
                    f"a grant of {value} {label} exceeds the ceiling of "
                    f"{ceiling}; no authorization may express it"
                )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "fund",
                    self.admission_record_id,
                    self.granted.wall_clock_seconds,
                    self.granted.gpu_hours,
                    self.granted.usd,
                    self.granted.model_tokens,
                    self.authority,
                ),
            )
