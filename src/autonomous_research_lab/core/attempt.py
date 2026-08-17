"""The action lifecycle: intent, attempt, outcome.

Three separate concepts:

``ResearchAction``
    Scientific intent. Semantic identity — the same intent is the same action.

``ActionAttempt``
    One occurrence of trying to execute an intent. Occurrence identity — a
    retry is a new attempt with a new id, carrying the same action.

``ActionOutcome``
    How one attempt ended: terminal status, what it produced, what it cost.

The lifecycle::

    (action proposed/selected)
        -> ActionAttempt QUEUED
        -> RUNNING
        -> SUCCEEDED | FAILED | CANCELLED | TIMED_OUT   (outcome attached)

The reason for the separation is stated once and enforced everywhere: **a
failed attempt must never make work look done.** Whether an action is complete
is a question about attempts with *succeeded* outcomes, not about whether the
action appears anywhere in a history log. ``ResearchState.history`` remains an
audit trail only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from .actions import ResearchAction
from .budget import NO_COST, ResourceCost
from .ids import occurrence_id


class AttemptStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        return self in {
            AttemptStatus.SUCCEEDED,
            AttemptStatus.FAILED,
            AttemptStatus.CANCELLED,
            AttemptStatus.TIMED_OUT,
        }


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """How one attempt ended. Only ever attached to a terminal attempt."""

    status: AttemptStatus
    produced: tuple[str, ...] = ()
    """Ids of the domain objects this attempt brought into being — results,
    evidence, hypotheses, claims — so a trajectory can be walked from decision
    to artefact."""

    error: str | None = None
    actual_cost: ResourceCost = NO_COST

    def __post_init__(self) -> None:
        if not self.status.is_terminal:
            raise ValueError(f"outcome status must be terminal, got {self.status}")
        if self.status is AttemptStatus.SUCCEEDED and self.error is not None:
            raise ValueError("a succeeded outcome cannot carry an error")


@dataclass(frozen=True, slots=True)
class ActionAttempt:
    """One occurrence of executing an action. Never reused across retries."""

    action: ResearchAction
    status: AttemptStatus = AttemptStatus.QUEUED
    outcome: ActionOutcome | None = None
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", occurrence_id("att"))
        if self.status.is_terminal:
            if self.outcome is None:
                raise ValueError(f"terminal attempt {self.id} requires an outcome")
            if self.outcome.status is not self.status:
                raise ValueError(
                    f"attempt status {self.status} disagrees with outcome "
                    f"status {self.outcome.status}"
                )
        elif self.outcome is not None:
            raise ValueError(f"non-terminal attempt {self.id} cannot have an outcome")

    def started(self) -> ActionAttempt:
        """The same attempt, now running. Identity is preserved."""
        if self.status is not AttemptStatus.QUEUED:
            raise ValueError(f"cannot start attempt in status {self.status}")
        return replace(self, status=AttemptStatus.RUNNING)

    def resolved(self, outcome: ActionOutcome) -> ActionAttempt:
        """The same attempt, terminated with ``outcome``. Identity is preserved."""
        if self.status.is_terminal:
            raise ValueError(f"attempt {self.id} is already terminal")
        return replace(self, status=outcome.status, outcome=outcome)

    @property
    def succeeded(self) -> bool:
        return self.status is AttemptStatus.SUCCEEDED
