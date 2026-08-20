"""The action lifecycle: intent, attempt, outcome.

Three separate concepts:

``ResearchAction``
    Scientific intent. Semantic identity — the same intent is the same action.

``ActionAttempt``
    One occurrence of trying to execute an intent. Occurrence identity — a
    retry is a new attempt with a new id, carrying the same action.

``ActionOutcome``
    How one attempt ended: terminal status, what it produced, what it cost.

``AttemptPhase``
    How far one attempt got in making itself durable. Orthogonal to
    status: a succeeded attempt whose result is still only in memory has
    not survived anything yet.

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


class AttemptPhase(StrEnum):
    """How far an attempt got in writing itself down.

    ``AttemptStatus`` says whether the science worked. This says whether
    the record of it would survive the process dying, which is a
    different question with a different answer at every moment in
    between::

        STARTED -> SUBMITTED -> OUTPUTS_DURABLE -> BUNDLE_DURABLE
                -> COMMITTED -> COMPLETED

    Phases may be skipped — an attempt that runs no job never reaches
    ``SUBMITTED`` — but they never go backwards, and every attempt
    begins at ``STARTED``.

    The first two are written *before* the thing they name, and the rest
    *after*. That asymmetry is deliberate. An intent recorded early can
    be checked afterwards, because the job id is derived rather than
    minted, so "was this ever submitted?" has an answer; a side effect
    nobody wrote down first is undiscoverable. A durability claim is the
    other way round: it is only true once the bytes are there.
    """

    STARTED = "started"
    """A reservation is about to be posted and the attempt is about to
    run. Carries the state it begins from, the amount to be held, and
    the job id it will use if it runs one."""

    SUBMITTED = "submitted"
    """The job is about to be handed to the executor. Recovery reattaches
    to exactly this job id, or finds it was never submitted."""

    OUTPUTS_DURABLE = "outputs_durable"
    """The result and its evidence are in the store. The work is bought
    and paid for; from here on nothing needs re-running."""

    BUNDLE_DURABLE = "bundle_durable"
    """The commit bundle is written. Applying it again produces the same
    successor, so from here recovery can finish without the runtime."""

    COMMITTED = "committed"
    """The debit is settled and the successor state is persisted.
    Carries what was reserved and what it actually cost."""

    COMPLETED = "completed"
    """Nothing is owed. The attempt is closed."""

    RELEASED = "released"
    """The attempt was abandoned before it cost anything, and its
    reservation was given back. The other way an attempt ends."""

    @property
    def is_terminal(self) -> bool:
        return self in {AttemptPhase.COMPLETED, AttemptPhase.RELEASED}


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
