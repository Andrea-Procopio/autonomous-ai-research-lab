"""The durable records of a funded research run.

Two of them, with deliberately different identity semantics.

:class:`ResearchRun` is the envelope: which admission was funded, by
which authorization, into which state. Its ``run_id`` is an *occurrence*
id, because two runs of one admission are two events and no content
could tell them apart. The record around that id is content-addressed
like every other record in the repository, so it re-derives on load and
a tampered envelope fails loudly.

:class:`BudgetEntry` is one movement on the run's ledger. Entries are
sequence-numbered from zero, and each names the id of the entry before
it: the chain makes truncation, reordering, and substitution loud, not
merely detectable by careful reading. Every entry also carries the
balance after it, so a replay has something to disagree with.

The one thing neither record holds is a scientific judgment. A grant
buys the chance to find out; it says nothing about what will be found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from ..core.budget import ResearchBudget, ResourceCost
from ..core.ids import content_id

MAX_REASON_CHARS: Final = 200
"""A ledger reason names what the money went to — an action, an attempt,
a grant's authority. It is a label, not a narrative."""


class EntryKind(StrEnum):
    """What one ledger entry did to the balance."""

    GRANT = "grant"
    DEBIT = "debit"


@dataclass(frozen=True, slots=True)
class BudgetEntry:
    """One movement on one run's ledger, written once and never revised.

    ``charge_id`` is the idempotency key: posting the same charge twice
    records one entry. The caller supplies it from something already
    unique — an attempt id for a debit, the authorization id for the
    grant — so idempotency needs no clock and no coordination.
    """

    run_id: str
    sequence: int
    kind: EntryKind
    amount: ResourceCost
    """How much moved. Both kinds use the cost type: a grant and a debit
    are the same four resource dimensions travelling in opposite
    directions, and one arithmetic path is easier to audit than two."""

    charge_id: str
    reason: str
    balance_after: ResearchBudget
    previous_entry_id: str
    """The id of entry ``sequence - 1``, or ``""`` for the grant. The
    chain is what makes a deleted middle entry loud."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("a ledger entry must name its run")
        if self.sequence < 0:
            raise ValueError(
                f"a ledger sequence starts at zero, got {self.sequence}"
            )
        if not self.charge_id.strip():
            raise ValueError(
                "a ledger entry must carry a charge id; without one it "
                "cannot be posted idempotently"
            )
        if not self.reason.strip():
            raise ValueError("a ledger entry must say what it paid for")
        if len(self.reason) > MAX_REASON_CHARS:
            raise ValueError(
                f"reason must be at most {MAX_REASON_CHARS} characters, "
                f"got {len(self.reason)}"
            )
        if self.amount.is_zero:
            raise ValueError(
                "a ledger entry moves resources; a zero entry records "
                "nothing and would still consume a sequence number"
            )
        if (self.sequence == 0) != (self.kind is EntryKind.GRANT):
            raise ValueError(
                "entry zero is the grant and the grant is entry zero; "
                f"got {self.kind} at sequence {self.sequence}"
            )
        if bool(self.previous_entry_id) == (self.sequence == 0):
            raise ValueError(
                "every entry but the first names its predecessor, and "
                "the first names none"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "bent",
                    self.run_id,
                    self.sequence,
                    str(self.kind),
                    self.amount.wall_clock_seconds,
                    self.amount.gpu_hours,
                    self.amount.usd,
                    self.amount.model_tokens,
                    self.charge_id,
                    self.reason,
                    self.balance_after.wall_clock_seconds,
                    self.balance_after.gpu_hours,
                    self.balance_after.usd,
                    self.balance_after.model_tokens,
                    self.previous_entry_id,
                ),
            )


@dataclass(frozen=True, slots=True)
class ResearchRun:
    """The run envelope: one admission, funded, with a state to run from.

    Written last, after the funded snapshot exists and the grant is on
    the ledger. A crash before it leaves an inert orphan snapshot and no
    run — "no envelope means no run" — and the re-run grants honestly
    from scratch.

    The lineage stamps are copied from the admission record after the
    door proves the reloaded state agrees with them, so the envelope
    describes its own run without reopening the admission store.
    """

    run_id: str
    directive_id: str
    authorization_id: str
    admission_record_id: str
    admitted_state_id: str
    funded_state_id: str
    granted: ResearchBudget
    grant_entry_id: str
    label: str
    authority: str
    question_id: str
    hypothesis_id: str
    prediction_ids: tuple[str, ...]
    id: str = field(default="")

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("directive_id", self.directive_id),
            ("authorization_id", self.authorization_id),
            ("admission_record_id", self.admission_record_id),
            ("admitted_state_id", self.admitted_state_id),
            ("funded_state_id", self.funded_state_id),
            ("grant_entry_id", self.grant_entry_id),
            ("label", self.label),
            ("authority", self.authority),
            ("question_id", self.question_id),
            ("hypothesis_id", self.hypothesis_id),
        ):
            if not value.strip():
                raise ValueError(f"a run envelope must name its {name}")
        if not self.prediction_ids:
            raise ValueError(
                "a run envelope names the admitted predictions; a state "
                "with none would be unfalsifiable and not worth funding"
            )
        if self.funded_state_id == self.admitted_state_id:
            raise ValueError(
                "the funded state is a successor of the admitted state, "
                "never the admitted state itself"
            )
        if self.granted.is_exhausted:
            raise ValueError(
                "a run envelope records a grant that can buy something; "
                "an exhausted grant funds no work"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "rune",
                    self.run_id,
                    self.directive_id,
                    self.authorization_id,
                    self.admission_record_id,
                    self.admitted_state_id,
                    self.funded_state_id,
                    self.granted.wall_clock_seconds,
                    self.granted.gpu_hours,
                    self.granted.usd,
                    self.granted.model_tokens,
                    self.grant_entry_id,
                    self.label,
                    self.authority,
                    self.question_id,
                    self.hypothesis_id,
                    self.prediction_ids,
                ),
            )
