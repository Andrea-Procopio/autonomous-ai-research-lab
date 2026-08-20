"""Where the loop reports what it spent.

The runtime charges the budget it carries on the state. That number is
the working remainder — fast, local, and gone the moment the process
is. A durable record of the same spend lives outside the runtime, and
this is the seam between them::

    hold the money  ->  do the work  ->  charge the state
                    ->  settle the hold  ->  require the two to agree

The seam is a protocol rather than an import for the usual reason: the
runtime depends on ``core`` alone, and the package that owns run
identity, funding, and the ledger sits above it. The loop knows only
that something can be billed and asked whether it agrees.

The hold is what a crash leaves behind. Money reserved and never
settled says both that an attempt was authorized and that nobody has yet
written down what it cost — which is exactly the question a recovering
process needs answered, and exactly the thing a bare debit cannot say.

Three rules the implementation must honor, and the loop relies on:

* **idempotent by charge id.** The loop reserves and settles once per
  attempt, and an attempt id is already unique, so a re-dispatched or
  replayed step cannot debit twice.
* **one answer per hold.** A reservation is settled or released, never
  both, and never re-opened once answered.
* **fail closed on disagreement.** ``require_balance`` raises rather
  than reconciling. Two records of one number that differ is a
  bookkeeping failure; a research runtime that silently picks one of
  them has stopped being able to say what it spent.

``None`` in the loop means no durable ledger is wired, and the run's
spend lives only on its state snapshots — the pre-existing behavior,
kept as the explicit ablation.
"""

from __future__ import annotations

from typing import Protocol

from ..core.budget import ResearchBudget, ResourceCost, Settlement


class SpendLedger(Protocol):
    """The durable side of the runtime's budget arithmetic."""

    def reserve(
        self, cost: ResourceCost, *, charge_id: str, reason: str
    ) -> object:
        """Hold ``cost`` against the budget without spending it. Must
        refuse what the available balance cannot cover: this is where a
        run is stopped for lack of money."""
        ...

    def settle(
        self, cost: ResourceCost, *, charge_id: str, reason: str
    ) -> Settlement:
        """Answer the hold on ``charge_id`` with what was actually spent,
        recorded in full even when it exceeds what was reserved. The
        settlement says whether the authorization was breached; acting on
        that is the caller's business."""
        ...

    def release(self, *, charge_id: str, reason: str) -> object:
        """Give back a hold nothing was spent against."""
        ...

    def holds(self, charge_id: str, /) -> bool:
        """Whether a reservation was ever posted for ``charge_id``."""
        ...

    def require_balance(self, expected: ResearchBudget) -> None:
        """Raise when the durable balance and ``expected`` disagree."""
        ...
