"""Where the loop reports what it spent.

The runtime charges the budget it carries on the state. That number is
the working remainder — fast, local, and gone the moment the process
is. A durable record of the same spend lives outside the runtime, and
this is the seam between them::

    charge the state  ->  post the debit  ->  require the two to agree

The seam is a protocol rather than an import for the usual reason: the
runtime depends on ``core`` alone, and the package that owns run
identity, funding, and the ledger sits above it. The loop knows only
that something can be billed and asked whether it agrees.

Two rules the implementation must honor, and the loop relies on:

* **idempotent by charge id.** The loop bills once per attempt, and an
  attempt id is already unique, so a re-dispatched or replayed step
  cannot debit twice.
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

from ..core.budget import ResearchBudget, ResourceCost


class SpendLedger(Protocol):
    """The durable side of the runtime's budget arithmetic."""

    def debit(
        self, cost: ResourceCost, *, charge_id: str, reason: str
    ) -> object:
        """Bill ``cost``. Posting one ``charge_id`` twice must record one
        debit and return what is already on the record."""
        ...

    def require_balance(self, expected: ResearchBudget) -> None:
        """Raise when the durable balance and ``expected`` disagree."""
        ...
