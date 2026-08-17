"""The commit bundle: everything one attempt asks the state to accept, at once.

An attempt's effects — the proposals it produced and the outcome it claims —
are one transaction, not a sequence of independent edits. Committing them one
by one leaves two gaps this object closes:

* a mid-sequence rejection strands earlier proposals in the state while the
  attempt resolves ``FAILED``, so a failed attempt half-changed the world;
* an outcome can claim ``produced`` ids that no committed object carries, so
  a successful action invents outputs that do not exist.

A :class:`CommitBundle` is the unit the transition layer accepts atomically:
either every proposal commits, the produced ids check out against what was
actually committed, and the attempt resolves with its outcome — or nothing
changes at all. The invariant it exists to enforce:

    A successful action cannot claim outputs that do not exist in the
    resulting state/store.

This is a domain-level commit boundary over an immutable state, not a
distributed transaction system.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attempt import ActionOutcome, AttemptStatus
from .proposals import Proposal


@dataclass(frozen=True, slots=True)
class CommitBundle:
    """The complete effect of one attempt, submitted for atomic commit.

    ``outcome`` must be terminal (enforced by :class:`ActionOutcome`). A
    bundle whose outcome is not ``SUCCEEDED`` carries no proposals and claims
    no produced ids — a failed attempt changed nothing, and saying so is the
    point.
    """

    attempt_id: str
    outcome: ActionOutcome
    proposals: tuple[Proposal, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome.status is not AttemptStatus.SUCCEEDED:
            if self.proposals:
                raise ValueError(
                    f"a {self.outcome.status} bundle cannot carry proposals; "
                    f"an unsuccessful attempt commits nothing"
                )
            if self.outcome.produced:
                raise ValueError(
                    f"a {self.outcome.status} bundle cannot claim produced ids"
                )
