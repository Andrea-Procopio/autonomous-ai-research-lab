"""The append-only budget ledger of one funded run.

Layout, under a program root::

    <root>/
    └── ledgers/
        └── <run_id>/
            ├── 000000.json      the grant
            ├── 000001.json      a debit
            └── ...

Five properties, each earning its mechanism.

**Append-only.** An entry file is published by hard-linking a scratch
file into place, which fails if the name is taken. Nothing rewrites an
entry, and a crash mid-write leaves an ignorable scratch file rather
than a corrupt ledger.

**Ordered and whole.** Sequence numbers are the filenames, so a gap is
visible without reading anything. Each entry also names the id of the
entry before it and the balance after itself, so a deleted middle entry,
a reordering, or a doctored amount contradicts the replay.

**Idempotent.** Every posting carries a ``charge_id`` the caller already
holds — an attempt id for a reservation and its settlement, the
authorization id for the grant. Posting the same charge twice for the
same kind of movement returns the entry already on the ledger and writes
nothing. The same charge id for a different amount is a conflict, never
a second debit. One charge id passes through the ledger once: reserved,
then settled or released, never both and never re-opened.

**Safe under concurrency.** The exclusive create *is* the lock. Two
debits racing for one sequence number cannot both win; the loser
reloads — the winner may have posted the very charge it was about to —
and retries against the new head. A debit the balance cannot cover
raises instead of overdrawing — except when it is settling a
reservation, where the money is already gone and refusing to write it
down would only hide it.

**Held, not spent.** A long attempt can die between paying for something
and recording what it bought. So money is *reserved* before the work
starts and the reservation is answered afterwards — by the debit that
settles it, or by the release that cancels it when recovery proves
nothing was spent. An interrupted attempt therefore leaves a visible
claim on the budget instead of a silence that reads as free money.
Reservations do not move the balance, because nothing has been spent
yet; they move what is *available*, which is the number to ask about
before committing to anything.

The ledger holds spend, not science. Nothing here knows what the money
bought.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

from ..core.budget import (
    InsufficientBudgetError,
    ResearchBudget,
    ResourceCost,
    Settlement,
)
from ..core.ids import occurrence_id
from .authorization import FundingAuthorization
from .records import BudgetEntry, EntryKind

_LEDGERS: Final = "ledgers"
_ENTRY_SUFFIX: Final = ".json"
_SEQUENCE_DIGITS: Final = 6
_MAX_POST_ATTEMPTS: Final = 16
"""How many times a losing writer re-reads the head and tries again.
Bounded so a pathological contender fails loudly instead of spinning."""


class LedgerConflictError(RuntimeError):
    """A write-once ledger entry would be overwritten, or one charge id
    was posted twice with different content."""


class LedgerIntegrityError(RuntimeError):
    """The stored ledger contradicts itself: a gap, a broken chain, a
    tampered entry, or a balance that does not survive replay."""


class LedgerMismatchError(RuntimeError):
    """The ledger's balance disagrees with a budget it should equal.

    Raised rather than reconciled: two records of one number that differ
    is a bookkeeping failure, and a research runtime that silently picks
    one of them has stopped being able to say what it spent.
    """


class LedgerContentionError(RuntimeError):
    """A posting lost its race too many times to be worth retrying."""


class BudgetLedger:
    """The ledger of one run, addressed by its root and run id."""

    def __init__(self, root: Path | str, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("a ledger belongs to a named run")
        self._run_id = run_id
        self._directory = Path(root) / _LEDGERS / run_id
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def directory(self) -> Path:
        return self._directory

    # -- reading ---------------------------------------------------------------

    def entries(self) -> tuple[BudgetEntry, ...]:
        """Every entry in sequence order, verified on the way out: ids
        re-derived, the chain walked, the balance replayed."""
        paths = sorted(self._directory.glob(f"*{_ENTRY_SUFFIX}"))
        entries: list[BudgetEntry] = []
        balance = ResearchBudget.zero()
        previous_id = ""
        for position, path in enumerate(paths):
            entry = self._read(path)
            if entry.sequence != position:
                raise LedgerIntegrityError(
                    f"ledger {self._run_id} jumps from sequence "
                    f"{position - 1} to {entry.sequence}; an entry is "
                    f"missing or misnamed"
                )
            if entry.run_id != self._run_id:
                raise LedgerIntegrityError(
                    f"entry {path.name} belongs to run {entry.run_id}, "
                    f"not {self._run_id}"
                )
            if entry.previous_entry_id != previous_id:
                raise LedgerIntegrityError(
                    f"entry {path.name} follows "
                    f"{entry.previous_entry_id or 'nothing'}, but the "
                    f"entry before it is {previous_id or 'nothing'}"
                )
            balance = _replayed(balance, entry)
            if balance != entry.balance_after:
                raise LedgerIntegrityError(
                    f"entry {path.name} records a balance of "
                    f"{entry.balance_after} but replaying the ledger "
                    f"gives {balance}"
                )
            entries.append(entry)
            previous_id = entry.id
        return tuple(entries)

    def balance(self) -> ResearchBudget:
        """What remains, replayed from the entries themselves. An empty
        ledger is exhausted: nothing has been granted yet."""
        entries = self.entries()
        return entries[-1].balance_after if entries else ResearchBudget.zero()

    def available(self) -> ResearchBudget:
        """What may still be committed: the balance, less every
        reservation still waiting to be answered.

        Never negative. A breached attempt can debit more than the
        balance held, leaving other reservations outstanding against
        nothing; that is a real event and it is recorded in full on the
        debit, but "less than nothing is available" is not a fact about
        money — it means nothing is available, which is what this
        returns.
        """
        entries = self.entries()
        return _minus(_balance_of(entries), _held(_open_reservations(entries)))

    def reserved(self) -> ResourceCost:
        """The total held by reservations nobody has answered yet."""
        return _held(self.reservations())

    def reservations(self) -> tuple[BudgetEntry, ...]:
        """Every reservation still open, in the order it was posted."""
        return _open_reservations(self.entries())

    def entry_for_charge(
        self, charge_id: str, *, kind: EntryKind | None = None
    ) -> BudgetEntry | None:
        """The first entry posted under ``charge_id``, or the first of
        that ``kind`` — one attempt id can name a reservation and the
        movement that answered it."""
        return next(
            (
                e
                for e in self.entries()
                if e.charge_id == charge_id
                and (kind is None or e.kind is kind)
            ),
            None,
        )

    def require_balance(self, expected: ResearchBudget) -> None:
        """Fail closed when the ledger and another record of the same
        number disagree."""
        actual = self.balance()
        if actual != expected:
            raise LedgerMismatchError(
                f"ledger {self._run_id} holds {actual} but the state it "
                f"bills for holds {expected}; refusing to guess which is "
                f"the truth"
            )

    # -- writing ---------------------------------------------------------------

    def grant(self, authorization: FundingAuthorization) -> BudgetEntry:
        """Post entry zero. Re-granting one authorization is a no-op that
        returns the entry already on the ledger."""
        if authorization.granted.is_exhausted:
            raise ValueError(
                "an exhausted grant would move nothing; there is no entry "
                "to write"
            )
        return self._post(
            kind=EntryKind.GRANT,
            amount=_as_cost(authorization.granted),
            charge_id=authorization.id,
            reason=f"grant authorized by {authorization.id}",
        )

    def debit(
        self, cost: ResourceCost, *, charge_id: str, reason: str
    ) -> BudgetEntry:
        """Bill ``cost`` against the balance. Posting the same charge id
        twice returns the first entry and writes nothing."""
        if cost.is_zero:
            raise ValueError(
                "a zero debit records nothing and would still consume a "
                "sequence number"
            )
        return self._post(
            kind=EntryKind.DEBIT,
            amount=cost,
            charge_id=charge_id,
            reason=reason,
        )

    def reserve(
        self, cost: ResourceCost, *, charge_id: str, reason: str
    ) -> BudgetEntry:
        """Hold ``cost`` against the budget without spending it.

        This is where a run is refused for lack of money, and the only
        place: a reservation the available balance cannot cover raises
        rather than authorizing work that cannot be paid for. Reserving
        the same charge twice returns the first reservation and holds
        nothing further.
        """
        if cost.is_zero:
            raise ValueError(
                "a zero reservation holds nothing and would still consume "
                "a sequence number"
            )
        return self._post(
            kind=EntryKind.RESERVATION,
            amount=cost,
            charge_id=charge_id,
            reason=reason,
        )

    def release(self, *, charge_id: str, reason: str) -> BudgetEntry:
        """Cancel an open reservation, spending nothing.

        The release carries the amount it gives back, so a reader of the
        ledger alone can see what stopped being held and when.
        """
        held = self._reservation_for(charge_id)
        return self._post(
            kind=EntryKind.RELEASE,
            amount=held.amount,
            charge_id=charge_id,
            reason=reason,
        )

    def settle(
        self, cost: ResourceCost, *, charge_id: str, reason: str
    ) -> Settlement:
        """Answer an open reservation with what the attempt actually cost.

        The debit is posted in full even when it exceeds the amount that
        was reserved, and even when it exceeds the balance — the money
        is gone either way, and a ledger that recorded the smaller number
        would be a ledger that hides overruns. The returned settlement
        says whether the authorization was breached; acting on that is
        the caller's job, not the ledger's.

        An attempt that cost nothing is released rather than debited: a
        zero debit records nothing and would still take a sequence
        number.

        Settling twice with the same figure settles once — recovery
        re-drives this after a crash, and a second debit for one attempt
        would be the exact fault the journal exists to prevent.
        """
        held = self._reservation_for(charge_id)
        entry = (
            self.release(charge_id=charge_id, reason=reason)
            if cost.is_zero
            else self._post(
                kind=EntryKind.DEBIT,
                amount=cost,
                charge_id=charge_id,
                reason=reason,
                overdraw=True,
            )
        )
        return Settlement(
            charge_id=charge_id,
            reserved=held.amount,
            actual=cost,
            entry_id=entry.id,
        )

    def _reservation_for(self, charge_id: str) -> BudgetEntry:
        """The reservation this charge was authorized by, open or not.

        Answered reservations are returned too, so that re-driving a
        settlement after a crash reaches the idempotent path instead of
        failing on a reservation its own debit has already closed.
        """
        held = self.entry_for_charge(charge_id, kind=EntryKind.RESERVATION)
        if held is None:
            raise LedgerIntegrityError(
                f"ledger {self._run_id} holds no reservation for "
                f"{charge_id}; every debit answers an authorization, and "
                f"there is none here to answer"
            )
        return held

    def _post(
        self,
        *,
        kind: EntryKind,
        amount: ResourceCost,
        charge_id: str,
        reason: str,
        overdraw: bool = False,
    ) -> BudgetEntry:
        for _ in range(_MAX_POST_ATTEMPTS):
            entries = self.entries()
            history = tuple(e for e in entries if e.charge_id == charge_id)
            _require_unanswered(history, kind, charge_id)
            existing = next((e for e in history if e.kind is kind), None)
            if existing is not None:
                return _require_same_posting(existing, kind, amount, reason)
            if kind is EntryKind.GRANT:
                if entries:
                    raise LedgerConflictError(
                        f"ledger {self._run_id} is already granted by "
                        f"{entries[0].charge_id}; a run is funded once"
                    )
                balance = _as_budget(amount)
            elif not entries:
                raise LedgerIntegrityError(
                    f"ledger {self._run_id} has no grant; a debit "
                    f"before the grant would bill nothing"
                )
            elif kind is EntryKind.DEBIT:
                balance = entries[-1].balance_after.spend(
                    amount, allow_overdraw=overdraw
                )
            else:
                # A reservation and a release move what is available,
                # not what is left.
                balance = entries[-1].balance_after
                if kind is EntryKind.RESERVATION:
                    _require_affordable(entries, amount, self._run_id)
            entry = BudgetEntry(
                run_id=self._run_id,
                sequence=len(entries),
                kind=kind,
                amount=amount,
                charge_id=charge_id,
                reason=reason,
                balance_after=balance,
                previous_entry_id=entries[-1].id if entries else "",
            )
            if self._publish(entry):
                return entry
            # Another writer took this sequence number. Re-read the head:
            # it may even hold this very charge, posted by whoever won.
        raise LedgerContentionError(
            f"ledger {self._run_id} lost {_MAX_POST_ATTEMPTS} races for a "
            f"sequence number; refusing to keep trying"
        )

    def _publish(self, entry: BudgetEntry) -> bool:
        """Hard-link the entry into place. ``False`` means the sequence
        was taken while this entry was being built."""
        target = self._path(entry.sequence)
        scratch = self._directory / f"{occurrence_id('bscratch')}.tmp"
        scratch.write_text(
            json.dumps(_entry_payload(entry), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        try:
            os.link(scratch, target)
        except FileExistsError:
            return False
        finally:
            scratch.unlink(missing_ok=True)
        return True

    # -- files -----------------------------------------------------------------

    def _path(self, sequence: int) -> Path:
        return self._directory / f"{sequence:0{_SEQUENCE_DIGITS}d}{_ENTRY_SUFFIX}"

    def _read(self, path: Path) -> BudgetEntry:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LedgerIntegrityError(
                f"ledger entry {path.name} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise LedgerIntegrityError(
                f"ledger entry {path.name} is not an object"
            )
        try:
            entry = _entry_from(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerIntegrityError(
                f"ledger entry {path.name} cannot be read: {exc}"
            ) from exc
        filed_as = payload.get("id")
        if filed_as != entry.id:
            raise LedgerIntegrityError(
                f"ledger entry {path.name} claims id {filed_as!r} but "
                f"re-derives {entry.id}; the file was edited"
            )
        return entry


def _replayed(balance: ResearchBudget, entry: BudgetEntry) -> ResearchBudget:
    """The balance after ``entry``, computed rather than trusted.

    Overdrawing is allowed here because this is a replay: the entry is
    already written, and a reader that refused to add up a ledger
    recording an overrun could not report the overrun.
    """
    if entry.kind is EntryKind.GRANT:
        return _as_budget(entry.amount)
    if entry.kind is EntryKind.DEBIT:
        return balance.spend(entry.amount, allow_overdraw=True)
    return balance


def _balance_of(entries: tuple[BudgetEntry, ...]) -> ResearchBudget:
    return entries[-1].balance_after if entries else ResearchBudget.zero()


_ANSWERS: Final = frozenset({EntryKind.DEBIT, EntryKind.RELEASE})
"""The two ways a reservation stops being open."""


def _open_reservations(
    entries: tuple[BudgetEntry, ...],
) -> tuple[BudgetEntry, ...]:
    answered = {e.charge_id for e in entries if e.kind in _ANSWERS}
    return tuple(
        entry
        for entry in entries
        if entry.kind is EntryKind.RESERVATION
        and entry.charge_id not in answered
    )


def _held(reservations: tuple[BudgetEntry, ...]) -> ResourceCost:
    total = ResourceCost()
    for entry in reservations:
        total = total + entry.amount
    return total


def _minus(balance: ResearchBudget, held: ResourceCost) -> ResearchBudget:
    """The balance less what is held, floored at nothing in every
    dimension."""
    return ResearchBudget(
        wall_clock_seconds=max(
            0.0, balance.wall_clock_seconds - held.wall_clock_seconds
        ),
        gpu_hours=max(0.0, balance.gpu_hours - held.gpu_hours),
        usd=max(0.0, balance.usd - held.usd),
        model_tokens=max(0, balance.model_tokens - held.model_tokens),
    )


def _require_unanswered(
    history: tuple[BudgetEntry, ...], kind: EntryKind, charge_id: str
) -> None:
    """A reservation is answered exactly once, and never re-opened.

    Re-posting the answer itself is the idempotent path and is handled
    by the caller. Everything else that touches an answered charge — a
    second reservation, a release after a debit, a debit after a release
    — is a bookkeeping error and says so.
    """
    answered = next((e for e in history if e.kind in _ANSWERS), None)
    if answered is not None and answered.kind is not kind:
        raise LedgerConflictError(
            f"charge {charge_id} is already answered by a "
            f"{answered.kind} of {answered.amount}; a reservation is "
            f"answered once, and money already settled is not held again"
        )


def _require_affordable(
    entries: tuple[BudgetEntry, ...], amount: ResourceCost, run_id: str
) -> None:
    """Where a run is refused for lack of money.

    Against what is available, not what is left: money another attempt
    is already holding has been promised, and promising it twice is how
    two attempts both believe they can afford to run.
    """
    available = _minus(_balance_of(entries), _held(_open_reservations(entries)))
    if not available.can_afford(amount):
        raise InsufficientBudgetError(
            f"ledger {run_id} cannot hold {amount}: {available} is "
            f"available once open reservations are counted"
        )


def _require_same_posting(
    existing: BudgetEntry, kind: EntryKind, amount: ResourceCost, reason: str
) -> BudgetEntry:
    """One charge id means one movement. Re-posting it identically is the
    idempotent path; re-posting it differently is a conflict."""
    if existing.kind is not kind or existing.amount != amount:
        raise LedgerConflictError(
            f"charge {existing.charge_id} is already on the ledger as "
            f"{existing.kind} of {existing.amount}; refusing to post it "
            f"again as {kind} of {amount}"
        )
    if existing.reason != reason:
        raise LedgerConflictError(
            f"charge {existing.charge_id} is already on the ledger for "
            f"{existing.reason!r}; refusing to re-post it for {reason!r}"
        )
    return existing


def _as_cost(budget: ResearchBudget) -> ResourceCost:
    return ResourceCost(
        wall_clock_seconds=budget.wall_clock_seconds,
        gpu_hours=budget.gpu_hours,
        usd=budget.usd,
        model_tokens=budget.model_tokens,
    )


def _as_budget(cost: ResourceCost) -> ResearchBudget:
    return ResearchBudget(
        wall_clock_seconds=cost.wall_clock_seconds,
        gpu_hours=cost.gpu_hours,
        usd=cost.usd,
        model_tokens=cost.model_tokens,
    )


def _entry_payload(entry: BudgetEntry) -> dict[str, object]:
    return {
        "id": entry.id,
        "run_id": entry.run_id,
        "sequence": entry.sequence,
        "kind": str(entry.kind),
        "amount": _amounts(
            entry.amount.wall_clock_seconds,
            entry.amount.gpu_hours,
            entry.amount.usd,
            entry.amount.model_tokens,
        ),
        "charge_id": entry.charge_id,
        "reason": entry.reason,
        "balance_after": _amounts(
            entry.balance_after.wall_clock_seconds,
            entry.balance_after.gpu_hours,
            entry.balance_after.usd,
            entry.balance_after.model_tokens,
        ),
        "previous_entry_id": entry.previous_entry_id,
    }


def _entry_from(payload: dict[str, object]) -> BudgetEntry:
    amount = _dimensions(payload, "amount")
    balance = _dimensions(payload, "balance_after")
    return BudgetEntry(
        run_id=_text(payload, "run_id"),
        sequence=_integer(payload, "sequence"),
        kind=EntryKind(_text(payload, "kind")),
        amount=ResourceCost(
            wall_clock_seconds=amount[0],
            gpu_hours=amount[1],
            usd=amount[2],
            model_tokens=int(amount[3]),
        ),
        charge_id=_text(payload, "charge_id"),
        reason=_text(payload, "reason"),
        balance_after=ResearchBudget(
            wall_clock_seconds=balance[0],
            gpu_hours=balance[1],
            usd=balance[2],
            model_tokens=int(balance[3]),
        ),
        previous_entry_id=_text(payload, "previous_entry_id"),
    )


def _amounts(
    wall_clock_seconds: float, gpu_hours: float, usd: float, model_tokens: int
) -> dict[str, float | int]:
    return {
        "wall_clock_seconds": wall_clock_seconds,
        "gpu_hours": gpu_hours,
        "usd": usd,
        "model_tokens": model_tokens,
    }


def _dimensions(
    payload: dict[str, object], key: str
) -> tuple[float, float, float, float]:
    raw = payload[key]
    if not isinstance(raw, dict):
        raise TypeError(f"{key} must be an object of resource dimensions")
    return (
        _number(raw, "wall_clock_seconds"),
        _number(raw, "gpu_hours"),
        _number(raw, "usd"),
        _number(raw, "model_tokens"),
    )


def _text(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _number(payload: dict[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)
