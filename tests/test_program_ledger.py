"""The append-only budget ledger.

Four properties are pinned here, because each is a mechanism rather than
a convention: append-only writing, ordered wholeness on replay,
idempotency by charge id, and safety when two writers post at once.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from autonomous_research_lab.core.budget import (
    InsufficientBudgetError,
    ResearchBudget,
    ResourceCost,
)
from autonomous_research_lab.program.authorization import FundingAuthorization
from autonomous_research_lab.program.ledger import (
    BudgetLedger,
    LedgerConflictError,
    LedgerIntegrityError,
    LedgerMismatchError,
)
from autonomous_research_lab.program.records import BudgetEntry, EntryKind

GRANT = ResearchBudget(wall_clock_seconds=1000.0, usd=100.0, model_tokens=10_000)


def authorization(**overrides: object) -> FundingAuthorization:
    fields: dict[str, object] = {
        "admission_record_id": "arun_1",
        "granted": GRANT,
        "authority": "Lab operator.",
    }
    fields.update(overrides)
    return FundingAuthorization(**fields)  # type: ignore[arg-type]


def granted_ledger(root: Path, run_id: str = "run_1") -> BudgetLedger:
    ledger = BudgetLedger(root, run_id)
    ledger.grant(authorization())
    return ledger


def test_a_fresh_ledger_is_empty_and_exhausted(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path, "run_1")
    assert ledger.entries() == ()
    assert ledger.balance().is_exhausted


def test_the_grant_is_entry_zero_and_sets_the_balance(tmp_path: Path) -> None:
    ledger = granted_ledger(tmp_path)

    entries = ledger.entries()
    assert len(entries) == 1
    assert entries[0].sequence == 0
    assert entries[0].kind is EntryKind.GRANT
    assert entries[0].previous_entry_id == ""
    assert ledger.balance() == GRANT


def test_a_debit_moves_the_balance_and_chains_to_its_predecessor(
    tmp_path: Path,
) -> None:
    ledger = granted_ledger(tmp_path)

    entry = ledger.debit(
        ResourceCost(usd=10.0, model_tokens=500),
        charge_id="att_1",
        reason="attempt att_1",
    )

    assert entry.sequence == 1
    assert entry.previous_entry_id == ledger.entries()[0].id
    assert ledger.balance().usd == 90.0
    assert ledger.balance().model_tokens == 9_500


def test_the_balance_replays_in_a_fresh_ledger_over_the_same_root(
    tmp_path: Path,
) -> None:
    ledger = granted_ledger(tmp_path)
    ledger.debit(ResourceCost(usd=10.0), charge_id="att_1", reason="a")
    ledger.debit(ResourceCost(usd=5.0), charge_id="att_2", reason="b")

    reloaded = BudgetLedger(tmp_path, "run_1")

    assert reloaded.balance() == ledger.balance()
    assert [e.id for e in reloaded.entries()] == [e.id for e in ledger.entries()]


class TestIdempotency:
    def test_the_same_charge_id_posts_once(self, tmp_path: Path) -> None:
        ledger = granted_ledger(tmp_path)
        cost = ResourceCost(usd=10.0)

        first = ledger.debit(cost, charge_id="att_1", reason="attempt att_1")
        second = ledger.debit(cost, charge_id="att_1", reason="attempt att_1")

        assert first == second
        assert len(ledger.entries()) == 2  # the grant and one debit
        assert ledger.balance().usd == 90.0

    def test_re_granting_one_authorization_credits_once(
        self, tmp_path: Path
    ) -> None:
        ledger = BudgetLedger(tmp_path, "run_1")

        first = ledger.grant(authorization())
        second = ledger.grant(authorization())

        assert first == second
        assert ledger.balance() == GRANT

    def test_one_charge_id_with_a_different_amount_is_a_conflict(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.debit(ResourceCost(usd=10.0), charge_id="att_1", reason="a")

        with pytest.raises(LedgerConflictError, match="already on the ledger"):
            ledger.debit(ResourceCost(usd=11.0), charge_id="att_1", reason="a")

    def test_one_charge_id_for_a_different_reason_is_a_conflict(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.debit(ResourceCost(usd=10.0), charge_id="att_1", reason="a")

        with pytest.raises(LedgerConflictError, match="re-post"):
            ledger.debit(ResourceCost(usd=10.0), charge_id="att_1", reason="b")

    def test_a_second_authorization_cannot_re_fund_a_run(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)

        with pytest.raises(LedgerConflictError, match="funded once"):
            ledger.grant(authorization(authority="Someone else."))


class TestRefusals:
    def test_a_debit_the_balance_cannot_cover_is_refused(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)

        with pytest.raises(InsufficientBudgetError):
            ledger.debit(
                ResourceCost(usd=101.0), charge_id="att_1", reason="too much"
            )

        assert len(ledger.entries()) == 1
        assert ledger.balance() == GRANT

    def test_a_debit_before_the_grant_is_refused(self, tmp_path: Path) -> None:
        ledger = BudgetLedger(tmp_path, "run_1")

        with pytest.raises(LedgerIntegrityError, match="no grant"):
            ledger.debit(ResourceCost(usd=1.0), charge_id="att_1", reason="a")

    def test_a_zero_movement_records_nothing(self, tmp_path: Path) -> None:
        ledger = granted_ledger(tmp_path)

        with pytest.raises(ValueError, match="zero debit"):
            ledger.debit(ResourceCost(), charge_id="att_1", reason="a")

    def test_an_exhausted_grant_is_refused(self, tmp_path: Path) -> None:
        ledger = BudgetLedger(tmp_path, "run_1")

        with pytest.raises(ValueError, match="exhausted grant"):
            ledger.grant(authorization(granted=ResearchBudget.zero()))


class TestTamperLoudness:
    def test_an_edited_entry_fails_to_load(self, tmp_path: Path) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.debit(ResourceCost(usd=10.0), charge_id="att_1", reason="a")
        path = ledger.directory / "000001.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["amount"]["usd"] = 1.0
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(LedgerIntegrityError, match="re-derives"):
            ledger.entries()

    def test_a_deleted_middle_entry_fails_to_load(self, tmp_path: Path) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.debit(ResourceCost(usd=10.0), charge_id="att_1", reason="a")
        ledger.debit(ResourceCost(usd=5.0), charge_id="att_2", reason="b")
        (ledger.directory / "000001.json").unlink()

        with pytest.raises(LedgerIntegrityError, match="missing or misnamed"):
            ledger.entries()

    def test_a_rewritten_balance_contradicts_the_replay(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        entry = ledger.debit(ResourceCost(usd=10.0), charge_id="att_1", reason="a")
        path = ledger.directory / "000001.json"
        # Re-derive the id so the entry passes its own identity check: the
        # replay, not the hash, is what must catch this.
        doctored = BudgetEntry(
            run_id=entry.run_id,
            sequence=entry.sequence,
            kind=entry.kind,
            amount=entry.amount,
            charge_id=entry.charge_id,
            reason=entry.reason,
            balance_after=ResearchBudget(
                wall_clock_seconds=1000.0, usd=95.0, model_tokens=10_000
            ),
            previous_entry_id=entry.previous_entry_id,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["balance_after"]["usd"] = 95.0
        payload["id"] = doctored.id
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(LedgerIntegrityError, match="replaying the ledger"):
            ledger.entries()

    def test_an_entry_from_another_run_fails_to_load(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        other = granted_ledger(tmp_path, run_id="run_2")
        stolen = (other.directory / "000000.json").read_text(encoding="utf-8")
        (ledger.directory / "000001.json").write_text(stolen, encoding="utf-8")

        with pytest.raises(LedgerIntegrityError):
            ledger.entries()


class TestReconciliation:
    def test_an_agreeing_balance_passes(self, tmp_path: Path) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.debit(ResourceCost(usd=10.0), charge_id="att_1", reason="a")

        ledger.require_balance(
            ResearchBudget(
                wall_clock_seconds=1000.0, usd=90.0, model_tokens=10_000
            )
        )

    def test_a_disagreeing_balance_fails_closed(self, tmp_path: Path) -> None:
        ledger = granted_ledger(tmp_path)

        with pytest.raises(LedgerMismatchError, match="which is the truth"):
            ledger.require_balance(ResearchBudget(usd=999.0))


def test_concurrent_debits_cannot_overspend(tmp_path: Path) -> None:
    """Twenty writers, each posting a distinct charge of 10 USD against a
    balance of 100. Exactly ten may succeed; the ledger never goes
    negative and the sequence numbers stay contiguous."""
    ledger = granted_ledger(tmp_path)

    def post(index: int) -> bool:
        try:
            BudgetLedger(tmp_path, "run_1").debit(
                ResourceCost(usd=10.0),
                charge_id=f"att_{index}",
                reason=f"attempt att_{index}",
            )
        except InsufficientBudgetError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = sum(pool.map(post, range(20)))

    entries = ledger.entries()  # verifies the chain and the replay
    assert accepted == 10
    assert len(entries) == 11  # the grant and ten debits
    assert [e.sequence for e in entries] == list(range(11))
    assert ledger.balance().usd == 0.0
    assert len({e.charge_id for e in entries}) == 11


def test_concurrent_postings_of_one_charge_debit_once(tmp_path: Path) -> None:
    """The same charge id from eight writers at once: one entry, one
    debit. Idempotency has to survive the race, not only the retry."""
    ledger = granted_ledger(tmp_path)

    def post(_: int) -> None:
        BudgetLedger(tmp_path, "run_1").debit(
            ResourceCost(usd=10.0), charge_id="att_1", reason="attempt att_1"
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(post, range(8)))

    assert len(ledger.entries()) == 2
    assert ledger.balance().usd == 90.0


def test_a_scratch_file_is_never_left_behind(tmp_path: Path) -> None:
    ledger = granted_ledger(tmp_path)
    ledger.debit(ResourceCost(usd=10.0), charge_id="att_1", reason="a")

    assert list(ledger.directory.glob("*.tmp")) == []


class TestReservations:
    """Money held against an attempt that has not finished.

    The point of the mechanism is what a crash leaves behind: a
    reservation nobody answered, which says both that an attempt was
    authorized and that nobody has yet written down what it cost.
    """

    def test_a_reservation_holds_money_without_spending_it(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)

        ledger.reserve(ResourceCost(usd=30.0), charge_id="att_1", reason="a")

        assert ledger.balance() == GRANT  # nothing is spent yet
        assert ledger.available().usd == 70.0
        assert ledger.reserved() == ResourceCost(usd=30.0)
        assert [e.charge_id for e in ledger.reservations()] == ["att_1"]

    def test_settling_spends_it_and_closes_the_reservation(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=30.0), charge_id="att_1", reason="a")

        settled = ledger.settle(
            ResourceCost(usd=12.0), charge_id="att_1", reason="a"
        )

        assert not settled.breached
        assert settled.reserved == ResourceCost(usd=30.0)
        assert settled.actual == ResourceCost(usd=12.0)
        assert ledger.balance().usd == 88.0
        assert ledger.available().usd == 88.0
        assert ledger.reservations() == ()

    def test_releasing_gives_it_back_unspent(self, tmp_path: Path) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=30.0), charge_id="att_1", reason="a")

        released = ledger.release(charge_id="att_1", reason="never started")

        assert released.kind is EntryKind.RELEASE
        assert released.amount == ResourceCost(usd=30.0)
        assert ledger.balance() == GRANT
        assert ledger.available() == GRANT
        assert ledger.reservations() == ()

    def test_reserving_twice_for_one_attempt_holds_once(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)

        first = ledger.reserve(
            ResourceCost(usd=30.0), charge_id="att_1", reason="a"
        )
        second = ledger.reserve(
            ResourceCost(usd=30.0), charge_id="att_1", reason="a"
        )

        assert first == second
        assert len(ledger.entries()) == 2
        assert ledger.available().usd == 70.0

    def test_the_available_balance_is_what_a_reservation_is_checked_against(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=80.0), charge_id="att_1", reason="a")

        # The balance is still 100, but 80 of it is promised elsewhere.
        with pytest.raises(InsufficientBudgetError, match="available"):
            ledger.reserve(
                ResourceCost(usd=30.0), charge_id="att_2", reason="b"
            )

        assert len(ledger.entries()) == 2

    def test_a_reservation_larger_than_the_grant_is_refused(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)

        with pytest.raises(InsufficientBudgetError):
            ledger.reserve(
                ResourceCost(usd=101.0), charge_id="att_1", reason="a"
            )

        assert len(ledger.entries()) == 1


class TestBreach:
    """What happens when an attempt costs more than it was authorized."""

    def test_the_whole_overrun_is_recorded(self, tmp_path: Path) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=10.0), charge_id="att_1", reason="a")

        settled = ledger.settle(
            ResourceCost(usd=25.0), charge_id="att_1", reason="a"
        )

        assert settled.breached
        assert settled.actual == ResourceCost(usd=25.0)
        # the debit is the real number, not the authorized one
        assert ledger.entries()[-1].amount == ResourceCost(usd=25.0)
        assert ledger.balance().usd == 75.0

    def test_an_overrun_past_the_whole_grant_still_lands_on_the_ledger(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=100.0), charge_id="att_1", reason="a")

        settled = ledger.settle(
            ResourceCost(usd=150.0), charge_id="att_1", reason="a"
        )

        assert settled.breached
        assert ledger.balance().usd == -50.0  # spent more than it had
        assert ledger.available().usd == 0.0  # but nothing is free
        assert ledger.entries()  # and the ledger still replays

    def test_an_unreserved_debit_is_still_refused_when_it_overdraws(
        self, tmp_path: Path
    ) -> None:
        """``settle`` may overdraw because the money is already gone.
        ``debit`` may not: it is asking to spend, not reporting."""
        ledger = granted_ledger(tmp_path)

        with pytest.raises(InsufficientBudgetError):
            ledger.debit(
                ResourceCost(usd=101.0), charge_id="att_1", reason="a"
            )


class TestOneAnswerPerReservation:
    def test_settling_without_a_reservation_is_refused(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)

        with pytest.raises(LedgerIntegrityError, match="no reservation"):
            ledger.settle(
                ResourceCost(usd=1.0), charge_id="att_1", reason="a"
            )

    def test_releasing_without_a_reservation_is_refused(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)

        with pytest.raises(LedgerIntegrityError, match="no reservation"):
            ledger.release(charge_id="att_1", reason="a")

    def test_a_settled_reservation_cannot_be_released_as_well(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=10.0), charge_id="att_1", reason="a")
        ledger.settle(ResourceCost(usd=4.0), charge_id="att_1", reason="a")

        with pytest.raises(LedgerConflictError, match="already answered"):
            ledger.release(charge_id="att_1", reason="a")

    def test_a_released_reservation_cannot_then_be_debited(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=10.0), charge_id="att_1", reason="a")
        ledger.release(charge_id="att_1", reason="never started")

        with pytest.raises(LedgerConflictError, match="already answered"):
            ledger.debit(ResourceCost(usd=4.0), charge_id="att_1", reason="a")

    def test_a_settled_attempt_is_not_reserved_again(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=10.0), charge_id="att_1", reason="a")
        ledger.settle(ResourceCost(usd=4.0), charge_id="att_1", reason="a")

        with pytest.raises(LedgerConflictError, match="already answered"):
            ledger.reserve(
                ResourceCost(usd=10.0), charge_id="att_1", reason="a"
            )

    def test_settling_twice_with_the_same_figure_settles_once(
        self, tmp_path: Path
    ) -> None:
        """Recovery re-drives a settlement it cannot prove happened."""
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=10.0), charge_id="att_1", reason="a")

        first = ledger.settle(
            ResourceCost(usd=4.0), charge_id="att_1", reason="a"
        )
        second = ledger.settle(
            ResourceCost(usd=4.0), charge_id="att_1", reason="a"
        )

        assert first == second
        assert len(ledger.entries()) == 3
        assert ledger.balance().usd == 96.0

    def test_settling_twice_for_different_amounts_is_a_conflict(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=10.0), charge_id="att_1", reason="a")
        ledger.settle(ResourceCost(usd=4.0), charge_id="att_1", reason="a")

        with pytest.raises(LedgerConflictError, match="already on the ledger"):
            ledger.settle(
                ResourceCost(usd=5.0), charge_id="att_1", reason="a"
            )

    def test_an_attempt_that_cost_nothing_is_released(
        self, tmp_path: Path
    ) -> None:
        ledger = granted_ledger(tmp_path)
        ledger.reserve(ResourceCost(usd=10.0), charge_id="att_1", reason="a")

        settled = ledger.settle(
            ResourceCost(), charge_id="att_1", reason="free"
        )

        assert not settled.breached
        assert ledger.entries()[-1].kind is EntryKind.RELEASE
        assert ledger.balance() == GRANT

    def test_the_invariant_holds_across_a_run(self, tmp_path: Path) -> None:
        """``available = balance - open reservations``, and every gap
        between the two names an attempt nobody has answered."""
        ledger = granted_ledger(tmp_path)

        def gap() -> float:
            return ledger.balance().usd - ledger.available().usd

        assert gap() == 0.0
        ledger.reserve(ResourceCost(usd=30.0), charge_id="att_1", reason="a")
        ledger.reserve(ResourceCost(usd=20.0), charge_id="att_2", reason="b")
        assert gap() == 50.0
        assert {e.charge_id for e in ledger.reservations()} == {
            "att_1",
            "att_2",
        }

        ledger.settle(ResourceCost(usd=25.0), charge_id="att_1", reason="a")
        assert gap() == 20.0
        assert [e.charge_id for e in ledger.reservations()] == ["att_2"]

        ledger.release(charge_id="att_2", reason="never started")
        assert gap() == 0.0
        assert ledger.reservations() == ()


class TestCompatibility:
    """A ledger written before reservations existed must read the same."""

    def preserved(self, tmp_path: Path) -> BudgetLedger:
        source = Path(__file__).parent / "data" / "task6a_ledger"
        run_id = "run_71303a08659a48a9"
        shutil.copytree(source, tmp_path / "ledgers" / run_id)
        return BudgetLedger(tmp_path, run_id)

    def test_a_preserved_task_6a_ledger_replays_unchanged(
        self, tmp_path: Path
    ) -> None:
        ledger = self.preserved(tmp_path)

        entries = ledger.entries()

        assert [str(e.kind) for e in entries] == ["grant", "debit"]
        assert entries[0].id == "bent_f2e3108a79f3c566"
        assert entries[1].id == "bent_9fd69d880e640b5f"
        assert ledger.balance() == ResearchBudget(
            wall_clock_seconds=84600.0,
            gpu_hours=99.5,
            usd=246.75,
            model_tokens=1_988_000,
        )

    def test_a_ledger_with_no_reservations_has_all_of_it_available(
        self, tmp_path: Path
    ) -> None:
        ledger = self.preserved(tmp_path)

        assert ledger.reservations() == ()
        assert ledger.reserved() == ResourceCost()
        assert ledger.available() == ledger.balance()

    def test_reading_it_changes_nothing_on_disk(self, tmp_path: Path) -> None:
        ledger = self.preserved(tmp_path)
        before = {
            path.name: path.read_bytes()
            for path in sorted(ledger.directory.glob("*.json"))
        }

        ledger.entries()
        ledger.available()

        after = {
            path.name: path.read_bytes()
            for path in sorted(ledger.directory.glob("*.json"))
        }
        assert after == before
