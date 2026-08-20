"""The append-only budget ledger.

Four properties are pinned here, because each is a mechanism rather than
a convention: append-only writing, ordered wholeness on replay,
idempotency by charge id, and safety when two writers post at once.
"""

from __future__ import annotations

import json
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
