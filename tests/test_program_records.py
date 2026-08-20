"""The run vocabulary: what a grant, a run, and a ledger entry may say.

Identity is the load-bearing part. A run is an *event* — two runs of one
admission are two runs, and no content distinguishes them — while every
record about it is content-addressed, so a tampered file re-derives a
different id and fails loudly on load.
"""

from __future__ import annotations

import pytest

from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.ids import occurrence_id
from autonomous_research_lab.program.authorization import (
    MAX_AUTHORITY_CHARS,
    MAX_USD,
    FundingAuthorization,
    UnauthorizedGrantError,
)
from autonomous_research_lab.program.directive import MAX_LABEL_CHARS, RunDirective
from autonomous_research_lab.program.records import (
    BudgetEntry,
    EntryKind,
    ResearchRun,
)

GRANT = ResearchBudget(wall_clock_seconds=3600.0, usd=25.0, model_tokens=200_000)


def make_authorization(**overrides: object) -> FundingAuthorization:
    fields: dict[str, object] = {
        "admission_record_id": "arun_1",
        "granted": GRANT,
        "authority": "Lab operator, standing compute allocation for August.",
    }
    fields.update(overrides)
    return FundingAuthorization(**fields)  # type: ignore[arg-type]


def make_run(**overrides: object) -> ResearchRun:
    fields: dict[str, object] = {
        "run_id": "run_1",
        "directive_id": "rdir_1",
        "authorization_id": "fund_1",
        "admission_record_id": "arun_1",
        "admitted_state_id": "st_admitted",
        "funded_state_id": "st_funded",
        "granted": GRANT,
        "grant_entry_id": "bent_0",
        "label": "first run",
        "authority": "Lab operator.",
        "question_id": "q_1",
        "hypothesis_id": "hyp_1",
        "prediction_ids": ("pred_1",),
    }
    fields.update(overrides)
    return ResearchRun(**fields)  # type: ignore[arg-type]


def make_entry(**overrides: object) -> BudgetEntry:
    fields: dict[str, object] = {
        "run_id": "run_1",
        "sequence": 1,
        "kind": EntryKind.DEBIT,
        "amount": ResourceCost(usd=1.0),
        "charge_id": "att_1",
        "reason": "attempt att_1",
        "balance_after": ResearchBudget(usd=24.0),
        "previous_entry_id": "bent_0",
    }
    fields.update(overrides)
    return BudgetEntry(**fields)  # type: ignore[arg-type]


class TestFundingAuthorization:
    def test_the_same_grant_on_the_same_authority_is_one_authorization(
        self,
    ) -> None:
        assert make_authorization().id == make_authorization().id

    def test_a_different_grant_is_a_different_authorization(self) -> None:
        other = make_authorization(granted=ResearchBudget(usd=26.0))
        assert other.id != make_authorization().id

    def test_a_grant_past_a_ceiling_cannot_be_expressed(self) -> None:
        with pytest.raises(UnauthorizedGrantError, match="exceeds the ceiling"):
            make_authorization(granted=ResearchBudget(usd=MAX_USD + 1.0))

    def test_a_negative_grant_cannot_be_expressed(self) -> None:
        with pytest.raises(UnauthorizedGrantError, match="cannot be negative"):
            make_authorization(granted=ResearchBudget(usd=-1.0))

    def test_an_authorization_must_name_its_admission_and_authority(self) -> None:
        with pytest.raises(ValueError, match="admission record"):
            make_authorization(admission_record_id="  ")
        with pytest.raises(ValueError, match="who authorized"):
            make_authorization(authority="")

    def test_the_authority_is_a_short_statement(self) -> None:
        with pytest.raises(ValueError, match="short statement"):
            make_authorization(authority="x" * (MAX_AUTHORITY_CHARS + 1))


class TestRunDirective:
    def test_directives_agreeing_on_everything_are_one_directive(self) -> None:
        first = RunDirective(
            admission_record_id="arun_1", authorization_id="fund_1", label="a"
        )
        second = RunDirective(
            admission_record_id="arun_1", authorization_id="fund_1", label="a"
        )
        assert first.id == second.id

    def test_a_different_label_is_a_different_run(self) -> None:
        first = RunDirective(
            admission_record_id="arun_1", authorization_id="fund_1", label="a"
        )
        second = RunDirective(
            admission_record_id="arun_1", authorization_id="fund_1", label="b"
        )
        assert first.id != second.id

    def test_a_run_must_say_what_it_is(self) -> None:
        with pytest.raises(ValueError, match="label"):
            RunDirective(
                admission_record_id="arun_1", authorization_id="fund_1", label=" "
            )
        with pytest.raises(ValueError, match="at most"):
            RunDirective(
                admission_record_id="arun_1",
                authorization_id="fund_1",
                label="x" * (MAX_LABEL_CHARS + 1),
            )


class TestResearchRun:
    def test_two_runs_of_one_admission_are_two_runs(self) -> None:
        first = make_run(run_id=occurrence_id("run"))
        second = make_run(run_id=occurrence_id("run"))
        assert first.run_id != second.run_id
        assert first.id != second.id

    def test_the_record_is_content_addressed_over_its_event(self) -> None:
        assert make_run().id == make_run().id

    def test_the_funded_state_is_never_the_admitted_state(self) -> None:
        with pytest.raises(ValueError, match="successor"):
            make_run(funded_state_id="st_admitted")

    def test_an_exhausted_grant_funds_nothing(self) -> None:
        with pytest.raises(ValueError, match="buy something"):
            make_run(granted=ResearchBudget.zero())

    def test_a_run_names_the_predictions_it_is_funded_to_test(self) -> None:
        with pytest.raises(ValueError, match="unfalsifiable"):
            make_run(prediction_ids=())


class TestBudgetEntry:
    def test_entry_zero_is_the_grant_and_the_grant_is_entry_zero(self) -> None:
        with pytest.raises(ValueError, match="entry zero is the grant"):
            make_entry(sequence=0, previous_entry_id="")
        with pytest.raises(ValueError, match="entry zero is the grant"):
            make_entry(kind=EntryKind.GRANT)

    def test_every_entry_but_the_first_names_its_predecessor(self) -> None:
        with pytest.raises(ValueError, match="predecessor"):
            make_entry(previous_entry_id="")

    def test_a_zero_entry_records_nothing(self) -> None:
        with pytest.raises(ValueError, match="zero entry"):
            make_entry(amount=ResourceCost())

    def test_an_entry_must_carry_a_charge_id_and_a_reason(self) -> None:
        with pytest.raises(ValueError, match="charge id"):
            make_entry(charge_id=" ")
        with pytest.raises(ValueError, match="what it paid for"):
            make_entry(reason="")

    def test_the_amount_and_the_balance_are_both_part_of_the_identity(
        self,
    ) -> None:
        assert make_entry().id == make_entry().id
        assert make_entry(amount=ResourceCost(usd=2.0)).id != make_entry().id
        assert (
            make_entry(balance_after=ResearchBudget(usd=23.0)).id
            != make_entry().id
        )
