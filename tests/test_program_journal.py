"""The attempt journal.

What is pinned here is the thing the journal exists for: after any one
write is lost, the record still says exactly how far the attempt got.
The mechanics it shares with the ledger and the stage log — hash
chaining, sequence numbering, tamper loudness — are pinned again rather
than assumed, because a shared shape is not a shared implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.core.attempt import AttemptPhase
from autonomous_research_lab.core.budget import ResourceCost
from autonomous_research_lab.program.journal import (
    AttemptEvent,
    JournalConflictError,
    JournalIntegrityError,
    RunJournal,
)

HELD = ResourceCost(usd=10.0, model_tokens=1000)


def started(journal: RunJournal, attempt_id: str = "att_1") -> AttemptEvent:
    return journal.record(
        attempt_id=attempt_id,
        phase=AttemptPhase.STARTED,
        state_id="st_origin",
        job_id="job_1",
        reserved=HELD,
    )


class TestRecording:
    def test_a_fresh_journal_knows_nothing(self, tmp_path: Path) -> None:
        journal = RunJournal(tmp_path, "run_1")

        assert journal.events() == ()
        assert journal.attempts() == ()
        assert journal.open_attempts() == ()
        assert journal.last_for("att_1") is None

    def test_an_attempt_begins_at_started(self, tmp_path: Path) -> None:
        journal = RunJournal(tmp_path, "run_1")

        event = started(journal)

        assert event.sequence == 0
        assert event.previous_event_id == ""
        assert event.state_id == "st_origin"
        assert event.reserved == HELD
        assert journal.attempts() == ("att_1",)
        assert journal.open_attempts() == ("att_1",)

    def test_the_phases_chain(self, tmp_path: Path) -> None:
        journal = RunJournal(tmp_path, "run_1")
        first = started(journal)

        second = journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.SUBMITTED,
            job_id="job_1",
        )

        assert second.sequence == 1
        assert second.previous_event_id == first.id

    def test_a_finished_attempt_is_not_open(self, tmp_path: Path) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.COMMITTED,
            state_id="st_next",
        )
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.COMPLETED,
            reserved=HELD,
            actual=ResourceCost(usd=4.0),
        )

        assert journal.open_attempts() == ()
        last = journal.last_for("att_1")
        assert last is not None
        assert last.phase is AttemptPhase.COMPLETED

    def test_a_released_attempt_is_not_open_either(
        self, tmp_path: Path
    ) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal)

        journal.record(attempt_id="att_1", phase=AttemptPhase.RELEASED)

        assert journal.open_attempts() == ()

    def test_two_attempts_are_tracked_apart(self, tmp_path: Path) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal, "att_1")
        started(journal, "att_2")
        journal.record(attempt_id="att_1", phase=AttemptPhase.RELEASED)

        assert journal.attempts() == ("att_1", "att_2")
        assert journal.open_attempts() == ("att_2",)

    def test_a_phase_carries_what_it_made_durable(
        self, tmp_path: Path
    ) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal)

        event = journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.OUTPUTS_DURABLE,
            produced=[("evidence", "ev_1"), ("result", "res_1")],
        )

        assert event.produced == (("evidence", "ev_1"), ("result", "res_1"))


class TestIdempotency:
    def test_recording_one_phase_twice_records_it_once(
        self, tmp_path: Path
    ) -> None:
        journal = RunJournal(tmp_path, "run_1")

        first = started(journal)
        second = started(journal)

        assert first == second
        assert len(journal.events()) == 1

    def test_a_repeat_may_explain_itself_differently(
        self, tmp_path: Path
    ) -> None:
        """Recovery re-drives a phase and says why; that is not a
        different phase."""
        journal = RunJournal(tmp_path, "run_1")
        started(journal)

        again = journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.STARTED,
            state_id="st_origin",
            job_id="job_1",
            reserved=HELD,
            detail="re-driven after recovery",
        )

        assert again.detail == ""  # the first record stands
        assert len(journal.events()) == 1

    def test_a_second_version_of_one_phase_is_a_conflict(
        self, tmp_path: Path
    ) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.BUNDLE_DURABLE,
            bundle_id="bun_1",
        )

        with pytest.raises(JournalConflictError, match="second version"):
            journal.record(
                attempt_id="att_1",
                phase=AttemptPhase.BUNDLE_DURABLE,
                bundle_id="bun_2",
            )


class TestTheLifecycleOnlyMovesForward:
    def test_an_attempt_begins_nowhere_else(self, tmp_path: Path) -> None:
        journal = RunJournal(tmp_path, "run_1")

        with pytest.raises(JournalConflictError, match="begins at"):
            journal.record(
                attempt_id="att_1",
                phase=AttemptPhase.SUBMITTED,
                job_id="job_1",
            )

    def test_a_phase_cannot_go_backwards(self, tmp_path: Path) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.BUNDLE_DURABLE,
            bundle_id="bun_1",
        )

        with pytest.raises(JournalConflictError, match="backwards"):
            journal.record(
                attempt_id="att_1",
                phase=AttemptPhase.SUBMITTED,
                job_id="job_1",
            )

    def test_phases_may_be_skipped(self, tmp_path: Path) -> None:
        """An attempt that runs no job never submits one."""
        journal = RunJournal(tmp_path, "run_1")
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.STARTED,
            state_id="st_origin",
            reserved=HELD,
        )

        event = journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.OUTPUTS_DURABLE,
            produced=[("result", "res_1")],
        )

        assert event.sequence == 1

    def test_a_committed_attempt_cannot_be_released(
        self, tmp_path: Path
    ) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.COMMITTED,
            state_id="st_next",
        )

        with pytest.raises(JournalConflictError, match="already committed"):
            journal.record(attempt_id="att_1", phase=AttemptPhase.RELEASED)

    def test_a_committed_attempt_cannot_be_abandoned_either(
        self, tmp_path: Path
    ) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.COMMITTED,
            state_id="st_next",
        )

        with pytest.raises(JournalConflictError, match="already committed"):
            journal.record(
                attempt_id="att_1",
                phase=AttemptPhase.ABANDONED,
                reserved=HELD,
                actual=ResourceCost(usd=4.0),
            )

    def test_a_released_attempt_cannot_claim_a_cost(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="second answer"):
            AttemptEvent(
                run_id="run_1",
                sequence=1,
                attempt_id="att_1",
                phase=AttemptPhase.RELEASED,
                actual=ResourceCost(usd=1.0),
                previous_event_id="aevt_0",
            )


class TestWhatEachPhasePromises:
    def test_a_started_attempt_names_where_it_begins(self) -> None:
        with pytest.raises(ValueError, match="begins from"):
            AttemptEvent(
                run_id="run_1",
                sequence=0,
                attempt_id="att_1",
                phase=AttemptPhase.STARTED,
                reserved=HELD,
            )

    def test_a_started_attempt_names_what_it_may_spend(self) -> None:
        with pytest.raises(ValueError, match="authorized to spend"):
            AttemptEvent(
                run_id="run_1",
                sequence=0,
                attempt_id="att_1",
                phase=AttemptPhase.STARTED,
                state_id="st_origin",
            )

    def test_a_submitted_attempt_names_its_job(self) -> None:
        with pytest.raises(ValueError, match="names its job"):
            AttemptEvent(
                run_id="run_1",
                sequence=1,
                attempt_id="att_1",
                phase=AttemptPhase.SUBMITTED,
                previous_event_id="aevt_0",
            )

    def test_a_durable_bundle_names_itself(self) -> None:
        with pytest.raises(ValueError, match="durable bundle has an id"):
            AttemptEvent(
                run_id="run_1",
                sequence=1,
                attempt_id="att_1",
                phase=AttemptPhase.BUNDLE_DURABLE,
                previous_event_id="aevt_0",
            )

    def test_a_committed_attempt_names_its_successor(self) -> None:
        with pytest.raises(ValueError, match="names the successor"):
            AttemptEvent(
                run_id="run_1",
                sequence=1,
                attempt_id="att_1",
                phase=AttemptPhase.COMMITTED,
                previous_event_id="aevt_0",
            )

    def test_only_the_closing_event_says_what_it_cost(self) -> None:
        with pytest.raises(ValueError, match="second answer"):
            AttemptEvent(
                run_id="run_1",
                sequence=1,
                attempt_id="att_1",
                phase=AttemptPhase.OUTPUTS_DURABLE,
                actual=ResourceCost(usd=1.0),
                previous_event_id="aevt_0",
            )


class TestBreach:
    def committed(self, journal: RunJournal, actual: ResourceCost) -> None:
        started(journal)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.COMMITTED,
            state_id="st_next",
        )
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.COMPLETED,
            reserved=HELD,
            actual=actual,
        )

    def test_staying_inside_the_authorization_is_not_a_breach(
        self, tmp_path: Path
    ) -> None:
        journal = RunJournal(tmp_path, "run_1")

        self.committed(journal, ResourceCost(usd=4.0, model_tokens=100))

        assert journal.breaches() == ()

    def test_a_breach_is_the_two_numbers_disagreeing(
        self, tmp_path: Path
    ) -> None:
        """Not a third field that could disagree with them."""
        journal = RunJournal(tmp_path, "run_1")

        self.committed(journal, ResourceCost(usd=4.0, model_tokens=99_000))

        breaches = journal.breaches()
        assert [e.attempt_id for e in breaches] == ["att_1"]
        assert breaches[0].actual.model_tokens == 99_000
        assert breaches[0].reserved.model_tokens == 1000


class TestTamperLoudness:
    def test_an_edited_event_fails_to_load(self, tmp_path: Path) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal)
        path = journal.directory / "000000.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["job_id"] = "job_somewhere_else"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(JournalIntegrityError, match="re-derives"):
            journal.events()

    def test_a_deleted_middle_event_fails_to_load(
        self, tmp_path: Path
    ) -> None:
        journal = RunJournal(tmp_path, "run_1")
        started(journal)
        journal.record(
            attempt_id="att_1", phase=AttemptPhase.SUBMITTED, job_id="job_1"
        )
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.OUTPUTS_DURABLE,
            produced=[("result", "res_1")],
        )
        (journal.directory / "000001.json").unlink()

        with pytest.raises(JournalIntegrityError, match="missing or misnamed"):
            journal.events()

    def test_an_event_from_another_run_is_refused(
        self, tmp_path: Path
    ) -> None:
        RunJournal(tmp_path, "run_other").record(
            attempt_id="att_1",
            phase=AttemptPhase.STARTED,
            state_id="st_origin",
            reserved=HELD,
        )
        misfiled = tmp_path / "journals" / "run_other" / "000000.json"
        target = tmp_path / "journals" / "run_1"
        target.mkdir(parents=True, exist_ok=True)
        (target / "000000.json").write_text(
            misfiled.read_text(encoding="utf-8"), encoding="utf-8"
        )

        with pytest.raises(JournalIntegrityError, match="belongs to run"):
            RunJournal(tmp_path, "run_1").events()
