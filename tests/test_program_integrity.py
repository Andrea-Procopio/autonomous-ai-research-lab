"""Verifying a whole run from cold.

Every test here writes a run in one store and verifies it through a
different one, because the claim under test is about what survives a
process, not about what a live object remembers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from autonomous_research_lab.core.attempt import AttemptPhase, SettlementBasis
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    ResultRef,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState, recording_lineage
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.program.authorization import FundingAuthorization
from autonomous_research_lab.program.directive import RunDirective
from autonomous_research_lab.program.integrity import (
    IntegrityIssue,
    IntegrityIssueKind,
    _check_lineage,
    _reachable_from_roots,
    verify_run,
)
from autonomous_research_lab.program.journal import RunJournal
from autonomous_research_lab.program.ledger import BudgetLedger
from autonomous_research_lab.program.records import ResearchRun
from autonomous_research_lab.program.store import ProgramStore

MANIFEST_FILENAME = "manifest.json"
GRANT = ResearchBudget(wall_clock_seconds=100.0, usd=10.0, model_tokens=1_000)
HELD = ResourceCost(usd=2.0)


def make_run_dir(root: Path, *, name: str = "job-1") -> Path:
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.log").write_text("ran\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    (run_dir / "metrics.json").write_text('{"x": 1}', encoding="utf-8")
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "metrics.json": hashlib.sha256(
                    (run_dir / "metrics.json").read_bytes()
                ).hexdigest()
            }
        ),
        encoding="utf-8",
    )
    return run_dir


QUESTION = ResearchQuestion(text="Is x one?")
HYPOTHESIS = Hypothesis(statement="x is one.", question_id=QUESTION.id)
PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="under the standard setup",
    metric="x",
    comparator=Comparator.GREATER_THAN,
    threshold=0.5,
)
SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="measure x",
    procedure="run the toy experiment once",
    metrics=("x",),
    seeds=(7,),
)


def make_result(run_dir: Path, *, job_id: str = "job_1") -> ExperimentResult:
    return ExperimentResult(
        spec_id=SPEC.id,
        job_id=job_id,
        status=ExperimentStatus.COMPLETED,
        command=("python", "experiment.py"),
        environment=Environment(python_version="3.11.9", platform="test"),
        metrics={"x": 1.0},
        seed=7,
        artifacts=(str(run_dir / "metrics.json"),),
        logs=(str(run_dir / "stdout.log"), str(run_dir / "stderr.log")),
    )


def write_run(root: Path) -> tuple[ExperimentResult, Evidence, ResearchState]:
    """One small but whole run root: a linked chain, a fact, a reading of
    it, and a state that references both. Whole on purpose — the chain
    checker runs here too, and a fixture with a dangling spec would test
    the fixture."""
    run_dir = make_run_dir(root)
    store = FileEvidenceStore(root)
    result = store.record_result(make_result(run_dir))
    evidence = store.record_evidence(
        Evidence(
            result_id=result.id,
            spec_id=result.spec_id,
            kind=EvidenceKind.MEASUREMENT,
            observation="x = 1.0 over one run, seed 7",
            metrics={"x": 1.0},
        )
    )
    seed = ResearchState(objective="measure x")
    # The whole lineage, not just its end: a snapshot whose parent is
    # missing is what `_check_lineage` exists to catch, and a fixture
    # that wrote one would be testing the verifier against a broken run
    # by accident.
    with recording_lineage() as derived:
        state = (
            seed.upsert_question(QUESTION)
            .upsert_hypothesis(HYPOTHESIS)
            .upsert_prediction(PREDICTION)
            .add_experiment(SPEC)
            .record_result(
                ResultRef(
                    result_id=result.id,
                    spec_id=result.spec_id,
                    status=result.status,
                )
            )
            .record_evidence(evidence.id)
        )
    states = FileStateStore(root)
    states.persist(seed)
    for successor in derived:
        states.persist(successor)
    return result, evidence, state


def test_a_written_run_verifies_from_a_cold_start(tmp_path: Path) -> None:
    write_run(tmp_path)

    report = verify_run(tmp_path)

    assert report.ok, report.issues
    assert report.states_checked == 7  # the whole lineage, not just its end
    assert report.results_checked == 1
    assert report.evidence_checked == 1
    assert report.blobs_checked == 3  # metrics.json, stdout, stderr


def test_an_empty_root_verifies_as_empty(tmp_path: Path) -> None:
    report = verify_run(tmp_path)

    assert report.ok
    assert report.states_checked == 0
    assert report.results_checked == 0


def test_the_run_directory_is_not_needed_afterwards(tmp_path: Path) -> None:
    """The durability claim, stated as a test: delete everything the
    executor left behind and the run still verifies."""
    write_run(tmp_path)
    run_dir = tmp_path / "runs"
    for path in sorted(run_dir.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    run_dir.rmdir()

    report = verify_run(tmp_path)

    assert report.ok, report.issues


class TestWhatItCatches:
    def test_a_deleted_blob(self, tmp_path: Path) -> None:
        result, _, _ = write_run(tmp_path)
        manifest = FileEvidenceStore(tmp_path).artifacts.get(result.id)
        assert manifest is not None
        FileEvidenceStore(tmp_path).artifacts.blob_path(
            manifest.entries[0].digest
        ).unlink()

        report = verify_run(tmp_path)

        assert not report.ok
        (issue,) = report.of_kind(IntegrityIssueKind.MISSING_BLOB)
        assert issue.subject_id == result.id

    def test_a_corrupt_blob(self, tmp_path: Path) -> None:
        result, _, _ = write_run(tmp_path)
        manifest = FileEvidenceStore(tmp_path).artifacts.get(result.id)
        assert manifest is not None
        path = FileEvidenceStore(tmp_path).artifacts.blob_path(
            manifest.entries[0].digest
        )
        path.write_bytes(b"different bytes entirely")

        report = verify_run(tmp_path)

        assert report.of_kind(IntegrityIssueKind.CORRUPT_BLOB)

    def test_an_edited_result_payload(self, tmp_path: Path) -> None:
        result, _, _ = write_run(tmp_path)
        path = tmp_path / "results" / f"{result.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metrics"]["x"] = 999.0
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        report = verify_run(tmp_path)

        (issue,) = report.of_kind(IntegrityIssueKind.UNREADABLE_RECORD)
        assert "edited" in issue.detail

    def test_a_tampered_snapshot(self, tmp_path: Path) -> None:
        _, _, state = write_run(tmp_path)
        path = tmp_path / "states" / f"{state.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["objective"] = "measure something else"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        report = verify_run(tmp_path)

        assert report.of_kind(IntegrityIssueKind.UNREADABLE_SNAPSHOT)

    def test_a_state_citing_a_fact_nobody_stored(self, tmp_path: Path) -> None:
        write_run(tmp_path)
        orphan = ResearchState(objective="measure x").record_result(
            ResultRef(
                result_id="res_missing",
                spec_id=SPEC.id,
                status=ExperimentStatus.COMPLETED,
            )
        )
        FileStateStore(tmp_path).persist(orphan)

        report = verify_run(tmp_path)

        issues = report.of_kind(IntegrityIssueKind.MISSING_FACT)
        assert any("res_missing" in issue.detail for issue in issues)

    def test_it_reports_everything_at_once(self, tmp_path: Path) -> None:
        """A verifier that stopped at the first problem would make a
        broken run take as many passes as it has faults."""
        result, _, _ = write_run(tmp_path)
        manifest = FileEvidenceStore(tmp_path).artifacts.get(result.id)
        assert manifest is not None
        for entry in manifest.entries:
            FileEvidenceStore(tmp_path).artifacts.blob_path(
                entry.digest
            ).unlink()

        report = verify_run(tmp_path)

        assert len(report.of_kind(IntegrityIssueKind.MISSING_BLOB)) == 3


class TestLineage:
    """A committed snapshot claims a parent. The claim has to hold, or a
    verifier saying "intact" is saying something it cannot know."""

    def test_a_whole_lineage_verifies(self, tmp_path: Path) -> None:
        write_run(tmp_path)

        report = verify_run(tmp_path)

        assert report.ok, report.issues
        assert report.states_checked == 7

    def test_a_parent_that_is_not_stored(self, tmp_path: Path) -> None:
        _, _, state = write_run(tmp_path)
        assert state.parent_id is not None
        (tmp_path / "states" / f"{state.parent_id}.json").unlink()

        report = verify_run(tmp_path)

        (issue,) = report.of_kind(IntegrityIssueKind.INCOMPLETE_LINEAGE)
        assert issue.subject_id == state.id
        assert "cannot be walked" in issue.detail

    def test_only_the_end_of_a_chain(self, tmp_path: Path) -> None:
        """What every writer did before the lineage was persisted: keep
        the head and let its ancestry go."""
        make_run_dir(tmp_path)
        seed = ResearchState(objective="measure x")
        head = seed.upsert_question(QUESTION).upsert_hypothesis(HYPOTHESIS)
        FileStateStore(tmp_path).persist(head)

        report = verify_run(tmp_path)

        assert not report.ok
        assert report.of_kind(IntegrityIssueKind.INCOMPLETE_LINEAGE)

    def test_a_ring_is_caught_rather_than_walked_forever(self) -> None:
        """A stored ring should be unreachable: a state's id covers its
        parent id, so a cycle would need two hashes each derived from
        the other. The guard exists anyway, because "cannot happen" and
        "will hang the verifier if it does" is a poor pair. Built here
        in memory, past the store that would refuse to write it.
        """
        first = ResearchState(objective="a")
        second = ResearchState(objective="b", parent_id=first.id)
        object.__setattr__(first, "parent_id", second.id)
        issues: list[IntegrityIssue] = []

        _check_lineage([first, second], issues)

        assert any("revisits" in issue.detail for issue in issues)
        assert all(
            issue.kind is IntegrityIssueKind.INCOMPLETE_LINEAGE
            for issue in issues
        )

    def test_cold_reconstruction_reaches_every_state(
        self, tmp_path: Path
    ) -> None:
        """The forward walk, run as its own traversal: from the roots,
        following children, the reader arrives at the head."""
        _, _, head = write_run(tmp_path)
        states = FileStateStore(tmp_path)
        loaded = [states.load(found) for found in states.state_ids()]

        roots = [state for state in loaded if state.parent_id is None]
        reached = _reachable_from_roots(loaded)

        assert len(roots) == 1
        assert head.id in reached
        assert len(reached) == len(loaded)


class TestTheFundedRun:
    def test_a_funded_run_verifies_including_its_ledger(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path)
        program, run = _fund(tmp_path)

        report = verify_run(tmp_path)

        assert report.ok, report.issues
        assert program.ledger_for(run.run_id).balance() == GRANT

    def test_a_ledger_that_no_longer_replays_is_reported(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path)
        program, run = _fund(tmp_path)
        entry = program.ledger_for(run.run_id).directory / "000000.json"
        payload = json.loads(entry.read_text(encoding="utf-8"))
        payload["balance_after"]["usd"] = 5.0
        entry.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        report = verify_run(tmp_path)

        assert report.of_kind(IntegrityIssueKind.LEDGER_ISSUE)

    def test_an_unreadable_envelope_is_reported_not_raised(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path)
        envelopes = tmp_path / "program" / "envelopes"
        envelopes.mkdir(parents=True, exist_ok=True)
        (envelopes / "rune_broken.json").write_text(
            json.dumps({"id": "rune_broken"}), encoding="utf-8"
        )

        report = verify_run(tmp_path)

        assert report.of_kind(IntegrityIssueKind.LEDGER_ISSUE)

    def test_a_run_that_has_spent_still_verifies(
        self, tmp_path: Path
    ) -> None:
        """The funded snapshot keeps the grant forever, so a run that has
        spent agrees with it no longer. What the balance must agree with
        is a state the run actually reached."""
        write_run(tmp_path)
        program, run = _fund(tmp_path)
        cost = ResourceCost(wall_clock_seconds=10.0, usd=1.0)
        program.ledger_for(run.run_id).debit(
            cost, charge_id="att_1", reason="one attempt"
        )
        # Both, as the funding stage does: a charged successor whose
        # parent is only in the program store has no lineage here.
        funded = program.state_store().load(run.funded_state_id)
        states = FileStateStore(tmp_path)
        states.persist(funded)
        states.persist(funded.charge(cost))

        report = verify_run(tmp_path)

        assert report.ok, report.issues

    def test_a_balance_no_snapshot_agrees_with_is_reported(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path)
        program, run = _fund(tmp_path)
        program.ledger_for(run.run_id).debit(
            ResourceCost(usd=1.0), charge_id="att_1", reason="one attempt"
        )

        report = verify_run(tmp_path)

        (issue,) = report.of_kind(IntegrityIssueKind.LEDGER_ISSUE)
        assert "nor the budget of any" in issue.detail

    def test_an_unfunded_root_skips_the_ledger_check(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path)

        report = verify_run(tmp_path)

        assert report.ok
        assert not report.of_kind(IntegrityIssueKind.LEDGER_ISSUE)


def _fund(root: Path) -> tuple[ProgramStore, ResearchRun]:
    """A funded run recorded directly: the bridge from an admission has
    its own tests, and this one is about what the verifier can read."""
    program = ProgramStore(root / "program")
    authorization = FundingAuthorization(
        admission_record_id="arun_1",
        granted=GRANT,
        authority="Lab operator.",
    )
    directive = RunDirective(
        admission_record_id="arun_1",
        authorization_id=authorization.id,
        label="integrity fixture",
    )
    admitted = ResearchState(objective="measure x")
    funded = admitted.fund(GRANT)
    program.record_directive(directive)
    program.record_authorization(authorization)
    program.persist_state(admitted)
    program.persist_state(funded)
    grant = program.ledger_for("run_fixture").grant(authorization)
    run = program.record_run(
        ResearchRun(
            run_id="run_fixture",
            directive_id=directive.id,
            authorization_id=authorization.id,
            admission_record_id="arun_1",
            admitted_state_id=admitted.id,
            funded_state_id=funded.id,
            granted=GRANT,
            grant_entry_id=grant.id,
            label=directive.label,
            authority=authorization.authority,
            question_id=QUESTION.id,
            hypothesis_id=HYPOTHESIS.id,
            prediction_ids=(PREDICTION.id,),
        )
    )
    return program, run


class TestAttemptLinks:
    """One row of the link table per test.

    Each plants a record that is internally valid and points at
    something that is not there, because that is the only shape this
    check can catch: every store underneath already refuses content that
    contradicts its own id.
    """

    def journalled(
        self, root: Path
    ) -> tuple[ProgramStore, str, RunJournal, BudgetLedger]:
        """A funded run with one attempt that began — the shape a healthy
        step leaves behind, plus the id of a state this root really
        holds."""
        program, run = _fund(root)
        journal = program.journal_for(run.run_id)
        ledger = program.ledger_for(run.run_id)
        _, _, state = write_run(root)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.STARTED,
            state_id=state.id,
            reserved=HELD,
        )
        ledger.reserve(HELD, charge_id="att_1", reason="attempt att_1")
        return program, state.id, journal, ledger

    def close(self, journal: RunJournal, ledger: BudgetLedger) -> None:
        ledger.settle(
            ResourceCost(usd=1.0), charge_id="att_1", reason="attempt att_1"
        )
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.COMPLETED,
            reserved=HELD,
            settled=ResourceCost(usd=1.0),
            basis=SettlementBasis.MEASURED,
        )

    def links(self, root: Path) -> tuple[IntegrityIssue, ...]:
        return verify_run(root, program_root=root / "program").of_kind(
            IntegrityIssueKind.ATTEMPT_LINK
        )

    def test_a_closed_attempt_links_up(self, tmp_path: Path) -> None:
        _, _, journal, ledger = self.journalled(tmp_path)
        self.close(journal, ledger)

        assert self.links(tmp_path) == ()

    def test_a_run_with_no_journal_is_not_faulted(
        self, tmp_path: Path
    ) -> None:
        """Everything written before the journal existed is such a run."""
        _fund(tmp_path)
        write_run(tmp_path)

        assert self.links(tmp_path) == ()

    def test_money_held_for_an_attempt_nobody_began(
        self, tmp_path: Path
    ) -> None:
        _, _, journal, ledger = self.journalled(tmp_path)
        self.close(journal, ledger)
        ledger.reserve(HELD, charge_id="att_ghost", reason="a")

        (issue,) = self.links(tmp_path)

        assert issue.subject_id == "att_ghost"
        assert "never began" in issue.detail

    def test_an_attempt_holding_nothing(self, tmp_path: Path) -> None:
        _, state_id, journal, ledger = self.journalled(tmp_path)
        self.close(journal, ledger)
        journal.record(
            attempt_id="att_unheld",
            phase=AttemptPhase.STARTED,
            state_id=state_id,
            reserved=HELD,
        )
        journal.record(
            attempt_id="att_unheld",
            phase=AttemptPhase.ABANDONED,
            reserved=HELD,
            settled=HELD,
            basis=SettlementBasis.CONSERVATIVE_MAX,
        )

        details = [issue.detail for issue in self.links(tmp_path)]

        assert any("nothing was held for it" in detail for detail in details)

    def test_an_attempt_released_before_anything_was_held_is_fine(
        self, tmp_path: Path
    ) -> None:
        """The one attempt with no reservation that is not a hole: a
        release says nothing was held *and* nothing was bought, which is
        the whole content of the phase."""
        _, state_id, journal, ledger = self.journalled(tmp_path)
        self.close(journal, ledger)
        journal.record(
            attempt_id="att_released",
            phase=AttemptPhase.STARTED,
            state_id=state_id,
            reserved=HELD,
        )
        journal.record(
            attempt_id="att_released", phase=AttemptPhase.RELEASED
        )

        assert self.links(tmp_path) == ()

    def test_a_debit_answering_no_reservation(self, tmp_path: Path) -> None:
        _, _, journal, ledger = self.journalled(tmp_path)
        self.close(journal, ledger)
        ledger.debit(
            ResourceCost(usd=1.0), charge_id="att_unauthorized", reason="a"
        )

        details = [issue.detail for issue in self.links(tmp_path)]

        assert any("answers no reservation" in detail for detail in details)

    def test_a_bundle_the_store_does_not_hold(self, tmp_path: Path) -> None:
        _, _, journal, ledger = self.journalled(tmp_path)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.BUNDLE_DURABLE,
            bundle_id="bun_nowhere",
        )
        self.close(journal, ledger)

        details = [issue.detail for issue in self.links(tmp_path)]

        assert any("is not in the store" in detail for detail in details)

    def test_a_committed_phase_naming_an_unstored_state(
        self, tmp_path: Path
    ) -> None:
        _, _, journal, ledger = self.journalled(tmp_path)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.COMMITTED,
            state_id="st_nowhere",
        )
        self.close(journal, ledger)

        details = [issue.detail for issue in self.links(tmp_path)]

        assert any("did not store" in detail for detail in details)

    def test_a_collected_job_that_left_no_run_directory(
        self, tmp_path: Path
    ) -> None:
        """An attempt cannot have collected outputs from a job that left
        nothing behind."""
        _, _, journal, ledger = self.journalled(tmp_path)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.SUBMITTED,
            job_id="job_nowhere",
        )
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.OUTPUTS_DURABLE,
            job_id="job_nowhere",
        )
        self.close(journal, ledger)
        (tmp_path / "runs").mkdir(exist_ok=True)

        details = [issue.detail for issue in self.links(tmp_path)]

        assert any("left no run directory" in detail for detail in details)

    def test_a_submitted_job_that_never_ran_is_not_faulted(
        self, tmp_path: Path
    ) -> None:
        """The note doing its job, not a broken link.

        ``SUBMITTED`` is written *before* the submission precisely so a
        later process can ask whether the job exists. A crash in between
        leaves the phase and no run directory, and "it never ran" is a
        complete answer — nothing was bought, and recovery closes the
        attempt on exactly that basis. Faulting it would fault the one
        outcome the phase was ordered that way to produce.
        """
        _, _, journal, ledger = self.journalled(tmp_path)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.SUBMITTED,
            job_id="job_nowhere",
        )
        self.close(journal, ledger)
        (tmp_path / "runs").mkdir(exist_ok=True)

        assert self.links(tmp_path) == ()

    def test_another_backend_is_not_faulted_for_the_local_layout(
        self, tmp_path: Path
    ) -> None:
        """No ``runs`` directory means the jobs live somewhere this check
        knows nothing about."""
        _, _, journal, ledger = self.journalled(tmp_path)
        journal.record(
            attempt_id="att_1",
            phase=AttemptPhase.SUBMITTED,
            job_id="job_elsewhere",
        )
        self.close(journal, ledger)
        shutil.rmtree(tmp_path / "runs")

        assert self.links(tmp_path) == ()

    def test_an_attempt_still_open_is_a_run_that_owes_something(
        self, tmp_path: Path
    ) -> None:
        self.journalled(tmp_path)  # started and never closed

        (issue,) = self.links(tmp_path)

        assert issue.subject_id == "att_1"
        assert "no terminal phase" in issue.detail
