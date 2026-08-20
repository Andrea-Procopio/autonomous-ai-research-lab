"""Verifying a whole run from cold.

Every test here writes a run in one store and verifies it through a
different one, because the claim under test is about what survives a
process, not about what a live object remembers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.program.authorization import FundingAuthorization
from autonomous_research_lab.program.directive import RunDirective
from autonomous_research_lab.program.integrity import (
    IntegrityIssueKind,
    verify_run,
)
from autonomous_research_lab.program.records import ResearchRun
from autonomous_research_lab.program.store import ProgramStore

MANIFEST_FILENAME = "manifest.json"
GRANT = ResearchBudget(wall_clock_seconds=100.0, usd=10.0, model_tokens=1_000)


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
    state = (
        ResearchState(objective="measure x")
        .upsert_question(QUESTION)
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
    FileStateStore(root).persist(state)
    return result, evidence, state


def test_a_written_run_verifies_from_a_cold_start(tmp_path: Path) -> None:
    write_run(tmp_path)

    report = verify_run(tmp_path)

    assert report.ok, report.issues
    assert report.states_checked == 1
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
        funded = program.state_store().load(run.funded_state_id)
        FileStateStore(tmp_path).persist(funded.charge(cost))

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
