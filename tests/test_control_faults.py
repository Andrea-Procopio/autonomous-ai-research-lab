"""Killing a step at every durable write, and finishing it afterwards.

One clean step is measured to find how many durable writes it makes.
Then the same step is run again once per write, stopped immediately
after that write, and recovered. Every stop must leave a run that
verifies, owes nothing, has charged each attempt exactly once, and can
take another step.

The sweep is the test. Any one of these positions passing proves little;
all of them passing is the claim.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path

import pytest

from autonomous_research_lab.control.controller import Controller
from autonomous_research_lab.control.lab import RuntimeRequest
from autonomous_research_lab.control.recovery import RecoveryReport, recover
from autonomous_research_lab.control.stage import Fact, StageName
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.program.integrity import IntegrityReport, verify_run
from autonomous_research_lab.program.records import EntryKind
from autonomous_research_lab.program.store import ProgramStore
from examples.canary_chain import walk
from examples.torn_step import KILLED
from faults import Faults, FaultyLab, SimulatedCrashError


class Funded:
    """A canary walked to a funded run, ready to take one step."""

    def __init__(self, root: Path) -> None:
        self.root = root
        walk(root, stop_after=StageName.FUNDING)
        controller = Controller(root)
        (investigation,) = controller.investigations.investigations()
        log = controller.investigations.log_for(
            investigation.investigation_id
        )
        self.head = log.facts().require(Fact.STATE_ID)
        self.program = ProgramStore(root / "program")
        (self.run,) = self.program.runs()

    def request(self) -> RuntimeRequest:
        return RuntimeRequest(
            root=self.root,
            evidence=FileEvidenceStore(self.root),
            states=FileStateStore(self.root),
            ledger=self.program.ledger_for(self.run.run_id),
            journal=self.program.journal_for(self.run.run_id),
            bundles=self.program.bundles(),
        )

    def step(self, from_state: str, *, after: int | None = None) -> Faults:
        """One step, stopped after the ``after``-th durable write."""
        faults = Faults(after=after)
        lab = FaultyLab(faults, self.run.run_id)
        runtime = lab.runtime(self.request())
        state = FileStateStore(self.root).load(from_state)
        with contextlib.suppress(SimulatedCrashError):
            runtime.step(state)
        return faults

    def recover(self) -> RecoveryReport:
        return recover(
            journal=self.program.journal_for(self.run.run_id),
            ledger=self.program.ledger_for(self.run.run_id),
            bundles=self.program.bundles(),
            states=FileStateStore(self.root),
            evidence=FileEvidenceStore(self.root),
            fallback_state_id=self.head,
        )

    def verify(self) -> IntegrityReport:
        return verify_run(self.root, program_root=self.root / "program")

    def debits(self) -> list[str]:
        entries = self.program.ledger_for(self.run.run_id).entries()
        return [e.charge_id for e in entries if e.kind is EntryKind.DEBIT]

    def jobs(self) -> list[str]:
        runs = self.root / "runs"
        return sorted(p.name for p in runs.iterdir()) if runs.is_dir() else []


def a_clean_step(tmp_path: Path) -> int:
    """How many durable writes one uninterrupted step makes."""
    funded = Funded(tmp_path / "measure")
    return funded.step(funded.head).count


def test_a_step_makes_several_durable_writes(tmp_path: Path) -> None:
    """The sweep below is only meaningful if there is something to sweep
    over."""
    assert a_clean_step(tmp_path) >= 6


@pytest.mark.parametrize("after", range(1, 12))
def test_a_step_stopped_at_any_write_recovers(
    tmp_path: Path, after: int
) -> None:
    funded = Funded(tmp_path)
    total = funded.step(funded.head, after=after).count
    if total < after:
        pytest.skip(f"a step makes fewer than {after} durable writes")

    report = funded.recover()

    # Nothing is owed, and nothing was charged twice.
    journal = funded.program.journal_for(funded.run.run_id)
    assert journal.open_attempts() == ()
    assert len(funded.debits()) == len(set(funded.debits()))
    # The run still verifies from cold, whatever it was stopped at.
    assert funded.verify().ok, funded.verify().issues
    # And it can go on.
    assert report.state_id


@pytest.mark.parametrize("after", range(1, 12))
def test_a_recovered_run_takes_another_step(
    tmp_path: Path, after: int
) -> None:
    """The point of recovering is to continue, not merely to tidy up."""
    funded = Funded(tmp_path)
    if funded.step(funded.head, after=after).count < after:
        pytest.skip(f"a step makes fewer than {after} durable writes")
    report = funded.recover()

    funded.step(report.state_id)

    assert funded.program.journal_for(
        funded.run.run_id
    ).open_attempts() == ()
    assert len(funded.debits()) == len(set(funded.debits()))
    assert funded.verify().ok, funded.verify().issues


def test_a_job_is_never_submitted_twice(tmp_path: Path) -> None:
    """Job ids are derived from attempt ids and the executor refuses a
    second submission of one. A retry after a crash is a new attempt, so
    it is a new job — and the crashed one is still on disk, collected or
    ruled out rather than run again."""
    funded = Funded(tmp_path)
    funded.step(funded.head, after=4)
    before = funded.jobs()
    report = funded.recover()

    funded.step(report.state_id)

    after_jobs = funded.jobs()
    assert len(after_jobs) == len(set(after_jobs))  # each job id once
    assert set(before) <= set(after_jobs)  # and the crashed one survives


def test_an_uninterrupted_step_needs_no_recovery(tmp_path: Path) -> None:
    funded = Funded(tmp_path)
    funded.step(funded.head)

    report = funded.recover()

    assert not report.recoveries
    assert funded.verify().ok


# -- and again with nothing shared but files -----------------------------------

REPO = Path(__file__).resolve().parent.parent


def torn_step(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.torn_step",
            "--run-root",
            str(root),
            *arguments,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


@pytest.mark.parametrize("after", [2, 5, 10, 14])
def test_one_process_dies_and_another_finishes_the_step(
    tmp_path: Path, after: int
) -> None:
    """The real thing: the first process is killed outright — no
    unwinding, no cleanup, no shared memory — and a second process that
    saw none of it reads the records and finishes the job.

    A handful of positions rather than all of them, because each costs
    two interpreters. The exhaustive sweep is the in-process one above;
    this is the proof that it is not an artifact of staying in one
    process.
    """
    root = tmp_path / "run"

    killed = torn_step(root, "--kill-after", str(after))
    assert killed.returncode == KILLED, killed.stderr
    assert "killed after write" in killed.stdout

    resumed = torn_step(root)

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "the two records agree: True" in resumed.stdout
    assert "intact" in resumed.stdout
