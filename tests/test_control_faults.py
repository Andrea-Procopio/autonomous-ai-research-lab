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
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autonomous_research_lab.control.controller import Controller
from autonomous_research_lab.control.lab import RuntimeRequest
from autonomous_research_lab.control.recovery import RecoveryReport, recover
from autonomous_research_lab.control.stage import Fact, StageName
from autonomous_research_lab.core.attempt import AttemptPhase, SettlementBasis
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.execution.executor import derive_job_id
from autonomous_research_lab.execution.salvage import LocalFinishedJobs
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.program.integrity import IntegrityReport, verify_run
from autonomous_research_lab.program.records import EntryKind
from autonomous_research_lab.program.store import ProgramStore
from examples.canary_chain import walk
from examples.torn_step import KILLED
from faults import Faults, FaultyLab, SimulatedCrashError


class Funded:
    """A canary walked to a funded run, ready to take one step."""

    def __init__(self, root: Path, *, repairs: bool = False) -> None:
        self.root = root
        self.repairs = repairs
        self.reached = ""
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
        lab = FaultyLab(faults, self.run.run_id, repairs=self.repairs)
        runtime = lab.runtime(self.request())
        state = FileStateStore(self.root).load(from_state)
        self.reached = from_state
        with contextlib.suppress(SimulatedCrashError):
            self.reached = runtime.step(state).state.id
        return faults

    def at_the_experiment(self) -> str:
        """One clean step past the funded head, which is the design, so
        the next step is the one that runs a job."""
        self.step(self.head)
        return self.reached

    def recover(self) -> RecoveryReport:
        return recover(
            journal=self.program.journal_for(self.run.run_id),
            ledger=self.program.ledger_for(self.run.run_id),
            bundles=self.program.bundles(),
            states=FileStateStore(self.root),
            evidence=FileEvidenceStore(self.root),
            fallback_state_id=self.head,
            jobs=LocalFinishedJobs(self.root / "runs"),
        )

    def verify(self) -> IntegrityReport:
        return verify_run(self.root, program_root=self.root / "program")

    def debits(self) -> list[str]:
        entries = self.program.ledger_for(self.run.run_id).entries()
        return [e.charge_id for e in entries if e.kind is EntryKind.DEBIT]

    def jobs(self) -> list[str]:
        runs = self.root / "runs"
        return sorted(p.name for p in runs.iterdir()) if runs.is_dir() else []

    def submitted(self) -> set[str]:
        """Every job the journal says was handed to the executor."""
        journal = self.program.journal_for(self.run.run_id)
        return {
            event.job_id
            for event in journal.events()
            if event.phase is AttemptPhase.SUBMITTED
        }

    def attempts_that_submitted(self) -> dict[str, str]:
        """Which attempt submitted which job."""
        journal = self.program.journal_for(self.run.run_id)
        return {
            event.attempt_id: event.job_id
            for event in journal.events()
            if event.phase is AttemptPhase.SUBMITTED
        }


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


# -- the step that repairs itself ----------------------------------------------
#
# The sweep above walks the *design* step, which runs no job. A step that
# executes one is longer, and the part of it that was never swept is the
# part a repair adds: a rerun submitted from inside the bounded repair
# loop. That rerun used to run with no attempt open for it and no note of
# it anywhere, so a process killed there lost the job outright — the
# spend charged to nothing, the outputs orphaned in the run directory
# with nothing on the record to find them by.


def a_repairing_step(tmp_path: Path) -> int:
    """How many durable writes a step that fails and repairs itself
    makes."""
    funded = Funded(tmp_path / "measure", repairs=True)
    return funded.step(funded.at_the_experiment()).count


def test_a_repair_rerun_is_an_attempt_of_its_own(tmp_path: Path) -> None:
    """The fix, stated as the record it leaves.

    Two jobs run in this step — the experiment that fails and the rerun
    that fixes it — and each is submitted by a different attempt, under
    an id derived from that attempt, written down before the job exists.
    """
    funded = Funded(tmp_path, repairs=True)

    funded.step(funded.at_the_experiment())

    submitted = funded.attempts_that_submitted()
    assert len(submitted) == 2, submitted
    for attempt_id, job_id in submitted.items():
        assert job_id == derive_job_id(attempt_id)
    # Both jobs really ran, and each is one the journal named first.
    assert set(funded.jobs()) == set(submitted.values())
    # Each attempt holds its own money and answers for it exactly once.
    assert len(funded.debits()) == len(set(funded.debits()))
    assert set(submitted) <= set(funded.debits())
    assert funded.verify().ok, funded.verify().issues


@pytest.mark.parametrize("after", range(1, 51))
def test_a_repairing_step_stopped_at_any_write_recovers(
    tmp_path: Path, after: int
) -> None:
    funded = Funded(tmp_path, repairs=True)
    total = funded.step(funded.at_the_experiment(), after=after).count
    if total < after:
        pytest.skip(f"a repairing step makes fewer than {after} writes")

    report = funded.recover()

    journal = funded.program.journal_for(funded.run.run_id)
    assert journal.open_attempts() == ()
    assert len(funded.debits()) == len(set(funded.debits()))
    assert funded.verify().ok, funded.verify().issues
    assert report.state_id
    # The claim this sweep exists for: wherever it died, no job ran that
    # the journal had not already written down. A job on disk that no
    # attempt submitted is work the run paid for and cannot account for.
    assert set(funded.jobs()) <= funded.submitted()


@pytest.mark.parametrize("after", range(1, 51))
def test_a_recovered_repairing_run_takes_another_step(
    tmp_path: Path, after: int
) -> None:
    """Recovering is for continuing, here too."""
    funded = Funded(tmp_path, repairs=True)
    if funded.step(funded.at_the_experiment(), after=after).count < after:
        pytest.skip(f"a repairing step makes fewer than {after} writes")
    report = funded.recover()

    funded.step(report.state_id)

    assert funded.program.journal_for(
        funded.run.run_id
    ).open_attempts() == ()
    assert len(funded.debits()) == len(set(funded.debits()))
    assert funded.verify().ok, funded.verify().issues
    assert set(funded.jobs()) <= funded.submitted()


# -- the job that finished before its record did --------------------------------
#
# The window between a job finishing and OUTPUTS_DURABLE landing is where
# collect-finished recovery earns its keep: the work is bought, the outputs
# are on disk, and nothing in the journal says so yet. The harness ticks a
# position the moment a job returns, so the sweep can stop exactly there.


def positions_labelled(tmp_path: Path, prefix: str) -> list[int]:
    """Where in a clean repairing step the writes matching ``prefix``
    land. The canary is deterministic, so a measuring run's positions
    hold for every later run."""
    funded = Funded(tmp_path / "positions", repairs=True)
    faults = funded.step(funded.at_the_experiment())
    assert faults.writes is not None
    return [
        index + 1
        for index, label in enumerate(faults.writes)
        if label.startswith(prefix)
    ]


def test_a_finished_job_is_collected_not_abandoned(tmp_path: Path) -> None:
    """Killed the instant its job finished, before anything recorded that
    it had: the attempt completes with the job's measured cost, and no
    conservative charge appears anywhere."""
    (position, *_rest) = positions_labelled(tmp_path, "job ")
    funded = Funded(tmp_path, repairs=True)
    funded.step(funded.at_the_experiment(), after=position)

    funded.recover()

    journal = funded.program.journal_for(funded.run.run_id)
    assert journal.open_attempts() == ()
    events = journal.events()
    assert not any(e.phase is AttemptPhase.ABANDONED for e in events)
    assert not any(
        e.basis is SettlementBasis.CONSERVATIVE_MAX for e in events
    )
    # The settlement is the job's own recorded cost, to the byte.
    (submitted,) = [e for e in events if e.phase is AttemptPhase.SUBMITTED]
    (completed,) = [
        e
        for e in events
        if e.phase is AttemptPhase.COMPLETED
        and e.attempt_id == submitted.attempt_id
    ]
    record = json.loads(
        (funded.root / "runs" / submitted.job_id / "job.json").read_text()
    )
    assert completed.settled.wall_clock_seconds == pytest.approx(
        record["result"]["cost"]["wall_clock_seconds"]
    )
    assert completed.basis is SettlementBasis.MEASURED
    assert funded.verify().ok, funded.verify().issues


def test_a_collected_attempt_is_not_re_run(tmp_path: Path) -> None:
    """The point of collecting: the work is kept, so continuing does not
    buy it twice."""
    (position, *_rest) = positions_labelled(tmp_path, "job ")
    funded = Funded(tmp_path, repairs=True)
    funded.step(funded.at_the_experiment(), after=position)
    report = funded.recover()
    journal = funded.program.journal_for(funded.run.run_id)
    salvaged = journal.attempts()

    funded.step(report.state_id)

    # The salvaged attempt's history is untouched; new work is new
    # attempts with new jobs, and nothing was submitted twice.
    for attempt_id in salvaged:
        assert journal.last_for(attempt_id) is not None
        last = journal.last_for(attempt_id)
        assert last is not None and last.phase.is_terminal
    assert len(funded.debits()) == len(set(funded.debits()))
    assert set(funded.jobs()) <= funded.submitted()
    assert funded.verify().ok, funded.verify().issues


def test_a_kill_after_outputs_durable_salvages_the_same_way(
    tmp_path: Path,
) -> None:
    """One phase later — the outputs are recorded, the bundle is not —
    the same collection finishes the attempt."""
    (position, *_rest) = positions_labelled(tmp_path, "outputs_durable")
    funded = Funded(tmp_path, repairs=True)
    funded.step(funded.at_the_experiment(), after=position)

    funded.recover()

    journal = funded.program.journal_for(funded.run.run_id)
    assert journal.open_attempts() == ()
    assert not any(
        e.phase is AttemptPhase.ABANDONED for e in journal.events()
    )
    assert funded.verify().ok, funded.verify().issues


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


@pytest.mark.parametrize("after", [18, 20, 21, 23])
def test_a_killed_repair_rerun_is_answered_by_the_next_process(
    tmp_path: Path, after: int
) -> None:
    """The same two processes, over the writes a repair rerun makes.

    These positions had no answer before the rerun was an attempt; two of
    them (the job finished, the outputs recorded) now have a *better*
    answer than abandonment — the next process finds the finished job on
    disk, collects it, and completes the attempt with its measured cost.
    Either way both open attempts are settled separately, and the run
    still verifies.
    """
    root = tmp_path / "run"

    killed = torn_step(root, "--repair", "--kill-after", str(after))
    assert killed.returncode == KILLED, killed.stderr

    resumed = torn_step(root)

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "the two records agree: True" in resumed.stdout
    assert "intact" in resumed.stdout
    # Two attempts were open, not one: the step's own and the rerun's,
    # and both got an answer. Which answer depends on where it died —
    # released before anything was held, abandoned once something was,
    # finished from a durable bundle — and that it got one at all is the
    # part that is new.
    assert "open attempts ['att_" in resumed.stdout
    assert resumed.stdout.count(" -> ") == 2, resumed.stdout
