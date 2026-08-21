"""Killing a step at one durable write, and finishing it afterwards.

The Task 6D artifact. Two commands, two processes, nothing shared but
files::

    python -m examples.torn_step --run-root /tmp/torn --kill-after 9
    python -m examples.torn_step --run-root /tmp/torn

The first walks the canary to a funded run, starts one step, and is
killed — really killed, ``os._exit`` with no unwinding and no cleanup —
the instant its ninth durable write lands. The second knows nothing
about the first except what is on disk. It reads the attempt journal,
answers whatever was left open, and prints the ledger, the journal and
the verifier side by side.

Run it at every position and the pattern is the whole point: wherever
the first process died, the second finds a run that owes nothing, has
charged each attempt exactly once, and verifies from cold.

``--repair`` tears a longer step instead: the one that runs a job, has
it fail, and repairs it inside the same step. The rerun is an attempt of
its own — its own reservation, its own ``STARTED``/``SUBMITTED``, its
own derived job id, all of it on disk before the job exists — which is
what lets a killed repair be answered rather than lost.

``--kill-after`` with no number lists the writes a clean step makes, so
there is no guessing about which positions exist.

The fault machinery lives here rather than under ``tests`` because the
tests import it from here: a driver that demonstrates a mechanism and a
harness that exercises it are the same code, and two copies of it would
drift.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from autonomous_research_lab.control.controller import Controller
from autonomous_research_lab.control.lab import RuntimeRequest
from autonomous_research_lab.control.recovery import recover
from autonomous_research_lab.control.stage import Fact, StageName
from autonomous_research_lab.core.attempt import SettlementBasis
from autonomous_research_lab.core.evidence import Evidence
from autonomous_research_lab.core.experiment import ExperimentResult
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.runner import JobRunner
from autonomous_research_lab.execution.salvage import LocalFinishedJobs
from autonomous_research_lab.literature.retrieval import LiteratureProvider
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.persistence.commit_store import CommitBundleStore
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.program.integrity import verify_run
from autonomous_research_lab.program.journal import RunJournal
from autonomous_research_lab.program.ledger import BudgetLedger
from autonomous_research_lab.program.store import ProgramStore
from autonomous_research_lab.runtime.providers import ModelProvider
from examples.canary_chain import walk
from examples.canary_lab import CanaryLab

_GUESSED = SettlementBasis.CONSERVATIVE_MAX

KILLED = 9
"""The exit code a killed process leaves. Chosen to be unmistakable: no
part of the system returns it."""


class SimulatedCrashError(BaseException):
    """The simulated crash. Carries the write it came after.

    A ``BaseException``, deliberately: a crash is not a failure a program
    observes, and nothing in the system may be allowed to catch and
    absorb it. Derived from ``RuntimeError`` it would be swallowed by the
    runtime's role-failure handling — ``except Exception`` around
    ``perform`` — and every position inside a role invocation would
    quietly test "the role raised" instead of "the process died". The
    hard variant (``os._exit``) needs no such care, which is the point of
    having both."""


@dataclass
class Faults:
    """A stop after the ``after``-th durable write, or none at all."""

    after: int | None = None
    hard: bool = False
    """Whether to die rather than raise. A raised exception unwinds, runs
    ``finally`` blocks and lets callers tidy up, which is exactly what a
    killed process does not do; ``os._exit`` is the honest simulation and
    the one the driver uses. Tests raise instead, because a test that
    exits takes the test runner with it."""

    writes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.writes is None:
            self.writes = []

    def tick(self, label: str) -> None:
        assert self.writes is not None
        self.writes.append(label)
        if self.after is None or len(self.writes) != self.after:
            return
        if self.hard:
            print(f"killed after write {self.after}: {label}", flush=True)
            os._exit(KILLED)
        raise SimulatedCrashError(
            f"stopped after write {self.after}: {label}"
        )

    @property
    def count(self) -> int:
        assert self.writes is not None
        return len(self.writes)

    def at(self, position: int) -> str:
        assert self.writes is not None
        return self.writes[position - 1]


class FaultyStates(FileStateStore):
    def __init__(self, root: Path | str, faults: Faults) -> None:
        super().__init__(root)
        self._faults = faults

    def persist(self, state: ResearchState) -> Path:
        path = super().persist(state)
        self._faults.tick(f"snapshot {state.id}")
        return path


class FaultyEvidence(FileEvidenceStore):
    def __init__(self, root: Path | str, faults: Faults) -> None:
        super().__init__(root)
        self._faults = faults

    def record_result(self, result: ExperimentResult) -> ExperimentResult:
        recorded = super().record_result(result)
        self._faults.tick(f"result {recorded.id}")
        return recorded

    def record_evidence(self, evidence: Evidence) -> Evidence:
        recorded = super().record_evidence(evidence)
        self._faults.tick(f"evidence {recorded.id}")
        return recorded


class FaultyLedger(BudgetLedger):
    def __init__(self, root: Path | str, run_id: str, faults: Faults) -> None:
        super().__init__(root, run_id)
        self._faults = faults

    def reserve(self, cost, *, charge_id, reason):  # type: ignore[no-untyped-def]
        entry = super().reserve(cost, charge_id=charge_id, reason=reason)
        self._faults.tick(f"reservation {charge_id}")
        return entry

    def settle(self, cost, *, charge_id, reason):  # type: ignore[no-untyped-def]
        settlement = super().settle(cost, charge_id=charge_id, reason=reason)
        self._faults.tick(f"settlement {charge_id}")
        return settlement

    def release(self, *, charge_id, reason):  # type: ignore[no-untyped-def]
        entry = super().release(charge_id=charge_id, reason=reason)
        self._faults.tick(f"release {charge_id}")
        return entry


class FaultyJournal(RunJournal):
    def __init__(self, root: Path | str, run_id: str, faults: Faults) -> None:
        super().__init__(root, run_id)
        self._faults = faults

    def record(self, **fields: object):  # type: ignore[no-untyped-def]
        event = super().record(**fields)  # type: ignore[arg-type]
        self._faults.tick(f"{event.phase} {event.attempt_id}")
        return event


@dataclass
class FaultyRunner:
    """Ticks the moment a job's outputs exist on disk.

    The window between a job finishing and ``OUTPUTS_DURABLE`` landing is
    exactly where collect-finished recovery earns its keep, and without a
    tick there the sweep cannot stop in it. Sits *inside* the journalling
    runner, so the order of ticks matches the order of durable facts:
    submitted, job's outputs, outputs recorded.
    """

    inner: JobRunner
    faults: Faults

    def run(
        self, job: ExperimentJob, attempt_id: str = "", /
    ) -> ExperimentResult:
        result = self.inner.run(job, attempt_id)
        self.faults.tick(f"job {job.id} finished")
        return result


class FaultyBundles(CommitBundleStore):
    def __init__(self, root: Path | str, faults: Faults) -> None:
        super().__init__(root)
        self._faults = faults

    def record(self, bundle):  # type: ignore[no-untyped-def]
        bundle_id = super().record(bundle)
        self._faults.tick(f"bundle {bundle_id}")
        return bundle_id


class FaultyLab:
    """The canary's instruments, writing to stores that can stop.

    A wrapper rather than a subclass: ``CanaryLab`` is a frozen dataclass
    and inheriting from one to add mutable state is a fight with the
    language for no benefit. Delegation also makes the point plainly —
    nothing about the *instruments* changes, only where their writes go.
    """

    def __init__(
        self, faults: Faults, run_id: str, *, repairs: bool = False
    ) -> None:
        self.faults = faults
        self.run_id = run_id
        self._inner = CanaryLab(repairs=repairs)

    def model_provider(self, stage: StageName, /) -> ModelProvider:
        return self._inner.model_provider(stage)

    def literature_provider(self) -> LiteratureProvider:
        return self._inner.literature_provider()

    def runtime(self, request: RuntimeRequest, /) -> ResearchRuntime:
        program = request.root / "program"
        self._inner = replace(
            self._inner,
            runner_middleware=lambda inner: FaultyRunner(inner, self.faults),
        )
        return self._inner.runtime(
            replace(
                request,
                evidence=FaultyEvidence(request.root, self.faults),
                states=FaultyStates(request.root, self.faults),
                ledger=FaultyLedger(program, self.run_id, self.faults),
                journal=FaultyJournal(program, self.run_id, self.faults),
                bundles=FaultyBundles(program, self.faults),
            )
        )


# -- the driver ----------------------------------------------------------------


def funded(root: Path) -> tuple[ProgramStore, str, str]:
    """Walk the canary to a funded run, if it is not one already."""
    walk(root, stop_after=StageName.FUNDING)
    controller = Controller(root)
    (investigation,) = controller.investigations.investigations()
    log = controller.investigations.log_for(investigation.investigation_id)
    program = ProgramStore(root / "program")
    (run,) = program.runs()
    return program, run.run_id, log.facts().require(Fact.STATE_ID)


def request_for(root: Path, program: ProgramStore, run_id: str) -> RuntimeRequest:
    return RuntimeRequest(
        root=root,
        evidence=FileEvidenceStore(root),
        states=FileStateStore(root),
        ledger=program.ledger_for(run_id),
        journal=program.journal_for(run_id),
        bundles=program.bundles(),
    )


def one_step(
    root: Path, from_state: str, faults: Faults, *, repairs: bool = False
) -> tuple[Faults, str]:
    program, run_id, _ = funded(root)
    runtime = FaultyLab(faults, run_id, repairs=repairs).runtime(
        request_for(root, program, run_id)
    )
    report = runtime.step(FileStateStore(root).load(from_state))
    return faults, report.state.id


def start_of_the_killable_step(root: Path, *, repairs: bool) -> str:
    """Where the step this driver tears begins.

    Plainly, the funded head: the canary's first step designs the
    experiment. With ``--repair`` it is one step further on, because the
    step worth tearing is the one that runs a job — fails, and repairs
    itself inside the same step. The design is taken cleanly first; it is
    the setting, not the subject.
    """
    _, _, head = funded(root)
    if not repairs:
        return head
    _, designed = one_step(root, head, Faults())
    return designed


def kill(root: Path, after: int, *, repairs: bool) -> int:
    start = start_of_the_killable_step(root, repairs=repairs)
    one_step(root, start, Faults(after=after, hard=True), repairs=repairs)
    print("the step finished; nothing was killed")
    return 0


def enumerate_writes(root: Path, *, repairs: bool) -> int:
    """List the durable writes one clean step makes, in a scratch root."""
    scratch = root / "scratch"
    start = start_of_the_killable_step(scratch, repairs=repairs)
    faults, _ = one_step(scratch, start, Faults(), repairs=repairs)
    assert faults.writes is not None
    for position, label in enumerate(faults.writes, 1):
        print(f"  {position:>3}  {label}")
    print(f"\n{faults.count} durable write(s) in one step")
    return 0


def resume(root: Path) -> int:
    program, run_id, head = funded(root)
    journal = program.journal_for(run_id)
    ledger = program.ledger_for(run_id)

    print(f"run           {run_id}")
    print(f"open attempts {list(journal.open_attempts()) or 'none'}")
    print(f"reservations  {[e.charge_id for e in ledger.reservations()] or 'none'}")
    print()

    report = recover(
        journal=journal,
        ledger=ledger,
        bundles=program.bundles(),
        states=FileStateStore(root),
        evidence=FileEvidenceStore(root),
        fallback_state_id=head,
        jobs=LocalFinishedJobs(root / "runs"),
    )
    print(f"recovery: {report.summary()}")
    for answered in report.recoveries:
        print(
            f"  {answered.attempt_id}: {answered.left_at} -> "
            f"{answered.resolution}"
        )
    print()

    print("ledger")
    for entry in ledger.entries():
        print(f"  {entry.sequence:>3}  {entry.kind:<12} {entry.charge_id}")
    print(f"  balance   {ledger.balance()}")
    print(f"  available {ledger.available()}")
    print()

    print("journal")
    for event in journal.events():
        basis = (
            "" if event.basis is SettlementBasis.NONE else f"  [{event.basis}]"
        )
        print(
            f"  {event.sequence:>3}  {event.phase:<16} "
            f"{event.attempt_id}{basis}"
        )
    unknown = [e for e in journal.events() if e.basis is _GUESSED]
    if unknown:
        print()
        print(
            "  the actual cost of the following is unknown; each was "
            "charged its authorization:"
        )
        for event in unknown:
            print(f"    {event.attempt_id}  settled {event.settled}")
    print()

    continued = FileStateStore(root).load(report.state_id)
    agree = continued.budget == ledger.balance()
    print(f"state {continued.id} holds {continued.budget}")
    print(f"the two records agree: {agree}")

    integrity = verify_run(root, program_root=root / "program")
    print()
    for issue in integrity.issues:
        print(f"  {issue.kind}: {issue.subject_id}: {issue.detail}")
    print("intact" if integrity.ok else "FATAL: the run does not verify.")
    return 0 if integrity.ok and agree else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, required=True, help="the run directory"
    )
    parser.add_argument(
        "--kill-after",
        type=int,
        nargs="?",
        const=0,
        help="die after this many durable writes; with no number, list them",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "tear the step that runs a job, fails, and repairs itself "
            "inside itself — the rerun is an attempt of its own"
        ),
    )
    arguments = parser.parse_args()
    root = arguments.run_root.resolve()
    if arguments.kill_after is None:
        return resume(root)
    if arguments.kill_after == 0:
        return enumerate_writes(root, repairs=arguments.repair)
    return kill(root, arguments.kill_after, repairs=arguments.repair)


if __name__ == "__main__":
    sys.exit(main())
