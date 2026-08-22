"""Finishing what a killed process started.

Each test plants the record a crash would have left and asks recovery to
answer it. What is pinned is the partition: a durable bundle recovers
completely, and everything earlier is paid for in full and closed with
nothing to show. And, in both directions, that running recovery twice
changes nothing.
"""

from __future__ import annotations

from pathlib import Path

from autonomous_research_lab.control import recovery as recovery_module
from autonomous_research_lab.control.recovery import (
    FinishedJobs,
    RecoveryReport,
    recover,
)
from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.attempt import (
    ActionAttempt,
    ActionOutcome,
    AttemptPhase,
    AttemptStatus,
    SettlementBasis,
)
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.commit import CommitBundle
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentStatus,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.proposals import HypothesisProposal
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.persistence.commit_store import CommitBundleStore
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.program.authorization import FundingAuthorization
from autonomous_research_lab.program.journal import RunJournal
from autonomous_research_lab.program.ledger import BudgetLedger
from autonomous_research_lab.program.records import EntryKind

GRANT = ResearchBudget(wall_clock_seconds=1000.0, usd=100.0, model_tokens=10_000)
HELD = ResourceCost(usd=10.0, model_tokens=500)
SPENT = ResourceCost(usd=4.0, model_tokens=200)
HYPOTHESIS = Hypothesis(statement="X causes Y.")
ACTION = ResearchAction(
    action_type=ResearchActionType.GENERATE_HYPOTHESIS, rationale="r"
)


class Crash:
    """The durable records a killed process leaves behind."""

    def __init__(self, root: Path) -> None:
        self.states = FileStateStore(root)
        self.evidence = InMemoryEvidenceStore()
        self.journal = RunJournal(root / "program", "run_1")
        self.ledger = BudgetLedger(root / "program", "run_1")
        self.bundles = CommitBundleStore(root / "program")
        self.ledger.grant(
            FundingAuthorization(
                admission_record_id="arun_1",
                granted=GRANT,
                authority="Lab operator.",
            )
        )
        self.head = ResearchState(objective="o").fund(GRANT)
        self.states.persist(self.head)
        self.attempt = ActionAttempt(action=ACTION).started()
        self.begun = self.head.begin_attempt(self.attempt)

    def started(self, job_id: str = "") -> None:
        """As far as `_open_attempt` gets: the begun state is durable,
        the phase is written, the money is held."""
        self.states.persist(self.begun)
        self.journal.record(
            attempt_id=self.attempt.id,
            phase=AttemptPhase.STARTED,
            state_id=self.begun.id,
            job_id=job_id,
            reserved=HELD,
        )
        self.ledger.reserve(HELD, charge_id=self.attempt.id, reason="a")

    def submitted(self, job_id: str, *, registered: str | None = None) -> None:
        """One step further: the job id was pre-registered on STARTED and
        the submission went out. ``registered`` lets a test disagree the
        two on purpose."""
        self.started(job_id if registered is None else registered)
        self.journal.record(
            attempt_id=self.attempt.id,
            phase=AttemptPhase.SUBMITTED,
            job_id=job_id,
        )

    def bundle(self) -> CommitBundle:
        return CommitBundle(
            attempt_id=self.attempt.id,
            outcome=ActionOutcome(
                status=AttemptStatus.SUCCEEDED,
                produced=(HYPOTHESIS.id,),
                actual_cost=SPENT,
            ),
            proposals=(HypothesisProposal(HYPOTHESIS, proposer="t"),),
        )

    def bundle_durable(self) -> str:
        self.started()
        bundle_id = self.bundles.record(self.bundle())
        self.journal.record(
            attempt_id=self.attempt.id,
            phase=AttemptPhase.BUNDLE_DURABLE,
            bundle_id=bundle_id,
        )
        return bundle_id

    def run(
        self,
        fallback: str | None = None,
        *,
        jobs: FinishedJobs | None = None,
    ) -> RecoveryReport:
        return recover(
            journal=self.journal,
            ledger=self.ledger,
            bundles=self.bundles,
            states=self.states,
            evidence=self.evidence,
            fallback_state_id=fallback or self.head.id,
            jobs=jobs,
        )

    def kinds(self) -> list[str]:
        return [str(e.kind) for e in self.ledger.entries()]


class TestNothingToDo:
    def test_an_untouched_run_recovers_nothing(self, tmp_path: Path) -> None:
        crash = Crash(tmp_path)

        report = crash.run()

        assert not report.anything_to_do
        assert report.state_id == crash.head.id
        assert crash.kinds() == ["grant"]


class TestAnAttemptWithNoBundle:
    """Nothing on disk says what it cost, and something was probably
    spent. It pays its authorization in full."""

    def test_it_is_abandoned_and_charged_the_whole_reservation(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.started()

        report = crash.run()

        (recovery,) = report.recoveries
        assert recovery.resolution is AttemptPhase.ABANDONED
        assert recovery.settled == HELD
        assert crash.ledger.balance().usd == 90.0
        assert crash.ledger.reservations() == ()

    def test_the_charge_is_not_recorded_as_a_measurement(
        self, tmp_path: Path
    ) -> None:
        """The ledger moved by the reservation, and nobody knows what the
        attempt cost. Recording the first as if it were the second would
        leave the accounting safe and the history false."""
        crash = Crash(tmp_path)
        crash.started()

        (recovery,) = crash.run().recoveries

        assert recovery.basis is SettlementBasis.CONSERVATIVE_MAX
        assert not recovery.actual_cost_known
        assert "unknown" in recovery.detail

    def test_a_conservative_charge_is_not_a_breach(
        self, tmp_path: Path
    ) -> None:
        """Only a measurement can overrun. A charge that *is* the
        authorization cannot exceed it, and calling it a breach would
        turn every crash into a budget incident."""
        crash = Crash(tmp_path)
        crash.started()

        (recovery,) = crash.run().recoveries

        assert not recovery.breached
        assert crash.journal.breaches() == ()

    def test_the_ledger_says_why_it_charged_what_it_did(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.started()

        crash.run()

        (debit,) = [
            e for e in crash.ledger.entries() if e.kind is EntryKind.DEBIT
        ]
        assert "actual cost unknown" in debit.reason

    def test_the_state_is_charged_the_same_amount(
        self, tmp_path: Path
    ) -> None:
        """Two records of one number. Letting one move without the other
        is the disagreement the next step would fail closed on."""
        crash = Crash(tmp_path)
        crash.started()

        report = crash.run()

        continued = crash.states.load(report.state_id)
        assert continued.budget == crash.ledger.balance()

    def test_nothing_is_released(self, tmp_path: Path) -> None:
        """Releasing would say the money was never spent, and nothing on
        disk supports that."""
        crash = Crash(tmp_path)
        crash.started()

        crash.run()

        assert "release" not in crash.kinds()
        assert crash.kinds() == ["grant", "reservation", "debit"]

    def test_the_journal_closes_it(self, tmp_path: Path) -> None:
        crash = Crash(tmp_path)
        crash.started()

        crash.run()

        assert crash.journal.open_attempts() == ()
        last = crash.journal.last_for(crash.attempt.id)
        assert last is not None
        assert last.phase is AttemptPhase.ABANDONED
        assert last.settled == HELD
        # ...and the record does not claim that is what it cost.
        assert last.basis is SettlementBasis.CONSERVATIVE_MAX
        assert not last.actual_cost_known

    def test_no_successor_is_invented(self, tmp_path: Path) -> None:
        """The work was bought; the reasoning that would have used it is
        gone, and recovery does not make it up."""
        crash = Crash(tmp_path)
        crash.started()

        report = crash.run()

        continued = crash.states.load(report.state_id)
        assert continued.hypotheses == ()


class TestADurableBundle:
    """The expensive case, and the one that recovers completely."""

    def test_the_bundle_is_applied_and_the_successor_persisted(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.bundle_durable()

        report = crash.run()

        (recovery,) = report.recoveries
        assert recovery.resolution is AttemptPhase.COMPLETED
        recovered = crash.states.load(report.state_id)
        assert [h.id for h in recovered.hypotheses] == [HYPOTHESIS.id]
        assert recovered.attempts[0].status is AttemptStatus.SUCCEEDED

    def test_it_settles_what_the_bundle_says_it_cost(
        self, tmp_path: Path
    ) -> None:
        """Not the reservation: here the run knows the real number, and
        the record says so."""
        crash = Crash(tmp_path)
        crash.bundle_durable()

        report = crash.run()

        assert report.recoveries[0].settled == SPENT
        assert report.recoveries[0].basis is SettlementBasis.MEASURED
        assert report.recoveries[0].actual_cost_known
        assert crash.ledger.balance().usd == 96.0
        recovered = crash.states.load(report.state_id)
        assert recovered.budget == crash.ledger.balance()

    def test_it_reaches_the_state_the_live_step_would_have(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.bundle_durable()

        report = crash.run()

        assert report.state_id == crash.states.load(report.state_id).id
        assert crash.journal.open_attempts() == ()

    def test_a_successor_already_stored_is_adopted_not_recomputed(
        self, tmp_path: Path
    ) -> None:
        """The crash between persisting the successor and journalling
        the phase after it."""
        crash = Crash(tmp_path)
        crash.bundle_durable()
        first = crash.run()
        before = set(crash.states.state_ids())

        # A second recovery, from where the first left the run, must
        # find nothing left to answer.
        again = crash.run(first.state_id)

        assert not again.anything_to_do
        assert again.state_id == first.state_id
        assert set(crash.states.state_ids()) == before


class TestRunningItTwice:
    def test_an_abandoned_attempt_is_not_charged_again(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.started()

        first = crash.run()
        balance = crash.ledger.balance()
        crash.run(first.state_id)

        assert crash.ledger.balance() == balance
        assert crash.kinds() == ["grant", "reservation", "debit"]

    def test_a_recovered_bundle_is_not_committed_again(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.bundle_durable()

        first = crash.run()
        crash.run(first.state_id)

        debits = [e for e in crash.ledger.entries() if e.kind is EntryKind.DEBIT]
        assert len(debits) == 1
        assert crash.states.load(first.state_id).hypotheses[0].id == HYPOTHESIS.id


class TestABreachIsReported:
    def test_spending_past_the_authorization_is_flagged(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.started()
        overrun = CommitBundle(
            attempt_id=crash.attempt.id,
            outcome=ActionOutcome(
                status=AttemptStatus.SUCCEEDED,
                produced=(HYPOTHESIS.id,),
                actual_cost=ResourceCost(usd=25.0),
            ),
            proposals=(HypothesisProposal(HYPOTHESIS, proposer="t"),),
        )
        bundle_id = crash.bundles.record(overrun)
        crash.journal.record(
            attempt_id=crash.attempt.id,
            phase=AttemptPhase.BUNDLE_DURABLE,
            bundle_id=bundle_id,
        )

        report = crash.run()

        assert report.breached
        assert crash.journal.breaches()
        # and the whole overrun is on the ledger, not the authorized part
        assert crash.ledger.balance().usd == 75.0


class TestAStepWhoseEventWasLost:
    def test_a_finished_step_is_adopted_rather_than_repeated(
        self, tmp_path: Path
    ) -> None:
        """A crash the journal cannot pinpoint: the step ran to the end
        and the stage event never landed."""
        crash = Crash(tmp_path)
        resolved = crash.begun.resolve_attempt(
            crash.attempt.resolved(
                ActionOutcome(status=AttemptStatus.SUCCEEDED)
            )
        )
        crash.states.persist(crash.begun)
        crash.states.persist(resolved)

        report = crash.run()

        assert report.anything_to_do
        assert report.adopted == resolved.id
        assert report.state_id == resolved.id
        assert "crash hid" in report.summary()

    def test_a_step_still_mid_attempt_is_not_adopted(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.states.persist(crash.begun)

        report = crash.run()

        assert not report.anything_to_do
        assert report.state_id == crash.head.id


class TestAdoptingTheRecordedSettlement:
    """A crash between the live settlement and the closing journal event
    leaves the debit on the ledger with the state-budget *delta* — which
    floating point can hold one ulp away from the bundle's own cost.
    The movement that happened is authoritative; recovery adopts it."""

    def test_a_settled_charge_is_adopted_not_rederived(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.bundle_durable()
        # What the dying step actually posted: the bundle's cost, one
        # ulp adrift — exactly what (before - (before - cost)) can give.
        drifted = ResourceCost(
            usd=SPENT.usd + 1e-13, model_tokens=SPENT.model_tokens
        )
        crash.ledger.settle(
            drifted,
            charge_id=crash.attempt.id,
            reason=f"attempt {crash.attempt.id}",
        )

        (recovery,) = crash.run().recoveries

        assert recovery.resolution is AttemptPhase.COMPLETED
        assert recovery.basis is SettlementBasis.MEASURED
        assert recovery.settled == drifted  # the ledger's figure, adopted
        # And twice is once.
        assert crash.run().recoveries == ()

    def test_a_released_charge_is_adopted_too(self, tmp_path: Path) -> None:
        """The live step can release (a zero delta) and die before the
        journal closes; recovery must not try to debit over the release."""
        crash = Crash(tmp_path)
        crash.bundle_durable()
        crash.ledger.release(
            charge_id=crash.attempt.id,
            reason=f"attempt {crash.attempt.id}",
        )

        (recovery,) = crash.run().recoveries

        assert recovery.resolution is AttemptPhase.COMPLETED
        assert recovery.settled.is_zero


class Answers:
    """A collector with a script: one job id it will call finished."""

    def __init__(self, job_id: str = "", result: ExperimentResult | None = None):
        self._job_id = job_id
        self._result = result

    def finished(self, job_id: str, /) -> ExperimentResult | None:
        return self._result if job_id == self._job_id else None


def fabricated_result(job_id: str, *, spec_id: str = "exp_unknown") -> ExperimentResult:
    """A terminal result the executor could have recorded — for a spec the
    state never registered, so the deterministic gate must refuse it."""
    return ExperimentResult(
        spec_id=spec_id,
        job_id=job_id,
        status=ExperimentStatus.COMPLETED,
        command=("python", "x.py"),
        environment=Environment(python_version="3", platform="test"),
        metrics={"value": 1.0},
        cost=ResourceCost(wall_clock_seconds=2.0),
    )


class TestCollectingFinishedJobs:
    """The refusal edges of the salvage arm. The happy path — a real job,
    a real gate pass, a completed attempt with the job's measured cost —
    is proven end-to-end by the fault sweep, killed at the exact write."""

    def test_without_a_collector_nothing_changes(self, tmp_path: Path) -> None:
        """``jobs=None`` is the pre-salvage behavior, verbatim."""
        crash = Crash(tmp_path)
        crash.submitted("job_1")

        (recovery,) = crash.run().recoveries

        assert recovery.resolution is AttemptPhase.ABANDONED
        assert recovery.basis is SettlementBasis.CONSERVATIVE_MAX

    def test_a_job_the_collector_cannot_prove_is_abandoned(
        self, tmp_path: Path
    ) -> None:
        crash = Crash(tmp_path)
        crash.submitted("job_1")

        (recovery,) = crash.run(jobs=Answers()).recoveries

        assert recovery.resolution is AttemptPhase.ABANDONED
        assert recovery.basis is SettlementBasis.CONSERVATIVE_MAX

    def test_a_submission_disagreeing_with_its_registration_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The STARTED event pre-registered one job id and SUBMITTED names
        another: two versions of one history, and salvage trusts neither."""
        crash = Crash(tmp_path)
        crash.submitted("job_other", registered="job_1")

        report = crash.run(
            jobs=Answers("job_other", fabricated_result("job_other"))
        )

        (recovery,) = report.recoveries
        assert recovery.resolution is AttemptPhase.ABANDONED
        assert recovery.basis is SettlementBasis.CONSERVATIVE_MAX

    def test_an_unregistered_submission_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A STARTED event with no job id pre-registered nothing; there is
        no intent on record for the submission to be checked against."""
        crash = Crash(tmp_path)
        crash.submitted("job_1", registered="")

        report = crash.run(jobs=Answers("job_1", fabricated_result("job_1")))

        (recovery,) = report.recoveries
        assert recovery.resolution is AttemptPhase.ABANDONED

    def test_a_gate_refused_result_salvages_a_failed_bundle(
        self, tmp_path: Path
    ) -> None:
        """The collected result names a spec the state never registered.
        Salvage commits the same failed bundle the live step would have:
        the attempt completes with the result's measured cost, and nothing
        enters the evidence store."""
        crash = Crash(tmp_path)
        crash.submitted("job_1")
        collected = fabricated_result("job_1")

        report = crash.run(jobs=Answers("job_1", collected))

        (recovery,) = report.recoveries
        assert recovery.resolution is AttemptPhase.COMPLETED
        assert recovery.basis is SettlementBasis.MEASURED
        assert recovery.settled == collected.cost
        assert crash.evidence.results() == ()
        # And twice is once: nothing is open, so a second pass is silent.
        assert crash.run(jobs=Answers("job_1", collected)).recoveries == ()

    def test_a_disagreeing_outputs_record_is_refused(
        self, tmp_path: Path
    ) -> None:
        """OUTPUTS_DURABLE names one result and the executor reports
        another — the records disagree, and salvage takes neither side."""
        crash = Crash(tmp_path)
        crash.submitted("job_1")
        crash.journal.record(
            attempt_id=crash.attempt.id,
            phase=AttemptPhase.OUTPUTS_DURABLE,
            job_id="job_1",
            produced=(("result", "res_someone_else"),),
        )

        report = crash.run(jobs=Answers("job_1", fabricated_result("job_1")))

        (recovery,) = report.recoveries
        assert recovery.resolution is AttemptPhase.ABANDONED
        assert recovery.basis is SettlementBasis.CONSERVATIVE_MAX


def test_recovery_reaches_no_executor(tmp_path: Path) -> None:
    """Recovery cannot resubmit because it cannot submit: it is handed
    records, never an executor. Even the collector arrives as a protocol
    with one question in it — did this job finish? — and the composition
    root supplies the implementation; nothing here can start work."""
    source = Path(recovery_module.__file__ or "")
    imports = [
        line
        for line in source.read_text().splitlines()
        if line.startswith(("from ", "import "))
    ]

    assert not [line for line in imports if "execution" in line]
