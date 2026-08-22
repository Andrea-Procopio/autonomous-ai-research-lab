"""Correctness invariants of the runtime loop: the deterministic validation
gate, honest accounting, engineering-failure signals, and budget discipline."""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.attempt import (
    AttemptPhase,
    AttemptStatus,
    SettlementBasis,
)
from autonomous_research_lab.core.budget import (
    NO_COST,
    ResearchBudget,
    ResourceCost,
    Settlement,
)
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
)
from autonomous_research_lab.core.proposals import Proposal, ResultProposal
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.orchestration import loop as runtime_loop
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import ResearchRuntime, StepReport
from autonomous_research_lab.persistence import FileStateStore
from autonomous_research_lab.persistence.commit_store import CommitBundleStore
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.frontier import build_frontier
from autonomous_research_lab.runtime.metrics import ProviderUsage, StepMetrics
from autonomous_research_lab.runtime.spend import SpendLedger

QUESTION = ResearchQuestion(text="Is the stream fair?")
HYPOTHESIS = Hypothesis(statement="The stream is biased.", question_id=QUESTION.id)

#: Writes the metric value it is given via config — a real process, so the
#: executor's manifest and provenance are genuine.
_SCRIPT = (
    "import json, os, pathlib; "
    "d = pathlib.Path(os.environ['ARL_RUN_DIR']); "
    "cfg = json.loads(pathlib.Path(os.environ['ARL_CONFIG']).read_text()); "
    "(d / 'metrics.json').write_text(json.dumps({'heads_rate': cfg['value']}))"
)
_FAILING_SCRIPT = "raise SystemExit(3)"


def _prepared_state(
    spec: ExperimentSpec,
    prediction: Prediction,
    *,
    budget: ResearchBudget | None = None,
) -> ResearchState:
    """A state whose frontier has exactly one open move: run ``spec``."""
    return (
        ResearchState(
            objective="fairness",
            budget=budget
            or ResearchBudget(
                wall_clock_seconds=3600.0, usd=10.0, model_tokens=200_000
            ),
        )
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(prediction)
        .add_experiment(spec)
    )


def _spec_and_prediction(
    *,
    threshold: float = 0.9,
    declared_metrics: tuple[str, ...] = ("heads_rate",),
    seeds: tuple[int, ...] = (7,),
) -> tuple[ExperimentSpec, Prediction]:
    prediction = Prediction(
        hypothesis_id=HYPOTHESIS.id,
        condition="one draw stream",
        metric="heads_rate",
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=threshold,
    )
    spec = ExperimentSpec(
        prediction_id=prediction.id,
        objective="measure the rate",
        procedure="run the stream and report",
        metrics=declared_metrics,
        seeds=seeds,
    )
    return spec, prediction


@dataclass
class StubEngineer(ResearchRole):
    """Executor seat driving a real LocalExecutor, with configurable faults."""

    executor: LocalExecutor
    value: float = 0.5
    omit_seed: bool = False
    tamper: bool = False
    fail_process: bool = False
    proposal_count: int = 1
    cost_override: ResourceCost | None = None
    raise_error: str | None = None
    misreport: dict[str, str] | None = None
    """Assigned spec id -> the spec id the returned result claims instead."""

    drop_manifest: bool = False
    performed: int = 0

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_ENGINEER

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset(
            {ResearchActionType.RUN_EXPERIMENT, ResearchActionType.REPLICATE}
        )

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        self.performed += 1
        if self.raise_error is not None:
            raise RuntimeError(self.raise_error)
        (spec,) = invocation.context.experiments
        proposals: list[Proposal] = []
        for index in range(self.proposal_count):
            seed = None if self.omit_seed else 100 + index
            script = _FAILING_SCRIPT if self.fail_process else _SCRIPT
            job = ExperimentJob(
                spec_id=(self.misreport or {}).get(spec.id, spec.id),
                command=(sys.executable, "-c", script),
                config={"value": self.value},
                seed=seed,
                timeout_seconds=60.0,
            )
            result = self.executor.collect(self.executor.submit(job))
            if self.tamper:
                run_dir = Path(result.logs[0]).parent
                (run_dir / "metrics.json").write_text(
                    json.dumps({"heads_rate": 0.99})
                )
            if self.drop_manifest:
                run_dir = Path(result.logs[0]).parent
                (run_dir / "manifest.json").unlink()
            if self.cost_override is not None:
                result = dataclasses.replace(result, cost=self.cost_override)
            proposals.append(
                ResultProposal(result=result, proposer="stub:engineer")
            )
        return tuple(proposals)


class SpyCritic(ResearchRole):
    """A critic that records whether it was ever consulted."""

    def __init__(self) -> None:
        self.performed = 0

    @property
    def name(self) -> RoleName:
        return RoleName.RESULT_ANALYST

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset({ResearchActionType.ANALYZE})

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        self.performed += 1
        return ()


class RogueCritic(SpyCritic):
    """A critic that tries to smuggle a forged result through its review —
    a ResultProposal is outside ANALYZE's output contract."""

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        self.performed += 1
        (result, *_) = invocation.context.results
        forged = dataclasses.replace(result, job_id="job_forged", id="")
        return (ResultProposal(result=forged, proposer="rogue:critic"),)


@dataclass
class ListSink:
    records: list[StepMetrics] = field(default_factory=list)

    def log(self, record: StepMetrics) -> None:
        self.records.append(record)


def _runtime(
    tmp_path: Path,
    engineer: StubEngineer,
    critic: ResearchRole,
    *,
    config: RuntimeConfig | None = None,
    usage: object | None = None,
    ledger: SpendLedger | None = None,
    journal: object | None = None,
) -> tuple[ResearchRuntime, InMemoryEvidenceStore, ListSink]:
    store = InMemoryEvidenceStore()
    sink = ListSink()
    recoverable = journal is not None
    runtime = ResearchRuntime(
        config=config or RuntimeConfig(),
        director=RuleBasedFrontierDirector(),
        roles={
            RoleName.RESEARCH_ENGINEER: engineer,
            RoleName.RESULT_ANALYST: critic,
        },
        store=store,
        metrics=sink,
        usage=usage,  # type: ignore[arg-type]
        ledger=ledger,
        journal=journal,  # type: ignore[arg-type]
        bundles=CommitBundleStore(tmp_path / "program") if recoverable else None,
        states=FileStateStore(tmp_path / "states") if recoverable else None,
    )
    return runtime, store, sink


def _assert_nothing_scientific_committed(
    report: StepReport, store: InMemoryEvidenceStore, critic: SpyCritic
) -> None:
    state = report.state
    assert state.results == ()
    assert state.prediction_tests == ()
    assert state.evidence_ids == ()
    assert store.results() == ()
    # The action is an engineering failure on the record...
    (attempt,) = state.attempts
    assert attempt.status is AttemptStatus.FAILED
    assert any("engineering failure" in note for note in report.notes)
    # ...and no critic was asked to opine on arithmetic.
    assert critic.performed == 0
    assert not report.critic_invoked
    assert report.critic_reasons == ()


def test_missing_declared_metric_never_enters_scientific_state(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction(
        declared_metrics=("heads_rate", "power_draw")
    )
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"))
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    _assert_nothing_scientific_committed(report, store, critic)
    (validation,) = report.validation
    assert any(
        c.name == "declared_metrics_present" for c in validation.failures
    )
    # The run's outputs survive for diagnosis even though nothing committed.
    run_dirs = list((tmp_path / "runs").iterdir())
    assert run_dirs and any(
        (d / "metrics.json").exists() for d in run_dirs
    )


def test_missing_seed_never_enters_scientific_state(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), omit_seed=True)
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    _assert_nothing_scientific_committed(report, store, critic)
    (validation,) = report.validation
    assert any(c.name == "seed_recorded" for c in validation.failures)


def test_artifact_tampering_is_rejected_before_commit(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), tamper=True)
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    _assert_nothing_scientific_committed(report, store, critic)
    (validation,) = report.validation
    assert any(c.name == "artifact_integrity" for c in validation.failures)


def test_valid_negative_results_commit_normally(tmp_path: Path) -> None:
    """A completed run that refutes the prediction is science, not trouble:
    it commits, is transcribed, and wakes neither critic nor debugger."""
    spec, prediction = _spec_and_prediction(threshold=0.55)
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), value=0.5)
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    state = report.state
    (ref,) = state.results
    (test,) = state.prediction_tests
    assert test.consistency is Consistency.INCONSISTENT
    assert len(state.evidence_ids) == 1
    assert store.get_result(ref.result_id).succeeded
    assert critic.performed == 0
    assert report.notes == ()  # no engineering note, no debugging noise
    (validation,) = report.validation
    assert validation.passed


def test_a_step_persists_every_state_it_derives(tmp_path: Path) -> None:
    """A snapshot store that only ever sees the head of a step holds
    states whose parents nobody can find. One step derives several — the
    attempt begun, each proposal committed, the attempt resolved — and
    all of them are written, oldest first, so the head is never on disk
    before its ancestry is.
    """
    spec, prediction = _spec_and_prediction(threshold=0.55)
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), value=0.5)
    runtime, _, _ = _runtime(tmp_path, engineer, SpyCritic())
    states = FileStateStore(tmp_path / "states")
    runtime.states = states
    start = _prepared_state(spec, prediction)
    states.persist(start)

    report = runtime.step(start)

    stored = {found: states.load(found) for found in states.state_ids()}
    assert len(stored) > 2, "a step derives more than its own head"
    assert report.state.id in stored
    orphans = [
        state.id
        for state in stored.values()
        # The starting state excepted: this fixture builds it in memory
        # and persists only it, which is the very habit under test —
        # what must hold is that the step adds no new orphan.
        if state.id != start.id
        and state.parent_id is not None
        and state.parent_id not in stored
    ]
    assert orphans == []
    walked = report.state
    while walked.id != start.id:
        assert walked.parent_id is not None
        walked = stored[walked.parent_id]  # every hop is on disk


def test_multiple_result_proposals_are_rejected_transactionally(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), proposal_count=2)
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    _assert_nothing_scientific_committed(report, store, critic)


def test_zero_result_proposals_are_an_output_contract_failure(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), proposal_count=0)
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    _assert_nothing_scientific_committed(report, store, critic)


def test_repeated_execution_failures_raise_the_deterministic_signal(
    tmp_path: Path,
) -> None:
    """Failed executions commit as execution records; the second failure of
    one experiment produces the engineering note — from the results
    themselves, not from attempt bookkeeping — and never a critic."""
    spec, prediction = _spec_and_prediction(seeds=(0, 1))
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), fail_process=True)
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)

    first = runtime.step(_prepared_state(spec, prediction))
    assert len(first.state.results) == 1
    assert not any("failed execution" in n for n in first.notes)

    second = runtime.step(first.state)
    assert len(second.state.results) == 2
    assert all(
        not store.get_result(ref.result_id).succeeded
        for ref in second.state.results
    )
    assert any(
        "2 failed execution(s)" in note for note in second.notes
    )
    # An engineering signal, not scientific critique — and no evidence.
    assert critic.performed == 0
    assert second.state.evidence_ids == ()
    # The note reaches the director through the next frontier.
    third_frontier_notes = build_frontier(
        second.state, open_decisions=second.notes
    ).open_decisions
    assert any("failed execution" in n for n in third_frontier_notes)


def test_role_exceptions_land_on_the_failure_path(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"), raise_error="GPU on fire"
    )
    critic = SpyCritic()
    runtime, _, sink = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    (attempt,) = report.state.attempts
    assert attempt.status is AttemptStatus.FAILED
    assert attempt.outcome is not None
    assert "GPU on fire" in str(attempt.outcome.error)
    assert any("GPU on fire" in note for note in report.notes)
    assert sink.records[-1].failures == 1
    # The failed attempt is on the next frontier, worth revisiting.
    assert build_frontier(report.state).failed_attempts


def test_an_unaffordable_action_never_invokes_its_role(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"))
    critic = SpyCritic()
    runtime, _, _ = _runtime(tmp_path, engineer, critic)
    poor = _prepared_state(
        spec,
        prediction,
        budget=ResearchBudget(wall_clock_seconds=1.0, model_tokens=100),
    )

    report = runtime.step(poor)

    assert report.halt_reason is not None
    assert "insufficient budget" in report.halt_reason
    assert engineer.performed == 0
    assert report.state.attempts == ()
    assert report.state.budget == poor.budget  # nothing spent, nothing done


def test_budget_overrun_bills_the_work_and_halts(tmp_path: Path) -> None:
    """An actual cost beyond the remaining budget cannot leave committed
    work unbilled, and cannot be billed at less than it cost: the whole
    figure is charged, the balance goes negative, and the program
    halts."""
    spec, prediction = _spec_and_prediction(threshold=0.45)
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        value=0.5,
        cost_override=ResourceCost(wall_clock_seconds=10_000.0),
    )
    critic = SpyCritic()
    runtime, _, _ = _runtime(tmp_path, engineer, critic)
    state = _prepared_state(
        spec,
        prediction,
        budget=ResearchBudget(wall_clock_seconds=400.0, model_tokens=100),
    )

    report = runtime.step(state)

    assert report.halt_reason == "budget breached: the run spent past its grant"
    assert report.state.results  # the work really committed...
    # ...and was billed in full: 400 held, 10,000 spent.
    assert report.state.budget.wall_clock_seconds == -9_600.0
    assert any("budget breach" in note for note in report.notes)


class FakeUsageSource:
    """Stands in for a provider adapter reporting real usage."""

    def __init__(self) -> None:
        self._pending = ProviderUsage(
            calls=2, input_tokens=1_200, output_tokens=340, model="fake-model-1"
        )

    def drain(self) -> ProviderUsage:
        usage, self._pending = self._pending, ProviderUsage()
        return usage


def test_a_provider_adapter_can_report_actual_usage(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction(threshold=0.45)
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), value=0.5)
    critic = SpyCritic()
    runtime, _, sink = _runtime(
        tmp_path, engineer, critic, usage=FakeUsageSource()
    )

    report = runtime.step(_prepared_state(spec, prediction))

    assert report.provider_usage.calls == 2
    assert report.provider_usage.model == "fake-model-1"
    record = sink.records[-1]
    assert record.provider_calls == 2
    assert record.input_tokens == 1_200
    assert record.output_tokens == 340
    assert record.model == "fake-model-1"
    # Conceptual invocations are still tracked separately and honestly.
    assert record.reasoning_invocations == 2


# -- operational record vs scientific state -----------------------------------


def test_validation_rejected_work_records_actual_cost_and_runtime(
    tmp_path: Path,
) -> None:
    """A run that executed but failed the gate keeps its operational record:
    the attempt carries the actual cost, the budget is billed the actual
    cost, and the runtime is metered — while nothing scientific commits."""
    spec, prediction = _spec_and_prediction()
    actual = ResourceCost(wall_clock_seconds=120.0)
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"), tamper=True, cost_override=actual
    )
    critic = SpyCritic()
    runtime, store, sink = _runtime(tmp_path, engineer, critic)
    before = _prepared_state(spec, prediction)

    report = runtime.step(before)

    _assert_nothing_scientific_committed(report, store, critic)
    (attempt,) = report.state.attempts
    assert attempt.outcome is not None
    assert attempt.outcome.actual_cost == actual  # actual, never the estimate
    spent = (
        before.budget.wall_clock_seconds
        - report.state.budget.wall_clock_seconds
    )
    assert spent == 120.0  # the rejected work is billed in full
    # The work that really ran is on the metrics record, commit or not.
    assert sink.records[-1].experiment_seconds > 0.0


def test_budget_overrun_from_rejected_work_is_billed_and_halts(
    tmp_path: Path,
) -> None:
    """Gate-rejected work whose actual cost exceeds the remaining budget
    is charged in full and halts the run — exactly like committed work.
    Rejection is not a discount."""
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        tamper=True,
        cost_override=ResourceCost(wall_clock_seconds=10_000.0),
    )
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)
    state = _prepared_state(
        spec,
        prediction,
        budget=ResearchBudget(wall_clock_seconds=400.0, model_tokens=100),
    )

    report = runtime.step(state)

    assert report.halt_reason == "budget breached: the run spent past its grant"
    assert report.state.budget.wall_clock_seconds == -9_600.0  # billed whole
    assert any("budget breach" in note for note in report.notes)
    assert report.state.results == ()  # and still nothing scientific
    assert store.results() == ()


# -- the critic's output contract ---------------------------------------------


def test_unauthorized_critic_output_cannot_commit(tmp_path: Path) -> None:
    """The critic is under the same mechanical output contract as every
    other seat: an unauthorized proposal rejects its entire bundle, the
    critic attempt fails, and no forged object reaches state or store."""
    spec, prediction = _spec_and_prediction(threshold=0.5)
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), value=0.9)
    critic = RogueCritic()
    # This scenario tests the output contract, not verification: run the
    # explicitly ablated lab so the scientific trigger still fires for an
    # unverified result.
    runtime, store, _ = _runtime(
        tmp_path,
        engineer,
        critic,
        config=RuntimeConfig(verification_governance_enabled=False),
    )

    report = runtime.step(_prepared_state(spec, prediction))

    assert critic.performed == 1  # the trigger fired and the critic ran
    assert report.critic_invoked
    analyze = next(
        a
        for a in report.state.attempts
        if a.action.action_type is ResearchActionType.ANALYZE
    )
    assert analyze.status is AttemptStatus.FAILED
    assert analyze.outcome is not None
    assert "output contract" in str(analyze.outcome.error)
    # Only the engineer's genuine result exists anywhere authoritative.
    (ref,) = report.state.results
    (recorded,) = store.results()
    assert recorded.id == ref.result_id
    assert recorded.job_id != "job_forged"


# -- invocation counting and exception attribution ----------------------------


def test_role_invocation_is_counted_once_on_success(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction(threshold=0.45)
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), value=0.5)
    critic = SpyCritic()
    runtime, _, sink = _runtime(tmp_path, engineer, critic)

    runtime.step(_prepared_state(spec, prediction))

    assert engineer.performed == 1
    assert sink.records[-1].reasoning_invocations == 2  # director + role


def test_role_invocation_is_counted_once_on_role_exception(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"), raise_error="GPU on fire"
    )
    critic = SpyCritic()
    runtime, _, sink = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    assert engineer.performed == 1
    assert sink.records[-1].reasoning_invocations == 2  # not double-counted
    assert any("raised during" in note for note in report.notes)


def test_unexpected_validation_exception_is_not_blamed_on_the_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug in the runtime's own post-role checks is a crash, not a role
    outcome: it propagates rather than masquerading as the role raising,
    and the single role invocation is never double-counted."""
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"))
    critic = SpyCritic()
    runtime, store, sink = _runtime(tmp_path, engineer, critic)

    def broken_gate(*args: object, **kwargs: object) -> object:
        raise RuntimeError("validator bug")

    monkeypatch.setattr(runtime_loop, "gate_results", broken_gate)

    with pytest.raises(RuntimeError, match="validator bug"):
        runtime.step(_prepared_state(spec, prediction))

    assert engineer.performed == 1  # the role ran once and was not blamed
    assert sink.records == []  # no record mislabels this as a role failure
    assert store.results() == ()


# -- identity of failed executions --------------------------------------------


def test_correctly_assigned_failed_execution_commits_as_failure_record(
    tmp_path: Path,
) -> None:
    """A failed run of the assigned experiment is an honest execution
    record: it commits with inconclusive standing and produces no claim."""
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), fail_process=True)
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    (attempt,) = report.state.attempts
    assert attempt.status is AttemptStatus.SUCCEEDED  # the record committed
    (ref,) = report.state.results
    assert not store.get_result(ref.result_id).succeeded
    (test,) = report.state.prediction_tests
    assert test.consistency is Consistency.INCONCLUSIVE
    assert report.state.evidence_ids == ()  # a record, not a claim
    (validation,) = report.validation
    assert validation.passed  # identity and provenance were still checked


def test_wrongly_assigned_failed_result_is_rejected_transactionally(
    tmp_path: Path,
) -> None:
    """An executor assigned experiment A cannot commit a failed result for
    experiment B: assignment identity is validated regardless of status."""
    spec, prediction = _spec_and_prediction()
    other_prediction = Prediction(
        hypothesis_id=HYPOTHESIS.id,
        condition="a different draw stream",
        metric="heads_rate",
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=0.9,
    )
    other_spec = ExperimentSpec(
        prediction_id=other_prediction.id,
        objective="measure the other rate",
        procedure="run the other stream and report",
        metrics=("heads_rate",),
        seeds=(7,),
    )
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        fail_process=True,
        misreport={spec.id: other_spec.id, other_spec.id: spec.id},
    )
    critic = SpyCritic()
    runtime, store, _ = _runtime(tmp_path, engineer, critic)
    state = (
        _prepared_state(spec, prediction)
        .upsert_prediction(other_prediction)
        .add_experiment(other_spec)
    )

    report = runtime.step(state)

    _assert_nothing_scientific_committed(report, store, critic)
    (validation,) = report.validation
    assert any(
        c.name == "result_matches_assignment" for c in validation.failures
    )


def test_failed_result_with_broken_provenance_is_rejected_but_costed(
    tmp_path: Path,
) -> None:
    """Losing the manifest breaks the executor contract even for a failed
    run: the result is barred from authoritative state, while its actual
    cost and runtime stay on the operational books."""
    spec, prediction = _spec_and_prediction()
    actual = ResourceCost(wall_clock_seconds=90.0)
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        fail_process=True,
        drop_manifest=True,
        cost_override=actual,
    )
    critic = SpyCritic()
    runtime, store, sink = _runtime(tmp_path, engineer, critic)

    report = runtime.step(_prepared_state(spec, prediction))

    _assert_nothing_scientific_committed(report, store, critic)
    (validation,) = report.validation
    assert any(c.name == "artifact_integrity" for c in validation.failures)
    (attempt,) = report.state.attempts
    assert attempt.outcome is not None
    assert attempt.outcome.actual_cost == actual
    assert sink.records[-1].experiment_seconds > 0.0


# -- the durable spend ledger ------------------------------------------------
# The loop charges the budget it carries on the state. When a durable
# ledger is wired, the same movement is posted there, keyed by the attempt
# that incurred it, and the two records must agree afterwards.


class RecordingLedger:
    """A ``SpendLedger`` that keeps its postings in memory. The durable
    implementation lives in ``program``, which the runtime deliberately
    cannot import; this pins the contract the loop relies on — including
    that every debit answers a hold taken before the work."""

    def __init__(self, balance: ResearchBudget) -> None:
        self.balance = balance
        self.postings: list[tuple[str, ResourceCost]] = []
        self.held: dict[str, ResourceCost] = {}
        self.released: list[str] = []

    def reserve(
        self, cost: ResourceCost, *, charge_id: str, reason: str
    ) -> object:
        del reason
        self.held.setdefault(charge_id, cost)
        return None

    def holds(self, charge_id: str, /) -> bool:
        return charge_id in self.held

    def settle(
        self, cost: ResourceCost, *, charge_id: str, reason: str
    ) -> Settlement:
        del reason
        reserved = self.held.get(charge_id)
        assert reserved is not None, "every debit answers a hold"
        for posted_id, posted_cost in self.postings:
            if posted_id == charge_id:
                assert posted_cost == cost, "one charge id, one movement"
                break
        else:
            if cost.is_zero:
                self.released.append(charge_id)
            else:
                self.postings.append((charge_id, cost))
                self.balance = self.balance.spend(cost, allow_overdraw=True)
        return Settlement(
            charge_id=charge_id, reserved=reserved, actual=cost, entry_id="bent_x"
        )

    def release(self, *, charge_id: str, reason: str) -> object:
        del reason
        self.released.append(charge_id)
        return None

    def require_balance(self, expected: ResearchBudget) -> None:
        if self.balance != expected:
            raise AssertionError(
                f"ledger holds {self.balance}, state holds {expected}"
            )


def test_a_step_posts_one_debit_for_what_it_charged(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"))
    state = _prepared_state(spec, prediction)
    ledger = RecordingLedger(state.budget)
    runtime, _, _ = _runtime(tmp_path, engineer, SpyCritic(), ledger=ledger)

    report = runtime.step(state)

    (attempt,) = report.state.attempts
    assert [charge_id for charge_id, _ in ledger.postings] == [attempt.id]
    charged = ledger.postings[0][1]
    assert charged.wall_clock_seconds == (
        state.budget.wall_clock_seconds - report.state.budget.wall_clock_seconds
    )
    assert ledger.balance == report.state.budget


def test_an_overrun_posts_the_whole_figure_to_the_ledger(
    tmp_path: Path,
) -> None:
    """The ledger records what was spent, not what could be afforded.

    Charging the affordable share would keep the two records agreeing by
    making both of them wrong, and the money above the budget would
    appear nowhere at all — which is exactly the failure a budget exists
    to make visible."""
    spec, prediction = _spec_and_prediction(threshold=0.45)
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        value=0.5,
        cost_override=ResourceCost(wall_clock_seconds=10_000.0),
    )
    state = _prepared_state(
        spec,
        prediction,
        budget=ResearchBudget(wall_clock_seconds=400.0, model_tokens=100),
    )
    ledger = RecordingLedger(state.budget)
    runtime, _, _ = _runtime(tmp_path, engineer, SpyCritic(), ledger=ledger)

    report = runtime.step(state)

    assert report.halt_reason == "budget breached: the run spent past its grant"
    (_, charged) = ledger.postings[0]
    assert charged.wall_clock_seconds == 10_000.0  # the whole overrun
    assert ledger.balance == report.state.budget
    assert report.state.budget.wall_clock_seconds == -9_600.0
    # and the breach is visible as two numbers that disagree
    assert charged.exceeds(ledger.held[report.state.attempts[0].id])


def test_a_ledger_that_disagrees_with_the_state_stops_the_step(
    tmp_path: Path,
) -> None:
    """A bookkeeping divergence is not a research outcome: it raises out
    of the step rather than becoming a halt reason the director reads."""
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"))
    state = _prepared_state(spec, prediction)
    # Deep enough to afford the charge, so what fails is the comparison
    # afterwards and not the debit itself.
    ledger = RecordingLedger(
        ResearchBudget(wall_clock_seconds=9_999.0, usd=99.0, model_tokens=999_999)
    )
    runtime, _, _ = _runtime(tmp_path, engineer, SpyCritic(), ledger=ledger)

    with pytest.raises(AssertionError, match="ledger holds"):
        runtime.step(state)


def test_without_a_ledger_the_loop_bills_the_state_as_before(
    tmp_path: Path,
) -> None:
    """``ledger=None`` is the ablation, and the default: the step charges
    the state exactly as it always has. Every other test in this file
    runs unledgered, so they are the rest of this assertion."""
    spec, prediction = _spec_and_prediction()
    state = _prepared_state(spec, prediction)
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"))
    runtime, _, _ = _runtime(tmp_path, engineer, SpyCritic())

    report = runtime.step(state)

    assert runtime.ledger is None
    assert report.state.results
    assert (
        report.state.budget.wall_clock_seconds
        < state.budget.wall_clock_seconds
    )


class RecordingJournal:
    """An ``AttemptJournal`` that keeps its phases in memory."""

    def __init__(self) -> None:
        self.phases: list[tuple[str, AttemptPhase]] = []
        self.costs: dict[str, tuple[ResourceCost, ResourceCost]] = {}
        self.bases: dict[str, SettlementBasis] = {}

    def record(
        self,
        *,
        attempt_id: str,
        phase: AttemptPhase,
        state_id: str = "",
        job_id: str = "",
        bundle_id: str = "",
        produced: object = (),
        reserved: ResourceCost = NO_COST,
        settled: ResourceCost = NO_COST,
        basis: SettlementBasis = SettlementBasis.NONE,
        detail: str = "",
    ) -> object:
        del state_id, job_id, bundle_id, produced, detail
        self.phases.append((attempt_id, phase))
        if phase is AttemptPhase.COMPLETED:
            self.costs[attempt_id] = (reserved, settled)
            self.bases[attempt_id] = basis
        return None

    def breaches(self) -> list[str]:
        return [
            attempt_id
            for attempt_id, (reserved, settled) in self.costs.items()
            if self.bases[attempt_id] is SettlementBasis.MEASURED
            and settled.exceeds(reserved)
        ]


def test_a_breach_is_two_numbers_on_the_journal_and_a_halt(
    tmp_path: Path,
) -> None:
    """Never hide expenditure: the run stops, and the record says by how
    much it went over rather than that it stopped."""
    spec, prediction = _spec_and_prediction(threshold=0.45)
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        value=0.5,
        cost_override=ResourceCost(wall_clock_seconds=10_000.0),
    )
    journal = RecordingJournal()
    state = _prepared_state(
        spec,
        prediction,
        budget=ResearchBudget(wall_clock_seconds=400.0, model_tokens=100),
    )
    runtime, _, _ = _runtime(
        tmp_path, engineer, SpyCritic(), journal=journal
    )

    report = runtime.step(state)

    (attempt,) = report.state.attempts
    assert report.halt_reason == "budget breached: the run spent past its grant"
    assert journal.breaches() == [attempt.id]
    reserved, settled = journal.costs[attempt.id]
    assert settled.wall_clock_seconds == 10_000.0
    assert settled.exceeds(reserved)
    assert journal.bases[attempt.id] is SettlementBasis.MEASURED


def test_an_attempt_within_its_authorization_is_no_breach(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"))
    journal = RecordingJournal()
    runtime, _, _ = _runtime(
        tmp_path, engineer, SpyCritic(), journal=journal
    )

    runtime.step(_prepared_state(spec, prediction))

    assert journal.breaches() == []
    assert [phase for _, phase in journal.phases] == [
        AttemptPhase.STARTED,
        AttemptPhase.BUNDLE_DURABLE,
        AttemptPhase.COMMITTED,
        AttemptPhase.COMPLETED,
    ]
