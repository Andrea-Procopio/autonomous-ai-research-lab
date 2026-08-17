"""Correctness invariants of the runtime loop: the deterministic validation
gate, honest accounting, engineering-failure signals, and budget discipline."""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.attempt import AttemptStatus
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
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
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import ResearchRuntime, StepReport
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.frontier import build_frontier
from autonomous_research_lab.runtime.metrics import ProviderUsage, StepMetrics

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
                spec_id=spec.id,
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


@dataclass
class ListSink:
    records: list[StepMetrics] = field(default_factory=list)

    def log(self, record: StepMetrics) -> None:
        self.records.append(record)


def _runtime(
    tmp_path: Path,
    engineer: StubEngineer,
    critic: SpyCritic,
    *,
    config: RuntimeConfig | None = None,
    usage: object | None = None,
) -> tuple[ResearchRuntime, InMemoryEvidenceStore, ListSink]:
    store = InMemoryEvidenceStore()
    sink = ListSink()
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
    work unbilled: the remainder is drained, the overrun is recorded, and
    the program halts."""
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

    assert report.halt_reason == "budget exhausted after cost overrun"
    assert report.state.results  # the work really committed...
    assert report.state.budget.wall_clock_seconds == 0.0  # ...and was billed
    assert any("budget overrun" in note for note in report.notes)


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
