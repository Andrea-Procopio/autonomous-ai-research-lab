"""Runtime integration of the scientific-debugging system.

The regression cases from the Phase 1 milestone:

A. crash          -> engineering failure -> bounded debug -> no false negative
C. silent bug     -> positive control fails -> implementation uncertain
D. true negative  -> full verification passes -> verified negative, no debug
E. bad methodology-> rejected before execution -> redesign, no debug
F. bad analysis   -> analytical failure, execution not blamed
G. persistent bug -> debug stops at the configured bound
H. cost accounting-> every failed/retried execution bills its actual cost

(Case B — malformed metrics as a deterministic engineering failure — lives
in ``test_failure_classifier.py`` against the real executor.)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.attempt import AttemptStatus
from autonomous_research_lab.core.budget import NO_COST, ResearchBudget
from autonomous_research_lab.core.experiment import (
    ExperimentResult,
    ExperimentSpec,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
)
from autonomous_research_lab.core.proposals import (
    AssessmentProposal,
    Proposal,
    ResultProposal,
)
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.failure_classifier import FailureDiagnosis
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.execution.runner import DirectJobRunner
from autonomous_research_lab.orchestration.debug_loop import (
    ExperimentDebugger,
    RepairProposal,
)
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.metrics import StepMetrics
from autonomous_research_lab.runtime.playbook import EmpiricalMLPlaybook
from autonomous_research_lab.runtime.preflight import require_preflight
from autonomous_research_lab.runtime.verification import (
    CheckState,
    ExperimentValidityStatus,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)

QUESTION = ResearchQuestion(text="Is the stream fair?")
HYPOTHESIS = Hypothesis(statement="The stream is biased.", question_id=QUESTION.id)

#: Echoes its config back as metrics — every value the tests need appears in
#: metrics.json via a real process, so provenance is genuine.
_ECHO = (
    "import json, os, pathlib; "
    "d = pathlib.Path(os.environ['ARL_RUN_DIR']); "
    "cfg = json.loads(pathlib.Path(os.environ['ARL_CONFIG']).read_text()); "
    "(d / 'metrics.json').write_text(json.dumps(cfg))"
)
_CRASH = "raise SystemExit(3)"

OVERFIT_CONTROL = PositiveControl(
    name="tiny_overfit",
    metric="overfit_acc",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.99,
    rationale="a faithful implementation must overfit the tiny probe set",
)


def _spec_and_prediction(
    *, threshold: float = 0.55, seeds: tuple[int, ...] = (7,)
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
        metrics=("heads_rate",),
        seeds=seeds,
    )
    return spec, prediction


def _prepared_state(
    spec: ExperimentSpec, prediction: Prediction
) -> ResearchState:
    return (
        ResearchState(
            objective="fairness",
            budget=ResearchBudget(
                wall_clock_seconds=3600.0, usd=10.0, model_tokens=200_000
            ),
        )
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(prediction)
        .add_experiment(spec)
    )


@dataclass
class StubEngineer(ResearchRole):
    """Executor seat driving a real LocalExecutor; per-invocation metrics."""

    executor: LocalExecutor
    runs: tuple[dict[str, float] | None, ...] = ({"heads_rate": 0.9},)
    """One entry per invocation: a metrics dict to echo, or ``None`` to crash."""

    preflight_command: tuple[str, ...] | None = None
    """When set, preflight this command before running — the seam that lets
    a test exercise the runtime's PreflightError handling."""

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
        (spec,) = invocation.context.experiments
        run = self.runs[min(self.performed, len(self.runs) - 1)]
        self.performed += 1
        script = _CRASH if run is None else _ECHO
        job = ExperimentJob(
            spec_id=spec.id,
            command=(sys.executable, "-c", script),
            config=dict(run or {}),
            seed=100 + self.performed,
            timeout_seconds=30.0,
        )
        if self.preflight_command is not None:
            probe = ExperimentJob(
                spec_id=spec.id, command=self.preflight_command, seed=job.seed
            )
            require_preflight(probe, spec)
        result = self.executor.collect(self.executor.submit(job))
        return (ResultProposal(result=result, proposer="stub:engineer"),)


class SpyCritic(ResearchRole):
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


class CherryPickingCritic(SpyCritic):
    """Cites only the evidence that favors the hypothesis — deliberately
    invalid analysis over a valid result family."""

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        self.performed += 1
        (hypothesis,) = invocation.context.hypotheses
        consistent_results = {
            t.result_id
            for t in invocation.context.prediction_tests
            if t.consistency is Consistency.CONSISTENT
        }
        favorable = tuple(
            e.id
            for e in invocation.context.evidence
            if e.result_id in consistent_results
        )
        return (
            AssessmentProposal(
                assessment=EpistemicAssessment(
                    subject_id=hypothesis.id,
                    verdict=AssessmentVerdict.SUPPORTED,
                    method="stub:cherry-picker:v1",
                    evidence_ids=favorable,
                    rationale="only the good runs, naturally",
                ),
                proposer="stub:cherry-picker",
            ),
        )


@dataclass
class StubMethodologyReviewer:
    state: CheckState
    detail: str = "review verdict"
    calls: int = 0

    def review(
        self,
        spec: ExperimentSpec,
        prediction: Prediction | None,
        *,
        objective: str,
    ) -> VerificationCheck:
        self.calls += 1
        return VerificationCheck(
            dimension=ValidityDimension.METHODOLOGY,
            name="methodological_validity",
            state=self.state,
            detail=self.detail,
        )


@dataclass
class StubImplementationVerifier:
    state: CheckState
    calls: int = 0

    def verify(
        self,
        spec: ExperimentSpec,
        result: ExperimentResult,
        prediction: Prediction | None,
        checks: tuple[VerificationCheck, ...],
    ) -> VerificationCheck:
        self.calls += 1
        return VerificationCheck(
            dimension=ValidityDimension.IMPLEMENTATION,
            name="implementation_faithfulness",
            state=self.state,
            detail="verifier verdict",
        )


@dataclass
class FixedRepair:
    """Rule-based repair strategy: always propose the same replacement run."""

    metrics: dict[str, float] | None
    proposals: int = 0

    def propose(
        self,
        spec: ExperimentSpec,
        failed: ExperimentResult,
        diagnosis: FailureDiagnosis,
        attempt_number: int,
    ) -> RepairProposal | None:
        self.proposals += 1
        script = _CRASH if self.metrics is None else _ECHO
        return RepairProposal(
            job=ExperimentJob(
                spec_id=spec.id,
                command=(sys.executable, "-c", script),
                config=dict(self.metrics or {}),
                seed=200 + self.proposals,
                timeout_seconds=30.0,
            ),
            rationale=f"replace the failing script (attempt {attempt_number})",
        )


@dataclass
class ListSink:
    records: list[StepMetrics] = field(default_factory=list)

    def log(self, record: StepMetrics) -> None:
        self.records.append(record)


def _runtime(
    tmp_path: Path,
    engineer: StubEngineer,
    *,
    critic: SpyCritic | None = None,
    config: RuntimeConfig | None = None,
    repair: FixedRepair | None = None,
    methodology: StubMethodologyReviewer | None = None,
    verifier: StubImplementationVerifier | None = None,
    controls: tuple[PositiveControl, ...] = (),
    playbook: EmpiricalMLPlaybook | None = None,
) -> tuple[ResearchRuntime, InMemoryEvidenceStore, ListSink, SpyCritic]:
    store = InMemoryEvidenceStore()
    sink = ListSink()
    critic = critic or SpyCritic()
    runtime = ResearchRuntime(
        config=config or RuntimeConfig(),
        director=RuleBasedFrontierDirector(),
        roles={
            RoleName.RESEARCH_ENGINEER: engineer,
            RoleName.RESULT_ANALYST: critic,
        },
        store=store,
        metrics=sink,
        playbook=playbook,
        debugger=(
            ExperimentDebugger(
                runner=DirectJobRunner(engineer.executor), strategy=repair
            )
            if repair is not None
            else None
        ),
        methodology_reviewer=methodology,
        implementation_verifier=verifier,
        control_source=(lambda spec: controls) if controls else None,
    )
    return runtime, store, sink, critic


# -- A: crash -> diagnose -> bounded debug -> valid execution -----------------


def test_crash_is_diagnosed_debugged_and_repaired(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), runs=(None,))
    repair = FixedRepair(metrics={"heads_rate": 0.9})
    runtime, store, sink, critic = _runtime(tmp_path, engineer, repair=repair)

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state

    # Diagnosis and repair are on the record.
    assert any("engineering failure diagnosed" in n for n in report.notes)
    assert any("debugging succeeded on attempt 1" in n for n in report.notes)
    assert report.debug_attempts == 1
    assert report.debug_resolved
    record = sink.records[-1]
    assert record.failure_category == "nonzero_exit"
    assert record.debug_attempts == 1
    assert record.debug_resolved

    # Both the failure and the repaired run are separate committed records.
    assert len(state.results) == 2
    statuses = {
        store.get_result(ref.result_id).succeeded for ref in state.results
    }
    assert statuses == {True, False}
    # The crash yielded an inconclusive test, never a scientific negative.
    consistencies = {t.consistency for t in state.prediction_tests}
    assert consistencies == {Consistency.INCONCLUSIVE, Consistency.CONSISTENT}
    # The retry is a distinct auditable DEBUG attempt.
    debug_attempts = [
        a
        for a in state.attempts
        if a.action.action_type is ResearchActionType.DEBUG
    ]
    assert len(debug_attempts) == 1
    assert debug_attempts[0].status is AttemptStatus.SUCCEEDED
    assert "repair attempt 1" in debug_attempts[0].action.rationale
    # No critic was consulted about an engineering matter.
    assert critic.performed == 0


def test_debugging_is_off_without_the_flag(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), runs=(None,))
    repair = FixedRepair(metrics={"heads_rate": 0.9})
    runtime, _, sink, _ = _runtime(
        tmp_path,
        engineer,
        repair=repair,
        config=RuntimeConfig(debug_enabled=False),
    )

    report = runtime.step(_prepared_state(spec, prediction))

    assert repair.proposals == 0
    assert report.debug_attempts == 0
    # The failure is still diagnosed and noted for the director.
    assert any("engineering failure diagnosed" in n for n in report.notes)
    assert sink.records[-1].failure_category == "nonzero_exit"


# -- C: silent bug -> implementation uncertain, not a negative ----------------


def test_silent_bug_fails_the_control_and_defers_the_negative(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction(threshold=0.5)
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        runs=({"heads_rate": 0.45, "overfit_acc": 0.5},),  # plausible, wrong
    )
    repair = FixedRepair(metrics={"heads_rate": 0.9})
    runtime, _, sink, critic = _runtime(
        tmp_path, engineer, repair=repair, controls=(OVERFIT_CONTROL,)
    )

    report = runtime.step(_prepared_state(spec, prediction))

    record = sink.records[-1]
    assert record.control_failures == 1
    assert (
        record.verification_status
        == ExperimentValidityStatus.IMPLEMENTATION_UNCERTAIN
    )
    assert record.negative_result_verdict == "deferred"
    assert any("implementation uncertain" in n for n in report.notes)
    assert any("not a scientific negative" in n for n in report.notes)
    # The observation is preserved (evidence committed), just not promoted.
    assert len(report.state.evidence_ids) == 1
    # A completed run never enters the debug loop, silent bug or not.
    assert repair.proposals == 0
    assert report.debug_attempts == 0
    assert critic.performed == 0


# -- D: correct experiment, true negative -> verified, no debugging -----------


def test_true_negative_with_full_verification_is_accepted(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction(threshold=0.5)
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        runs=({"heads_rate": 0.45, "overfit_acc": 1.0},),
    )
    repair = FixedRepair(metrics={"heads_rate": 0.9})
    methodology = StubMethodologyReviewer(CheckState.PASS)
    verifier = StubImplementationVerifier(CheckState.PASS)
    runtime, store, sink, critic = _runtime(
        tmp_path,
        engineer,
        repair=repair,
        methodology=methodology,
        verifier=verifier,
        controls=(OVERFIT_CONTROL,),
    )

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state

    record = sink.records[-1]
    assert record.verification_status == ExperimentValidityStatus.VERIFIED
    assert record.negative_result_verdict == "accepted"
    assert any("verified scientific negative" in n for n in report.notes)
    # The negative evidence is preserved.
    (test,) = state.prediction_tests
    assert test.consistency is Consistency.INCONSISTENT
    (evidence_id,) = state.evidence_ids
    assert store.get_evidence(evidence_id).result_id == state.results[0].result_id
    # Debugging was NOT activated: a valid negative is evidence, not a bug.
    assert repair.proposals == 0
    assert report.debug_attempts == 0
    assert record.debug_attempts == 0
    # Verification stayed selective: methodology once (pre-run, cached);
    # the verifier was not needed because the controls resolved the
    # implementation dimension deterministically.
    assert methodology.calls == 1
    assert verifier.calls == 0
    assert critic.performed == 0


def test_uncovered_negative_triggers_the_implementation_verifier(
    tmp_path: Path,
) -> None:
    """Without controls, a conclusive negative is exactly the event that
    justifies one semantic verification call."""
    spec, prediction = _spec_and_prediction(threshold=0.5)
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"), runs=({"heads_rate": 0.45},)
    )
    methodology = StubMethodologyReviewer(CheckState.PASS)
    verifier = StubImplementationVerifier(CheckState.PASS)
    runtime, _, sink, _ = _runtime(
        tmp_path, engineer, methodology=methodology, verifier=verifier
    )

    report = runtime.step(_prepared_state(spec, prediction))

    assert verifier.calls == 1
    record = sink.records[-1]
    assert record.verification_status == ExperimentValidityStatus.VERIFIED
    assert record.negative_result_verdict == "accepted"
    assert any("verified scientific negative" in n for n in report.notes)


def test_verifier_rejection_defers_the_negative(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction(threshold=0.5)
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"), runs=({"heads_rate": 0.45},)
    )
    verifier = StubImplementationVerifier(CheckState.FAIL)
    runtime, _, sink, _ = _runtime(tmp_path, engineer, verifier=verifier)

    report = runtime.step(_prepared_state(spec, prediction))

    record = sink.records[-1]
    assert record.implementation_rejected
    assert record.negative_result_verdict == "deferred"
    assert any("implementation rejected" in n for n in report.notes)


# -- E: methodological failure -> redesign, never debug, never a negative ----


def test_methodologically_invalid_design_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"))
    repair = FixedRepair(metrics={"heads_rate": 0.9})
    methodology = StubMethodologyReviewer(
        CheckState.FAIL,
        detail="heads_rate cannot answer a question about draw independence",
    )
    runtime, store, sink, critic = _runtime(
        tmp_path, engineer, repair=repair, methodology=methodology
    )

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state

    # Nothing executed, nothing was debugged, nothing scientific concluded.
    assert engineer.performed == 0
    assert repair.proposals == 0
    assert state.results == ()
    assert state.prediction_tests == ()
    assert state.evidence_ids == ()
    assert store.results() == ()
    (attempt,) = state.attempts
    assert attempt.status is AttemptStatus.FAILED
    assert attempt.outcome is not None
    assert "redesign the experiment" in str(attempt.outcome.error)
    record = sink.records[-1]
    assert record.methodology_rejected
    assert record.negative_result_verdict == ""
    assert any("methodological failure" in n for n in report.notes)
    assert any("redesign the experiment" in n for n in report.notes)
    assert critic.performed == 0

    # The verdict is cached: the next step re-rejects without a new review.
    second = runtime.step(state)
    assert methodology.calls == 1
    assert any("methodological failure" in n for n in second.notes)


# -- F: analytical failure -> redo analysis, execution not blamed -------------


def test_cherry_picking_analysis_is_rejected_without_blaming_execution(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction(threshold=0.5, seeds=(3, 0))
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        runs=({"heads_rate": 0.51}, {"heads_rate": 0.49}),
    )
    critic = CherryPickingCritic()
    runtime, _, sink, _ = _runtime(
        tmp_path,
        engineer,
        critic=critic,
        playbook=EmpiricalMLPlaybook(),
        # This scenario isolates ANALYSIS validity, so the (unwired)
        # verification governance is explicitly ablated — otherwise the
        # promotion gate would fail closed before the analysis gate runs.
        config=RuntimeConfig(verification_governance_enabled=False),
    )

    first = runtime.step(_prepared_state(spec, prediction))
    second = runtime.step(first.state)  # the contradiction triggers the critic

    assert critic.performed == 1
    record = sink.records[-1]
    assert record.analysis_rejected
    assert any("analytical failure" in n for n in second.notes)
    assert any("redo the analysis" in n for n in second.notes)
    # The invalid judgment never entered authoritative scientific state:
    # the gate rejected it before commit, and the critic attempt failed.
    assert second.state.assessments == ()
    analyze_attempt = next(
        a
        for a in second.state.attempts
        if a.action.action_type is ResearchActionType.ANALYZE
    )
    assert analyze_attempt.status is AttemptStatus.FAILED
    # Execution is not blamed: both runs stay valid committed results and
    # no engineering-failure machinery was touched.
    assert record.failure_category == ""
    assert record.debug_attempts == 0
    assert not any("engineering failure" in n for n in second.notes)
    assert len(second.state.results) == 2


# -- G: the bound holds at the runtime level ----------------------------------


def test_persistent_failure_stops_at_max_debug_attempts(tmp_path: Path) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), runs=(None,))
    repair = FixedRepair(metrics=None)  # every repair crashes again
    runtime, store, sink, _ = _runtime(
        tmp_path,
        engineer,
        repair=repair,
        config=RuntimeConfig(max_debug_attempts=2),
    )

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state

    assert repair.proposals == 2
    assert report.debug_attempts == 2
    assert not report.debug_resolved
    assert any("stopped after 2 attempt(s)" in n for n in report.notes)
    # Initial failure + two failed retries: three separate honest records.
    assert len(state.results) == 3
    assert all(
        not store.get_result(ref.result_id).succeeded for ref in state.results
    )
    debug_attempts = [
        a
        for a in state.attempts
        if a.action.action_type is ResearchActionType.DEBUG
    ]
    assert len(debug_attempts) == 2
    record = sink.records[-1]
    assert record.debug_attempts == 2
    assert not record.debug_resolved


# -- H: cost accounting -------------------------------------------------------


def test_failed_and_retried_executions_bill_their_actual_cost(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(LocalExecutor(tmp_path / "runs"), runs=(None,))
    repair = FixedRepair(metrics=None)
    runtime, store, _, _ = _runtime(
        tmp_path,
        engineer,
        repair=repair,
        config=RuntimeConfig(max_debug_attempts=2),
    )
    before = _prepared_state(spec, prediction)

    report = runtime.step(before)
    state = report.state

    # Every execution that ran — the failure and both failed retries — is
    # committed, and the budget was billed the sum of their actual costs.
    executed = tuple(store.get_result(ref.result_id) for ref in state.results)
    assert len(executed) == 3
    total = sum(r.cost.wall_clock_seconds for r in executed)
    assert total > 0.0
    spent = (
        before.budget.wall_clock_seconds - state.budget.wall_clock_seconds
    )
    assert spent == pytest.approx(total)
    # Each attempt's outcome carries its own actual cost.
    for attempt in state.attempts:
        assert attempt.outcome is not None
        assert attempt.outcome.actual_cost.wall_clock_seconds > 0.0


def test_preflight_rejection_prevents_execution_and_bills_nothing(
    tmp_path: Path,
) -> None:
    spec, prediction = _spec_and_prediction()
    engineer = StubEngineer(
        LocalExecutor(tmp_path / "runs"),
        preflight_command=("definitely-not-a-binary-anywhere",),
    )
    runtime, store, sink, _ = _runtime(tmp_path, engineer)
    before = _prepared_state(spec, prediction)

    report = runtime.step(before)
    state = report.state

    assert state.results == ()
    assert store.results() == ()
    (attempt,) = state.attempts
    assert attempt.status is AttemptStatus.FAILED
    assert attempt.outcome is not None
    assert attempt.outcome.actual_cost == NO_COST
    assert state.budget == before.budget  # nothing ran, nothing billed
    record = sink.records[-1]
    assert record.preflight_failed
    assert record.failure_category == "preflight"
    assert any("preflight rejected" in n for n in report.notes)
