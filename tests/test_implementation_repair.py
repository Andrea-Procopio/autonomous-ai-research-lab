"""Bounded implementation repair: entered only on implementation-invalidity
evidence, never on scientific disappointment.

The two critical invariants of this milestone:

1. a scientifically disappointing but valid result cannot enter debugging;
2. a completed result with independent evidence of an implementation bug
   can enter bounded implementation repair — and the repaired run earns its
   own verification while the original stays preserved.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.budget import ResearchBudget
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
from autonomous_research_lab.core.proposals import Proposal, ResultProposal
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.orchestration.debug_loop import (
    ExperimentDebugger,
    ImplementationRepairTrigger,
    RepairKind,
    RepairProposal,
    ScientificOutcomeError,
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
from autonomous_research_lab.runtime.verification import (
    CheckState,
    ExperimentValidityStatus,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)

QUESTION = ResearchQuestion(text="Is the stream fair?")
HYPOTHESIS = Hypothesis(statement="The stream is biased.", question_id=QUESTION.id)

_ECHO = (
    "import json, os, pathlib; "
    "d = pathlib.Path(os.environ['ARL_RUN_DIR']); "
    "cfg = json.loads(pathlib.Path(os.environ['ARL_CONFIG']).read_text()); "
    "(d / 'metrics.json').write_text(json.dumps(cfg))"
)

OVERFIT_CONTROL = PositiveControl(
    name="tiny_overfit",
    metric="overfit_acc",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.99,
    rationale="a faithful implementation must overfit the tiny probe set",
)

BUGGY = {"heads_rate": 0.45, "overfit_acc": 0.5}
FIXED = {"heads_rate": 0.45, "overfit_acc": 1.0}


def _spec_and_prediction() -> tuple[ExperimentSpec, Prediction]:
    prediction = Prediction(
        hypothesis_id=HYPOTHESIS.id,
        condition="one draw stream",
        metric="heads_rate",
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=0.5,
    )
    spec = ExperimentSpec(
        prediction_id=prediction.id,
        objective="measure the rate",
        procedure="run the stream and report",
        metrics=("heads_rate",),
        seeds=(7,),
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


def _job(spec: ExperimentSpec, metrics: dict[str, float], seed: int) -> ExperimentJob:
    return ExperimentJob(
        spec_id=spec.id,
        command=(sys.executable, "-c", _ECHO),
        config=dict(metrics),
        seed=seed,
        timeout_seconds=30.0,
    )


@dataclass
class StubEngineer(ResearchRole):
    executor: LocalExecutor
    metrics_payload: dict[str, float]
    performed: int = 0

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_ENGINEER

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset({ResearchActionType.RUN_EXPERIMENT})

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        self.performed += 1
        (spec,) = invocation.context.experiments
        job = _job(spec, self.metrics_payload, seed=7)
        result = self.executor.collect(self.executor.submit(job))
        return (ResultProposal(result=result, proposer="stub:engineer"),)


@dataclass
class ImplFix:
    """Implementation-repair strategy double: reruns with fixed metrics."""

    metrics: dict[str, float] | None
    proposals: int = 0
    triggers: list[ImplementationRepairTrigger] = field(default_factory=list)

    def propose(
        self,
        spec: ExperimentSpec,
        invalid: ExperimentResult,
        trigger: ImplementationRepairTrigger,
        attempt_number: int,
    ) -> RepairProposal | None:
        self.triggers.append(trigger)
        self.proposals += 1
        if self.metrics is None:
            return None
        return RepairProposal(
            job=_job(spec, self.metrics, seed=200 + self.proposals),
            rationale=f"fix the miscount (attempt {attempt_number})",
        )


@dataclass
class NoExecutionRepair:
    """Execution-repair strategy that must never be consulted here."""

    proposals: int = 0

    def propose(
        self,
        spec: ExperimentSpec,
        failed: ExperimentResult,
        diagnosis: object,
        attempt_number: int,
    ) -> RepairProposal | None:
        self.proposals += 1
        return None


@dataclass
class QueueVerifier:
    """Implementation verifier returning queued verdicts, in order."""

    verdicts: list[CheckState]
    calls: int = 0

    def verify(
        self,
        spec: ExperimentSpec,
        result: ExperimentResult,
        prediction: Prediction | None,
        checks: tuple[VerificationCheck, ...],
    ) -> VerificationCheck:
        state = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return VerificationCheck(
            dimension=ValidityDimension.IMPLEMENTATION,
            name="implementation_faithfulness",
            state=state,
            detail="queued verdict",
        )


@dataclass
class PassMethodology:
    def review(
        self,
        spec: ExperimentSpec,
        prediction: Prediction | None,
        *,
        objective: str,
    ) -> VerificationCheck:
        return VerificationCheck(
            dimension=ValidityDimension.METHODOLOGY,
            name="methodological_validity",
            state=CheckState.PASS,
        )


@dataclass
class ListSink:
    records: list[StepMetrics] = field(default_factory=list)

    def log(self, record: StepMetrics) -> None:
        self.records.append(record)


def _runtime(
    tmp_path: Path,
    metrics_payload: dict[str, float],
    *,
    impl_strategy: ImplFix | None,
    verifier: QueueVerifier | None = None,
    controls: bool = True,
    methodology: bool = False,
    config: RuntimeConfig | None = None,
) -> tuple[ResearchRuntime, InMemoryEvidenceStore, ListSink]:
    store = InMemoryEvidenceStore()
    sink = ListSink()
    executor = LocalExecutor(tmp_path / "runs")
    runtime = ResearchRuntime(
        config=config or RuntimeConfig(),
        director=RuleBasedFrontierDirector(),
        roles={
            RoleName.RESEARCH_ENGINEER: StubEngineer(executor, metrics_payload),
        },
        store=store,
        metrics=sink,
        debugger=ExperimentDebugger(
            executor=executor,
            strategy=NoExecutionRepair(),
            implementation_strategy=impl_strategy,
        ),
        implementation_verifier=verifier,
        methodology_reviewer=PassMethodology() if methodology else None,
        control_source=(
            (lambda spec: (OVERFIT_CONTROL,)) if controls else None
        ),
    )
    return runtime, store, sink


# -- critical invariant 1: disappointment alone never repairs -----------------


def test_verified_negative_never_enters_implementation_repair(
    tmp_path: Path,
) -> None:
    impl = ImplFix(metrics=FIXED)
    runtime, _, sink = _runtime(
        tmp_path, FIXED, impl_strategy=impl, methodology=True
    )
    spec, prediction = _spec_and_prediction()

    report = runtime.step(_prepared_state(spec, prediction))

    record = sink.records[-1]
    assert record.verification_status == ExperimentValidityStatus.VERIFIED
    assert record.negative_result_verdict == "accepted"
    (test,) = report.state.prediction_tests
    assert test.consistency is Consistency.INCONSISTENT  # disappointing...
    assert impl.proposals == 0  # ...and untouchable by repair
    assert report.implementation_debug_attempts == 0
    assert record.implementation_debug_attempts == 0


# -- critical invariant 2: implementation evidence opens bounded repair -------


def test_completed_silent_bug_enters_bounded_implementation_repair(
    tmp_path: Path,
) -> None:
    impl = ImplFix(metrics=FIXED)
    runtime, store, sink = _runtime(tmp_path, BUGGY, impl_strategy=impl)
    spec, prediction = _spec_and_prediction()
    before = _prepared_state(spec, prediction)

    report = runtime.step(before)
    state = report.state

    # The repair ran, bounded, and resolved.
    assert impl.proposals == 1
    assert report.implementation_debug_attempts == 1
    assert report.implementation_debug_resolved
    record = sink.records[-1]
    assert record.implementation_debug_attempts == 1
    assert record.implementation_debug_resolved
    assert any(
        "implementation repair succeeded on attempt 1" in n
        for n in report.notes
    )

    # The trigger carried implementation-invalidity evidence, not outcomes.
    (trigger,) = impl.triggers
    assert all(
        c.dimension is ValidityDimension.IMPLEMENTATION for c in trigger.checks
    )
    assert any(c.state is CheckState.FAIL for c in trigger.checks)

    # Both runs are separate committed records; the original is preserved
    # with its adverse verdict, the rerun earned its own (case G and H).
    assert len(state.results) == 2
    original_id, retry_id = (
        state.results[0].result_id,
        state.results[1].result_id,
    )
    original_verdict = runtime.verifications.get(original_id)
    retry_verdict = runtime.verifications.get(retry_id)
    assert original_verdict is not None and retry_verdict is not None
    assert (
        original_verdict.validity
        is ExperimentValidityStatus.IMPLEMENTATION_UNCERTAIN
    )
    assert retry_verdict.report.dimension_state(
        ValidityDimension.IMPLEMENTATION
    ) is CheckState.PASS
    assert store.get_result(original_id).metrics["overfit_acc"] == 0.5

    # The rerun is a distinct auditable DEBUG attempt naming its basis.
    (debug_attempt,) = [
        a
        for a in state.attempts
        if a.action.action_type is ResearchActionType.DEBUG
    ]
    assert "implementation invalidity" in debug_attempt.action.rationale
    assert "fix the miscount" in debug_attempt.action.rationale

    # Case I: every execution billed its actual cost.
    executed = tuple(store.get_result(ref.result_id) for ref in state.results)
    total = sum(r.cost.wall_clock_seconds for r in executed)
    spent = before.budget.wall_clock_seconds - state.budget.wall_clock_seconds
    assert total > 0.0
    assert spent == pytest.approx(total)


def test_semantic_verifier_fail_also_opens_repair(tmp_path: Path) -> None:
    impl = ImplFix(metrics=FIXED)
    verifier = QueueVerifier([CheckState.FAIL, CheckState.PASS])
    runtime, _, sink = _runtime(
        tmp_path,
        {"heads_rate": 0.45},
        impl_strategy=impl,
        verifier=verifier,
        controls=False,
    )
    spec, prediction = _spec_and_prediction()

    report = runtime.step(_prepared_state(spec, prediction))

    assert verifier.calls == 2  # original FAIL, rerun independently PASSed
    assert impl.proposals == 1
    assert report.implementation_debug_resolved
    assert sink.records[-1].implementation_rejected


def test_persistent_implementation_bug_stops_at_the_bound(
    tmp_path: Path,
) -> None:
    impl = ImplFix(metrics=BUGGY)  # every "fix" is still buggy
    runtime, _, _ = _runtime(
        tmp_path,
        BUGGY,
        impl_strategy=impl,
        config=RuntimeConfig(max_debug_attempts=2),
    )
    spec, prediction = _spec_and_prediction()

    report = runtime.step(_prepared_state(spec, prediction))

    assert impl.proposals == 2
    assert report.implementation_debug_attempts == 2
    assert not report.implementation_debug_resolved
    assert any(
        "implementation repair stopped after 2 attempt(s)" in n
        for n in report.notes
    )
    # Original plus both bounded retries, all preserved.
    assert len(report.state.results) == 3


def test_fresh_failures_yield_fresh_triggers(tmp_path: Path) -> None:
    """Each bounded attempt answers the *latest* run's evidence: a rerun
    that still fails its controls produces a new trigger anchored to that
    rerun, never a stale re-read of the original result."""
    impl = ImplFix(metrics=BUGGY)  # first "fix" is still buggy
    runtime, _, _ = _runtime(
        tmp_path,
        BUGGY,
        impl_strategy=impl,
        config=RuntimeConfig(max_debug_attempts=2),
    )
    spec, prediction = _spec_and_prediction()

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state

    assert impl.proposals == 2
    first_trigger, second_trigger = impl.triggers
    original_id, first_retry_id, _ = (ref.result_id for ref in state.results)
    assert first_trigger.result_id == original_id
    # The second attempt's trigger indicts the first retry, not the
    # original — the loop reasons from each fresh verification report.
    assert second_trigger.result_id == first_retry_id
    assert second_trigger.result_id != original_id
    assert any(c.state is CheckState.FAIL for c in second_trigger.checks)


@dataclass
class CrashingThenFixedImplRepair(ImplFix):
    """First reimplementation crashes outright — an execution failure."""

    def propose(
        self,
        spec: ExperimentSpec,
        invalid: ExperimentResult,
        trigger: ImplementationRepairTrigger,
        attempt_number: int,
    ) -> RepairProposal | None:
        self.triggers.append(trigger)
        self.proposals += 1
        return RepairProposal(
            job=ExperimentJob(
                spec_id=spec.id,
                command=(sys.executable, "-c", "raise SystemExit(3)"),
                seed=300 + self.proposals,
                timeout_seconds=30.0,
            ),
            rationale="rewrite the counter (which crashes)",
        )


@dataclass
class FixedExecutionRepair:
    """Execution-repair strategy that reruns with the fixed metrics."""

    metrics: dict[str, float]
    proposals: int = 0

    def propose(
        self,
        spec: ExperimentSpec,
        failed: ExperimentResult,
        diagnosis: object,
        attempt_number: int,
    ) -> RepairProposal | None:
        self.proposals += 1
        return RepairProposal(
            job=_job(spec, self.metrics, seed=400 + self.proposals),
            rationale="relaunch the rewritten counter",
        )


def test_crashed_reimplementation_transitions_to_execution_repair(
    tmp_path: Path,
) -> None:
    """A reimplementation that crashes is an execution failure: it is
    diagnosed by the classifier and repaired with execution-repair
    semantics — inside the same bounded episode."""
    impl = CrashingThenFixedImplRepair(metrics=None)
    execution = FixedExecutionRepair(metrics=FIXED)
    store = InMemoryEvidenceStore()
    sink = ListSink()
    executor = LocalExecutor(tmp_path / "runs")
    runtime = ResearchRuntime(
        config=RuntimeConfig(max_debug_attempts=3),
        director=RuleBasedFrontierDirector(),
        roles={
            RoleName.RESEARCH_ENGINEER: StubEngineer(executor, BUGGY),
        },
        store=store,
        metrics=sink,
        debugger=ExperimentDebugger(
            executor=executor,
            strategy=execution,
            implementation_strategy=impl,
        ),
        control_source=lambda spec: (OVERFIT_CONTROL,),
    )
    spec, prediction = _spec_and_prediction()
    before = _prepared_state(spec, prediction)

    report = runtime.step(before)
    state = report.state

    # Attempt 1: implementation strategy (crashes). Attempt 2: the crash is
    # diagnosed and handed to the execution strategy, which recovers a
    # completed run whose fresh verification passes its controls.
    assert impl.proposals == 1
    assert execution.proposals == 1
    assert report.implementation_debug_attempts == 2
    assert report.implementation_debug_resolved
    assert any(
        "reimplementation crashed — execution failure diagnosed" in n
        for n in report.notes
    )
    assert any(
        "implementation repair succeeded on attempt 2" in n
        for n in report.notes
    )
    # Original + crashed rerun + recovered rerun, all preserved and billed.
    assert len(state.results) == 3
    executed = tuple(store.get_result(ref.result_id) for ref in state.results)
    statuses = [r.succeeded for r in executed]
    assert statuses == [True, False, True]
    spent = (
        before.budget.wall_clock_seconds - state.budget.wall_clock_seconds
    )
    assert spent == pytest.approx(
        sum(r.cost.wall_clock_seconds for r in executed)
    )
    # The recovered run earned its own verification record.
    final_verdict = runtime.verifications.get(executed[2].id)
    assert final_verdict is not None
    assert final_verdict.report.dimension_state(
        ValidityDimension.IMPLEMENTATION
    ) is CheckState.PASS


def test_invalid_negative_then_verified_positive_regression(
    tmp_path: Path,
) -> None:
    """THE consistency scenario after silent-bug repair.

    Run A: completes, PredictionTest INCONSISTENT, control FAIL →
    IMPLEMENTATION_UNCERTAIN. Repair produces run B: completes,
    PredictionTest CONSISTENT, controls PASS → VERIFIED. Everything stays
    recorded; only B participates in scientific inference.
    """
    from autonomous_research_lab.core.assessment import (
        AssessmentVerdict,
        EpistemicAssessment,
    )
    from autonomous_research_lab.core.proposals import AssessmentProposal
    from autonomous_research_lab.orchestration.loop import (
        PromotionError,
        _conclusive_evidence_ids,
        _standing_notes,
    )
    from autonomous_research_lab.runtime.frontier import (
        build_frontier,
        find_contradictions,
    )
    from autonomous_research_lab.runtime.verification import OutcomeStanding
    from autonomous_research_lab.runtime.verification_store import (
        ScientificAdmissibility,
    )

    impl = ImplFix(metrics={"heads_rate": 0.52, "overfit_acc": 1.0})
    runtime, store, _ = _runtime(
        tmp_path, BUGGY, impl_strategy=impl, methodology=True
    )
    spec, prediction = _spec_and_prediction()

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state
    assert report.implementation_debug_resolved

    # -- RECORDED: every fact about A and B persists, immutably -------------
    a_ref, b_ref = state.results
    run_a = store.get_result(a_ref.result_id)
    run_b = store.get_result(b_ref.result_id)
    assert run_a.metrics["heads_rate"] == 0.45  # A result stored
    assert run_b.metrics["heads_rate"] == 0.52  # B result stored
    tests = {t.result_id: t.consistency for t in state.prediction_tests}
    assert tests[run_a.id] is Consistency.INCONSISTENT  # A test stored
    assert tests[run_b.id] is Consistency.CONSISTENT  # B test stored
    evidence_of = {
        store.get_evidence(eid).result_id: eid for eid in state.evidence_ids
    }
    assert run_a.id in evidence_of  # A evidence stored
    assert run_b.id in evidence_of  # B evidence stored
    verdict_a = runtime.verifications.get(run_a.id)
    verdict_b = runtime.verifications.get(run_b.id)
    assert verdict_a is not None and verdict_b is not None
    assert (
        verdict_a.validity
        is ExperimentValidityStatus.IMPLEMENTATION_UNCERTAIN
    )  # A adverse record stored
    assert verdict_a.standing is OutcomeStanding.OBSERVED_UNRESOLVED
    assert verdict_b.validity is ExperimentValidityStatus.VERIFIED

    # -- ADMISSIBLE: only B participates in scientific inference ------------
    policy = ScientificAdmissibility(
        verifications=runtime.verifications, governance_enabled=True
    )
    assert not policy(run_a.id)
    assert policy(run_b.id)

    frontier = build_frontier(state, admissible=policy)
    assert prediction not in frontier.unresolved_predictions  # resolved by B
    assert frontier.contradictions == ()  # A opposes B mechanically only
    assert find_contradictions(state, admissible=policy) == ()
    assert len(find_contradictions(state)) == 1  # ...and history shows it

    # A raised no scientific critic escalation, tier, or synthesis.
    assert report.critic_reasons == ()
    assert not report.critic_invoked
    assert report.synthesis is None

    # A conclusive assessment citing only B passes BOTH gates — the
    # promotion/coverage deadlock is gone.
    citing_b = AssessmentProposal(
        assessment=EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.SUPPORTED,
            method="test:direct",
            evidence_ids=(evidence_of[run_b.id],),
        ),
        proposer="test",
    )
    runtime._gate_promotions((citing_b,))  # no raise
    runtime._gate_analysis(state, (citing_b,))  # no raise: A not required
    # Coverage owed = admissible family only; the mechanical family is wider.
    assert _conclusive_evidence_ids(
        state, store, HYPOTHESIS.id, admissible=policy
    ) == (evidence_of[run_b.id],)
    assert len(_conclusive_evidence_ids(state, store, HYPOTHESIS.id)) == 2

    # Citing A conclusively is still blocked — inspectable, never trusted.
    citing_a = AssessmentProposal(
        assessment=EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.SUPPORTED,
            method="test:direct",
            evidence_ids=(evidence_of[run_a.id],),
        ),
        proposer="test",
    )
    with pytest.raises(PromotionError):
        runtime._gate_promotions((citing_a,))

    # Case G: A stays visible to reasoning, annotated with its standing.
    (note,) = _standing_notes(
        (store.get_evidence(evidence_of[run_a.id]),), runtime.verifications
    )
    assert "implementation_uncertain" in note
    assert "observed_unresolved" in note


def test_repair_is_off_without_debug_enabled(tmp_path: Path) -> None:
    impl = ImplFix(metrics=FIXED)
    runtime, _, _ = _runtime(
        tmp_path,
        BUGGY,
        impl_strategy=impl,
        config=RuntimeConfig(debug_enabled=False),
    )
    spec, prediction = _spec_and_prediction()
    report = runtime.step(_prepared_state(spec, prediction))
    assert impl.proposals == 0
    assert report.implementation_debug_attempts == 0


# -- structural rejection of outcome-based repair (case D) --------------------


def test_trigger_cannot_be_built_from_scientific_outcomes() -> None:
    prediction_check = VerificationCheck(
        dimension=ValidityDimension.ANALYSIS,
        name="prediction_test",
        state=CheckState.FAIL,
        detail="observed 0.45 vs ge 0.5 — scientifically disappointing",
    )
    with pytest.raises(ValueError, match="not repair evidence"):
        ImplementationRepairTrigger(result_id="res_x", checks=(prediction_check,))

    passing_only = VerificationCheck(
        dimension=ValidityDimension.IMPLEMENTATION,
        name="positive_control:probe",
        state=CheckState.PASS,
    )
    with pytest.raises(ValueError, match="at least one FAILED"):
        ImplementationRepairTrigger(result_id="res_x", checks=(passing_only,))

    with pytest.raises(ValueError, match="none was given"):
        ImplementationRepairTrigger(result_id="res_x", checks=())


def test_debug_still_refuses_completed_results(tmp_path: Path) -> None:
    spec, _ = _spec_and_prediction()
    executor = LocalExecutor(tmp_path / "runs")
    completed = executor.collect(
        executor.submit(_job(spec, BUGGY, seed=7))
    )
    assert completed.succeeded
    debugger = ExperimentDebugger(
        executor=executor,
        strategy=NoExecutionRepair(),
        implementation_strategy=ImplFix(metrics=FIXED),
    )
    with pytest.raises(ScientificOutcomeError):
        debugger.debug(spec, completed)


def test_repair_implementation_requires_a_strategy(tmp_path: Path) -> None:
    spec, _ = _spec_and_prediction()
    executor = LocalExecutor(tmp_path / "runs")
    completed = executor.collect(executor.submit(_job(spec, BUGGY, seed=7)))
    debugger = ExperimentDebugger(
        executor=executor, strategy=NoExecutionRepair()
    )
    trigger = ImplementationRepairTrigger(
        result_id=completed.id,
        checks=(OVERFIT_CONTROL.evaluate(completed.metrics),),
    )
    with pytest.raises(RuntimeError, match="no implementation repair strategy"):
        debugger.repair_implementation(spec, completed, trigger)


def test_repair_implementation_session_is_typed_and_audited(
    tmp_path: Path,
) -> None:
    spec, _ = _spec_and_prediction()
    executor = LocalExecutor(tmp_path / "runs")
    completed = executor.collect(executor.submit(_job(spec, BUGGY, seed=7)))
    trigger = ImplementationRepairTrigger(
        result_id=completed.id,
        checks=(OVERFIT_CONTROL.evaluate(completed.metrics),),
    )
    debugger = ExperimentDebugger(
        executor=executor,
        strategy=NoExecutionRepair(),
        implementation_strategy=ImplFix(metrics=FIXED),
    )

    session = debugger.repair_implementation(spec, completed, trigger)

    assert session.kind is RepairKind.IMPLEMENTATION
    assert session.trigger is trigger
    assert session.initial_diagnosis is None
    assert session.resolved
    (attempt,) = session.attempts
    assert attempt.diagnosis is None
    assert attempt.basis == "implementation invalidity"
    assert attempt.result.succeeded
    # The original completed result is untouched by the repair.
    assert completed.metrics["overfit_acc"] == 0.5
