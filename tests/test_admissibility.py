"""Scientific admissibility: mechanically conclusive != scientifically
admissible.

A result may remain permanently recorded as historical fact without being
allowed to participate in scientific inference. These tests pin the
governed control plane — frontier resolution, contradiction detection,
critic triggering — against the one canonical policy.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.experiment import (
    ExperimentSpec,
    ExperimentStatus,
    ResultRef,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.core.proposals import Proposal, ResultProposal
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.orchestration.critic_trigger import CriticTrigger
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.frontier import (
    build_frontier,
    find_contradictions,
)
from autonomous_research_lab.runtime.metrics import StepMetrics
from autonomous_research_lab.runtime.playbook import EmpiricalMLPlaybook
from autonomous_research_lab.runtime.verification import (
    CheckState,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
    VerificationReport,
)
from autonomous_research_lab.runtime.verification_store import (
    InMemoryVerificationStore,
    ScientificAdmissibility,
    VerificationRecord,
)

QUESTION = ResearchQuestion(text="Is the stream fair?")
HYPOTHESIS = Hypothesis(statement="The stream is biased.", question_id=QUESTION.id)

PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="one draw stream",
    metric="heads_rate",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.5,
)
SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="measure the rate",
    procedure="run the stream and report",
    metrics=("heads_rate",),
    seeds=(3, 0),
)


def _report(implementation: CheckState) -> VerificationReport:
    return VerificationReport(
        checks=tuple(
            VerificationCheck(dimension=d, name=f"{d}_check", state=s)
            for d, s in {
                ValidityDimension.EXECUTION: CheckState.PASS,
                ValidityDimension.IMPLEMENTATION: implementation,
                ValidityDimension.METHODOLOGY: CheckState.PASS,
                ValidityDimension.ANALYSIS: CheckState.PASS,
            }.items()
        )
    )


def _policy(
    verified: tuple[str, ...] = (),
    invalid: tuple[str, ...] = (),
    *,
    governance: bool = True,
) -> ScientificAdmissibility:
    store = InMemoryVerificationStore()
    for result_id in verified:
        store.record(
            VerificationRecord(
                result_id=result_id,
                spec_id=SPEC.id,
                report=_report(CheckState.PASS),
            )
        )
    for result_id in invalid:
        store.record(
            VerificationRecord(
                result_id=result_id,
                spec_id=SPEC.id,
                report=_report(CheckState.FAIL),
            )
        )
    return ScientificAdmissibility(
        verifications=store, governance_enabled=governance
    )


def _state_with_tests(
    *tests: tuple[str, Consistency, float]
) -> ResearchState:
    state = (
        ResearchState(objective="fairness")
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(PREDICTION)
        .add_experiment(SPEC)
    )
    for result_id, consistency, observed in tests:
        state = state.record_result(
            ResultRef(
                result_id=result_id,
                spec_id=SPEC.id,
                status=ExperimentStatus.COMPLETED,
            )
        )
        state = state.record_prediction_test(
            PredictionTest(
                prediction_id=PREDICTION.id,
                result_id=result_id,
                metric=PREDICTION.metric,
                observed=observed,
                consistency=consistency,
            )
        )
    return state


# -- the policy itself --------------------------------------------------------


def test_admissibility_semantics() -> None:
    policy = _policy(verified=("res_ok",), invalid=("res_bad",))
    assert policy("res_ok")
    assert not policy("res_bad")  # adverse record
    assert not policy("res_missing")  # missing record fails closed
    ablated = _policy(invalid=("res_bad",), governance=False)
    assert ablated("res_bad") and ablated("res_missing")  # explicit ablation


# -- frontier resolution (cases A, B, C) --------------------------------------


def test_invalid_only_conclusive_test_leaves_prediction_unresolved() -> None:
    state = _state_with_tests(("res_a", Consistency.INCONSISTENT, 0.45))
    policy = _policy(invalid=("res_a",))
    frontier = build_frontier(state, admissible=policy)
    assert PREDICTION in frontier.unresolved_predictions
    # The mechanical test stays fully on the record.
    (test,) = state.prediction_tests
    assert test.consistency is Consistency.INCONSISTENT


def test_missing_record_under_governance_does_not_resolve() -> None:
    state = _state_with_tests(("res_a", Consistency.CONSISTENT, 0.55))
    policy = _policy()  # empty store: no record at all
    frontier = build_frontier(state, admissible=policy)
    assert PREDICTION in frontier.unresolved_predictions


def test_explicit_ablation_preserves_legacy_resolution() -> None:
    state = _state_with_tests(("res_a", Consistency.CONSISTENT, 0.55))
    ablated = _policy(governance=False)
    assert PREDICTION not in build_frontier(
        state, admissible=ablated
    ).unresolved_predictions
    # And the legacy default (no policy at all) behaves identically.
    assert PREDICTION not in build_frontier(state).unresolved_predictions


def test_verified_conclusive_test_resolves() -> None:
    state = _state_with_tests(("res_a", Consistency.CONSISTENT, 0.55))
    policy = _policy(verified=("res_a",))
    assert PREDICTION not in build_frontier(
        state, admissible=policy
    ).unresolved_predictions


# -- contradictions (case D and its inverse) ----------------------------------


def test_invalid_negative_plus_verified_positive_is_no_contradiction() -> None:
    state = _state_with_tests(
        ("res_bad", Consistency.INCONSISTENT, 0.45),
        ("res_ok", Consistency.CONSISTENT, 0.55),
    )
    policy = _policy(verified=("res_ok",), invalid=("res_bad",))
    assert find_contradictions(state, admissible=policy) == ()
    # Mechanically the record is still mixed — history is not rewritten.
    assert len(find_contradictions(state)) == 1


def test_two_verified_opposing_results_still_contradict() -> None:
    state = _state_with_tests(
        ("res_neg", Consistency.INCONSISTENT, 0.49),
        ("res_pos", Consistency.CONSISTENT, 0.51),
    )
    policy = _policy(verified=("res_neg", "res_pos"))
    (contradiction,) = find_contradictions(state, admissible=policy)
    assert contradiction.subject_id == PREDICTION.id


# -- critic triggering (case E and contradiction gating) ----------------------


def test_invalid_large_effect_raises_no_scientific_trigger() -> None:
    state = _state_with_tests(("res_bad", Consistency.CONSISTENT, 0.9))
    (test,) = state.prediction_tests
    trigger = CriticTrigger()
    # Legacy reads the huge deviation as a large effect...
    assert any(
        "unexpectedly large effect" in r
        for r in trigger.reasons(state, test=test)
    )
    # ...but an implementation-invalid result raises no scientific reasons.
    policy = _policy(invalid=("res_bad",))
    assert trigger.reasons(state, test=test, admissible=policy) == ()


def test_contradiction_counting_uses_admissible_tests_only() -> None:
    state = _state_with_tests(
        ("res_bad", Consistency.INCONSISTENT, 0.45),
        ("res_ok", Consistency.CONSISTENT, 0.51),
    )
    ok_test = next(
        t for t in state.prediction_tests if t.result_id == "res_ok"
    )
    trigger = CriticTrigger()
    mixed_policy = _policy(verified=("res_ok",), invalid=("res_bad",))
    assert not any(
        "contradictory replications" in r
        for r in trigger.reasons(state, test=ok_test, admissible=mixed_policy)
    )
    both_verified = _policy(verified=("res_ok", "res_bad"))
    assert any(
        "contradictory replications" in r
        for r in trigger.reasons(state, test=ok_test, admissible=both_verified)
    )


def test_director_request_survives_inadmissibility() -> None:
    state = _state_with_tests(("res_bad", Consistency.CONSISTENT, 0.9))
    (test,) = state.prediction_tests
    reasons = CriticTrigger().reasons(
        state,
        test=test,
        director_request="please double-check",
        admissible=_policy(invalid=("res_bad",)),
    )
    assert reasons == ("director request: please double-check",)


# -- runtime case D: a fully verified contradiction still escalates -----------

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
)


@dataclass
class SeqEngineer(ResearchRole):
    executor: LocalExecutor
    runs: tuple[dict[str, float], ...]
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
        run = self.runs[min(self.performed, len(self.runs) - 1)]
        self.performed += 1
        (spec,) = invocation.context.experiments
        job = ExperimentJob(
            spec_id=spec.id,
            command=(sys.executable, "-c", _ECHO),
            config=dict(run),
            seed=100 + self.performed,
            timeout_seconds=30.0,
        )
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


def test_fully_verified_contradiction_still_escalates(tmp_path: Path) -> None:
    """Case D: governance must not mute *real* science — two verified
    opposing runs still register as a contradiction, wake the critic, and
    trigger synthesis."""
    critic = SpyCritic()
    sink = ListSink()
    runtime = ResearchRuntime(
        config=RuntimeConfig(),
        director=RuleBasedFrontierDirector(),
        roles={
            RoleName.RESEARCH_ENGINEER: SeqEngineer(
                LocalExecutor(tmp_path / "runs"),
                runs=(
                    {"heads_rate": 0.51, "overfit_acc": 1.0},
                    {"heads_rate": 0.49, "overfit_acc": 1.0},
                ),
            ),
            RoleName.RESULT_ANALYST: critic,
        },
        store=InMemoryEvidenceStore(),
        metrics=sink,
        playbook=EmpiricalMLPlaybook(),
        methodology_reviewer=PassMethodology(),
        control_source=lambda spec: (OVERFIT_CONTROL,),
    )
    state = (
        ResearchState(
            objective="fairness",
            budget=ResearchBudget(
                wall_clock_seconds=3600.0, usd=10.0, model_tokens=200_000
            ),
        )
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(PREDICTION)
        .add_experiment(SPEC)
    )

    first = runtime.step(state)
    second = runtime.step(first.state)  # the replication contradicts

    assert critic.performed == 1
    assert second.critic_invoked
    assert any("contradictory replications" in r for r in second.critic_reasons)
    assert second.synthesis is not None  # contradiction-triggered slow loop
    policy = ScientificAdmissibility(
        verifications=runtime.verifications, governance_enabled=True
    )
    (contradiction,) = find_contradictions(second.state, admissible=policy)
    assert contradiction.subject_id == PREDICTION.id
