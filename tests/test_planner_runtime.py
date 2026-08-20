"""The model-backed planner inside the real runtime loop.

Deterministic end-to-end coverage of the Task 4 slice: a verified baseline
becomes admissible evidence; the planning director hands the seat to the
planner; one fake-provider decision commits its whole proposition chain
atomically through the governed commit; the engineer runs the planned
experiment with the planner-selected template; verification produces a
durable record; the second decision stops the program through the loop's
own halt path. Failure modes stay in their lanes: a gate-rejected decision
is a failed attempt that mutates nothing and reaches no engineer, and an
unaffordable planning step halts on the existing budget gate with zero
provider calls.

Runtimes here are rebuilt between steps around shared stores — states are
values, and the fake provider's script must be authored after the ids it
cites exist. The stores, not the runtime object, carry the continuity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autonomous_research_lab.core.attempt import AttemptStatus
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.experiment import (
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.execution.binding import HostPythonBinding
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.execution.runner import DirectJobRunner
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.orchestration.planning import PlanningDirector
from autonomous_research_lab.roles.base import RoleName
from autonomous_research_lab.roles.engineer import (
    ImplementationTemplate,
    ModelBackedEngineer,
)
from autonomous_research_lab.roles.planner import (
    ModelBackedPlanner,
    TemplateCapability,
    TemplateCatalog,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.implementation_store import (
    ImplementationStore,
)
from autonomous_research_lab.runtime.metrics import StepMetrics
from autonomous_research_lab.runtime.planning_store import (
    PlanningAction,
    PlanningStore,
    StopReason,
)
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    ScriptedReply,
    UsageLedger,
)
from autonomous_research_lab.runtime.verification import (
    CheckState,
    ExperimentValidityStatus,
    OutcomeStanding,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)
from autonomous_research_lab.runtime.verification_store import (
    InMemoryVerificationStore,
)

QUESTION = ResearchQuestion(text="does the fixture stream lean heads?")
HYPOTHESIS = Hypothesis(
    statement="the fixture stream is biased toward heads",
    question_id=QUESTION.id,
)
PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="one fixture stream",
    metric="heads_rate",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.5,
)
BASELINE_SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="measure the fixture heads rate",
    procedure="run the fixture and report heads_rate and tiny_acc",
    metrics=("heads_rate", "tiny_acc"),
    seeds=(7,),
    estimated_cost=ResourceCost(wall_clock_seconds=60.0),
)

BASELINE_TEMPLATE = ImplementationTemplate(
    name="baseline-template-v1", source="# baseline start\n"
)
PLANNED_TEMPLATE = ImplementationTemplate(
    name="planned-template-v1", source="# planned start\n"
)
OVERFIT_CONTROL = PositiveControl(
    name="tiny_overfit",
    metric="tiny_acc",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.99,
)
CATALOG = TemplateCatalog(
    entries=(
        TemplateCapability(
            template=BASELINE_TEMPLATE,
            metrics=("heads_rate", "tiny_acc"),
            estimated_cost=ResourceCost(wall_clock_seconds=60.0),
            control=OVERFIT_CONTROL,
        ),
        TemplateCapability(
            template=PLANNED_TEMPLATE,
            metrics=("heads_rate", "tails_rate", "tiny_acc"),
            estimated_cost=ResourceCost(wall_clock_seconds=60.0),
            control=OVERFIT_CONTROL,
        ),
    )
)

GOOD_SOURCE = """\
import json
import os
from pathlib import Path

metrics = {"heads_rate": 0.75, "tails_rate": 0.25, "tiny_acc": 1.0}
run_dir = Path(os.environ["ARL_RUN_DIR"])
(run_dir / "metrics.json").write_text(json.dumps(metrics))
"""


def _engineer_reply() -> str:
    return json.dumps(
        {
            "files": [{"path": "experiment.py", "content": GOOD_SOURCE}],
            "rationale": "deterministic fixture",
        }
    )


def _decision(evidence_id: str, **overrides: Any) -> str:
    payload: dict[str, Any] = {
        "action": "new_experiment",
        "question_id": QUESTION.id,
        "rationale": "the baseline is verified; measure the tails side",
        "evidence_ids": [evidence_id],
        "hypothesis_id": "",
        "hypothesis_statement": "the tails rate mirrors the heads bias",
        "prediction_condition": "on the same fixture stream",
        "prediction_metric": "tails_rate",
        "prediction_comparator": "le",
        "prediction_threshold": 0.5,
        "prediction_tolerance": 0,
        "prediction_expectation": "tails stays below half",
        "experiment_objective": "measure the fixture tails rate",
        "experiment_procedure": (
            "run the fixture and report heads_rate, tails_rate and tiny_acc"
        ),
        "experiment_metrics": ["tails_rate", "heads_rate", "tiny_acc"],
        "experiment_baselines": ["the verified heads-rate baseline"],
        "experiment_controls": ["tiny subset must be memorized"],
        "experiment_seeds": [13],
        "template_id": PLANNED_TEMPLATE.id,
        "target_experiment_id": "",
        "replication_seed": -1,
        "removed_component": "",
        "stop_reason": "none",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _stop_decision(evidence_id: str) -> str:
    return _decision(
        evidence_id,
        action="stop",
        hypothesis_statement="",
        prediction_condition="",
        prediction_metric="",
        prediction_comparator="none",
        prediction_threshold=0,
        prediction_tolerance=0,
        prediction_expectation="",
        experiment_objective="",
        experiment_procedure="",
        experiment_metrics=[],
        experiment_baselines=[],
        experiment_controls=[],
        experiment_seeds=[],
        template_id="",
        stop_reason="question_resolved",
        rationale="both sides of the fixture stream are now characterized",
    )


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
            detail="fixture design reviewed at wiring time",
        )


class ListSink:
    def __init__(self) -> None:
        self.records: list[StepMetrics] = []

    def log(self, record: StepMetrics) -> None:
        self.records.append(record)


class Stores:
    """The continuity between rebuilt runtimes: every durable artifact."""

    def __init__(self, tmp_path: Path) -> None:
        self.evidence = InMemoryEvidenceStore()
        self.verifications = InMemoryVerificationStore()
        self.implementations = ImplementationStore(tmp_path / "implementations")
        self.plans = PlanningStore(tmp_path / "planning")
        self.executor = LocalExecutor(tmp_path / "runs")
        self.sink = ListSink()


def _runtime(
    stores: Stores,
    *,
    engineer_replies: tuple[ScriptedReply | str, ...] = (),
    planner_replies: tuple[ScriptedReply | str, ...] = (),
) -> ResearchRuntime:
    ledger = UsageLedger()
    engineer = ModelBackedEngineer(
        provider=FakeModelProvider(engineer_replies),
        model="test-model",
        runner=DirectJobRunner(stores.executor),
        ledger=ledger,
        store=stores.implementations,
        binding=HostPythonBinding(timeout_seconds=60.0),
        template=BASELINE_TEMPLATE,
        template_resolver=lambda spec: next(
            (
                entry.template
                for record in stores.plans.records()
                if record.spec_id == spec.id and record.template_id
                for entry in [CATALOG.get(record.template_id)]
                if entry is not None
            ),
            None,
        ),
    )
    planner = ModelBackedPlanner(
        provider=FakeModelProvider(planner_replies),
        model="test-model",
        ledger=ledger,
        store=stores.plans,
        catalog=CATALOG,
    )
    return ResearchRuntime(
        config=RuntimeConfig(),
        director=PlanningDirector(plans=stores.plans),
        roles={
            RoleName.RESEARCH_ENGINEER: engineer,
            RoleName.RESEARCH_DIRECTOR: planner,
        },
        store=stores.evidence,
        metrics=stores.sink,
        usage=ledger,
        methodology_reviewer=PassMethodology(),
        control_source=lambda _spec: (OVERFIT_CONTROL,),
        verifications=stores.verifications,
    )


def _initial_state() -> ResearchState:
    return (
        ResearchState(
            objective=QUESTION.text,
            budget=ResearchBudget(
                wall_clock_seconds=3600.0, usd=10.0, model_tokens=200_000
            ),
        )
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(PREDICTION)
        .add_experiment(BASELINE_SPEC)
    )


def test_the_full_slice_plans_runs_verifies_and_stops(tmp_path: Path) -> None:
    stores = Stores(tmp_path)

    # Step 1 — the baseline runs and verifies: genuine admissible evidence.
    baseline = _runtime(stores, engineer_replies=(_engineer_reply(),))
    state = baseline.step(_initial_state()).state
    (evidence_id,) = state.evidence_ids
    result_id = stores.evidence.get_evidence(evidence_id).result_id
    verdict = stores.verifications.get(result_id)
    assert verdict is not None
    assert verdict.standing is OutcomeStanding.VERIFIED_EVIDENCE

    # Step 2 — nothing pending: the planner is invoked and its decision
    # commits hypothesis, prediction and experiment in one atomic bundle.
    planning = _runtime(stores, planner_replies=(_decision(evidence_id),))
    report = planning.step(state)
    state = report.state
    from autonomous_research_lab.core.actions import ResearchActionType

    attempt = next(
        a
        for a in state.attempts
        if a.action.action_type is ResearchActionType.PLAN_NEXT_ACTION
    )
    assert attempt.status is AttemptStatus.SUCCEEDED
    assert len(state.hypotheses) == 2
    assert len(state.predictions) == 2
    assert len(state.experiments) == 2
    (decision,) = stores.plans.records()
    assert decision.action is PlanningAction.NEW_EXPERIMENT
    assert decision.evidence_ids == (evidence_id,)
    planned_spec = state.experiment(decision.spec_id)
    assert planned_spec is not None
    assert planned_spec.estimated_cost == ResourceCost(wall_clock_seconds=60.0)

    # Step 3 — the planner's experiment traverses the engineer with the
    # planner-selected template.
    execution = _runtime(stores, engineer_replies=(_engineer_reply(),))
    state = execution.step(state).state
    planned_ref = next(
        ref for ref in state.results if ref.spec_id == decision.spec_id
    )
    assert planned_ref.status is ExperimentStatus.COMPLETED
    implementation = next(
        record
        for record in stores.implementations.records()
        if record.spec_id == decision.spec_id
    )
    assert implementation.template_id == PLANNED_TEMPLATE.id  # cross-check
    planned_verdict = stores.verifications.get(planned_ref.result_id)
    assert planned_verdict is not None
    assert planned_verdict.validity is ExperimentValidityStatus.VERIFIED
    assert stores.plans.is_dispatched(decision.id)

    # Step 4 — the second planner decision, over the updated state: stop.
    second = _runtime(stores, planner_replies=(_stop_decision(evidence_id),))
    state = second.step(state).state
    stop = next(
        record
        for record in stores.plans.records()
        if record.action is PlanningAction.STOP
    )
    assert stop.stop_reason is StopReason.QUESTION_RESOLVED
    assert len(state.experiments) == 2  # a stop commits no experiment

    # Step 5 — the stop reaches the loop's own halt path, typed reason
    # in the halt rationale.
    halting = _runtime(stores)
    final = halting.step(state)
    assert final.halt_reason is not None
    assert "planner stop: question_resolved" in final.halt_reason
    assert stop.id in final.halt_reason


def test_a_gate_rejected_decision_mutates_nothing_and_reaches_no_engineer(
    tmp_path: Path,
) -> None:
    stores = Stores(tmp_path)
    bad = _decision("ev_nonexistent")
    worse = _decision("ev_nonexistent", template_id="tmpl_forged")
    runtime = _runtime(stores, planner_replies=(bad, worse))

    state = (
        ResearchState(
            objective=QUESTION.text,
            budget=ResearchBudget(
                wall_clock_seconds=3600.0, usd=10.0, model_tokens=200_000
            ),
        )
        .upsert_question(QUESTION)
    )
    report = runtime.step(state)

    attempt = next(iter(report.state.attempts))
    assert attempt.status is AttemptStatus.FAILED
    assert report.state.hypotheses == ()
    assert report.state.predictions == ()
    assert report.state.experiments == ()
    assert stores.plans.records() == ()  # nothing accepted
    assert len(stores.plans.rejected()) == 2  # every attempt durable
    assert stores.implementations.records() == ()  # no engineer involvement
    assert any("failure" in note for note in report.notes)


def test_an_unaffordable_planning_step_halts_before_any_model_call(
    tmp_path: Path,
) -> None:
    stores = Stores(tmp_path)
    runtime = _runtime(stores, planner_replies=(_decision("ev_x"),))
    state = ResearchState(
        objective=QUESTION.text,
        budget=ResearchBudget(
            wall_clock_seconds=10.0, usd=0.01, model_tokens=100
        ),
    ).upsert_question(QUESTION)

    report = runtime.step(state)

    assert report.halt_reason is not None
    assert "insufficient budget" in report.halt_reason
    planner = runtime.roles[RoleName.RESEARCH_DIRECTOR]
    assert isinstance(planner, ModelBackedPlanner)
    assert stores.plans.records() == ()
    assert stores.plans.rejected() == ()
    metrics = stores.sink.records[-1]
    assert metrics.provider_calls == 0
