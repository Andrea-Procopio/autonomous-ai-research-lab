"""The model-backed planner, deterministically.

Everything here runs on :class:`FakeModelProvider`. The invariants under
test: one decision per invocation with full provenance in the planning
store; gate rejections are preserved and earn exactly one corrective call;
provider accounting reaches the ledger exactly once on success and on
failure; replicate and stop decisions commit nothing; and the rendered
request is a pure function of the projection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autonomous_research_lab.core.actions import (
    ResearchAction,
    ResearchActionType,
)
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.proposals import ProposalKind, kind_of
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.roles.base import RoleContext, RoleInvocation, RoleName
from autonomous_research_lab.roles.engineer import ImplementationTemplate
from autonomous_research_lab.roles.planner import (
    ModelBackedPlanner,
    PlannerContractError,
    PlanningRejectedError,
    TemplateCapability,
    TemplateCatalog,
)
from autonomous_research_lab.runtime.metrics import ProviderUsage
from autonomous_research_lab.runtime.planning_store import (
    PlanningAction,
    PlanningStore,
    StopReason,
)
from autonomous_research_lab.runtime.providers import (
    CallAccounting,
    FakeModelProvider,
    ProviderTransportError,
    ScriptedReply,
    UsageLedger,
)

QUESTION = ResearchQuestion(text="does scaling help?", importance="core")
HYPOTHESIS = Hypothesis(
    statement="standardizing features improves accuracy",
    rationale="distance metrics are scale-sensitive",
    question_id=QUESTION.id,
)
PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="on the synthetic blobs data",
    metric="test_accuracy",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.9,
)
BASELINE_SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="measure baseline accuracy",
    procedure="train knn on standardized synthetic blobs and score it",
    metrics=("test_accuracy", "tiny_subset_accuracy"),
    baselines=("chance accuracy 0.5",),
    controls=("tiny subset must be memorized",),
    seeds=(7, 11),
    estimated_cost=ResourceCost(wall_clock_seconds=120.0),
)
RESULT = ExperimentResult(
    spec_id=BASELINE_SPEC.id,
    job_id="job-fixture-1",
    status=ExperimentStatus.COMPLETED,
    command=("python", "experiment.py"),
    environment=Environment(
        python_version="3.11", platform="test", git_commit="", git_dirty=False
    ),
    metrics={"test_accuracy": 0.93, "tiny_subset_accuracy": 1.0},
    seed=7,
)
GOOD_EVIDENCE = Evidence(
    result_id=RESULT.id,
    spec_id=BASELINE_SPEC.id,
    kind=EvidenceKind.MEASUREMENT,
    observation="test_accuracy 0.93 at seed 7",
)

TEMPLATE = ImplementationTemplate(
    name="catalog-classification-v1", source="# classification template\n"
)
CATALOG = TemplateCatalog(
    entries=(
        TemplateCapability(
            template=TEMPLATE,
            metrics=("test_accuracy", "tiny_subset_accuracy", "accuracy_drop"),
            estimated_cost=ResourceCost(wall_clock_seconds=120.0),
        ),
    )
)


def _context() -> RoleContext:
    return RoleContext(
        objective=QUESTION.text,
        questions=(QUESTION,),
        hypotheses=(HYPOTHESIS,),
        predictions=(PREDICTION,),
        experiments=(BASELINE_SPEC,),
        results=(RESULT,),
        evidence=(GOOD_EVIDENCE,),
        admissible_evidence_ids=(GOOD_EVIDENCE.id,),
        remaining_budget=ResearchBudget(
            wall_clock_seconds=1000.0, usd=5.0, model_tokens=100_000
        ),
    )


def _invocation(context: RoleContext | None = None) -> RoleInvocation:
    return RoleInvocation(
        role=RoleName.RESEARCH_DIRECTOR,
        assignment=ResearchAction(
            action_type=ResearchActionType.PLAN_NEXT_ACTION,
            rationale="select the next scientific action",
        ),
        context=context if context is not None else _context(),
        allowed_actions=frozenset({ResearchActionType.PLAN_NEXT_ACTION}),
        expected_output=frozenset(
            {
                ProposalKind.HYPOTHESIS,
                ProposalKind.PREDICTION,
                ProposalKind.EXPERIMENT,
            }
        ),
    )


def _decision(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": "new_experiment",
        "question_id": QUESTION.id,
        "rationale": "the baseline is verified; test robustness next",
        "evidence_ids": [GOOD_EVIDENCE.id],
        "hypothesis_id": "",
        "hypothesis_statement": (
            "accuracy survives moderate label noise on this data"
        ),
        "prediction_condition": "with 10 percent label noise",
        "prediction_metric": "accuracy_drop",
        "prediction_comparator": "le",
        "prediction_threshold": 0.15,
        "prediction_tolerance": 0,
        "prediction_expectation": "the drop stays small",
        "experiment_objective": "measure accuracy under label noise",
        "experiment_procedure": (
            "train on noisy labels and compare against the clean score"
        ),
        "experiment_metrics": ["accuracy_drop", "tiny_subset_accuracy"],
        "experiment_baselines": ["clean-label accuracy from the baseline"],
        "experiment_controls": ["tiny subset must be memorized"],
        "experiment_seeds": [13],
        "template_id": TEMPLATE.id,
        "target_experiment_id": "",
        "replication_seed": -1,
        "removed_component": "",
        "stop_reason": "none",
    }
    payload.update(overrides)
    return payload


def _reply(**overrides: Any) -> str:
    return json.dumps(_decision(**overrides))


def _planner(
    tmp_path: Path,
    replies: tuple[ScriptedReply | str, ...],
    *,
    repairs: int = 1,
) -> tuple[ModelBackedPlanner, FakeModelProvider, UsageLedger, PlanningStore]:
    provider = FakeModelProvider(replies)
    ledger = UsageLedger()
    store = PlanningStore(tmp_path / "planning")
    planner = ModelBackedPlanner(
        provider=provider,
        model="test-model",
        ledger=ledger,
        store=store,
        catalog=CATALOG,
        max_corrective_calls=repairs,
    )
    return planner, provider, ledger, store


# -- accepted decisions -----------------------------------------------------------


def test_a_valid_decision_yields_proposals_and_a_provenance_record(
    tmp_path: Path,
) -> None:
    planner, provider, ledger, store = _planner(tmp_path, (_reply(),))
    invocation = _invocation()

    proposals = planner.perform(invocation)

    kinds = tuple(kind_of(p) for p in proposals)
    assert kinds == (
        ProposalKind.HYPOTHESIS,
        ProposalKind.PREDICTION,
        ProposalKind.EXPERIMENT,
    )
    assert all(invocation.permits(p) for p in proposals)

    (record,) = store.records()
    assert record.action is PlanningAction.NEW_EXPERIMENT
    assert record.invocation_id == invocation.id
    assert record.question_id == QUESTION.id
    assert record.evidence_ids == (GOOD_EVIDENCE.id,)
    assert record.template_id == TEMPLATE.id
    assert record.repair_count == 0
    assert record.request_fingerprint == provider.calls[0].fingerprint
    assert record.response_id.startswith("mcall_")
    assert record.provider == "fake"
    assert record.served_model == "test-model"
    assert record.latency_seconds >= 0.0
    assert record.input_tokens > 0 and record.output_tokens > 0
    assert record.nominal_cost_usd is None  # unknown, never invented

    drained = ledger.drain()
    assert drained.calls == 1  # exactly one success, recorded once


def test_the_request_is_a_pure_function_of_the_projection(
    tmp_path: Path,
) -> None:
    """Identical authoritative projection, identical request fingerprint —
    the planner reads state, never conversation history."""
    first, first_provider, _, _ = _planner(tmp_path / "a", (_reply(),))
    second, second_provider, _, _ = _planner(tmp_path / "b", (_reply(),))

    first.perform(_invocation(_context()))
    second.perform(_invocation(_context()))

    assert (
        first_provider.calls[0].fingerprint
        == second_provider.calls[0].fingerprint
    )


def test_a_replicate_decision_returns_no_proposals(tmp_path: Path) -> None:
    reply = _reply(
        action="replicate",
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
        target_experiment_id=BASELINE_SPEC.id,
        replication_seed=11,
    )
    planner, _, _, store = _planner(tmp_path, (reply,))

    proposals = planner.perform(_invocation())

    assert proposals == ()
    (record,) = store.records()
    assert record.action is PlanningAction.REPLICATE
    assert record.spec_id == BASELINE_SPEC.id
    assert record.replication_seed == 11


def test_a_stop_decision_commits_nothing_and_types_its_reason(
    tmp_path: Path,
) -> None:
    reply = _reply(
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
    )
    planner, _, _, store = _planner(tmp_path, (reply,))

    proposals = planner.perform(_invocation())

    assert proposals == ()
    (record,) = store.records()
    assert record.action is PlanningAction.STOP
    assert record.stop_reason is StopReason.QUESTION_RESOLVED
    assert record.spec_id == ""  # no hidden experiment


# -- the corrective call ------------------------------------------------------------


def test_a_gate_rejected_decision_earns_exactly_one_corrective_call(
    tmp_path: Path,
) -> None:
    bad = _reply(template_id="tmpl_forged")
    planner, provider, ledger, store = _planner(tmp_path, (bad, _reply()))

    proposals = planner.perform(_invocation())

    assert len(proposals) == 3
    assert len(provider.calls) == 2
    repair = provider.calls[1]
    assert repair.metadata["planning_repair"] == "1"
    assert "unknown_template" in repair.messages[-1].content
    (rejected,) = store.rejected()
    reasons = rejected["reasons"]
    assert isinstance(reasons, list)
    assert reasons[0]["rule"] == "unknown_template"
    (record,) = store.records()
    assert record.repair_count == 1
    assert ledger.drain().calls == 2  # both calls billed, each once


def test_repair_is_bounded_and_every_attempt_is_preserved(
    tmp_path: Path,
) -> None:
    bad = _reply(template_id="tmpl_forged")
    worse = _reply(evidence_ids=["ev_missing"])
    planner, provider, ledger, store = _planner(tmp_path, (bad, worse))

    with pytest.raises(PlanningRejectedError, match="unknown_evidence"):
        planner.perform(_invocation())

    assert len(provider.calls) == 2  # never a third call
    assert len(store.rejected()) == 2
    assert store.records() == ()  # nothing accepted, nothing committed
    assert ledger.drain().calls == 2


def test_scientific_disagreement_never_triggers_repair(tmp_path: Path) -> None:
    """A gate-valid decision that predicts a *negative* outcome for the
    standing hypothesis commits on the first call — there is no code path
    from 'valid but unwelcome' to a corrective call."""
    contrarian = _reply(
        hypothesis_statement=(
            "the baseline gain disappears under label noise"
        ),
        prediction_comparator="ge",
        prediction_threshold=0.5,
        rationale="the verified baseline may be an artifact of clean labels",
    )
    planner, provider, _, store = _planner(tmp_path, (contrarian,))

    proposals = planner.perform(_invocation())

    assert len(provider.calls) == 1
    assert len(proposals) == 3
    (record,) = store.records()
    assert record.repair_count == 0


# -- provider failures ---------------------------------------------------------------


def test_a_provider_failure_is_billed_once_and_leaves_no_record(
    tmp_path: Path,
) -> None:
    error = ProviderTransportError("the endpoint is down", status_code=503)
    error.with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=42, output_tokens=0, model="test-model"
            ),
            latency_seconds=0.5,
            request_id="req-1",
            model="test-model",
        )
    )
    planner, _, ledger, store = _planner(
        tmp_path, (ScriptedReply(text="", error=error),)
    )

    with pytest.raises(ProviderTransportError):
        planner.perform(_invocation())

    drained = ledger.drain()
    assert drained.calls == 1
    assert drained.input_tokens == 42  # the billed failure, exactly once
    assert store.records() == ()
    assert store.rejected() == ()


# -- contract errors -------------------------------------------------------------------


def test_an_unsupported_action_raises_before_any_model_call(
    tmp_path: Path,
) -> None:
    planner, provider, _, _ = _planner(tmp_path, (_reply(),))
    invocation = RoleInvocation(
        role=RoleName.RESEARCH_DIRECTOR,
        assignment=ResearchAction(
            action_type=ResearchActionType.GENERATE_HYPOTHESIS,
            rationale="not planning work",
        ),
        context=_context(),
        allowed_actions=frozenset({ResearchActionType.GENERATE_HYPOTHESIS}),
        expected_output=frozenset({ProposalKind.HYPOTHESIS}),
    )
    with pytest.raises(PlannerContractError, match="plan_next_action"):
        planner.perform(invocation)
    assert provider.calls == ()


def test_a_context_without_a_question_raises_before_any_model_call(
    tmp_path: Path,
) -> None:
    planner, provider, _, _ = _planner(tmp_path, (_reply(),))
    with pytest.raises(PlannerContractError, match="research question"):
        planner.perform(_invocation(RoleContext(objective="empty")))
    assert provider.calls == ()
