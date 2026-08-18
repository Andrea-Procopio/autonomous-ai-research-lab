"""The deterministic planning gate and expansion, with no provider at all.

Every rule the task charter names is exercised as a pure function of the
typed projection, the trusted catalog, and one flat payload. The fixture
context holds a full baseline chain: one question, one hypothesis, one
prediction, one experiment with seeds (7, 11), one completed result at
seed 7, one admissible and one inadmissible piece of evidence, and a
remaining budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
from autonomous_research_lab.core.proposals import (
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
)
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.roles.base import RoleContext
from autonomous_research_lab.roles.engineer import ImplementationTemplate
from autonomous_research_lab.roles.planner import (
    PLANNING_SCHEMA,
    TemplateCapability,
    TemplateCatalog,
    check_decision,
    expand_decision,
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
    metrics={"test_accuracy": 0.93},
)
UNVERIFIED_EVIDENCE = Evidence(
    result_id="res_unverified0001",
    spec_id=BASELINE_SPEC.id,
    kind=EvidenceKind.MEASUREMENT,
    observation="an observation whose result never passed verification",
)

TEMPLATE_A = ImplementationTemplate(
    name="catalog-classification-v1", source="# classification template\n"
)
TEMPLATE_B = ImplementationTemplate(
    name="catalog-regression-v1", source="# regression template\n"
)
CATALOG = TemplateCatalog(
    entries=(
        TemplateCapability(
            template=TEMPLATE_A,
            metrics=(
                "test_accuracy",
                "tiny_subset_accuracy",
                "accuracy_drop",
            ),
            estimated_cost=ResourceCost(wall_clock_seconds=120.0),
            description="stdlib classification experiments",
        ),
        TemplateCapability(
            template=TEMPLATE_B,
            metrics=("test_mse", "noiseless_tiny_mse"),
            estimated_cost=ResourceCost(wall_clock_seconds=120.0),
            description="stdlib regression experiments",
        ),
    )
)


def _context(**overrides: Any) -> RoleContext:
    values: dict[str, Any] = {
        "objective": QUESTION.text,
        "questions": (QUESTION,),
        "hypotheses": (HYPOTHESIS,),
        "predictions": (PREDICTION,),
        "experiments": (BASELINE_SPEC,),
        "results": (RESULT,),
        "evidence": (GOOD_EVIDENCE, UNVERIFIED_EVIDENCE),
        "admissible_evidence_ids": (GOOD_EVIDENCE.id,),
        "remaining_budget": ResearchBudget(
            wall_clock_seconds=1000.0, usd=5.0, model_tokens=100_000
        ),
    }
    values.update(overrides)
    return RoleContext(**values)


def _decision(**overrides: Any) -> dict[str, Any]:
    """A gate-valid new-experiment decision; overrides carve out the rest."""
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
        "template_id": TEMPLATE_A.id,
        "target_experiment_id": "",
        "replication_seed": -1,
        "removed_component": "",
        "stop_reason": "none",
    }
    payload.update(overrides)
    return payload


def _replication(**overrides: Any) -> dict[str, Any]:
    payload = _decision(
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
    payload.update(overrides)
    return payload


def _stop(**overrides: Any) -> dict[str, Any]:
    payload = _replication(
        action="stop",
        target_experiment_id="",
        replication_seed=-1,
        stop_reason="question_resolved",
    )
    payload.update(overrides)
    return payload


def _rules(payload: dict[str, Any], **context_overrides: Any) -> set[str]:
    rejections = check_decision(
        payload, context=_context(**context_overrides), catalog=CATALOG
    )
    return {r.rule for r in rejections}


# -- accepted decisions ---------------------------------------------------------


def test_a_valid_new_experiment_passes_and_expands_in_chain_order() -> None:
    payload = _decision()
    assert _rules(payload) == set()

    expanded = expand_decision(
        payload, context=_context(), catalog=CATALOG, proposer="planner:test"
    )
    hypothesis, prediction, experiment = expanded.proposals
    assert isinstance(hypothesis, HypothesisProposal)
    assert isinstance(prediction, PredictionProposal)
    assert isinstance(experiment, ExperimentProposal)
    assert prediction.prediction.hypothesis_id == hypothesis.hypothesis.id
    assert experiment.spec.prediction_id == prediction.prediction.id
    # The cost is stamped from the trusted catalog, never model-authored.
    assert experiment.spec.estimated_cost == ResourceCost(
        wall_clock_seconds=120.0
    )
    assert expanded.spec_id == experiment.spec.id


def test_a_valid_replication_names_the_next_unused_seed() -> None:
    payload = _replication()
    assert _rules(payload) == set()

    expanded = expand_decision(
        payload, context=_context(), catalog=CATALOG, proposer="planner:test"
    )
    assert expanded.proposals == ()  # nothing new to commit
    assert expanded.spec_id == BASELINE_SPEC.id
    assert expanded.prediction_id == PREDICTION.id


def test_a_stop_produces_no_proposition_and_no_experiment() -> None:
    payload = _stop()
    assert _rules(payload) == set()
    expanded = expand_decision(
        payload, context=_context(), catalog=CATALOG, proposer="planner:test"
    )
    assert expanded.proposals == ()
    assert expanded.spec_id == ""


def test_reusing_an_existing_hypothesis_is_a_two_proposal_chain() -> None:
    payload = _decision(
        hypothesis_id=HYPOTHESIS.id, hypothesis_statement=""
    )
    assert _rules(payload) == set()
    expanded = expand_decision(
        payload, context=_context(), catalog=CATALOG, proposer="planner:test"
    )
    kinds = tuple(type(p).__name__ for p in expanded.proposals)
    assert kinds == ("PredictionProposal", "ExperimentProposal")
    assert expanded.hypothesis_id == HYPOTHESIS.id


# -- reference integrity ----------------------------------------------------------


def test_unknown_references_are_each_named() -> None:
    assert "unknown_question" in _rules(_decision(question_id="q_missing"))
    assert "unknown_hypothesis" in _rules(
        _decision(hypothesis_id="hyp_missing", hypothesis_statement="")
    )
    assert "unknown_evidence" in _rules(
        _decision(evidence_ids=["ev_missing"])
    )
    assert "unknown_target_experiment" in _rules(
        _replication(target_experiment_id="exp_missing")
    )


def test_a_hypothesis_answering_a_different_question_is_inconsistent() -> None:
    other = ResearchQuestion(text="a second question", importance="side")
    rules = _rules(
        _decision(
            question_id=other.id, hypothesis_id=HYPOTHESIS.id,
            hypothesis_statement="",
        ),
        questions=(QUESTION, other),
    )
    assert "inconsistent_chain" in rules


def test_inadmissible_evidence_cannot_ground_a_decision() -> None:
    assert "inadmissible_evidence_cited" in _rules(
        _decision(evidence_ids=[UNVERIFIED_EVIDENCE.id])
    )
    assert "inadmissible_evidence_cited" in _rules(_decision(evidence_ids=[]))


# -- templates and metrics ---------------------------------------------------------


def test_templates_outside_the_catalog_are_unsupported() -> None:
    assert "unknown_template" in _rules(_decision(template_id="tmpl_forged"))


def test_metrics_a_template_cannot_measure_are_rejected() -> None:
    assert "undeclared_metric" in _rules(
        _decision(experiment_metrics=["accuracy_drop", "test_mse"])
    )


def test_the_prediction_metric_must_be_declared_by_the_experiment() -> None:
    assert "undeclared_metric" in _rules(
        _decision(prediction_metric="tiny_subset_accuracy",
                  experiment_metrics=["accuracy_drop"])
    )


# -- falsifiability -----------------------------------------------------------------


def test_unfalsifiable_predictions_are_rejected() -> None:
    assert "unfalsifiable_prediction" in _rules(
        _decision(prediction_comparator="none")
    )
    assert "unfalsifiable_prediction" in _rules(
        _decision(prediction_metric="")
    )
    assert "unfalsifiable_prediction" in _rules(
        _decision(prediction_condition="")
    )
    assert "unfalsifiable_prediction" in _rules(
        _decision(prediction_comparator="approx", prediction_tolerance=0)
    )


# -- duplicates, seeds, replication ---------------------------------------------------


def test_a_duplicate_experiment_is_rejected_by_content_identity() -> None:
    duplicate = _decision(
        hypothesis_id=HYPOTHESIS.id,
        hypothesis_statement="",
        prediction_condition=PREDICTION.condition,
        prediction_metric=PREDICTION.metric,
        prediction_comparator="ge",
        prediction_threshold=0.9,
        prediction_expectation="",
        experiment_objective=BASELINE_SPEC.objective,
        experiment_procedure=BASELINE_SPEC.procedure,
        experiment_metrics=list(BASELINE_SPEC.metrics),
        experiment_seeds=list(BASELINE_SPEC.seeds),
    )
    assert "duplicate_experiment" in _rules(duplicate)


def test_replication_seed_rules_mirror_the_engineer() -> None:
    assert _rules(_replication(replication_seed=11)) == set()  # next unused
    assert "seed_already_used" in _rules(_replication(replication_seed=7))
    assert "seed_policy_mismatch" in _rules(
        _replication(replication_seed=99)  # not declared
    )


def test_replication_with_every_seed_used_is_rejected() -> None:
    second = ExperimentResult(
        spec_id=BASELINE_SPEC.id,
        job_id="job-fixture-2",
        status=ExperimentStatus.COMPLETED,
        command=("python", "experiment.py"),
        environment=RESULT.environment,
        metrics={"test_accuracy": 0.91, "tiny_subset_accuracy": 1.0},
        seed=11,
    )
    rules = _rules(
        _replication(replication_seed=11), results=(RESULT, second)
    )
    assert "seed_already_used" in rules


def test_an_experiment_without_seeds_is_rejected() -> None:
    assert "inconsistent_chain" in _rules(_decision(experiment_seeds=[]))


# -- ablations -----------------------------------------------------------------------


def _ablation(**overrides: Any) -> dict[str, Any]:
    payload = _decision(
        action="ablation",
        hypothesis_id=HYPOTHESIS.id,
        hypothesis_statement="",
        prediction_condition="without standardization",
        prediction_metric="test_accuracy",
        prediction_comparator="lt",
        prediction_threshold=0.9,
        experiment_objective="measure accuracy without standardization",
        experiment_procedure=(
            "train knn on raw synthetic blobs, skipping the standardized "
            "step, and score it"
        ),
        experiment_metrics=["test_accuracy", "tiny_subset_accuracy"],
        experiment_seeds=[7],
        target_experiment_id=BASELINE_SPEC.id,
        removed_component="standardized",
    )
    payload.update(overrides)
    return payload


def test_a_valid_ablation_names_its_parent_in_the_record() -> None:
    payload = _ablation()
    assert _rules(payload) == set()
    expanded = expand_decision(
        payload, context=_context(), catalog=CATALOG, proposer="planner:test"
    )
    experiment = expanded.proposals[-1]
    assert isinstance(experiment, ExperimentProposal)
    assert any(
        "ablation of" in baseline and BASELINE_SPEC.id in baseline
        for baseline in experiment.spec.baselines
    )


def test_invalid_ablations_are_rejected() -> None:
    assert "invalid_ablation_parent" in _rules(
        _ablation(target_experiment_id="exp_missing")
    )
    assert "unnamed_removed_component" in _rules(
        _ablation(removed_component="")
    )
    assert "unnamed_removed_component" in _rules(
        _ablation(removed_component="a component the parent never had")
    )


# -- sentinel discipline ---------------------------------------------------------


def test_a_stop_may_hide_no_experiment() -> None:
    smuggled = _stop(
        experiment_objective="a hidden experiment",
        experiment_metrics=["test_accuracy"],
    )
    assert "inapplicable_field" in _rules(smuggled)


def test_a_replication_may_smuggle_no_new_propositions() -> None:
    assert "inapplicable_field" in _rules(
        _replication(hypothesis_statement="a new idea on the side")
    )


def test_a_new_experiment_may_not_carry_replication_fields() -> None:
    assert "inapplicable_field" in _rules(_decision(replication_seed=13))
    assert "inapplicable_field" in _rules(
        _decision(target_experiment_id=BASELINE_SPEC.id)
    )
    assert "inapplicable_field" in _rules(
        _decision(stop_reason="question_resolved")
    )


def test_a_stop_requires_its_typed_reason() -> None:
    assert "inconsistent_chain" in _rules(_stop(stop_reason="none"))


# -- budget --------------------------------------------------------------------------


def test_budget_violations_are_rejected() -> None:
    broke = ResearchBudget(wall_clock_seconds=10.0, usd=0.1, model_tokens=100)
    assert "budget_insufficient" in _rules(
        _decision(), remaining_budget=broke
    )
    assert "budget_insufficient" in _rules(
        _replication(), remaining_budget=broke
    )


# -- the schema surface ----------------------------------------------------------------


def test_the_schema_has_no_slot_for_results_or_infrastructure() -> None:
    """Most of the planner's must-not list is enforced by absence: no
    property could carry an observed value, a command, a path, a
    dependency, or a container setting."""
    declared = PLANNING_SCHEMA.json_schema["properties"]
    assert isinstance(declared, Mapping)
    properties = set(declared)
    assert properties == {
        "action", "question_id", "rationale", "evidence_ids",
        "hypothesis_id", "hypothesis_statement",
        "prediction_condition", "prediction_metric", "prediction_comparator",
        "prediction_threshold", "prediction_tolerance",
        "prediction_expectation",
        "experiment_objective", "experiment_procedure", "experiment_metrics",
        "experiment_baselines", "experiment_controls", "experiment_seeds",
        "template_id", "target_experiment_id", "replication_seed",
        "removed_component", "stop_reason",
    }
    # Closed on the wire: _normalize_closed ran at construction.
    assert PLANNING_SCHEMA.json_schema["additionalProperties"] is False
