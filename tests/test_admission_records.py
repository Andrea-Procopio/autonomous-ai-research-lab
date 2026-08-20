"""The admission vocabulary: one named selection, one admitted seed.

These tests pin the structural claims: a directive names exactly one
selection run record; every identity is deterministic content
addressing; the operationalization quotes are non-empty and bounded;
the provenance split between inherited and operator-stated requirements
cannot be crossed; the record admits no numeric judgment field anywhere;
and the core prediction identity trap the encoding gate exists for is
documented as a test.
"""

from __future__ import annotations

import dataclasses

import pytest

from autonomous_research_lab.admission import (
    CLAIM_KINDS,
    MAX_ARM_CHARS,
    MAX_REQUIREMENT_CHARS,
    MECHANICAL_READING,
    AdmissionDirective,
    AdmissionRecord,
    GroundedSupport,
    OperationalPrediction,
    Requirement,
    RequirementSource,
    SupportSource,
)
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.mapping.records import CallProvenance


def _directive(**overrides: object) -> AdmissionDirective:
    values: dict[str, object] = {
        "selection_run_record_id": "srun_0000000000000001",
        "scheduling_requirement": "Batch-scheduled execution.",
        "job_duration_requirement": "Jobs bounded to two days.",
        "checkpoint_requirement": "Checkpoint and resume required.",
    }
    values.update(overrides)
    return AdmissionDirective(**values)  # type: ignore[arg-type]


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_0000000000000001",
        response_id="mcall_000000000000001",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id=None,
        latency_seconds=0.1,
        input_tokens=100,
        output_tokens=50,
        repair_count=0,
    )


def _support() -> GroundedSupport:
    return GroundedSupport(
        source=SupportSource.CANDIDATE,
        field_path="predictions[0].text",
        quote="ablating them drops accuracy",
    )


def _operational(**overrides: object) -> OperationalPrediction:
    values: dict[str, object] = {
        "prediction_text": "Ablating top heads drops accuracy more.",
        "condition": "Few-shot classification with trained scalars.",
        "base_metric": "few-shot accuracy",
        "expected_higher_arm": "accuracy drop from ablating top heads",
        "expected_lower_arm": "accuracy drop from ablating bottom heads",
        "contrary_observation": "similar drop for both ablations",
        "support": (_support(),),
    }
    values.update(overrides)
    return OperationalPrediction(**values)  # type: ignore[arg-type]


def _inherited() -> Requirement:
    return Requirement(
        source=RequirementSource.CANDIDATE_RESOURCES,
        record_id="idea_0000000000000001",
        field_path="resources.compute",
        quote="~100 GPU-hours on a single A100",
    )


def _operator() -> Requirement:
    return Requirement(
        source=RequirementSource.ADMISSION_DIRECTIVE,
        record_id="adir_0000000000000001",
        field_path="scheduling_requirement",
        quote="Batch-scheduled execution.",
    )


def _record(**overrides: object) -> AdmissionRecord:
    values: dict[str, object] = {
        "run_id": "adm_0000000000000001",
        "directive_id": "adir_0000000000000001",
        "selection_run_record_id": "srun_0000000000000001",
        "selection_run_id": "sel_0000000000000001",
        "selection_directive_id": "sdir_0000000000000001",
        "prior_art_run_record_id": "prun_0000000000000001",
        "prior_art_run_id": "pac_0000000000000001",
        "selected_prior_art_assessment_id": "paa_0000000000000001",
        "ideation_run_record_id": "irun_0000000000000001",
        "ideation_run_id": "idg_0000000000000001",
        "direction_id": "dir_0000000000000001",
        "snapshot_id": "cfp_0000000000000001",
        "map_run_id": "map_0000000000000001",
        "map_assessment_id": "madq_000000000000001",
        "selected_candidate_id": "idea_0000000000000001",
        "operational_predictions": (_operational(),),
        "measurements": ("few-shot accuracy", "prefix matching score"),
        "controls": ("random head ablation",),
        "comparison_targets": ("LoRA fine-tuning",),
        "evaluation_protocol": "Seeded runs across tasks and seeds.",
        "inherited_requirements": (_inherited(),),
        "operator_requirements": (_operator(),),
        "mechanical_reading": MECHANICAL_READING,
        "question_id": "q_00000000000000001",
        "hypothesis_id": "hyp_0000000000000001",
        "prediction_ids": ("pred_000000000000001",),
        "state_id": "st_0000000000000001",
        "provenance": _provenance(),
        "model_calls": 1,
        "input_tokens": 100,
        "output_tokens": 50,
    }
    values.update(overrides)
    return AdmissionRecord(**values)  # type: ignore[arg-type]


# -- directive ----------------------------------------------------------------


def test_a_directive_names_exactly_one_selection_run() -> None:
    with pytest.raises(ValueError, match="selection run record"):
        _directive(selection_run_record_id="   ")


def test_directive_identity_is_deterministic() -> None:
    assert _directive().id == _directive().id
    assert _directive().id.startswith("adir_")


def test_each_directive_field_perturbs_the_identity() -> None:
    base = _directive().id
    assert _directive(scheduling_requirement="Different.").id != base
    assert _directive(job_duration_requirement="Different.").id != base
    assert _directive(max_model_calls=3).id != base


def test_operator_statements_are_bounded() -> None:
    with pytest.raises(ValueError, match="operator fact"):
        _directive(checkpoint_requirement=" ")
    with pytest.raises(ValueError, match="short statement"):
        _directive(scheduling_requirement="x" * (MAX_REQUIREMENT_CHARS + 1))


def test_the_call_budget_is_bounded() -> None:
    with pytest.raises(ValueError, match="max_model_calls"):
        _directive(max_model_calls=0)
    with pytest.raises(ValueError, match="max_model_calls"):
        _directive(max_model_calls=5)


# -- claim kinds --------------------------------------------------------------


def test_claim_kinds_is_a_closed_map_of_known_labels() -> None:
    labels = set(CLAIM_KINDS.values())
    assert labels == {
        "record_quotation",
        "operational_interpretation",
        "record_restatement",
        "deterministic_copy",
    }
    # The one prose seat of model judgment is named as exactly that.
    assert (
        CLAIM_KINDS["operational_prediction.condition"]
        == "operational_interpretation"
    )
    # The importance copy is documented: cfp_alignment, not contribution.
    assert CLAIM_KINDS["question.importance"] == "deterministic_copy"


# -- value objects ------------------------------------------------------------


def test_support_requires_path_and_quote() -> None:
    with pytest.raises(ValueError, match="field path"):
        GroundedSupport(
            source=SupportSource.CANDIDATE, field_path=" ", quote="q"
        )
    with pytest.raises(ValueError, match="verbatim quote"):
        GroundedSupport(
            source=SupportSource.DIRECTION, field_path="scope", quote="  "
        )


def test_an_operationalization_requires_all_prose() -> None:
    for name in (
        "prediction_text",
        "condition",
        "base_metric",
        "expected_higher_arm",
        "expected_lower_arm",
        "contrary_observation",
    ):
        with pytest.raises(ValueError, match=name):
            _operational(**{name: "   "})


def test_arms_are_bounded_and_distinct() -> None:
    with pytest.raises(ValueError, match="at most"):
        _operational(expected_higher_arm="x" * (MAX_ARM_CHARS + 1))
    with pytest.raises(ValueError, match="arms must differ"):
        _operational(
            expected_higher_arm="the same arm",
            expected_lower_arm="  The  SAME arm ",
        )


def test_an_operationalization_carries_support() -> None:
    with pytest.raises(ValueError, match="grounded"):
        _operational(support=())


def test_a_requirement_is_self_contained_provenance() -> None:
    for name in ("record_id", "field_path", "quote"):
        values = {
            "source": RequirementSource.SELECTION_DIRECTIVE,
            "record_id": "sdir_0000000000000001",
            "field_path": "compute_constraint",
            "quote": "2-4 GPUs",
        }
        values[name] = "  "
        with pytest.raises(ValueError, match=name):
            Requirement(**values)  # type: ignore[arg-type]


# -- the admission record -----------------------------------------------------


def test_the_record_requires_at_least_one_encoding() -> None:
    with pytest.raises(ValueError, match="unfalsifiable"):
        _record(operational_predictions=(), prediction_ids=())


def test_prediction_ids_align_index_for_index() -> None:
    with pytest.raises(ValueError, match="index-for-index"):
        _record(prediction_ids=())
    with pytest.raises(ValueError, match="no duplicates"):
        _record(
            operational_predictions=(
                _operational(),
                _operational(base_metric="prefix matching score"),
            ),
            prediction_ids=(
                "pred_000000000000001",
                "pred_000000000000001",
            ),
        )
    with pytest.raises(ValueError, match="pred_ prefix"):
        _record(prediction_ids=("res_0000000000000001",))


def test_the_provenance_split_is_never_crossed() -> None:
    smuggled_in = Requirement(
        source=RequirementSource.ADMISSION_DIRECTIVE,
        record_id="adir_0000000000000001",
        field_path="scheduling_requirement",
        quote="Batch-scheduled execution.",
    )
    with pytest.raises(ValueError, match="never presented as inherited"):
        _record(inherited_requirements=(smuggled_in,))
    smuggled_out = Requirement(
        source=RequirementSource.CANDIDATE_RESOURCES,
        record_id="idea_0000000000000001",
        field_path="resources.compute",
        quote="~100 GPU-hours",
    )
    with pytest.raises(ValueError, match="operator-stated"):
        _record(operator_requirements=(smuggled_out,))


def test_the_mechanical_reading_is_pinned() -> None:
    with pytest.raises(ValueError, match="sign_only"):
        _record(mechanical_reading="effect_size")


def test_the_copied_surfaces_must_be_present() -> None:
    for name in ("measurements", "controls", "comparison_targets"):
        with pytest.raises(ValueError, match=r"cannot\s+be empty"):
            _record(**{name: ()})
        with pytest.raises(ValueError, match="non-empty"):
            _record(**{name: ("ok", "  ")})
    with pytest.raises(ValueError, match="evaluation protocol"):
        _record(evaluation_protocol="  ")


def test_the_created_ids_carry_their_prefixes() -> None:
    for name, bad in (
        ("run_id", "sel_0000000000000001"),
        ("question_id", "hyp_0000000000000001"),
        ("hypothesis_id", "q_00000000000000001"),
        ("state_id", "pred_000000000000001"),
    ):
        with pytest.raises(ValueError, match="prefix"):
            _record(**{name: bad})


def test_the_spend_shape_is_enforced() -> None:
    with pytest.raises(ValueError, match="at least the gated call"):
        _record(model_calls=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        _record(output_tokens=-1)


def test_record_identity_is_deterministic_content_addressing() -> None:
    assert _record().id == _record().id
    assert _record().id.startswith("arun_")
    doctored = _record(evaluation_protocol="A different protocol.")
    assert doctored.id != _record().id


def test_no_numeric_judgment_field_exists_anywhere() -> None:
    """Score-free is structural: no admission-defined dataclass carries
    a float or a score-shaped name. (CallProvenance's latency lives in
    mapping and measures the wire, not a judgment.)"""
    banned_names = ("score", "rank", "rating", "weight", "confidence")
    for kind in (
        AdmissionDirective,
        GroundedSupport,
        OperationalPrediction,
        Requirement,
        AdmissionRecord,
    ):
        for entry in dataclasses.fields(kind):
            assert "float" not in str(entry.type), (kind, entry.name)
            for banned in banned_names:
                assert banned not in entry.name, (kind, entry.name)


def test_two_core_predictions_differing_only_in_expectation_share_an_id() -> (
    None
):
    """The trap the encoding gate exists for: core Prediction's content
    id EXCLUDES the prose expectation, so two encodings with the same
    mechanical tuple silently collide. The gate therefore deduplicates
    on the mechanical tuple, never on the prose."""
    first = Prediction(
        hypothesis_id="hyp_0000000000000001",
        condition="c",
        metric="difference in accuracy: top minus bottom",
        comparator=Comparator.GREATER_THAN,
        threshold=0.0,
        expectation="one prose reading",
    )
    second = Prediction(
        hypothesis_id="hyp_0000000000000001",
        condition="c",
        metric="difference in accuracy: top minus bottom",
        comparator=Comparator.GREATER_THAN,
        threshold=0.0,
        expectation="a very different prose reading",
    )
    assert first.id == second.id
