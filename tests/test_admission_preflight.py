"""The admission preflight: an incoherent directive refuses before any
call, naming every violation at once."""

from __future__ import annotations

import pytest

from autonomous_research_lab.admission import (
    AdmissionDirective,
    AdmissionPreflightError,
    check_admission_coherence,
)
from autonomous_research_lab.admission.records import (
    MAX_ENCODINGS_PER_PREDICTION,
)


def _directive(**overrides: object) -> AdmissionDirective:
    values: dict[str, object] = {
        "selection_run_record_id": "srun_0000000000000001",
        "scheduling_requirement": "Batch-scheduled execution.",
        "job_duration_requirement": "Jobs bounded to two days.",
        "checkpoint_requirement": "Checkpoint and resume required.",
    }
    values.update(overrides)
    return AdmissionDirective(**values)  # type: ignore[arg-type]


def test_the_default_wiring_is_coherent_for_the_live_shape() -> None:
    plan = check_admission_coherence(
        directive=_directive(),
        prediction_count=1,
        max_output_tokens=16_384,
        max_corrective_calls=1,
    )
    assert plan.worst_case_calls == 2
    assert plan.max_encodings == MAX_ENCODINGS_PER_PREDICTION
    assert plan.worst_case_output_tokens <= 16_384


def test_a_call_budget_the_stage_cannot_fit_refuses() -> None:
    with pytest.raises(AdmissionPreflightError, match="allows 1"):
        check_admission_coherence(
            directive=_directive(max_model_calls=1),
            prediction_count=1,
            max_output_tokens=16_384,
            max_corrective_calls=1,
        )


def test_a_reply_window_the_encodings_cannot_fit_refuses() -> None:
    with pytest.raises(AdmissionPreflightError, match="output tokens"):
        check_admission_coherence(
            directive=_directive(),
            prediction_count=50,
            max_output_tokens=16_384,
            max_corrective_calls=1,
        )


def test_a_candidate_without_predictions_refuses() -> None:
    with pytest.raises(AdmissionPreflightError, match="nothing admissible"):
        check_admission_coherence(
            directive=_directive(),
            prediction_count=0,
            max_output_tokens=16_384,
            max_corrective_calls=1,
        )


def test_every_violation_is_named_in_one_error() -> None:
    with pytest.raises(AdmissionPreflightError) as caught:
        check_admission_coherence(
            directive=_directive(max_model_calls=1),
            prediction_count=50,
            max_output_tokens=16_384,
            max_corrective_calls=1,
        )
    message = str(caught.value)
    assert "allows 1" in message
    assert "output tokens" in message


def test_the_plan_states_what_it_reserved() -> None:
    plan = check_admission_coherence(
        directive=_directive(),
        prediction_count=2,
        max_output_tokens=16_384,
        max_corrective_calls=1,
    )
    assert plan.prediction_count == 2
    assert plan.max_encodings == 2 * MAX_ENCODINGS_PER_PREDICTION
    assert plan.worst_case_output_tokens == 500 + plan.max_encodings * 450
