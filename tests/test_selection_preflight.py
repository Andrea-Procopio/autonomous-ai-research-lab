"""The selection preflight: a directive that cannot finish the work its
own settings allow is refused before any call, every violation named."""

from __future__ import annotations

import pytest

from autonomous_research_lab.selection.directive import SelectionDirective
from autonomous_research_lab.selection.preflight import (
    GATED_STAGES,
    STAGE1_BASE_OUTPUT_TOKENS,
    STAGE1_TOKENS_PER_CANDIDATE,
    STAGE1_TOKENS_PER_PAIR,
    SelectionPreflightError,
    check_selection_coherence,
)

ENVELOPE = 16384


def _directive(**overrides: object) -> SelectionDirective:
    values: dict[str, object] = {
        "prior_art_run_record_id": "prun_1",
        "compute_constraint": "One CPU workstation.",
        "data_constraint": "Public datasets only.",
        "time_constraint": "Runs finish within hours.",
        "experimental_constraint": "Containerized seeded runs.",
    }
    values.update(overrides)
    return SelectionDirective(**values)  # type: ignore[arg-type]


def test_the_default_directive_is_coherent_for_three_eligible() -> None:
    plan = check_selection_coherence(
        directive=_directive(),
        eligible_count=3,
        max_output_tokens=ENVELOPE,
        max_corrective_calls=1,
    )
    assert plan.pairs == 3
    assert plan.worst_stage1_output_tokens == (
        STAGE1_BASE_OUTPUT_TOKENS
        + 3 * STAGE1_TOKENS_PER_CANDIDATE
        + 3 * STAGE1_TOKENS_PER_PAIR
    )
    assert plan.worst_stage1_output_tokens <= ENVELOPE
    assert plan.worst_calls_total == GATED_STAGES * 2


def test_the_ceiling_of_six_still_fits_the_proven_envelope() -> None:
    plan = check_selection_coherence(
        directive=_directive(max_eligible_candidates=6),
        eligible_count=6,
        max_output_tokens=ENVELOPE,
        max_corrective_calls=1,
    )
    assert plan.worst_stage1_output_tokens <= ENVELOPE


def test_the_pairwise_review_must_fit_its_reply_budget() -> None:
    with pytest.raises(SelectionPreflightError, match="comparative review"):
        check_selection_coherence(
            directive=_directive(),
            eligible_count=3,
            max_output_tokens=4096,
            max_corrective_calls=1,
        )


def test_an_oversized_eligible_set_is_refused_before_any_call() -> None:
    with pytest.raises(SelectionPreflightError, match="exceed the"):
        check_selection_coherence(
            directive=_directive(max_eligible_candidates=2),
            eligible_count=3,
            max_output_tokens=ENVELOPE,
            max_corrective_calls=1,
        )


def test_the_call_budget_must_cover_both_stages_and_correctives() -> None:
    with pytest.raises(SelectionPreflightError, match="worst-case calls"):
        check_selection_coherence(
            directive=_directive(max_model_calls=3),
            eligible_count=2,
            max_output_tokens=ENVELOPE,
            max_corrective_calls=1,
        )


def test_every_violation_is_collected_in_one_refusal() -> None:
    with pytest.raises(SelectionPreflightError) as caught:
        check_selection_coherence(
            directive=_directive(
                max_eligible_candidates=2, max_model_calls=2
            ),
            eligible_count=5,
            max_output_tokens=4096,
            max_corrective_calls=1,
        )
    message = str(caught.value)
    assert "exceed the directive's cap" in message
    assert "comparative review" in message
    assert "worst-case calls" in message


def test_a_coherent_plan_states_its_reserved_calls() -> None:
    plan = check_selection_coherence(
        directive=_directive(),
        eligible_count=1,
        max_output_tokens=ENVELOPE,
        max_corrective_calls=1,
    )
    assert plan.eligible == 1
    assert plan.pairs == 0
    assert plan.worst_calls_total == 4
    assert plan.output_token_envelope == ENVELOPE
