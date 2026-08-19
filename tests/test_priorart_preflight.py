"""The budget-coherence preflight: a directive that cannot complete its
own promised work is refused before the first call, with every violated
inequality named together."""

from __future__ import annotations

import pytest

from autonomous_research_lab.ideation.records import (
    AddressedProblem,
    CandidateIdea,
    DataRequirement,
    DataStatus,
    Prediction,
    ResourceEstimate,
    TargetedTheme,
    problem_key,
    theme_key,
)
from autonomous_research_lab.mapping.adequacy import SupportTier
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    ProblemKind,
    ThemeEra,
)
from autonomous_research_lab.priorart.assessment import PriorArtThresholds
from autonomous_research_lab.priorart.directive import PriorArtDirective
from autonomous_research_lab.priorart.preflight import (
    PriorArtPreflightError,
    check_budget_coherence,
)


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_0",
        response_id="mcall_0",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id=None,
        latency_seconds=0.25,
        input_tokens=10,
        output_tokens=10,
        repair_count=0,
    )


def _candidate(cited: int) -> CandidateIdea:
    statement = "head-level mechanisms of in-context learning are untested"
    return CandidateIdea(
        run_id="idg_1",
        title="Do Reweighting Scalars Select Induction Heads?",
        research_question="Does head reweighting select induction heads?",
        proposed_contribution="a causal test of head reweighting",
        mechanism="reweighting amplifies specialized induction heads",
        hypothesis="reweighted heads carry the in-context ability",
        grounding="the cited records report head specialization",
        predictions=(
            Prediction(
                text="ablating reweighted heads drops accuracy",
                falsifier="accuracy unchanged after ablation",
            ),
        ),
        datasets=(
            DataRequirement(
                name="synthetic sequences",
                status=DataStatus.NEW_REQUIREMENT,
                role="probe tasks",
            ),
        ),
        metrics=("accuracy",),
        evaluation_protocol="held-out probes",
        baselines=("dense fine-tuning",),
        ablations=("random head subsets",),
        resources=ResourceEstimate(
            compute="one GPU-day", data="synthetic", implementation="small"
        ),
        risks=("effects may not localize",),
        cfp_alignment="matches the adaptation topic",
        aligned_topics=("adaptation",),
        uncertainty="the mechanism may be diffuse",
        search_terms=("attention head reweighting",),
        addressed_problems=(
            AddressedProblem(
                key=problem_key(statement),
                statement=statement,
                kind=ProblemKind.OPEN_PROBLEM,
                tier=SupportTier.TENTATIVE,
            ),
        ),
        targeted_themes=(
            TargetedTheme(
                key=theme_key("adaptation"),
                name="adaptation",
                era=ThemeEra.RECENT,
            ),
        ),
        cited_source_ids=tuple(f"lit_c{index}" for index in range(cited)),
        cited_recent=cited,
        cited_foundational=0,
        cited_undated=0,
        provenance=_provenance(),
    )


def _directive(**overrides: object) -> PriorArtDirective:
    defaults: dict[str, object] = {
        "ideation_run_record_id": "irun_1",
        "cutoff_date": "2026-08-18",
        "recent_window_start": "2025-08-18",
    }
    defaults.update(overrides)
    return PriorArtDirective(**defaults)  # type: ignore[arg-type]


def _check(
    *,
    directive: PriorArtDirective | None = None,
    candidates: int = 3,
    cited: int = 5,
    thresholds: PriorArtThresholds | None = None,
    screening_batch_size: int = 12,
    max_corrective_calls: int = 1,
) -> object:
    return check_budget_coherence(
        directive=directive or _directive(),
        candidates=tuple(_candidate(cited) for _ in range(candidates)),
        thresholds=thresholds or PriorArtThresholds(),
        screening_batch_size=screening_batch_size,
        max_corrective_calls=max_corrective_calls,
    )


def test_the_default_directive_is_coherent_for_three_candidates() -> None:
    plan = check_budget_coherence(
        directive=_directive(),
        candidates=tuple(_candidate(5) for _ in range(3)),
        thresholds=PriorArtThresholds(),
        screening_batch_size=12,
        max_corrective_calls=1,
    )
    assert plan.worst_pool_per_candidate == 35
    assert plan.screening_capacity == 35
    assert plan.worst_screening_calls_per_candidate == 4
    assert plan.worst_calls_per_candidate == 12
    assert plan.worst_calls_total == 36


def test_retrieval_may_not_exceed_the_screening_cap() -> None:
    # Six families at five results plus six cited works can pool 36
    # against the cap of 35: the Task 5D.1 mechanical-truncation shape,
    # now inexpressible as an executed run.
    with pytest.raises(PriorArtPreflightError, match="mechanically"):
        _check(cited=6)
    _check(cited=5)


def test_comparison_must_be_reachable() -> None:
    with pytest.raises(PriorArtPreflightError, match="min_compared_works"):
        _check(
            directive=_directive(max_compared_works=1),
            thresholds=PriorArtThresholds(min_compared_works=2),
        )


def test_the_source_threshold_must_be_screenable() -> None:
    # A pool that clears min_unique_sources must fit the screening cap,
    # or DISTINGUISHED is unreachable by construction. Retrieval must
    # shrink alongside so only this rule fires.
    with pytest.raises(PriorArtPreflightError, match="unreachable"):
        _check(
            directive=_directive(
                results_per_query=1, max_screened_per_candidate=8
            ),
            cited=1,
            thresholds=PriorArtThresholds(min_unique_sources=9),
        )


def test_the_call_budget_must_cover_the_worst_case() -> None:
    with pytest.raises(PriorArtPreflightError, match="worst-case calls"):
        _check(directive=_directive(max_model_calls=35))
    _check(directive=_directive(max_model_calls=36))
    # One candidate needs exactly twelve at the defaults.
    with pytest.raises(PriorArtPreflightError, match="worst-case calls"):
        _check(directive=_directive(max_model_calls=11), candidates=1)
    _check(directive=_directive(max_model_calls=12), candidates=1)


def test_every_violation_is_collected_in_one_refusal() -> None:
    with pytest.raises(PriorArtPreflightError) as caught:
        _check(
            directive=_directive(
                max_screened_per_candidate=10,
                max_compared_works=1,
                max_model_calls=6,
            ),
            thresholds=PriorArtThresholds(
                min_unique_sources=15, min_compared_works=2
            ),
        )
    message = str(caught.value)
    assert "mechanically" in message
    assert "min_compared_works" in message
    assert "unreachable" in message
    assert "worst-case calls" in message


def test_the_five_candidate_default_portfolio_is_detected() -> None:
    # The ideation layer defaults to five candidates; at the worst case
    # that is 60 calls against the hard ceiling of 36. The preflight
    # names the mismatch instead of letting the run abort mid-flight.
    with pytest.raises(PriorArtPreflightError, match="worst-case calls"):
        _check(candidates=5)
