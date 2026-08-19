"""The prior-art vocabulary: pinned enums, bounded directives, coherent
comparisons, and coverage arithmetic that cannot lie."""

from __future__ import annotations

import pytest

from autonomous_research_lab.literature.retrieval import ResultOrdering
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    SupportLocation,
)
from autonomous_research_lab.priorart.directive import (
    COMPARED_WORKS_CEILING,
    MODEL_CALLS_CEILING,
    RESULTS_PER_QUERY_CEILING,
    SCREENED_PER_CANDIDATE_CEILING,
    PriorArtDirective,
)
from autonomous_research_lab.priorart.records import (
    DIMENSIONS,
    ComparisonDimension,
    DimensionComparison,
    PriorArtCoverage,
    PriorArtQueryExecution,
    PriorArtQueryFamily,
    PriorArtRunRecord,
    PriorArtScreeningRecord,
    SimilarityDecision,
    SimilarityLabel,
    WorkComparison,
)


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id=None,
        latency_seconds=0.25,
        input_tokens=100,
        output_tokens=50,
        repair_count=0,
    )


def _dimension(
    dimension: ComparisonDimension = ComparisonDimension.MECHANISM,
) -> DimensionComparison:
    return DimensionComparison(
        dimension=dimension,
        candidate_position="reweights attention heads directly",
        prior_work_position="prunes attention heads after training",
        support_location=SupportLocation.ABSTRACT,
        support_snippet="prunes attention heads",
    )


def _comparison(**overrides: object) -> WorkComparison:
    defaults: dict[str, object] = {
        "run_id": "pac_1",
        "candidate_id": "idea_1",
        "source_id": "lit_1",
        "known_prior_art": False,
        "dimensions": tuple(_dimension(entry) for entry in DIMENSIONS),
        "overlap_features": ("both intervene on attention heads",),
        "material_differences": ("reweighting versus pruning",),
        "similarity": SimilarityLabel.RELATED,
        "provenance": _provenance(),
    }
    defaults.update(overrides)
    return WorkComparison(**defaults)  # type: ignore[arg-type]


def _coverage(**overrides: object) -> PriorArtCoverage:
    defaults: dict[str, object] = {
        "families_executed": tuple(
            family.value for family in PriorArtQueryFamily
        ),
        "queries_executed": 6,
        "total_retrieved": 30,
        "unique_sources": 20,
        "overlap": 14,
        "saturation": 0.5,
        "post_cutoff_excluded": 2,
        "undated_sources": 1,
        "abstract_level": 15,
        "metadata_level": 3,
        "known_prior_art_listed": 4,
        "known_prior_art_recovered": 2,
        "screened": 18,
        "potential_overlap": 3,
        "related": 4,
        "unrelated": 10,
        "undecidable": 1,
        "metadata_ambiguous": 1,
        "screening_truncated": 0,
        "compared_works": 4,
    }
    defaults.update(overrides)
    return PriorArtCoverage(**defaults)  # type: ignore[arg-type]


# -- pinned vocabulary --------------------------------------------------------


def test_the_query_families_are_pinned() -> None:
    assert [family.value for family in PriorArtQueryFamily] == [
        "mechanism",
        "problem_mechanism",
        "evaluation_setup",
        "synonyms_legacy",
        "competing_approaches",
        "recent",
    ]


def test_the_comparison_dimensions_are_pinned() -> None:
    assert [dimension.value for dimension in ComparisonDimension] == [
        "scientific_question",
        "mechanism",
        "data_setting",
        "evaluation_protocol",
        "claimed_contribution",
    ]
    assert tuple(ComparisonDimension) == DIMENSIONS


def test_the_judgment_vocabularies_are_pinned() -> None:
    assert [decision.value for decision in SimilarityDecision] == [
        "potential_overlap",
        "related",
        "unrelated",
        "undecidable",
    ]
    assert [label.value for label in SimilarityLabel] == [
        "substantial_match",
        "related",
        "distinct",
    ]


# -- the directive ------------------------------------------------------------


def test_a_directive_rejects_a_malformed_cutoff() -> None:
    with pytest.raises(ValueError, match="ISO date"):
        PriorArtDirective(
            ideation_run_record_id="irun_1",
            cutoff_date="August 2026",
            recent_window_start="2025-08-18",
        )
    with pytest.raises(ValueError, match="ISO date"):
        PriorArtDirective(
            ideation_run_record_id="irun_1",
            cutoff_date="2026-08-18",
            recent_window_start="",
        )


def test_the_recent_window_cannot_start_after_the_cutoff() -> None:
    with pytest.raises(ValueError, match="cannot start after"):
        PriorArtDirective(
            ideation_run_record_id="irun_1",
            cutoff_date="2026-08-18",
            recent_window_start="2026-08-19",
        )


def test_directive_budgets_are_bounded_by_construction() -> None:
    for label, ceiling in (
        ("results_per_query", RESULTS_PER_QUERY_CEILING),
        ("max_screened_per_candidate", SCREENED_PER_CANDIDATE_CEILING),
        ("max_compared_works", COMPARED_WORKS_CEILING),
        ("max_model_calls", MODEL_CALLS_CEILING),
    ):
        with pytest.raises(ValueError, match=label):
            PriorArtDirective(
                ideation_run_record_id="irun_1",
                cutoff_date="2026-08-18",
                recent_window_start="2025-08-18",
                **{label: ceiling + 1},  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match=label):
            PriorArtDirective(
                ideation_run_record_id="irun_1",
                cutoff_date="2026-08-18",
                recent_window_start="2025-08-18",
                **{label: 0},  # type: ignore[arg-type]
            )


def test_directive_identity_is_content_addressed() -> None:
    first = PriorArtDirective(
        ideation_run_record_id="irun_1",
        cutoff_date="2026-08-18",
        recent_window_start="2025-08-18",
    )
    same = PriorArtDirective(
        ideation_run_record_id="irun_1",
        cutoff_date="2026-08-18",
        recent_window_start="2025-08-18",
    )
    other = PriorArtDirective(
        ideation_run_record_id="irun_1",
        cutoff_date="2026-08-17",
        recent_window_start="2025-08-18",
    )
    assert first.id == same.id
    assert first.id.startswith("pdir_")
    assert first.id != other.id


# -- query executions ---------------------------------------------------------


def test_a_query_execution_records_the_executed_text() -> None:
    execution = PriorArtQueryExecution(
        run_id="pac_1",
        candidate_id="idea_1",
        family=PriorArtQueryFamily.MECHANISM,
        text="attention head reweighting",
        from_date="",
        to_date="2026-08-18",
        query_fingerprint="litq_1",
        search_record_id="lits_1",
        retrieved=5,
        new_unique=4,
        from_cache=False,
        ordering=ResultOrdering.INFLUENCE,
    )
    assert execution.id.startswith("pqx_")
    with pytest.raises(ValueError, match="executed text"):
        PriorArtQueryExecution(
            run_id="pac_1",
            candidate_id="idea_1",
            family=PriorArtQueryFamily.MECHANISM,
            text="   ",
            from_date="",
            to_date="2026-08-18",
            query_fingerprint="litq_1",
            search_record_id="lits_1",
            retrieved=5,
            new_unique=4,
            from_cache=False,
            ordering=ResultOrdering.INFLUENCE,
        )


def test_plan_provenance_binds_plan_and_renderer() -> None:
    def _execution(**overrides: object) -> PriorArtQueryExecution:
        defaults: dict[str, object] = {
            "run_id": "pac_1",
            "candidate_id": "idea_1",
            "family": PriorArtQueryFamily.MECHANISM,
            "text": '("attention head reweighting")',
            "from_date": "",
            "to_date": "2026-08-18",
            "query_fingerprint": "litq_1",
            "search_record_id": "lits_1",
            "retrieved": 5,
            "new_unique": 4,
            "from_cache": False,
            "ordering": ResultOrdering.INFLUENCE,
        }
        defaults.update(overrides)
        return PriorArtQueryExecution(**defaults)  # type: ignore[arg-type]

    planned = _execution(
        plan_groups=(("attention head reweighting",),),
        renderer="boolean-v1",
    )
    legacy = _execution()
    # A plan without its renderer (or vice versa) is unconstructible.
    with pytest.raises(ValueError, match="renderer"):
        _execution(plan_groups=(("term",),))
    with pytest.raises(ValueError, match="renderer"):
        _execution(renderer="boolean-v1")
    # The plan joins the identity only when present, so a pre-5D.1
    # record keeps deriving its original id.
    assert planned.id != legacy.id
    assert legacy.id == _execution().id


# -- screening ----------------------------------------------------------------


def test_a_screening_judgment_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        PriorArtScreeningRecord(
            run_id="pac_1",
            candidate_id="idea_1",
            source_id="lit_1",
            known_prior_art=False,
            decision=SimilarityDecision.RELATED,
            reason="  ",
            provenance=_provenance(),
        )


# -- comparisons --------------------------------------------------------------


def test_a_dimension_comparison_requires_its_texts() -> None:
    for label in (
        "candidate_position",
        "prior_work_position",
        "support_snippet",
    ):
        with pytest.raises(ValueError, match=label):
            DimensionComparison(
                dimension=ComparisonDimension.MECHANISM,
                candidate_position=(
                    "" if label == "candidate_position" else "reweights"
                ),
                prior_work_position=(
                    "" if label == "prior_work_position" else "prunes"
                ),
                support_location=SupportLocation.ABSTRACT,
                support_snippet="" if label == "support_snippet" else "text",
            )


def test_a_comparison_covers_each_dimension_exactly_once() -> None:
    with pytest.raises(ValueError, match="five dimensions"):
        _comparison(dimensions=tuple(_dimension(d) for d in DIMENSIONS[:-1]))
    with pytest.raises(ValueError, match="five dimensions"):
        _comparison(
            dimensions=(
                *tuple(_dimension(d) for d in DIMENSIONS[:-1]),
                _dimension(DIMENSIONS[0]),
            )
        )


def test_similarity_coherence_is_structural() -> None:
    # A match must name what overlaps; a distinction must name what
    # differs; RELATED needs both. The contradiction is unconstructible.
    with pytest.raises(ValueError, match="overlapping features"):
        _comparison(
            similarity=SimilarityLabel.SUBSTANTIAL_MATCH,
            overlap_features=(),
        )
    with pytest.raises(ValueError, match="material differences"):
        _comparison(
            similarity=SimilarityLabel.DISTINCT, material_differences=()
        )
    with pytest.raises(ValueError, match="overlapping features"):
        _comparison(similarity=SimilarityLabel.RELATED, overlap_features=())
    assert (
        _comparison(
            similarity=SimilarityLabel.SUBSTANTIAL_MATCH,
            material_differences=(),
        ).similarity
        is SimilarityLabel.SUBSTANTIAL_MATCH
    )
    assert (
        _comparison(
            similarity=SimilarityLabel.DISTINCT, overlap_features=()
        ).similarity
        is SimilarityLabel.DISTINCT
    )


def test_comparison_identity_is_content_addressed() -> None:
    assert _comparison().id == _comparison().id
    assert _comparison().id.startswith("pcmp_")
    assert _comparison().id != _comparison(source_id="lit_2").id


# -- coverage -----------------------------------------------------------------


def test_coverage_arithmetic_is_validated() -> None:
    assert _coverage().overlap == 14
    with pytest.raises(ValueError, match="overlap"):
        _coverage(overlap=13)
    with pytest.raises(ValueError, match="access-level split"):
        _coverage(abstract_level=14)
    with pytest.raises(ValueError, match="partition the in-cutoff pool"):
        _coverage(screening_truncated=1)
    with pytest.raises(ValueError, match="screening decisions"):
        _coverage(unrelated=9)
    with pytest.raises(ValueError, match="negative"):
        _coverage(undecidable=-1, unrelated=12)
    with pytest.raises(ValueError, match="saturation"):
        _coverage(saturation=1.5)
    with pytest.raises(ValueError, match="recovered"):
        _coverage(known_prior_art_recovered=5)
    with pytest.raises(ValueError, match="metadata_ambiguous"):
        _coverage(metadata_ambiguous=4)
    with pytest.raises(ValueError, match="screened works"):
        _coverage(compared_works=19)


def test_coverage_counts_each_family_once() -> None:
    with pytest.raises(ValueError, match="each family once"):
        _coverage(
            families_executed=("mechanism", "mechanism"), queries_executed=2
        )
    with pytest.raises(ValueError, match="undercount"):
        _coverage(queries_executed=5)


# -- the run record -----------------------------------------------------------


def _run_record(**overrides: object) -> PriorArtRunRecord:
    defaults: dict[str, object] = {
        "run_id": "pac_1",
        "directive_id": "pdir_1",
        "ideation_run_record_id": "irun_1",
        "ideation_run_id": "idg_1",
        "assessment_id": "madq_1",
        "map_run_id": "map_1",
        "snapshot_id": "cfp_1",
        "candidate_ids": ("idea_1", "idea_2"),
        "prior_art_assessment_ids": ("paa_1", "paa_2"),
        "query_execution_ids": ("pqx_1",),
        "screening_ids": ("pscr_1",),
        "comparison_ids": ("pcmp_1",),
        "model_calls": 8,
        "input_tokens": 1000,
        "output_tokens": 500,
    }
    defaults.update(overrides)
    return PriorArtRunRecord(**defaults)  # type: ignore[arg-type]


def test_a_run_record_pairs_each_candidate_with_one_assessment() -> None:
    assert _run_record().id.startswith("prun_")
    with pytest.raises(ValueError, match="exactly one assessment"):
        _run_record(prior_art_assessment_ids=("paa_1",))
    with pytest.raises(ValueError, match="refusal portfolio never enters"):
        _run_record(candidate_ids=(), prior_art_assessment_ids=())
    with pytest.raises(ValueError, match="each id once"):
        _run_record(
            candidate_ids=("idea_1", "idea_1"),
            prior_art_assessment_ids=("paa_1", "paa_2"),
        )
