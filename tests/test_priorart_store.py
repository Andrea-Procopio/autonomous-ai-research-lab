"""The prior-art store: write-once, verify-on-repeat, tamper-loud, with
one assessment per candidate per run and one account of each run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.literature.retrieval import ResultOrdering
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    SupportLocation,
)
from autonomous_research_lab.priorart.assessment import (
    PriorArtAssessment,
    PriorArtReason,
    PriorArtReasonCode,
    PriorArtThresholds,
    PriorArtVerdict,
)
from autonomous_research_lab.priorart.directive import PriorArtDirective
from autonomous_research_lab.priorart.records import (
    DIMENSIONS,
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
from autonomous_research_lab.priorart.store import (
    PriorArtConflictError,
    PriorArtIntegrityError,
    PriorArtStore,
)

RUN = "pac_1"


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id="req-9",
        latency_seconds=0.25,
        input_tokens=100,
        output_tokens=50,
        repair_count=1,
    )


def _directive() -> PriorArtDirective:
    return PriorArtDirective(
        ideation_run_record_id="irun_1",
        cutoff_date="2026-08-18",
        recent_window_start="2025-08-18",
    )


def _execution() -> PriorArtQueryExecution:
    return PriorArtQueryExecution(
        run_id=RUN,
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


def _screening() -> PriorArtScreeningRecord:
    return PriorArtScreeningRecord(
        run_id=RUN,
        candidate_id="idea_1",
        source_id="lit_1",
        known_prior_art=True,
        decision=SimilarityDecision.POTENTIAL_OVERLAP,
        reason="same intervention family",
        provenance=_provenance(),
    )


def _comparison() -> WorkComparison:
    return WorkComparison(
        run_id=RUN,
        candidate_id="idea_1",
        source_id="lit_1",
        known_prior_art=True,
        dimensions=tuple(
            DimensionComparison(
                dimension=dimension,
                candidate_position="reweights attention heads",
                prior_work_position="prunes attention heads",
                support_location=SupportLocation.ABSTRACT,
                support_snippet="prunes attention heads",
            )
            for dimension in DIMENSIONS
        ),
        overlap_features=("both intervene on attention heads",),
        material_differences=("reweighting versus pruning",),
        similarity=SimilarityLabel.RELATED,
        provenance=_provenance(),
    )


def _coverage() -> PriorArtCoverage:
    return PriorArtCoverage(
        families_executed=tuple(
            family.value for family in PriorArtQueryFamily
        ),
        queries_executed=6,
        total_retrieved=12,
        unique_sources=12,
        overlap=1,
        saturation=0.25,
        post_cutoff_excluded=1,
        undated_sources=0,
        abstract_level=10,
        metadata_level=1,
        known_prior_art_listed=1,
        known_prior_art_recovered=1,
        screened=11,
        potential_overlap=1,
        related=2,
        unrelated=7,
        undecidable=1,
        metadata_ambiguous=1,
        screening_truncated=0,
        compared_works=1,
    )


def _assessment(candidate_id: str = "idea_1") -> PriorArtAssessment:
    return PriorArtAssessment(
        run_id=RUN,
        candidate_id=candidate_id,
        directive_id=_directive().id,
        verdict=PriorArtVerdict.NOVELTY_UNRESOLVED,
        overlapping_work_ids=(),
        compared_work_ids=("lit_1",),
        reasons=(
            PriorArtReason(
                PriorArtReasonCode.METADATA_AMBIGUITY,
                "metadata-only source lit_7 screened as undecidable",
            ),
        ),
        thresholds=PriorArtThresholds(),
        coverage=_coverage(),
    )


def _run_record(run_id: str = RUN) -> PriorArtRunRecord:
    return PriorArtRunRecord(
        run_id=run_id,
        directive_id=_directive().id,
        ideation_run_record_id="irun_1",
        ideation_run_id="idg_1",
        assessment_id="madq_1",
        map_run_id="map_1",
        snapshot_id="cfp_1",
        candidate_ids=("idea_1",),
        prior_art_assessment_ids=(_assessment().id,),
        query_execution_ids=(_execution().id,),
        screening_ids=(_screening().id,),
        comparison_ids=(_comparison().id,),
        model_calls=4,
        input_tokens=1000,
        output_tokens=400,
    )


def test_every_record_kind_reloads_identically(tmp_path: Path) -> None:
    store = PriorArtStore(tmp_path / "priorart")
    directive = store.record_directive(_directive())
    execution = store.record_query_execution(_execution())
    screening = store.record_screening(_screening())
    comparison = store.record_comparison(_comparison())
    assessment = store.record_assessment(_assessment())
    run = store.record_run(_run_record())

    fresh = PriorArtStore(tmp_path / "priorart")
    assert fresh.get_directive(directive.id) == directive
    assert fresh.get_query_execution(execution.id) == execution
    assert fresh.get_screening(screening.id) == screening
    assert fresh.get_comparison(comparison.id) == comparison
    assert fresh.get_assessment(assessment.id) == assessment
    assert fresh.get_run(run.id) == run
    assert fresh.assessments() == (assessment,)
    assert fresh.runs() == (run,)


def test_records_are_write_once_and_verify_on_repeat(
    tmp_path: Path,
) -> None:
    store = PriorArtStore(tmp_path / "priorart")
    comparison = store.record_comparison(_comparison())
    # Identical re-recording is a no-op.
    assert store.record_comparison(_comparison()) == comparison
    # A doctored file under the same name makes re-recording a conflict.
    path = (
        tmp_path / "priorart" / "comparisons" / f"{comparison.id}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["similarity"] = "distinct"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    with pytest.raises(PriorArtConflictError, match="never rewritten"):
        store.record_comparison(_comparison())


def test_a_tampered_record_fails_on_load(tmp_path: Path) -> None:
    store = PriorArtStore(tmp_path / "priorart")
    assessment = store.record_assessment(_assessment())
    path = (
        tmp_path / "priorart" / "assessments" / f"{assessment.id}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["verdict"] = "distinguished"
    payload["reasons"] = []
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    fresh = PriorArtStore(tmp_path / "priorart")
    with pytest.raises(PriorArtIntegrityError, match="no longer matches"):
        fresh.get_assessment(assessment.id)


def test_one_assessment_per_candidate_per_run_is_enforced(
    tmp_path: Path,
) -> None:
    store = PriorArtStore(tmp_path / "priorart")
    store.record_assessment(_assessment())
    second = PriorArtAssessment(
        run_id=RUN,
        candidate_id="idea_1",
        directive_id=_directive().id,
        verdict=PriorArtVerdict.DISTINGUISHED,
        overlapping_work_ids=(),
        compared_work_ids=("lit_1",),
        reasons=(),
        thresholds=PriorArtThresholds(),
        coverage=_coverage(),
    )
    with pytest.raises(PriorArtConflictError, match="second verdict"):
        store.record_assessment(second)
    # The same candidate in another run is a legitimate new assessment.
    other_run = PriorArtAssessment(
        run_id="pac_2",
        candidate_id="idea_1",
        directive_id=_directive().id,
        verdict=PriorArtVerdict.DISTINGUISHED,
        overlapping_work_ids=(),
        compared_work_ids=("lit_1",),
        reasons=(),
        thresholds=PriorArtThresholds(),
        coverage=_coverage(),
    )
    store.record_assessment(other_run)
    assert store.assessment_for_candidate(RUN, "idea_1") == _assessment()
    assert store.assessment_for_candidate("pac_2", "idea_1") == other_run
    assert store.assessment_for_candidate("pac_3", "idea_1") is None


def test_one_run_record_per_run_is_enforced(tmp_path: Path) -> None:
    store = PriorArtStore(tmp_path / "priorart")
    store.record_run(_run_record())
    second = PriorArtRunRecord(
        run_id=RUN,
        directive_id=_directive().id,
        ideation_run_record_id="irun_1",
        ideation_run_id="idg_1",
        assessment_id="madq_1",
        map_run_id="map_1",
        snapshot_id="cfp_1",
        candidate_ids=("idea_1",),
        prior_art_assessment_ids=(_assessment().id,),
        query_execution_ids=(),
        screening_ids=(),
        comparison_ids=(),
        model_calls=9,
        input_tokens=1,
        output_tokens=1,
    )
    with pytest.raises(PriorArtConflictError, match="second account"):
        store.record_run(second)
    store.record_run(_run_record(run_id="pac_2"))
    assert len(store.runs()) == 2


def test_rejected_payloads_are_preserved_as_data(tmp_path: Path) -> None:
    store = PriorArtStore(tmp_path / "priorart")
    path = store.preserve_rejected(
        run_id=RUN,
        stage="comparison",
        reasons=(("unsupported_claim", "snippet not in abstract"),),
        request_fingerprint="mreq_1",
        response_id="mcall_2",
        payload={"comparisons": [{"source_id": "lit_9"}]},
        repair=1,
    )
    assert path.exists()
    (rejected,) = store.rejected()
    assert rejected["run_id"] == RUN
    assert rejected["stage"] == "comparison"
    assert rejected["reasons"] == [
        {"rule": "unsupported_claim", "detail": "snippet not in abstract"}
    ]
    assert rejected["repair"] == 1
    assert rejected["payload"] == {"comparisons": [{"source_id": "lit_9"}]}


def test_absent_kinds_read_as_empty(tmp_path: Path) -> None:
    store = PriorArtStore(tmp_path / "priorart")
    assert store.get_directive("pdir_missing") is None
    assert store.assessments() == ()
    assert store.runs() == ()
    assert store.rejected() == ()
