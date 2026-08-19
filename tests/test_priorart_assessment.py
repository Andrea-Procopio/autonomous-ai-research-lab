"""The deterministic prior-art verdict: fail-closed aggregation, the
complete-diagnosis discipline, and the one door into a challenge run."""

from __future__ import annotations

from pathlib import Path

import pytest

from autonomous_research_lab.ideation.records import (
    AddressedProblem,
    CandidateIdea,
    DataRequirement,
    DataStatus,
    IdeationRunRecord,
    PortfolioReport,
    Prediction,
    ResourceEstimate,
    TargetedTheme,
    problem_key,
    theme_key,
)
from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.mapping.adequacy import SupportTier
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    ProblemKind,
    SupportLocation,
    ThemeEra,
)
from autonomous_research_lab.priorart.assessment import (
    MissingCandidatePortfolioError,
    PriorArtAssessment,
    PriorArtReason,
    PriorArtReasonCode,
    PriorArtThresholds,
    PriorArtVerdict,
    assess_prior_art,
    require_candidates_for_prior_art,
)
from autonomous_research_lab.priorart.records import (
    DIMENSIONS,
    ComparisonDimension,
    DimensionComparison,
    OverlapHypothesis,
    PriorArtCoverage,
    PriorArtQueryFamily,
    PriorArtScreeningRecord,
    SimilarityDecision,
    SimilarityLabel,
    WorkComparison,
)

RUN = "pac_1"
CANDIDATE = "idea_1"
DIRECTIVE = "pdir_1"


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


def _screening(
    source_id: str, decision: SimilarityDecision
) -> PriorArtScreeningRecord:
    return PriorArtScreeningRecord(
        run_id=RUN,
        candidate_id=CANDIDATE,
        source_id=source_id,
        known_prior_art=False,
        decision=decision,
        reason="screened against the candidate's mechanism",
        provenance=_provenance(),
    )


def _comparison(
    source_id: str,
    similarity: SimilarityLabel = SimilarityLabel.RELATED,
) -> WorkComparison:
    return WorkComparison(
        run_id=RUN,
        candidate_id=CANDIDATE,
        source_id=source_id,
        known_prior_art=False,
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
        material_differences=("reweighting versus pruning",)
        if similarity is not SimilarityLabel.SUBSTANTIAL_MATCH
        else (),
        similarity=similarity,
        provenance=_provenance(),
    )


_DECISIONS: dict[str, SimilarityDecision] = {
    "lit_1": SimilarityDecision.POTENTIAL_OVERLAP,
    "lit_2": SimilarityDecision.POTENTIAL_OVERLAP,
    "lit_3": SimilarityDecision.RELATED,
    "lit_4": SimilarityDecision.RELATED,
    **{
        f"lit_{index}": SimilarityDecision.UNRELATED
        for index in range(5, 13)
    },
}


def _coverage(**overrides: object) -> PriorArtCoverage:
    defaults: dict[str, object] = {
        "families_executed": tuple(
            family.value for family in PriorArtQueryFamily
        ),
        "queries_executed": 6,
        "total_retrieved": 12,
        "unique_sources": 12,
        "overlap": 2,
        "saturation": 0.5,
        "post_cutoff_excluded": 0,
        "undated_sources": 0,
        "abstract_level": 12,
        "metadata_level": 0,
        "known_prior_art_listed": 2,
        "known_prior_art_recovered": 1,
        "screened": 12,
        "potential_overlap": 2,
        "related": 2,
        "unrelated": 8,
        "undecidable": 0,
        "metadata_ambiguous": 0,
        "screening_truncated": 0,
        "compared_works": 2,
    }
    defaults.update(overrides)
    return PriorArtCoverage(**defaults)  # type: ignore[arg-type]


def _distinguished_inputs() -> dict[str, object]:
    return {
        "run_id": RUN,
        "candidate_id": CANDIDATE,
        "directive_id": DIRECTIVE,
        "screenings": tuple(
            _screening(source_id, decision)
            for source_id, decision in _DECISIONS.items()
        ),
        "comparisons": (_comparison("lit_1"), _comparison("lit_2")),
        "coverage": _coverage(),
        "metadata_source_ids": frozenset(),
        "thresholds": PriorArtThresholds(),
    }


def _assess(**overrides: object) -> PriorArtAssessment:
    inputs = _distinguished_inputs()
    inputs.update(overrides)
    return assess_prior_art(**inputs)  # type: ignore[arg-type]


def _codes(assessment: PriorArtAssessment) -> set[PriorArtReasonCode]:
    return {reason.code for reason in assessment.reasons}


# -- the verdict rules --------------------------------------------------------


def test_adequate_coverage_with_material_differences_distinguishes() -> None:
    assessment = _assess()
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED
    assert assessment.reasons == ()
    assert assessment.overlapping_work_ids == ()
    assert assessment.compared_work_ids == ("lit_1", "lit_2")
    assert assessment.id.startswith("paa_")


def test_one_grounded_substantial_match_yields_overlapping() -> None:
    assessment = _assess(
        comparisons=(
            _comparison("lit_1", SimilarityLabel.SUBSTANTIAL_MATCH),
            _comparison("lit_2"),
        )
    )
    assert assessment.verdict is PriorArtVerdict.OVERLAPPING
    assert assessment.overlapping_work_ids == ("lit_1",)


def test_overlapping_takes_precedence_over_coverage_reasons() -> None:
    # One grounded counterexample falsifies regardless of how thin the
    # search was — and the thin coverage stays on the record.
    assessment = _assess(
        comparisons=(
            _comparison("lit_1", SimilarityLabel.SUBSTANTIAL_MATCH),
            _comparison("lit_2"),
        ),
        coverage=_coverage(
            unique_sources=4,
            overlap=10,
            abstract_level=4,
            screened=4,
            unrelated=0,
        ),
    )
    assert assessment.verdict is PriorArtVerdict.OVERLAPPING
    assert PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES in _codes(assessment)


def test_an_incomplete_family_sweep_is_unresolved() -> None:
    assessment = _assess(
        coverage=_coverage(
            families_executed=("mechanism", "recent"), queries_executed=2
        )
    )
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert _codes(assessment) == {
        PriorArtReasonCode.FAMILY_COVERAGE_INCOMPLETE
    }


def test_a_thin_pool_is_unresolved() -> None:
    assessment = _assess(
        coverage=_coverage(
            unique_sources=9,
            overlap=5,
            abstract_level=9,
            screened=9,
            unrelated=5,
        )
    )
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert _codes(assessment) == {PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES}


def test_excessive_undecidable_screens_are_unresolved() -> None:
    decisions = dict(_DECISIONS)
    for source_id in ("lit_5", "lit_6", "lit_7", "lit_8", "lit_9"):
        decisions[source_id] = SimilarityDecision.UNDECIDABLE
    assessment = _assess(
        screenings=tuple(
            _screening(source_id, decision)
            for source_id, decision in decisions.items()
        ),
        coverage=_coverage(unrelated=3, undecidable=5),
    )
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert _codes(assessment) == {PriorArtReasonCode.EXCESSIVE_UNCERTAINTY}


def test_a_metadata_undecidable_is_coverage_not_a_blocker() -> None:
    # The Task 5D.1 live evidence: every blocking metadata screen was
    # an undecidable title-only screen — the access level restated,
    # never an overlap signal. Undecidability without an attested
    # hypothesis is a coverage fact, and differentiation stays
    # reachable beside it.
    decisions = dict(_DECISIONS)
    decisions["lit_5"] = SimilarityDecision.UNDECIDABLE
    assessment = _assess(
        screenings=tuple(
            _screening(source_id, decision)
            for source_id, decision in decisions.items()
        ),
        metadata_source_ids=frozenset({"lit_5"}),
        coverage=_coverage(
            abstract_level=11,
            metadata_level=1,
            unrelated=7,
            undecidable=1,
        ),
    )
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED
    assert assessment.reasons == ()


def test_uncertainty_is_measured_where_text_exists() -> None:
    # Two of twelve abstract-level screens undecidable (0.1667) beside
    # five metadata-only undecidables: the old whole-pool reading
    # (7/17 = 0.41) would have fired on the missing abstracts alone;
    # the abstract-level basis does not.
    decisions = dict(_DECISIONS)
    decisions["lit_9"] = SimilarityDecision.UNDECIDABLE
    decisions["lit_10"] = SimilarityDecision.UNDECIDABLE
    for index in range(13, 18):
        decisions[f"lit_{index}"] = SimilarityDecision.UNDECIDABLE
    assessment = _assess(
        screenings=tuple(
            _screening(source_id, decision)
            for source_id, decision in decisions.items()
        ),
        metadata_source_ids=frozenset(
            f"lit_{index}" for index in range(13, 18)
        ),
        coverage=_coverage(
            total_retrieved=17,
            unique_sources=17,
            abstract_level=12,
            metadata_level=5,
            screened=17,
            unrelated=6,
            undecidable=7,
        ),
    )
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED
    assert assessment.reasons == ()


def test_the_thin_pool_rule_measures_the_screenable_pool() -> None:
    # Post-cutoff works can never be screened, so they cannot count
    # toward the pool that grounds differentiation.
    nine = tuple(
        _screening(source_id, decision)
        for source_id, decision in tuple(_DECISIONS.items())[:9]
    )
    blocked = _assess(
        screenings=nine,
        coverage=_coverage(
            post_cutoff_excluded=3,
            abstract_level=9,
            screened=9,
            unrelated=5,
        ),
    )
    assert blocked.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert _codes(blocked) == {PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES}
    assert "excluded post-cutoff" in blocked.reasons[0].detail
    ten = tuple(
        _screening(source_id, decision)
        for source_id, decision in tuple(_DECISIONS.items())[:10]
    )
    passing = _assess(
        screenings=ten,
        coverage=_coverage(
            unique_sources=13,
            overlap=1,
            post_cutoff_excluded=3,
            abstract_level=10,
            screened=10,
            unrelated=6,
        ),
    )
    assert passing.verdict is PriorArtVerdict.DISTINGUISHED


def test_a_material_ambiguity_names_its_attested_claim() -> None:
    attested = PriorArtScreeningRecord(
        run_id=RUN,
        candidate_id=CANDIDATE,
        source_id="lit_2",
        known_prior_art=False,
        decision=SimilarityDecision.POTENTIAL_OVERLAP,
        reason="the title names the candidate's exact mechanism",
        provenance=_provenance(),
        overlap_hypothesis=OverlapHypothesis(
            candidate_claim="reweights attention heads",
            source_text="attention head reweighting",
            support_location=SupportLocation.TITLE,
            dimension=ComparisonDimension.MECHANISM,
            rationale="the title claims the proposed core mechanism",
        ),
    )
    assessment = _assess(
        screenings=tuple(
            attested
            if source_id == "lit_2"
            else _screening(source_id, decision)
            for source_id, decision in _DECISIONS.items()
        ),
        comparisons=(_comparison("lit_1"),),
        metadata_source_ids=frozenset({"lit_2"}),
        coverage=_coverage(
            abstract_level=11,
            metadata_level=1,
            metadata_ambiguous=1,
            compared_works=1,
        ),
    )
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert _codes(assessment) == {PriorArtReasonCode.METADATA_AMBIGUITY}
    detail = assessment.reasons[0].detail
    assert "claim at risk" in detail
    assert "reweights attention heads" in detail
    assert "mechanism" in detail


def test_a_metadata_potential_overlap_forces_unresolved() -> None:
    # lit_2 has no abstract yet screened as possibly overlapping: it
    # could contain direct prior art, so DISTINGUISHED is unreachable.
    assessment = _assess(
        comparisons=(_comparison("lit_1"),),
        metadata_source_ids=frozenset({"lit_2"}),
        coverage=_coverage(
            abstract_level=11,
            metadata_level=1,
            metadata_ambiguous=1,
            compared_works=1,
        ),
    )
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert _codes(assessment) == {PriorArtReasonCode.METADATA_AMBIGUITY}
    assert "lit_2" in assessment.reasons[0].detail


def test_no_comparable_work_is_unresolved_not_distinguished() -> None:
    decisions = {
        source_id: SimilarityDecision.UNRELATED for source_id in _DECISIONS
    }
    assessment = _assess(
        screenings=tuple(
            _screening(source_id, decision)
            for source_id, decision in decisions.items()
        ),
        comparisons=(),
        coverage=_coverage(
            potential_overlap=0, related=0, unrelated=12, compared_works=0
        ),
    )
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert _codes(assessment) == {PriorArtReasonCode.NO_COMPARABLE_WORK}
    assert "never proof of novelty" in assessment.reasons[0].detail


def test_an_uncompared_potential_overlap_is_unresolved() -> None:
    assessment = _assess(
        comparisons=(_comparison("lit_1"),),
        coverage=_coverage(compared_works=1),
    )
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert _codes(assessment) == {
        PriorArtReasonCode.UNCOMPARED_POTENTIAL_OVERLAP
    }
    assert "lit_2" in assessment.reasons[0].detail


def test_truncated_screening_is_unresolved() -> None:
    decisions = dict(_DECISIONS)
    del decisions["lit_12"]
    assessment = _assess(
        screenings=tuple(
            _screening(source_id, decision)
            for source_id, decision in decisions.items()
        ),
        coverage=_coverage(
            screened=11, screening_truncated=1, unrelated=7
        ),
    )
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert _codes(assessment) == {PriorArtReasonCode.SCREENING_TRUNCATED}


def test_every_fired_rule_is_recorded_together() -> None:
    assessment = _assess(
        comparisons=(),
        coverage=_coverage(
            families_executed=("mechanism",),
            queries_executed=1,
            unique_sources=5,
            overlap=9,
            abstract_level=5,
            screened=5,
            unrelated=1,
            compared_works=0,
        ),
    )
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert {
        PriorArtReasonCode.FAMILY_COVERAGE_INCOMPLETE,
        PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES,
        PriorArtReasonCode.NO_COMPARABLE_WORK,
        PriorArtReasonCode.UNCOMPARED_POTENTIAL_OVERLAP,
    } <= _codes(assessment)


# -- input consistency --------------------------------------------------------


def test_records_from_another_run_or_candidate_are_refused() -> None:
    stray = PriorArtScreeningRecord(
        run_id="pac_other",
        candidate_id=CANDIDATE,
        source_id="lit_1",
        known_prior_art=False,
        decision=SimilarityDecision.UNRELATED,
        reason="from another run",
        provenance=_provenance(),
    )
    with pytest.raises(ValueError, match="another run"):
        _assess(screenings=(stray,))
    with pytest.raises(ValueError, match="another run"):
        _assess(
            comparisons=(
                WorkComparison(
                    run_id=RUN,
                    candidate_id="idea_other",
                    source_id="lit_1",
                    known_prior_art=False,
                    dimensions=_comparison("lit_1").dimensions,
                    overlap_features=("overlap",),
                    material_differences=("difference",),
                    similarity=SimilarityLabel.RELATED,
                    provenance=_provenance(),
                ),
            )
        )


def test_the_coverage_must_match_the_records() -> None:
    with pytest.raises(ValueError, match="recorded comparisons"):
        _assess(comparisons=(_comparison("lit_1"),))
    with pytest.raises(ValueError, match="metadata-ambiguous"):
        _assess(metadata_source_ids=frozenset({"lit_1"}))


# -- the assessment invariants ------------------------------------------------


def test_the_verdict_shape_invariants_are_structural() -> None:
    good = _assess()
    with pytest.raises(ValueError, match="imply each other"):
        PriorArtAssessment(
            run_id=RUN,
            candidate_id=CANDIDATE,
            directive_id=DIRECTIVE,
            verdict=PriorArtVerdict.OVERLAPPING,
            overlapping_work_ids=(),
            compared_work_ids=("lit_1",),
            reasons=(),
            thresholds=PriorArtThresholds(),
            coverage=_coverage(),
        )
    with pytest.raises(ValueError, match="unresolved reasons"):
        PriorArtAssessment(
            run_id=RUN,
            candidate_id=CANDIDATE,
            directive_id=DIRECTIVE,
            verdict=PriorArtVerdict.DISTINGUISHED,
            overlapping_work_ids=(),
            compared_work_ids=("lit_1",),
            reasons=(
                PriorArtReason(
                    PriorArtReasonCode.SCREENING_TRUNCATED, "one source"
                ),
            ),
            thresholds=PriorArtThresholds(),
            coverage=_coverage(),
        )
    with pytest.raises(ValueError, match="names why"):
        PriorArtAssessment(
            run_id=RUN,
            candidate_id=CANDIDATE,
            directive_id=DIRECTIVE,
            verdict=PriorArtVerdict.NOVELTY_UNRESOLVED,
            overlapping_work_ids=(),
            compared_work_ids=("lit_1",),
            reasons=(),
            thresholds=PriorArtThresholds(),
            coverage=_coverage(),
        )
    with pytest.raises(ValueError, match="among the compared"):
        PriorArtAssessment(
            run_id=RUN,
            candidate_id=CANDIDATE,
            directive_id=DIRECTIVE,
            verdict=PriorArtVerdict.OVERLAPPING,
            overlapping_work_ids=("lit_9",),
            compared_work_ids=("lit_1",),
            reasons=(),
            thresholds=PriorArtThresholds(),
            coverage=_coverage(),
        )
    assert good.verdict is PriorArtVerdict.DISTINGUISHED


def test_thresholds_are_validated_and_travel_with_the_verdict() -> None:
    with pytest.raises(ValueError, match="min_unique_sources"):
        PriorArtThresholds(min_unique_sources=0)
    with pytest.raises(ValueError, match="fraction"):
        PriorArtThresholds(max_undecidable_fraction=0.0)
    strict = _assess(thresholds=PriorArtThresholds(min_unique_sources=13))
    assert strict.thresholds.min_unique_sources == 13
    assert strict.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED


# -- the door -----------------------------------------------------------------


def _statement() -> str:
    return "reweighting mechanisms remain untested at scale"


def _idea(run_id: str = "idg_1") -> CandidateIdea:
    statement = _statement()
    return CandidateIdea(
        run_id=run_id,
        title="A Candidate",
        research_question="Does reweighting select induction heads?",
        proposed_contribution="a causal test of head reweighting",
        mechanism="reweighting amplifies induction heads",
        hypothesis="reweighted heads carry in-context ability",
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
        cited_source_ids=("lit_a",),
        cited_recent=1,
        cited_foundational=0,
        cited_undated=0,
        provenance=_provenance(),
    )


def _portfolio(candidates: int) -> PortfolioReport:
    return PortfolioReport(
        problems_total=2,
        problems_addressed=1 if candidates else 0,
        problems_unaddressed=1 if candidates else 2,
        unaddressed_statements=("an unaddressed problem",)
        if candidates
        else ("an unaddressed problem", _statement()),
        addressed_multi_source=0,
        addressed_tentative=1 if candidates else 0,
        addressed_single_source_limitation=0,
        addressed_contradicted=0,
        candidates=candidates,
        distinct_sources_cited=candidates,
        themes_targeted=candidates,
        distinct_problem_sets=candidates,
        distinct_theme_sets=candidates,
        distinct_dataset_sets=candidates,
        distinct_metric_sets=candidates,
    )


def _run_record(
    candidate_ids: tuple[str, ...], run_id: str = "idg_1"
) -> IdeationRunRecord:
    return IdeationRunRecord(
        run_id=run_id,
        directive_id="idir_1",
        assessment_id="madq_1",
        map_run_id="map_1",
        snapshot_id="cfp_1",
        direction_id="dir_1",
        candidate_ids=candidate_ids,
        refusal_justification=""
        if candidate_ids
        else "the mapped problems are already saturated",
        diversity_rationale="one candidate per problem"
        if candidate_ids
        else "",
        model_calls=2,
        input_tokens=100,
        output_tokens=50,
        portfolio=_portfolio(len(candidate_ids)),
    )


def test_the_door_admits_a_loadable_portfolio(tmp_path: Path) -> None:
    store = IdeationStore(tmp_path / "ideation")
    idea = store.record_idea(_idea())
    record = store.record_run(_run_record((idea.id,)))
    admitted = require_candidates_for_prior_art(store, record.id)
    assert admitted == record


def test_the_door_refuses_an_unknown_run_record(tmp_path: Path) -> None:
    store = IdeationStore(tmp_path / "ideation")
    with pytest.raises(
        MissingCandidatePortfolioError, match="no ideation run record"
    ):
        require_candidates_for_prior_art(store, "irun_missing")


def test_the_door_refuses_a_refusal_portfolio(tmp_path: Path) -> None:
    store = IdeationStore(tmp_path / "ideation")
    record = store.record_run(_run_record(()))
    with pytest.raises(
        MissingCandidatePortfolioError, match="honest refusal"
    ):
        require_candidates_for_prior_art(store, record.id)


def test_the_door_refuses_a_partial_portfolio(tmp_path: Path) -> None:
    store = IdeationStore(tmp_path / "ideation")
    record = store.record_run(_run_record((_idea().id,)))
    with pytest.raises(
        MissingCandidatePortfolioError, match="partial portfolio"
    ):
        require_candidates_for_prior_art(store, record.id)
