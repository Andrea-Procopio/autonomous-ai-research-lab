"""The selection door and the trusted partition: one named prior-art
run, every record loaded and cross-checked before any model call, and
eligibility that the named run's verdicts alone decide."""

from __future__ import annotations

from pathlib import Path

import pytest

from autonomous_research_lab.ideation.direction import DirectionRecord
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
    ThemeEra,
)
from autonomous_research_lab.priorart.assessment import (
    PriorArtAssessment,
    PriorArtReason,
    PriorArtReasonCode,
    PriorArtThresholds,
    PriorArtVerdict,
)
from autonomous_research_lab.priorart.records import (
    PriorArtCoverage,
    PriorArtQueryFamily,
    PriorArtRunRecord,
)
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.selection.eligibility import (
    MissingChallengedPortfolioError,
    partition_by_verdict,
    require_challenged_portfolio_for_selection,
)

RUN = "pac_1"


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


def _candidate(title: str) -> CandidateIdea:
    statement = "head-level mechanisms of in-context learning are untested"
    return CandidateIdea(
        run_id="idg_1",
        title=title,
        research_question=f"{title}?",
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
        cited_source_ids=("lit_c1",),
        cited_recent=1,
        cited_foundational=0,
        cited_undated=0,
        provenance=_provenance(),
    )


def _portfolio(candidates: int) -> PortfolioReport:
    return PortfolioReport(
        problems_total=1,
        problems_addressed=1,
        problems_unaddressed=0,
        unaddressed_statements=(),
        addressed_multi_source=0,
        addressed_tentative=1,
        addressed_single_source_limitation=0,
        addressed_contradicted=0,
        candidates=candidates,
        distinct_sources_cited=1,
        themes_targeted=1,
        distinct_problem_sets=1,
        distinct_theme_sets=1,
        distinct_dataset_sets=1,
        distinct_metric_sets=1,
    )


def _direction() -> DirectionRecord:
    return DirectionRecord(
        run_id="idg_1",
        snapshot_id="cfp_1",
        scope="mechanistic accounts of in-context learning",
        topics=("adaptation",),
        constraints=("empirical papers",),
        relevant_dates=("2026-01-31",),
        provenance=_provenance(),
    )


def _ideation_run(
    candidate_ids: tuple[str, ...], direction_id: str
) -> IdeationRunRecord:
    return IdeationRunRecord(
        run_id="idg_1",
        directive_id="idir_1",
        assessment_id="madq_1",
        map_run_id="map_1",
        snapshot_id="cfp_1",
        direction_id=direction_id,
        candidate_ids=candidate_ids,
        refusal_justification="",
        diversity_rationale="a small portfolio",
        model_calls=2,
        input_tokens=100,
        output_tokens=50,
        portfolio=_portfolio(len(candidate_ids)),
    )


def _coverage() -> PriorArtCoverage:
    return PriorArtCoverage(
        families_executed=tuple(
            family.value for family in PriorArtQueryFamily
        ),
        queries_executed=6,
        total_retrieved=12,
        unique_sources=12,
        overlap=2,
        saturation=0.5,
        post_cutoff_excluded=0,
        undated_sources=0,
        abstract_level=12,
        metadata_level=0,
        known_prior_art_listed=2,
        known_prior_art_recovered=1,
        screened=12,
        potential_overlap=2,
        related=2,
        unrelated=8,
        undecidable=0,
        metadata_ambiguous=0,
        screening_truncated=0,
        compared_works=2,
    )


def _assessment(
    candidate_id: str,
    verdict: PriorArtVerdict,
    run_id: str = RUN,
) -> PriorArtAssessment:
    overlapping = (
        ("lit_1",) if verdict is PriorArtVerdict.OVERLAPPING else ()
    )
    reasons = (
        (
            PriorArtReason(
                PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES, "thin pool"
            ),
        )
        if verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
        else ()
    )
    return PriorArtAssessment(
        run_id=run_id,
        candidate_id=candidate_id,
        directive_id="pdir_1",
        verdict=verdict,
        overlapping_work_ids=overlapping,
        compared_work_ids=("lit_1", "lit_2"),
        reasons=reasons,
        thresholds=PriorArtThresholds(),
        coverage=_coverage(),
    )


def _prior_art_run(
    candidate_ids: tuple[str, ...],
    assessment_ids: tuple[str, ...],
    ideation_run_record_id: str,
    run_id: str = RUN,
) -> PriorArtRunRecord:
    return PriorArtRunRecord(
        run_id=run_id,
        directive_id="pdir_1",
        ideation_run_record_id=ideation_run_record_id,
        ideation_run_id="idg_1",
        assessment_id="madq_1",
        map_run_id="map_1",
        snapshot_id="cfp_1",
        candidate_ids=candidate_ids,
        prior_art_assessment_ids=assessment_ids,
        query_execution_ids=(),
        screening_ids=(),
        comparison_ids=(),
        model_calls=3,
        input_tokens=300,
        output_tokens=150,
    )


def _stores(
    tmp_path: Path,
    verdicts: tuple[PriorArtVerdict, ...],
) -> tuple[PriorArtStore, IdeationStore, PriorArtRunRecord]:
    """Record one challenged portfolio with one candidate per verdict
    and return the wired stores plus the durable run record."""
    ideation_store = IdeationStore(tmp_path / "ideation")
    prior_art_store = PriorArtStore(tmp_path / "priorart")
    direction = ideation_store.record_direction(_direction())
    candidates = tuple(
        ideation_store.record_idea(_candidate(f"Candidate {index}"))
        for index in range(len(verdicts))
    )
    candidate_ids = tuple(candidate.id for candidate in candidates)
    ideation_run = ideation_store.record_run(
        _ideation_run(candidate_ids, direction.id)
    )
    assessments = tuple(
        prior_art_store.record_prior_art_assessment(
            _assessment(candidate_id, verdict)
        )
        for candidate_id, verdict in zip(
            candidate_ids, verdicts, strict=True
        )
    )
    record = prior_art_store.record_run(
        _prior_art_run(
            candidate_ids,
            tuple(assessment.id for assessment in assessments),
            ideation_run.id,
        )
    )
    return prior_art_store, ideation_store, record


# -- the door -------------------------------------------------------------------


def test_the_door_returns_the_portfolio_in_record_order(
    tmp_path: Path,
) -> None:
    prior_art, ideation, record = _stores(
        tmp_path,
        (PriorArtVerdict.DISTINGUISHED, PriorArtVerdict.OVERLAPPING),
    )
    inputs = require_challenged_portfolio_for_selection(
        prior_art, ideation, record.id
    )
    assert inputs.prior_art_run == record
    assert tuple(c.id for c in inputs.candidates) == record.candidate_ids
    assert tuple(a.id for a in inputs.assessments) == (
        record.prior_art_assessment_ids
    )
    assert inputs.direction.id == inputs.ideation_run.direction_id


def test_a_missing_run_record_is_refused(tmp_path: Path) -> None:
    prior_art, ideation, _ = _stores(
        tmp_path, (PriorArtVerdict.DISTINGUISHED,)
    )
    with pytest.raises(
        MissingChallengedPortfolioError, match="no prior-art run record"
    ):
        require_challenged_portfolio_for_selection(
            prior_art, ideation, "prun_missing"
        )


def test_a_missing_assessment_is_refused(tmp_path: Path) -> None:
    prior_art, ideation, record = _stores(
        tmp_path, (PriorArtVerdict.DISTINGUISHED,)
    )
    ghost = _prior_art_run(
        record.candidate_ids,
        ("paa_never_recorded",),
        record.ideation_run_record_id,
        run_id="pac_ghost",
    )
    prior_art.record_run(ghost)
    with pytest.raises(
        MissingChallengedPortfolioError, match="partial challenge"
    ):
        require_challenged_portfolio_for_selection(
            prior_art, ideation, ghost.id
        )


def test_an_assessment_from_another_run_is_refused(tmp_path: Path) -> None:
    prior_art, ideation, record = _stores(
        tmp_path, (PriorArtVerdict.DISTINGUISHED,)
    )
    foreign = prior_art.record_prior_art_assessment(
        _assessment(
            record.candidate_ids[0],
            PriorArtVerdict.DISTINGUISHED,
            run_id="pac_other",
        )
    )
    mixed = _prior_art_run(
        record.candidate_ids,
        (foreign.id,),
        record.ideation_run_record_id,
        run_id="pac_mixed",
    )
    prior_art.record_run(mixed)
    with pytest.raises(
        MissingChallengedPortfolioError, match="another run or candidate"
    ):
        require_challenged_portfolio_for_selection(
            prior_art, ideation, mixed.id
        )


def test_an_assessment_for_another_candidate_is_refused(
    tmp_path: Path,
) -> None:
    prior_art, ideation, record = _stores(
        tmp_path,
        (PriorArtVerdict.DISTINGUISHED, PriorArtVerdict.DISTINGUISHED),
    )
    crossed = _prior_art_run(
        record.candidate_ids,
        (
            record.prior_art_assessment_ids[1],
            record.prior_art_assessment_ids[0],
        ),
        record.ideation_run_record_id,
        run_id="pac_crossed",
    )
    prior_art.record_run(crossed)
    with pytest.raises(
        MissingChallengedPortfolioError, match="another run or candidate"
    ):
        require_challenged_portfolio_for_selection(
            prior_art, ideation, crossed.id
        )


def test_a_partial_portfolio_is_refused(tmp_path: Path) -> None:
    prior_art, _ideation, record = _stores(
        tmp_path, (PriorArtVerdict.DISTINGUISHED,)
    )
    fresh_ideation = IdeationStore(tmp_path / "empty_ideation")
    fresh_ideation.record_direction(_direction())
    fresh_ideation.record_run(
        _ideation_run(record.candidate_ids, _direction().id)
    )
    with pytest.raises(
        MissingChallengedPortfolioError, match="partial portfolio"
    ):
        require_challenged_portfolio_for_selection(
            prior_art, fresh_ideation, record.id
        )


def test_a_missing_ideation_run_is_refused(tmp_path: Path) -> None:
    prior_art, _, record = _stores(
        tmp_path, (PriorArtVerdict.DISTINGUISHED,)
    )
    with pytest.raises(MissingChallengedPortfolioError, match="lineage"):
        require_challenged_portfolio_for_selection(
            prior_art, IdeationStore(tmp_path / "empty"), record.id
        )


def test_a_missing_direction_is_refused(tmp_path: Path) -> None:
    prior_art, ideation, record = _stores(
        tmp_path, (PriorArtVerdict.DISTINGUISHED,)
    )
    bare = IdeationStore(tmp_path / "bare_ideation")
    for candidate_id in record.candidate_ids:
        idea = ideation.get_idea(candidate_id)
        assert idea is not None
        bare.record_idea(idea)
    bare.record_run(
        _ideation_run(record.candidate_ids, _direction().id)
    )
    with pytest.raises(MissingChallengedPortfolioError, match="haystack"):
        require_challenged_portfolio_for_selection(
            prior_art, bare, record.id
        )


# -- the partition ----------------------------------------------------------------


def test_partition_is_by_the_named_runs_verdict_alone(
    tmp_path: Path,
) -> None:
    prior_art, ideation, record = _stores(
        tmp_path,
        (
            PriorArtVerdict.DISTINGUISHED,
            PriorArtVerdict.OVERLAPPING,
            PriorArtVerdict.NOVELTY_UNRESOLVED,
        ),
    )
    inputs = require_challenged_portfolio_for_selection(
        prior_art, ideation, record.id
    )
    partition = partition_by_verdict(inputs)
    assert tuple(c.id for c in partition.eligible) == (
        record.candidate_ids[0],
    )
    assert partition.eligible_assessments[0].verdict is (
        PriorArtVerdict.DISTINGUISHED
    )
    by_id = {entry.candidate_id: entry for entry in partition.ineligible}
    overlapping = by_id[record.candidate_ids[1]]
    assert overlapping.verdict is PriorArtVerdict.OVERLAPPING
    assert overlapping.overlapping_work_ids == ("lit_1",)
    unresolved = by_id[record.candidate_ids[2]]
    assert unresolved.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert unresolved.reasons[0].code is (
        PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES
    )


def test_a_distinguished_verdict_in_another_run_does_not_leak(
    tmp_path: Path,
) -> None:
    """The named run is the whole eligibility universe: a candidate
    distinguished by some other challenge stays ineligible here."""
    prior_art, ideation, record = _stores(
        tmp_path, (PriorArtVerdict.NOVELTY_UNRESOLVED,)
    )
    candidate_id = record.candidate_ids[0]
    other = prior_art.record_prior_art_assessment(
        _assessment(
            candidate_id, PriorArtVerdict.DISTINGUISHED, run_id="pac_later"
        )
    )
    prior_art.record_run(
        _prior_art_run(
            (candidate_id,),
            (other.id,),
            record.ideation_run_record_id,
            run_id="pac_later",
        )
    )
    inputs = require_challenged_portfolio_for_selection(
        prior_art, ideation, record.id
    )
    partition = partition_by_verdict(inputs)
    assert partition.eligible == ()
    assert partition.ineligible[0].verdict is (
        PriorArtVerdict.NOVELTY_UNRESOLVED
    )
