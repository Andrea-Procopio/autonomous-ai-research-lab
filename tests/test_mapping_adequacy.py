"""The deterministic adequacy assessment: pass and fail outcomes, typed
reasons, support tiers, durable reload, and the Task 5C guard. All
records are synthetic; no network, no model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.mapping.adequacy import (
    AdequacyReasonCode,
    AdequacyStatus,
    AdequacyThresholds,
    InadequateFieldMapError,
    MapAdequacyAssessment,
    SupportTier,
    assess_adequacy,
    require_adequate_for_idea_generation,
    support_tier,
)
from autonomous_research_lab.mapping.brief import SourceEra
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    CoverageReport,
    ExtractionRecord,
    FieldMapRecord,
    GroupEntry,
    ProblemEntry,
    ProblemInventoryRecord,
    ProblemKind,
    ScreeningDecision,
    ScreeningRecord,
    SupportLocation,
    ThemeEntry,
    ThemeEra,
)
from autonomous_research_lab.mapping.store import (
    MappingIntegrityError,
    MappingStore,
)

RUN = "map_1"
BRIEF_ID = "brief_1"


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        provider="fake",
        requested_model="m",
        served_model="m",
        provider_request_id=None,
        latency_seconds=0.1,
        input_tokens=10,
        output_tokens=5,
        repair_count=0,
    )


def _screening(source_id: str, decision: ScreeningDecision) -> ScreeningRecord:
    return ScreeningRecord(
        run_id=RUN,
        source_id=source_id,
        decision=decision,
        reason="screened",
        provenance=_provenance(),
    )


def _extraction(
    source_id: str,
    era: SourceEra = SourceEra.RECENT,
    *,
    sufficient: bool = True,
    access: str = "abstract",
) -> ExtractionRecord:
    return ExtractionRecord(
        run_id=RUN,
        source_id=source_id,
        era=era,
        access_level=access,
        support_location=(
            SupportLocation.ABSTRACT
            if access == "abstract"
            else SupportLocation.TITLE
        ),
        sufficient_support=sufficient,
        insufficiency_reason="" if sufficient else "metadata-only access",
        methods=("a method",) if sufficient else (),
        datasets=(),
        metrics=(),
        evaluation_protocols=(),
        baselines=(),
        reported_results=(),
        limitations=(),
        future_work=(),
        open_problems=(),
        provenance=_provenance() if sufficient else None,
    )


def _field_map(
    themes: tuple[tuple[str, tuple[str, ...]], ...],
) -> FieldMapRecord:
    return FieldMapRecord(
        run_id=RUN,
        brief_id=BRIEF_ID,
        themes=tuple(
            ThemeEntry(
                name=name,
                summary="synthesized.",
                era=ThemeEra.BOTH,
                source_ids=source_ids,
            )
            for name, source_ids in themes
        ),
        approaches=(
            GroupEntry(name="an approach", summary="s.", source_ids=("s1",)),
        ),
        evaluation_practices=(),
        relationships=(),
        recent_source_ids=(),
        foundational_source_ids=(),
        undated_source_ids=(),
        provenance=_provenance(),
    )


def _problem(
    statement: str,
    kind: ProblemKind = ProblemKind.OPEN_PROBLEM,
    supporting: tuple[str, ...] = ("s1",),
    conflicting: tuple[str, ...] = (),
) -> ProblemEntry:
    return ProblemEntry(
        statement=statement,
        kind=kind,
        grounding="grounded in the cited reports",
        supporting_source_ids=supporting,
        conflicting_source_ids=conflicting,
    )


def _inventory(*problems: ProblemEntry) -> ProblemInventoryRecord:
    return ProblemInventoryRecord(
        run_id=RUN,
        brief_id=BRIEF_ID,
        problems=problems,
        provenance=_provenance(),
    )


def _coverage(**overrides: object) -> CoverageReport:
    defaults: dict[str, object] = {
        "queries_executed": 7,
        "total_retrieved": 40,
        "unique_sources": 30,
        "screened": 12,
        "screening_truncated": 0,
        "relevant": 9,
        "excluded": 3,
        "uncertain": 0,
        "abstract_level": 12,
        "metadata_level": 0,
        "extraction_eligible": 9,
        "extracted": 9,
        "extraction_truncated": 0,
        "insufficient_support": 0,
        "saturation": 0.2,
    }
    defaults.update(overrides)
    return CoverageReport(**defaults)  # type: ignore[arg-type]


#: A corpus that clears every default bar: 9 relevant of 12 screened,
#: spread over 3 families, 8 grounded (4 recent + 4 foundational), one
#: multi-source theme, one multi-source problem.
_SOURCES = tuple(f"s{i}" for i in range(1, 10))


def _adequate_inputs() -> dict[str, object]:
    screenings = [
        _screening(s, ScreeningDecision.RELEVANT) for s in _SOURCES
    ] + [
        _screening("x1", ScreeningDecision.EXCLUDED),
        _screening("x2", ScreeningDecision.EXCLUDED),
        _screening("x3", ScreeningDecision.EXCLUDED),
    ]
    extractions = [
        _extraction(
            s,
            SourceEra.RECENT if i % 2 == 0 else SourceEra.FOUNDATIONAL,
        )
        for i, s in enumerate(_SOURCES[:8])
    ]
    field_map = _field_map(
        (
            ("Cross-paper theme", ("s1", "s2", "s3")),
            ("A narrower theme", ("s4",)),
        )
    )
    inventory = _inventory(
        _problem("a multi-source problem", supporting=("s1", "s2")),
        _problem(
            "one paper's compute limitation",
            kind=ProblemKind.COMPUTE_LIMITATION,
            supporting=("s3",),
        ),
    )
    return {
        "run_id": RUN,
        "brief_id": BRIEF_ID,
        "screenings": screenings,
        "extractions": extractions,
        "field_map": field_map,
        "inventory": inventory,
        "family_sources": {
            "recent": ("s1", "s2", "s3", "x1"),
            "foundational": ("s4", "s5", "s6", "x2"),
            "limitations_open_problems": ("s7", "s8", "s9", "x3"),
        },
        "coverage": _coverage(),
        "thresholds": AdequacyThresholds(),
    }


def _assess(**overrides: object) -> MapAdequacyAssessment:
    inputs = _adequate_inputs()
    inputs.update(overrides)
    return assess_adequacy(**inputs)  # type: ignore[arg-type]


def _codes(assessment: MapAdequacyAssessment) -> set[AdequacyReasonCode]:
    return {reason.code for reason in assessment.reasons}


# -- the pass and its determinism ---------------------------------------------


def test_an_adequate_corpus_passes_with_no_reasons() -> None:
    assessment = _assess()
    assert assessment.status is AdequacyStatus.ADEQUATE_FOR_IDEA_GENERATION
    assert assessment.reasons == ()
    assert assessment.metrics.relevant_sources == 9
    assert assessment.metrics.grounded_sources == 8


def test_the_verdict_is_deterministic_with_stable_identity() -> None:
    first, second = _assess(), _assess()
    assert first == second
    assert first.id == second.id
    assert first.id.startswith("madq_")


def test_the_thresholds_travel_with_the_assessment() -> None:
    bar = AdequacyThresholds(min_relevant_sources=3, min_grounded_sources=2)
    assessment = _assess(thresholds=bar)
    assert assessment.thresholds == bar


# -- fail outcomes, one rule at a time ----------------------------------------


def test_too_few_relevant_sources_fails() -> None:
    inputs = _adequate_inputs()
    screenings = [
        _screening(s, ScreeningDecision.RELEVANT) for s in _SOURCES[:4]
    ] + [
        _screening(f"x{i}", ScreeningDecision.EXCLUDED) for i in range(8)
    ]
    assessment = _assess(screenings=screenings)
    assert assessment.status is AdequacyStatus.INSUFFICIENT_COVERAGE
    assert AdequacyReasonCode.TOO_FEW_RELEVANT in _codes(assessment)
    del inputs


def test_source_count_alone_is_never_sufficient() -> None:
    """Twenty relevant sources, but every one from a single query family
    and no cross-paper structure anywhere: high count, poor coverage."""
    many = tuple(f"m{i}" for i in range(20))
    screenings = [_screening(s, ScreeningDecision.RELEVANT) for s in many]
    extractions = [_extraction(s, SourceEra.RECENT) for s in many[:8]]
    field_map = _field_map((("Single-source theme", ("m1",)),))
    inventory = _inventory(_problem("a tentative one", supporting=("m1",)))
    assessment = _assess(
        screenings=screenings,
        extractions=extractions,
        field_map=field_map,
        inventory=inventory,
        family_sources={"recent": many},
    )
    codes = _codes(assessment)
    assert assessment.status is AdequacyStatus.INSUFFICIENT_COVERAGE
    assert AdequacyReasonCode.FAMILY_COVERAGE_THIN in codes
    assert AdequacyReasonCode.THEME_SUPPORT_THIN in codes
    assert AdequacyReasonCode.PROBLEM_SUPPORT_THIN in codes
    assert AdequacyReasonCode.FOUNDATIONAL_COVERAGE_THIN in codes


def test_recent_foundational_imbalance_fails() -> None:
    extractions = [
        _extraction(s, SourceEra.RECENT) for s in _SOURCES[:8]
    ]  # everything recent, nothing foundational
    assessment = _assess(extractions=extractions)
    assert AdequacyReasonCode.FOUNDATIONAL_COVERAGE_THIN in _codes(assessment)


def test_abstract_access_limitations_fail_grounding() -> None:
    """Relevant sources whose access level supports no extraction leave
    the corpus ungrounded, however many were screened relevant."""
    extractions = [
        _extraction(s, SourceEra.RECENT) for s in _SOURCES[:3]
    ] + [
        _extraction(s, sufficient=False, access="metadata")
        for s in _SOURCES[3:]
    ]
    assessment = _assess(extractions=extractions)
    codes = _codes(assessment)
    assert AdequacyReasonCode.TOO_FEW_GROUNDED in codes
    detail = next(
        r.detail
        for r in assessment.reasons
        if r.code is AdequacyReasonCode.TOO_FEW_GROUNDED
    )
    assert "metadata-only" in detail


def test_excessive_uncertainty_fails() -> None:
    screenings = [
        _screening(s, ScreeningDecision.RELEVANT) for s in _SOURCES
    ] + [
        _screening(f"u{i}", ScreeningDecision.UNCERTAIN) for i in range(9)
    ]
    assessment = _assess(screenings=screenings)
    assert AdequacyReasonCode.EXCESSIVE_UNCERTAINTY in _codes(assessment)


def test_budget_truncation_is_named_in_the_diagnosis() -> None:
    screenings = [
        _screening(s, ScreeningDecision.RELEVANT) for s in _SOURCES[:4]
    ]
    assessment = _assess(
        screenings=screenings,
        coverage=_coverage(screening_truncated=20),
    )
    detail = next(
        r.detail
        for r in assessment.reasons
        if r.code is AdequacyReasonCode.TOO_FEW_RELEVANT
    )
    assert "budget-truncated" in detail


def test_a_cross_paper_claim_on_one_source_fails() -> None:
    """Defense in depth below the inventory gate: a conflicting-findings
    problem that does not actually span two distinct sources."""
    inventory = _inventory(
        _problem("a multi-source problem", supporting=("s1", "s2")),
        _problem(
            "papers allegedly disagree",
            kind=ProblemKind.CONFLICTING_FINDINGS,
            supporting=("s3",),
            conflicting=(),
        ),
    )
    assessment = _assess(inventory=inventory)
    assert AdequacyReasonCode.CROSS_PAPER_PATTERN_UNSUPPORTED in _codes(
        assessment
    )


# -- support tiers -------------------------------------------------------------


def test_support_tiers_distinguish_the_required_kinds() -> None:
    single_limitation = _problem(
        "one paper's data limitation",
        kind=ProblemKind.DATA_LIMITATION,
        supporting=("s1",),
    )
    tentative = _problem("an inferred gap", supporting=("s1",))
    multi = _problem("independently reported", supporting=("s1", "s2"))
    contradicted = _problem(
        "contested finding",
        kind=ProblemKind.CONFLICTING_FINDINGS,
        supporting=("s1",),
        conflicting=("s2",),
    )
    assert support_tier(single_limitation) is (
        SupportTier.SINGLE_SOURCE_LIMITATION
    )
    assert support_tier(tentative) is SupportTier.TENTATIVE
    assert support_tier(multi) is SupportTier.MULTI_SOURCE
    assert support_tier(contradicted) is SupportTier.CONTRADICTED


def test_single_source_limitations_are_not_field_wide_consensus() -> None:
    """A one-source limitation may stay in the inventory, but it cannot
    satisfy the cross-paper support requirement by itself."""
    inventory = _inventory(
        _problem(
            "one paper's compute limitation",
            kind=ProblemKind.COMPUTE_LIMITATION,
            supporting=("s1",),
        ),
        _problem("an inferred gap", supporting=("s2",)),
    )
    assessment = _assess(inventory=inventory)
    assert AdequacyReasonCode.PROBLEM_SUPPORT_THIN in _codes(assessment)
    tiers = {p.tier for p in assessment.problem_support}
    assert tiers == {
        SupportTier.SINGLE_SOURCE_LIMITATION,
        SupportTier.TENTATIVE,
    }


def test_a_contradiction_counts_as_cross_paper_support() -> None:
    inventory = _inventory(
        _problem(
            "contested finding",
            kind=ProblemKind.CONFLICTING_FINDINGS,
            supporting=("s1",),
            conflicting=("s2",),
        ),
    )
    assessment = _assess(inventory=inventory)
    assert AdequacyReasonCode.PROBLEM_SUPPORT_THIN not in _codes(assessment)
    assert assessment.metrics.contradicted_problems == 1


# -- durability and the guard --------------------------------------------------


def test_the_assessment_survives_reload_with_identical_verdict(
    tmp_path: Path,
) -> None:
    store = MappingStore(tmp_path)
    insufficient = _assess(
        screenings=[_screening("s1", ScreeningDecision.RELEVANT)]
    )
    store.record_adequacy(insufficient)

    fresh = MappingStore(tmp_path)
    reloaded = fresh.get_adequacy(insufficient.id)
    assert reloaded == insufficient
    assert reloaded is not None
    assert reloaded.status is AdequacyStatus.INSUFFICIENT_COVERAGE
    assert reloaded.reasons == insufficient.reasons
    assert fresh.adequacy_for_run(RUN) == insufficient


def test_a_tampered_assessment_fails_on_load(tmp_path: Path) -> None:
    store = MappingStore(tmp_path)
    assessment = _assess()
    store.record_adequacy(assessment)
    path = tmp_path / "assessments" / f"{assessment.id}.json"
    payload = json.loads(path.read_text())
    payload["status"] = "adequate_for_idea_generation"
    payload["reasons"] = []
    path.write_text(json.dumps(payload))
    # (The original was already adequate; flip a metric instead.)
    payload["metrics"]["relevant_sources"] = 999
    path.write_text(json.dumps(payload))

    with pytest.raises(MappingIntegrityError, match="re-derives"):
        MappingStore(tmp_path).get_adequacy(assessment.id)


def test_the_task_5c_guard_opens_only_for_adequate_maps(
    tmp_path: Path,
) -> None:
    store = MappingStore(tmp_path)

    with pytest.raises(InadequateFieldMapError, match="no adequacy"):
        require_adequate_for_idea_generation(store, "madq_missing")

    adequate = _assess()
    store.record_adequacy(adequate)
    assert (
        require_adequate_for_idea_generation(store, adequate.id)
        == adequate
    )

    insufficient = _assess(
        screenings=[_screening("s1", ScreeningDecision.RELEVANT)]
    )
    store.record_adequacy(insufficient)
    with pytest.raises(InadequateFieldMapError) as caught:
        require_adequate_for_idea_generation(store, insufficient.id)
    assert caught.value.reasons == insufficient.reasons
    assert AdequacyReasonCode.TOO_FEW_RELEVANT in {
        r.code for r in caught.value.reasons
    }


def test_thresholds_are_validated() -> None:
    with pytest.raises(ValueError):
        AdequacyThresholds(min_relevant_sources=0)
    with pytest.raises(ValueError):
        AdequacyThresholds(max_uncertain_fraction=0.0)
    with pytest.raises(ValueError):
        AdequacyThresholds(max_uncertain_fraction=1.5)
