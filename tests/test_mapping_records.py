"""The mapping vocabulary: bounded briefs, trusted era classification,
and record invariants. All fixtures are synthetic; no network, no model."""

from __future__ import annotations

import pytest

from autonomous_research_lab.mapping.brief import (
    QueryFamily,
    ResearchBrief,
    SourceEra,
    classify_era,
)
from autonomous_research_lab.mapping.records import (
    CLAIM_KINDS,
    CallProvenance,
    ExtractionRecord,
    ProblemEntry,
    ProblemKind,
    ScreeningDecision,
    ScreeningRecord,
    SupportLocation,
)


def _brief(**overrides: object) -> ResearchBrief:
    defaults: dict[str, object] = {
        "topic": "in-context learning",
        "cutoff_date": "2026-08-18",
        "recent_window_start": "2026-01-01",
    }
    defaults.update(overrides)
    return ResearchBrief(**defaults)  # type: ignore[arg-type]


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


def _extraction(**overrides: object) -> ExtractionRecord:
    defaults: dict[str, object] = {
        "run_id": "map_1",
        "source_id": "lit_1",
        "era": SourceEra.RECENT,
        "access_level": "abstract",
        "support_location": SupportLocation.ABSTRACT,
        "sufficient_support": True,
        "insufficiency_reason": "",
        "methods": ("prompt adaptation",),
        "datasets": (),
        "metrics": (),
        "evaluation_protocols": (),
        "baselines": (),
        "reported_results": (),
        "limitations": (),
        "future_work": (),
        "open_problems": (),
        "provenance": _provenance(),
    }
    defaults.update(overrides)
    return ExtractionRecord(**defaults)  # type: ignore[arg-type]


# -- the brief ----------------------------------------------------------------


def test_a_brief_is_bounded_and_content_addressed() -> None:
    first = _brief(workshop_hints=("efficient adaptation",))
    again = _brief(workshop_hints=("efficient adaptation",))
    assert first.id == again.id
    assert first.id.startswith("brief_")
    assert _brief(topic="other").id != first.id


@pytest.mark.parametrize(
    "overrides",
    [
        {"topic": "  "},
        {"cutoff_date": "August 2026"},
        {"recent_window_start": "2026-09-01"},  # window starts after cutoff
        {"workshop_hints": ("",)},
        {"max_queries_per_family": 0},
        {"max_queries_per_family": 99},
        {"results_per_query": 0},
        {"results_per_query": 1000},
        {"max_screened_sources": 0},
        {"max_extracted_sources": 10_000},
        {"max_model_calls": 0},
        {"max_model_calls": 10_000},
    ],
)
def test_an_unbounded_or_malformed_brief_cannot_be_built(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _brief(**overrides)


def test_trusted_date_ranges_per_family() -> None:
    brief = _brief()
    assert brief.date_range(QueryFamily.RECENT) == ("2026-01-01", "2026-08-18")
    assert brief.date_range(QueryFamily.FOUNDATIONAL) == ("", "2025-12-31")
    assert brief.date_range(QueryFamily.METHODS) == ("", "2026-08-18")


def test_era_classification_is_deterministic_and_honest() -> None:
    brief = _brief()
    assert classify_era("2026-03-01", brief) is SourceEra.RECENT
    assert classify_era("2026-01-01", brief) is SourceEra.RECENT  # inclusive
    assert classify_era("2025-12-31", brief) is SourceEra.FOUNDATIONAL
    assert classify_era("2019-06-01", brief) is SourceEra.FOUNDATIONAL
    assert classify_era(None, brief) is SourceEra.UNDATED


# -- records ------------------------------------------------------------------


def test_every_model_authored_category_carries_a_claim_kind() -> None:
    """The epistemic labels are structural: each category is stamped by
    trusted code, and the vocabulary distinguishes the required kinds."""
    kinds = set(CLAIM_KINDS.values())
    assert {
        "author_reported_claim",
        "author_reported_limitation",
        "mapper_synthesis",
        "inferred_open_problem",
    } <= kinds
    assert CLAIM_KINDS["extraction.limitations"] == "author_reported_limitation"
    assert CLAIM_KINDS["inventory.problems"] == "inferred_open_problem"


def test_a_screening_record_requires_a_reason() -> None:
    with pytest.raises(ValueError):
        ScreeningRecord(
            run_id="map_1",
            source_id="lit_1",
            decision=ScreeningDecision.RELEVANT,
            reason="  ",
            provenance=_provenance(),
        )


def test_an_extraction_cannot_claim_support_and_extract_nothing() -> None:
    with pytest.raises(ValueError):
        _extraction(methods=())


def test_an_insufficiency_record_asserts_nothing() -> None:
    honest = _extraction(
        sufficient_support=False,
        insufficiency_reason="abstract reports no methods or results",
        methods=(),
    )
    assert honest.methods == ()

    with pytest.raises(ValueError):
        _extraction(
            sufficient_support=False,
            insufficiency_reason="reason",
            methods=("smuggled claim",),
        )
    with pytest.raises(ValueError):
        _extraction(
            sufficient_support=False, insufficiency_reason="  ", methods=()
        )


def test_extraction_identity_is_deterministic() -> None:
    assert _extraction().id == _extraction().id
    assert _extraction(methods=("other method",)).id != _extraction().id


def test_a_problem_requires_support() -> None:
    with pytest.raises(ValueError):
        ProblemEntry(
            statement="robustness under distribution shift is unresolved",
            kind=ProblemKind.OPEN_PROBLEM,
            grounding="both papers report it",
            supporting_source_ids=(),
        )
