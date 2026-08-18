"""The deterministic mapping gates, rule by rule. All payloads are
synthetic; the sources are static Task 5A records. No network, no model."""

from __future__ import annotations

from collections.abc import Mapping

from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureSource,
)
from autonomous_research_lab.mapping.brief import ResearchBrief, SourceEra
from autonomous_research_lab.mapping.gates import (
    check_extraction,
    check_field_map,
    check_inventory,
    check_queries,
    check_screening,
)

BRIEF = ResearchBrief(
    topic="in-context learning",
    cutoff_date="2026-08-18",
    recent_window_start="2026-01-01",
    max_queries_per_family=2,
)

ABSTRACT = (
    "We study prompt adaptation for in-context learning. On the GLUE "
    "benchmark (v2, validation split, https://gluebenchmark.com, MIT "
    "license) our method reaches 88.5 accuracy over 3 seeds, but "
    "degrades under distribution shift."
)


def _source(**overrides: object) -> LiteratureSource:
    defaults: dict[str, object] = {
        "provider": "openalex",
        "provider_id": "W1",
        "title": "Prompt Adaptation for In-Context Learning",
        "authors": ("Ada Lovelace",),
        "publication_date": "2026-07-01",
        "publication_year": 2026,
        "venue": "Journal of Things",
        "work_type": "article",
        "abstract": ABSTRACT,
        "doi": None,
        "arxiv_id": None,
        "provider_url": "https://openalex.org/W1",
        "landing_page_url": None,
        "pdf_url": None,
        "cited_by_count": None,
        "referenced_work_ids": (),
        "access_level": AccessLevel.ABSTRACT,
    }
    defaults.update(overrides)
    return LiteratureSource(**defaults)  # type: ignore[arg-type]


SOURCE = _source()


def _rules(rejections: tuple[object, ...]) -> set[str]:
    return {r.rule for r in rejections}  # type: ignore[attr-defined]


# -- queries ------------------------------------------------------------------


def _queries_payload(*pairs: tuple[str, str]) -> Mapping[str, object]:
    return {
        "queries": [
            {"family": family, "text": text} for family, text in pairs
        ]
    }


_VALID_QUERIES = _queries_payload(
    ("recent", "in-context learning"),
    ("foundational", "meta-learning"),
    ("limitations_open_problems", "in-context learning limitations"),
)


def test_a_valid_query_proposal_passes() -> None:
    assert check_queries(_VALID_QUERIES, brief=BRIEF) == ()


def test_missing_required_families_are_rejected() -> None:
    payload = _queries_payload(("recent", "in-context learning"))
    assert "missing_family" in _rules(check_queries(payload, brief=BRIEF))


def test_query_budget_and_duplicates_are_rejected() -> None:
    over = _queries_payload(
        ("recent", "a query"),
        ("recent", "b query"),
        ("recent", "c query"),
        ("foundational", "meta-learning"),
        ("limitations_open_problems", "limits"),
    )
    assert "budget_violation" in _rules(check_queries(over, brief=BRIEF))

    duplicated = _queries_payload(
        ("recent", "In-Context Learning"),
        ("recent", "in-context learning"),
        ("foundational", "meta-learning"),
        ("limitations_open_problems", "limits"),
    )
    assert "duplicate_finding" in _rules(
        check_queries(duplicated, brief=BRIEF)
    )


def test_unusable_query_text_is_rejected() -> None:
    payload = _queries_payload(
        ("recent", "  "),
        ("foundational", "meta-learning"),
        ("limitations_open_problems", "x" * 400),
    )
    rules = _rules(check_queries(payload, brief=BRIEF))
    assert "empty_finding" in rules
    assert "budget_violation" in rules


# -- screening ----------------------------------------------------------------


def _decision(
    source_id: str, decision: str = "relevant", reason: str = "on topic"
) -> dict[str, object]:
    return {"source_id": source_id, "decision": decision, "reason": reason}


def test_a_complete_screening_batch_passes() -> None:
    payload = {"decisions": [_decision("lit_a"), _decision("lit_b", "excluded")]}
    assert (
        check_screening(payload, expected_source_ids=("lit_a", "lit_b")) == ()
    )


def test_screening_must_cover_the_batch_exactly_once() -> None:
    payload = {"decisions": [_decision("lit_a"), _decision("lit_a")]}
    rules = _rules(
        check_screening(payload, expected_source_ids=("lit_a", "lit_b"))
    )
    assert "duplicate_finding" in rules
    assert "missing_decision" in rules

    stray = {"decisions": [_decision("lit_zzz")]}
    rules = _rules(check_screening(stray, expected_source_ids=("lit_a",)))
    assert "unknown_source" in rules
    assert "missing_decision" in rules


def test_screening_reasons_are_required_and_modest() -> None:
    payload = {
        "decisions": [
            _decision("lit_a", reason="  "),
            _decision("lit_b", reason="exhaustive coverage of the field"),
        ]
    }
    rules = _rules(
        check_screening(payload, expected_source_ids=("lit_a", "lit_b"))
    )
    assert "empty_finding" in rules
    assert "coverage_language" in rules


# -- extraction ---------------------------------------------------------------


def _extraction_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_id": SOURCE.id,
        "support_location": "abstract",
        "sufficient_support": True,
        "insufficiency_reason": "",
        "methods": ["prompt adaptation"],
        "datasets": [
            {
                "name": "GLUE",
                "task": "language understanding",
                "version": "v2",
                "split": "validation",
                "subset": "",
                "preprocessing": "",
                "size": "",
                "availability": "public",
                "url": "https://gluebenchmark.com",
                "license": "MIT",
            }
        ],
        "metrics": ["accuracy"],
        "evaluation_protocols": ["evaluation over 3 seeds"],
        "baselines": [],
        "reported_results": ["reaches 88.5 accuracy on GLUE"],
        "limitations": [
            {"text": "degrades under distribution shift", "kind": "generalization"}
        ],
        "future_work": [],
        "open_problems": ["robustness under distribution shift"],
    }
    payload.update(overrides)
    return payload


def test_a_grounded_extraction_passes() -> None:
    assert check_extraction(_extraction_payload(), source=SOURCE) == ()


def test_the_wrong_source_id_is_rejected() -> None:
    payload = _extraction_payload(source_id="lit_other")
    assert "unknown_source" in _rules(
        check_extraction(payload, source=SOURCE)
    )


def test_access_level_mismatches_fail_closed() -> None:
    full_text = _extraction_payload(support_location="full_text")
    assert "access_level_mismatch" in _rules(
        check_extraction(full_text, source=SOURCE)
    )

    metadata_only = _source(
        provider_id="W2", abstract=None, access_level=AccessLevel.METADATA
    )
    payload = _extraction_payload(source_id=metadata_only.id)
    assert "access_level_mismatch" in _rules(
        check_extraction(payload, source=metadata_only)
    )


def test_ungrounded_numbers_are_rejected() -> None:
    payload = _extraction_payload(
        reported_results=["reaches 97.1 accuracy on GLUE"]
    )
    assert "ungrounded_number" in _rules(
        check_extraction(payload, source=SOURCE)
    )


def test_a_number_inside_a_larger_number_is_not_grounding() -> None:
    """The abstract says 88.5; a claim of 88 is not the same token."""
    payload = _extraction_payload(reported_results=["reaches 88 accuracy"])
    assert "ungrounded_number" in _rules(
        check_extraction(payload, source=SOURCE)
    )


def test_unsupported_dataset_details_are_rejected() -> None:
    fabricated_name = _extraction_payload(
        datasets=[
            {
                "name": "SuperDuperBench",
                "task": "",
                "version": "",
                "split": "",
                "subset": "",
                "preprocessing": "",
                "size": "",
                "availability": "unreported",
                "url": "",
                "license": "",
            }
        ]
    )
    assert "unsupported_claim" in _rules(
        check_extraction(fabricated_name, source=SOURCE)
    )

    fabricated_detail = _extraction_payload()
    datasets = fabricated_detail["datasets"]
    assert isinstance(datasets, list)
    datasets[0]["license"] = "Apache-2.0"
    rules = _rules(check_extraction(fabricated_detail, source=SOURCE))
    # The factual detail fields get the stronger verbatim check, which
    # subsumes number grounding.
    assert "unsupported_claim" in rules


def test_a_non_http_dataset_url_is_malformed() -> None:
    payload = _extraction_payload()
    datasets = payload["datasets"]
    assert isinstance(datasets, list)
    datasets[0]["url"] = "gluebenchmark.com"
    assert "malformed_finding" in _rules(
        check_extraction(payload, source=SOURCE)
    )


def test_duplicate_findings_are_rejected() -> None:
    payload = _extraction_payload(
        methods=["prompt adaptation", "Prompt  Adaptation"]
    )
    assert "duplicate_finding" in _rules(
        check_extraction(payload, source=SOURCE)
    )


def test_the_insufficiency_outcome_is_all_or_nothing() -> None:
    honest = _extraction_payload(
        sufficient_support=False,
        insufficiency_reason="the abstract reports no substantive detail",
        methods=[],
        datasets=[],
        metrics=[],
        evaluation_protocols=[],
        baselines=[],
        reported_results=[],
        limitations=[],
        future_work=[],
        open_problems=[],
    )
    assert check_extraction(honest, source=SOURCE) == ()

    smuggled = _extraction_payload(
        sufficient_support=False,
        insufficiency_reason="nothing here",
    )
    assert "conflicting_record" in _rules(
        check_extraction(smuggled, source=SOURCE)
    )

    contradictory = _extraction_payload(insufficiency_reason="but also this")
    assert "conflicting_record" in _rules(
        check_extraction(contradictory, source=SOURCE)
    )

    empty_but_confident = _extraction_payload(
        methods=[],
        datasets=[],
        metrics=[],
        evaluation_protocols=[],
        baselines=[],
        reported_results=[],
        limitations=[],
        future_work=[],
        open_problems=[],
    )
    assert "empty_finding" in _rules(
        check_extraction(empty_but_confident, source=SOURCE)
    )


def test_coverage_language_is_rejected_everywhere() -> None:
    payload = _extraction_payload(
        open_problems=["a systematic review would settle this"]
    )
    assert "coverage_language" in _rules(
        check_extraction(payload, source=SOURCE)
    )


# -- the field map ------------------------------------------------------------


_ERAS = {"lit_r": SourceEra.RECENT, "lit_f": SourceEra.FOUNDATIONAL}
_ACCESSIBLE = {
    "lit_r": "prompt adaptation reaches 88.5 accuracy",
    "lit_f": "meta-learning with 3 seeds",
}


def _field_map_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "themes": [
            {
                "name": "Prompt adaptation",
                "summary": "Recent adaptation methods.",
                "era": "recent",
                "source_ids": ["lit_r"],
            },
            {
                "name": "Meta-learning foundations",
                "summary": "Foundational episodic training.",
                "era": "foundational",
                "source_ids": ["lit_f"],
            },
        ],
        "approaches": [
            {
                "name": "Gradient-free adaptation",
                "summary": "Adaptation without weight updates.",
                "source_ids": ["lit_r"],
            }
        ],
        "evaluation_practices": [],
        "relationships": [
            {
                "kind": "builds_on",
                "from_theme": "Prompt adaptation",
                "to_theme": "Meta-learning foundations",
                "note": "adaptation reuses episodic ideas",
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_a_grounded_field_map_passes() -> None:
    assert (
        check_field_map(
            _field_map_payload(), eras=_ERAS, accessible=_ACCESSIBLE
        )
        == ()
    )


def test_unknown_sources_and_missing_support_are_rejected() -> None:
    unknown = _field_map_payload(
        approaches=[
            {"name": "X", "summary": "Y.", "source_ids": ["lit_ghost"]}
        ]
    )
    assert "unknown_source" in _rules(
        check_field_map(unknown, eras=_ERAS, accessible=_ACCESSIBLE)
    )

    ungrounded = _field_map_payload(
        approaches=[{"name": "X", "summary": "Y.", "source_ids": []}]
    )
    assert "missing_support" in _rules(
        check_field_map(ungrounded, eras=_ERAS, accessible=_ACCESSIBLE)
    )


def test_era_claims_must_match_the_trusted_classification() -> None:
    payload = _field_map_payload()
    themes = payload["themes"]
    assert isinstance(themes, list)
    themes[0]["era"] = "foundational"  # cited source classifies as recent
    assert "era_mismatch" in _rules(
        check_field_map(payload, eras=_ERAS, accessible=_ACCESSIBLE)
    )

    mixed = _field_map_payload()
    themes = mixed["themes"]
    assert isinstance(themes, list)
    themes[0]["source_ids"] = ["lit_r", "lit_f"]
    themes[0]["era"] = "recent"  # mixed citations classify as both
    assert "era_mismatch" in _rules(
        check_field_map(mixed, eras=_ERAS, accessible=_ACCESSIBLE)
    )


def test_duplicate_themes_and_bad_relationships_are_rejected() -> None:
    duplicated = _field_map_payload()
    themes = duplicated["themes"]
    assert isinstance(themes, list)
    themes[1]["name"] = "prompt adaptation"  # casefold duplicate
    assert "duplicate_finding" in _rules(
        check_field_map(duplicated, eras=_ERAS, accessible=_ACCESSIBLE)
    )

    dangling = _field_map_payload(
        relationships=[
            {
                "kind": "contrasts_with",
                "from_theme": "Prompt adaptation",
                "to_theme": "No Such Theme",
                "note": "n",
            }
        ]
    )
    assert "unknown_theme" in _rules(
        check_field_map(dangling, eras=_ERAS, accessible=_ACCESSIBLE)
    )

    reflexive = _field_map_payload(
        relationships=[
            {
                "kind": "builds_on",
                "from_theme": "Prompt adaptation",
                "to_theme": "Prompt adaptation",
                "note": "n",
            }
        ]
    )
    assert "conflicting_record" in _rules(
        check_field_map(reflexive, eras=_ERAS, accessible=_ACCESSIBLE)
    )


def test_field_map_numbers_must_be_grounded_in_cited_sources() -> None:
    payload = _field_map_payload()
    themes = payload["themes"]
    assert isinstance(themes, list)
    themes[0]["summary"] = "Reaches 99.9 accuracy."
    assert "ungrounded_number" in _rules(
        check_field_map(payload, eras=_ERAS, accessible=_ACCESSIBLE)
    )


# -- the inventory ------------------------------------------------------------


def _inventory_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "problems": [
            {
                "statement": "robustness under distribution shift is open",
                "kind": "open_problem",
                "grounding": "the recent paper reports degradation",
                "supporting_source_ids": ["lit_r"],
                "conflicting_source_ids": [],
            }
        ]
    }
    payload.update(overrides)
    return payload


def test_a_grounded_inventory_passes() -> None:
    assert (
        check_inventory(
            _inventory_payload(), eras=_ERAS, accessible=_ACCESSIBLE
        )
        == ()
    )


def test_inventory_reference_and_conflict_rules() -> None:
    unknown = _inventory_payload()
    problems = unknown["problems"]
    assert isinstance(problems, list)
    problems[0]["supporting_source_ids"] = ["lit_ghost"]
    assert "unknown_source" in _rules(
        check_inventory(unknown, eras=_ERAS, accessible=_ACCESSIBLE)
    )

    both_sides = _inventory_payload()
    problems = both_sides["problems"]
    assert isinstance(problems, list)
    problems[0]["conflicting_source_ids"] = ["lit_r"]
    assert "conflicting_record" in _rules(
        check_inventory(both_sides, eras=_ERAS, accessible=_ACCESSIBLE)
    )

    unconflicted = _inventory_payload()
    problems = unconflicted["problems"]
    assert isinstance(problems, list)
    problems[0]["kind"] = "conflicting_findings"
    assert "missing_support" in _rules(
        check_inventory(unconflicted, eras=_ERAS, accessible=_ACCESSIBLE)
    )


def test_inventory_statements_are_unique_grounded_and_modest() -> None:
    payload = _inventory_payload(
        problems=[
            {
                "statement": "Robustness is open",
                "kind": "open_problem",
                "grounding": "reported degradation",
                "supporting_source_ids": ["lit_r"],
                "conflicting_source_ids": [],
            },
            {
                "statement": "robustness  is open",
                "kind": "open_problem",
                "grounding": "same",
                "supporting_source_ids": ["lit_r"],
                "conflicting_source_ids": [],
            },
            {
                "statement": "a 42.7 point gap remains",
                "kind": "missing_comparison",
                "grounding": "no such number is reported",
                "supporting_source_ids": ["lit_f"],
                "conflicting_source_ids": [],
            },
            {
                "statement": "this problem has never been studied",
                "kind": "open_problem",
                "grounding": "novelty claim",
                "supporting_source_ids": ["lit_r"],
                "conflicting_source_ids": [],
            },
        ]
    )
    rules = _rules(
        check_inventory(payload, eras=_ERAS, accessible=_ACCESSIBLE)
    )
    assert "duplicate_finding" in rules
    assert "ungrounded_number" in rules
    assert "coverage_language" in rules


def test_an_empty_inventory_is_rejected() -> None:
    assert "empty_finding" in _rules(
        check_inventory(
            {"problems": []}, eras=_ERAS, accessible=_ACCESSIBLE
        )
    )


# -- transport-corrupted text (Task 5B.1, observed live) ----------------------


def test_control_characters_are_rejected_with_an_actionable_message() -> None:
    """OBSERVED live (2026-08-18): a non-ASCII dash in a year range
    arrived as U+0002, splitting '2026' into an ungroundable '026'. The
    gate must name the corruption and the fix, not just the symptom."""
    payload = _extraction_payload(
        methods=["prompt adaptation", "work from 20252026 on GLUE"]
    )
    rejections = check_extraction(payload, source=SOURCE)
    rules = {r.rule for r in rejections}
    assert "corrupted_text" in rules
    detail = next(
        r.detail for r in rejections if r.rule == "corrupted_text"
    )
    assert "U+0002" in detail
    assert "ASCII" in detail


def test_corrupted_query_text_is_rejected_before_retrieval() -> None:
    payload = _queries_payload(
        ("recent", "in-contextlearning"),
        ("foundational", "meta-learning"),
        ("limitations_open_problems", "limits of adaptation"),
    )
    assert "corrupted_text" in _rules(check_queries(payload, brief=BRIEF))
