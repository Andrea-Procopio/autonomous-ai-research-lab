"""The deterministic prior-art gates: query discipline, screening held
to the rendered universe, comparisons held verbatim to accessible text,
and the Task 5C.1 identifier lesson carried forward."""

from __future__ import annotations

from collections.abc import Mapping

from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureSource,
)
from autonomous_research_lab.mapping.gates import MappingRejection
from autonomous_research_lab.priorart.gates import (
    check_comparisons,
    check_prior_art_queries,
    check_similarity_screening,
)
from autonomous_research_lab.priorart.records import (
    DIMENSIONS,
)

CUTOFF = "2026-08-18"

ABSTRACT = (
    "We prune attention heads after training and report a 12.5 point "
    "gain over the dense baseline on 3 synthetic tasks."
)


def _source(**overrides: object) -> LiteratureSource:
    defaults: dict[str, object] = {
        "provider": "scripted",
        "provider_id": "W1",
        "title": "Pruning Attention Heads for Efficient Inference",
        "authors": ("Ada Lovelace",),
        "publication_date": "2026-01-15",
        "publication_year": 2026,
        "venue": "Journal of Examples",
        "work_type": "article",
        "abstract": ABSTRACT,
        "doi": None,
        "arxiv_id": None,
        "provider_url": "https://example.org/W1",
        "landing_page_url": None,
        "pdf_url": None,
        "cited_by_count": 10,
        "referenced_work_ids": (),
        "access_level": AccessLevel.ABSTRACT,
    }
    defaults.update(overrides)
    return LiteratureSource(**defaults)  # type: ignore[arg-type]


def _rules(rejections: tuple[MappingRejection, ...]) -> set[str]:
    return {entry.rule for entry in rejections}


# -- the query gate -----------------------------------------------------------


def _queries_payload(
    **replacements: str,
) -> dict[str, list[dict[str, str]]]:
    texts = {
        "mechanism": "attention head reweighting scalars",
        "problem_mechanism": "in-context learning head reweighting",
        "evaluation_setup": "prefix matching copying accuracy benchmark",
        "synonyms_legacy": "soft head masking gating",
        "competing_approaches": "head pruning LoRA adapters",
        "recent": "attention head reweighting",
    }
    texts.update(replacements)
    return {
        "queries": [
            {"family": family, "text": text}
            for family, text in texts.items()
        ]
    }


def test_a_full_query_slate_passes() -> None:
    assert (
        check_prior_art_queries(
            _queries_payload(), max_queries_per_family=1
        )
        == ()
    )


def test_a_missing_family_is_rejected() -> None:
    payload = {
        "queries": [
            entry
            for entry in _queries_payload()["queries"]
            if entry["family"] != "synonyms_legacy"
        ]
    }
    rejections = check_prior_art_queries(payload, max_queries_per_family=1)
    assert _rules(rejections) == {"missing_family"}
    assert "synonyms_legacy" in rejections[0].detail


def test_an_over_budget_family_is_rejected() -> None:
    payload = {
        "queries": [
            *_queries_payload()["queries"],
            {"family": "recent", "text": "another recent query"},
        ]
    }
    assert _rules(
        check_prior_art_queries(payload, max_queries_per_family=1)
    ) == {"budget_violation"}


def test_duplicate_queries_are_rejected() -> None:
    payload = {
        "queries": [
            *_queries_payload()["queries"],
            {"family": "recent", "text": "Attention  Head\tReweighting"},
        ]
    }
    assert "duplicate_finding" in _rules(
        check_prior_art_queries(payload, max_queries_per_family=2)
    )


def test_query_text_bounds_are_enforced() -> None:
    assert "empty_finding" in _rules(
        check_prior_art_queries(
            _queries_payload(recent="  "), max_queries_per_family=1
        )
    )
    assert "malformed_finding" in _rules(
        check_prior_art_queries(
            _queries_payload(recent="x" * 301), max_queries_per_family=1
        )
    )


def test_control_characters_in_queries_are_rejected() -> None:
    assert "corrupted_text" in _rules(
        check_prior_art_queries(
            _queries_payload(recent="reweighting\x00heads"),
            max_queries_per_family=1,
        )
    )


# -- the screening gate -------------------------------------------------------


def _screens_payload(
    *entries: Mapping[str, str],
) -> Mapping[str, object]:
    return {"screens": list(entries)}


def _screen(
    source_id: str = "lit_1",
    decision: str = "related",
    reason: str = "also intervenes on attention heads",
) -> Mapping[str, str]:
    return {"source_id": source_id, "decision": decision, "reason": reason}


ACCESSIBLE = {"lit_1": ABSTRACT.casefold(), "lit_2": "another abstract"}


def test_a_grounded_screen_passes() -> None:
    assert (
        check_similarity_screening(
            _screens_payload(_screen(), _screen(source_id="lit_2")),
            accessible=ACCESSIBLE,
            candidate_tokens=frozenset(),
            known_ids=frozenset(ACCESSIBLE),
        )
        == ()
    )


def test_an_unknown_source_screen_is_rejected() -> None:
    rejections = check_similarity_screening(
        _screens_payload(
            _screen(),
            _screen(source_id="lit_2"),
            _screen(source_id="lit_9"),
        ),
        accessible=ACCESSIBLE,
        candidate_tokens=frozenset(),
        known_ids=frozenset(ACCESSIBLE),
    )
    assert _rules(rejections) == {"unknown_source"}


def test_every_rendered_source_is_screened_exactly_once() -> None:
    assert _rules(
        check_similarity_screening(
            _screens_payload(_screen()),
            accessible=ACCESSIBLE,
            candidate_tokens=frozenset(),
            known_ids=frozenset(ACCESSIBLE),
        )
    ) == {"missing_decision"}
    assert "duplicate_finding" in _rules(
        check_similarity_screening(
            _screens_payload(
                _screen(), _screen(), _screen(source_id="lit_2")
            ),
            accessible=ACCESSIBLE,
            candidate_tokens=frozenset(),
            known_ids=frozenset(ACCESSIBLE),
        )
    )


def test_a_screen_reason_may_not_claim_novelty() -> None:
    rejections = check_similarity_screening(
        _screens_payload(
            _screen(reason="no prior work resembles the candidate"),
            _screen(source_id="lit_2"),
        ),
        accessible=ACCESSIBLE,
        candidate_tokens=frozenset(),
        known_ids=frozenset(ACCESSIBLE),
    )
    assert "novelty_claim" in _rules(rejections)


def test_ungrounded_numbers_in_reasons_are_rejected() -> None:
    grounded = check_similarity_screening(
        _screens_payload(
            _screen(reason="reports a 12.5 point gain like the candidate"),
            _screen(source_id="lit_2"),
        ),
        accessible=ACCESSIBLE,
        candidate_tokens=frozenset(),
        known_ids=frozenset(ACCESSIBLE),
    )
    assert grounded == ()
    rejections = check_similarity_screening(
        _screens_payload(
            _screen(reason="reports a 99 point gain"),
            _screen(source_id="lit_2"),
        ),
        accessible=ACCESSIBLE,
        candidate_tokens=frozenset(),
        known_ids=frozenset(ACCESSIBLE),
    )
    assert _rules(rejections) == {"ungrounded_number"}


# -- the comparison gate ------------------------------------------------------


def _dimension_entry(
    dimension: str,
    *,
    snippet: str = "prune attention heads",
    location: str = "abstract",
) -> dict[str, str]:
    return {
        "dimension": dimension,
        "candidate_position": "reweights attention heads directly",
        "prior_work_position": "prunes attention heads after training",
        "support_location": location,
        "support_snippet": snippet,
    }


def _comparison_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "source_id": "lit_1",
        "similarity": "related",
        "overlap_features": ["both intervene on attention heads"],
        "material_differences": ["reweighting versus pruning"],
        "dimensions": [
            _dimension_entry(dimension.value) for dimension in DIMENSIONS
        ],
    }
    entry.update(overrides)
    return entry


SOURCES = {"lit_1": _source()}


def _check(
    entry: dict[str, object],
    *,
    sources: Mapping[str, LiteratureSource] = SOURCES,
    candidate_tokens: frozenset[str] = frozenset(),
    known_ids: frozenset[str] = frozenset(SOURCES),
) -> tuple[MappingRejection, ...]:
    return check_comparisons(
        {"comparisons": [entry]},
        sources=sources,
        cutoff_date=CUTOFF,
        candidate_tokens=candidate_tokens,
        known_ids=known_ids,
    )


def test_a_grounded_comparison_passes() -> None:
    assert _check(_comparison_entry()) == ()


def test_an_unknown_work_comparison_is_rejected() -> None:
    rejections = _check(_comparison_entry(source_id="lit_9"))
    assert "unknown_source" in _rules(rejections)
    # The rendered work is then also uncompared.
    assert "missing_decision" in _rules(rejections)


def test_every_rendered_work_is_compared_exactly_once() -> None:
    assert _rules(
        check_comparisons(
            {"comparisons": []},
            sources=SOURCES,
            cutoff_date=CUTOFF,
            candidate_tokens=frozenset(),
            known_ids=frozenset(),
        )
    ) == {"missing_decision"}
    rejections = check_comparisons(
        {"comparisons": [_comparison_entry(), _comparison_entry()]},
        sources=SOURCES,
        cutoff_date=CUTOFF,
        candidate_tokens=frozenset(),
        known_ids=frozenset(),
    )
    assert "duplicate_finding" in _rules(rejections)
    assert "budget_violation" in _rules(rejections)


def test_a_missing_dimension_is_rejected() -> None:
    entry = _comparison_entry(
        dimensions=[
            _dimension_entry(dimension.value)
            for dimension in DIMENSIONS[:-1]
        ]
    )
    rejections = _check(entry)
    assert "missing_dimension" in _rules(rejections)
    assert "claimed_contribution" in rejections[-1].detail


def test_a_snippet_absent_from_the_named_part_is_unsupported() -> None:
    entry = _comparison_entry(
        dimensions=[
            _dimension_entry(
                dimension.value,
                snippet=(
                    "a sweeping conclusion the abstract never states"
                    if dimension is DIMENSIONS[0]
                    else "prune attention heads"
                ),
            )
            for dimension in DIMENSIONS
        ]
    )
    assert "unsupported_claim" in _rules(_check(entry))


def test_a_title_snippet_must_come_from_the_title() -> None:
    good = _comparison_entry(
        dimensions=[
            _dimension_entry(
                dimension.value,
                snippet=(
                    "Pruning Attention Heads"
                    if dimension is DIMENSIONS[0]
                    else "prune attention heads"
                ),
                location=(
                    "title" if dimension is DIMENSIONS[0] else "abstract"
                ),
            )
            for dimension in DIMENSIONS
        ]
    )
    assert _check(good) == ()
    bad = _comparison_entry(
        dimensions=[
            _dimension_entry(
                dimension.value,
                # In the abstract, not the title.
                snippet=(
                    "dense baseline"
                    if dimension is DIMENSIONS[0]
                    else "prune attention heads"
                ),
                location=(
                    "title" if dimension is DIMENSIONS[0] else "abstract"
                ),
            )
            for dimension in DIMENSIONS
        ]
    )
    assert "unsupported_claim" in _rules(_check(bad))


def test_unretrieved_support_locations_are_access_mismatches() -> None:
    full_text = _comparison_entry(
        dimensions=[
            _dimension_entry(
                dimension.value,
                location=(
                    "full_text" if dimension is DIMENSIONS[0] else "abstract"
                ),
            )
            for dimension in DIMENSIONS
        ]
    )
    assert "access_level_mismatch" in _rules(_check(full_text))
    metadata_only = {
        "lit_1": _source(abstract=None, access_level=AccessLevel.METADATA)
    }
    assert "access_level_mismatch" in _rules(
        _check(_comparison_entry(), sources=metadata_only)
    )


def test_a_post_cutoff_source_is_rejected() -> None:
    late = {"lit_1": _source(publication_date="2026-08-19")}
    assert "post_cutoff_source" in _rules(
        _check(_comparison_entry(), sources=late)
    )
    undated = {"lit_1": _source(publication_date=None)}
    assert "post_cutoff_source" not in _rules(
        _check(_comparison_entry(), sources=undated)
    )


def test_a_match_without_features_is_ungrounded_overlap() -> None:
    entry = _comparison_entry(
        similarity="substantial_match", overlap_features=[]
    )
    assert "ungrounded_overlap" in _rules(_check(entry))


def test_a_distinction_without_differences_contradicts_itself() -> None:
    entry = _comparison_entry(similarity="distinct", material_differences=[])
    assert "similarity_contradiction" in _rules(_check(entry))
    related = _comparison_entry(similarity="related", material_differences=[])
    assert "similarity_contradiction" in _rules(_check(related))


def test_known_ids_are_stripped_before_number_extraction() -> None:
    # The Task 5C.1 regression, mirrored: a known source id pasted into
    # prose must not fire on fragments of its own hex.
    entry = _comparison_entry(
        overlap_features=["both intervene on heads, as lit_452f82f87 shows"]
    )
    assert (
        _check(entry, known_ids=frozenset({"lit_452f82f87"})) == ()
    )


def test_a_fabricated_id_still_reads_as_ungrounded_numbers() -> None:
    entry = _comparison_entry(
        overlap_features=["both intervene on heads, as lit_452f82f87 shows"]
    )
    assert "ungrounded_number" in _rules(_check(entry, known_ids=frozenset()))


def test_prior_work_numbers_are_held_to_the_source_alone() -> None:
    candidate_tokens = frozenset({"7"})
    entry = _comparison_entry(
        dimensions=[
            _dimension_entry(dimension.value)
            | {
                "prior_work_position": (
                    "prunes attention heads across 7 settings"
                )
            }
            for dimension in DIMENSIONS
        ]
    )
    # A candidate-side number cannot launder a claim about the source.
    assert "ungrounded_number" in _rules(
        _check(entry, candidate_tokens=candidate_tokens)
    )
    differences = _comparison_entry(
        material_differences=["the candidate spans 7 settings, this work 3"]
    )
    assert _check(differences, candidate_tokens=candidate_tokens) == ()


def test_material_differences_may_not_claim_novelty() -> None:
    entry = _comparison_entry(
        material_differences=["the candidate is novel in its reweighting"]
    )
    assert "novelty_claim" in _rules(_check(entry))


def test_all_fired_rules_are_returned_together() -> None:
    entry = _comparison_entry(
        similarity="substantial_match",
        overlap_features=[],
        material_differences=[],
        dimensions=[
            _dimension_entry(
                dimension.value,
                snippet=(
                    "never stated"
                    if dimension is DIMENSIONS[0]
                    else "prune attention heads"
                ),
            )
            for dimension in DIMENSIONS[:-1]
        ],
    )
    rules = _rules(_check(entry))
    assert {
        "ungrounded_overlap",
        "missing_dimension",
        "unsupported_claim",
    } <= rules
