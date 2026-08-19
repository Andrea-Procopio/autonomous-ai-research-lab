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


# -- the query-plan gate ------------------------------------------------------

#: What the candidate's rendered record contains — plan anchors are
#: checked against this.
CANDIDATE_HAYSTACK = (
    "reweights attention heads to select semantic induction heads for "
    "in-context learning via prefix matching and copying; search "
    "terms: attention head reweighting; head ablation"
)


def _group(*alternatives: str) -> dict[str, list[str]]:
    return {"alternatives": list(alternatives)}


def _plan(
    family: str, *groups: dict[str, list[str]]
) -> dict[str, object]:
    return {"family": family, "groups": list(groups)}


def _plans_payload(
    **replacements: dict[str, object],
) -> dict[str, list[dict[str, object]]]:
    plans = {
        "mechanism": _plan(
            "mechanism",
            _group("attention head reweighting", "head gating"),
            _group("induction heads"),
        ),
        "problem_mechanism": _plan(
            "problem_mechanism",
            _group("in-context learning"),
            _group("attention head reweighting"),
        ),
        "evaluation_setup": _plan(
            "evaluation_setup",
            _group("prefix matching", "copying"),
            _group("in-context learning"),
        ),
        "synonyms_legacy": _plan(
            "synonyms_legacy",
            _group("attention heads"),
            _group("soft masking", "gating", "pruning"),
        ),
        "competing_approaches": _plan(
            "competing_approaches",
            _group("attention heads"),
            _group("LoRA", "adapters", "head pruning"),
        ),
        "recent": _plan(
            "recent", _group("attention head reweighting")
        ),
    }
    plans.update(replacements)
    return {"queries": list(plans.values())}


def _check_queries(
    payload: dict[str, list[dict[str, object]]],
    *,
    max_queries_per_family: int = 1,
) -> tuple[MappingRejection, ...]:
    return check_prior_art_queries(
        payload,
        max_queries_per_family=max_queries_per_family,
        candidate_haystack=CANDIDATE_HAYSTACK,
    )


def test_a_full_plan_slate_passes() -> None:
    assert _check_queries(_plans_payload()) == ()


def test_a_missing_family_is_rejected() -> None:
    payload = {
        "queries": [
            plan
            for plan in _plans_payload()["queries"]
            if plan["family"] != "synonyms_legacy"
        ]
    }
    rejections = _check_queries(payload)
    assert _rules(rejections) == {"missing_family"}
    assert "synonyms_legacy" in rejections[0].detail


def test_an_over_budget_family_is_rejected() -> None:
    payload = {
        "queries": [
            *_plans_payload()["queries"],
            _plan("recent", _group("induction heads")),
        ]
    }
    assert _rules(_check_queries(payload)) == {"budget_violation"}


def test_a_reordered_duplicate_plan_is_still_a_duplicate() -> None:
    # Canonicalization makes ordering non-semantic: the same plan with
    # groups and alternatives shuffled is the same plan.
    payload = {
        "queries": [
            *_plans_payload()["queries"],
            _plan(
                "mechanism",
                _group("Induction Heads"),
                _group("head gating", "attention head reweighting"),
            ),
        ]
    }
    rejections = _check_queries(payload, max_queries_per_family=2)
    assert "duplicate_finding" in _rules(rejections)


def test_the_observed_failure_shape_cannot_execute() -> None:
    # The Task 5D live defect: ten-plus terms intended as one search.
    # As ten conjoined groups it is excessive conjunctivity; as one
    # opaque string it is not expressible in the schema at all.
    terms = (
        "learned",
        "attention",
        "head",
        "reweighting",
        "scalars",
        "semantic",
        "induction",
        "heads",
        "prefix matching",
        "copying",
    )
    payload = _plans_payload(
        mechanism=_plan("mechanism", *(_group(term) for term in terms))
    )
    rejections = _check_queries(payload)
    assert "excessive_conjunctivity" in _rules(rejections)


def test_group_and_term_bounds_are_enforced() -> None:
    over_alternatives = _plans_payload(
        recent=_plan(
            "recent",
            _group("attention heads", "a1", "a2", "a3", "a4"),
        )
    )
    assert "budget_violation" in _rules(_check_queries(over_alternatives))
    too_long = _plans_payload(
        recent=_plan("recent", _group("attention heads", "x" * 61))
    )
    assert "malformed_finding" in _rules(_check_queries(too_long))
    empty_group = _plans_payload(
        recent=_plan("recent", _group("attention heads"), _group())
    )
    assert "empty_finding" in _rules(_check_queries(empty_group))
    no_groups = _plans_payload(recent=_plan("recent"))
    assert "empty_finding" in _rules(_check_queries(no_groups))
    blank_term = _plans_payload(
        recent=_plan("recent", _group("attention heads", "  "))
    )
    assert "empty_finding" in _rules(_check_queries(blank_term))


def test_boolean_syntax_in_terms_is_rejected() -> None:
    for bad in (
        'induction "heads"',
        "(induction heads)",
        "induction OR heads",
        "heads AND reweighting",
        "not heads",
        "induction head*",
        "heads | gating",
        "heads~2",
    ):
        payload = _plans_payload(
            recent=_plan("recent", _group("attention heads", bad))
        )
        assert "unsupported_syntax" in _rules(_check_queries(payload)), bad


def test_duplicate_alternatives_after_normalization_are_rejected() -> None:
    payload = _plans_payload(
        recent=_plan(
            "recent",
            _group("attention heads", "Attention  Heads"),
        )
    )
    assert "duplicate_finding" in _rules(_check_queries(payload))


def test_a_plan_without_a_candidate_anchor_is_rejected() -> None:
    payload = _plans_payload(
        recent=_plan(
            "recent",
            _group("protein folding"),
            _group("crystallography"),
        )
    )
    rejections = _check_queries(payload)
    assert "missing_support" in _rules(rejections)
    assert "anchor" in "; ".join(r.detail for r in rejections)


def test_an_over_rendered_plan_is_rejected_before_execution() -> None:
    # Every alternative legal on its own, but the rendered expression
    # exceeds the ceiling: rejected deterministically, never truncated.
    wide = [
        _group(*(f"some formidable established phrase {g}{a}" for a in range(4)))
        for g in range(3)
    ]
    wide[0]["alternatives"][0] = "attention head reweighting"
    payload = _plans_payload(recent=_plan("recent", *wide))
    assert "budget_violation" in _rules(_check_queries(payload))


def test_control_characters_in_terms_are_rejected() -> None:
    payload = _plans_payload(
        recent=_plan("recent", _group("attention\x00heads"))
    )
    assert "corrupted_text" in _rules(_check_queries(payload))


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
