"""The trusted query-rendering boundary: explicit Boolean semantics,
canonical order-invariance, bounds, and the family-to-intent mapping."""

from __future__ import annotations

from autonomous_research_lab.priorart.plan import (
    MAX_ALTERNATIVES_PER_GROUP,
    MAX_CONCEPT_GROUPS,
    MAX_RENDERED_CHARS,
    MAX_TERM_CHARS,
    REQUIRED_INTENTS,
    canonical_groups,
    render_query,
)
from autonomous_research_lab.priorart.records import PriorArtQueryFamily


def test_synonyms_become_alternatives_not_requirements() -> None:
    # One concept with three names is ONE group: any name suffices.
    rendered = render_query(
        (("attention head reweighting", "head gating", "head masking"),)
    )
    assert rendered == (
        '("attention head reweighting" OR "head gating" OR '
        '"head masking")'
    )
    assert " AND " not in rendered


def test_independent_concepts_remain_conjunctive() -> None:
    rendered = render_query(
        (
            ("attention head reweighting",),
            ("in-context learning", "few-shot learning"),
        )
    )
    assert rendered == (
        '("attention head reweighting") AND '
        '("few-shot learning" OR "in-context learning")'
    )


def test_multi_word_terms_are_exact_phrases() -> None:
    # Every alternative is quoted — single words included — so no term
    # can be re-read as an operator by the search engine.
    rendered = render_query((("reweighting",), ("induction heads",)))
    assert '"reweighting"' in rendered
    assert '"induction heads"' in rendered


def test_rendering_is_order_invariant() -> None:
    # Ordering is non-semantic (AND and OR are commutative): the same
    # plan in any order renders — and fingerprints — identically.
    forward = render_query(
        (("gating", "reweighting"), ("in-context learning",))
    )
    backward = render_query(
        (("in-context learning",), ("reweighting", "gating"))
    )
    assert forward == backward
    assert canonical_groups(
        (("Gating", "reweighting "), ("in-context learning",))
    ) == canonical_groups(
        (("in-context  learning",), ("reweighting", "gating"))
    )


def test_canonicalization_deduplicates_within_a_group() -> None:
    assert canonical_groups((("Heads", "heads", "HEADS"),)) == (
        ("heads",),
    )


def test_the_bounds_exclude_the_observed_failure_shape() -> None:
    # The Task 5D live queries conflated ~10 concepts; the bound is set
    # far below that, and the widest legal render stays inside the
    # rendered ceiling.
    assert MAX_CONCEPT_GROUPS == 3
    widest = tuple(
        tuple(f"term {group}{alt} extra words" for alt in range(4))
        for group in range(MAX_CONCEPT_GROUPS)
    )
    assert len(widest[0]) == MAX_ALTERNATIVES_PER_GROUP
    rendered = render_query(widest)
    assert len(rendered) <= MAX_RENDERED_CHARS
    assert MAX_TERM_CHARS * MAX_ALTERNATIVES_PER_GROUP > 100


def test_every_required_intent_maps_to_an_executed_family() -> None:
    # The Task 5D specification names seven search intents; the six
    # families must serve all of them, explicitly.
    families = {family.value for family in PriorArtQueryFamily}
    assert set(REQUIRED_INTENTS.values()) <= families
    assert set(REQUIRED_INTENTS) == {
        "exact_mechanism",
        "problem_plus_mechanism",
        "task_dataset_metric",
        "synonyms_and_older_terminology",
        "closest_cited_work",
        "competing_approaches",
        "latest_work_to_cutoff",
    }
    # Every family serves at least one intent — no dead family.
    assert set(REQUIRED_INTENTS.values()) == families
