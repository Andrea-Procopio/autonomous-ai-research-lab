"""Deterministic, conservative deduplication over literature snapshots.

All fixtures here are synthetic — the rules under test are this package's
own, not any provider's. No test opens a network connection.
"""

from __future__ import annotations

import pytest

from autonomous_research_lab.literature.dedup import deduplicate
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureSource,
)


def _source(
    provider_id: str,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    title: str | None = "A Sufficiently Long Study of Things",
    year: int | None = 2026,
    authors: tuple[str, ...] = ("Ada Lovelace",),
    cited_by_count: int | None = None,
) -> LiteratureSource:
    return LiteratureSource(
        provider="openalex",
        provider_id=provider_id,
        title=title,
        authors=authors,
        publication_date=None,
        publication_year=year,
        venue=None,
        work_type=None,
        abstract=None,
        doi=doi,
        arxiv_id=arxiv_id,
        provider_url=f"https://example.org/{provider_id}",
        landing_page_url=None,
        pdf_url=None,
        cited_by_count=cited_by_count,
        referenced_work_ids=(),
        access_level=AccessLevel.METADATA,
    )


# -- exact canonical identifiers ----------------------------------------------


def test_a_shared_doi_is_a_duplicate() -> None:
    first = _source("W1", doi="10.1/x")
    second = _source("W2", doi="10.1/x", title="Retitled in a Later Snapshot")
    report = deduplicate([first, second])

    assert report.total == 2
    assert report.representative_ids == (first.id,)
    assert report.duplicate_count == 1
    (group,) = report.groups
    assert group.source_ids == (first.id, second.id)
    assert "doi" in group.matched_on
    assert report.conflicts == ()


def test_a_shared_arxiv_id_is_a_duplicate() -> None:
    first = _source("W1", arxiv_id="2103.14030")
    second = _source("W2", arxiv_id="2103.14030")
    report = deduplicate([first, second])
    assert report.duplicate_count == 1
    assert "arxiv" in report.groups[0].matched_on


def test_two_snapshots_of_one_work_are_duplicates() -> None:
    """Same provider id, drifted volatile metadata: one work, two
    snapshots, and one may carry an identifier the other predates."""
    early = _source("W1", cited_by_count=10)
    late = _source("W1", doi="10.1/x", cited_by_count=178)
    assert early.id != late.id
    report = deduplicate([early, late])
    assert report.duplicate_count == 1
    assert "provider_id" in report.groups[0].matched_on


def test_distinct_works_stay_distinct() -> None:
    report = deduplicate(
        [
            _source("W1", doi="10.1/x", title="On Apples and Their Orchards"),
            _source("W2", doi="10.1/y", title="On Oranges and Their Groves"),
        ]
    )
    assert report.duplicate_count == 0
    assert report.groups == ()
    assert report.conflicts == ()


# -- the safe title fallback --------------------------------------------------


def test_identifierless_snapshots_match_on_title_year_and_first_author() -> None:
    first = _source("W1", authors=("Ajieh, Frank",))
    second = _source("W2", authors=("Frank Ajieh",))  # byline convention varies
    report = deduplicate([first, second])
    assert report.duplicate_count == 1
    assert report.groups[0].matched_on == ("title",)


@pytest.mark.parametrize(
    "second",
    [
        _source("W2", year=2025),  # different year
        _source("W2", authors=("Grace Hopper",)),  # different first author
        _source("W2", year=None),  # missing year: no key at all
        _source("W2", authors=()),  # missing authors: no key at all
        _source("W2", title=None),  # missing title: no key at all
    ],
)
def test_the_title_key_requires_every_leg(second: LiteratureSource) -> None:
    report = deduplicate([_source("W1"), second])
    assert report.duplicate_count == 0


def test_a_short_generic_title_never_fuses_works() -> None:
    report = deduplicate(
        [_source("W1", title="Overview"), _source("W2", title="Overview")]
    )
    assert report.duplicate_count == 0


def test_title_matching_never_speaks_where_an_identifier_exists() -> None:
    """A snapshot with a canonical identifier is decided by identifiers
    alone: an identifier-less snapshot with the same title stays
    separate rather than being fused on weaker evidence."""
    with_doi = _source("W1", doi="10.1/x")
    without = _source("W2")
    report = deduplicate([with_doi, without])
    assert report.duplicate_count == 0


def test_punctuation_and_case_do_not_defeat_the_title_key() -> None:
    report = deduplicate(
        [
            _source("W1", title="In-Context Learning: A Survey"),
            _source("W2", title="in context learning — a survey"),
        ]
    )
    assert report.duplicate_count == 1


# -- conflicts are never silently merged --------------------------------------


def test_conflicting_dois_under_one_arxiv_id_refuse_to_merge() -> None:
    first = _source("W1", arxiv_id="2103.14030", doi="10.1/x")
    second = _source("W2", arxiv_id="2103.14030", doi="10.1/y")
    report = deduplicate([first, second])

    assert report.duplicate_count == 0
    assert report.representative_ids == (first.id, second.id)
    (conflict,) = report.conflicts
    assert conflict.kind == "arxiv"
    assert conflict.key == "2103.14030"
    assert conflict.source_ids == (first.id, second.id)
    assert "10.1/x" in conflict.detail
    assert "10.1/y" in conflict.detail


def test_conflicting_dois_across_snapshots_of_one_work_are_reported() -> None:
    """The provider correcting a work's DOI between retrievals is a
    contradiction to surface, not a tie to break silently."""
    report = deduplicate(
        [_source("W1", doi="10.1/x"), _source("W1", doi="10.1/y")]
    )
    assert report.duplicate_count == 0
    (conflict,) = report.conflicts
    assert conflict.kind == "provider_id"
    assert "conflicting DOIs" in conflict.detail


def test_conflicting_arxiv_ids_under_one_doi_refuse_to_merge() -> None:
    report = deduplicate(
        [
            _source("W1", doi="10.1/x", arxiv_id="2103.14030"),
            _source("W2", doi="10.1/x", arxiv_id="2109.00001"),
        ]
    )
    assert report.duplicate_count == 0
    (conflict,) = report.conflicts
    assert "arXiv ids" in conflict.detail


# -- determinism --------------------------------------------------------------


def test_the_report_is_deterministic_and_order_grounded() -> None:
    sources = [
        _source("W1", doi="10.1/x"),
        _source("W2", doi="10.1/y", title="On Oranges and Their Groves"),
        _source("W3", doi="10.1/x", cited_by_count=5),
    ]
    first = deduplicate(sources)
    second = deduplicate(sources)
    assert first == second
    assert first.representative_ids == (sources[0].id, sources[1].id)

    reversed_report = deduplicate(list(reversed(sources)))
    assert reversed_report.duplicate_count == first.duplicate_count
    assert reversed_report.representative_ids[0] == sources[2].id


def test_transitive_duplicates_form_one_group() -> None:
    """A joins B on DOI, B joins C on arXiv id: one work, one group."""
    a = _source("W1", doi="10.1/x")
    b = _source("W2", doi="10.1/x", arxiv_id="2103.14030")
    c = _source("W3", arxiv_id="2103.14030")
    report = deduplicate([a, b, c])
    assert report.representative_ids == (a.id,)
    (group,) = report.groups
    assert group.source_ids == (a.id, b.id, c.id)
    assert group.matched_on == ("arxiv", "doi")


def test_an_empty_input_yields_an_empty_report() -> None:
    report = deduplicate([])
    assert report.total == 0
    assert report.representative_ids == ()
    assert report.groups == ()
    assert report.conflicts == ()
