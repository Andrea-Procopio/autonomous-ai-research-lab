"""The literature-retrieval seam: bounded queries, canonical identifiers,
and the normalized source contract.

No test here opens a network connection. Identifier fixtures are labelled
per case: forms marked OBSERVED were returned by the real OpenAlex API on
2026-08-18; the rest are documentation-derived (the DOI handbook's
case-insensitivity, arXiv's id grammar) or synthetic edge cases.
"""

from __future__ import annotations

import pytest

from autonomous_research_lab.literature.retrieval import (
    PAGE_SIZE_CEILING,
    RESULT_CEILING,
    AccessLevel,
    LiteratureProviderError,
    LiteratureQuery,
    LiteratureSource,
    MalformedLiteratureResponseError,
    ResultOrdering,
    RetrievedSearch,
    ScriptedLiteratureProvider,
    normalize_arxiv_id,
    normalize_doi,
)


def _source(**overrides: object) -> LiteratureSource:
    defaults: dict[str, object] = {
        "provider": "openalex",
        "provider_id": "W1",
        "title": "A Study of Things",
        "authors": ("Ada Lovelace",),
        "publication_date": "2026-06-10",
        "publication_year": 2026,
        "venue": "Journal of Things",
        "work_type": "article",
        "abstract": None,
        "doi": None,
        "arxiv_id": None,
        "provider_url": "https://openalex.org/W1",
        "landing_page_url": None,
        "pdf_url": None,
        "cited_by_count": None,
        "referenced_work_ids": (),
        "access_level": AccessLevel.METADATA,
    }
    defaults.update(overrides)
    return LiteratureSource(**defaults)  # type: ignore[arg-type]


# -- the bounded query --------------------------------------------------------


def test_a_query_is_bounded_by_construction() -> None:
    query = LiteratureQuery(
        text="in-context learning",
        from_date="2026-05-01",
        to_date="2026-08-18",
        per_page=25,
        max_results=75,
    )
    assert query.per_page <= PAGE_SIZE_CEILING
    assert query.max_results <= RESULT_CEILING


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": "   "},
        {"from_date": "June 2026"},
        {"to_date": "2026-13-01"},
        {"from_date": "2026-08-18", "to_date": "2026-05-01"},
        {"per_page": 0},
        {"per_page": PAGE_SIZE_CEILING + 1},
        {"max_results": 0},
        {"max_results": RESULT_CEILING + 1},
        {"timeout_seconds": 0.0},
    ],
)
def test_an_unbounded_or_malformed_query_cannot_be_built(
    overrides: dict[str, object],
) -> None:
    fields: dict[str, object] = {"text": "topic"}
    fields.update(overrides)
    with pytest.raises(ValueError):
        LiteratureQuery(**fields)  # type: ignore[arg-type]


def test_the_fingerprint_covers_what_determines_the_result_set() -> None:
    base = LiteratureQuery(text="topic", from_date="2026-01-01", max_results=50)
    same = LiteratureQuery(text="topic", from_date="2026-01-01", max_results=50)
    assert base.fingerprint == same.fingerprint

    for variant in (
        LiteratureQuery(text="other", from_date="2026-01-01", max_results=50),
        LiteratureQuery(text="topic", from_date="2026-02-01", max_results=50),
        LiteratureQuery(text="topic", from_date="2026-01-01", max_results=60),
        LiteratureQuery(
            text="topic", from_date="2026-01-01", max_results=50, per_page=10
        ),
    ):
        assert variant.fingerprint != base.fingerprint


def test_patience_does_not_change_the_fingerprint() -> None:
    """The result set is decided server-side; two queries differing only in
    timeout ask for the same results and replay from the same search."""
    quick = LiteratureQuery(text="topic", timeout_seconds=5.0)
    patient = LiteratureQuery(text="topic", timeout_seconds=120.0)
    assert quick.fingerprint == patient.fingerprint


def test_the_default_ordering_keeps_the_pre_ordering_fingerprint() -> None:
    """The backward-compatibility guarantee, pinned against a preserved
    live record: this exact query completed the Task 5A live proof on
    2026-08-18 under fingerprint litq_41cbe09e73f99e67, before result
    orderings existed. A recency query must still derive that
    fingerprint, or every existing corpus stops replaying."""
    query = LiteratureQuery(
        text="in-context learning",
        from_date="2026-05-01",
        to_date="2026-08-18",
        per_page=25,
        max_results=75,
    )
    assert query.ordering is ResultOrdering.RECENCY
    assert query.fingerprint == "litq_41cbe09e73f99e67"


def test_an_influence_ordering_is_a_distinct_result_set() -> None:
    recency = LiteratureQuery(text="topic")
    influence = LiteratureQuery(
        text="topic", ordering=ResultOrdering.INFLUENCE
    )
    assert recency.fingerprint != influence.fingerprint


# -- canonical identifiers ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # OBSERVED: OpenAlex reports DOIs as resolver URLs.
        ("https://doi.org/10.5281/zenodo.20632938", "10.5281/zenodo.20632938"),
        ("https://doi.org/10.48550/arxiv.2608.14274", "10.48550/arxiv.2608.14274"),
        # Documentation-derived: DOIs are case-insensitive by spec.
        ("https://doi.org/10.5281/ZENODO.1", "10.5281/zenodo.1"),
        ("doi:10.1000/ABC", "10.1000/abc"),
        ("10.1234/plain", "10.1234/plain"),
        ("http://dx.doi.org/10.1/x", "10.1/x"),
        # Synthetic: things that must not become plausible-looking keys.
        ("not-a-doi", None),
        ("https://doi.org/garbage", None),
        ("10.nothing", None),
        ("", None),
        (None, None),
    ],
)
def test_doi_normalization(raw: str | None, expected: str | None) -> None:
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # OBSERVED: the two shapes arXiv identity actually arrives in.
        ("10.48550/arxiv.2608.14274", "2608.14274"),
        ("http://arxiv.org/abs/1706.08749", "1706.08749"),
        ("https://arxiv.org/pdf/1706.08749", "1706.08749"),
        # Documentation-derived: labels, versions, and the old-style grammar.
        ("arXiv:2103.14030", "2103.14030"),
        ("2103.14030v2", "2103.14030"),
        ("https://arxiv.org/abs/2103.14030v1", "2103.14030"),
        ("https://arxiv.org/pdf/2103.14030v1.pdf", "2103.14030"),
        ("math.GT/0309136", "math.gt/0309136"),
        # Synthetic: near misses stay None rather than becoming keys.
        ("https://example.org/abs/2103.14030", None),
        ("https://arxiv.org/things/2103.14030", None),
        ("10.5281/zenodo.20632938", None),
        ("wat", None),
        ("", None),
        (None, None),
    ],
)
def test_arxiv_normalization(raw: str | None, expected: str | None) -> None:
    assert normalize_arxiv_id(raw) == expected


# -- the normalized source ----------------------------------------------------


def test_content_identity_is_deterministic_over_the_whole_record() -> None:
    first = _source()
    again = _source()
    assert first.id == again.id
    assert first.id.startswith("lit_")
    assert _source(title="A Different Study").id != first.id
    assert _source(cited_by_count=3).id != first.id


def test_the_access_level_matches_what_was_actually_retrieved() -> None:
    with_abstract = _source(
        abstract="We study things.", access_level=AccessLevel.ABSTRACT
    )
    assert with_abstract.access_level is AccessLevel.ABSTRACT

    with pytest.raises(ValueError):
        _source(abstract=None, access_level=AccessLevel.ABSTRACT)
    with pytest.raises(ValueError):
        _source(abstract=None, access_level=AccessLevel.FULL_TEXT)
    with pytest.raises(ValueError):
        _source(abstract="We study things.", access_level=AccessLevel.METADATA)
    with pytest.raises(ValueError):
        _source(abstract="   ", access_level=AccessLevel.ABSTRACT)


def test_a_source_names_its_provider_and_identifier() -> None:
    with pytest.raises(ValueError):
        _source(provider=" ")
    with pytest.raises(ValueError):
        _source(provider_id="")


def test_missing_optional_metadata_is_representable_as_absent() -> None:
    bare = _source(
        title=None,
        authors=(),
        publication_date=None,
        publication_year=None,
        venue=None,
        work_type=None,
        cited_by_count=None,
    )
    assert bare.title is None
    assert bare.authors == ()
    assert bare.cited_by_count is None
    assert bare.access_level is AccessLevel.METADATA


# -- the completed retrieval --------------------------------------------------


def _retrieved(**overrides: object) -> RetrievedSearch:
    defaults: dict[str, object] = {
        "provider": "scripted",
        "retrieved_at": "2026-08-18T12:00:00+00:00",
        "request_params": {"search": "topic"},
        "total_count": 2,
        "pages_fetched": 1,
        "page_identifiers": ("ray-1",),
        "rate_limit": {},
        "truncated": False,
        "sources": (_source(),),
    }
    defaults.update(overrides)
    return RetrievedSearch(**defaults)  # type: ignore[arg-type]


def test_a_retrieval_accounts_for_every_page() -> None:
    with pytest.raises(ValueError):
        _retrieved(pages_fetched=2, page_identifiers=("ray-1",))
    with pytest.raises(ValueError):
        _retrieved(pages_fetched=0, page_identifiers=())


# -- the scripted provider ----------------------------------------------------


def test_the_scripted_provider_serves_in_order_and_records_queries() -> None:
    outcomes = (
        _retrieved(),
        MalformedLiteratureResponseError("scripted failure"),
    )
    provider = ScriptedLiteratureProvider(outcomes)
    first = provider.search(LiteratureQuery(text="one"))
    assert first is outcomes[0]

    with pytest.raises(LiteratureProviderError, match="scripted failure"):
        provider.search(LiteratureQuery(text="two"))
    with pytest.raises(LiteratureProviderError, match="scripted for 2"):
        provider.search(LiteratureQuery(text="three"))
    assert [query.text for query in provider.queries] == ["one", "two", "three"]
