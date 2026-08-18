"""The OpenAlex adapter, tested against captured and documented shapes.

Fixture provenance, kept honest per payload below: the success envelope,
the two work shapes (a venue-published article and an arXiv-DOI preprint),
the rate-limit headers, and the HTML 404 body are sanitized copies of
responses the real endpoint returned on 2026-08-18 (abstracts shortened,
author lists trimmed). The JSON ``{"error", "message"}`` envelope, the 403
refusal, and the 429 semantics were NOT captured live: they are derived
from the official error-handling documentation (help.openalex.org/api).

No test opens a network connection, and no test touches a real credential:
keys in this file are obvious dummies.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Mapping
from typing import Any

import pytest

from autonomous_research_lab.literature import openalex
from autonomous_research_lab.literature.openalex import (
    SORTS,
    OpenAlexProvider,
    _normalize_work,
    _reconstruct_abstract,
)
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureAuthenticationError,
    LiteratureConfigurationError,
    LiteratureQuery,
    LiteratureRateLimitError,
    LiteratureTimeoutError,
    LiteratureTransportError,
    MalformedLiteratureResponseError,
)

# -- fixtures -----------------------------------------------------------------

#: OBSERVED live (2026-08-18, sanitized): a venue-published article. The
#: abstract index is shortened; the structure is verbatim.
_WORK_ARTICLE: dict[str, Any] = {
    "id": "https://openalex.org/W4288283362",
    "doi": "https://doi.org/10.5281/zenodo.20632938",
    "title": "Spiral Time: A Geometric Reframing of Temporal Structure",
    "display_name": "Spiral Time: A Geometric Reframing of Temporal Structure",
    "publication_year": 2026,
    "publication_date": "2026-06-10",
    "language": "en",
    "type": "article",
    "cited_by_count": 178,
    "ids": {
        "openalex": "https://openalex.org/W4288283362",
        "doi": "https://doi.org/10.5281/zenodo.20632938",
    },
    "primary_location": {
        "landing_page_url": "https://github.com/frankajieh-ship-it/spiral-time",
        "pdf_url": None,
        "version": "submittedVersion",
        "source": {
            "id": "https://openalex.org/S4406922384",
            "display_name": "Open MIND",
            "type": "repository",
        },
    },
    "authorships": [
        {
            "author_position": "first",
            "author": {
                "id": None,
                "display_name": "Ajieh, Frank",
                "orcid": "https://orcid.org/0009-0005-9019-843X",
            },
            "raw_author_name": "Frank Ajieh",
        }
    ],
    "abstract_inverted_index": {
        "Spiral": [0],
        "Time": [1],
        "reframes": [2],
        "temporal": [3],
        "structure.": [4],
    },
    "referenced_works": [],
    "relevance_score": 816.3272,
}

#: OBSERVED live (2026-08-18, sanitized): an arXiv-hosted preprint whose
#: arXiv identity arrives as a ``10.48550/arxiv.…`` DOI. Authorship is
#: synthetic (not captured for this work); everything else is verbatim.
_WORK_ARXIV: dict[str, Any] = {
    "id": "https://openalex.org/W2254118105",
    "doi": "https://doi.org/10.48550/arxiv.2608.14274",
    "title": "Solving QBF by Clause Selection",
    "publication_year": 2026,
    "publication_date": "2026-08-14",
    "type": "preprint",
    "cited_by_count": 0,
    "ids": {
        "openalex": "https://openalex.org/W2254118105",
        "doi": "https://doi.org/10.48550/arxiv.2608.14274",
        "mag": "2254118105",
    },
    "primary_location": {
        "landing_page_url": "https://doi.org/10.48550/arxiv.2608.14274",
        "pdf_url": None,
        "version": None,
        "source": {
            "id": "https://openalex.org/S4306400194",
            "display_name": "arXiv (Cornell University)",
            "type": "repository",
        },
    },
    "authorships": [
        {"author": {"display_name": "A. Researcher"}, "raw_author_name": None}
    ],
    "abstract_inverted_index": {"We": [0], "select": [1], "clauses.": [2]},
    "referenced_works": ["https://openalex.org/W1234567890"],
}

#: Synthetic: the sparsest work the protocol admits — identity only.
_WORK_BARE: dict[str, Any] = {"id": "https://openalex.org/W99"}

#: OBSERVED live (2026-08-18): rate-limit headers on every reply.
_PAGE_HEADERS = {
    "content-type": "application/json",
    "cf-ray": "a2d3183bfd64c674-EWR",
    "x-ratelimit-limit": "1000",
    "x-ratelimit-remaining": "990",
    "x-ratelimit-credits-used": "10",
    "x-ratelimit-reset": "18673",
    "x-ratelimit-cost-usd": "0.001",
    "x-ratelimit-remaining-usd": "0.099",
}

#: OBSERVED live (2026-08-18): a 404 whose body is HTML, not the
#: documented JSON envelope.
_HTML_404 = (
    b"<!doctype html>\n<html lang=en>\n<title>404 Not Found</title>\n"
    b"<h1>Not Found</h1>\n<p>The requested URL was not found.</p>\n"
)

#: Documentation-derived: the JSON error envelope and its fields.
_JSON_400 = json.dumps(
    {"error": "Invalid filter", "message": "Unknown filter field: author_name."}
).encode()
_JSON_403 = json.dumps(
    {"error": "Forbidden", "message": "You don't have access to this resource."}
).encode()
_JSON_429 = json.dumps(
    {"error": "Too Many Requests", "message": "Daily credit budget exceeded."}
).encode()


def _page(
    results: list[dict[str, Any]],
    *,
    count: int = 1,
    next_cursor: str | None = None,
) -> bytes:
    """The observed success envelope: ``meta`` plus ``results``."""
    return json.dumps(
        {
            "meta": {
                "count": count,
                "db_response_time_ms": 127,
                "page": 1,
                "per_page": len(results),
                "groups_count": None,
                "next_cursor": next_cursor,
            },
            "results": results,
        }
    ).encode()


# -- the connection stub (mirrors tests/test_muse_provider.py) ----------------


class _FakeReply:
    def __init__(
        self, body: bytes, headers: dict[str, str], *, status: int = 200
    ) -> None:
        self._body = body
        self._offset = 0
        self.status = status
        self.headers = headers

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk


class _FakeConnection:
    """The ``_connect`` seam, captured: records the one request and serves
    one scripted reply or raises one scripted exception."""

    def __init__(self, base_url: str, outcome: _FakeReply | Exception) -> None:
        self.base_url = base_url
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False
        self._outcome = outcome

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.requests.append((method, url, dict(headers or {})))

    def getresponse(self) -> _FakeReply:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    def close(self) -> None:
        self.closed = True


def _stub_pages(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[_FakeReply | Exception],
) -> list[_FakeConnection]:
    """One scripted outcome per page request, each on a fresh connection,
    exactly as the adapter opens them."""
    seen: list[_FakeConnection] = []
    queue = list(outcomes)

    def fake_connect(base_url: str, timeout: float) -> _FakeConnection:
        outcome = queue.pop(0) if queue else _FakeReply(b"{}", {})
        connection = _FakeConnection(base_url, outcome)
        seen.append(connection)
        return connection

    monkeypatch.setattr(openalex, "_connect", fake_connect)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    return seen


def _params_of(connection: _FakeConnection) -> dict[str, str]:
    _, target, _ = connection.requests[0]
    path, _, query = target.partition("?")
    assert path == "/works"
    return {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}


def _query(**overrides: object) -> LiteratureQuery:
    defaults: dict[str, object] = {
        "text": "machine learning",
        "from_date": "2026-06-01",
        "to_date": "2026-08-18",
        "per_page": 25,
        "max_results": 50,
    }
    defaults.update(overrides)
    return LiteratureQuery(**defaults)  # type: ignore[arg-type]


# -- normalization ------------------------------------------------------------


def test_the_observed_article_normalizes_completely() -> None:
    source = _normalize_work(_WORK_ARTICLE)
    assert source.provider == "openalex"
    assert source.provider_id == "W4288283362"
    assert source.title == (
        "Spiral Time: A Geometric Reframing of Temporal Structure"
    )
    assert source.authors == ("Ajieh, Frank",)
    assert source.publication_date == "2026-06-10"
    assert source.publication_year == 2026
    assert source.venue == "Open MIND"
    assert source.work_type == "article"
    assert source.abstract == "Spiral Time reframes temporal structure."
    assert source.doi == "10.5281/zenodo.20632938"
    assert source.arxiv_id is None
    assert source.provider_url == "https://openalex.org/W4288283362"
    assert source.landing_page_url == (
        "https://github.com/frankajieh-ship-it/spiral-time"
    )
    assert source.pdf_url is None
    assert source.cited_by_count == 178
    assert source.referenced_work_ids == ()
    assert source.access_level is AccessLevel.ABSTRACT


def test_the_observed_arxiv_preprint_yields_its_arxiv_id() -> None:
    source = _normalize_work(_WORK_ARXIV)
    assert source.doi == "10.48550/arxiv.2608.14274"
    assert source.arxiv_id == "2608.14274"
    assert source.work_type == "preprint"
    assert source.venue == "arXiv (Cornell University)"
    assert source.authors == ("A. Researcher",)
    assert source.referenced_work_ids == ("W1234567890",)


def test_an_arxiv_landing_url_also_yields_the_id() -> None:
    """OBSERVED shape: some arXiv-hosted works carry a non-arXiv DOI and
    reveal arXiv identity only through their location URLs."""
    work = dict(_WORK_ARXIV)
    work["doi"] = "https://doi.org/10.5281/zenodo.21836329"
    work["primary_location"] = {
        "landing_page_url": "http://arxiv.org/abs/1706.08749",
        "pdf_url": "https://arxiv.org/pdf/1706.08749",
    }
    source = _normalize_work(work)
    assert source.doi == "10.5281/zenodo.21836329"
    assert source.arxiv_id == "1706.08749"


def test_missing_optional_metadata_stays_absent() -> None:
    """Nothing reported, nothing invented: no fabricated defaults."""
    source = _normalize_work(_WORK_BARE)
    assert source.provider_id == "W99"
    assert source.title is None
    assert source.authors == ()
    assert source.publication_date is None
    assert source.publication_year is None
    assert source.venue is None
    assert source.work_type is None
    assert source.abstract is None
    assert source.doi is None
    assert source.arxiv_id is None
    assert source.landing_page_url is None
    assert source.pdf_url is None
    assert source.cited_by_count is None
    assert source.referenced_work_ids == ()
    assert source.access_level is AccessLevel.METADATA


def test_a_work_without_an_id_is_a_malformed_reply() -> None:
    with pytest.raises(MalformedLiteratureResponseError):
        _normalize_work({"title": "No identity"})
    with pytest.raises(MalformedLiteratureResponseError):
        _normalize_work("not an object")


def test_a_malformed_abstract_index_yields_no_abstract() -> None:
    """No partial reconstruction masquerading as the abstract."""
    assert _reconstruct_abstract({"word": ["not-an-int"]}) is None
    assert _reconstruct_abstract({"word": "not-a-list"}) is None
    assert _reconstruct_abstract({}) is None
    assert _reconstruct_abstract(None) is None
    assert _reconstruct_abstract({"b": [1], "a": [0]}) == "a b"


# -- the request --------------------------------------------------------------


def test_the_request_carries_dates_sort_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OBSERVED wire form (2026-08-18): the text matches in
    title_and_abstract — plain ``search=`` is fulltext matching and
    returned 4.6M works for a 6.3k-work phrase — with dates as sibling
    filter clauses and an explicit sort."""
    seen = _stub_pages(
        monkeypatch, [_FakeReply(_page([_WORK_ARTICLE]), _PAGE_HEADERS)]
    )
    OpenAlexProvider().search(_query())

    params = _params_of(seen[0])
    assert "search" not in params
    assert params["filter"] == (
        "title_and_abstract.search:machine learning,"
        "from_publication_date:2026-06-01,to_publication_date:2026-08-18"
    )
    assert params["sort"] == "publication_date:desc"
    assert params["per_page"] == "25"
    assert params["cursor"] == "*"


def test_each_ordering_maps_to_its_documented_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OBSERVED (2026-08-18): cited_by_count:desc works with search text
    and cursor paging, returning descending counts."""
    from autonomous_research_lab.literature.retrieval import ResultOrdering

    for ordering, expected in SORTS.items():
        seen = _stub_pages(
            monkeypatch, [_FakeReply(_page([_WORK_ARTICLE]), _PAGE_HEADERS)]
        )
        OpenAlexProvider().search(_query(ordering=ordering))
        assert _params_of(seen[0])["sort"] == expected
    assert SORTS[ResultOrdering.INFLUENCE] == "cited_by_count:desc"


def test_filter_syntax_characters_in_query_text_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commas separate filter clauses and colons separate keys from
    values; either inside the text would corrupt the filter."""
    seen = _stub_pages(
        monkeypatch, [_FakeReply(_page([_WORK_ARTICLE]), _PAGE_HEADERS)]
    )
    OpenAlexProvider().search(
        _query(text='TACTICL: compression, "in-context learning"')
    )
    params = _params_of(seen[0])
    assert params["filter"].startswith(
        'title_and_abstract.search:TACTICL compression "in-context learning",'
    )


def test_an_unbounded_date_side_is_omitted_not_defaulted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _stub_pages(
        monkeypatch, [_FakeReply(_page([_WORK_ARTICLE]), _PAGE_HEADERS)]
    )
    OpenAlexProvider().search(_query(from_date="", to_date=""))
    assert _params_of(seen[0])["filter"] == (
        "title_and_abstract.search:machine learning"
    )


def test_the_api_key_is_a_header_and_never_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _stub_pages(
        monkeypatch, [_FakeReply(_page([_WORK_ARTICLE]), _PAGE_HEADERS)]
    )
    monkeypatch.setenv("OPENALEX_API_KEY", "dummy-not-a-real-key")
    retrieved = OpenAlexProvider().search(_query())

    _, target, headers = seen[0].requests[0]
    assert headers["Authorization"] == "Bearer dummy-not-a-real-key"
    assert "dummy" not in target
    assert all("dummy" not in v for v in retrieved.request_params.values())
    assert all("dummy" not in v for v in retrieved.rate_limit.values())


def test_anonymous_use_sends_no_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _stub_pages(
        monkeypatch, [_FakeReply(_page([_WORK_ARTICLE]), _PAGE_HEADERS)]
    )
    OpenAlexProvider().search(_query())
    _, _, headers = seen[0].requests[0]
    assert "Authorization" not in headers


def test_an_unsupported_scheme_is_local_misconfiguration() -> None:
    with pytest.raises(LiteratureConfigurationError):
        OpenAlexProvider(base_url="ftp://api.openalex.org").search(_query())


# -- pagination and bounds ----------------------------------------------------


def test_cursor_paging_respects_the_result_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _stub_pages(
        monkeypatch,
        [
            _FakeReply(
                _page([_WORK_ARTICLE, _WORK_ARXIV], count=9, next_cursor="cur2"),
                _PAGE_HEADERS,
            ),
            _FakeReply(
                _page([_WORK_BARE], count=9, next_cursor="cur3"), _PAGE_HEADERS
            ),
        ],
    )
    retrieved = OpenAlexProvider().search(_query(per_page=2, max_results=3))

    assert [s.provider_id for s in retrieved.sources] == [
        "W4288283362",
        "W2254118105",
        "W99",
    ]
    assert retrieved.pages_fetched == 2
    assert retrieved.page_identifiers == (
        "a2d3183bfd64c674-EWR",
        "a2d3183bfd64c674-EWR",
    )
    assert retrieved.total_count == 9
    assert retrieved.truncated is True
    assert retrieved.rate_limit["openalex:credits_used_total"] == "20"
    assert _params_of(seen[0])["cursor"] == "*"
    second = {
        k: v[0]
        for k, v in urllib.parse.parse_qs(
            seen[1].requests[0][1].partition("?")[2]
        ).items()
    }
    assert second["cursor"] == "cur2"
    assert second["per_page"] == "1"  # only one seat left in the budget


def test_an_overfull_page_is_trimmed_to_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pages(
        monkeypatch,
        [_FakeReply(_page([_WORK_ARTICLE, _WORK_ARXIV], count=2), _PAGE_HEADERS)],
    )
    retrieved = OpenAlexProvider().search(_query(per_page=1, max_results=1))
    assert len(retrieved.sources) == 1
    assert retrieved.truncated is True


def test_empty_results_end_the_search_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pages(monkeypatch, [_FakeReply(_page([], count=0), _PAGE_HEADERS)])
    retrieved = OpenAlexProvider().search(_query())
    assert retrieved.sources == ()
    assert retrieved.pages_fetched == 1
    assert retrieved.total_count == 0
    assert retrieved.truncated is False


def test_an_exhausted_cursor_ends_the_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _stub_pages(
        monkeypatch,
        [_FakeReply(_page([_WORK_ARTICLE], count=1, next_cursor=None), _PAGE_HEADERS)],
    )
    retrieved = OpenAlexProvider().search(_query())
    assert len(retrieved.sources) == 1
    assert retrieved.truncated is False
    assert len(seen) == 1


# -- failure translation ------------------------------------------------------


def test_a_rate_limit_retries_once_honoring_the_servers_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    naps: list[float] = []
    _stub_pages(
        monkeypatch,
        [
            _FakeReply(_JSON_429, {"retry-after": "1.5"}, status=429),
            _FakeReply(_page([_WORK_ARTICLE]), _PAGE_HEADERS),
        ],
    )
    retrieved = OpenAlexProvider(sleep=naps.append).search(_query())
    assert len(retrieved.sources) == 1
    assert naps == [1.5]


def test_a_persistent_rate_limit_raises_after_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    naps: list[float] = []
    seen = _stub_pages(
        monkeypatch,
        [
            _FakeReply(_JSON_429, {}, status=429),
            _FakeReply(_JSON_429, {}, status=429),
        ],
    )
    with pytest.raises(LiteratureRateLimitError, match="Daily credit budget"):
        OpenAlexProvider(sleep=naps.append).search(_query())
    assert naps == [2.0]  # the documented first backoff step
    assert len(seen) == 2


def test_a_server_wait_beyond_the_cap_is_not_slept_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    naps: list[float] = []
    seen = _stub_pages(
        monkeypatch,
        [_FakeReply(_JSON_429, {"retry-after": "3600"}, status=429)],
    )
    with pytest.raises(LiteratureRateLimitError) as caught:
        OpenAlexProvider(sleep=naps.append).search(_query())
    assert naps == []
    assert len(seen) == 1
    assert caught.value.retry_after_seconds == 3600.0


def test_a_server_error_retries_once_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    naps: list[float] = []
    seen = _stub_pages(
        monkeypatch,
        [
            _FakeReply(b"upstream burp", {}, status=503),
            _FakeReply(b"upstream burp", {}, status=503),
        ],
    )
    with pytest.raises(LiteratureTransportError) as caught:
        OpenAlexProvider(sleep=naps.append).search(_query())
    assert caught.value.status_code == 503
    assert naps == [2.0]
    assert len(seen) == 2


def test_client_errors_are_typed_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _stub_pages(monkeypatch, [_FakeReply(_JSON_400, {}, status=400)])
    with pytest.raises(LiteratureTransportError) as caught:
        OpenAlexProvider().search(_query())
    assert caught.value.status_code == 400
    assert caught.value.provider_error == "Invalid filter"
    assert "Unknown filter field" in str(caught.value)
    assert len(seen) == 1


def test_the_observed_html_404_degrades_to_the_status_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pages(monkeypatch, [_FakeReply(_HTML_404, {}, status=404)])
    with pytest.raises(LiteratureTransportError, match="HTTP 404") as caught:
        OpenAlexProvider().search(_query())
    assert caught.value.provider_error is None


def test_an_access_refusal_is_an_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pages(monkeypatch, [_FakeReply(_JSON_403, {}, status=403)])
    with pytest.raises(LiteratureAuthenticationError) as caught:
        OpenAlexProvider().search(_query())
    assert caught.value.status_code == 403
    assert caught.value.provider_error == "Forbidden"


def test_a_timeout_is_typed_with_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pages(monkeypatch, [TimeoutError()])
    with pytest.raises(LiteratureTimeoutError) as caught:
        OpenAlexProvider().search(_query(timeout_seconds=7.0))
    assert caught.value.timeout_seconds == 7.0


def test_a_connection_failure_is_transport_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _stub_pages(monkeypatch, [ConnectionRefusedError("refused")])
    with pytest.raises(LiteratureTransportError, match="could not reach"):
        OpenAlexProvider().search(_query())
    assert len(seen) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"\xff\xfe\x01 not utf-8",
        b"not json at all",
        b"[]",
        b'{"results": []}',
        b'{"meta": {"count": 0}}',
        b'{"meta": {"count": 0}, "results": "wat"}',
    ],
)
def test_malformed_and_non_utf8_replies_are_typed(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    _stub_pages(monkeypatch, [_FakeReply(body, _PAGE_HEADERS)])
    with pytest.raises(MalformedLiteratureResponseError):
        OpenAlexProvider().search(_query())


def test_an_absurdly_large_body_is_a_fault_not_a_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pages(
        monkeypatch, [_FakeReply(_page([_WORK_ARTICLE]), _PAGE_HEADERS)]
    )
    monkeypatch.setattr(openalex, "_MAX_BODY_BYTES", 64)
    with pytest.raises(MalformedLiteratureResponseError, match="exceeds"):
        OpenAlexProvider().search(_query())


def test_every_connection_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_pages(
        monkeypatch,
        [
            _FakeReply(_JSON_429, {"retry-after": "0.1"}, status=429),
            _FakeReply(_page([_WORK_ARTICLE]), _PAGE_HEADERS),
        ],
    )
    OpenAlexProvider(sleep=lambda _: None).search(_query())
    assert all(connection.closed for connection in seen)
