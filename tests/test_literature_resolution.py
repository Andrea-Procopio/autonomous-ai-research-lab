"""Access resolution (Task 5G): the lawful path to a missing abstract.

The Atom parsing is pure and pinned against captured response shapes;
the resolving provider is exercised with an injected fetch, so nothing
here touches a network. The live arXiv wire contract is qualified
separately, like OpenAlex's was.
"""

from __future__ import annotations

from autonomous_research_lab.literature.arxiv import summary_from
from autonomous_research_lab.literature.resolution import (
    AccessResolvingProvider,
)
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureQuery,
    LiteratureSource,
    RetrievedSearch,
    ScriptedLiteratureProvider,
)
from autonomous_research_lab.literature.store import LiteratureStore

ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: search_query=&amp;id_list=1512.03385</title>
  <entry>
    <id>http://arxiv.org/abs/1512.03385v1</id>
    <title>Deep Residual Learning for Image Recognition</title>
    <summary>  Deeper neural networks are more difficult to train.
We present a residual learning framework to ease the training
of networks.
</summary>
  </entry>
</feed>
"""

EMPTY_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: search_query=&amp;id_list=nonsense</title>
</feed>
"""

ERROR_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#bad_id</id>
    <title>Error</title>
    <summary>incorrect id format</summary>
  </entry>
</feed>
"""


class TestSummaryParsing:
    def test_a_found_work_yields_its_collapsed_abstract(self) -> None:
        assert summary_from(ATOM) == (
            "Deeper neural networks are more difficult to train. We "
            "present a residual learning framework to ease the training "
            "of networks."
        )

    def test_an_empty_feed_yields_none(self) -> None:
        assert summary_from(EMPTY_FEED) is None

    def test_an_error_entry_yields_none(self) -> None:
        assert summary_from(ERROR_FEED) is None

    def test_malformed_xml_yields_none(self) -> None:
        assert summary_from(b"this is not xml <") is None

    def test_a_blank_summary_yields_none(self) -> None:
        blank = ATOM.replace(
            b"<summary>  Deeper neural networks are more difficult to train.\n"
            b"We present a residual learning framework to ease the training\n"
            b"of networks.\n</summary>",
            b"<summary>   </summary>",
        )
        assert summary_from(blank) is None


def source(
    *,
    provider_id: str = "W1",
    abstract: str | None = None,
    arxiv_id: str | None = "1512.03385",
) -> LiteratureSource:
    return LiteratureSource(
        provider="openalex",
        provider_id=provider_id,
        title="Deep Residual Learning",
        authors=("Kaiming He",),
        publication_date="2015-12-10",
        publication_year=2015,
        venue="CVPR",
        work_type="article",
        abstract=abstract,
        doi="10.1109/cvpr.2016.90",
        arxiv_id=arxiv_id,
        provider_url="https://openalex.org/W1",
        landing_page_url="https://arxiv.org/abs/1512.03385",
        pdf_url=None,
        cited_by_count=100000,
        referenced_work_ids=(),
        access_level=(
            AccessLevel.METADATA if abstract is None else AccessLevel.ABSTRACT
        ),
    )


def retrieved(*sources: LiteratureSource) -> RetrievedSearch:
    return RetrievedSearch(
        provider="openalex",
        retrieved_at="2026-08-23T00:00:00+00:00",
        request_params={"filter": "title.search:residual"},
        total_count=len(sources),
        pages_fetched=1,
        page_identifiers=("",),
        rate_limit={},
        truncated=False,
        sources=sources,
    )


def query() -> LiteratureQuery:
    return LiteratureQuery(text="residual learning", max_results=5)


class TestAccessResolvingProvider:
    def test_a_metadata_only_work_with_an_arxiv_id_is_upgraded(self) -> None:
        bare = source(abstract=None)
        provider = AccessResolvingProvider(
            ScriptedLiteratureProvider((retrieved(bare),), name="openalex"),
            fetch=lambda arxiv_id: f"the abstract of {arxiv_id}",
        )
        result = provider.search(query())
        upgraded = result.sources[0]
        assert upgraded.access_level is AccessLevel.ABSTRACT
        assert upgraded.abstract == "the abstract of 1512.03385"
        assert upgraded.id != bare.id
        assert upgraded.provider_id == bare.provider_id
        assert upgraded.doi == bare.doi
        assert result.provider == "openalex"

    def test_a_failed_fetch_leaves_the_source_honestly_metadata(
        self,
    ) -> None:
        bare = source(abstract=None)
        provider = AccessResolvingProvider(
            ScriptedLiteratureProvider((retrieved(bare),), name="openalex"),
            fetch=lambda _: None,
        )
        result = provider.search(query())
        assert result.sources[0] is bare

    def test_an_abstract_bearing_source_is_never_touched(self) -> None:
        readable = source(abstract="Already retrieved.")
        calls: list[str] = []

        def fetch(arxiv_id: str) -> str | None:
            calls.append(arxiv_id)
            return "should never be used"

        provider = AccessResolvingProvider(
            ScriptedLiteratureProvider(
                (retrieved(readable),), name="openalex"
            ),
            fetch=fetch,
        )
        result = provider.search(query())
        assert result is not None
        assert result.sources[0] is readable
        assert calls == []

    def test_a_metadata_work_without_an_arxiv_id_stays_as_it_is(
        self,
    ) -> None:
        bare = source(abstract=None, arxiv_id=None)
        provider = AccessResolvingProvider(
            ScriptedLiteratureProvider((retrieved(bare),), name="openalex"),
            fetch=lambda _: "never",
        )
        assert provider.search(query()).sources[0] is bare

    def test_an_untouched_search_is_returned_verbatim(self) -> None:
        readable = source(abstract="Already retrieved.")
        search = retrieved(readable)
        provider = AccessResolvingProvider(
            ScriptedLiteratureProvider((search,), name="openalex"),
            fetch=lambda _: None,
        )
        assert provider.search(query()) is search

    def test_an_upgraded_source_survives_the_store(
        self, tmp_path: object
    ) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        provider = AccessResolvingProvider(
            ScriptedLiteratureProvider(
                (retrieved(source(abstract=None)),), name="openalex"
            ),
            fetch=lambda _: "A recovered abstract.",
        )
        upgraded = provider.search(query()).sources[0]
        store = LiteratureStore(tmp_path / "literature")
        store.record_source(upgraded)
        loaded = store.get_source(upgraded.id)
        assert loaded == upgraded
        assert loaded.access_level is AccessLevel.ABSTRACT

    def test_the_name_is_the_inner_providers(self) -> None:
        provider = AccessResolvingProvider(
            ScriptedLiteratureProvider((), name="openalex"),
            fetch=lambda _: None,
        )
        assert provider.name == "openalex"


class TestBlankArxivId:
    def test_a_blank_arxiv_id_is_never_fetched(self) -> None:
        bare = source(abstract=None, arxiv_id=" ")
        calls: list[str] = []

        def fetch(arxiv_id: str) -> str | None:
            calls.append(arxiv_id)
            return "never"

        provider = AccessResolvingProvider(
            ScriptedLiteratureProvider((retrieved(bare),), name="openalex"),
            fetch=fetch,
        )
        assert provider.search(query()).sources[0] is bare
        assert calls == []
