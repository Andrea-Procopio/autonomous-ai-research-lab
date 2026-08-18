"""The literature store and corpus: write-once provenance, tamper
detection, bounded growth, and cache replay with zero network calls.

All payloads here are synthetic — the store's contract is about identity
and durability, not about any provider's wire format. No test opens a
network connection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.literature import openalex
from autonomous_research_lab.literature.corpus import LiteratureCorpus
from autonomous_research_lab.literature.openalex import OpenAlexProvider
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureQuery,
    LiteratureSource,
    RetrievedSearch,
    ScriptedLiteratureProvider,
)
from autonomous_research_lab.literature.store import (
    LiteratureConflictError,
    LiteratureCorpusLimitError,
    LiteratureIntegrityError,
    LiteratureStore,
    search_record_from,
)


def _source(provider_id: str = "W1", **overrides: object) -> LiteratureSource:
    defaults: dict[str, object] = {
        "provider": "scripted",
        "provider_id": provider_id,
        "title": f"Paper {provider_id}",
        "authors": ("Ada Lovelace",),
        "publication_date": "2026-06-10",
        "publication_year": 2026,
        "venue": "Journal of Things",
        "work_type": "article",
        "abstract": "We report things.",
        "doi": f"10.1000/{provider_id.lower()}",
        "arxiv_id": None,
        "provider_url": f"https://example.org/{provider_id}",
        "landing_page_url": None,
        "pdf_url": None,
        "cited_by_count": 3,
        "referenced_work_ids": ("W7", "W8"),
        "access_level": AccessLevel.ABSTRACT,
    }
    defaults.update(overrides)
    return LiteratureSource(**defaults)  # type: ignore[arg-type]


def _query(**overrides: object) -> LiteratureQuery:
    defaults: dict[str, object] = {
        "text": "topic",
        "from_date": "2026-05-01",
        "to_date": "2026-08-18",
        "per_page": 2,
        "max_results": 4,
    }
    defaults.update(overrides)
    return LiteratureQuery(**defaults)  # type: ignore[arg-type]


def _retrieved(
    *sources: LiteratureSource, retrieved_at: str = "2026-08-18T12:00:00+00:00"
) -> RetrievedSearch:
    return RetrievedSearch(
        provider="scripted",
        retrieved_at=retrieved_at,
        request_params={"search": "topic", "sort": "date:desc"},
        total_count=200,
        pages_fetched=1,
        page_identifiers=("ray-1",),
        rate_limit={"scripted:credits": "10"},
        truncated=True,
        sources=tuple(sources),
    )


# -- write-once sources -------------------------------------------------------


def test_sources_are_write_once_and_verify_on_repeat(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path)
    source = _source()
    assert store.record_source(source) == source
    assert store.record_source(source) == source  # identical re-record: no-op
    assert store.get_source(source.id) == source

    imposter = _source(title="A Different Paper", id=source.id)
    with pytest.raises(LiteratureConflictError, match="never rewritten"):
        store.record_source(imposter)


def test_a_tampered_source_fails_on_load(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path)
    source = store.record_source(_source())
    path = tmp_path / "sources" / f"{source.id}.json"
    payload = json.loads(path.read_text())
    payload["cited_by_count"] = 9_999
    path.write_text(json.dumps(payload))

    with pytest.raises(LiteratureIntegrityError, match="re-derives"):
        store.get_source(source.id)


def test_the_source_bound_fails_loudly(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path, max_sources=2)
    store.record_source(_source("W1"))
    second = store.record_source(_source("W2"))
    with pytest.raises(LiteratureCorpusLimitError):
        store.record_source(_source("W3"))
    # An identical re-record of held content is still a no-op at the cap.
    assert store.record_source(second) == second


# -- write-once searches ------------------------------------------------------


def test_a_search_is_recorded_sources_first(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path)
    record = search_record_from(_query(), _retrieved(_source()))
    with pytest.raises(LiteratureConflictError, match="sources-first"):
        store.record_search(record)


def test_a_recorded_search_replays_and_round_trips(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path)
    query = _query()
    first, second = _source("W1"), _source("W2")
    retrieved = _retrieved(first, second)
    for source in retrieved.sources:
        store.record_source(source)
    record = store.record_search(search_record_from(query, retrieved))

    fresh = LiteratureStore(tmp_path)
    replayed = fresh.completed_search("scripted", query.fingerprint)
    assert replayed == record
    assert replayed is not None
    assert replayed.id == record.id  # identity recomputed, not trusted
    assert replayed.source_ids == (first.id, second.id)
    assert replayed.provider_work_ids == ("W1", "W2")
    assert fresh.load_sources(replayed) == (first, second)
    # Serialization is deterministic: a re-record of the reloaded record
    # is byte-identical, hence a no-op.
    assert fresh.record_search(replayed) == record


def test_a_second_completed_search_for_the_same_query_conflicts(
    tmp_path: Path,
) -> None:
    store = LiteratureStore(tmp_path)
    query = _query()
    source = store.record_source(_source())
    first = store.record_search(
        search_record_from(query, _retrieved(source))
    )
    rerun = search_record_from(
        query, _retrieved(source, retrieved_at="2026-08-18T13:00:00+00:00")
    )
    assert rerun.id != first.id  # a later run is a distinct record
    with pytest.raises(LiteratureConflictError, match="already replays"):
        store.record_search(rerun)

    # A deliberate re-run is recordable as data without stealing the
    # replay seat.
    store.record_search(rerun, replayable=False)
    assert store.completed_search("scripted", query.fingerprint) == first
    assert len(store.searches()) == 2


def test_the_search_bound_fails_loudly(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path, max_searches=1)
    source = store.record_source(_source())
    store.record_search(search_record_from(_query(), _retrieved(source)))
    with pytest.raises(LiteratureCorpusLimitError):
        store.record_search(
            search_record_from(_query(text="other topic"), _retrieved(source))
        )


def test_a_tampered_search_fails_on_load(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path)
    source = store.record_source(_source())
    record = store.record_search(
        search_record_from(_query(), _retrieved(source))
    )
    path = tmp_path / "searches" / f"{record.id}.json"
    payload = json.loads(path.read_text())
    payload["source_ids"] = []
    payload["provider_work_ids"] = []
    path.write_text(json.dumps(payload))

    with pytest.raises(LiteratureIntegrityError, match="re-derives"):
        store.get_search(record.id)


def test_a_search_citing_a_vanished_source_fails_on_load(
    tmp_path: Path,
) -> None:
    store = LiteratureStore(tmp_path)
    source = store.record_source(_source())
    record = store.record_search(
        search_record_from(_query(), _retrieved(source))
    )
    (tmp_path / "sources" / f"{source.id}.json").unlink()
    with pytest.raises(LiteratureIntegrityError, match="no longer holds"):
        store.load_sources(record)


def test_a_replay_index_pointing_nowhere_fails_loudly(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path)
    query = _query()
    source = store.record_source(_source())
    record = store.record_search(search_record_from(query, _retrieved(source)))
    (tmp_path / "searches" / f"{record.id}.json").unlink()
    with pytest.raises(LiteratureIntegrityError, match="no longer holds"):
        store.completed_search("scripted", query.fingerprint)


def test_a_replay_index_for_the_wrong_query_fails_loudly(
    tmp_path: Path,
) -> None:
    """A forged index entry pointing the query at some *other* completed
    search must not replay it: the record's own fingerprint is checked."""
    store = LiteratureStore(tmp_path)
    query, other = _query(), _query(text="other topic")
    source = store.record_source(_source())
    store.record_search(search_record_from(query, _retrieved(source)))
    other_record = store.record_search(
        search_record_from(other, _retrieved(source)), replayable=False
    )

    index_path = next((tmp_path / "queries").glob("*.json"))
    forged = json.loads(index_path.read_text())
    forged["search_id"] = other_record.id
    index_path.write_text(json.dumps(forged))

    with pytest.raises(LiteratureIntegrityError, match="different query"):
        store.completed_search("scripted", query.fingerprint)


def test_an_unknown_query_is_a_miss_not_an_error(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path)
    assert store.completed_search("scripted", "litq_missing") is None


# -- the corpus ---------------------------------------------------------------


def test_a_live_search_is_recorded_and_an_identical_one_replays(
    tmp_path: Path,
) -> None:
    query = _query()
    retrieved = _retrieved(_source("W1"), _source("W2"))
    provider = ScriptedLiteratureProvider([retrieved])
    corpus = LiteratureCorpus(LiteratureStore(tmp_path), provider)

    live = corpus.search(query)
    assert live.from_cache is False
    assert live.sources == retrieved.sources

    replay = corpus.search(query)
    assert replay.from_cache is True
    assert replay.record == live.record
    assert replay.sources == live.sources
    # The provider was consulted exactly once: the replay cost nothing.
    assert len(provider.queries) == 1


def test_a_cache_hit_makes_zero_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The literal wire-level guarantee, proven against the real adapter:
    with the completed search on disk, the connection seam is never
    touched."""
    query = _query()
    store = LiteratureStore(tmp_path)
    source = _source(provider="openalex")
    store.record_source(source)
    retrieved = RetrievedSearch(
        provider="openalex",
        retrieved_at="2026-08-18T12:00:00+00:00",
        request_params={"search": "topic"},
        total_count=1,
        pages_fetched=1,
        page_identifiers=("ray-1",),
        rate_limit={},
        truncated=False,
        sources=(source,),
    )
    store.record_search(search_record_from(query, retrieved))

    def refuse_network(base_url: str, timeout: float) -> object:
        raise AssertionError("a cache hit must not open a connection")

    monkeypatch.setattr(openalex, "_connect", refuse_network)
    corpus = LiteratureCorpus(store, OpenAlexProvider())
    replay = corpus.search(query)
    assert replay.from_cache is True
    assert replay.sources == (source,)


def test_different_queries_do_not_share_a_cache_seat(tmp_path: Path) -> None:
    provider = ScriptedLiteratureProvider(
        [_retrieved(_source("W1")), _retrieved(_source("W2"))]
    )
    corpus = LiteratureCorpus(LiteratureStore(tmp_path), provider)
    corpus.search(_query())
    second = corpus.search(_query(text="other topic"))
    assert second.from_cache is False
    assert len(provider.queries) == 2


def test_the_cache_is_keyed_by_provider_as_well(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path)
    first = LiteratureCorpus(
        store, ScriptedLiteratureProvider([_retrieved(_source())])
    )
    first.search(_query())

    other_retrieval = RetrievedSearch(
        provider="other",
        retrieved_at="2026-08-18T12:00:00+00:00",
        request_params={},
        total_count=0,
        pages_fetched=1,
        page_identifiers=("",),
        rate_limit={},
        truncated=False,
        sources=(),
    )
    other = LiteratureCorpus(
        store,
        ScriptedLiteratureProvider([other_retrieval], name="other"),
    )
    assert other.search(_query()).from_cache is False


def test_a_provider_misreporting_its_name_is_refused(tmp_path: Path) -> None:
    provider = ScriptedLiteratureProvider(
        [_retrieved(_source())], name="masquerade"
    )
    corpus = LiteratureCorpus(LiteratureStore(tmp_path), provider)
    with pytest.raises(LiteratureConflictError, match="wrong name"):
        corpus.search(_query())
