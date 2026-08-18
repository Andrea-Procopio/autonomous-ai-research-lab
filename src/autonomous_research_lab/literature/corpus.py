"""The reproducible local corpus: cache-or-live over one provider and one
store.

One rule: an identical completed search is never re-fetched. The corpus
looks the query's fingerprint up in the store first; a hit replays the
recorded search and its sources — zero network calls, tamper-checked on
load — and only a miss reaches the provider, after which the retrieval is
recorded sources-first and indexed for every later identical query.

The corpus stores and replays; it never interprets. What the papers mean,
whether they matter, and what to do about them are later stages' questions,
asked of the durable records rather than of the network.
"""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import LiteratureProvider, LiteratureQuery, LiteratureSource
from .store import (
    LiteratureConflictError,
    LiteratureSearchRecord,
    LiteratureStore,
    search_record_from,
)


@dataclass(frozen=True, slots=True)
class CorpusSearchResult:
    """One answered query: the durable search record, its sources in
    provider order, and whether the network was involved at all."""

    record: LiteratureSearchRecord
    sources: tuple[LiteratureSource, ...]
    from_cache: bool


class LiteratureCorpus:
    """Cache-or-live literature search over one provider and one store."""

    def __init__(
        self, store: LiteratureStore, provider: LiteratureProvider
    ) -> None:
        self._store = store
        self._provider = provider

    @property
    def store(self) -> LiteratureStore:
        return self._store

    def search(self, query: LiteratureQuery) -> CorpusSearchResult:
        """Answer ``query`` from the store when it already holds a
        completed identical search, from the provider otherwise."""
        cached = self._store.completed_search(
            self._provider.name, query.fingerprint
        )
        if cached is not None:
            return CorpusSearchResult(
                record=cached,
                sources=self._store.load_sources(cached),
                from_cache=True,
            )
        retrieved = self._provider.search(query)
        if retrieved.provider != self._provider.name:
            raise LiteratureConflictError(
                f"provider {self._provider.name!r} returned a retrieval "
                f"claiming provider {retrieved.provider!r}; refusing to "
                f"record provenance under the wrong name"
            )
        for source in retrieved.sources:
            self._store.record_source(source)
        record = self._store.record_search(search_record_from(query, retrieved))
        return CorpusSearchResult(
            record=record, sources=retrieved.sources, from_cache=False
        )
