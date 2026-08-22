"""Access resolution: upgrade a metadata-only work before it is recorded.

Task 5G closes the gap Task 5D.2 named: an attested material ambiguity
on a metadata-only work had no lawful in-repo path to an abstract. The
path is retrieval-time enrichment by trusted code — a wrapper around
the real provider that, for each returned work carrying an arXiv id
but no abstract, fetches the work's own abstract from its arXiv
listing and rebuilds the source at ``ABSTRACT`` access.

Why at retrieval time and nowhere else: a ``LiteratureSource`` is
content-addressed over every field and the store is write-once, so an
in-place upgrade is impossible by design — a resolved source is a
different record. Resolving before anything is recorded means the
stored corpus simply contains the better source, and every downstream
reference (screening, extraction, citations, the bibliography) names
it consistently.

Honesty: the fetched text is the work's own abstract, on the listing
the index's record names via ``arxiv_id`` — the same work, not a
substitute. The model is never involved; a fetch that fails leaves the
source honestly metadata-only, because resolution is an enrichment,
never a gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .arxiv import fetch_abstract
from .retrieval import (
    AccessLevel,
    LiteratureProvider,
    LiteratureQuery,
    LiteratureSource,
    RetrievedSearch,
)


class AccessResolvingProvider(LiteratureProvider):
    """The real provider, with metadata-only works resolved on the way
    out. Provider identity stays the index that found the work — the
    resolution is part of retrieval, not a second source of works."""

    def __init__(
        self,
        inner: LiteratureProvider,
        *,
        fetch: Callable[[str], str | None] = fetch_abstract,
    ) -> None:
        self._inner = inner
        self._fetch = fetch

    @property
    def name(self) -> str:
        return self._inner.name

    def search(self, query: LiteratureQuery) -> RetrievedSearch:
        retrieved = self._inner.search(query)
        resolved = tuple(
            self._resolved(source) for source in retrieved.sources
        )
        if all(
            after is before
            for after, before in zip(resolved, retrieved.sources, strict=True)
        ):
            return retrieved
        return replace(retrieved, sources=resolved)

    def _resolved(self, source: LiteratureSource) -> LiteratureSource:
        if source.access_level is not AccessLevel.METADATA:
            return source
        if source.arxiv_id is None or not source.arxiv_id.strip():
            return source
        abstract = self._fetch(source.arxiv_id)
        if abstract is None:
            return source
        return replace(
            source,
            abstract=abstract,
            access_level=AccessLevel.ABSTRACT,
            id="",
        )
