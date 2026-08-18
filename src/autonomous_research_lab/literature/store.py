"""Durable literature provenance: what was searched, what came back, and in
what order — mirroring the planning and implementation stores.

Three write-once layers under one injected root:

* **Source records** (``sources/<id>.json``) — normalized
  :class:`~.retrieval.LiteratureSource` snapshots, content-addressed: the
  same reported metadata is the same file, however many searches returned
  it. Two snapshots of one paper that differ (a grown citation count) are
  two records; recognizing them as one *work* is the deduplication layer's
  job, never the store's.
* **Search records** (``searches/<id>.json``) — one completed bounded
  search: the exact query, the provider parameters actually sent, the
  retrieval timestamp, pagination and rate-limit observations, and the
  returned source ids in provider order. A search record may only be
  written after every source it references — provenance is recorded
  sources-first, so a search can never cite a snapshot the store does not
  hold.
* **The replay index** (``queries/<key>.json``) — query fingerprint (plus
  provider) to search id, write-once: one completed search per query is
  the replayable one, and an identical later query is served from the
  record instead of the network.

Every record's id is recomputed from what was read, never trusted from the
file: a record that no longer hashes to its name fails loudly on load.
Writes are write-once and verify-on-repeat: identical re-recording is a
no-op, different content under the same name raises. The store is bounded —
capped counts of searches and sources — so a corpus cannot grow without
limit; write-once means eviction is off the table, so the cap fails loudly
instead of silently discarding.

Nothing here may ever hold a credential: records store queries, parameters,
identifiers and metadata text, not keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..core.ids import content_id
from ..core.types import freeze_mapping
from .retrieval import (
    AccessLevel,
    LiteratureQuery,
    LiteratureSource,
    ResultOrdering,
    RetrievedSearch,
)

_SEARCHES_DIRNAME: Final = "searches"
_SOURCES_DIRNAME: Final = "sources"
_QUERIES_DIRNAME: Final = "queries"
_RECORD_SUFFIX: Final = ".json"

DEFAULT_MAX_SEARCHES: Final = 1_024
DEFAULT_MAX_SOURCES: Final = 100_000


class LiteratureConflictError(RuntimeError):
    """A write-once literature artifact would be overwritten with different
    content, or a record would cite a source the store does not hold."""


class LiteratureIntegrityError(RuntimeError):
    """A stored literature record no longer matches its own identity."""


class LiteratureCorpusLimitError(RuntimeError):
    """The bounded corpus is full. Write-once storage cannot evict, so the
    bound fails loudly; a larger investigation needs a new corpus root or
    an explicitly larger bound."""


@dataclass(frozen=True, slots=True)
class LiteratureSearchRecord:
    """One completed bounded search with its complete provenance.

    The id derives from every field — the retrieval timestamp included, so
    two identical searches completed at different times are distinct
    records — while the *replay* identity is ``query_fingerprint``, which
    covers only what determines the result set.
    """

    provider: str
    query_text: str
    from_date: str
    to_date: str
    per_page: int
    max_results: int
    query_fingerprint: str
    retrieved_at: str
    request_params: Mapping[str, str]
    total_count: int | None
    pages_fetched: int
    page_identifiers: tuple[str, ...]
    rate_limit: Mapping[str, str]
    truncated: bool
    source_ids: tuple[str, ...]
    """Normalized source content ids, in provider order — the corpus's
    ground truth for what this search returned, and in what order."""

    provider_work_ids: tuple[str, ...]
    """The provider's own work ids, same order, so the record stays
    interpretable against the provider without loading every source."""

    ordering: ResultOrdering = ResultOrdering.RECENCY
    """How the provider was asked to order the slice. Recency is the
    historical default; like the query fingerprint, it joins the record
    id and the serialized payload only when it is not that default, so
    every record written before orderings existed still re-derives its
    own id on load."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_params", freeze_mapping(self.request_params)
        )
        object.__setattr__(self, "rate_limit", freeze_mapping(self.rate_limit))
        if len(self.provider_work_ids) != len(self.source_ids):
            raise ValueError(
                "provider_work_ids and source_ids must align one-to-one"
            )
        if not self.id:
            parts: tuple[object, ...] = (
                self.provider,
                self.query_text,
                self.from_date,
                self.to_date,
                self.per_page,
                self.max_results,
                self.query_fingerprint,
                self.retrieved_at,
                self.request_params,
                self.total_count,
                self.pages_fetched,
                self.page_identifiers,
                self.rate_limit,
                self.truncated,
                self.source_ids,
                self.provider_work_ids,
            )
            if self.ordering is not ResultOrdering.RECENCY:
                parts = (*parts, self.ordering)
            object.__setattr__(self, "id", content_id("lits", *parts))


def search_record_from(
    query: LiteratureQuery, retrieved: RetrievedSearch
) -> LiteratureSearchRecord:
    """The durable record of one completed retrieval of one query."""
    return LiteratureSearchRecord(
        provider=retrieved.provider,
        query_text=query.text,
        from_date=query.from_date,
        to_date=query.to_date,
        per_page=query.per_page,
        max_results=query.max_results,
        query_fingerprint=query.fingerprint,
        retrieved_at=retrieved.retrieved_at,
        request_params=retrieved.request_params,
        total_count=retrieved.total_count,
        pages_fetched=retrieved.pages_fetched,
        page_identifiers=retrieved.page_identifiers,
        rate_limit=retrieved.rate_limit,
        truncated=retrieved.truncated,
        source_ids=tuple(source.id for source in retrieved.sources),
        provider_work_ids=tuple(
            source.provider_id for source in retrieved.sources
        ),
        ordering=query.ordering,
    )


class LiteratureStore:
    """File-backed, write-once, bounded storage for literature searches and
    normalized sources, under one injected root."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> None:
        if max_searches < 1 or max_sources < 1:
            raise ValueError("store bounds must be positive")
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_searches = max_searches
        self._max_sources = max_sources

    @property
    def root(self) -> Path:
        return self._root

    # -- sources ---------------------------------------------------------------

    def _source_path(self, source_id: str) -> Path:
        return self._root / _SOURCES_DIRNAME / f"{source_id}{_RECORD_SUFFIX}"

    def record_source(self, source: LiteratureSource) -> LiteratureSource:
        """Store one normalized source, write-once. Identical re-recording
        is a no-op; different content under the same id raises."""
        existing = self.get_source(source.id)
        if existing is not None:
            if existing != source:
                raise LiteratureConflictError(
                    f"literature source {source.id} is already recorded with "
                    f"different content; records are never rewritten"
                )
            return existing
        if self.source_count() >= self._max_sources:
            raise LiteratureCorpusLimitError(
                f"the corpus already holds {self._max_sources} sources; "
                f"refusing to grow without bound"
            )
        path = self._source_path(source.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_source_payload(source), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return source

    def get_source(self, source_id: str) -> LiteratureSource | None:
        path = self._source_path(source_id)
        if not path.exists():
            return None
        # The id is recomputed from what was read, never trusted from the
        # file: a record that no longer hashes to its name fails loudly.
        source = _source_from(json.loads(path.read_text(encoding="utf-8")))
        if source.id != source_id:
            raise LiteratureIntegrityError(
                f"literature source filed under {source_id} re-derives id "
                f"{source.id}; refusing to load a record that no longer "
                f"matches its name"
            )
        return source

    def source_count(self) -> int:
        directory = self._root / _SOURCES_DIRNAME
        if not directory.is_dir():
            return 0
        return sum(1 for _ in directory.glob(f"*{_RECORD_SUFFIX}"))

    def sources(self) -> tuple[LiteratureSource, ...]:
        directory = self._root / _SOURCES_DIRNAME
        if not directory.exists():
            return ()
        loaded = []
        for path in sorted(directory.glob(f"*{_RECORD_SUFFIX}")):
            source = self.get_source(path.stem)
            assert source is not None  # listed from the directory just above
            loaded.append(source)
        return tuple(loaded)

    # -- searches --------------------------------------------------------------

    def _search_path(self, search_id: str) -> Path:
        return self._root / _SEARCHES_DIRNAME / f"{search_id}{_RECORD_SUFFIX}"

    def record_search(
        self, record: LiteratureSearchRecord, *, replayable: bool = True
    ) -> LiteratureSearchRecord:
        """Store one completed search, write-once, sources-first: every
        source the record cites must already be recorded.

        ``replayable`` also files the search under its query fingerprint so
        an identical later query replays from the record. One completed
        search per query holds that seat; recording a second one for the
        same query is a conflict unless ``replayable=False`` says this
        retrieval is deliberately a re-run.
        """
        for source_id in record.source_ids:
            if not self._source_path(source_id).exists():
                raise LiteratureConflictError(
                    f"search {record.id} cites source {source_id}, which is "
                    f"not recorded; provenance is recorded sources-first"
                )
        existing = self.get_search(record.id)
        if existing is not None:
            if existing != record:
                raise LiteratureConflictError(
                    f"literature search {record.id} is already recorded with "
                    f"different content; records are never rewritten"
                )
        else:
            if self.search_count() >= self._max_searches:
                raise LiteratureCorpusLimitError(
                    f"the corpus already holds {self._max_searches} "
                    f"searches; refusing to grow without bound"
                )
            path = self._search_path(record.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(_search_payload(record), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if replayable:
            self._index_query(record)
        return record

    def get_search(self, search_id: str) -> LiteratureSearchRecord | None:
        path = self._search_path(search_id)
        if not path.exists():
            return None
        record = _search_from(json.loads(path.read_text(encoding="utf-8")))
        if record.id != search_id:
            raise LiteratureIntegrityError(
                f"literature search filed under {search_id} re-derives id "
                f"{record.id}; refusing to load a record that no longer "
                f"matches its name"
            )
        return record

    def search_count(self) -> int:
        directory = self._root / _SEARCHES_DIRNAME
        if not directory.is_dir():
            return 0
        return sum(1 for _ in directory.glob(f"*{_RECORD_SUFFIX}"))

    def searches(self) -> tuple[LiteratureSearchRecord, ...]:
        directory = self._root / _SEARCHES_DIRNAME
        if not directory.exists():
            return ()
        loaded = []
        for path in sorted(directory.glob(f"*{_RECORD_SUFFIX}")):
            record = self.get_search(path.stem)
            assert record is not None  # listed from the directory just above
            loaded.append(record)
        return tuple(loaded)

    def load_sources(
        self, record: LiteratureSearchRecord
    ) -> tuple[LiteratureSource, ...]:
        """The record's sources, in provider order. A cited source the
        store cannot produce — or produces tampered — fails loudly."""
        loaded = []
        for source_id in record.source_ids:
            source = self.get_source(source_id)
            if source is None:
                raise LiteratureIntegrityError(
                    f"search {record.id} cites source {source_id}, which the "
                    f"store no longer holds"
                )
            loaded.append(source)
        return tuple(loaded)

    # -- the replay index ------------------------------------------------------

    def _query_key(self, provider: str, query_fingerprint: str) -> str:
        return content_id("litidx", provider, query_fingerprint)

    def _query_path(self, provider: str, query_fingerprint: str) -> Path:
        key = self._query_key(provider, query_fingerprint)
        return self._root / _QUERIES_DIRNAME / f"{key}{_RECORD_SUFFIX}"

    def _index_query(self, record: LiteratureSearchRecord) -> None:
        path = self._query_path(record.provider, record.query_fingerprint)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("search_id") != record.id:
                raise LiteratureConflictError(
                    f"query {record.query_fingerprint} already replays from "
                    f"search {payload.get('search_id')}; a deliberate re-run "
                    f"must be recorded with replayable=False"
                )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "provider": record.provider,
                    "query_fingerprint": record.query_fingerprint,
                    "search_id": record.id,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def completed_search(
        self, provider: str, query_fingerprint: str
    ) -> LiteratureSearchRecord | None:
        """The replayable completed search for this query, or ``None``. An
        index entry whose record is missing, tampered, or filed for a
        different query fails loudly rather than replaying the wrong
        search."""
        path = self._query_path(provider, query_fingerprint)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        search_id = payload.get("search_id")
        record = self.get_search(str(search_id)) if search_id else None
        if record is None:
            raise LiteratureIntegrityError(
                f"query {query_fingerprint} points at search {search_id}, "
                f"which the store no longer holds"
            )
        if (
            record.query_fingerprint != query_fingerprint
            or record.provider != provider
        ):
            raise LiteratureIntegrityError(
                f"query {query_fingerprint} points at search {record.id}, "
                f"which records a different query or provider"
            )
        return record


# -- serialization ------------------------------------------------------------


def _source_payload(source: LiteratureSource) -> dict[str, object]:
    return {
        "id": source.id,
        "provider": source.provider,
        "provider_id": source.provider_id,
        "title": source.title,
        "authors": list(source.authors),
        "publication_date": source.publication_date,
        "publication_year": source.publication_year,
        "venue": source.venue,
        "work_type": source.work_type,
        "abstract": source.abstract,
        "doi": source.doi,
        "arxiv_id": source.arxiv_id,
        "provider_url": source.provider_url,
        "landing_page_url": source.landing_page_url,
        "pdf_url": source.pdf_url,
        "cited_by_count": source.cited_by_count,
        "referenced_work_ids": list(source.referenced_work_ids),
        "access_level": source.access_level.value,
    }


def _source_from(payload: Mapping[str, object]) -> LiteratureSource:
    authors = payload["authors"]
    referenced = payload["referenced_work_ids"]
    assert isinstance(authors, list)
    assert isinstance(referenced, list)
    return LiteratureSource(
        provider=str(payload["provider"]),
        provider_id=str(payload["provider_id"]),
        title=_opt_str(payload["title"]),
        authors=tuple(str(item) for item in authors),
        publication_date=_opt_str(payload["publication_date"]),
        publication_year=_opt_int(payload["publication_year"]),
        venue=_opt_str(payload["venue"]),
        work_type=_opt_str(payload["work_type"]),
        abstract=_opt_str(payload["abstract"]),
        doi=_opt_str(payload["doi"]),
        arxiv_id=_opt_str(payload["arxiv_id"]),
        provider_url=str(payload["provider_url"]),
        landing_page_url=_opt_str(payload["landing_page_url"]),
        pdf_url=_opt_str(payload["pdf_url"]),
        cited_by_count=_opt_int(payload["cited_by_count"]),
        referenced_work_ids=tuple(str(item) for item in referenced),
        access_level=AccessLevel(str(payload["access_level"])),
    )


def _search_payload(record: LiteratureSearchRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": record.id,
        "provider": record.provider,
        "query_text": record.query_text,
        "from_date": record.from_date,
        "to_date": record.to_date,
        "per_page": record.per_page,
        "max_results": record.max_results,
        "query_fingerprint": record.query_fingerprint,
        "retrieved_at": record.retrieved_at,
        "request_params": dict(record.request_params),
        "total_count": record.total_count,
        "pages_fetched": record.pages_fetched,
        "page_identifiers": list(record.page_identifiers),
        "rate_limit": dict(record.rate_limit),
        "truncated": record.truncated,
        "source_ids": list(record.source_ids),
        "provider_work_ids": list(record.provider_work_ids),
    }
    if record.ordering is not ResultOrdering.RECENCY:
        # Mirrors the id rule: recency records keep the exact byte
        # layout they had before orderings existed, so an identical
        # re-record of a v1 file stays a no-op.
        payload["ordering"] = record.ordering.value
    return payload


def _search_from(payload: Mapping[str, object]) -> LiteratureSearchRecord:
    request_params = payload["request_params"]
    rate_limit = payload["rate_limit"]
    page_identifiers = payload["page_identifiers"]
    source_ids = payload["source_ids"]
    provider_work_ids = payload["provider_work_ids"]
    assert isinstance(request_params, Mapping)
    assert isinstance(rate_limit, Mapping)
    assert isinstance(page_identifiers, list)
    assert isinstance(source_ids, list)
    assert isinstance(provider_work_ids, list)
    return LiteratureSearchRecord(
        provider=str(payload["provider"]),
        query_text=str(payload["query_text"]),
        from_date=str(payload["from_date"]),
        to_date=str(payload["to_date"]),
        per_page=int(str(payload["per_page"])),
        max_results=int(str(payload["max_results"])),
        query_fingerprint=str(payload["query_fingerprint"]),
        retrieved_at=str(payload["retrieved_at"]),
        request_params={
            str(k): str(v) for k, v in request_params.items()
        },
        total_count=_opt_int(payload["total_count"]),
        pages_fetched=int(str(payload["pages_fetched"])),
        page_identifiers=tuple(str(item) for item in page_identifiers),
        rate_limit={str(k): str(v) for k, v in rate_limit.items()},
        truncated=bool(payload["truncated"]),
        source_ids=tuple(str(item) for item in source_ids),
        provider_work_ids=tuple(str(item) for item in provider_work_ids),
        ordering=ResultOrdering(
            str(payload.get("ordering", ResultOrdering.RECENCY.value))
        ),
    )


def _opt_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _opt_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
