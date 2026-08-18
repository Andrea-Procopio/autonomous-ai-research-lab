"""Task 5A live proof: one real bounded literature search, preserved,
reloaded, and replayed.

One trajectory, live, with nothing mocked::

    bounded query (broad ML topic, explicit date range, result budget)
      -> OpenAlex /works search, cursor-paged, deterministically sorted
      -> normalized source records + a durable write-once search record
      -> reload from disk with fresh objects (identities recomputed)
      -> the identical query again: replayed from the corpus, zero
         network calls
      -> deterministic deduplication report over the retrieved slice

Success is defined by correct state transitions and provenance — records
that round-trip to the same identities and a replay that never touches the
network — never by what the papers say. Requires nothing but outbound
HTTPS: OpenAlex needs no credential for basic use (an optional
``OPENALEX_API_KEY`` raises the daily budget; this driver never prints or
stores it). Run with::

    python -m examples.live_literature --run-root live_runs/task5a-<date>

The run root is expected to sit under the gitignored ``live_runs/``; the
records it accumulates are live payloads and must not be committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autonomous_research_lab.literature.corpus import LiteratureCorpus
from autonomous_research_lab.literature.dedup import deduplicate
from autonomous_research_lab.literature.openalex import OpenAlexProvider
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureProvider,
    LiteratureQuery,
    RetrievedSearch,
)
from autonomous_research_lab.literature.store import LiteratureStore

QUERY = LiteratureQuery(
    text="in-context learning",
    from_date="2026-05-01",
    to_date="2026-08-18",
    per_page=25,
    max_results=75,
)


class CountingProvider(LiteratureProvider):
    """Wraps the real adapter to make 'zero network calls on replay' an
    assertable fact rather than a claim."""

    def __init__(self, inner: LiteratureProvider) -> None:
        self._inner = inner
        self.searches = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def search(self, query: LiteratureQuery) -> RetrievedSearch:
        self.searches += 1
        return self._inner.search(query)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="run directory (put it under the gitignored live_runs/)",
    )
    arguments = parser.parse_args()
    corpus_root = arguments.run_root.resolve() / "literature"

    provider = CountingProvider(OpenAlexProvider())
    store = LiteratureStore(corpus_root)
    corpus = LiteratureCorpus(store, provider)

    print("== Task 5A live proof: bounded literature retrieval ==")
    print("provider   : openalex (api.openalex.org, credential-free)")
    print(f"query      : {QUERY.text!r}")
    print(f"filters    : publication_date in [{QUERY.from_date}, {QUERY.to_date}]")
    print(f"budgets    : per_page={QUERY.per_page} max_results={QUERY.max_results}")
    print(f"fingerprint: {QUERY.fingerprint}")
    print(f"corpus root: {corpus_root}")
    print()

    live = corpus.search(QUERY)
    record = live.record
    if live.from_cache:
        print("NOTE: this query was already completed in this corpus; the")
        print("run below reports the recorded search rather than a new one.")
    print("-- retrieval --")
    print(f"retrieved_at   : {record.retrieved_at}")
    print(f"request params : {dict(record.request_params)}")
    print(f"pages fetched  : {record.pages_fetched}")
    print(f"page ids       : {record.page_identifiers}")
    print(f"total matching : {record.total_count} (provider-reported)")
    print(f"returned       : {len(live.sources)} (truncated={record.truncated})")
    print(f"rate limit     : {dict(record.rate_limit)}")
    print(f"search record  : {record.id}")
    print()

    with_doi = sum(1 for s in live.sources if s.doi is not None)
    with_arxiv = sum(1 for s in live.sources if s.arxiv_id is not None)
    with_abstract = sum(
        1 for s in live.sources if s.access_level is AccessLevel.ABSTRACT
    )
    with_date = sum(1 for s in live.sources if s.publication_date is not None)
    n = max(1, len(live.sources))
    print("-- identifier and metadata coverage --")
    print(f"provider ids   : {len(live.sources)}/{len(live.sources)}")
    print(f"DOIs           : {with_doi}/{len(live.sources)} ({100 * with_doi // n}%)")
    print(f"arXiv ids      : {with_arxiv}/{len(live.sources)}")
    print(f"abstracts      : {with_abstract}/{len(live.sources)}")
    print(f"metadata-only  : {len(live.sources) - with_abstract}/{len(live.sources)}")
    print(f"pub. dates     : {with_date}/{len(live.sources)}")
    print()

    # Reload everything through fresh objects: every identity is recomputed
    # from the bytes on disk and must land exactly where it started.
    fresh = LiteratureStore(corpus_root)
    reloaded = fresh.completed_search(provider.name, QUERY.fingerprint)
    assert reloaded is not None, "the completed search must be replayable"
    assert reloaded == record, "the search record must round-trip identically"
    resources = fresh.load_sources(reloaded)
    assert resources == live.sources, "sources must round-trip identically"
    assert [s.id for s in resources] == list(record.source_ids)
    print("-- durability --")
    print(f"reloaded search {reloaded.id} and {len(resources)} sources from")
    print("disk via fresh objects; every recomputed identity matched.")
    print()

    searches_before = provider.searches
    replay = corpus.search(QUERY)
    assert replay.from_cache, "an identical completed search must replay"
    assert provider.searches == searches_before, (
        "a replay must make zero network calls"
    )
    assert replay.record == record and replay.sources == live.sources
    print("-- cache --")
    print(
        f"identical query replayed from the corpus: from_cache=True, "
        f"provider consulted {provider.searches} time(s) in total."
    )
    print()

    report = deduplicate(live.sources)
    print("-- deduplication --")
    print(f"snapshots      : {report.total}")
    print(f"distinct works : {len(report.representative_ids)}")
    print(f"duplicates     : {report.duplicate_count}")
    for group in report.groups:
        print(f"  group {group.source_ids} matched on {group.matched_on}")
    print(f"conflicts      : {len(report.conflicts)}")
    for conflict in report.conflicts:
        print(f"  {conflict.kind} {conflict.key!r}: {conflict.detail}")
    print()

    print("-- known coverage limitations --")
    print("* abstracts arrive as an inverted index and some are absent for")
    print("  legal reasons; original whitespace is not recoverable.")
    print("* relevance ranking is not stable across index updates; this")
    print("  search sorts by publication_date:desc and the record preserves")
    print("  provider order, but a re-run on a later index may differ —")
    print("  which is exactly why the completed search replays from disk.")
    print("* recent-paper coverage depends on OpenAlex ingestion lag from")
    print("  Crossref/arXiv; the provider-reported total is an estimate.")
    print("* anonymous use spends a documented daily credit budget; the")
    print("  rate-limit observations above record what this run cost.")
    print()
    print("PASSED: retrieval, durable provenance, reload, replay, dedup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
