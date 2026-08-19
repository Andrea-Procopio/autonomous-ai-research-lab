"""Task 5D live proof: an adversarial prior-art challenge over the
preserved Task 5C candidate portfolio.

Three roots: the preserved Task 5B.1 run (read-only — its literature
store resolves the candidates' cited sources), the preserved Task 5C run
(read-only — the immutable portfolio under challenge), and a fresh run
root that receives the prior-art records and the fresh challenge
corpus. Fresh candidate-specific OpenAlex searches are the point: the
challenge retrieves its own evidence rather than reusing the map's::

    the preserved candidate portfolio
      -> require_candidates_for_prior_art     (before any model call)
      -> per candidate: one gated query-proposal call,
         trusted retrieval (dates, ordering, budgets from the directive),
         cited-source injection + dedup, cutoff filter,
         gated similarity screening, one gated comparison call,
         trusted coverage and the deterministic verdict
      -> one run record with full lineage, durably recorded

Success is defined by correct records and fail-closed verdicts — an
OVERLAPPING or NOVELTY_UNRESOLVED result is a successful scientific
outcome, and nothing here is tuned to preserve the candidates. Requires
MUSE_API_KEY (or MODEL_API_KEY), outbound HTTPS to Muse and
api.openalex.org (OPENALEX_API_KEY optional — raises the daily credit
budget). Run with::

    python -m examples.live_task5d \\
        --map-root live_runs/task5b1-2026-08-18 \\
        --idea-root live_runs/task5c-2026-08-19 \\
        --run-root live_runs/task5d-<date>

The run root must sit under the gitignored ``live_runs/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.literature.corpus import LiteratureCorpus
from autonomous_research_lab.literature.openalex import OpenAlexProvider
from autonomous_research_lab.literature.retrieval import (
    LiteratureProvider,
    LiteratureQuery,
    RetrievedSearch,
)
from autonomous_research_lab.literature.store import LiteratureStore
from autonomous_research_lab.priorart.challenger import (
    PriorArtChallenger,
    PriorArtRunResult,
)
from autonomous_research_lab.priorart.directive import PriorArtDirective
from autonomous_research_lab.priorart.records import PriorArtQueryFamily
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.runtime.muse import (
    KEY_ENV_VARS,
    MUSE_SPARK_1_2,
    MuseSparkProvider,
)
from autonomous_research_lab.runtime.providers import UsageLedger

#: The preserved Task 5C portfolio this challenge enters through, and
#: its full lineage. Ids, never paths: the records prove their own
#: identity wherever the stores live.
IDEATION_RUN_RECORD_ID = "irun_23e08da6543a3bb6"
ASSESSMENT_ID = "madq_1bbb287a57486d0f"
MAP_RUN_ID = "map_4414ba86ab4e468d"
SNAPSHOT_ID = "cfp_1cad211804dc2cb8"
EXPECTED_CANDIDATE_IDS = (
    "idea_1e1fa63952cc0d91",
    "idea_1fd7f11cbcdda4ca",
    "idea_79acb8c5851b8839",
)

#: The cutoff matches the Task 5B.1 brief's: prior art is assessed as of
#: the day the map's own retrieval stopped, so the challenge and the
#: grounding describe the same literature moment. The recent window
#: mirrors the brief's one-year recency convention.
DIRECTIVE = PriorArtDirective(
    ideation_run_record_id=IDEATION_RUN_RECORD_ID,
    cutoff_date="2026-08-18",
    recent_window_start="2025-08-18",
)

#: What producing this challenge's entire input cost (the Task 5C
#: ideation run). The challenge spend below is on top.
IDEATION_BASELINE = {
    "model_calls": 3,
    "input_tokens": 18_970,
    "output_tokens": 16_148,
}


class _CountingProvider(LiteratureProvider):
    """Pass-through wrapper that counts live searches, so replay can
    prove zero network calls as a fact rather than a claim."""

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
        "--map-root",
        type=Path,
        required=True,
        help="the preserved Task 5B.1 run root (read-only input)",
    )
    parser.add_argument(
        "--idea-root",
        type=Path,
        required=True,
        help="the preserved Task 5C run root (read-only input)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="run directory (put it under the gitignored live_runs/)",
    )
    arguments = parser.parse_args()
    map_root = arguments.map_root.resolve()
    idea_root = arguments.idea_root.resolve()
    run_root = arguments.run_root.resolve()

    if not any(os.environ.get(name, "").strip() for name in KEY_ENV_VARS):
        print(
            f"FATAL: no Muse API key in the environment "
            f"({' or '.join(KEY_ENV_VARS)})"
        )
        return 1
    ideation_store = IdeationStore(idea_root / "ideation")
    if ideation_store.get_run(IDEATION_RUN_RECORD_ID) is None:
        print(
            f"FATAL: {idea_root} does not hold run record "
            f"{IDEATION_RUN_RECORD_ID}; point --idea-root at the "
            f"preserved task5c run root"
        )
        return 1
    known_literature = LiteratureStore(map_root / "literature")
    if known_literature.source_count() == 0:
        print(
            f"FATAL: {map_root} holds no literature sources; point "
            f"--map-root at the preserved task5b1 run root"
        )
        return 1

    counting = _CountingProvider(OpenAlexProvider())
    corpus = LiteratureCorpus(
        LiteratureStore(run_root / "literature"), counting
    )
    ledger = UsageLedger()
    challenger = PriorArtChallenger(
        provider=MuseSparkProvider(),
        model=MUSE_SPARK_1_2,
        ledger=ledger,
        ideation_store=ideation_store,
        known_literature=known_literature,
        corpus=corpus,
        store=PriorArtStore(run_root / "priorart"),
        # 16384, the Task 5C lesson: the comparison reply is one large
        # JSON object over several works; the budget has to fit it.
        max_output_tokens=16384,
        temperature=0.0,
        request_timeout_seconds=240.0,
    )

    print("== Task 5D live proof: adversarial prior-art challenge ==")
    print(f"directive  : {DIRECTIVE.id}")
    print(f"portfolio  : {IDEATION_RUN_RECORD_ID} (assessment "
          f"{ASSESSMENT_ID}, map run {MAP_RUN_ID}, cfp {SNAPSHOT_ID})")
    print(f"cutoff     : {DIRECTIVE.cutoff_date} (recent window from "
          f"{DIRECTIVE.recent_window_start})")
    print(f"budgets    : {DIRECTIVE.results_per_query} results/query, "
          f"{DIRECTIVE.max_screened_per_candidate} screened and "
          f"{DIRECTIVE.max_compared_works} compared per candidate, "
          f"{DIRECTIVE.max_model_calls} model calls")
    print(f"map root   : {map_root} (read-only)")
    print(f"idea root  : {idea_root} (read-only)")
    print(f"run root   : {run_root}")
    print("provider   : openalex (api.openalex.org, credential-free; "
          "fresh candidate-specific searches)")
    print()

    started = time.monotonic()
    result = challenger.run(DIRECTIVE)
    elapsed = time.monotonic() - started
    _report(result, run_root, elapsed, counting.searches)
    _verify_reload(result, run_root)
    _verify_replay(result, run_root, counting)
    print("PASSED: the door walked, every candidate was challenged with")
    print("fresh bounded retrieval, and a deterministic fail-closed")
    print("verdict was durably recorded per candidate. OVERLAPPING and")
    print("NOVELTY_UNRESOLVED are successful outcomes; only")
    print("infrastructure, gate exhaustion, or integrity failures are")
    print("not.")
    return 0


def _report(
    result: PriorArtRunResult,
    run_root: Path,
    elapsed: float,
    live_searches: int,
) -> None:
    record = result.run_record
    for candidate, assessment in zip(
        result.candidates, result.assessments, strict=True
    ):
        coverage = assessment.coverage
        print(f"-- candidate: {candidate.title} --")
        print(f"id         : {candidate.id}")
        executions = [
            e for e in result.executions if e.candidate_id == candidate.id
        ]
        for execution in executions:
            window = (
                f"{execution.from_date or 'open'}..{execution.to_date}"
            )
            print(
                f"query      : [{execution.family.value}] "
                f"{execution.text!r} ({window}, "
                f"{execution.ordering.value}) -> {execution.retrieved} "
                f"retrieved, {execution.new_unique} new"
                f"{' (cache)' if execution.from_cache else ''}"
            )
        print(
            f"pool       : {coverage.total_retrieved} retrieved + "
            f"{coverage.known_prior_art_listed} cited -> "
            f"{coverage.unique_sources} unique "
            f"({coverage.overlap} overlapping appearances, saturation "
            f"{coverage.saturation}); {coverage.post_cutoff_excluded} "
            f"post-cutoff excluded; {coverage.abstract_level} abstract / "
            f"{coverage.metadata_level} metadata-only; "
            f"{coverage.known_prior_art_recovered} cited source(s) "
            f"re-surfaced by fresh search"
        )
        print(
            f"screening  : {coverage.screened} screened "
            f"({coverage.screening_truncated} truncated) -> "
            f"{coverage.potential_overlap} potential overlap, "
            f"{coverage.related} related, {coverage.unrelated} "
            f"unrelated, {coverage.undecidable} undecidable; "
            f"{coverage.metadata_ambiguous} metadata-ambiguous"
        )
        comparisons = [
            entry
            for entry in result.comparisons
            if entry.candidate_id == candidate.id
        ]
        for comparison in comparisons:
            known = " (cited by the candidate)" if (
                comparison.known_prior_art
            ) else ""
            print(
                f"nearest    : {comparison.source_id} "
                f"[{comparison.similarity.value}]{known}"
            )
            for dimension in comparison.dimensions:
                print(
                    f"  {dimension.dimension.value}: candidate "
                    f"{dimension.candidate_position} | prior work "
                    f"{dimension.prior_work_position} "
                    f"[{dimension.support_location.value}: "
                    f"{dimension.support_snippet!r}]"
                )
            for feature in comparison.overlap_features:
                print(f"  overlap : {feature}")
            for difference in comparison.material_differences:
                print(f"  differs : {difference}")
        print(f"verdict    : {assessment.verdict.value.upper()}")
        for reason in assessment.reasons:
            print(f"  reason   : [{reason.code.value}] {reason.detail}")
        if assessment.overlapping_work_ids:
            print(
                f"  overlaps : "
                f"{', '.join(assessment.overlapping_work_ids)}"
            )
        print()

    print("-- spend (on top of the Task 5C ideation baseline) --")
    credits = _openalex_credits(run_root)
    repairs = _repair_total(result)
    rejected = len(PriorArtStore(run_root / "priorart").rejected())
    print(
        f"model calls={record.model_calls}; tokens="
        f"{record.input_tokens}in/{record.output_tokens}out "
        f"(the portfolio cost {IDEATION_BASELINE['model_calls']} calls, "
        f"{IDEATION_BASELINE['input_tokens']}in/"
        f"{IDEATION_BASELINE['output_tokens']}out)"
    )
    print(
        f"openalex   : {live_searches} live searches, "
        f"{credits} credits; wall clock {elapsed:.0f}s; "
        f"repairs={repairs}; rejected payloads preserved={rejected}"
    )
    print()
    print("-- limitations (by construction) --")
    print("* every verdict describes this bounded corpus: DISTINGUISHED")
    print("  means materially differentiated from the closest works this")
    print("  search surfaced under the recorded cutoff -- never proof of")
    print("  novelty; absence from this corpus is not novelty.")
    print("* comparisons are grounded in title/abstract text only; a")
    print("  metadata-only source that might overlap forces unresolved.")
    print("* citation counts ordered retrieval; they are not evidence of")
    print("  correctness or relevance.")
    print("* the candidates themselves are untouched: novelty on the 5C")
    print("  records stays structurally UNASSESSED, and these verdicts")
    print("  live beside them, not on them.")
    print()


def _openalex_credits(run_root: Path) -> str:
    store = LiteratureStore(run_root / "literature")
    total = 0
    for search in store.searches():
        total += int(
            search.rate_limit.get("openalex:credits_used_total", "0")
        )
    return str(total)


def _repair_total(result: PriorArtRunResult) -> int:
    seen: set[str] = set()
    total = 0
    provenances = [
        *(record.provenance for record in result.screenings),
        *(record.provenance for record in result.comparisons),
    ]
    for provenance in provenances:
        if provenance.response_id not in seen:
            seen.add(provenance.response_id)
            total += provenance.repair_count
    return total


def _verify_reload(result: PriorArtRunResult, run_root: Path) -> None:
    fresh = PriorArtStore(run_root / "priorart")
    record = result.run_record
    assert fresh.get_run(record.id) == record, "the run must reload intact"
    assert fresh.get_directive(DIRECTIVE.id) == DIRECTIVE
    assert record.candidate_ids == EXPECTED_CANDIDATE_IDS
    assert record.assessment_id == ASSESSMENT_ID
    assert record.map_run_id == MAP_RUN_ID
    assert record.snapshot_id == SNAPSHOT_ID
    for assessment in result.assessments:
        reloaded = fresh.get_prior_art_assessment(assessment.id)
        assert reloaded == assessment, "verdicts must reload intact"
    for execution in result.executions:
        assert fresh.get_query_execution(execution.id) == execution
    for screening in result.screenings:
        assert fresh.get_screening(screening.id) == screening
    for comparison in result.comparisons:
        assert fresh.get_comparison(comparison.id) == comparison
    print("-- durability --")
    print("reloaded the run, directive, every assessment, execution,")
    print("screening, and comparison via fresh objects: identical")
    print("identity throughout; the preserved 5B.1 and 5C inputs were")
    print("opened read-only and are byte-identical.")
    print()


def _verify_replay(
    result: PriorArtRunResult, run_root: Path, counting: _CountingProvider
) -> None:
    """Rebuild every executed query from its durable record and re-run
    it against the same corpus root: every one must replay from the
    store with zero further network calls."""
    corpus = LiteratureCorpus(
        LiteratureStore(run_root / "literature"), counting
    )
    before = counting.searches
    for execution in result.executions:
        query = LiteratureQuery(
            text=execution.text,
            from_date=execution.from_date,
            to_date=execution.to_date,
            per_page=min(DIRECTIVE.results_per_query, 25),
            max_results=DIRECTIVE.results_per_query,
            ordering=execution.ordering,
        )
        assert query.fingerprint == execution.query_fingerprint, (
            "the durable execution record must rebuild its exact query"
        )
        replay = corpus.search(query)
        assert replay.from_cache, "an identical query must replay"
        assert replay.record.id == execution.search_record_id
    assert counting.searches == before, "replay must make no network calls"
    families = sorted(
        {e.family for e in result.executions},
        key=list(PriorArtQueryFamily).index,
    )
    print("-- replay --")
    print(
        f"re-ran all {len(result.executions)} recorded queries "
        f"({len(families)} families) from their durable records: every "
        f"one replayed from the corpus with zero network calls."
    )
    print()


if __name__ == "__main__":
    sys.exit(main())
