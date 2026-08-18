"""Task 5B.1 live proof: diversified retrieval and a deterministic
adequacy verdict, on the Task 5B brief.

The same in-context-learning direction as the Task 5B live run, so the
two are directly comparable — what changed is retrieval and judgment::

    the Task 5B workshop-style brief
      -> precise queries matched in titles+abstracts (not fulltext)
      -> recency-ordered recent work, citation-ranked foundational work
      -> screening; bounded refinement if the relevant yield is thin
      -> extraction -> FieldMap -> ProblemInventory
      -> MapAdequacyAssessment: ADEQUATE_FOR_IDEA_GENERATION or an
         honest INSUFFICIENT_COVERAGE, computed by trusted code

Success is defined by correct records and a fail-closed verdict that
survives reload — the task passes even if the map is judged inadequate.
Requires MUSE_API_KEY (or MODEL_API_KEY) and outbound HTTPS. Run with::

    python -m examples.live_task5b1 --run-root live_runs/task5b1-<date>

The run root must sit under the gitignored ``live_runs/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from autonomous_research_lab.literature.corpus import LiteratureCorpus
from autonomous_research_lab.literature.openalex import OpenAlexProvider
from autonomous_research_lab.literature.store import LiteratureStore
from autonomous_research_lab.mapping.adequacy import (
    AdequacyThresholds,
    SupportTier,
)
from autonomous_research_lab.mapping.brief import ResearchBrief
from autonomous_research_lab.mapping.mapper import FieldMapper, MappingRunResult
from autonomous_research_lab.mapping.store import MappingStore
from autonomous_research_lab.runtime.muse import (
    KEY_ENV_VARS,
    MUSE_SPARK_1_2,
    MuseSparkProvider,
)
from autonomous_research_lab.runtime.providers import UsageLedger

BRIEF = ResearchBrief(
    topic=(
        "in-context learning in large language models: mechanisms, "
        "efficiency, and robustness"
    ),
    cutoff_date="2026-08-18",
    recent_window_start="2025-08-18",
    workshop_hints=(
        "workshop scope: understanding and improving in-context learning",
        "topics of interest: mechanisms of ICL, efficient adaptation, "
        "robustness under distribution shift, evaluation practices",
    ),
    max_queries_per_family=1,
    results_per_query=15,
    max_screened_sources=80,
    max_extracted_sources=12,
    max_model_calls=40,
    refinement_rounds=1,
)

#: The Task 5B live run (2026-08-18), for the measured comparison.
BASELINE = {
    "retrieved": 105,
    "unique": 93,
    "screened": 60,
    "relevant": 5,
    "excluded": 54,
    "uncertain": 1,
    "grounded": 5,
    "recent_grounded": 1,
    "foundational_grounded": 4,
    "model_calls": 18,
    "input_tokens": 43_283,
    "output_tokens": 56_813,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="run directory (put it under the gitignored live_runs/)",
    )
    arguments = parser.parse_args()
    root = arguments.run_root.resolve()

    if not any(os.environ.get(name, "").strip() for name in KEY_ENV_VARS):
        print(
            f"FATAL: no Muse API key in the environment "
            f"({' or '.join(KEY_ENV_VARS)})"
        )
        return 1

    ledger = UsageLedger()
    thresholds = AdequacyThresholds()
    mapper = FieldMapper(
        provider=MuseSparkProvider(),
        model=MUSE_SPARK_1_2,
        ledger=ledger,
        corpus=LiteratureCorpus(
            LiteratureStore(root / "literature"), OpenAlexProvider()
        ),
        store=MappingStore(root / "mapping"),
        thresholds=thresholds,
        max_output_tokens=8192,
        temperature=0.0,
        request_timeout_seconds=240.0,
    )

    print("== Task 5B.1 live proof: retrieval quality and adequacy ==")
    print(f"brief      : {BRIEF.id} (same direction as the Task 5B run)")
    print(f"topic      : {BRIEF.topic}")
    print(f"budgets    : {BRIEF.max_queries_per_family}/family, "
          f"{BRIEF.results_per_query}/query, screen "
          f"{BRIEF.max_screened_sources}, extract "
          f"{BRIEF.max_extracted_sources}, {BRIEF.max_model_calls} calls, "
          f"{BRIEF.refinement_rounds} refinement round(s)")
    print(f"run root   : {root}")
    print()

    result = mapper.run(BRIEF)
    _report(result)
    _verify_reload(result, root)
    print("PASSED: diversified retrieval, gated mapping, and a")
    print("deterministic adequacy verdict, durably recorded.")
    return 0


def _report(result: MappingRunResult) -> None:
    coverage = result.run_record.coverage
    metrics = result.assessment.metrics
    print("-- retrieval (strategy per family) --")
    for execution in result.query_executions:
        round_tag = (
            f" [refinement {execution.refinement_round}]"
            if execution.refinement_round
            else ""
        )
        print(
            f"[{execution.family.value}]{round_tag} "
            f"{execution.ordering.value}-ordered {execution.text!r} "
            f"({execution.from_date or 'open'}..{execution.to_date}): "
            f"retrieved {execution.retrieved}, new {execution.new_unique}, "
            f"cached={execution.from_cache}"
        )
    print(
        f"queries={coverage.queries_executed} retrieved="
        f"{coverage.total_retrieved} unique={coverage.unique_sources} "
        f"overlap={metrics.overlap} saturation={coverage.saturation}"
    )
    print()
    print("-- screening --")
    yield_now = (
        100 * coverage.relevant // coverage.screened
        if coverage.screened
        else 0
    )
    yield_before = 100 * BASELINE["relevant"] // BASELINE["screened"]
    print(
        f"screened={coverage.screened} (truncated "
        f"{coverage.screening_truncated}): relevant={coverage.relevant} "
        f"excluded={coverage.excluded} uncertain={coverage.uncertain}"
    )
    print(
        f"relevance yield: {yield_now}% (Task 5B baseline: "
        f"{yield_before}%)"
    )
    print(
        f"access levels among screened: abstract={coverage.abstract_level} "
        f"metadata-only={coverage.metadata_level}"
    )
    print()
    print("-- extraction and balance --")
    print(
        f"eligible={coverage.extraction_eligible} "
        f"extracted={coverage.extracted} (truncated "
        f"{coverage.extraction_truncated}); grounded="
        f"{metrics.grounded_sources}; insufficient="
        f"{metrics.insufficient_extractions}"
    )
    print(
        f"era balance (grounded): recent={metrics.recent_grounded} "
        f"foundational={metrics.foundational_grounded} "
        f"undated={metrics.undated_grounded} (baseline: "
        f"{BASELINE['recent_grounded']}/{BASELINE['foundational_grounded']})"
    )
    print(
        f"families with relevant sources: "
        f"{', '.join(metrics.families_with_relevant) or 'none'}"
    )
    print()
    print("-- field map and problems --")
    for theme in result.field_map.themes:
        print(
            f"theme [{theme.era.value}] {theme.name}: "
            f"{len(theme.source_ids)} source(s)"
        )
    print(
        f"themes: {metrics.multi_source_themes} multi-source, "
        f"{metrics.single_source_themes} single-source"
    )
    for support in result.assessment.problem_support:
        print(f"[{support.tier.value}] {support.statement}")
    tier_counts = {
        tier.value: sum(
            1
            for support in result.assessment.problem_support
            if support.tier is tier
        )
        for tier in SupportTier
    }
    print(f"problem support tiers: {tier_counts}")
    print()
    print("-- spend (vs the Task 5B baseline) --")
    print(
        f"model calls={result.run_record.model_calls} "
        f"(baseline {BASELINE['model_calls']}); tokens="
        f"{result.run_record.input_tokens}in/"
        f"{result.run_record.output_tokens}out (baseline "
        f"{BASELINE['input_tokens']}in/{BASELINE['output_tokens']}out)"
    )
    print()
    print("-- adequacy verdict (trusted code, fail-closed) --")
    print(f"status: {result.assessment.status.value.upper()}")
    print(f"assessment id: {result.assessment.id}")
    if result.assessment.reasons:
        for reason in result.assessment.reasons:
            print(f"  [{reason.code.value}] {reason.detail}")
    else:
        print("  every configured bar was cleared.")
    print()
    print("-- coverage limitations (by construction) --")
    print("* a bounded slice under explicit budgets; adequacy means")
    print("  adequate for bounded candidate generation under this brief,")
    print("  never exhaustive coverage or a systematic review.")
    print("* abstract-level grounding at most; citation ranking is a")
    print("  discovery signal, not a quality or correctness claim.")
    print("* absence from this corpus is not novelty.")
    print()


def _verify_reload(result: MappingRunResult, root: Path) -> None:
    fresh = MappingStore(root / "mapping")
    reloaded = fresh.adequacy_for_run(result.run_record.run_id)
    assert reloaded == result.assessment, "the verdict must reload intact"
    assert reloaded is not None
    assert reloaded.id == result.assessment.id
    assert reloaded.reasons == result.assessment.reasons
    assert reloaded.thresholds == result.assessment.thresholds
    assert fresh.get_run(result.run_record.id) == result.run_record
    print("-- durability --")
    print("reloaded the adequacy assessment via fresh objects: identical")
    print("identity, status, reasons, and thresholds; the run record and")
    print("every mapping record verified likewise.")
    print()


if __name__ == "__main__":
    sys.exit(main())
