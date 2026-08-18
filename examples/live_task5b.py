"""Task 5B live proof: one real, bounded, Muse-backed field-mapping run.

One trajectory, live, with nothing mocked and nothing hand-edited::

    a workshop-style research brief (broad ML topic + CFP-like hints)
      -> Muse proposes focused queries    (validated, dated by trusted code)
      -> OpenAlex retrieval through the Task 5A corpus (cache-or-live)
      -> Muse screens every retrieved source   (all verdicts preserved)
      -> Muse extracts per abstract-level relevant source, under the
         verbatim-grounding gate; metadata-only sources become
         deterministic insufficient-support records with no model call
      -> one FieldMap, one ProblemInventory    (gated, source-grounded)
      -> deterministic coverage accounting and one durable run record

Success is defined by correct state transitions and provenance — every
record durable, write-once, and reloadable to the same identities — never
by whether the map is flattering. Requires MUSE_API_KEY (or
MODEL_API_KEY) in the environment and outbound HTTPS; no Docker. Run
with::

    python -m examples.live_task5b --run-root live_runs/task5b-<date>

The run root must sit under the gitignored ``live_runs/``; everything it
accumulates is a live payload and must not be committed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from autonomous_research_lab.literature.corpus import LiteratureCorpus
from autonomous_research_lab.literature.openalex import OpenAlexProvider
from autonomous_research_lab.literature.store import LiteratureStore
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
    max_screened_sources=60,
    max_extracted_sources=10,
    max_model_calls=30,
)


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
    corpus = LiteratureCorpus(
        LiteratureStore(root / "literature"), OpenAlexProvider()
    )
    store = MappingStore(root / "mapping")
    mapper = FieldMapper(
        provider=MuseSparkProvider(),
        model=MUSE_SPARK_1_2,
        ledger=ledger,
        corpus=corpus,
        store=store,
        max_output_tokens=8192,
        temperature=0.0,
        request_timeout_seconds=240.0,
    )

    print("== Task 5B live proof: evidence-grounded field mapping ==")
    print(f"brief      : {BRIEF.id}")
    print(f"topic      : {BRIEF.topic}")
    print(f"cutoff     : {BRIEF.cutoff_date}; recent window from "
          f"{BRIEF.recent_window_start}")
    print(f"budgets    : {BRIEF.max_queries_per_family}/family, "
          f"{BRIEF.results_per_query}/query, screen "
          f"{BRIEF.max_screened_sources}, extract "
          f"{BRIEF.max_extracted_sources}, {BRIEF.max_model_calls} calls")
    print(f"run root   : {root}")
    print()

    result = mapper.run(BRIEF)
    _report(result, store)
    _verify_reload(result, root)
    print("PASSED: field map and problem inventory produced, gated, and")
    print("durably recorded with intact provenance.")
    return 0


def _report(result: MappingRunResult, store: MappingStore) -> None:
    coverage = result.run_record.coverage
    print("-- retrieval --")
    for execution in result.query_executions:
        print(
            f"[{execution.family.value}] {execution.text!r} "
            f"({execution.from_date or 'open'}..{execution.to_date}): "
            f"retrieved {execution.retrieved}, new {execution.new_unique}, "
            f"cached={execution.from_cache}"
        )
    overlap = coverage.total_retrieved - coverage.unique_sources
    print(
        f"queries={coverage.queries_executed} retrieved="
        f"{coverage.total_retrieved} unique={coverage.unique_sources} "
        f"overlap={overlap} saturation={coverage.saturation}"
    )
    print()
    print("-- screening --")
    print(
        f"screened={coverage.screened} (truncated "
        f"{coverage.screening_truncated}): relevant={coverage.relevant} "
        f"excluded={coverage.excluded} uncertain={coverage.uncertain}"
    )
    print(
        f"access levels among screened: abstract={coverage.abstract_level} "
        f"metadata-only={coverage.metadata_level}"
    )
    print()
    print("-- extraction --")
    print(
        f"eligible={coverage.extraction_eligible} "
        f"extracted={coverage.extracted} (truncated "
        f"{coverage.extraction_truncated}); insufficient accessible "
        f"support={coverage.insufficient_support}"
    )
    print()
    print("-- field map --")
    for theme in result.field_map.themes:
        print(
            f"theme [{theme.era.value}] {theme.name}: "
            f"{len(theme.source_ids)} source(s)"
        )
    print(
        f"approaches={len(result.field_map.approaches)} "
        f"evaluation_practices="
        f"{len(result.field_map.evaluation_practices)} "
        f"relationships={len(result.field_map.relationships)}"
    )
    print(
        f"era split: recent={len(result.field_map.recent_source_ids)} "
        f"foundational={len(result.field_map.foundational_source_ids)} "
        f"undated={len(result.field_map.undated_source_ids)}"
    )
    print()
    print("-- problem inventory --")
    for problem in result.inventory.problems:
        print(f"[{problem.kind.value}] {problem.statement}")
        print(f"  supported by: {', '.join(problem.supporting_source_ids)}")
        if problem.conflicting_source_ids:
            print(
                f"  conflicting: "
                f"{', '.join(problem.conflicting_source_ids)}"
            )
    print()
    print("-- spend and provenance --")
    rejected = store.rejected()
    # Screening records share their batch call's provenance, so per-call
    # statistics dedup by response occurrence id.
    calls = {
        provenance.response_id: provenance
        for provenance in (
            result.field_map.provenance,
            result.inventory.provenance,
            *(s.provenance for s in result.screenings),
            *(
                e.provenance
                for e in result.extractions
                if e.provenance is not None
            ),
        )
    }
    repaired = sum(1 for p in calls.values() if p.repair_count > 0)
    latency = sum(p.latency_seconds for p in calls.values())
    print(
        f"model calls={result.run_record.model_calls} tokens="
        f"{result.run_record.input_tokens}in/"
        f"{result.run_record.output_tokens}out"
    )
    print(
        f"gate rejections preserved={len(rejected)}; repaired accepted "
        f"calls={repaired}; accepted-call latency ~{latency:.0f}s"
    )
    print(
        f"served model: {result.field_map.provenance.served_model}; "
        f"provider: {result.field_map.provenance.provider}"
    )
    print()
    print("-- coverage limitations (by construction) --")
    print("* a bounded slice: a handful of keyword queries, capped results,")
    print("  capped screening and extraction; not a systematic search.")
    print("* abstract-level grounding at most: nothing here rests on")
    print("  methods sections, tables, or appendices.")
    print("* foundational retrieval is date-scoped and recency-sorted, not")
    print("  citation-ranked; era labels follow the brief's window.")
    print("* the provider-reported match totals vastly exceed what was")
    print("  screened; no claim of exhaustiveness or novelty is made.")
    print()


def _verify_reload(result: MappingRunResult, root: Path) -> None:
    fresh = MappingStore(root / "mapping")
    assert fresh.get_run(result.run_record.id) == result.run_record
    assert fresh.get_field_map(result.field_map.id) == result.field_map
    assert fresh.get_inventory(result.inventory.id) == result.inventory
    assert len(fresh.screenings()) == len(result.screenings)
    assert len(fresh.extractions()) == len(result.extractions)
    for execution in result.query_executions:
        assert fresh.get_query_execution(execution.id) == execution
    print("-- durability --")
    print("reloaded the run, field map, inventory, and every screening,")
    print("extraction, and query record via fresh objects; every")
    print("recomputed identity matched.")
    print()


if __name__ == "__main__":
    sys.exit(main())
