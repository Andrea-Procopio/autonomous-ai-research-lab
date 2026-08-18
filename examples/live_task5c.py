"""Task 5C live proof: gated candidate generation over the assessed
Task 5B.1 field map, directed by a real workshop call.

The mapping root is the preserved Task 5B.1 run — opened read-only; the
generator calls nothing but its accessors — and the ideation records go
to a fresh root. Zero literature retrieval happens: ideation reads the
durable records, so the only network traffic is Muse::

    the preserved ADEQUATE assessment + a real CFP snapshot
      -> require_adequate_for_idea_generation   (before any model call)
      -> one gated direction-extraction call
      -> one gated candidate-portfolio call
      -> tier/era/novelty stamping by trusted code
      -> honest portfolio accounting, durably recorded

Success is defined by correct records that survive reload — the task
passes on a thin portfolio, and on an honest grounded refusal, exactly
as it passes on a rich one. Requires MUSE_API_KEY (or MODEL_API_KEY)
and outbound HTTPS to Muse only. Run with::

    python -m examples.live_task5c \\
        --map-root live_runs/task5b1-2026-08-18 \\
        --run-root live_runs/task5c-<date>

The run root must sit under the gitignored ``live_runs/``.

The CFP snapshot below is the call of the NeurIPS 2026 workshop
"Foundations of LLM Post-Training in Changing Environments"
(fllmpt-work.shop), captured 2026-08-19 via a summarizing fetch of the
workshop's pages; the snapshot records that capture verbatim with its
URL and date, and its hash seals exactly what the model was shown.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from autonomous_research_lab.ideation.direction import CfpSnapshot
from autonomous_research_lab.ideation.directive import IdeationDirective
from autonomous_research_lab.ideation.generator import (
    IdeaGenerator,
    IdeationRunResult,
)
from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.mapping.adequacy import (
    require_adequate_for_idea_generation,
)
from autonomous_research_lab.mapping.store import MappingStore
from autonomous_research_lab.runtime.muse import (
    KEY_ENV_VARS,
    MUSE_SPARK_1_2,
    MuseSparkProvider,
)
from autonomous_research_lab.runtime.providers import UsageLedger

#: The preserved Task 5B.1 verdict this run enters through, and the
#: mapping run it assessed. Ids, never paths: the records prove their
#: own identity wherever the store lives.
ASSESSMENT_ID = "madq_1bbb287a57486d0f"
MAP_RUN_ID = "map_4414ba86ab4e468d"

CFP_TEXT = """\
Workshop on Foundations of LLM Post-Training in Changing Environments
A NeurIPS 2026 workshop. Paris, December 12-13, 2026.

Large language models are inherently multitask systems, adapted through
methods such as instruction tuning, preference-based updates, and domain
adaptation. Deployed systems face evolving tasks, data distribution
drift, and shifting feedback signals, yet current post-training practice
is largely heuristic, with incomplete understanding of statistical
identifiability, optimization dynamics, robustness to misspecification,
and trade-offs between adaptation and capability preservation.

Topics of interest include:
- preference feedback as data: statistical modeling of preference
  signals, heterogeneous annotators, and noise
- robustness and valid inference: conditions for reliable post-training
  updates and uncertainty quantification methods
- adaptive data collection and feedback loops: sequential data
  collection effects and selection bias in evolving contexts
- adaptation mechanisms and limits: understanding parameter-efficient
  methods and capability preservation trade-offs

Submissions: long papers up to 8 pages excluding references and
appendices; short papers up to 4 pages. PDF using the NeurIPS 2026
style file. Double-blind review; the workshop is non-archival and
welcomes dual submissions with disclosure. Papers are evaluated on
novelty, significance, technical quality, and clarity.

Important dates:
- abstract registration: August 23, 2026
- paper submission: August 30, 2026
- author notification: October 2026
- workshop: December 12-13, 2026
"""

SNAPSHOT = CfpSnapshot(
    source_url="https://www.fllmpt-work.shop/call/",
    supplied_at="2026-08-19",
    text=CFP_TEXT,
)

DIRECTIVE = IdeationDirective(
    assessment_id=ASSESSMENT_ID, snapshot_id=SNAPSHOT.id
)

#: The Task 5B.1 mapping run's spend — what building this candidate
#: generation's entire input cost. The ideation spend below is on top.
MAPPING_BASELINE = {
    "model_calls": 24,
    "input_tokens": 47_289,
    "output_tokens": 90_972,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-root",
        type=Path,
        required=True,
        help="the preserved Task 5B.1 run root (read-only input)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="run directory (put it under the gitignored live_runs/)",
    )
    arguments = parser.parse_args()
    map_root = arguments.map_root.resolve()
    run_root = arguments.run_root.resolve()

    if not any(os.environ.get(name, "").strip() for name in KEY_ENV_VARS):
        print(
            f"FATAL: no Muse API key in the environment "
            f"({' or '.join(KEY_ENV_VARS)})"
        )
        return 1
    map_store = MappingStore(map_root / "mapping")
    if map_store.get_adequacy(ASSESSMENT_ID) is None:
        print(
            f"FATAL: {map_root} does not hold assessment {ASSESSMENT_ID}; "
            f"point --map-root at the preserved task5b1 run root"
        )
        return 1

    ledger = UsageLedger()
    generator = IdeaGenerator(
        provider=MuseSparkProvider(),
        model=MUSE_SPARK_1_2,
        ledger=ledger,
        map_store=map_store,
        store=IdeationStore(run_root / "ideation"),
        max_output_tokens=8192,
        temperature=0.0,
        request_timeout_seconds=240.0,
    )

    print("== Task 5C live proof: grounded candidate generation ==")
    print(f"directive  : {DIRECTIVE.id}")
    print(f"assessment : {ASSESSMENT_ID} (map run {MAP_RUN_ID})")
    print(f"cfp        : {SNAPSHOT.id} from {SNAPSHOT.source_url}")
    print(f"budgets    : {DIRECTIVE.max_candidates} candidates, "
          f"{DIRECTIVE.max_model_calls} model calls")
    print(f"map root   : {map_root} (read-only)")
    print(f"run root   : {run_root}")
    print()

    result = generator.run(DIRECTIVE, SNAPSHOT)
    _report(result)
    _verify_reload(result, run_root, map_root)
    print("PASSED: the guard walked, the call read into a grounded")
    print("direction, and a gated, tier-stamped candidate portfolio (or")
    print("an honest refusal) was durably recorded. A thin portfolio is")
    print("a pass; only infrastructure, gate exhaustion, or integrity")
    print("failures are not.")
    return 0


def _report(result: IdeationRunResult) -> None:
    print("-- the guard (trusted code, fail-closed) --")
    print(f"status: {result.assessment.status.value.upper()}")
    print(
        f"grounded sources: {result.assessment.metrics.grounded_sources} "
        f"({result.assessment.metrics.recent_grounded} recent / "
        f"{result.assessment.metrics.foundational_grounded} foundational); "
        f"problems: {len(result.assessment.problem_support)}"
    )
    print()
    print("-- the extracted direction (gated against the snapshot) --")
    direction = result.direction
    print(f"sha256(call text): {result.snapshot.text_sha256}")
    print(f"scope: {direction.scope}")
    for topic in direction.topics:
        print(f"topic: {topic}")
    for constraint in direction.constraints:
        print(f"constraint: {constraint}")
    for date in direction.relevant_dates:
        print(f"date: {date}")
    print(
        f"direction call: repair_count="
        f"{direction.provenance.repair_count}, latency="
        f"{direction.provenance.latency_seconds:.1f}s"
    )
    print()
    record = result.run_record
    if not result.ideas:
        print("-- honest refusal --")
        print(record.refusal_justification)
        print()
    for index, idea in enumerate(result.ideas, start=1):
        print(f"-- candidate {index}: {idea.title} --")
        print(f"question   : {idea.research_question}")
        for problem in idea.addressed_problems:
            print(f"problem    : [{problem.tier.value}] {problem.statement}")
        for theme in idea.targeted_themes:
            print(f"theme      : [{theme.era.value}] {theme.name}")
        print(f"mechanism  : {idea.mechanism}")
        print(f"hypothesis : {idea.hypothesis}")
        for prediction in idea.predictions:
            print(f"prediction : {prediction.text}")
            print(f"falsifier  : {prediction.falsifier}")
        for dataset in idea.datasets:
            print(
                f"data       : {dataset.name} [{dataset.status.value}] - "
                f"{dataset.role}"
            )
        print(f"metrics    : {', '.join(idea.metrics)}")
        print(f"evaluation : {idea.evaluation_protocol}")
        print(f"baselines  : {', '.join(idea.baselines)}")
        print(f"ablations  : {', '.join(idea.ablations)}")
        print(
            f"resources  : compute {idea.resources.compute}; data "
            f"{idea.resources.data}; implementation "
            f"{idea.resources.implementation}"
        )
        for risk in idea.risks:
            print(f"risk       : {risk}")
        print(f"cfp fit    : {idea.cfp_alignment}")
        print(f"  aligned  : {', '.join(idea.aligned_topics)}")
        print(f"uncertainty: {idea.uncertainty}")
        print(f"5d terms   : {'; '.join(idea.search_terms)}")
        print(
            f"sources    : {', '.join(idea.cited_source_ids)} "
            f"({idea.cited_recent} recent / {idea.cited_foundational} "
            f"foundational / {idea.cited_undated} undated)"
        )
        print(f"novelty    : {idea.novelty_status.value} (structural)")
        print()
    portfolio = record.portfolio
    print("-- portfolio accounting (trusted code) --")
    print(
        f"candidates={portfolio.candidates}; problems addressed="
        f"{portfolio.problems_addressed}/{portfolio.problems_total} "
        f"(multi_source={portfolio.addressed_multi_source}, "
        f"contradicted={portfolio.addressed_contradicted}, "
        f"tentative={portfolio.addressed_tentative}, "
        f"single_source_limitation="
        f"{portfolio.addressed_single_source_limitation})"
    )
    for statement in portfolio.unaddressed_statements:
        print(f"unaddressed: {statement}")
    print(
        f"diversity  : {portfolio.distinct_problem_sets} problem sets, "
        f"{portfolio.distinct_theme_sets} theme sets, "
        f"{portfolio.distinct_dataset_sets} dataset sets, "
        f"{portfolio.distinct_metric_sets} metric sets across "
        f"{portfolio.candidates} candidate(s); "
        f"{portfolio.distinct_sources_cited} distinct source(s) cited"
    )
    if record.diversity_rationale:
        print(f"rationale  : {record.diversity_rationale}")
    print()
    print("-- spend (on top of the Task 5B.1 mapping baseline) --")
    print(
        f"model calls={record.model_calls}; tokens="
        f"{record.input_tokens}in/{record.output_tokens}out "
        f"(the mapped input cost {MAPPING_BASELINE['model_calls']} calls, "
        f"{MAPPING_BASELINE['input_tokens']}in/"
        f"{MAPPING_BASELINE['output_tokens']}out)"
    )
    print()
    print("-- limitations (by construction) --")
    print("* candidates are conjectures read off a bounded map: not")
    print("  proposals, not scientific state, and carrying no novelty")
    print("  claim -- novelty stays UNASSESSED until the Task 5D")
    print("  prior-art challenge; absence from this corpus is not")
    print("  novelty.")
    print("* support tiers travel with every idea, so a single-paper")
    print("  limitation can never be presented as field consensus.")
    print("* the grounding surface is the mapping run's gated claim")
    print("  texts, at most abstract-deep -- never full papers.")
    print("* the CFP constrains relevance only; it is not evidence.")
    print()


def _verify_reload(
    result: IdeationRunResult, run_root: Path, map_root: Path
) -> None:
    fresh = IdeationStore(run_root / "ideation")
    record = result.run_record
    assert fresh.get_run(record.id) == record, "the run must reload intact"
    assert fresh.get_snapshot(result.snapshot.id) == result.snapshot
    assert fresh.get_directive(result.directive.id) == result.directive
    assert fresh.get_direction(result.direction.id) == result.direction
    for idea in result.ideas:
        assert fresh.get_idea(idea.id) == idea, "ideas must reload intact"
    assert fresh.runs_for_assessment(ASSESSMENT_ID) == (record,)
    # The read-only input still walks the guard, byte-identical.
    reloaded = require_adequate_for_idea_generation(
        MappingStore(map_root / "mapping"), ASSESSMENT_ID
    )
    assert reloaded == result.assessment
    print("-- durability --")
    print("reloaded the run, snapshot, directive, direction, and every")
    print("candidate via fresh objects: identical identity throughout;")
    print("the preserved mapping input still walks the adequacy guard")
    print("unchanged.")
    print()


if __name__ == "__main__":
    sys.exit(main())
