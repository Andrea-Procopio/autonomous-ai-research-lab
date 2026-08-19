"""Counterfactual replay: the preserved Task 5D.1 evidence re-judged
under the Task 5D.2 calibrated verdict rules, before any new live run.

Strictly read-only and print-only: the preserved records are loaded,
re-assessed in memory through the current :func:`assess_prior_art`, and
compared against the stored verdicts. Nothing is written anywhere — a
counterfactual is analysis, not a record.

What this can conclude: whether the evidence the 5D.1 run actually
recorded still blocks under the calibrated semantics — the narrowed
material metadata-ambiguity rule, the abstract-level uncertainty basis,
and the screenable-pool source threshold. What it cannot conclude: what
the new metadata screening gate would have found. The old screening
records carry no overlap hypotheses and cannot be re-gated, so a
counterfactual DISTINGUISHED means "the preserved evidence no longer
blocks" — never "the new gate found nothing". That is exactly why the
Task 5D.2 live rerun follows this replay instead of replacing it.

Run with::

    python -m examples.replay_task5d1_counterfactual \\
        --run-root live_runs/task5d1-2026-08-19 \\
        --map-root live_runs/task5b1-2026-08-18
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from autonomous_research_lab.literature.store import LiteratureStore
from autonomous_research_lab.priorart.assessment import (
    PriorArtAssessment,
    assess_prior_art,
)
from autonomous_research_lab.priorart.records import (
    PriorArtScreeningRecord,
    SimilarityDecision,
    WorkComparison,
)
from autonomous_research_lab.priorart.store import PriorArtStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="the preserved Task 5D.1 run root (read-only input)",
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        required=True,
        help="the preserved Task 5B.1 run root (read-only input; "
        "resolves cited-only sources)",
    )
    arguments = parser.parse_args()
    run_root = arguments.run_root.resolve()
    map_root = arguments.map_root.resolve()

    store = PriorArtStore(run_root / "priorart")
    runs = store.runs()
    if len(runs) != 1:
        print(f"FATAL: {run_root} holds {len(runs)} run records, not 1")
        return 1
    (record,) = runs
    fresh_literature = LiteratureStore(run_root / "literature")
    known_literature = LiteratureStore(map_root / "literature")

    screenings: list[PriorArtScreeningRecord] = []
    for screening_id in record.screening_ids:
        screening = store.get_screening(screening_id)
        assert screening is not None, screening_id
        screenings.append(screening)
    comparisons: list[WorkComparison] = []
    for comparison_id in record.comparison_ids:
        comparison = store.get_comparison(comparison_id)
        assert comparison is not None, comparison_id
        comparisons.append(comparison)

    print("== Task 5D.1 counterfactual under the 5D.2 calibrated rules ==")
    print(f"run        : {record.run_id} ({record.id})")
    print(f"directive  : {record.directive_id}")
    print("mode       : read-only, print-only; nothing is written")
    print()

    changed = 0
    for candidate_id in record.candidate_ids:
        stored = store.assessment_for_candidate(
            record.run_id, candidate_id
        )
        assert stored is not None, candidate_id
        candidate_screenings = tuple(
            entry
            for entry in screenings
            if entry.candidate_id == candidate_id
        )
        candidate_comparisons = tuple(
            entry
            for entry in comparisons
            if entry.candidate_id == candidate_id
        )
        metadata_ids = _metadata_ids(
            candidate_screenings, fresh_literature, known_literature
        )
        material = sum(
            1
            for entry in candidate_screenings
            if entry.source_id in metadata_ids
            and entry.decision is SimilarityDecision.POTENTIAL_OVERLAP
        )
        counterfactual = assess_prior_art(
            run_id=stored.run_id,
            candidate_id=stored.candidate_id,
            directive_id=stored.directive_id,
            screenings=candidate_screenings,
            comparisons=candidate_comparisons,
            coverage=replace(
                stored.coverage, metadata_ambiguous=material
            ),
            metadata_source_ids=metadata_ids,
            thresholds=stored.thresholds,
        )
        changed += _report(stored, counterfactual)
        print()

    print(f"-- {changed} of {len(record.candidate_ids)} verdicts change --")
    print("Caveat: the preserved screening records carry no overlap")
    print("hypotheses and cannot be re-gated. A counterfactual")
    print("DISTINGUISHED means the preserved evidence no longer blocks,")
    print("never that the new metadata gate would have found nothing;")
    print("the Task 5D.2 live rerun answers that. No record was written")
    print("or modified by this replay.")
    return 0


def _metadata_ids(
    screenings: tuple[PriorArtScreeningRecord, ...],
    fresh: LiteratureStore,
    known: LiteratureStore,
) -> frozenset[str]:
    """Trusted code's account of which screened sources carried no
    abstract, rebuilt from the preserved literature stores."""
    metadata: set[str] = set()
    for entry in screenings:
        source = fresh.get_source(entry.source_id) or known.get_source(
            entry.source_id
        )
        assert source is not None, (
            f"{entry.source_id} is in neither preserved literature store"
        )
        if source.abstract is None:
            metadata.add(entry.source_id)
    return frozenset(metadata)


def _report(
    stored: PriorArtAssessment, counterfactual: PriorArtAssessment
) -> int:
    print(f"-- candidate {stored.candidate_id} --")
    print(f"stored     : {stored.verdict.value.upper()}")
    for reason in stored.reasons:
        print(f"  reason   : [{reason.code.value}] {reason.detail}")
    print(f"calibrated : {counterfactual.verdict.value.upper()}")
    for reason in counterfactual.reasons:
        print(f"  reason   : [{reason.code.value}] {reason.detail}")
    if counterfactual.verdict is stored.verdict and (
        tuple(r.code for r in counterfactual.reasons)
        == tuple(r.code for r in stored.reasons)
    ):
        print("  change   : none")
        return 0
    dropped = {r.code for r in stored.reasons} - {
        r.code for r in counterfactual.reasons
    }
    for code in sorted(code.value for code in dropped):
        print(f"  released : {code} no longer fires under calibration")
    return 1 if counterfactual.verdict is not stored.verdict else 0


if __name__ == "__main__":
    sys.exit(main())
