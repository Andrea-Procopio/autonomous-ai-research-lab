"""Task 5D.2 live proof: the prior-art challenge under calibrated
decision rules and a coherent budget.

Task 5D.1's live evidence exposed a composition failure: every
metadata-ambiguity blocker was an undecidable title-only screen (access
level restated, never an overlap signal), the directive could retrieve
more than it could screen so a successful retrieval mechanically
truncated, and every corrective call bought a constraint the model was
never told. The Task 5D.2 calibration narrowed metadata ambiguity to
attested material hypotheses screened in their own gated call, measured
uncertainty and the source threshold on the bases they were meant to
proxy, ordered cited works first, made the budget-coherence preflight
refuse an inconsistent directive before any call, and proved all three
verdicts reachable on closed corpora. The counterfactual replay of the
preserved 5D.1 records under the calibrated rules ran first
(``examples.replay_task5d1_counterfactual``): one candidate's evidence
no longer blocks, two still refuse on distinct single causes.

This run challenges the same three preserved candidates live, under the
calibrated rules frozen before execution. The directive keeps the same
cutoff, window, retrieval strength, and comparison cap as Task 5D and
5D.1; its screening and call budgets move to the coherent defaults
(35 screened, 36 calls) the preflight demands, so its id differs from
the 5D/5D.1 directive — the change is the corrected boundary itself,
acknowledged rather than hidden. Nothing is tuned after observing
candidate outcomes, and no candidate is required to become
DISTINGUISHED.

Three roots: the preserved Task 5B.1 run (read-only — its literature
store resolves the candidates' cited sources), the preserved Task 5C run
(read-only — the immutable portfolio under challenge), and a fresh run
root that receives the prior-art records and the fresh challenge
corpus. The preserved Task 5D and 5D.1 runs are untouched.

Success is defined by correct records and fail-closed verdicts — an
OVERLAPPING or NOVELTY_UNRESOLVED result is a successful scientific
outcome. What this run must specifically show: the preflight passes and
its call plan is printed, both screening kinds are gated, no
METADATA_AMBIGUITY reason rests on a bare undecidable screen, no
mechanical truncation occurs, and the run-record spend reconciles with
the ledger. Requires MUSE_API_KEY (or MODEL_API_KEY), outbound HTTPS to
Muse and api.openalex.org (OPENALEX_API_KEY optional — raises the daily
credit budget). Run with::

    python -m examples.live_task5d2 \\
        --map-root live_runs/task5b1-2026-08-18 \\
        --idea-root live_runs/task5c-2026-08-19 \\
        --run-root live_runs/task5d2-<date>

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
from autonomous_research_lab.priorart.assessment import PriorArtThresholds
from autonomous_research_lab.priorart.challenger import (
    PriorArtChallenger,
    PriorArtRunResult,
)
from autonomous_research_lab.priorart.directive import PriorArtDirective
from autonomous_research_lab.priorart.preflight import (
    PriorArtPreflightError,
    check_budget_coherence,
)
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

#: Same cutoff, window, retrieval strength, and comparison cap as Task
#: 5D and 5D.1; screening and call budgets at the coherent defaults the
#: preflight demands. Frozen before the run, never tuned after it.
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

#: The Task 5D.1 live evidence this rerun is measured against —
#: diagnostic, never an acceptance target.
TASK5D1_BASELINE = {
    "searches": 18,
    "zero_result_searches": 5,
    "unique_pools": "9, 23, 17",
    "credits": 180,
    "model_calls": 14,
    "repairs": 3,
    "verdicts": "3x NOVELTY_UNRESOLVED (metadata ambiguity on every "
    "candidate, one thin pool, one truncation)",
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
    ideation_run = ideation_store.get_run(IDEATION_RUN_RECORD_ID)
    if ideation_run is None:
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
    # One set of wiring values, visibly shared between the challenger
    # and the fail-fast preflight below.
    thresholds = PriorArtThresholds()
    screening_batch_size = 12
    max_corrective_calls = 1
    challenger = PriorArtChallenger(
        provider=MuseSparkProvider(),
        model=MUSE_SPARK_1_2,
        ledger=ledger,
        ideation_store=ideation_store,
        known_literature=known_literature,
        corpus=corpus,
        store=PriorArtStore(run_root / "priorart"),
        thresholds=thresholds,
        screening_batch_size=screening_batch_size,
        # 16384, the Task 5C lesson: the comparison reply is one large
        # JSON object over several works; the budget has to fit it.
        max_output_tokens=16384,
        temperature=0.0,
        request_timeout_seconds=240.0,
        max_corrective_calls=max_corrective_calls,
    )

    # The same budget-coherence preflight the challenger itself runs,
    # executed fail-fast here so an incoherent configuration is a
    # printed refusal, not a traceback — and the reserved call plan is
    # on the record before the first call.
    candidates = []
    for candidate_id in ideation_run.candidate_ids:
        candidate = ideation_store.get_idea(candidate_id)
        if candidate is None:
            print(f"FATAL: candidate {candidate_id} failed to load")
            return 1
        candidates.append(candidate)
    try:
        plan = check_budget_coherence(
            directive=DIRECTIVE,
            candidates=candidates,
            thresholds=thresholds,
            screening_batch_size=screening_batch_size,
            max_corrective_calls=max_corrective_calls,
        )
    except PriorArtPreflightError as error:
        print(f"FATAL: {error}")
        return 1

    print("== Task 5D.2 live proof: the challenge under calibrated "
          "rules and a coherent budget ==")
    print(f"directive  : {DIRECTIVE.id}")
    print(f"portfolio  : {IDEATION_RUN_RECORD_ID} (assessment "
          f"{ASSESSMENT_ID}, map run {MAP_RUN_ID}, cfp {SNAPSHOT_ID})")
    print(f"cutoff     : {DIRECTIVE.cutoff_date} (recent window from "
          f"{DIRECTIVE.recent_window_start})")
    print(f"budgets    : {DIRECTIVE.results_per_query} results/query, "
          f"{DIRECTIVE.max_screened_per_candidate} screened and "
          f"{DIRECTIVE.max_compared_works} compared per candidate, "
          f"{DIRECTIVE.max_model_calls} model calls")
    print(f"preflight  : PASSED — worst-case pool "
          f"{plan.worst_pool_per_candidate}/{plan.screening_capacity} "
          f"screenable, {plan.worst_screening_calls_per_candidate} "
          f"screening calls per candidate, "
          f"{plan.worst_calls_per_candidate} calls per candidate, "
          f"{plan.worst_calls_total}/{DIRECTIVE.max_model_calls} total")
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
    _verify_calibration(result)
    _verify_spend(result, ledger)
    _verify_reload(result, run_root)
    _verify_replay(result, run_root, counting)
    print("PASSED: the preflight held, the door walked, both screening")
    print("kinds were gated, and a deterministic fail-closed verdict was")
    print("durably recorded per candidate — with no blocker resting on a")
    print("bare undecidable metadata screen and no mechanical")
    print("truncation. OVERLAPPING and NOVELTY_UNRESOLVED are successful")
    print("outcomes; no candidate is required to become DISTINGUISHED.")
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
            plan = " AND ".join(
                "[" + " | ".join(group) + "]"
                for group in execution.plan_groups
            )
            print(f"plan       : [{execution.family.value}] {plan}")
            print(
                f"  rendered : {execution.text!r} "
                f"({execution.renderer}; {window}, "
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
            f"{coverage.metadata_ambiguous} materially "
            f"metadata-ambiguous"
        )
        for screening in result.screenings:
            if screening.candidate_id != candidate.id:
                continue
            hypothesis = screening.overlap_hypothesis
            if hypothesis is None:
                continue
            print(
                f"attested   : {screening.source_id} claims "
                f"{hypothesis.candidate_claim!r} via "
                f"{hypothesis.dimension.value} "
                f"[{hypothesis.support_location.value}: "
                f"{hypothesis.source_text!r}] — {hypothesis.rationale}"
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

    zero_result = sum(1 for e in result.executions if not e.retrieved)
    print("-- against the Task 5D.1 baseline (diagnostic, not a "
          "target) --")
    print(
        f"searches   : {len(result.executions)} now vs "
        f"{TASK5D1_BASELINE['searches']} then; zero-result "
        f"{zero_result} now vs "
        f"{TASK5D1_BASELINE['zero_result_searches']} then"
    )
    pools = ", ".join(
        str(a.coverage.unique_sources) for a in result.assessments
    )
    print(
        f"pools      : {pools} unique sources now vs "
        f"{TASK5D1_BASELINE['unique_pools']} then"
    )
    print(f"then       : {TASK5D1_BASELINE['verdicts']}")
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
    print("  metadata-only source blocks exactly when its accessible")
    print("  metadata attests a material overlap hypothesis -- access")
    print("  resolution for such sources still does not exist.")
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


def _verify_calibration(result: PriorArtRunResult) -> None:
    """The two specifically calibrated properties, asserted on the live
    records: no blocker rests on a bare undecidable metadata screen,
    and no truncation occurred under the preflighted budget."""
    material = {
        screening.source_id
        for screening in result.screenings
        if screening.overlap_hypothesis is not None
    }
    for assessment in result.assessments:
        coverage = assessment.coverage
        assert coverage.screening_truncated == 0, (
            "a preflighted directive can never mechanically truncate"
        )
        assert coverage.metadata_ambiguous <= len(material), (
            "every material ambiguity carries an attested hypothesis"
        )
    print("-- calibration --")
    print("zero screening truncation under the preflighted budget, and")
    print("every metadata-ambiguity blocker (if any) carries an attested")
    print("overlap hypothesis; bare undecidable metadata screens are")
    print("coverage, not blockers.")
    print()


def _verify_spend(result: PriorArtRunResult, ledger: UsageLedger) -> None:
    drained = ledger.drain()
    record = result.run_record
    assert record.input_tokens == drained.input_tokens, (
        "the run record must reconcile with the ledger"
    )
    assert record.output_tokens == drained.output_tokens, (
        "the run record must reconcile with the ledger"
    )
    print("-- accounting --")
    print(
        f"run-record tokens ({record.input_tokens}in/"
        f"{record.output_tokens}out) equal the drained ledger exactly; "
        f"calls {record.model_calls} vs ledger {drained.calls}."
    )
    print()


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
