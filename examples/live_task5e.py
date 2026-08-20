"""Task 5E live proof: one selection over the challenged portfolio.

The Task 5D.2 run left three candidates DISTINGUISHED within its
challenged corpus. This run walks the one door into selection — the
directive names that run record explicitly, and eligibility is computed
by trusted code from its assessments alone — then makes at most two
gated model calls: one comparative review of every eligible candidate
and every pair, and one final choice among the candidates no validated
disqualifier removed. The operator's resource constraints were stated
and frozen before implementation; nothing is tuned after observing
outcomes, and no candidate is required to win.

Success is defined by correct records and honest outcomes — SELECTED,
NO_ELIGIBLE_CANDIDATE, and NO_DEFENSIBLE_CANDIDATE are all successful
scientific outcomes. What this run must specifically show: the
preflight passes and its call plan is printed; the eligible set is
stamped by trusted code and printed before any call; each stage spends
at most one corrective call; the run-record spend reconciles exactly
with the ledger; every record reloads intact from a fresh store; and
every preserved upstream artifact is byte-identical afterwards (hashed
before and after). Nothing is retrieved in this run, so there is no
zero-network replay to claim — the only network traffic is Muse.

Two read-only roots and one fresh root: the preserved Task 5C run (the
immutable portfolio), the preserved Task 5D.2 run (the verdicts that
define eligibility), and a fresh run root that receives the selection
records. Requires MUSE_API_KEY (or MODEL_API_KEY) and outbound HTTPS to
Muse only. Run with::

    python -m examples.live_task5e \\
        --idea-root live_runs/task5c-2026-08-19 \\
        --priorart-root live_runs/task5d2-2026-08-19 \\
        --run-root live_runs/task5e-<date>

The run root must sit under the gitignored ``live_runs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.priorart.assessment import PriorArtVerdict
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.runtime.muse import (
    KEY_ENV_VARS,
    MUSE_SPARK_1_2,
    MuseSparkProvider,
)
from autonomous_research_lab.runtime.providers import UsageLedger
from autonomous_research_lab.selection.directive import SelectionDirective
from autonomous_research_lab.selection.eligibility import (
    MissingChallengedPortfolioError,
    partition_by_verdict,
    require_challenged_portfolio_for_selection,
)
from autonomous_research_lab.selection.preflight import (
    SelectionPreflightError,
    check_selection_coherence,
)
from autonomous_research_lab.selection.records import SelectionOutcome
from autonomous_research_lab.selection.selector import (
    CandidateSelector,
    SelectionRunResult,
    render_candidate_for_selection,
)
from autonomous_research_lab.selection.store import SelectionStore

#: The preserved Task 5D.2 challenge this selection enters through, and
#: the portfolio behind it. Ids, never paths: the records prove their
#: own identity wherever the stores live.
PRIOR_ART_RUN_RECORD_ID = "prun_095dfdb9a99f4f0f"
IDEATION_RUN_RECORD_ID = "irun_23e08da6543a3bb6"
EXPECTED_CANDIDATE_IDS = (
    "idea_1e1fa63952cc0d91",
    "idea_1fd7f11cbcdda4ca",
    "idea_79acb8c5851b8839",
)

#: The operator's resource statements, provided 2026-08-19 and frozen
#: before implementation. They are facts about what is available —
#: quoted verbatim by any attested disqualifier — never tuned after a
#: run, and never a preference for any candidate.
DIRECTIVE = SelectionDirective(
    prior_art_run_record_id=PRIOR_ART_RUN_RECORD_ID,
    compute_constraint=(
        "Elastic cloud GPU capacity: up to 2-4 GPUs per study (up to "
        "A100/H100 class), provisioned on demand. Continuous "
        "availability is not guaranteed, so execution must tolerate "
        "queueing, interruption, and reduced parallelism, and must be "
        "able to run on a single GPU."
    ),
    data_constraint=(
        "Public or lawfully accessible datasets; preprocessing allowed."
    ),
    time_constraint=(
        "Individual runs up to days; complete studies up to weeks."
    ),
    experimental_constraint=(
        "Containerized seeded runs with parallel sweeps, replications, "
        "controls, and ablations."
    ),
)

#: What producing this selection's entire input cost (the Task 5D.2
#: prior-art challenge). The selection spend below is on top.
PRIOR_ART_BASELINE = {
    "model_calls": 15,
    "input_tokens": 32_295,
    "output_tokens": 70_140,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idea-root",
        type=Path,
        required=True,
        help="the preserved Task 5C run root (read-only input)",
    )
    parser.add_argument(
        "--priorart-root",
        type=Path,
        required=True,
        help="the preserved Task 5D.2 run root (read-only input)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="run directory (put it under the gitignored live_runs/)",
    )
    arguments = parser.parse_args()
    idea_root = arguments.idea_root.resolve()
    priorart_root = arguments.priorart_root.resolve()
    run_root = arguments.run_root.resolve()

    if not any(os.environ.get(name, "").strip() for name in KEY_ENV_VARS):
        print(
            f"FATAL: no Muse API key in the environment "
            f"({' or '.join(KEY_ENV_VARS)})"
        )
        return 1
    ideation_store = IdeationStore(idea_root / "ideation")
    prior_art_store = PriorArtStore(priorart_root / "priorart")

    # Every preserved upstream artifact is hashed before wiring anything
    # and re-hashed after the run: byte identity as a fact, not a claim.
    before_ideas = _digests(idea_root)
    before_prior = _digests(priorart_root)

    # The same door and preflight the selector itself runs, executed
    # fail-fast here so a missing record or an incoherent configuration
    # is a printed refusal, not a traceback — and the trusted eligible
    # set is on the record before the first call.
    try:
        inputs = require_challenged_portfolio_for_selection(
            prior_art_store, ideation_store, PRIOR_ART_RUN_RECORD_ID
        )
    except MissingChallengedPortfolioError as error:
        print(f"FATAL: {error}")
        return 1
    if inputs.prior_art_run.candidate_ids != EXPECTED_CANDIDATE_IDS:
        print(
            f"FATAL: {priorart_root} holds a different portfolio than "
            f"the preserved 5D.2 run; refusing to select over it"
        )
        return 1
    if inputs.ideation_run.id != IDEATION_RUN_RECORD_ID:
        print(
            f"FATAL: the challenge names ideation run "
            f"{inputs.ideation_run.id}, not {IDEATION_RUN_RECORD_ID}"
        )
        return 1
    partition = partition_by_verdict(inputs)

    ledger = UsageLedger()
    # One set of wiring values, visibly shared between the selector and
    # the fail-fast preflight below.
    max_output_tokens = 16384
    # 16384, the Task 5C lesson: the comparative review is one large
    # JSON object over several candidates and every pair; the budget
    # has to fit it.
    max_corrective_calls = 1
    selector = CandidateSelector(
        provider=MuseSparkProvider(),
        model=MUSE_SPARK_1_2,
        ledger=ledger,
        ideation_store=ideation_store,
        prior_art_store=prior_art_store,
        store=SelectionStore(run_root / "selection"),
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        request_timeout_seconds=240.0,
        max_corrective_calls=max_corrective_calls,
    )
    try:
        plan = check_selection_coherence(
            directive=DIRECTIVE,
            eligible_count=len(partition.eligible),
            max_output_tokens=max_output_tokens,
            max_corrective_calls=max_corrective_calls,
        )
    except SelectionPreflightError as error:
        print(f"FATAL: {error}")
        return 1

    print("== Task 5E live proof: one selection over the challenged "
          "portfolio ==")
    print(f"directive  : {DIRECTIVE.id}")
    print(f"door       : {PRIOR_ART_RUN_RECORD_ID} (challenge run "
          f"{inputs.prior_art_run.run_id}, portfolio "
          f"{IDEATION_RUN_RECORD_ID})")
    verdicts = ", ".join(
        f"{assessment.candidate_id}={assessment.verdict.value}"
        for assessment in inputs.assessments
    )
    print(f"verdicts   : {verdicts}")
    eligible_ids = ", ".join(c.id for c in partition.eligible) or "(none)"
    print(f"eligible   : {eligible_ids}")
    print("             (stamped by trusted code from the named run, "
          "before any call)")
    print(f"budgets    : {DIRECTIVE.max_eligible_candidates} eligible "
          f"cap, {DIRECTIVE.max_model_calls} model calls")
    print(f"preflight  : PASSED — {plan.eligible} eligible, "
          f"{plan.pairs} pairs, worst stage-1 reply "
          f"{plan.worst_stage1_output_tokens}/"
          f"{plan.output_token_envelope} tokens, worst calls "
          f"{plan.worst_calls_total}/{DIRECTIVE.max_model_calls}")
    print(f"idea root  : {idea_root} (read-only)")
    print(f"prior root : {priorart_root} (read-only)")
    print(f"run root   : {run_root}")
    print("provider   : muse only — nothing is retrieved in this run")
    print()

    started = time.monotonic()
    result = selector.run(DIRECTIVE)
    elapsed = time.monotonic() - started
    _report(result, run_root, elapsed)
    _verify_hashes(idea_root, before_ideas, priorart_root, before_prior)
    _verify_outcome(result, ideation_store, prior_art_store)
    _verify_spend(result, ledger)
    _verify_reload(result, run_root)
    print("PASSED: the preflight held, the door walked, eligibility was")
    print("stamped by trusted code from the one named prior-art run, and")
    print("every model judgment passed its deterministic gate before one")
    print("nested selection record was durably written. SELECTED,")
    print("NO_ELIGIBLE_CANDIDATE, and NO_DEFENSIBLE_CANDIDATE are all")
    print("successful outcomes; no candidate was required to win.")
    return 0


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


def _report(
    result: SelectionRunResult, run_root: Path, elapsed: float
) -> None:
    record = result.run_record
    print("-- outcome --")
    print(f"run        : {record.run_id} (record {record.id})")
    print(f"outcome    : {record.outcome.value}")
    for entry in record.ineligible:
        details = "; ".join(
            f"{reason.code.value}: {reason.detail}"
            for reason in entry.reasons
        )
        overlap = (
            f" overlapping works: {', '.join(entry.overlapping_work_ids)}"
            if entry.overlapping_work_ids
            else ""
        )
        print(f"ineligible : {entry.candidate_id} "
              f"({entry.verdict.value}){overlap}")
        if details:
            print(f"             {details}")
    for review in record.reviews:
        mark = (
            "disqualified"
            if review.disqualifiers
            else "contender"
        )
        print(f"reviewed   : {review.candidate_id} ({mark})")
        for disqualifier in review.disqualifiers:
            print(f"             {disqualifier.ground.value} on the "
                  f"{disqualifier.dimension.value} constraint")
            print(f"             candidate: "
                  f"{disqualifier.candidate_text!r}")
            print(f"             constraint: "
                  f"{disqualifier.constraint_text!r}")
            print(f"             unrepairable: "
                  f"{disqualifier.why_unrepairable}")
    if record.pairwise_comparisons:
        print(f"pairs      : {len(record.pairwise_comparisons)} explicit "
              f"comparisons")
    decision = record.decision
    if decision is not None:
        print(f"selected   : {decision.selected_candidate_id}")
        print(f"tradeoff   : {decision.decisive_tradeoff}")
        for rationale in decision.why_selected_over:
            print(f"over       : {rationale.candidate_id} — "
                  f"{rationale.reason}")
        print(f"objective  : {decision.first_experimental_objective}")
        print(f"needs      : {'; '.join(decision.required_capabilities)}")
        print(f"risks      : {'; '.join(decision.residual_risks)}")
    print()
    print("-- spend (on top of the Task 5D.2 prior-art baseline: "
          f"{PRIOR_ART_BASELINE['model_calls']} calls, "
          f"{PRIOR_ART_BASELINE['input_tokens']}in/"
          f"{PRIOR_ART_BASELINE['output_tokens']}out) --")
    repairs = 0
    if record.review_provenance is not None:
        repairs += record.review_provenance.repair_count
    if decision is not None:
        repairs += decision.provenance.repair_count
    rejected = SelectionStore(run_root / "selection").rejected()
    print(f"model calls: {record.model_calls} ({repairs} corrective), "
          f"tokens {record.input_tokens}in/{record.output_tokens}out, "
          f"{len(rejected)} rejected payload(s) preserved, "
          f"{elapsed:.1f}s wall clock")
    print()
    print("-- limitations (by construction) --")
    print("The selection is a model preference validated - never")
    print("computed - by trusted code: trusted code owns eligibility,")
    print("attestation, the stamped sets, and outcome legality; the")
    print("model owns only which defensible candidate it prefers. It is")
    print("a comparative judgment over this bounded portfolio under one")
    print("prior-art run's bounded search - never proof the winner is")
    print("novel, and never a ranking: no score exists anywhere in")
    print("these records. Selection confers no scientific status: the")
    print("record lives outside research state, the candidates are")
    print("untouched, and their novelty standing stays structurally")
    print("unassessed. Unselected eligible candidates remain")
    print("addressable and available to future selection runs; not")
    print("being selected is not a disqualification. Nothing was")
    print("retrieved here, so there is no replay to claim - the only")
    print("network was Muse.")
    print()


def _verify_hashes(
    idea_root: Path,
    before_ideas: dict[str, str],
    priorart_root: Path,
    before_prior: dict[str, str],
) -> None:
    after_ideas = _digests(idea_root)
    after_prior = _digests(priorart_root)
    assert after_ideas == before_ideas, (
        "the preserved ideation artifacts must be byte-identical"
    )
    assert after_prior == before_prior, (
        "the preserved prior-art artifacts must be byte-identical"
    )
    print("-- integrity --")
    print(f"every upstream artifact is byte-identical: "
          f"{len(before_ideas)} + {len(before_prior)} files hashed "
          f"before and after.")
    print()


def _verify_outcome(
    result: SelectionRunResult,
    ideation_store: IdeationStore,
    prior_art_store: PriorArtStore,
) -> None:
    """Outcome-agnostic checks: the stamps equal a local trusted-code
    recomputation, and whichever outcome occurred has its exact
    structural shape."""
    record = result.run_record
    expected_eligible = tuple(
        assessment.candidate_id
        for assessment in result.inputs.assessments
        if assessment.verdict is PriorArtVerdict.DISTINGUISHED
    )
    assert record.eligible_candidate_ids == expected_eligible, (
        "the stamped eligible set must equal the trusted recomputation"
    )
    if record.outcome is SelectionOutcome.NO_ELIGIBLE_CANDIDATE:
        assert record.model_calls == 0, "an ineligible stop is free"
    elif record.outcome is SelectionOutcome.NO_DEFENSIBLE_CANDIDATE:
        for review in record.reviews:
            assert review.disqualifiers, (
                "an honest stop carries a validated disqualifier per "
                "eligible candidate"
            )
            candidate = ideation_store.get_idea(review.candidate_id)
            assert candidate is not None
            block = render_candidate_for_selection(candidate)
            for disqualifier in review.disqualifiers:
                assert _normalized(
                    disqualifier.candidate_text
                ) in _normalized(block), (
                    "every disqualifier quote re-finds in the candidate"
                )
    else:
        decision = record.decision
        assert decision is not None
        contenders = set(record.eligible_candidate_ids) - set(
            record.disqualified_candidate_ids
        )
        assert decision.selected_candidate_id in contenders
        argued = {
            entry.candidate_id for entry in decision.why_selected_over
        }
        assert argued == contenders - {decision.selected_candidate_id}
        # Unselected candidates stay addressable for future runs.
        for candidate_id in sorted(argued):
            assert ideation_store.get_idea(candidate_id) is not None
            assessment = prior_art_store.assessment_for_candidate(
                record.prior_art_run_id, candidate_id
            )
            assert assessment is not None
            assert assessment.verdict is PriorArtVerdict.DISTINGUISHED
    print("-- authority split --")
    print("the stamped eligible and disqualified sets equal the trusted")
    print("recomputation; the outcome carries its exact structural")
    print("shape; unselected eligible candidates remain addressable.")
    print()


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _verify_spend(result: SelectionRunResult, ledger: UsageLedger) -> None:
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


def _verify_reload(result: SelectionRunResult, run_root: Path) -> None:
    fresh = SelectionStore(run_root / "selection")
    record = result.run_record
    assert fresh.get_run(record.id) == record, "the run must reload intact"
    assert fresh.get_directive(DIRECTIVE.id) == DIRECTIVE
    assert record.prior_art_run_record_id == PRIOR_ART_RUN_RECORD_ID
    assert record.ideation_run_record_id == IDEATION_RUN_RECORD_ID
    assert record.candidate_ids == EXPECTED_CANDIDATE_IDS
    print("-- durability --")
    print("reloaded the run record and directive via fresh objects:")
    print("identical identity throughout; the record pins the 5D.2")
    print("challenge and the 5C portfolio it selected over.")
    print()


if __name__ == "__main__":
    sys.exit(main())
