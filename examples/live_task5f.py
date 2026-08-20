"""Task 5F live proof: one governed admission of the selected candidate.

The Task 5E run selected one candidate over two undisqualified
alternatives. This run walks the one door into admission — the directive
names that selection run record explicitly, and trusted code re-verifies
the complete lineage behind it (selection, prior art, ideation,
direction, CFP snapshot) before anything else — then makes at most one
gated model call (plus at most one corrective) to translate the
candidate's recorded predictions into machine-checkable core
predictions under the sign-only neutral encoding. Everything else the
initial state holds is a deterministic verbatim copy. Nothing is tuned
toward preferred hypothesis wording; a fail-closed refusal is an honest
result.

What this run must specifically show: the door and preflight pass and
are printed; the admitted state is a bare linked seed (question,
hypothesis, encoded predictions, zero budget, nothing else); the record
spend reconciles exactly with the ledger; the admission reloads intact
from a fresh store; re-running the completed directive replays the
stored result with ZERO provider calls; and every preserved upstream
artifact is byte-identical afterwards (hashed before and after).
Nothing is retrieved in this run, so there is no zero-network replay of
retrieval to claim — the only network traffic is Muse.

Three read-only roots and one fresh root: the preserved Task 5C run
(the immutable portfolio), the preserved Task 5D.2 run (the verdicts),
the preserved Task 5E run (the selection), and a fresh run root that
receives the admission records and the state snapshot. Requires
MUSE_API_KEY (or MODEL_API_KEY) and outbound HTTPS to Muse only. Run
with::

    python -m examples.live_task5f \\
        --idea-root live_runs/task5c-2026-08-19 \\
        --priorart-root live_runs/task5d2-2026-08-19 \\
        --selection-root live_runs/task5e-2026-08-20 \\
        --run-root live_runs/task5f-<date>

The run root must sit under the gitignored ``live_runs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from autonomous_research_lab.admission.admitter import (
    AdmissionRunResult,
    CandidateAdmitter,
)
from autonomous_research_lab.admission.directive import AdmissionDirective
from autonomous_research_lab.admission.door import (
    AdmissionRefusedError,
    require_selected_candidate_for_admission,
)
from autonomous_research_lab.admission.preflight import (
    AdmissionPreflightError,
    check_admission_coherence,
)
from autonomous_research_lab.admission.store import AdmissionStore
from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.prediction import Comparator
from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.runtime.muse import (
    KEY_ENV_VARS,
    MUSE_SPARK_1_2,
    MuseSparkProvider,
)
from autonomous_research_lab.runtime.providers import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    UsageLedger,
)
from autonomous_research_lab.selection.store import SelectionStore

#: The preserved Task 5E selection this admission enters through, and
#: the candidate it selected. Ids, never paths: the records prove their
#: own identity wherever the stores live.
SELECTION_RUN_RECORD_ID = "srun_fd598b0eb2e3e80b"
EXPECTED_SELECTED_CANDIDATE_ID = "idea_1e1fa63952cc0d91"

#: The operator's execution-environment statements, provided 2026-08-20
#: and frozen before the run. They are operator facts about where the
#: admitted work will eventually run — recorded as operator-stated
#: requirements, never presented as inherited from any upstream record,
#: and never implemented in this task.
DIRECTIVE = AdmissionDirective(
    selection_run_record_id=SELECTION_RUN_RECORD_ID,
    scheduling_requirement=(
        "Batch-scheduled execution on shared infrastructure (e.g. "
        "Slurm-class schedulers) with variable GPU availability."
    ),
    job_duration_requirement=(
        "Individual jobs bounded to at most two days of wall-clock time."
    ),
    checkpoint_requirement=(
        "All long-running work must support checkpointing and resume "
        "across job boundaries."
    ),
)

#: What producing this admission's input cost (the Task 5E selection).
#: The admission spend below is on top.
SELECTION_BASELINE = {
    "model_calls": 2,
    "input_tokens": 7_496,
    "output_tokens": 10_833,
}


class _RefusingProvider(ModelProvider):
    """A provider that refuses every call: anything completed against it
    was completed without the model — exactly what the replay must
    prove."""

    @property
    def name(self) -> str:
        return "refusing"

    def invoke(self, _request: ModelRequest) -> ModelResponse:
        raise AssertionError(
            "the replay of a completed admission must not call the "
            "provider"
        )


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
        "--selection-root",
        type=Path,
        required=True,
        help="the preserved Task 5E run root (read-only input)",
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
    selection_root = arguments.selection_root.resolve()
    run_root = arguments.run_root.resolve()

    if not any(os.environ.get(name, "").strip() for name in KEY_ENV_VARS):
        print(
            f"FATAL: no Muse API key in the environment "
            f"({' or '.join(KEY_ENV_VARS)})"
        )
        return 1
    ideation_store = IdeationStore(idea_root / "ideation")
    prior_art_store = PriorArtStore(priorart_root / "priorart")
    selection_store = SelectionStore(selection_root / "selection")

    # Every preserved upstream artifact is hashed before wiring anything
    # and re-hashed after the run: byte identity as a fact, not a claim.
    before = {
        "ideation": _digests(idea_root),
        "prior art": _digests(priorart_root),
        "selection": _digests(selection_root),
    }

    # The same door the admitter itself runs, executed fail-fast here so
    # a missing record or a lineage disagreement is a printed refusal,
    # not a traceback — and the selected candidate is on the record
    # before the first call.
    try:
        inputs = require_selected_candidate_for_admission(
            selection_store,
            prior_art_store,
            ideation_store,
            SELECTION_RUN_RECORD_ID,
        )
    except AdmissionRefusedError as error:
        print(f"FATAL: {error}")
        return 1
    if inputs.selected.id != EXPECTED_SELECTED_CANDIDATE_ID:
        print(
            f"FATAL: the selection record names "
            f"{inputs.selected.id}, not the preserved winner "
            f"{EXPECTED_SELECTED_CANDIDATE_ID}"
        )
        return 1

    ledger = UsageLedger()
    # One set of wiring values, visibly shared between the admitter and
    # the fail-fast preflight below.
    max_output_tokens = 16384
    # 16384, the Task 5C lesson: one JSON object over every encoding;
    # the budget has to fit it.
    max_corrective_calls = 1
    store = AdmissionStore(run_root / "admission")
    admitter = CandidateAdmitter(
        provider=MuseSparkProvider(),
        model=MUSE_SPARK_1_2,
        ledger=ledger,
        ideation_store=ideation_store,
        prior_art_store=prior_art_store,
        selection_store=selection_store,
        store=store,
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        request_timeout_seconds=240.0,
        max_corrective_calls=max_corrective_calls,
    )
    try:
        plan = check_admission_coherence(
            directive=DIRECTIVE,
            prediction_count=len(inputs.selected.predictions),
            max_output_tokens=max_output_tokens,
            max_corrective_calls=max_corrective_calls,
        )
    except AdmissionPreflightError as error:
        print(f"FATAL: {error}")
        return 1

    print("== Task 5F live proof: one governed admission of the selected "
          "candidate ==")
    print(f"directive  : {DIRECTIVE.id}")
    print(f"door       : {SELECTION_RUN_RECORD_ID} (selection run "
          f"{inputs.selection_run.run_id}, outcome "
          f"{inputs.selection_run.outcome.value})")
    print(f"selected   : {inputs.selected.id}")
    print(f"             {inputs.selected.title!r}")
    print(f"lineage    : prior art {inputs.prior_art_run.id}, ideation "
          f"{inputs.ideation_run.id},")
    print(f"             direction {inputs.direction.id}, snapshot "
          f"{inputs.snapshot.id},")
    print(f"             map run {inputs.ideation_run.map_run_id}, map "
          f"adequacy {inputs.ideation_run.assessment_id}")
    print(f"predictions: {len(inputs.selected.predictions)} recorded, "
          f"{len(inputs.selected.metrics)} declared metrics")
    print(f"preflight  : PASSED — worst reply "
          f"{plan.worst_case_output_tokens}/{max_output_tokens} tokens, "
          f"worst calls {plan.worst_case_calls}/"
          f"{DIRECTIVE.max_model_calls}")
    print(f"idea root  : {idea_root} (read-only)")
    print(f"prior root : {priorart_root} (read-only)")
    print(f"sel root   : {selection_root} (read-only)")
    print(f"run root   : {run_root}")
    print("provider   : muse only — nothing is retrieved in this run")
    print()

    started = time.monotonic()
    result = admitter.run(DIRECTIVE)
    elapsed = time.monotonic() - started
    _report(result, store, elapsed)
    _verify_hashes(
        before,
        {
            "ideation": _digests(idea_root),
            "prior art": _digests(priorart_root),
            "selection": _digests(selection_root),
        },
    )
    _verify_state(result, ideation_store)
    _verify_spend(result, ledger)
    _verify_reload(result, run_root)
    _verify_replay(
        result,
        ideation_store,
        prior_art_store,
        selection_store,
        store,
    )
    print("PASSED: the door verified the whole lineage, the preflight")
    print("held, one gated translation passed its deterministic gate,")
    print("and the admitted state was written all-or-nothing beside its")
    print("write-once record. An admission is a governed translation of")
    print("a validated model preference — DISTINGUISHED, SELECTED, and")
    print("ADMITTED are rungs of bounded process, and none of them means")
    print("true, novel, or empirically supported.")
    return 0


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


def _report(
    result: AdmissionRunResult, store: AdmissionStore, elapsed: float
) -> None:
    record = result.record
    state = result.state
    print("-- outcome --")
    print(f"run        : {record.run_id} (record {record.id})")
    print(f"admitted   : {record.selected_candidate_id}")
    print(f"state      : {record.state_id}")
    print(f"objective  : {state.objective}")
    (question,) = state.questions
    (hypothesis,) = state.hypotheses
    print(f"question   : {question.id}: {question.text}")
    print(f"hypothesis : {hypothesis.id}: {hypothesis.statement}")
    for entry, prediction_id in zip(
        record.operational_predictions, record.prediction_ids, strict=True
    ):
        print(f"prediction : {prediction_id}")
        print(f"             encodes: {entry.prediction_text!r}")
        print(f"             condition: {entry.condition}")
        print(f"             metric: difference in {entry.base_metric}: "
              f"{entry.expected_higher_arm} minus "
              f"{entry.expected_lower_arm} > 0")
        print(f"             contrary: {entry.contrary_observation!r}")
        for link in entry.support:
            print(f"             support: {link.source.value} "
                  f"{link.field_path}: {link.quote!r}")
    print(f"mechanical : {record.mechanical_reading} — comparative prose "
          f"is weakened to the sign of the difference; effect sizes are "
          f"the planner's work")
    print("-- requirements (by provenance, all verbatim quotes) --")
    for requirement in record.inherited_requirements:
        print(f"inherited  : [{requirement.source.value}] "
              f"{requirement.field_path}: {requirement.quote!r}")
    for requirement in record.operator_requirements:
        print(f"operator   : {requirement.field_path}: "
              f"{requirement.quote!r}")
    print()
    print("-- spend (on top of the Task 5E selection baseline: "
          f"{SELECTION_BASELINE['model_calls']} calls, "
          f"{SELECTION_BASELINE['input_tokens']}in/"
          f"{SELECTION_BASELINE['output_tokens']}out) --")
    rejected = store.rejected()
    print(f"model calls: {record.model_calls} "
          f"({record.provenance.repair_count} corrective), tokens "
          f"{record.input_tokens}in/{record.output_tokens}out, "
          f"{len(rejected)} rejected payload(s) preserved, "
          f"{elapsed:.1f}s wall clock")
    print()
    print("-- limitations (by construction) --")
    print("An admission is a governed translation of a validated model")
    print("preference - never truth, novelty, or empirical support: the")
    print("state holds propositions only, with no result, evidence,")
    print("assessment, or claim, and a zero budget (budget assignment is")
    print("later operator work). The encoded predictions commit to the")
    print("sign of a difference, not an effect size - choosing real")
    print("thresholds is the planner's job. The execution requirements")
    print("are stated capabilities for later work, not implementations;")
    print("nothing here schedules, checkpoints, or runs anything. The")
    print("gate's number check reads digits, not number words - the")
    print("operator statements are trusted inputs either way. The")
    print("candidate records, the selection record, and every other")
    print("upstream artifact stay byte-identical; unselected candidates")
    print("remain addressable. Nothing was retrieved here, so there is")
    print("no retrieval replay to claim - the only network was Muse.")
    print()


def _verify_hashes(
    before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]
) -> None:
    for name, digests in before.items():
        assert after[name] == digests, (
            f"the preserved {name} artifacts must be byte-identical"
        )
    total = sum(len(digests) for digests in before.values())
    print("-- integrity --")
    print(f"every upstream artifact is byte-identical: {total} files "
          f"hashed before and after across three read-only roots.")
    print()


def _verify_state(
    result: AdmissionRunResult, ideation_store: IdeationStore
) -> None:
    """The admitted seed, checked as a fact: bare, linked, verbatim."""
    record = result.record
    state = result.state
    assert result.inputs is not None
    candidate = result.inputs.selected
    decision = result.inputs.selection_run.decision
    assert decision is not None

    assert state.parent_id is None
    assert state.id == record.state_id
    assert state.budget == ResearchBudget.zero()
    (question,) = state.questions
    (hypothesis,) = state.hypotheses
    assert question.text == candidate.research_question
    assert question.importance == candidate.cfp_alignment
    assert hypothesis.statement == candidate.hypothesis
    assert hypothesis.rationale == candidate.mechanism
    assert hypothesis.question_id == question.id
    assert state.objective == decision.first_experimental_objective
    assert len(state.predictions) >= 1
    for prediction in state.predictions:
        assert prediction.hypothesis_id == hypothesis.id
        assert prediction.comparator is Comparator.GREATER_THAN
        assert prediction.threshold == 0.0
        assert prediction.metric.startswith("difference in ")
    for collection in (
        state.experiments,
        state.results,
        state.evidence_ids,
        state.prediction_tests,
        state.claims,
        state.evidence_links,
        state.assessments,
        state.attempts,
        state.history,
    ):
        assert collection == (), "the admitted seed holds propositions only"
    assert record.measurements == candidate.metrics
    assert record.controls == candidate.ablations
    assert record.comparison_targets == candidate.baselines

    reloaded = ideation_store.get_idea(candidate.id)
    assert reloaded is not None
    assert reloaded.novelty_status.value == "unassessed"
    print("-- the admitted seed --")
    print("bare and linked: question <- hypothesis <- encoded")
    print("predictions, every scientific copy verbatim from the")
    print("candidate or the selection decision, zero budget, empty")
    print("judgment collections, and the candidate's novelty standing")
    print("untouched.")
    print()


def _verify_spend(result: AdmissionRunResult, ledger: UsageLedger) -> None:
    drained = ledger.drain()
    record = result.record
    assert record.input_tokens == drained.input_tokens, (
        "the admission record must reconcile with the ledger"
    )
    assert record.output_tokens == drained.output_tokens, (
        "the admission record must reconcile with the ledger"
    )
    print("-- accounting --")
    print(
        f"admission-record tokens ({record.input_tokens}in/"
        f"{record.output_tokens}out) equal the drained ledger exactly; "
        f"calls {record.model_calls} vs ledger {drained.calls}."
    )
    print()


def _verify_reload(result: AdmissionRunResult, run_root: Path) -> None:
    fresh = AdmissionStore(run_root / "admission")
    record, state = fresh.get_admitted_state(result.record.id)
    assert record == result.record, "the record must reload intact"
    assert state == result.state, "the state must reload intact"
    assert fresh.get_directive(DIRECTIVE.id) == DIRECTIVE
    assert record.selection_run_record_id == SELECTION_RUN_RECORD_ID
    assert record.selected_candidate_id == EXPECTED_SELECTED_CANDIDATE_ID
    print("-- durability --")
    print("reloaded the admission record, the state snapshot, and the")
    print("directive via fresh objects: identical identity throughout;")
    print("the record pins the 5E selection and the whole lineage")
    print("behind it.")
    print()


def _verify_replay(
    result: AdmissionRunResult,
    ideation_store: IdeationStore,
    prior_art_store: PriorArtStore,
    selection_store: SelectionStore,
    store: AdmissionStore,
) -> None:
    """The idempotence proof: the same directive returns the stored
    result through a provider that refuses every call."""
    replayer = CandidateAdmitter(
        provider=_RefusingProvider(),
        model=MUSE_SPARK_1_2,
        ledger=UsageLedger(),
        ideation_store=ideation_store,
        prior_art_store=prior_art_store,
        selection_store=selection_store,
        store=store,
    )
    replayed = replayer.run(DIRECTIVE)
    assert replayed.replayed
    assert replayed.record == result.record
    assert replayed.state == result.state
    print("-- replay --")
    print("re-running the completed directive returned the stored record")
    print("and state through a provider that refuses every call: zero")
    print("model calls, zero spend, nothing rewritten.")
    print()


if __name__ == "__main__":
    sys.exit(main())
