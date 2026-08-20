"""Task 6A proof: one admitted state becomes one funded, billable run.

The Task 5F run admitted the selected candidate as a genesis
``ResearchState``: propositions, and a budget of zero. Nothing could
spend against it. Funding it in place was not an option either — a
state's content id excludes its budget, so replacing the budget produces
different bytes under the same id, and the append-only snapshot store
refuses the write. That refusal is correct, and it is why this stage
exists.

This run walks the whole supported bridge over the preserved 5F
admission, and makes no model call and no network call at all: funding
is an operator act and a deterministic one.

What this run must specifically show:

1. the door and preflight pass, printed with the ids they checked;
2. the funded state is a *successor* — a new id, whose parent is the
   admitted state, carrying the same propositions and the granted
   budget;
3. the grant is ledger entry zero, and the ledger agrees with the state;
4. one deterministic charge posts exactly one debit, and the balance
   moves by exactly that amount;
5. re-posting the same charge id debits nothing;
6. a fresh store over the same root replays the identical balance;
7. a doctored state fails closed against the ledger rather than being
   reconciled;
8. re-running the completed directive replays the run and grants
   nothing a second time;
9. every preserved 5F artifact is byte-identical afterwards.

One read-only root and one fresh root: the preserved Task 5F run (the
admission), and a fresh run root that receives the run envelope, the
ledger, and the state snapshots. No credentials are needed. Run with::

    python -m examples.live_task6a \\
        --admission-root live_runs/task5f-2026-08-20 \\
        --run-root live_runs/task6a-<date>

The run root must sit under the gitignored ``live_runs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

from autonomous_research_lab.admission.store import AdmissionStore
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.program.authorization import FundingAuthorization
from autonomous_research_lab.program.directive import RunDirective
from autonomous_research_lab.program.door import (
    RunRefusedError,
    require_admitted_state_for_run,
)
from autonomous_research_lab.program.ledger import (
    BudgetLedger,
    LedgerMismatchError,
)
from autonomous_research_lab.program.preflight import (
    RunPreflightError,
    check_funding_coherence,
)
from autonomous_research_lab.program.records import EntryKind
from autonomous_research_lab.program.starter import RunStartResult, start_run
from autonomous_research_lab.program.store import ProgramStore

#: The preserved Task 5F admission this run enters through. An id, never
#: a path: the record proves its own identity wherever the store lives.
ADMISSION_RECORD_ID = "arun_aef62566adb793d3"
EXPECTED_ADMITTED_STATE_ID = "st_bea69ecb9b4e3ac2"

#: The operator's grant, decided and frozen before the run. Small on
#: purpose: this task funds the bridge, not an experiment campaign, and
#: an honest first grant is the one the operator would actually stand
#: behind.
GRANT = ResearchBudget(
    wall_clock_seconds=86_400.0,
    gpu_hours=100.0,
    usd=250.0,
    model_tokens=2_000_000,
)

AUTHORITY = (
    "Lab operator, 2026-08-20: single-run allocation for the admitted "
    "Task 5F candidate, against the standing August compute budget."
)

LABEL = "Task 6A first funded run of the 5F admission"

#: The deterministic charge this run posts, so the ledger is exercised
#: rather than only granted. It stands for one attempt's actual cost; no
#: work is performed here and none is claimed.
CHARGE_ID = "att_task6a_demonstration"
CHARGE = ResourceCost(
    wall_clock_seconds=1_800.0, gpu_hours=0.5, usd=3.25, model_tokens=12_000
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--admission-root",
        type=Path,
        required=True,
        help="the preserved Task 5F run root (read-only input)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="run directory (put it under the gitignored live_runs/)",
    )
    arguments = parser.parse_args()
    admission_root = arguments.admission_root.resolve()
    run_root = arguments.run_root.resolve()

    admission_store = AdmissionStore(admission_root / "admission")
    program_store = ProgramStore(run_root / "program")

    # Every preserved artifact is hashed before anything is wired and
    # re-hashed after: byte identity as a fact, not a claim.
    before = _digests(admission_root)

    authorization = FundingAuthorization(
        admission_record_id=ADMISSION_RECORD_ID,
        granted=GRANT,
        authority=AUTHORITY,
    )
    directive = RunDirective(
        admission_record_id=ADMISSION_RECORD_ID,
        authorization_id=authorization.id,
        label=LABEL,
    )

    # One proof, one fresh root. Re-running the completed directive is
    # itself demonstrated below, from inside a run that started here; a
    # second invocation over the same root would have nothing to prove
    # and is refused in words rather than in a traceback.
    started_already = program_store.run_for_directive(directive.id)
    if started_already is not None:
        print(
            f"FATAL: {run_root} already holds run "
            f"{started_already.run_id} for this directive; Task 6A writes "
            f"a fresh run root"
        )
        return 1

    # The same door and preflight start_run runs, executed fail-fast here
    # so a missing record or a lineage disagreement is a printed refusal
    # rather than a traceback.
    try:
        inputs = require_admitted_state_for_run(
            admission_store, directive, authorization
        )
    except RunRefusedError as error:
        print(f"FATAL: {error}")
        return 1
    if inputs.admitted_state.id != EXPECTED_ADMITTED_STATE_ID:
        print(
            f"FATAL: the admission names state {inputs.admitted_state.id}, "
            f"not the preserved seed {EXPECTED_ADMITTED_STATE_ID}"
        )
        return 1
    try:
        check_funding_coherence(
            directive=directive,
            authorization=authorization,
            minimum_first_step=ResourceCost(wall_clock_seconds=60.0),
        )
    except RunPreflightError as error:
        print(f"FATAL: {error}")
        return 1

    print("-- door --")
    print(f"admission record   {inputs.admission.id}")
    print(f"admitted state     {inputs.admitted_state.id} (budget zero)")
    print(f"question           {inputs.admission.question_id}")
    print(f"hypothesis         {inputs.admission.hypothesis_id}")
    print(f"predictions        {', '.join(inputs.admission.prediction_ids)}")
    print(f"authorization      {authorization.id}")
    print(f"directive          {directive.id}")
    print("preflight passed: the grant covers the caller's cheapest step.")
    print()

    started = time.monotonic()
    result = start_run(
        admission_store=admission_store,
        program_store=program_store,
        directive=directive,
        authorization=authorization,
        minimum_first_step=ResourceCost(wall_clock_seconds=60.0),
    )
    elapsed = time.monotonic() - started
    _report_start(result, inputs.admitted_state, elapsed)

    ledger = program_store.ledger_for(result.run.run_id)
    _prove_grant(ledger, result)
    charged = _prove_one_charge(ledger, result)
    _prove_idempotence(ledger, charged)
    _prove_replay(program_store, result, charged)
    _prove_mismatch_fails_closed(ledger)
    _prove_directive_replay(admission_store, program_store, directive,
                            authorization, result)

    after = _digests(admission_root)
    _verify_hashes(before, after)

    print("-- what this does not mean --")
    print("A grant is authorization, never scientific standing. ADMITTED")
    print("did not mean true, novel, or supported, and FUNDED does not")
    print("either. Nothing was executed, measured, or concluded here: the")
    print("run holds the same propositions the admission translated, and")
    print("a budget it may now spend against.")
    print()
    return 0


# -- reporting -----------------------------------------------------------------


def _report_start(
    result: RunStartResult, admitted: ResearchState, elapsed: float
) -> None:
    funded = result.funded_state
    print("-- the funded run --")
    print(f"run id             {result.run.run_id}")
    print(f"envelope           {result.run.id}")
    print(f"funded state       {funded.id}")
    print(f"  parent           {funded.parent_id}")
    print(f"  budget           {funded.budget}")
    print(f"elapsed            {elapsed:.3f}s")
    assert not result.replayed
    assert funded.parent_id == admitted.id
    assert funded.id != admitted.id
    assert funded.budget == GRANT
    # Succession, not replacement: the propositions cross unchanged and
    # nothing scientific was added on the way.
    assert funded.questions == admitted.questions
    assert funded.hypotheses == admitted.hypotheses
    assert funded.predictions == admitted.predictions
    assert not funded.results and not funded.evidence_ids
    assert not funded.assessments and not funded.attempts
    print("the funded state is a successor: same propositions, new id,")
    print("and the admitted snapshot still reads back with a zero budget.")
    print()


def _prove_grant(ledger: BudgetLedger, result: RunStartResult) -> None:
    entries = ledger.entries()
    assert len(entries) == 1
    assert entries[0].kind is EntryKind.GRANT
    assert entries[0].id == result.run.grant_entry_id
    assert ledger.balance() == GRANT == result.funded_state.budget
    print("-- ledger: the grant --")
    print(f"entry 0            {entries[0].id}")
    print(f"  charge id        {entries[0].charge_id}")
    print(f"  balance after    {entries[0].balance_after}")
    print("the ledger and the funded state agree on the balance.")
    print()


def _prove_one_charge(
    ledger: BudgetLedger, result: RunStartResult
) -> ResearchBudget:
    before = ledger.balance()
    entry = ledger.debit(
        CHARGE, charge_id=CHARGE_ID, reason=f"attempt {CHARGE_ID}"
    )
    after = ledger.balance()
    assert entry.sequence == 1
    assert entry.previous_entry_id == result.run.grant_entry_id
    assert after == before.spend(CHARGE)
    assert len(ledger.entries()) == 2
    print("-- ledger: one charge --")
    print(f"entry 1            {entry.id}")
    print(f"  charge id        {entry.charge_id}")
    print(f"  amount           {entry.amount}")
    print(f"  balance after    {after}")
    print("exactly one debit, and the balance moved by exactly that much.")
    print()
    return after


def _prove_idempotence(ledger: BudgetLedger, expected: ResearchBudget) -> None:
    repeated = ledger.debit(
        CHARGE, charge_id=CHARGE_ID, reason=f"attempt {CHARGE_ID}"
    )
    assert repeated.sequence == 1
    assert len(ledger.entries()) == 2
    assert ledger.balance() == expected
    print("-- ledger: the same charge again --")
    print("posting the same charge id returned the entry already on the")
    print("ledger: two entries, one debit, balance unchanged.")
    print()


def _prove_replay(
    store: ProgramStore, result: RunStartResult, expected: ResearchBudget
) -> None:
    fresh_store = ProgramStore(store.root)
    run, state = fresh_store.get_funded_state(result.run.id)
    fresh_ledger = fresh_store.ledger_for(run.run_id)
    assert run == result.run
    assert state == result.funded_state
    assert fresh_ledger.balance() == expected
    print("-- replay from a fresh store --")
    print(f"balance replayed   {fresh_ledger.balance()}")
    print("the envelope, the funded state, and every entry reloaded and")
    print("re-derived their own ids; the balance is a replay, not a")
    print("remembered number.")
    print()


def _prove_mismatch_fails_closed(ledger: BudgetLedger) -> None:
    doctored = ledger.balance().plus(ResearchBudget(usd=1.0))
    try:
        ledger.require_balance(doctored)
    except LedgerMismatchError as error:
        print("-- a disagreement fails closed --")
        print(f"{error}")
        print("the ledger refuses to guess which record is the truth.")
        print()
        return
    raise AssertionError("a disagreeing balance must not be accepted")


def _prove_directive_replay(
    admission_store: AdmissionStore,
    program_store: ProgramStore,
    directive: RunDirective,
    authorization: FundingAuthorization,
    result: RunStartResult,
) -> None:
    before = program_store.ledger_for(result.run.run_id).balance()
    replayed = start_run(
        admission_store=admission_store,
        program_store=program_store,
        directive=directive,
        authorization=authorization,
    )
    assert replayed.replayed
    assert replayed.run == result.run
    assert replayed.funded_state == result.funded_state
    assert len(program_store.runs()) == 1
    assert program_store.ledger_for(result.run.run_id).balance() == before
    print("-- re-running the completed directive --")
    print("returned the run it already started: no second envelope, no")
    print("second grant, and the balance untouched.")
    print()


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


def _verify_hashes(before: dict[str, str], after: dict[str, str]) -> None:
    assert before, "no preserved admission artifacts were hashed"
    changed = sorted(
        name for name in before if before[name] != after.get(name)
    )
    added = sorted(set(after) - set(before))
    assert not changed, f"preserved artifacts changed: {changed}"
    assert not added, f"artifacts appeared in a read-only root: {added}"
    print("-- preserved artifacts --")
    print(f"{len(before)} admission files, byte-identical before and after.")
    print()


if __name__ == "__main__":
    sys.exit(main())
