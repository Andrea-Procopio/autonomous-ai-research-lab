"""The bridge: one admitted state becomes one funded run.

:func:`start_run` is the only supported route from the analysis chain
into the runtime, and the fixed sequence it performs is the whole
contract::

    RunDirective + FundingAuthorization
      -> replay?   a completed directive returns the run it already
                   started, with no second grant and nothing rewritten
      -> require_admitted_state_for_run   (the door; before any write)
      -> check_funding_coherence          (the preflight; before the grant)
      -> preserve the admitted snapshot into the run root
      -> funded = admitted.fund(grant)    (a SUCCESSOR, never a replacement)
      -> persist the funded snapshot and read it back
      -> ledger entry zero: the grant
      -> write the run envelope             (last)

The ordering is admission's, one level up, and for the same reason. The
envelope is written last, so a crash before it leaves an inert orphan
snapshot and an ungranted or granted-but-unenveloped ledger — "no
envelope means no run" — and the re-run starts honestly rather than
inheriting a half-built one. The grant is idempotent by authorization
id, so the re-run's ledger cannot end up double-credited either.

Copying the admitted snapshot into the run root is deliberate: the
snapshot store is content-addressed and verifies on repeat, so the copy
is a byte-identical no-op that cannot diverge, and it leaves the run
root holding the whole lineage from the genesis state onward.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..admission.store import AdmissionStore
from ..core.budget import NO_COST, ResourceCost
from ..core.ids import occurrence_id
from ..core.state import ResearchState
from .authorization import FundingAuthorization
from .directive import RunDirective
from .door import RunInputs, require_admitted_state_for_run
from .preflight import RunPlan, check_funding_coherence
from .records import ResearchRun
from .store import ProgramStore


@dataclass(frozen=True, slots=True)
class RunStartResult:
    """What starting a run produced. ``replayed`` is ``True`` when the
    directive had already started its run and nothing was written."""

    run: ResearchRun
    funded_state: ResearchState
    inputs: RunInputs | None
    plan: RunPlan | None
    replayed: bool


def start_run(
    *,
    admission_store: AdmissionStore,
    program_store: ProgramStore,
    directive: RunDirective,
    authorization: FundingAuthorization,
    minimum_first_step: ResourceCost = NO_COST,
) -> RunStartResult:
    """Fund one admitted state into one run, or replay the run this
    directive already started."""
    existing = program_store.run_for_directive(directive.id)
    if existing is not None:
        run, funded = program_store.get_funded_state(existing.id)
        return RunStartResult(
            run=run,
            funded_state=funded,
            inputs=None,
            plan=None,
            replayed=True,
        )

    inputs = require_admitted_state_for_run(
        admission_store, directive, authorization
    )
    plan = check_funding_coherence(
        directive=directive,
        authorization=authorization,
        minimum_first_step=minimum_first_step,
    )

    program_store.record_directive(directive)
    program_store.record_authorization(authorization)

    admitted = inputs.admitted_state
    program_store.persist_state(admitted)
    funded = admitted.fund(authorization.granted)
    program_store.persist_state(funded)

    run_id = occurrence_id("run")
    grant_entry = program_store.ledger_for(run_id).grant(authorization)

    run = program_store.record_run(
        ResearchRun(
            run_id=run_id,
            directive_id=directive.id,
            authorization_id=authorization.id,
            admission_record_id=inputs.admission.id,
            admitted_state_id=admitted.id,
            funded_state_id=funded.id,
            granted=authorization.granted,
            grant_entry_id=grant_entry.id,
            label=directive.label,
            authority=authorization.authority,
            question_id=inputs.admission.question_id,
            hypothesis_id=inputs.admission.hypothesis_id,
            prediction_ids=inputs.admission.prediction_ids,
        )
    )
    return RunStartResult(
        run=run,
        funded_state=funded,
        inputs=inputs,
        plan=plan,
        replayed=False,
    )
