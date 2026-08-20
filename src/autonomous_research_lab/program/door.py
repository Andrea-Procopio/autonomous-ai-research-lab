"""The one door into a funded run, before anything is written.

:func:`require_admitted_state_for_run` reuses admission's own accessor —
:meth:`~..admission.store.AdmissionStore.get_admitted_state` — so the
definition of "an admitted state" is never forked, then adds what
funding alone must prove: the authorization was issued against *this*
admission, the reloaded state is the seed the record describes, and it
is unfunded.

The cross-record equalities are the point, exactly as they are one stage
up. Every loaded record is individually tamper-loud, so a forged record
must be self-consistent — but a self-consistent authorization could
still name a different admission, and a self-consistent admission record
could still point at a snapshot holding somebody else's propositions.
Each check below closes one such seam, and every refusal names the ids
that disagree.

A failed door writes nothing: no grant, no snapshot, no envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..admission.records import AdmissionRecord
from ..admission.store import AdmissionStore
from ..core.state import ResearchState
from .authorization import FundingAuthorization
from .directive import RunDirective


class RunRefusedError(RuntimeError):
    """The one refusal the door raises: a missing admission, an
    authorization for a different admission, an admitted state that
    disagrees with its record, or a seed that is already funded."""


@dataclass(frozen=True, slots=True)
class RunInputs:
    """Everything the door loaded and cross-checked."""

    directive: RunDirective
    authorization: FundingAuthorization
    admission: AdmissionRecord
    admitted_state: ResearchState


def require_admitted_state_for_run(
    admission_store: AdmissionStore,
    directive: RunDirective,
    authorization: FundingAuthorization,
) -> RunInputs:
    """The single entrance to a run: one named admission record whose
    state reloads and matches, funded by an authorization issued against
    that same admission."""
    if directive.authorization_id != authorization.id:
        raise RunRefusedError(
            f"directive {directive.id} names authorization "
            f"{directive.authorization_id}, but was handed "
            f"{authorization.id}"
        )
    if authorization.admission_record_id != directive.admission_record_id:
        raise RunRefusedError(
            f"authorization {authorization.id} funds admission "
            f"{authorization.admission_record_id}, but directive "
            f"{directive.id} runs {directive.admission_record_id}; a "
            f"grant is issued against one admission"
        )
    record = admission_store.get_record(directive.admission_record_id)
    if record is None:
        raise RunRefusedError(
            f"no admission record {directive.admission_record_id} in this "
            f"store; a run enters through a durable admission"
        )
    # The admission accessor is the only way in: it loads the record
    # first and the state through it, and refuses a seed whose budget is
    # not zero. Reusing it keeps one definition of an admitted state.
    record, state = admission_store.get_admitted_state(record.id)

    if state.id != record.state_id:
        raise RunRefusedError(
            f"admission {record.id} names state {record.state_id} but "
            f"loaded {state.id}"
        )
    if not state.budget.is_exhausted:
        # Defense in depth: the accessor above already refuses a seed
        # whose budget is not zero, and lets that integrity error travel
        # as itself. Kept so that widening the accessor could not quietly
        # widen what may be funded.
        raise RunRefusedError(
            f"admitted state {state.id} already carries a budget; a seed "
            f"is funded once, by succession, and never in place"
        )
    if state.parent_id is not None:
        raise RunRefusedError(
            f"admitted state {state.id} has parent {state.parent_id}; a "
            f"genesis state has none, and only a genesis state is funded "
            f"into a run"
        )

    stamped_questions = (record.question_id,)
    loaded_questions = tuple(question.id for question in state.questions)
    stamped_hypotheses = (record.hypothesis_id,)
    loaded_hypotheses = tuple(h.id for h in state.hypotheses)
    loaded_predictions = tuple(p.id for p in state.predictions)
    for label, stamped, loaded in (
        ("question", stamped_questions, loaded_questions),
        ("hypothesis", stamped_hypotheses, loaded_hypotheses),
        ("predictions", tuple(record.prediction_ids), loaded_predictions),
    ):
        if stamped != loaded:
            raise RunRefusedError(
                f"admission {record.id} stamps {label} {list(stamped)}, "
                f"but its state holds {list(loaded)}; a record that "
                f"disagrees with its own state cannot be funded"
            )
    if state.results or state.evidence_ids or state.assessments:
        raise RunRefusedError(
            f"admitted state {state.id} carries results, evidence, or "
            f"judgments; a genesis state holds propositions only"
        )

    return RunInputs(
        directive=directive,
        authorization=authorization,
        admission=record,
        admitted_state=state,
    )
