"""The one door into an admission, before any model call.

:func:`require_selected_candidate_for_admission` reuses the selection
door — :func:`~..selection.eligibility.require_challenged_portfolio_for_selection`
plus :func:`~..selection.eligibility.partition_by_verdict` — so the
definition of "challenged portfolio" is never forked, then adds what
admission alone must prove: the named selection run ended ``SELECTED``,
its stamps equal the reloaded reality, and the records agree with each
other, not only each with itself.

The cross-record equalities are the point. Every loaded record is
individually tamper-loud (ids re-derive on load), so a forged record
must be self-consistent — but a self-consistent selection record could
still name a different portfolio, a different direction, or a different
run than the records it points at. Each equality below closes one such
seam, and every refusal names the ids that disagree.

Lineage depth, resolved deliberately: everything reachable through the
selection, prior-art, and ideation stores is loaded and verified — the
CFP snapshot and the direction included. The mapping ids
(``map_…``/``madq_…``) are cross-checked between the ideation and
prior-art runs and carried verbatim onto the admission record; the
mapping store itself is never loaded. A failed door produces no model
call and no state.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ideation.direction import CfpSnapshot, DirectionRecord
from ..ideation.records import CandidateIdea, IdeationRunRecord
from ..ideation.store import IdeationStore
from ..priorart.assessment import PriorArtAssessment
from ..priorart.records import PriorArtRunRecord
from ..priorart.store import PriorArtStore
from ..selection.directive import SelectionDirective
from ..selection.eligibility import (
    EligibilityPartition,
    MissingChallengedPortfolioError,
    partition_by_verdict,
    require_challenged_portfolio_for_selection,
)
from ..selection.records import SelectionOutcome, SelectionRunRecord
from ..selection.store import SelectionStore


class AdmissionRefusedError(RuntimeError):
    """The one refusal the door raises: a missing or stop-outcome
    selection run, an unloadable lineage record, or any cross-record
    disagreement. Raised before any model call, naming what failed."""


@dataclass(frozen=True, slots=True)
class AdmissionInputs:
    """Everything the door loaded and cross-checked. ``selected`` is the
    exact candidate the selection decision names, and
    ``selected_assessment`` its own prior-art assessment."""

    selection_run: SelectionRunRecord
    selection_directive: SelectionDirective
    prior_art_run: PriorArtRunRecord
    ideation_run: IdeationRunRecord
    direction: DirectionRecord
    snapshot: CfpSnapshot
    candidates: tuple[CandidateIdea, ...]
    assessments: tuple[PriorArtAssessment, ...]
    partition: EligibilityPartition
    selected: CandidateIdea
    selected_assessment: PriorArtAssessment


def require_selected_candidate_for_admission(
    selection_store: SelectionStore,
    prior_art_store: PriorArtStore,
    ideation_store: IdeationStore,
    record_id: str,
) -> AdmissionInputs:
    """The single entrance to an admission: one named selection run
    record whose outcome is ``SELECTED`` and whose complete lineage
    loads, cross-checks, and recomputes. Every refusal happens before
    any model call and names what failed."""
    run = selection_store.get_run(record_id)
    if run is None:
        raise AdmissionRefusedError(
            f"no selection run record {record_id} in this store; an "
            f"admission enters through a durable selection"
        )
    if run.outcome is not SelectionOutcome.SELECTED:
        raise AdmissionRefusedError(
            f"selection run {record_id} ended {run.outcome.value}; an "
            f"honest stop is a settled outcome, not an admissible "
            f"selection"
        )
    directive = selection_store.get_directive(run.directive_id)
    if directive is None:
        raise AdmissionRefusedError(
            f"selection directive {run.directive_id}, named by run "
            f"{record_id}, is not in this store; the frozen resource "
            f"constraints are part of the admitted lineage"
        )
    try:
        inputs = require_challenged_portfolio_for_selection(
            prior_art_store, ideation_store, run.prior_art_run_record_id
        )
    except MissingChallengedPortfolioError as exc:
        raise AdmissionRefusedError(
            f"admission of selection run {record_id} refused: {exc}"
        ) from exc

    prior_art_run = inputs.prior_art_run
    ideation_run = inputs.ideation_run
    direction = inputs.direction
    for label, stamped, loaded in (
        ("prior_art_run_id", run.prior_art_run_id, prior_art_run.run_id),
        (
            "ideation_run_record_id",
            run.ideation_run_record_id,
            ideation_run.id,
        ),
        ("ideation_run_id", run.ideation_run_id, ideation_run.run_id),
        ("direction_id", run.direction_id, direction.id),
        ("snapshot_id", prior_art_run.snapshot_id, ideation_run.snapshot_id),
        ("map_run_id", prior_art_run.map_run_id, ideation_run.map_run_id),
        (
            "map_assessment_id",
            prior_art_run.assessment_id,
            ideation_run.assessment_id,
        ),
        ("direction.run_id", direction.run_id, ideation_run.run_id),
    ):
        if stamped != loaded:
            raise AdmissionRefusedError(
                f"selection run {record_id} names {label} {stamped}, but "
                f"the loaded lineage carries {loaded}; a lineage that "
                f"disagrees with itself cannot be admitted"
            )
    if tuple(run.candidate_ids) != tuple(prior_art_run.candidate_ids) or (
        tuple(run.prior_art_assessment_ids)
        != tuple(prior_art_run.prior_art_assessment_ids)
    ):
        raise AdmissionRefusedError(
            f"selection run {record_id} names a different portfolio than "
            f"prior-art run {prior_art_run.id}; a selection over one "
            f"portfolio cannot admit from another"
        )

    partition = partition_by_verdict(inputs)
    recomputed = tuple(candidate.id for candidate in partition.eligible)
    if tuple(run.eligible_candidate_ids) != recomputed:
        raise AdmissionRefusedError(
            f"selection run {record_id} stamps eligible "
            f"{list(run.eligible_candidate_ids)}, but the named prior-art "
            f"run's verdicts recompute to {list(recomputed)}; eligibility "
            f"is computed, never copied"
        )

    decision = run.decision
    assert decision is not None  # structural: SELECTED carries a decision
    selected = decision.selected_candidate_id
    selected_candidate: CandidateIdea | None = None
    selected_assessment: PriorArtAssessment | None = None
    for candidate, assessment in zip(
        inputs.candidates, inputs.assessments, strict=True
    ):
        if candidate.id == selected:
            selected_candidate = candidate
            selected_assessment = assessment
    if selected_candidate is None or selected_assessment is None:
        raise AdmissionRefusedError(
            f"selection run {record_id} selected {selected}, which the "
            f"reloaded portfolio does not contain"
        )
    if selected_candidate.run_id != ideation_run.run_id:
        raise AdmissionRefusedError(
            f"candidate {selected} belongs to ideation run "
            f"{selected_candidate.run_id}, not {ideation_run.run_id}; a "
            f"candidate from another run cannot be admitted"
        )

    snapshot = ideation_store.get_snapshot(direction.snapshot_id)
    if snapshot is None:
        raise AdmissionRefusedError(
            f"CFP snapshot {direction.snapshot_id} is not in this store; "
            f"without it the lineage does not reach the call that "
            f"governed the direction"
        )

    return AdmissionInputs(
        selection_run=run,
        selection_directive=directive,
        prior_art_run=prior_art_run,
        ideation_run=ideation_run,
        direction=direction,
        snapshot=snapshot,
        candidates=inputs.candidates,
        assessments=inputs.assessments,
        partition=partition,
        selected=selected_candidate,
        selected_assessment=selected_assessment,
    )
