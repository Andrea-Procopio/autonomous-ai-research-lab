"""The one door into a selection run, and the trusted partition behind it.

:func:`require_challenged_portfolio_for_selection` mirrors
:func:`~..priorart.assessment.require_candidates_for_prior_art` one stage
down: a selection run enters through a durable prior-art run record whose
assessments and candidates all load and cross-check, or it does not
start. There is no "latest assessment" inference anywhere in this
package — no record carries a clock that could define "latest" — so the
directive names one run, and an assessment outside that run does not
exist for this selection. That is what makes staleness well-defined.

:func:`partition_by_verdict` is trusted code all the way down: a
candidate is eligible exactly when its assessment in the named run is
``DISTINGUISHED``. Eligibility means the candidate may be selected — it
does not guarantee selection and never claims absolute novelty. Every
other candidate is copied into an :class:`~.records.IneligibleCandidate`
with its own verdict's grounded specifics, so an empty eligible set
explains itself without a model call.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ideation.direction import DirectionRecord
from ..ideation.records import CandidateIdea, IdeationRunRecord
from ..ideation.store import IdeationStore
from ..priorart.assessment import PriorArtAssessment, PriorArtVerdict
from ..priorart.records import PriorArtRunRecord
from ..priorart.store import PriorArtStore
from .records import IneligibleCandidate


class MissingChallengedPortfolioError(RuntimeError):
    """The one refusal the door raises: no durable prior-art run record,
    an unloadable assessment, candidate, or direction, or an assessment
    that does not belong to the named run."""


@dataclass(frozen=True, slots=True)
class SelectionInputs:
    """Everything the door loaded and cross-checked, in record order.
    ``candidates[i]`` carries ``assessments[i]``."""

    prior_art_run: PriorArtRunRecord
    ideation_run: IdeationRunRecord
    direction: DirectionRecord
    candidates: tuple[CandidateIdea, ...]
    assessments: tuple[PriorArtAssessment, ...]


@dataclass(frozen=True, slots=True)
class EligibilityPartition:
    """Trusted code's split of the portfolio: the candidates a selection
    may choose among, their assessments in the same order, and every
    other candidate with the verdict that kept it out."""

    eligible: tuple[CandidateIdea, ...]
    eligible_assessments: tuple[PriorArtAssessment, ...]
    ineligible: tuple[IneligibleCandidate, ...]


def require_challenged_portfolio_for_selection(
    prior_art_store: PriorArtStore,
    ideation_store: IdeationStore,
    run_record_id: str,
) -> SelectionInputs:
    """The single entrance to a selection run: a durable prior-art run
    record whose assessments, candidates, and governing direction all
    load and cross-check. Every refusal happens before any model call
    and names the record that failed."""
    record = prior_art_store.get_run(run_record_id)
    if record is None:
        raise MissingChallengedPortfolioError(
            f"no prior-art run record {run_record_id} in this store; a "
            f"selection enters through a durable challenge"
        )
    ideation_run = ideation_store.get_run(record.ideation_run_record_id)
    if ideation_run is None:
        raise MissingChallengedPortfolioError(
            f"ideation run record {record.ideation_run_record_id}, named "
            f"by challenge {run_record_id}, is not in this store; "
            f"refusing a portfolio without its lineage"
        )
    direction = ideation_store.get_direction(ideation_run.direction_id)
    if direction is None:
        raise MissingChallengedPortfolioError(
            f"direction record {ideation_run.direction_id} is not in this "
            f"store; without the governing call's recorded reading the "
            f"scope gate has no haystack"
        )
    candidates: list[CandidateIdea] = []
    assessments: list[PriorArtAssessment] = []
    for candidate_id, assessment_id in zip(
        record.candidate_ids, record.prior_art_assessment_ids, strict=True
    ):
        assessment = prior_art_store.get_prior_art_assessment(assessment_id)
        if assessment is None:
            raise MissingChallengedPortfolioError(
                f"assessment {assessment_id} named by challenge "
                f"{run_record_id} is not in this store; refusing a "
                f"partial challenge"
            )
        if (
            assessment.candidate_id != candidate_id
            or assessment.run_id != record.run_id
        ):
            raise MissingChallengedPortfolioError(
                f"assessment {assessment_id} judges candidate "
                f"{assessment.candidate_id} in run {assessment.run_id}, "
                f"not candidate {candidate_id} in run {record.run_id}; an "
                f"assessment from another run or candidate cannot enter "
                f"selection"
            )
        candidate = ideation_store.get_idea(candidate_id)
        if candidate is None:
            raise MissingChallengedPortfolioError(
                f"candidate {candidate_id} named by challenge "
                f"{run_record_id} is not in this store; refusing a "
                f"partial portfolio"
            )
        candidates.append(candidate)
        assessments.append(assessment)
    return SelectionInputs(
        prior_art_run=record,
        ideation_run=ideation_run,
        direction=direction,
        candidates=tuple(candidates),
        assessments=tuple(assessments),
    )


def partition_by_verdict(inputs: SelectionInputs) -> EligibilityPartition:
    """Split the portfolio by the named run's verdicts alone. Pure and
    deterministic: the model can argue about the eligible set later, but
    it can never author it."""
    eligible: list[CandidateIdea] = []
    eligible_assessments: list[PriorArtAssessment] = []
    ineligible: list[IneligibleCandidate] = []
    for candidate, assessment in zip(
        inputs.candidates, inputs.assessments, strict=True
    ):
        if assessment.verdict is PriorArtVerdict.DISTINGUISHED:
            eligible.append(candidate)
            eligible_assessments.append(assessment)
            continue
        ineligible.append(
            IneligibleCandidate(
                candidate_id=candidate.id,
                assessment_id=assessment.id,
                verdict=assessment.verdict,
                reasons=assessment.reasons,
                overlapping_work_ids=assessment.overlapping_work_ids,
            )
        )
    return EligibilityPartition(
        eligible=tuple(eligible),
        eligible_assessments=tuple(eligible_assessments),
        ineligible=tuple(ineligible),
    )
