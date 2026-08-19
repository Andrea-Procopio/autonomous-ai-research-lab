"""The selection records: everything a selection run may durably claim.

A selection is a comparative judgment over one challenged portfolio —
never a fact about which idea is objectively best, never a novelty
certificate, and never scientific state. Nothing here is ``Evidence``, a
``Hypothesis``, or any other scientific-state proposition; nothing here
touches the candidate records it judges — the portfolio stays immutable,
its ``NoveltyStatus`` stays ``UNASSESSED``, and admission to research
state remains a later task behind the governed commit.

The authority split is structural. Trusted code decides validity: the
eligible set (``DISTINGUISHED`` in the one named prior-art run, nothing
else), the disqualified set, the exact partition, the outcome's legality,
and every count. The model decides only which non-disqualified eligible
candidate it prefers, and that preference is labeled
``comparative_preference`` in :data:`CLAIM_KINDS` — an artifact-grounded
preference validated, never computed, by trusted code. No record here
carries a numeric score of any kind: score-free justification is
structural, not a gate's opinion.

Identity follows the house rules: run ids are occurrences, every stored
record is content-addressed over all of its fields, provider provenance
and spend included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from typing import Final

from ..core.ids import content_id
from ..mapping.records import CallProvenance
from ..priorart.assessment import PriorArtReason, PriorArtVerdict

#: The structural epistemic label of each model-authored record category.
#: Everything else on these records — the eligible and disqualified sets,
#: the partition, the outcome, and the spend — is trusted-code
#: computation and carries no model authorship at all.
CLAIM_KINDS: Final = {
    "review.prior_art_verdict": "record_restatement",
    "review.scientific_importance": "candidate_grounded_judgment",
    "review.falsifiability_in_practice": "candidate_grounded_judgment",
    "review.diagnosticity": "candidate_grounded_judgment",
    "review.expected_information_gain": "candidate_grounded_judgment",
    "review.evaluation_quality": "candidate_grounded_judgment",
    "review.feasibility_within_directive": "candidate_grounded_judgment",
    "review.prior_art_differentiation": "candidate_grounded_judgment",
    "review.cfp_relevance": "candidate_grounded_judgment",
    "review.execution_risk": "candidate_grounded_judgment",
    "review.portfolio_redundancy": "candidate_grounded_judgment",
    "review.disqualifier.ground": "candidate_grounded_judgment",
    "review.disqualifier.candidate_text": "record_quotation",
    "review.disqualifier.constraint_text": "record_quotation",
    "review.disqualifier.why_unrepairable": "candidate_grounded_judgment",
    "pair.comparison": "candidate_grounded_judgment",
    "decision.selected_candidate_id": "comparative_preference",
    "decision.decisive_tradeoff": "comparative_preference",
    "decision.why_selected_over": "comparative_preference",
    "decision.first_experimental_objective": "design_target",
    "decision.required_capabilities": "design_target",
    "decision.residual_risks": "candidate_grounded_judgment",
}

REVIEW_FIELDS: Final = (
    "scientific_importance",
    "falsifiability_in_practice",
    "diagnosticity",
    "expected_information_gain",
    "evaluation_quality",
    "feasibility_within_directive",
    "prior_art_differentiation",
    "cfp_relevance",
    "execution_risk",
    "portfolio_redundancy",
)
"""The ten judgment fields every comparative review covers, in prose.
The stage schema is built from this tuple, so the schema and the record
cannot drift apart."""


class SelectionOutcome(StrEnum):
    """How one selection run ended. The two stops are structurally
    distinct: ``NO_ELIGIBLE_CANDIDATE`` is decided by trusted code from
    the named run's verdicts before any model call, while
    ``NO_DEFENSIBLE_CANDIDATE`` needs a validated hard disqualifier for
    every eligible candidate. Neither is ever conflated with the other,
    and no fourth outcome exists."""

    SELECTED = "selected"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"
    NO_DEFENSIBLE_CANDIDATE = "no_defensible_candidate"


class DisqualificationGround(StrEnum):
    """The only grounds on which an eligible candidate may be
    disqualified. Narrow on purpose: weakness relative to another
    candidate, uncertainty between close candidates, current repository
    limitations, and implementation difficulty are judgments for the
    review fields, never disqualifiers.

    Candidates that survived the ideation gates carry falsifiers,
    metrics, baselines, and CFP alignment as construction invariants, so
    every ground but ``RESOURCES_EXCEED_DIRECTIVE`` is a near-unreachable
    fail-closed guard — kept expressible so a real defect against the
    directive's stated envelope still has a name."""

    NOT_FALSIFIABLE_IN_PRACTICE = "not_falsifiable_in_practice"
    RESOURCES_EXCEED_DIRECTIVE = "resources_exceed_directive"
    OUTCOME_NOT_MEASURABLE = "outcome_not_measurable"
    NO_CREDIBLE_BASELINE = "no_credible_baseline"
    OUTSIDE_CFP_SCOPE = "outside_cfp_scope"


class DisqualifierDimension(StrEnum):
    """Which constraint text a disqualifier quotes: one of the
    directive's four statements, or the governing call's recorded
    direction for scope. The gate re-finds ``constraint_text`` in
    exactly the named haystack."""

    COMPUTE = "compute"
    DATA = "data"
    TIME = "time"
    EXPERIMENTAL = "experimental"
    SCOPE = "scope"


GROUND_DIMENSIONS: Final[
    dict[DisqualificationGround, frozenset[DisqualifierDimension]]
] = {
    DisqualificationGround.NOT_FALSIFIABLE_IN_PRACTICE: frozenset(
        {
            DisqualifierDimension.COMPUTE,
            DisqualifierDimension.DATA,
            DisqualifierDimension.TIME,
            DisqualifierDimension.EXPERIMENTAL,
        }
    ),
    DisqualificationGround.RESOURCES_EXCEED_DIRECTIVE: frozenset(
        {
            DisqualifierDimension.COMPUTE,
            DisqualifierDimension.DATA,
            DisqualifierDimension.TIME,
            DisqualifierDimension.EXPERIMENTAL,
        }
    ),
    DisqualificationGround.OUTCOME_NOT_MEASURABLE: frozenset(
        {DisqualifierDimension.EXPERIMENTAL}
    ),
    DisqualificationGround.NO_CREDIBLE_BASELINE: frozenset(
        {DisqualifierDimension.EXPERIMENTAL, DisqualifierDimension.DATA}
    ),
    DisqualificationGround.OUTSIDE_CFP_SCOPE: frozenset(
        {DisqualifierDimension.SCOPE}
    ),
}
"""Which dimensions can carry each ground: a scope objection quotes the
direction, a resource objection quotes a resource statement. A
mismatched pair is unconstructible."""


@dataclass(frozen=True, slots=True)
class HardDisqualifier:
    """One attested account of why one eligible candidate cannot be
    pursued under this directive — the overlap-hypothesis discipline
    applied to resources. Both text ends are held to recorded text by
    the gate: ``candidate_text`` is re-found verbatim in the candidate's
    own rendered record, ``constraint_text`` in the haystack the
    dimension names. Without validation a disqualifier is a gate
    rejection, never a stop."""

    ground: DisqualificationGround
    dimension: DisqualifierDimension
    candidate_text: str
    constraint_text: str
    why_unrepairable: str
    """Why no repair short of changing the candidate resolves the
    conflict — the model's own claim, held to the strict text rules."""

    def __post_init__(self) -> None:
        for label, value in (
            ("candidate_text", self.candidate_text),
            ("constraint_text", self.constraint_text),
            ("why_unrepairable", self.why_unrepairable),
        ):
            if not value.strip():
                raise ValueError(f"a disqualifier requires {label}")
        if self.dimension not in GROUND_DIMENSIONS[self.ground]:
            raise ValueError(
                f"a {self.ground.value} disqualifier cannot quote the "
                f"{self.dimension.value} constraint"
            )


@dataclass(frozen=True, slots=True)
class CandidateReview:
    """One candidate's comparative review: the model's restatement of
    the recorded prior-art verdict (gate-held to the assessment), ten
    prose judgments, and any attested disqualifiers. Prose only — there
    is nowhere to put a score."""

    candidate_id: str
    prior_art_verdict: PriorArtVerdict
    scientific_importance: str
    falsifiability_in_practice: str
    diagnosticity: str
    expected_information_gain: str
    evaluation_quality: str
    feasibility_within_directive: str
    prior_art_differentiation: str
    cfp_relevance: str
    execution_risk: str
    portfolio_redundancy: str
    disqualifiers: tuple[HardDisqualifier, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("a review names its candidate")
        for name in REVIEW_FIELDS:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"a review requires {name}")
        grounds = [entry.ground for entry in self.disqualifiers]
        if len(set(grounds)) != len(grounds):
            raise ValueError(
                "a review carries at most one disqualifier per ground"
            )


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    """One explicit comparison of two eligible candidates. Trusted code
    stamps the canonical order, so a reversed duplicate of the same pair
    is unconstructible."""

    first_candidate_id: str
    second_candidate_id: str
    comparison: str

    def __post_init__(self) -> None:
        if self.first_candidate_id == self.second_candidate_id:
            raise ValueError("a pair compares two distinct candidates")
        if not self.first_candidate_id < self.second_candidate_id:
            raise ValueError(
                "pairs are stored in canonical id order; trusted code "
                "stamps the order, never the model"
            )
        if not self.comparison.strip():
            raise ValueError("a pair requires an explicit comparison")


@dataclass(frozen=True, slots=True)
class IneligibleCandidate:
    """Trusted code's account of why one candidate never entered the
    review: its verdict in the named run, with that assessment's own
    grounded specifics copied forward so the stop explains itself. The
    assessment invariants are re-enforced on the copy."""

    candidate_id: str
    assessment_id: str
    verdict: PriorArtVerdict
    reasons: tuple[PriorArtReason, ...]
    overlapping_work_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.verdict is PriorArtVerdict.DISTINGUISHED:
            raise ValueError(
                "a distinguished candidate is eligible, not ineligible"
            )
        if (self.verdict is PriorArtVerdict.OVERLAPPING) != bool(
            self.overlapping_work_ids
        ):
            raise ValueError(
                "an OVERLAPPING verdict and a grounded overlapping work "
                "imply each other"
            )
        if (
            self.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
            and not self.reasons
        ):
            raise ValueError("an unresolved verdict names why")


@dataclass(frozen=True, slots=True)
class SelectionRationale:
    """Why the winner beats one specific alternative."""

    candidate_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("a rationale names the alternative")
        if not self.reason.strip():
            raise ValueError("a rationale states its reason")


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """The model's preference among the non-disqualified eligible
    candidates, with the decisive tradeoff, an entry against every
    alternative, and what pursuing the winner first requires."""

    selected_candidate_id: str
    decisive_tradeoff: str
    why_selected_over: tuple[SelectionRationale, ...]
    first_experimental_objective: str
    required_capabilities: tuple[str, ...]
    residual_risks: tuple[str, ...]
    provenance: CallProvenance

    def __post_init__(self) -> None:
        for label, value in (
            ("selected_candidate_id", self.selected_candidate_id),
            ("decisive_tradeoff", self.decisive_tradeoff),
            (
                "first_experimental_objective",
                self.first_experimental_objective,
            ),
        ):
            if not value.strip():
                raise ValueError(f"a decision requires {label}")
        for label, items in (
            ("required_capabilities", self.required_capabilities),
            ("residual_risks", self.residual_risks),
        ):
            if not items:
                raise ValueError(f"a decision requires {label}")
            if any(not entry.strip() for entry in items):
                raise ValueError(f"{label} entries must be non-empty")
            if len(set(items)) != len(items):
                raise ValueError(f"{label} lists each entry once")
        others = [entry.candidate_id for entry in self.why_selected_over]
        if len(set(others)) != len(others):
            raise ValueError("each alternative is argued against once")
        if self.selected_candidate_id in others:
            raise ValueError(
                "the winner is not an alternative to itself"
            )


@dataclass(frozen=True, slots=True)
class SelectionRunRecord:
    """The completed selection: what was asked, the trusted-code-stamped
    partition, everything the model judged, the outcome, and the spend —
    one record per run, written once, after the outcome is settled.

    The outcome shapes are structural. ``NO_ELIGIBLE_CANDIDATE`` exists
    exactly when the named run distinguished nothing: zero reviews, zero
    pairs, zero calls, zero spend, and every ineligible candidate named
    with its verdict. ``NO_DEFENSIBLE_CANDIDATE`` exists exactly when
    every eligible candidate's review carries a validated disqualifier —
    a stop with a defensible candidate remaining cannot be recorded,
    whatever code path produced it. ``SELECTED`` exists exactly when one
    non-disqualified eligible candidate carries the decision, argued
    against every other contender."""

    run_id: str
    directive_id: str
    prior_art_run_record_id: str
    prior_art_run_id: str
    ideation_run_record_id: str
    ideation_run_id: str
    direction_id: str
    """The gated CFP reading whose rendered text is the scope
    disqualifier's haystack; pinned so the exact haystack reloads."""

    candidate_ids: tuple[str, ...]
    prior_art_assessment_ids: tuple[str, ...]
    eligible_candidate_ids: tuple[str, ...]
    ineligible: tuple[IneligibleCandidate, ...]
    disqualified_candidate_ids: tuple[str, ...]
    reviews: tuple[CandidateReview, ...]
    pairwise_comparisons: tuple[PairwiseComparison, ...]
    review_provenance: CallProvenance | None
    outcome: SelectionOutcome
    decision: SelectionDecision | None
    model_calls: int
    input_tokens: int
    output_tokens: int
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.candidate_ids:
            raise ValueError(
                "a selection run names the challenged portfolio; a "
                "refusal portfolio never enters"
            )
        if len(self.prior_art_assessment_ids) != len(self.candidate_ids):
            raise ValueError(
                "every candidate carries exactly one assessment, in "
                "record order"
            )
        for label, items in (
            ("candidate_ids", self.candidate_ids),
            ("prior_art_assessment_ids", self.prior_art_assessment_ids),
            ("eligible_candidate_ids", self.eligible_candidate_ids),
            ("disqualified_candidate_ids", self.disqualified_candidate_ids),
        ):
            if len(set(items)) != len(items):
                raise ValueError(f"{label} lists each id once")
        eligible = set(self.eligible_candidate_ids)
        ineligible_ids = [entry.candidate_id for entry in self.ineligible]
        if len(set(ineligible_ids)) != len(ineligible_ids):
            raise ValueError("ineligible lists each candidate once")
        if eligible | set(ineligible_ids) != set(self.candidate_ids) or (
            eligible & set(ineligible_ids)
        ):
            raise ValueError(
                "eligible and ineligible candidates must partition the "
                "portfolio exactly"
            )
        disqualified = set(self.disqualified_candidate_ids)
        if not disqualified <= eligible:
            raise ValueError(
                "only an eligible candidate can be disqualified"
            )
        carrying = {
            review.candidate_id
            for review in self.reviews
            if review.disqualifiers
        }
        if disqualified != carrying:
            raise ValueError(
                "the disqualified set must be exactly the candidates "
                "whose reviews carry validated disqualifiers"
            )
        for label, count in (
            ("model_calls", self.model_calls),
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if count < 0:
                raise ValueError(f"{label} cannot be negative")
        if self.outcome is SelectionOutcome.NO_ELIGIBLE_CANDIDATE:
            self._check_no_eligible()
        else:
            self._check_reviewed(eligible, disqualified)
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "srun",
                    self.run_id,
                    self.directive_id,
                    self.prior_art_run_record_id,
                    self.prior_art_run_id,
                    self.ideation_run_record_id,
                    self.ideation_run_id,
                    self.direction_id,
                    self.candidate_ids,
                    self.prior_art_assessment_ids,
                    self.eligible_candidate_ids,
                    tuple(_ineligible_key(e) for e in self.ineligible),
                    self.disqualified_candidate_ids,
                    tuple(_review_key(r) for r in self.reviews),
                    tuple(_pair_key(p) for p in self.pairwise_comparisons),
                    (
                        self.review_provenance.response_id
                        if self.review_provenance is not None
                        else ""
                    ),
                    self.outcome,
                    _decision_key(self.decision),
                    self.model_calls,
                    self.input_tokens,
                    self.output_tokens,
                ),
            )

    def _check_no_eligible(self) -> None:
        if self.eligible_candidate_ids:
            raise ValueError(
                "an eligible candidate contradicts NO_ELIGIBLE_CANDIDATE"
            )
        if self.reviews or self.pairwise_comparisons:
            raise ValueError(
                "nothing was reviewable, so nothing can carry a review"
            )
        if self.review_provenance is not None or self.decision is not None:
            raise ValueError(
                "an ineligible portfolio is settled by trusted code "
                "alone; no model call can have happened"
            )
        if self.model_calls or self.input_tokens or self.output_tokens:
            raise ValueError(
                "a run that never called the model cannot have spent"
            )

    def _check_reviewed(
        self, eligible: set[str], disqualified: set[str]
    ) -> None:
        reviewed = [review.candidate_id for review in self.reviews]
        if len(set(reviewed)) != len(reviewed) or set(reviewed) != eligible:
            raise ValueError(
                "reviews must cover every eligible candidate exactly once"
            )
        expected_pairs = {
            pair for pair in combinations(sorted(eligible), 2)
        }
        recorded_pairs = {
            (pair.first_candidate_id, pair.second_candidate_id)
            for pair in self.pairwise_comparisons
        }
        if (
            len(recorded_pairs) != len(self.pairwise_comparisons)
            or recorded_pairs != expected_pairs
        ):
            raise ValueError(
                "pairwise comparisons must cover every eligible pair "
                "exactly once"
            )
        if self.review_provenance is None:
            raise ValueError("a reviewed run records its call provenance")
        if self.outcome is SelectionOutcome.NO_DEFENSIBLE_CANDIDATE:
            if disqualified != eligible:
                raise ValueError(
                    "NO_DEFENSIBLE_CANDIDATE requires a validated "
                    "disqualifier for every eligible candidate; a "
                    "defensible candidate remains"
                )
            if self.decision is not None:
                raise ValueError("an honest stop carries no decision")
            if self.model_calls < 1:
                raise ValueError(
                    "a reviewed stop spent at least the review call"
                )
            return
        if self.decision is None:
            raise ValueError("a SELECTED outcome carries the decision")
        winner = self.decision.selected_candidate_id
        contenders = eligible - disqualified
        if winner not in contenders:
            raise ValueError(
                "the winner must be an eligible candidate no validated "
                "disqualifier removed"
            )
        others = {
            entry.candidate_id for entry in self.decision.why_selected_over
        }
        if others != contenders - {winner}:
            raise ValueError(
                "the decision argues against exactly the other "
                "contenders"
            )
        if self.model_calls < 2:
            raise ValueError("a selection spent both gated stages")


def _disqualifier_key(entry: HardDisqualifier) -> tuple[object, ...]:
    return (
        entry.ground,
        entry.dimension,
        entry.candidate_text,
        entry.constraint_text,
        entry.why_unrepairable,
    )


def _review_key(review: CandidateReview) -> tuple[object, ...]:
    return (
        review.candidate_id,
        review.prior_art_verdict,
        *(getattr(review, name) for name in REVIEW_FIELDS),
        tuple(_disqualifier_key(entry) for entry in review.disqualifiers),
    )


def _pair_key(pair: PairwiseComparison) -> tuple[object, ...]:
    return (
        pair.first_candidate_id,
        pair.second_candidate_id,
        pair.comparison,
    )


def _ineligible_key(entry: IneligibleCandidate) -> tuple[object, ...]:
    return (
        entry.candidate_id,
        entry.assessment_id,
        entry.verdict,
        tuple((reason.code, reason.detail) for reason in entry.reasons),
        entry.overlapping_work_ids,
    )


def _decision_key(decision: SelectionDecision | None) -> tuple[object, ...]:
    if decision is None:
        return ()
    return (
        decision.selected_candidate_id,
        decision.decisive_tradeoff,
        tuple(
            (entry.candidate_id, entry.reason)
            for entry in decision.why_selected_over
        ),
        decision.first_experimental_objective,
        decision.required_capabilities,
        decision.residual_risks,
        decision.provenance.response_id,
    )
