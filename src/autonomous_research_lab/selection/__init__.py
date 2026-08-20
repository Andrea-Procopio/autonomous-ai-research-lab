"""Candidate selection: which challenged candidate to pursue, if any.

The chain: a :class:`SelectionDirective` names one prior-art run record —
the only door in — and trusted code computes eligibility from that run's
assessments alone (``DISTINGUISHED`` and nothing else; a verdict in any
other run does not exist for this selection). Two gated model stages
follow: one comparative review of every eligible candidate and every
pair, then one final choice among the candidates no validated
disqualifier removed. One nested run record preserves everything.

Three outcomes, structurally distinct: ``SELECTED``,
``NO_ELIGIBLE_CANDIDATE`` (decided by trusted code, zero model calls,
zero spend), and ``NO_DEFENSIBLE_CANDIDATE`` (every eligible candidate
carries a validated, verbatim-attested disqualifier). A stop with a
defensible candidate remaining cannot be recorded.

The authority split: trusted code owns validity — the eligible set, the
disqualified set, the partition, the outcome's legality, and every
gate — while the model owns only which defensible candidate it prefers.
A selection is a recorded preference over a bounded, challenged
portfolio, never a fact about which idea is best and never proof of
novelty. It stays outside ``ResearchState``, the candidate records stay
untouched, and their ``NoveltyStatus`` stays ``UNASSESSED``.
"""

from .directive import (
    ELIGIBLE_CANDIDATES_CEILING,
    MAX_CONSTRAINT_CHARS,
    MODEL_CALLS_CEILING,
    SelectionDirective,
)
from .eligibility import (
    EligibilityPartition,
    MissingChallengedPortfolioError,
    SelectionInputs,
    partition_by_verdict,
    require_challenged_portfolio_for_selection,
)
from .gates import (
    MappingRejection,
    check_comparative_review,
    check_selection_decision,
)
from .preflight import (
    GATED_STAGES,
    STAGE1_BASE_OUTPUT_TOKENS,
    STAGE1_TOKENS_PER_CANDIDATE,
    STAGE1_TOKENS_PER_PAIR,
    STAGE2_BASE_OUTPUT_TOKENS,
    STAGE2_TOKENS_PER_ALTERNATIVE,
    SelectionCallPlan,
    SelectionPreflightError,
    check_selection_coherence,
)
from .records import (
    CLAIM_KINDS,
    GROUND_DIMENSIONS,
    REVIEW_FIELDS,
    CandidateReview,
    DisqualificationGround,
    DisqualifierDimension,
    HardDisqualifier,
    IneligibleCandidate,
    PairwiseComparison,
    SelectionDecision,
    SelectionOutcome,
    SelectionRationale,
    SelectionRunRecord,
)
from .selector import (
    COMPARATIVE_REVIEW_INSTRUCTION,
    COMPARATIVE_REVIEW_SCHEMA,
    SELECTION_DECISION_INSTRUCTION,
    SELECTION_DECISION_SCHEMA,
    CandidateSelector,
    SelectionBudgetError,
    SelectionRejectedError,
    SelectionRunResult,
    render_candidate_for_selection,
)
from .store import (
    SelectionConflictError,
    SelectionIntegrityError,
    SelectionStore,
)

__all__ = [
    "CLAIM_KINDS",
    "COMPARATIVE_REVIEW_INSTRUCTION",
    "COMPARATIVE_REVIEW_SCHEMA",
    "ELIGIBLE_CANDIDATES_CEILING",
    "GATED_STAGES",
    "GROUND_DIMENSIONS",
    "MAX_CONSTRAINT_CHARS",
    "MODEL_CALLS_CEILING",
    "REVIEW_FIELDS",
    "SELECTION_DECISION_INSTRUCTION",
    "SELECTION_DECISION_SCHEMA",
    "STAGE1_BASE_OUTPUT_TOKENS",
    "STAGE1_TOKENS_PER_CANDIDATE",
    "STAGE1_TOKENS_PER_PAIR",
    "STAGE2_BASE_OUTPUT_TOKENS",
    "STAGE2_TOKENS_PER_ALTERNATIVE",
    "CandidateReview",
    "CandidateSelector",
    "DisqualificationGround",
    "DisqualifierDimension",
    "EligibilityPartition",
    "HardDisqualifier",
    "IneligibleCandidate",
    "MappingRejection",
    "MissingChallengedPortfolioError",
    "PairwiseComparison",
    "SelectionBudgetError",
    "SelectionCallPlan",
    "SelectionConflictError",
    "SelectionDecision",
    "SelectionDirective",
    "SelectionInputs",
    "SelectionIntegrityError",
    "SelectionOutcome",
    "SelectionPreflightError",
    "SelectionRationale",
    "SelectionRejectedError",
    "SelectionRunRecord",
    "SelectionRunResult",
    "SelectionStore",
    "check_comparative_review",
    "check_selection_coherence",
    "check_selection_decision",
    "partition_by_verdict",
    "render_candidate_for_selection",
    "require_challenged_portfolio_for_selection",
]
