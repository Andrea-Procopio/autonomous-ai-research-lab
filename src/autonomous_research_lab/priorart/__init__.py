"""The adversarial prior-art challenge over a candidate portfolio
(Task 5D).

The chain: a durable ideation run record enters through
:func:`require_candidates_for_prior_art` (the one door), and for each
candidate the challenger makes three gated calls — proposed searches
that trusted code dates, orders, and executes through the Task 5A
corpus; similarity screens of the deduplicated pool; and nearest-work
comparisons across five explicit dimensions, each held verbatim to the
source's accessible text. Trusted code computes the coverage and the
verdict: ``OVERLAPPING`` needs one grounded substantial match,
``DISTINGUISHED`` needs adequate recorded coverage and explicit material
differences against the closest works, and everything else is honestly
``NOVELTY_UNRESOLVED`` — the fail-closed default, and a successful
outcome.

A verdict describes this bounded corpus, never the world's literature:
``DISTINGUISHED`` is not proof of novelty, absence from the corpus is
not novelty, and citation counts only order retrieval. Nothing here
creates ``Evidence`` or any other scientific-state proposition, nothing
here can reach ``ResearchState``, and nothing here touches the candidate
records it challenges — all pinned by the structural tests. Admission
through the governed commit belongs to later tasks reading these durable
records.
"""

from ..mapping.gates import MappingRejection
from .assessment import (
    MissingCandidatePortfolioError,
    PriorArtAssessment,
    PriorArtReason,
    PriorArtReasonCode,
    PriorArtThresholds,
    PriorArtVerdict,
    assess_prior_art,
    require_candidates_for_prior_art,
)
from .challenger import (
    COMPARISON_INSTRUCTION,
    COMPARISON_SCHEMA,
    METADATA_SCREENING_INSTRUCTION,
    METADATA_SCREENING_SCHEMA,
    PRIOR_ART_QUERY_SCHEMA,
    PRIOR_ART_RETRIEVAL_STRATEGIES,
    QUERY_INSTRUCTION,
    SCREENING_INSTRUCTION,
    SIMILARITY_SCREENING_SCHEMA,
    PriorArtBudgetError,
    PriorArtChallenger,
    PriorArtContractError,
    PriorArtRejectedError,
    PriorArtRunResult,
)
from .directive import (
    COMPARED_WORKS_CEILING,
    MODEL_CALLS_CEILING,
    RESULTS_PER_QUERY_CEILING,
    SCREENED_PER_CANDIDATE_CEILING,
    PriorArtDirective,
)
from .gates import (
    check_comparisons,
    check_metadata_screening,
    check_prior_art_queries,
    check_similarity_screening,
)
from .plan import (
    MAX_ALTERNATIVES_PER_GROUP,
    MAX_CONCEPT_GROUPS,
    MAX_RENDERED_CHARS,
    MAX_TERM_CHARS,
    MIN_TERM_CHARS,
    RENDERER_VERSION,
    REQUIRED_INTENTS,
    canonical_groups,
    render_query,
)
from .preflight import (
    PriorArtCallPlan,
    PriorArtPreflightError,
    check_budget_coherence,
)
from .records import (
    CLAIM_KINDS,
    DIMENSIONS,
    ComparisonDimension,
    DimensionComparison,
    OverlapHypothesis,
    PriorArtCoverage,
    PriorArtQueryExecution,
    PriorArtQueryFamily,
    PriorArtRunRecord,
    PriorArtScreeningRecord,
    SimilarityDecision,
    SimilarityLabel,
    WorkComparison,
)
from .store import (
    PriorArtConflictError,
    PriorArtIntegrityError,
    PriorArtStore,
)

__all__ = [
    "CLAIM_KINDS",
    "COMPARED_WORKS_CEILING",
    "COMPARISON_INSTRUCTION",
    "COMPARISON_SCHEMA",
    "DIMENSIONS",
    "MAX_ALTERNATIVES_PER_GROUP",
    "MAX_CONCEPT_GROUPS",
    "MAX_RENDERED_CHARS",
    "MAX_TERM_CHARS",
    "METADATA_SCREENING_INSTRUCTION",
    "METADATA_SCREENING_SCHEMA",
    "MIN_TERM_CHARS",
    "MODEL_CALLS_CEILING",
    "PRIOR_ART_QUERY_SCHEMA",
    "PRIOR_ART_RETRIEVAL_STRATEGIES",
    "QUERY_INSTRUCTION",
    "RENDERER_VERSION",
    "REQUIRED_INTENTS",
    "RESULTS_PER_QUERY_CEILING",
    "SCREENED_PER_CANDIDATE_CEILING",
    "SCREENING_INSTRUCTION",
    "SIMILARITY_SCREENING_SCHEMA",
    "ComparisonDimension",
    "DimensionComparison",
    "MappingRejection",
    "MissingCandidatePortfolioError",
    "OverlapHypothesis",
    "PriorArtAssessment",
    "PriorArtBudgetError",
    "PriorArtCallPlan",
    "PriorArtChallenger",
    "PriorArtConflictError",
    "PriorArtContractError",
    "PriorArtCoverage",
    "PriorArtDirective",
    "PriorArtIntegrityError",
    "PriorArtPreflightError",
    "PriorArtQueryExecution",
    "PriorArtQueryFamily",
    "PriorArtReason",
    "PriorArtReasonCode",
    "PriorArtRejectedError",
    "PriorArtRunRecord",
    "PriorArtRunResult",
    "PriorArtScreeningRecord",
    "PriorArtStore",
    "PriorArtThresholds",
    "PriorArtVerdict",
    "SimilarityDecision",
    "SimilarityLabel",
    "WorkComparison",
    "assess_prior_art",
    "canonical_groups",
    "check_budget_coherence",
    "check_comparisons",
    "check_metadata_screening",
    "check_prior_art_queries",
    "check_similarity_screening",
    "render_query",
    "require_candidates_for_prior_art",
]
