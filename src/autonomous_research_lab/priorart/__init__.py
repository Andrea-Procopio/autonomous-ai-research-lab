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
from .directive import (
    COMPARED_WORKS_CEILING,
    MODEL_CALLS_CEILING,
    RESULTS_PER_QUERY_CEILING,
    SCREENED_PER_CANDIDATE_CEILING,
    PriorArtDirective,
)
from .gates import (
    QUERY_TEXT_MAX_CHARS,
    QUERY_TEXT_MIN_CHARS,
    check_comparisons,
    check_prior_art_queries,
    check_similarity_screening,
)
from .records import (
    CLAIM_KINDS,
    DIMENSIONS,
    ComparisonDimension,
    DimensionComparison,
    PriorArtCoverage,
    PriorArtQueryExecution,
    PriorArtQueryFamily,
    PriorArtRunRecord,
    PriorArtScreeningRecord,
    SimilarityDecision,
    SimilarityLabel,
    WorkComparison,
)

__all__ = [
    "CLAIM_KINDS",
    "COMPARED_WORKS_CEILING",
    "DIMENSIONS",
    "MODEL_CALLS_CEILING",
    "QUERY_TEXT_MAX_CHARS",
    "QUERY_TEXT_MIN_CHARS",
    "RESULTS_PER_QUERY_CEILING",
    "SCREENED_PER_CANDIDATE_CEILING",
    "ComparisonDimension",
    "DimensionComparison",
    "MappingRejection",
    "MissingCandidatePortfolioError",
    "PriorArtAssessment",
    "PriorArtCoverage",
    "PriorArtDirective",
    "PriorArtQueryExecution",
    "PriorArtQueryFamily",
    "PriorArtReason",
    "PriorArtReasonCode",
    "PriorArtRunRecord",
    "PriorArtScreeningRecord",
    "PriorArtThresholds",
    "PriorArtVerdict",
    "SimilarityDecision",
    "SimilarityLabel",
    "WorkComparison",
    "assess_prior_art",
    "check_comparisons",
    "check_prior_art_queries",
    "check_similarity_screening",
    "require_candidates_for_prior_art",
]
