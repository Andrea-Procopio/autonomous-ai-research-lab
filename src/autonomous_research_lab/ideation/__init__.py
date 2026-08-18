"""Candidate idea generation over the assessed literature map (Task 5C).

The chain: a durable ``MapAdequacyAssessment`` and a supplied CFP
snapshot enter through :func:`require_adequate_for_idea_generation` (the
mapping package's one door), one gated call reads the call text into a
structured direction, one gated call proposes a bounded candidate
portfolio, and trusted code stamps everything a candidate carries beyond
its own words — problem statements, kinds, and support tiers; theme names
and eras; the cited sources' era mix; and the structurally ``UNASSESSED``
novelty status. Honest refusal is a first-class outcome: a grounded
justification for zero candidates is recorded, not retried.

Candidate ideas are conjectures, not scientific state: nothing here
creates ``Evidence``, ``ResearchQuestion``, ``Hypothesis``, or any other
scientific-state proposition, and nothing here can reach
``ResearchState`` — both directions pinned by the structural tests.
Admission through the governed commit, like novelty assessment, belongs
to later tasks reading these durable records.
"""

from ..mapping.gates import MappingRejection
from .direction import (
    MAX_SNAPSHOT_CHARS,
    CfpSnapshot,
    DirectionRecord,
)
from .directive import (
    MAX_CANDIDATES_CEILING,
    MODEL_CALLS_CEILING,
    IdeationDirective,
)
from .gates import (
    NOVELTY_PHRASES,
    check_candidates,
    check_direction,
    check_novelty_language,
    claim_text_of,
)
from .records import (
    CLAIM_KINDS,
    AddressedProblem,
    CandidateIdea,
    DataRequirement,
    DataStatus,
    IdeationRunRecord,
    NoveltyStatus,
    PortfolioReport,
    Prediction,
    ResourceEstimate,
    TargetedTheme,
    problem_key,
    theme_key,
)
from .store import (
    IdeationConflictError,
    IdeationIntegrityError,
    IdeationStore,
)

__all__ = [
    "CLAIM_KINDS",
    "MAX_CANDIDATES_CEILING",
    "MAX_SNAPSHOT_CHARS",
    "MODEL_CALLS_CEILING",
    "NOVELTY_PHRASES",
    "AddressedProblem",
    "CandidateIdea",
    "CfpSnapshot",
    "DataRequirement",
    "DataStatus",
    "DirectionRecord",
    "IdeationConflictError",
    "IdeationDirective",
    "IdeationIntegrityError",
    "IdeationRunRecord",
    "IdeationStore",
    "MappingRejection",
    "NoveltyStatus",
    "PortfolioReport",
    "Prediction",
    "ResourceEstimate",
    "TargetedTheme",
    "check_candidates",
    "check_direction",
    "check_novelty_language",
    "claim_text_of",
    "problem_key",
    "theme_key",
]
