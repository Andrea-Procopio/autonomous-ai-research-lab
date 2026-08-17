"""Pure domain types.

This package has no dependencies on any sibling package -- orchestration,
roles, search, execution, evidence storage and knowledge all depend on ``core``
and never the reverse. Keeping the dependency graph a DAG rooted here is what
allows the scientific vocabulary to stay stable while the machinery around it
is replaced.
"""

from .actions import ResearchAction, ResearchActionType
from .budget import InsufficientBudgetError, ResearchBudget, ResourceCost
from .claim import Claim, ClaimStatus, EvidenceLink, EvidenceRelation
from .evidence import Evidence, EvidenceKind
from .experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    ResultRef,
)
from .hypothesis import Hypothesis, HypothesisStatus
from .ids import content_id
from .question import QuestionStatus, ResearchQuestion
from .state import ResearchState
from .types import ConfigValue

__all__ = [
    "Claim",
    "ClaimStatus",
    "ConfigValue",
    "Environment",
    "Evidence",
    "EvidenceKind",
    "EvidenceLink",
    "EvidenceRelation",
    "ExperimentResult",
    "ExperimentSpec",
    "ExperimentStatus",
    "Hypothesis",
    "HypothesisStatus",
    "InsufficientBudgetError",
    "QuestionStatus",
    "ResearchAction",
    "ResearchActionType",
    "ResearchBudget",
    "ResearchQuestion",
    "ResearchState",
    "ResourceCost",
    "ResultRef",
    "content_id",
]
