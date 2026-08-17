"""Pure domain types.

This package has no dependencies on any sibling package — orchestration,
roles, search, execution, evidence storage and knowledge all depend on ``core``
and never the reverse. Keeping the dependency graph a DAG rooted here is what
allows the scientific vocabulary to stay stable while the machinery around it
is replaced. External validation libraries stay out for the same reason:
validating model output is a boundary concern, and the boundary translates
into these types rather than replacing them.
"""

from .actions import ResearchAction, ResearchActionType
from .assessment import AssessmentVerdict, EpistemicAssessment
from .attempt import ActionAttempt, ActionOutcome, AttemptStatus
from .budget import (
    NO_COST,
    InsufficientBudgetError,
    ResearchBudget,
    ResourceCost,
)
from .claim import Claim, EvidenceLink, EvidenceRelation
from .decision import (
    ActionCandidate,
    ActionUtility,
    DecisionRecord,
    EvaluatedCandidate,
)
from .evidence import Evidence, EvidenceKind
from .experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    ResultRef,
)
from .hypothesis import Hypothesis, HypothesisStatus
from .ids import content_id, occurrence_id
from .prediction import Comparator, Prediction, PredictionStatus
from .proposals import (
    AssessmentProposal,
    ClaimProposal,
    EvidenceProposal,
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    Proposal,
    ResultProposal,
)
from .question import QuestionStatus, ResearchQuestion
from .serialize import to_jsonable
from .state import ResearchState
from .types import ConfigValue

__all__ = [
    "NO_COST",
    "ActionAttempt",
    "ActionCandidate",
    "ActionOutcome",
    "ActionUtility",
    "AssessmentProposal",
    "AssessmentVerdict",
    "AttemptStatus",
    "Claim",
    "ClaimProposal",
    "Comparator",
    "ConfigValue",
    "DecisionRecord",
    "Environment",
    "EpistemicAssessment",
    "EvaluatedCandidate",
    "Evidence",
    "EvidenceKind",
    "EvidenceLink",
    "EvidenceProposal",
    "EvidenceRelation",
    "ExperimentProposal",
    "ExperimentResult",
    "ExperimentSpec",
    "ExperimentStatus",
    "Hypothesis",
    "HypothesisProposal",
    "HypothesisStatus",
    "InsufficientBudgetError",
    "Prediction",
    "PredictionProposal",
    "PredictionStatus",
    "Proposal",
    "QuestionStatus",
    "ResearchAction",
    "ResearchActionType",
    "ResearchBudget",
    "ResearchQuestion",
    "ResearchState",
    "ResourceCost",
    "ResultProposal",
    "ResultRef",
    "content_id",
    "occurrence_id",
    "to_jsonable",
]
