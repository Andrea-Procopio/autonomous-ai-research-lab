"""Pure domain types.

This package has no dependencies on any sibling package — orchestration,
roles, search, execution, evidence storage, knowledge and persistence all
depend on ``core`` and never the reverse. Keeping the dependency graph a DAG
rooted here is what allows the scientific vocabulary to stay stable while the
machinery around it is replaced. External validation libraries stay out for
the same reason: validating model output is a boundary concern, and the
boundary translates into these types rather than replacing them.
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
from .commit import CommitBundle
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
from .hypothesis import Hypothesis
from .ids import content_id, occurrence_id
from .prediction import Comparator, Consistency, Prediction, PredictionTest
from .proposals import (
    AssessmentProposal,
    ClaimProposal,
    EvidenceProposal,
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    Proposal,
    ProposalKind,
    QuestionProposal,
    ResultProposal,
    kind_of,
    payload_ids,
)
from .question import QuestionStatus, ResearchQuestion
from .replication import (
    ReplicationGroup,
    group_replications,
    replication_group_of,
)
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
    "CommitBundle",
    "Comparator",
    "ConfigValue",
    "Consistency",
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
    "InsufficientBudgetError",
    "Prediction",
    "PredictionProposal",
    "PredictionTest",
    "Proposal",
    "ProposalKind",
    "QuestionProposal",
    "QuestionStatus",
    "ReplicationGroup",
    "ResearchAction",
    "ResearchActionType",
    "ResearchBudget",
    "ResearchQuestion",
    "ResearchState",
    "ResourceCost",
    "ResultProposal",
    "ResultRef",
    "content_id",
    "group_replications",
    "kind_of",
    "occurrence_id",
    "payload_ids",
    "replication_group_of",
    "to_jsonable",
]
