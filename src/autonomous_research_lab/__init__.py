"""Autonomous AI Research Lab.

Infrastructure for autonomous systems that conduct scientific research, built
around one commitment: the system optimises for reliable scientific knowledge,
not for output that looks like knowledge.

Package layout (dependencies point downward only):

``core``
    Domain vocabulary: state, actions and their attempts, hypotheses,
    predictions, experiments, evidence, claims, assessments, proposals,
    decisions, budgets. Depends on nothing else.
``evidence``
    Append-only storage of what actually happened.
``execution``
    Turning an experiment design into a running process, anywhere.
``knowledge``
    Read models over accumulated results — today, the claim-evidence graph.
``search``
    Selection policies over evaluated candidates.
``roles``
    Specialized agents with their own objectives and authority. Roles produce
    proposals; they never mutate state.
``orchestration``
    Candidate generation, utility evaluation, decision wiring, the transition
    layer that commits proposals, and trajectory logging.
``publication``
    Reporting. Deliberately empty for now.
"""

from .core import (
    ActionAttempt,
    ActionCandidate,
    ActionOutcome,
    ActionUtility,
    AssessmentVerdict,
    AttemptStatus,
    Claim,
    Comparator,
    DecisionRecord,
    EpistemicAssessment,
    EvaluatedCandidate,
    Evidence,
    EvidenceKind,
    EvidenceLink,
    EvidenceRelation,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    Hypothesis,
    HypothesisStatus,
    Prediction,
    PredictionStatus,
    Proposal,
    ResearchAction,
    ResearchActionType,
    ResearchBudget,
    ResearchQuestion,
    ResearchState,
    ResourceCost,
    ResultRef,
)

__all__ = [
    "ActionAttempt",
    "ActionCandidate",
    "ActionOutcome",
    "ActionUtility",
    "AssessmentVerdict",
    "AttemptStatus",
    "Claim",
    "Comparator",
    "DecisionRecord",
    "EpistemicAssessment",
    "EvaluatedCandidate",
    "Evidence",
    "EvidenceKind",
    "EvidenceLink",
    "EvidenceRelation",
    "ExperimentResult",
    "ExperimentSpec",
    "ExperimentStatus",
    "Hypothesis",
    "HypothesisStatus",
    "Prediction",
    "PredictionStatus",
    "Proposal",
    "ResearchAction",
    "ResearchActionType",
    "ResearchBudget",
    "ResearchQuestion",
    "ResearchState",
    "ResourceCost",
    "ResultRef",
    "__version__",
]

__version__ = "0.0.2"
