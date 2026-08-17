"""Autonomous AI Research Lab.

Infrastructure for autonomous systems that conduct scientific research, built
around one commitment: the system optimises for reliable scientific knowledge,
not for output that looks like knowledge.

Package layout (dependencies point downward only):

``core``
    Domain vocabulary: state, actions, hypotheses, experiments, evidence,
    claims, budgets. Depends on nothing else.
``evidence``
    Append-only storage of what actually happened.
``execution``
    Turning an experiment design into a running process, anywhere.
``knowledge``
    Read models over accumulated results -- today, the claim-evidence graph.
``search``
    Policies over scientific states and actions.
``roles``
    Specialized agents with their own objectives and authority.
``orchestration``
    Choosing the next action, and who performs it.
``publication``
    Reporting. Deliberately empty for now.
"""

from .core import (
    Claim,
    ClaimStatus,
    Evidence,
    EvidenceKind,
    EvidenceLink,
    EvidenceRelation,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    Hypothesis,
    HypothesisStatus,
    ResearchAction,
    ResearchActionType,
    ResearchBudget,
    ResearchQuestion,
    ResearchState,
    ResourceCost,
    ResultRef,
)

__all__ = [
    "Claim",
    "ClaimStatus",
    "Evidence",
    "EvidenceKind",
    "EvidenceLink",
    "EvidenceRelation",
    "ExperimentResult",
    "ExperimentSpec",
    "ExperimentStatus",
    "Hypothesis",
    "HypothesisStatus",
    "ResearchAction",
    "ResearchActionType",
    "ResearchBudget",
    "ResearchQuestion",
    "ResearchState",
    "ResourceCost",
    "ResultRef",
    "__version__",
]

__version__ = "0.0.1"
