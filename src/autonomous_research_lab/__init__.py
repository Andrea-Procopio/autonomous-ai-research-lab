"""Autonomous AI Research Lab.

Infrastructure for autonomous systems that conduct scientific research, built
around one commitment: the system optimises for reliable scientific knowledge,
not for output that looks like knowledge.

Package layout (dependencies point downward only):

``core``
    Domain vocabulary: state, actions and their attempts, questions,
    hypotheses, predictions and their tests, experiments, evidence, claims,
    assessments, proposals, commit bundles, decisions, budgets, replication
    groups. Depends on nothing else.
``evidence``
    Append-only storage of what actually happened.
``execution``
    Turning an experiment design into a running process, anywhere.
``knowledge``
    Read models over accumulated results — today, the claim-evidence graph.
``persistence``
    Content-addressed snapshots of research states on the local filesystem.
``runtime``
    The cost-aware layer: the frontier view, deterministic validation,
    reasoning tiers and escalation, runtime metrics, playbooks, and the
    development/held-out evaluation seam.
``literature``
    Bounded retrieval from a real scholarly index: snapshot records,
    deduplication, write-once search provenance, a replayable corpus.
``mapping``
    What the literature adds up to: gated field maps and problem
    inventories, judged by a trusted-code adequacy verdict.
``ideation``
    What might be worth investigating: CFP-directed, gated candidate
    generation over the assessed map.
``priorart``
    Whether it was already done: the adversarial challenge with a
    deterministic fail-closed verdict per candidate.
``selection``
    Which candidate to pursue, if any: gated choice among the
    ``DISTINGUISHED`` survivors of one named challenge, with three
    honest outcomes — a preference, never proof.
``admission``
    The governed bridge into research state: one named SELECTED
    selection, verified through its whole lineage, becomes a bare
    initial state of propositions — a translation, never a promotion.
``search``
    Selection policies over evaluated action candidates.
``roles``
    Specialized agents with their own objectives and authority. Roles receive
    explicit invocations and produce proposals; they never mutate state.
``orchestration``
    The director (single-invocation fast path and the decomposed path),
    the research runtime loop, deterministic routing, critic and synthesis
    triggers, the atomic transition layer, and trajectory logging.
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
    CommitBundle,
    Comparator,
    Consistency,
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
    Prediction,
    PredictionTest,
    Proposal,
    ProposalKind,
    ReplicationGroup,
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
    "CommitBundle",
    "Comparator",
    "Consistency",
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
    "Prediction",
    "PredictionTest",
    "Proposal",
    "ProposalKind",
    "ReplicationGroup",
    "ResearchAction",
    "ResearchActionType",
    "ResearchBudget",
    "ResearchQuestion",
    "ResearchState",
    "ResourceCost",
    "ResultRef",
    "__version__",
]

__version__ = "0.0.4"
