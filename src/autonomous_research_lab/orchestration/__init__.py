"""Orchestration: deciding, committing, and remembering why.

* ``director`` — the reasoning seat: the single-invocation fast path
  (:class:`FrontierDirector`) and the decomposed generator/evaluator/policy
  path (:class:`ResearchDirector`), kept for the ablation;
* ``loop`` — the research runtime: one director invocation, deterministic
  routing, one role invocation, Tier-0 validation and evidence reading,
  event-triggered critique, cadenced synthesis;
* ``routing`` — static action-type → seat routing, zero model calls;
* ``critic_trigger`` — deterministic rules for when a result earns a critic;
* ``debug_loop`` — the bounded repair loop for failed executions, entered by
  failure diagnosis and never by scientific outcome;
* ``review`` — role-backed adapters for the runtime's semantic verification
  hooks (methodology review, implementation verification);
* ``synthesis`` — the slow loop's cadence and review types;
* ``candidates`` / ``evaluation`` — the decomposed path's pieces;
* ``transitions`` — the only path from a proposal to a state change;
* ``assignment`` — suitability-based routing, retained for when routing has
  calibration data behind it;
* ``trajectory`` — every decision preserved for later evaluation.
"""

from .assignment import RegistryAssigner, RoleAssigner
from .candidates import CandidateGenerator, RuleBasedCandidateGenerator
from .critic_trigger import CriticTrigger
from .debug_loop import (
    DebugAttempt,
    DebugSession,
    ExperimentDebugger,
    ImplementationRepairStrategy,
    ImplementationRepairTrigger,
    RepairKind,
    RepairProposal,
    RepairStrategy,
    ScientificOutcomeError,
    is_debuggable,
)
from .director import (
    Decision,
    Deliberation,
    FrontierDirector,
    ResearchDirector,
    RuleBasedFrontierDirector,
    ValuedCandidate,
    deliberation_record,
)
from .evaluation import HeuristicUtilityEvaluator, UtilityEvaluator
from .loop import (
    MissingRoleError,
    PromotionError,
    ResearchRuntime,
    RunOutcome,
    StepReport,
)
from .review import (
    RoleBackedImplementationVerifier,
    RoleBackedMethodologyReviewer,
)
from .routing import STATIC_ROUTES, expected_proposals, route
from .synthesis import (
    SynthesisRecommendation,
    SynthesisReview,
    SynthesisTrigger,
)
from .trajectory import JsonlTrajectoryLogger
from .transitions import TransitionError, commit, commit_bundle

__all__ = [
    "STATIC_ROUTES",
    "CandidateGenerator",
    "CriticTrigger",
    "DebugAttempt",
    "DebugSession",
    "Decision",
    "Deliberation",
    "ExperimentDebugger",
    "FrontierDirector",
    "HeuristicUtilityEvaluator",
    "ImplementationRepairStrategy",
    "ImplementationRepairTrigger",
    "JsonlTrajectoryLogger",
    "MissingRoleError",
    "PromotionError",
    "RegistryAssigner",
    "RepairKind",
    "RepairProposal",
    "RepairStrategy",
    "ResearchDirector",
    "ResearchRuntime",
    "RoleAssigner",
    "RoleBackedImplementationVerifier",
    "RoleBackedMethodologyReviewer",
    "RuleBasedCandidateGenerator",
    "RuleBasedFrontierDirector",
    "RunOutcome",
    "ScientificOutcomeError",
    "StepReport",
    "SynthesisRecommendation",
    "SynthesisReview",
    "SynthesisTrigger",
    "TransitionError",
    "UtilityEvaluator",
    "ValuedCandidate",
    "commit",
    "commit_bundle",
    "deliberation_record",
    "expected_proposals",
    "is_debuggable",
    "route",
]
