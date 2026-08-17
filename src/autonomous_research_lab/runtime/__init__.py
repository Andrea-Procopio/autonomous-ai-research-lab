"""The runtime layer: rich scientific state, sparse model calls.

This package holds the pieces that keep the number of model invocations
decoupled from the richness of the domain model:

* ``config`` — typed feature flags, so optional machinery is measurable;
* ``frontier`` — the derived view a director deliberates over;
* ``validation`` — Tier-0 deterministic checks and readings of results;
* ``verification`` — experiment validity, orthogonal to scientific outcome:
  check states, positive controls, semantic review hooks, the
  negative-result gate;
* ``preflight`` — cheap deterministic checks before expensive execution;
* ``escalation`` — reasoning tiers, coarse valuations, and the rule that
  picks the cheapest sufficient tier;
* ``metrics`` — per-step resource accounting, persisted from step one;
* ``playbook`` — an advisory research prior, never a pipeline;
* ``evaluators`` — the development / held-out evaluation seam.

Everything here depends on ``core`` only.
"""

from .config import RuntimeConfig
from .escalation import (
    COMPACT_METHOD,
    CompactValuation,
    EscalationPolicy,
    EscalationSignals,
    Level,
    ReasoningTier,
    as_action_utility,
)
from .evaluators import (
    EvaluationHooks,
    HeldOutAccess,
    HeldOutAccessError,
    ObjectiveEvaluator,
)
from .frontier import (
    AdmissibilityCheck,
    Contradiction,
    ResearchFrontier,
    build_frontier,
    find_contradictions,
)
from .metrics import (
    NO_USAGE,
    JsonlRuntimeMetrics,
    MetricsSink,
    ProviderUsage,
    StepMetrics,
    UsageSource,
)
from .playbook import (
    EmpiricalMLPlaybook,
    Playbook,
    PlaybookAdvice,
    PlaybookStage,
)
from .preflight import (
    DEFAULT_PREFLIGHT_CHECKS,
    JobLike,
    PreflightCheck,
    PreflightError,
    require_preflight,
    run_preflight,
)
from .validation import (
    MANIFEST_FILENAME,
    ReplicationSummary,
    ValidationCheck,
    ValidationReport,
    evidence_from_result,
    replication_summary,
    sha256_of,
    validate_result,
    verify_artifact_integrity,
)
from .verification import (
    CheckState,
    ControlSource,
    ExperimentValidityStatus,
    ImplementationVerifier,
    MethodologyReviewer,
    OutcomeStanding,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
    VerificationReport,
    derive_validity,
    evaluate_controls,
    outcome_standing,
    verify_analysis_coverage,
)
from .verification_store import (
    FileVerificationStore,
    InMemoryVerificationStore,
    ScientificAdmissibility,
    VerificationConflictError,
    VerificationIntegrityError,
    VerificationRecord,
    VerificationStore,
)

__all__ = [
    "COMPACT_METHOD",
    "DEFAULT_PREFLIGHT_CHECKS",
    "MANIFEST_FILENAME",
    "NO_USAGE",
    "AdmissibilityCheck",
    "CheckState",
    "CompactValuation",
    "Contradiction",
    "ControlSource",
    "EmpiricalMLPlaybook",
    "EscalationPolicy",
    "EscalationSignals",
    "EvaluationHooks",
    "ExperimentValidityStatus",
    "FileVerificationStore",
    "HeldOutAccess",
    "HeldOutAccessError",
    "ImplementationVerifier",
    "InMemoryVerificationStore",
    "JobLike",
    "JsonlRuntimeMetrics",
    "Level",
    "MethodologyReviewer",
    "MetricsSink",
    "ObjectiveEvaluator",
    "OutcomeStanding",
    "Playbook",
    "PlaybookAdvice",
    "PlaybookStage",
    "PositiveControl",
    "PreflightCheck",
    "PreflightError",
    "ProviderUsage",
    "ReasoningTier",
    "ReplicationSummary",
    "ResearchFrontier",
    "RuntimeConfig",
    "ScientificAdmissibility",
    "StepMetrics",
    "UsageSource",
    "ValidationCheck",
    "ValidationReport",
    "ValidityDimension",
    "VerificationCheck",
    "VerificationConflictError",
    "VerificationIntegrityError",
    "VerificationRecord",
    "VerificationReport",
    "VerificationStore",
    "as_action_utility",
    "build_frontier",
    "derive_validity",
    "evaluate_controls",
    "evidence_from_result",
    "find_contradictions",
    "outcome_standing",
    "replication_summary",
    "require_preflight",
    "run_preflight",
    "sha256_of",
    "validate_result",
    "verify_analysis_coverage",
    "verify_artifact_integrity",
]
