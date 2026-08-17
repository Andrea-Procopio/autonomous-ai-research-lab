"""The runtime layer: rich scientific state, sparse model calls.

This package holds the pieces that keep the number of model invocations
decoupled from the richness of the domain model:

* ``config`` — typed feature flags, so optional machinery is measurable;
* ``frontier`` — the derived view a director deliberates over;
* ``validation`` — Tier-0 deterministic checks and readings of results;
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

__all__ = [
    "COMPACT_METHOD",
    "MANIFEST_FILENAME",
    "NO_USAGE",
    "CompactValuation",
    "Contradiction",
    "EmpiricalMLPlaybook",
    "EscalationPolicy",
    "EscalationSignals",
    "EvaluationHooks",
    "HeldOutAccess",
    "HeldOutAccessError",
    "JsonlRuntimeMetrics",
    "Level",
    "MetricsSink",
    "ObjectiveEvaluator",
    "Playbook",
    "PlaybookAdvice",
    "PlaybookStage",
    "ProviderUsage",
    "ReasoningTier",
    "ReplicationSummary",
    "ResearchFrontier",
    "RuntimeConfig",
    "StepMetrics",
    "UsageSource",
    "ValidationCheck",
    "ValidationReport",
    "as_action_utility",
    "build_frontier",
    "evidence_from_result",
    "find_contradictions",
    "replication_summary",
    "sha256_of",
    "validate_result",
    "verify_artifact_integrity",
]
