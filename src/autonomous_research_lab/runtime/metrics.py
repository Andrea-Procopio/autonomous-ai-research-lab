"""Runtime metrics: what each decision actually cost, recorded from step one.

The question this project ultimately wants to answer about its own
architecture is ``scientific progress / resource spent`` — and the marginal
value of each component (critic, playbook, synthesis, escalation) is a
question about these records. So every research step writes one structured
record.

Two kinds of call counting, deliberately not one number:

``reasoning_invocations``
    What the loop can actually enforce: how many times it invoked a
    reasoning seat this step — director deliberation, role performance,
    critic review, synthesis. With rule-based roles these invocations make
    zero model calls; with model-backed roles each is *at least* one.

``provider_calls`` / tokens / ``model``
    What a provider adapter reports having actually spent. Zero and empty
    until a provider exists, and never inferred: the loop cannot constrain
    or count calls a role's adapter makes internally, so these numbers come
    only from a :class:`UsageSource` that the adapter feeds.

Same storage philosophy as the trajectory log: one JSON object per line in a
local file, no database, no dashboard. Analysis code reads it later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from ..core.serialize import to_jsonable
from .escalation import ReasoningTier


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    """Actual model usage as reported by a provider adapter."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    def __add__(self, other: ProviderUsage) -> ProviderUsage:
        return ProviderUsage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            model=other.model or self.model,
        )


NO_USAGE = ProviderUsage()


class UsageSource(Protocol):
    """Where provider-reported usage enters the runtime. A future provider
    adapter accumulates usage as roles call it; the loop drains it once per
    step. Rule-based runs have no source, and their records honestly say
    zero."""

    def drain(self) -> ProviderUsage:
        """Usage accumulated since the last drain."""
        ...


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """The resource accounting of one runtime step, keyed to its decision."""

    decision_id: str
    action_type: str
    outcome_status: str
    reasoning_tier: ReasoningTier
    reasoning_invocations: int
    """Reasoning-seat invocations made by the loop this step (director,
    role, critic, synthesis). The enforceable quantity."""

    provider_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    """Provider-reported actuals; zero / empty without a provider adapter."""

    wall_clock_seconds: float = 0.0
    experiment_seconds: float = 0.0
    """Runtime of every execution the step's role actually performed —
    including work whose scientific commit was later rejected. Time spent
    is time spent."""

    estimated_usd: float = 0.0
    failures: int = 0
    critic_invoked: bool = False
    critic_reasons: tuple[str, ...] = ()
    synthesis_invoked: bool = False

    failure_category: str = ""
    """Deterministic classification of this step's execution failure, when
    one occurred (``execution.failure_classifier`` categories)."""

    debug_attempts: int = 0
    debug_resolved: bool = False
    """Debug success means a *valid execution* was recovered. It is a
    statement about engineering, never about the scientific outcome — a
    repaired experiment yielding a valid negative is a debugging success."""

    verification_status: str = ""
    """``ExperimentValidityStatus`` of this step's committed completed
    result, when verification ran."""

    preflight_failed: bool = False
    control_failures: int = 0
    methodology_rejected: bool = False
    implementation_rejected: bool = False
    analysis_rejected: bool = False
    negative_result_verdict: str = ""
    """For a conclusive negative outcome this step: ``"accepted"`` when it
    was promoted to verified scientific evidence, ``"deferred"`` when the
    observation was preserved with validity unresolved. Empty otherwise."""
    branch_count: int = 1
    """Concurrent lines of investigation this step advanced. 1 until
    branching exists; recorded so the ablation can see it change."""

    rationale: str = ""
    """The raw decision rationale, preserved verbatim."""

    notes: tuple[str, ...] = ()
    """Deterministic runtime notes: validation failures, engineering
    failures, budget overruns."""


class MetricsSink(Protocol):
    """Anything that accepts step metrics — the JSONL sink below, or a
    remote collector later."""

    def log(self, record: StepMetrics) -> None: ...


class JsonlRuntimeMetrics:
    """Append-only JSONL sink for step metrics, timestamped at the edge."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def log(self, record: StepMetrics) -> None:
        payload = to_jsonable(record)
        assert isinstance(payload, dict)
        payload["logged_at"] = datetime.now(tz=UTC).isoformat()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def read(self) -> tuple[dict[str, object], ...]:
        if not self._path.exists():
            return ()
        with self._path.open(encoding="utf-8") as handle:
            return tuple(json.loads(line) for line in handle if line.strip())
