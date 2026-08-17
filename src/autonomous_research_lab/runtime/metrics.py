"""Runtime metrics: what each decision actually cost, recorded from step one.

The question this project ultimately wants to answer about its own
architecture is ``scientific progress / resource spent`` — and the marginal
value of each component (critic, playbook, synthesis, escalation) is a
question about these records. So every research step writes one structured
record: conceptual model calls, tokens where known, wall-clock, experiment
compute, whether the critic fired and why, the reasoning tier, and the
outcome.

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
class StepMetrics:
    """The resource accounting of one runtime step, keyed to its decision."""

    decision_id: str
    action_type: str
    outcome_status: str
    reasoning_tier: ReasoningTier
    llm_calls: int
    """Conceptual model invocations this step: director deliberation, role
    performances, critic review, synthesis. Counted by the runtime, so the
    accounting exists before any real provider does."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    """Zero / empty until a provider reports real numbers."""

    wall_clock_seconds: float = 0.0
    experiment_seconds: float = 0.0
    estimated_usd: float = 0.0
    failures: int = 0
    critic_invoked: bool = False
    critic_reasons: tuple[str, ...] = ()
    synthesis_invoked: bool = False
    branch_count: int = 1
    """Concurrent lines of investigation this step advanced. 1 until
    branching exists; recorded so the ablation can see it change."""

    rationale: str = ""
    """The raw decision rationale, preserved verbatim."""

    notes: tuple[str, ...] = ()


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
