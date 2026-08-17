"""Experiment design and the immutable record of what was actually run.

Three separate concepts, deliberately not merged:

``ExperimentSpec``
    The *scientific* design: what is being tested, against what baseline, by
    what procedure, and what observation would falsify it. Contains no commands,
    paths, or infrastructure.

``ExperimentResult``
    The immutable record of one execution. Produced only by an executor, from
    a process that actually ran. No component may synthesise one from
    reasoning.

``Environment``
    The provenance needed to attempt a re-run.

The binding between a spec and a runnable process lives in
:mod:`autonomous_research_lab.execution`, which keeps scientific design free of
infrastructure detail.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .budget import NO_COST, ResourceCost
from .ids import content_id
from .types import ConfigValue, freeze_mapping


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        }


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    hypothesis_id: str
    objective: str
    procedure: str
    metrics: tuple[str, ...]
    """Names of the metrics the experiment is expected to emit. Declared up
    front so that a result reporting different metrics is detectable rather
    than reinterpreted after the fact."""

    falsification_criterion: str
    """The condition on ``metrics`` that would count against the hypothesis,
    fixed before the run so it cannot be adjusted to fit the outcome."""

    baselines: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (0,)
    estimated_cost: ResourceCost = NO_COST
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("experiment spec must declare at least one metric")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "exp",
                    self.hypothesis_id,
                    self.objective,
                    self.procedure,
                    self.metrics,
                    self.falsification_criterion,
                    self.seeds,
                ),
            )


@dataclass(frozen=True, slots=True)
class Environment:
    """Provenance of the machine and code that produced a result."""

    python_version: str
    platform: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    """``True`` when the working tree had uncommitted changes at run time, i.e.
    when ``git_commit`` does not fully describe the code that ran."""


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """What actually happened. Append-only, never edited in place."""

    spec_id: str
    job_id: str
    status: ExperimentStatus
    command: tuple[str, ...]
    environment: Environment
    metrics: Mapping[str, float] = field(default_factory=dict)
    config: Mapping[str, ConfigValue] = field(default_factory=dict)
    seed: int | None = None
    artifacts: tuple[str, ...] = ()
    logs: tuple[str, ...] = ()
    runtime_seconds: float = 0.0
    cost: ResourceCost = NO_COST
    exit_code: int | None = None
    failure_reason: str | None = None
    """Present on failure. A failed run is a recorded outcome, not an absence
    of one."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))
        object.__setattr__(self, "config", freeze_mapping(self.config))
        if not self.id:
            object.__setattr__(self, "id", content_id("res", self.job_id))

    @property
    def succeeded(self) -> bool:
        return self.status is ExperimentStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class ResultRef:
    """A state's reference to a result held in the evidence store.

    States are beliefs and may branch; results are facts and are shared. A
    state therefore carries a reference plus the little that orchestration
    needs to reason without loading the payload.
    """

    result_id: str
    spec_id: str
    status: ExperimentStatus
