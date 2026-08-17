"""Execution contracts.

An :class:`ExperimentJob` is the infrastructure-facing binding of a scientific
:class:`~autonomous_research_lab.core.experiment.ExperimentSpec` to something a
machine can run. Keeping them separate means a spec can be re-bound to a
different backend -- laptop today, cluster later -- without editing the science.

The :class:`Executor` interface is deliberately three methods. It is shaped for
asynchronous, long-running, remote work, even though the only implementation
today is local and synchronous: ``submit`` returns a handle rather than a
result, so no caller can be written in a way that assumes the result is
already available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from ..core.experiment import ExperimentResult
from ..core.ids import content_id
from ..core.types import ConfigValue, freeze_mapping


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


class UnknownJobError(KeyError):
    """Raised when a job id is not known to the executor."""


class JobNotFinishedError(RuntimeError):
    """Raised when collecting a job that has not reached a terminal state."""


@dataclass(frozen=True, slots=True)
class ExperimentJob:
    spec_id: str
    command: tuple[str, ...]
    working_dir: str
    config: Mapping[str, ConfigValue] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    seed: int | None = None
    timeout_seconds: float | None = None
    attempt: int = 0
    """Discriminates replications of an otherwise identical job, so that
    re-running the same spec produces a distinct record rather than colliding
    with the original."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", freeze_mapping(self.config))
        object.__setattr__(self, "env", freeze_mapping(self.env))
        if not self.command:
            raise ValueError("job command must be non-empty")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "job",
                    self.spec_id,
                    self.command,
                    self.config,
                    self.seed,
                    self.attempt,
                ),
            )


class Executor(ABC):
    @abstractmethod
    def submit(self, job: ExperimentJob) -> str:
        """Enqueue ``job`` and return its id."""

    @abstractmethod
    def status(self, job_id: str) -> JobStatus: ...

    @abstractmethod
    def collect(self, job_id: str) -> ExperimentResult:
        """Return the immutable record of a terminated job."""
