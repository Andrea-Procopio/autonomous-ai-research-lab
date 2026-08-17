"""Execution contracts.

An :class:`ExperimentJob` is the infrastructure-facing binding of a scientific
:class:`~autonomous_research_lab.core.experiment.ExperimentSpec` to something a
machine can run. Keeping them separate means a spec can be re-bound to a
different backend — laptop today, cluster later — without editing the science.

A job is an **occurrence**, not a semantic object: submitting the same
spec twice — a retry, a replication — is two events, so two identically
configured jobs carry distinct ids, and each job may be submitted exactly
once. This is what keeps every execution a distinct record with its own
provenance.

The :class:`Executor` interface is deliberately three methods, shaped for
asynchronous, long-running, remote work even though the only implementation
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
from ..core.ids import occurrence_id
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


class DuplicateJobError(RuntimeError):
    """Raised when a job object is submitted more than once. A retry is a new
    event: construct a new job."""


@dataclass(frozen=True, slots=True)
class ExperimentJob:
    spec_id: str
    command: tuple[str, ...]
    working_dir: str | None = None
    """Where the process runs. ``None`` means the executor provides an
    isolated, job-private working directory — the default, so a job only
    touches shared code when it explicitly asks to."""

    config: Mapping[str, ConfigValue] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    seed: int | None = None
    timeout_seconds: float | None = None
    required_artifacts: tuple[str, ...] = ()
    """Paths (relative to the run directory) the process must produce. A run
    that exits zero without them is recorded as a failure: a declared output
    that does not exist is not a success with a caveat, it is a failure."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", freeze_mapping(self.config))
        object.__setattr__(self, "env", freeze_mapping(self.env))
        if not self.command:
            raise ValueError("job command must be non-empty")
        if not self.id:
            object.__setattr__(self, "id", occurrence_id("job"))


class Executor(ABC):
    @abstractmethod
    def submit(self, job: ExperimentJob) -> str:
        """Enqueue ``job`` and return its id. Each job submits at most once."""

    @abstractmethod
    def status(self, job_id: str) -> JobStatus: ...

    @abstractmethod
    def collect(self, job_id: str) -> ExperimentResult:
        """Return the immutable record of a terminated job."""
