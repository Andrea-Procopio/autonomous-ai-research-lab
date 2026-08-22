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

The id may be *derived* rather than minted, and that is not a departure
from occurrence identity: :func:`derive_job_id` derives it from the
attempt, which is itself an occurrence. What it buys is recomputability.
A caller that writes the id down before submitting can ask afterwards —
from a different process, with nothing in memory — whether that exact
job was ever submitted, which is the difference between recovering an
interrupted run and paying for it twice.

The :class:`Executor` interface is deliberately three methods, shaped for
asynchronous, long-running, remote work even though the only implementation
today is local and synchronous: ``submit`` returns a handle rather than a
result, so no caller can be written in a way that assumes the result is
already available.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePath
from typing import Final

from ..core.experiment import ExperimentResult
from ..core.ids import content_id, occurrence_id
from ..core.types import ConfigValue, freeze_mapping

DEFAULT_TIMEOUT_SECONDS: Final = 3600.0
"""Every job has a finite timeout. A job that genuinely needs longer states
so explicitly; "run forever" is not an option the contract offers."""


def derive_job_id(attempt_id: str) -> str:
    """The id the job of ``attempt_id`` will have, computed before it runs.

    One attempt submits at most one job through this route, so the
    attempt alone determines the id. A retry is a new attempt — the
    domain says so — and therefore a new job id, which is exactly why
    recovery may collect a job but must never resubmit one.
    """
    if not attempt_id.strip():
        raise ValueError("a derived job id needs the attempt it belongs to")
    return content_id("job", attempt_id)


def job_id_for_attempt(attempt_id: str) -> str:
    """The derived id when an attempt is named, and none when it is not.

    The "none" case is not an oversight. A job run outside any accounted
    attempt has nobody who will need to find it again, and minting an
    occurrence id for it is honest about that.
    """
    return derive_job_id(attempt_id) if attempt_id.strip() else ""


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
    """Environment variables passed to the process, explicitly. Executors do
    not hand the host environment to jobs; anything a job needs beyond the
    executor's small allowlist is declared here."""

    seed: int | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    required_artifacts: tuple[str, ...] = ()
    """Paths (relative to the run directory) the process must produce. A run
    that exits zero without them is recorded as a failure: a declared output
    that does not exist is not a success with a caveat, it is a failure.
    Paths must stay inside the run directory: absolute paths and ``..``
    segments are rejected at construction, and executors re-check the
    resolved path (symlinks included) at collection time."""

    gpu_count: int = 0
    """GPUs this job occupies for its whole wall-clock life. Occupancy,
    not utilization: the accounting question is what the lab could not
    schedule elsewhere while this ran, not what the kernels achieved.
    Zero for CPU jobs, and the executor bills ``gpu_hours`` from it."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", freeze_mapping(self.config))
        object.__setattr__(self, "env", freeze_mapping(self.env))
        if not self.command:
            raise ValueError("job command must be non-empty")
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be positive and finite")
        if self.gpu_count < 0:
            raise ValueError("gpu_count cannot be negative")
        for relative in self.required_artifacts:
            path = PurePath(relative)
            if not relative.strip() or path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"required artifact {relative!r} must be a relative path "
                    f"inside the run directory"
                )
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
