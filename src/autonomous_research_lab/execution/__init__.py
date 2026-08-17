"""Running experiments, and the interface that keeps where they run swappable."""

from .executor import (
    Executor,
    ExperimentJob,
    JobNotFinishedError,
    JobStatus,
    UnknownJobError,
)
from .local import LocalExecutor, MalformedMetricsError

__all__ = [
    "Executor",
    "ExperimentJob",
    "JobNotFinishedError",
    "JobStatus",
    "LocalExecutor",
    "MalformedMetricsError",
    "UnknownJobError",
]
