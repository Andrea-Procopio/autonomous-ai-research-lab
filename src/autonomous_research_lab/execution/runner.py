"""Who submits a job, and who merely asks for one to be run.

A role prepares work; it does not perform side effects. That is the
proposal invariant the whole architecture rests on, and the engineer was
quietly the exception: it built a job and then submitted, polled and
collected it itself. Nothing about the *science* required that, and two
things argued against it. A role holding an executor can launch work
nobody outside it recorded, and — the reason this changed — the
boundaries either side of a submission are exactly where an interrupted
run needs a durable note, which is trusted code's job to write.

So the engineer takes a :class:`JobRunner` instead of an
:class:`~.executor.Executor`. It hands over a job and receives a result;
it cannot poll, cannot collect, and cannot submit anything twice.

:class:`DirectJobRunner` is what that used to mean, extracted unchanged:
submit, wait, collect. A runtime that journals the two boundaries wraps
it rather than replacing it, so the difference between a recoverable run
and an ordinary one is which runner was wired, not which code path ran.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final, Protocol

from ..core.budget import ResourceCost
from ..core.experiment import ExperimentResult
from .executor import (
    Executor,
    ExperimentJob,
    JobNotFinishedError,
    UnknownJobError,
)

DEFAULT_POLL_SECONDS: Final = 0.05


class JobRunner(Protocol):
    """Run one prepared job to completion and return what happened.

    The parameters are positional-only so an implementation may name them
    whatever reads best; ``attempt_id`` is the attempt the job belongs to,
    which a journaling runner needs and a direct one ignores.
    """

    def run(
        self, job: ExperimentJob, attempt_id: str = "", /
    ) -> ExperimentResult: ...


@dataclass(frozen=True, slots=True)
class DirectJobRunner:
    """Submit, wait, collect — the executor contract, used plainly."""

    executor: Executor
    poll_seconds: float = DEFAULT_POLL_SECONDS

    def run(
        self, job: ExperimentJob, _attempt_id: str = "", /
    ) -> ExperimentResult:
        job_id = self.executor.submit(job)
        while not self.executor.status(job_id).is_terminal:
            time.sleep(self.poll_seconds)
        return self.executor.collect(job_id)


@dataclass(frozen=True, slots=True)
class ExecutorJobFacts:
    """The two questions recovery asks about a job, and nothing else.

    Deliberately narrow. Recovery lives in the composition root and must
    decide whether an interrupted attempt bought anything; it needs to
    know whether the job exists and what it cost, not what it measured.
    Handing it a result would hand it the science too.
    """

    executor: Executor

    def exists(self, job_id: str, /) -> bool:
        """Whether the executor has ever heard of this job."""
        try:
            self.executor.status(job_id)
        except UnknownJobError:
            return False
        return True

    def cost_of(self, job_id: str, /) -> ResourceCost | None:
        """What the job cost, or ``None`` when it cannot be collected —
        a job whose process died leaves a record saying it began and
        nothing saying how it ended."""
        try:
            return self.executor.collect(job_id).cost
        except (UnknownJobError, JobNotFinishedError):
            return None
