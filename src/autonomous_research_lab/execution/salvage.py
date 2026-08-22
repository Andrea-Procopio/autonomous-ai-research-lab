"""Finding out whether an interrupted attempt's job finished.

Recovery's one question about a job, answered from the local executor's
durable layout: did it finish, and with what full result? Never a
submission channel — the collector has no ``submit``, which is what
keeps "recovery never resubmits a job" structural rather than promised.

The path from an orphaned run directory to an answer is two existing
moves: :meth:`~.local.LocalExecutor.reap` closes the books on a job
whose submitter provably died, and ``collect`` reads the terminal
record. A job that cannot be proven finished — unknown, still running,
unprovably dead, or behind a record this process cannot read — yields
``None``, and recovery's conservative arm handles it exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.experiment import ExperimentResult
from .executor import JobNotFinishedError, UnknownJobError
from .local import LocalExecutor, MalformedJobRecordError


@dataclass(frozen=True, slots=True)
class LocalFinishedJobs:
    """The finished-job question over the local ``runs/`` layout."""

    run_root: Path

    def finished(self, job_id: str, /) -> ExperimentResult | None:
        executor = LocalExecutor(self.run_root)
        try:
            if not executor.reap(job_id).is_terminal:
                return None
            return executor.collect(job_id)
        except (UnknownJobError, JobNotFinishedError):
            return None
        except MalformedJobRecordError:
            # The safe direction: an unreadable record salvages nothing,
            # the attempt is charged conservatively, and the verifier
            # still reports the broken record for what it is.
            return None


__all__ = ["LocalFinishedJobs"]
