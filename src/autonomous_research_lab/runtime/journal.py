"""Where the loop writes down what an attempt is doing.

The runtime charges a budget it carries on the state and reports the
spend through :mod:`.spend`. This is the other half of the same seam: a
record of *what the money was for* and how far the attempt got in making
it durable, written as it happens rather than reconstructed afterwards.

The two together are what make a killed process recoverable::

    journal the intent  ->  do the thing  ->  journal that it is durable

A ledger alone cannot close the gap. It says money moved; it cannot say
whether the thing the money bought was ever written down, so a process
that dies between paying and recording leaves two records disagreeing
and nothing to decide between them. The journal is that deciding record.

It is a protocol rather than an import for the reason ``spend`` is: the
runtime depends on ``core`` alone, and the package that owns run
identity and durable records sits above it. The loop knows only that
something can be told what phase an attempt reached.

``None`` in the loop means no journal is wired, and the run keeps the
pre-existing behavior — recoverable between steps and not inside one.
It is the same explicit ablation ``ledger=None`` already is.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from ..core.attempt import AttemptPhase, SettlementBasis
from ..core.budget import NO_COST, ResourceCost
from ..core.experiment import ExperimentResult
from ..execution.executor import ExperimentJob
from ..execution.runner import JobRunner


class AttemptJournal(Protocol):
    """The durable record of one run's attempts, phase by phase."""

    def record(
        self,
        *,
        attempt_id: str,
        phase: AttemptPhase,
        state_id: str = "",
        job_id: str = "",
        bundle_id: str = "",
        produced: Iterable[tuple[str, str]] = (),
        reserved: ResourceCost = NO_COST,
        settled: ResourceCost = NO_COST,
        basis: SettlementBasis = SettlementBasis.NONE,
        detail: str = "",
    ) -> object:
        """Write down that ``attempt_id`` reached ``phase``.

        Recording one phase twice must record it once and return what is
        already written: recovery re-drives phases it cannot prove
        happened, and a journal that grew a duplicate every time would
        stop being able to say what the attempt did.
        """
        ...


@dataclass(frozen=True, slots=True)
class JournalingJobRunner:
    """A runner that writes down a submission before making it.

    The two phases either side of a job are the only ones the runtime
    cannot see from outside a role, which is why they are recorded here
    rather than in the loop. The order is the one the whole journal
    keeps: ``SUBMITTED`` before the submission, because a job nobody
    wrote down first cannot be found afterwards, and
    ``OUTPUTS_DURABLE`` after the collection, because a durability claim
    is only true once the bytes exist.

    A job with no attempt behind it is run and not journaled: there is
    no reservation it answers and nobody who would come looking.
    """

    inner: JobRunner
    journal: AttemptJournal

    def run(
        self, job: ExperimentJob, attempt_id: str = "", /
    ) -> ExperimentResult:
        if not attempt_id:
            return self.inner.run(job, attempt_id)
        self.journal.record(
            attempt_id=attempt_id,
            phase=AttemptPhase.SUBMITTED,
            job_id=job.id,
            detail=f"submitting {job.spec_id}",
        )
        result = self.inner.run(job, attempt_id)
        self.journal.record(
            attempt_id=attempt_id,
            phase=AttemptPhase.OUTPUTS_DURABLE,
            job_id=job.id,
            produced=(("result", result.id),),
            detail=f"{result.status}",
        )
        return result
