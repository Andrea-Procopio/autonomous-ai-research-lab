"""The bounded debug loop: repair invalid executions, never disappointing ones.

::

    failed execution
        -> deterministic diagnosis (execution/failure_classifier)
        -> repair proposal (with its rationale, on the record)
        -> rerun (a NEW job — a retry is a new auditable occurrence)
        -> still failing? next bounded attempt
        -> STOP after max_attempts

Two invariants, enforced structurally rather than promised:

* **entry is by failure diagnosis, never by scientific outcome** —
  :meth:`ExperimentDebugger.debug` refuses a completed result outright
  (:class:`ScientificOutcomeError`), so ``while result_is_bad: debug()``
  cannot be written against this interface. A run that executed correctly
  and produced a disappointing number is not debuggable here, full stop;
* **every attempt is preserved** — the failing result, its diagnosis, the
  repair rationale, and the rerun's own result (cost included) each survive
  as separate records. Debugging succeeds when it produces a *valid*
  execution; whether that execution then supports or refutes the prediction
  is not this loop's business.

The debugger holds an executor and a :class:`RepairStrategy`; the strategy
is where a model-backed engineer will eventually sit, and a rule-based one
sits today. Either way each proposal states its rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.experiment import ExperimentResult, ExperimentSpec
from ..execution.executor import Executor, ExperimentJob
from ..execution.failure_classifier import (
    FailureCategory,
    FailureDiagnosis,
    diagnose_failure,
)


class ScientificOutcomeError(ValueError):
    """Raised when a completed run is handed to the debugger. A valid
    execution with a disappointing outcome is evidence, not a defect."""


@dataclass(frozen=True, slots=True)
class RepairProposal:
    """One proposed fix: a fresh job (new occurrence id) plus why it should
    help. The job must target the same spec — repair changes the plumbing,
    never the science."""

    job: ExperimentJob
    rationale: str


class RepairStrategy(Protocol):
    """Proposes the next repair given everything known about the failure.
    Returning ``None`` means no further repair is worth attempting."""

    def propose(
        self,
        spec: ExperimentSpec,
        failed: ExperimentResult,
        diagnosis: FailureDiagnosis,
        attempt_number: int,
    ) -> RepairProposal | None: ...


@dataclass(frozen=True, slots=True)
class DebugAttempt:
    """One auditable repair attempt: what was diagnosed, what was tried and
    why, and what actually happened when it reran."""

    number: int
    diagnosis: FailureDiagnosis
    repair_rationale: str
    result: ExperimentResult


@dataclass(frozen=True, slots=True)
class DebugSession:
    """The immutable record of one bounded debug loop."""

    spec_id: str
    initial_result_id: str
    initial_diagnosis: FailureDiagnosis
    attempts: tuple[DebugAttempt, ...]
    resolved: bool
    """True when the final attempt produced a valid (completed) execution.
    A debugging success — which says nothing about the scientific outcome."""

    stop_reason: str


@dataclass(frozen=True, slots=True)
class ExperimentDebugger:
    """Runs the bounded repair loop against a real executor.

    ``max_attempts`` is a hard ceiling; callers may pass a lower bound per
    call but never raise it above the configured limit.
    """

    executor: Executor
    strategy: RepairStrategy
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def debug(
        self,
        spec: ExperimentSpec,
        failed: ExperimentResult,
        *,
        max_attempts: int | None = None,
    ) -> DebugSession:
        """Attempt to repair one failed execution of ``spec``.

        Each iteration re-diagnoses the latest failure, asks the strategy
        for a repair, and reruns as a brand-new job. Stops on the first
        valid execution, on a strategy give-up, or at the attempt bound.
        """
        if failed.succeeded:
            raise ScientificOutcomeError(
                f"result {failed.id} is a completed execution; its outcome "
                f"is scientific evidence and may not be debugged"
            )
        if failed.spec_id != spec.id:
            raise ValueError(
                f"result {failed.id} ran {failed.spec_id}, not {spec.id}"
            )
        limit = self.max_attempts
        if max_attempts is not None:
            limit = min(limit, max_attempts)

        initial_diagnosis = diagnose_failure(failed)
        attempts: list[DebugAttempt] = []
        current = failed
        diagnosis = initial_diagnosis
        stop_reason = f"attempt limit of {limit} reached"

        for number in range(1, limit + 1):
            proposal = self.strategy.propose(spec, current, diagnosis, number)
            if proposal is None:
                stop_reason = "repair strategy proposed no further fix"
                break
            if proposal.job.spec_id != spec.id:
                raise ValueError(
                    f"repair proposal targets {proposal.job.spec_id}, "
                    f"not the spec being debugged ({spec.id})"
                )
            rerun = self.executor.collect(self.executor.submit(proposal.job))
            attempts.append(
                DebugAttempt(
                    number=number,
                    diagnosis=diagnosis,
                    repair_rationale=proposal.rationale,
                    result=rerun,
                )
            )
            if rerun.succeeded:
                return DebugSession(
                    spec_id=spec.id,
                    initial_result_id=failed.id,
                    initial_diagnosis=initial_diagnosis,
                    attempts=tuple(attempts),
                    resolved=True,
                    stop_reason=f"valid execution on attempt {number}",
                )
            current = rerun
            diagnosis = diagnose_failure(rerun)

        return DebugSession(
            spec_id=spec.id,
            initial_result_id=failed.id,
            initial_diagnosis=initial_diagnosis,
            attempts=tuple(attempts),
            resolved=False,
            stop_reason=stop_reason,
        )


def is_debuggable(diagnosis: FailureDiagnosis) -> bool:
    """Whether a diagnosis warrants entering the debug loop at all: an
    engineering failure of some category — never a completed run."""
    return diagnosis.category is not FailureCategory.NONE
