"""The bounded repair loops: fix invalid experiments, never disappointing ones.

Two structurally separate entries into one bounded repair machine:

**Execution repair** — a failed or cancelled process::

    failed execution
        -> deterministic diagnosis (execution/failure_classifier)
        -> repair proposal (with its rationale, on the record)
        -> rerun (a NEW job — a retry is a new auditable occurrence)
        -> still failing? next bounded attempt
        -> STOP after max_attempts

**Implementation repair** — a *completed* run with independent evidence of
an implementation bug (a failed positive control, a deterministic invariant
violation, an implementation-verifier FAIL). Entry requires an
:class:`ImplementationRepairTrigger`, whose constructor accepts **only**
implementation-dimension verification checks with at least one ``FAIL``. A
prediction test, a small effect, an underperforming baseline — none of
these can be expressed as such a trigger, so metric direction cannot become
a repair criterion by construction.

The invariants, enforced structurally rather than promised:

* **scientific outcome alone never triggers repair** —
  :meth:`ExperimentDebugger.debug` refuses a completed result outright
  (:class:`ScientificOutcomeError`), and
  :meth:`ExperimentDebugger.repair_implementation` demands a typed trigger
  carrying implementation-invalidity evidence. ``while result_is_bad:
  debug()`` cannot be written against either interface;
* **every attempt is preserved** — the invalid result, its diagnosis or
  trigger, the repair rationale, and each rerun's own result (cost
  included) survive as separate records. The original completed-but-invalid
  result is never deleted or rewritten;
* **repair success means a valid execution / valid implementation was
  recovered** — never a positive scientific outcome. A rerun from
  implementation repair must *earn* its own verification; nothing is
  transferred from the run it replaces.

The debugger holds an executor plus one strategy per entry path; the
strategies are where a model-backed engineer will eventually sit, and
rule-based ones sit today. Either way each proposal states its rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..core.experiment import ExperimentResult, ExperimentSpec
from ..execution.executor import Executor, ExperimentJob
from ..execution.failure_classifier import (
    FailureCategory,
    FailureDiagnosis,
    diagnose_failure,
)
from ..runtime.verification import (
    CheckState,
    ValidityDimension,
    VerificationCheck,
)


class ScientificOutcomeError(ValueError):
    """Raised when a completed run is handed to the debugger without
    implementation-invalidity evidence. A valid execution with a
    disappointing outcome is evidence, not a defect."""


@dataclass(frozen=True, slots=True)
class RepairProposal:
    """One proposed fix: a fresh job (new occurrence id) plus why it should
    help. The job must target the same spec — repair changes the plumbing,
    never the science."""

    job: ExperimentJob
    rationale: str


class RepairStrategy(Protocol):
    """Proposes the next execution repair given everything known about the
    failure. Returning ``None`` means no further repair is worth attempting."""

    def propose(
        self,
        spec: ExperimentSpec,
        failed: ExperimentResult,
        diagnosis: FailureDiagnosis,
        attempt_number: int,
    ) -> RepairProposal | None: ...


class RepairKind(StrEnum):
    EXECUTION = "execution"
    IMPLEMENTATION = "implementation"


@dataclass(frozen=True, slots=True)
class ImplementationRepairTrigger:
    """The typed entry ticket into implementation repair.

    Constructible **only** from implementation-dimension verification checks
    of which at least one is ``FAIL`` — a failed positive control, a
    deterministic invariant violation, a verifier's FAIL. A prediction
    test, a metric value, or any check on another dimension is rejected at
    construction, which is what makes "the result was disappointing" an
    impossible justification for repair.
    """

    result_id: str
    """The completed result whose implementation the evidence indicts."""

    checks: tuple[VerificationCheck, ...]

    def __post_init__(self) -> None:
        if not self.checks:
            raise ValueError(
                "implementation repair requires implementation-invalidity "
                "evidence; none was given"
            )
        foreign = [
            c.name
            for c in self.checks
            if c.dimension is not ValidityDimension.IMPLEMENTATION
        ]
        if foreign:
            raise ValueError(
                f"implementation repair accepts only implementation-dimension "
                f"checks; scientific outcomes (prediction tests, metric "
                f"direction, effect size) are not repair evidence — rejected: "
                f"{', '.join(foreign)}"
            )
        if not any(c.state is CheckState.FAIL for c in self.checks):
            raise ValueError(
                "implementation repair requires at least one FAILED "
                "implementation check; uncertainty alone does not indict "
                "the implementation"
            )

    @property
    def rationale(self) -> str:
        failed = [c for c in self.checks if c.state is CheckState.FAIL]
        return "; ".join(
            f"{c.name} failed" + (f" ({c.detail})" if c.detail else "")
            for c in failed
        )


class ImplementationRepairStrategy(Protocol):
    """Proposes the next reimplementation attempt for a run whose process
    completed but whose implementation the trigger's evidence indicts.
    Returning ``None`` means no further repair is worth attempting."""

    def propose(
        self,
        spec: ExperimentSpec,
        invalid: ExperimentResult,
        trigger: ImplementationRepairTrigger,
        attempt_number: int,
    ) -> RepairProposal | None: ...


@dataclass(frozen=True, slots=True)
class DebugAttempt:
    """One auditable repair attempt: what it was based on, what was tried
    and why, and what actually happened when it reran."""

    number: int
    diagnosis: FailureDiagnosis | None
    """The execution-failure diagnosis this attempt answered; ``None`` for
    implementation-repair attempts, whose basis is the session's trigger."""

    repair_rationale: str
    result: ExperimentResult

    @property
    def basis(self) -> str:
        """What this attempt was answering, for audit rendering."""
        if self.diagnosis is not None:
            return str(self.diagnosis.category)
        return "implementation invalidity"


@dataclass(frozen=True, slots=True)
class DebugSession:
    """The immutable record of one bounded repair loop."""

    spec_id: str
    kind: RepairKind
    initial_result_id: str
    initial_diagnosis: FailureDiagnosis | None
    """Present for execution repair; ``None`` for implementation repair,
    where ``trigger`` carries the basis instead."""

    trigger: ImplementationRepairTrigger | None
    """Present for implementation repair; ``None`` for execution repair."""

    attempts: tuple[DebugAttempt, ...]
    resolved: bool
    """True when the final attempt produced a completed execution. For
    implementation repair the rerun must additionally *earn* fresh
    verification before its implementation counts as recovered — that
    judgment belongs to the verification layer, not this loop. Either way,
    resolution says nothing about the scientific outcome."""

    stop_reason: str


@dataclass(frozen=True, slots=True)
class ExperimentDebugger:
    """Runs the bounded repair loops against a real executor.

    ``max_attempts`` is a hard ceiling; callers may pass a lower bound per
    call but never raise it above the configured limit. The two entry
    paths — :meth:`debug` for failed executions,
    :meth:`repair_implementation` for completed runs indicted by
    implementation-invalidity evidence — share the rerun machinery but not
    their preconditions.
    """

    executor: Executor
    strategy: RepairStrategy
    implementation_strategy: ImplementationRepairStrategy | None = None
    """Absent by default: a debugger without one can repair executions but
    can never touch a completed run."""

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
                f"is scientific evidence and may not be debugged — "
                f"implementation repair requires an explicit "
                f"ImplementationRepairTrigger"
            )
        if failed.spec_id != spec.id:
            raise ValueError(
                f"result {failed.id} ran {failed.spec_id}, not {spec.id}"
            )
        initial_diagnosis = diagnose_failure(failed)
        attempts: list[DebugAttempt] = []
        current = failed
        diagnosis = initial_diagnosis
        limit = self._limit(max_attempts)
        stop_reason = f"attempt limit of {limit} reached"

        for number in range(1, limit + 1):
            proposal = self.strategy.propose(spec, current, diagnosis, number)
            if proposal is None:
                stop_reason = "repair strategy proposed no further fix"
                break
            rerun = self._rerun(spec, proposal)
            attempts.append(
                DebugAttempt(
                    number=number,
                    diagnosis=diagnosis,
                    repair_rationale=proposal.rationale,
                    result=rerun,
                )
            )
            if rerun.succeeded:
                return self._session(
                    RepairKind.EXECUTION, spec, failed, initial_diagnosis,
                    None, attempts, resolved=True,
                    stop_reason=f"valid execution on attempt {number}",
                )
            current = rerun
            diagnosis = diagnose_failure(rerun)

        return self._session(
            RepairKind.EXECUTION, spec, failed, initial_diagnosis, None,
            attempts, resolved=False, stop_reason=stop_reason,
        )

    def repair_implementation(
        self,
        spec: ExperimentSpec,
        invalid: ExperimentResult,
        trigger: ImplementationRepairTrigger,
        *,
        max_attempts: int | None = None,
    ) -> DebugSession:
        """Attempt to repair the *implementation* behind a completed run.

        Entry demands a :class:`ImplementationRepairTrigger` — typed
        implementation-invalidity evidence — so this path is structurally
        unreachable from a prediction test or a disappointing metric. The
        original result is never touched: each attempt is a fresh job, and
        a completed rerun must earn its own verification afterwards; this
        loop only reports that a rerun completed.
        """
        if self.implementation_strategy is None:
            raise RuntimeError(
                "no implementation repair strategy is configured; a debugger "
                "without one can never touch a completed run"
            )
        if invalid.spec_id != spec.id:
            raise ValueError(
                f"result {invalid.id} ran {invalid.spec_id}, not {spec.id}"
            )
        attempts: list[DebugAttempt] = []
        current = invalid
        limit = self._limit(max_attempts)
        stop_reason = f"attempt limit of {limit} reached"

        for number in range(1, limit + 1):
            proposal = self.implementation_strategy.propose(
                spec, current, trigger, number
            )
            if proposal is None:
                stop_reason = "repair strategy proposed no further fix"
                break
            rerun = self._rerun(spec, proposal)
            attempts.append(
                DebugAttempt(
                    number=number,
                    diagnosis=None,
                    repair_rationale=proposal.rationale,
                    result=rerun,
                )
            )
            if rerun.succeeded:
                return self._session(
                    RepairKind.IMPLEMENTATION, spec, invalid, None, trigger,
                    attempts, resolved=True,
                    stop_reason=(
                        f"reimplementation completed on attempt {number}; "
                        f"its validity must now be earned by fresh "
                        f"verification"
                    ),
                )
            current = rerun

        return self._session(
            RepairKind.IMPLEMENTATION, spec, invalid, None, trigger,
            attempts, resolved=False, stop_reason=stop_reason,
        )

    def _limit(self, max_attempts: int | None) -> int:
        if max_attempts is None:
            return self.max_attempts
        return min(self.max_attempts, max_attempts)

    def _rerun(
        self, spec: ExperimentSpec, proposal: RepairProposal
    ) -> ExperimentResult:
        if proposal.job.spec_id != spec.id:
            raise ValueError(
                f"repair proposal targets {proposal.job.spec_id}, "
                f"not the spec being debugged ({spec.id})"
            )
        return self.executor.collect(self.executor.submit(proposal.job))

    def _session(
        self,
        kind: RepairKind,
        spec: ExperimentSpec,
        initial: ExperimentResult,
        initial_diagnosis: FailureDiagnosis | None,
        trigger: ImplementationRepairTrigger | None,
        attempts: list[DebugAttempt],
        *,
        resolved: bool,
        stop_reason: str,
    ) -> DebugSession:
        return DebugSession(
            spec_id=spec.id,
            kind=kind,
            initial_result_id=initial.id,
            initial_diagnosis=initial_diagnosis,
            trigger=trigger,
            attempts=tuple(attempts),
            resolved=resolved,
            stop_reason=stop_reason,
        )


def is_debuggable(diagnosis: FailureDiagnosis) -> bool:
    """Whether a diagnosis warrants entering the debug loop at all: an
    engineering failure of some category — never a completed run."""
    return diagnosis.category is not FailureCategory.NONE
