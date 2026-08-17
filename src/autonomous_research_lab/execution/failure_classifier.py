"""Deterministic first-pass diagnosis of failed executions.

Given a failed :class:`~autonomous_research_lab.core.experiment.ExperimentResult`
— its structured failure reason, exit code, and preserved stderr — classify
the *engineering* failure so the response can be chosen mechanically: raise a
timeout, install a dependency, fix a path, repair the metrics contract.

This classifier is deliberately conservative and deliberately blind to
science. It only ever runs on executions that did not complete validly, and
a completed run is answered with :attr:`FailureCategory.NONE` no matter what
its metrics say: a disappointing number is not a failure, and no heuristic
here is allowed to say otherwise. When the signals are ambiguous the answer
is :attr:`FailureCategory.UNKNOWN` with :attr:`Repairability.UNCERTAIN` —
an honest "diagnose by hand", never a guess dressed as a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from ..core.experiment import ExperimentResult


class FailureCategory(StrEnum):
    NONE = "none"
    """The run completed. Whatever its metrics say is science, not failure."""

    TIMEOUT = "timeout"
    LAUNCH = "launch"
    """The process could not be started at all."""

    OUT_OF_MEMORY = "out_of_memory"
    IMPORT_ERROR = "import_error"
    MISSING_PATH = "missing_path"
    MISSING_METRICS = "missing_metrics"
    MALFORMED_METRICS = "malformed_metrics"
    MISSING_ARTIFACT = "missing_artifact"
    NONZERO_EXIT = "nonzero_exit"
    UNKNOWN = "unknown"


class Repairability(StrEnum):
    REPAIRABLE = "repairable"
    """A known engineering fix exists (path, dependency, contract, limits)."""

    UNCERTAIN = "uncertain"
    """Failure is real but its cause is not established; diagnose further."""

    NOT_APPLICABLE = "not_applicable"
    """Nothing to repair: the run completed."""


@dataclass(frozen=True, slots=True)
class FailureDiagnosis:
    category: FailureCategory
    repairability: Repairability
    rationale: str
    evidence: tuple[str, ...] = ()
    """The exact signals the diagnosis rests on — failure-reason text,
    matched stderr lines — so the classification is auditable."""


#: Substrings of the executor's structured ``failure_reason``, checked first
#: because they are produced by our own code and therefore unambiguous.
_REASON_SIGNALS: Final[tuple[tuple[str, FailureCategory], ...]] = (
    ("timed out after", FailureCategory.TIMEOUT),
    ("could not launch command", FailureCategory.LAUNCH),
    ("wrote no metrics.json", FailureCategory.MISSING_METRICS),
    ("is not valid JSON", FailureCategory.MALFORMED_METRICS),
    ("must contain a JSON object", FailureCategory.MALFORMED_METRICS),
    ("is not a number", FailureCategory.MALFORMED_METRICS),
    ("is not finite", FailureCategory.MALFORMED_METRICS),
    ("required artifact", FailureCategory.MISSING_ARTIFACT),
)

#: Substrings matched against the run's preserved stderr, most specific
#: first. Only consulted when the failure reason alone is ambiguous.
_STDERR_SIGNALS: Final[tuple[tuple[str, FailureCategory], ...]] = (
    ("CUDA out of memory", FailureCategory.OUT_OF_MEMORY),
    ("MemoryError", FailureCategory.OUT_OF_MEMORY),
    ("Out of memory", FailureCategory.OUT_OF_MEMORY),
    ("ModuleNotFoundError", FailureCategory.IMPORT_ERROR),
    ("ImportError", FailureCategory.IMPORT_ERROR),
    ("FileNotFoundError", FailureCategory.MISSING_PATH),
    ("No such file or directory", FailureCategory.MISSING_PATH),
)

_STDERR_TAIL_BYTES: Final = 16_384


def diagnose_failure(result: ExperimentResult) -> FailureDiagnosis:
    """Classify one failed execution from its structured record.

    ``result.failure_reason`` (written by the executor) is trusted first;
    the preserved stderr log is scanned second; a bare nonzero exit with no
    recognizable signal stays ``UNKNOWN``/``UNCERTAIN``.
    """
    if result.succeeded:
        return FailureDiagnosis(
            category=FailureCategory.NONE,
            repairability=Repairability.NOT_APPLICABLE,
            rationale=(
                "the run completed; its scientific outcome is not a failure "
                "and is not classified here"
            ),
        )

    reason = result.failure_reason or ""
    for signal, category in _REASON_SIGNALS:
        if signal in reason:
            return FailureDiagnosis(
                category=category,
                repairability=Repairability.REPAIRABLE,
                rationale=f"executor recorded: {reason}",
                evidence=(reason,),
            )

    stderr_hits = _stderr_evidence(result)
    for signal, category in _STDERR_SIGNALS:
        matched = tuple(line for line in stderr_hits if signal in line)
        if matched:
            return FailureDiagnosis(
                category=category,
                repairability=Repairability.REPAIRABLE,
                rationale=f"stderr shows {signal!r}",
                evidence=(reason, *matched) if reason else matched,
            )

    if "exited with code" in reason:
        return FailureDiagnosis(
            category=FailureCategory.NONZERO_EXIT,
            repairability=Repairability.UNCERTAIN,
            rationale=(
                f"{reason}; no recognizable cause in stderr — diagnose "
                f"before repairing"
            ),
            evidence=(reason,),
        )

    return FailureDiagnosis(
        category=FailureCategory.UNKNOWN,
        repairability=Repairability.UNCERTAIN,
        rationale=reason or f"run ended {result.status} without a recorded reason",
        evidence=(reason,) if reason else (),
    )


def _stderr_evidence(result: ExperimentResult) -> tuple[str, ...]:
    """The last lines of the run's preserved stderr, if it still exists.
    A missing log yields no evidence — never an error: classification must
    work on whatever survives."""
    if len(result.logs) < 2:
        return ()
    path = Path(result.logs[1])
    try:
        raw = path.read_bytes()
    except OSError:
        return ()
    tail = raw[-_STDERR_TAIL_BYTES:].decode("utf-8", errors="replace")
    return tuple(line for line in tail.splitlines() if line.strip())
