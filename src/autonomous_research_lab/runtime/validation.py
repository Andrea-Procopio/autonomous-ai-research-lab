"""Deterministic validation of executed results — Tier 0, before any model.

The ground-truth hierarchy this runtime enforces::

    executed result / artifact
      > deterministic validation
        > artifact-grounded independent judgment
          > LLM opinion

Everything in this module is the second rung: checks a machine can perform
directly on a result, its spec, and the files on disk. No model is ever asked
whether a metric is present, whether a number is finite, or whether a result
belongs to the experiment that claims it — code answers those, and only the
*semantic* questions that survive these checks are worth a critic's tokens.

Also here: :func:`evidence_from_result`, the deterministic factual reading of
a completed result. Restating measured metrics as an ``Evidence`` object is a
transcription, not a judgment, so it costs zero model calls.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..core.evidence import Evidence, EvidenceKind
from ..core.experiment import ExperimentResult, ExperimentSpec, ExperimentStatus
from ..core.prediction import Consistency, Prediction, PredictionTest

MANIFEST_FILENAME = "manifest.json"
"""Part of the executor's run-directory contract (mirrored in
``execution.local``, which this package deliberately does not import):
relative artifact path -> sha256."""


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[ValidationCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def validate_result(
    spec: ExperimentSpec,
    result: ExperimentResult,
    *,
    prediction: Prediction | None = None,
) -> ValidationReport:
    """Every mechanical check one result admits, in one pass.

    A failed run yields a mostly-failing report — correctly: the report says
    what the result can and cannot support, it does not decide what to do
    about it.
    """
    checks: list[ValidationCheck] = []

    checks.append(
        ValidationCheck(
            name="result_matches_spec",
            passed=result.spec_id == spec.id,
            detail=f"result claims spec {result.spec_id}, validated against {spec.id}",
        )
    )
    # An orphaned run's exit code is honestly None — the submitter died
    # before observing it, and the executor's reap decided COMPLETED from
    # the contract's own evidence (metrics that parse, declared artifacts
    # present). Requiring a watched exit here would refuse every salvaged
    # job for carrying the truthful record of how it was recovered.
    exit_ok = result.exit_code == 0 or (
        result.exit_code is None and result.failure_reason is None
    )
    checks.append(
        ValidationCheck(
            name="process_completed",
            passed=result.status is ExperimentStatus.COMPLETED and exit_ok,
            detail=result.failure_reason
            or (
                f"status {result.status}, exit unobserved (orphaned run "
                f"finalized from its own evidence)"
                if result.exit_code is None
                else f"status {result.status}, exit code {result.exit_code}"
            ),
        )
    )

    missing = tuple(m for m in spec.metrics if m not in result.metrics)
    checks.append(
        ValidationCheck(
            name="declared_metrics_present",
            passed=not missing,
            detail=f"missing: {', '.join(missing)}" if missing else "",
        )
    )
    if prediction is not None:
        checks.append(
            ValidationCheck(
                name="prediction_metric_present",
                passed=prediction.metric in result.metrics,
                detail=f"prediction is stated in {prediction.metric!r}",
            )
        )

    non_finite = tuple(
        name for name, value in result.metrics.items() if not math.isfinite(value)
    )
    checks.append(
        ValidationCheck(
            name="metrics_finite",
            passed=not non_finite,
            detail=f"non-finite: {', '.join(non_finite)}" if non_finite else "",
        )
    )

    checks.append(
        ValidationCheck(
            name="seed_recorded",
            passed=result.seed is not None,
            detail="a result without its seed cannot be replicated exactly",
        )
    )

    return ValidationReport(checks=tuple(checks))


def verify_artifact_integrity(result: ExperimentResult) -> ValidationCheck:
    """Re-hash the result's artifacts against the manifest its run wrote.

    Detects post-hoc edits to experiment outputs. A result without a manifest
    (an older or foreign executor) fails the check explicitly rather than
    passing silently.
    """
    run_dir = _run_dir_of(result)
    if run_dir is None:
        return ValidationCheck(
            name="artifact_integrity",
            passed=False,
            detail="result carries no logs; its run directory cannot be located",
        )
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return ValidationCheck(
            name="artifact_integrity",
            passed=False,
            detail=f"no {MANIFEST_FILENAME} next to the run's logs",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, recorded in sorted(manifest.items()):
        path = run_dir / str(relative)
        if not path.is_file():
            return ValidationCheck(
                name="artifact_integrity",
                passed=False,
                detail=f"manifest names {relative}, which no longer exists",
            )
        if sha256_of(path) != recorded:
            return ValidationCheck(
                name="artifact_integrity",
                passed=False,
                detail=f"{relative} no longer hashes to its recorded digest",
            )
    return ValidationCheck(name="artifact_integrity", passed=True)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir_of(result: ExperimentResult) -> Path | None:
    if not result.logs:
        return None
    return Path(result.logs[0]).parent


@dataclass(frozen=True, slots=True)
class ReplicationSummary:
    """Basic consistency statistics over one replication family, one metric."""

    metric: str
    values: tuple[float, ...]
    spread: float
    """max - min over the observed values."""

    consistent_within: float
    """The tolerance the summary was asked about."""

    @property
    def consistent(self) -> bool:
        return self.spread <= self.consistent_within


def replication_summary(
    results: Iterable[ExperimentResult], metric: str, *, tolerance: float
) -> ReplicationSummary:
    """Deterministic agreement check across replications of one protocol.

    Statistics here are deliberately primitive — spread against a stated
    tolerance. Anything richer (power, intervals) is a statistician's job,
    performed on the same family.
    """
    values = tuple(
        r.metrics[metric]
        for r in results
        if r.succeeded and metric in r.metrics
    )
    spread = (max(values) - min(values)) if values else 0.0
    return ReplicationSummary(
        metric=metric,
        values=values,
        spread=spread,
        consistent_within=tolerance,
    )


def evidence_from_result(
    result: ExperimentResult,
    *,
    test: PredictionTest | None = None,
) -> Evidence:
    """The deterministic factual reading of one result. Zero model calls.

    A failed run reads as FAILURE evidence; a completed run whose mechanical
    prediction test came back inconsistent reads as NULL_RESULT; anything
    else is a MEASUREMENT. The observation restates what was measured —
    wording that draws a conclusion belongs in a claim, proposed by someone
    prepared to sign it.
    """
    if not result.succeeded:
        return Evidence(
            result_id=result.id,
            spec_id=result.spec_id,
            kind=EvidenceKind.FAILURE,
            observation=(
                f"Run did not complete: "
                f"{result.failure_reason or result.status}."
            ),
        )
    readings = ", ".join(
        f"{name}={value:g}" for name, value in sorted(result.metrics.items())
    )
    kind = (
        EvidenceKind.NULL_RESULT
        if test is not None and test.consistency is Consistency.INCONSISTENT
        else EvidenceKind.MEASUREMENT
    )
    return Evidence(
        result_id=result.id,
        spec_id=result.spec_id,
        kind=kind,
        observation=f"Observed {readings} (seed {result.seed}).",
        metrics=dict(result.metrics),
    )
