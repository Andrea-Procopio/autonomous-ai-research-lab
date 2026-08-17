"""Deterministic validation catches machine-checkable problems — no model calls."""

from __future__ import annotations

import json
from pathlib import Path

from autonomous_research_lab.core.evidence import EvidenceKind
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.runtime.validation import (
    MANIFEST_FILENAME,
    evidence_from_result,
    replication_summary,
    sha256_of,
    validate_result,
    verify_artifact_integrity,
)

ENV = Environment(python_version="3.11", platform="test")
PREDICTION = Prediction(
    hypothesis_id="hyp_x",
    condition="test",
    metric="accuracy",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.9,
)
SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="measure accuracy",
    procedure="run the model",
    metrics=("accuracy", "loss"),
)


def _result(
    *,
    metrics: dict[str, float],
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
    exit_code: int | None = 0,
    spec_id: str | None = None,
    seed: int | None = 7,
    logs: tuple[str, ...] = (),
) -> ExperimentResult:
    return ExperimentResult(
        spec_id=spec_id or SPEC.id,
        job_id="job_test",
        status=status,
        command=("run",),
        environment=ENV,
        metrics=metrics,
        seed=seed,
        logs=logs,
        exit_code=exit_code,
    )


def test_clean_result_passes_every_check() -> None:
    report = validate_result(
        SPEC,
        _result(metrics={"accuracy": 0.95, "loss": 0.1}),
        prediction=PREDICTION,
    )
    assert report.passed
    assert report.failures == ()


def test_missing_declared_metric_is_caught() -> None:
    report = validate_result(SPEC, _result(metrics={"accuracy": 0.95}))
    assert not report.passed
    assert any(c.name == "declared_metrics_present" for c in report.failures)


def test_non_finite_metric_is_caught() -> None:
    report = validate_result(
        SPEC, _result(metrics={"accuracy": float("nan"), "loss": 0.1})
    )
    assert any(c.name == "metrics_finite" for c in report.failures)


def test_result_from_a_different_spec_is_caught() -> None:
    report = validate_result(
        SPEC, _result(metrics={"accuracy": 0.95, "loss": 0.1}, spec_id="exp_other")
    )
    assert any(c.name == "result_matches_spec" for c in report.failures)


def test_failed_process_is_caught() -> None:
    report = validate_result(
        SPEC,
        _result(metrics={}, status=ExperimentStatus.FAILED, exit_code=1),
    )
    assert any(c.name == "process_completed" for c in report.failures)


def test_missing_seed_is_caught() -> None:
    report = validate_result(
        SPEC, _result(metrics={"accuracy": 0.95, "loss": 0.1}, seed=None)
    )
    assert any(c.name == "seed_recorded" for c in report.failures)


def test_artifact_integrity_detects_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "output.txt"
    artifact.write_text("measured")
    (run_dir / "stdout.log").write_text("")
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps({"output.txt": sha256_of(artifact)})
    )
    result = _result(
        metrics={"accuracy": 0.95, "loss": 0.1},
        logs=(str(run_dir / "stdout.log"),),
    )

    assert verify_artifact_integrity(result).passed

    artifact.write_text("edited after the fact")
    check = verify_artifact_integrity(result)
    assert not check.passed
    assert "output.txt" in check.detail


def test_artifact_integrity_fails_without_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "stdout.log").write_text("")
    result = _result(
        metrics={"accuracy": 0.95, "loss": 0.1},
        logs=(str(tmp_path / "stdout.log"),),
    )
    assert not verify_artifact_integrity(result).passed


def test_replication_summary_is_deterministic_arithmetic() -> None:
    results = tuple(
        _result(metrics={"accuracy": value, "loss": 0.1})
        for value in (0.90, 0.92, 0.91)
    )
    summary = replication_summary(results, "accuracy", tolerance=0.05)
    assert summary.values == (0.90, 0.92, 0.91)
    assert abs(summary.spread - 0.02) < 1e-12
    assert summary.consistent

    tight = replication_summary(results, "accuracy", tolerance=0.01)
    assert not tight.consistent


def test_evidence_reading_is_a_transcription_not_a_judgment() -> None:
    result = _result(metrics={"accuracy": 0.95, "loss": 0.1})
    evidence = evidence_from_result(result)
    assert evidence.result_id == result.id
    assert evidence.kind is EvidenceKind.MEASUREMENT
    assert evidence.metrics["accuracy"] == 0.95
    assert "accuracy=0.95" in evidence.observation

    inconsistent = PredictionTest(
        prediction_id=PREDICTION.id,
        result_id=result.id,
        metric="accuracy",
        observed=0.95,
        consistency=Consistency.INCONSISTENT,
    )
    assert (
        evidence_from_result(result, test=inconsistent).kind
        is EvidenceKind.NULL_RESULT
    )

    failed = _result(metrics={}, status=ExperimentStatus.FAILED, exit_code=1)
    assert evidence_from_result(failed).kind is EvidenceKind.FAILURE
