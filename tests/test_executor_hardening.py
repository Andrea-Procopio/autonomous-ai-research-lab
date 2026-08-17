"""The executor's no-silent-success guarantees, checked against real processes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from autonomous_research_lab.core.experiment import ExperimentStatus
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import (
    MANIFEST_FILENAME,
    WORKSPACE_DIRNAME,
    LocalExecutor,
)
from autonomous_research_lab.runtime.validation import verify_artifact_integrity

_WRITE_METRICS = (
    "import json, os, pathlib; "
    "d = pathlib.Path(os.environ['ARL_RUN_DIR']); "
    "(d / 'metrics.json').write_text(json.dumps({'m': 1.0}))"
)


def _job(script: str, **kwargs: object) -> ExperimentJob:
    return ExperimentJob(
        spec_id="exp_hardening",
        command=(sys.executable, "-c", script),
        **kwargs,  # type: ignore[arg-type]
    )


def test_jobs_run_in_an_isolated_workspace_by_default(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path)
    job = _job(
        _WRITE_METRICS + "; pathlib.Path('scratch.txt').write_text('local')"
    )
    result = executor.collect(executor.submit(job))

    assert result.succeeded
    workspace_file = tmp_path / job.id / WORKSPACE_DIRNAME / "scratch.txt"
    assert workspace_file.exists()  # cwd was the job-private workspace
    assert str(workspace_file) in result.artifacts


def test_missing_required_artifact_is_a_failure_not_a_success(
    tmp_path: Path,
) -> None:
    executor = LocalExecutor(tmp_path)
    job = _job(_WRITE_METRICS, required_artifacts=("model.ckpt",))
    result = executor.collect(executor.submit(job))

    assert result.status is ExperimentStatus.FAILED
    assert result.failure_reason is not None
    assert "model.ckpt" in result.failure_reason
    # The run's outputs are preserved for diagnosis regardless.
    assert (tmp_path / job.id / "metrics.json").exists()


def test_produced_required_artifact_passes(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path)
    job = _job(
        _WRITE_METRICS + "; (d / 'model.ckpt').write_text('weights')",
        required_artifacts=("model.ckpt",),
    )
    result = executor.collect(executor.submit(job))
    assert result.succeeded


def test_non_finite_metrics_are_a_failure(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path)
    job = _job(
        "import os, pathlib; "
        "d = pathlib.Path(os.environ['ARL_RUN_DIR']); "
        "(d / 'metrics.json').write_text('{\"m\": NaN}')"
    )
    result = executor.collect(executor.submit(job))

    assert result.status is ExperimentStatus.FAILED
    assert result.failure_reason is not None
    assert "finite" in result.failure_reason


def test_every_run_writes_an_artifact_manifest(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path)
    job = _job(_WRITE_METRICS)
    result = executor.collect(executor.submit(job))

    manifest_path = tmp_path / job.id / MANIFEST_FILENAME
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "metrics.json" in manifest
    assert verify_artifact_integrity(result).passed


def test_tampered_artifacts_no_longer_verify(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path)
    job = _job(_WRITE_METRICS)
    result = executor.collect(executor.submit(job))

    (tmp_path / job.id / "metrics.json").write_text('{"m": 2.0}')
    check = verify_artifact_integrity(result)
    assert not check.passed
    assert "metrics.json" in check.detail
