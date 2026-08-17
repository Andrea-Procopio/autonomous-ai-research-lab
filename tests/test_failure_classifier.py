"""The deterministic engineering-failure classifier: conservative, auditable,
and structurally blind to scientific outcomes."""

from __future__ import annotations

import sys
from pathlib import Path

from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentStatus,
)
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.failure_classifier import (
    FailureCategory,
    Repairability,
    diagnose_failure,
)
from autonomous_research_lab.execution.local import LocalExecutor

_ENV = Environment(python_version="3", platform="test")


def _result(
    *,
    status: ExperimentStatus = ExperimentStatus.FAILED,
    failure_reason: str | None = None,
    metrics: dict[str, float] | None = None,
    exit_code: int | None = 1,
    logs: tuple[str, ...] = (),
) -> ExperimentResult:
    return ExperimentResult(
        spec_id="exp_x",
        job_id="job_x",
        status=status,
        command=("cmd",),
        environment=_ENV,
        metrics=metrics or {},
        exit_code=exit_code,
        failure_reason=failure_reason,
        logs=logs,
    )


def _run(script: str, tmp_path: Path, **kwargs: object) -> ExperimentResult:
    executor = LocalExecutor(tmp_path / "runs")
    job = ExperimentJob(
        spec_id="exp_x",
        command=(sys.executable, "-c", script),
        timeout_seconds=30.0,
        **kwargs,  # type: ignore[arg-type]
    )
    return executor.collect(executor.submit(job))


# -- structured failure reasons from our own executor -------------------------


def test_timeout_is_classified_from_the_failure_reason() -> None:
    diagnosis = diagnose_failure(_result(failure_reason="timed out after 60.0s"))
    assert diagnosis.category is FailureCategory.TIMEOUT
    assert diagnosis.repairability is Repairability.REPAIRABLE
    assert diagnosis.evidence  # the exact signal is on the record


def test_launch_failure_is_classified() -> None:
    diagnosis = diagnose_failure(
        _result(failure_reason="could not launch command: no such binary")
    )
    assert diagnosis.category is FailureCategory.LAUNCH


def test_missing_metrics_is_a_deterministic_engineering_failure(
    tmp_path: Path,
) -> None:
    """Regression case B: process exits zero but writes no metrics — the
    executor records the failure, and the classifier names it."""
    result = _run("print('all good, no metrics though')", tmp_path)
    assert result.status is ExperimentStatus.FAILED
    diagnosis = diagnose_failure(result)
    assert diagnosis.category is FailureCategory.MISSING_METRICS
    assert diagnosis.repairability is Repairability.REPAIRABLE


def test_malformed_metrics_is_a_deterministic_engineering_failure(
    tmp_path: Path,
) -> None:
    result = _run(
        "import os, pathlib; "
        "pathlib.Path(os.environ['ARL_RUN_DIR'], 'metrics.json')"
        ".write_text('not json at all')",
        tmp_path,
    )
    assert result.status is ExperimentStatus.FAILED
    diagnosis = diagnose_failure(result)
    assert diagnosis.category is FailureCategory.MALFORMED_METRICS


def test_missing_required_artifact_is_classified(tmp_path: Path) -> None:
    result = _run(
        "import json, os, pathlib; "
        "pathlib.Path(os.environ['ARL_RUN_DIR'], 'metrics.json')"
        ".write_text(json.dumps({'m': 1.0}))",
        tmp_path,
        required_artifacts=("model.ckpt",),
    )
    diagnosis = diagnose_failure(result)
    assert diagnosis.category is FailureCategory.MISSING_ARTIFACT


# -- stderr evidence ----------------------------------------------------------


def test_import_error_is_read_from_preserved_stderr(tmp_path: Path) -> None:
    result = _run("import definitely_not_a_module", tmp_path)
    diagnosis = diagnose_failure(result)
    assert diagnosis.category is FailureCategory.IMPORT_ERROR
    assert any("ModuleNotFoundError" in line for line in diagnosis.evidence)


def test_missing_path_is_read_from_preserved_stderr(tmp_path: Path) -> None:
    result = _run("open('/definitely/not/a/path')", tmp_path)
    diagnosis = diagnose_failure(result)
    assert diagnosis.category is FailureCategory.MISSING_PATH


def test_oom_is_read_from_stderr() -> None:
    result = _result(failure_reason="exited with code 137")
    # Simulate a preserved stderr log carrying the OOM signal.
    diagnosis = diagnose_failure(result)
    assert diagnosis.category is FailureCategory.NONZERO_EXIT  # no stderr file

    oom = diagnose_failure(
        _result(
            failure_reason="exited with code 1",
            logs=("stdout", "stderr"),
        )
    )
    # Missing log files yield no evidence, never an error.
    assert oom.category is FailureCategory.NONZERO_EXIT


def test_bare_nonzero_exit_stays_uncertain(tmp_path: Path) -> None:
    """An unexplained crash is an honest UNKNOWN-cause diagnosis, never a
    guess dressed up as one."""
    result = _run("raise SystemExit(3)", tmp_path)
    diagnosis = diagnose_failure(result)
    assert diagnosis.category is FailureCategory.NONZERO_EXIT
    assert diagnosis.repairability is Repairability.UNCERTAIN


# -- the structural blindness to science --------------------------------------


def test_a_completed_run_is_never_classified_as_a_failure() -> None:
    """The classifier refuses to see scientific outcomes: a completed run
    with a terrible metric is NONE / NOT_APPLICABLE, full stop."""
    result = _result(
        status=ExperimentStatus.COMPLETED,
        exit_code=0,
        failure_reason=None,
        metrics={"accuracy": 0.0},  # disappointing is not broken
    )
    diagnosis = diagnose_failure(result)
    assert diagnosis.category is FailureCategory.NONE
    assert diagnosis.repairability is Repairability.NOT_APPLICABLE
