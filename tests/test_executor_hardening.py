"""The executor's no-silent-success guarantees, checked against real processes."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from autonomous_research_lab.core.experiment import ExperimentStatus
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import (
    HOME_DIRNAME,
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


def test_host_secrets_are_absent_from_the_child_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child sees a small allowlist plus what the job declares — never
    the whole host environment."""
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "hunter2")
    executor = LocalExecutor(tmp_path)
    job = _job(
        "import json, os, pathlib; "
        "d = pathlib.Path(os.environ['ARL_RUN_DIR']); "
        "(d / 'env.json').write_text(json.dumps(sorted(os.environ))); "
        + _WRITE_METRICS.replace("import json, os, pathlib; ", ""),
        env={"DECLARED_FOR_JOB": "yes"},
    )
    result = executor.collect(executor.submit(job))

    assert result.succeeded
    seen = set(json.loads((tmp_path / job.id / "env.json").read_text()))
    assert "SUPER_SECRET_TOKEN" not in seen
    assert "DECLARED_FOR_JOB" in seen  # explicit passthrough still works
    assert {"ARL_RUN_DIR", "ARL_CONFIG"} <= seen


def test_child_home_and_xdg_directories_are_job_private(tmp_path: Path) -> None:
    """SDK default-credential discovery under ``~`` (``~/.aws``,
    ``~/.config/gcloud``, ...) lands in a job-private directory inside the
    run directory — never the host user's home. Recovery isolation, not
    containment: generated code can still open absolute host paths."""
    executor = LocalExecutor(tmp_path)
    job = _job(
        "import json, os, pathlib; "
        "d = pathlib.Path(os.environ['ARL_RUN_DIR']); "
        "(d / 'home.json').write_text(json.dumps({"
        "'home': str(pathlib.Path.home()), "
        "'xdg_config': os.environ['XDG_CONFIG_HOME'], "
        "'xdg_cache': os.environ['XDG_CACHE_HOME']})); "
        + _WRITE_METRICS.replace("import json, os, pathlib; ", "")
    )
    result = executor.collect(executor.submit(job))

    assert result.succeeded
    seen = json.loads((tmp_path / job.id / "home.json").read_text())
    child_home = Path(seen["home"]).resolve()
    assert child_home == (tmp_path / job.id / HOME_DIRNAME).resolve()
    assert child_home != Path.home().resolve()
    for xdg in (seen["xdg_config"], seen["xdg_cache"]):
        assert Path(xdg).resolve().is_relative_to(child_home)


def test_escaping_required_artifact_paths_are_rejected_up_front() -> None:
    for bad in ("/etc/passwd", "../outside.txt", "a/../../outside.txt", " "):
        with pytest.raises(ValueError, match="required artifact"):
            ExperimentJob(
                spec_id="exp_bad",
                command=("true",),
                required_artifacts=(bad,),
            )


def test_symlink_escape_of_a_required_artifact_is_a_failure(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("host file")
    executor = LocalExecutor(tmp_path / "runs")
    job = _job(
        _WRITE_METRICS
        + f"; (d / 'model.ckpt').symlink_to({str(outside)!r})",
        required_artifacts=("model.ckpt",),
    )
    result = executor.collect(executor.submit(job))

    assert result.status is ExperimentStatus.FAILED
    assert result.failure_reason is not None
    assert "resolves outside" in result.failure_reason


def test_escaping_symlink_artifacts_are_not_collected_or_hashed(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("host file")
    executor = LocalExecutor(tmp_path / "runs")
    job = _job(
        _WRITE_METRICS
        + f"; (d / 'leak.txt').symlink_to({str(outside)!r})"
    )
    result = executor.collect(executor.submit(job))

    assert result.succeeded
    assert not any(a.endswith("leak.txt") for a in result.artifacts)
    manifest = json.loads(
        (tmp_path / "runs" / job.id / MANIFEST_FILENAME).read_text()
    )
    assert "leak.txt" not in manifest


def test_timeout_terminates_the_whole_process_group(tmp_path: Path) -> None:
    """A timed-out experiment may not leave grandchildren running."""
    executor = LocalExecutor(tmp_path)
    body = (
        "import os, subprocess, sys, time, pathlib; "
        "d = pathlib.Path(os.environ['ARL_RUN_DIR']); "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)']); "
        "(d / 'child.pid').write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    job = _job(body, timeout_seconds=1.5)
    result = executor.collect(executor.submit(job))

    assert result.status is ExperimentStatus.FAILED
    assert "timed out" in str(result.failure_reason)
    child_pid = int((tmp_path / job.id / "child.pid").read_text())
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break  # the grandchild is gone
        time.sleep(0.1)
    else:
        os.kill(child_pid, signal.SIGKILL)  # clean up before failing
        raise AssertionError("grandchild survived the job timeout")


def test_every_job_has_a_finite_timeout() -> None:
    job = ExperimentJob(spec_id="exp_t", command=("true",))
    assert job.timeout_seconds == 3600.0
    with pytest.raises(ValueError, match="timeout"):
        ExperimentJob(spec_id="exp_t", command=("true",), timeout_seconds=0.0)
    with pytest.raises(ValueError, match="timeout"):
        ExperimentJob(
            spec_id="exp_t", command=("true",), timeout_seconds=float("inf")
        )
