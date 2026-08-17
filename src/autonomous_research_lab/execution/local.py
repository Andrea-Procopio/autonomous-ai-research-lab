"""Local subprocess executor -- the reference implementation of :class:`Executor`.

The contract between the lab and an experiment process is deliberately narrow:

* the lab sets ``ARL_RUN_DIR``, ``ARL_CONFIG`` (path to a JSON file) and, when
  a seed is fixed, ``ARL_SEED``;
* the process writes ``metrics.json`` -- a flat JSON object of numbers -- into
  ``ARL_RUN_DIR``;
* anything else it writes into the run directory is collected as an artifact.

Metrics are *read from a file a process wrote*. No component may hand metrics
to the executor, and no reasoning step may edit them afterwards. That is the
whole point of routing every number through here.

Execution is synchronous: ``submit`` runs the job to completion before
returning its id. That is a property of this backend, not of the interface --
callers still have to poll ``status`` and call ``collect``, so they remain
correct when a genuinely asynchronous backend replaces it.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from ..core.budget import ResourceCost
from ..core.experiment import Environment, ExperimentResult, ExperimentStatus
from .executor import (
    DuplicateJobError,
    Executor,
    ExperimentJob,
    JobStatus,
    UnknownJobError,
)

METRICS_FILENAME = "metrics.json"
CONFIG_FILENAME = "config.json"
STDOUT_FILENAME = "stdout.log"
STDERR_FILENAME = "stderr.log"

_STATUS_MAP = {
    JobStatus.SUCCEEDED: ExperimentStatus.COMPLETED,
    JobStatus.FAILED: ExperimentStatus.FAILED,
    JobStatus.CANCELLED: ExperimentStatus.CANCELLED,
}


class MalformedMetricsError(RuntimeError):
    """Raised when an experiment's ``metrics.json`` is not a flat map of numbers."""


class LocalExecutor(Executor):
    def __init__(self, run_root: Path | str) -> None:
        self._run_root = Path(run_root)
        self._run_root.mkdir(parents=True, exist_ok=True)
        self._results: dict[str, ExperimentResult] = {}
        self._status: dict[str, JobStatus] = {}

    def submit(self, job: ExperimentJob) -> str:
        if job.id in self._status:
            raise DuplicateJobError(
                f"job {job.id} was already submitted; a retry is a new event — "
                f"construct a new job"
            )
        self._status[job.id] = JobStatus.RUNNING
        run_dir = self._run_root / job.id
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / CONFIG_FILENAME
        config_path.write_text(json.dumps(dict(job.config), indent=2, sort_keys=True))

        env = dict(os.environ)
        env.update(job.env)
        env["ARL_RUN_DIR"] = str(run_dir)
        env["ARL_CONFIG"] = str(config_path)
        if job.seed is not None:
            env["ARL_SEED"] = str(job.seed)

        started = time.monotonic()
        failure_reason: str | None = None
        try:
            completed = subprocess.run(
                job.command,
                cwd=job.working_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=job.timeout_seconds,
                check=False,
            )
            exit_code: int | None = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            exit_code = None
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            failure_reason = f"timed out after {job.timeout_seconds}s"
        except OSError as exc:
            exit_code = None
            stdout, stderr = "", ""
            failure_reason = f"could not launch command: {exc}"
        runtime = time.monotonic() - started

        (run_dir / STDOUT_FILENAME).write_text(stdout)
        (run_dir / STDERR_FILENAME).write_text(stderr)

        metrics: Mapping[str, float] = {}
        if exit_code == 0:
            try:
                metrics = _read_metrics(run_dir / METRICS_FILENAME)
            except (MalformedMetricsError, FileNotFoundError) as exc:
                failure_reason = str(exc)
        elif failure_reason is None:
            failure_reason = f"exited with code {exit_code}"

        job_status = (
            JobStatus.SUCCEEDED if failure_reason is None else JobStatus.FAILED
        )
        self._status[job.id] = job_status

        result = ExperimentResult(
            spec_id=job.spec_id,
            job_id=job.id,
            status=_STATUS_MAP[job_status],
            command=job.command,
            environment=_capture_environment(job.working_dir),
            metrics=metrics,
            config=job.config,
            seed=job.seed,
            artifacts=_collect_artifacts(run_dir),
            logs=(str(run_dir / STDOUT_FILENAME), str(run_dir / STDERR_FILENAME)),
            runtime_seconds=runtime,
            cost=ResourceCost(wall_clock_seconds=runtime),
            exit_code=exit_code,
            failure_reason=failure_reason,
        )
        self._results[job.id] = result
        return job.id

    def status(self, job_id: str) -> JobStatus:
        try:
            return self._status[job_id]
        except KeyError as exc:
            raise UnknownJobError(job_id) from exc

    def collect(self, job_id: str) -> ExperimentResult:
        try:
            return self._results[job_id]
        except KeyError as exc:
            raise UnknownJobError(job_id) from exc


def _read_metrics(path: Path) -> Mapping[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"experiment wrote no {path.name}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MalformedMetricsError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MalformedMetricsError(f"{path.name} must contain a JSON object")
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MalformedMetricsError(f"metric {key!r} is not a number: {value!r}")
        metrics[str(key)] = float(value)
    return metrics


def _collect_artifacts(run_dir: Path) -> tuple[str, ...]:
    reserved = {STDOUT_FILENAME, STDERR_FILENAME, CONFIG_FILENAME}
    return tuple(
        sorted(
            str(p) for p in run_dir.rglob("*") if p.is_file() and p.name not in reserved
        )
    )


def _capture_environment(working_dir: str) -> Environment:
    commit = _git(working_dir, "rev-parse", "HEAD")
    dirty: bool | None = None
    if commit is not None:
        porcelain = _git(working_dir, "status", "--porcelain")
        dirty = bool(porcelain)
    return Environment(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        git_commit=commit,
        git_dirty=dirty,
    )


def _git(working_dir: str, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
