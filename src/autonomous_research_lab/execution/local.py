"""Local subprocess executor -- the reference implementation of :class:`Executor`.

The contract between the lab and an experiment process is deliberately narrow:

* the lab sets ``ARL_RUN_DIR``, ``ARL_CONFIG`` (path to a JSON file) and, when
  a seed is fixed, ``ARL_SEED``;
* the process writes ``metrics.json`` -- a flat JSON object of finite numbers
  -- into ``ARL_RUN_DIR``;
* anything else it writes into the run directory is collected as an artifact.

Metrics are *read from a file a process wrote*. No component may hand metrics
to the executor, and no reasoning step may edit them afterwards. That is the
whole point of routing every number through here.

There is no silent success. A run that exits zero but writes no metrics, a
metric that is not a finite number, or a declared required artifact that is
missing or escapes the run directory -- each is recorded as a failure, with
the run directory preserved for diagnosis. Every run also gets a
``manifest.json`` of artifact hashes, so post-hoc edits to experiment outputs
are detectable later.

Isolation here is **job-private recovery isolation, not a security
sandbox**: a job with no ``working_dir`` runs in a job-private ``workspace/``
inside its run directory so concurrent jobs cannot trample each other; the
child process receives a small allowlisted environment plus ``job.env``
rather than the whole host environment; ``HOME`` (and the XDG base
directories) point at a job-private ``home/`` inside the run directory, so
standard SDKs that discover credentials and configuration under ``~`` --
``~/.aws``, ``~/.config/gcloud`` and the like -- find an empty job-private
directory rather than the host user's; artifact paths are confined to the
run directory (symlinks that resolve outside it are excluded); every job has
a finite timeout, and on timeout the whole process group is terminated, not
only the immediate child. None of this constrains a malicious process --
generated code can still deliberately open absolute host paths, and
containment at that level is a future remote-executor concern.

Execution is synchronous: ``submit`` runs the job to completion before
returning its id. That is a property of this backend, not of the interface --
callers still have to poll ``status`` and call ``collect``, so they remain
correct when a genuinely asynchronous backend replaces it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

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
MANIFEST_FILENAME = "manifest.json"
WORKSPACE_DIRNAME = "workspace"
HOME_DIRNAME = "home"

#: The only host environment variables a child process inherits. Everything
#: else -- credentials, tokens, cloud configuration -- must be passed
#: explicitly via ``job.env`` to reach an experiment. ``HOME`` is
#: deliberately absent: the child gets a job-private home directory instead,
#: so SDK default-credential discovery cannot reach the host user's files.
ENV_ALLOWLIST: Final = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "COMSPEC",
    "PATHEXT",
)

_TERMINATE_GRACE_SECONDS: Final = 5.0

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

        if job.working_dir is not None:
            working_dir = job.working_dir
        else:
            workspace = run_dir / WORKSPACE_DIRNAME
            workspace.mkdir(parents=True, exist_ok=True)
            working_dir = str(workspace)

        config_path = run_dir / CONFIG_FILENAME
        config_path.write_text(json.dumps(dict(job.config), indent=2, sort_keys=True))

        # Explicit environment: a small allowlist plus what the job declares.
        # Host credentials never reach an experiment implicitly.
        env = {
            name: os.environ[name]
            for name in ENV_ALLOWLIST
            if name in os.environ
        }
        # Job-private home: ``Path.home()`` and XDG-style configuration and
        # cache lookups resolve inside the run directory, not to the host
        # user's home (recovery isolation, not filesystem containment).
        # ``USERPROFILE`` covers ``Path.home()`` on Windows.
        home_dir = run_dir / HOME_DIRNAME
        home_dir.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home_dir)
        env["USERPROFILE"] = str(home_dir)
        env["XDG_CONFIG_HOME"] = str(home_dir / ".config")
        env["XDG_CACHE_HOME"] = str(home_dir / ".cache")
        env["XDG_DATA_HOME"] = str(home_dir / ".local" / "share")
        env["XDG_STATE_HOME"] = str(home_dir / ".local" / "state")
        env.update(job.env)
        env["ARL_RUN_DIR"] = str(run_dir)
        env["ARL_CONFIG"] = str(config_path)
        if job.seed is not None:
            env["ARL_SEED"] = str(job.seed)

        started = time.monotonic()
        exit_code, stdout, stderr, failure_reason = _run_process(
            job.command, working_dir, env, job.timeout_seconds
        )
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

        if failure_reason is None:
            failure_reason = _check_required_artifacts(
                run_dir, job.required_artifacts
            )

        job_status = (
            JobStatus.SUCCEEDED if failure_reason is None else JobStatus.FAILED
        )
        self._status[job.id] = job_status

        # Everything the run left behind is preserved and hashed -- failures
        # included, because a failed run's outputs are diagnostic evidence.
        artifacts = _collect_artifacts(run_dir)
        _write_manifest(run_dir, artifacts)

        result = ExperimentResult(
            spec_id=job.spec_id,
            job_id=job.id,
            status=_STATUS_MAP[job_status],
            command=job.command,
            environment=_capture_environment(working_dir),
            metrics=metrics,
            config=job.config,
            seed=job.seed,
            artifacts=artifacts,
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
        if not math.isfinite(value):
            raise MalformedMetricsError(
                f"metric {key!r} is not finite: {value!r}"
            )
        metrics[str(key)] = float(value)
    return metrics


def _run_process(
    command: tuple[str, ...],
    working_dir: str,
    env: dict[str, str],
    timeout_seconds: float,
) -> tuple[int | None, str, str, str | None]:
    """Run one job process to completion or timeout.

    The child starts in its own session (POSIX), so a timeout terminates the
    entire process group -- an experiment that forked workers does not leave
    them running after its record says it timed out.
    """
    try:
        process = subprocess.Popen(
            command,
            cwd=working_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        return None, "", "", f"could not launch command: {exc}"
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return process.returncode, stdout, stderr, None
    except subprocess.TimeoutExpired:
        _terminate_group(process)
        stdout, stderr = process.communicate()
        return None, stdout, stderr, f"timed out after {timeout_seconds}s"


def _terminate_group(process: subprocess.Popen[str]) -> None:
    """SIGTERM the whole process group, escalate to SIGKILL after a grace
    period. Falls back to killing the immediate child where process groups
    are unavailable."""
    if os.name != "posix":  # pragma: no cover - windows fallback
        process.kill()
        return
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:  # pragma: no cover - already gone
        return
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:  # pragma: no cover - already gone
        return
    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)


def _within(run_dir: Path, path: Path) -> bool:
    """Whether ``path`` -- symlinks resolved -- stays inside ``run_dir``."""
    try:
        return path.resolve().is_relative_to(run_dir.resolve())
    except OSError:  # pragma: no cover - unresolvable path
        return False


def _check_required_artifacts(
    run_dir: Path, required: tuple[str, ...]
) -> str | None:
    """Missing or escaping declared outputs make the run a failure.

    ``ExperimentJob`` already rejects absolute and ``..`` paths; this is the
    runtime half of the check, which symlinks can only fail, not bypass."""
    for relative in required:
        target = run_dir / relative
        if not target.is_file():
            return f"required artifact(s) not produced: {relative}"
        if not _within(run_dir, target):
            return (
                f"required artifact {relative!r} resolves outside the run "
                f"directory"
            )
    return None


def _collect_artifacts(run_dir: Path) -> tuple[str, ...]:
    """Every file the run left inside its directory. Symlinks resolving
    outside the run directory are excluded: what they point at was not
    produced by this run, and hashing it would launder foreign content into
    the run's manifest."""
    reserved = {
        STDOUT_FILENAME,
        STDERR_FILENAME,
        CONFIG_FILENAME,
        MANIFEST_FILENAME,
    }
    return tuple(
        sorted(
            str(p)
            for p in run_dir.rglob("*")
            if p.is_file() and p.name not in reserved and _within(run_dir, p)
        )
    )


def _write_manifest(run_dir: Path, artifacts: tuple[str, ...]) -> None:
    """Record a sha256 per artifact, keyed by run-dir-relative path, so that
    later edits to experiment outputs are detectable rather than invisible."""
    manifest = {
        str(Path(artifact).relative_to(run_dir)): hashlib.sha256(
            Path(artifact).read_bytes()
        ).hexdigest()
        for artifact in artifacts
    }
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
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
