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

Every job also leaves a durable record of itself, ``job.json``, in its run
directory: written before the process starts and rewritten once it ends.
That is what makes a job findable by a process that did not submit it. A
run interrupted mid-step is resumed by a fresh executor that can still
answer ``status`` and ``collect`` for the job the dead process launched --
and, just as importantly, still refuses to submit that job id a second
time. Without it the in-memory maps die with the process and an expensive
completed job becomes indistinguishable from one that never ran.
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
from ..core.types import ConfigValue
from .executor import (
    DuplicateJobError,
    Executor,
    ExperimentJob,
    JobNotFinishedError,
    JobStatus,
    UnknownJobError,
)

METRICS_FILENAME = "metrics.json"
CONFIG_FILENAME = "config.json"
STDOUT_FILENAME = "stdout.log"
STDERR_FILENAME = "stderr.log"
MANIFEST_FILENAME = "manifest.json"
JOB_RECORD_FILENAME = "job.json"
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


class MalformedJobRecordError(RuntimeError):
    """Raised when a job's durable record cannot be read.

    Loud rather than treated as absence: a record that exists but does not
    parse means a job may have run, and reporting the job as unknown would
    invite resubmitting it."""


class LocalExecutor(Executor):
    def __init__(self, run_root: Path | str) -> None:
        self._run_root = Path(run_root)
        self._run_root.mkdir(parents=True, exist_ok=True)
        self._results: dict[str, ExperimentResult] = {}
        self._status: dict[str, JobStatus] = {}

    def submit(self, job: ExperimentJob) -> str:
        # Both records are consulted: the in-memory one for this process,
        # the durable one for a job some earlier process launched. A
        # derived job id makes that second case reachable, and resubmitting
        # it would be the double-charge this whole mechanism exists to
        # prevent.
        if job.id in self._status or self._record_path(job.id).is_file():
            raise DuplicateJobError(
                f"job {job.id} was already submitted; a retry is a new event — "
                f"construct a new job"
            )
        self._status[job.id] = JobStatus.RUNNING
        run_dir = self._run_root / job.id
        run_dir.mkdir(parents=True, exist_ok=True)
        # Written before the process starts: a job nobody wrote down first
        # is a job no later process can find.
        self._write_record(job, JobStatus.RUNNING, None)

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
        self._write_record(job, job_status, result)
        return job.id

    def status(self, job_id: str) -> JobStatus:
        if job_id in self._status:
            return self._status[job_id]
        return self._durable(job_id)[0]

    def collect(self, job_id: str) -> ExperimentResult:
        found = self._results.get(job_id)
        if found is not None:
            return found
        status, result = self._durable(job_id)
        if result is None:
            raise JobNotFinishedError(
                f"job {job_id} is recorded as {status}; the process that "
                f"submitted it left no result to collect"
            )
        return result

    # -- the durable record ----------------------------------------------------

    def _record_path(self, job_id: str) -> Path:
        return self._run_root / job_id / JOB_RECORD_FILENAME

    def _write_record(
        self,
        job: ExperimentJob,
        status: JobStatus,
        result: ExperimentResult | None,
    ) -> None:
        """Publish what is known about this job so far.

        Rewritten rather than appended: unlike a ledger this is not a
        history, it is the executor's current answer about one job, and
        the run directory beside it already holds the evidence.
        """
        path = self._record_path(job.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_suffix(".tmp")
        scratch.write_text(
            json.dumps(
                _record_payload(job, status, result), indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        scratch.replace(path)

    def _durable(self, job_id: str) -> tuple[JobStatus, ExperimentResult | None]:
        path = self._record_path(job_id)
        if not path.is_file():
            raise UnknownJobError(job_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MalformedJobRecordError(
                f"the durable record of job {job_id} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise MalformedJobRecordError(
                f"the durable record of job {job_id} is not an object"
            )
        try:
            return _record_from(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedJobRecordError(
                f"the durable record of job {job_id} cannot be read: {exc}"
            ) from exc


def _record_payload(
    job: ExperimentJob, status: JobStatus, result: ExperimentResult | None
) -> dict[str, object]:
    """What one job was and how it ended.

    The job's ``env`` is deliberately absent. It is the one field a caller
    may fill with a secret, and a record written to disk is exactly where
    a secret must not be; the run directory records what ran, not what it
    was given to authenticate with.
    """
    payload: dict[str, object] = {
        "job_id": job.id,
        "spec_id": job.spec_id,
        "status": str(status),
        "command": list(job.command),
        "config": dict(job.config),
        "seed": job.seed,
    }
    if result is None:
        return payload
    payload["result"] = {
        "status": str(result.status),
        "environment": {
            "python_version": result.environment.python_version,
            "platform": result.environment.platform,
            "git_commit": result.environment.git_commit,
            "git_dirty": result.environment.git_dirty,
        },
        "metrics": dict(result.metrics),
        "artifacts": list(result.artifacts),
        "logs": list(result.logs),
        "runtime_seconds": result.runtime_seconds,
        "cost": {
            "wall_clock_seconds": result.cost.wall_clock_seconds,
            "gpu_hours": result.cost.gpu_hours,
            "usd": result.cost.usd,
            "model_tokens": result.cost.model_tokens,
        },
        "exit_code": result.exit_code,
        "failure_reason": result.failure_reason,
    }
    return payload


def _record_from(
    payload: Mapping[str, object],
) -> tuple[JobStatus, ExperimentResult | None]:
    status = JobStatus(_text(payload, "status"))
    body = payload.get("result")
    if body is None:
        return status, None
    if not isinstance(body, dict):
        raise TypeError("result must be an object")
    environment = body["environment"]
    cost = body["cost"]
    if not isinstance(environment, dict) or not isinstance(cost, dict):
        raise TypeError("environment and cost must be objects")
    return status, ExperimentResult(
        spec_id=_text(payload, "spec_id"),
        job_id=_text(payload, "job_id"),
        status=ExperimentStatus(_text(body, "status")),
        command=tuple(_strings(payload, "command")),
        environment=Environment(
            python_version=_text(environment, "python_version"),
            platform=_text(environment, "platform"),
            git_commit=_optional_text(environment, "git_commit"),
            git_dirty=_optional_bool(environment, "git_dirty"),
        ),
        metrics=_numbers(body, "metrics"),
        config=_config(payload, "config"),
        seed=_optional_int(payload, "seed"),
        artifacts=tuple(_strings(body, "artifacts")),
        logs=tuple(_strings(body, "logs")),
        runtime_seconds=_number_at(body, "runtime_seconds"),
        cost=ResourceCost(
            wall_clock_seconds=_number_at(cost, "wall_clock_seconds"),
            gpu_hours=_number_at(cost, "gpu_hours"),
            usd=_number_at(cost, "usd"),
            model_tokens=int(_number_at(cost, "model_tokens")),
        ),
        exit_code=_optional_int(body, "exit_code"),
        failure_reason=_optional_text(body, "failure_reason"),
    )


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{key} must be a string or null")


def _optional_bool(payload: Mapping[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"{key} must be a boolean or null")


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise TypeError(f"{key} must be an integer or null")


def _number_at(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)


def _strings(payload: Mapping[str, object], key: str) -> list[str]:
    values = payload[key]
    if not isinstance(values, list) or any(
        not isinstance(item, str) for item in values
    ):
        raise TypeError(f"{key} must be a list of strings")
    return [str(item) for item in values]


def _numbers(payload: Mapping[str, object], key: str) -> dict[str, float]:
    values = payload[key]
    if not isinstance(values, dict):
        raise TypeError(f"{key} must be an object of numbers")
    return {str(name): _number_at(values, name) for name in values}


def _config(
    payload: Mapping[str, object], key: str
) -> dict[str, ConfigValue]:
    values = payload[key]
    if not isinstance(values, dict):
        raise TypeError(f"{key} must be an object")
    read: dict[str, ConfigValue] = {}
    for name, value in values.items():
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"config value {name} is not a scalar")
        read[str(name)] = value
    return read


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
    the run's manifest. The executor's own files are excluded for a
    different reason: the job record is rewritten when the run ends, so
    hashing it would guarantee a manifest that no longer matches."""
    reserved = {
        STDOUT_FILENAME,
        STDERR_FILENAME,
        CONFIG_FILENAME,
        MANIFEST_FILENAME,
        JOB_RECORD_FILENAME,
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
