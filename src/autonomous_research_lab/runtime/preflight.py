"""Preflight: cheap deterministic checks that run before expensive execution.

A job that cannot possibly succeed — a command that does not resolve, a
dataset path that does not exist, a seed the spec declared but the job never
received — should fail in milliseconds, not after an hour of compute. Every
check here is Tier 0: pure inspection of the job and spec, no model calls,
no process launched.

Checks are deliberately not universal. Each one decides for itself whether
it applies (``NOT_APPLICABLE`` otherwise), and the default list is small;
experiment-specific checks (smoke tests, shape/range invariants on a tiny
input) implement the same :class:`PreflightCheck` protocol and are passed in
by whoever builds the job.

This module depends on ``core`` only. It inspects jobs through the
structural :class:`JobLike` protocol rather than importing the execution
package, mirroring how ``runtime.validation`` mirrors the executor's
manifest contract without importing it — ``ExperimentJob`` satisfies the
protocol as-is.
"""

from __future__ import annotations

import itertools
import shutil
import site
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from ..core.experiment import ExperimentSpec
from ..core.types import ConfigValue
from .verification import (
    CheckState,
    ValidityDimension,
    VerificationCheck,
    VerificationReport,
)

#: Config keys whose string values name filesystem inputs. A convention, not
#: an inference: jobs that want their paths preflighted use these suffixes.
PATH_KEY_SUFFIXES = ("_path", "_file", "_dir")


class JobLike(Protocol):
    """The slice of an execution job preflight needs to see."""

    @property
    def command(self) -> tuple[str, ...]: ...

    @property
    def config(self) -> Mapping[str, ConfigValue]: ...

    @property
    def seed(self) -> int | None: ...

    @property
    def working_dir(self) -> str | None: ...

    @property
    def required_artifacts(self) -> tuple[str, ...]: ...


class PreflightCheck(Protocol):
    def check(self, job: JobLike, spec: ExperimentSpec | None) -> VerificationCheck: ...


class PreflightError(RuntimeError):
    """Raised by :func:`require_preflight` when a deterministic preflight
    check fails. Carries the full report; the execution never started, so a
    handler bills no compute for it."""

    def __init__(self, report: VerificationReport):
        failed = ", ".join(c.name for c in report.failures)
        super().__init__(f"preflight failed: {failed}")
        self.report = report


def _check(name: str, state: CheckState, detail: str = "") -> VerificationCheck:
    return VerificationCheck(
        dimension=ValidityDimension.EXECUTION,
        name=f"preflight:{name}",
        state=state,
        detail=detail,
    )


class CommandResolvable:
    """The command's executable must exist — on PATH or as a file."""

    def check(
        self,
        job: JobLike,
        spec: ExperimentSpec | None,  # noqa: ARG002 - job-only check
    ) -> VerificationCheck:
        executable = job.command[0]
        if shutil.which(executable) is not None or Path(executable).is_file():
            return _check("command_resolvable", CheckState.PASS)
        return _check(
            "command_resolvable",
            CheckState.FAIL,
            f"command {executable!r} resolves to no executable",
        )


class WorkingDirectoryExists:
    """A declared working directory must exist. ``None`` means the executor
    provides an isolated one, so there is nothing to check."""

    def check(
        self,
        job: JobLike,
        spec: ExperimentSpec | None,  # noqa: ARG002 - job-only check
    ) -> VerificationCheck:
        if job.working_dir is None:
            return _check(
                "working_dir_exists",
                CheckState.NOT_APPLICABLE,
                "executor provides a job-private workspace",
            )
        if Path(job.working_dir).is_dir():
            return _check("working_dir_exists", CheckState.PASS)
        return _check(
            "working_dir_exists",
            CheckState.FAIL,
            f"working directory {job.working_dir!r} does not exist",
        )


class ConfigPathsExist:
    """Config entries following the ``*_path`` / ``*_file`` / ``*_dir``
    convention must name existing filesystem inputs."""

    def check(
        self,
        job: JobLike,
        spec: ExperimentSpec | None,  # noqa: ARG002 - job-only check
    ) -> VerificationCheck:
        declared = {
            key: value
            for key, value in job.config.items()
            if isinstance(value, str) and key.endswith(PATH_KEY_SUFFIXES)
        }
        if not declared:
            return _check(
                "config_paths_exist",
                CheckState.NOT_APPLICABLE,
                "no path-convention config keys declared",
            )
        missing = sorted(
            f"{key}={value!r}"
            for key, value in declared.items()
            if not Path(value).exists()
        )
        if missing:
            return _check(
                "config_paths_exist",
                CheckState.FAIL,
                f"missing input(s): {', '.join(missing)}",
            )
        return _check("config_paths_exist", CheckState.PASS)


class SeedPropagated:
    """A spec that declares seeds must see one reach the job."""

    def check(self, job: JobLike, spec: ExperimentSpec | None) -> VerificationCheck:
        if spec is None or not spec.seeds:
            return _check(
                "seed_propagated", CheckState.NOT_APPLICABLE, "spec declares no seeds"
            )
        if job.seed is None:
            return _check(
                "seed_propagated",
                CheckState.FAIL,
                f"spec declares seeds {spec.seeds}, job carries none",
            )
        if job.seed not in spec.seeds:
            return _check(
                "seed_propagated",
                CheckState.UNCERTAIN,
                f"job seed {job.seed} is not among declared seeds {spec.seeds}",
            )
        return _check("seed_propagated", CheckState.PASS)


class PthFilesVisible:
    """The launching interpreter's site-packages ``.pth`` files must not
    carry the macOS hidden flag.

    CPython >= 3.11.9 hardening makes ``site.py`` silently *skip* a hidden
    ``.pth`` file, so an editable install's source path never reaches
    ``sys.path`` and every child spawned as ``[sys.executable, "-m", ...]``
    dies with ``ModuleNotFoundError`` — observed live on 2026-08-18, when
    iCloud Drive Desktop sync re-flagged ``.venv`` between runs and the
    seed-29 ridge replication failed at launch. The cause is external to
    the repository, so this check only diagnoses: it never mutates flags,
    files, or host configuration. Remediation belongs to the operator:
    ``chflags -R nohidden <venv>`` clears the flags until the sync daemon
    reapplies them; the durable fix is moving the repository out of an
    iCloud-synced folder.
    """

    def check(
        self,
        job: JobLike,
        spec: ExperimentSpec | None,  # noqa: ARG002 - job-only check
    ) -> VerificationCheck:
        if not _imports_the_lab_package(job.command):
            return _check(
                "pth_files_visible",
                CheckState.NOT_APPLICABLE,
                "job does not import the lab package through this "
                "interpreter's site-packages",
            )
        hidden = _hidden_pth_files(_site_package_dirs())
        if hidden is None:
            return _check(
                "pth_files_visible",
                CheckState.NOT_APPLICABLE,
                "no .pth stat result exposes a hidden flag (non-BSD "
                "platform, or no .pth files under site-packages)",
            )
        if hidden:
            return _check(
                "pth_files_visible",
                CheckState.FAIL,
                f"hidden .pth file(s): {', '.join(hidden)}; CPython >= "
                f"3.11.9 site.py skips hidden .pth files, so the paths "
                f"they add never reach sys.path and child interpreters "
                f"raise ModuleNotFoundError; the flag is set externally "
                f"(iCloud Desktop sync observed) — remediate outside the "
                f"lab: 'chflags -R nohidden <venv>' clears it until the "
                f"sync daemon reapplies it; durably, move the repository "
                f"out of an iCloud-synced folder. This check changes "
                f"nothing itself.",
            )
        return _check("pth_files_visible", CheckState.PASS)


def _imports_the_lab_package(command: tuple[str, ...]) -> bool:
    """Whether the job runs a lab module through this interpreter — the
    exact shape that depends on the editable install's ``.pth``. A trusted
    experiment source run directly (``python experiment.py``) never
    imports the package, so its jobs are exempt."""
    if len(command) < 3:
        return False
    try:
        same = Path(command[0]).resolve() == Path(sys.executable).resolve()
    except OSError:
        return False
    if not same:
        return False
    for flag, module in itertools.pairwise(command[1:]):
        if flag == "-m" and module.split(".")[0] == "autonomous_research_lab":
            return True
    return False


def _site_package_dirs() -> tuple[str, ...]:
    """The running interpreter's site-packages directories; empty when the
    interpreter does not expose them (some embedded builds)."""
    getter = getattr(site, "getsitepackages", None)
    if not callable(getter):
        return ()
    try:
        listed = getter()
    except OSError:
        return ()
    return tuple(entry for entry in listed if Path(entry).is_dir())


def _hidden_pth_files(directories: Sequence[str]) -> tuple[str, ...] | None:
    """The hidden-flagged ``.pth`` files under ``directories``, or ``None``
    when no stat result exposed ``st_flags`` at all — a platform that
    cannot express the condition yields no verdict, not a manufactured
    PASS."""
    saw_flags = False
    hidden: list[str] = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.pth")):
            try:
                flags = getattr(path.lstat(), "st_flags", None)
            except OSError:
                continue
            if not isinstance(flags, int):
                continue
            saw_flags = True
            if flags & stat.UF_HIDDEN:
                hidden.append(str(path))
    return tuple(hidden) if saw_flags else None


DEFAULT_PREFLIGHT_CHECKS: tuple[PreflightCheck, ...] = (
    CommandResolvable(),
    WorkingDirectoryExists(),
    ConfigPathsExist(),
    SeedPropagated(),
    PthFilesVisible(),
)


def run_preflight(
    job: JobLike,
    spec: ExperimentSpec | None = None,
    *,
    checks: Sequence[PreflightCheck] = DEFAULT_PREFLIGHT_CHECKS,
) -> VerificationReport:
    """Run every check; the report says what applied and what failed."""
    return VerificationReport(
        checks=tuple(check.check(job, spec) for check in checks)
    )


def require_preflight(
    job: JobLike,
    spec: ExperimentSpec | None = None,
    *,
    checks: Sequence[PreflightCheck] = DEFAULT_PREFLIGHT_CHECKS,
) -> VerificationReport:
    """Run preflight and raise :class:`PreflightError` on any failure —
    the executor-boundary guard that keeps a doomed job from spending."""
    report = run_preflight(job, spec, checks=checks)
    if not report.passed:
        raise PreflightError(report)
    return report
