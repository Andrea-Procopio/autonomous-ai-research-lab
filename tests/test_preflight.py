"""Preflight: deterministic pre-execution checks that stop doomed jobs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.runtime.preflight import (
    PreflightError,
    require_preflight,
    run_preflight,
)
from autonomous_research_lab.runtime.verification import (
    CheckState,
    VerificationCheck,
)


def _spec(seeds: tuple[int, ...] = (7,)) -> ExperimentSpec:
    return ExperimentSpec(
        prediction_id="pred_x",
        objective="measure",
        procedure="run",
        metrics=("m",),
        seeds=seeds,
    )


def _job(**kwargs: object) -> ExperimentJob:
    return ExperimentJob(
        spec_id="exp_x",
        command=(sys.executable, "-c", "pass"),
        **kwargs,  # type: ignore[arg-type]
    )


def _state_of(
    report_checks: tuple[VerificationCheck, ...], name: str
) -> CheckState:
    return next(c.state for c in report_checks if c.name == f"preflight:{name}")


def test_a_well_formed_job_passes_preflight() -> None:
    report = run_preflight(_job(seed=7), _spec())
    assert report.passed
    assert _state_of(report.checks, "command_resolvable") is CheckState.PASS
    assert _state_of(report.checks, "seed_propagated") is CheckState.PASS


def test_unresolvable_command_fails_before_any_execution() -> None:
    job = ExperimentJob(
        spec_id="exp_x", command=("definitely-not-a-binary-anywhere",), seed=7
    )
    report = run_preflight(job, _spec())
    assert not report.passed
    assert _state_of(report.checks, "command_resolvable") is CheckState.FAIL


def test_missing_declared_input_path_fails(tmp_path: Path) -> None:
    job = _job(seed=7, config={"data_path": str(tmp_path / "nope.csv")})
    report = run_preflight(job, _spec())
    assert _state_of(report.checks, "config_paths_exist") is CheckState.FAIL

    existing = tmp_path / "data.csv"
    existing.write_text("x")
    ok = run_preflight(_job(seed=7, config={"data_path": str(existing)}), _spec())
    assert _state_of(ok.checks, "config_paths_exist") is CheckState.PASS


def test_path_convention_is_opt_in_not_inferred() -> None:
    """Config values that do not follow the *_path/*_file/*_dir convention
    are not guessed at: the check reports NOT_APPLICABLE, never a verdict."""
    report = run_preflight(_job(seed=7, config={"value": "hello"}), _spec())
    assert _state_of(report.checks, "config_paths_exist") is CheckState.NOT_APPLICABLE


def test_seed_declared_but_not_propagated_fails() -> None:
    report = run_preflight(_job(), _spec(seeds=(1, 2)))
    assert _state_of(report.checks, "seed_propagated") is CheckState.FAIL


def test_undeclared_seed_is_uncertain_not_failed() -> None:
    report = run_preflight(_job(seed=99), _spec(seeds=(1, 2)))
    assert _state_of(report.checks, "seed_propagated") is CheckState.UNCERTAIN
    assert report.passed  # uncertain does not block execution


def test_no_spec_makes_spec_checks_not_applicable() -> None:
    report = run_preflight(_job())
    assert _state_of(report.checks, "seed_propagated") is CheckState.NOT_APPLICABLE


def test_require_preflight_raises_with_the_full_report() -> None:
    job = ExperimentJob(spec_id="exp_x", command=("no-such-binary-xyz",))
    with pytest.raises(PreflightError) as excinfo:
        require_preflight(job, _spec())
    assert "command_resolvable" in str(excinfo.value)
    assert not excinfo.value.report.passed


def test_require_preflight_returns_the_report_on_success() -> None:
    report = require_preflight(_job(seed=7), _spec())
    assert report.passed


# -- the hidden-.pth diagnosis -------------------------------------------------

_DARWIN = sys.platform == "darwin"


def _shim_job(**kwargs: object) -> ExperimentJob:
    """The exact command shape ContainerBinding produces: the lab's own
    interpreter importing a lab module, which depends on the .pth."""
    return ExperimentJob(
        spec_id="exp_x",
        command=(
            sys.executable,
            "-m",
            "autonomous_research_lab.execution.container_shim",
            "--image",
            "img",
        ),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.skipif(not _DARWIN, reason="UF_HIDDEN is a macOS/BSD file flag")
def test_a_hidden_pth_file_fails_preflight_with_the_remediation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import stat as stat_module

    from autonomous_research_lab.runtime import preflight

    # The skipif above handles runtime; the platform block is for the
    # type checker, whose Linux stubs have no os.chflags — mypy prunes
    # platform-guarded branches instead of checking or flagging them.
    if sys.platform == "darwin":
        pth = tmp_path / "_editable_fake.pth"
        pth.write_text("/nowhere/src\n")
        # A tmp file, never the real venv.
        os.chflags(pth, stat_module.UF_HIDDEN)
        monkeypatch.setattr(
            preflight, "_site_package_dirs", lambda: (str(tmp_path),)
        )

        check = preflight.PthFilesVisible().check(_shim_job(seed=7), _spec())

        assert check.state is CheckState.FAIL
        assert "_editable_fake.pth" in check.detail
        assert "site.py" in check.detail
        # The remediation is named, not applied.
        assert "chflags" in check.detail


@pytest.mark.skipif(not _DARWIN, reason="UF_HIDDEN is a macOS/BSD file flag")
def test_visible_pth_files_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autonomous_research_lab.runtime import preflight

    (tmp_path / "_editable_fake.pth").write_text("/nowhere/src\n")
    monkeypatch.setattr(
        preflight, "_site_package_dirs", lambda: (str(tmp_path),)
    )
    check = preflight.PthFilesVisible().check(_shim_job(seed=7), _spec())
    assert check.state is CheckState.PASS


def test_platforms_without_file_flags_stay_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no stat result exposes st_flags the check yields no verdict —
    inert, never a manufactured PASS."""
    from autonomous_research_lab.runtime import preflight

    monkeypatch.setattr(
        preflight, "_hidden_pth_files", lambda directories: None
    )
    check = preflight.PthFilesVisible().check(_shim_job(seed=7), _spec())
    assert check.state is CheckState.NOT_APPLICABLE


def test_jobs_that_do_not_import_the_lab_are_not_applicable() -> None:
    """HostPythonBinding jobs run a trusted source directly and never
    import the package, so the .pth's health is not theirs."""
    from autonomous_research_lab.runtime.preflight import PthFilesVisible

    direct = _job(seed=7)  # (sys.executable, "-c", "pass")
    check = PthFilesVisible().check(direct, _spec())
    assert check.state is CheckState.NOT_APPLICABLE

    foreign = ExperimentJob(
        spec_id="exp_x",
        command=("definitely-not-python", "-m", "autonomous_research_lab.x"),
        seed=7,
    )
    assert (
        PthFilesVisible().check(foreign, _spec()).state
        is CheckState.NOT_APPLICABLE
    )
