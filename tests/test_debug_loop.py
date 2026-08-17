"""The bounded debug loop, against real processes: repair, audit, bound —
and the structural refusal to debug scientific outcomes."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autonomous_research_lab.core.experiment import (
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.failure_classifier import (
    FailureCategory,
    FailureDiagnosis,
    diagnose_failure,
)
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.orchestration.debug_loop import (
    ExperimentDebugger,
    RepairProposal,
    RepairStrategy,
    ScientificOutcomeError,
    is_debuggable,
)

_OK = (
    "import json, os, pathlib; "
    "pathlib.Path(os.environ['ARL_RUN_DIR'], 'metrics.json')"
    ".write_text(json.dumps({'m': 1.0}))"
)
_CRASH = "raise SystemExit(3)"


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        prediction_id="pred_x",
        objective="measure",
        procedure="run",
        metrics=("m",),
        seeds=(7,),
    )


def _job(spec: ExperimentSpec, script: str) -> ExperimentJob:
    return ExperimentJob(
        spec_id=spec.id,
        command=(sys.executable, "-c", script),
        seed=7,
        timeout_seconds=30.0,
    )


@dataclass
class ScriptStrategy(RepairStrategy):
    """Rule-based repair: propose the scripts in order, recording rationale."""

    spec: ExperimentSpec
    scripts: tuple[str, ...]
    proposals: list[int] = field(default_factory=list)

    def propose(
        self,
        spec: ExperimentSpec,
        failed: ExperimentResult,
        diagnosis: FailureDiagnosis,
        attempt_number: int,
    ) -> RepairProposal | None:
        self.proposals.append(attempt_number)
        index = len(self.proposals) - 1
        if index >= len(self.scripts):
            return None
        return RepairProposal(
            job=_job(self.spec, self.scripts[index]),
            rationale=f"attempt {attempt_number}: swap in script {index}",
        )


def _failed_run(executor: LocalExecutor, spec: ExperimentSpec) -> ExperimentResult:
    result = executor.collect(executor.submit(_job(spec, _CRASH)))
    assert result.status is ExperimentStatus.FAILED
    return result


def test_a_repairable_failure_is_repaired_and_fully_audited(
    tmp_path: Path,
) -> None:
    spec = _spec()
    executor = LocalExecutor(tmp_path / "runs")
    failed = _failed_run(executor, spec)
    strategy = ScriptStrategy(spec, (_CRASH, _OK))
    debugger = ExperimentDebugger(executor=executor, strategy=strategy)

    session = debugger.debug(spec, failed)

    assert session.resolved
    assert len(session.attempts) == 2
    first, second = session.attempts
    # Retry numbering, diagnosis, rationale, and each rerun's own result
    # (with its cost) survive per attempt.
    assert (first.number, second.number) == (1, 2)
    assert first.diagnosis is not None
    assert first.diagnosis.category is FailureCategory.NONZERO_EXIT
    assert "swap in script" in first.repair_rationale
    assert not first.result.succeeded
    assert second.result.succeeded
    assert second.result.cost.wall_clock_seconds > 0.0
    # The initial failure's record and run directory are untouched.
    assert session.initial_result_id == failed.id
    assert Path(failed.logs[1]).exists()
    # Every rerun was a new job occurrence with its own run directory.
    run_dirs = {Path(a.result.logs[0]).parent for a in session.attempts}
    assert len(run_dirs) == 2
    assert Path(failed.logs[0]).parent not in run_dirs


def test_the_debug_loop_stops_at_the_configured_bound(tmp_path: Path) -> None:
    """Regression case G: persistent failure stops after max attempts."""
    spec = _spec()
    executor = LocalExecutor(tmp_path / "runs")
    failed = _failed_run(executor, spec)
    strategy = ScriptStrategy(spec, (_CRASH,) * 10)
    debugger = ExperimentDebugger(
        executor=executor, strategy=strategy, max_attempts=3
    )

    session = debugger.debug(spec, failed)

    assert not session.resolved
    assert len(session.attempts) == 3
    assert "limit of 3" in session.stop_reason
    assert strategy.proposals == [1, 2, 3]


def test_callers_can_narrow_but_never_widen_the_bound(tmp_path: Path) -> None:
    spec = _spec()
    executor = LocalExecutor(tmp_path / "runs")
    failed = _failed_run(executor, spec)
    debugger = ExperimentDebugger(
        executor=executor,
        strategy=ScriptStrategy(spec, (_CRASH,) * 10),
        max_attempts=2,
    )
    session = debugger.debug(spec, failed, max_attempts=5)
    assert len(session.attempts) == 2  # the configured ceiling held


def test_a_strategy_that_gives_up_stops_the_loop(tmp_path: Path) -> None:
    spec = _spec()
    executor = LocalExecutor(tmp_path / "runs")
    failed = _failed_run(executor, spec)
    debugger = ExperimentDebugger(
        executor=executor, strategy=ScriptStrategy(spec, ())
    )
    session = debugger.debug(spec, failed)
    assert not session.resolved
    assert session.attempts == ()
    assert "no further fix" in session.stop_reason


def test_a_completed_result_is_refused_outright(tmp_path: Path) -> None:
    """The critical invariant: `while result_is_scientifically_bad: debug()`
    cannot exist against this interface. A completed run — however
    disappointing its metrics — is not debuggable."""
    spec = _spec()
    executor = LocalExecutor(tmp_path / "runs")
    completed = executor.collect(executor.submit(_job(spec, _OK)))
    assert completed.succeeded
    debugger = ExperimentDebugger(
        executor=executor, strategy=ScriptStrategy(spec, (_OK,))
    )
    with pytest.raises(ScientificOutcomeError, match="scientific evidence"):
        debugger.debug(spec, completed)


def test_is_debuggable_follows_the_diagnosis_never_the_metrics(
    tmp_path: Path,
) -> None:
    spec = _spec()
    executor = LocalExecutor(tmp_path / "runs")
    failed = _failed_run(executor, spec)
    assert is_debuggable(diagnose_failure(failed))
    completed = executor.collect(executor.submit(_job(spec, _OK)))
    assert not is_debuggable(diagnose_failure(completed))


def test_a_repair_proposal_may_not_switch_experiments(tmp_path: Path) -> None:
    spec = _spec()
    other = ExperimentSpec(
        prediction_id="pred_y",
        objective="other",
        procedure="other",
        metrics=("m",),
    )
    executor = LocalExecutor(tmp_path / "runs")
    failed = _failed_run(executor, spec)

    class Swapping(ScriptStrategy):
        def propose(
            self,
            spec: ExperimentSpec,
            failed: ExperimentResult,
            diagnosis: FailureDiagnosis,
            attempt_number: int,
        ) -> RepairProposal | None:
            return RepairProposal(
                job=_job(other, _OK), rationale="sneakily run something else"
            )

    debugger = ExperimentDebugger(
        executor=executor, strategy=Swapping(spec, ())
    )
    with pytest.raises(ValueError, match="not the spec being debugged"):
        debugger.debug(spec, failed)
