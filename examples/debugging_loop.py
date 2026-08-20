"""The scientific-debugging system, end to end, on three contrasting cases.

Everything here is rule-based test doubles standing where model-backed roles
will sit — this demo shows the *machinery* (diagnosis, bounded repair,
verification, the negative-result gate, the methodology gate), not
autonomous model-driven research.

Case A — **broken experiment**: the run crashes on a missing data file. The
deterministic classifier names the failure, the bounded debug loop repairs
the path and reruns, and a valid execution is recovered. Debugging succeeded
— which says nothing about which way the science came out.

Case B — **correct experiment, genuine negative**: the run is valid, the
positive control passes, methodology was approved before execution, and the
pre-registered prediction still fails. Verification passes on every
dimension, so the negative is promoted to *verified scientific evidence* —
and the debug loop is never touched, because a disappointing result is not
a defect.

Case C — **technically correct, methodologically invalid**: the design
measures the wrong metric for the stated question. The methodology gate
rejects it before a single second of compute is spent: no debugging, no
false scientific conclusion — the surfaced response is *redesign*.

Run with::

    python examples/debugging_loop.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import mkdtemp

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.experiment import ExperimentResult, ExperimentSpec
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.proposals import Proposal, ResultProposal
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.failure_classifier import FailureDiagnosis
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.orchestration.debug_loop import (
    ExperimentDebugger,
    RepairProposal,
)
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import ResearchRuntime, StepReport
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.preflight import require_preflight
from autonomous_research_lab.runtime.verification import (
    CheckState,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)
from autonomous_research_lab.runtime.verification_store import (
    FileVerificationStore,
)

QUESTION = ResearchQuestion(
    text="Does the sampler's draw stream lean toward heads?"
)
HYPOTHESIS = Hypothesis(
    statement="The stream is biased toward heads (rate >= 0.60).",
    question_id=QUESTION.id,
)

#: The experiment process: reads a data file of 0/1 draws, reports the heads
#: rate plus a positive-control metric (a probe the implementation must get
#: exactly right if it is counting at all).
_EXPERIMENT = """
import json, os, pathlib
cfg = json.loads(pathlib.Path(os.environ["ARL_CONFIG"]).read_text())
draws = [int(t) for t in pathlib.Path(cfg["data_path"]).read_text().split()]
probe = [1, 1, 1, 1]  # a known all-heads probe: its rate must be 1.0
metrics = {
    "heads_rate": sum(draws) / len(draws),
    "probe_rate": sum(probe) / len(probe),
    "n_draws": len(draws),
}
run_dir = pathlib.Path(os.environ["ARL_RUN_DIR"])
(run_dir / "metrics.json").write_text(json.dumps(metrics))
"""

PROBE_CONTROL = PositiveControl(
    name="all_heads_probe",
    metric="probe_rate",
    comparator=Comparator.APPROXIMATELY,
    threshold=1.0,
    tolerance=1e-9,
    rationale="an implementation that miscounts cannot rate the probe 1.0",
)


# -- rule-based doubles -------------------------------------------------------


class DemoEngineer(ResearchRole):
    """Executor seat: preflights, then runs the experiment script against a
    configured data path (which case A deliberately breaks)."""

    def __init__(
        self, executor: LocalExecutor, data_path: Path, *, preflight: bool
    ) -> None:
        self._executor = executor
        self._data_path = data_path
        self._preflight = preflight

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_ENGINEER

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset({ResearchActionType.RUN_EXPERIMENT})

    def suitability(
        self,
        state: ResearchState,  # noqa: ARG002 - demo roles fit everything
        action: ResearchAction,  # noqa: ARG002
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        (spec,) = invocation.context.experiments
        job = ExperimentJob(
            spec_id=spec.id,
            command=(sys.executable, "-c", _EXPERIMENT),
            config={"data_path": str(self._data_path)},
            seed=spec.seeds[0],
            timeout_seconds=60.0,
        )
        if self._preflight:
            require_preflight(job, spec)
        result = self._executor.collect(self._executor.submit(job))
        return (ResultProposal(result=result, proposer="demo:engineer"),)


class PathRepair:
    """Rule-based repair strategy: on a missing-path failure, point the job
    at the real data file and rerun."""

    def __init__(self, correct_path: Path) -> None:
        self._correct_path = correct_path

    def propose(
        self,
        spec: ExperimentSpec,
        failed: ExperimentResult,  # noqa: ARG002 - the rule needs the diagnosis
        diagnosis: FailureDiagnosis,
        attempt_number: int,  # noqa: ARG002
    ) -> RepairProposal | None:
        return RepairProposal(
            job=ExperimentJob(
                spec_id=spec.id,
                command=(sys.executable, "-c", _EXPERIMENT),
                config={"data_path": str(self._correct_path)},
                seed=spec.seeds[0],
                timeout_seconds=60.0,
            ),
            rationale=(
                f"diagnosis was {diagnosis.category}: repoint data_path at "
                f"the file that actually exists"
            ),
        )


class RuleBasedMethodologist:
    """Approves designs whose measured metric can answer the question; the
    heads-bias question needs ``heads_rate``, not a latency reading."""

    def review(
        self,
        spec: ExperimentSpec,  # noqa: ARG002 - the rule reads the prediction
        prediction: Prediction | None,
        *,
        objective: str,  # noqa: ARG002
    ) -> VerificationCheck:
        valid = prediction is not None and prediction.metric == "heads_rate"
        return VerificationCheck(
            dimension=ValidityDimension.METHODOLOGY,
            name="methodological_validity",
            state=CheckState.PASS if valid else CheckState.FAIL,
            detail=(
                "the prediction is stated in the metric the question is about"
                if valid
                else (
                    f"the design tests "
                    f"{prediction.metric if prediction else 'nothing'!r}, "
                    f"which cannot answer a question about heads bias"
                )
            ),
        )


class RuleBasedVerifier:
    """Implementation verifier double: trusts the probe-control arithmetic
    it is shown; with the control green it finds nothing to falsify."""

    def verify(
        self,
        spec: ExperimentSpec,  # noqa: ARG002 - rule reads only the checks
        result: ExperimentResult,  # noqa: ARG002
        prediction: Prediction | None,  # noqa: ARG002
        checks: tuple[VerificationCheck, ...],
    ) -> VerificationCheck:
        control_failed = any(
            c.name.startswith("positive_control") and c.state is CheckState.FAIL
            for c in checks
        )
        return VerificationCheck(
            dimension=ValidityDimension.IMPLEMENTATION,
            name="implementation_faithfulness",
            state=CheckState.FAIL if control_failed else CheckState.PASS,
            detail=(
                "a failed positive control is a silent-bug signal"
                if control_failed
                else "spec, config and metrics agree; probe control is green"
            ),
        )


# -- wiring -------------------------------------------------------------------


def _state(spec: ExperimentSpec, prediction: Prediction) -> ResearchState:
    return (
        ResearchState(
            objective="Characterize the sampler's draw stream.",
            budget=ResearchBudget(
                wall_clock_seconds=3600.0, usd=5.0, model_tokens=100_000
            ),
        )
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(prediction)
        .add_experiment(spec)
    )


def _design(metric: str) -> tuple[ExperimentSpec, Prediction]:
    prediction = Prediction(
        hypothesis_id=HYPOTHESIS.id,
        condition="the recorded draw stream",
        metric=metric,
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=0.60,
        expectation=f"{metric} is at least 0.60",
    )
    spec = ExperimentSpec(
        prediction_id=prediction.id,
        objective="Estimate the stream's heads rate.",
        procedure="Read the recorded draws and report the observed rates.",
        metrics=(metric, "probe_rate", "n_draws"),
        seeds=(7,),
    )
    return spec, prediction


def _runtime(
    root: Path, *, data_path: Path, repair_to: Path | None, preflight: bool = True
) -> ResearchRuntime:
    executor = LocalExecutor(root / "runs")
    config = RuntimeConfig(max_debug_attempts=3)
    return ResearchRuntime(
        config=config,
        director=RuleBasedFrontierDirector(),
        roles={
            RoleName.RESEARCH_ENGINEER: DemoEngineer(
                executor,
                data_path,
                preflight=preflight and config.preflight_enabled,
            ),
        },
        store=FileEvidenceStore(root),
        debugger=(
            ExperimentDebugger(
                executor=executor,
                strategy=PathRepair(repair_to),
                max_attempts=config.max_debug_attempts,
            )
            if repair_to is not None
            else None
        ),
        methodology_reviewer=RuleBasedMethodologist(),
        implementation_verifier=RuleBasedVerifier(),
        control_source=lambda spec: (PROBE_CONTROL,),  # noqa: ARG005
        # Verdicts are durable: one JSON per result, reloadable later.
        verifications=FileVerificationStore(root / "verifications"),
    )


def _print_step(title: str, report: StepReport) -> None:
    print(f"== {title} ==")
    for note in report.notes:
        print(f"  note: {note}")
    for verification in report.verification:
        for check in verification.checks:
            print(f"  check [{check.state.value:>14}] {check.name}")
    if report.debug_attempts:
        print(
            f"  debug attempts: {report.debug_attempts} "
            f"(resolved: {report.debug_resolved})"
        )
    print()


def main() -> None:
    root = Path(mkdtemp())
    data = root / "draws.txt"
    # 40 heads in 80 draws: honestly fair, so the >=0.60 prediction fails.
    data.write_text(" ".join(["1", "0"] * 40))

    # A: the config points at a path that does not exist; the classifier
    # diagnoses it and the bounded debug loop repairs it. Preflight would
    # (correctly) stop this doomed job before launch; case A is about
    # repairing a mid-flight failure, so this engineer runs without it.
    spec, prediction = _design("heads_rate")
    runtime = _runtime(
        root / "case_a",
        data_path=root / "missing.txt",
        repair_to=data,
        preflight=False,
    )
    report_a = runtime.step(_state(spec, prediction))
    _print_step("case A: broken experiment -> diagnose -> repair -> valid", report_a)

    # B: same (correct) design against the real data; the genuine negative
    # passes every verification dimension and is preserved as evidence.
    runtime_b = _runtime(root / "case_b", data_path=data, repair_to=data)
    report_b = runtime_b.step(_state(spec, prediction))
    _print_step("case B: valid experiment, genuine negative result", report_b)

    # C: a design measuring the wrong metric for the question. The
    # methodology gate rejects it before execution.
    bad_spec, bad_prediction = _design("probe_rate")
    runtime_c = _runtime(root / "case_c", data_path=data, repair_to=data)
    report_c = runtime_c.step(_state(bad_spec, bad_prediction))
    _print_step("case C: methodologically invalid -> redesign, no debug", report_c)

    print("summary")
    print(f"  A resolved by debugging: {report_a.debug_resolved}")
    accepted = any(
        "verified scientific negative" in note for note in report_b.notes
    )
    print(
        f"  B negative verdict: {'accepted' if accepted else 'deferred'} "
        f"(debug attempts: {report_b.debug_attempts})"
    )
    print(
        f"  C executed: {bool(report_c.state.results)} "
        f"(methodology rejected before execution)"
    )
    stored = FileVerificationStore(root / "case_b" / "verifications").records()
    for verdict in stored:
        print(
            f"  durable verdict on disk: {verdict.result_id} -> "
            f"{verdict.validity} ({verdict.standing})"
        )


if __name__ == "__main__":
    main()
