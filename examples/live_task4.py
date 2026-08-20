"""Task 4 live trajectory: planning -> engineering -> execution -> planning.

One autonomous trajectory, live, with nothing mocked::

    manually supplied research question + human-authored baseline design
      -> the baseline runs through Muse + the disposable container and is
         verified normally (genuine admissible evidence, nothing hand-built)
      -> ModelBackedPlanner (Muse Spark 1.2, one structured decision over a
         deterministic projection of the authoritative state)
      -> deterministic planning gate -> atomic governed commit
      -> ModelBackedEngineer implements the planner's experiment with the
         planner-selected trusted template -> contained execution
      -> Phase 1 verification -> scientific admissibility
      -> the planner is invoked again over the updated state

The planner receives genuine alternatives: a trusted catalog of three
templates with distinct metric vocabularies, an unused declared seed on
the baseline (a real replication gap), the option to ablate the baseline
procedure, and a typed stop. Nothing prompts it toward any one of them.

Requires: a running Docker daemon (colima works), the pinned image already
pulled, and MUSE_API_KEY (or MODEL_API_KEY) in the environment. Run with::

    python -m examples.live_task4 --run-root /path/to/run

Run book: ``colima start`` before, ``colima stop`` after; the driver
manages neither. Exits 0 only if the trajectory produced at least two
accepted planning decisions and the first non-stop decision traversed the
engineer, the container, and verification with intact provenance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.execution.binding import ContainerBinding
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.execution.runner import DirectJobRunner
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.orchestration.planning import PlanningDirector
from autonomous_research_lab.orchestration.trajectory import (
    JsonlTrajectoryLogger,
)
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.roles.base import RoleName
from autonomous_research_lab.roles.engineer import (
    ImplementationTemplate,
    ModelBackedEngineer,
)
from autonomous_research_lab.roles.planner import (
    ModelBackedPlanner,
    TemplateCapability,
    TemplateCatalog,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.implementation_store import (
    ImplementationStore,
)
from autonomous_research_lab.runtime.metrics import JsonlRuntimeMetrics
from autonomous_research_lab.runtime.muse import (
    KEY_ENV_VARS,
    MUSE_SPARK_1_2,
    MuseSparkProvider,
)
from autonomous_research_lab.runtime.planning_store import (
    PlanningAction,
    PlanningStore,
)
from autonomous_research_lab.runtime.preflight import PthFilesVisible
from autonomous_research_lab.runtime.providers import UsageLedger
from autonomous_research_lab.runtime.verification import (
    CheckState,
    PositiveControl,
)
from autonomous_research_lab.runtime.verification_store import (
    FileVerificationStore,
    ScientificAdmissibility,
)
from examples.trajectory_campaign import StructuralMethodology

DEFAULT_IMAGE = (
    "python@sha256:"
    "9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7"
)

_EXPERIMENTS = Path(__file__).parent / "experiments"

TINY_OVERFIT_CONTROL = PositiveControl(
    name="tiny_subset_overfit",
    metric="tiny_subset_accuracy",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.99,
    rationale="a faithful trainer must fit 8 linearly separable points",
)

OBJECTIVE = (
    "Characterize when a from-scratch logistic-regression learner recovers "
    "a synthetic two-blob decision boundary, and how robust that recovery "
    "is."
)

BASELINE_PROCEDURE = (
    "Generate a two-class 2D synthetic dataset with random.Random(ARL_SEED): "
    "400 training and 200 test points, half of each set drawn from a "
    "Gaussian blob centered at (-2.0, 0.0) with unit variance (label 0) and "
    "half from a blob centered at (+2.0, 0.0) with unit variance (label 1). "
    "Train logistic regression (two weights and a bias) from scratch with "
    "batch gradient descent on the training set: learning rate 0.1, 300 "
    "epochs. Report test_accuracy as the fraction of test points classified "
    "correctly at threshold 0.5, and train_loss_final as the mean "
    "cross-entropy on the training set after training. As a positive "
    "control, train a fresh classifier the same way on only the first 8 "
    "training points and report tiny_subset_accuracy as its accuracy on "
    "those same 8 points. Also report n_train and n_test."
)


def _catalog() -> TemplateCatalog:
    def _template(name: str, filename: str) -> ImplementationTemplate:
        return ImplementationTemplate(
            name=name,
            source=(_EXPERIMENTS / filename).read_text(encoding="utf-8"),
        )

    run_cost = ResourceCost(wall_clock_seconds=300.0)
    return TemplateCatalog(
        entries=(
            TemplateCapability(
                template=_template(
                    "task3-synthetic-blobs-v1", "task3_template.py"
                ),
                metrics=(
                    "test_accuracy",
                    "train_loss_final",
                    "tiny_subset_accuracy",
                    "n_train",
                    "n_test",
                ),
                estimated_cost=run_cost,
                description=(
                    "seeded two-blob classification: train a stdlib "
                    "learner and score held-out accuracy"
                ),
                control=TINY_OVERFIT_CONTROL,
            ),
            TemplateCapability(
                template=_template(
                    "task4-effect-separation-v1", "task4_template_effect.py"
                ),
                metrics=(
                    "accuracy_gap",
                    "close_blobs_accuracy",
                    "far_blobs_accuracy",
                    "tiny_subset_accuracy",
                    "n_train",
                    "n_test",
                ),
                estimated_cost=run_cost,
                description=(
                    "class-separation effect: train and score the same "
                    "learner on a close and a far blob pair and report "
                    "the accuracy gap"
                ),
                control=TINY_OVERFIT_CONTROL,
            ),
            TemplateCapability(
                template=_template(
                    "task4-label-noise-v1", "task4_template_robustness.py"
                ),
                metrics=(
                    "clean_test_accuracy",
                    "noisy_test_accuracy",
                    "accuracy_drop",
                    "tiny_subset_accuracy",
                    "n_train",
                    "n_test",
                ),
                estimated_cost=run_cost,
                description=(
                    "label-noise robustness: flip a seeded fraction of "
                    "training labels and report the held-out accuracy drop"
                ),
                control=TINY_OVERFIT_CONTROL,
            ),
        )
    )


def build_state() -> tuple[ResearchState, ExperimentSpec, Prediction]:
    question = ResearchQuestion(
        text=(
            "Under what conditions does a from-scratch logistic-regression "
            "learner recover a linearly separable synthetic decision "
            "boundary?"
        ),
        importance=(
            "The first live proof that an evidence-grounded planner can "
            "select the lab's next experiment autonomously."
        ),
    )
    hypothesis = Hypothesis(
        statement=(
            "Batch gradient-descent logistic regression separates two "
            "well-separated seeded Gaussian blobs with high held-out "
            "accuracy."
        ),
        rationale="The classes are linearly separable by construction.",
        assumptions=("The seeded generator produces the stated blobs.",),
        question_id=question.id,
    )
    prediction = Prediction(
        hypothesis_id=hypothesis.id,
        condition=(
            "400 train / 200 test points from unit-variance Gaussian blobs "
            "centered at (-2,0) and (+2,0), seeded from ARL_SEED"
        ),
        metric="test_accuracy",
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=0.9,
        expectation="Held-out accuracy of at least 0.9.",
    )
    spec = ExperimentSpec(
        prediction_id=prediction.id,
        objective="measure baseline held-out accuracy on separated blobs",
        procedure=BASELINE_PROCEDURE,
        metrics=(
            "test_accuracy",
            "train_loss_final",
            "tiny_subset_accuracy",
            "n_train",
            "n_test",
        ),
        baselines=("chance accuracy 0.5 for balanced two-class data",),
        controls=(
            "tiny_subset_accuracy >= 0.99: a faithful trainer must fit 8 "
            "separable points",
        ),
        seeds=(7, 11),
        estimated_cost=ResourceCost(wall_clock_seconds=300.0),
    )
    state = (
        ResearchState(
            objective=OBJECTIVE,
            budget=ResearchBudget(
                wall_clock_seconds=3600.0, usd=10.0, model_tokens=400_000
            ),
        )
        .upsert_question(question)
        .upsert_hypothesis(hypothesis)
        .upsert_prediction(prediction)
        .add_experiment(spec)
    )
    return state, spec, prediction


class CatalogControls:
    """Positive controls served per spec: the baseline's control for the
    human-authored spec, the planner-selected template's control for
    planned specs (resolved through the planning store)."""

    def __init__(
        self,
        baseline_spec_id: str,
        plans: PlanningStore,
        catalog: TemplateCatalog,
    ) -> None:
        self._baseline_spec_id = baseline_spec_id
        self._plans = plans
        self._catalog = catalog

    def __call__(self, spec: ExperimentSpec) -> tuple[PositiveControl, ...]:
        if spec.id == self._baseline_spec_id:
            return (TINY_OVERFIT_CONTROL,)
        for record in self._plans.records():
            if record.spec_id == spec.id and record.template_id:
                entry = self._catalog.get(record.template_id)
                if entry is not None and entry.control is not None:
                    return (entry.control,)
        return ()


def _fail_fast(docker_host: str) -> str | None:
    """Deterministic environment diagnosis before any model spend."""
    if not any(name for name in KEY_ENV_VARS if _env_present(name)):
        return (
            "no Muse API key in the environment: set MUSE_API_KEY (or "
            "MODEL_API_KEY)"
        )
    socket_path = docker_host.removeprefix("unix://")
    if docker_host.startswith("unix://") and not Path(socket_path).exists():
        return (
            f"docker socket {socket_path} does not exist — start the "
            f"container VM first (colima start)"
        )
    probe = ExperimentJob(
        spec_id="probe",
        command=(
            sys.executable,
            "-m",
            "autonomous_research_lab.execution.container_shim",
        ),
        timeout_seconds=1.0,
    )
    check = PthFilesVisible().check(probe, None)
    if check.state is CheckState.FAIL:
        return check.detail
    return None


def _env_present(name: str) -> bool:
    import os

    return bool(os.environ.get(name, "").strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--docker-host",
        default=f"unix://{Path.home()}/.colima/default/docker.sock",
    )
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args(argv)

    problem = _fail_fast(args.docker_host)
    if problem is not None:
        print(f"refusing to start: {problem}")
        return 1

    # Absolute from the start: the container shim child validates the
    # source tree from a job-private working directory, and docker mounts
    # need absolute paths — a relative --run-root would break both.
    root: Path = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    state, baseline_spec, _prediction = build_state()
    catalog = _catalog()
    plans = PlanningStore(root / "planning")
    ledger = UsageLedger()
    implementations = ImplementationStore(root / "implementations")

    def resolve_template(spec: ExperimentSpec) -> ImplementationTemplate | None:
        for record in plans.records():
            if record.spec_id == spec.id and record.template_id:
                entry = catalog.get(record.template_id)
                if entry is not None:
                    return entry.template
        return None

    engineer = ModelBackedEngineer(
        provider=MuseSparkProvider(),
        model=MUSE_SPARK_1_2,
        runner=DirectJobRunner(LocalExecutor(root / "runs")),
        ledger=ledger,
        store=implementations,
        binding=ContainerBinding(
            image=args.image,
            docker_host=args.docker_host,
            timeout_seconds=180.0,
        ),
        template=catalog.entries[0].template,
        template_resolver=resolve_template,
        request_timeout_seconds=args.request_timeout,
    )
    planner = ModelBackedPlanner(
        provider=MuseSparkProvider(),
        model=MUSE_SPARK_1_2,
        ledger=ledger,
        store=plans,
        catalog=catalog,
        request_timeout_seconds=args.request_timeout,
        # Muse Spark padded four consecutive planning replies past the
        # 4096 default in the 2026-08-18 partial-3 attempt (finish_reason
        # 'length'); match the engineer's headroom.
        max_output_tokens=8192,
    )
    runtime = ResearchRuntime(
        config=RuntimeConfig(),
        director=PlanningDirector(plans=plans),
        roles={
            RoleName.RESEARCH_ENGINEER: engineer,
            RoleName.RESEARCH_DIRECTOR: planner,
        },
        store=FileEvidenceStore(root),
        states=FileStateStore(root),
        trajectory=JsonlTrajectoryLogger(root / "trajectory.jsonl"),
        metrics=JsonlRuntimeMetrics(root / "metrics.jsonl"),
        usage=ledger,
        methodology_reviewer=StructuralMethodology(),
        control_source=CatalogControls(baseline_spec.id, plans, catalog),
        verifications=FileVerificationStore(root / "verifications"),
    )

    halt: str | None = None
    for index in range(args.max_steps):
        report = runtime.step(state)
        state = report.state
        selected = report.deliberation.selected
        action = selected.action.action_type.value if selected else "none"
        print(f"step {index + 1}: {action} -> {report.halt_reason or 'ok'}")
        for note in report.notes:
            print(f"  note: {note}")
        if report.halt_reason is not None:
            halt = report.halt_reason
            break
        if (
            len(plans.records()) >= 2
            and not plans.open_decisions()
            and not _has_execution_work(state)
        ):
            halt = "trajectory complete: both planning decisions resolved"
            break

    return _report(runtime, state, plans, implementations, halt, root)


def _has_execution_work(state: ResearchState) -> bool:
    ran = {ref.spec_id for ref in state.results}
    return any(spec.id not in ran for spec in state.experiments)


def _report(
    runtime: ResearchRuntime,
    state: ResearchState,
    plans: PlanningStore,
    implementations: ImplementationStore,
    halt: str | None,
    root: Path,
) -> int:
    print("\n== Task 4 live autonomous trajectory ==")
    print(f"run root: {root}")
    print(f"halt: {halt}")

    decisions = plans.records()
    print(f"\naccepted planning decisions: {len(decisions)}")
    for record in decisions:
        print(f"decision {record.id}:")
        print(f"  action: {record.action.value}")
        print(f"  question: {record.question_id}")
        print(f"  cited evidence: {list(record.evidence_ids)}")
        print(
            f"  chain: hypothesis={record.hypothesis_id or '-'} "
            f"prediction={record.prediction_id or '-'} "
            f"spec={record.spec_id or '-'}"
        )
        print(
            f"  template: {record.template_id or '-'}  "
            f"replication_seed: {record.replication_seed!r}  "
            f"stop: {record.stop_reason.value if record.stop_reason else '-'}"
        )
        print(f"  repairs: {record.repair_count}")
        print(f"  request fingerprint: {record.request_fingerprint}")
        print(f"  response occurrence: {record.response_id}")
        print(f"  provider request id: {record.provider_request_id}")
        print(f"  served model: {record.served_model}")
        print(
            f"  latency: {record.latency_seconds:.2f}s  tokens: "
            f"in={record.input_tokens} out={record.output_tokens}  "
            f"nominal cost: {record.nominal_cost_usd!r}"
        )
        print(f"  rationale: {record.rationale[:300]}")
        print(f"  dispatched: {plans.is_dispatched(record.id)}")
    rejections = plans.rejected()
    print(f"rejected planning attempts: {len(rejections)}")
    for rejected in rejections:
        reasons = rejected.get("reasons", [])
        assert isinstance(reasons, list)
        names = ", ".join(str(entry.get("rule")) for entry in reasons)
        print(f"  rejected (repair {rejected.get('repair')}): {names}")

    admissible = ScientificAdmissibility(
        verifications=runtime.verifications, governance_enabled=True
    )
    verified = 0
    for ref in state.results:
        result = runtime.store.get_result(ref.result_id)
        verdict = runtime.verifications.get(result.id)
        standing = verdict.standing.value if verdict else "unrecorded"
        print(
            f"result {result.id} (spec {result.spec_id}, seed "
            f"{result.seed!r}): {result.status.value}, verification "
            f"{standing}, admissible {admissible(result.id)}"
        )
        if verdict is not None and admissible(result.id):
            verified += 1

    non_stop = next(
        (d for d in decisions if d.action is not PlanningAction.STOP), None
    )
    traversed = False
    if non_stop is not None:
        target_spec = non_stop.spec_id
        implementation = next(
            (
                record
                for record in implementations.records()
                if record.spec_id == target_spec
                and record.invocation_id  # planner-era record
            ),
            None,
        )
        has_result = any(ref.spec_id == target_spec for ref in state.results)
        template_ok = (
            non_stop.action is PlanningAction.REPLICATE
            or (
                implementation is not None
                and implementation.template_id == non_stop.template_id
            )
        )
        traversed = has_result and template_ok
        if implementation is not None:
            print(
                f"\nplanned implementation {implementation.id}: template "
                f"{implementation.template_id} (decision names "
                f"{non_stop.template_id or '-'})"
            )

    ok = len(decisions) >= 2 and verified >= 1 and (
        non_stop is None or traversed
    )
    print(
        f"\nTRAJECTORY: decisions={len(decisions)} verified_results="
        f"{verified} first_non_stop_traversed={traversed}"
    )
    print("LIVE PROOF: " + ("PASSED" if ok else "NOT PASSED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
