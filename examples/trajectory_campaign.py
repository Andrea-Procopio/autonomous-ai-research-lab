"""The Task 3 trajectory campaign: does the slice generalize?

Four bounded, offline, seeded ML experiments, every one implemented live
by the model-backed engineer from ONE generic template
(``experiments/campaign_template.py``) that knows nothing about any task —
which is the point: Task 3 proved the vertical slice with a purpose-built
template, and this campaign measures whether the engineer, the
deterministic gates, and the verification layer hold up across designs
they were not written for.

The four tasks, chosen to cover the campaign brief:

* ``separable-knn`` — k-nearest-neighbours on separable blobs, with an
  explicit majority-class **baseline** metric;
* ``xor-perceptron`` — a **genuine negative**: a single-layer perceptron
  cannot learn XOR labels; the pre-registered prediction is expected to
  fail while an instrument control proves the trainer itself works;
* ``ridge-replication`` — ridge regression **replicated across three
  seeds** (one run + two replications of one protocol);
* ``scaling-ablation`` — one **ablation**: kNN with and without feature
  standardization under a dominant nuisance coordinate, predicted on the
  gain.

Per task the runtime is wired exactly as the live slice was — Muse behind
the engineer, container execution, file-backed state/verification stores,
positive controls, and a deterministic *structural* methodology review —
plus the rule-based demo scientist/critic seats so the loop can synthesize
and assess its way to a natural stop (those seats make zero provider
calls, and the metrics record them as such).

Measured per task and in aggregate: implementation success, generation-
repair rate, admissibility rate, exact token usage (Muse nominal cost is
unknown; tokens are the ground truth), and runtime.

Run from the repository root (module mode, so the demo seats import)::

    python -m examples.trajectory_campaign --campaign-root <dir> [--only slug]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.execution.binding import ContainerBinding, JobBinding
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.execution.runner import DirectJobRunner
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import (
    MissingRoleError,
    ResearchRuntime,
    StepReport,
)
from autonomous_research_lab.orchestration.trajectory import JsonlTrajectoryLogger
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.roles.base import RoleName
from autonomous_research_lab.roles.engineer import (
    ImplementationTemplate,
    ModelBackedEngineer,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.implementation_store import ImplementationStore
from autonomous_research_lab.runtime.metrics import (
    JsonlRuntimeMetrics,
    StepMetrics,
)
from autonomous_research_lab.runtime.muse import MUSE_SPARK_1_2, MuseSparkProvider
from autonomous_research_lab.runtime.providers import ModelProvider, UsageLedger
from autonomous_research_lab.runtime.verification import (
    CheckState,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)
from autonomous_research_lab.runtime.verification_store import (
    FileVerificationStore,
    ScientificAdmissibility,
)
from examples.runtime_loop import DemoCritic, DemoScientist

DEFAULT_IMAGE = (
    "python@sha256:"
    "9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7"
)

TEMPLATE_PATH = Path(__file__).parent / "experiments" / "campaign_template.py"


# -- task definitions ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CampaignTask:
    """One pre-registered experiment: the full scientific chain plus its
    instrument controls, fixed before anything runs."""

    slug: str
    question: ResearchQuestion
    hypothesis: Hypothesis
    prediction: Prediction
    spec: ExperimentSpec
    controls: tuple[PositiveControl, ...]

    def initial_state(self) -> ResearchState:
        return (
            ResearchState(
                objective=self.question.text,
                budget=ResearchBudget(
                    wall_clock_seconds=3600.0, usd=10.0, model_tokens=200_000
                ),
            )
            .upsert_question(self.question)
            .upsert_hypothesis(self.hypothesis)
            .upsert_prediction(self.prediction)
            .add_experiment(self.spec)
        )


def _task(
    *,
    slug: str,
    question: str,
    importance: str,
    hypothesis: str,
    rationale: str,
    condition: str,
    metric: str,
    comparator: Comparator,
    threshold: float,
    expectation: str,
    objective: str,
    procedure: str,
    metrics: tuple[str, ...],
    baselines: tuple[str, ...],
    controls_text: tuple[str, ...],
    seeds: tuple[int, ...],
    controls: tuple[PositiveControl, ...],
) -> CampaignTask:
    q = ResearchQuestion(text=question, importance=importance)
    h = Hypothesis(statement=hypothesis, rationale=rationale, question_id=q.id)
    p = Prediction(
        hypothesis_id=h.id,
        condition=condition,
        metric=metric,
        comparator=comparator,
        threshold=threshold,
        expectation=expectation,
    )
    spec = ExperimentSpec(
        prediction_id=p.id,
        objective=objective,
        procedure=procedure,
        metrics=metrics,
        baselines=baselines,
        controls=controls_text,
        seeds=seeds,
    )
    return CampaignTask(
        slug=slug,
        question=q,
        hypothesis=h,
        prediction=p,
        spec=spec,
        controls=controls,
    )


SEPARABLE_KNN = _task(
    slug="separable-knn",
    question=(
        "Does nearest-neighbour voting recover well-separated class "
        "structure without any trained parameters?"
    ),
    importance=(
        "A baseline-anchored positive task for the campaign: if the slice "
        "cannot implement kNN it cannot implement anything."
    ),
    hypothesis=(
        "k-nearest-neighbours with k=5 separates two well-separated "
        "Gaussian blobs far better than the majority-class baseline."
    ),
    rationale="The blobs are 3 standard deviations apart by construction.",
    condition=(
        "300 train / 200 test points, unit-variance blobs centered at "
        "(-1.5, 0) and (+1.5, 0), seeded from ARL_SEED"
    ),
    metric="test_accuracy",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.85,
    expectation="Held-out kNN accuracy of at least 0.85.",
    objective="Measure kNN test accuracy against the majority baseline.",
    procedure=(
        "Generate a two-class 2D dataset with random.Random(ARL_SEED): 300 "
        "training and 200 test points, half of each split drawn from a "
        "Gaussian blob centered at (-1.5, 0.0) and half from a blob "
        "centered at (+1.5, 0.0), both with standard deviation 1.0 per "
        "coordinate (use gauss), labels 0 and 1. Implement k-nearest-"
        "neighbours classification from scratch with k=5 and Euclidean "
        "distance, classifying each test point by majority vote of the 5 "
        "nearest training points. Report test_accuracy (fraction of test "
        "points classified correctly), majority_baseline_accuracy (accuracy "
        "on the test set of always predicting the most common training "
        "label), and tiny_subset_accuracy: using only the first 8 training "
        "points as the reference set, classify those same 8 points with "
        "k=1; each point is its own nearest neighbour at distance zero, so "
        "a correct implementation reports 1.0. Also report n_train and "
        "n_test."
    ),
    metrics=(
        "test_accuracy",
        "majority_baseline_accuracy",
        "tiny_subset_accuracy",
        "n_train",
        "n_test",
    ),
    baselines=("majority class: about 0.5 on balanced test data",),
    controls_text=(
        "tiny_subset_accuracy >= 0.99: k=1 self-classification is exact",
    ),
    seeds=(11,),
    controls=(
        PositiveControl(
            name="knn_self_match",
            metric="tiny_subset_accuracy",
            comparator=Comparator.GREATER_OR_EQUAL,
            threshold=0.99,
            rationale="each reference point is its own nearest neighbour",
        ),
    ),
)

XOR_PERCEPTRON = _task(
    slug="xor-perceptron",
    question="Can a single-layer perceptron learn an XOR parity labeling?",
    importance=(
        "The campaign's genuine negative: the pre-registered prediction is "
        "expected to fail while the instrument control passes, and the "
        "verified negative must survive as evidence."
    ),
    hypothesis=(
        "A single-layer perceptron can learn the XOR parity labeling of "
        "the plane to high held-out accuracy."
    ),
    rationale=(
        "Deliberately the classic falsified claim: XOR is not linearly "
        "separable, so the hypothesis should be refuted by the run."
    ),
    condition=(
        "400 train / 200 test points uniform in [-1, 1]^2, label 1 iff "
        "x0*x1 > 0, 50 perceptron epochs at learning rate 0.1, seeded "
        "from ARL_SEED"
    ),
    metric="test_accuracy",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.9,
    expectation=(
        "Held-out accuracy of at least 0.9 (expected NOT to hold; chance "
        "is about 0.5 on this labeling)."
    ),
    objective=(
        "Test a linear classifier against a labeling that is not linearly "
        "separable, with a separable control proving the trainer works."
    ),
    procedure=(
        "Generate a two-class 2D dataset with random.Random(ARL_SEED): 400 "
        "training and 200 test points; each point has coordinates x0 and "
        "x1 drawn uniformly from [-1, 1] (use uniform), and its label is 1 "
        "if x0*x1 > 0 else 0. Train a single-layer perceptron: weights w0, "
        "w1 and bias b all start at 0; the prediction for a point is 1 if "
        "w0*x0 + w1*x1 + b > 0 else 0; for each training example in order, "
        "if the prediction is wrong, update w0 += 0.1*(label - "
        "prediction)*x0, w1 += 0.1*(label - prediction)*x1, b += "
        "0.1*(label - prediction). Run 50 epochs over the training set. "
        "Report test_accuracy of the trained perceptron on the test set "
        "under the XOR labels. As an instrument control, relabel the same "
        "points with label 1 if x0 + x1 > 0 else 0, train a fresh "
        "perceptron the same way on the relabeled training set, and report "
        "separable_sanity_accuracy as its accuracy on the relabeled test "
        "set. Also report n_train and n_test."
    ),
    metrics=(
        "test_accuracy",
        "separable_sanity_accuracy",
        "n_train",
        "n_test",
    ),
    baselines=("chance: about 0.5, XOR labels are balanced in expectation",),
    controls_text=(
        "separable_sanity_accuracy >= 0.9: the same trainer must solve a "
        "linearly separable relabeling of the same points",
    ),
    seeds=(23,),
    controls=(
        PositiveControl(
            name="separable_sanity",
            metric="separable_sanity_accuracy",
            comparator=Comparator.GREATER_OR_EQUAL,
            threshold=0.9,
            rationale=(
                "a perceptron that cannot solve a separable problem is a "
                "broken instrument, not evidence about XOR"
            ),
        ),
    ),
)

RIDGE_REPLICATION = _task(
    slug="ridge-replication",
    question=(
        "Does gradient-descent ridge regression recover a noisy linear "
        "relationship at noise-floor error, consistently across seeds?"
    ),
    importance=(
        "The campaign's replication family: three seeds, one protocol, "
        "three independent pieces of evidence."
    ),
    hypothesis=(
        "Full-batch gradient descent with a small L2 penalty fits y = "
        "2.5x - 1 + noise to held-out error near the noise floor."
    ),
    rationale="The model class contains the true function.",
    condition=(
        "120 train / 60 test points, x uniform in [-2, 2], Gaussian noise "
        "sd 0.5, learning rate 0.05, 500 epochs, L2 0.001 on w, seeded "
        "from ARL_SEED"
    ),
    metric="test_mse",
    comparator=Comparator.LESS_OR_EQUAL,
    threshold=0.5,
    expectation=(
        "Held-out mean squared error at most 0.5 (noise variance is 0.25)."
    ),
    objective=(
        "Fit a 1D ridge regression by gradient descent and measure "
        "held-out error, replicated across declared seeds."
    ),
    procedure=(
        "Generate a 1D regression dataset with random.Random(ARL_SEED): "
        "120 training and 60 test points with x drawn uniformly from "
        "[-2, 2] (use uniform) and y = 2.5*x - 1.0 + e where e is Gaussian "
        "noise with standard deviation 0.5 (use gauss). Fit y = w*x + b by "
        "full-batch gradient descent on mean squared error with an L2 "
        "penalty of 0.001 on w (never on b): learning rate 0.05, 500 "
        "epochs, w and b initialized to 0. Report test_mse (mean squared "
        "error on the test set), learned_w and learned_b. As an instrument "
        "control, fit a fresh model the same way to 8 noise-free points "
        "y = 2.5*x - 1.0 with x evenly spaced from -2 to 2, and report "
        "noiseless_tiny_mse as that model's mean squared error on those "
        "same 8 points, which must be near zero. Also report n_train and "
        "n_test."
    ),
    metrics=(
        "test_mse",
        "learned_w",
        "learned_b",
        "noiseless_tiny_mse",
        "n_train",
        "n_test",
    ),
    baselines=(
        "predicting the training mean: mse near var(y), about 8.6 here",
    ),
    controls_text=(
        "noiseless_tiny_mse <= 0.01: the fitter must drive noise-free "
        "error to near zero",
    ),
    seeds=(5, 13, 29),
    controls=(
        PositiveControl(
            name="noiseless_fit",
            metric="noiseless_tiny_mse",
            comparator=Comparator.LESS_OR_EQUAL,
            threshold=0.01,
            rationale="noise-free data must be fit almost exactly",
        ),
    ),
)

SCALING_ABLATION = _task(
    slug="scaling-ablation",
    question=(
        "How much of kNN's accuracy under a dominant nuisance feature is "
        "owed to feature standardization?"
    ),
    importance=(
        "The campaign's ablation: one component removed (standardization), "
        "the prediction pre-registered on the gain."
    ),
    hypothesis=(
        "With a nuisance coordinate 100 times the informative scale, "
        "standardizing features is what makes kNN work."
    ),
    rationale=(
        "Euclidean distance is scale-sensitive; the nuisance axis drowns "
        "the informative one unless standardized away."
    ),
    condition=(
        "300 train / 200 test points, informative x1 blobs at -1.5/+1.5 "
        "(sd 1), nuisance x0 with sd 100, kNN k=5, seeded from ARL_SEED"
    ),
    metric="ablation_gain",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.2,
    expectation=(
        "Standardized kNN beats raw kNN by at least 0.2 accuracy."
    ),
    objective=(
        "Measure kNN accuracy with and without feature standardization "
        "under a dominant nuisance coordinate; the ablated variant is the "
        "raw-coordinates run."
    ),
    procedure=(
        "Generate a two-class 2D dataset with random.Random(ARL_SEED): 300 "
        "training and 200 test points. For each point, draw its "
        "informative coordinate x1 with gauss: half of each split from "
        "mean -1.5 (label 0) and half from mean +1.5 (label 1), standard "
        "deviation 1.0; draw its nuisance coordinate x0 with gauss from "
        "mean 0.0 and standard deviation 100.0. Classify the test points "
        "with k-nearest-neighbours (k=5, Euclidean distance, majority "
        "vote) in two ways. raw_accuracy: use the raw coordinates. "
        "standardized_accuracy: standardize each coordinate first "
        "(subtract the training mean and divide by the training standard "
        "deviation of that coordinate, applying the training statistics "
        "to the test points too). Report raw_accuracy, "
        "standardized_accuracy, and ablation_gain = standardized_accuracy "
        "- raw_accuracy. As an instrument control, report "
        "self_knn_accuracy: with k=1 and the standardized first 8 training "
        "points as the reference set, classify those same 8 points, which "
        "must give 1.0. Also report n_train and n_test."
    ),
    metrics=(
        "raw_accuracy",
        "standardized_accuracy",
        "ablation_gain",
        "self_knn_accuracy",
        "n_train",
        "n_test",
    ),
    baselines=(
        "chance: about 0.5; the raw-coordinates run is the ablated variant",
    ),
    controls_text=("self_knn_accuracy >= 0.99: k=1 self-match is exact",),
    seeds=(41,),
    controls=(
        PositiveControl(
            name="knn_self_match",
            metric="self_knn_accuracy",
            comparator=Comparator.GREATER_OR_EQUAL,
            threshold=0.99,
            rationale="each reference point is its own nearest neighbour",
        ),
    ),
)

CAMPAIGN_TASKS: tuple[CampaignTask, ...] = (
    SEPARABLE_KNN,
    XOR_PERCEPTRON,
    RIDGE_REPLICATION,
    SCALING_ABLATION,
)


# -- deterministic wiring ------------------------------------------------------


class StructuralMethodology:
    """Tier-0 structural methodology review, shared by every campaign task.

    Checks what code can check about a design: the pre-registered metric is
    declared, a baseline is stated, an instrument control is declared. The
    scientific substance of each design was human-authored with the
    campaign; this review is structural, and says so."""

    def review(
        self,
        spec: ExperimentSpec,
        prediction: Prediction | None,
        *,
        objective: str,  # noqa: ARG002 - structural review only
    ) -> VerificationCheck:
        problems: list[str] = []
        if prediction is None:
            problems.append("no prediction reaches the review")
        elif prediction.metric not in spec.metrics:
            problems.append(
                f"prediction metric {prediction.metric!r} is not declared"
            )
        if not spec.controls:
            problems.append("no instrument control declared")
        if not spec.baselines:
            problems.append("no baseline stated")
        if problems:
            return VerificationCheck(
                dimension=ValidityDimension.METHODOLOGY,
                name="methodological_validity",
                state=CheckState.FAIL,
                detail="; ".join(problems),
            )
        return VerificationCheck(
            dimension=ValidityDimension.METHODOLOGY,
            name="methodological_validity",
            state=CheckState.PASS,
            detail=(
                "structural review: pre-registered metric declared, "
                "baseline stated, instrument control declared "
                "(design substance human-authored with the campaign)"
            ),
        )


class ControlTable:
    """Per-spec positive controls, looked up by spec id."""

    def __init__(self, tasks: tuple[CampaignTask, ...]) -> None:
        self._by_spec = {task.spec.id: task.controls for task in tasks}

    def __call__(self, spec: ExperimentSpec) -> tuple[PositiveControl, ...]:
        return self._by_spec.get(spec.id, ())


@dataclass
class TeeSink:
    """Step metrics into memory (for measurement) and JSONL (for the record)."""

    jsonl: JsonlRuntimeMetrics
    records: list[StepMetrics] = field(default_factory=list)

    def log(self, record: StepMetrics) -> None:
        self.records.append(record)
        self.jsonl.log(record)


@dataclass(frozen=True, slots=True)
class TaskRun:
    task: CampaignTask
    reports: tuple[StepReport, ...]
    runtime: ResearchRuntime
    implementations: ImplementationStore
    metrics: tuple[StepMetrics, ...]
    stopped_by: str


def run_task(
    task: CampaignTask,
    root: Path,
    *,
    provider: ModelProvider,
    model: str,
    binding: JobBinding,
    request_timeout: float = 240.0,
    max_steps: int = 16,
) -> TaskRun:
    """One task's full trajectory: steps until the director stops, the
    budget halts, or the step cap is hit."""
    root.mkdir(parents=True, exist_ok=True)
    ledger = UsageLedger()
    implementations = ImplementationStore(root / "implementations")
    engineer = ModelBackedEngineer(
        provider=provider,
        model=model,
        runner=DirectJobRunner(LocalExecutor(root / "runs")),
        ledger=ledger,
        store=implementations,
        binding=binding,
        template=ImplementationTemplate(
            name="campaign-generic-v1",
            source=TEMPLATE_PATH.read_text(encoding="utf-8"),
        ),
        request_timeout_seconds=request_timeout,
    )
    sink = TeeSink(jsonl=JsonlRuntimeMetrics(root / "metrics.jsonl"))
    runtime = ResearchRuntime(
        config=RuntimeConfig(),
        director=RuleBasedFrontierDirector(),
        roles={
            RoleName.RESEARCH_ENGINEER: engineer,
            RoleName.RESEARCH_DIRECTOR: DemoScientist(
                threshold=task.prediction.threshold, seeds=task.spec.seeds
            ),
            RoleName.RESULT_ANALYST: DemoCritic(),
        },
        store=FileEvidenceStore(root),
        states=FileStateStore(root),
        trajectory=JsonlTrajectoryLogger(root / "trajectory.jsonl"),
        metrics=sink,
        usage=ledger,
        methodology_reviewer=StructuralMethodology(),
        control_source=ControlTable(CAMPAIGN_TASKS),
        verifications=FileVerificationStore(root / "verifications"),
    )

    state = task.initial_state()
    reports: list[StepReport] = []
    stopped_by = f"step cap of {max_steps} reached"
    for _ in range(max_steps):
        try:
            report = runtime.step(state)
        except MissingRoleError as exc:  # defensive: all seats are wired
            stopped_by = f"unroutable action: {exc}"
            break
        reports.append(report)
        state = report.state
        if report.halt_reason is not None:
            stopped_by = report.halt_reason
            break
    return TaskRun(
        task=task,
        reports=tuple(reports),
        runtime=runtime,
        implementations=implementations,
        metrics=tuple(sink.records),
        stopped_by=stopped_by,
    )


# -- measurement ---------------------------------------------------------------


def measure(run: TaskRun) -> dict[str, object]:
    """The campaign's numbers for one task, computed from the records the
    runtime already keeps — nothing is measured that is not on the record."""
    state = run.reports[-1].state if run.reports else run.task.initial_state()
    actions = Counter(m.action_type for m in run.metrics)
    engineer_invocations = actions["run_experiment"] + actions["replicate"]

    results = [
        run.runtime.store.get_result(ref.result_id) for ref in state.results
    ]
    completed = [r for r in results if r.succeeded]
    failed = [r for r in results if not r.succeeded]
    admissible = ScientificAdmissibility(
        verifications=run.runtime.verifications, governance_enabled=True
    )
    admitted = [r for r in completed if admissible(r.id)]

    tests = state.tests_for(run.task.prediction.id)
    provider_calls = sum(m.provider_calls for m in run.metrics)
    generation_repairs = len(run.implementations.rejected())

    return {
        "slug": run.task.slug,
        "stopped_by": run.stopped_by,
        "steps": len(run.reports),
        "actions": dict(actions),
        "engineer_invocations": engineer_invocations,
        "results_committed": len(results),
        "results_completed": len(completed),
        "results_failed": len(failed),
        "implementation_success_rate": (
            len(completed) / engineer_invocations if engineer_invocations else None
        ),
        "generation_repairs": generation_repairs,
        "generation_repair_rate": (
            generation_repairs / provider_calls if provider_calls else None
        ),
        "execution_debug_attempts": sum(m.debug_attempts for m in run.metrics),
        "admissible_results": len(admitted),
        "admissibility_rate": (
            len(admitted) / len(completed) if completed else None
        ),
        "prediction_tests": [t.consistency.value for t in tests],
        "verification": {
            r.id: (
                record.validity.value
                if (record := run.runtime.verifications.get(r.id)) is not None
                else "unrecorded"
            )
            for r in results
        },
        "claims": len(state.claims),
        "assessments": len(state.assessments),
        "provider_calls": provider_calls,
        "input_tokens": sum(m.input_tokens for m in run.metrics),
        "output_tokens": sum(m.output_tokens for m in run.metrics),
        "experiment_seconds": round(
            sum(m.experiment_seconds for m in run.metrics), 3
        ),
        "wall_clock_seconds": round(
            sum(m.wall_clock_seconds for m in run.metrics), 3
        ),
        "notes": sorted({n for m in run.metrics for n in m.notes}),
    }


def _print_summary(summary: dict[str, object]) -> None:
    slug = summary["slug"]
    print(f"-- {slug} --")
    for key in (
        "stopped_by",
        "steps",
        "actions",
        "engineer_invocations",
        "results_completed",
        "results_failed",
        "implementation_success_rate",
        "generation_repairs",
        "generation_repair_rate",
        "execution_debug_attempts",
        "admissible_results",
        "admissibility_rate",
        "prediction_tests",
        "verification",
        "claims",
        "assessments",
        "provider_calls",
        "input_tokens",
        "output_tokens",
        "experiment_seconds",
        "wall_clock_seconds",
    ):
        print(f"  {key}: {summary[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--only", default=None, help="run a single task by slug"
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--docker-host",
        default=f"unix://{Path.home()}/.colima/default/docker.sock",
    )
    parser.add_argument("--request-timeout", type=float, default=240.0)
    args = parser.parse_args(argv)

    tasks = tuple(
        t for t in CAMPAIGN_TASKS if args.only is None or t.slug == args.only
    )
    if not tasks:
        print(f"no task named {args.only!r}", file=sys.stderr)
        return 2

    summaries: list[dict[str, object]] = []
    for task in tasks:
        binding = ContainerBinding(
            image=args.image,
            docker_host=args.docker_host,
            timeout_seconds=180.0,
        )
        run = run_task(
            task,
            args.campaign_root / task.slug,
            provider=MuseSparkProvider(),
            model=MUSE_SPARK_1_2,
            binding=binding,
            request_timeout=args.request_timeout,
        )
        summary = measure(run)
        summaries.append(summary)
        _print_summary(summary)

    out = args.campaign_root / "campaign_summary.json"
    existing: list[dict[str, object]] = []
    if out.exists():
        existing = [
            s
            for s in json.loads(out.read_text(encoding="utf-8"))
            if s.get("slug") not in {x["slug"] for x in summaries}
        ]
    out.write_text(
        json.dumps(existing + summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"summary written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
