"""Task 3 live vertical slice: spec -> Muse -> contained execution -> evidence.

One runtime step, live, with nothing mocked::

    existing ExperimentSpec
      -> MuseSparkProvider (one structured-output call, validated locally)
      -> deterministic source validation + preserved implementation
      -> ExperimentJob bound to a disposable container (network none,
         read-only source mount, dropped capabilities, finite limits)
      -> LocalExecutor runs it and reads metrics.json
      -> Tier-0 validation gate -> atomic commit -> mechanical prediction
         test -> deterministic evidence transcription
      -> verification record (execution / implementation / methodology /
         analysis) -> scientific admissibility

The experiment is a deliberately inexpensive, fully offline, seeded ML
task: logistic regression trained from scratch (stdlib only) on synthetic
2D Gaussian blobs, with a tiny-subset overfit metric as the positive
control. The science is modest on purpose — what is being proven live is
the *slice*, end to end, with real provenance at every joint.

Requires: a running Docker daemon (colima works), the pinned image already
pulled, and MUSE_API_KEY (or MODEL_API_KEY) in the environment. Run with::

    python examples/live_task3.py --run-root /path/to/run

Exits 0 only if the step reaches scientifically admissible verified
evidence; any lesser outcome is reported and exits 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.execution.binding import ContainerBinding
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.execution.runner import DirectJobRunner
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import ResearchRuntime, StepReport
from autonomous_research_lab.orchestration.trajectory import JsonlTrajectoryLogger
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.roles.base import RoleName
from autonomous_research_lab.roles.engineer import (
    ImplementationTemplate,
    ModelBackedEngineer,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.implementation_store import ImplementationStore
from autonomous_research_lab.runtime.metrics import JsonlRuntimeMetrics
from autonomous_research_lab.runtime.muse import MUSE_SPARK_1_2, MuseSparkProvider
from autonomous_research_lab.runtime.providers import UsageLedger
from autonomous_research_lab.runtime.verification import (
    CheckState,
    OutcomeStanding,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)
from autonomous_research_lab.runtime.verification_store import (
    FileVerificationStore,
    ScientificAdmissibility,
)

#: The environment that runs generated code, pinned by digest: the image
#: recorded is the image that ran.
DEFAULT_IMAGE = (
    "python@sha256:"
    "9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7"
)

TEMPLATE_PATH = Path(__file__).parent / "experiments" / "task3_template.py"

OBJECTIVE = (
    "Establish that the lab's model-backed engineer can implement and run "
    "a small seeded learning experiment end to end."
)

PROCEDURE = (
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


def build_state() -> tuple[ResearchState, ExperimentSpec, Prediction]:
    question = ResearchQuestion(
        text=(
            "Can a from-scratch logistic-regression learner recover a "
            "linearly separable synthetic decision boundary?"
        ),
        importance=(
            "The first live proof that model-generated implementations can "
            "produce scientifically admissible evidence in this lab."
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
        objective=OBJECTIVE,
        procedure=PROCEDURE,
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
        seeds=(7,),
    )
    state = (
        ResearchState(
            objective=OBJECTIVE,
            budget=ResearchBudget(
                wall_clock_seconds=3600.0, usd=10.0, model_tokens=200_000
            ),
        )
        .upsert_question(question)
        .upsert_hypothesis(hypothesis)
        .upsert_prediction(prediction)
        .add_experiment(spec)
    )
    return state, spec, prediction


class TemplateTaskMethodology:
    """Deterministic methodology verdict for the one fixed template task.

    The design was human-reviewed when this file was written: held-out
    accuracy of the learner under test, a stated chance baseline, a
    pre-registered threshold, and an instrument control. Encoding that
    review as code keeps the live slice honest about where the judgment
    came from — no model reviewed this design."""

    def review(
        self,
        spec: ExperimentSpec,
        prediction: Prediction | None,
        *,
        objective: str,  # noqa: ARG002 - fixed-task reviewer
    ) -> VerificationCheck:
        sound = (
            prediction is not None
            and prediction.metric in spec.metrics
            and "tiny_subset_accuracy" in spec.metrics
        )
        return VerificationCheck(
            dimension=ValidityDimension.METHODOLOGY,
            name="methodological_validity",
            state=CheckState.PASS if sound else CheckState.FAIL,
            detail=(
                "human-authored fixed template task: held-out accuracy "
                "measures the hypothesis, chance baseline stated, "
                "positive control declared"
                if sound
                else "spec no longer matches the reviewed template task"
            ),
        )


TINY_OVERFIT_CONTROL = PositiveControl(
    name="tiny_subset_overfit",
    metric="tiny_subset_accuracy",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.99,
    rationale="a faithful trainer must fit 8 linearly separable points",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--docker-host",
        default=f"unix://{Path.home()}/.colima/default/docker.sock",
    )
    parser.add_argument("--request-timeout", type=float, default=240.0)
    args = parser.parse_args(argv)
    root: Path = args.run_root
    root.mkdir(parents=True, exist_ok=True)

    template = ImplementationTemplate(
        name="task3-synthetic-blobs-v1",
        source=TEMPLATE_PATH.read_text(encoding="utf-8"),
    )
    ledger = UsageLedger()
    implementations = ImplementationStore(root / "implementations")
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
        template=template,
        request_timeout_seconds=args.request_timeout,
    )
    verifications = FileVerificationStore(root / "verifications")
    runtime = ResearchRuntime(
        config=RuntimeConfig(),
        director=RuleBasedFrontierDirector(),
        roles={RoleName.RESEARCH_ENGINEER: engineer},
        store=FileEvidenceStore(root),
        states=FileStateStore(root),
        trajectory=JsonlTrajectoryLogger(root / "trajectory.jsonl"),
        metrics=JsonlRuntimeMetrics(root / "metrics.jsonl"),
        usage=ledger,
        methodology_reviewer=TemplateTaskMethodology(),
        control_source=lambda _spec: (TINY_OVERFIT_CONTROL,),
        verifications=verifications,
    )

    state, _spec, prediction = build_state()
    report = runtime.step(state)
    return _report(report, runtime, prediction, implementations, root)


def _report(
    report: StepReport,
    runtime: ResearchRuntime,
    prediction: Prediction,
    implementations: ImplementationStore,
    root: Path,
) -> int:
    state = report.state
    selected = report.deliberation.selected
    print("== Task 3 live vertical slice ==")
    print(f"run root: {root}")
    print(f"selected action: {selected.action.action_type if selected else None}")
    print(f"reasoning invocations: {report.reasoning_invocations}")
    usage = report.provider_usage
    print(
        f"provider usage: calls={usage.calls} in={usage.input_tokens} "
        f"out={usage.output_tokens} model={usage.model!r}"
    )

    for record in implementations.records():
        print("implementation record:")
        print(f"  id: {record.id}")
        print(f"  source_id: {record.source_id}")
        print(f"  manifest: {dict(record.manifest)}")
        print(f"  template: {record.template_id} sha256={record.template_sha256}")
        print(f"  request fingerprint: {record.request_fingerprint}")
        print(f"  response occurrence: {record.response_id}")
        print(f"  provider request id: {record.provider_request_id}")
        print(f"  provider/model: {record.provider} / {record.served_model}")
        print(f"  entrypoint: {record.entrypoint}  seed: {record.seed}")
        print(f"  rationale: {record.rationale[:300]}")

    ok = False
    for ref in state.results:
        result = runtime.store.get_result(ref.result_id)
        print(f"result {result.id}: {result.status} exit={result.exit_code}")
        print(f"  metrics: {dict(result.metrics)}")
        print(f"  config: {dict(result.config)}")
        test = state.test_for_result(prediction.id, result.id)
        if test is not None:
            print(f"  prediction test: {test.consistency} — {test.detail}")
        verdict_record = runtime.verifications.get(result.id)
        if verdict_record is not None:
            print(
                f"  verification: {verdict_record.validity} -> "
                f"{verdict_record.standing}"
            )
            for check in verdict_record.report.checks:
                print(
                    f"    [{check.dimension}] {check.name}: {check.state}"
                    f" — {check.detail}"
                )
            admissible = ScientificAdmissibility(
                verifications=runtime.verifications, governance_enabled=True
            )(result.id)
            print(f"  scientifically admissible: {admissible}")
            ok = (
                result.succeeded
                and verdict_record.standing is OutcomeStanding.VERIFIED_EVIDENCE
                and admissible
                and usage.calls >= 1
            )

    for evidence_id in state.evidence_ids:
        evidence = runtime.store.get_evidence(evidence_id)
        print(f"evidence {evidence.id} ({evidence.kind}): {evidence.observation}")
    for note in report.notes:
        print(f"note: {note}")

    verdict = "reached verified evidence" if ok else "did NOT reach verified evidence"
    print(f"LIVE SLICE: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
