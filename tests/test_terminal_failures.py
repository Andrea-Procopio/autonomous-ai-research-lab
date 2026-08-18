"""Terminal failures end the run after one durable record.

The preserved Task 4 partials exposed three loops of repeated dispatch
that no retry inside the run could improve: ten billed 401 calls on a
rotated key (partial-1), ten generations preflight-rejected by the same
hidden-``.pth`` host condition (partial-2), and ten planner attempts over
a state that could not satisfy the gate (partial-3). These tests pin the
correction: a rejected credential or a local configuration failure halts
after exactly one provider call, one durable failed attempt and zero
scientific-state mutation; the stable environment preflight failure halts
instead of re-dispatching; and an ordinary transient failure — a
truncated generation — still earns its retry and can succeed.
"""

from __future__ import annotations

import json
from pathlib import Path

from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.experiment import (
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.execution.binding import HostPythonBinding
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.orchestration.planning import PlanningDirector
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
from autonomous_research_lab.runtime.metrics import StepMetrics
from autonomous_research_lab.runtime.planning_store import PlanningStore
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    InvalidModelResponseError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ScriptedReply,
    UsageLedger,
)
from autonomous_research_lab.runtime.verification import (
    CheckState,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)
from autonomous_research_lab.runtime.verification_store import (
    InMemoryVerificationStore,
)

QUESTION = ResearchQuestion(text="does the fixture stream lean heads?")
HYPOTHESIS = Hypothesis(
    statement="the fixture stream is biased toward heads",
    question_id=QUESTION.id,
)
PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="one fixture stream",
    metric="heads_rate",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.5,
)
SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="measure the fixture heads rate",
    procedure="run the fixture and report heads_rate and tiny_acc",
    metrics=("heads_rate", "tiny_acc"),
    seeds=(7,),
    estimated_cost=ResourceCost(wall_clock_seconds=60.0),
)
TEMPLATE = ImplementationTemplate(
    name="terminal-fixture-template-v1", source="# start\n"
)
OVERFIT_CONTROL = PositiveControl(
    name="tiny_overfit",
    metric="tiny_acc",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.99,
)
CATALOG = TemplateCatalog(
    entries=(
        TemplateCapability(
            template=TEMPLATE,
            metrics=("heads_rate", "tiny_acc"),
            estimated_cost=ResourceCost(wall_clock_seconds=60.0),
            control=OVERFIT_CONTROL,
        ),
    )
)

GOOD_SOURCE = """\
import json
import os
from pathlib import Path

metrics = {"heads_rate": 0.75, "tiny_acc": 1.0}
run_dir = Path(os.environ["ARL_RUN_DIR"])
(run_dir / "metrics.json").write_text(json.dumps(metrics))
"""


def _engineer_reply() -> str:
    return json.dumps(
        {
            "files": [{"path": "experiment.py", "content": GOOD_SOURCE}],
            "rationale": "deterministic fixture",
        }
    )


def _auth_error() -> ProviderAuthenticationError:
    return ProviderAuthenticationError(
        "the Muse endpoint returned HTTP 401: Unauthorized",
        status_code=401,
        provider_error="invalid_api_key",
        request_id="req-observed-401",
    )


class PassMethodology:
    def review(
        self,
        spec: ExperimentSpec,
        prediction: Prediction | None,
        *,
        objective: str,
    ) -> VerificationCheck:
        return VerificationCheck(
            dimension=ValidityDimension.METHODOLOGY,
            name="methodological_validity",
            state=CheckState.PASS,
            detail="fixture design reviewed at wiring time",
        )


class _FailingCheck:
    """A preflight check that always fails under a chosen name — the
    stable environment name, or a job-defect name a later generation
    could fix."""

    def __init__(self, name: str) -> None:
        self._name = name

    def check(
        self, job: object, spec: ExperimentSpec | None
    ) -> VerificationCheck:
        return VerificationCheck(
            dimension=ValidityDimension.EXECUTION,
            name=self._name,
            state=CheckState.FAIL,
            detail="deterministic fixture failure",
        )


class ListSink:
    def __init__(self) -> None:
        self.records: list[StepMetrics] = []

    def log(self, record: StepMetrics) -> None:
        self.records.append(record)


class Rig:
    """One runtime over fresh stores, with every assertion surface kept."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        engineer_replies: tuple[ScriptedReply | str, ...],
        preflight_checks: tuple[object, ...] | None = None,
    ) -> None:
        self.ledger = UsageLedger()
        self.engineer_provider = FakeModelProvider(engineer_replies)
        self.planner_provider = FakeModelProvider(())
        self.plans = PlanningStore(tmp_path / "planning")
        self.sink = ListSink()
        extra: dict[str, object] = {}
        if preflight_checks is not None:
            extra["preflight_checks"] = preflight_checks
        engineer = ModelBackedEngineer(
            provider=self.engineer_provider,
            model="test-model",
            executor=LocalExecutor(tmp_path / "runs"),
            ledger=self.ledger,
            store=ImplementationStore(tmp_path / "implementations"),
            binding=HostPythonBinding(timeout_seconds=60.0),
            template=TEMPLATE,
            **extra,  # type: ignore[arg-type]
        )
        planner = ModelBackedPlanner(
            provider=self.planner_provider,
            model="test-model",
            ledger=self.ledger,
            store=self.plans,
            catalog=CATALOG,
        )
        self.runtime = ResearchRuntime(
            config=RuntimeConfig(),
            director=PlanningDirector(plans=self.plans),
            roles={
                RoleName.RESEARCH_ENGINEER: engineer,
                RoleName.RESEARCH_DIRECTOR: planner,
            },
            store=InMemoryEvidenceStore(),
            metrics=self.sink,
            usage=self.ledger,
            methodology_reviewer=PassMethodology(),
            control_source=lambda _spec: (OVERFIT_CONTROL,),
            verifications=InMemoryVerificationStore(),
        )

    def recorded_provider_calls(self) -> int:
        return sum(record.provider_calls for record in self.sink.records)


def _initial_state() -> ResearchState:
    return (
        ResearchState(
            objective=QUESTION.text,
            budget=ResearchBudget(
                wall_clock_seconds=3600.0, usd=10.0, model_tokens=200_000
            ),
        )
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(PREDICTION)
        .add_experiment(SPEC)
    )


def _assert_no_scientific_mutation(state: ResearchState) -> None:
    assert len(state.hypotheses) == 1
    assert len(state.predictions) == 1
    assert len(state.experiments) == 1
    assert state.results == ()
    assert state.evidence_ids == ()


def test_a_rejected_credential_halts_after_exactly_one_call(
    tmp_path: Path,
) -> None:
    rig = Rig(
        tmp_path,
        engineer_replies=tuple(
            ScriptedReply(error=_auth_error()) for _ in range(5)
        ),
    )

    outcome = rig.runtime.run(_initial_state(), max_steps=10)

    # A ten-step allowance produced exactly one provider call: the loop
    # halted on the first refusal instead of re-billing it nine times.
    assert len(rig.engineer_provider.calls) == 1
    assert len(rig.planner_provider.calls) == 0
    assert "terminal provider failure" in outcome.halt_reason
    assert "credential" in outcome.halt_reason
    assert len(outcome.reports) == 1

    # One durable failed attempt, zero scientific-state mutation.
    (attempt,) = outcome.state.attempts
    assert attempt.outcome is not None
    assert "401" in str(attempt.outcome.error)
    _assert_no_scientific_mutation(outcome.state)

    # No planner involvement of any kind: no decision, no rejection.
    assert rig.plans.records() == ()
    assert rig.plans.rejected() == ()

    # Accounting is exactly once, and unknown is not zero: the refusal
    # carried no usage, so nothing entered the recorded spend.
    assert rig.recorded_provider_calls() == 0


def test_a_configuration_failure_is_terminal_and_mutates_nothing(
    tmp_path: Path,
) -> None:
    rig = Rig(
        tmp_path,
        engineer_replies=tuple(
            ScriptedReply(
                error=ProviderConfigurationError("MUSE_API_KEY is not set")
            )
            for _ in range(5)
        ),
    )

    outcome = rig.runtime.run(_initial_state(), max_steps=10)

    assert len(rig.engineer_provider.calls) == 1
    assert "terminal configuration failure" in outcome.halt_reason
    _assert_no_scientific_mutation(outcome.state)


def test_the_hidden_pth_environment_failure_is_not_redispatched(
    tmp_path: Path,
) -> None:
    rig = Rig(
        tmp_path,
        engineer_replies=tuple(_engineer_reply() for _ in range(5)),
        preflight_checks=(_FailingCheck("preflight:pth_files_visible"),),
    )

    outcome = rig.runtime.run(_initial_state(), max_steps=6)

    # Partial-2 billed ten generations against the same broken host; now
    # the first diagnosis halts the run.
    assert len(rig.engineer_provider.calls) == 1
    assert "terminal environment failure" in outcome.halt_reason
    assert "pth_files_visible" in outcome.halt_reason
    _assert_no_scientific_mutation(outcome.state)
    # The one successful generation was recorded exactly once.
    assert rig.recorded_provider_calls() == 1


def test_a_job_defect_preflight_failure_stays_retryable(
    tmp_path: Path,
) -> None:
    """Only the stable environment check is terminal: a preflight failure
    the next generation could fix is re-dispatched on the loop's own
    economics."""
    rig = Rig(
        tmp_path,
        engineer_replies=tuple(_engineer_reply() for _ in range(5)),
        preflight_checks=(_FailingCheck("preflight:command_resolvable"),),
    )

    outcome = rig.runtime.run(_initial_state(), max_steps=2)

    assert len(rig.engineer_provider.calls) == 2
    assert outcome.halt_reason == "step limit of 2 reached"


def test_one_transient_failure_then_success_still_works(
    tmp_path: Path,
) -> None:
    rig = Rig(
        tmp_path,
        engineer_replies=(
            ScriptedReply(
                error=InvalidModelResponseError(
                    "the generation was truncated (finish_reason 'length')"
                )
            ),
            _engineer_reply(),
        ),
    )

    state = _initial_state()
    first = rig.runtime.step(state)
    assert first.halt_reason is None  # transient: the run continues
    second = rig.runtime.step(first.state)

    assert len(rig.engineer_provider.calls) == 2
    (ref,) = second.state.results
    assert ref.status is ExperimentStatus.COMPLETED
    assert len(second.state.evidence_ids) == 1
    # The failure recorded no spend (unknown is not zero); the success
    # recorded exactly one call.
    assert rig.recorded_provider_calls() == 1
