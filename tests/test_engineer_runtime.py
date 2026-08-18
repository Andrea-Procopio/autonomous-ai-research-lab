"""The model-backed engineer inside the real runtime loop.

Deterministic end-to-end coverage of the Task 3 slice: the director selects
the run, the engineer turns a (fake) model reply into preserved source and
an executed job, the Tier-0 gate validates, the commit is atomic, evidence
is transcribed, verification produces a durable record, and — with a
passing methodology review and positive control — the deterministic fixture
reaches ``VERIFIED_EVIDENCE``. Failure modes stay in their lanes: provider
failure commits nothing scientific, a crashed run is an honest execution
failure entering only the typed repair path, and a verified negative is
evidence, not a debugging trigger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from autonomous_research_lab.core.attempt import AttemptStatus
from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.evidence import EvidenceKind
from autonomous_research_lab.core.experiment import ExperimentSpec, ExperimentStatus
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
)
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.execution.binding import HostPythonBinding
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.roles.base import RoleName
from autonomous_research_lab.roles.engineer import (
    ENTRYPOINT,
    ImplementationTemplate,
    ModelBackedEngineer,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.implementation_store import ImplementationStore
from autonomous_research_lab.runtime.metrics import ProviderUsage, StepMetrics
from autonomous_research_lab.runtime.providers import (
    CallAccounting,
    FakeModelProvider,
    ProviderTimeoutError,
    ScriptedReply,
    UsageLedger,
)
from autonomous_research_lab.runtime.verification import (
    CheckState,
    ExperimentValidityStatus,
    OutcomeStanding,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)
from autonomous_research_lab.runtime.verification_store import (
    ScientificAdmissibility,
)

TEMPLATE = ImplementationTemplate(name="loop-template-v1", source="# start\n")

QUESTION = ResearchQuestion(text="Does the fixture stream lean heads?")
HYPOTHESIS = Hypothesis(
    statement="The fixture stream is biased toward heads.",
    question_id=QUESTION.id,
)

OVERFIT_CONTROL = PositiveControl(
    name="tiny_overfit",
    metric="tiny_acc",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.99,
)


def _source(heads_rate: float) -> str:
    return f"""\
import json
import os
from pathlib import Path

metrics = {{"heads_rate": {heads_rate}, "tiny_acc": 1.0}}
run_dir = Path(os.environ["ARL_RUN_DIR"])
(run_dir / "metrics.json").write_text(json.dumps(metrics))
"""


CRASHING_SOURCE = """\
import sys

sys.exit(3)
"""


def _reply(source: str) -> str:
    return json.dumps(
        {
            "files": [{"path": "experiment.py", "content": source}],
            "rationale": "deterministic fixture",
        }
    )


def _spec_and_prediction() -> tuple[ExperimentSpec, Prediction]:
    prediction = Prediction(
        hypothesis_id=HYPOTHESIS.id,
        condition="one fixture stream",
        metric="heads_rate",
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=0.5,
    )
    spec = ExperimentSpec(
        prediction_id=prediction.id,
        objective="measure the fixture heads rate",
        procedure="run the fixture and report heads_rate and tiny_acc",
        metrics=("heads_rate", "tiny_acc"),
        seeds=(7,),
    )
    return spec, prediction


def _prepared_state(
    spec: ExperimentSpec, prediction: Prediction
) -> ResearchState:
    return (
        ResearchState(
            objective="fixture bias",
            budget=ResearchBudget(
                wall_clock_seconds=3600.0, usd=10.0, model_tokens=200_000
            ),
        )
        .upsert_question(QUESTION)
        .upsert_hypothesis(HYPOTHESIS)
        .upsert_prediction(prediction)
        .add_experiment(spec)
    )


@dataclass
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


@dataclass
class ListSink:
    records: list[StepMetrics] = field(default_factory=list)

    def log(self, record: StepMetrics) -> None:
        self.records.append(record)


def _runtime(
    tmp_path: Path, replies: tuple[ScriptedReply | str, ...]
) -> tuple[ResearchRuntime, ImplementationStore, ListSink]:
    ledger = UsageLedger()
    implementations = ImplementationStore(tmp_path / "implementations")
    engineer = ModelBackedEngineer(
        provider=FakeModelProvider(replies),
        model="test-model",
        executor=LocalExecutor(tmp_path / "runs"),
        ledger=ledger,
        store=implementations,
        binding=HostPythonBinding(timeout_seconds=60.0),
        template=TEMPLATE,
    )
    sink = ListSink()
    runtime = ResearchRuntime(
        config=RuntimeConfig(),
        director=RuleBasedFrontierDirector(),
        roles={RoleName.RESEARCH_ENGINEER: engineer},
        store=InMemoryEvidenceStore(),
        metrics=sink,
        usage=ledger,
        methodology_reviewer=PassMethodology(),
        control_source=lambda _spec: (OVERFIT_CONTROL,),
    )
    return runtime, implementations, sink


# -- 10. the full slice reaches verified evidence ------------------------------


def test_the_slice_commits_transcribes_verifies_and_admits(
    tmp_path: Path,
) -> None:
    runtime, implementations, sink = _runtime(
        tmp_path, (_reply(_source(heads_rate=0.75)),)
    )
    spec, prediction = _spec_and_prediction()

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state

    # The attempt succeeded and exactly one result committed.
    attempt = next(iter(state.attempts))
    assert attempt.status is AttemptStatus.SUCCEEDED
    (ref,) = state.results
    assert ref.status is ExperimentStatus.COMPLETED
    result = runtime.store.get_result(ref.result_id)
    assert result.metrics == {"heads_rate": 0.75, "tiny_acc": 1.0}

    # The mechanical prediction test read the pre-registered comparison.
    test = state.test_for_result(prediction.id, result.id)
    assert test is not None
    assert test.consistency is Consistency.CONSISTENT

    # Deterministic transcription into evidence, kind MEASUREMENT.
    (evidence_id,) = state.evidence_ids
    evidence = runtime.store.get_evidence(evidence_id)
    assert evidence.kind is EvidenceKind.MEASUREMENT
    assert evidence.result_id == result.id

    # The durable verification record: every dimension resolved.
    verdict = runtime.verifications.get(result.id)
    assert verdict is not None
    assert verdict.validity is ExperimentValidityStatus.VERIFIED
    assert verdict.standing is OutcomeStanding.VERIFIED_EVIDENCE
    assert ScientificAdmissibility(
        verifications=runtime.verifications, governance_enabled=True
    )(result.id)

    # Implementation provenance survived the loop and names this result.
    (record,) = implementations.records()
    assert result.config["implementation_id"] == record.id
    assert result.config["source_id"] == record.source_id
    assert (
        implementations.source_dir(record.source_id) / ENTRYPOINT
    ).read_text() == _source(heads_rate=0.75)

    # Provider-reported usage reached the step metrics through the ledger.
    metrics = sink.records[-1]
    assert metrics.provider_calls == 1
    assert metrics.input_tokens > 0
    assert metrics.output_tokens > 0
    assert metrics.model == "test-model"
    assert report.provider_usage.calls == 1


# -- 9/11. a verified negative is evidence, not a repair trigger ---------------


def test_a_verified_negative_is_accepted_and_never_debugged(
    tmp_path: Path,
) -> None:
    runtime, _, sink = _runtime(tmp_path, (_reply(_source(heads_rate=0.25)),))
    spec, prediction = _spec_and_prediction()

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state

    (ref,) = state.results
    result = runtime.store.get_result(ref.result_id)
    test = state.test_for_result(prediction.id, result.id)
    assert test is not None
    assert test.consistency is Consistency.INCONSISTENT

    verdict = runtime.verifications.get(result.id)
    assert verdict is not None
    assert verdict.standing is OutcomeStanding.VERIFIED_EVIDENCE
    (evidence_id,) = state.evidence_ids
    assert (
        runtime.store.get_evidence(evidence_id).kind is EvidenceKind.NULL_RESULT
    )

    metrics = sink.records[-1]
    assert metrics.negative_result_verdict == "accepted"
    assert metrics.debug_attempts == 0
    assert metrics.implementation_debug_attempts == 0
    assert metrics.failure_category == ""
    assert any("verified scientific negative" in note for note in report.notes)


# -- 6. provider failure commits nothing scientific ----------------------------


def test_provider_failure_creates_no_result_and_no_evidence(
    tmp_path: Path,
) -> None:
    error = ProviderTimeoutError("deadline exceeded", timeout_seconds=240.0)
    error.with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=21, output_tokens=0, model="test-model"
            ),
            latency_seconds=240.0,
        )
    )
    runtime, implementations, sink = _runtime(
        tmp_path, (ScriptedReply(error=error),)
    )
    spec, prediction = _spec_and_prediction()

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state

    attempt = next(iter(state.attempts))
    assert attempt.status is AttemptStatus.FAILED
    assert state.results == ()
    assert state.evidence_ids == ()
    assert implementations.records() == ()
    assert any("engineering failure" in note for note in report.notes)
    # The failed call's accounting still reached the step metrics.
    metrics = sink.records[-1]
    assert metrics.provider_calls == 1
    assert metrics.input_tokens == 21
    assert metrics.failures == 1


# -- 11. a crashed run is an honest execution failure --------------------------


def test_a_crashed_run_is_recorded_and_enters_the_typed_repair_path(
    tmp_path: Path,
) -> None:
    runtime, implementations, sink = _runtime(
        tmp_path, (_reply(CRASHING_SOURCE),)
    )
    spec, prediction = _spec_and_prediction()

    report = runtime.step(_prepared_state(spec, prediction))
    state = report.state

    # The failed execution is committed as a fact, not hidden.
    (ref,) = state.results
    assert ref.status is ExperimentStatus.FAILED
    result = runtime.store.get_result(ref.result_id)
    assert result.exit_code == 3
    assert result.metrics == {}
    # Its implementation remains preserved and linked.
    (record,) = implementations.records()
    assert result.config["implementation_id"] == record.id

    # Deterministic diagnosis ran; with no debugger wired, no repair —
    # and the failure never becomes evidence or a verification record.
    assert any(
        "engineering failure diagnosed" in note for note in report.notes
    )
    assert state.evidence_ids == ()
    assert runtime.verifications.get(result.id) is None
    metrics = sink.records[-1]
    assert metrics.failure_category == "nonzero_exit"
    assert metrics.debug_attempts == 0
