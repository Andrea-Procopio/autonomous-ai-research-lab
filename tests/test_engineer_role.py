"""The model-backed engineer, deterministically.

Everything here runs on :class:`FakeModelProvider` and harmless fixture
source. The invariants under test: metrics come only from executed
processes; unsafe or malformed model output never executes and is
preserved; accounting reaches the ledger exactly once on success and on
failure; provenance identifies the exact implementation that ran; and two
executions are two occurrences even with identical source.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.proposals import ProposalKind, ResultProposal
from autonomous_research_lab.execution.binding import HostPythonBinding
from autonomous_research_lab.execution.executor import (
    Executor,
    ExperimentJob,
    JobStatus,
)
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.roles.base import (
    RoleContext,
    RoleInvocation,
    RoleName,
)
from autonomous_research_lab.roles.engineer import (
    ENGINEER_INSTRUCTION,
    ENTRYPOINT,
    MAX_SOURCE_BYTES,
    PROPOSAL_SCHEMA,
    EngineerContractError,
    ImplementationRejectedError,
    ImplementationTemplate,
    ModelBackedEngineer,
)
from autonomous_research_lab.runtime.implementation_store import (
    ImplementationStore,
    SourceFile,
    source_tree_id,
)
from autonomous_research_lab.runtime.metrics import NO_USAGE, ProviderUsage
from autonomous_research_lab.runtime.preflight import PreflightError
from autonomous_research_lab.runtime.providers import (
    CallAccounting,
    FakeModelProvider,
    ProviderTransportError,
    ScriptedReply,
    StructuredOutputError,
    UsageLedger,
)

TEMPLATE = ImplementationTemplate(
    name="test-template-v1", source="# complete me\n"
)

#: Harmless fixture source obeying the executor contract. The numbers it
#: writes are deliberately different from anything a reply's rationale says.
GOOD_SOURCE = """\
import json
import os
from pathlib import Path

metrics = {"score": 0.25, "tiny_acc": 1.0}
run_dir = Path(os.environ["ARL_RUN_DIR"])
(run_dir / "metrics.json").write_text(json.dumps(metrics))
"""


def _spec(seeds: tuple[int, ...] = (7,)) -> ExperimentSpec:
    return ExperimentSpec(
        prediction_id="prd_0000000000000001",
        objective="measure the fixture score",
        procedure="write the fixture metrics through the process contract",
        metrics=("score", "tiny_acc"),
        seeds=seeds,
    )


def _reply(
    source: str = GOOD_SOURCE,
    rationale: str = "fixture implementation; score=0.99 is expected",
    path: str = ENTRYPOINT,
) -> str:
    return json.dumps(
        {"files": [{"path": path, "content": source}], "rationale": rationale}
    )


def _payload(*files: tuple[str, str], rationale: str = "fixture") -> str:
    return json.dumps(
        {
            "files": [{"path": p, "content": c} for p, c in files],
            "rationale": rationale,
        }
    )


class ForbiddenExecutor(Executor):
    """An executor that must never be reached — the assertion that
    rejection happens strictly before execution."""

    def submit(self, job: ExperimentJob) -> str:
        raise AssertionError("execution must not be reached")

    def status(self, job_id: str) -> JobStatus:  # pragma: no cover - unreached
        raise AssertionError("execution must not be reached")

    def collect(self, job_id: str) -> ExperimentResult:  # pragma: no cover
        raise AssertionError("execution must not be reached")


def _engineer(
    tmp_path: Path,
    replies: tuple[ScriptedReply | str, ...],
    *,
    executor: Executor | None = None,
    repairs: int = 1,
) -> tuple[ModelBackedEngineer, FakeModelProvider, UsageLedger, ImplementationStore]:
    provider = FakeModelProvider(replies)
    ledger = UsageLedger()
    store = ImplementationStore(tmp_path / "implementations")
    engineer = ModelBackedEngineer(
        provider=provider,
        model="test-model",
        executor=executor or LocalExecutor(tmp_path / "runs"),
        ledger=ledger,
        store=store,
        binding=HostPythonBinding(timeout_seconds=60.0),
        template=TEMPLATE,
        max_generation_repairs=repairs,
    )
    return engineer, provider, ledger, store


def _invocation(
    spec: ExperimentSpec,
    *,
    action_type: ResearchActionType = ResearchActionType.RUN_EXPERIMENT,
    results: tuple[ExperimentResult, ...] = (),
    experiments: tuple[ExperimentSpec, ...] | None = None,
) -> RoleInvocation:
    action = ResearchAction(
        action_type=action_type, rationale="assigned", targets=(spec.id,)
    )
    return RoleInvocation(
        role=RoleName.RESEARCH_ENGINEER,
        assignment=action,
        context=RoleContext(
            objective="test objective",
            experiments=(spec,) if experiments is None else experiments,
            results=results,
        ),
        allowed_actions=frozenset({action_type}),
        expected_output=frozenset({ProposalKind.RESULT}),
    )


def _prior_result(spec: ExperimentSpec, seed: int) -> ExperimentResult:
    return ExperimentResult(
        spec_id=spec.id,
        job_id=f"job_prior_{seed}",
        status=ExperimentStatus.COMPLETED,
        command=("python", "experiment.py"),
        environment=Environment(python_version="3.11", platform="test"),
        metrics={"score": 0.5, "tiny_acc": 1.0},
        seed=seed,
    )


# -- 1. the happy path ---------------------------------------------------------


def test_a_valid_reply_becomes_preserved_source_and_one_result(
    tmp_path: Path,
) -> None:
    engineer, _, _, store = _engineer(tmp_path, (_reply(),))
    spec = _spec()

    proposals = engineer.perform(_invocation(spec))

    (proposal,) = proposals
    assert isinstance(proposal, ResultProposal)
    result = proposal.result
    assert result.succeeded
    assert result.seed == 7
    # The source that ran is preserved, byte-exact and content-addressed.
    (record,) = store.records()
    source_path = store.source_dir(record.source_id) / ENTRYPOINT
    assert source_path.read_text() == GOOD_SOURCE
    assert record.manifest[ENTRYPOINT] == SourceFile(ENTRYPOINT, GOOD_SOURCE).sha256
    # The result links back to the implementation through job config.
    assert result.config["implementation_id"] == record.id
    assert result.config["source_id"] == record.source_id


def test_the_request_carries_spec_seed_contract_and_template(
    tmp_path: Path,
) -> None:
    engineer, provider, _, _ = _engineer(tmp_path, (_reply(),))
    spec = _spec()
    engineer.perform(_invocation(spec))

    (request,) = provider.calls
    assert request.instruction == ENGINEER_INSTRUCTION
    assert request.schema is PROPOSAL_SCHEMA
    (message,) = request.messages
    assert spec.procedure in message.content
    assert "score, tiny_acc" in message.content
    assert "seed for this run: 7" in message.content
    for contract_term in ("ARL_RUN_DIR", "ARL_CONFIG", "ARL_SEED", "metrics.json"):
        assert contract_term in message.content
    assert TEMPLATE.source in message.content
    assert request.metadata["spec_id"] == spec.id


# -- 2. metrics come from the process, never the model -------------------------


def test_metrics_come_from_the_executed_process_not_the_model(
    tmp_path: Path,
) -> None:
    engineer, _, _, _ = _engineer(
        tmp_path,
        (_reply(rationale="score=0.99, definitely; tiny_acc=0.0"),),
    )
    (proposal,) = engineer.perform(_invocation(_spec()))
    assert isinstance(proposal, ResultProposal)
    # What the process wrote — not what the rationale asserted.
    assert proposal.result.metrics == {"score": 0.25, "tiny_acc": 1.0}


def test_the_output_schema_has_nowhere_to_put_metrics_or_results() -> None:
    properties = PROPOSAL_SCHEMA.json_schema["properties"]
    assert isinstance(properties, Mapping)
    assert set(properties) == {"files", "rationale"}


def test_a_reply_smuggling_extra_fields_is_a_schema_violation(
    tmp_path: Path,
) -> None:
    smuggled = json.dumps(
        {
            "files": [{"path": ENTRYPOINT, "content": GOOD_SOURCE}],
            "rationale": "fixture",
            "metrics": {"score": 0.99},
        }
    )
    engineer, _, _, store = _engineer(
        tmp_path, (smuggled,), executor=ForbiddenExecutor()
    )
    with pytest.raises(StructuredOutputError):
        engineer.perform(_invocation(_spec()))
    assert store.records() == ()


# -- 3. the role holds no authority over state or results ----------------------


def test_the_result_is_the_executors_object_verbatim(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path / "runs")
    engineer, _, _, _ = _engineer(tmp_path, (_reply(),), executor=executor)
    (proposal,) = engineer.perform(_invocation(_spec()))
    assert isinstance(proposal, ResultProposal)
    result = proposal.result
    assert result is executor.collect(result.job_id)


# -- 4. deterministic rejection before execution -------------------------------


@pytest.mark.parametrize(
    ("reply", "reason_fragment"),
    [
        (_payload(("/tmp/evil.py", "x = 1\n")), "absolute path"),
        (_payload(("../evil.py", "x = 1\n")), "traversal"),
        (_payload(("helper.py", "x = 1\n")), "outside the allowlist"),
        (
            _payload((ENTRYPOINT, "x = 1\n"), ("helper.py", "y = 2\n")),
            "outside the allowlist",
        ),
        (
            _payload((ENTRYPOINT, "x = 1\n"), (ENTRYPOINT, "y = 2\n")),
            "duplicate",
        ),
        (_payload(rationale="nothing"), "no files"),
        (_payload((ENTRYPOINT, "def broken(:\n")), "does not compile"),
        (
            _payload((ENTRYPOINT, "# " + "x" * MAX_SOURCE_BYTES)),
            "byte limit",
        ),
        (_payload((ENTRYPOINT, "x = '\x00'")), "NUL"),
    ],
)
def test_unsafe_or_malformed_source_is_rejected_before_execution(
    tmp_path: Path, reply: str, reason_fragment: str
) -> None:
    engineer, _, _, store = _engineer(
        tmp_path, (reply,), executor=ForbiddenExecutor(), repairs=0
    )
    with pytest.raises(ImplementationRejectedError, match=reason_fragment):
        engineer.perform(_invocation(_spec()))
    # Nothing executed, nothing recorded as an implementation — but the
    # refused attempt is preserved as data, reason and payload included.
    assert store.records() == ()
    (rejected,) = store.rejected()
    assert reason_fragment in str(rejected["reason"])


def test_non_json_output_fails_closed_before_execution(tmp_path: Path) -> None:
    engineer, _, _, store = _engineer(
        tmp_path, ("this is not json",), executor=ForbiddenExecutor()
    )
    with pytest.raises(StructuredOutputError):
        engineer.perform(_invocation(_spec()))
    assert store.records() == ()


def test_missing_and_multiple_specs_are_contract_errors(tmp_path: Path) -> None:
    engineer, provider, _, _ = _engineer(
        tmp_path, (), executor=ForbiddenExecutor()
    )
    spec = _spec()
    with pytest.raises(EngineerContractError, match="exactly one"):
        engineer.perform(_invocation(spec, experiments=()))
    with pytest.raises(EngineerContractError, match="exactly one"):
        engineer.perform(_invocation(spec, experiments=(spec, _spec(seeds=(3,)))))
    assert provider.calls == ()  # rejected before any model spend


def test_unsupported_actions_are_contract_errors(tmp_path: Path) -> None:
    engineer, provider, _, _ = _engineer(
        tmp_path, (), executor=ForbiddenExecutor()
    )
    with pytest.raises(EngineerContractError, match="not analyze"):
        engineer.perform(
            _invocation(_spec(), action_type=ResearchActionType.ANALYZE)
        )
    assert provider.calls == ()


def test_a_failing_preflight_prevents_execution_but_keeps_the_record(
    tmp_path: Path,
) -> None:
    class UnresolvableBinding:
        def bind(
            self,
            *,
            spec_id: str,
            source_dir: Path,
            entrypoint: str,
            config: object,
            seed: int | None,
            job_id: str = "",
        ) -> ExperimentJob:
            return ExperimentJob(
                spec_id=spec_id,
                command=("no-such-binary-anywhere", str(source_dir / entrypoint)),
                seed=seed,
                id=job_id,
            )

    provider = FakeModelProvider((_reply(),))
    store = ImplementationStore(tmp_path / "implementations")
    engineer = ModelBackedEngineer(
        provider=provider,
        model="test-model",
        executor=ForbiddenExecutor(),
        ledger=UsageLedger(),
        store=store,
        binding=UnresolvableBinding(),
        template=TEMPLATE,
    )
    with pytest.raises(PreflightError):
        engineer.perform(_invocation(_spec()))
    # The implementation attempt is preserved even though it never launched.
    assert len(store.records()) == 1


# -- bounded generation repair: the one model-backed retry ---------------------


def test_a_rejected_generation_earns_exactly_one_corrective_call(
    tmp_path: Path,
) -> None:
    corrupted = _payload((ENTRYPOINT, "x = '\x00'"))  # the failure seen live
    engineer, provider, ledger, store = _engineer(
        tmp_path, (corrupted, _reply())
    )
    (proposal,) = engineer.perform(_invocation(_spec()))
    assert isinstance(proposal, ResultProposal)
    assert proposal.result.succeeded

    # The rejected first attempt is preserved; the repaired one executed.
    (rejected,) = store.rejected()
    assert "NUL" in str(rejected["reason"])
    assert len(store.records()) == 1
    # The corrective request carried the failed reply and the exact reason.
    first, second = provider.calls
    assert len(second.messages) == len(first.messages) + 2
    assert "NUL" in second.messages[-1].content
    assert second.metadata["generation_repair"] == "1"
    # Both calls are on the books.
    assert ledger.drain().calls == 2


def test_generation_repair_is_bounded(tmp_path: Path) -> None:
    corrupted = _payload((ENTRYPOINT, "x = '\x00'"))
    engineer, _, ledger, store = _engineer(
        tmp_path, (corrupted, corrupted), executor=ForbiddenExecutor()
    )
    with pytest.raises(ImplementationRejectedError, match="NUL"):
        engineer.perform(_invocation(_spec()))
    assert len(store.rejected()) == 2  # both refused attempts preserved
    assert store.records() == ()
    assert ledger.drain().calls == 2


# -- 5/6. accounting reaches the ledger exactly once ---------------------------


def test_success_accounting_reaches_the_ledger_exactly_once(
    tmp_path: Path,
) -> None:
    engineer, _, ledger, _ = _engineer(tmp_path, (_reply(),))
    engineer.perform(_invocation(_spec()))
    drained = ledger.drain()
    assert drained.calls == 1
    assert drained.output_tokens > 0
    assert drained.model == "test-model"
    assert ledger.drain() == NO_USAGE  # nothing recorded twice


def test_failure_accounting_reaches_the_ledger_exactly_once(
    tmp_path: Path,
) -> None:
    error = ProviderTransportError("upstream 500", status_code=500)
    error.with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=11, output_tokens=3, model="test-model"
            ),
            latency_seconds=0.2,
        )
    )
    engineer, _, ledger, store = _engineer(
        tmp_path, (ScriptedReply(error=error),), executor=ForbiddenExecutor()
    )
    with pytest.raises(ProviderTransportError):
        engineer.perform(_invocation(_spec()))
    drained = ledger.drain()
    assert drained == ProviderUsage(
        calls=1, input_tokens=11, output_tokens=3, model="test-model"
    )
    assert ledger.drain() == NO_USAGE
    # Provider failure produced nothing: no source, no record, no result.
    assert store.records() == ()
    assert store.rejected() == ()


# -- 7. provenance -------------------------------------------------------------


def test_provenance_identifies_the_exact_implementation(tmp_path: Path) -> None:
    reply = ScriptedReply(
        text=_reply(rationale="the rationale"),
        model="test-model-served",
        request_id="prov-req-1",
    )
    engineer, provider, _, store = _engineer(tmp_path, (reply,))
    invocation = _invocation(_spec())
    (proposal,) = engineer.perform(invocation)
    assert isinstance(proposal, ResultProposal)

    (record,) = store.records()
    (request,) = provider.calls
    assert record.invocation_id == invocation.id
    assert record.spec_id == invocation.context.experiments[0].id
    assert record.template_id == TEMPLATE.id
    assert record.template_sha256 == TEMPLATE.sha256
    assert record.source_id == source_tree_id(
        (SourceFile(ENTRYPOINT, GOOD_SOURCE),)
    )
    assert record.request_fingerprint == request.fingerprint
    assert record.response_id.startswith("mcall_")
    assert record.provider == "fake"
    assert record.requested_model == "test-model"
    assert record.served_model == "test-model-served"
    assert record.provider_request_id == "prov-req-1"
    assert record.rationale == "the rationale"
    assert record.entrypoint == ENTRYPOINT
    assert record.command == proposal.result.command
    assert record.seed == proposal.result.seed == 7
    # Round-trips from disk identically, so provenance survives the process.
    assert store.get(record.id) == record


# -- 8. occurrence identity ----------------------------------------------------


def test_two_executions_of_identical_source_are_two_occurrences(
    tmp_path: Path,
) -> None:
    engineer, _, _, store = _engineer(tmp_path, (_reply(), _reply()))
    spec = _spec()
    (first,) = engineer.perform(_invocation(spec))
    (second,) = engineer.perform(_invocation(spec))
    assert isinstance(first, ResultProposal)
    assert isinstance(second, ResultProposal)
    assert first.result.id != second.result.id
    assert first.result.job_id != second.result.job_id

    one, two = store.records()
    assert one.source_id == two.source_id  # identical source ...
    assert one.id != two.id  # ... two generation events
    assert len(list(store.source_dir(one.source_id).iterdir())) == 1


def test_replicate_picks_the_next_unused_seed(tmp_path: Path) -> None:
    engineer, _, _, _ = _engineer(tmp_path, (_reply(),))
    spec = _spec(seeds=(3, 9))
    (proposal,) = engineer.perform(
        _invocation(
            spec,
            action_type=ResearchActionType.REPLICATE,
            results=(_prior_result(spec, seed=3),),
        )
    )
    assert isinstance(proposal, ResultProposal)
    assert proposal.result.seed == 9


# -- 9. a negative result is a result ------------------------------------------


def test_a_disappointing_result_returns_normally(tmp_path: Path) -> None:
    negative = GOOD_SOURCE.replace('"score": 0.25', '"score": 0.01')
    engineer, _, _, _ = _engineer(tmp_path, (_reply(source=negative),))
    (proposal,) = engineer.perform(_invocation(_spec()))
    assert isinstance(proposal, ResultProposal)
    assert proposal.result.succeeded  # completed; its value is not our call
    assert proposal.result.metrics["score"] == 0.01
