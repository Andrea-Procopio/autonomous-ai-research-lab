"""The scientific-promotion gate: raw observation != verified support.

The invariant under test: an observation whose verification is unresolved
or adverse is preserved and inspectable, but cannot silently become trusted
support for a claim or a conclusive assessment. Results never verified at
all (the ablated lab) keep legacy semantics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.attempt import AttemptStatus
from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.claim import Claim, EvidenceLink, EvidenceRelation
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    Proposal,
    ResultProposal,
)
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import (
    PromotionError,
    ResearchRuntime,
)
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.metrics import StepMetrics
from autonomous_research_lab.runtime.verification import (
    CheckState,
    ExperimentValidityStatus,
    OutcomeStanding,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
    VerificationReport,
)
from autonomous_research_lab.runtime.verification_store import (
    verification_record,
)

QUESTION = ResearchQuestion(text="Is the stream fair?")
HYPOTHESIS = Hypothesis(statement="The stream is biased.", question_id=QUESTION.id)

_ECHO = (
    "import json, os, pathlib; "
    "d = pathlib.Path(os.environ['ARL_RUN_DIR']); "
    "cfg = json.loads(pathlib.Path(os.environ['ARL_CONFIG']).read_text()); "
    "(d / 'metrics.json').write_text(json.dumps(cfg))"
)

OVERFIT_CONTROL = PositiveControl(
    name="tiny_overfit",
    metric="overfit_acc",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.99,
)


def _spec_and_prediction() -> tuple[ExperimentSpec, Prediction]:
    prediction = Prediction(
        hypothesis_id=HYPOTHESIS.id,
        condition="one draw stream",
        metric="heads_rate",
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=0.5,
    )
    spec = ExperimentSpec(
        prediction_id=prediction.id,
        objective="measure the rate",
        procedure="run the stream and report",
        metrics=("heads_rate",),
        seeds=(7,),
    )
    return spec, prediction


def _prepared_state(
    spec: ExperimentSpec, prediction: Prediction
) -> ResearchState:
    return (
        ResearchState(
            objective="fairness",
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
class StubEngineer(ResearchRole):
    executor: LocalExecutor
    metrics_payload: dict[str, float]
    performed: int = 0

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_ENGINEER

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset({ResearchActionType.RUN_EXPERIMENT})

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        self.performed += 1
        (spec,) = invocation.context.experiments
        job = ExperimentJob(
            spec_id=spec.id,
            command=(sys.executable, "-c", _ECHO),
            config=dict(self.metrics_payload),
            seed=7,
            timeout_seconds=30.0,
        )
        result = self.executor.collect(self.executor.submit(job))
        return (ResultProposal(result=result, proposer="stub:engineer"),)


@dataclass
class StubScientist(ResearchRole):
    """Scientist seat that synthesizes every evidence into a claim, citing
    it with a configurable relation — the promotion attempt under test."""

    relation: EvidenceRelation = EvidenceRelation.SUPPORTS
    invocations: list[RoleInvocation] = field(default_factory=list)

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_DIRECTOR

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset({ResearchActionType.SYNTHESIZE_FINDING})

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        self.invocations.append(invocation)
        (evidence,) = invocation.context.evidence
        (hypothesis,) = invocation.context.hypotheses
        claim = Claim(
            statement=hypothesis.statement,
            scope="the recorded draw stream",
            hypothesis_id=hypothesis.id,
        )
        link = EvidenceLink(
            claim_id=claim.id,
            evidence_id=evidence.id,
            relation=self.relation,
            rationale="synthesized by the stub scientist",
        )
        return (
            ClaimProposal(claim=claim, links=(link,), proposer="stub:scientist"),
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
        )


@dataclass
class ListSink:
    records: list[StepMetrics] = field(default_factory=list)

    def log(self, record: StepMetrics) -> None:
        self.records.append(record)


def _runtime(
    tmp_path: Path,
    metrics_payload: dict[str, float],
    *,
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
    methodology: PassMethodology | None = None,
) -> tuple[ResearchRuntime, StubScientist, ListSink]:
    scientist = StubScientist(relation=relation)
    sink = ListSink()
    runtime = ResearchRuntime(
        config=RuntimeConfig(),
        director=RuleBasedFrontierDirector(),
        roles={
            RoleName.RESEARCH_ENGINEER: StubEngineer(
                LocalExecutor(tmp_path / "runs"), metrics_payload
            ),
            RoleName.RESEARCH_DIRECTOR: scientist,
        },
        store=InMemoryEvidenceStore(),
        metrics=sink,
        methodology_reviewer=methodology or PassMethodology(),
        control_source=lambda spec: (OVERFIT_CONTROL,),
    )
    return runtime, scientist, sink


# -- case A: verified evidence supports normally ------------------------------


def test_verified_evidence_may_support_a_claim(tmp_path: Path) -> None:
    runtime, _, sink = _runtime(
        tmp_path, {"heads_rate": 0.45, "overfit_acc": 1.0}
    )
    spec, prediction = _spec_and_prediction()

    first = runtime.step(_prepared_state(spec, prediction))
    assert sink.records[-1].verification_status == (
        ExperimentValidityStatus.VERIFIED
    )
    second = runtime.step(first.state)  # synthesize the verified negative

    (claim,) = second.state.claims
    (link,) = second.state.evidence_links
    assert link.relation is EvidenceRelation.SUPPORTS
    assert claim.hypothesis_id == HYPOTHESIS.id
    assert not sink.records[-1].promotion_blocked


# -- cases B/E/F: unresolved observation is preserved, never promoted ---------


def test_unresolved_evidence_cannot_silently_support_a_claim(
    tmp_path: Path,
) -> None:
    runtime, _, sink = _runtime(
        tmp_path, {"heads_rate": 0.45, "overfit_acc": 0.5}  # silent bug
    )
    spec, prediction = _spec_and_prediction()

    first = runtime.step(_prepared_state(spec, prediction))
    (result_ref,) = first.state.results
    verdict = runtime.verifications.get(result_ref.result_id)
    assert verdict is not None
    assert verdict.validity is ExperimentValidityStatus.IMPLEMENTATION_UNCERTAIN
    assert verdict.standing is OutcomeStanding.OBSERVED_UNRESOLVED

    second = runtime.step(first.state)  # the synthesis attempt

    # The promotion was rejected: no claim, no link, failed attempt.
    assert second.state.claims == ()
    assert second.state.evidence_links == ()
    synthesis_attempt = next(
        a
        for a in second.state.attempts
        if a.action.action_type is ResearchActionType.SYNTHESIZE_FINDING
    )
    assert synthesis_attempt.status is AttemptStatus.FAILED
    assert sink.records[-1].promotion_blocked
    assert any("scientific promotion blocked" in n for n in second.notes)

    # Case E: the raw observation is fully preserved and inspectable —
    # evidence in state and store, verification verdict durable.
    (evidence_id,) = second.state.evidence_ids
    evidence = runtime.store.get_evidence(evidence_id)
    assert evidence.result_id == result_ref.result_id
    assert evidence.metrics["heads_rate"] == 0.45
    assert runtime.verifications.get(result_ref.result_id) == verdict


def test_scientist_sees_the_unresolved_standing_in_its_context(
    tmp_path: Path,
) -> None:
    """A reasoning seat can distinguish verified from observed-unresolved
    without global state: its context notes carry the durable standing."""
    runtime, scientist, _ = _runtime(
        tmp_path, {"heads_rate": 0.45, "overfit_acc": 0.5}
    )
    spec, prediction = _spec_and_prediction()
    first = runtime.step(_prepared_state(spec, prediction))
    runtime.step(first.state)

    (invocation,) = scientist.invocations
    assert any(
        "observed_unresolved" in note and "implementation_uncertain" in note
        for note in invocation.context.notes
    )


def test_inconclusive_citation_of_unresolved_evidence_is_allowed(
    tmp_path: Path,
) -> None:
    """Inspection and factual annotation stay open: only trusted bearing
    (supports/contradicts) requires verification."""
    runtime, _, sink = _runtime(
        tmp_path,
        {"heads_rate": 0.45, "overfit_acc": 0.5},
        relation=EvidenceRelation.INCONCLUSIVE,
    )
    spec, prediction = _spec_and_prediction()
    first = runtime.step(_prepared_state(spec, prediction))
    second = runtime.step(first.state)

    (link,) = second.state.evidence_links
    assert link.relation is EvidenceRelation.INCONCLUSIVE
    assert not sink.records[-1].promotion_blocked


# -- the gate itself, across every validity status ----------------------------


def _gate_fixture(
    validity_report: VerificationReport | None,
) -> tuple[ResearchRuntime, Evidence]:
    store = InMemoryEvidenceStore()
    runtime = ResearchRuntime(
        config=RuntimeConfig(),
        director=RuleBasedFrontierDirector(),
        roles={},
        store=store,
    )
    result = ExperimentResult(
        spec_id="exp_g",
        job_id="job_g",
        status=ExperimentStatus.COMPLETED,
        command=("cmd",),
        environment=Environment(python_version="3", platform="test"),
        metrics={"heads_rate": 0.48},
        exit_code=0,
        seed=0,
    )
    store.record_result(result)
    evidence = store.record_evidence(
        Evidence(
            result_id=result.id,
            spec_id=result.spec_id,
            kind=EvidenceKind.MEASUREMENT,
            observation="heads_rate = 0.48",
        )
    )
    if validity_report is not None:
        runtime.verifications.record(
            verification_record(result.id, result.spec_id, validity_report)
        )
    return runtime, evidence


def _single_dimension_report(
    dimension: ValidityDimension, state: CheckState
) -> VerificationReport:
    baseline = {
        ValidityDimension.EXECUTION: CheckState.PASS,
        ValidityDimension.IMPLEMENTATION: CheckState.PASS,
        ValidityDimension.METHODOLOGY: CheckState.PASS,
        ValidityDimension.ANALYSIS: CheckState.PASS,
    }
    baseline[dimension] = state
    return VerificationReport(
        checks=tuple(
            VerificationCheck(dimension=d, name=f"{d}_check", state=s)
            for d, s in baseline.items()
        )
    )


def _supporting_assessment(evidence: Evidence) -> AssessmentProposal:
    return AssessmentProposal(
        assessment=EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.SUPPORTED,
            method="test:direct",
            evidence_ids=(evidence.id,),
        ),
        proposer="test",
    )


def test_every_non_verified_status_blocks_conclusive_use() -> None:
    """Cases B, C and D of the downstream-gating table: implementation
    uncertainty, methodological invalidity, and plain unresolved validity
    all block trusted use — deterministically, with no model consulted."""
    adverse = (
        _single_dimension_report(ValidityDimension.IMPLEMENTATION, CheckState.FAIL),
        _single_dimension_report(ValidityDimension.METHODOLOGY, CheckState.FAIL),
        _single_dimension_report(ValidityDimension.ANALYSIS, CheckState.FAIL),
        _single_dimension_report(
            ValidityDimension.METHODOLOGY, CheckState.UNCERTAIN
        ),
    )
    for report in adverse:
        runtime, evidence = _gate_fixture(report)
        with pytest.raises(PromotionError, match="cannot serve as"):
            runtime._gate_promotions((_supporting_assessment(evidence),))


def test_verified_status_permits_conclusive_use() -> None:
    runtime, evidence = _gate_fixture(
        _single_dimension_report(ValidityDimension.ANALYSIS, CheckState.PASS)
    )
    runtime._gate_promotions((_supporting_assessment(evidence),))  # no raise


def test_undetermined_assessments_remain_open_to_any_observation() -> None:
    runtime, evidence = _gate_fixture(
        _single_dimension_report(ValidityDimension.IMPLEMENTATION, CheckState.FAIL)
    )
    undetermined = AssessmentProposal(
        assessment=EpistemicAssessment(
            subject_id=HYPOTHESIS.id,
            verdict=AssessmentVerdict.UNDETERMINED,
            method="test:direct",
            evidence_ids=(evidence.id,),
        ),
        proposer="test",
    )
    runtime._gate_promotions((undetermined,))  # no raise: it claims nothing


def test_never_verified_results_keep_legacy_semantics() -> None:
    """A result with no verification record predates or was excluded from
    verification: the gate has no verdict to enforce and stays silent."""
    runtime, evidence = _gate_fixture(None)
    runtime._gate_promotions((_supporting_assessment(evidence),))  # no raise
