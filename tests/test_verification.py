"""The experiment-validity model: check states, status derivation, the
outcome gate's orthogonality, positive controls, and analysis coverage."""

from __future__ import annotations

import dataclasses

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.prediction import Comparator
from autonomous_research_lab.core.proposals import AssessmentProposal, Proposal
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.orchestration.review import (
    RoleBackedImplementationVerifier,
    RoleBackedMethodologyReviewer,
)
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.runtime.verification import (
    CheckState,
    ExperimentValidityStatus,
    OutcomeStanding,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
    VerificationReport,
    derive_validity,
    evaluate_controls,
    outcome_standing,
    verify_analysis_coverage,
)

_D = ValidityDimension
_S = CheckState


def _check(dimension: _D, state: _S, name: str = "c") -> VerificationCheck:
    return VerificationCheck(dimension=dimension, name=name, state=state)


def _report(**dimensions: _S) -> VerificationReport:
    return VerificationReport(
        checks=tuple(
            _check(_D(name), state, name=f"{name}_check")
            for name, state in dimensions.items()
        )
    )


ALL_PASS = _report(
    execution=_S.PASS,
    implementation=_S.PASS,
    methodology=_S.PASS,
    analysis=_S.PASS,
)


# -- dimension aggregation ----------------------------------------------------


def test_any_fail_dominates_a_dimension() -> None:
    report = VerificationReport(
        checks=(
            _check(_D.IMPLEMENTATION, _S.PASS, "reviewer_says_fine"),
            _check(_D.IMPLEMENTATION, _S.FAIL, "control_failed"),
        )
    )
    # A semantic PASS can never wash out a deterministic FAIL.
    assert report.dimension_state(_D.IMPLEMENTATION) is _S.FAIL


def test_uncertain_blocks_pass() -> None:
    report = VerificationReport(
        checks=(
            _check(_D.METHODOLOGY, _S.PASS),
            _check(_D.METHODOLOGY, _S.UNCERTAIN, "unresolved"),
        )
    )
    assert report.dimension_state(_D.METHODOLOGY) is _S.UNCERTAIN


def test_an_unchecked_dimension_is_not_applicable_never_passing() -> None:
    report = _report(execution=_S.PASS)
    assert report.dimension_state(_D.ANALYSIS) is _S.NOT_APPLICABLE


# -- validity derivation ------------------------------------------------------


def test_all_dimensions_resolved_is_verified() -> None:
    assert derive_validity(ALL_PASS) is ExperimentValidityStatus.VERIFIED


def test_execution_failure_dominates_everything() -> None:
    report = _report(
        execution=_S.FAIL,
        implementation=_S.FAIL,
        methodology=_S.FAIL,
        analysis=_S.FAIL,
    )
    assert derive_validity(report) is ExperimentValidityStatus.ENGINEERING_FAILED


def test_methodology_failure_means_redesign_not_verify() -> None:
    report = _report(
        execution=_S.PASS,
        implementation=_S.UNCERTAIN,
        methodology=_S.FAIL,
        analysis=_S.PASS,
    )
    assert (
        derive_validity(report)
        is ExperimentValidityStatus.METHODOLOGICALLY_INVALID
    )


def test_uncertain_implementation_is_implementation_uncertain() -> None:
    report = _report(
        execution=_S.PASS,
        implementation=_S.UNCERTAIN,
        methodology=_S.PASS,
        analysis=_S.PASS,
    )
    assert (
        derive_validity(report)
        is ExperimentValidityStatus.IMPLEMENTATION_UNCERTAIN
    )


def test_analysis_failure_is_analytically_invalid() -> None:
    report = _report(
        execution=_S.PASS,
        implementation=_S.PASS,
        methodology=_S.PASS,
        analysis=_S.FAIL,
    )
    assert derive_validity(report) is ExperimentValidityStatus.ANALYTICALLY_INVALID


def test_unresolved_dimensions_yield_unverified_not_verified() -> None:
    """The honest intermediate state: outcome observed, validity unresolved."""
    report = _report(
        execution=_S.PASS,
        implementation=_S.PASS,
        methodology=_S.UNCERTAIN,
        analysis=_S.PASS,
    )
    assert derive_validity(report) is ExperimentValidityStatus.UNVERIFIED


# -- the gate is orthogonal to outcome ----------------------------------------


def test_verified_validity_promotes_any_outcome_to_evidence() -> None:
    """VERIFIED x negative prediction test = a valid scientific negative.
    The gate never asks which way the result went."""
    assert (
        outcome_standing(ExperimentValidityStatus.VERIFIED)
        is OutcomeStanding.VERIFIED_EVIDENCE
    )


def test_every_unresolved_status_defers_promotion() -> None:
    for status in ExperimentValidityStatus:
        if status is ExperimentValidityStatus.VERIFIED:
            continue
        assert outcome_standing(status) is OutcomeStanding.OBSERVED_UNRESOLVED


# -- positive controls --------------------------------------------------------


def test_a_passing_control_is_implementation_evidence() -> None:
    control = PositiveControl(
        name="tiny_overfit",
        metric="train_acc",
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=0.99,
        rationale="a faithful learner must overfit ten points",
    )
    (check,) = evaluate_controls((control,), {"train_acc": 1.0})
    assert check.dimension is _D.IMPLEMENTATION
    assert check.state is _S.PASS


def test_a_failing_control_fails_the_implementation_dimension() -> None:
    control = PositiveControl(
        name="lr_zero_frozen",
        metric="param_delta",
        comparator=Comparator.APPROXIMATELY,
        threshold=0.0,
        tolerance=1e-9,
    )
    (check,) = evaluate_controls((control,), {"param_delta": 0.3})
    assert check.state is _S.FAIL
    assert "param_delta" in check.detail


def test_a_control_without_its_metric_is_uncertain() -> None:
    control = PositiveControl(
        name="c",
        metric="absent",
        comparator=Comparator.GREATER_THAN,
        threshold=0.0,
    )
    (check,) = evaluate_controls((control,), {"other": 1.0})
    assert check.state is _S.UNCERTAIN


# -- analysis coverage --------------------------------------------------------


def test_full_coverage_passes() -> None:
    check = verify_analysis_coverage(
        cited_evidence_ids=("e1", "e2"),
        conclusive_evidence_ids=("e1", "e2"),
    )
    assert check.state is _S.PASS


def test_cherry_picking_fails_the_analysis_dimension() -> None:
    check = verify_analysis_coverage(
        cited_evidence_ids=("e_good",),
        conclusive_evidence_ids=("e_good", "e_bad"),
    )
    assert check.state is _S.FAIL
    assert "e_bad" in check.detail


def test_nothing_conclusive_is_not_applicable() -> None:
    check = verify_analysis_coverage(
        cited_evidence_ids=(), conclusive_evidence_ids=()
    )
    assert check.state is _S.NOT_APPLICABLE


# -- role-backed review adapters ----------------------------------------------


class _VerdictRole(ResearchRole):
    """A rule-based reviewer seat returning a fixed verdict."""

    def __init__(self, verdict: AssessmentVerdict | None):
        self._verdict = verdict
        self.invocations: list[RoleInvocation] = []

    @property
    def name(self) -> RoleName:
        return RoleName.RESULT_ANALYST

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset({ResearchActionType.ANALYZE, ResearchActionType.FALSIFY})

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        self.invocations.append(invocation)
        if self._verdict is None:
            return ()
        (spec,) = invocation.context.experiments
        return (
            AssessmentProposal(
                assessment=EpistemicAssessment(
                    subject_id=spec.id,
                    verdict=self._verdict,
                    method="stub:reviewer:v1",
                    rationale="fixed verdict for the test",
                ),
                proposer="stub:reviewer",
            ),
        )


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        prediction_id="pred_x",
        objective="measure",
        procedure="run",
        metrics=("m",),
    )


def _result() -> ExperimentResult:
    return ExperimentResult(
        spec_id=_spec().id,
        job_id="job_x",
        status=ExperimentStatus.COMPLETED,
        command=("cmd",),
        environment=Environment(python_version="3", platform="test"),
        metrics={"m": 1.0},
        exit_code=0,
        seed=0,
    )


def test_role_backed_methodology_review_maps_verdicts_to_states() -> None:
    for verdict, expected in (
        (AssessmentVerdict.SUPPORTED, _S.PASS),
        (AssessmentVerdict.REFUTED, _S.FAIL),
        (AssessmentVerdict.UNDETERMINED, _S.UNCERTAIN),
    ):
        reviewer = RoleBackedMethodologyReviewer(role=_VerdictRole(verdict))
        check = reviewer.review(_spec(), None, objective="obj")
        assert check.dimension is _D.METHODOLOGY
        assert check.state is expected


def test_role_backed_verifier_uses_falsify_and_reads_the_verdict() -> None:
    role = _VerdictRole(AssessmentVerdict.REFUTED)
    verifier = RoleBackedImplementationVerifier(role=role)
    check = verifier.verify(_spec(), _result(), None, ())
    assert check.dimension is _D.IMPLEMENTATION
    assert check.state is _S.FAIL
    (invocation,) = role.invocations
    assert invocation.assignment.action_type is ResearchActionType.FALSIFY


def test_a_reviewer_with_no_answer_is_uncertain_not_a_verdict() -> None:
    reviewer = RoleBackedMethodologyReviewer(role=_VerdictRole(None))
    assert reviewer.review(_spec(), None, objective="obj").state is _S.UNCERTAIN


def test_a_raising_reviewer_is_uncertain() -> None:
    class Exploding(_VerdictRole):
        def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
            raise RuntimeError("reviewer crashed")

    reviewer = RoleBackedMethodologyReviewer(role=Exploding(None))
    check = reviewer.review(_spec(), None, objective="obj")
    assert check.state is _S.UNCERTAIN
    assert "reviewer crashed" in check.detail


def test_review_verdicts_are_never_committed_state() -> None:
    """The adapter reads the assessment and discards it — a review verdict
    is a check state, not an epistemic claim entering the record."""
    role = _VerdictRole(AssessmentVerdict.SUPPORTED)
    RoleBackedMethodologyReviewer(role=role).review(_spec(), None, objective="o")
    # Nothing here has (or could have) touched any ResearchState: the
    # adapter returns a VerificationCheck and holds no state reference.
    (invocation,) = role.invocations
    assert dataclasses.is_dataclass(invocation.context)
