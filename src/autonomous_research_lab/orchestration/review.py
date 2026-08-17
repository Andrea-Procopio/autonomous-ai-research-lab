"""Role-backed semantic review: reusing the role abstraction for verification.

The runtime's semantic hooks
(:class:`~autonomous_research_lab.runtime.verification.MethodologyReviewer`,
:class:`~autonomous_research_lab.runtime.verification.ImplementationVerifier`)
are plain protocols. These adapters implement them on top of an existing
:class:`~autonomous_research_lab.roles.base.ResearchRole` — no new permanent
agent, no new action type, no change to ``core``:

* methodology review is an ``ANALYZE`` invocation of the reviewer seat over
  the spec, its prediction, and the objective;
* implementation verification is a ``FALSIFY`` invocation — the verifier is
  asked to *break* the implementation, handed the spec, the result, and the
  deterministic check states already gathered.

The role answers, per its normal output contract, with an
``AssessmentProposal``; the adapter reads its verdict as a check state and
**discards the proposal** — a review verdict feeds a
:class:`~autonomous_research_lab.runtime.verification.VerificationReport`,
it is not an epistemic claim about a hypothesis and never commits to state.
A role that raises, or answers with nothing usable, yields ``UNCERTAIN``:
a reviewer failing to review is an absent determination, not a verdict.

Model judgment cannot override deterministic ground truth by construction
here: these adapters only ever write the implementation / methodology
dimensions, and dimension aggregation treats any deterministic ``FAIL`` as
final regardless of what a reviewer says.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.actions import ResearchAction, ResearchActionType
from ..core.assessment import AssessmentVerdict
from ..core.experiment import ExperimentResult, ExperimentSpec
from ..core.prediction import Prediction
from ..core.proposals import AssessmentProposal
from ..roles.base import ResearchRole, RoleContext, RoleInvocation
from ..runtime.verification import (
    CheckState,
    ValidityDimension,
    VerificationCheck,
)
from .routing import expected_proposals

_VERDICT_TO_STATE = {
    AssessmentVerdict.SUPPORTED: CheckState.PASS,
    AssessmentVerdict.PLAUSIBLE: CheckState.PASS,
    AssessmentVerdict.REFUTED: CheckState.FAIL,
    AssessmentVerdict.CONTESTED: CheckState.UNCERTAIN,
    AssessmentVerdict.UNDETERMINED: CheckState.UNCERTAIN,
}


def _reviewed(
    role: ResearchRole,
    invocation: RoleInvocation,
    *,
    dimension: ValidityDimension,
    name: str,
) -> VerificationCheck:
    try:
        proposals = role.perform(invocation)
    except Exception as exc:  # a reviewer failing is an absent determination
        return VerificationCheck(
            dimension=dimension,
            name=name,
            state=CheckState.UNCERTAIN,
            detail=f"reviewer {role.name} failed to review: {exc}",
        )
    assessment = next(
        (p.assessment for p in proposals if isinstance(p, AssessmentProposal)),
        None,
    )
    if assessment is None:
        return VerificationCheck(
            dimension=dimension,
            name=name,
            state=CheckState.UNCERTAIN,
            detail=f"reviewer {role.name} returned no assessment",
        )
    return VerificationCheck(
        dimension=dimension,
        name=name,
        state=_VERDICT_TO_STATE[assessment.verdict],
        detail=f"{assessment.method}: {assessment.rationale or assessment.verdict}",
    )


@dataclass(frozen=True, slots=True)
class RoleBackedMethodologyReviewer:
    """Adapts a role into the runtime's ``MethodologyReviewer`` protocol.

    The question posed: even perfectly implemented, would this experiment
    answer the stated scientific question? A ``REFUTED`` assessment means
    *redesign the experiment* — not debug it, not record a negative.
    """

    role: ResearchRole

    def review(
        self, spec: ExperimentSpec, prediction: Prediction | None, *, objective: str
    ) -> VerificationCheck:
        action = ResearchAction(
            action_type=ResearchActionType.ANALYZE,
            rationale=(
                "methodology review: would this design, perfectly "
                "implemented, validly answer the scientific question? "
                "Judge construct and metric validity, baselines, comparison "
                "fairness, dataset/regime relevance, confounds, and "
                "statistical adequacy."
            ),
            targets=(spec.id,),
        )
        invocation = RoleInvocation(
            role=self.role.name,
            assignment=action,
            context=RoleContext(
                objective=objective,
                predictions=(prediction,) if prediction is not None else (),
                experiments=(spec,),
                notes=(action.rationale,),
            ),
            allowed_actions=frozenset({ResearchActionType.ANALYZE}),
            expected_output=expected_proposals(ResearchActionType.ANALYZE),
        )
        return _reviewed(
            self.role,
            invocation,
            dimension=ValidityDimension.METHODOLOGY,
            name="methodological_validity",
        )


@dataclass(frozen=True, slots=True)
class RoleBackedImplementationVerifier:
    """Adapts a role into the runtime's ``ImplementationVerifier`` protocol.

    A ``FALSIFY`` assignment: actively hunt for silent bugs — wrong
    objective, ignored variables, wrong metric computation, leaked data, bad
    splits, train/eval mode errors, broken updates, spec/implementation
    mismatch — that could make plausible metrics misleading.
    """

    role: ResearchRole

    def verify(
        self,
        spec: ExperimentSpec,
        result: ExperimentResult,
        prediction: Prediction | None,
        checks: tuple[VerificationCheck, ...],
    ) -> VerificationCheck:
        action = ResearchAction(
            action_type=ResearchActionType.FALSIFY,
            rationale=(
                "implementation verification: does the run faithfully "
                "realize the spec? Actively search for silent bugs that "
                "would produce plausible but misleading metrics."
            ),
            targets=(result.id,),
        )
        invocation = RoleInvocation(
            role=self.role.name,
            assignment=action,
            context=RoleContext(
                objective=spec.objective,
                predictions=(prediction,) if prediction is not None else (),
                experiments=(spec,),
                results=(result,),
                notes=(
                    action.rationale,
                    *(
                        f"deterministic check {c.name}: {c.state}"
                        + (f" ({c.detail})" if c.detail else "")
                        for c in checks
                    ),
                ),
            ),
            allowed_actions=frozenset({ResearchActionType.FALSIFY}),
            expected_output=expected_proposals(ResearchActionType.FALSIFY),
        )
        return _reviewed(
            self.role,
            invocation,
            dimension=ValidityDimension.IMPLEMENTATION,
            name="implementation_faithfulness",
        )
