"""The deterministic science seats, and the fixed-region enforcement.

The model holds exactly one seat in this lab — the engineer's. The
scientist that designs the experiment and the analyst that records the
verdict are trusted code, for the same reason the canary's are: the
funded state already carries the pre-registered prediction, and a
designer's whole legitimate job is to copy its metric verbatim into a
spec the catalog can serve. A model inventing anything there would be
answering a different question than the one the state committed to.
(The planner takes this seat in Task 7B, once evidence exists for its
gate to cite.)

:class:`FixedRegionCheck` is the other half of the narrow-slot design:
a completion that edits one byte of a fenced fixed region — the
seeding, the data loading, the probe, the metrics writing — is refused
before execution, after the rejected payload is preserved. Trusted
measurement stays trusted because nothing else can touch it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.claim import Claim, EvidenceLink, EvidenceRelation
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.prediction import Consistency, PredictionTest
from autonomous_research_lab.core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    ExperimentProposal,
    Proposal,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.roles.planner import TemplateCatalog
from autonomous_research_lab.runtime.implementation_store import (
    ImplementationStore,
)
from autonomous_research_lab.runtime.preflight import JobLike
from autonomous_research_lab.runtime.verification import (
    CheckState,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)

from .catalog import entry_for_metric, fixed_regions

SEEDS = (11, 23, 47)


class VisionScientist(ResearchRole):
    """Designs the one experiment each admitted prediction asks for, and
    reads results back as claims. It invents no metric: the admitted
    contrast is the metric, copied verbatim, and the catalog's entry for
    it supplies everything else the spec declares."""

    def __init__(self, catalog: TemplateCatalog) -> None:
        self._catalog = catalog

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_DIRECTOR

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset(
            {
                ResearchActionType.DESIGN_EXPERIMENT,
                ResearchActionType.SYNTHESIZE_FINDING,
            }
        )

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        del state, action
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        if (
            invocation.assignment.action_type
            is ResearchActionType.DESIGN_EXPERIMENT
        ):
            return self._design(invocation)
        return self._synthesize(invocation)

    def _design(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        targets = set(invocation.assignment.targets)
        prediction = next(
            (
                found
                for found in invocation.context.predictions
                if found.id in targets
            ),
            invocation.context.predictions[0],
        )
        entry = entry_for_metric(self._catalog, prediction.metric)
        spec = ExperimentSpec(
            prediction_id=prediction.id,
            objective=(
                "Measure the pre-registered linear probe contrast the "
                "admitted prediction names."
            ),
            procedure=(
                "Run the catalog template for this contrast: train the "
                "designated arm on the seeded subset, freeze both arms, "
                "fit the linear probe, and report the admitted difference "
                "on held-out images."
            ),
            metrics=entry.metrics,
            baselines=("the untrained comparison arm",),
            controls=(
                "tiny-subset overfit control: the probe must fit a "
                "memorizable subset to at least 0.95 top-1",
            ),
            seeds=SEEDS,
            estimated_cost=entry.estimated_cost,
        )
        return (ExperimentProposal(spec=spec, proposer="vision:scientist"),)

    def _synthesize(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        evidence = invocation.context.evidence[0]
        hypothesis = invocation.context.hypotheses[0]
        spec = invocation.context.experiments[0]
        test = next(
            found
            for found in invocation.context.prediction_tests
            if found.result_id == evidence.result_id
        )
        claim = Claim(
            statement=hypothesis.statement,
            scope=spec.procedure,
            hypothesis_id=hypothesis.id,
        )
        link = EvidenceLink(
            claim_id=claim.id,
            evidence_id=evidence.id,
            relation={
                Consistency.CONSISTENT: EvidenceRelation.SUPPORTS,
                Consistency.INCONSISTENT: EvidenceRelation.CONTRADICTS,
                Consistency.INCONCLUSIVE: EvidenceRelation.INCONCLUSIVE,
            }[test.consistency],
            rationale=(
                f"the pre-registered prediction tested {test.consistency}: "
                f"{test.detail}"
            ),
        )
        return (
            ClaimProposal(
                claim=claim, links=(link,), proposer="vision:scientist"
            ),
        )


class VisionAnalyst(ResearchRole):
    """Puts a judgment about the hypothesis on the record when asked."""

    @property
    def name(self) -> RoleName:
        return RoleName.RESULT_ANALYST

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset(
            {ResearchActionType.ANALYZE, ResearchActionType.ASSESS_CLAIM}
        )

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        del state, action
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        verdict = _verdict(invocation.context.prediction_tests)
        if (
            invocation.assignment.action_type
            is ResearchActionType.ASSESS_CLAIM
        ):
            subject = invocation.context.claims[0].id
        else:
            subject = invocation.context.hypotheses[0].id
        assessment = EpistemicAssessment(
            subject_id=subject,
            verdict=verdict,
            method="vision-analyst:v1",
            evidence_ids=tuple(
                found.id for found in invocation.context.evidence
            ),
            scope=(
                invocation.context.experiments[0].procedure
                if invocation.context.experiments
                else ""
            ),
            rationale="; ".join(invocation.context.notes)
            or "read as recorded",
        )
        return (
            AssessmentProposal(assessment=assessment, proposer="vision:analyst"),
        )


def _verdict(tests: Sequence[PredictionTest]) -> AssessmentVerdict:
    consistent = any(
        test.consistency is Consistency.CONSISTENT for test in tests
    )
    inconsistent = any(
        test.consistency is Consistency.INCONSISTENT for test in tests
    )
    if consistent and inconsistent:
        return AssessmentVerdict.CONTESTED
    if inconsistent:
        return AssessmentVerdict.REFUTED
    if consistent:
        return AssessmentVerdict.SUPPORTED
    return AssessmentVerdict.UNDETERMINED


@dataclass(frozen=True, slots=True)
class VisionControls:
    """The catalog's control for any spec the catalog served."""

    catalog: TemplateCatalog

    def __call__(self, spec: ExperimentSpec) -> tuple[PositiveControl, ...]:
        for entry in self.catalog.entries:
            if entry.control is not None and set(entry.metrics) <= set(
                spec.metrics
            ):
                return (entry.control,)
        return ()


@dataclass(frozen=True, slots=True)
class FixedRegionCheck:
    """Preflight: the completion preserved every fixed region, byte for
    byte, of the template it was completed from.

    Runs after the implementation record is durable, so a tampering
    completion is preserved, refused before execution, and never
    silently retried. ``NOT_APPLICABLE`` for jobs with no implementation
    id — foreign jobs are not this check's business.
    """

    store: ImplementationStore
    catalog: TemplateCatalog

    def check(
        self, job: JobLike, spec: ExperimentSpec | None
    ) -> VerificationCheck:
        del spec
        implementation_id = job.config.get("implementation_id")
        if not isinstance(implementation_id, str) or not implementation_id:
            return _check(
                CheckState.NOT_APPLICABLE, "no implementation declared"
            )
        record = self.store.get(implementation_id)
        if record is None:
            return _check(
                CheckState.FAIL,
                f"implementation {implementation_id} is not on record",
            )
        entry = self.catalog.get(record.template_id)
        if entry is None:
            return _check(
                CheckState.FAIL,
                f"completion claims template {record.template_id}, which "
                f"this catalog did not issue",
            )
        source_path = (
            self.store.source_dir(record.source_id) / record.entrypoint
        )
        if not source_path.is_file():
            return _check(
                CheckState.FAIL,
                f"the persisted source {record.source_id} has no "
                f"{record.entrypoint}",
            )
        completed = fixed_regions(source_path.read_text())
        expected = fixed_regions(entry.template.source)
        if completed != expected:
            return _check(
                CheckState.FAIL,
                "the completion altered a fixed region; trusted "
                "measurement code is not the model's to edit",
            )
        return _check(
            CheckState.PASS,
            f"all {len(expected)} fixed region(s) preserved",
        )


def _check(state: CheckState, detail: str) -> VerificationCheck:
    return VerificationCheck(
        dimension=ValidityDimension.EXECUTION,
        name="preflight:fixed_regions",
        state=state,
        detail=detail,
    )
