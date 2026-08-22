"""The deterministic science seats, and the fixed-region enforcement.

The model holds exactly one seat in this lab — the engineer's. The
scientist that designs the experiment and the analyst that records the
verdict are trusted code, for the same reason the canary's are: the
funded state already carries the pre-registered prediction, and a
designer's whole legitimate job is to copy its metric verbatim into a
spec the catalog can serve. A model inventing anything there would be
answering a different question than the one the state committed to.
(Since Task 7B the planner shares this seat: once verified evidence
exists, the composite director in ``direction.py`` hands consultations
to the model-backed planner, and this scientist keeps the design and
synthesis work.)

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
from autonomous_research_lab.core.prediction import (
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    ExperimentProposal,
    Proposal,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleContext,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.roles.engineer import (
    ENTRYPOINT,
    ImplementationTemplate,
)
from autonomous_research_lab.roles.planner import TemplateCatalog
from autonomous_research_lab.runtime.implementation_store import (
    ImplementationStore,
    SourceFile,
)
from autonomous_research_lab.runtime.preflight import JobLike
from autonomous_research_lab.runtime.statistics import assess_family
from autonomous_research_lab.runtime.verification import (
    CheckState,
    PositiveControl,
    ValidityDimension,
    VerificationCheck,
)

from .catalog import entry_for_metric, fixed_regions

SEEDS = (11, 23, 47, 71, 83)


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


class VisionStatistician(ResearchRole):
    """The result-analyst seat, with the arithmetic done properly.

    For ASSESS_CLAIM: exact inference over the claim's replication
    family — the raw per-seed observations the enriched context now
    carries — with the comparison count pinned across the hypothesis's
    tested predictions and every figure stated in the assessment's own
    rationale. The verdict is trusted code's; no model authors a number.

    Citation discipline, exactly as the gates demand: the hypothesis-wide
    admissible conclusive family, in full, for every verdict — coverage
    applies to UNDETERMINED too, and promotion refuses any inadmissible
    citation. The statistic itself is computed over admissible tests
    only, because an inadmissible observation counted-but-uncited would
    slip both gates.

    ANALYZE (the critic path) keeps the prior analyst behavior,
    delegated unchanged.
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        self._alpha = alpha
        self._fallback = VisionAnalyst()

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
        if (
            invocation.assignment.action_type
            is not ResearchActionType.ASSESS_CLAIM
        ):
            return self._fallback.perform(invocation)
        context = invocation.context
        claim = context.claims[0]
        admissible_ids = set(context.admissible_evidence_ids)
        admissible_result_ids = {
            found.result_id
            for found in context.evidence
            if found.id in admissible_ids
        }
        seed_of = {
            result.id: result.seed for result in context.results
        }

        def family(prediction: Prediction) -> tuple[PredictionTest, ...]:
            tests = [
                test
                for test in context.prediction_tests
                if test.prediction_id == prediction.id
                and test.consistency is not Consistency.INCONCLUSIVE
                and test.result_id in admissible_result_ids
            ]
            tests.sort(
                key=lambda test: (
                    seed_of.get(test.result_id) is None,
                    seed_of.get(test.result_id) or 0,
                )
            )
            return tuple(tests)

        comparisons = (
            sum(1 for found in context.predictions if family(found)) or 1
        )
        own = _own_predictions(context, claim)
        assessed = [
            assess_family(
                prediction,
                family(prediction),
                alpha=self._alpha,
                comparisons=comparisons,
            )
            for prediction in own
        ]
        verdict = _combined(tuple(v for v, _ in assessed))
        rationale = " | ".join(stats.render() for _, stats in assessed)
        spec = next(
            (
                found
                for found in context.experiments
                if own and found.prediction_id == own[0].id
            ),
            None,
        )
        assessment = EpistemicAssessment(
            subject_id=claim.id,
            verdict=verdict,
            method="statistician:exact-sign-v1",
            # The hypothesis-wide admissible conclusive family, in
            # context order — the one citation set both gates accept.
            evidence_ids=tuple(
                found.id
                for found in context.evidence
                if found.id in admissible_ids
            ),
            scope=spec.procedure if spec is not None else "",
            rationale=rationale or "no admissible conclusive observations",
        )
        return (
            AssessmentProposal(
                assessment=assessment, proposer="vision:statistician"
            ),
        )


def _own_predictions(
    context: RoleContext, claim: Claim
) -> tuple[Prediction, ...]:
    """The predictions the claim's own evidence tested: links → evidence
    → spec → prediction. Falls back to every shown prediction when the
    join yields nothing, so a sparsely-linked claim is still judged on
    the record rather than on air."""
    linked_evidence = {
        link.evidence_id
        for link in context.evidence_links
        if link.claim_id == claim.id
    }
    linked_specs = {
        found.spec_id
        for found in context.evidence
        if found.id in linked_evidence
    }
    prediction_ids = {
        spec.prediction_id
        for spec in context.experiments
        if spec.id in linked_specs
    }
    own = tuple(
        found for found in context.predictions if found.id in prediction_ids
    )
    return own or tuple(context.predictions)


_SEVERITY = (
    AssessmentVerdict.REFUTED,
    AssessmentVerdict.CONTESTED,
    AssessmentVerdict.UNDETERMINED,
    AssessmentVerdict.PLAUSIBLE,
    AssessmentVerdict.SUPPORTED,
)


def _combined(verdicts: tuple[AssessmentVerdict, ...]) -> AssessmentVerdict:
    """Worst-of across a claim's own predictions: a claim is only as
    settled as its least settled family."""
    if not verdicts:
        return AssessmentVerdict.UNDETERMINED
    for severe in _SEVERITY:
        if severe in verdicts:
            return severe
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
class FixedRegionReview:
    """The fixed-region rule, where it can still be fixed.

    Runs inside the engineer's bounded generation-repair loop, handed
    the resolved template and the parsed completion before anything
    persists: a violation earns the model one corrective call carrying
    exactly what it must not touch. :class:`FixedRegionCheck` below
    keeps the same judgment as a preflight backstop — defense in depth,
    not redundancy: the review gives feedback, the check refuses
    execution.
    """

    def review(
        self,
        template: ImplementationTemplate,
        files: tuple[SourceFile, ...],
    ) -> str:
        completed = next(
            (found for found in files if found.path == ENTRYPOINT), None
        )
        if completed is None:
            return ""  # the parse gate already refuses a missing entrypoint
        if fixed_regions(completed.content) != fixed_regions(template.source):
            return (
                "the completion altered a fenced fixed region; everything "
                "between ARL-FIXED-BEGIN and ARL-FIXED-END — seeding, data "
                "loading, splits, the probe, the control, and the metrics "
                "writing — is trusted measurement code and must be "
                "reproduced byte-for-byte; complete only the build_encoder "
                "slot between ARL-SLOT-BEGIN and ARL-SLOT-END"
            )
        return ""


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
