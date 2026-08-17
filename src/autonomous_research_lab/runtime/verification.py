"""Experiment validity, kept orthogonal to scientific outcome.

The taxonomy this module makes explicit::

    ENGINEERING FAILURE          the process misbehaved       -> repair execution
    IMPLEMENTATION FAILURE       the code ran but is wrong    -> verify / debug
    METHODOLOGICAL FAILURE       the experiment is wrong      -> redesign
    ANALYTICAL FAILURE           the inference is wrong       -> redo analysis
    VERIFIED                     all four dimensions resolved -> evidence

Validity and outcome are orthogonal axes: ``VERIFIED`` plus an inconsistent
prediction test is a *valid scientific negative*; ``IMPLEMENTATION_UNCERTAIN``
plus the same test is a debugging question, not negative evidence. Nothing
here looks at whether a result is pleasing — only at whether it can be
trusted.

Checks use explicit states (:class:`CheckState`) rather than confidence
floats: ``PASS`` / ``FAIL`` are determinations, ``UNCERTAIN`` is an honest
absence of one, ``NOT_APPLICABLE`` keeps universal check lists from
manufacturing verdicts about experiments they do not apply to. Deterministic
checks always outrank semantic review: a reviewer's ``PASS`` can never wash
out a failed control, because dimension aggregation treats any ``FAIL`` as
final.

No general system can prove the total absence of silent scientific bugs.
What this layer provides is layered defence — deterministic validation,
positive controls, selective independent judgment — and an explicit record
of which layers actually ran, so an unverified result is never mistaken for
a verified one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..core.experiment import ExperimentResult, ExperimentSpec
from ..core.prediction import Comparator, Prediction


class CheckState(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


class ValidityDimension(StrEnum):
    """The four axes on which an experiment can be invalid."""

    EXECUTION = "execution"
    IMPLEMENTATION = "implementation"
    METHODOLOGY = "methodology"
    ANALYSIS = "analysis"


class ExperimentValidityStatus(StrEnum):
    ENGINEERING_FAILED = "engineering_failed"
    IMPLEMENTATION_UNCERTAIN = "implementation_uncertain"
    METHODOLOGICALLY_INVALID = "methodologically_invalid"
    ANALYTICALLY_INVALID = "analytically_invalid"
    UNVERIFIED = "unverified"
    """Outcome observed, validity unresolved: no dimension failed, but at
    least one was never determined. The honest intermediate state — an
    observation in this status is preserved, never promoted."""

    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    """One named check on one validity dimension."""

    dimension: ValidityDimension
    name: str
    state: CheckState
    detail: str = ""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """The immutable record of every verification check one result faced."""

    checks: tuple[VerificationCheck, ...]

    @property
    def failures(self) -> tuple[VerificationCheck, ...]:
        return tuple(c for c in self.checks if c.state is CheckState.FAIL)

    @property
    def passed(self) -> bool:
        return not self.failures

    def dimension_state(self, dimension: ValidityDimension) -> CheckState:
        """The aggregate state of one dimension: any FAIL is final, any
        UNCERTAIN blocks PASS, and a dimension with no applicable checks is
        NOT_APPLICABLE — never silently passing."""
        states = {
            c.state for c in self.checks if c.dimension is dimension
        } - {CheckState.NOT_APPLICABLE}
        if CheckState.FAIL in states:
            return CheckState.FAIL
        if CheckState.UNCERTAIN in states:
            return CheckState.UNCERTAIN
        if CheckState.PASS in states:
            return CheckState.PASS
        return CheckState.NOT_APPLICABLE


def derive_validity(report: VerificationReport) -> ExperimentValidityStatus:
    """Collapse a report into one validity status, worst dimension first.

    Precedence encodes the response ladder: a failed execution is diagnosed
    before the implementation is questioned; a definitively invalid design
    (redesign) outranks an uncertain implementation (verify); analysis is
    judged only on an otherwise sound experiment. ``VERIFIED`` requires a
    positive determination on every dimension — an axis nobody checked
    yields ``UNVERIFIED``, not a pass.
    """
    execution = report.dimension_state(ValidityDimension.EXECUTION)
    implementation = report.dimension_state(ValidityDimension.IMPLEMENTATION)
    methodology = report.dimension_state(ValidityDimension.METHODOLOGY)
    analysis = report.dimension_state(ValidityDimension.ANALYSIS)

    if execution is not CheckState.PASS:
        return ExperimentValidityStatus.ENGINEERING_FAILED
    if methodology is CheckState.FAIL:
        return ExperimentValidityStatus.METHODOLOGICALLY_INVALID
    if implementation is CheckState.FAIL or implementation is CheckState.UNCERTAIN:
        return ExperimentValidityStatus.IMPLEMENTATION_UNCERTAIN
    if analysis is CheckState.FAIL:
        return ExperimentValidityStatus.ANALYTICALLY_INVALID
    if CheckState.PASS == implementation == methodology == analysis:
        return ExperimentValidityStatus.VERIFIED
    return ExperimentValidityStatus.UNVERIFIED


class OutcomeStanding(StrEnum):
    """Whether an observed outcome may serve as scientific evidence.

    Applies identically to positive and negative outcomes — the gate is
    about validity, never about which way the result went.
    """

    VERIFIED_EVIDENCE = "verified_evidence"
    OBSERVED_UNRESOLVED = "observed_unresolved"


def outcome_standing(validity: ExperimentValidityStatus) -> OutcomeStanding:
    """The negative-result gate (and, symmetrically, the positive one).

    A conclusive outcome becomes strong scientific evidence only when every
    validity dimension is resolved. Anything less preserves the observation
    without promoting it — and never, under any status, routes it to
    debugging merely for being disappointing.
    """
    if validity is ExperimentValidityStatus.VERIFIED:
        return OutcomeStanding.VERIFIED_EVIDENCE
    return OutcomeStanding.OBSERVED_UNRESOLVED


# -- positive controls --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositiveControl:
    """An experiment-specific invariant that a faithful implementation must
    satisfy — a tiny dataset that must be overfittable, a zero learning rate
    that must change nothing, a shuffled-label run that must degrade.

    Deterministic by construction: the control is a pre-stated comparison on
    a metric the run reports. A control that fails makes the implementation
    *uncertain*; it is never read as a scientific negative, because it tests
    the instrument, not the hypothesis.
    """

    name: str
    metric: str
    comparator: Comparator
    threshold: float
    tolerance: float = 0.0
    rationale: str = ""

    def evaluate(self, metrics: Mapping[str, float]) -> VerificationCheck:
        observed = metrics.get(self.metric)
        if observed is None:
            return VerificationCheck(
                dimension=ValidityDimension.IMPLEMENTATION,
                name=f"positive_control:{self.name}",
                state=CheckState.UNCERTAIN,
                detail=f"run reported no value for {self.metric!r}",
            )
        held = _compare(observed, self.comparator, self.threshold, self.tolerance)
        return VerificationCheck(
            dimension=ValidityDimension.IMPLEMENTATION,
            name=f"positive_control:{self.name}",
            state=CheckState.PASS if held else CheckState.FAIL,
            detail=(
                f"observed {self.metric}={observed:g} vs {self.comparator} "
                f"{self.threshold:g}"
                + (f" (tolerance {self.tolerance:g})" if self.tolerance else "")
                + (f"; {self.rationale}" if self.rationale else "")
            ),
        )


def _compare(
    value: float, comparator: Comparator, threshold: float, tolerance: float
) -> bool:
    match comparator:
        case Comparator.LESS_THAN:
            return value < threshold
        case Comparator.LESS_OR_EQUAL:
            return value <= threshold
        case Comparator.GREATER_THAN:
            return value > threshold
        case Comparator.GREATER_OR_EQUAL:
            return value >= threshold
        case Comparator.APPROXIMATELY:
            return abs(value - threshold) <= tolerance
    raise AssertionError(f"unhandled comparator {comparator}")


def evaluate_controls(
    controls: Iterable[PositiveControl], metrics: Mapping[str, float]
) -> tuple[VerificationCheck, ...]:
    """Evaluate every control against one run's reported metrics."""
    return tuple(control.evaluate(metrics) for control in controls)


class ControlSource(Protocol):
    """Where experiment-specific controls come from. ML-specific control
    libraries stay outside ``core``; the runtime only needs a lookup."""

    def __call__(self, spec: ExperimentSpec) -> tuple[PositiveControl, ...]: ...


# -- semantic review hooks ----------------------------------------------------
#
# The two questions code cannot answer. Both are Protocols so the runtime can
# hold a rule-based double today and a role-backed (model) reviewer tomorrow
# — and so either can be removed for an ablation by leaving the field None.
# Their answers feed one dimension each of the report; they can never
# overturn a deterministic FAIL elsewhere.


class MethodologyReviewer(Protocol):
    """Even if perfectly implemented, would this experiment answer the
    intended scientific question? (Construct/metric validity, baselines,
    confounds, regime relevance, statistical adequacy, scope.)"""

    def review(
        self, spec: ExperimentSpec, prediction: Prediction | None, *, objective: str
    ) -> VerificationCheck: ...


class ImplementationVerifier(Protocol):
    """Does the implementation faithfully realize the spec — and is there a
    silent bug (wrong loss, leaked data, bad split, dead gradient, wrong
    baseline) that would produce plausible but misleading metrics?"""

    def verify(
        self,
        spec: ExperimentSpec,
        result: ExperimentResult,
        prediction: Prediction | None,
        checks: tuple[VerificationCheck, ...],
    ) -> VerificationCheck: ...


# -- analysis validity --------------------------------------------------------


def verify_analysis_coverage(
    *,
    cited_evidence_ids: Iterable[str],
    conclusive_evidence_ids: Iterable[str],
) -> VerificationCheck:
    """Deterministic guard against post-hoc run selection.

    An analysis (an assessment, an aggregation) that cites only a subset of
    the conclusive observations available to it is cherry-picking until
    shown otherwise. The response to a failure is *redo the analysis over
    the full family* — never rerun experiments until the result changes.
    """
    cited = set(cited_evidence_ids)
    conclusive = set(conclusive_evidence_ids)
    ignored = conclusive - cited
    if not conclusive:
        return VerificationCheck(
            dimension=ValidityDimension.ANALYSIS,
            name="analysis_coverage",
            state=CheckState.NOT_APPLICABLE,
            detail="no conclusive observations to cover",
        )
    if ignored:
        return VerificationCheck(
            dimension=ValidityDimension.ANALYSIS,
            name="analysis_coverage",
            state=CheckState.FAIL,
            detail=(
                f"analysis cites {len(cited & conclusive)} of "
                f"{len(conclusive)} conclusive observation(s); ignored: "
                f"{', '.join(sorted(ignored))}"
            ),
        )
    return VerificationCheck(
        dimension=ValidityDimension.ANALYSIS,
        name="analysis_coverage",
        state=CheckState.PASS,
        detail=f"all {len(conclusive)} conclusive observation(s) considered",
    )
