"""The admission records: everything an admission run may durably claim.

An admission is a governed translation of one validated selection into
the initial scientific state — never a promotion. ``DISTINGUISHED``
meant differentiated within one bounded prior-art corpus; ``SELECTED``
meant preferred within one constrained portfolio; ``ADMITTED`` means
converted into the governed research state. None of the three means
true, novel, or empirically supported, and the admitted state holds
propositions only: no result, evidence, assessment, or claim is
expressible anywhere in this vocabulary.

The authority split is structural. Trusted code owns every identifier,
the lineage, the selected candidate, every deterministic copy (the
question, the hypothesis, the objective, the measurement/control/
comparison surface, every requirement quote), the construction of every
core record, and the spend. The model owns only the operationalization
wording: for each of the candidate's recorded predictions, a condition,
the two comparison arms, a contrary restatement, and field-path
traceability links. ``condition`` is the one prose seat of model
judgment — named as such in :data:`CLAIM_KINDS`, held to text
discipline and to its own grounded support, and never mistaken for a
trusted-code conclusion.

The neutral encoding: trusted code turns each operationalization into a
core ``Prediction`` with ``metric = "difference in {base_metric}:
{expected_higher_arm} minus {expected_lower_arm}"``,
``comparator = GREATER_THAN`` and ``threshold = 0.0`` — structural
constants, so the model never authors a number and the encoded
commitment is exactly the sign of the difference, nothing more.
:data:`MECHANICAL_READING` stamps that weakening on the record so a
later assessor cannot read a marginal delta as confirming stronger
prose. The templated metric string is an exact-match contract: a future
experiment spec must declare it verbatim among its metrics, and the
executor must report it verbatim as a metrics key.

Identity follows the house rules: run ids are occurrences, the stored
record is content-addressed over all of its fields, provider provenance
and spend included, and the persisted initial state is part of the
write-once artifact set — a record whose state snapshot is gone fails
loudly forever rather than being quietly rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from ..core.ids import content_id
from ..mapping.records import CallProvenance

MAX_ARM_CHARS: Final = 120
"""Arms are embedded in the templated metric string, which future
experiment specs and executors must reproduce byte-for-byte as a metrics
key; the bound keeps that contract usable."""

MAX_ENCODINGS_PER_PREDICTION: Final = 3
"""A recorded prediction may carry several observables (the live 5E
winner's single prediction carries two), so one encoding per prediction
would silently discard content; the cap is what lets the preflight's
token arithmetic bound the reply."""

MECHANICAL_READING: Final = "sign_only"
"""What the encoded predictions actually commit to: the sign of the
difference between the two arms. Comparative prose like "substantially
more" is deliberately weakened to "> 0" rather than inventing an effect
size nobody recorded; choosing real thresholds is the planner's work."""

#: The structural epistemic label of each record category. The
#: ``deterministic_copy`` entries document exactly which candidate or
#: selection field each trusted-code copy came from — in particular that
#: a question's ``importance`` is the candidate's ``cfp_alignment``
#: (its why-it-matters statement), not its ``proposed_contribution``.
CLAIM_KINDS: Final = {
    "operational_prediction.prediction_text": "record_quotation",
    "operational_prediction.condition": "operational_interpretation",
    "operational_prediction.base_metric": "record_restatement",
    "operational_prediction.expected_higher_arm": "record_quotation",
    "operational_prediction.expected_lower_arm": "record_quotation",
    "operational_prediction.contrary_observation": "record_quotation",
    "operational_prediction.support.quote": "record_quotation",
    "requirement.quote": "record_quotation",
    "question.text": "deterministic_copy",
    "question.importance": "deterministic_copy",
    "hypothesis.statement": "deterministic_copy",
    "hypothesis.rationale": "deterministic_copy",
    "state.objective": "deterministic_copy",
    "measurements": "deterministic_copy",
    "controls": "deterministic_copy",
    "comparison_targets": "deterministic_copy",
    "evaluation_protocol": "deterministic_copy",
    "prediction.metric": "operational_interpretation",
}


class SupportSource(StrEnum):
    """Where a grounding quote may come from: the selected candidate's
    own record, or the direction that governs its scope. Nothing else is
    quotable — in particular no sibling candidate, no assessment prose,
    and no retrieved source."""

    CANDIDATE = "candidate"
    DIRECTION = "direction"


class RequirementSource(StrEnum):
    """Where an execution-capability requirement was stated. Inherited
    requirements come from the records the admission verified; operator
    requirements come from the admission directive itself. The two are
    never presented as each other."""

    CANDIDATE_RESOURCES = "candidate_resources"
    SELECTION_DIRECTIVE = "selection_directive"
    SELECTION_DECISION = "selection_decision"
    ADMISSION_DIRECTIVE = "admission_directive"


@dataclass(frozen=True, slots=True)
class GroundedSupport:
    """One traceability link: a verbatim quote at a named field path of
    the candidate or its direction. The gate re-finds the quote in that
    field's text — not anywhere else."""

    source: SupportSource
    field_path: str
    quote: str

    def __post_init__(self) -> None:
        if not self.field_path.strip():
            raise ValueError("support names the field path it quotes")
        if not self.quote.strip():
            raise ValueError("support carries the verbatim quote")


@dataclass(frozen=True, slots=True)
class OperationalPrediction:
    """One machine-checkable encoding of one recorded candidate
    prediction. Everything quoted is re-found by the gate; the two arms
    are embedded verbatim in the derived core prediction's metric."""

    prediction_text: str
    """The candidate prediction being encoded, quoted verbatim — the
    coverage key the gate matches by normalized equality."""

    condition: str
    base_metric: str
    expected_higher_arm: str
    expected_lower_arm: str
    contrary_observation: str
    """A restatement of the recorded falsifier: what observation would
    count against the hypothesis. Re-found in that prediction's own
    falsifier text and nowhere else."""

    support: tuple[GroundedSupport, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("prediction_text", self.prediction_text),
            ("condition", self.condition),
            ("base_metric", self.base_metric),
            ("expected_higher_arm", self.expected_higher_arm),
            ("expected_lower_arm", self.expected_lower_arm),
            ("contrary_observation", self.contrary_observation),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be non-empty")
        for label, arm in (
            ("expected_higher_arm", self.expected_higher_arm),
            ("expected_lower_arm", self.expected_lower_arm),
        ):
            if len(arm) > MAX_ARM_CHARS:
                raise ValueError(
                    f"{label} must be at most {MAX_ARM_CHARS} characters, "
                    f"got {len(arm)}"
                )
        if _folded(self.expected_higher_arm) == _folded(
            self.expected_lower_arm
        ):
            raise ValueError(
                "the two arms must differ; a difference of an arm with "
                "itself commits to nothing"
            )
        if not self.support:
            raise ValueError(
                "an operationalization carries at least one grounded "
                "support link"
            )


@dataclass(frozen=True, slots=True)
class Requirement:
    """One execution-capability requirement, quoted verbatim from the
    record that stated it. Self-contained provenance: the source kind,
    the id of the record quoted, and the field path within it."""

    source: RequirementSource
    record_id: str
    field_path: str
    quote: str

    def __post_init__(self) -> None:
        for label, value in (
            ("record_id", self.record_id),
            ("field_path", self.field_path),
            ("quote", self.quote),
        ):
            if not value.strip():
                raise ValueError(f"a requirement's {label} must be non-empty")


@dataclass(frozen=True, slots=True)
class AdmissionRecord:
    """The completed admission: the full verified lineage, the grounded
    operationalization, every deterministic copy, the created core
    record ids, the initial state id, and the spend — one record per
    selection run, ever, written once after the state snapshot exists.

    The record and the state snapshot it names are one write-once
    artifact set: the public accessor loads the record first and the
    state through it, so an admitted state is never exposed without its
    record, and a record whose snapshot is missing or tampered fails
    loudly forever."""

    run_id: str
    directive_id: str
    selection_run_record_id: str
    selection_run_id: str
    selection_directive_id: str
    prior_art_run_record_id: str
    prior_art_run_id: str
    selected_prior_art_assessment_id: str
    ideation_run_record_id: str
    ideation_run_id: str
    direction_id: str
    snapshot_id: str
    map_run_id: str
    map_assessment_id: str
    """The mapping lineage ids, taken from the ideation run after the
    door proves the prior-art run carries identical copies. Explicit
    names (not ideation's bare ``assessment_id``) because this record
    also names a prior-art assessment. The mapping store itself is
    never loaded — carried and cross-checked, not resolved."""

    selected_candidate_id: str
    operational_predictions: tuple[OperationalPrediction, ...]
    measurements: tuple[str, ...]
    controls: tuple[str, ...]
    comparison_targets: tuple[str, ...]
    evaluation_protocol: str
    inherited_requirements: tuple[Requirement, ...]
    operator_requirements: tuple[Requirement, ...]
    mechanical_reading: str
    question_id: str
    hypothesis_id: str
    prediction_ids: tuple[str, ...]
    state_id: str
    provenance: CallProvenance
    model_calls: int
    input_tokens: int
    output_tokens: int
    id: str = field(default="")

    def __post_init__(self) -> None:
        for label, value in (
            ("run_id", self.run_id),
            ("directive_id", self.directive_id),
            ("selection_run_record_id", self.selection_run_record_id),
            ("selection_run_id", self.selection_run_id),
            ("selection_directive_id", self.selection_directive_id),
            ("prior_art_run_record_id", self.prior_art_run_record_id),
            ("prior_art_run_id", self.prior_art_run_id),
            (
                "selected_prior_art_assessment_id",
                self.selected_prior_art_assessment_id,
            ),
            ("ideation_run_record_id", self.ideation_run_record_id),
            ("ideation_run_id", self.ideation_run_id),
            ("direction_id", self.direction_id),
            ("snapshot_id", self.snapshot_id),
            ("map_run_id", self.map_run_id),
            ("map_assessment_id", self.map_assessment_id),
            ("selected_candidate_id", self.selected_candidate_id),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be non-empty")
        if not self.operational_predictions:
            raise ValueError(
                "an admission encodes at least one machine-checkable "
                "prediction; a seed with none would be unfalsifiable"
            )
        for label, items in (
            ("measurements", self.measurements),
            ("controls", self.controls),
            ("comparison_targets", self.comparison_targets),
        ):
            if not items:
                raise ValueError(
                    f"{label} are copied from the candidate and cannot "
                    f"be empty"
                )
            if any(not entry.strip() for entry in items):
                raise ValueError(f"every entry in {label} must be non-empty")
        if not self.evaluation_protocol.strip():
            raise ValueError("the evaluation protocol copy must be non-empty")
        if not self.inherited_requirements:
            raise ValueError(
                "the candidate's own resource statements always exist, "
                "so inherited requirements cannot be empty"
            )
        for entry in self.inherited_requirements:
            if entry.source is RequirementSource.ADMISSION_DIRECTIVE:
                raise ValueError(
                    "an operator statement is never presented as inherited"
                )
        if not self.operator_requirements:
            raise ValueError(
                "the directive's operator statements are always quoted"
            )
        for entry in self.operator_requirements:
            if entry.source is not RequirementSource.ADMISSION_DIRECTIVE:
                raise ValueError(
                    "an inherited requirement is never presented as "
                    "operator-stated"
                )
        if self.mechanical_reading != MECHANICAL_READING:
            raise ValueError(
                f"the encoded commitment is {MECHANICAL_READING!r} by "
                f"construction; nothing else can be claimed"
            )
        for label, value, prefix in (
            ("run_id", self.run_id, "adm_"),
            ("question_id", self.question_id, "q_"),
            ("hypothesis_id", self.hypothesis_id, "hyp_"),
            ("state_id", self.state_id, "st_"),
        ):
            if not value.startswith(prefix):
                raise ValueError(f"{label} must carry the {prefix} prefix")
        if len(self.prediction_ids) != len(self.operational_predictions):
            raise ValueError(
                "prediction ids align index-for-index with the "
                "operationalizations that produced them"
            )
        if len(set(self.prediction_ids)) != len(self.prediction_ids):
            raise ValueError(
                "two operationalizations derived the same prediction id; "
                "the encoding admits no duplicates"
            )
        for prediction_id in self.prediction_ids:
            if not prediction_id.startswith("pred_"):
                raise ValueError("prediction ids carry the pred_ prefix")
        if self.model_calls < 1:
            raise ValueError("an admission spent at least the gated call")
        for label, count in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if count < 0:
                raise ValueError(f"{label} cannot be negative")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "arun",
                    self.run_id,
                    self.directive_id,
                    self.selection_run_record_id,
                    self.selection_run_id,
                    self.selection_directive_id,
                    self.prior_art_run_record_id,
                    self.prior_art_run_id,
                    self.selected_prior_art_assessment_id,
                    self.ideation_run_record_id,
                    self.ideation_run_id,
                    self.direction_id,
                    self.snapshot_id,
                    self.map_run_id,
                    self.map_assessment_id,
                    self.selected_candidate_id,
                    tuple(
                        _operational_key(entry)
                        for entry in self.operational_predictions
                    ),
                    self.measurements,
                    self.controls,
                    self.comparison_targets,
                    self.evaluation_protocol,
                    tuple(
                        _requirement_key(entry)
                        for entry in self.inherited_requirements
                    ),
                    tuple(
                        _requirement_key(entry)
                        for entry in self.operator_requirements
                    ),
                    self.mechanical_reading,
                    self.question_id,
                    self.hypothesis_id,
                    self.prediction_ids,
                    self.state_id,
                    self.provenance.response_id,
                    self.model_calls,
                    self.input_tokens,
                    self.output_tokens,
                ),
            )


def _folded(text: str) -> str:
    return " ".join(text.casefold().split())


def _support_key(entry: GroundedSupport) -> tuple[object, ...]:
    return (entry.source, entry.field_path, entry.quote)


def _operational_key(entry: OperationalPrediction) -> tuple[object, ...]:
    return (
        entry.prediction_text,
        entry.condition,
        entry.base_metric,
        entry.expected_higher_arm,
        entry.expected_lower_arm,
        entry.contrary_observation,
        tuple(_support_key(link) for link in entry.support),
    )


def _requirement_key(entry: Requirement) -> tuple[object, ...]:
    return (entry.source, entry.record_id, entry.field_path, entry.quote)
