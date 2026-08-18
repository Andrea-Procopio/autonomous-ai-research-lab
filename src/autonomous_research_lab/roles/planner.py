"""The model-backed research planner: authoritative state in, one decision out.

The second concrete :class:`~autonomous_research_lab.roles.base.ResearchRole`
with a model behind it, for the ``RESEARCH_DIRECTOR`` seat. One invocation
performs one narrow slice of work::

    deterministic projection of ResearchState (never conversation history)
      -> one structured model call            (schema-validated locally)
      -> deterministic planning gate          (typed rejection rules)
      -> durable PlanningRecord               (full provider provenance)
      -> proposals for the governed commit    (or none, for replicate/stop)

The model's authority is deliberately narrow. It selects exactly one next
scientific action — a new falsifiable experiment, a replication of an
existing experiment, an ablation derived from one, or a justified stop —
and states the proposition chain that action rests on. It writes no code,
executes nothing, constructs no results, evidence, or verification records,
chooses no commands, paths, dependencies, or container settings (the schema
has no slot for any of them), and its experiment costs are stamped from the
trusted template catalog, never taken from the reply.

A decision that fails the deterministic gate is preserved as data and earns
at most one corrective call carrying the exact rejection rules. Scientific
disagreement is not a gate rule: a valid decision the lab finds boring,
disappointing, or grounded in negative evidence commits like any other.

Provider failures are runtime failures: accounting reaches the
:class:`~autonomous_research_lab.runtime.providers.UsageLedger` exactly
once and the typed error re-raises — no record, no proposals, no state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ..core.actions import ResearchAction, ResearchActionType
from ..core.budget import ResourceCost
from ..core.experiment import ExperimentSpec
from ..core.hypothesis import Hypothesis
from ..core.prediction import Comparator, Prediction
from ..core.proposals import (
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    Proposal,
)
from ..core.state import ResearchState
from ..runtime.planning_store import (
    PlanningAction,
    PlanningRecord,
    PlanningStore,
    StopReason,
)
from ..runtime.providers import (
    Message,
    MessageRole,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    OutputSchema,
    UsageLedger,
)
from ..runtime.verification import PositiveControl
from .base import (
    ResearchRole,
    RoleContext,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from .engineer import ImplementationTemplate

#: How many of the most recent results the planning projection includes —
#: the explicit bound that keeps the prompt finite as history grows.
MAX_PROJECTED_RESULTS: Final = 32

_NO_COMPARATOR: Final = "none"
_NO_STOP: Final = "none"
_NO_SEED: Final = -1


class PlannerContractError(RuntimeError):
    """The invocation itself is unusable — an unsupported action, or a
    context without a research question. Deterministic, raised before any
    model call; no spend, no side effects."""


class PlanningRejectedError(RuntimeError):
    """The model's decision failed the deterministic planning gate and was
    refused before anything reached the governed commit. Every attempt
    (payload and rules) is preserved in the planning store."""


@dataclass(frozen=True, slots=True)
class PlanningRejection:
    """One deterministic gate rule that fired, with its specifics."""

    rule: str
    detail: str


@dataclass(frozen=True, slots=True)
class TemplateCapability:
    """One trusted template and what it can measure — the catalog entry the
    planner may select and the gate holds it to."""

    template: ImplementationTemplate
    metrics: tuple[str, ...]
    estimated_cost: ResourceCost
    description: str = ""
    control: PositiveControl | None = None
    """The instrument control a faithful implementation from this template
    must satisfy, for wiring into the verification pipeline."""

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("a template capability must declare metrics")


@dataclass(frozen=True, slots=True)
class TemplateCatalog:
    """The explicitly supplied set of trusted templates. Nothing outside it
    is selectable, and the planner never authors a template."""

    entries: tuple[TemplateCapability, ...]

    def __post_init__(self) -> None:
        ids = [entry.template.id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("template ids in a catalog must be unique")

    def get(self, template_id: str) -> TemplateCapability | None:
        return next(
            (e for e in self.entries if e.template.id == template_id), None
        )


#: The whole output contract of the planning call: one flat decision. The
#: supported schema subset has no oneOf, so inapplicable fields carry typed
#: sentinels ("", [], -1, "none") and the gate enforces the per-action
#: discipline mechanically — which is also what makes "no hidden experiment
#: inside a stop" a checkable equality rather than prose. There is no slot
#: anywhere for an observed value, a result, a command, a path, a
#: dependency, or a container setting.
PLANNING_SCHEMA: Final = OutputSchema(
    name="planning_decision",
    json_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["new_experiment", "replicate", "ablation", "stop"],
            },
            "question_id": {
                "type": "string",
                "description": "Id of the research question this serves.",
            },
            "rationale": {
                "type": "string",
                "description": (
                    "Concise scientific rationale, grounded in the cited "
                    "evidence ids."
                ),
            },
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Ids of ADMISSIBLE evidence the rationale relies on."
                ),
            },
            "hypothesis_id": {
                "type": "string",
                "description": (
                    "Existing hypothesis id, or \"\" when proposing a new "
                    "one."
                ),
            },
            "hypothesis_statement": {
                "type": "string",
                "description": (
                    "New falsifiable hypothesis, or \"\" when hypothesis_id "
                    "is given."
                ),
            },
            "prediction_condition": {"type": "string"},
            "prediction_metric": {"type": "string"},
            "prediction_comparator": {
                "type": "string",
                "enum": ["lt", "le", "gt", "ge", "approx", "none"],
            },
            "prediction_threshold": {"type": "number"},
            "prediction_tolerance": {"type": "number"},
            "prediction_expectation": {"type": "string"},
            "experiment_objective": {"type": "string"},
            "experiment_procedure": {"type": "string"},
            "experiment_metrics": {
                "type": "array",
                "items": {"type": "string"},
            },
            "experiment_baselines": {
                "type": "array",
                "items": {"type": "string"},
            },
            "experiment_controls": {
                "type": "array",
                "items": {"type": "string"},
            },
            "experiment_seeds": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "template_id": {
                "type": "string",
                "description": (
                    "Id of the trusted template from the supplied catalog, "
                    "or \"\"."
                ),
            },
            "target_experiment_id": {
                "type": "string",
                "description": (
                    "Existing experiment id for replicate/ablation, else "
                    "\"\"."
                ),
            },
            "replication_seed": {
                "type": "integer",
                "description": (
                    "The next unused declared seed for a replication; -1 "
                    "otherwise."
                ),
            },
            "removed_component": {
                "type": "string",
                "description": (
                    "For ablation: the component of the parent procedure "
                    "removed; \"\" otherwise."
                ),
            },
            "stop_reason": {
                "type": "string",
                "enum": [
                    "none",
                    "budget_insufficient",
                    "question_resolved",
                    "hypothesis_refuted",
                    "no_informative_next_experiment",
                ],
            },
        },
        "required": [
            "action",
            "question_id",
            "rationale",
            "evidence_ids",
            "hypothesis_id",
            "hypothesis_statement",
            "prediction_condition",
            "prediction_metric",
            "prediction_comparator",
            "prediction_threshold",
            "prediction_tolerance",
            "prediction_expectation",
            "experiment_objective",
            "experiment_procedure",
            "experiment_metrics",
            "experiment_baselines",
            "experiment_controls",
            "experiment_seeds",
            "template_id",
            "target_experiment_id",
            "replication_seed",
            "removed_component",
            "stop_reason",
        ],
        "additionalProperties": False,
    },
)

PLANNER_INSTRUCTION: Final = (
    "You are the research planner of an autonomous scientific laboratory. "
    "From the authoritative research state you are shown — and nothing "
    "else — you select exactly ONE next action: a new falsifiable "
    "experiment, a replication of an existing experiment at its next "
    "unused declared seed, an ablation of an existing experiment that "
    "removes one named component of its procedure, or a justified stop. "
    "You return exactly the JSON the schema requires. Ground your "
    "rationale only in evidence marked ADMISSIBLE, citing its ids. Never "
    "invent results, metrics values, or observations; never propose work "
    "the remaining budget cannot afford; a new or ablated experiment must "
    "use one template from the supplied catalog and declare only metrics "
    "that template can measure, including the prediction's metric. Fields "
    "that do not apply to your chosen action MUST carry their sentinel "
    "values exactly: empty string for unused text fields, empty arrays "
    "for unused lists, -1 for replication_seed, \"none\" for "
    "prediction_comparator and stop_reason, and 0 for unused numeric "
    "thresholds and tolerances. A stop decision carries only its typed "
    "stop_reason, cited evidence, and rationale."
)


# -- the deterministic planning gate -------------------------------------------


def check_decision(
    payload: Mapping[str, object],
    *,
    context: RoleContext,
    catalog: TemplateCatalog,
) -> tuple[PlanningRejection, ...]:
    """Every deterministic rule the decision violates — all of them, so a
    corrective call receives complete feedback. Pure: reads only the typed
    projection, the trusted catalog, and the payload."""
    rejections: list[PlanningRejection] = []
    action = str(payload["action"])

    _check_references(payload, context, rejections)
    _check_sentinels(action, payload, rejections)

    if action == PlanningAction.STOP.value:
        if str(payload["stop_reason"]) == _NO_STOP:
            rejections.append(
                PlanningRejection(
                    "inconsistent_chain",
                    "a stop decision requires a typed stop_reason",
                )
            )
    elif action == PlanningAction.REPLICATE.value:
        _check_replication(payload, context, rejections)
    else:  # new_experiment / ablation
        _check_chain_fields(payload, rejections)
        _check_template(payload, catalog, rejections)
        if action == PlanningAction.ABLATION.value:
            _check_ablation(payload, context, rejections)
        if not rejections:
            _check_derived_chain(payload, context, catalog, rejections)

    return tuple(rejections)


def _check_references(
    payload: Mapping[str, object],
    context: RoleContext,
    rejections: list[PlanningRejection],
) -> None:
    if not str(payload["rationale"]).strip():
        rejections.append(
            PlanningRejection(
                "inconsistent_chain", "every decision requires a rationale"
            )
        )
    question_id = str(payload["question_id"])
    if question_id not in {q.id for q in context.questions}:
        rejections.append(
            PlanningRejection(
                "unknown_question",
                f"question {question_id!r} is not in the research state",
            )
        )
    cited = _strings(payload["evidence_ids"])
    if not cited:
        rejections.append(
            PlanningRejection(
                "inadmissible_evidence_cited",
                "a decision must cite at least one admissible evidence id",
            )
        )
    known = {e.id for e in context.evidence}
    admissible = set(context.admissible_evidence_ids)
    for evidence_id in cited:
        if evidence_id not in known:
            rejections.append(
                PlanningRejection(
                    "unknown_evidence",
                    f"evidence {evidence_id!r} is not in the research state",
                )
            )
        elif evidence_id not in admissible:
            rejections.append(
                PlanningRejection(
                    "inadmissible_evidence_cited",
                    f"evidence {evidence_id!r} does not stand as verified "
                    f"scientific evidence and cannot ground a decision",
                )
            )


#: Which sentinel-checked fields each action legitimately uses. Everything
#: else must carry its sentinel — mechanically, so a stop can hide no
#: experiment and a replication can smuggle no new propositions.
_CHAIN_FIELDS: Final = frozenset(
    {
        "hypothesis_id",
        "hypothesis_statement",
        "prediction_condition",
        "prediction_metric",
        "prediction_comparator",
        "prediction_threshold",
        "prediction_tolerance",
        "prediction_expectation",
        "experiment_objective",
        "experiment_procedure",
        "experiment_metrics",
        "experiment_baselines",
        "experiment_controls",
        "experiment_seeds",
        "template_id",
    }
)

_APPLICABLE: Final[Mapping[str, frozenset[str]]] = {
    PlanningAction.NEW_EXPERIMENT.value: _CHAIN_FIELDS,
    PlanningAction.ABLATION.value: _CHAIN_FIELDS
    | {"target_experiment_id", "removed_component"},
    PlanningAction.REPLICATE.value: frozenset(
        {"target_experiment_id", "replication_seed"}
    ),
    PlanningAction.STOP.value: frozenset({"stop_reason"}),
}

_SENTINELS: Final[tuple[tuple[str, object], ...]] = (
    ("hypothesis_id", ""),
    ("hypothesis_statement", ""),
    ("prediction_condition", ""),
    ("prediction_metric", ""),
    ("prediction_comparator", _NO_COMPARATOR),
    ("prediction_threshold", 0),
    ("prediction_tolerance", 0),
    ("prediction_expectation", ""),
    ("experiment_objective", ""),
    ("experiment_procedure", ""),
    ("experiment_metrics", ()),
    ("experiment_baselines", ()),
    ("experiment_controls", ()),
    ("experiment_seeds", ()),
    ("template_id", ""),
    ("target_experiment_id", ""),
    ("replication_seed", _NO_SEED),
    ("removed_component", ""),
    ("stop_reason", _NO_STOP),
)


def _check_sentinels(
    action: str,
    payload: Mapping[str, object],
    rejections: list[PlanningRejection],
) -> None:
    applicable = _APPLICABLE[action]
    for field_name, sentinel in _SENTINELS:
        if field_name in applicable:
            continue
        value = payload[field_name]
        if isinstance(value, Sequence) and not isinstance(value, str):
            value = tuple(value)
        if value != sentinel:
            rejections.append(
                PlanningRejection(
                    "inapplicable_field",
                    f"{field_name} does not apply to a {action} decision "
                    f"and must carry its sentinel, got {value!r}",
                )
            )


def _check_chain_fields(
    payload: Mapping[str, object], rejections: list[PlanningRejection]
) -> None:
    has_reference = bool(str(payload["hypothesis_id"]).strip())
    has_statement = bool(str(payload["hypothesis_statement"]).strip())
    if has_reference == has_statement:
        rejections.append(
            PlanningRejection(
                "inconsistent_chain",
                "exactly one of hypothesis_id and hypothesis_statement "
                "must be given",
            )
        )
    if not str(payload["prediction_condition"]).strip():
        rejections.append(
            PlanningRejection(
                "unfalsifiable_prediction",
                "a prediction requires the condition under which it holds",
            )
        )
    if not str(payload["prediction_metric"]).strip():
        rejections.append(
            PlanningRejection(
                "unfalsifiable_prediction", "a prediction requires a metric"
            )
        )
    comparator = str(payload["prediction_comparator"])
    if comparator == _NO_COMPARATOR:
        rejections.append(
            PlanningRejection(
                "unfalsifiable_prediction",
                "a prediction requires a comparison operator",
            )
        )
    threshold = _number(payload["prediction_threshold"])
    tolerance = _number(payload["prediction_tolerance"])
    if not math.isfinite(threshold):
        rejections.append(
            PlanningRejection(
                "unfalsifiable_prediction",
                "the prediction threshold must be finite",
            )
        )
    if tolerance < 0:
        rejections.append(
            PlanningRejection(
                "unfalsifiable_prediction", "tolerance must be non-negative"
            )
        )
    if comparator == Comparator.APPROXIMATELY.value and tolerance <= 0:
        rejections.append(
            PlanningRejection(
                "unfalsifiable_prediction",
                "an approx prediction with zero tolerance can never be "
                "checked against a measured value",
            )
        )
    if not str(payload["experiment_objective"]).strip():
        rejections.append(
            PlanningRejection(
                "inconsistent_chain", "an experiment requires an objective"
            )
        )
    if not str(payload["experiment_procedure"]).strip():
        rejections.append(
            PlanningRejection(
                "inconsistent_chain", "an experiment requires a procedure"
            )
        )
    metrics = _strings(payload["experiment_metrics"])
    if not metrics:
        rejections.append(
            PlanningRejection(
                "inconsistent_chain",
                "an experiment must declare at least one metric",
            )
        )
    metric = str(payload["prediction_metric"])
    if metric and metrics and metric not in metrics:
        rejections.append(
            PlanningRejection(
                "undeclared_metric",
                f"the prediction metric {metric!r} is not among the "
                f"experiment's declared metrics",
            )
        )
    seeds = _integers(payload["experiment_seeds"])
    if not seeds:
        rejections.append(
            PlanningRejection(
                "inconsistent_chain",
                "an experiment must declare at least one seed",
            )
        )
    if len(seeds) != len(set(seeds)) or any(s < 0 for s in seeds):
        rejections.append(
            PlanningRejection(
                "inconsistent_chain",
                "declared seeds must be distinct non-negative integers",
            )
        )


def _check_template(
    payload: Mapping[str, object],
    catalog: TemplateCatalog,
    rejections: list[PlanningRejection],
) -> None:
    template_id = str(payload["template_id"])
    capability = catalog.get(template_id)
    if capability is None:
        rejections.append(
            PlanningRejection(
                "unknown_template",
                f"template {template_id!r} is not in the trusted catalog",
            )
        )
        return
    undeclared = [
        m
        for m in _strings(payload["experiment_metrics"])
        if m not in capability.metrics
    ]
    if undeclared:
        rejections.append(
            PlanningRejection(
                "undeclared_metric",
                f"metric(s) {', '.join(undeclared)} are not declared by "
                f"template {template_id}",
            )
        )


def _check_ablation(
    payload: Mapping[str, object],
    context: RoleContext,
    rejections: list[PlanningRejection],
) -> None:
    target_id = str(payload["target_experiment_id"])
    parent = next(
        (spec for spec in context.experiments if spec.id == target_id), None
    )
    if parent is None:
        rejections.append(
            PlanningRejection(
                "invalid_ablation_parent",
                f"ablation parent {target_id!r} is not an experiment in the "
                f"research state",
            )
        )
        return
    removed = str(payload["removed_component"]).strip()
    if not removed:
        rejections.append(
            PlanningRejection(
                "unnamed_removed_component",
                "an ablation must name the component it removes",
            )
        )
        return
    named_in_parent = removed.lower() in parent.procedure.lower() or any(
        removed.lower() in control.lower() for control in parent.controls
    )
    if not named_in_parent:
        rejections.append(
            PlanningRejection(
                "unnamed_removed_component",
                f"removed component {removed!r} does not appear in the "
                f"parent experiment's procedure or controls",
            )
        )


def _check_replication(
    payload: Mapping[str, object],
    context: RoleContext,
    rejections: list[PlanningRejection],
) -> None:
    target_id = str(payload["target_experiment_id"])
    target = next(
        (spec for spec in context.experiments if spec.id == target_id), None
    )
    if target is None:
        rejections.append(
            PlanningRejection(
                "unknown_target_experiment",
                f"replication target {target_id!r} is not an experiment in "
                f"the research state",
            )
        )
        return
    seed = _number(payload["replication_seed"])
    seed = int(seed)
    used = {r.seed for r in context.results if r.spec_id == target.id}
    expected = next((s for s in target.seeds if s not in used), None)
    if expected is None:
        rejections.append(
            PlanningRejection(
                "seed_already_used",
                f"every declared seed of {target_id} has already been run; "
                f"replication needs an unused declared seed",
            )
        )
        return
    if seed not in target.seeds:
        rejections.append(
            PlanningRejection(
                "seed_policy_mismatch",
                f"seed {seed} is not among the declared seeds "
                f"{target.seeds}",
            )
        )
        return
    if seed in used:
        rejections.append(
            PlanningRejection(
                "seed_already_used",
                f"seed {seed} has already produced a result for {target_id}",
            )
        )
        return
    if seed != expected:
        rejections.append(
            PlanningRejection(
                "seed_policy_mismatch",
                f"the deterministic seed policy requires the first unused "
                f"declared seed, {expected}, not {seed}",
            )
        )
    if (
        context.remaining_budget is not None
        and not context.remaining_budget.can_afford(target.estimated_cost)
    ):
        rejections.append(
            PlanningRejection(
                "budget_insufficient",
                f"the remaining budget cannot afford replicating "
                f"{target_id}",
            )
        )


def _check_derived_chain(
    payload: Mapping[str, object],
    context: RoleContext,
    catalog: TemplateCatalog,
    rejections: list[PlanningRejection],
) -> None:
    """Checks that need the actual chain: hypothesis reference integrity,
    duplicate detection by content id, and affordability of the stamped
    cost. Runs only when the field-level checks all passed, so the domain
    constructors cannot raise."""
    reference = str(payload["hypothesis_id"]).strip()
    if reference and reference not in {h.id for h in context.hypotheses}:
        rejections.append(
            PlanningRejection(
                "unknown_hypothesis",
                f"hypothesis {reference!r} is not in the research state",
            )
        )
        return
    if reference:
        referenced = next(
            h for h in context.hypotheses if h.id == reference
        )
        if referenced.question_id != str(payload["question_id"]):
            rejections.append(
                PlanningRejection(
                    "inconsistent_chain",
                    f"hypothesis {reference} answers question "
                    f"{referenced.question_id}, not the cited question",
                )
            )
            return
    _, _, spec = derive_chain(payload, catalog)
    if spec.id in {existing.id for existing in context.experiments}:
        rejections.append(
            PlanningRejection(
                "duplicate_experiment",
                f"the proposed experiment derives the same identity as "
                f"existing experiment {spec.id}; propose a replication "
                f"explicitly instead",
            )
        )
    if (
        context.remaining_budget is not None
        and not context.remaining_budget.can_afford(spec.estimated_cost)
    ):
        rejections.append(
            PlanningRejection(
                "budget_insufficient",
                "the remaining budget cannot afford the proposed "
                "experiment's catalog-stamped cost",
            )
        )


# -- deterministic expansion ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpandedDecision:
    """What one accepted decision puts in front of the governed commit,
    plus the ids the planning record references."""

    proposals: tuple[Proposal, ...]
    hypothesis_id: str = ""
    prediction_id: str = ""
    spec_id: str = ""


def derive_chain(
    payload: Mapping[str, object], catalog: TemplateCatalog
) -> tuple[Hypothesis | None, Prediction, ExperimentSpec]:
    """The proposition chain a gate-valid new-experiment or ablation
    decision determines — one derivation shared by the gate's duplicate
    check and the expansion, so they cannot disagree."""
    reference = str(payload["hypothesis_id"]).strip()
    hypothesis: Hypothesis | None = None
    if reference:
        hypothesis_id = reference
    else:
        hypothesis = Hypothesis(
            statement=str(payload["hypothesis_statement"]),
            rationale=str(payload["rationale"]),
            question_id=str(payload["question_id"]),
        )
        hypothesis_id = hypothesis.id
    prediction = Prediction(
        hypothesis_id=hypothesis_id,
        condition=str(payload["prediction_condition"]),
        metric=str(payload["prediction_metric"]),
        comparator=Comparator(str(payload["prediction_comparator"])),
        threshold=float(_number(payload["prediction_threshold"])),
        tolerance=float(_number(payload["prediction_tolerance"])),
        expectation=str(payload["prediction_expectation"]),
    )
    capability = catalog.get(str(payload["template_id"]))
    assert capability is not None  # the gate checked template membership
    baselines = _strings(payload["experiment_baselines"])
    if str(payload["action"]) == PlanningAction.ABLATION.value:
        # The scientific record itself names the parent and the removal.
        baselines = (
            *baselines,
            f"ablation of {payload['target_experiment_id']}: removed "
            f"{payload['removed_component']}",
        )
    spec = ExperimentSpec(
        prediction_id=prediction.id,
        objective=str(payload["experiment_objective"]),
        procedure=str(payload["experiment_procedure"]),
        metrics=_strings(payload["experiment_metrics"]),
        baselines=baselines,
        controls=_strings(payload["experiment_controls"]),
        seeds=_integers(payload["experiment_seeds"]),
        estimated_cost=capability.estimated_cost,
    )
    return hypothesis, prediction, spec


def expand_decision(
    payload: Mapping[str, object],
    *,
    context: RoleContext,
    catalog: TemplateCatalog,
    proposer: str,
) -> ExpandedDecision:
    """The deterministic expansion of one gate-accepted decision. Order in
    the bundle is (hypothesis?, prediction, experiment) so the atomic
    commit sees every referent before its dependent."""
    action = str(payload["action"])
    if action == PlanningAction.STOP.value:
        return ExpandedDecision(proposals=())
    if action == PlanningAction.REPLICATE.value:
        target_id = str(payload["target_experiment_id"])
        target = next(s for s in context.experiments if s.id == target_id)
        prediction = next(
            (p for p in context.predictions if p.id == target.prediction_id),
            None,
        )
        return ExpandedDecision(
            proposals=(),
            hypothesis_id=(
                prediction.hypothesis_id if prediction is not None else ""
            ),
            prediction_id=target.prediction_id,
            spec_id=target.id,
        )
    hypothesis, prediction, spec = derive_chain(payload, catalog)
    proposals: list[Proposal] = []
    if hypothesis is not None:
        proposals.append(
            HypothesisProposal(hypothesis=hypothesis, proposer=proposer)
        )
    proposals.append(
        PredictionProposal(prediction=prediction, proposer=proposer)
    )
    proposals.append(ExperimentProposal(spec=spec, proposer=proposer))
    return ExpandedDecision(
        proposals=tuple(proposals),
        hypothesis_id=prediction.hypothesis_id,
        prediction_id=prediction.id,
        spec_id=spec.id,
    )


# -- the role -------------------------------------------------------------------


class ModelBackedPlanner(ResearchRole):
    """See the module docstring; construction is explicit wiring, and every
    collaborator is injected — provider, ledger, store, catalog — so tests
    and live runs differ only in what is plugged in."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        ledger: UsageLedger,
        store: PlanningStore,
        catalog: TemplateCatalog,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        request_timeout_seconds: float = 240.0,
        max_corrective_calls: int = 1,
    ) -> None:
        self._provider = provider
        self._model = model
        self._ledger = ledger
        self._store = store
        self._catalog = catalog
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._request_timeout_seconds = request_timeout_seconds
        self._max_corrective_calls = max_corrective_calls

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_DIRECTOR

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset({ResearchActionType.PLAN_NEXT_ACTION})

    def suitability(
        self,
        state: ResearchState,  # noqa: ARG002 - static seat, static fit
        action: ResearchAction,
    ) -> RoleSuitability:
        fits = action.action_type in self.supported_actions
        return RoleSuitability(
            value=1.0 if fits else 0.0,
            rationale="the one model-backed planning seat",
        )

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        if invocation.assignment.action_type is not (
            ResearchActionType.PLAN_NEXT_ACTION
        ):
            raise PlannerContractError(
                f"the planner performs plan_next_action, not "
                f"{invocation.assignment.action_type}"
            )
        if not invocation.context.questions:
            raise PlannerContractError(
                "the planner needs at least one research question in its "
                "context"
            )

        request = self._request(invocation)
        response = self._invoke(request)
        # Bounded corrective call: a decision the deterministic gate
        # rejects earns at most ``max_corrective_calls`` retries, each
        # carrying every rule that fired. Every rejected attempt is
        # preserved before the retry happens. Only gate rules trigger this
        # path — a valid decision the lab merely dislikes has no route to
        # a second call.
        repairs = 0
        while True:
            payload = response.structured
            if payload is None:
                rejections: tuple[PlanningRejection, ...] = (
                    PlanningRejection(
                        "no_structured_payload",
                        "the reply carried no structured payload",
                    ),
                )
            else:
                rejections = check_decision(
                    payload, context=invocation.context, catalog=self._catalog
                )
            if not rejections:
                break
            self._store.preserve_rejected(
                invocation_id=invocation.id,
                reasons=tuple((r.rule, r.detail) for r in rejections),
                request_fingerprint=response.request_fingerprint,
                response_id=response.id,
                payload=payload if payload is not None else response.text,
                repair=repairs,
            )
            if repairs >= self._max_corrective_calls:
                raise PlanningRejectedError(
                    "planning decision rejected by the deterministic gate: "
                    + "; ".join(f"{r.rule}: {r.detail}" for r in rejections)
                )
            repairs += 1
            request = _repair_request(request, response, rejections, repairs)
            response = self._invoke(request)

        assert payload is not None  # an empty payload never passes the gate
        expanded = expand_decision(
            payload,
            context=invocation.context,
            catalog=self._catalog,
            proposer=f"planner:{self._provider.name}:{self._model}",
        )
        stop_value = str(payload["stop_reason"])
        record = PlanningRecord(
            invocation_id=invocation.id,
            action=PlanningAction(str(payload["action"])),
            question_id=str(payload["question_id"]),
            rationale=str(payload["rationale"]),
            evidence_ids=_strings(payload["evidence_ids"]),
            hypothesis_id=expanded.hypothesis_id,
            prediction_id=expanded.prediction_id,
            spec_id=expanded.spec_id,
            parent_experiment_id=(
                str(payload["target_experiment_id"])
                if str(payload["action"]) == PlanningAction.ABLATION.value
                else ""
            ),
            removed_component=str(payload["removed_component"]),
            replication_seed=(
                int(_number(payload["replication_seed"]))
                if str(payload["action"]) == PlanningAction.REPLICATE.value
                else None
            ),
            template_id=str(payload["template_id"]),
            stop_reason=(
                StopReason(stop_value) if stop_value != _NO_STOP else None
            ),
            repair_count=repairs,
            request_fingerprint=response.request_fingerprint,
            response_id=response.id,
            provider=response.provider,
            requested_model=self._model,
            served_model=response.model,
            provider_request_id=response.request_id,
            latency_seconds=response.latency_seconds,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            nominal_cost_usd=(
                response.nominal_cost.usd
                if response.nominal_cost is not None
                else None
            ),
        )
        self._store.record(record)
        return expanded.proposals

    # -- the model call ------------------------------------------------------

    def _request(self, invocation: RoleInvocation) -> ModelRequest:
        return ModelRequest(
            model=self._model,
            instruction=PLANNER_INSTRUCTION,
            messages=(
                Message(
                    role=MessageRole.USER,
                    content=render_planning_context(
                        invocation.context, self._catalog
                    ),
                ),
            ),
            schema=PLANNING_SCHEMA,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            timeout_seconds=self._request_timeout_seconds,
            metadata={
                "invocation_id": invocation.id,
                "action": invocation.assignment.action_type.value,
            },
        )

    def _invoke(self, request: ModelRequest) -> ModelResponse:
        """One provider call, with its accounting always reaching the
        ledger: the response on success, the attached cost on failure —
        recorded exactly once — before the typed error propagates."""
        try:
            response = self._provider.invoke(request)
        except ModelProviderError as error:
            self._ledger.record_failure(error)
            raise
        self._ledger.record(response)
        return response


def _repair_request(
    base: ModelRequest,
    failed: ModelResponse,
    rejections: tuple[PlanningRejection, ...],
    attempt: int,
) -> ModelRequest:
    """One corrective request: the failed reply plus every deterministic
    rule that fired. Gate rules only — never scientific taste."""
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    feedback = (
        f"Your decision was rejected by the deterministic planning gate. "
        f"Nothing was committed. The rules that fired:\n{rules}\n"
        f"Return one corrected decision now, satisfying every original "
        f"constraint, including the sentinel discipline for fields that "
        f"do not apply to your chosen action."
    )
    return ModelRequest(
        model=base.model,
        instruction=base.instruction,
        messages=(
            *base.messages,
            Message(
                role=MessageRole.ASSISTANT,
                content=failed.text or "(empty reply)",
            ),
            Message(role=MessageRole.USER, content=feedback),
        ),
        schema=base.schema,
        max_output_tokens=base.max_output_tokens,
        temperature=base.temperature,
        timeout_seconds=base.timeout_seconds,
        metadata={**base.metadata, "planning_repair": str(attempt)},
    )


# -- the deterministic projection renderer ---------------------------------------


def render_planning_context(
    context: RoleContext, catalog: TemplateCatalog
) -> str:
    """The bounded projection as text: authoritative state only, iterated
    in state order with fixed formatting, so identical state renders an
    identical prompt and an identical request fingerprint."""
    lines: list[str] = ["## Research questions"]
    for question in context.questions:
        lines.append(f"- {question.id}: {question.text}")
    lines.append("\n## Hypotheses")
    if not context.hypotheses:
        lines.append("- none yet")
    for hypothesis in context.hypotheses:
        lines.append(
            f"- {hypothesis.id} (question {hypothesis.question_id}): "
            f"{hypothesis.statement}"
        )
    lines.append("\n## Predictions")
    if not context.predictions:
        lines.append("- none yet")
    for prediction in context.predictions:
        lines.append(
            f"- {prediction.id} (hypothesis {prediction.hypothesis_id}): "
            f"{prediction.metric} {prediction.comparator.value} "
            f"{prediction.threshold!r} under: {prediction.condition}"
        )
    lines.append("\n## Experiments")
    if not context.experiments:
        lines.append("- none yet")
    for spec in context.experiments:
        lines.append(
            f"- {spec.id} (prediction {spec.prediction_id}): {spec.objective}"
        )
        lines.append(f"  procedure: {spec.procedure}")
        lines.append(f"  metrics: {', '.join(spec.metrics)}")
        lines.append(
            f"  declared seeds: {list(spec.seeds)!r}; controls: "
            f"{'; '.join(spec.controls) if spec.controls else 'none'}"
        )
    lines.append("\n## Results")
    if not context.results:
        lines.append("- none yet")
    for result in context.results:
        metrics = ", ".join(
            f"{key}={result.metrics[key]!r}" for key in sorted(result.metrics)
        )
        lines.append(
            f"- {result.id} (experiment {result.spec_id}, seed "
            f"{result.seed!r}, status {result.status.value}): "
            f"{metrics or 'no metrics'}"
        )
    lines.append("\n## Prediction tests")
    if not context.prediction_tests:
        lines.append("- none yet")
    for test in context.prediction_tests:
        lines.append(
            f"- prediction {test.prediction_id} vs result {test.result_id}: "
            f"{test.consistency.value} (observed {test.observed!r})"
        )
    lines.append("\n## Evidence")
    if not context.evidence:
        lines.append("- none yet")
    admissible = set(context.admissible_evidence_ids)
    for evidence in context.evidence:
        standing = (
            "ADMISSIBLE" if evidence.id in admissible else "INADMISSIBLE"
        )
        lines.append(
            f"- {evidence.id} [{standing}] (result {evidence.result_id}, "
            f"experiment {evidence.spec_id}, kind {evidence.kind.value}): "
            f"{evidence.observation}"
        )
    lines.append("\n## Epistemic assessments")
    if not context.assessments:
        lines.append("- none yet")
    for assessment in context.assessments:
        lines.append(
            f"- {assessment.subject_id}: {assessment.verdict.value} "
            f"({assessment.rationale})"
        )
    lines.append("\n## Standing notes and contradictions")
    if not context.notes:
        lines.append("- none")
    for note in context.notes:
        lines.append(f"- {note}")
    budget = context.remaining_budget
    lines.append("\n## Remaining budget")
    if budget is None:
        lines.append("- not disclosed")
    else:
        lines.append(
            f"- wall_clock_seconds={budget.wall_clock_seconds!r}, "
            f"gpu_hours={budget.gpu_hours!r}, usd={budget.usd!r}, "
            f"model_tokens={budget.model_tokens}"
        )
    lines.append("\n## Trusted template catalog")
    for entry in catalog.entries:
        cost = entry.estimated_cost
        lines.append(
            f"- {entry.template.id} ({entry.template.name}): "
            f"{entry.description or 'no description'}"
        )
        lines.append(f"  measurable metrics: {', '.join(entry.metrics)}")
        lines.append(
            f"  estimated cost per run: "
            f"wall_clock_seconds={cost.wall_clock_seconds!r}, "
            f"usd={cost.usd!r}, model_tokens={cost.model_tokens}"
        )
        if entry.control is not None:
            control = entry.control
            lines.append(
                f"  instrument control (declare its metric): "
                f"{control.metric} {control.comparator.value} "
                f"{control.threshold!r} — {control.rationale}"
            )
    lines.append(
        "\nDecide exactly one next action now and return the JSON decision."
    )
    return "\n".join(lines)


# -- payload coercion helpers -----------------------------------------------------


def _strings(value: object) -> tuple[str, ...]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return tuple(str(item) for item in value)


def _integers(value: object) -> tuple[int, ...]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return tuple(int(_number(item)) for item in value)


def _number(value: object) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return value


__all__ = [
    "MAX_PROJECTED_RESULTS",
    "PLANNER_INSTRUCTION",
    "PLANNING_SCHEMA",
    "ExpandedDecision",
    "ModelBackedPlanner",
    "PlannerContractError",
    "PlanningRejectedError",
    "PlanningRejection",
    "TemplateCapability",
    "TemplateCatalog",
    "check_decision",
    "derive_chain",
    "expand_decision",
    "render_planning_context",
]
