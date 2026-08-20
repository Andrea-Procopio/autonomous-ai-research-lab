"""The admitter: one gated translation from a selection to a state.

``run()`` is the public admission operation, and it is all-or-nothing:
directive, replay, conflict, door, preflight, one gated model call (at
most one corrective), trusted construction of every core record, the
state snapshot with a read-back verification, and the write-once
admission record — in that order. A failure anywhere leaves nothing but
the directive and the preserved rejections; a crash between the
snapshot and the record leaves an inert orphan snapshot ("no record
means not admitted"), and the re-run spends one fresh gated call —
never claimed to be free.

The model's schema holds only the operationalization: for each recorded
prediction, a condition, a base metric chosen from the candidate's own
list, the two comparison arms in the candidate's own words, a contrary
restatement of the recorded falsifier, and grounded support links.
There is no field for a hypothesis, a question, an id, a number, a
requirement, or a different candidate. Everything else the state needs
is a deterministic verbatim copy stamped by trusted code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from ..core.hypothesis import Hypothesis
from ..core.ids import occurrence_id
from ..core.prediction import Comparator, Prediction
from ..core.question import ResearchQuestion
from ..core.state import ResearchState
from ..ideation.direction import DirectionRecord
from ..ideation.records import CandidateIdea
from ..ideation.store import IdeationStore
from ..mapping.records import CallProvenance
from ..priorart.store import PriorArtStore
from ..runtime.providers import (
    Message,
    MessageRole,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    OutputSchema,
    StructuredOutputError,
    UsageLedger,
)
from ..selection.store import SelectionStore
from .directive import AdmissionDirective
from .door import (
    AdmissionInputs,
    require_selected_candidate_for_admission,
)
from .gates import (
    _REFERENCE,
    MappingRejection,
    _normalized,
    _number_tokens,
    _without_known_ids,
    check_operationalization,
)
from .preflight import check_admission_coherence
from .records import (
    MECHANICAL_READING,
    AdmissionRecord,
    GroundedSupport,
    OperationalPrediction,
    Requirement,
    RequirementSource,
    SupportSource,
)
from .store import AdmissionConflictError, AdmissionStore


class AdmissionRejectedError(RuntimeError):
    """The gate refused the payload after the bounded corrective call.
    Everything refused is preserved; nothing was recorded."""


class AdmissionBudgetError(RuntimeError):
    """A call would exceed the directive's model-call budget. The
    refusal happens before the call, never after."""


# -- output contract ------------------------------------------------------------

_SUPPORT_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "source": {
            "type": "string",
            "enum": [source.value for source in SupportSource],
        },
        "field_path": {
            "type": "string",
            "description": (
                "One of the labeled field paths shown; direction fields "
                "carry the 'direction.' prefix."
            ),
        },
        "quote": {
            "type": "string",
            "description": (
                "A verbatim fragment of that field's shown text; the "
                "gate re-finds it there."
            ),
        },
    },
    "required": ["source", "field_path", "quote"],
}


def operationalization_schema(metrics: tuple[str, ...]) -> OutputSchema:
    """The per-run output contract: the base metric is an enum over the
    candidate's own declared metrics, so an invented observable is
    unexpressible. No numeric field, no id field, no stop shape."""
    return OutputSchema(
        name="admission_operationalization",
        json_schema={
            "type": "object",
            "properties": {
                "operational_predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prediction_text": {
                                "type": "string",
                                "description": (
                                    "One recorded prediction, quoted "
                                    "verbatim from predictions[i].text."
                                ),
                            },
                            "condition": {"type": "string"},
                            "base_metric": {
                                "type": "string",
                                "enum": list(metrics),
                            },
                            "expected_higher_arm": {"type": "string"},
                            "expected_lower_arm": {"type": "string"},
                            "contrary_observation": {
                                "type": "string",
                                "description": (
                                    "A verbatim fragment of that "
                                    "prediction's falsifier."
                                ),
                            },
                            "support": {
                                "type": "array",
                                "items": _SUPPORT_SCHEMA,
                            },
                        },
                        "required": [
                            "prediction_text",
                            "condition",
                            "base_metric",
                            "expected_higher_arm",
                            "expected_lower_arm",
                            "contrary_observation",
                            "support",
                        ],
                    },
                }
            },
            "required": ["operational_predictions"],
        },
    )


_TEXT_NOTE: Final = (
    " In prose, prefer plain punctuation - simple hyphens and straight "
    "quotes - because decorative dashes are corrupted in transit and the "
    "gate rejects control characters; legitimate Unicode in names, "
    "technical terms, and titles is preserved."
)

OPERATIONALIZATION_INSTRUCTION: Final = (
    "You translate ONE selected research candidate's recorded "
    "predictions into machine-checkable form. Translate, never invent: "
    "for each prediction shown under predictions[i].text, return one to "
    "three encodings. In each, quote the prediction verbatim as "
    "prediction_text; state the condition under which the comparison is "
    "asserted; choose base_metric from the shown metric list; name the "
    "two compared arms in the candidate's own words - both must "
    "re-appear verbatim in the shown fields, and expected_higher_arm is "
    "the one the prediction expects to score higher on the base metric; "
    "restate the recorded falsifier verbatim as contrary_observation; "
    "and attach at least one support link quoting a shown field at its "
    "labeled path. Trusted code turns each encoding into the "
    "pre-registered commitment 'the difference (higher arm minus lower "
    "arm) is greater than zero' - you never state a threshold, an "
    "effect size, or any number the shown fields do not contain. Cover "
    "every recorded prediction. Do not mention record identifiers "
    "except those shown, do not claim novelty or guaranteed success, "
    "and keep every field to one or two short sentences."
    + _TEXT_NOTE
)


@dataclass(frozen=True, slots=True)
class AdmissionRunResult:
    """Everything one admission concluded: the durable record, the
    admitted state, and — on a fresh run — the verified inputs.
    ``replayed`` is True when a completed directive returned its stored
    result with zero provider calls."""

    record: AdmissionRecord
    state: ResearchState
    directive: AdmissionDirective
    inputs: AdmissionInputs | None
    replayed: bool


class _Spend:
    """Mutable within one run: what the run has consumed so far."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


# -- rendering (prompt == haystack, per field) ----------------------------------


def build_field_texts(
    candidate: CandidateIdea, direction: DirectionRecord
) -> dict[str, str]:
    """The closed grounding surface: every quotable field path mapped to
    the exact text the prompt shows for it. Direction paths carry the
    ``direction.`` prefix; everything else is the candidate's."""
    texts: dict[str, str] = {
        "title": candidate.title,
        "research_question": candidate.research_question,
        "proposed_contribution": candidate.proposed_contribution,
        "mechanism": candidate.mechanism,
        "hypothesis": candidate.hypothesis,
        "grounding": candidate.grounding,
    }
    for index, prediction in enumerate(candidate.predictions):
        texts[f"predictions[{index}].text"] = prediction.text
        texts[f"predictions[{index}].falsifier"] = prediction.falsifier
    for index, metric in enumerate(candidate.metrics):
        texts[f"metrics[{index}]"] = metric
    texts["evaluation_protocol"] = candidate.evaluation_protocol
    for label, entries in (
        ("baselines", candidate.baselines),
        ("ablations", candidate.ablations),
        ("risks", candidate.risks),
        ("aligned_topics", candidate.aligned_topics),
    ):
        for index, entry in enumerate(entries):
            texts[f"{label}[{index}]"] = entry
    for index, dataset in enumerate(candidate.datasets):
        texts[f"datasets[{index}].name"] = dataset.name
        texts[f"datasets[{index}].role"] = dataset.role
    texts["resources.compute"] = candidate.resources.compute
    texts["resources.data"] = candidate.resources.data
    texts["resources.implementation"] = candidate.resources.implementation
    texts["cfp_alignment"] = candidate.cfp_alignment
    texts["uncertainty"] = candidate.uncertainty
    texts["direction.scope"] = direction.scope
    for index, topic in enumerate(direction.topics):
        texts[f"direction.topics[{index}]"] = topic
    for index, constraint in enumerate(direction.constraints):
        texts[f"direction.constraints[{index}]"] = constraint
    return texts


def render_admission_context(
    candidate_id: str,
    field_texts: Mapping[str, str],
    inputs: AdmissionInputs,
    directive: AdmissionDirective,
) -> str:
    """The one prompt body. Each groundable field appears as
    ``path: text`` — the exact strings the gate re-finds — followed by
    the seven binding statements, so every number the operationalization
    may honestly use is on screen."""
    selection_directive = inputs.selection_directive
    lines = [
        f"## Selected candidate {candidate_id}",
        "Groundable fields (quote them at these exact paths):",
        *(f"{path}: {text}" for path, text in field_texts.items()),
        "",
        "## Binding statements (context, not quotable field paths)",
        f"compute constraint: {selection_directive.compute_constraint}",
        f"data constraint: {selection_directive.data_constraint}",
        f"time constraint: {selection_directive.time_constraint}",
        (
            f"experimental constraint: "
            f"{selection_directive.experimental_constraint}"
        ),
        f"scheduling requirement: {directive.scheduling_requirement}",
        f"job duration requirement: {directive.job_duration_requirement}",
        f"checkpoint requirement: {directive.checkpoint_requirement}",
    ]
    return "\n".join(lines)


def encoded_metric(entry: OperationalPrediction) -> str:
    """The templated metric string — an exact-match contract: a future
    experiment spec must declare it verbatim among its metrics, and the
    executor must report it verbatim as a metrics key."""
    return (
        f"difference in {entry.base_metric}: "
        f"{entry.expected_higher_arm} minus {entry.expected_lower_arm}"
    )


class CandidateAdmitter:
    """Runs the admission pipeline for one directive. The ideation,
    prior-art, and selection stores are read-only inputs; only the
    admission store (its state snapshots included) is written."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        ledger: UsageLedger,
        ideation_store: IdeationStore,
        prior_art_store: PriorArtStore,
        selection_store: SelectionStore,
        store: AdmissionStore,
        max_output_tokens: int = 16384,
        # 16384, the Task 5C lesson: one JSON object over every
        # encoding; the budget has to fit it.
        temperature: float = 0.0,
        request_timeout_seconds: float = 240.0,
        max_corrective_calls: int = 1,
    ) -> None:
        self._provider = provider
        self._model = model
        self._ledger = ledger
        self._ideation_store = ideation_store
        self._prior_art_store = prior_art_store
        self._selection_store = selection_store
        self._store = store
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._request_timeout_seconds = request_timeout_seconds
        self._max_corrective_calls = max_corrective_calls

    def run(self, directive: AdmissionDirective) -> AdmissionRunResult:
        """Execute one admission: replay a completed directive at zero
        calls, refuse a second admission of an admitted selection, or
        run the door, the one gated call, and the all-or-nothing
        write."""
        self._store.record_directive(directive)

        # Replay reads the admission root alone: a completed admission
        # must return its stored result even when the upstream stores
        # are no longer mounted.
        stored = self._store.record_for_directive(directive.id)
        if stored is not None:
            record, state = self._store.get_admitted_state(stored.id)
            return AdmissionRunResult(
                record=record,
                state=state,
                directive=directive,
                inputs=None,
                replayed=True,
            )
        occupied = self._store.record_for_selection_run(
            directive.selection_run_record_id
        )
        if occupied is not None:
            raise AdmissionConflictError(
                f"selection run {directive.selection_run_record_id} is "
                f"already admitted as {occupied.id} under directive "
                f"{occupied.directive_id}; an admitted selection is "
                f"never silently replaced"
            )

        inputs = require_selected_candidate_for_admission(
            self._selection_store,
            self._prior_art_store,
            self._ideation_store,
            directive.selection_run_record_id,
        )
        candidate = inputs.selected
        # Defense in depth: the driver preflights the same wiring before
        # any spend, and no preflighted run can reach the runtime
        # budget error below.
        check_admission_coherence(
            directive=directive,
            prediction_count=len(candidate.predictions),
            max_output_tokens=self._max_output_tokens,
            max_corrective_calls=self._max_corrective_calls,
        )

        run_id = occurrence_id("adm")
        spend = _Spend(directive.max_model_calls)
        field_texts = build_field_texts(candidate, inputs.direction)
        candidate_block = "\n".join(
            text
            for path, text in field_texts.items()
            if not path.startswith("direction.")
        )
        falsifier_by_prediction = {
            _normalized(prediction.text): prediction.falsifier
            for prediction in candidate.predictions
        }
        context = render_admission_context(
            candidate.id, field_texts, inputs, directive
        )
        # Every id the prompt actually shows is quotable; everything
        # else id-shaped is a fabricated reference. Ids are stripped
        # before number extraction because hex ids are full of digits.
        known_ids = frozenset(_REFERENCE.findall(context)) | {candidate.id}
        haystack_tokens = _number_tokens(
            _without_known_ids(context, known_ids)
        )

        def gate(
            payload: Mapping[str, object],
        ) -> tuple[MappingRejection, ...]:
            return check_operationalization(
                payload,
                field_texts=field_texts,
                candidate_block=candidate_block,
                falsifier_by_prediction=falsifier_by_prediction,
                metrics=candidate.metrics,
                haystack_tokens=haystack_tokens,
                known_ids=known_ids,
            )

        payload, provenance = self._gated_call(
            self._request(
                OPERATIONALIZATION_INSTRUCTION,
                context,
                operationalization_schema(candidate.metrics),
                run_id,
            ),
            gate=gate,
            run_id=run_id,
            spend=spend,
        )
        encodings = _parse_operationalizations(payload, candidate)

        question = ResearchQuestion(
            text=candidate.research_question,
            importance=candidate.cfp_alignment,
        )
        hypothesis = Hypothesis(
            statement=candidate.hypothesis,
            rationale=candidate.mechanism,
            question_id=question.id,
        )
        predictions = tuple(
            Prediction(
                hypothesis_id=hypothesis.id,
                condition=entry.condition,
                metric=encoded_metric(entry),
                comparator=Comparator.GREATER_THAN,
                threshold=0.0,
                tolerance=0.0,
                expectation=entry.prediction_text,
            )
            for entry in encodings
        )
        prediction_ids = tuple(entry.id for entry in predictions)
        if len(set(prediction_ids)) != len(prediction_ids):
            # Defense in depth: the gate already rejects duplicate
            # mechanical tuples, and core Prediction identity excludes
            # the prose expectation — a collision here would silently
            # merge two commitments.
            raise AdmissionRejectedError(
                "two encodings derived the same prediction identity; "
                "refusing to admit a state that silently merges them"
            )
        decision = inputs.selection_run.decision
        assert decision is not None  # structural: the door required SELECTED
        state = ResearchState(
            objective=decision.first_experimental_objective,
            questions=(question,),
            hypotheses=(hypothesis,),
            predictions=predictions,
        )
        # The admitted seed holds propositions only — structural here,
        # asserted anyway so no refactoring can quietly widen it.
        assert state.parent_id is None
        assert state.budget.is_exhausted
        assert not state.experiments
        assert not state.results
        assert not state.evidence_ids
        assert not state.prediction_tests
        assert not state.claims
        assert not state.evidence_links
        assert not state.assessments
        assert not state.attempts
        assert not state.history

        inherited, operator = _requirements(inputs, directive)
        self._store.persist_state(state)
        record = self._store.record_admission(
            AdmissionRecord(
                run_id=run_id,
                directive_id=directive.id,
                selection_run_record_id=inputs.selection_run.id,
                selection_run_id=inputs.selection_run.run_id,
                selection_directive_id=inputs.selection_directive.id,
                prior_art_run_record_id=inputs.prior_art_run.id,
                prior_art_run_id=inputs.prior_art_run.run_id,
                selected_prior_art_assessment_id=(
                    inputs.selected_assessment.id
                ),
                ideation_run_record_id=inputs.ideation_run.id,
                ideation_run_id=inputs.ideation_run.run_id,
                direction_id=inputs.direction.id,
                snapshot_id=inputs.snapshot.id,
                map_run_id=inputs.ideation_run.map_run_id,
                map_assessment_id=inputs.ideation_run.assessment_id,
                selected_candidate_id=candidate.id,
                operational_predictions=encodings,
                measurements=candidate.metrics,
                controls=candidate.ablations,
                comparison_targets=candidate.baselines,
                evaluation_protocol=candidate.evaluation_protocol,
                inherited_requirements=inherited,
                operator_requirements=operator,
                mechanical_reading=MECHANICAL_READING,
                question_id=question.id,
                hypothesis_id=hypothesis.id,
                prediction_ids=prediction_ids,
                state_id=state.id,
                provenance=provenance,
                model_calls=spend.calls,
                input_tokens=spend.input_tokens,
                output_tokens=spend.output_tokens,
            )
        )
        return AdmissionRunResult(
            record=record,
            state=state,
            directive=directive,
            inputs=inputs,
            replayed=False,
        )

    # -- the call machinery (the selector's, with admission wording) --------

    def _request(
        self,
        instruction: str,
        content: str,
        schema: OutputSchema,
        run_id: str,
    ) -> ModelRequest:
        return ModelRequest(
            model=self._model,
            instruction=instruction,
            messages=(Message(role=MessageRole.USER, content=content),),
            schema=schema,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            timeout_seconds=self._request_timeout_seconds,
            metadata={
                "admission_run": run_id,
                "stage": "operationalization",
            },
        )

    def _invoke(self, request: ModelRequest, spend: _Spend) -> ModelResponse:
        """One provider call: budget checked before, accounting reaching
        the ledger exactly once — the response on success, the attached
        cost on failure — before any error propagates. The run's own
        spend folds failed-call usage too: a billed schema violation
        stays on the admission record exactly as it stays on the
        ledger."""
        if spend.calls >= spend.limit:
            raise AdmissionBudgetError(
                f"the directive's model-call budget ({spend.limit}) is "
                f"spent; refusing the call that would exceed it"
            )
        spend.calls += 1
        try:
            response = self._provider.invoke(request)
        except ModelProviderError as error:
            if self._ledger.record_failure(error):
                assert error.accounting is not None
                spend.input_tokens += error.accounting.usage.input_tokens
                spend.output_tokens += error.accounting.usage.output_tokens
            raise
        self._ledger.record(response)
        spend.input_tokens += response.usage.input_tokens
        spend.output_tokens += response.usage.output_tokens
        return response

    def _attempt(
        self, request: ModelRequest, spend: _Spend
    ) -> tuple[ModelResponse | None, StructuredOutputError | None]:
        """One call whose schema violation is a correctable outcome, not
        an abort: exactly :class:`StructuredOutputError` is caught (its
        accounting already reached the ledger in ``_invoke``); every
        other provider failure propagates."""
        try:
            return self._invoke(request, spend), None
        except StructuredOutputError as error:
            return None, error

    def _gated_call(
        self,
        request: ModelRequest,
        *,
        gate: Callable[[Mapping[str, object]], tuple[MappingRejection, ...]],
        run_id: str,
        spend: _Spend,
    ) -> tuple[Mapping[str, object], CallProvenance]:
        """One structured call under one deterministic gate, with the
        bounded corrective-call discipline: a schema violation and a
        gate rejection earn the same treatment — the payload is
        preserved, at most ``max_corrective_calls`` retries carry the
        exact rules that fired, and only mechanical rules — never a
        preferred wording — trigger the retry."""
        stage = "operationalization"
        response, schema_error = self._attempt(request, spend)
        repairs = 0
        while True:
            if schema_error is not None:
                payload: Mapping[str, object] | None = None
                rejections: tuple[MappingRejection, ...] = (
                    MappingRejection(
                        "invalid_structured_output",
                        f"the reply violated the output schema: "
                        f"{schema_error}",
                    ),
                )
                fingerprint, response_id = request.fingerprint, ""
                raw: object = str(schema_error)
            else:
                assert response is not None
                payload = response.structured
                if payload is None:
                    rejections = (
                        MappingRejection(
                            "no_structured_payload",
                            "the reply carried no structured payload",
                        ),
                    )
                else:
                    rejections = gate(payload)
                fingerprint = response.request_fingerprint
                response_id = response.id
                raw = response.text
            if not rejections:
                break
            self._store.preserve_rejected(
                run_id=run_id,
                stage=stage,
                reasons=tuple((r.rule, r.detail) for r in rejections),
                request_fingerprint=fingerprint,
                response_id=response_id,
                payload=payload if payload is not None else raw,
                repair=repairs,
            )
            if repairs >= self._max_corrective_calls:
                if schema_error is not None:
                    raise schema_error
                raise AdmissionRejectedError(
                    f"{stage} payload rejected by the deterministic gate: "
                    + "; ".join(
                        f"{r.rule}: {r.detail}" for r in rejections
                    )
                )
            repairs += 1
            request = _repair_request(request, response, rejections, repairs)
            response, schema_error = self._attempt(request, spend)
        assert response is not None
        assert payload is not None  # an absent payload never passes the gate
        return payload, CallProvenance(
            request_fingerprint=response.request_fingerprint,
            response_id=response.id,
            provider=response.provider,
            requested_model=self._model,
            served_model=response.model,
            provider_request_id=response.request_id,
            latency_seconds=response.latency_seconds,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            repair_count=repairs,
        )


def _repair_request(
    base: ModelRequest,
    failed: ModelResponse | None,
    rejections: tuple[MappingRejection, ...],
    attempt: int,
) -> ModelRequest:
    """One corrective request: the failed reply (when one was parseable
    at all) plus every deterministic rule that fired. Mechanical rules
    only — never a preferred wording."""
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    feedback = (
        f"Your output was rejected by the deterministic admission gate. "
        f"Nothing was recorded. The rules that fired:\n{rules}\n"
        f"Return one corrected output now, satisfying every original "
        f"constraint. Quote the shown fields verbatim at their labeled "
        f"paths; drop anything you cannot ground."
    )
    previous = (
        failed.text
        if failed is not None and failed.text
        else "(the previous reply did not satisfy the output schema)"
    )
    return ModelRequest(
        model=base.model,
        instruction=base.instruction,
        messages=(
            *base.messages,
            Message(role=MessageRole.ASSISTANT, content=previous),
            Message(role=MessageRole.USER, content=feedback),
        ),
        schema=base.schema,
        max_output_tokens=base.max_output_tokens,
        temperature=base.temperature,
        timeout_seconds=base.timeout_seconds,
        metadata={**base.metadata, "admission_repair": str(attempt)},
    )


# -- payload parsing (gate-validated input only) --------------------------------


def _parse_operationalizations(
    payload: Mapping[str, object], candidate: CandidateIdea
) -> tuple[OperationalPrediction, ...]:
    """Rebuild the accepted encodings in canonical order: by the record
    order of the predictions they encode, then by payload order. The
    gate already proved coverage, grounding, and the caps."""
    prediction_order = {
        _normalized(prediction.text): index
        for index, prediction in enumerate(candidate.predictions)
    }
    entries: list[tuple[int, int, OperationalPrediction]] = []
    raw_entries = payload.get("operational_predictions")
    assert isinstance(raw_entries, (list, tuple))
    for position, raw in enumerate(raw_entries):
        assert isinstance(raw, Mapping)
        support_raw = raw.get("support")
        assert isinstance(support_raw, (list, tuple))
        support = tuple(
            GroundedSupport(
                source=SupportSource(str(link["source"])),
                field_path=str(link["field_path"]),
                quote=str(link["quote"]),
            )
            for link in support_raw
            if isinstance(link, Mapping)
        )
        entry = OperationalPrediction(
            prediction_text=str(raw["prediction_text"]),
            condition=str(raw["condition"]),
            base_metric=str(raw["base_metric"]),
            expected_higher_arm=str(raw["expected_higher_arm"]),
            expected_lower_arm=str(raw["expected_lower_arm"]),
            contrary_observation=str(raw["contrary_observation"]),
            support=support,
        )
        entries.append(
            (
                prediction_order[_normalized(entry.prediction_text)],
                position,
                entry,
            )
        )
    entries.sort(key=lambda item: (item[0], item[1]))
    return tuple(entry for _, _, entry in entries)


def _requirements(
    inputs: AdmissionInputs, directive: AdmissionDirective
) -> tuple[tuple[Requirement, ...], tuple[Requirement, ...]]:
    """Every execution-capability requirement, split by provenance and
    quoted verbatim by trusted code. The model authors none of them."""
    candidate = inputs.selected
    selection_directive = inputs.selection_directive
    decision = inputs.selection_run.decision
    assert decision is not None
    inherited = [
        Requirement(
            source=RequirementSource.CANDIDATE_RESOURCES,
            record_id=candidate.id,
            field_path=f"resources.{name}",
            quote=quote,
        )
        for name, quote in (
            ("compute", candidate.resources.compute),
            ("data", candidate.resources.data),
            ("implementation", candidate.resources.implementation),
        )
    ]
    inherited.extend(
        Requirement(
            source=RequirementSource.SELECTION_DIRECTIVE,
            record_id=selection_directive.id,
            field_path=name,
            quote=quote,
        )
        for name, quote in (
            ("compute_constraint", selection_directive.compute_constraint),
            ("data_constraint", selection_directive.data_constraint),
            ("time_constraint", selection_directive.time_constraint),
            (
                "experimental_constraint",
                selection_directive.experimental_constraint,
            ),
        )
    )
    inherited.extend(
        Requirement(
            source=RequirementSource.SELECTION_DECISION,
            record_id=inputs.selection_run.id,
            field_path=f"decision.required_capabilities[{index}]",
            quote=quote,
        )
        for index, quote in enumerate(decision.required_capabilities)
    )
    operator = tuple(
        Requirement(
            source=RequirementSource.ADMISSION_DIRECTIVE,
            record_id=directive.id,
            field_path=name,
            quote=quote,
        )
        for name, quote in (
            ("scheduling_requirement", directive.scheduling_requirement),
            (
                "job_duration_requirement",
                directive.job_duration_requirement,
            ),
            ("checkpoint_requirement", directive.checkpoint_requirement),
        )
    )
    return tuple(inherited), operator
