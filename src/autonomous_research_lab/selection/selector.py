"""The candidate selector: two gated model stages over one challenged
portfolio, or no model call at all.

    directive -> door -> trusted partition
        -> zero eligible?  record NO_ELIGIBLE_CANDIDATE, done (0 calls)
        -> preflight -> stage 1: comparative review (every candidate,
           every pair, attested disqualifiers) under its gate
        -> every eligible candidate disqualified?  record
           NO_DEFENSIBLE_CANDIDATE, done (no second call)
        -> stage 2: one winner among the contenders under its gate
        -> record SELECTED

The metadata-screening rationale applies to the stage split: the large
comparative review is separate from the small decision, so a stage-2
rejection never redoes stage 1. Each stage gets at most one corrective
call carrying the exact mechanical rules that fired — never a preferred
verdict. Every rejected payload is preserved before any retry is
decided.

Trusted code owns the eligible set, the disqualified stamp, the outcome,
and the spend; the model owns the judgments and the final preference.
The upstream ideation and prior-art stores are read-only here — only the
selection store is written — and the candidate records come out of a run
byte-identical to how they went in.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ..core.ids import occurrence_id
from ..ideation.records import CandidateIdea
from ..ideation.store import IdeationStore
from ..mapping.records import CallProvenance
from ..priorart.assessment import PriorArtAssessment, PriorArtVerdict
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
from .directive import SelectionDirective
from .eligibility import (
    EligibilityPartition,
    SelectionInputs,
    partition_by_verdict,
    require_challenged_portfolio_for_selection,
)
from .gates import (
    MappingRejection,
    check_comparative_review,
    check_selection_decision,
)
from .preflight import check_selection_coherence
from .records import (
    REVIEW_FIELDS,
    CandidateReview,
    DisqualificationGround,
    DisqualifierDimension,
    HardDisqualifier,
    PairwiseComparison,
    SelectionDecision,
    SelectionOutcome,
    SelectionRationale,
    SelectionRunRecord,
)
from .store import SelectionStore


class SelectionRejectedError(RuntimeError):
    """The gate refused a stage's payload after the bounded corrective
    call. Everything refused is preserved; nothing was recorded."""


class SelectionBudgetError(RuntimeError):
    """A call would exceed the directive's model-call budget. The
    refusal happens before the call, never after."""


# -- output contracts -----------------------------------------------------------

_DISQUALIFIER_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "ground": {
            "type": "string",
            "enum": [ground.value for ground in DisqualificationGround],
        },
        "dimension": {
            "type": "string",
            "enum": [entry.value for entry in DisqualifierDimension],
            "description": (
                "Which constraint the conflict is with; names the text "
                "constraint_text must quote."
            ),
        },
        "candidate_text": {
            "type": "string",
            "description": (
                "A verbatim fragment of the candidate's own rendered "
                "record; the gate re-finds it there."
            ),
        },
        "constraint_text": {
            "type": "string",
            "description": (
                "A verbatim quote of the named constraint; the gate "
                "re-finds it there."
            ),
        },
        "why_unrepairable": {"type": "string"},
    },
    "required": [
        "ground",
        "dimension",
        "candidate_text",
        "constraint_text",
        "why_unrepairable",
    ],
}

COMPARATIVE_REVIEW_SCHEMA: Final = OutputSchema(
    name="selection_comparative_review",
    json_schema={
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "prior_art_verdict": {
                            "type": "string",
                            "enum": [
                                verdict.value
                                for verdict in PriorArtVerdict
                            ],
                        },
                        **{
                            name: {"type": "string"}
                            for name in REVIEW_FIELDS
                        },
                        "disqualifiers": {
                            "type": "array",
                            "items": _DISQUALIFIER_SCHEMA,
                        },
                    },
                    "required": [
                        "candidate_id",
                        "prior_art_verdict",
                        *REVIEW_FIELDS,
                        "disqualifiers",
                    ],
                },
            },
            "pairwise_comparisons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "first_candidate_id": {"type": "string"},
                        "second_candidate_id": {"type": "string"},
                        "comparison": {"type": "string"},
                    },
                    "required": [
                        "first_candidate_id",
                        "second_candidate_id",
                        "comparison",
                    ],
                },
            },
        },
        "required": ["reviews", "pairwise_comparisons"],
    },
)

SELECTION_DECISION_SCHEMA: Final = OutputSchema(
    name="selection_decision",
    json_schema={
        "type": "object",
        "properties": {
            "selected_candidate_id": {"type": "string"},
            "decisive_tradeoff": {"type": "string"},
            "why_selected_over": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["candidate_id", "reason"],
                },
            },
            "first_experimental_objective": {"type": "string"},
            "required_capabilities": {
                "type": "array",
                "items": {"type": "string"},
            },
            "residual_risks": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "selected_candidate_id",
            "decisive_tradeoff",
            "why_selected_over",
            "first_experimental_objective",
            "required_capabilities",
            "residual_risks",
        ],
    },
)
"""No numeric field and no stop shape anywhere in either schema: a score
and a model-authored stop are structurally inexpressible, the way the
engineer schema has nowhere to put a metric."""


# -- instructions ---------------------------------------------------------------

_TEXT_NOTE: Final = (
    " In prose, prefer plain punctuation - simple hyphens and straight "
    "quotes - because decorative dashes are corrupted in transit and the "
    "gate rejects control characters; legitimate Unicode in names, "
    "technical terms, and titles is preserved."
)

COMPARATIVE_REVIEW_INSTRUCTION: Final = (
    "You review ALL listed research candidates together for selection "
    "under one directive. Every candidate shown carries a "
    "'distinguished' verdict recorded by a bounded prior-art challenge: "
    "restate that recorded verdict exactly in prior_art_verdict, and in "
    "prose call it 'distinguished within the challenged corpus' - the "
    "verdict describes one bounded search, so any wording that claims "
    "more is rejected by the gate. Review every candidate exactly once, "
    "covering all ten judgment fields in one or two short sentences "
    "each, and compare EVERY pair of candidates exactly once, "
    "explicitly - a reply that overruns the output budget is lost "
    "entirely, so be brief everywhere. Use only numbers that appear in "
    "the shown records and constraints, and give no scores, rankings, "
    "or percentages of preference - there is nowhere to put one. A hard "
    "disqualifier is NARROW: only the listed grounds qualify, and only "
    "when the candidate's own quoted text conflicts irreparably with "
    "the quoted constraint. Quote both verbatim - the gate re-finds "
    "each in the rendered record and the named constraint - name the "
    "dimension, and say why no repair short of changing the candidate "
    "resolves the conflict. Weakness relative to another candidate, "
    "uncertainty between close candidates, current repository "
    "limitations, and implementation difficulty are never "
    "disqualifiers: put those in the judgment fields. At most one "
    "disqualifier per ground per candidate; most candidates deserve "
    "none."
    + _TEXT_NOTE
)

SELECTION_DECISION_INSTRUCTION: Final = (
    "You choose exactly ONE of the listed candidates to pursue. The "
    "list already excludes every disqualified candidate, and there is "
    "no other outcome to express: name the winner, state the decisive "
    "tradeoff, and explain why the winner beats EACH listed "
    "alternative, one short entry per alternative. Then state the "
    "winner's first experimental objective, the capabilities pursuing "
    "it requires, and the residual risks - short plain entries, no "
    "scores anywhere. Ground yourself in the reviews and comparisons "
    "shown; use only numbers that appear in the shown records and "
    "constraints; in prose say 'distinguished within the challenged "
    "corpus' rather than any stronger wording, and state differences "
    "rather than claiming any candidate is without precedent."
    + _TEXT_NOTE
)


# -- the service ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectionRunResult:
    """Everything one selection run concluded, mirroring what is
    durable: the run record plus the loaded inputs behind it."""

    run_record: SelectionRunRecord
    directive: SelectionDirective
    inputs: SelectionInputs
    partition: EligibilityPartition


class _Spend:
    """Mutable within one run: what the run has consumed so far."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


class CandidateSelector:
    """Runs the selection pipeline for one directive. The ideation and
    prior-art stores are read-only inputs; only the selection store is
    written."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        ledger: UsageLedger,
        ideation_store: IdeationStore,
        prior_art_store: PriorArtStore,
        store: SelectionStore,
        max_output_tokens: int = 16384,
        # 16384, the Task 5C lesson: the comparative review is one large
        # JSON object over several candidates and every pair; the budget
        # has to fit it.
        temperature: float = 0.0,
        request_timeout_seconds: float = 240.0,
        max_corrective_calls: int = 1,
    ) -> None:
        self._provider = provider
        self._model = model
        self._ledger = ledger
        self._ideation_store = ideation_store
        self._prior_art_store = prior_art_store
        self._store = store
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._request_timeout_seconds = request_timeout_seconds
        self._max_corrective_calls = max_corrective_calls

    def run(self, directive: SelectionDirective) -> SelectionRunResult:
        """Execute one selection run: door, partition, and either the
        deterministic ineligible stop or the two gated stages."""
        self._store.record_directive(directive)
        inputs = require_challenged_portfolio_for_selection(
            self._prior_art_store,
            self._ideation_store,
            directive.prior_art_run_record_id,
        )
        partition = partition_by_verdict(inputs)
        run_id = occurrence_id("sel")
        spend = _Spend(directive.max_model_calls)

        if not partition.eligible:
            # Trusted code alone settles an ineligible portfolio: the
            # named run's verdicts are on the record, so the stop needs
            # no judgment, no call, and no spend.
            record = self._record(
                run_id,
                directive,
                inputs,
                partition,
                reviews=(),
                pairs=(),
                review_provenance=None,
                outcome=SelectionOutcome.NO_ELIGIBLE_CANDIDATE,
                decision=None,
                spend=spend,
            )
            return SelectionRunResult(record, directive, inputs, partition)

        # Defense in depth: the driver preflights the same wiring before
        # any spend, and no preflighted run can reach the runtime
        # budget error below.
        check_selection_coherence(
            directive=directive,
            eligible_count=len(partition.eligible),
            max_output_tokens=self._max_output_tokens,
            max_corrective_calls=self._max_corrective_calls,
        )

        candidate_blocks = {
            candidate.id: render_candidate_for_selection(candidate)
            for candidate in partition.eligible
        }
        assessment_blocks = tuple(
            render_assessment_for_selection(assessment)
            for assessment in partition.eligible_assessments
        )
        constraint_haystacks = {
            DisqualifierDimension.COMPUTE.value: (
                directive.compute_constraint
            ),
            DisqualifierDimension.DATA.value: directive.data_constraint,
            DisqualifierDimension.TIME.value: directive.time_constraint,
            DisqualifierDimension.EXPERIMENTAL.value: (
                directive.experimental_constraint
            ),
            DisqualifierDimension.SCOPE.value: (
                inputs.direction.rendered_text()
            ),
        }
        known_ids = frozenset(candidate_blocks) | {
            problem.key
            for candidate in partition.eligible
            for problem in candidate.addressed_problems
        } | {
            theme.key
            for candidate in partition.eligible
            for theme in candidate.targeted_themes
        }
        rendered_context = "\n".join(
            (
                *candidate_blocks.values(),
                *assessment_blocks,
                render_constraints(directive),
                inputs.direction.rendered_text(),
            )
        )
        haystack_tokens = _number_tokens(
            _without_known_ids(rendered_context, known_ids)
        )

        def review_gate(
            payload: Mapping[str, object],
        ) -> tuple[MappingRejection, ...]:
            return check_comparative_review(
                payload,
                candidate_blocks=candidate_blocks,
                assessment_verdicts={
                    assessment.candidate_id: assessment.verdict.value
                    for assessment in partition.eligible_assessments
                },
                constraint_haystacks=constraint_haystacks,
                haystack_tokens=haystack_tokens,
                known_ids=known_ids,
            )

        review_payload, review_provenance = self._gated_call(
            self._request(
                COMPARATIVE_REVIEW_INSTRUCTION,
                render_review_context(
                    tuple(candidate_blocks.values()),
                    assessment_blocks,
                    directive,
                    inputs.direction.rendered_text(),
                ),
                COMPARATIVE_REVIEW_SCHEMA,
                run_id,
                "comparative_review",
            ),
            gate=review_gate,
            stage="comparative_review",
            run_id=run_id,
            spend=spend,
        )
        eligible_order = tuple(
            candidate.id for candidate in partition.eligible
        )
        reviews = _parse_reviews(review_payload, eligible_order)
        pairs = _parse_pairs(review_payload)
        disqualified = tuple(
            review.candidate_id
            for review in reviews
            if review.disqualifiers
        )

        if set(disqualified) == set(eligible_order):
            # Every eligible candidate carries a validated disqualifier:
            # trusted code concludes the honest stop, and no decision
            # call is spent asking the model to choose among nothing.
            record = self._record(
                run_id,
                directive,
                inputs,
                partition,
                reviews=reviews,
                pairs=pairs,
                review_provenance=review_provenance,
                outcome=SelectionOutcome.NO_DEFENSIBLE_CANDIDATE,
                decision=None,
                spend=spend,
            )
            return SelectionRunResult(record, directive, inputs, partition)

        contender_ids = tuple(
            candidate_id
            for candidate_id in eligible_order
            if candidate_id not in set(disqualified)
        )

        def decision_gate(
            payload: Mapping[str, object],
        ) -> tuple[MappingRejection, ...]:
            return check_selection_decision(
                payload,
                eligible_ids=frozenset(eligible_order),
                disqualified_ids=frozenset(disqualified),
                haystack_tokens=haystack_tokens,
                known_ids=known_ids,
            )

        decision_payload, decision_provenance = self._gated_call(
            self._request(
                SELECTION_DECISION_INSTRUCTION,
                render_decision_context(
                    tuple(
                        candidate_blocks[candidate_id]
                        for candidate_id in contender_ids
                    ),
                    contender_ids,
                    reviews,
                    pairs,
                    directive,
                    disqualified,
                ),
                SELECTION_DECISION_SCHEMA,
                run_id,
                "selection_decision",
            ),
            gate=decision_gate,
            stage="selection_decision",
            run_id=run_id,
            spend=spend,
        )
        decision = _parse_decision(decision_payload, decision_provenance)
        record = self._record(
            run_id,
            directive,
            inputs,
            partition,
            reviews=reviews,
            pairs=pairs,
            review_provenance=review_provenance,
            outcome=SelectionOutcome.SELECTED,
            decision=decision,
            spend=spend,
        )
        return SelectionRunResult(record, directive, inputs, partition)

    # -- record assembly -------------------------------------------------------

    def _record(
        self,
        run_id: str,
        directive: SelectionDirective,
        inputs: SelectionInputs,
        partition: EligibilityPartition,
        *,
        reviews: tuple[CandidateReview, ...],
        pairs: tuple[PairwiseComparison, ...],
        review_provenance: CallProvenance | None,
        outcome: SelectionOutcome,
        decision: SelectionDecision | None,
        spend: _Spend,
    ) -> SelectionRunRecord:
        record = SelectionRunRecord(
            run_id=run_id,
            directive_id=directive.id,
            prior_art_run_record_id=inputs.prior_art_run.id,
            prior_art_run_id=inputs.prior_art_run.run_id,
            ideation_run_record_id=inputs.prior_art_run.ideation_run_record_id,
            ideation_run_id=inputs.prior_art_run.ideation_run_id,
            direction_id=inputs.direction.id,
            candidate_ids=inputs.prior_art_run.candidate_ids,
            prior_art_assessment_ids=(
                inputs.prior_art_run.prior_art_assessment_ids
            ),
            eligible_candidate_ids=tuple(
                candidate.id for candidate in partition.eligible
            ),
            ineligible=partition.ineligible,
            disqualified_candidate_ids=tuple(
                review.candidate_id
                for review in reviews
                if review.disqualifiers
            ),
            reviews=reviews,
            pairwise_comparisons=pairs,
            review_provenance=review_provenance,
            outcome=outcome,
            decision=decision,
            model_calls=spend.calls,
            input_tokens=spend.input_tokens,
            output_tokens=spend.output_tokens,
        )
        return self._store.record_run(record)

    # -- provider plumbing -------------------------------------------------------

    def _request(
        self,
        instruction: str,
        content: str,
        schema: OutputSchema,
        run_id: str,
        stage: str,
    ) -> ModelRequest:
        return ModelRequest(
            model=self._model,
            instruction=instruction,
            messages=(Message(role=MessageRole.USER, content=content),),
            schema=schema,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            timeout_seconds=self._request_timeout_seconds,
            metadata={"selection_run": run_id, "stage": stage},
        )

    def _invoke(self, request: ModelRequest, spend: _Spend) -> ModelResponse:
        """One provider call: budget checked before, accounting reaching
        the ledger exactly once — the response on success, the attached
        cost on failure — before any error propagates. The run's own
        spend folds failed-call usage too: a billed schema violation
        stays on the run record exactly as it stays on the ledger."""
        if spend.calls >= spend.limit:
            raise SelectionBudgetError(
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
        stage: str,
        run_id: str,
        spend: _Spend,
    ) -> tuple[Mapping[str, object], CallProvenance]:
        """One structured call under one deterministic gate, with the
        bounded corrective-call discipline: a schema violation and a
        gate rejection earn the same treatment — the payload is
        preserved, at most ``max_corrective_calls`` retries carry the
        exact rules that fired, and only mechanical rules — never a
        preferred candidate — trigger the retry."""
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
                raise SelectionRejectedError(
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
    only — never a preferred candidate."""
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    feedback = (
        f"Your output was rejected by the deterministic selection gate. "
        f"Nothing was recorded. The rules that fired:\n{rules}\n"
        f"Return one corrected output now, satisfying every original "
        f"constraint. Judge only the listed candidates; quote their "
        f"rendered records and the stated constraints verbatim; drop "
        f"anything you cannot ground."
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
        metadata={**base.metadata, "selection_repair": str(attempt)},
    )


# -- payload parsing (gate-validated input only) ---------------------------------


def _parse_reviews(
    payload: Mapping[str, object], eligible_order: tuple[str, ...]
) -> tuple[CandidateReview, ...]:
    """Rebuild the accepted reviews in eligible (record) order. The gate
    already proved exactly one review per eligible candidate."""
    entries: dict[str, Mapping[str, object]] = {}
    for raw in _sequence(payload, "reviews"):
        assert isinstance(raw, Mapping)
        entries[str(raw["candidate_id"])] = raw
    reviews = []
    for candidate_id in eligible_order:
        entry = entries[candidate_id]
        fields = {name: str(entry[name]) for name in REVIEW_FIELDS}
        disqualifiers = []
        for raw in _sequence(entry, "disqualifiers"):
            assert isinstance(raw, Mapping)
            disqualifiers.append(
                HardDisqualifier(
                    ground=DisqualificationGround(str(raw["ground"])),
                    dimension=DisqualifierDimension(str(raw["dimension"])),
                    candidate_text=str(raw["candidate_text"]),
                    constraint_text=str(raw["constraint_text"]),
                    why_unrepairable=str(raw["why_unrepairable"]),
                )
            )
        reviews.append(
            CandidateReview(
                candidate_id=candidate_id,
                prior_art_verdict=PriorArtVerdict(
                    str(entry["prior_art_verdict"])
                ),
                disqualifiers=tuple(disqualifiers),
                **fields,
            )
        )
    return tuple(reviews)


def _parse_pairs(
    payload: Mapping[str, object],
) -> tuple[PairwiseComparison, ...]:
    """Rebuild the accepted pairs in canonical order: trusted code
    stamps the id order within each pair and sorts the pairs."""
    pairs = []
    for raw in _sequence(payload, "pairwise_comparisons"):
        assert isinstance(raw, Mapping)
        first = str(raw["first_candidate_id"])
        second = str(raw["second_candidate_id"])
        pairs.append(
            PairwiseComparison(
                first_candidate_id=min(first, second),
                second_candidate_id=max(first, second),
                comparison=str(raw["comparison"]),
            )
        )
    return tuple(
        sorted(
            pairs,
            key=lambda pair: (
                pair.first_candidate_id,
                pair.second_candidate_id,
            ),
        )
    )


def _parse_decision(
    payload: Mapping[str, object], provenance: CallProvenance
) -> SelectionDecision:
    rationales = []
    for raw in _sequence(payload, "why_selected_over"):
        assert isinstance(raw, Mapping)
        rationales.append(
            SelectionRationale(
                candidate_id=str(raw["candidate_id"]),
                reason=str(raw["reason"]),
            )
        )
    return SelectionDecision(
        selected_candidate_id=str(payload["selected_candidate_id"]),
        decisive_tradeoff=str(payload["decisive_tradeoff"]),
        why_selected_over=tuple(
            sorted(rationales, key=lambda entry: entry.candidate_id)
        ),
        first_experimental_objective=str(
            payload["first_experimental_objective"]
        ),
        required_capabilities=_string_items(
            payload, "required_capabilities"
        ),
        residual_risks=_string_items(payload, "residual_risks"),
        provenance=provenance,
    )


def _sequence(payload: Mapping[str, object], key: str) -> tuple[object, ...]:
    """A validated payload arrives deep-frozen (arrays as tuples); this
    reads either shape and refuses anything else."""
    value = payload.get(key, ())
    assert isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    )
    return tuple(value)


def _string_items(
    payload: Mapping[str, object], key: str
) -> tuple[str, ...]:
    return tuple(str(entry) for entry in _sequence(payload, key))


# -- rendering ------------------------------------------------------------------


def render_candidate_for_selection(candidate: CandidateIdea) -> str:
    """The one candidate block both stages render and the gate re-finds
    quotes in: the full structured record — resources, risks, and CFP
    alignment included, because a resource disqualifier must be able to
    quote the resource text it objects to. The prompt block and the
    attestation haystack are the same string, so the gate can never
    demand a quote the model was not shown."""
    lines = [
        f"## Candidate {candidate.id}",
        f"title: {candidate.title}",
        f"research question: {candidate.research_question}",
        f"proposed contribution: {candidate.proposed_contribution}",
        f"mechanism: {candidate.mechanism}",
        f"hypothesis: {candidate.hypothesis}",
        f"grounding: {candidate.grounding}",
        "predictions:",
        *(
            f"- {p.text} (falsifier: {p.falsifier})"
            for p in candidate.predictions
        ),
        "datasets:",
        *(
            f"- {d.name} ({d.status.value}): {d.role}"
            for d in candidate.datasets
        ),
        f"metrics: {'; '.join(candidate.metrics)}",
        f"evaluation protocol: {candidate.evaluation_protocol}",
        f"baselines: {'; '.join(candidate.baselines)}",
        f"ablations: {'; '.join(candidate.ablations)}",
        f"resources: compute: {candidate.resources.compute}; data: "
        f"{candidate.resources.data}; implementation: "
        f"{candidate.resources.implementation}",
        f"risks: {'; '.join(candidate.risks)}",
        f"cfp alignment: {candidate.cfp_alignment}",
        f"aligned topics: {'; '.join(candidate.aligned_topics)}",
        "addressed problems:",
        *(
            f"- {p.key}: {p.statement} ({p.kind.value}, {p.tier.value})"
            for p in candidate.addressed_problems
        ),
        "targeted themes:",
        *(
            f"- {t.key}: {t.name} ({t.era.value})"
            for t in candidate.targeted_themes
        ),
        f"cited sources: {candidate.cited_recent} recent, "
        f"{candidate.cited_foundational} foundational, "
        f"{candidate.cited_undated} undated",
        f"uncertainty: {candidate.uncertainty}",
    ]
    return "\n".join(lines)


def render_assessment_for_selection(assessment: PriorArtAssessment) -> str:
    """The trusted account of one eligible candidate's challenge: the
    recorded verdict and the coverage that earned it. Deliberately free
    of every banned phrase, so an honest reply quoting this block never
    trips the language gates."""
    coverage = assessment.coverage
    return "\n".join(
        (
            f"## Prior-art verdict for {assessment.candidate_id}",
            f"verdict: {assessment.verdict.value} (within the challenged "
            f"corpus of this one bounded run; never a statement about "
            f"the world's literature)",
            f"compared works: {coverage.compared_works}",
            f"unique sources screened: {coverage.screened} of "
            f"{coverage.unique_sources}",
            f"query families executed: "
            f"{len(coverage.families_executed)}",
        )
    )


def render_constraints(directive: SelectionDirective) -> str:
    """The directive's four resource statements, labeled by the
    dimension a disqualifier would name. Rendered verbatim: the gate
    re-finds constraint quotes in exactly this text."""
    return "\n".join(
        (
            "## Directive constraints (what is actually available)",
            f"compute: {directive.compute_constraint}",
            f"data: {directive.data_constraint}",
            f"time: {directive.time_constraint}",
            f"experimental: {directive.experimental_constraint}",
        )
    )


def render_review_context(
    candidate_blocks: Sequence[str],
    assessment_blocks: Sequence[str],
    directive: SelectionDirective,
    direction_text: str,
) -> str:
    return "\n\n".join(
        (
            *candidate_blocks,
            *assessment_blocks,
            render_constraints(directive),
            "## Governing call direction (the scope constraint)\n"
            + direction_text,
            "Review every candidate above and compare every pair "
            "exactly once, as schema JSON.",
        )
    )


def render_decision_context(
    contender_blocks: Sequence[str],
    contender_ids: Sequence[str],
    reviews: Sequence[CandidateReview],
    pairs: Sequence[PairwiseComparison],
    directive: SelectionDirective,
    disqualified: Sequence[str],
) -> str:
    review_lines = []
    for review in reviews:
        if review.candidate_id not in contender_ids:
            continue
        review_lines.append(f"### Review of {review.candidate_id}")
        review_lines.extend(
            f"{name}: {getattr(review, name)}" for name in REVIEW_FIELDS
        )
    pair_lines = [
        f"- {pair.first_candidate_id} vs {pair.second_candidate_id}: "
        f"{pair.comparison}"
        for pair in pairs
        if pair.first_candidate_id in contender_ids
        and pair.second_candidate_id in contender_ids
    ]
    settled = (
        "## Already settled (not offered)\n"
        + "\n".join(
            f"- {candidate_id}: a validated disqualifier removed it"
            for candidate_id in disqualified
        )
        if disqualified
        else ""
    )
    sections = [
        *contender_blocks,
        "## Accepted comparative reviews\n" + "\n".join(review_lines),
    ]
    if pair_lines:
        sections.append(
            "## Accepted pairwise comparisons\n" + "\n".join(pair_lines)
        )
    if settled:
        sections.append(settled)
    sections.append(render_constraints(directive))
    sections.append(
        "Choose exactly one candidate from the listed contenders, as "
        "schema JSON."
    )
    return "\n\n".join(sections)


# Mirrors of the mapping gates' private number helpers, kept private here
# for the same reason.
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")


def _number_tokens(text: str) -> frozenset[str]:
    return frozenset(_NUMBER.findall(text))


def _without_known_ids(text: str, known_ids: frozenset[str]) -> str:
    cleaned = text
    for identifier in sorted(known_ids, key=len, reverse=True):
        cleaned = cleaned.replace(identifier, " ")
    return cleaned
