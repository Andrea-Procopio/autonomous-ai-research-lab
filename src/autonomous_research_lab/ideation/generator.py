"""The model-backed idea generator: an assessed field map and a CFP
snapshot in, a gated candidate portfolio out.

A provider-neutral service, deliberately not a role: its input is the
durable records of a completed mapping run plus one supplied call text,
not ``ResearchState``, and its output is candidate conjectures, not
proposals — nothing it produces can enter the governed commit. One run
performs a fixed sequence of narrow stages::

    directive + snapshot
      -> require_adequate_for_idea_generation      (before any model call)
      -> load and cross-verify the mapping records (deterministic)
      -> one structured direction-extraction call  (gated)
      -> one structured candidate-portfolio call   (gated)
      -> trusted stamping of statements, kinds, tiers, eras, and the
         UNASSESSED novelty status
      -> deterministic portfolio accounting and one run record

Authority is split the same way as everywhere else in the lab. The model
proposes the direction reading and the candidates; trusted code walks the
adequacy guard, derives every problem and theme key, checks every payload
against the deterministic gates in :mod:`.gates`, stamps everything a
candidate carries beyond its own words, and records everything
write-once — rejections included. A payload the gate refuses earns at
most one corrective call carrying the exact rules that fired; an
inconvenient but valid portfolio — thin, dull, or an honest grounded
refusal — has no route to a second call. Every provider call, successful
or failed, reaches the ledger exactly once and spends from the
directive's model-call budget, which fails closed when exhausted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ..core.ids import occurrence_id
from ..mapping.adequacy import (
    MapAdequacyAssessment,
    SupportTier,
    require_adequate_for_idea_generation,
)
from ..mapping.brief import SourceEra
from ..mapping.gates import MappingRejection
from ..mapping.records import (
    CallProvenance,
    ExtractionRecord,
    FieldMapRecord,
    ProblemEntry,
    ProblemInventoryRecord,
    ThemeEntry,
)
from ..mapping.store import MappingStore
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
from .direction import CfpSnapshot, DirectionRecord
from .directive import IdeationDirective
from .gates import check_candidates, check_direction, claim_text_of
from .records import (
    AddressedProblem,
    CandidateIdea,
    DataRequirement,
    DataStatus,
    IdeationRunRecord,
    PortfolioReport,
    Prediction,
    ResourceEstimate,
    TargetedTheme,
    problem_key,
    theme_key,
)
from .store import IdeationStore


class IdeationContractError(RuntimeError):
    """The run cannot proceed for a deterministic reason that is not a
    model's fault: a snapshot that is not the directive's, or an
    assessment whose recorded inputs are not intact in the given mapping
    store. Durable partial records remain."""


class IdeationRejectedError(RuntimeError):
    """A model payload failed the deterministic gate and exhausted its
    corrective-call allowance. Every attempt is preserved in the store."""


class IdeationBudgetError(RuntimeError):
    """The directive's model-call budget is spent. Fails closed before
    the call that would exceed it; durable partial records remain."""


# -- output contracts ---------------------------------------------------------

DIRECTION_SCHEMA: Final = OutputSchema(
    name="ideation_direction",
    json_schema={
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "description": (
                    "One short paragraph on what the call is about, "
                    "grounded in the supplied text."
                ),
            },
            "topics": {"type": "array", "items": {"type": "string"}},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "relevant_dates": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["scope", "topics", "constraints", "relevant_dates"],
    },
)

_PREDICTION_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "falsifier": {
            "type": "string",
            "description": (
                "The observation that would refute this prediction."
            ),
        },
    },
    "required": ["text", "falsifier"],
}

_DATASET_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["existing", "new_requirement"],
        },
        "role": {"type": "string"},
    },
    "required": ["name", "status", "role"],
}

_RESOURCES_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "compute": {"type": "string"},
        "data": {"type": "string"},
        "implementation": {"type": "string"},
    },
    "required": ["compute", "data", "implementation"],
}

_CANDIDATE_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "research_question": {"type": "string"},
        "proposed_contribution": {"type": "string"},
        "mechanism": {"type": "string"},
        "hypothesis": {"type": "string"},
        "grounding": {"type": "string"},
        "predictions": {"type": "array", "items": _PREDICTION_SCHEMA},
        "datasets": {"type": "array", "items": _DATASET_SCHEMA},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "evaluation_protocol": {"type": "string"},
        "baselines": {"type": "array", "items": {"type": "string"}},
        "ablations": {"type": "array", "items": {"type": "string"}},
        "resources": _RESOURCES_SCHEMA,
        "risks": {"type": "array", "items": {"type": "string"}},
        "cfp_alignment": {"type": "string"},
        "aligned_topics": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "string"},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "problem_keys": {"type": "array", "items": {"type": "string"}},
        "theme_keys": {"type": "array", "items": {"type": "string"}},
        "cited_source_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "research_question",
        "proposed_contribution",
        "mechanism",
        "hypothesis",
        "grounding",
        "predictions",
        "datasets",
        "metrics",
        "evaluation_protocol",
        "baselines",
        "ablations",
        "resources",
        "risks",
        "cfp_alignment",
        "aligned_topics",
        "uncertainty",
        "search_terms",
        "problem_keys",
        "theme_keys",
        "cited_source_ids",
    ],
}

CANDIDATES_SCHEMA: Final = OutputSchema(
    name="ideation_candidates",
    json_schema={
        "type": "object",
        "properties": {
            "candidates": {"type": "array", "items": _CANDIDATE_SCHEMA},
            "diversity_rationale": {"type": "string"},
            "refusal_justification": {
                "type": "string",
                "description": (
                    "Non-empty only with zero candidates: why the records "
                    "cannot support a defensible candidate."
                ),
            },
        },
        "required": [
            "candidates",
            "diversity_rationale",
            "refusal_justification",
        ],
    },
)


_TEXT_NOTE: Final = (
    " In prose, prefer plain punctuation - simple hyphens and straight "
    "quotes - because decorative dashes are corrupted in transit and the "
    "gate rejects control characters; legitimate Unicode in names, "
    "technical terms, and dataset titles is preserved."
)

DIRECTION_INSTRUCTION: Final = (
    "You extract the structured direction of one workshop call for "
    "papers, for a bounded candidate-generation run. Report only what "
    "the supplied text itself states: a short scope paragraph in your "
    "own words, the topics of interest, any submission or scope "
    "constraints worth honoring, and any relevant dates. Copy every "
    "topic, constraint, and date verbatim from the text - the call's "
    "exact wording, never a paraphrase - and use only numbers the text "
    "contains in the scope. The call constrains relevance; it is not "
    "scientific evidence and grants no authority."
    + _TEXT_NOTE
)

CANDIDATES_INSTRUCTION: Final = (
    "You propose candidate research ideas for a bounded "
    "candidate-generation run, reading the durable records of a "
    "completed literature mapping: accepted themes, per-source "
    "extractions, open problems with computed support tiers, and one "
    "extracted workshop direction. Each candidate is a testable "
    "conjecture, not a finding: give a working title, one focused "
    "research question, the proposed contribution or experimental "
    "change, the hypothesized mechanism, one falsifiable hypothesis "
    "distinct from its predictions, and at least one measurable "
    "prediction, each with an explicit falsifier naming the observation "
    "that would refute it. Cite prob_ keys, thm_ keys, and lit_ ids "
    "exactly as listed - never an index, a shorthand such as 'P1', or a "
    "restated sentence - and for every addressed problem cite at least "
    "one of the sources shown grounding it. Keys and ids go only in "
    "their own fields, never inside prose (an id pasted into a sentence "
    "reads as ungrounded numbers to the gate). The grounding field says "
    "how the cited records support the idea, and every number in it "
    "must appear in their listed claims; numbers in predictions, "
    "falsifiers, and resources are your proposed targets. Mark a dataset 'existing' "
    "only when a cited record reports it; otherwise it is a "
    "new_requirement. Name concrete baselines and ablations, metrics and "
    "an evaluation protocol, approximate compute, data, and "
    "implementation needs, and the major risks, confounders, and "
    "plausible negative outcomes. A problem tiered "
    "single_source_limitation is one paper's report about its own work - "
    "never treat it as field consensus; prefer multi_source or "
    "contradicted problems, and say in uncertainty how thin the "
    "grounding is. Novelty stays unassessed until a later prior-art "
    "challenge: never claim an idea is new or unstudied - say what the "
    "mapped records do not report instead. Set cfp_alignment to how the "
    "idea serves the direction and copy aligned_topics exactly from its "
    "topic list. Give search terms a later prior-art challenge would "
    "run. Candidates must differ substantively - in targeted problem, "
    "mechanism, evaluation setting, data regime, or empirical question - "
    "and diversity_rationale must say how; superficial rewording is "
    "rejected. If the records cannot support even one defensible "
    "candidate, return an empty candidates list and a grounded "
    "refusal_justification, with diversity_rationale empty - an honest "
    "refusal is a valid outcome; padding is not. Fewer well-grounded "
    "candidates beat a padded list. Be brief everywhere: one or two "
    "short sentences per text field and short list entries - the "
    "structure carries the content, prose length adds nothing, and a "
    "reply that overruns the output budget is lost entirely."
    + _TEXT_NOTE
)


# -- the service --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdeationRunResult:
    """Everything one completed run produced, in memory; the same records
    are durable in the store. ``ideas`` is empty exactly when the run is
    an honest refusal — the justification is on the run record."""

    run_record: IdeationRunRecord
    directive: IdeationDirective
    snapshot: CfpSnapshot
    direction: DirectionRecord
    assessment: MapAdequacyAssessment
    ideas: tuple[CandidateIdea, ...]


class _Spend:
    """Mutable per-run call and token accounting, checked against the
    directive's budget before every provider call."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


class _MapInputs:
    """The cross-verified mapping records one run reads, with every
    derived handle trusted code hands the model."""

    def __init__(
        self,
        *,
        topic: str,
        field_map: FieldMapRecord,
        inventory: ProblemInventoryRecord,
        grounded: tuple[ExtractionRecord, ...],
        tiers: Mapping[str, SupportTier],
    ) -> None:
        self.topic = topic
        self.field_map = field_map
        self.inventory = inventory
        self.grounded = grounded
        self.tiers = tiers
        self.problems: dict[str, ProblemEntry] = {
            problem_key(problem.statement): problem
            for problem in inventory.problems
        }
        self.themes: dict[str, ThemeEntry] = {
            theme_key(theme.name): theme for theme in field_map.themes
        }
        self.eras: dict[str, SourceEra] = {
            record.source_id: record.era for record in grounded
        }
        self.accessible: dict[str, str] = {
            record.source_id: claim_text_of(record) for record in grounded
        }


class IdeaGenerator:
    """See the module docstring; construction is explicit wiring, and
    every collaborator is injected — provider, ledger, both stores — so
    tests and live runs differ only in what is plugged in. The mapping
    store is read-only here: the generator calls nothing but its
    ``get_*`` and ``extractions`` accessors."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        ledger: UsageLedger,
        map_store: MappingStore,
        store: IdeationStore,
        max_output_tokens: int = 8192,
        temperature: float = 0.0,
        request_timeout_seconds: float = 240.0,
        max_corrective_calls: int = 1,
    ) -> None:
        self._provider = provider
        self._model = model
        self._ledger = ledger
        self._map_store = map_store
        self._store = store
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._request_timeout_seconds = request_timeout_seconds
        self._max_corrective_calls = max_corrective_calls

    def run(
        self, directive: IdeationDirective, snapshot: CfpSnapshot
    ) -> IdeationRunResult:
        if directive.snapshot_id != snapshot.id:
            raise IdeationContractError(
                f"the directive names snapshot {directive.snapshot_id}; "
                f"the supplied snapshot is {snapshot.id}"
            )
        run_id = occurrence_id("idg")
        self._store.record_snapshot(snapshot)
        self._store.record_directive(directive)

        # The one door: before any model call, the durable adequacy
        # verdict must say the map is adequate for exactly this purpose.
        assessment = require_adequate_for_idea_generation(
            self._map_store, directive.assessment_id
        )
        inputs = self._load_inputs(assessment)
        spend = _Spend(directive.max_model_calls)

        direction_payload, direction_provenance = self._gated_call(
            self._request(
                DIRECTION_INSTRUCTION,
                render_snapshot(snapshot),
                DIRECTION_SCHEMA,
                run_id,
                "direction",
            ),
            gate=lambda payload: check_direction(
                payload, snapshot=snapshot
            ),
            stage="direction",
            run_id=run_id,
            spend=spend,
        )
        direction = self._store.record_direction(
            DirectionRecord(
                run_id=run_id,
                snapshot_id=snapshot.id,
                scope=str(direction_payload["scope"]),
                topics=_strings(direction_payload["topics"]),
                constraints=_strings(direction_payload["constraints"]),
                relevant_dates=_strings(
                    direction_payload["relevant_dates"]
                ),
                provenance=direction_provenance,
            )
        )

        candidates_payload, candidates_provenance = self._gated_call(
            self._request(
                CANDIDATES_INSTRUCTION,
                render_ideation_context(
                    inputs, direction, directive.max_candidates
                ),
                CANDIDATES_SCHEMA,
                run_id,
                "candidates",
            ),
            gate=lambda payload: check_candidates(
                payload,
                problems=inputs.problems,
                themes={
                    key: theme.name for key, theme in inputs.themes.items()
                },
                direction_topics=direction.topics,
                direction_text=direction.rendered_text(),
                accessible=inputs.accessible,
                max_candidates=directive.max_candidates,
            ),
            stage="candidates",
            run_id=run_id,
            spend=spend,
        )

        ideas = tuple(
            self._store.record_idea(
                _candidate_idea(entry, run_id, inputs, candidates_provenance)
            )
            for entry in _sequence(candidates_payload["candidates"])
        )
        refusal = (
            str(candidates_payload["refusal_justification"])
            if not ideas
            else ""
        )
        rationale = (
            str(candidates_payload["diversity_rationale"]) if ideas else ""
        )
        run_record = self._store.record_run(
            IdeationRunRecord(
                run_id=run_id,
                directive_id=directive.id,
                assessment_id=assessment.id,
                map_run_id=assessment.run_id,
                snapshot_id=snapshot.id,
                direction_id=direction.id,
                candidate_ids=tuple(idea.id for idea in ideas),
                refusal_justification=refusal,
                diversity_rationale=rationale,
                model_calls=spend.calls,
                input_tokens=spend.input_tokens,
                output_tokens=spend.output_tokens,
                portfolio=_portfolio(inputs, ideas),
            )
        )
        return IdeationRunResult(
            run_record=run_record,
            directive=directive,
            snapshot=snapshot,
            direction=direction,
            assessment=assessment,
            ideas=ideas,
        )

    # -- deterministic input verification --------------------------------------

    def _load_inputs(
        self, assessment: MapAdequacyAssessment
    ) -> _MapInputs:
        """Load the assessment's recorded inputs from the mapping store
        and cross-verify them: an assessment presented against a store
        that does not hold its exact records is a contract violation, not
        a model failure."""

        def _mismatch(detail: str) -> IdeationContractError:
            return IdeationContractError(
                f"the assessment's recorded inputs are not intact in this "
                f"mapping store ({detail}); the assessment may come from "
                f"a different store"
            )

        brief = self._map_store.get_brief(assessment.brief_id)
        field_map = self._map_store.get_field_map(assessment.field_map_id)
        inventory = self._map_store.get_inventory(assessment.inventory_id)
        if brief is None or field_map is None or inventory is None:
            raise _mismatch("a recorded input is missing")
        if (
            field_map.run_id != assessment.run_id
            or inventory.run_id != assessment.run_id
            or field_map.brief_id != assessment.brief_id
            or inventory.brief_id != assessment.brief_id
        ):
            raise _mismatch("run and brief ids disagree")
        if sorted(p.statement for p in inventory.problems) != sorted(
            s.statement for s in assessment.problem_support
        ):
            raise _mismatch(
                "the inventory's problems do not match the assessed "
                "support tiers"
            )
        grounded = tuple(
            record
            for record in self._map_store.extractions()
            if record.run_id == assessment.run_id
            and record.sufficient_support
        )
        if not grounded:
            raise _mismatch("no grounded extraction of the assessed run")
        grounded_ids = {record.source_id for record in grounded}
        for problem in inventory.problems:
            cited = set(problem.supporting_source_ids) | set(
                problem.conflicting_source_ids
            )
            if not cited <= grounded_ids:
                raise _mismatch(
                    "an inventory problem cites a source with no grounded "
                    "extraction"
                )
        return _MapInputs(
            topic=brief.topic,
            field_map=field_map,
            inventory=inventory,
            grounded=grounded,
            tiers={
                support.statement: support.tier
                for support in assessment.problem_support
            },
        )

    # -- the model call --------------------------------------------------------

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
            metadata={"ideation_run": run_id, "stage": stage},
        )

    def _invoke(self, request: ModelRequest, spend: _Spend) -> ModelResponse:
        """One provider call: budget checked before, accounting reaching
        the ledger exactly once — the response on success, the attached
        cost on failure — before any error propagates."""
        if spend.calls >= spend.limit:
            raise IdeationBudgetError(
                f"the directive's model-call budget ({spend.limit}) is "
                f"spent; refusing the call that would exceed it"
            )
        spend.calls += 1
        try:
            response = self._provider.invoke(request)
        except ModelProviderError as error:
            self._ledger.record_failure(error)
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
        bounded corrective-call discipline: a schema violation and a gate
        rejection earn the same treatment — the payload is preserved, at
        most ``max_corrective_calls`` retries carry the exact rules that
        fired, and only mechanical rules — never taste — trigger the
        retry."""
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
                raise IdeationRejectedError(
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
    """One corrective request: the failed reply (when one was parseable at
    all) plus every deterministic rule that fired. Mechanical rules only —
    never analytic taste."""
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    feedback = (
        f"Your output was rejected by the deterministic ideation gate. "
        f"Nothing was recorded. The rules that fired:\n{rules}\n"
        f"Return one corrected output now, satisfying every original "
        f"constraint. Cite only the listed keys and ids; ground every "
        f"claim in the records you were shown; drop anything you cannot "
        f"ground."
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
        metadata={**base.metadata, "ideation_repair": str(attempt)},
    )


# -- record construction from gate-accepted payloads ---------------------------


def _prediction(item: object) -> Prediction:
    assert isinstance(item, Mapping)
    return Prediction(
        text=str(item["text"]), falsifier=str(item["falsifier"])
    )


def _data_requirement(item: object) -> DataRequirement:
    assert isinstance(item, Mapping)
    return DataRequirement(
        name=str(item["name"]),
        status=DataStatus(str(item["status"])),
        role=str(item["role"]),
    )


def _candidate_idea(
    entry: object,
    run_id: str,
    inputs: _MapInputs,
    provenance: CallProvenance,
) -> CandidateIdea:
    assert isinstance(entry, Mapping)
    cited = _strings(entry["cited_source_ids"])
    eras = [inputs.eras[source_id] for source_id in cited]
    resources = entry["resources"]
    assert isinstance(resources, Mapping)
    return CandidateIdea(
        run_id=run_id,
        title=str(entry["title"]),
        research_question=str(entry["research_question"]),
        proposed_contribution=str(entry["proposed_contribution"]),
        mechanism=str(entry["mechanism"]),
        hypothesis=str(entry["hypothesis"]),
        grounding=str(entry["grounding"]),
        predictions=tuple(
            _prediction(item) for item in _sequence(entry["predictions"])
        ),
        datasets=tuple(
            _data_requirement(item)
            for item in _sequence(entry["datasets"])
        ),
        metrics=_strings(entry["metrics"]),
        evaluation_protocol=str(entry["evaluation_protocol"]),
        baselines=_strings(entry["baselines"]),
        ablations=_strings(entry["ablations"]),
        resources=ResourceEstimate(
            compute=str(resources["compute"]),
            data=str(resources["data"]),
            implementation=str(resources["implementation"]),
        ),
        risks=_strings(entry["risks"]),
        cfp_alignment=str(entry["cfp_alignment"]),
        aligned_topics=_strings(entry["aligned_topics"]),
        uncertainty=str(entry["uncertainty"]),
        search_terms=_strings(entry["search_terms"]),
        addressed_problems=tuple(
            AddressedProblem(
                key=key,
                statement=inputs.problems[key].statement,
                kind=inputs.problems[key].kind,
                tier=inputs.tiers[inputs.problems[key].statement],
            )
            for key in _strings(entry["problem_keys"])
        ),
        targeted_themes=tuple(
            TargetedTheme(
                key=key,
                name=inputs.themes[key].name,
                era=inputs.themes[key].era,
            )
            for key in _strings(entry["theme_keys"])
        ),
        cited_source_ids=cited,
        cited_recent=sum(1 for era in eras if era is SourceEra.RECENT),
        cited_foundational=sum(
            1 for era in eras if era is SourceEra.FOUNDATIONAL
        ),
        cited_undated=sum(1 for era in eras if era is SourceEra.UNDATED),
        provenance=provenance,
    )


def _portfolio(
    inputs: _MapInputs, ideas: tuple[CandidateIdea, ...]
) -> PortfolioReport:
    addressed = {
        problem.statement
        for idea in ideas
        for problem in idea.addressed_problems
    }
    unaddressed = tuple(
        problem.statement
        for problem in inputs.inventory.problems
        if problem.statement not in addressed
    )
    tier_counts = {tier: 0 for tier in SupportTier}
    for statement in addressed:
        tier_counts[inputs.tiers[statement]] += 1
    return PortfolioReport(
        problems_total=len(inputs.inventory.problems),
        problems_addressed=len(addressed),
        problems_unaddressed=len(unaddressed),
        unaddressed_statements=unaddressed,
        addressed_multi_source=tier_counts[SupportTier.MULTI_SOURCE],
        addressed_tentative=tier_counts[SupportTier.TENTATIVE],
        addressed_single_source_limitation=tier_counts[
            SupportTier.SINGLE_SOURCE_LIMITATION
        ],
        addressed_contradicted=tier_counts[SupportTier.CONTRADICTED],
        candidates=len(ideas),
        distinct_sources_cited=len(
            {sid for idea in ideas for sid in idea.cited_source_ids}
        ),
        themes_targeted=len(
            {theme.name for idea in ideas for theme in idea.targeted_themes}
        ),
        distinct_problem_sets=len(
            {
                frozenset(p.key for p in idea.addressed_problems)
                for idea in ideas
            }
        ),
        distinct_theme_sets=len(
            {
                frozenset(t.key for t in idea.targeted_themes)
                for idea in ideas
            }
        ),
        distinct_dataset_sets=len(
            {
                frozenset(d.name.casefold() for d in idea.datasets)
                for idea in ideas
            }
        ),
        distinct_metric_sets=len(
            {
                frozenset(m.casefold() for m in idea.metrics)
                for idea in ideas
            }
        ),
    )


# -- prompt rendering ---------------------------------------------------------


def render_snapshot(snapshot: CfpSnapshot) -> str:
    lines = [
        "## Workshop call text",
        f"- source: {snapshot.source_url}",
        f"- supplied: {snapshot.supplied_at}",
        f"- sha256: {snapshot.text_sha256}",
        "",
        snapshot.text,
        "",
        "Extract the structured direction now as schema JSON.",
    ]
    return "\n".join(lines)


def render_ideation_context(
    inputs: _MapInputs,
    direction: DirectionRecord,
    max_candidates: int,
) -> str:
    lines = [
        f"## Topic\n{inputs.topic}",
        "\n## Workshop direction (constrains relevance; not evidence)",
        f"scope: {direction.scope}",
        "topics (copy exactly into aligned_topics):",
        *(f"- {topic}" for topic in direction.topics),
    ]
    if direction.constraints:
        lines.append("constraints:")
        lines.extend(f"- {item}" for item in direction.constraints)
    if direction.relevant_dates:
        lines.append("relevant dates:")
        lines.extend(f"- {item}" for item in direction.relevant_dates)
    lines.append(
        "\n## Accepted themes (cite these thm_ keys and no others)"
    )
    for key, theme in inputs.themes.items():
        lines.append(f"\n### {key}")
        lines.append(f"name: {theme.name} [era: {theme.era.value}]")
        lines.append(f"summary: {theme.summary}")
        lines.append(f"sources: {', '.join(theme.source_ids)}")
    lines.append(
        "\n## Extracted sources (cite these lit_ ids and no others; the "
        "listed claims are the only grounding surface)"
    )
    for record in inputs.grounded:
        lines.append(f"\n### {record.source_id} [era: {record.era.value}]")
        for label, items in (
            ("methods", record.methods),
            ("datasets", tuple(d.name for d in record.datasets)),
            ("metrics", record.metrics),
            ("evaluation", record.evaluation_protocols),
            ("baselines", record.baselines),
            ("reported results", record.reported_results),
            (
                "limitations",
                tuple(
                    f"[{entry.kind.value}] {entry.text}"
                    for entry in record.limitations
                ),
            ),
            ("future work", record.future_work),
            ("open problems", record.open_problems),
        ):
            if items:
                lines.append(f"{label}: " + " | ".join(items))
    lines.append(
        "\n## Open problems (cite these prob_ keys and no others)"
    )
    for key, problem in inputs.problems.items():
        tier = inputs.tiers[problem.statement]
        lines.append(
            f"\n### {key} [tier: {tier.value}, kind: {problem.kind.value}]"
        )
        lines.append(f"statement: {problem.statement}")
        lines.append(f"grounding: {problem.grounding}")
        lines.append(
            f"supporting: {', '.join(problem.supporting_source_ids)}"
        )
        if problem.conflicting_source_ids:
            lines.append(
                f"conflicting: "
                f"{', '.join(problem.conflicting_source_ids)}"
            )
    lines.append(
        f"\nPropose at most {max_candidates} candidates now as schema "
        f"JSON."
    )
    return "\n".join(lines)


def _sequence(value: object) -> Sequence[object]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return value


def _strings(value: object) -> tuple[str, ...]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return tuple(str(item) for item in value)
