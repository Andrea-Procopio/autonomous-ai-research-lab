"""The prior-art challenger: three gated calls per candidate, trusted
execution and a deterministic verdict around them.

One run challenges every candidate of one immutable portfolio:

```
PriorArtDirective
  -> require_candidates_for_prior_art        (before any model call)
  -> per candidate:
       one gated query-proposal call         (model supplies text only)
       trusted retrieval through the corpus  (dates, ordering, budgets)
       cited-source injection + deduplication, cutoff filter
       gated similarity screening in batches (abstract-level and
         metadata-only sources apart, cited works first)
       one gated nearest-work comparison call
       trusted coverage + assess_prior_art   (the deterministic verdict)
  -> one run record with full lineage
```

The adversarial division of authority is the mapper's, sharpened: the
model proposes search text, judges similarity, and reads accessible
text into dimension comparisons — and nothing else. Trusted code owns
every date, every ordering, the deduplicated pool, the cutoff, which
sources are even rendered for comparison (metadata-only sources never
are), the known-prior-art stamp, the coverage, and the verdict. There
are no refinement rounds: a retrieval too thin to distinguish against
is honestly ``NOVELTY_UNRESOLVED``, and tuning the search until a
candidate survives would be exactly the failure this stage exists to
prevent. A verdict the caller dislikes has no route to a second call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final

from ..core.ids import occurrence_id
from ..ideation.records import CandidateIdea, IdeationRunRecord
from ..ideation.store import IdeationStore
from ..literature.corpus import LiteratureCorpus
from ..literature.dedup import deduplicate
from ..literature.retrieval import LiteratureQuery, LiteratureSource, ResultOrdering
from ..literature.store import LiteratureStore
from ..mapping.gates import MappingRejection, accessible_text_of
from ..mapping.records import CallProvenance, SupportLocation
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
from .assessment import (
    PriorArtAssessment,
    PriorArtThresholds,
    assess_prior_art,
    require_candidates_for_prior_art,
)
from .directive import PriorArtDirective
from .gates import (
    _number_tokens,
    check_comparisons,
    check_metadata_screening,
    check_prior_art_queries,
    check_similarity_screening,
)
from .plan import (
    MAX_ALTERNATIVES_PER_GROUP,
    MAX_CONCEPT_GROUPS,
    RENDERER_VERSION,
    canonical_groups,
    render_query,
)
from .records import (
    DIMENSIONS,
    ComparisonDimension,
    DimensionComparison,
    OverlapHypothesis,
    PriorArtCoverage,
    PriorArtQueryExecution,
    PriorArtQueryFamily,
    PriorArtRunRecord,
    PriorArtScreeningRecord,
    SimilarityDecision,
    SimilarityLabel,
    WorkComparison,
)
from .store import PriorArtStore


class PriorArtContractError(RuntimeError):
    """The run's inputs violate the challenge contract: a cited source
    the known-literature store does not hold, or collaborators wired
    against the wrong records."""


class PriorArtRejectedError(RuntimeError):
    """A model payload failed the deterministic gate after the bounded
    corrective call; everything rejected is preserved."""


class PriorArtBudgetError(RuntimeError):
    """The directive's model-call budget would be exceeded; the refusal
    happens before the call, never after."""


#: The retrieval strategy trusted code fixes per family: influence
#: surfaces the canonical prior work a date sort buries; recency finds
#: the concurrent work. A discovery signal, never a quality claim.
PRIOR_ART_RETRIEVAL_STRATEGIES: Final[
    Mapping[PriorArtQueryFamily, ResultOrdering]
] = {
    PriorArtQueryFamily.MECHANISM: ResultOrdering.INFLUENCE,
    PriorArtQueryFamily.PROBLEM_MECHANISM: ResultOrdering.RECENCY,
    PriorArtQueryFamily.EVALUATION_SETUP: ResultOrdering.RECENCY,
    PriorArtQueryFamily.SYNONYMS_LEGACY: ResultOrdering.INFLUENCE,
    PriorArtQueryFamily.COMPETING_APPROACHES: ResultOrdering.INFLUENCE,
    PriorArtQueryFamily.RECENT: ResultOrdering.RECENCY,
}

_ABSTRACT_RENDER_CHARS: Final = 1500

# -- output contracts ---------------------------------------------------------

_FAMILIES: Final = tuple(family.value for family in PriorArtQueryFamily)

PRIOR_ART_QUERY_SCHEMA: Final = OutputSchema(
    name="prior_art_queries",
    json_schema={
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "family": {
                            "type": "string",
                            "enum": list(_FAMILIES),
                        },
                        "groups": {
                            "type": "array",
                            "description": (
                                f"Concept groups, conjoined: a result "
                                f"must match every group. Keep to 2; "
                                f"{MAX_CONCEPT_GROUPS} is the hard cap."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "alternatives": {
                                        "type": "array",
                                        "description": (
                                            f"Alternative terms or "
                                            f"phrases for this ONE "
                                            f"concept; a result needs "
                                            f"any one of them. At most "
                                            f"{MAX_ALTERNATIVES_PER_GROUP} "
                                            f"per group. Plain words "
                                            f"only — no quotes, "
                                            f"parentheses, or operator "
                                            f"words; trusted code "
                                            f"renders the Boolean."
                                        ),
                                        "items": {"type": "string"},
                                    }
                                },
                                "required": ["alternatives"],
                            },
                        },
                    },
                    "required": ["family", "groups"],
                },
            }
        },
        "required": ["queries"],
    },
)

SIMILARITY_SCREENING_SCHEMA: Final = OutputSchema(
    name="prior_art_screening",
    json_schema={
        "type": "object",
        "properties": {
            "screens": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "decision": {
                            "type": "string",
                            "enum": [
                                decision.value
                                for decision in SimilarityDecision
                            ],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["source_id", "decision", "reason"],
                },
            }
        },
        "required": ["screens"],
    },
)

_OVERLAP_HYPOTHESIS_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "candidate_claim": {
            "type": "string",
            "description": (
                "A verbatim fragment of the candidate record naming the "
                "claim at risk; the gate re-finds it there."
            ),
        },
        "source_text": {
            "type": "string",
            "description": (
                "A verbatim quote from the named part of the source; "
                "the gate re-finds it there."
            ),
        },
        "support_location": {
            "type": "string",
            "enum": [location.value for location in SupportLocation],
        },
        "dimension": {
            "type": "string",
            "enum": [dimension.value for dimension in ComparisonDimension],
        },
        "rationale": {
            "type": "string",
            "description": (
                "Why this overlap could reach the candidate's core "
                "contribution rather than only its background."
            ),
        },
    },
    "required": [
        "candidate_claim",
        "source_text",
        "support_location",
        "dimension",
        "rationale",
    ],
}

METADATA_SCREENING_SCHEMA: Final = OutputSchema(
    name="prior_art_metadata_screening",
    json_schema={
        "type": "object",
        "properties": {
            "screens": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "decision": {
                            "type": "string",
                            "enum": [
                                decision.value
                                for decision in SimilarityDecision
                            ],
                        },
                        "reason": {"type": "string"},
                        "overlap_hypothesis": _OVERLAP_HYPOTHESIS_SCHEMA,
                    },
                    "required": ["source_id", "decision", "reason"],
                },
            }
        },
        "required": ["screens"],
    },
)

_DIMENSION_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "dimension": {
            "type": "string",
            "enum": [dimension.value for dimension in ComparisonDimension],
        },
        "candidate_position": {"type": "string"},
        "prior_work_position": {"type": "string"},
        "support_location": {
            "type": "string",
            "enum": [location.value for location in SupportLocation],
        },
        "support_snippet": {
            "type": "string",
            "description": (
                "A verbatim quote from the named part of the source; the "
                "gate re-finds it there."
            ),
        },
    },
    "required": [
        "dimension",
        "candidate_position",
        "prior_work_position",
        "support_location",
        "support_snippet",
    ],
}

COMPARISON_SCHEMA: Final = OutputSchema(
    name="prior_art_comparisons",
    json_schema={
        "type": "object",
        "properties": {
            "comparisons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "similarity": {
                            "type": "string",
                            "enum": [
                                label.value for label in SimilarityLabel
                            ],
                        },
                        "overlap_features": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "material_differences": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "dimensions": {
                            "type": "array",
                            "items": _DIMENSION_SCHEMA,
                        },
                    },
                    "required": [
                        "source_id",
                        "similarity",
                        "overlap_features",
                        "material_differences",
                        "dimensions",
                    ],
                },
            }
        },
        "required": ["comparisons"],
    },
)


_TEXT_NOTE: Final = (
    " In prose, prefer plain punctuation - simple hyphens and straight "
    "quotes - because decorative dashes are corrupted in transit and the "
    "gate rejects control characters; legitimate Unicode in names, "
    "technical terms, and titles is preserved."
)

QUERY_INSTRUCTION: Final = (
    "You design literature searches whose job is to FALSIFY a research "
    "candidate's differentiation: find the prior work most likely to "
    "have already done what the candidate proposes. Propose exactly one "
    "search plan per listed family: the exact mechanism or "
    "intervention; the problem combined with the mechanism; the task, "
    "dataset, metric, or evaluation combination; synonyms and older "
    "terminology for the same idea (the strongest prior art often "
    "predates today's vocabulary); the closest cited work and competing "
    "approaches; and recent work. A plan is a list of concept groups "
    "with strict Boolean meaning: a result must match EVERY group, and "
    "within a group any ONE alternative suffices. So use few groups - "
    f"one or two, {MAX_CONCEPT_GROUPS} at most - and put synonyms, "
    "rephrasings, and older terminology in the SAME group as "
    f"alternatives, at most {MAX_ALTERNATIVES_PER_GROUP} per group, "
    "never as extra groups: every extra group narrows the search, and "
    "a search conjoining many terms returns nothing. A multi-word "
    "alternative is matched as an exact phrase, so prefer short "
    "established phrases; at least one alternative must come from the "
    "candidate's own record. Plain terms only: no dates, no quotes, no "
    "parentheses, no AND/OR/NOT - trusted code renders the Boolean "
    "expression, sets every date range, and executes every search."
    + _TEXT_NOTE
)

SCREENING_INSTRUCTION: Final = (
    "You screen retrieved papers for similarity to ONE research "
    "candidate, from title and accessible abstract alone. This is not "
    "topical relevance: judge whether the paper may have already done "
    "what the candidate proposes. Decide every listed source exactly "
    "once: potential_overlap when the paper may anticipate the "
    "candidate's core contribution; related when it is clearly nearby "
    "work that does not anticipate it; unrelated when it is not nearby "
    "work; undecidable when the accessible text cannot settle the "
    "question - undecidable is honest, never a failure. Give one short "
    "reason per source, grounded in its shown text; use only numbers "
    "that appear in the source's text or the candidate's record, and "
    "never assert that no prior work exists."
    + _TEXT_NOTE
)

METADATA_SCREENING_INSTRUCTION: Final = (
    "You screen retrieved papers for similarity to ONE research "
    "candidate from bibliographic metadata alone - title, year, venue; "
    "no abstract was retrieved for these sources. Decide every listed "
    "source exactly once. Choose potential_overlap ONLY when the shown "
    "metadata itself supports a specific, material concern that the "
    "paper may have already done what the candidate proposes; such a "
    "decision must carry an overlap_hypothesis quoting the candidate "
    "claim at risk verbatim from the candidate record, the supporting "
    "source text verbatim from the named accessible part, the "
    "overlapping dimension, and why the concern reaches the "
    "candidate's core contribution. Sharing a broad topic, a common "
    "dataset, or generic terms is never material on its own: decide "
    "related or unrelated instead. When the metadata cannot settle "
    "the question, undecidable is honest, never a failure, and "
    "carries no hypothesis. Give one short reason per source, "
    "grounded in its shown text; use only numbers that appear in the "
    "source's text or the candidate's record, and never assert that "
    "no prior work exists."
    + _TEXT_NOTE
)

COMPARISON_INSTRUCTION: Final = (
    "You compare ONE research candidate against the closest retrieved "
    "prior works, dimension by dimension, from each work's title and "
    "abstract alone. For every listed work cover all five dimensions - "
    "scientific_question, mechanism, data_setting, evaluation_protocol, "
    "claimed_contribution - each with what the candidate proposes, what "
    "the work's accessible text reports, and a short verbatim quote "
    "from the named part (title or abstract) that your reading rests "
    "on; the gate re-finds every quote, so copy exactly. Then name the "
    "overlapping features and the material differences as short "
    "entries, and label the work substantial_match only when its "
    "accessible text indicates it already substantially does what the "
    "candidate proposes; related when it is nearby without "
    "anticipating; distinct otherwise. A label must agree with your own "
    "lists - a match names overlaps, a distinction names differences. "
    "Describe the prior work only from its shown text, using only its "
    "own numbers; never claim the candidate is novel - state "
    "differences instead. Be brief everywhere: one short sentence per "
    "position and short list entries - a reply that overruns the "
    "output budget is lost entirely."
    + _TEXT_NOTE
)


# -- the service --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PriorArtRunResult:
    """Everything one completed challenge produced, in memory; the same
    records are durable in the store."""

    run_record: PriorArtRunRecord
    directive: PriorArtDirective
    ideation_run: IdeationRunRecord
    candidates: tuple[CandidateIdea, ...]
    assessments: tuple[PriorArtAssessment, ...]
    executions: tuple[PriorArtQueryExecution, ...]
    screenings: tuple[PriorArtScreeningRecord, ...]
    comparisons: tuple[WorkComparison, ...]


class _Spend:
    """Mutable per-run call and token accounting, checked against the
    directive's budget before every provider call."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


class _Pool:
    """One candidate's deduplicated working pool, built by trusted code:
    fresh retrieval joined with the candidate's own cited sources,
    bridged by exact identifiers (DOI, arXiv id, provider id) so one
    paper never appears twice under two snapshot ids."""

    def __init__(
        self,
        *,
        fresh: Sequence[LiteratureSource],
        cited: Sequence[LiteratureSource],
        cutoff_date: str,
    ) -> None:
        combined = list(fresh)
        present = {source.id for source in combined}
        combined.extend(
            source for source in cited if source.id not in present
        )
        by_id = {source.id: source for source in combined}
        report = deduplicate(combined)
        group_of: dict[str, tuple[str, ...]] = {}
        for group in report.groups:
            for source_id in group.source_ids:
                group_of[source_id] = group.source_ids
        cited_ids = set(source.id for source in cited)
        fresh_ids = {source.id for source in fresh}
        self.representatives: list[LiteratureSource] = []
        self.known_prior_art: dict[str, bool] = {}
        self.recovered = 0
        for rep_id in report.representative_ids:
            members = set(group_of.get(rep_id, (rep_id,)))
            self.representatives.append(by_id[rep_id])
            known = bool(members & cited_ids)
            self.known_prior_art[rep_id] = known
            if known and members & fresh_ids:
                self.recovered += 1
        in_cutoff = [
            source
            for source in self.representatives
            if source.publication_date is None
            or source.publication_date <= cutoff_date
        ]
        # Known prior art screens first: the candidate's own cited works
        # are the most likely falsifiers, so a screening truncation must
        # cost the fresh tail, never them (the Task 5D.1 live defect —
        # cited works sorted last, and exactly they were truncated).
        self.in_cutoff = [
            *(s for s in in_cutoff if self.known_prior_art[s.id]),
            *(s for s in in_cutoff if not self.known_prior_art[s.id]),
        ]
        self.post_cutoff_excluded = len(self.representatives) - len(
            self.in_cutoff
        )
        self.metadata_ids = frozenset(
            source.id for source in self.in_cutoff if source.abstract is None
        )


class PriorArtChallenger:
    """See the module docstring; construction is explicit wiring, and
    every collaborator is injected. The ideation store is read-only
    here; ``known_literature`` (the mapping run's corpus, holding the
    candidates' cited sources) is read-only too; only ``corpus`` — the
    fresh challenge corpus — and ``store`` are written."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        ledger: UsageLedger,
        ideation_store: IdeationStore,
        known_literature: LiteratureStore,
        corpus: LiteratureCorpus,
        store: PriorArtStore,
        thresholds: PriorArtThresholds | None = None,
        screening_batch_size: int = 10,
        max_output_tokens: int = 16384,
        temperature: float = 0.0,
        request_timeout_seconds: float = 240.0,
        max_corrective_calls: int = 1,
    ) -> None:
        self._provider = provider
        self._model = model
        self._ledger = ledger
        self._ideation_store = ideation_store
        self._known_literature = known_literature
        self._corpus = corpus
        self._store = store
        self._thresholds = thresholds or PriorArtThresholds()
        self._screening_batch_size = screening_batch_size
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._request_timeout_seconds = request_timeout_seconds
        self._max_corrective_calls = max_corrective_calls

    def run(self, directive: PriorArtDirective) -> PriorArtRunResult:
        self._store.record_directive(directive)

        # The one door: before any model call, the durable ideation run
        # record must hold a loadable candidate portfolio.
        ideation_run = require_candidates_for_prior_art(
            self._ideation_store, directive.ideation_run_record_id
        )
        candidates: list[CandidateIdea] = []
        for candidate_id in ideation_run.candidate_ids:
            candidate = self._ideation_store.get_idea(candidate_id)
            assert candidate is not None  # the door verified loadability
            candidates.append(candidate)

        run_id = occurrence_id("pac")
        spend = _Spend(directive.max_model_calls)
        assessments: list[PriorArtAssessment] = []
        executions: list[PriorArtQueryExecution] = []
        screenings: list[PriorArtScreeningRecord] = []
        comparisons: list[WorkComparison] = []
        for candidate in candidates:
            assessment = self._challenge(
                run_id,
                directive,
                candidate,
                spend,
                executions,
                screenings,
                comparisons,
            )
            assessments.append(assessment)

        run_record = self._store.record_run(
            PriorArtRunRecord(
                run_id=run_id,
                directive_id=directive.id,
                ideation_run_record_id=ideation_run.id,
                ideation_run_id=ideation_run.run_id,
                assessment_id=ideation_run.assessment_id,
                map_run_id=ideation_run.map_run_id,
                snapshot_id=ideation_run.snapshot_id,
                candidate_ids=tuple(c.id for c in candidates),
                prior_art_assessment_ids=tuple(a.id for a in assessments),
                query_execution_ids=tuple(e.id for e in executions),
                screening_ids=tuple(s.id for s in screenings),
                comparison_ids=tuple(c.id for c in comparisons),
                model_calls=spend.calls,
                input_tokens=spend.input_tokens,
                output_tokens=spend.output_tokens,
            )
        )
        return PriorArtRunResult(
            run_record=run_record,
            directive=directive,
            ideation_run=ideation_run,
            candidates=tuple(candidates),
            assessments=tuple(assessments),
            executions=tuple(executions),
            screenings=tuple(screenings),
            comparisons=tuple(comparisons),
        )

    # -- one candidate's challenge ---------------------------------------------

    def _challenge(
        self,
        run_id: str,
        directive: PriorArtDirective,
        candidate: CandidateIdea,
        spend: _Spend,
        executions: list[PriorArtQueryExecution],
        screenings: list[PriorArtScreeningRecord],
        comparisons: list[WorkComparison],
    ) -> PriorArtAssessment:
        candidate_block = render_candidate(candidate)
        candidate_tokens = _number_tokens(candidate_block.casefold())

        # Stage 1: one gated query-proposal call; trusted execution.
        queries_payload, _ = self._gated_call(
            self._request(
                QUERY_INSTRUCTION,
                render_candidate_for_queries(candidate_block),
                PRIOR_ART_QUERY_SCHEMA,
                run_id,
                "queries",
            ),
            gate=partial(
                check_prior_art_queries,
                max_queries_per_family=1,
                candidate_haystack=candidate_block,
            ),
            stage="queries",
            run_id=run_id,
            spend=spend,
        )
        candidate_executions, fresh = self._retrieve(
            run_id, directive, candidate.id, queries_payload
        )
        executions.extend(candidate_executions)

        # Trusted pooling: cited sources join the pool — a candidate
        # overlapping work it itself cites is the most likely falsifier.
        pool = _Pool(
            fresh=fresh,
            cited=self._cited_sources(candidate),
            cutoff_date=directive.cutoff_date,
        )
        known_ids = (
            frozenset(source.id for source in pool.representatives)
            | frozenset(candidate.cited_source_ids)
            | frozenset({candidate.id})
            | frozenset(p.key for p in candidate.addressed_problems)
            | frozenset(t.key for t in candidate.targeted_themes)
        )

        # Stage 2: gated similarity screening over the bounded pool.
        to_screen = pool.in_cutoff[: directive.max_screened_per_candidate]
        candidate_screenings = self._screen(
            run_id,
            candidate,
            candidate_block,
            candidate_tokens,
            known_ids,
            pool,
            to_screen,
            spend,
        )
        screenings.extend(candidate_screenings)

        # Stage 3: one gated comparison call over the closest
        # abstract-level works; metadata-only sources are never rendered.
        to_compare = _comparison_pool(
            candidate_screenings, pool, directive.max_compared_works
        )
        candidate_comparisons = self._compare(
            run_id,
            candidate,
            candidate_block,
            candidate_tokens,
            known_ids,
            directive.cutoff_date,
            pool,
            to_compare,
            spend,
        )
        comparisons.extend(candidate_comparisons)

        # Trusted coverage and the deterministic verdict.
        coverage = _coverage(
            candidate,
            candidate_executions,
            pool,
            to_screen,
            candidate_screenings,
            candidate_comparisons,
        )
        assessment = assess_prior_art(
            run_id=run_id,
            candidate_id=candidate.id,
            directive_id=directive.id,
            screenings=candidate_screenings,
            comparisons=candidate_comparisons,
            coverage=coverage,
            metadata_source_ids=pool.metadata_ids,
            thresholds=self._thresholds,
        )
        return self._store.record_prior_art_assessment(assessment)

    # -- trusted execution -----------------------------------------------------

    def _cited_sources(
        self, candidate: CandidateIdea
    ) -> tuple[LiteratureSource, ...]:
        sources = []
        for source_id in candidate.cited_source_ids:
            source = self._known_literature.get_source(source_id)
            if source is None:
                raise PriorArtContractError(
                    f"candidate {candidate.id} cites {source_id}, which "
                    f"the known-literature store does not hold; the "
                    f"challenge reads the same corpus the mapping run "
                    f"recorded"
                )
            sources.append(source)
        return tuple(sources)

    def _retrieve(
        self,
        run_id: str,
        directive: PriorArtDirective,
        candidate_id: str,
        payload: Mapping[str, object],
    ) -> tuple[list[PriorArtQueryExecution], list[LiteratureSource]]:
        """Trusted execution of accepted queries, in payload order:
        dates and retrieval strategy from the directive and the family —
        never from the model — retrieval through the challenge corpus
        (cache-or-live), one durable execution record each."""
        executions: list[PriorArtQueryExecution] = []
        seen: set[str] = set()
        fresh_ordered: list[LiteratureSource] = []
        entries = payload["queries"]
        assert isinstance(entries, Sequence)
        for item in entries:
            assert isinstance(item, Mapping)
            family = PriorArtQueryFamily(str(item["family"]))
            groups = canonical_groups(_accepted_groups(item))
            text = render_query(groups)
            from_date = (
                directive.recent_window_start
                if family is PriorArtQueryFamily.RECENT
                else ""
            )
            ordering = PRIOR_ART_RETRIEVAL_STRATEGIES[family]
            query = LiteratureQuery(
                text=text,
                from_date=from_date,
                to_date=directive.cutoff_date,
                per_page=min(directive.results_per_query, 25),
                max_results=directive.results_per_query,
                ordering=ordering,
            )
            result = self._corpus.search(query)
            new = [s for s in result.sources if s.id not in seen]
            execution = self._store.record_query_execution(
                PriorArtQueryExecution(
                    run_id=run_id,
                    candidate_id=candidate_id,
                    family=family,
                    text=text,
                    from_date=from_date,
                    to_date=directive.cutoff_date,
                    query_fingerprint=query.fingerprint,
                    search_record_id=result.record.id,
                    retrieved=len(result.sources),
                    new_unique=len(new),
                    from_cache=result.from_cache,
                    ordering=ordering,
                    plan_groups=groups,
                    renderer=RENDERER_VERSION,
                )
            )
            executions.append(execution)
            for source in new:
                seen.add(source.id)
                fresh_ordered.append(source)
        return executions, fresh_ordered

    def _screen(
        self,
        run_id: str,
        candidate: CandidateIdea,
        candidate_block: str,
        candidate_tokens: frozenset[str],
        known_ids: frozenset[str],
        pool: _Pool,
        to_screen: Sequence[LiteratureSource],
        spend: _Spend,
    ) -> list[PriorArtScreeningRecord]:
        """Two screening tasks, two precise instructions: abstract-level
        sources under the similarity gate, metadata-only sources under
        the material-ambiguity gate — a small batch whose rejection
        never forces re-doing the abstract screens."""
        records: list[PriorArtScreeningRecord] = []
        abstract_level = [
            source for source in to_screen if source.id not in pool.metadata_ids
        ]
        metadata_only = [
            source for source in to_screen if source.id in pool.metadata_ids
        ]
        for start in range(
            0, len(abstract_level), self._screening_batch_size
        ):
            batch = abstract_level[start : start + self._screening_batch_size]
            accessible = {
                source.id: accessible_text_of(source) for source in batch
            }
            payload, provenance = self._gated_call(
                self._request(
                    SCREENING_INSTRUCTION,
                    render_screening_batch(candidate_block, batch),
                    SIMILARITY_SCREENING_SCHEMA,
                    run_id,
                    "screening",
                ),
                gate=partial(
                    check_similarity_screening,
                    accessible=accessible,
                    candidate_tokens=candidate_tokens,
                    known_ids=known_ids,
                ),
                stage="screening",
                run_id=run_id,
                spend=spend,
            )
            screens = payload["screens"]
            assert isinstance(screens, Sequence)
            for item in screens:
                assert isinstance(item, Mapping)
                source_id = str(item["source_id"])
                records.append(
                    self._store.record_screening(
                        PriorArtScreeningRecord(
                            run_id=run_id,
                            candidate_id=candidate.id,
                            source_id=source_id,
                            known_prior_art=pool.known_prior_art[source_id],
                            decision=SimilarityDecision(
                                str(item["decision"])
                            ),
                            reason=str(item["reason"]),
                            provenance=provenance,
                        )
                    )
                )
        for start in range(
            0, len(metadata_only), self._screening_batch_size
        ):
            batch = metadata_only[start : start + self._screening_batch_size]
            sources = {source.id: source for source in batch}
            payload, provenance = self._gated_call(
                self._request(
                    METADATA_SCREENING_INSTRUCTION,
                    render_metadata_screening_batch(candidate_block, batch),
                    METADATA_SCREENING_SCHEMA,
                    run_id,
                    "metadata_screening",
                ),
                gate=partial(
                    check_metadata_screening,
                    sources=sources,
                    candidate_haystack=candidate_block,
                    candidate_tokens=candidate_tokens,
                    known_ids=known_ids,
                ),
                stage="metadata_screening",
                run_id=run_id,
                spend=spend,
            )
            screens = payload["screens"]
            assert isinstance(screens, Sequence)
            for item in screens:
                assert isinstance(item, Mapping)
                source_id = str(item["source_id"])
                hypothesis = item.get("overlap_hypothesis")
                records.append(
                    self._store.record_screening(
                        PriorArtScreeningRecord(
                            run_id=run_id,
                            candidate_id=candidate.id,
                            source_id=source_id,
                            known_prior_art=pool.known_prior_art[source_id],
                            decision=SimilarityDecision(
                                str(item["decision"])
                            ),
                            reason=str(item["reason"]),
                            provenance=provenance,
                            overlap_hypothesis=(
                                _overlap_hypothesis(hypothesis)
                                if hypothesis is not None
                                else None
                            ),
                        )
                    )
                )
        return records

    def _compare(
        self,
        run_id: str,
        candidate: CandidateIdea,
        candidate_block: str,
        candidate_tokens: frozenset[str],
        known_ids: frozenset[str],
        cutoff_date: str,
        pool: _Pool,
        to_compare: Sequence[LiteratureSource],
        spend: _Spend,
    ) -> list[WorkComparison]:
        if not to_compare:
            return []
        sources = {source.id: source for source in to_compare}
        payload, provenance = self._gated_call(
            self._request(
                COMPARISON_INSTRUCTION,
                render_comparison_batch(candidate_block, to_compare),
                COMPARISON_SCHEMA,
                run_id,
                "comparison",
            ),
            gate=partial(
                check_comparisons,
                sources=sources,
                cutoff_date=cutoff_date,
                candidate_tokens=candidate_tokens,
                known_ids=known_ids,
            ),
            stage="comparison",
            run_id=run_id,
            spend=spend,
        )
        records: list[WorkComparison] = []
        entries = payload["comparisons"]
        assert isinstance(entries, Sequence)
        for item in entries:
            assert isinstance(item, Mapping)
            source_id = str(item["source_id"])
            dimensions = item["dimensions"]
            assert isinstance(dimensions, Sequence)
            records.append(
                self._store.record_comparison(
                    WorkComparison(
                        run_id=run_id,
                        candidate_id=candidate.id,
                        source_id=source_id,
                        known_prior_art=pool.known_prior_art[source_id],
                        dimensions=tuple(
                            _dimension(entry) for entry in dimensions
                        ),
                        overlap_features=_strings(item["overlap_features"]),
                        material_differences=_strings(
                            item["material_differences"]
                        ),
                        similarity=SimilarityLabel(str(item["similarity"])),
                        provenance=provenance,
                    )
                )
            )
        return records

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
            metadata={"prior_art_run": run_id, "stage": stage},
        )

    def _invoke(self, request: ModelRequest, spend: _Spend) -> ModelResponse:
        """One provider call: budget checked before, accounting reaching
        the ledger exactly once — the response on success, the attached
        cost on failure — before any error propagates. The run's own
        spend folds failed-call usage too: a billed schema violation
        stays on the run record exactly as it stays on the ledger."""
        if spend.calls >= spend.limit:
            raise PriorArtBudgetError(
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
        preferred verdict — trigger the retry."""
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
                raise PriorArtRejectedError(
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
    only — never a preferred verdict."""
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    feedback = (
        f"Your output was rejected by the deterministic prior-art gate. "
        f"Nothing was recorded. The rules that fired:\n{rules}\n"
        f"Return one corrected output now, satisfying every original "
        f"constraint. Judge only the listed sources; quote their "
        f"accessible text verbatim; drop anything you cannot ground."
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
        metadata={**base.metadata, "prior_art_repair": str(attempt)},
    )


# -- trusted pooling, ordering, and coverage -----------------------------------


def _comparison_pool(
    screenings: Sequence[PriorArtScreeningRecord],
    pool: _Pool,
    max_compared_works: int,
) -> list[LiteratureSource]:
    """The closest works, chosen by trusted code: every abstract-level
    potential overlap first, then abstract-level related work, in
    screening order, bounded by the directive. Metadata-only sources
    are never comparable — their ambiguity is the verdict's problem,
    not the model's."""
    by_id = {source.id: source for source in pool.in_cutoff}
    chosen: list[LiteratureSource] = []
    for wanted in (
        SimilarityDecision.POTENTIAL_OVERLAP,
        SimilarityDecision.RELATED,
    ):
        for record in screenings:
            if len(chosen) >= max_compared_works:
                return chosen
            if record.decision is not wanted:
                continue
            if record.source_id in pool.metadata_ids:
                continue
            source = by_id[record.source_id]
            if source not in chosen:
                chosen.append(source)
    return chosen


def _coverage(
    candidate: CandidateIdea,
    executions: Sequence[PriorArtQueryExecution],
    pool: _Pool,
    to_screen: Sequence[LiteratureSource],
    screenings: Sequence[PriorArtScreeningRecord],
    comparisons: Sequence[WorkComparison],
) -> PriorArtCoverage:
    decisions = {
        decision: sum(
            1 for record in screenings if record.decision is decision
        )
        for decision in SimilarityDecision
    }
    last = executions[-1] if executions else None
    saturation = (
        round(1.0 - last.new_unique / last.retrieved, 4)
        if last is not None and last.retrieved
        else 1.0
    )
    in_cutoff = pool.in_cutoff
    return PriorArtCoverage(
        families_executed=tuple(
            family.value
            for family in PriorArtQueryFamily
            if any(e.family is family for e in executions)
        ),
        queries_executed=len(executions),
        total_retrieved=sum(e.retrieved for e in executions),
        unique_sources=len(pool.representatives),
        overlap=sum(e.retrieved for e in executions)
        + len(candidate.cited_source_ids)
        - len(pool.representatives),
        saturation=saturation,
        post_cutoff_excluded=pool.post_cutoff_excluded,
        undated_sources=sum(
            1 for source in in_cutoff if source.publication_date is None
        ),
        abstract_level=sum(
            1 for source in in_cutoff if source.abstract is not None
        ),
        metadata_level=len(pool.metadata_ids),
        known_prior_art_listed=len(candidate.cited_source_ids),
        known_prior_art_recovered=pool.recovered,
        screened=len(to_screen),
        potential_overlap=decisions[SimilarityDecision.POTENTIAL_OVERLAP],
        related=decisions[SimilarityDecision.RELATED],
        unrelated=decisions[SimilarityDecision.UNRELATED],
        undecidable=decisions[SimilarityDecision.UNDECIDABLE],
        metadata_ambiguous=sum(
            1
            for record in screenings
            if record.source_id in pool.metadata_ids
            and record.decision is SimilarityDecision.POTENTIAL_OVERLAP
        ),
        screening_truncated=len(in_cutoff) - len(to_screen),
        compared_works=len(comparisons),
    )


# -- rendering ----------------------------------------------------------------


def render_candidate(candidate: CandidateIdea) -> str:
    """The one candidate block every stage renders: the full structured
    record, so a screen or comparison judges the actual proposal, not a
    summary of it."""
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
        f"search terms: {'; '.join(candidate.search_terms)}",
        f"uncertainty: {candidate.uncertainty}",
    ]
    return "\n".join(lines)


def render_candidate_for_queries(candidate_block: str) -> str:
    families = "\n".join(
        f"- {family.value}" for family in PriorArtQueryFamily
    )
    return (
        f"{candidate_block}\n\n## Query families (exactly one search "
        f"plan each)\n{families}\n\nPropose one plan per family as "
        f"schema JSON: concept groups conjoined, alternatives within a "
        f"group as alternatives."
    )


def render_screening_batch(
    candidate_block: str, sources: Sequence[LiteratureSource]
) -> str:
    lines = [
        candidate_block,
        "\n## Sources to screen (title + accessible abstract only)",
    ]
    for source in sources:
        lines.append(f"\n### {source.id}")
        lines.append(f"title: {source.title or '(no title reported)'}")
        lines.append(
            f"year: {source.publication_year!r}; venue: "
            f"{source.venue or '(unreported)'}; type: "
            f"{source.work_type or '(unreported)'}"
        )
        if source.abstract is None:
            lines.append("abstract: (not retrieved — metadata-only access)")
        else:
            lines.append(f"abstract: {_clip(source.abstract)}")
    lines.append(
        "\nScreen every source above against the candidate, exactly once "
        "each, as schema JSON."
    )
    return "\n".join(lines)


def render_metadata_screening_batch(
    candidate_block: str, sources: Sequence[LiteratureSource]
) -> str:
    lines = [
        candidate_block,
        "\n## Metadata-only sources to screen (no abstract retrieved)",
    ]
    for source in sources:
        lines.append(f"\n### {source.id}")
        lines.append(f"title: {source.title or '(no title reported)'}")
        lines.append(
            f"year: {source.publication_year!r}; venue: "
            f"{source.venue or '(unreported)'}; type: "
            f"{source.work_type or '(unreported)'}"
        )
    lines.append(
        "\nScreen every source above against the candidate, exactly once "
        "each, as schema JSON; a potential_overlap decision carries its "
        "attested overlap_hypothesis."
    )
    return "\n".join(lines)


def render_comparison_batch(
    candidate_block: str, sources: Sequence[LiteratureSource]
) -> str:
    lines = [
        candidate_block,
        "\n## Closest works to compare (title + accessible abstract)",
    ]
    for source in sources:
        lines.append(f"\n### {source.id}")
        lines.append(f"title: {source.title or '(no title reported)'}")
        lines.append(
            f"date: {source.publication_date or '(unreported)'}; venue: "
            f"{source.venue or '(unreported)'}"
        )
        assert source.abstract is not None  # metadata-only never rendered
        lines.append(f"abstract: {_clip(source.abstract)}")
    dimensions = ", ".join(d.value for d in DIMENSIONS)
    lines.append(
        f"\nCompare the candidate against every work above, exactly once "
        f"each, across all five dimensions ({dimensions}), as schema "
        f"JSON."
    )
    return "\n".join(lines)


def _accepted_groups(
    item: Mapping[str, object],
) -> tuple[tuple[str, ...], ...]:
    groups = item["groups"]
    assert isinstance(groups, Sequence)
    projected = []
    for group in groups:
        assert isinstance(group, Mapping)
        alternatives = group["alternatives"]
        assert isinstance(alternatives, Sequence)
        projected.append(tuple(str(term) for term in alternatives))
    return tuple(projected)


def _clip(text: str) -> str:
    if len(text) <= _ABSTRACT_RENDER_CHARS:
        return text
    return text[:_ABSTRACT_RENDER_CHARS] + " [truncated for screening]"


def _dimension(entry: object) -> DimensionComparison:
    assert isinstance(entry, Mapping)
    return DimensionComparison(
        dimension=ComparisonDimension(str(entry["dimension"])),
        candidate_position=str(entry["candidate_position"]),
        prior_work_position=str(entry["prior_work_position"]),
        support_location=SupportLocation(str(entry["support_location"])),
        support_snippet=str(entry["support_snippet"]),
    )


def _overlap_hypothesis(entry: object) -> OverlapHypothesis:
    assert isinstance(entry, Mapping)
    return OverlapHypothesis(
        candidate_claim=str(entry["candidate_claim"]),
        source_text=str(entry["source_text"]),
        support_location=SupportLocation(str(entry["support_location"])),
        dimension=ComparisonDimension(str(entry["dimension"])),
        rationale=str(entry["rationale"]),
    )


def _strings(value: object) -> tuple[str, ...]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return tuple(str(entry) for entry in value)
