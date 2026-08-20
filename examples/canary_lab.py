"""A laboratory with no network, no clock, and no model — the canary's.

The canary carries a synthetic brief through all seven stages of the
chain. That needs instruments, and these are they: a literature provider
that answers from a small invented corpus, a model that answers by
computing a reply from the request, and a runtime whose roles and
trusted templates are ordinary Python.

Two choices make it deterministic in a way a scripted list is not.

**The literature provider answers by query, not by turn.** A stage that
re-runs after a crash issues the same searches and gets the same works
back, whatever order the rest of the walk happened in.

**The model answers by schema, not by turn.** Every reply is derived
from the request — which sources the prompt shows, which candidate it
names — so the fixture cannot drift out of step with a stage that makes
one more call than it used to. It is a function, not a script.

Everything the model says is consistent with what the corpus says,
because the gates check that: numbers must appear in the cited text,
dataset names must appear verbatim, comparison snippets must be quoted
from the abstract they claim. A canary whose model contradicted its own
corpus would be rejected by the stages, which is the system working.

This is a fixture, not a demonstration of research. The findings are
invented and mean nothing. What it demonstrates is the machinery: that
seven stages can be walked, interrupted anywhere, and resumed.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from autonomous_research_lab.control.lab import RuntimeRequest
from autonomous_research_lab.control.stage import StageName
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
    ResultProposal,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.execution.executor import (
    ExperimentJob,
    job_id_for_attempt,
)
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.execution.runner import (
    DirectJobRunner,
    JobRunner,
)
from autonomous_research_lab.ideation.records import problem_key, theme_key
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureProvider,
    LiteratureQuery,
    LiteratureSource,
    RetrievedSearch,
)
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.priorart.records import ComparisonDimension
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.journal import JournalingJobRunner
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from autonomous_research_lab.selection.records import REVIEW_FIELDS

PROVIDER: Final = "canary"
RETRIEVED_AT: Final = "2026-08-18T12:00:00+00:00"

#: One vocabulary for every abstract in the corpus. The gates hold the
#: model to the cited text, so a fixture whose replies use words the
#: sources do not contain is a fixture the stages reject. Sharing one
#: vocabulary makes that impossible by construction.
_ABSTRACT: Final = (
    "We study {phrase} for in-context learning. The method reweights "
    "attention heads and is evaluated on held-out prompts, where we "
    "report accuracy under distribution shift. We compare against "
    "adapters and LoRA baselines on public benchmarks. Limitations: "
    "results degrade under distribution shift, and evaluation covers "
    "public benchmarks only."
)


def _source(
    provider_id: str, *, phrase: str, title: str, date: str
) -> LiteratureSource:
    return LiteratureSource(
        provider=PROVIDER,
        provider_id=provider_id,
        title=title,
        authors=("Ada Lovelace", "Alan Turing"),
        publication_date=date,
        publication_year=int(date[:4]),
        venue="Journal of Invented Results",
        work_type="article",
        abstract=_ABSTRACT.format(phrase=phrase),
        doi=None,
        arxiv_id=None,
        provider_url=f"https://example.invalid/{provider_id}",
        landing_page_url=None,
        pdf_url=None,
        cited_by_count=None,
        referenced_work_ids=(),
        access_level=AccessLevel.ABSTRACT,
    )


# -- the corpus the field map is built from ------------------------------------

RECENT = tuple(
    _source(
        f"recent-{index}",
        phrase=f"prompt adaptation variant {index}",
        title=f"Prompt Adaptation for In-Context Learning ({index})",
        date=f"2026-0{index + 1}-01",
    )
    for index in range(1, 5)
)
FOUNDATIONAL = tuple(
    _source(
        f"foundational-{index}",
        phrase=f"episodic meta-learning variant {index}",
        title=f"Episodic Meta-Learning for Rapid Adaptation ({index})",
        date=f"201{index + 5}-06-01",
    )
    for index in range(1, 5)
)
UNCERTAIN = _source(
    "uncertain-1",
    phrase="adaptation in changing environments",
    title="Adaptation in Changing Environments",
    date="2026-04-04",
)
EXCLUDED = _source(
    "excluded-1",
    phrase="deep-sea coral reproduction",
    title="Deep-Sea Coral Reproduction",
    date="2026-05-05",
)

RELEVANT: Final = (*RECENT, *FOUNDATIONAL)
MAP_CORPUS: Final = (*RELEVANT, UNCERTAIN, EXCLUDED)

# -- the corpus the prior-art challenge screens --------------------------------

PRIOR_WORK: Final = tuple(
    _source(
        f"prior-{index}",
        phrase=f"attention head reweighting approach {index}",
        title=f"Reweighting Attention Heads for Adaptation ({index})",
        date=f"2025-{index:02d}-01" if index <= 12 else "2024-01-01",
    )
    for index in range(1, 13)
)

BY_ID: Final = {
    source.id: source for source in (*MAP_CORPUS, *PRIOR_WORK)
}

# -- the literature provider ---------------------------------------------------

#: Which works each search finds, keyed by a phrase the query text
#: carries. The mapping stage proposes three queries; the prior-art
#: challenge proposes six per candidate, and every one of them retrieves
#: from the same invented pile of near-work.
ANSWERS: Final[tuple[tuple[str, tuple[LiteratureSource, ...]], ...]] = (
    ("prompt adaptation", (*RECENT, UNCERTAIN)),
    ("episodic meta-learning", FOUNDATIONAL),
    ("in-context learning limitations", (RECENT[0], RECENT[1], EXCLUDED)),
    ("attention head reweighting", PRIOR_WORK[0:3]),
    ("head gating", PRIOR_WORK[3:6]),
    ("induction heads", PRIOR_WORK[6:9]),
    ("head masking", PRIOR_WORK[9:12]),
    ("adapters", PRIOR_WORK[0:3]),
    ("distribution shift", PRIOR_WORK[3:6]),
)

#: One search term per prior-art family, chosen so the six searches
#: between them turn up the whole invented pile of near-work and overlap
#: enough for the saturation figure to mean something.
PRIOR_ART_TERMS: Final = (
    ("mechanism", "attention head reweighting"),
    ("problem_mechanism", "head gating"),
    ("evaluation_setup", "induction heads"),
    ("synonyms_legacy", "head masking"),
    ("competing_approaches", "adapters"),
    ("recent", "distribution shift"),
)

#: A phrase every abstract in the corpus contains, so a comparison can
#: quote it and the gate can find it where the comparison says it is.
SNIPPET: Final = "reweights attention heads and is evaluated on held-out prompts"


def _retrieved(sources: Sequence[LiteratureSource]) -> RetrievedSearch:
    return RetrievedSearch(
        provider=PROVIDER,
        retrieved_at=RETRIEVED_AT,
        request_params={"search": "canary"},
        total_count=len(sources),
        pages_fetched=1,
        page_identifiers=("page-1",),
        rate_limit={},
        truncated=False,
        sources=tuple(sources),
    )


class CanaryLiterature(LiteratureProvider):
    """Answers by what the query says, never by how many came before."""

    @property
    def name(self) -> str:
        return PROVIDER

    def search(self, query: LiteratureQuery) -> RetrievedSearch:
        text = query.text.casefold()
        found: list[LiteratureSource] = []
        for phrase, sources in ANSWERS:
            if phrase in text:
                found.extend(
                    source for source in sources if source not in found
                )
        return _retrieved(found[: query.max_results])


# -- the model -----------------------------------------------------------------


def _shown(request: ModelRequest) -> list[LiteratureSource]:
    """Every source of ours the prompt actually shows, in prompt order.

    Reading the request rather than counting calls is what makes the
    fixture immune to a stage making one more call than it used to.
    """
    text = "\n".join(message.content for message in request.messages)
    positions = [
        (text.index(source_id), BY_ID[source_id])
        for source_id in BY_ID
        if source_id in text
    ]
    return [source for _, source in sorted(positions)]


class CanaryModel(ModelProvider):
    """One deterministic answer per schema, computed from the request."""

    @property
    def name(self) -> str:
        return PROVIDER

    def invoke(self, request: ModelRequest) -> ModelResponse:
        if request.schema is None:
            raise AssertionError("the canary answers structured calls only")
        answer = ANSWERERS.get(request.schema.name)
        if answer is None:
            raise AssertionError(
                f"the canary has no answer for schema "
                f"{request.schema.name!r}"
            )
        # Delegating to the fake provider keeps usage and latency derived
        # by the same fixed rules every other deterministic test uses.
        return FakeModelProvider([answer(request)], name=PROVIDER).invoke(
            request
        )


def _mapping_queries(request: ModelRequest) -> str:
    del request
    return json.dumps(
        {
            "queries": [
                {"family": "recent", "text": "prompt adaptation"},
                {"family": "foundational", "text": "episodic meta-learning"},
                {
                    "family": "limitations_open_problems",
                    "text": "in-context learning limitations",
                },
            ]
        }
    )


def _mapping_screening(request: ModelRequest) -> str:
    decisions = []
    for source in _shown(request):
        if source is EXCLUDED:
            decision = "excluded"
        elif source is UNCERTAIN:
            decision = "uncertain"
        else:
            decision = "relevant"
        decisions.append(
            {
                "source_id": source.id,
                "decision": decision,
                "reason": f"screened as {decision} from the shown abstract",
            }
        )
    return json.dumps({"decisions": decisions})


def _mapping_extraction(request: ModelRequest) -> str:
    (source,) = _shown(request)
    return json.dumps(
        {
            "source_id": source.id,
            "support_location": "abstract",
            "sufficient_support": True,
            "insufficiency_reason": "",
            "methods": ["attention head reweighting"],
            "datasets": [
                {
                    "name": "public benchmarks",
                    "task": "in-context learning",
                    "version": "",
                    "split": "",
                    "subset": "",
                    "preprocessing": "",
                    "size": "",
                    "availability": "unreported",
                    "url": "",
                    "license": "",
                }
            ],
            "metrics": ["accuracy"],
            "evaluation_protocols": ["evaluated on held-out prompts"],
            "baselines": ["adapters", "LoRA"],
            "reported_results": ["report accuracy under distribution shift"],
            "limitations": [
                {
                    "text": "results degrade under distribution shift",
                    "kind": "generalization",
                }
            ],
            "future_work": [],
            "open_problems": ["robustness under distribution shift"],
        }
    )


def _mapping_field_map(request: ModelRequest) -> str:
    """A map over exactly the sources the prompt shows.

    Built from the request rather than from the corpus, because a brief
    that extracts fewer sources shows fewer here — and a theme citing a
    source the run never extracted is rejected by the gate, correctly.
    """
    shown = _shown(request)
    recent = [source for source in shown if source in RECENT]
    older = [source for source in shown if source in FOUNDATIONAL]
    themes = []
    if recent:
        themes.append(
            {
                "name": "Prompt adaptation",
                "summary": (
                    "Recent work reweights attention heads to adapt "
                    "without weight updates."
                ),
                "era": "recent",
                "source_ids": [source.id for source in recent],
            }
        )
    if older:
        themes.append(
            {
                "name": "Meta-learning foundations",
                "summary": (
                    "Earlier work trains episodically for rapid adaptation."
                ),
                "era": "foundational",
                "source_ids": [source.id for source in older],
            }
        )
    return json.dumps(
        {
            "themes": themes,
            "approaches": [
                {
                    "name": "Gradient-free adaptation",
                    "summary": "Adaptation without updating any weights.",
                    "source_ids": [source.id for source in shown[:2]],
                }
            ],
            "evaluation_practices": [
                {
                    "name": "Held-out prompt accuracy",
                    "summary": "Accuracy measured on held-out prompts.",
                    "source_ids": [source.id for source in shown[:2]],
                }
            ],
            "relationships": (
                [
                    {
                        "kind": "builds_on",
                        "from_theme": "Prompt adaptation",
                        "to_theme": "Meta-learning foundations",
                        "note": "adaptation reuses episodic ideas",
                    }
                ]
                if len(themes) == 2
                else []
            ),
        }
    )


P_OPEN: Final = (
    "robustness of in-context learning under distribution shift is "
    "unresolved"
)
P_CONFLICT: Final = (
    "whether episodic training is required for rapid adaptation is "
    "contested"
)


def _mapping_inventory(request: ModelRequest) -> str:
    """The problems the shown extractions support, and no others."""
    shown = _shown(request)
    recent = [source for source in shown if source in RECENT]
    older = [source for source in shown if source in FOUNDATIONAL]
    problems: list[dict[str, object]] = []
    if recent:
        problems.append(
            {
                "statement": P_OPEN,
                "kind": "open_problem",
                "grounding": (
                    "the recent sources report degradation under "
                    "distribution shift"
                ),
                "supporting_source_ids": [
                    source.id for source in recent[:3]
                ],
                "conflicting_source_ids": [],
            }
        )
    if older and recent:
        problems.append(
            {
                "statement": P_CONFLICT,
                "kind": "conflicting_findings",
                "grounding": (
                    "the foundational sources train episodically; the "
                    "recent ones adapt prompts without it"
                ),
                "supporting_source_ids": [source.id for source in older[:2]],
                "conflicting_source_ids": [recent[0].id],
            }
        )
    return json.dumps({"problems": problems})


def _ideation_direction(request: ModelRequest) -> str:
    del request
    return json.dumps(
        {
            "scope": (
                "A call for work on how in-context learning adapts and on "
                "how well it holds up when the distribution moves."
            ),
            "topics": [
                "mechanisms of in-context learning",
                "efficient adaptation",
                "robustness under distribution shift",
            ],
            "constraints": ["Submissions are limited to 9 pages"],
            "relevant_dates": [],
        }
    )


def _candidate(
    *,
    title: str,
    question: str,
    mechanism: str,
    hypothesis: str,
    prediction: str,
    falsifier: str,
    problem: str,
    theme: str,
) -> dict[str, object]:
    return {
        "title": title,
        "research_question": question,
        "proposed_contribution": (
            "A held-out evaluation of the effect under distribution shift."
        ),
        "mechanism": mechanism,
        "hypothesis": hypothesis,
        "grounding": (
            "The cited records report accuracy under distribution shift on "
            "held-out prompts."
        ),
        "predictions": [{"text": prediction, "falsifier": falsifier}],
        "datasets": [
            {
                "name": "public benchmarks",
                "status": "existing",
                "role": "held-out evaluation",
            }
        ],
        "metrics": ["accuracy"],
        "evaluation_protocol": (
            "Adapt on one distribution, evaluate on held-out prompts from "
            "another."
        ),
        "baselines": ["adapters", "LoRA"],
        "ablations": ["remove the reweighting"],
        "resources": {
            "compute": "one CPU core",
            "data": "public benchmarks",
            "implementation": "a small standard-library harness",
        },
        "risks": ["the effect may vanish under distribution shift"],
        "cfp_alignment": (
            "Addresses the call's interest in mechanisms of in-context "
            "learning."
        ),
        "aligned_topics": ["mechanisms of in-context learning"],
        "uncertainty": (
            "Grounded in abstract-level claims from the cited records only."
        ),
        # Every family's anchor term, because the query gate requires
        # each proposed search to be grounded in the candidate's own
        # record rather than in whatever the model felt like searching.
        "search_terms": [term for _, term in PRIOR_ART_TERMS],
        "problem_keys": [problem_key(problem)],
        "theme_keys": [theme_key(theme)],
        "cited_source_ids": [RECENT[0].id, FOUNDATIONAL[0].id],
    }


def _ideation_candidates(request: ModelRequest) -> str:
    del request
    return json.dumps(
        {
            "refusal_justification": "",
            "diversity_rationale": (
                "One candidate tests whether the effect survives a shift; "
                "the other tests whether the training route matters at all."
            ),
            "candidates": [
                _candidate(
                    title="Reweighting under distribution shift",
                    question=(
                        "Does attention head reweighting keep its gains "
                        "when the distribution moves?"
                    ),
                    mechanism=(
                        "Reweighted heads may carry distribution-specific "
                        "information."
                    ),
                    hypothesis=(
                        "Reweighting keeps most of its in-distribution gain "
                        "on held-out prompts."
                    ),
                    prediction=(
                        "Accuracy on held-out prompts stays above the "
                        "adapters baseline."
                    ),
                    falsifier=(
                        "Accuracy on held-out prompts falls below the "
                        "adapters baseline."
                    ),
                    problem=P_OPEN,
                    theme="Prompt adaptation",
                ),
                _candidate(
                    title="Is episodic training required for adaptation",
                    question=(
                        "Does episodic training change how well adaptation "
                        "survives a shift?"
                    ),
                    mechanism=(
                        "Episodic training and prompt adaptation may reach "
                        "the same behaviour by different routes."
                    ),
                    hypothesis=(
                        "Episodically trained models and prompt-adapted "
                        "ones differ on held-out prompts."
                    ),
                    prediction=(
                        "The two approaches rank differently on held-out "
                        "prompts."
                    ),
                    falsifier=(
                        "The two approaches rank identically on every "
                        "held-out prompt set."
                    ),
                    problem=P_CONFLICT,
                    theme="Meta-learning foundations",
                ),
            ]
        }
    )


def _prior_art_queries(request: ModelRequest) -> str:
    del request
    return json.dumps(
        {
            "queries": [
                {"family": family, "groups": [{"alternatives": [term]}]}
                for family, term in PRIOR_ART_TERMS
            ]
        }
    )


#: The three near-works the challenge compares against. Everything else
#: screens unrelated, so the comparison stays about the closest work
#: rather than about everything retrieved.
NEAREST: Final = frozenset(source.id for source in PRIOR_WORK[:3])


def _prior_art_screening(request: ModelRequest) -> str:
    screens = [
        {
            "source_id": source.id,
            "decision": "related" if source.id in NEAREST else "unrelated",
            "reason": "judged from the shown title and abstract",
        }
        for source in _shown(request)
    ]
    return json.dumps({"screens": screens})


def _prior_art_comparisons(request: ModelRequest) -> str:
    comparisons = [
        {
            "source_id": source.id,
            "similarity": "related",
            "overlap_features": ["both reweight attention heads"],
            "material_differences": [
                "the prior work does not evaluate under distribution shift"
            ],
            "dimensions": [
                {
                    "dimension": dimension.value,
                    "candidate_position": (
                        "measures accuracy on held-out prompts under "
                        "distribution shift"
                    ),
                    "prior_work_position": (
                        "reweights attention heads and reports held-out "
                        "accuracy"
                    ),
                    "support_location": "abstract",
                    "support_snippet": SNIPPET,
                }
                for dimension in ComparisonDimension
            ],
        }
        for source in _shown(request)
    ]
    return json.dumps({"comparisons": comparisons})


_CANDIDATE_ID = re.compile(r"\bidea_[0-9a-f]{16}\b")


def _candidates_named(request: ModelRequest) -> list[str]:
    """The candidate ids the prompt names, first mention first.

    The same trick as :func:`_shown`, for records that are not sources:
    read the request instead of counting the calls before it.
    """
    text = "\n".join(message.content for message in request.messages)
    seen: list[str] = []
    for found in _CANDIDATE_ID.findall(text):
        if found not in seen:
            seen.append(found)
    return seen


def _selection_review(request: ModelRequest) -> str:
    candidates = _candidates_named(request)
    reviews = []
    for candidate_id in candidates:
        entry: dict[str, object] = {
            "candidate_id": candidate_id,
            "prior_art_verdict": "distinguished",
            "disqualifiers": [],
        }
        for name in REVIEW_FIELDS:
            entry[name] = (
                f"{name.replace('_', ' ')} is within the stated constraints "
                f"for this candidate"
            )
        reviews.append(entry)
    pairs = [
        {
            "first_candidate_id": first,
            "second_candidate_id": second,
            "comparison": (
                "both fit the constraints; they differ in how directly "
                "they test the shift"
            ),
        }
        for index, first in enumerate(candidates)
        for second in candidates[index + 1 :]
    ]
    return json.dumps({"reviews": reviews, "pairwise_comparisons": pairs})


def _selection_decision(request: ModelRequest) -> str:
    winner, *others = _candidates_named(request)
    return json.dumps(
        {
            "selected_candidate_id": winner,
            "decisive_tradeoff": (
                "the cheaper falsifier wins under the stated compute "
                "constraint"
            ),
            "why_selected_over": [
                {
                    "candidate_id": other,
                    "reason": "it answers its question with fewer runs",
                }
                for other in others
            ],
            "first_experimental_objective": (
                "measure accuracy on held-out prompts before and after the "
                "shift"
            ),
            "required_capabilities": ["a seeded synthetic data generator"],
            "residual_risks": ["the effect may be specific to the fixture"],
        }
    )


#: For each candidate: its prediction, its falsifier, and the two arms
#: the prediction compares -- every one of them a phrase that appears
#: verbatim somewhere in that candidate's own record, because the
#: admission gate re-finds each of them there.
ARMS: Final = (
    (
        "Accuracy on held-out prompts stays above the adapters baseline.",
        "Accuracy on held-out prompts falls below the adapters baseline.",
        "attention head reweighting",
        "adapters",
    ),
    (
        "The two approaches rank differently on held-out prompts.",
        "The two approaches rank identically on every held-out prompt set.",
        "Episodic training",
        "prompt adaptation",
    ),
)


def _admission_operationalization(request: ModelRequest) -> str:
    text = "\n".join(message.content for message in request.messages)
    prediction, falsifier, higher, lower = next(
        entry for entry in ARMS if entry[0] in text
    )
    return json.dumps(
        {
            "operational_predictions": [
                {
                    "prediction_text": prediction,
                    "condition": (
                        "on held-out prompts drawn from the shifted "
                        "distribution"
                    ),
                    "base_metric": "accuracy",
                    "expected_higher_arm": higher,
                    "expected_lower_arm": lower,
                    "contrary_observation": falsifier,
                    "support": [
                        {
                            "source": "candidate",
                            "field_path": "predictions[0].text",
                            "quote": prediction,
                        }
                    ],
                }
            ]
        }
    )


ANSWERERS: Final = {
    "mapping_queries": _mapping_queries,
    "mapping_screening": _mapping_screening,
    "mapping_extraction": _mapping_extraction,
    "mapping_field_map": _mapping_field_map,
    "mapping_inventory": _mapping_inventory,
    "ideation_direction": _ideation_direction,
    "ideation_candidates": _ideation_candidates,
    "prior_art_queries": _prior_art_queries,
    "prior_art_screening": _prior_art_screening,
    "prior_art_comparisons": _prior_art_comparisons,
    "selection_comparative_review": _selection_review,
    "selection_decision": _selection_decision,
    "admission_operationalization": _admission_operationalization,
}


# -- the runtime ---------------------------------------------------------------

EXPERIMENT = Path(__file__).resolve().parent / "experiments" / "canary_shift.py"


class CanaryScientist(ResearchRole):
    """Designs the one experiment the admitted prediction asks for, and
    reads the result back as a claim.

    It invents no metric: the pre-registered contrast the admission
    recorded is the metric, and the design carries it verbatim. A role
    that named its own observable would be answering a different
    question from the one the state committed to.
    """

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
        prediction = invocation.context.predictions[0]
        spec = ExperimentSpec(
            prediction_id=prediction.id,
            objective=(
                "Measure the pre-registered accuracy contrast on held-out "
                "prompts."
            ),
            procedure=(
                "Draw a seeded synthetic evaluation of both arms and report "
                "the difference in accuracy between them."
            ),
            metrics=(prediction.metric, "treatment_accuracy",
                     "control_accuracy"),
            baselines=("the adapters arm",),
            seeds=(11, 23),
        )
        return (ExperimentProposal(spec=spec, proposer="canary:scientist"),)

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
                claim=claim, links=(link,), proposer="canary:scientist"
            ),
        )


class CanaryEngineer(ResearchRole):
    """Prepares the designed experiment; trusted code runs it."""

    def __init__(self, runner: JobRunner) -> None:
        self._runner = runner

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_ENGINEER

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset(
            {ResearchActionType.RUN_EXPERIMENT, ResearchActionType.REPLICATE}
        )

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        del state, action
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        spec = invocation.context.experiments[0]
        used = {result.seed for result in invocation.context.results}
        seed = next(
            (value for value in spec.seeds if value not in used), spec.seeds[0]
        )
        job = ExperimentJob(
            spec_id=spec.id,
            command=(sys.executable, str(EXPERIMENT)),
            config={"metric": spec.metrics[0]},
            seed=seed,
            timeout_seconds=120.0,
            required_artifacts=("metrics.json",),
            id=job_id_for_attempt(invocation.attempt_id),
        )
        result = self._runner.run(job, invocation.attempt_id)
        return (ResultProposal(result=result, proposer="executor:local"),)


class CanaryAnalyst(ResearchRole):
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
            claim = invocation.context.claims[0]
            subject = claim.id
        else:
            subject = invocation.context.hypotheses[0].id
        assessment = EpistemicAssessment(
            subject_id=subject,
            verdict=verdict,
            method="canary-analyst:v1",
            evidence_ids=tuple(
                found.id for found in invocation.context.evidence
            ),
            scope=(
                invocation.context.experiments[0].procedure
                if invocation.context.experiments
                else ""
            ),
            rationale="; ".join(invocation.context.notes) or "read as recorded",
        )
        return (
            AssessmentProposal(assessment=assessment, proposer="canary:analyst")
        ,)


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


# -- the lab -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanaryLab:
    """Instruments with no outside world behind them."""

    def model_provider(self, _stage: StageName) -> ModelProvider:
        return CanaryModel()

    def literature_provider(self) -> LiteratureProvider:
        return CanaryLiterature()

    def runtime(self, request: RuntimeRequest) -> ResearchRuntime:
        return ResearchRuntime(
            # No verification component is wired, so governance is switched
            # off explicitly rather than inferred from its absence: the
            # ablated lab is a stated configuration, never an accident.
            config=RuntimeConfig(verification_governance_enabled=False),
            director=RuleBasedFrontierDirector(),
            roles={
                RoleName.RESEARCH_DIRECTOR: CanaryScientist(),
                RoleName.RESEARCH_ENGINEER: CanaryEngineer(
                    JournalingJobRunner(
                        inner=DirectJobRunner(
                            LocalExecutor(request.root / "runs")
                        ),
                        journal=request.journal,
                    )
                ),
                RoleName.RESULT_ANALYST: CanaryAnalyst(),
            },
            store=request.evidence,
            states=request.states,
            ledger=request.ledger,
            journal=request.journal,
            bundles=request.bundles,
        )


def lab() -> CanaryLab:
    return CanaryLab()

