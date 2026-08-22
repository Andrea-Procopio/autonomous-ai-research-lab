"""Scripted instruments for the vision chain: stages one through six.

The canary pattern (see ``examples/canary_lab.py``), spoken in this
lab's vocabulary: an invented literature corpus about visual
representation learning, one shared abstract template so every quote a
gate re-finds is genuinely there, and one deterministic answer per
schema, computed from the request rather than from call order. The
candidate this corpus admits carries the prediction the vision lab's
templates measure verbatim::

    difference in linear probe accuracy:
        trained encoder minus randomly initialized encoder

so a walk from this brief reaches real training with zero network and
zero model spend. Every string relationship the deterministic gates
check — quotes in abstracts, arms in candidate records, terms in search
plans — is internal to this module, which is what makes it a fixture
rather than a mock: the stages run for real, and this merely answers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Final

from autonomous_research_lab.ideation.records import problem_key, theme_key
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureProvider,
    LiteratureQuery,
    LiteratureSource,
    RetrievedSearch,
)
from autonomous_research_lab.priorart.records import ComparisonDimension
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from autonomous_research_lab.selection.records import REVIEW_FIELDS

PROVIDER: Final = "vision-scripted"
RETRIEVED_AT: Final = "2026-08-18T12:00:00+00:00"

#: One vocabulary for every abstract in the corpus. The gates hold the
#: model to the cited text, so a fixture whose replies use words the
#: sources do not contain is a fixture the stages reject. Sharing one
#: vocabulary makes that impossible by construction.
_ABSTRACT: Final = (
    "We study {phrase} for visual representation learning. The method "
    "trains a small convolutional encoder and is evaluated with a "
    "linear probe on held-out images, where we report linear probe "
    "accuracy against random features. We compare against randomly "
    "initialized encoder and handcrafted feature baselines on public "
    "image benchmarks. Limitations: results degrade at larger image "
    "sizes, and evaluation covers public image benchmarks only."
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
        phrase=f"contrastive pretraining variant {index}",
        title=f"Contrastive Pretraining for Visual Representations ({index})",
        date=f"2026-0{index + 1}-01",
    )
    for index in range(1, 5)
)
FOUNDATIONAL = tuple(
    _source(
        f"foundational-{index}",
        phrase=f"supervised pretraining variant {index}",
        title=f"Supervised Pretraining for Transfer ({index})",
        date=f"201{index + 5}-06-01",
    )
    for index in range(1, 5)
)
UNCERTAIN = _source(
    "uncertain-1",
    phrase="representations in changing environments",
    title="Representations in Changing Environments",
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
        phrase=f"trained convolutional encoder approach {index}",
        title=f"Training Convolutional Encoders for Transfer ({index})",
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
    ("contrastive pretraining", (*RECENT, UNCERTAIN)),
    ("supervised pretraining", FOUNDATIONAL),
    ("linear probe limitations", (RECENT[0], RECENT[1], EXCLUDED)),
    ("trained convolutional encoder", PRIOR_WORK[0:3]),
    ("random feature baselines", PRIOR_WORK[3:6]),
    ("frozen feature probes", PRIOR_WORK[6:9]),
    ("untrained encoder baselines", PRIOR_WORK[9:12]),
    ("handcrafted feature baselines", PRIOR_WORK[0:3]),
    ("larger image sizes", PRIOR_WORK[3:6]),
)

#: One search term per prior-art family, chosen so the six searches
#: between them turn up the whole invented pile of near-work and overlap
#: enough for the saturation figure to mean something.
PRIOR_ART_TERMS: Final = (
    ("mechanism", "trained convolutional encoder"),
    ("problem_mechanism", "random feature baselines"),
    ("evaluation_setup", "frozen feature probes"),
    ("synonyms_legacy", "untrained encoder baselines"),
    ("competing_approaches", "handcrafted feature baselines"),
    ("recent", "larger image sizes"),
)

#: A phrase every abstract in the corpus contains, so a comparison can
#: quote it and the gate can find it where the comparison says it is.
SNIPPET: Final = "trains a small convolutional encoder and is evaluated with a"


def _retrieved(sources: Sequence[LiteratureSource]) -> RetrievedSearch:
    return RetrievedSearch(
        provider=PROVIDER,
        retrieved_at=RETRIEVED_AT,
        request_params={"search": "vision-scripted"},
        total_count=len(sources),
        pages_fetched=1,
        page_identifiers=("page-1",),
        rate_limit={},
        truncated=False,
        sources=tuple(sources),
    )


class VisionScriptedLiterature(LiteratureProvider):
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


class VisionScriptedModel(ModelProvider):
    """One deterministic answer per schema, computed from the request.

    ``engineer_reply`` — a complete implementation-proposal JSON string —
    lets the same scripted provider serve the model-backed engineer: the
    composition layer builds it from the catalog's template with the one
    slot filled by a trusted fixture body, so the whole chain runs with
    zero network while the engineer's contract stays the production one.
    """

    def __init__(self, engineer_reply: str | None = None) -> None:
        self._engineer_reply = engineer_reply

    @property
    def name(self) -> str:
        return PROVIDER

    def invoke(self, request: ModelRequest) -> ModelResponse:
        if request.schema is None:
            raise AssertionError("the scripted model answers structured calls only")
        if request.schema.name == "implementation_proposal":
            if self._engineer_reply is None:
                raise AssertionError(
                    "this scripted model was built without an engineer "
                    "reply; the lab factory supplies one"
                )
            return FakeModelProvider(
                [self._engineer_reply], name=PROVIDER
            ).invoke(request)
        answer = ANSWERERS.get(request.schema.name)
        if answer is None:
            raise AssertionError(
                f"the scripted model has no answer for schema "
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
                {"family": "recent", "text": "contrastive pretraining"},
                {"family": "foundational", "text": "supervised pretraining"},
                {
                    "family": "limitations_open_problems",
                    "text": "linear probe limitations",
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
            "methods": ["training a small convolutional encoder"],
            "datasets": [
                {
                    "name": "public image benchmarks",
                    "task": "visual representation learning",
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
            "metrics": ["linear probe accuracy"],
            "evaluation_protocols": [
                "evaluated with a linear probe on held-out images"
            ],
            "baselines": ["randomly initialized encoder", "handcrafted feature"],
            "reported_results": [
                "report linear probe accuracy against random features"
            ],
            "limitations": [
                {
                    "text": "results degrade at larger image sizes",
                    "kind": "generalization",
                }
            ],
            "future_work": [],
            "open_problems": ["robustness at larger image sizes"],
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
                "name": "Contrastive pretraining",
                "summary": (
                    "Recent work trains small convolutional encoders "
                    "judged by linear probes."
                ),
                "era": "recent",
                "source_ids": [source.id for source in recent],
            }
        )
    if older:
        themes.append(
            {
                "name": "Supervised pretraining foundations",
                "summary": (
                    "Earlier work pretrains with labels for transfer."
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
                    "name": "Frozen encoder evaluation",
                    "summary": "Encoders judged without further training.",
                    "source_ids": [source.id for source in shown[:2]],
                }
            ],
            "evaluation_practices": [
                {
                    "name": "Held-out linear probe accuracy",
                    "summary": "Linear probe accuracy measured on held-out images.",
                    "source_ids": [source.id for source in shown[:2]],
                }
            ],
            "relationships": (
                [
                    {
                        "kind": "builds_on",
                        "from_theme": "Contrastive pretraining",
                        "to_theme": "Supervised pretraining foundations",
                        "note": "contrastive work reuses supervised ideas",
                    }
                ]
                if len(themes) == 2
                else []
            ),
        }
    )


P_OPEN: Final = (
    "whether trained encoders keep their linear probe advantage on "
    "held-out images is unresolved"
)
P_CONFLICT: Final = (
    "whether label supervision is required for transferable "
    "representations is contested"
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
                    "the recent sources report linear probe accuracy "
                    "against random features"
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
                    "the foundational sources pretrain with labels; the "
                    "recent ones train contrastively without them"
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
                "A call for work on how visual representations are learned "
                "and on how much of their quality training explains."
            ),
            "topics": [
                "visual representation learning",
                "efficient pretraining",
                "evaluation with linear probes",
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
            "A held-out linear probe evaluation of the training effect."
        ),
        "mechanism": mechanism,
        "hypothesis": hypothesis,
        "grounding": (
            "The cited records report linear probe accuracy on held-out "
            "images."
        ),
        "predictions": [{"text": prediction, "falsifier": falsifier}],
        "datasets": [
            {
                "name": "public image benchmarks",
                "status": "existing",
                "role": "held-out probe evaluation",
            }
        ],
        "metrics": ["linear probe accuracy"],
        "evaluation_protocol": (
            "Train an encoder on one split, evaluate it with a linear probe "
            "on held-out images."
        ),
        "baselines": ["randomly initialized encoder", "handcrafted feature"],
        "ablations": ["remove the training"],
        "resources": {
            "compute": "one CPU core or one small GPU",
            "data": "a locally staged CIFAR-10 copy",
            "implementation": "a small torch training harness",
        },
        "risks": ["the effect may vanish at this scale"],
        "cfp_alignment": (
            "Addresses the call's interest in visual representation "
            "learning."
        ),
        "aligned_topics": ["visual representation learning"],
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
                "One candidate tests whether training explains the probe "
                "advantage; the other whether the label route matters."
            ),
            "candidates": [
                _candidate(
                    title="Trained versus random encoders",
                    question=(
                        "Does a trained encoder keep its linear probe "
                        "advantage over a randomly initialized encoder?"
                    ),
                    mechanism=(
                        "Training may concentrate class-relevant structure "
                        "into the encoder's features."
                    ),
                    hypothesis=(
                        "A trained encoder keeps most of its probe advantage "
                        "on held-out images."
                    ),
                    prediction=(
                        "Linear probe accuracy for the trained encoder stays "
                        "above the randomly initialized encoder."
                    ),
                    falsifier=(
                        "Linear probe accuracy for the trained encoder falls "
                        "below the randomly initialized encoder."
                    ),
                    problem=P_OPEN,
                    theme="Contrastive pretraining",
                ),
                _candidate(
                    title="Is label supervision required for transfer",
                    question=(
                        "Does label supervision change how well "
                        "representations transfer?"
                    ),
                    mechanism=(
                        "Label supervision and contrastive training may reach "
                        "the same features by different routes."
                    ),
                    hypothesis=(
                        "Supervised and contrastive encoders differ on "
                        "held-out images."
                    ),
                    prediction=(
                        "The two training routes rank differently on "
                        "held-out images."
                    ),
                    falsifier=(
                        "The two training routes rank identically on every "
                        "held-out image set."
                    ),
                    problem=P_CONFLICT,
                    theme="Supervised pretraining foundations",
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
            "overlap_features": ["both train convolutional encoders"],
            "material_differences": [
                "the prior work never compares against a randomly initialized encoder"
            ],
            "dimensions": [
                {
                    "dimension": dimension.value,
                    "candidate_position": (
                        "measures linear probe accuracy against a randomly "
                        "initialized encoder"
                    ),
                    "prior_work_position": (
                        "trains a convolutional encoder and reports linear "
                        "probe accuracy"
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
                "they test the training effect"
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
                "measure linear probe accuracy for the trained and the "
                "randomly initialized encoder"
            ),
            "required_capabilities": ["a staged CIFAR-10 copy and a torch harness"],
            "residual_risks": ["the effect may be specific to small encoders"],
        }
    )


#: For each candidate: its prediction, its falsifier, and the two arms
#: the prediction compares -- every one of them a phrase that appears
#: verbatim somewhere in that candidate's own record, because the
#: admission gate re-finds each of them there.
ARMS: Final = (
    (
        "Linear probe accuracy for the trained encoder stays above the "
        "randomly initialized encoder.",
        "Linear probe accuracy for the trained encoder falls below the "
        "randomly initialized encoder.",
        "trained encoder",
        "randomly initialized encoder",
    ),
    (
        "The two training routes rank differently on held-out images.",
        "The two training routes rank identically on every held-out "
        "image set.",
        "label supervision",
        "contrastive training",
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
                        "on held-out images from the evaluation split"
                    ),
                    "base_metric": "linear probe accuracy",
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


_HYPOTHESIS_LINE: Final = re.compile(
    r"- (hyp_[0-9a-f]{16}) \(question (q_[0-9a-f]{16})\)"
)
_ADMISSIBLE_LINE: Final = re.compile(r"- (ev_[0-9a-f]{16}) \[ADMISSIBLE\]")
_TEMPLATE_ID: Final = re.compile(r"\btmpl_[0-9a-f]{16}\b")
_METRICS_LINE: Final = re.compile(r"measurable metrics: (.+)")

_SHARPENED_FLOOR: Final = 0.01
"""The effect-size floor the scripted planner pre-registers. Chosen
below every contrast either trainer produces at the declared seeds, so
the sharpened prediction is a genuine claim that genuinely holds."""


def _planning_sentinels() -> dict[str, object]:
    """Every schema key with its sentinel; a decision overrides some."""
    return {
        "action": "stop",
        "rationale": "",
        "question_id": "",
        "evidence_ids": [],
        "hypothesis_id": "",
        "hypothesis_statement": "",
        "prediction_condition": "",
        "prediction_metric": "",
        "prediction_comparator": "none",
        "prediction_threshold": 0,
        "prediction_tolerance": 0,
        "prediction_expectation": "",
        "experiment_objective": "",
        "experiment_procedure": "",
        "experiment_metrics": [],
        "experiment_baselines": [],
        "experiment_controls": [],
        "experiment_seeds": [],
        "template_id": "",
        "target_experiment_id": "",
        "replication_seed": -1,
        "removed_component": "",
        "stop_reason": "none",
    }


def _planning_decision(request: ModelRequest) -> str:
    """The scripted planner: sharpen the admitted claim, then stop.

    Computed from the rendered context, never from call order. The first
    consultation arrives once the bootstrap arc has verified evidence on
    the record; it pre-registers a sharper bound on the admitted
    contrast — comparator and threshold the planner's own, run through
    the same trusted template at a fresh seed. The second consultation
    sees that sharpened prediction already in the context and stops with
    a typed reason. Both cite the admissible evidence, because the gate
    requires every decision to.
    """
    text = "\n".join(message.content for message in request.messages)
    evidence = [found.group(1) for found in _ADMISSIBLE_LINE.finditer(text)]
    seen: list[str] = []
    for found in evidence:
        if found not in seen:
            seen.append(found)
    hypothesis = _HYPOTHESIS_LINE.search(text)
    template = _TEMPLATE_ID.search(text)
    metrics_line = _METRICS_LINE.search(text)
    assert hypothesis is not None, (
        "the scripted planner needs a hypothesis line in the context"
    )
    payload = _planning_sentinels()
    payload["evidence_ids"] = seen
    # The reference checks apply to every action, a stop included: each
    # decision names the question it is about and cites its evidence.
    payload["question_id"] = hypothesis.group(2)

    if f" ge {_SHARPENED_FLOOR}" in text:
        payload["action"] = "stop"
        payload["stop_reason"] = "question_resolved"
        payload["rationale"] = (
            "the admitted contrast held at its sign and at the "
            "pre-registered floor; a further run would spend without "
            "changing the answer"
        )
        return json.dumps(payload)

    assert template and metrics_line, (
        "the scripted planner needs a template and its metrics in the "
        "rendered context"
    )
    metric_names = metrics_line.group(1).split(", ")
    primary = metric_names[0]
    payload.update(
        {
            "action": "new_experiment",
            "rationale": (
                "every verified run shows the trained encoder ahead by "
                "well over one point of probe accuracy; pre-register "
                "that margin as a floor and test it at a fresh seed"
            ),
            "hypothesis_id": hypothesis.group(1),
            "prediction_condition": (
                "on held-out images from the evaluation split, at a seed "
                "no earlier run used"
            ),
            "prediction_metric": primary,
            "prediction_comparator": "ge",
            "prediction_threshold": _SHARPENED_FLOOR,
            "prediction_expectation": (
                f"the trained encoder's probe advantage is at least "
                f"{_SHARPENED_FLOOR}"
            ),
            "experiment_objective": (
                "Hold the admitted contrast to a pre-registered "
                "effect-size floor."
            ),
            "experiment_procedure": (
                "Re-run the trusted template for this contrast at a "
                "fresh seed and compare the observed difference against "
                "the pre-registered floor."
            ),
            "experiment_metrics": metric_names,
            "experiment_baselines": ["the untrained comparison arm"],
            "experiment_controls": [
                "tiny-subset overfit control: the probe must fit a "
                "memorizable subset to at least 0.95 top-1"
            ],
            "experiment_seeds": [61],
            "template_id": template.group(0),
        }
    )
    return json.dumps(payload)


def _manuscript_prose(request: ModelRequest) -> str:
    """Prose computed from the shown packet, so the manuscript gates
    pass by construction: every number and citation the fixture writes
    is harvested verbatim from the packet markdown the author sent, and
    the template text around them is digit-free and heading-free. Where
    a pattern is absent the fixture writes numberless prose — the same
    honest fallback the instruction asks of a live model."""
    shown = request.messages[0].content if request.messages else ""
    citations = re.findall(r"\[(lits_[0-9a-f]{16})\]", shown)
    family = re.search(r"n=(\d+), consistent", shown)
    p_value = re.search(r"one-sided p=([0-9.eE+-]+),", shown)
    verdict = re.search(r"\*\*([A-Z]+)\*\* \(statistician", shown)
    cited = " ".join(f"[{name}]" for name in dict.fromkeys(citations))
    standing = (
        f"the recorded verdict is {verdict.group(1).lower()}"
        if verdict
        else "no claim was conclusively judged"
    )
    family_clause = (
        f"a replication family of n={family.group(1)} seeded runs, "
        f"reaching a one-sided exact sign test of p={p_value.group(1)}"
        if family and p_value
        else "the replication family the packet records"
    )
    payload = {
        "abstract": (
            f"We report the registered contrast exactly as the evidence "
            f"packet records it: {standing}, over {family_clause}. Every "
            f"figure in this manuscript is assembled by trusted code "
            f"from the verified record."
        ),
        "introduction": (
            f"Prior work motivates the registered question {cited}. "
            f"This report states only what the packet's verified record "
            f"supports, in the recorded verdict's own words."
        ),
        "method_narrative": (
            "The experiment ran the pre-registered trusted template at "
            "the recorded seeds; the registration below is rendered "
            "from the record, not restated."
        ),
        "discussion": (
            f"Across {family_clause}, {standing}. We draw no conclusion "
            f"beyond the recorded verdicts and figures."
        ),
        "limitations": (
            "The evidence covers the recorded seeds, dataset, and "
            "training budget only; no claim extends past the packet's "
            "stated scope, and the underpowered families remain exactly "
            "as judged."
        ),
    }
    return json.dumps(payload)


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
    "planning_decision": _planning_decision,
    "manuscript_prose": _manuscript_prose,
}
