"""The deterministic candidate gates, rule by rule. All payloads are
synthetic; the problems, themes, and claim texts are static mapping
records. No network, no model."""

from __future__ import annotations

from autonomous_research_lab.ideation.gates import (
    check_candidates,
    claim_text_of,
)
from autonomous_research_lab.ideation.records import problem_key, theme_key
from autonomous_research_lab.mapping.brief import SourceEra
from autonomous_research_lab.mapping.records import (
    DatasetUse,
    ExtractionRecord,
    Limitation,
    LimitationKind,
    ProblemEntry,
    ProblemKind,
    SupportLocation,
)

CLAIMS_A = (
    "attention head reweighting improves accuracy by 2.4 points on glue\n"
    "adapter-based fine-tuning\naccuracy"
)
CLAIMS_B = (
    "kalman filter view of in-context learning\n"
    "evaluated only on synthetic regression with 3 seeds"
)
ACCESSIBLE = {"lit_a": CLAIMS_A, "lit_b": CLAIMS_B}

P_MULTI = ProblemEntry(
    statement=(
        "Head-level mechanisms of in-context learning remain unclear."
    ),
    kind=ProblemKind.OPEN_PROBLEM,
    grounding="Both papers report unresolved mechanisms across 14 tasks.",
    supporting_source_ids=("lit_a", "lit_b"),
)
P_LIMIT = ProblemEntry(
    statement=(
        "Kalman filtering results rest on synthetic regression only."
    ),
    kind=ProblemKind.DATA_LIMITATION,
    grounding=(
        "One paper reports evaluation on synthetic regression with 3 "
        "seeds only."
    ),
    supporting_source_ids=("lit_b",),
)
P_CONFLICT = ProblemEntry(
    statement="Theoretical accounts of in-context learning disagree.",
    kind=ProblemKind.CONFLICTING_FINDINGS,
    grounding="The two papers give incompatible accounts.",
    supporting_source_ids=("lit_a",),
    conflicting_source_ids=("lit_b",),
)
PROBLEMS = {
    problem_key(problem.statement): problem
    for problem in (P_MULTI, P_LIMIT, P_CONFLICT)
}
THEMES = {theme_key("Mechanistic accounts"): "Mechanistic accounts"}
TOPICS = (
    "mechanisms of in-context learning",
    "efficient adaptation and fine-tuning",
)
DIRECTION_TEXT = (
    "A call about mechanisms and adaptation.\n"
    + "\n".join(TOPICS)
    + "\nSubmissions are limited to 9 pages."
)


def _candidate(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "title": "Head reweighting under domain shift",
        "research_question": (
            "Does attention head reweighting keep its gains under "
            "distribution shift?"
        ),
        "proposed_contribution": (
            "An out-of-domain evaluation of head reweighting."
        ),
        "mechanism": (
            "Reweighted heads carry task identity that may be domain "
            "specific."
        ),
        "hypothesis": (
            "Reweighting keeps most of its in-domain gain out of domain."
        ),
        "grounding": (
            "One cited record reports reweighting improving accuracy by "
            "2.4 points on GLUE."
        ),
        "predictions": [
            {
                "text": "Out-of-domain accuracy drops by at most 5 points.",
                "falsifier": (
                    "Out-of-domain accuracy drops by more than 5 points."
                ),
            }
        ],
        "datasets": [
            {
                "name": "GLUE",
                "status": "existing",
                "role": "in-domain evaluation",
            }
        ],
        "metrics": ["accuracy"],
        "evaluation_protocol": (
            "Adapt in one domain, evaluate in a held-out domain."
        ),
        "baselines": ["full fine-tuning"],
        "ablations": ["remove the reweighting scalars"],
        "resources": {
            "compute": "a few GPU days",
            "data": "public benchmarks",
            "implementation": "a small adapter patch",
        },
        "risks": ["the effect may vanish under shift"],
        "cfp_alignment": "Addresses the mechanisms topic of the call.",
        "aligned_topics": ["mechanisms of in-context learning"],
        "uncertainty": (
            "Grounded in abstract-level claims from two papers only."
        ),
        "search_terms": ["attention head reweighting robustness"],
        "problem_keys": [problem_key(P_MULTI.statement)],
        "theme_keys": [theme_key("Mechanistic accounts")],
        "cited_source_ids": ["lit_a", "lit_b"],
    }
    values.update(overrides)
    return values


def _second_candidate() -> dict[str, object]:
    return _candidate(
        title="Reconciling filtering and compositional accounts",
        research_question=(
            "Which theoretical account predicts behaviour on shared "
            "tasks?"
        ),
        mechanism=(
            "The two accounts imply different error patterns on the "
            "same prompts."
        ),
        hypothesis=(
            "The accounts disagree measurably on at least one shared "
            "task family."
        ),
        grounding=(
            "The cited records give incompatible accounts of in-context "
            "learning."
        ),
        problem_keys=[problem_key(P_CONFLICT.statement)],
        aligned_topics=["efficient adaptation and fine-tuning"],
        datasets=[
            {
                "name": "a new shared diagnostic suite",
                "status": "new_requirement",
                "role": "head-to-head comparison",
            }
        ],
    )


def _payload(
    *candidates: dict[str, object],
    rationale: str = "The candidates target distinct problems.",
    refusal: str = "",
) -> dict[str, object]:
    return {
        "candidates": list(candidates),
        "diversity_rationale": rationale,
        "refusal_justification": refusal,
    }


def _check(payload: dict[str, object], **overrides: object) -> tuple[object, ...]:
    arguments: dict[str, object] = {
        "problems": PROBLEMS,
        "themes": THEMES,
        "direction_topics": TOPICS,
        "direction_text": DIRECTION_TEXT,
        "accessible": ACCESSIBLE,
        "max_candidates": 3,
    }
    arguments.update(overrides)
    return check_candidates(payload, **arguments)  # type: ignore[arg-type]


def _rules(rejections: tuple[object, ...]) -> set[str]:
    return {r.rule for r in rejections}  # type: ignore[attr-defined]


# -- the pass cases -----------------------------------------------------------


def test_a_grounded_candidate_payload_passes() -> None:
    assert _check(_payload(_candidate())) == ()
    assert _check(_payload(_candidate(), _second_candidate())) == ()


def test_an_honest_refusal_passes() -> None:
    refusal = _payload(
        rationale="",
        refusal=(
            "The mapped records report single-paper limitations only; no "
            "defensible candidate follows from them."
        ),
    )
    assert _check(refusal) == ()


def test_legitimate_unicode_survives_the_gate() -> None:
    accented = _candidate(
        baselines=["full fine-tuning", "Ondřej's café-style baseline"],
        uncertainty=(
            "Grounded in two abstracts; effect sizes ≥ the reported "
            "2.4 gain are conjecture."
        ),
    )
    assert _check(_payload(accented)) == ()


# -- refusal discipline -------------------------------------------------------


def test_zero_candidates_require_a_justification() -> None:
    assert _rules(_check(_payload(rationale=""))) == {"empty_finding"}


def test_a_refusal_carries_no_diversity_rationale() -> None:
    conflicted = _payload(
        rationale="still explaining diversity",
        refusal="The records are too thin.",
    )
    assert _rules(_check(conflicted)) == {"conflicting_record"}


def test_a_portfolio_and_a_refusal_cannot_coexist() -> None:
    both = _payload(_candidate(), refusal="but also no")
    assert "conflicting_record" in _rules(_check(both))


def test_a_refusal_justification_is_still_gated() -> None:
    corrupted = _payload(rationale="", refusal="too\x02thin")
    assert "corrupted_text" in _rules(_check(corrupted))
    invented = _payload(
        rationale="",
        refusal="Only 42 of the mapped problems are usable.",
    )
    assert "ungrounded_number" in _rules(_check(invented))


# -- bounds and blanks --------------------------------------------------------


def test_more_candidates_than_allowed_is_rejected() -> None:
    crowded = _payload(_candidate(), _second_candidate())
    assert "budget_violation" in _rules(_check(crowded, max_candidates=1))


def test_blank_fields_are_rejected() -> None:
    assert "empty_finding" in _rules(_check(_payload(_candidate(title=" "))))
    assert "empty_finding" in _rules(
        _check(_payload(_candidate(metrics=[])))
    )
    assert "empty_finding" in _rules(
        _check(_payload(_candidate(predictions=[])))
    )
    missing_resource = _candidate(
        resources={
            "compute": "",
            "data": "public",
            "implementation": "small",
        }
    )
    assert "empty_finding" in _rules(_check(_payload(missing_resource)))
    assert "empty_finding" in _rules(
        _check(_payload(_candidate(datasets=[])))
    )


# -- canonical references -----------------------------------------------------


def test_unknown_problem_keys_are_rejected_with_the_advice() -> None:
    shorthand = _candidate(problem_keys=["P1"])
    rejections = _check(_payload(shorthand))
    assert "unknown_problem" in _rules(rejections)
    details = " ".join(r.detail for r in rejections)  # type: ignore[attr-defined]
    assert "never an index" in details


def test_unknown_theme_keys_and_sources_and_topics_are_rejected() -> None:
    assert "unknown_theme" in _rules(
        _check(_payload(_candidate(theme_keys=["T1"])))
    )
    assert "unknown_source" in _rules(
        _check(_payload(_candidate(cited_source_ids=["lit_a", "lit_zz"])))
    )
    assert "unknown_topic" in _rules(
        _check(_payload(_candidate(aligned_topics=["something else"])))
    )


def test_missing_references_are_rejected() -> None:
    assert "missing_support" in _rules(
        _check(_payload(_candidate(problem_keys=[])))
    )
    assert "missing_support" in _rules(
        _check(_payload(_candidate(cited_source_ids=[])))
    )
    assert "missing_support" in _rules(
        _check(_payload(_candidate(aligned_topics=[])))
    )


def test_each_addressed_problem_needs_a_grounding_citation() -> None:
    # P_LIMIT is grounded by lit_b alone; citing only lit_a addresses it
    # with none of its own sources.
    decorative = _candidate(
        problem_keys=[problem_key(P_LIMIT.statement)],
        cited_source_ids=["lit_a"],
        grounding="One record reports reweighting gains of 2.4 points.",
        datasets=[
            {
                "name": "GLUE",
                "status": "existing",
                "role": "evaluation",
            }
        ],
    )
    assert "missing_support" in _rules(_check(_payload(decorative)))


# -- scoped number grounding --------------------------------------------------


def test_ungrounded_numbers_in_grounding_are_rejected() -> None:
    invented = _candidate(
        grounding=(
            "One cited record reports reweighting improving accuracy by "
            "88.8 points."
        )
    )
    assert _rules(_check(_payload(invented))) == {"ungrounded_number"}


def test_numbers_in_predictions_are_design_targets() -> None:
    targeted = _candidate(
        predictions=[
            {
                "text": (
                    "Reweighting recovers at least 80 percent of the "
                    "adapter gain out of domain."
                ),
                "falsifier": (
                    "Reweighting recovers less than 80 percent of the "
                    "gain on every evaluated shift."
                ),
            }
        ]
    )
    assert _check(_payload(targeted)) == ()


def test_numbers_grounded_in_problem_texts_pass() -> None:
    # "14" appears only in P_MULTI's grounding text, in no cited claim
    # text — the problem's own gated words are legitimate grounding.
    grounded = _candidate(
        grounding=(
            "The records report unresolved mechanisms across 14 tasks "
            "and a 2.4 point reweighting gain."
        ),
    )
    assert _check(_payload(grounded)) == ()
    unrelated = _candidate(
        problem_keys=[problem_key(P_CONFLICT.statement)],
        grounding=(
            "The records report unresolved mechanisms across 14 tasks."
        ),
    )
    # The same "14" is ungrounded once the candidate stops addressing
    # the problem whose grounding text carries it.
    assert "ungrounded_number" in _rules(_check(_payload(unrelated)))


def test_cfp_alignment_numbers_come_from_the_direction() -> None:
    grounded = _candidate(
        cfp_alignment=(
            "Fits the call, whose submissions are limited to 9 pages."
        )
    )
    assert _check(_payload(grounded)) == ()
    invented = _candidate(
        cfp_alignment="Fits all 12 tracks of the call."
    )
    assert "ungrounded_number" in _rules(_check(_payload(invented)))


# -- claim language -----------------------------------------------------------


def test_novelty_language_is_rejected() -> None:
    branded = _candidate(title="A novel reweighting method")
    assert "novelty_claim" in _rules(_check(_payload(branded)))
    sota = _candidate(
        proposed_contribution=(
            "Beat the state-of-the-art on every benchmark."
        )
    )
    assert "novelty_claim" in _rules(_check(_payload(sota)))
    pioneering = _candidate(
        uncertainty="This would be the first to test reweighting."
    )
    assert "novelty_claim" in _rules(_check(_payload(pioneering)))


def test_a_bare_first_is_not_a_novelty_claim() -> None:
    stepwise = _candidate(
        proposed_contribution=(
            "As a first step, evaluate reweighting out of domain."
        )
    )
    assert _check(_payload(stepwise)) == ()


def test_coverage_language_is_rejected() -> None:
    sweeping = _candidate(
        grounding=(
            "An exhaustive reading of the cited records shows a 2.4 "
            "point gain."
        )
    )
    assert "coverage_language" in _rules(_check(_payload(sweeping)))


def test_control_characters_are_rejected_with_an_actionable_message() -> None:
    corrupted = _candidate(mechanism="reweighting\x02composition")
    rejections = _check(_payload(corrupted))
    assert "corrupted_text" in _rules(rejections)
    details = " ".join(r.detail for r in rejections)  # type: ignore[attr-defined]
    assert "U+0002" in details


# -- falsifiability -----------------------------------------------------------


def test_circular_predictions_are_rejected() -> None:
    self_falsifying = _candidate(
        predictions=[
            {
                "text": "Accuracy stays within 5 points.",
                "falsifier": "Accuracy stays within 5 points.",
            }
        ]
    )
    assert "circular_finding" in _rules(_check(_payload(self_falsifying)))
    restated = _candidate(
        predictions=[
            {
                "text": (
                    "Reweighting keeps most of its in-domain gain out "
                    "of domain."
                ),
                "falsifier": "The gain disappears out of domain.",
            }
        ]
    )
    assert "circular_finding" in _rules(_check(_payload(restated)))


# -- duplicates and diversity -------------------------------------------------


def test_duplicate_candidates_are_rejected() -> None:
    twice = _payload(_candidate(), _candidate())
    rules = _rules(_check(twice))
    assert "duplicate_finding" in rules
    assert "insufficient_diversity" in rules


def test_superficial_renaming_is_insufficient_diversity() -> None:
    renamed = _candidate(
        title="Reweighting robustness, revisited",
        research_question="Is head reweighting robust to shift?",
        hypothesis="The reweighting gain survives domain shift.",
    )
    rules = _rules(_check(_payload(_candidate(), renamed)))
    assert "insufficient_diversity" in rules
    assert "duplicate_finding" not in rules


def test_duplicate_entries_within_one_candidate_are_rejected() -> None:
    assert "duplicate_finding" in _rules(
        _check(_payload(_candidate(cited_source_ids=["lit_a", "lit_a"])))
    )
    assert "duplicate_finding" in _rules(
        _check(
            _payload(_candidate(metrics=["accuracy", "Accuracy"]))
        )
    )


# -- dataset semantics --------------------------------------------------------


def test_existing_datasets_must_be_reported_by_cited_records() -> None:
    invented = _candidate(
        datasets=[
            {
                "name": "SuperBench-X",
                "status": "existing",
                "role": "evaluation",
            }
        ]
    )
    rejections = _check(_payload(invented))
    assert _rules(rejections) == {"unsupported_claim"}
    details = " ".join(r.detail for r in rejections)  # type: ignore[attr-defined]
    assert "new_requirement" in details


def test_a_new_requirement_needs_no_grounding() -> None:
    required = _candidate(
        datasets=[
            {
                "name": "a new cross-domain probe suite",
                "status": "new_requirement",
                "role": "out-of-domain evaluation",
            }
        ]
    )
    assert _check(_payload(required)) == ()


# -- the grounding surface ----------------------------------------------------


def test_claim_text_of_collects_the_extractions_claims() -> None:
    record = ExtractionRecord(
        run_id="map_1",
        source_id="lit_a",
        era=SourceEra.RECENT,
        access_level="abstract",
        support_location=SupportLocation.ABSTRACT,
        sufficient_support=True,
        insufficiency_reason="",
        methods=("Attention Head Reweighting",),
        datasets=(DatasetUse(name="GLUE", version="v2"),),
        metrics=("accuracy",),
        evaluation_protocols=(),
        baselines=("LoRA",),
        reported_results=("improves accuracy by 2.4 points",),
        limitations=(
            Limitation(
                text="evaluated on English only", kind=LimitationKind.DATA
            ),
        ),
        future_work=(),
        open_problems=(),
        provenance=None,
    )
    haystack = claim_text_of(record)
    for expected in (
        "attention head reweighting",
        "glue",
        "v2",
        "lora",
        "2.4",
        "evaluated on english only",
    ):
        assert expected in haystack


def test_the_gate_never_raises_on_taste() -> None:
    # A dull, thin, single-candidate portfolio addressing only a
    # single-source limitation passes: taste is not a rule. (The default
    # existing-GLUE dataset is replaced because lit_b reports no GLUE.)
    thin = _candidate(
        problem_keys=[problem_key(P_LIMIT.statement)],
        cited_source_ids=["lit_b"],
        grounding=(
            "The cited record reports evaluation on synthetic regression "
            "with 3 seeds only."
        ),
        datasets=[
            {
                "name": "a broader regression suite",
                "status": "new_requirement",
                "role": "evaluation beyond synthetic data",
            }
        ],
    )
    assert _check(_payload(thin)) == ()
