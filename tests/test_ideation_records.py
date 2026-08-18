"""The ideation vocabulary: bounded directives, derived keys, stamped
candidate records, portfolio accounting, and the structural novelty
status. All records are synthetic; no network, no model."""

from __future__ import annotations

import pytest

from autonomous_research_lab.ideation.directive import IdeationDirective
from autonomous_research_lab.ideation.records import (
    CLAIM_KINDS,
    AddressedProblem,
    CandidateIdea,
    DataRequirement,
    DataStatus,
    IdeationRunRecord,
    NoveltyStatus,
    PortfolioReport,
    Prediction,
    ResourceEstimate,
    TargetedTheme,
    problem_key,
    theme_key,
)
from autonomous_research_lab.mapping.adequacy import SupportTier
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    ProblemKind,
    ThemeEra,
)

RUN = "idg_1"
STATEMENT = "Head-level mechanisms of in-context learning remain unclear."
THEME = "Mechanistic accounts"


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id=None,
        latency_seconds=0.25,
        input_tokens=100,
        output_tokens=200,
        repair_count=0,
    )


def _addressed() -> AddressedProblem:
    return AddressedProblem(
        key=problem_key(STATEMENT),
        statement=STATEMENT,
        kind=ProblemKind.OPEN_PROBLEM,
        tier=SupportTier.MULTI_SOURCE,
    )


def _targeted() -> TargetedTheme:
    return TargetedTheme(
        key=theme_key(THEME), name=THEME, era=ThemeEra.RECENT
    )


def _idea(**overrides: object) -> CandidateIdea:
    values: dict[str, object] = {
        "run_id": RUN,
        "title": "Reweighting under shift",
        "research_question": "Does head reweighting survive domain shift?",
        "proposed_contribution": "Evaluate reweighting out of domain.",
        "mechanism": "Reweighted heads carry task identity.",
        "hypothesis": "Reweighting keeps most of its gain out of domain.",
        "grounding": "The cited records report head reweighting gains.",
        "predictions": (
            Prediction(
                text="Out-of-domain accuracy stays within 5 points.",
                falsifier="Accuracy drops by more than 5 points.",
            ),
        ),
        "datasets": (
            DataRequirement(
                name="GLUE",
                status=DataStatus.EXISTING,
                role="in-domain evaluation",
            ),
        ),
        "metrics": ("accuracy",),
        "evaluation_protocol": "Train in domain, evaluate out of domain.",
        "baselines": ("LoRA",),
        "ablations": ("remove the reweighting scalars",),
        "resources": ResourceEstimate(
            compute="single GPU days",
            data="public benchmarks only",
            implementation="small patch to an adapter library",
        ),
        "risks": ("the effect may not transfer",),
        "cfp_alignment": "Targets the mechanisms topic.",
        "aligned_topics": ("mechanisms of in-context learning",),
        "uncertainty": "Grounded in two abstracts only.",
        "search_terms": ("attention head reweighting domain shift",),
        "addressed_problems": (_addressed(),),
        "targeted_themes": (_targeted(),),
        "cited_source_ids": ("lit_a", "lit_b"),
        "cited_recent": 1,
        "cited_foundational": 1,
        "cited_undated": 0,
        "provenance": _provenance(),
    }
    values.update(overrides)
    return CandidateIdea(**values)  # type: ignore[arg-type]


def _portfolio(**overrides: object) -> PortfolioReport:
    values: dict[str, object] = {
        "problems_total": 3,
        "problems_addressed": 1,
        "problems_unaddressed": 2,
        "unaddressed_statements": ("p2", "p3"),
        "addressed_multi_source": 1,
        "addressed_tentative": 0,
        "addressed_single_source_limitation": 0,
        "addressed_contradicted": 0,
        "candidates": 2,
        "distinct_sources_cited": 2,
        "themes_targeted": 1,
        "distinct_problem_sets": 2,
        "distinct_theme_sets": 1,
        "distinct_dataset_sets": 2,
        "distinct_metric_sets": 1,
    }
    values.update(overrides)
    return PortfolioReport(**values)  # type: ignore[arg-type]


def _run_record(**overrides: object) -> IdeationRunRecord:
    values: dict[str, object] = {
        "run_id": RUN,
        "directive_id": "idir_1",
        "assessment_id": "madq_1",
        "map_run_id": "map_1",
        "snapshot_id": "cfp_1",
        "direction_id": "dir_1",
        "candidate_ids": ("idea_1", "idea_2"),
        "refusal_justification": "",
        "diversity_rationale": "They target distinct problems.",
        "model_calls": 2,
        "input_tokens": 100,
        "output_tokens": 200,
        "portfolio": _portfolio(),
    }
    values.update(overrides)
    return IdeationRunRecord(**values)  # type: ignore[arg-type]


# -- the directive ------------------------------------------------------------


def test_a_directive_is_bounded_and_content_addressed() -> None:
    directive = IdeationDirective(
        assessment_id="madq_1", snapshot_id="cfp_1"
    )
    again = IdeationDirective(assessment_id="madq_1", snapshot_id="cfp_1")
    other = IdeationDirective(assessment_id="madq_2", snapshot_id="cfp_1")
    assert directive.id.startswith("idir_")
    assert directive.id == again.id
    assert directive.id != other.id


def test_an_unbounded_or_malformed_directive_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="assessment"):
        IdeationDirective(assessment_id="  ", snapshot_id="cfp_1")
    with pytest.raises(ValueError, match="snapshot"):
        IdeationDirective(assessment_id="madq_1", snapshot_id="")
    with pytest.raises(ValueError, match="max_candidates"):
        IdeationDirective(
            assessment_id="madq_1", snapshot_id="cfp_1", max_candidates=0
        )
    with pytest.raises(ValueError, match="max_candidates"):
        IdeationDirective(
            assessment_id="madq_1", snapshot_id="cfp_1", max_candidates=99
        )
    with pytest.raises(ValueError, match="max_model_calls"):
        IdeationDirective(
            assessment_id="madq_1", snapshot_id="cfp_1", max_model_calls=0
        )
    with pytest.raises(ValueError, match="max_model_calls"):
        IdeationDirective(
            assessment_id="madq_1", snapshot_id="cfp_1", max_model_calls=99
        )


# -- derived keys and claim kinds ---------------------------------------------


def test_problem_and_theme_keys_are_deterministic_content_identity() -> None:
    assert problem_key(STATEMENT).startswith("prob_")
    assert theme_key(THEME).startswith("thm_")
    assert problem_key(STATEMENT) == problem_key(STATEMENT)
    # Exactness is the contract: a restated sentence is a different key.
    assert problem_key(STATEMENT) != problem_key(STATEMENT + " ")
    assert theme_key(THEME) != theme_key(THEME.lower())


def test_every_model_authored_category_carries_a_claim_kind() -> None:
    assert CLAIM_KINDS["idea.mechanism"] == "candidate_conjecture"
    assert CLAIM_KINDS["idea.predictions"] == "candidate_prediction"
    assert CLAIM_KINDS["idea.falsifiers"] == "design_target"
    assert CLAIM_KINDS["idea.grounding"] == "grounding_synthesis"
    assert CLAIM_KINDS["direction.topics"] == "cfp_reported"
    assert CLAIM_KINDS["direction.scope"] == "extractor_synthesis"
    assert CLAIM_KINDS["run.refusal_justification"] == "grounding_synthesis"


# -- the stamped references ---------------------------------------------------


def test_an_addressed_problem_binds_key_and_statement() -> None:
    with pytest.raises(ValueError, match="does not derive"):
        AddressedProblem(
            key=problem_key("another statement"),
            statement=STATEMENT,
            kind=ProblemKind.OPEN_PROBLEM,
            tier=SupportTier.TENTATIVE,
        )
    with pytest.raises(ValueError, match="statement"):
        AddressedProblem(
            key=problem_key("  "),
            statement="  ",
            kind=ProblemKind.OPEN_PROBLEM,
            tier=SupportTier.TENTATIVE,
        )
    with pytest.raises(ValueError, match="does not derive"):
        TargetedTheme(key=theme_key("Other"), name=THEME, era=ThemeEra.BOTH)


def test_a_prediction_requires_its_falsifier() -> None:
    with pytest.raises(ValueError, match="falsifier"):
        Prediction(text="Accuracy improves.", falsifier="   ")
    with pytest.raises(ValueError, match="text"):
        Prediction(text="", falsifier="Accuracy drops.")


def test_a_data_requirement_names_its_data_and_role() -> None:
    with pytest.raises(ValueError, match="named"):
        DataRequirement(
            name=" ", status=DataStatus.EXISTING, role="evaluation"
        )
    with pytest.raises(ValueError, match="role"):
        DataRequirement(
            name="GLUE", status=DataStatus.NEW_REQUIREMENT, role=""
        )


# -- the candidate record -----------------------------------------------------


def test_a_candidate_idea_validates_its_grounding() -> None:
    with pytest.raises(ValueError, match="title"):
        _idea(title="   ")
    with pytest.raises(ValueError, match="addressed_problems"):
        _idea(addressed_problems=())
    with pytest.raises(ValueError, match="cited_source_ids"):
        _idea(cited_source_ids=(), cited_recent=0, cited_foundational=0)
    with pytest.raises(ValueError, match="at most once"):
        _idea(cited_source_ids=("lit_a", "lit_a"))
    with pytest.raises(ValueError, match="at most once"):
        _idea(addressed_problems=(_addressed(), _addressed()))
    with pytest.raises(ValueError, match="era mix"):
        _idea(cited_recent=2)
    with pytest.raises(ValueError, match="era mix"):
        _idea(cited_recent=-1, cited_foundational=3)


def test_novelty_is_structurally_unassessed() -> None:
    # The enum holds exactly one value: no generation-time record can
    # even express an assessed novelty status until Task 5D defines it.
    assert [status.value for status in NoveltyStatus] == ["unassessed"]
    assert _idea().novelty_status is NoveltyStatus.UNASSESSED


def test_candidate_idea_identity_is_deterministic() -> None:
    first = _idea()
    again = _idea()
    changed = _idea(grounding="A different grounding narrative.")
    assert first.id.startswith("idea_")
    assert first.id == again.id
    assert first.id != changed.id


# -- the portfolio and the run record -----------------------------------------


def test_a_portfolio_report_is_internally_consistent() -> None:
    with pytest.raises(ValueError, match="partition the inventory"):
        _portfolio(problems_addressed=2)
    with pytest.raises(ValueError, match="named"):
        _portfolio(unaddressed_statements=("p2",))
    with pytest.raises(ValueError, match="tier profile"):
        _portfolio(addressed_multi_source=0)
    with pytest.raises(ValueError, match="distinct_problem_sets"):
        _portfolio(distinct_problem_sets=3)
    # An honest refusal's portfolio: zero candidates, everything
    # unaddressed, every distinct-count zero.
    refusal = _portfolio(
        problems_addressed=0,
        problems_unaddressed=3,
        unaddressed_statements=("p1", "p2", "p3"),
        addressed_multi_source=0,
        candidates=0,
        distinct_sources_cited=0,
        themes_targeted=0,
        distinct_problem_sets=0,
        distinct_theme_sets=0,
        distinct_dataset_sets=0,
        distinct_metric_sets=0,
    )
    assert refusal.candidates == 0


def test_a_run_record_is_refusal_or_portfolio_never_both() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _run_record(refusal_justification="the records were too thin")
    with pytest.raises(ValueError, match="exactly one"):
        _run_record(candidate_ids=(), diversity_rationale="")
    with pytest.raises(ValueError, match="diversity rationale"):
        _run_record(
            candidate_ids=(),
            refusal_justification="the records were too thin",
        )
    with pytest.raises(ValueError, match="exactly the recorded"):
        _run_record(candidate_ids=("idea_1",))
    refusal = _run_record(
        candidate_ids=(),
        refusal_justification="the records were too thin",
        diversity_rationale="",
        portfolio=_portfolio(
            problems_addressed=0,
            problems_unaddressed=3,
            unaddressed_statements=("p1", "p2", "p3"),
            addressed_multi_source=0,
            candidates=0,
            distinct_sources_cited=0,
            themes_targeted=0,
            distinct_problem_sets=0,
            distinct_theme_sets=0,
            distinct_dataset_sets=0,
            distinct_metric_sets=0,
        ),
    )
    assert refusal.candidate_ids == ()


def test_a_run_record_is_content_addressed() -> None:
    first = _run_record()
    again = _run_record()
    changed = _run_record(output_tokens=999)
    assert first.id.startswith("irun_")
    assert first.id == again.id
    assert first.id != changed.id
