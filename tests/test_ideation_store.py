"""The write-once ideation store: recomputed identity on load, loud
tamper detection, one direction and one run record per run, rejected
payloads preserved as data. No network, no model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.ideation.direction import (
    CfpSnapshot,
    DirectionRecord,
)
from autonomous_research_lab.ideation.directive import IdeationDirective
from autonomous_research_lab.ideation.records import (
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
from autonomous_research_lab.ideation.store import (
    IdeationConflictError,
    IdeationIntegrityError,
    IdeationStore,
)
from autonomous_research_lab.mapping.adequacy import SupportTier
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    ProblemKind,
    ThemeEra,
)

RUN = "idg_1"

SNAPSHOT = CfpSnapshot(
    source_url="https://example.org/cfp",
    supplied_at="2026-08-19T10:00:00",
    text="Workshop on in-context learning.\n- mechanisms\n",
)

DIRECTIVE = IdeationDirective(assessment_id="madq_1", snapshot_id=SNAPSHOT.id)


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


def _direction(scope: str = "About mechanisms.") -> DirectionRecord:
    return DirectionRecord(
        run_id=RUN,
        snapshot_id=SNAPSHOT.id,
        scope=scope,
        topics=("mechanisms",),
        constraints=(),
        relevant_dates=(),
        provenance=_provenance(),
    )


STATEMENT = "Mechanisms remain unclear."


def _idea(title: str = "Reweighting under shift") -> CandidateIdea:
    return CandidateIdea(
        run_id=RUN,
        title=title,
        research_question="Does reweighting survive shift?",
        proposed_contribution="Evaluate it out of domain.",
        mechanism="Heads carry task identity.",
        hypothesis="The gain mostly survives.",
        grounding="The cited record reports reweighting gains.",
        predictions=(
            Prediction(
                text="Accuracy stays close.",
                falsifier="Accuracy collapses.",
            ),
        ),
        datasets=(
            DataRequirement(
                name="GLUE",
                status=DataStatus.EXISTING,
                role="evaluation",
            ),
        ),
        metrics=("accuracy",),
        evaluation_protocol="Adapt in domain, test out of domain.",
        baselines=("LoRA",),
        ablations=("remove scalars",),
        resources=ResourceEstimate(
            compute="single GPU days",
            data="public benchmarks",
            implementation="small patch",
        ),
        risks=("no transfer",),
        cfp_alignment="Targets the mechanisms topic.",
        aligned_topics=("mechanisms",),
        uncertainty="Two abstracts only.",
        search_terms=("head reweighting shift",),
        addressed_problems=(
            AddressedProblem(
                key=problem_key(STATEMENT),
                statement=STATEMENT,
                kind=ProblemKind.OPEN_PROBLEM,
                tier=SupportTier.MULTI_SOURCE,
            ),
        ),
        targeted_themes=(
            TargetedTheme(
                key=theme_key("Mechanisms"),
                name="Mechanisms",
                era=ThemeEra.RECENT,
            ),
        ),
        cited_source_ids=("lit_a",),
        cited_recent=1,
        cited_foundational=0,
        cited_undated=0,
        provenance=_provenance(),
    )


def _run_record(idea: CandidateIdea, **overrides: object) -> IdeationRunRecord:
    values: dict[str, object] = {
        "run_id": RUN,
        "directive_id": DIRECTIVE.id,
        "assessment_id": "madq_1",
        "map_run_id": "map_1",
        "snapshot_id": SNAPSHOT.id,
        "direction_id": _direction().id,
        "candidate_ids": (idea.id,),
        "refusal_justification": "",
        "diversity_rationale": "One candidate, one problem.",
        "model_calls": 2,
        "input_tokens": 100,
        "output_tokens": 200,
        "portfolio": PortfolioReport(
            problems_total=1,
            problems_addressed=1,
            problems_unaddressed=0,
            unaddressed_statements=(),
            addressed_multi_source=1,
            addressed_tentative=0,
            addressed_single_source_limitation=0,
            addressed_contradicted=0,
            candidates=1,
            distinct_sources_cited=1,
            themes_targeted=1,
            distinct_problem_sets=1,
            distinct_theme_sets=1,
            distinct_dataset_sets=1,
            distinct_metric_sets=1,
        ),
    }
    values.update(overrides)
    return IdeationRunRecord(**values)  # type: ignore[arg-type]


def test_every_record_kind_round_trips_with_recomputed_identity(
    tmp_path: Path,
) -> None:
    store = IdeationStore(tmp_path / "ideation")
    idea = _idea()
    direction = _direction()
    run_record = _run_record(idea)
    store.record_snapshot(SNAPSHOT)
    store.record_directive(DIRECTIVE)
    store.record_direction(direction)
    store.record_idea(idea)
    store.record_run(run_record)
    # Identical re-recording is a no-op, not a conflict.
    store.record_idea(idea)
    store.record_run(run_record)

    fresh = IdeationStore(tmp_path / "ideation")
    assert fresh.get_snapshot(SNAPSHOT.id) == SNAPSHOT
    assert fresh.get_directive(DIRECTIVE.id) == DIRECTIVE
    assert fresh.get_direction(direction.id) == direction
    assert fresh.get_idea(idea.id) == idea
    assert fresh.get_run(run_record.id) == run_record
    assert fresh.ideas() == (idea,)
    assert fresh.runs() == (run_record,)
    assert fresh.runs_for_assessment("madq_1") == (run_record,)
    assert fresh.runs_for_assessment("madq_other") == ()
    assert fresh.get_idea("idea_missing") is None


def test_a_tampered_record_fails_loudly_on_load(tmp_path: Path) -> None:
    store = IdeationStore(tmp_path / "ideation")
    idea = store.record_idea(_idea())
    path = tmp_path / "ideation" / "ideas" / f"{idea.id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["title"] = "A quietly improved title"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IdeationIntegrityError, match="re-derives"):
        IdeationStore(tmp_path / "ideation").get_idea(idea.id)


def test_write_once_refuses_different_content_under_one_id(
    tmp_path: Path,
) -> None:
    store = IdeationStore(tmp_path / "ideation")
    idea = store.record_idea(_idea())
    path = tmp_path / "ideation" / "ideas" / f"{idea.id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["uncertainty"] = "rewritten on disk"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IdeationConflictError, match="never rewritten"):
        store.record_idea(_idea())


def test_one_run_holds_one_direction_and_one_run_record(
    tmp_path: Path,
) -> None:
    store = IdeationStore(tmp_path / "ideation")
    store.record_direction(_direction())
    with pytest.raises(IdeationConflictError, match="second reading"):
        store.record_direction(_direction(scope="A different reading."))
    idea = store.record_idea(_idea())
    store.record_run(_run_record(idea))
    with pytest.raises(IdeationConflictError, match="second account"):
        store.record_run(_run_record(idea, output_tokens=999))


def test_rejected_payloads_are_preserved_as_data(tmp_path: Path) -> None:
    store = IdeationStore(tmp_path / "ideation")
    store.preserve_rejected(
        run_id=RUN,
        stage="candidates",
        reasons=(("unknown_problem", "cited P1"),),
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        payload={"candidates": [{"title": "x"}]},
        repair=0,
    )
    (rejected,) = IdeationStore(tmp_path / "ideation").rejected()
    assert rejected["stage"] == "candidates"
    assert rejected["reasons"] == [
        {"rule": "unknown_problem", "detail": "cited P1"}
    ]
    assert rejected["repair"] == 0
    assert rejected["request_fingerprint"] == "mreq_1"
