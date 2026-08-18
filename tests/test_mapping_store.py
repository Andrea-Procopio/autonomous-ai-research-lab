"""The mapping store: write-once semantics, per-run consistency, durable
reload with identity recomputation, and tamper detection. All payloads
are synthetic; no network, no model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.mapping.brief import (
    QueryFamily,
    ResearchBrief,
    SourceEra,
)
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    CoverageReport,
    DatasetAvailability,
    DatasetUse,
    ExtractionRecord,
    FieldMapRecord,
    GroupEntry,
    Limitation,
    LimitationKind,
    MappingRunRecord,
    ProblemEntry,
    ProblemInventoryRecord,
    ProblemKind,
    QueryExecution,
    RelationshipKind,
    ScreeningDecision,
    ScreeningRecord,
    SupportLocation,
    ThemeEntry,
    ThemeEra,
    ThemeRelationship,
)
from autonomous_research_lab.mapping.store import (
    MappingConflictError,
    MappingIntegrityError,
    MappingStore,
)

BRIEF = ResearchBrief(
    topic="in-context learning",
    cutoff_date="2026-08-18",
    recent_window_start="2026-01-01",
    workshop_hints=("efficient adaptation",),
)


def _provenance(response_id: str = "mcall_1") -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_1",
        response_id=response_id,
        provider="fake",
        requested_model="m",
        served_model="m-served",
        provider_request_id="req-7",
        latency_seconds=0.25,
        input_tokens=100,
        output_tokens=50,
        repair_count=1,
    )


def _screening(
    source_id: str = "lit_1", response_id: str = "mcall_1"
) -> ScreeningRecord:
    return ScreeningRecord(
        run_id="map_1",
        source_id=source_id,
        decision=ScreeningDecision.RELEVANT,
        reason="on topic",
        provenance=_provenance(response_id),
    )


def _extraction(
    source_id: str = "lit_1", run_id: str = "map_1"
) -> ExtractionRecord:
    return ExtractionRecord(
        run_id=run_id,
        source_id=source_id,
        era=SourceEra.RECENT,
        access_level="abstract",
        support_location=SupportLocation.ABSTRACT,
        sufficient_support=True,
        insufficiency_reason="",
        methods=("prompt adaptation",),
        datasets=(
            DatasetUse(
                name="GLUE",
                task="language understanding",
                version="v2",
                availability=DatasetAvailability.PUBLIC,
            ),
        ),
        metrics=("accuracy",),
        evaluation_protocols=(),
        baselines=("fine-tuning",),
        reported_results=("reaches 88.5 accuracy",),
        limitations=(
            Limitation(
                text="degrades under shift",
                kind=LimitationKind.GENERALIZATION,
            ),
        ),
        future_work=(),
        open_problems=("robustness under shift",),
        provenance=_provenance(),
    )


def _field_map() -> FieldMapRecord:
    return FieldMapRecord(
        run_id="map_1",
        brief_id=BRIEF.id,
        themes=(
            ThemeEntry(
                name="Prompt adaptation",
                summary="Recent adaptation methods.",
                era=ThemeEra.RECENT,
                source_ids=("lit_1",),
            ),
        ),
        approaches=(
            GroupEntry(
                name="Gradient-free",
                summary="No weight updates.",
                source_ids=("lit_1",),
            ),
        ),
        evaluation_practices=(),
        relationships=(
            ThemeRelationship(
                kind=RelationshipKind.BUILDS_ON,
                from_theme="Prompt adaptation",
                to_theme="Prompt adaptation base",
                note="n",
            ),
        ),
        recent_source_ids=("lit_1",),
        foundational_source_ids=(),
        undated_source_ids=(),
        provenance=_provenance(),
    )


def _inventory() -> ProblemInventoryRecord:
    return ProblemInventoryRecord(
        run_id="map_1",
        brief_id=BRIEF.id,
        problems=(
            ProblemEntry(
                statement="robustness under shift is open",
                kind=ProblemKind.OPEN_PROBLEM,
                grounding="reported degradation",
                supporting_source_ids=("lit_1",),
            ),
        ),
        provenance=_provenance(),
    )


def _run_record() -> MappingRunRecord:
    return MappingRunRecord(
        run_id="map_1",
        brief_id=BRIEF.id,
        query_execution_ids=("qrun_1",),
        screening_ids=("scrn_1",),
        extraction_ids=("extr_1",),
        field_map_id="fmap_1",
        inventory_id="pinv_1",
        model_calls=6,
        input_tokens=1000,
        output_tokens=400,
        coverage=CoverageReport(
            queries_executed=3,
            total_retrieved=6,
            unique_sources=5,
            screened=5,
            screening_truncated=0,
            relevant=3,
            excluded=1,
            uncertain=1,
            abstract_level=4,
            metadata_level=1,
            extraction_eligible=3,
            extracted=3,
            extraction_truncated=0,
            insufficient_support=1,
            saturation=0.5,
        ),
    )


def test_every_record_kind_round_trips_with_recomputed_identity(
    tmp_path: Path,
) -> None:
    store = MappingStore(tmp_path)
    brief = store.record_brief(BRIEF)
    execution = store.record_query_execution(
        QueryExecution(
            run_id="map_1",
            family=QueryFamily.RECENT,
            text="in-context learning",
            from_date="2026-01-01",
            to_date="2026-08-18",
            query_fingerprint="litq_1",
            search_record_id="lits_1",
            retrieved=3,
            new_unique=3,
            from_cache=False,
        )
    )
    screening = store.record_screening(_screening())
    extraction = store.record_extraction(_extraction())
    field_map = store.record_field_map(_field_map())
    inventory = store.record_inventory(_inventory())
    run = store.record_run(_run_record())

    fresh = MappingStore(tmp_path)
    assert fresh.get_brief(brief.id) == brief
    assert fresh.get_query_execution(execution.id) == execution
    assert fresh.get_screening(screening.id) == screening
    assert fresh.get_extraction(extraction.id) == extraction
    assert fresh.get_field_map(field_map.id) == field_map
    assert fresh.get_inventory(inventory.id) == inventory
    assert fresh.get_run(run.id) == run
    assert fresh.runs() == (run,)
    # Identical re-recording of reloaded records is a no-op everywhere.
    assert fresh.record_extraction(extraction) == extraction
    assert fresh.record_run(run) == run


def test_deterministic_insufficiency_records_round_trip(
    tmp_path: Path,
) -> None:
    """provenance=None (no model call) must survive serialization."""
    store = MappingStore(tmp_path)
    record = ExtractionRecord(
        run_id="map_1",
        source_id="lit_meta",
        era=SourceEra.UNDATED,
        access_level="metadata",
        support_location=SupportLocation.TITLE,
        sufficient_support=False,
        insufficiency_reason="metadata-only access",
        methods=(),
        datasets=(),
        metrics=(),
        evaluation_protocols=(),
        baselines=(),
        reported_results=(),
        limitations=(),
        future_work=(),
        open_problems=(),
        provenance=None,
    )
    store.record_extraction(record)
    assert MappingStore(tmp_path).get_extraction(record.id) == record


def test_write_once_refuses_different_content_under_one_id(
    tmp_path: Path,
) -> None:
    store = MappingStore(tmp_path)
    record = store.record_screening(_screening())
    path = tmp_path / "screenings" / f"{record.id}.json"
    payload = json.loads(path.read_text())
    payload["reason"] = "silently improved reason"
    path.write_text(json.dumps(payload))

    with pytest.raises(MappingIntegrityError, match="re-derives"):
        MappingStore(tmp_path).get_screening(record.id)


def test_one_run_holds_one_verdict_per_source(tmp_path: Path) -> None:
    store = MappingStore(tmp_path)
    store.record_screening(_screening())
    second_opinion = ScreeningRecord(
        run_id="map_1",
        source_id="lit_1",
        decision=ScreeningDecision.EXCLUDED,
        reason="changed my mind",
        provenance=_provenance("mcall_2"),
    )
    with pytest.raises(MappingConflictError, match="already screened"):
        store.record_screening(second_opinion)

    store.record_extraction(_extraction())
    drifted = _extraction()
    drifted = ExtractionRecord(
        run_id="map_1",
        source_id="lit_1",
        era=SourceEra.RECENT,
        access_level="abstract",
        support_location=SupportLocation.ABSTRACT,
        sufficient_support=True,
        insufficiency_reason="",
        methods=("a different method",),
        datasets=(),
        metrics=(),
        evaluation_protocols=(),
        baselines=(),
        reported_results=(),
        limitations=(),
        future_work=(),
        open_problems=(),
        provenance=_provenance("mcall_3"),
    )
    with pytest.raises(MappingConflictError, match="already extracted"):
        store.record_extraction(drifted)

    # A different run screening the same source is fine.
    other_run = _extraction(run_id="map_2")
    store.record_extraction(other_run)


def test_rejected_payloads_are_preserved_as_data(tmp_path: Path) -> None:
    store = MappingStore(tmp_path)
    store.preserve_rejected(
        run_id="map_1",
        stage="extraction",
        reasons=(("ungrounded_number", "97.1 is not in the text"),),
        request_fingerprint="mreq_9",
        response_id="mcall_9",
        payload={"source_id": "lit_1", "reported_results": ["97.1"]},
        repair=0,
    )
    (rejected,) = store.rejected()
    assert rejected["stage"] == "extraction"
    assert rejected["repair"] == 0
    reasons = rejected["reasons"]
    assert isinstance(reasons, list)
    assert reasons[0]["rule"] == "ungrounded_number"
    assert rejected["response_id"] == "mcall_9"
