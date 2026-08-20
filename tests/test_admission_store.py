"""The admission store: write-once, tamper-loud, one admission ever.

Beyond the house write-once rules, two uniqueness scans are pinned here:
one admission record per admission directive, and one per selection run
record — ever. The admitted state snapshot is part of the write-once
artifact set: the only public accessor loads the record first and the
state through it, so a state is never exposed without its record and a
record whose snapshot is gone fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.admission import (
    AdmissionConflictError,
    AdmissionDirective,
    AdmissionIntegrityError,
    AdmissionRecord,
    AdmissionStore,
    GroundedSupport,
    OperationalPrediction,
    Requirement,
    RequirementSource,
    SupportSource,
)
from autonomous_research_lab.admission.records import MECHANICAL_READING
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.mapping.records import CallProvenance
from autonomous_research_lab.persistence.state_store import SnapshotError


def _directive(**overrides: object) -> AdmissionDirective:
    values: dict[str, object] = {
        "selection_run_record_id": "srun_0000000000000001",
        "scheduling_requirement": "Batch-scheduled execution.",
        "job_duration_requirement": "Jobs bounded to two days.",
        "checkpoint_requirement": "Checkpoint and resume required.",
    }
    values.update(overrides)
    return AdmissionDirective(**values)  # type: ignore[arg-type]


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_0000000000000001",
        response_id="mcall_000000000000001",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id="req-1",
        latency_seconds=0.1,
        input_tokens=100,
        output_tokens=50,
        repair_count=1,
    )


def _operational() -> OperationalPrediction:
    return OperationalPrediction(
        prediction_text="Ablating top heads drops accuracy more.",
        condition="Few-shot classification with trained scalars.",
        base_metric="few-shot accuracy",
        expected_higher_arm="accuracy drop from ablating top heads",
        expected_lower_arm="accuracy drop from ablating bottom heads",
        contrary_observation="similar drop for both ablations",
        support=(
            GroundedSupport(
                source=SupportSource.CANDIDATE,
                field_path="predictions[0].text",
                quote="ablating them drops accuracy",
            ),
            GroundedSupport(
                source=SupportSource.DIRECTION,
                field_path="scope",
                quote="post-training mechanisms",
            ),
        ),
    )


def _state() -> ResearchState:
    question = ResearchQuestion(text="Do scalars select induction heads?")
    hypothesis = Hypothesis(statement="They do.", question_id=question.id)
    return ResearchState(
        objective="Measure the correlation.",
        questions=(question,),
        hypotheses=(hypothesis,),
    )


def _record(state: ResearchState, **overrides: object) -> AdmissionRecord:
    values: dict[str, object] = {
        "run_id": "adm_0000000000000001",
        "directive_id": _directive().id,
        "selection_run_record_id": "srun_0000000000000001",
        "selection_run_id": "sel_0000000000000001",
        "selection_directive_id": "sdir_0000000000000001",
        "prior_art_run_record_id": "prun_0000000000000001",
        "prior_art_run_id": "pac_0000000000000001",
        "selected_prior_art_assessment_id": "paa_0000000000000001",
        "ideation_run_record_id": "irun_0000000000000001",
        "ideation_run_id": "idg_0000000000000001",
        "direction_id": "dir_0000000000000001",
        "snapshot_id": "cfp_0000000000000001",
        "map_run_id": "map_0000000000000001",
        "map_assessment_id": "madq_000000000000001",
        "selected_candidate_id": "idea_0000000000000001",
        "operational_predictions": (_operational(),),
        "measurements": ("few-shot accuracy",),
        "controls": ("random head ablation",),
        "comparison_targets": ("LoRA fine-tuning",),
        "evaluation_protocol": "Seeded runs across tasks and seeds.",
        "inherited_requirements": (
            Requirement(
                source=RequirementSource.CANDIDATE_RESOURCES,
                record_id="idea_0000000000000001",
                field_path="resources.compute",
                quote="~100 GPU-hours on a single A100",
            ),
        ),
        "operator_requirements": (
            Requirement(
                source=RequirementSource.ADMISSION_DIRECTIVE,
                record_id=_directive().id,
                field_path="scheduling_requirement",
                quote="Batch-scheduled execution.",
            ),
        ),
        "mechanical_reading": MECHANICAL_READING,
        "question_id": "q_00000000000000001",
        "hypothesis_id": "hyp_0000000000000001",
        "prediction_ids": ("pred_000000000000001",),
        "state_id": state.id,
        "provenance": _provenance(),
        "model_calls": 2,
        "input_tokens": 100,
        "output_tokens": 50,
    }
    values.update(overrides)
    return AdmissionRecord(**values)  # type: ignore[arg-type]


def test_every_kind_round_trips_identically(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    directive = store.record_directive(_directive())
    state = _state()
    store.persist_state(state)
    record = store.record_admission(_record(state))

    assert store.get_directive(directive.id) == directive
    assert store.get_record(record.id) == record
    loaded_record, loaded_state = store.get_admitted_state(record.id)
    assert loaded_record == record
    assert loaded_state == state


def test_identical_rerecording_is_a_noop(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    state = _state()
    store.persist_state(state)
    record = _record(state)
    store.record_admission(record)
    store.record_admission(_record(state))
    assert len(store.records()) == 1
    assert record.id == _record(state).id


def test_a_conflicting_rewrite_is_refused(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    state = _state()
    record = _record(state)
    store.record_admission(record)
    path = tmp_path / "records" / f"{record.id}.json"
    payload = json.loads(path.read_text())
    payload["evaluation_protocol"] = "doctored"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    with pytest.raises(AdmissionConflictError, match="never rewritten"):
        store.record_admission(record)


def test_a_tampered_record_fails_loudly_on_fresh_reload(
    tmp_path: Path,
) -> None:
    store = AdmissionStore(tmp_path)
    state = _state()
    record = _record(state)
    store.record_admission(record)
    path = tmp_path / "records" / f"{record.id}.json"
    payload = json.loads(path.read_text())
    payload["evaluation_protocol"] = "doctored praise"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    fresh = AdmissionStore(tmp_path)
    with pytest.raises(AdmissionIntegrityError, match="no longer matches"):
        fresh.get_record(record.id)


def test_missing_records_return_none(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    assert store.get_directive("adir_missing") is None
    assert store.get_record("arun_missing") is None
    assert store.record_for_directive("adir_missing") is None
    assert store.record_for_selection_run("srun_missing") is None


def test_one_admission_per_directive_ever(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    state = _state()
    store.record_admission(_record(state))
    second = _record(
        state,
        run_id="adm_0000000000000002",
        selection_run_record_id="srun_0000000000000002",
    )
    with pytest.raises(AdmissionConflictError, match="already produced"):
        store.record_admission(second)


def test_an_admitted_selection_is_never_silently_replaced(
    tmp_path: Path,
) -> None:
    store = AdmissionStore(tmp_path)
    state = _state()
    first = store.record_admission(_record(state))
    second = _record(
        state,
        run_id="adm_0000000000000002",
        directive_id=_directive(scheduling_requirement="Different.").id,
    )
    with pytest.raises(
        AdmissionConflictError, match="never silently replaced"
    ) as caught:
        store.record_admission(second)
    assert first.id in str(caught.value)


def test_a_state_is_never_exposed_without_its_record(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    state = _state()
    store.persist_state(state)
    # The snapshot exists, but no admission record names it: the public
    # accessor refuses — "no record" means "not admitted".
    with pytest.raises(AdmissionIntegrityError, match=r"never\s+exposed"):
        store.get_admitted_state("arun_0000000000000001")


def test_a_record_without_its_snapshot_fails_loudly(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    state = _state()
    path = store.persist_state(state)
    record = store.record_admission(_record(state))
    path.unlink()

    with pytest.raises(SnapshotError, match="no snapshot"):
        store.get_admitted_state(record.id)


def test_snapshots_live_under_the_admission_root(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    path = store.persist_state(_state())
    assert path.parent == tmp_path / "states"


def test_rejected_payloads_are_preserved_verbatim(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    store.preserve_rejected(
        run_id="adm_0000000000000001",
        stage="operationalization",
        reasons=(("missing_support", "arm not re-found"),),
        request_fingerprint="mreq_0000000000000001",
        response_id="mcall_000000000000001",
        payload={"operational_predictions": []},
        repair=0,
    )
    preserved = store.rejected()
    assert len(preserved) == 1
    assert preserved[0]["stage"] == "operationalization"
    assert preserved[0]["reasons"] == [
        {"rule": "missing_support", "detail": "arm not re-found"}
    ]
    assert preserved[0]["payload"] == {"operational_predictions": []}


def test_lookup_by_directive_and_selection_run(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    state = _state()
    record = store.record_admission(_record(state))
    by_directive = store.record_for_directive(record.directive_id)
    by_selection = store.record_for_selection_run(
        record.selection_run_record_id
    )
    assert by_directive == record
    assert by_selection == record
