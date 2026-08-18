"""The planning store: write-once decisions, durable rejections, dispatch."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from autonomous_research_lab.runtime.planning_store import (
    PlanningAction,
    PlanningConflictError,
    PlanningIntegrityError,
    PlanningRecord,
    PlanningStore,
    StopReason,
)


def _record(**overrides: object) -> PlanningRecord:
    values: dict[str, object] = {
        "invocation_id": "inv_1",
        "action": PlanningAction.NEW_EXPERIMENT,
        "question_id": "q_1",
        "rationale": "the baseline is verified; probe robustness",
        "evidence_ids": ("ev_1",),
        "hypothesis_id": "hyp_1",
        "prediction_id": "pred_1",
        "spec_id": "exp_1",
        "template_id": "tmpl_1",
        "request_fingerprint": "mreq_1",
        "response_id": "mcall_1",
        "provider": "fake",
        "requested_model": "test-model",
        "served_model": "test-model",
        "latency_seconds": 1.25,
        "input_tokens": 100,
        "output_tokens": 50,
    }
    values.update(overrides)
    return PlanningRecord(**values)  # type: ignore[arg-type]


def test_records_round_trip_and_are_write_once(tmp_path: Path) -> None:
    store = PlanningStore(tmp_path)
    record = _record()
    store.record(record)
    store.record(record)  # identical re-record is a no-op

    loaded = store.get(record.id)
    assert loaded == record
    assert loaded is not None and loaded.nominal_cost_usd is None

    conflicting = replace(record, rationale="a different rationale", id=record.id)
    with pytest.raises(PlanningConflictError, match="never rewritten"):
        store.record(conflicting)


def test_a_tampered_record_fails_integrity_on_read(tmp_path: Path) -> None:
    store = PlanningStore(tmp_path)
    record = _record()
    store.record(record)
    path = tmp_path / "decisions" / f"{record.id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rationale"] = "quietly rewritten"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PlanningIntegrityError, match="no longer matches"):
        store.get(record.id)


def test_stop_records_carry_their_typed_reason(tmp_path: Path) -> None:
    store = PlanningStore(tmp_path)
    stop = _record(
        action=PlanningAction.STOP,
        hypothesis_id="",
        prediction_id="",
        spec_id="",
        template_id="",
        stop_reason=StopReason.QUESTION_RESOLVED,
    )
    store.record(stop)
    loaded = store.get(stop.id)
    assert loaded is not None
    assert loaded.stop_reason is StopReason.QUESTION_RESOLVED


def test_dispatch_markers_are_write_once_and_gate_open_decisions(
    tmp_path: Path,
) -> None:
    store = PlanningStore(tmp_path)
    record = _record()
    store.record(record)

    assert store.open_decisions() == (record,)
    store.mark_dispatched(record.id, "run_experiment emitted")
    assert store.is_dispatched(record.id)
    assert store.open_decisions() == ()

    with pytest.raises(PlanningConflictError, match="already dispatched"):
        store.mark_dispatched(record.id, "again")
    with pytest.raises(PlanningConflictError, match="unknown"):
        store.mark_dispatched("plan_0000000000000000", "never recorded")


def test_rejected_attempts_are_durable_with_every_rule(tmp_path: Path) -> None:
    store = PlanningStore(tmp_path)
    store.preserve_rejected(
        invocation_id="inv_1",
        reasons=(
            ("unknown_template", "template 'tmpl_x' is not in the catalog"),
            ("budget_insufficient", "cannot afford the proposed run"),
        ),
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        payload={"action": "new_experiment"},
        repair=0,
    )

    (rejected,) = store.rejected()
    reasons = rejected["reasons"]
    assert isinstance(reasons, list)
    assert {entry["rule"] for entry in reasons} == {
        "unknown_template",
        "budget_insufficient",
    }
    assert rejected["repair"] == 0
    assert rejected["request_fingerprint"] == "mreq_1"
