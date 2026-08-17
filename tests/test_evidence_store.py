from __future__ import annotations

import dataclasses

import pytest

from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentStatus,
)
from autonomous_research_lab.evidence.store import (
    EvidenceConflictError,
    InMemoryEvidenceStore,
    UnknownRecordError,
)

ENVIRONMENT = Environment(python_version="3.11.0", platform="test")


def make_result(**overrides: object) -> ExperimentResult:
    defaults: dict[str, object] = {
        "spec_id": "exp_1",
        "job_id": "job_1",
        "status": ExperimentStatus.COMPLETED,
        "command": ("python", "run.py"),
        "environment": ENVIRONMENT,
        "metrics": {"accuracy": 0.5},
    }
    return ExperimentResult(**(defaults | overrides))  # type: ignore[arg-type]


def test_recording_the_same_record_twice_is_idempotent() -> None:
    store = InMemoryEvidenceStore()
    store.record_result(make_result())
    store.record_result(make_result())
    assert len(store.results()) == 1


def test_a_result_cannot_be_rewritten() -> None:
    """The central invariant: an id keeps its content forever. Reinterpretation
    is allowed; revision of the record is not."""
    store = InMemoryEvidenceStore()
    original = store.record_result(make_result())
    tampered = dataclasses.replace(original, metrics={"accuracy": 0.99})

    with pytest.raises(EvidenceConflictError):
        store.record_result(tampered)

    assert store.get_result(original.id).metrics["accuracy"] == 0.5


def test_metrics_on_a_stored_result_are_read_only() -> None:
    store = InMemoryEvidenceStore()
    result = store.record_result(make_result())
    with pytest.raises(TypeError):
        result.metrics["accuracy"] = 0.99  # type: ignore[index]


def test_evidence_must_point_at_a_recorded_result() -> None:
    """Evidence with no recorded result behind it is an assertion, not
    evidence."""
    store = InMemoryEvidenceStore()
    orphan = Evidence(
        result_id="res_missing",
        spec_id="exp_1",
        kind=EvidenceKind.MEASUREMENT,
        observation="accuracy was 0.5",
    )
    with pytest.raises(UnknownRecordError):
        store.record_evidence(orphan)


def test_unknown_ids_raise() -> None:
    store = InMemoryEvidenceStore()
    with pytest.raises(UnknownRecordError):
        store.get_result("res_nope")
