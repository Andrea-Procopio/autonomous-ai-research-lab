"""Durable verification records: id-to-content stability, file roundtrip."""

from __future__ import annotations

from pathlib import Path

import pytest

from autonomous_research_lab.runtime.verification import (
    CheckState,
    ExperimentValidityStatus,
    OutcomeStanding,
    ValidityDimension,
    VerificationCheck,
    VerificationReport,
)
from autonomous_research_lab.runtime.verification_store import (
    FileVerificationStore,
    InMemoryVerificationStore,
    VerificationConflictError,
    VerificationIntegrityError,
    VerificationRecord,
)


def _report(implementation: CheckState) -> VerificationReport:
    return VerificationReport(
        checks=(
            VerificationCheck(
                dimension=ValidityDimension.EXECUTION,
                name="deterministic_validation",
                state=CheckState.PASS,
            ),
            VerificationCheck(
                dimension=ValidityDimension.IMPLEMENTATION,
                name="positive_control:probe",
                state=implementation,
                detail="probe reading",
            ),
            VerificationCheck(
                dimension=ValidityDimension.METHODOLOGY,
                name="methodological_validity",
                state=CheckState.PASS,
            ),
            VerificationCheck(
                dimension=ValidityDimension.ANALYSIS,
                name="raw_result_reading",
                state=CheckState.PASS,
            ),
        )
    )


def _record(result_id: str, report: VerificationReport) -> VerificationRecord:
    return VerificationRecord(
        result_id=result_id, spec_id="exp_a", report=report
    )


def test_record_derives_validity_and_standing() -> None:
    verified = _record("res_a", _report(CheckState.PASS))
    assert verified.validity is ExperimentValidityStatus.VERIFIED
    assert verified.standing is OutcomeStanding.VERIFIED_EVIDENCE

    uncertain = _record("res_b", _report(CheckState.FAIL))
    assert (
        uncertain.validity is ExperimentValidityStatus.IMPLEMENTATION_UNCERTAIN
    )
    assert uncertain.standing is OutcomeStanding.OBSERVED_UNRESOLVED


def test_in_memory_store_is_idempotent_and_conflict_safe() -> None:
    store = InMemoryVerificationStore()
    record = _record("res_a", _report(CheckState.PASS))
    store.record(record)
    store.record(record)  # identical re-record is a no-op
    assert store.get("res_a") == record
    assert store.get("res_missing") is None

    different = _record("res_a", _report(CheckState.FAIL))
    with pytest.raises(VerificationConflictError, match="never rewritten"):
        store.record(different)
    # The original verdict survives the attempted rewrite.
    assert store.get("res_a") == record


def test_file_store_roundtrips_and_survives_reinstantiation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "verifications"
    store = FileVerificationStore(root)
    record = _record("res_a", _report(CheckState.FAIL))
    store.record(record)

    # A fresh store over the same directory reads back an identical record:
    # the verdict survives beyond any in-memory step accounting.
    reopened = FileVerificationStore(root)
    loaded = reopened.get("res_a")
    assert loaded == record
    assert loaded is not None
    assert loaded.report.dimension_state(
        ValidityDimension.IMPLEMENTATION
    ) is CheckState.FAIL

    with pytest.raises(VerificationConflictError):
        reopened.record(
            _record("res_a", _report(CheckState.PASS))
        )


def test_records_are_internally_canonical() -> None:
    """The report is the single source of truth: the verdict is derived at
    construction and cannot be supplied, so an inconsistent record (report
    says FAIL, verdict says VERIFIED) is unconstructible by type."""
    record = VerificationRecord(
        result_id="res_a", spec_id="exp_a", report=_report(CheckState.FAIL)
    )
    assert (
        record.validity is ExperimentValidityStatus.IMPLEMENTATION_UNCERTAIN
    )
    with pytest.raises(TypeError):
        VerificationRecord(  # type: ignore[call-arg]
            result_id="res_a",
            spec_id="exp_a",
            report=_report(CheckState.FAIL),
            validity=ExperimentValidityStatus.VERIFIED,
        )


def test_tampered_disk_verdict_fails_loudly(tmp_path: Path) -> None:
    """A serialized record whose stored verdict disagrees with its own
    report is corruption: it raises on load, it never becomes trusted."""
    store = FileVerificationStore(tmp_path)
    record = VerificationRecord(
        result_id="res_a", spec_id="exp_a", report=_report(CheckState.FAIL)
    )
    store.record(record)

    path = tmp_path / "res_a.json"
    tampered = path.read_text().replace(
        '"validity": "implementation_uncertain"', '"validity": "verified"'
    )
    assert tampered != path.read_text()  # the edit really happened
    path.write_text(tampered)

    reopened = FileVerificationStore(tmp_path)
    with pytest.raises(VerificationIntegrityError, match="refusing to load"):
        reopened.get("res_a")


def test_file_store_lists_all_records(tmp_path: Path) -> None:
    store = FileVerificationStore(tmp_path)
    a = _record("res_a", _report(CheckState.PASS))
    b = _record("res_b", _report(CheckState.FAIL))
    store.record(a)
    store.record(b)
    assert set(store.records()) == {a, b}
