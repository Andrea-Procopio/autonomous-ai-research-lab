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
    verification_record,
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


def test_record_derives_validity_and_standing() -> None:
    verified = verification_record("res_a", "exp_a", _report(CheckState.PASS))
    assert verified.validity is ExperimentValidityStatus.VERIFIED
    assert verified.standing is OutcomeStanding.VERIFIED_EVIDENCE

    uncertain = verification_record("res_b", "exp_a", _report(CheckState.FAIL))
    assert (
        uncertain.validity is ExperimentValidityStatus.IMPLEMENTATION_UNCERTAIN
    )
    assert uncertain.standing is OutcomeStanding.OBSERVED_UNRESOLVED


def test_in_memory_store_is_idempotent_and_conflict_safe() -> None:
    store = InMemoryVerificationStore()
    record = verification_record("res_a", "exp_a", _report(CheckState.PASS))
    store.record(record)
    store.record(record)  # identical re-record is a no-op
    assert store.get("res_a") == record
    assert store.get("res_missing") is None

    different = verification_record("res_a", "exp_a", _report(CheckState.FAIL))
    with pytest.raises(VerificationConflictError, match="never rewritten"):
        store.record(different)
    # The original verdict survives the attempted rewrite.
    assert store.get("res_a") == record


def test_file_store_roundtrips_and_survives_reinstantiation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "verifications"
    store = FileVerificationStore(root)
    record = verification_record("res_a", "exp_a", _report(CheckState.FAIL))
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
            verification_record("res_a", "exp_a", _report(CheckState.PASS))
        )


def test_file_store_lists_all_records(tmp_path: Path) -> None:
    store = FileVerificationStore(tmp_path)
    a = verification_record("res_a", "exp_a", _report(CheckState.PASS))
    b = verification_record("res_b", "exp_a", _report(CheckState.FAIL))
    store.record(a)
    store.record(b)
    assert set(store.records()) == {a, b}
