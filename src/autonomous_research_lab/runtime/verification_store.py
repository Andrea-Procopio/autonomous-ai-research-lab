"""Durable verification records, keyed by result id.

A :class:`~.verification.VerificationReport` computed inside one runtime step
is not enough to *control* downstream scientific use: the question "was
result X verified?" must be answerable later — at claim-commit time, in a
later step, in a resumed session. This module makes the verdict a durable
record alongside the evidence it governs.

The store observes the same invariant as the evidence store: **an id never
maps to different content.** Re-recording an identical record is a no-op;
re-recording a different one raises. A result's verification verdict is
part of the permanent record — repair produces a *new* result with its own
record, it never rewrites the old one.

Absence is meaningful and deliberate: a result with **no** record was run
with the verification layer ablated (or predates it), and downstream gating
treats it under legacy semantics. A result *with* a record is governed by
it — there is no way to shed an adverse verdict short of running a new
experiment that earns its own.

Two implementations, same philosophy as the rest of persistence: in-memory
for tests and ablations, one small JSON file per record for durability. No
database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .verification import (
    CheckState,
    ExperimentValidityStatus,
    OutcomeStanding,
    ValidityDimension,
    VerificationCheck,
    VerificationReport,
    derive_validity,
    outcome_standing,
)


class VerificationConflictError(RuntimeError):
    """Raised when a result id is re-recorded with different content."""


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    """The durable verification verdict of one result.

    ``validity`` and ``standing`` are derivable from ``report`` but stored
    explicitly: the record is the authority downstream code consults, and a
    consumer should not need the derivation rules to read it.
    """

    result_id: str
    spec_id: str
    report: VerificationReport
    validity: ExperimentValidityStatus
    standing: OutcomeStanding


def verification_record(
    result_id: str, spec_id: str, report: VerificationReport
) -> VerificationRecord:
    """Build the record for one report, deriving validity and standing."""
    validity = derive_validity(report)
    return VerificationRecord(
        result_id=result_id,
        spec_id=spec_id,
        report=report,
        validity=validity,
        standing=outcome_standing(validity),
    )


class VerificationStore(Protocol):
    def record(self, record: VerificationRecord) -> VerificationRecord:
        """Store one record. Idempotent for identical content; a different
        record under the same result id raises
        :class:`VerificationConflictError`."""
        ...

    def get(self, result_id: str) -> VerificationRecord | None:
        """The record governing ``result_id``, or ``None`` when the result
        was never verified (verification ablated / predates the layer)."""
        ...

    def records(self) -> tuple[VerificationRecord, ...]: ...


class InMemoryVerificationStore:
    def __init__(self) -> None:
        self._records: dict[str, VerificationRecord] = {}

    def record(self, record: VerificationRecord) -> VerificationRecord:
        existing = self._records.get(record.result_id)
        if existing is not None:
            if existing != record:
                raise VerificationConflictError(
                    f"result {record.result_id} already has a different "
                    f"verification record; verdicts are never rewritten"
                )
            return existing
        self._records[record.result_id] = record
        return record

    def get(self, result_id: str) -> VerificationRecord | None:
        return self._records.get(result_id)

    def records(self) -> tuple[VerificationRecord, ...]:
        return tuple(self._records.values())


class FileVerificationStore:
    """One JSON file per record, named by result id — the same shape as the
    state snapshot store: local files, reconstructible offline, no database."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, result_id: str) -> Path:
        return self._root / f"{result_id}.json"

    def record(self, record: VerificationRecord) -> VerificationRecord:
        existing = self.get(record.result_id)
        if existing is not None:
            if existing != record:
                raise VerificationConflictError(
                    f"result {record.result_id} already has a different "
                    f"verification record; verdicts are never rewritten"
                )
            return existing
        payload = {
            "result_id": record.result_id,
            "spec_id": record.spec_id,
            "validity": record.validity.value,
            "standing": record.standing.value,
            "checks": [
                {
                    "dimension": check.dimension.value,
                    "name": check.name,
                    "state": check.state.value,
                    "detail": check.detail,
                }
                for check in record.report.checks
            ],
        }
        self._path(record.result_id).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return record

    def get(self, result_id: str) -> VerificationRecord | None:
        path = self._path(result_id)
        if not path.exists():
            return None
        return _parse(json.loads(path.read_text(encoding="utf-8")))

    def records(self) -> tuple[VerificationRecord, ...]:
        return tuple(
            _parse(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self._root.glob("*.json"))
        )


def _parse(payload: dict[str, object]) -> VerificationRecord:
    checks_payload = payload["checks"]
    assert isinstance(checks_payload, list)
    checks: list[VerificationCheck] = []
    for entry in checks_payload:
        assert isinstance(entry, dict)
        checks.append(
            VerificationCheck(
                dimension=ValidityDimension(str(entry["dimension"])),
                name=str(entry["name"]),
                state=CheckState(str(entry["state"])),
                detail=str(entry["detail"]),
            )
        )
    return VerificationRecord(
        result_id=str(payload["result_id"]),
        spec_id=str(payload["spec_id"]),
        report=VerificationReport(checks=tuple(checks)),
        validity=ExperimentValidityStatus(str(payload["validity"])),
        standing=OutcomeStanding(str(payload["standing"])),
    )
