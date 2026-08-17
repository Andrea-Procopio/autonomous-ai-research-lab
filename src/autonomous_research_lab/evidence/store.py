"""Append-only storage for results and evidence.

The store is the system's ground truth. Its single invariant: once a record is
written under an id, that id never maps to different content. Re-recording
identical content is a no-op, so retries and replays are safe; re-recording
*different* content under the same id raises.

The store is shared across branches of a research search. Beliefs fork;
observations do not.
"""

from __future__ import annotations

from typing import Protocol

from ..core.evidence import Evidence
from ..core.experiment import ExperimentResult


class EvidenceConflictError(RuntimeError):
    """Raised on an attempt to overwrite an existing record with new content."""


class UnknownRecordError(KeyError):
    """Raised when a requested result or evidence id is absent."""


class EvidenceStore(Protocol):
    """Storage contract. Implementations may be in-memory, file-backed, or
    remote; nothing above this interface may assume which."""

    def record_result(self, result: ExperimentResult) -> ExperimentResult: ...

    def record_evidence(self, evidence: Evidence) -> Evidence: ...

    def get_result(self, result_id: str) -> ExperimentResult: ...

    def get_evidence(self, evidence_id: str) -> Evidence: ...

    def results(self) -> tuple[ExperimentResult, ...]: ...

    def evidence(self) -> tuple[Evidence, ...]: ...


class InMemoryEvidenceStore:
    """Reference implementation. Adequate for a single process; a durable
    backend slots in behind :class:`EvidenceStore` without touching callers."""

    def __init__(self) -> None:
        self._results: dict[str, ExperimentResult] = {}
        self._evidence: dict[str, Evidence] = {}

    def record_result(self, result: ExperimentResult) -> ExperimentResult:
        existing = self._results.get(result.id)
        if existing is not None:
            if existing != result:
                raise EvidenceConflictError(
                    f"result {result.id} already recorded with different content"
                )
            return existing
        self._results[result.id] = result
        return result

    def record_evidence(self, evidence: Evidence) -> Evidence:
        existing = self._evidence.get(evidence.id)
        if existing is not None:
            if existing != evidence:
                raise EvidenceConflictError(
                    f"evidence {evidence.id} already recorded with different content"
                )
            return existing
        if evidence.result_id not in self._results:
            raise UnknownRecordError(
                f"evidence {evidence.id} references unrecorded result "
                f"{evidence.result_id}"
            )
        self._evidence[evidence.id] = evidence
        return evidence

    def get_result(self, result_id: str) -> ExperimentResult:
        try:
            return self._results[result_id]
        except KeyError as exc:
            raise UnknownRecordError(result_id) from exc

    def get_evidence(self, evidence_id: str) -> Evidence:
        try:
            return self._evidence[evidence_id]
        except KeyError as exc:
            raise UnknownRecordError(evidence_id) from exc

    def results(self) -> tuple[ExperimentResult, ...]:
        return tuple(self._results.values())

    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence.values())
