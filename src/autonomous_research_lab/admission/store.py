"""Durable storage for admissions, mirroring the selection store.

One directory per record kind — directives and completed admissions —
plus ``rejected/`` for every gate-refused model payload, and ``states/``
for the admitted initial state snapshots, managed by the house
:class:`~..persistence.state_store.FileStateStore` derived from the same
root. The state store is deliberately not injectable: the promise that
an admitted state is never exposed without its admission record depends
on the record and the snapshot living under one root, loaded through one
accessor.

Writes are write-once and verify-on-repeat: identical re-recording is a
no-op, different content under the same id raises. Ids are recomputed
from what was read, never trusted from the file, so a tampered record
fails loudly on load. Two internal-consistency rules go beyond plain
write-once: at most one admission record per admission directive, and at
most one per selection run record — ever. A second admission of the same
selection is a conflict to raise, not a record to file alongside the
first; an admitted selection is never silently replaced.

Nothing here may ever hold a credential: records store fingerprints,
ids, token counts and text, not keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ..core.ids import occurrence_id
from ..core.state import ResearchState
from ..mapping.records import CallProvenance
from ..persistence import FileStateStore
from .directive import AdmissionDirective
from .records import (
    AdmissionRecord,
    GroundedSupport,
    OperationalPrediction,
    Requirement,
    RequirementSource,
    SupportSource,
)

_RECORD_SUFFIX: Final = ".json"

_DIRECTIVES: Final = "directives"
_RECORDS: Final = "records"
_REJECTED: Final = "rejected"


class AdmissionConflictError(RuntimeError):
    """A write-once admission artifact would be overwritten with
    different content, or a selection run would be admitted twice."""


class AdmissionIntegrityError(RuntimeError):
    """A stored admission record no longer matches its own identity."""


class AdmissionStore:
    """File-backed, write-once storage for admissions under one injected
    root, including the admitted state snapshots."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._states = FileStateStore(self._root)

    @property
    def root(self) -> Path:
        return self._root

    # -- generic write-once machinery -----------------------------------------

    def _path(self, kind: str, record_id: str) -> Path:
        return self._root / kind / f"{record_id}{_RECORD_SUFFIX}"

    def _write_once(
        self, kind: str, record_id: str, payload: Mapping[str, object]
    ) -> None:
        path = self._path(kind, record_id)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise AdmissionConflictError(
                    f"{kind} record {record_id} is already recorded with "
                    f"different content; records are never rewritten"
                )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _load(self, kind: str, record_id: str) -> Mapping[str, object] | None:
        path = self._path(kind, record_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, Mapping)
        return payload

    def _ids(self, kind: str) -> tuple[str, ...]:
        directory = self._root / kind
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(path.stem for path in directory.glob(f"*{_RECORD_SUFFIX}"))
        )

    @staticmethod
    def _verify(kind: str, filed_as: str, rederived: str) -> None:
        # The id is recomputed from what was read, never trusted from the
        # file: a record that no longer hashes to its name fails loudly.
        if filed_as != rederived:
            raise AdmissionIntegrityError(
                f"{kind} record filed under {filed_as} re-derives id "
                f"{rederived}; refusing to load a record that no longer "
                f"matches its name"
            )

    # -- directives ------------------------------------------------------------

    def record_directive(
        self, directive: AdmissionDirective
    ) -> AdmissionDirective:
        self._write_once(
            _DIRECTIVES, directive.id, _directive_payload(directive)
        )
        return directive

    def get_directive(self, directive_id: str) -> AdmissionDirective | None:
        payload = self._load(_DIRECTIVES, directive_id)
        if payload is None:
            return None
        directive = _directive_from(payload)
        self._verify(_DIRECTIVES, directive_id, directive.id)
        return directive

    # -- completed admissions --------------------------------------------------

    def record_admission(self, record: AdmissionRecord) -> AdmissionRecord:
        for existing_id in self._ids(_RECORDS):
            if existing_id == record.id:
                continue
            existing = self.get_record(existing_id)
            assert existing is not None
            if existing.directive_id == record.directive_id:
                raise AdmissionConflictError(
                    f"directive {record.directive_id} already produced "
                    f"admission {existing.id}; a second account of one "
                    f"admission is a conflict, not a record"
                )
            if (
                existing.selection_run_record_id
                == record.selection_run_record_id
            ):
                raise AdmissionConflictError(
                    f"selection run {record.selection_run_record_id} is "
                    f"already admitted as {existing.id}; an admitted "
                    f"selection is never silently replaced"
                )
        self._write_once(_RECORDS, record.id, _record_payload(record))
        return record

    def get_record(self, record_id: str) -> AdmissionRecord | None:
        payload = self._load(_RECORDS, record_id)
        if payload is None:
            return None
        record = _record_from(payload)
        self._verify(_RECORDS, record_id, record.id)
        return record

    def records(self) -> tuple[AdmissionRecord, ...]:
        loaded = []
        for record_id in self._ids(_RECORDS):
            record = self.get_record(record_id)
            assert record is not None
            loaded.append(record)
        return tuple(loaded)

    def record_for_directive(
        self, directive_id: str
    ) -> AdmissionRecord | None:
        for record in self.records():
            if record.directive_id == directive_id:
                return record
        return None

    def record_for_selection_run(
        self, selection_run_record_id: str
    ) -> AdmissionRecord | None:
        for record in self.records():
            if record.selection_run_record_id == selection_run_record_id:
                return record
        return None

    # -- admitted states -------------------------------------------------------

    def persist_state(self, state: ResearchState) -> Path:
        """Persist the snapshot, then read it back and require equality
        before anything may reference it. The read-back is load-bearing:
        a ``ResearchState``'s content id deliberately excludes its
        budget, so id verification alone would not catch every divergent
        byte."""
        path = self._states.persist(state)
        if self._states.load(state.id) != state:
            raise AdmissionIntegrityError(
                f"snapshot {state.id} did not read back as the state "
                f"that was written; refusing to reference it"
            )
        return path

    def get_admitted_state(
        self, record_id: str
    ) -> tuple[AdmissionRecord, ResearchState]:
        """The one accessor: the record first, the state through it. An
        admitted state is never exposed without its admission record,
        and a record whose snapshot is missing or tampered fails loudly
        (the snapshot is part of the write-once artifact set)."""
        record = self.get_record(record_id)
        if record is None:
            raise AdmissionIntegrityError(
                f"no admission record {record_id}; a state is never "
                f"exposed without its record"
            )
        state = self._states.load(record.state_id)
        if not state.budget.is_exhausted:
            # The state's content id excludes the budget, so a doctored
            # budget would reload silently; an admitted seed's budget is
            # zero by construction, which makes the check exact.
            raise AdmissionIntegrityError(
                f"admitted state {record.state_id} carries a non-zero "
                f"budget; an admitted seed never does"
            )
        return record, state

    # -- rejected attempts -----------------------------------------------------

    def preserve_rejected(
        self,
        *,
        run_id: str,
        stage: str,
        reasons: tuple[tuple[str, str], ...],
        request_fingerprint: str,
        response_id: str,
        payload: object,
        repair: int,
    ) -> Path:
        """Preserve one gate-rejected model payload as data: the stage,
        every rule that fired, the call's provenance handles, and the
        raw payload. Returns the file written."""
        directory = self._root / _REJECTED
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{occurrence_id('arej')}{_RECORD_SUFFIX}"
        path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "stage": stage,
                    "reasons": [
                        {"rule": rule, "detail": detail}
                        for rule, detail in reasons
                    ],
                    "request_fingerprint": request_fingerprint,
                    "response_id": response_id,
                    "payload": _jsonable(payload),
                    "repair": repair,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def rejected(self) -> tuple[Mapping[str, object], ...]:
        directory = self._root / _REJECTED
        if not directory.exists():
            return ()
        return tuple(
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob(f"*{_RECORD_SUFFIX}"))
        )


# -- serialization ------------------------------------------------------------


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _provenance_payload(provenance: CallProvenance) -> dict[str, object]:
    return {
        "request_fingerprint": provenance.request_fingerprint,
        "response_id": provenance.response_id,
        "provider": provenance.provider,
        "requested_model": provenance.requested_model,
        "served_model": provenance.served_model,
        "provider_request_id": provenance.provider_request_id,
        "latency_seconds": provenance.latency_seconds,
        "input_tokens": provenance.input_tokens,
        "output_tokens": provenance.output_tokens,
        "repair_count": provenance.repair_count,
    }


def _provenance_from(payload: object) -> CallProvenance:
    assert isinstance(payload, Mapping)
    request_id = payload["provider_request_id"]
    return CallProvenance(
        request_fingerprint=str(payload["request_fingerprint"]),
        response_id=str(payload["response_id"]),
        provider=str(payload["provider"]),
        requested_model=str(payload["requested_model"]),
        served_model=str(payload["served_model"]),
        provider_request_id=(
            str(request_id) if request_id is not None else None
        ),
        latency_seconds=float(str(payload["latency_seconds"])),
        input_tokens=int(str(payload["input_tokens"])),
        output_tokens=int(str(payload["output_tokens"])),
        repair_count=int(str(payload["repair_count"])),
    )


def _directive_payload(directive: AdmissionDirective) -> dict[str, object]:
    return {
        "id": directive.id,
        "selection_run_record_id": directive.selection_run_record_id,
        "scheduling_requirement": directive.scheduling_requirement,
        "job_duration_requirement": directive.job_duration_requirement,
        "checkpoint_requirement": directive.checkpoint_requirement,
        "max_model_calls": directive.max_model_calls,
    }


def _directive_from(payload: Mapping[str, object]) -> AdmissionDirective:
    return AdmissionDirective(
        selection_run_record_id=str(payload["selection_run_record_id"]),
        scheduling_requirement=str(payload["scheduling_requirement"]),
        job_duration_requirement=str(payload["job_duration_requirement"]),
        checkpoint_requirement=str(payload["checkpoint_requirement"]),
        max_model_calls=int(str(payload["max_model_calls"])),
    )


def _support_payload(entry: GroundedSupport) -> dict[str, object]:
    return {
        "source": entry.source.value,
        "field_path": entry.field_path,
        "quote": entry.quote,
    }


def _support_from(payload: Mapping[str, object]) -> GroundedSupport:
    return GroundedSupport(
        source=SupportSource(str(payload["source"])),
        field_path=str(payload["field_path"]),
        quote=str(payload["quote"]),
    )


def _operational_payload(entry: OperationalPrediction) -> dict[str, object]:
    return {
        "prediction_text": entry.prediction_text,
        "condition": entry.condition,
        "base_metric": entry.base_metric,
        "expected_higher_arm": entry.expected_higher_arm,
        "expected_lower_arm": entry.expected_lower_arm,
        "contrary_observation": entry.contrary_observation,
        "support": [_support_payload(link) for link in entry.support],
    }


def _operational_from(payload: Mapping[str, object]) -> OperationalPrediction:
    return OperationalPrediction(
        prediction_text=str(payload["prediction_text"]),
        condition=str(payload["condition"]),
        base_metric=str(payload["base_metric"]),
        expected_higher_arm=str(payload["expected_higher_arm"]),
        expected_lower_arm=str(payload["expected_lower_arm"]),
        contrary_observation=str(payload["contrary_observation"]),
        support=tuple(
            _support_from(link) for link in _entries(payload, "support")
        ),
    )


def _requirement_payload(entry: Requirement) -> dict[str, object]:
    return {
        "source": entry.source.value,
        "record_id": entry.record_id,
        "field_path": entry.field_path,
        "quote": entry.quote,
    }


def _requirement_from(payload: Mapping[str, object]) -> Requirement:
    return Requirement(
        source=RequirementSource(str(payload["source"])),
        record_id=str(payload["record_id"]),
        field_path=str(payload["field_path"]),
        quote=str(payload["quote"]),
    )


def _record_payload(record: AdmissionRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "directive_id": record.directive_id,
        "selection_run_record_id": record.selection_run_record_id,
        "selection_run_id": record.selection_run_id,
        "selection_directive_id": record.selection_directive_id,
        "prior_art_run_record_id": record.prior_art_run_record_id,
        "prior_art_run_id": record.prior_art_run_id,
        "selected_prior_art_assessment_id": (
            record.selected_prior_art_assessment_id
        ),
        "ideation_run_record_id": record.ideation_run_record_id,
        "ideation_run_id": record.ideation_run_id,
        "direction_id": record.direction_id,
        "snapshot_id": record.snapshot_id,
        "map_run_id": record.map_run_id,
        "map_assessment_id": record.map_assessment_id,
        "selected_candidate_id": record.selected_candidate_id,
        "operational_predictions": [
            _operational_payload(entry)
            for entry in record.operational_predictions
        ],
        "measurements": list(record.measurements),
        "controls": list(record.controls),
        "comparison_targets": list(record.comparison_targets),
        "evaluation_protocol": record.evaluation_protocol,
        "inherited_requirements": [
            _requirement_payload(entry)
            for entry in record.inherited_requirements
        ],
        "operator_requirements": [
            _requirement_payload(entry)
            for entry in record.operator_requirements
        ],
        "mechanical_reading": record.mechanical_reading,
        "question_id": record.question_id,
        "hypothesis_id": record.hypothesis_id,
        "prediction_ids": list(record.prediction_ids),
        "state_id": record.state_id,
        "provenance": _provenance_payload(record.provenance),
        "model_calls": record.model_calls,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
    }


def _record_from(payload: Mapping[str, object]) -> AdmissionRecord:
    return AdmissionRecord(
        run_id=str(payload["run_id"]),
        directive_id=str(payload["directive_id"]),
        selection_run_record_id=str(payload["selection_run_record_id"]),
        selection_run_id=str(payload["selection_run_id"]),
        selection_directive_id=str(payload["selection_directive_id"]),
        prior_art_run_record_id=str(payload["prior_art_run_record_id"]),
        prior_art_run_id=str(payload["prior_art_run_id"]),
        selected_prior_art_assessment_id=str(
            payload["selected_prior_art_assessment_id"]
        ),
        ideation_run_record_id=str(payload["ideation_run_record_id"]),
        ideation_run_id=str(payload["ideation_run_id"]),
        direction_id=str(payload["direction_id"]),
        snapshot_id=str(payload["snapshot_id"]),
        map_run_id=str(payload["map_run_id"]),
        map_assessment_id=str(payload["map_assessment_id"]),
        selected_candidate_id=str(payload["selected_candidate_id"]),
        operational_predictions=tuple(
            _operational_from(entry)
            for entry in _entries(payload, "operational_predictions")
        ),
        measurements=_strings(payload, "measurements"),
        controls=_strings(payload, "controls"),
        comparison_targets=_strings(payload, "comparison_targets"),
        evaluation_protocol=str(payload["evaluation_protocol"]),
        inherited_requirements=tuple(
            _requirement_from(entry)
            for entry in _entries(payload, "inherited_requirements")
        ),
        operator_requirements=tuple(
            _requirement_from(entry)
            for entry in _entries(payload, "operator_requirements")
        ),
        mechanical_reading=str(payload["mechanical_reading"]),
        question_id=str(payload["question_id"]),
        hypothesis_id=str(payload["hypothesis_id"]),
        prediction_ids=_strings(payload, "prediction_ids"),
        state_id=str(payload["state_id"]),
        provenance=_provenance_from(payload["provenance"]),
        model_calls=int(str(payload["model_calls"])),
        input_tokens=int(str(payload["input_tokens"])),
        output_tokens=int(str(payload["output_tokens"])),
    )


def _entries(
    payload: Mapping[str, object], key: str
) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key, [])
    assert isinstance(value, list)
    entries = []
    for entry in value:
        assert isinstance(entry, Mapping)
        entries.append(entry)
    return tuple(entries)


def _strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    assert isinstance(value, list)
    return tuple(str(entry) for entry in value)
