"""Durable storage for field-mapping runs, mirroring the planning store.

One directory per record kind — briefs, query executions, screenings,
extractions, field maps, problem inventories, completed runs — plus
``rejected/`` for every gate-refused model payload, preserved as data with
the provenance of the call that produced it. Writes are write-once and
verify-on-repeat: identical re-recording is a no-op, different content
under the same id raises. Ids are recomputed from what was read, never
trusted from the file, so a tampered record fails loudly on load.

Two internal-consistency rules go beyond plain write-once: a run may hold
at most one screening and at most one extraction per source — a second,
different verdict about the same source in the same run is a conflict to
raise, not a record to file alongside the first.

Nothing here may ever hold a credential: records store fingerprints, ids,
token counts and text, not keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ..core.ids import occurrence_id
from .brief import QueryFamily, ResearchBrief, SourceEra
from .records import (
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

_RECORD_SUFFIX: Final = ".json"

_BRIEFS: Final = "briefs"
_QUERY_RUNS: Final = "query_runs"
_SCREENINGS: Final = "screenings"
_EXTRACTIONS: Final = "extractions"
_FIELD_MAPS: Final = "fieldmaps"
_INVENTORIES: Final = "inventories"
_RUNS: Final = "runs"
_REJECTED: Final = "rejected"


class MappingConflictError(RuntimeError):
    """A write-once mapping artifact would be overwritten with different
    content, or a run would hold two verdicts about one source."""


class MappingIntegrityError(RuntimeError):
    """A stored mapping record no longer matches its own identity."""


class MappingStore:
    """File-backed, write-once storage for one or more mapping runs under
    one injected root."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

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
                raise MappingConflictError(
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
            raise MappingIntegrityError(
                f"{kind} record filed under {filed_as} re-derives id "
                f"{rederived}; refusing to load a record that no longer "
                f"matches its name"
            )

    # -- briefs ----------------------------------------------------------------

    def record_brief(self, brief: ResearchBrief) -> ResearchBrief:
        self._write_once(_BRIEFS, brief.id, _brief_payload(brief))
        return brief

    def get_brief(self, brief_id: str) -> ResearchBrief | None:
        payload = self._load(_BRIEFS, brief_id)
        if payload is None:
            return None
        brief = _brief_from(payload)
        self._verify(_BRIEFS, brief_id, brief.id)
        return brief

    # -- query executions ------------------------------------------------------

    def record_query_execution(self, execution: QueryExecution) -> QueryExecution:
        self._write_once(
            _QUERY_RUNS, execution.id, _query_payload(execution)
        )
        return execution

    def get_query_execution(self, execution_id: str) -> QueryExecution | None:
        payload = self._load(_QUERY_RUNS, execution_id)
        if payload is None:
            return None
        execution = _query_from(payload)
        self._verify(_QUERY_RUNS, execution_id, execution.id)
        return execution

    # -- screenings ------------------------------------------------------------

    def record_screening(self, record: ScreeningRecord) -> ScreeningRecord:
        for existing_id in self._ids(_SCREENINGS):
            if existing_id == record.id:
                continue
            existing = self.get_screening(existing_id)
            assert existing is not None
            if (
                existing.run_id == record.run_id
                and existing.source_id == record.source_id
            ):
                raise MappingConflictError(
                    f"run {record.run_id} already screened source "
                    f"{record.source_id} ({existing.decision.value}); a "
                    f"second verdict is a conflict, not a record"
                )
        self._write_once(_SCREENINGS, record.id, _screening_payload(record))
        return record

    def get_screening(self, record_id: str) -> ScreeningRecord | None:
        payload = self._load(_SCREENINGS, record_id)
        if payload is None:
            return None
        record = _screening_from(payload)
        self._verify(_SCREENINGS, record_id, record.id)
        return record

    def screenings(self) -> tuple[ScreeningRecord, ...]:
        loaded = []
        for record_id in self._ids(_SCREENINGS):
            record = self.get_screening(record_id)
            assert record is not None
            loaded.append(record)
        return tuple(loaded)

    # -- extractions -----------------------------------------------------------

    def record_extraction(self, record: ExtractionRecord) -> ExtractionRecord:
        for existing_id in self._ids(_EXTRACTIONS):
            if existing_id == record.id:
                continue
            existing = self.get_extraction(existing_id)
            assert existing is not None
            if (
                existing.run_id == record.run_id
                and existing.source_id == record.source_id
            ):
                raise MappingConflictError(
                    f"run {record.run_id} already extracted source "
                    f"{record.source_id}; a second extraction is a "
                    f"conflict, not a record"
                )
        self._write_once(_EXTRACTIONS, record.id, _extraction_payload(record))
        return record

    def get_extraction(self, record_id: str) -> ExtractionRecord | None:
        payload = self._load(_EXTRACTIONS, record_id)
        if payload is None:
            return None
        record = _extraction_from(payload)
        self._verify(_EXTRACTIONS, record_id, record.id)
        return record

    def extractions(self) -> tuple[ExtractionRecord, ...]:
        loaded = []
        for record_id in self._ids(_EXTRACTIONS):
            record = self.get_extraction(record_id)
            assert record is not None
            loaded.append(record)
        return tuple(loaded)

    # -- field maps and inventories --------------------------------------------

    def record_field_map(self, record: FieldMapRecord) -> FieldMapRecord:
        self._write_once(_FIELD_MAPS, record.id, _field_map_payload(record))
        return record

    def get_field_map(self, record_id: str) -> FieldMapRecord | None:
        payload = self._load(_FIELD_MAPS, record_id)
        if payload is None:
            return None
        record = _field_map_from(payload)
        self._verify(_FIELD_MAPS, record_id, record.id)
        return record

    def record_inventory(
        self, record: ProblemInventoryRecord
    ) -> ProblemInventoryRecord:
        self._write_once(_INVENTORIES, record.id, _inventory_payload(record))
        return record

    def get_inventory(self, record_id: str) -> ProblemInventoryRecord | None:
        payload = self._load(_INVENTORIES, record_id)
        if payload is None:
            return None
        record = _inventory_from(payload)
        self._verify(_INVENTORIES, record_id, record.id)
        return record

    # -- completed runs --------------------------------------------------------

    def record_run(self, record: MappingRunRecord) -> MappingRunRecord:
        self._write_once(_RUNS, record.id, _run_payload(record))
        return record

    def get_run(self, record_id: str) -> MappingRunRecord | None:
        payload = self._load(_RUNS, record_id)
        if payload is None:
            return None
        record = _run_from(payload)
        self._verify(_RUNS, record_id, record.id)
        return record

    def runs(self) -> tuple[MappingRunRecord, ...]:
        loaded = []
        for record_id in self._ids(_RUNS):
            record = self.get_run(record_id)
            assert record is not None
            loaded.append(record)
        return tuple(loaded)

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
        every rule that fired, the call's provenance handles, and the raw
        payload. Returns the file written."""
        directory = self._root / _REJECTED
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{occurrence_id('mrej')}{_RECORD_SUFFIX}"
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


def _brief_payload(brief: ResearchBrief) -> dict[str, object]:
    return {
        "id": brief.id,
        "topic": brief.topic,
        "cutoff_date": brief.cutoff_date,
        "recent_window_start": brief.recent_window_start,
        "workshop_hints": list(brief.workshop_hints),
        "max_queries_per_family": brief.max_queries_per_family,
        "results_per_query": brief.results_per_query,
        "max_screened_sources": brief.max_screened_sources,
        "max_extracted_sources": brief.max_extracted_sources,
        "max_model_calls": brief.max_model_calls,
    }


def _brief_from(payload: Mapping[str, object]) -> ResearchBrief:
    hints = payload["workshop_hints"]
    assert isinstance(hints, list)
    return ResearchBrief(
        topic=str(payload["topic"]),
        cutoff_date=str(payload["cutoff_date"]),
        recent_window_start=str(payload["recent_window_start"]),
        workshop_hints=tuple(str(hint) for hint in hints),
        max_queries_per_family=int(str(payload["max_queries_per_family"])),
        results_per_query=int(str(payload["results_per_query"])),
        max_screened_sources=int(str(payload["max_screened_sources"])),
        max_extracted_sources=int(str(payload["max_extracted_sources"])),
        max_model_calls=int(str(payload["max_model_calls"])),
    )


def _query_payload(execution: QueryExecution) -> dict[str, object]:
    return {
        "id": execution.id,
        "run_id": execution.run_id,
        "family": execution.family.value,
        "text": execution.text,
        "from_date": execution.from_date,
        "to_date": execution.to_date,
        "query_fingerprint": execution.query_fingerprint,
        "search_record_id": execution.search_record_id,
        "retrieved": execution.retrieved,
        "new_unique": execution.new_unique,
        "from_cache": execution.from_cache,
    }


def _query_from(payload: Mapping[str, object]) -> QueryExecution:
    return QueryExecution(
        run_id=str(payload["run_id"]),
        family=QueryFamily(str(payload["family"])),
        text=str(payload["text"]),
        from_date=str(payload["from_date"]),
        to_date=str(payload["to_date"]),
        query_fingerprint=str(payload["query_fingerprint"]),
        search_record_id=str(payload["search_record_id"]),
        retrieved=int(str(payload["retrieved"])),
        new_unique=int(str(payload["new_unique"])),
        from_cache=bool(payload["from_cache"]),
    )


def _screening_payload(record: ScreeningRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "source_id": record.source_id,
        "decision": record.decision.value,
        "reason": record.reason,
        "provenance": _provenance_payload(record.provenance),
    }


def _screening_from(payload: Mapping[str, object]) -> ScreeningRecord:
    return ScreeningRecord(
        run_id=str(payload["run_id"]),
        source_id=str(payload["source_id"]),
        decision=ScreeningDecision(str(payload["decision"])),
        reason=str(payload["reason"]),
        provenance=_provenance_from(payload["provenance"]),
    )


def _extraction_payload(record: ExtractionRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "source_id": record.source_id,
        "era": record.era.value,
        "access_level": record.access_level,
        "support_location": record.support_location.value,
        "sufficient_support": record.sufficient_support,
        "insufficiency_reason": record.insufficiency_reason,
        "methods": list(record.methods),
        "datasets": [
            {
                "name": d.name,
                "task": d.task,
                "version": d.version,
                "split": d.split,
                "subset": d.subset,
                "preprocessing": d.preprocessing,
                "size": d.size,
                "availability": d.availability.value,
                "url": d.url,
                "license": d.license,
            }
            for d in record.datasets
        ],
        "metrics": list(record.metrics),
        "evaluation_protocols": list(record.evaluation_protocols),
        "baselines": list(record.baselines),
        "reported_results": list(record.reported_results),
        "limitations": [
            {"text": entry.text, "kind": entry.kind.value}
            for entry in record.limitations
        ],
        "future_work": list(record.future_work),
        "open_problems": list(record.open_problems),
        "provenance": (
            _provenance_payload(record.provenance)
            if record.provenance is not None
            else None
        ),
    }


def _extraction_from(payload: Mapping[str, object]) -> ExtractionRecord:
    datasets = payload["datasets"]
    limitations = payload["limitations"]
    provenance = payload["provenance"]
    assert isinstance(datasets, list)
    assert isinstance(limitations, list)
    return ExtractionRecord(
        run_id=str(payload["run_id"]),
        source_id=str(payload["source_id"]),
        era=SourceEra(str(payload["era"])),
        access_level=str(payload["access_level"]),
        support_location=SupportLocation(str(payload["support_location"])),
        sufficient_support=bool(payload["sufficient_support"]),
        insufficiency_reason=str(payload["insufficiency_reason"]),
        methods=_strings(payload["methods"]),
        datasets=tuple(
            DatasetUse(
                name=str(entry["name"]),
                task=str(entry["task"]),
                version=str(entry["version"]),
                split=str(entry["split"]),
                subset=str(entry["subset"]),
                preprocessing=str(entry["preprocessing"]),
                size=str(entry["size"]),
                availability=DatasetAvailability(str(entry["availability"])),
                url=str(entry["url"]),
                license=str(entry["license"]),
            )
            for entry in datasets
        ),
        metrics=_strings(payload["metrics"]),
        evaluation_protocols=_strings(payload["evaluation_protocols"]),
        baselines=_strings(payload["baselines"]),
        reported_results=_strings(payload["reported_results"]),
        limitations=tuple(
            Limitation(
                text=str(entry["text"]),
                kind=LimitationKind(str(entry["kind"])),
            )
            for entry in limitations
        ),
        future_work=_strings(payload["future_work"]),
        open_problems=_strings(payload["open_problems"]),
        provenance=(
            _provenance_from(provenance) if provenance is not None else None
        ),
    )


def _field_map_payload(record: FieldMapRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "brief_id": record.brief_id,
        "themes": [
            {
                "name": t.name,
                "summary": t.summary,
                "era": t.era.value,
                "source_ids": list(t.source_ids),
            }
            for t in record.themes
        ],
        "approaches": [
            {
                "name": g.name,
                "summary": g.summary,
                "source_ids": list(g.source_ids),
            }
            for g in record.approaches
        ],
        "evaluation_practices": [
            {
                "name": g.name,
                "summary": g.summary,
                "source_ids": list(g.source_ids),
            }
            for g in record.evaluation_practices
        ],
        "relationships": [
            {
                "kind": r.kind.value,
                "from_theme": r.from_theme,
                "to_theme": r.to_theme,
                "note": r.note,
            }
            for r in record.relationships
        ],
        "recent_source_ids": list(record.recent_source_ids),
        "foundational_source_ids": list(record.foundational_source_ids),
        "undated_source_ids": list(record.undated_source_ids),
        "provenance": _provenance_payload(record.provenance),
    }


def _field_map_from(payload: Mapping[str, object]) -> FieldMapRecord:
    themes = payload["themes"]
    approaches = payload["approaches"]
    practices = payload["evaluation_practices"]
    relationships = payload["relationships"]
    assert isinstance(themes, list)
    assert isinstance(approaches, list)
    assert isinstance(practices, list)
    assert isinstance(relationships, list)
    return FieldMapRecord(
        run_id=str(payload["run_id"]),
        brief_id=str(payload["brief_id"]),
        themes=tuple(
            ThemeEntry(
                name=str(entry["name"]),
                summary=str(entry["summary"]),
                era=ThemeEra(str(entry["era"])),
                source_ids=_strings(entry["source_ids"]),
            )
            for entry in themes
        ),
        approaches=tuple(
            GroupEntry(
                name=str(entry["name"]),
                summary=str(entry["summary"]),
                source_ids=_strings(entry["source_ids"]),
            )
            for entry in approaches
        ),
        evaluation_practices=tuple(
            GroupEntry(
                name=str(entry["name"]),
                summary=str(entry["summary"]),
                source_ids=_strings(entry["source_ids"]),
            )
            for entry in practices
        ),
        relationships=tuple(
            ThemeRelationship(
                kind=RelationshipKind(str(entry["kind"])),
                from_theme=str(entry["from_theme"]),
                to_theme=str(entry["to_theme"]),
                note=str(entry["note"]),
            )
            for entry in relationships
        ),
        recent_source_ids=_strings(payload["recent_source_ids"]),
        foundational_source_ids=_strings(payload["foundational_source_ids"]),
        undated_source_ids=_strings(payload["undated_source_ids"]),
        provenance=_provenance_from(payload["provenance"]),
    )


def _inventory_payload(record: ProblemInventoryRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "brief_id": record.brief_id,
        "problems": [
            {
                "statement": p.statement,
                "kind": p.kind.value,
                "grounding": p.grounding,
                "supporting_source_ids": list(p.supporting_source_ids),
                "conflicting_source_ids": list(p.conflicting_source_ids),
            }
            for p in record.problems
        ],
        "provenance": _provenance_payload(record.provenance),
    }


def _inventory_from(payload: Mapping[str, object]) -> ProblemInventoryRecord:
    problems = payload["problems"]
    assert isinstance(problems, list)
    return ProblemInventoryRecord(
        run_id=str(payload["run_id"]),
        brief_id=str(payload["brief_id"]),
        problems=tuple(
            ProblemEntry(
                statement=str(entry["statement"]),
                kind=ProblemKind(str(entry["kind"])),
                grounding=str(entry["grounding"]),
                supporting_source_ids=_strings(
                    entry["supporting_source_ids"]
                ),
                conflicting_source_ids=_strings(
                    entry["conflicting_source_ids"]
                ),
            )
            for entry in problems
        ),
        provenance=_provenance_from(payload["provenance"]),
    )


def _run_payload(record: MappingRunRecord) -> dict[str, object]:
    coverage = record.coverage
    return {
        "id": record.id,
        "run_id": record.run_id,
        "brief_id": record.brief_id,
        "query_execution_ids": list(record.query_execution_ids),
        "screening_ids": list(record.screening_ids),
        "extraction_ids": list(record.extraction_ids),
        "field_map_id": record.field_map_id,
        "inventory_id": record.inventory_id,
        "model_calls": record.model_calls,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "coverage": {
            "queries_executed": coverage.queries_executed,
            "total_retrieved": coverage.total_retrieved,
            "unique_sources": coverage.unique_sources,
            "screened": coverage.screened,
            "screening_truncated": coverage.screening_truncated,
            "relevant": coverage.relevant,
            "excluded": coverage.excluded,
            "uncertain": coverage.uncertain,
            "abstract_level": coverage.abstract_level,
            "metadata_level": coverage.metadata_level,
            "extraction_eligible": coverage.extraction_eligible,
            "extracted": coverage.extracted,
            "extraction_truncated": coverage.extraction_truncated,
            "insufficient_support": coverage.insufficient_support,
            "saturation": coverage.saturation,
        },
    }


def _run_from(payload: Mapping[str, object]) -> MappingRunRecord:
    coverage = payload["coverage"]
    assert isinstance(coverage, Mapping)
    return MappingRunRecord(
        run_id=str(payload["run_id"]),
        brief_id=str(payload["brief_id"]),
        query_execution_ids=_strings(payload["query_execution_ids"]),
        screening_ids=_strings(payload["screening_ids"]),
        extraction_ids=_strings(payload["extraction_ids"]),
        field_map_id=str(payload["field_map_id"]),
        inventory_id=str(payload["inventory_id"]),
        model_calls=int(str(payload["model_calls"])),
        input_tokens=int(str(payload["input_tokens"])),
        output_tokens=int(str(payload["output_tokens"])),
        coverage=CoverageReport(
            queries_executed=int(str(coverage["queries_executed"])),
            total_retrieved=int(str(coverage["total_retrieved"])),
            unique_sources=int(str(coverage["unique_sources"])),
            screened=int(str(coverage["screened"])),
            screening_truncated=int(str(coverage["screening_truncated"])),
            relevant=int(str(coverage["relevant"])),
            excluded=int(str(coverage["excluded"])),
            uncertain=int(str(coverage["uncertain"])),
            abstract_level=int(str(coverage["abstract_level"])),
            metadata_level=int(str(coverage["metadata_level"])),
            extraction_eligible=int(str(coverage["extraction_eligible"])),
            extracted=int(str(coverage["extracted"])),
            extraction_truncated=int(str(coverage["extraction_truncated"])),
            insufficient_support=int(str(coverage["insufficient_support"])),
            saturation=float(str(coverage["saturation"])),
        ),
    )


def _strings(value: object) -> tuple[str, ...]:
    assert isinstance(value, list)
    return tuple(str(item) for item in value)
