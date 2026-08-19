"""Durable storage for prior-art challenge runs, mirroring the ideation
store.

One directory per record kind — directives, query executions, similarity
screenings, work comparisons, per-candidate assessments, completed runs —
plus ``rejected/`` for every gate-refused model payload, preserved as
data with the provenance of the call that produced it. Writes are
write-once and verify-on-repeat: identical re-recording is a no-op,
different content under the same id raises. Ids are recomputed from what
was read, never trusted from the file, so a tampered record fails loudly
on load.

Two internal-consistency rules go beyond plain write-once: a run may
hold at most one assessment per candidate and at most one completed-run
record — a second, different verdict on the same candidate in the same
run, or a second account of the same run, is a conflict to raise, not a
record to file alongside the first.

Nothing here may ever hold a credential: records store fingerprints,
ids, token counts and text, not keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ..core.ids import occurrence_id
from ..literature.retrieval import ResultOrdering
from ..mapping.records import CallProvenance, SupportLocation
from .assessment import (
    PriorArtAssessment,
    PriorArtReason,
    PriorArtReasonCode,
    PriorArtThresholds,
    PriorArtVerdict,
)
from .directive import PriorArtDirective
from .records import (
    ComparisonDimension,
    DimensionComparison,
    OverlapHypothesis,
    PriorArtCoverage,
    PriorArtQueryExecution,
    PriorArtQueryFamily,
    PriorArtRunRecord,
    PriorArtScreeningRecord,
    SimilarityDecision,
    SimilarityLabel,
    WorkComparison,
)

_RECORD_SUFFIX: Final = ".json"

_DIRECTIVES: Final = "directives"
_EXECUTIONS: Final = "executions"
_SCREENINGS: Final = "screenings"
_COMPARISONS: Final = "comparisons"
_ASSESSMENTS: Final = "assessments"
_RUNS: Final = "runs"
_REJECTED: Final = "rejected"


class PriorArtConflictError(RuntimeError):
    """A write-once prior-art artifact would be overwritten with
    different content, or a run would hold two assessments of one
    candidate or two run records."""


class PriorArtIntegrityError(RuntimeError):
    """A stored prior-art record no longer matches its own identity."""


class PriorArtStore:
    """File-backed, write-once storage for one or more prior-art
    challenge runs under one injected root."""

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
                raise PriorArtConflictError(
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
            raise PriorArtIntegrityError(
                f"{kind} record filed under {filed_as} re-derives id "
                f"{rederived}; refusing to load a record that no longer "
                f"matches its name"
            )

    # -- directives ------------------------------------------------------------

    def record_directive(
        self, directive: PriorArtDirective
    ) -> PriorArtDirective:
        self._write_once(
            _DIRECTIVES, directive.id, _directive_payload(directive)
        )
        return directive

    def get_directive(self, directive_id: str) -> PriorArtDirective | None:
        payload = self._load(_DIRECTIVES, directive_id)
        if payload is None:
            return None
        directive = _directive_from(payload)
        self._verify(_DIRECTIVES, directive_id, directive.id)
        return directive

    # -- query executions ------------------------------------------------------

    def record_query_execution(
        self, record: PriorArtQueryExecution
    ) -> PriorArtQueryExecution:
        self._write_once(_EXECUTIONS, record.id, _execution_payload(record))
        return record

    def get_query_execution(
        self, record_id: str
    ) -> PriorArtQueryExecution | None:
        payload = self._load(_EXECUTIONS, record_id)
        if payload is None:
            return None
        record = _execution_from(payload)
        self._verify(_EXECUTIONS, record_id, record.id)
        return record

    # -- screenings ------------------------------------------------------------

    def record_screening(
        self, record: PriorArtScreeningRecord
    ) -> PriorArtScreeningRecord:
        self._write_once(_SCREENINGS, record.id, _screening_payload(record))
        return record

    def get_screening(
        self, record_id: str
    ) -> PriorArtScreeningRecord | None:
        payload = self._load(_SCREENINGS, record_id)
        if payload is None:
            return None
        record = _screening_from(payload)
        self._verify(_SCREENINGS, record_id, record.id)
        return record

    # -- comparisons -----------------------------------------------------------

    def record_comparison(self, record: WorkComparison) -> WorkComparison:
        self._write_once(_COMPARISONS, record.id, _comparison_payload(record))
        return record

    def get_comparison(self, record_id: str) -> WorkComparison | None:
        payload = self._load(_COMPARISONS, record_id)
        if payload is None:
            return None
        record = _comparison_from(payload)
        self._verify(_COMPARISONS, record_id, record.id)
        return record

    # -- assessments -----------------------------------------------------------

    def record_prior_art_assessment(
        self, record: PriorArtAssessment
    ) -> PriorArtAssessment:
        for existing_id in self._ids(_ASSESSMENTS):
            if existing_id == record.id:
                continue
            existing = self.get_prior_art_assessment(existing_id)
            assert existing is not None
            if (
                existing.run_id == record.run_id
                and existing.candidate_id == record.candidate_id
            ):
                raise PriorArtConflictError(
                    f"run {record.run_id} already assessed candidate "
                    f"{record.candidate_id}; a second verdict is a "
                    f"conflict, not a record"
                )
        self._write_once(_ASSESSMENTS, record.id, _assessment_payload(record))
        return record

    def get_prior_art_assessment(self, record_id: str) -> PriorArtAssessment | None:
        payload = self._load(_ASSESSMENTS, record_id)
        if payload is None:
            return None
        record = _assessment_from(payload)
        self._verify(_ASSESSMENTS, record_id, record.id)
        return record

    def prior_art_assessments(self) -> tuple[PriorArtAssessment, ...]:
        loaded = []
        for record_id in self._ids(_ASSESSMENTS):
            record = self.get_prior_art_assessment(record_id)
            assert record is not None
            loaded.append(record)
        return tuple(loaded)

    def assessment_for_candidate(
        self, run_id: str, candidate_id: str
    ) -> PriorArtAssessment | None:
        for record in self.prior_art_assessments():
            if (
                record.run_id == run_id
                and record.candidate_id == candidate_id
            ):
                return record
        return None

    # -- completed runs --------------------------------------------------------

    def record_run(self, record: PriorArtRunRecord) -> PriorArtRunRecord:
        for existing_id in self._ids(_RUNS):
            if existing_id == record.id:
                continue
            existing = self.get_run(existing_id)
            assert existing is not None
            if existing.run_id == record.run_id:
                raise PriorArtConflictError(
                    f"run {record.run_id} is already recorded; a second "
                    f"account of one run is a conflict, not a record"
                )
        self._write_once(_RUNS, record.id, _run_payload(record))
        return record

    def get_run(self, record_id: str) -> PriorArtRunRecord | None:
        payload = self._load(_RUNS, record_id)
        if payload is None:
            return None
        record = _run_from(payload)
        self._verify(_RUNS, record_id, record.id)
        return record

    def runs(self) -> tuple[PriorArtRunRecord, ...]:
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
        every rule that fired, the call's provenance handles, and the
        raw payload. Returns the file written."""
        directory = self._root / _REJECTED
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{occurrence_id('prej')}{_RECORD_SUFFIX}"
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


def _directive_payload(directive: PriorArtDirective) -> dict[str, object]:
    return {
        "id": directive.id,
        "ideation_run_record_id": directive.ideation_run_record_id,
        "cutoff_date": directive.cutoff_date,
        "recent_window_start": directive.recent_window_start,
        "results_per_query": directive.results_per_query,
        "max_screened_per_candidate": directive.max_screened_per_candidate,
        "max_compared_works": directive.max_compared_works,
        "max_model_calls": directive.max_model_calls,
    }


def _directive_from(payload: Mapping[str, object]) -> PriorArtDirective:
    return PriorArtDirective(
        ideation_run_record_id=str(payload["ideation_run_record_id"]),
        cutoff_date=str(payload["cutoff_date"]),
        recent_window_start=str(payload["recent_window_start"]),
        results_per_query=int(str(payload["results_per_query"])),
        max_screened_per_candidate=int(
            str(payload["max_screened_per_candidate"])
        ),
        max_compared_works=int(str(payload["max_compared_works"])),
        max_model_calls=int(str(payload["max_model_calls"])),
    )


def _execution_payload(record: PriorArtQueryExecution) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": record.id,
        "run_id": record.run_id,
        "candidate_id": record.candidate_id,
        "family": record.family.value,
        "text": record.text,
        "from_date": record.from_date,
        "to_date": record.to_date,
        "query_fingerprint": record.query_fingerprint,
        "search_record_id": record.search_record_id,
        "retrieved": record.retrieved,
        "new_unique": record.new_unique,
        "from_cache": record.from_cache,
        "ordering": record.ordering.value,
    }
    # Pre-5D.1 records carried no plan; the keys appear only when the
    # fields do, so the old files stay byte-identical and re-derivable.
    if record.plan_groups or record.renderer:
        payload["plan_groups"] = [
            list(group) for group in record.plan_groups
        ]
        payload["renderer"] = record.renderer
    return payload


def _execution_from(payload: Mapping[str, object]) -> PriorArtQueryExecution:
    return PriorArtQueryExecution(
        run_id=str(payload["run_id"]),
        candidate_id=str(payload["candidate_id"]),
        family=PriorArtQueryFamily(str(payload["family"])),
        text=str(payload["text"]),
        from_date=str(payload["from_date"]),
        to_date=str(payload["to_date"]),
        query_fingerprint=str(payload["query_fingerprint"]),
        search_record_id=str(payload["search_record_id"]),
        retrieved=int(str(payload["retrieved"])),
        new_unique=int(str(payload["new_unique"])),
        from_cache=bool(payload["from_cache"]),
        ordering=ResultOrdering(str(payload["ordering"])),
        plan_groups=tuple(
            _strings(group) for group in _group_lists(payload)
        ),
        renderer=str(payload.get("renderer", "")),
    )


def _group_lists(payload: Mapping[str, object]) -> list[object]:
    groups = payload.get("plan_groups", [])
    assert isinstance(groups, list)
    return groups


def _screening_payload(record: PriorArtScreeningRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": record.id,
        "run_id": record.run_id,
        "candidate_id": record.candidate_id,
        "source_id": record.source_id,
        "known_prior_art": record.known_prior_art,
        "decision": record.decision.value,
        "reason": record.reason,
        "provenance": _provenance_payload(record.provenance),
    }
    # Pre-5D.2 records carried no hypothesis; the key appears only when
    # the field does, so the old files stay byte-identical and
    # re-derivable.
    if record.overlap_hypothesis is not None:
        hypothesis = record.overlap_hypothesis
        payload["overlap_hypothesis"] = {
            "candidate_claim": hypothesis.candidate_claim,
            "source_text": hypothesis.source_text,
            "support_location": hypothesis.support_location.value,
            "dimension": hypothesis.dimension.value,
            "rationale": hypothesis.rationale,
        }
    return payload


def _screening_from(
    payload: Mapping[str, object],
) -> PriorArtScreeningRecord:
    hypothesis = payload.get("overlap_hypothesis")
    return PriorArtScreeningRecord(
        run_id=str(payload["run_id"]),
        candidate_id=str(payload["candidate_id"]),
        source_id=str(payload["source_id"]),
        known_prior_art=bool(payload["known_prior_art"]),
        decision=SimilarityDecision(str(payload["decision"])),
        reason=str(payload["reason"]),
        provenance=_provenance_from(payload["provenance"]),
        overlap_hypothesis=(
            _hypothesis_from(hypothesis) if hypothesis is not None else None
        ),
    )


def _hypothesis_from(payload: object) -> OverlapHypothesis:
    assert isinstance(payload, Mapping)
    return OverlapHypothesis(
        candidate_claim=str(payload["candidate_claim"]),
        source_text=str(payload["source_text"]),
        support_location=SupportLocation(str(payload["support_location"])),
        dimension=ComparisonDimension(str(payload["dimension"])),
        rationale=str(payload["rationale"]),
    )


def _comparison_payload(record: WorkComparison) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "candidate_id": record.candidate_id,
        "source_id": record.source_id,
        "known_prior_art": record.known_prior_art,
        "dimensions": [
            {
                "dimension": entry.dimension.value,
                "candidate_position": entry.candidate_position,
                "prior_work_position": entry.prior_work_position,
                "support_location": entry.support_location.value,
                "support_snippet": entry.support_snippet,
            }
            for entry in record.dimensions
        ],
        "overlap_features": list(record.overlap_features),
        "material_differences": list(record.material_differences),
        "similarity": record.similarity.value,
        "provenance": _provenance_payload(record.provenance),
    }


def _comparison_from(payload: Mapping[str, object]) -> WorkComparison:
    dimensions = payload["dimensions"]
    assert isinstance(dimensions, list)
    return WorkComparison(
        run_id=str(payload["run_id"]),
        candidate_id=str(payload["candidate_id"]),
        source_id=str(payload["source_id"]),
        known_prior_art=bool(payload["known_prior_art"]),
        dimensions=tuple(
            DimensionComparison(
                dimension=ComparisonDimension(str(entry["dimension"])),
                candidate_position=str(entry["candidate_position"]),
                prior_work_position=str(entry["prior_work_position"]),
                support_location=SupportLocation(
                    str(entry["support_location"])
                ),
                support_snippet=str(entry["support_snippet"]),
            )
            for entry in dimensions
        ),
        overlap_features=_strings(payload["overlap_features"]),
        material_differences=_strings(payload["material_differences"]),
        similarity=SimilarityLabel(str(payload["similarity"])),
        provenance=_provenance_from(payload["provenance"]),
    )


def _coverage_payload(coverage: PriorArtCoverage) -> dict[str, object]:
    return {
        "families_executed": list(coverage.families_executed),
        "queries_executed": coverage.queries_executed,
        "total_retrieved": coverage.total_retrieved,
        "unique_sources": coverage.unique_sources,
        "overlap": coverage.overlap,
        "saturation": coverage.saturation,
        "post_cutoff_excluded": coverage.post_cutoff_excluded,
        "undated_sources": coverage.undated_sources,
        "abstract_level": coverage.abstract_level,
        "metadata_level": coverage.metadata_level,
        "known_prior_art_listed": coverage.known_prior_art_listed,
        "known_prior_art_recovered": coverage.known_prior_art_recovered,
        "screened": coverage.screened,
        "potential_overlap": coverage.potential_overlap,
        "related": coverage.related,
        "unrelated": coverage.unrelated,
        "undecidable": coverage.undecidable,
        "metadata_ambiguous": coverage.metadata_ambiguous,
        "screening_truncated": coverage.screening_truncated,
        "compared_works": coverage.compared_works,
    }


def _coverage_from(payload: object) -> PriorArtCoverage:
    assert isinstance(payload, Mapping)
    return PriorArtCoverage(
        families_executed=_strings(payload["families_executed"]),
        queries_executed=int(str(payload["queries_executed"])),
        total_retrieved=int(str(payload["total_retrieved"])),
        unique_sources=int(str(payload["unique_sources"])),
        overlap=int(str(payload["overlap"])),
        saturation=float(str(payload["saturation"])),
        post_cutoff_excluded=int(str(payload["post_cutoff_excluded"])),
        undated_sources=int(str(payload["undated_sources"])),
        abstract_level=int(str(payload["abstract_level"])),
        metadata_level=int(str(payload["metadata_level"])),
        known_prior_art_listed=int(str(payload["known_prior_art_listed"])),
        known_prior_art_recovered=int(
            str(payload["known_prior_art_recovered"])
        ),
        screened=int(str(payload["screened"])),
        potential_overlap=int(str(payload["potential_overlap"])),
        related=int(str(payload["related"])),
        unrelated=int(str(payload["unrelated"])),
        undecidable=int(str(payload["undecidable"])),
        metadata_ambiguous=int(str(payload["metadata_ambiguous"])),
        screening_truncated=int(str(payload["screening_truncated"])),
        compared_works=int(str(payload["compared_works"])),
    )


def _assessment_payload(record: PriorArtAssessment) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "candidate_id": record.candidate_id,
        "directive_id": record.directive_id,
        "verdict": record.verdict.value,
        "overlapping_work_ids": list(record.overlapping_work_ids),
        "compared_work_ids": list(record.compared_work_ids),
        "reasons": [
            {"code": reason.code.value, "detail": reason.detail}
            for reason in record.reasons
        ],
        "thresholds": {
            "min_unique_sources": record.thresholds.min_unique_sources,
            "max_undecidable_fraction": (
                record.thresholds.max_undecidable_fraction
            ),
            "min_compared_works": record.thresholds.min_compared_works,
        },
        "coverage": _coverage_payload(record.coverage),
    }


def _assessment_from(payload: Mapping[str, object]) -> PriorArtAssessment:
    reasons = payload["reasons"]
    thresholds = payload["thresholds"]
    assert isinstance(reasons, list)
    assert isinstance(thresholds, Mapping)
    return PriorArtAssessment(
        run_id=str(payload["run_id"]),
        candidate_id=str(payload["candidate_id"]),
        directive_id=str(payload["directive_id"]),
        verdict=PriorArtVerdict(str(payload["verdict"])),
        overlapping_work_ids=_strings(payload["overlapping_work_ids"]),
        compared_work_ids=_strings(payload["compared_work_ids"]),
        reasons=tuple(
            PriorArtReason(
                code=PriorArtReasonCode(str(entry["code"])),
                detail=str(entry["detail"]),
            )
            for entry in reasons
        ),
        thresholds=PriorArtThresholds(
            min_unique_sources=int(str(thresholds["min_unique_sources"])),
            max_undecidable_fraction=float(
                str(thresholds["max_undecidable_fraction"])
            ),
            min_compared_works=int(str(thresholds["min_compared_works"])),
        ),
        coverage=_coverage_from(payload["coverage"]),
    )


def _run_payload(record: PriorArtRunRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "directive_id": record.directive_id,
        "ideation_run_record_id": record.ideation_run_record_id,
        "ideation_run_id": record.ideation_run_id,
        "assessment_id": record.assessment_id,
        "map_run_id": record.map_run_id,
        "snapshot_id": record.snapshot_id,
        "candidate_ids": list(record.candidate_ids),
        "prior_art_assessment_ids": list(record.prior_art_assessment_ids),
        "query_execution_ids": list(record.query_execution_ids),
        "screening_ids": list(record.screening_ids),
        "comparison_ids": list(record.comparison_ids),
        "model_calls": record.model_calls,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
    }


def _run_from(payload: Mapping[str, object]) -> PriorArtRunRecord:
    return PriorArtRunRecord(
        run_id=str(payload["run_id"]),
        directive_id=str(payload["directive_id"]),
        ideation_run_record_id=str(payload["ideation_run_record_id"]),
        ideation_run_id=str(payload["ideation_run_id"]),
        assessment_id=str(payload["assessment_id"]),
        map_run_id=str(payload["map_run_id"]),
        snapshot_id=str(payload["snapshot_id"]),
        candidate_ids=_strings(payload["candidate_ids"]),
        prior_art_assessment_ids=_strings(
            payload["prior_art_assessment_ids"]
        ),
        query_execution_ids=_strings(payload["query_execution_ids"]),
        screening_ids=_strings(payload["screening_ids"]),
        comparison_ids=_strings(payload["comparison_ids"]),
        model_calls=int(str(payload["model_calls"])),
        input_tokens=int(str(payload["input_tokens"])),
        output_tokens=int(str(payload["output_tokens"])),
    )


def _strings(value: object) -> tuple[str, ...]:
    assert isinstance(value, list)
    return tuple(str(entry) for entry in value)
