"""Durable storage for candidate-generation runs, mirroring the mapping
store.

One directory per record kind — CFP snapshots, directives, extracted
directions, candidate ideas, completed runs — plus ``rejected/`` for
every gate-refused model payload, preserved as data with the provenance
of the call that produced it. Writes are write-once and verify-on-repeat:
identical re-recording is a no-op, different content under the same id
raises. Ids are recomputed from what was read, never trusted from the
file, so a tampered record fails loudly on load.

Two internal-consistency rules go beyond plain write-once: a run may hold
at most one extracted direction and at most one completed-run record — a
second, different reading of the same snapshot in the same run, or a
second account of the same run, is a conflict to raise, not a record to
file alongside the first.

Nothing here may ever hold a credential: records store fingerprints, ids,
token counts and text, not keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ..core.ids import occurrence_id
from ..mapping.adequacy import SupportTier
from ..mapping.records import CallProvenance, ProblemKind, ThemeEra
from .direction import CfpSnapshot, DirectionRecord
from .directive import IdeationDirective
from .records import (
    AddressedProblem,
    CandidateIdea,
    DataRequirement,
    DataStatus,
    IdeationRunRecord,
    NoveltyStatus,
    PortfolioReport,
    Prediction,
    ResourceEstimate,
    TargetedTheme,
)

_RECORD_SUFFIX: Final = ".json"

_SNAPSHOTS: Final = "snapshots"
_DIRECTIVES: Final = "directives"
_DIRECTIONS: Final = "directions"
_IDEAS: Final = "ideas"
_RUNS: Final = "runs"
_REJECTED: Final = "rejected"


class IdeationConflictError(RuntimeError):
    """A write-once ideation artifact would be overwritten with different
    content, or a run would hold two directions or two run records."""


class IdeationIntegrityError(RuntimeError):
    """A stored ideation record no longer matches its own identity."""


class IdeationStore:
    """File-backed, write-once storage for one or more candidate-
    generation runs under one injected root."""

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
                raise IdeationConflictError(
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
            raise IdeationIntegrityError(
                f"{kind} record filed under {filed_as} re-derives id "
                f"{rederived}; refusing to load a record that no longer "
                f"matches its name"
            )

    # -- snapshots -------------------------------------------------------------

    def record_snapshot(self, snapshot: CfpSnapshot) -> CfpSnapshot:
        self._write_once(
            _SNAPSHOTS, snapshot.id, _snapshot_payload(snapshot)
        )
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> CfpSnapshot | None:
        payload = self._load(_SNAPSHOTS, snapshot_id)
        if payload is None:
            return None
        snapshot = _snapshot_from(payload)
        self._verify(_SNAPSHOTS, snapshot_id, snapshot.id)
        return snapshot

    # -- directives ------------------------------------------------------------

    def record_directive(
        self, directive: IdeationDirective
    ) -> IdeationDirective:
        self._write_once(
            _DIRECTIVES, directive.id, _directive_payload(directive)
        )
        return directive

    def get_directive(self, directive_id: str) -> IdeationDirective | None:
        payload = self._load(_DIRECTIVES, directive_id)
        if payload is None:
            return None
        directive = _directive_from(payload)
        self._verify(_DIRECTIVES, directive_id, directive.id)
        return directive

    # -- directions ------------------------------------------------------------

    def record_direction(self, record: DirectionRecord) -> DirectionRecord:
        for existing_id in self._ids(_DIRECTIONS):
            if existing_id == record.id:
                continue
            existing = self.get_direction(existing_id)
            assert existing is not None
            if existing.run_id == record.run_id:
                raise IdeationConflictError(
                    f"run {record.run_id} already extracted a direction; a "
                    f"second reading is a conflict, not a record"
                )
        self._write_once(_DIRECTIONS, record.id, _direction_payload(record))
        return record

    def get_direction(self, record_id: str) -> DirectionRecord | None:
        payload = self._load(_DIRECTIONS, record_id)
        if payload is None:
            return None
        record = _direction_from(payload)
        self._verify(_DIRECTIONS, record_id, record.id)
        return record

    # -- candidate ideas -------------------------------------------------------

    def record_idea(self, record: CandidateIdea) -> CandidateIdea:
        self._write_once(_IDEAS, record.id, _idea_payload(record))
        return record

    def get_idea(self, record_id: str) -> CandidateIdea | None:
        payload = self._load(_IDEAS, record_id)
        if payload is None:
            return None
        record = _idea_from(payload)
        self._verify(_IDEAS, record_id, record.id)
        return record

    def ideas(self) -> tuple[CandidateIdea, ...]:
        loaded = []
        for record_id in self._ids(_IDEAS):
            record = self.get_idea(record_id)
            assert record is not None
            loaded.append(record)
        return tuple(loaded)

    # -- completed runs --------------------------------------------------------

    def record_run(self, record: IdeationRunRecord) -> IdeationRunRecord:
        for existing_id in self._ids(_RUNS):
            if existing_id == record.id:
                continue
            existing = self.get_run(existing_id)
            assert existing is not None
            if existing.run_id == record.run_id:
                raise IdeationConflictError(
                    f"run {record.run_id} is already recorded; a second "
                    f"account of one run is a conflict, not a record"
                )
        self._write_once(_RUNS, record.id, _run_payload(record))
        return record

    def get_run(self, record_id: str) -> IdeationRunRecord | None:
        payload = self._load(_RUNS, record_id)
        if payload is None:
            return None
        record = _run_from(payload)
        self._verify(_RUNS, record_id, record.id)
        return record

    def runs(self) -> tuple[IdeationRunRecord, ...]:
        loaded = []
        for record_id in self._ids(_RUNS):
            record = self.get_run(record_id)
            assert record is not None
            loaded.append(record)
        return tuple(loaded)

    def runs_for_assessment(
        self, assessment_id: str
    ) -> tuple[IdeationRunRecord, ...]:
        """Every completed run over one assessment — plural on purpose:
        two runs of one directive are two legitimate occurrences."""
        return tuple(
            record
            for record in self.runs()
            if record.assessment_id == assessment_id
        )

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
        path = directory / f"{occurrence_id('irej')}{_RECORD_SUFFIX}"
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


def _snapshot_payload(snapshot: CfpSnapshot) -> dict[str, object]:
    return {
        "id": snapshot.id,
        "source_url": snapshot.source_url,
        "supplied_at": snapshot.supplied_at,
        "text": snapshot.text,
        "text_sha256": snapshot.text_sha256,
    }


def _snapshot_from(payload: Mapping[str, object]) -> CfpSnapshot:
    return CfpSnapshot(
        source_url=str(payload["source_url"]),
        supplied_at=str(payload["supplied_at"]),
        text=str(payload["text"]),
        text_sha256=str(payload["text_sha256"]),
    )


def _directive_payload(directive: IdeationDirective) -> dict[str, object]:
    return {
        "id": directive.id,
        "assessment_id": directive.assessment_id,
        "snapshot_id": directive.snapshot_id,
        "max_candidates": directive.max_candidates,
        "max_model_calls": directive.max_model_calls,
    }


def _directive_from(payload: Mapping[str, object]) -> IdeationDirective:
    return IdeationDirective(
        assessment_id=str(payload["assessment_id"]),
        snapshot_id=str(payload["snapshot_id"]),
        max_candidates=int(str(payload["max_candidates"])),
        max_model_calls=int(str(payload["max_model_calls"])),
    )


def _direction_payload(record: DirectionRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "snapshot_id": record.snapshot_id,
        "scope": record.scope,
        "topics": list(record.topics),
        "constraints": list(record.constraints),
        "relevant_dates": list(record.relevant_dates),
        "provenance": _provenance_payload(record.provenance),
    }


def _direction_from(payload: Mapping[str, object]) -> DirectionRecord:
    return DirectionRecord(
        run_id=str(payload["run_id"]),
        snapshot_id=str(payload["snapshot_id"]),
        scope=str(payload["scope"]),
        topics=_strings(payload["topics"]),
        constraints=_strings(payload["constraints"]),
        relevant_dates=_strings(payload["relevant_dates"]),
        provenance=_provenance_from(payload["provenance"]),
    )


def _idea_payload(record: CandidateIdea) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "title": record.title,
        "research_question": record.research_question,
        "proposed_contribution": record.proposed_contribution,
        "mechanism": record.mechanism,
        "hypothesis": record.hypothesis,
        "grounding": record.grounding,
        "predictions": [
            {"text": p.text, "falsifier": p.falsifier}
            for p in record.predictions
        ],
        "datasets": [
            {"name": d.name, "status": d.status.value, "role": d.role}
            for d in record.datasets
        ],
        "metrics": list(record.metrics),
        "evaluation_protocol": record.evaluation_protocol,
        "baselines": list(record.baselines),
        "ablations": list(record.ablations),
        "resources": {
            "compute": record.resources.compute,
            "data": record.resources.data,
            "implementation": record.resources.implementation,
        },
        "risks": list(record.risks),
        "cfp_alignment": record.cfp_alignment,
        "aligned_topics": list(record.aligned_topics),
        "uncertainty": record.uncertainty,
        "search_terms": list(record.search_terms),
        "addressed_problems": [
            {
                "key": p.key,
                "statement": p.statement,
                "kind": p.kind.value,
                "tier": p.tier.value,
            }
            for p in record.addressed_problems
        ],
        "targeted_themes": [
            {"key": t.key, "name": t.name, "era": t.era.value}
            for t in record.targeted_themes
        ],
        "cited_source_ids": list(record.cited_source_ids),
        "cited_recent": record.cited_recent,
        "cited_foundational": record.cited_foundational,
        "cited_undated": record.cited_undated,
        "novelty_status": record.novelty_status.value,
        "provenance": _provenance_payload(record.provenance),
    }


def _idea_from(payload: Mapping[str, object]) -> CandidateIdea:
    predictions = payload["predictions"]
    datasets = payload["datasets"]
    resources = payload["resources"]
    addressed = payload["addressed_problems"]
    themes = payload["targeted_themes"]
    assert isinstance(predictions, list)
    assert isinstance(datasets, list)
    assert isinstance(resources, Mapping)
    assert isinstance(addressed, list)
    assert isinstance(themes, list)
    return CandidateIdea(
        run_id=str(payload["run_id"]),
        title=str(payload["title"]),
        research_question=str(payload["research_question"]),
        proposed_contribution=str(payload["proposed_contribution"]),
        mechanism=str(payload["mechanism"]),
        hypothesis=str(payload["hypothesis"]),
        grounding=str(payload["grounding"]),
        predictions=tuple(
            Prediction(
                text=str(entry["text"]), falsifier=str(entry["falsifier"])
            )
            for entry in predictions
        ),
        datasets=tuple(
            DataRequirement(
                name=str(entry["name"]),
                status=DataStatus(str(entry["status"])),
                role=str(entry["role"]),
            )
            for entry in datasets
        ),
        metrics=_strings(payload["metrics"]),
        evaluation_protocol=str(payload["evaluation_protocol"]),
        baselines=_strings(payload["baselines"]),
        ablations=_strings(payload["ablations"]),
        resources=ResourceEstimate(
            compute=str(resources["compute"]),
            data=str(resources["data"]),
            implementation=str(resources["implementation"]),
        ),
        risks=_strings(payload["risks"]),
        cfp_alignment=str(payload["cfp_alignment"]),
        aligned_topics=_strings(payload["aligned_topics"]),
        uncertainty=str(payload["uncertainty"]),
        search_terms=_strings(payload["search_terms"]),
        addressed_problems=tuple(
            AddressedProblem(
                key=str(entry["key"]),
                statement=str(entry["statement"]),
                kind=ProblemKind(str(entry["kind"])),
                tier=SupportTier(str(entry["tier"])),
            )
            for entry in addressed
        ),
        targeted_themes=tuple(
            TargetedTheme(
                key=str(entry["key"]),
                name=str(entry["name"]),
                era=ThemeEra(str(entry["era"])),
            )
            for entry in themes
        ),
        cited_source_ids=_strings(payload["cited_source_ids"]),
        cited_recent=int(str(payload["cited_recent"])),
        cited_foundational=int(str(payload["cited_foundational"])),
        cited_undated=int(str(payload["cited_undated"])),
        novelty_status=NoveltyStatus(str(payload["novelty_status"])),
        provenance=_provenance_from(payload["provenance"]),
    )


def _run_payload(record: IdeationRunRecord) -> dict[str, object]:
    portfolio = record.portfolio
    return {
        "id": record.id,
        "run_id": record.run_id,
        "directive_id": record.directive_id,
        "assessment_id": record.assessment_id,
        "map_run_id": record.map_run_id,
        "snapshot_id": record.snapshot_id,
        "direction_id": record.direction_id,
        "candidate_ids": list(record.candidate_ids),
        "refusal_justification": record.refusal_justification,
        "diversity_rationale": record.diversity_rationale,
        "model_calls": record.model_calls,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "portfolio": {
            "problems_total": portfolio.problems_total,
            "problems_addressed": portfolio.problems_addressed,
            "problems_unaddressed": portfolio.problems_unaddressed,
            "unaddressed_statements": list(
                portfolio.unaddressed_statements
            ),
            "addressed_multi_source": portfolio.addressed_multi_source,
            "addressed_tentative": portfolio.addressed_tentative,
            "addressed_single_source_limitation": (
                portfolio.addressed_single_source_limitation
            ),
            "addressed_contradicted": portfolio.addressed_contradicted,
            "candidates": portfolio.candidates,
            "distinct_sources_cited": portfolio.distinct_sources_cited,
            "themes_targeted": portfolio.themes_targeted,
            "distinct_problem_sets": portfolio.distinct_problem_sets,
            "distinct_theme_sets": portfolio.distinct_theme_sets,
            "distinct_dataset_sets": portfolio.distinct_dataset_sets,
            "distinct_metric_sets": portfolio.distinct_metric_sets,
        },
    }


def _run_from(payload: Mapping[str, object]) -> IdeationRunRecord:
    portfolio = payload["portfolio"]
    assert isinstance(portfolio, Mapping)
    return IdeationRunRecord(
        run_id=str(payload["run_id"]),
        directive_id=str(payload["directive_id"]),
        assessment_id=str(payload["assessment_id"]),
        map_run_id=str(payload["map_run_id"]),
        snapshot_id=str(payload["snapshot_id"]),
        direction_id=str(payload["direction_id"]),
        candidate_ids=_strings(payload["candidate_ids"]),
        refusal_justification=str(payload["refusal_justification"]),
        diversity_rationale=str(payload["diversity_rationale"]),
        model_calls=int(str(payload["model_calls"])),
        input_tokens=int(str(payload["input_tokens"])),
        output_tokens=int(str(payload["output_tokens"])),
        portfolio=PortfolioReport(
            problems_total=int(str(portfolio["problems_total"])),
            problems_addressed=int(str(portfolio["problems_addressed"])),
            problems_unaddressed=int(
                str(portfolio["problems_unaddressed"])
            ),
            unaddressed_statements=_strings(
                portfolio["unaddressed_statements"]
            ),
            addressed_multi_source=int(
                str(portfolio["addressed_multi_source"])
            ),
            addressed_tentative=int(str(portfolio["addressed_tentative"])),
            addressed_single_source_limitation=int(
                str(portfolio["addressed_single_source_limitation"])
            ),
            addressed_contradicted=int(
                str(portfolio["addressed_contradicted"])
            ),
            candidates=int(str(portfolio["candidates"])),
            distinct_sources_cited=int(
                str(portfolio["distinct_sources_cited"])
            ),
            themes_targeted=int(str(portfolio["themes_targeted"])),
            distinct_problem_sets=int(
                str(portfolio["distinct_problem_sets"])
            ),
            distinct_theme_sets=int(str(portfolio["distinct_theme_sets"])),
            distinct_dataset_sets=int(
                str(portfolio["distinct_dataset_sets"])
            ),
            distinct_metric_sets=int(
                str(portfolio["distinct_metric_sets"])
            ),
        ),
    )


def _strings(value: object) -> tuple[str, ...]:
    assert isinstance(value, list)
    return tuple(str(item) for item in value)
