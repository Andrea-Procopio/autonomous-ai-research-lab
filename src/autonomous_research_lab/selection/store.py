"""Durable storage for selection runs, mirroring the prior-art store.

One directory per record kind — directives and completed runs — plus
``rejected/`` for every gate-refused model payload, preserved as data
with the provenance of the call that produced it. A selection run's
entire outcome is one nested record: the comparative review is one
inseparable joint judgment over the whole eligible set, so splitting its
parts into separate files would fabricate an independence the call never
had, and write-once becomes atomic — the whole nest is content-addressed
together.

Writes are write-once and verify-on-repeat: identical re-recording is a
no-op, different content under the same id raises. Ids are recomputed
from what was read, never trusted from the file, so a tampered record
fails loudly on load. One internal-consistency rule goes beyond plain
write-once: at most one completed-run record per run — a second account
of the same run is a conflict to raise, not a record to file alongside
the first. Two selection runs over the same portfolio are two
occurrences, both durable: an unselected candidate stays available to
future runs.

Nothing here may ever hold a credential: records store fingerprints,
ids, token counts and text, not keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ..core.ids import occurrence_id
from ..mapping.records import CallProvenance
from ..priorart.assessment import (
    PriorArtReason,
    PriorArtReasonCode,
    PriorArtVerdict,
)
from .directive import SelectionDirective
from .records import (
    REVIEW_FIELDS,
    CandidateReview,
    DisqualificationGround,
    DisqualifierDimension,
    HardDisqualifier,
    IneligibleCandidate,
    PairwiseComparison,
    SelectionDecision,
    SelectionOutcome,
    SelectionRationale,
    SelectionRunRecord,
)

_RECORD_SUFFIX: Final = ".json"

_DIRECTIVES: Final = "directives"
_RUNS: Final = "runs"
_REJECTED: Final = "rejected"


class SelectionConflictError(RuntimeError):
    """A write-once selection artifact would be overwritten with
    different content, or a run would hold two run records."""


class SelectionIntegrityError(RuntimeError):
    """A stored selection record no longer matches its own identity."""


class SelectionStore:
    """File-backed, write-once storage for one or more selection runs
    under one injected root."""

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
                raise SelectionConflictError(
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
            raise SelectionIntegrityError(
                f"{kind} record filed under {filed_as} re-derives id "
                f"{rederived}; refusing to load a record that no longer "
                f"matches its name"
            )

    # -- directives ------------------------------------------------------------

    def record_directive(
        self, directive: SelectionDirective
    ) -> SelectionDirective:
        self._write_once(
            _DIRECTIVES, directive.id, _directive_payload(directive)
        )
        return directive

    def get_directive(self, directive_id: str) -> SelectionDirective | None:
        payload = self._load(_DIRECTIVES, directive_id)
        if payload is None:
            return None
        directive = _directive_from(payload)
        self._verify(_DIRECTIVES, directive_id, directive.id)
        return directive

    # -- completed runs ----------------------------------------------------------

    def record_run(self, record: SelectionRunRecord) -> SelectionRunRecord:
        for existing_id in self._ids(_RUNS):
            if existing_id == record.id:
                continue
            existing = self.get_run(existing_id)
            assert existing is not None
            if existing.run_id == record.run_id:
                raise SelectionConflictError(
                    f"run {record.run_id} is already recorded; a second "
                    f"account of one run is a conflict, not a record"
                )
        self._write_once(_RUNS, record.id, _run_payload(record))
        return record

    def get_run(self, record_id: str) -> SelectionRunRecord | None:
        payload = self._load(_RUNS, record_id)
        if payload is None:
            return None
        record = _run_from(payload)
        self._verify(_RUNS, record_id, record.id)
        return record

    def runs(self) -> tuple[SelectionRunRecord, ...]:
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
        path = directory / f"{occurrence_id('srej')}{_RECORD_SUFFIX}"
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


def _directive_payload(directive: SelectionDirective) -> dict[str, object]:
    return {
        "id": directive.id,
        "prior_art_run_record_id": directive.prior_art_run_record_id,
        "compute_constraint": directive.compute_constraint,
        "data_constraint": directive.data_constraint,
        "time_constraint": directive.time_constraint,
        "experimental_constraint": directive.experimental_constraint,
        "max_eligible_candidates": directive.max_eligible_candidates,
        "max_model_calls": directive.max_model_calls,
    }


def _directive_from(payload: Mapping[str, object]) -> SelectionDirective:
    return SelectionDirective(
        prior_art_run_record_id=str(payload["prior_art_run_record_id"]),
        compute_constraint=str(payload["compute_constraint"]),
        data_constraint=str(payload["data_constraint"]),
        time_constraint=str(payload["time_constraint"]),
        experimental_constraint=str(payload["experimental_constraint"]),
        max_eligible_candidates=int(str(payload["max_eligible_candidates"])),
        max_model_calls=int(str(payload["max_model_calls"])),
    )


def _disqualifier_payload(entry: HardDisqualifier) -> dict[str, object]:
    return {
        "ground": entry.ground.value,
        "dimension": entry.dimension.value,
        "candidate_text": entry.candidate_text,
        "constraint_text": entry.constraint_text,
        "why_unrepairable": entry.why_unrepairable,
    }


def _disqualifier_from(payload: Mapping[str, object]) -> HardDisqualifier:
    return HardDisqualifier(
        ground=DisqualificationGround(str(payload["ground"])),
        dimension=DisqualifierDimension(str(payload["dimension"])),
        candidate_text=str(payload["candidate_text"]),
        constraint_text=str(payload["constraint_text"]),
        why_unrepairable=str(payload["why_unrepairable"]),
    )


def _review_payload(review: CandidateReview) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": review.candidate_id,
        "prior_art_verdict": review.prior_art_verdict.value,
        "disqualifiers": [
            _disqualifier_payload(entry) for entry in review.disqualifiers
        ],
    }
    for name in REVIEW_FIELDS:
        payload[name] = str(getattr(review, name))
    return payload


def _review_from(payload: Mapping[str, object]) -> CandidateReview:
    fields: dict[str, str] = {
        name: str(payload[name]) for name in REVIEW_FIELDS
    }
    return CandidateReview(
        candidate_id=str(payload["candidate_id"]),
        prior_art_verdict=PriorArtVerdict(str(payload["prior_art_verdict"])),
        disqualifiers=tuple(
            _disqualifier_from(entry)
            for entry in _entries(payload, "disqualifiers")
        ),
        **fields,
    )


def _pair_payload(pair: PairwiseComparison) -> dict[str, object]:
    return {
        "first_candidate_id": pair.first_candidate_id,
        "second_candidate_id": pair.second_candidate_id,
        "comparison": pair.comparison,
    }


def _pair_from(payload: Mapping[str, object]) -> PairwiseComparison:
    return PairwiseComparison(
        first_candidate_id=str(payload["first_candidate_id"]),
        second_candidate_id=str(payload["second_candidate_id"]),
        comparison=str(payload["comparison"]),
    )


def _ineligible_payload(entry: IneligibleCandidate) -> dict[str, object]:
    return {
        "candidate_id": entry.candidate_id,
        "assessment_id": entry.assessment_id,
        "verdict": entry.verdict.value,
        "reasons": [
            {"code": reason.code.value, "detail": reason.detail}
            for reason in entry.reasons
        ],
        "overlapping_work_ids": list(entry.overlapping_work_ids),
    }


def _ineligible_from(payload: Mapping[str, object]) -> IneligibleCandidate:
    return IneligibleCandidate(
        candidate_id=str(payload["candidate_id"]),
        assessment_id=str(payload["assessment_id"]),
        verdict=PriorArtVerdict(str(payload["verdict"])),
        reasons=tuple(
            PriorArtReason(
                code=PriorArtReasonCode(str(entry["code"])),
                detail=str(entry["detail"]),
            )
            for entry in _entries(payload, "reasons")
        ),
        overlapping_work_ids=_strings(payload, "overlapping_work_ids"),
    )


def _decision_payload(decision: SelectionDecision) -> dict[str, object]:
    return {
        "selected_candidate_id": decision.selected_candidate_id,
        "decisive_tradeoff": decision.decisive_tradeoff,
        "why_selected_over": [
            {"candidate_id": entry.candidate_id, "reason": entry.reason}
            for entry in decision.why_selected_over
        ],
        "first_experimental_objective": (
            decision.first_experimental_objective
        ),
        "required_capabilities": list(decision.required_capabilities),
        "residual_risks": list(decision.residual_risks),
        "provenance": _provenance_payload(decision.provenance),
    }


def _decision_from(payload: Mapping[str, object]) -> SelectionDecision:
    return SelectionDecision(
        selected_candidate_id=str(payload["selected_candidate_id"]),
        decisive_tradeoff=str(payload["decisive_tradeoff"]),
        why_selected_over=tuple(
            SelectionRationale(
                candidate_id=str(entry["candidate_id"]),
                reason=str(entry["reason"]),
            )
            for entry in _entries(payload, "why_selected_over")
        ),
        first_experimental_objective=str(
            payload["first_experimental_objective"]
        ),
        required_capabilities=_strings(payload, "required_capabilities"),
        residual_risks=_strings(payload, "residual_risks"),
        provenance=_provenance_from(payload["provenance"]),
    )


def _run_payload(record: SelectionRunRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": record.id,
        "run_id": record.run_id,
        "directive_id": record.directive_id,
        "prior_art_run_record_id": record.prior_art_run_record_id,
        "prior_art_run_id": record.prior_art_run_id,
        "ideation_run_record_id": record.ideation_run_record_id,
        "ideation_run_id": record.ideation_run_id,
        "direction_id": record.direction_id,
        "candidate_ids": list(record.candidate_ids),
        "prior_art_assessment_ids": list(record.prior_art_assessment_ids),
        "eligible_candidate_ids": list(record.eligible_candidate_ids),
        "ineligible": [
            _ineligible_payload(entry) for entry in record.ineligible
        ],
        "disqualified_candidate_ids": list(
            record.disqualified_candidate_ids
        ),
        "reviews": [_review_payload(review) for review in record.reviews],
        "pairwise_comparisons": [
            _pair_payload(pair) for pair in record.pairwise_comparisons
        ],
        "outcome": record.outcome.value,
        "model_calls": record.model_calls,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
    }
    # A NO_ELIGIBLE run has neither; the keys appear only when the
    # fields do, so every file re-derives byte-identically.
    if record.review_provenance is not None:
        payload["review_provenance"] = _provenance_payload(
            record.review_provenance
        )
    if record.decision is not None:
        payload["decision"] = _decision_payload(record.decision)
    return payload


def _run_from(payload: Mapping[str, object]) -> SelectionRunRecord:
    review_provenance = payload.get("review_provenance")
    decision = payload.get("decision")
    assert decision is None or isinstance(decision, Mapping)
    return SelectionRunRecord(
        run_id=str(payload["run_id"]),
        directive_id=str(payload["directive_id"]),
        prior_art_run_record_id=str(payload["prior_art_run_record_id"]),
        prior_art_run_id=str(payload["prior_art_run_id"]),
        ideation_run_record_id=str(payload["ideation_run_record_id"]),
        ideation_run_id=str(payload["ideation_run_id"]),
        direction_id=str(payload["direction_id"]),
        candidate_ids=_strings(payload, "candidate_ids"),
        prior_art_assessment_ids=_strings(
            payload, "prior_art_assessment_ids"
        ),
        eligible_candidate_ids=_strings(payload, "eligible_candidate_ids"),
        ineligible=tuple(
            _ineligible_from(entry)
            for entry in _entries(payload, "ineligible")
        ),
        disqualified_candidate_ids=_strings(
            payload, "disqualified_candidate_ids"
        ),
        reviews=tuple(
            _review_from(entry) for entry in _entries(payload, "reviews")
        ),
        pairwise_comparisons=tuple(
            _pair_from(entry)
            for entry in _entries(payload, "pairwise_comparisons")
        ),
        review_provenance=(
            _provenance_from(review_provenance)
            if review_provenance is not None
            else None
        ),
        outcome=SelectionOutcome(str(payload["outcome"])),
        decision=_decision_from(decision) if decision is not None else None,
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
