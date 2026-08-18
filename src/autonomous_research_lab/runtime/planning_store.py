"""Durable storage for planning decisions, mirroring the implementation store.

One planner invocation yields at most one accepted :class:`PlanningRecord` —
the decision, its scientific references, and the full provider provenance
(request fingerprint, response occurrence id, served model, provider request
id, latency, exact reported token usage). Rejected attempts are preserved as
data under ``rejected/``, never silently discarded, and never expanded into
proposals.

``dispatched/`` holds write-once markers: the deterministic planning
director marks a decision dispatched when it emits the follow-up action the
decision calls for, so a decision is acted on exactly once across steps —
and, because the marker is a file, across process restarts too.

``attempts/`` counts dispatch attempts per decision, one write-once file
each, so the director's bounded retry of a failed dispatch spends from a
budget that survives a resume: a restarted run continues the count where
the previous process left it instead of starting a fresh allowance.

Nothing here may ever hold a credential: records store fingerprints, ids,
token counts and text, not keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from ..core.ids import content_id, occurrence_id

_DECISIONS_DIRNAME: Final = "decisions"
_REJECTED_DIRNAME: Final = "rejected"
_DISPATCHED_DIRNAME: Final = "dispatched"
_ATTEMPTS_DIRNAME: Final = "attempts"
_RECORD_SUFFIX: Final = ".json"


class PlanningConflictError(RuntimeError):
    """A write-once planning artifact would be overwritten with different
    content. Records and dispatch markers are never rewritten."""


class PlanningIntegrityError(RuntimeError):
    """A stored planning record no longer matches its own identity."""


class PlanningAction(StrEnum):
    """The four decisions the planner may make — exactly one per invocation."""

    NEW_EXPERIMENT = "new_experiment"
    REPLICATE = "replicate"
    ABLATION = "ablation"
    STOP = "stop"


class StopReason(StrEnum):
    """Typed reasons a planner may stop the investigation. A stop is a
    scientific decision with a nameable ground, never a shrug."""

    BUDGET_INSUFFICIENT = "budget_insufficient"
    QUESTION_RESOLVED = "question_resolved"
    HYPOTHESIS_REFUTED = "hypothesis_refuted"
    NO_INFORMATIVE_NEXT_EXPERIMENT = "no_informative_next_experiment"


@dataclass(frozen=True, slots=True)
class PlanningRecord:
    """One accepted planning decision with its complete provenance.

    Scientific references (``hypothesis_id`` / ``prediction_id`` /
    ``spec_id``) are the ids of the objects the decision's expansion put in
    front of the governed commit — or, for a replication, the existing
    target. The id derives from every field including the response
    occurrence id, so two identical decisions from distinct calls are
    distinct records.
    """

    invocation_id: str
    action: PlanningAction
    question_id: str
    rationale: str
    evidence_ids: tuple[str, ...]
    hypothesis_id: str = ""
    prediction_id: str = ""
    spec_id: str = ""
    parent_experiment_id: str = ""
    removed_component: str = ""
    replication_seed: int | None = None
    template_id: str = ""
    stop_reason: StopReason | None = None
    repair_count: int = 0
    request_fingerprint: str = ""
    response_id: str = ""
    provider: str = ""
    requested_model: str = ""
    served_model: str = ""
    provider_request_id: str | None = None
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    nominal_cost_usd: float | None = None
    """``None`` unless the provider reported a trustworthy price — unknown
    is not zero."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "plan",
                    self.invocation_id,
                    self.action,
                    self.question_id,
                    self.rationale,
                    self.evidence_ids,
                    self.hypothesis_id,
                    self.prediction_id,
                    self.spec_id,
                    self.parent_experiment_id,
                    self.removed_component,
                    self.replication_seed,
                    self.template_id,
                    self.stop_reason,
                    self.repair_count,
                    self.request_fingerprint,
                    self.response_id,
                ),
            )


class PlanningStore:
    """File-backed, write-once storage for planning decisions, rejected
    attempts, and dispatch markers, under one injected root."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # -- accepted decisions ----------------------------------------------------

    def _record_path(self, decision_id: str) -> Path:
        return self._root / _DECISIONS_DIRNAME / f"{decision_id}{_RECORD_SUFFIX}"

    def record(self, record: PlanningRecord) -> PlanningRecord:
        """Store one decision, write-once. Identical re-recording is a
        no-op; different content under the same id raises."""
        existing = self.get(record.id)
        if existing is not None:
            if existing != record:
                raise PlanningConflictError(
                    f"planning decision {record.id} is already recorded "
                    f"with different content; records are never rewritten"
                )
            return existing
        path = self._record_path(record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_to_payload(record), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def get(self, decision_id: str) -> PlanningRecord | None:
        path = self._record_path(decision_id)
        if not path.exists():
            return None
        # The id is recomputed from what was read, never trusted from the
        # file: a record that no longer hashes to its name fails loudly.
        record = _from_payload(json.loads(path.read_text(encoding="utf-8")))
        if record.id != decision_id:
            raise PlanningIntegrityError(
                f"decision filed under {decision_id} re-derives id "
                f"{record.id}; refusing to load a record that no longer "
                f"matches its name"
            )
        return record

    def records(self) -> tuple[PlanningRecord, ...]:
        directory = self._root / _DECISIONS_DIRNAME
        if not directory.exists():
            return ()
        loaded = []
        for path in sorted(directory.glob(f"*{_RECORD_SUFFIX}")):
            record = self.get(path.stem)
            assert record is not None  # listed from the directory just above
            loaded.append(record)
        return tuple(loaded)

    # -- dispatch markers --------------------------------------------------------

    def _marker_path(self, decision_id: str) -> Path:
        return (
            self._root / _DISPATCHED_DIRNAME / f"{decision_id}{_RECORD_SUFFIX}"
        )

    def mark_dispatched(self, decision_id: str, note: str) -> None:
        """Record, write-once, that the decision's follow-up action was
        emitted. Marking twice with any content is a conflict: a decision
        is acted on exactly once."""
        if self.get(decision_id) is None:
            raise PlanningConflictError(
                f"cannot dispatch unknown planning decision {decision_id}"
            )
        path = self._marker_path(decision_id)
        if path.exists():
            raise PlanningConflictError(
                f"planning decision {decision_id} is already dispatched"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"decision_id": decision_id, "note": note},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def is_dispatched(self, decision_id: str) -> bool:
        return self._marker_path(decision_id).exists()

    # -- dispatch attempts -------------------------------------------------------

    def _attempts_dir(self, decision_id: str) -> Path:
        return self._root / _ATTEMPTS_DIRNAME / decision_id

    def dispatch_attempts(self, decision_id: str) -> int:
        """How many dispatch attempts are durably recorded for the
        decision. Zero for a decision never attempted — and for an unknown
        one: absence of record is absence of attempts."""
        directory = self._attempts_dir(decision_id)
        if not directory.is_dir():
            return 0
        return sum(
            1 for _ in directory.glob(f"attempt-*{_RECORD_SUFFIX}")
        )

    def record_dispatch_attempt(self, decision_id: str) -> int:
        """Durably count one more dispatch attempt for the decision and
        return the new total. One write-once file per attempt: a resumed
        run continues the count where the previous process left it, so a
        dispatch budget cannot be reset by restarting."""
        if self.get(decision_id) is None:
            raise PlanningConflictError(
                f"cannot record a dispatch attempt for unknown planning "
                f"decision {decision_id}"
            )
        number = self.dispatch_attempts(decision_id) + 1
        path = (
            self._attempts_dir(decision_id)
            / f"attempt-{number:04d}{_RECORD_SUFFIX}"
        )
        if path.exists():
            raise PlanningConflictError(
                f"dispatch attempt {number} for planning decision "
                f"{decision_id} is already recorded"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"attempt": number, "decision_id": decision_id},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return number

    def open_decisions(self) -> tuple[PlanningRecord, ...]:
        """Accepted decisions whose follow-up has not been emitted yet, in
        deterministic (id) order. In the single-planner flow at most one is
        ever open."""
        return tuple(
            record
            for record in self.records()
            if not self.is_dispatched(record.id)
        )

    # -- rejected attempts -------------------------------------------------------

    def preserve_rejected(
        self,
        *,
        invocation_id: str,
        reasons: tuple[tuple[str, str], ...],
        request_fingerprint: str,
        response_id: str,
        payload: object,
        repair: int,
    ) -> Path:
        """Preserve one gate-rejected planning attempt as data: every rule
        that fired with its detail, the provenance of the call, and the raw
        payload. Returns the file written."""
        directory = self._root / _REJECTED_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{occurrence_id('rej')}{_RECORD_SUFFIX}"
        path.write_text(
            json.dumps(
                {
                    "invocation_id": invocation_id,
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
        directory = self._root / _REJECTED_DIRNAME
        if not directory.exists():
            return ()
        return tuple(
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob(f"*{_RECORD_SUFFIX}"))
        )


def _jsonable(value: object) -> object:
    """A JSON-serializable copy of ``value``; anything unrepresentable is
    preserved as its ``repr`` rather than dropped."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _to_payload(record: PlanningRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "invocation_id": record.invocation_id,
        "action": record.action.value,
        "question_id": record.question_id,
        "rationale": record.rationale,
        "evidence_ids": list(record.evidence_ids),
        "hypothesis_id": record.hypothesis_id,
        "prediction_id": record.prediction_id,
        "spec_id": record.spec_id,
        "parent_experiment_id": record.parent_experiment_id,
        "removed_component": record.removed_component,
        "replication_seed": record.replication_seed,
        "template_id": record.template_id,
        "stop_reason": (
            record.stop_reason.value if record.stop_reason is not None else None
        ),
        "repair_count": record.repair_count,
        "request_fingerprint": record.request_fingerprint,
        "response_id": record.response_id,
        "provider": record.provider,
        "requested_model": record.requested_model,
        "served_model": record.served_model,
        "provider_request_id": record.provider_request_id,
        "latency_seconds": record.latency_seconds,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "nominal_cost_usd": record.nominal_cost_usd,
    }


def _from_payload(payload: Mapping[str, object]) -> PlanningRecord:
    evidence_ids = payload["evidence_ids"]
    replication_seed = payload["replication_seed"]
    stop_reason = payload["stop_reason"]
    provider_request_id = payload["provider_request_id"]
    nominal = payload["nominal_cost_usd"]
    assert isinstance(evidence_ids, list)
    return PlanningRecord(
        invocation_id=str(payload["invocation_id"]),
        action=PlanningAction(str(payload["action"])),
        question_id=str(payload["question_id"]),
        rationale=str(payload["rationale"]),
        evidence_ids=tuple(str(item) for item in evidence_ids),
        hypothesis_id=str(payload["hypothesis_id"]),
        prediction_id=str(payload["prediction_id"]),
        spec_id=str(payload["spec_id"]),
        parent_experiment_id=str(payload["parent_experiment_id"]),
        removed_component=str(payload["removed_component"]),
        replication_seed=(
            int(replication_seed)
            if isinstance(replication_seed, int)
            and not isinstance(replication_seed, bool)
            else None
        ),
        template_id=str(payload["template_id"]),
        stop_reason=(
            StopReason(str(stop_reason)) if stop_reason is not None else None
        ),
        repair_count=int(str(payload["repair_count"])),
        request_fingerprint=str(payload["request_fingerprint"]),
        response_id=str(payload["response_id"]),
        provider=str(payload["provider"]),
        requested_model=str(payload["requested_model"]),
        served_model=str(payload["served_model"]),
        provider_request_id=(
            str(provider_request_id)
            if provider_request_id is not None
            else None
        ),
        latency_seconds=float(str(payload["latency_seconds"])),
        input_tokens=int(str(payload["input_tokens"])),
        output_tokens=int(str(payload["output_tokens"])),
        nominal_cost_usd=(
            float(str(nominal)) if nominal is not None else None
        ),
    )
