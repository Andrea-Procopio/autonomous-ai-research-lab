"""The durable evidence store: facts on disk, verified by digest.

Layout, under a run root::

    <root>/
    ├── results/<result_id>.json     one execution record
    ├── evidence/<evidence_id>.json  one factual reading
    ├── artifacts/…                  the manifests (the artifact store's)
    └── blobs/…                      the bytes (the artifact store's)

Same invariant as the in-memory store, and the same errors: an id never
maps to different content, re-recording identical content is a no-op,
and evidence naming an unrecorded result is refused.

Ordering is the point
---------------------

``record_result`` stores the bytes, then the manifest, then the fact::

    artifacts.ingest(result)      blobs, then the manifest record
    write results/<id>.json       the fact itself
    -> the caller persists the state that references it

A state can therefore only ever reference a result whose outputs are
already durable. The store refuses the whole record if any file is
refused, so a half-stored result does not exist.

Why a payload digest
--------------------

Every other file store here detects tampering by recomputing the
record's content id on load. That would prove nothing about these two:

* ``ExperimentResult.id`` derives from its job id alone — a result is an
  *event*, and two identical runs are two results — so edited metrics
  still re-derive the same id;
* ``Evidence.id`` covers its result, kind, and observation, but not its
  ``metrics`` or ``spec_id`` — one result read two ways is two readings.

Both are right as domain identity and useless as integrity checks. So
each record carries ``payload_digest``: a sha256 over its own canonical
JSON, recomputed on load. An edited file fails loudly instead of
returning a fact nobody wrote.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ..core.budget import ResourceCost
from ..core.evidence import Evidence, EvidenceKind
from ..core.experiment import Environment, ExperimentResult, ExperimentStatus
from ..core.types import ConfigValue
from .artifacts import ArtifactStore, FileArtifactStore
from .store import EvidenceConflictError, UnknownRecordError

_RESULTS: Final = "results"
_EVIDENCE: Final = "evidence"
_RECORD_SUFFIX: Final = ".json"
_DIGEST_KEY: Final = "payload_digest"


class EvidenceIntegrityError(RuntimeError):
    """A stored record no longer matches its own payload digest. The
    domain id cannot catch this — see the module docstring — so the
    store carries its own."""


class FileEvidenceStore:
    """File-backed results and evidence, with artifact storage injected."""

    def __init__(
        self, root: Path | str, *, artifacts: ArtifactStore | None = None
    ) -> None:
        self._root = Path(root)
        self._results = self._root / _RESULTS
        self._evidence_dir = self._root / _EVIDENCE
        self._results.mkdir(parents=True, exist_ok=True)
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts = (
            artifacts if artifacts is not None else FileArtifactStore(self._root)
        )
        # Records are write-once, so a read cache can never go stale.
        self._result_cache: dict[str, ExperimentResult] = {}
        self._evidence_cache: dict[str, Evidence] = {}

    @property
    def root(self) -> Path:
        return self._root

    @property
    def artifacts(self) -> ArtifactStore:
        return self._artifacts

    # -- writing ---------------------------------------------------------------

    def record_result(self, result: ExperimentResult) -> ExperimentResult:
        existing = self._read_result(result.id)
        if existing is not None:
            if existing != result:
                raise EvidenceConflictError(
                    f"result {result.id} already recorded with different "
                    f"content"
                )
            # The artifacts may still be missing if an earlier attempt was
            # interrupted between the blobs and the record.
            self._artifacts.ingest(result)
            return existing
        # Bytes first: a recorded fact whose outputs are not yet durable
        # would be exactly the gap this store exists to close.
        self._artifacts.ingest(result)
        self._write(self._result_path(result.id), _result_payload(result))
        self._result_cache[result.id] = result
        return result

    def record_evidence(self, evidence: Evidence) -> Evidence:
        existing = self._read_evidence(evidence.id)
        if existing is not None:
            if existing != evidence:
                raise EvidenceConflictError(
                    f"evidence {evidence.id} already recorded with different "
                    f"content"
                )
            return existing
        if self._read_result(evidence.result_id) is None:
            raise UnknownRecordError(
                f"evidence {evidence.id} references unrecorded result "
                f"{evidence.result_id}"
            )
        self._write(
            self._evidence_path(evidence.id), _evidence_payload(evidence)
        )
        self._evidence_cache[evidence.id] = evidence
        return evidence

    # -- reading ---------------------------------------------------------------

    def get_result(self, result_id: str) -> ExperimentResult:
        result = self._read_result(result_id)
        if result is None:
            raise UnknownRecordError(result_id)
        return result

    def get_evidence(self, evidence_id: str) -> Evidence:
        evidence = self._read_evidence(evidence_id)
        if evidence is None:
            raise UnknownRecordError(evidence_id)
        return evidence

    def results(self) -> tuple[ExperimentResult, ...]:
        return tuple(
            self.get_result(path.stem)
            for path in sorted(self._results.glob(f"*{_RECORD_SUFFIX}"))
        )

    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(
            self.get_evidence(path.stem)
            for path in sorted(self._evidence_dir.glob(f"*{_RECORD_SUFFIX}"))
        )

    def result_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(path.stem for path in self._results.glob(f"*{_RECORD_SUFFIX}"))
        )

    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                path.stem
                for path in self._evidence_dir.glob(f"*{_RECORD_SUFFIX}")
            )
        )

    # -- files -----------------------------------------------------------------

    def _result_path(self, result_id: str) -> Path:
        return self._results / f"{result_id}{_RECORD_SUFFIX}"

    def _evidence_path(self, evidence_id: str) -> Path:
        return self._evidence_dir / f"{evidence_id}{_RECORD_SUFFIX}"

    def _write(self, path: Path, payload: dict[str, object]) -> None:
        payload[_DIGEST_KEY] = digest_of(payload)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _read_result(self, result_id: str) -> ExperimentResult | None:
        cached = self._result_cache.get(result_id)
        if cached is not None:
            return cached
        payload = self._load(self._result_path(result_id))
        if payload is None:
            return None
        result = _result_from(payload)
        if result.id != result_id:
            raise EvidenceIntegrityError(
                f"result filed under {result_id} reconstructs as {result.id}"
            )
        self._result_cache[result_id] = result
        return result

    def _read_evidence(self, evidence_id: str) -> Evidence | None:
        cached = self._evidence_cache.get(evidence_id)
        if cached is not None:
            return cached
        payload = self._load(self._evidence_path(evidence_id))
        if payload is None:
            return None
        evidence = _evidence_from(payload)
        if evidence.id != evidence_id:
            raise EvidenceIntegrityError(
                f"evidence filed under {evidence_id} reconstructs as "
                f"{evidence.id}"
            )
        self._evidence_cache[evidence_id] = evidence
        return evidence

    def _load(self, path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceIntegrityError(
                f"{path.name} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise EvidenceIntegrityError(f"{path.name} is not an object")
        stored = payload.pop(_DIGEST_KEY, None)
        recomputed = digest_of(payload)
        if stored != recomputed:
            raise EvidenceIntegrityError(
                f"{path.name} carries digest {stored!r} but its contents "
                f"hash to {recomputed}; the file was edited"
            )
        return payload


def digest_of(payload: Mapping[str, object]) -> str:
    """The canonical digest of one record's payload, excluding the digest
    field itself."""
    body = {key: value for key, value in payload.items() if key != _DIGEST_KEY}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# -- payloads ------------------------------------------------------------------


def _result_payload(result: ExperimentResult) -> dict[str, object]:
    return {
        "id": result.id,
        "spec_id": result.spec_id,
        "job_id": result.job_id,
        "status": str(result.status),
        "command": list(result.command),
        "environment": {
            "python_version": result.environment.python_version,
            "platform": result.environment.platform,
            "git_commit": result.environment.git_commit,
            "git_dirty": result.environment.git_dirty,
        },
        "metrics": dict(result.metrics),
        "config": dict(result.config),
        "seed": result.seed,
        "artifacts": list(result.artifacts),
        "logs": list(result.logs),
        "runtime_seconds": result.runtime_seconds,
        "cost": {
            "wall_clock_seconds": result.cost.wall_clock_seconds,
            "gpu_hours": result.cost.gpu_hours,
            "usd": result.cost.usd,
            "model_tokens": result.cost.model_tokens,
        },
        "exit_code": result.exit_code,
        "failure_reason": result.failure_reason,
    }


def _result_from(payload: dict[str, object]) -> ExperimentResult:
    environment = _object(payload, "environment")
    cost = _object(payload, "cost")
    return ExperimentResult(
        spec_id=_text(payload, "spec_id"),
        job_id=_text(payload, "job_id"),
        status=ExperimentStatus(_text(payload, "status")),
        command=_strings(payload, "command"),
        environment=Environment(
            python_version=_text(environment, "python_version"),
            platform=_text(environment, "platform"),
            git_commit=_optional_text(environment, "git_commit"),
            git_dirty=_optional_bool(environment, "git_dirty"),
        ),
        metrics=_floats(payload, "metrics"),
        config=_config(payload, "config"),
        seed=_optional_int(payload, "seed"),
        artifacts=_strings(payload, "artifacts"),
        logs=_strings(payload, "logs"),
        runtime_seconds=_number(payload, "runtime_seconds"),
        cost=ResourceCost(
            wall_clock_seconds=_number(cost, "wall_clock_seconds"),
            gpu_hours=_number(cost, "gpu_hours"),
            usd=_number(cost, "usd"),
            model_tokens=int(_number(cost, "model_tokens")),
        ),
        exit_code=_optional_int(payload, "exit_code"),
        failure_reason=_optional_text(payload, "failure_reason"),
    )


def _evidence_payload(evidence: Evidence) -> dict[str, object]:
    return {
        "id": evidence.id,
        "result_id": evidence.result_id,
        "spec_id": evidence.spec_id,
        "kind": str(evidence.kind),
        "observation": evidence.observation,
        "metrics": dict(evidence.metrics),
    }


def _evidence_from(payload: dict[str, object]) -> Evidence:
    return Evidence(
        result_id=_text(payload, "result_id"),
        spec_id=_text(payload, "spec_id"),
        kind=EvidenceKind(_text(payload, "kind")),
        observation=_text(payload, "observation"),
        metrics=_floats(payload, "metrics"),
    )


# -- field readers -------------------------------------------------------------
# Boundary code, deliberately explicit: every field is checked as it is
# read, so codec drift surfaces as an error rather than a wrong fact.


def _object(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise EvidenceIntegrityError(f"{key} must be an object")
    return value


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise EvidenceIntegrityError(f"{key} must be a string")
    return value


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvidenceIntegrityError(f"{key} must be a string or null")
    return value


def _optional_bool(payload: Mapping[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise EvidenceIntegrityError(f"{key} must be a boolean or null")
    return value


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvidenceIntegrityError(f"{key} must be an integer or null")
    return value


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceIntegrityError(f"{key} must be a number")
    return float(value)


def _strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise EvidenceIntegrityError(f"{key} must be a list of strings")
    return tuple(str(item) for item in value)


def _floats(payload: Mapping[str, object], key: str) -> dict[str, float]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise EvidenceIntegrityError(f"{key} must be an object of numbers")
    read: dict[str, float] = {}
    for name, entry in value.items():
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise EvidenceIntegrityError(f"{key}.{name} must be a number")
        read[str(name)] = float(entry)
    return read


def _config(payload: Mapping[str, object], key: str) -> dict[str, ConfigValue]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise EvidenceIntegrityError(f"{key} must be an object")
    read: dict[str, ConfigValue] = {}
    for name, entry in value.items():
        if entry is not None and not isinstance(entry, (str, int, float, bool)):
            raise EvidenceIntegrityError(
                f"{key}.{name} must be a string, number, boolean, or null"
            )
        read[str(name)] = entry
    return read
