"""Durable storage for research runs, mirroring the admission store.

One directory per record kind — directives, authorizations, and run
envelopes — plus ``ledgers/`` for the append-only budget entries
(:class:`~.ledger.BudgetLedger` owns those) and ``states/`` for the state
snapshots, managed by the house
:class:`~..persistence.state_store.FileStateStore` derived from the same
root. The state store is deliberately not injectable, for the same
reason it is not injectable in admission: the promise that a funded
state is never exposed without its run envelope depends on the record
and the snapshot living under one root, loaded through one accessor.

Writes are write-once and verify-on-repeat: identical re-recording is a
no-op, different content under the same id raises. Ids are recomputed
from what was read, never trusted from the file. Two internal rules go
beyond plain write-once: at most one run envelope per run directive —
which is what makes re-running a completed directive a replay — and at
most one per run id.

Unlike admission, an admission record may back *several* runs. Each
needs its own directive, so each is a stated act rather than a repeated
command.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ..core.budget import ResearchBudget
from ..core.state import ResearchState
from ..persistence import FileStateStore
from .authorization import FundingAuthorization
from .directive import RunDirective
from .ledger import BudgetLedger
from .records import ResearchRun

_RECORD_SUFFIX: Final = ".json"

_DIRECTIVES: Final = "directives"
_AUTHORIZATIONS: Final = "authorizations"
_RUNS: Final = "runs"


class ProgramConflictError(RuntimeError):
    """A write-once run artifact would be overwritten with different
    content, or a directive would start two runs."""


class ProgramIntegrityError(RuntimeError):
    """A stored run record no longer matches its own identity, or its
    snapshot is missing."""


class ProgramStore:
    """File-backed, write-once storage for one or more runs under one
    injected root."""

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
                raise ProgramConflictError(
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
            raise ProgramIntegrityError(
                f"{kind} record filed under {filed_as} re-derives id "
                f"{rederived}; refusing to load a record that no longer "
                f"matches its name"
            )

    # -- directives ------------------------------------------------------------

    def record_directive(self, directive: RunDirective) -> RunDirective:
        self._write_once(
            _DIRECTIVES, directive.id, _directive_payload(directive)
        )
        return directive

    def get_directive(self, directive_id: str) -> RunDirective | None:
        payload = self._load(_DIRECTIVES, directive_id)
        if payload is None:
            return None
        directive = _directive_from(payload)
        self._verify(_DIRECTIVES, directive_id, directive.id)
        return directive

    # -- authorizations --------------------------------------------------------

    def record_authorization(
        self, authorization: FundingAuthorization
    ) -> FundingAuthorization:
        self._write_once(
            _AUTHORIZATIONS,
            authorization.id,
            _authorization_payload(authorization),
        )
        return authorization

    def get_authorization(
        self, authorization_id: str
    ) -> FundingAuthorization | None:
        payload = self._load(_AUTHORIZATIONS, authorization_id)
        if payload is None:
            return None
        authorization = _authorization_from(payload)
        self._verify(_AUTHORIZATIONS, authorization_id, authorization.id)
        return authorization

    # -- run envelopes ---------------------------------------------------------

    def record_run(self, run: ResearchRun) -> ResearchRun:
        """Write the envelope last. Two runs may share an admission, but
        never a directive and never a run id."""
        for existing in self.runs():
            if existing.id == run.id:
                continue
            if existing.directive_id == run.directive_id:
                raise ProgramConflictError(
                    f"directive {run.directive_id} already started run "
                    f"{existing.run_id}; re-running it replays that run"
                )
            if existing.run_id == run.run_id:
                raise ProgramConflictError(
                    f"run {run.run_id} is already recorded as "
                    f"{existing.id}; a run is enveloped once"
                )
        self._write_once(_RUNS, run.id, _run_payload(run))
        return run

    def get_run(self, record_id: str) -> ResearchRun | None:
        payload = self._load(_RUNS, record_id)
        if payload is None:
            return None
        run = _run_from(payload)
        self._verify(_RUNS, record_id, run.id)
        return run

    def runs(self) -> tuple[ResearchRun, ...]:
        loaded = []
        for record_id in self._ids(_RUNS):
            run = self.get_run(record_id)
            assert run is not None
            loaded.append(run)
        return tuple(loaded)

    def run_for_directive(self, directive_id: str) -> ResearchRun | None:
        """The replay lookup: a completed directive names its run."""
        return next(
            (r for r in self.runs() if r.directive_id == directive_id), None
        )

    def runs_for_admission(self, admission_record_id: str) -> tuple[
        ResearchRun, ...
    ]:
        return tuple(
            r for r in self.runs() if r.admission_record_id == admission_record_id
        )

    # -- states and ledgers ----------------------------------------------------

    def persist_state(self, state: ResearchState) -> Path:
        """Persist the snapshot, then read it back and require equality
        before anything may reference it. The read-back is load-bearing
        here for the same reason it is in admission, and one reason more:
        a state's content id excludes its budget, and the budget is
        precisely what this package writes."""
        path = self._states.persist(state)
        if self._states.load(state.id) != state:
            raise ProgramIntegrityError(
                f"snapshot {state.id} did not read back as the state that "
                f"was written; refusing to reference it"
            )
        return path

    def get_funded_state(
        self, record_id: str
    ) -> tuple[ResearchRun, ResearchState]:
        """The one accessor: the envelope first, the state through it. A
        funded state is never exposed without the run that authorized it,
        and a run whose snapshot is missing or tampered fails loudly."""
        run = self.get_run(record_id)
        if run is None:
            raise ProgramIntegrityError(
                f"no run record {record_id}; a funded state is never "
                f"exposed without its run"
            )
        state = self._states.load(run.funded_state_id)
        if state.budget != run.granted:
            # The state's content id excludes the budget, so a doctored
            # budget reloads silently. The envelope records what was
            # granted, which makes the check exact at the moment of
            # funding — every later balance lives on the ledger.
            raise ProgramIntegrityError(
                f"funded state {run.funded_state_id} holds "
                f"{state.budget}, but run {run.run_id} granted "
                f"{run.granted}"
            )
        return run, state

    def state_store(self) -> FileStateStore:
        """The run's snapshot store, for the runtime to persist into: the
        whole lineage from the admitted seed onward lives under one root."""
        return self._states

    def ledger_for(self, run_id: str) -> BudgetLedger:
        return BudgetLedger(self._root, run_id)


# -- payloads ------------------------------------------------------------------


def _directive_payload(directive: RunDirective) -> dict[str, object]:
    return {
        "id": directive.id,
        "admission_record_id": directive.admission_record_id,
        "authorization_id": directive.authorization_id,
        "label": directive.label,
    }


def _directive_from(payload: Mapping[str, object]) -> RunDirective:
    return RunDirective(
        admission_record_id=_text(payload, "admission_record_id"),
        authorization_id=_text(payload, "authorization_id"),
        label=_text(payload, "label"),
    )


def _authorization_payload(
    authorization: FundingAuthorization,
) -> dict[str, object]:
    return {
        "id": authorization.id,
        "admission_record_id": authorization.admission_record_id,
        "granted": _budget_payload(authorization.granted),
        "authority": authorization.authority,
    }


def _authorization_from(payload: Mapping[str, object]) -> FundingAuthorization:
    return FundingAuthorization(
        admission_record_id=_text(payload, "admission_record_id"),
        granted=_budget_from(payload, "granted"),
        authority=_text(payload, "authority"),
    )


def _run_payload(run: ResearchRun) -> dict[str, object]:
    return {
        "id": run.id,
        "run_id": run.run_id,
        "directive_id": run.directive_id,
        "authorization_id": run.authorization_id,
        "admission_record_id": run.admission_record_id,
        "admitted_state_id": run.admitted_state_id,
        "funded_state_id": run.funded_state_id,
        "granted": _budget_payload(run.granted),
        "grant_entry_id": run.grant_entry_id,
        "label": run.label,
        "authority": run.authority,
        "question_id": run.question_id,
        "hypothesis_id": run.hypothesis_id,
        "prediction_ids": list(run.prediction_ids),
    }


def _run_from(payload: Mapping[str, object]) -> ResearchRun:
    return ResearchRun(
        run_id=_text(payload, "run_id"),
        directive_id=_text(payload, "directive_id"),
        authorization_id=_text(payload, "authorization_id"),
        admission_record_id=_text(payload, "admission_record_id"),
        admitted_state_id=_text(payload, "admitted_state_id"),
        funded_state_id=_text(payload, "funded_state_id"),
        granted=_budget_from(payload, "granted"),
        grant_entry_id=_text(payload, "grant_entry_id"),
        label=_text(payload, "label"),
        authority=_text(payload, "authority"),
        question_id=_text(payload, "question_id"),
        hypothesis_id=_text(payload, "hypothesis_id"),
        prediction_ids=_strings(payload, "prediction_ids"),
    )


def _budget_payload(budget: ResearchBudget) -> dict[str, float | int]:
    return {
        "wall_clock_seconds": budget.wall_clock_seconds,
        "gpu_hours": budget.gpu_hours,
        "usd": budget.usd,
        "model_tokens": budget.model_tokens,
    }


def _budget_from(payload: Mapping[str, object], key: str) -> ResearchBudget:
    raw = payload[key]
    if not isinstance(raw, Mapping):
        raise ProgramIntegrityError(f"{key} must be an object of dimensions")
    return ResearchBudget(
        wall_clock_seconds=_number(raw, "wall_clock_seconds"),
        gpu_hours=_number(raw, "gpu_hours"),
        usd=_number(raw, "usd"),
        model_tokens=int(_number(raw, "model_tokens")),
    )


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ProgramIntegrityError(f"{key} must be a string")
    return value


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramIntegrityError(f"{key} must be a number")
    return float(value)


def _strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ProgramIntegrityError(f"{key} must be a list of strings")
    return tuple(str(item) for item in value)
