"""One investigation, and the durable record of what it was asked to do.

Layout, under a control root::

    <root>/
    ├── configs/<cfg_…>.json           the run config, verbatim
    ├── investigations/<invr_…>.json   one record per ``arl run``
    └── logs/<inv_…>/…                 the stage events (see events.py)

An :class:`Investigation` is an *event*: one attempt to carry one brief
to a terminal state. Its ``investigation_id`` is therefore an occurrence
id, because pursuing the same brief twice is two attempts and no content
could tell them apart — the same split :class:`~..program.records.ResearchRun`
makes between its ``run_id`` and its content-addressed record.

The config is stored verbatim and addressed by its own content, and the
investigation names it. Two consequences, both wanted: the parameters of
a run are provenance rather than a file somebody might edit afterwards,
and two investigations started from the same config visibly share one
config record instead of two copies that might differ by a space.

The record holds no status. What happened is the event log's business,
and a status field here would be a second answer to a question that
already has one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..core.ids import content_id
from .events import StageLog
from .stage import StageName

_RECORD_SUFFIX: Final = ".json"
_CONFIGS: Final = "configs"
_INVESTIGATIONS: Final = "investigations"

MAX_LABEL_CHARS: Final = 120
"""A label names the investigation for a human reading ``arl status``.
It is a handle, not a description of the research."""


class InvestigationConflictError(RuntimeError):
    """A write-once control record would be overwritten with different
    content."""


class InvestigationIntegrityError(RuntimeError):
    """A stored control record no longer matches its own identity."""


@dataclass(frozen=True, slots=True)
class Investigation:
    """One run of the chain, from a config to a terminal state."""

    investigation_id: str
    config_id: str
    label: str
    stop_after: str = ""
    """The scope the config declared: the stage this investigation is
    not meant to go past, or empty for the whole chain. Recorded, and so
    binding on every walk — an investigation meant to reach a funded run
    and stop is not talked into experimenting by being resumed. The
    ``--stop-after`` flag is a different thing: a brake on one walk,
    deliberately not recorded."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        for name, value in (
            ("investigation_id", self.investigation_id),
            ("config_id", self.config_id),
            ("label", self.label),
        ):
            if not value.strip():
                raise ValueError(f"an investigation must name its {name}")
        if len(self.label) > MAX_LABEL_CHARS:
            raise ValueError(
                f"label must be at most {MAX_LABEL_CHARS} characters, "
                f"got {len(self.label)}"
            )
        if self.stop_after and self.stop_after not in set(StageName):
            raise ValueError(
                f"stop_after names no stage of the chain: {self.stop_after!r}"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "invr",
                    self.investigation_id,
                    self.config_id,
                    self.label,
                    self.stop_after,
                ),
            )


class InvestigationStore:
    """File-backed, write-once storage for investigations and the configs
    they were started from, under one injected control root."""

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
                raise InvestigationConflictError(
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

    # -- configs ---------------------------------------------------------------

    def record_config(self, payload: Mapping[str, object]) -> str:
        """Store one config verbatim and return its content id.

        Verbatim on purpose: the controller reads the config through a
        typed codec, but what it *stores* is the operator's own document,
        so a later reader can see what was asked for rather than what the
        codec understood of it.
        """
        config_id = content_id("cfg", payload)
        self._write_once(_CONFIGS, config_id, payload)
        return config_id

    def get_config(self, config_id: str) -> Mapping[str, object] | None:
        payload = self._load(_CONFIGS, config_id)
        if payload is None:
            return None
        rederived = content_id("cfg", payload)
        if rederived != config_id:
            raise InvestigationIntegrityError(
                f"config filed under {config_id} re-derives {rederived}; "
                f"the file was edited after the run was started"
            )
        return payload

    # -- investigations --------------------------------------------------------

    def record(self, investigation: Investigation) -> Investigation:
        existing = self.get(investigation.investigation_id)
        if existing is not None and existing != investigation:
            raise InvestigationConflictError(
                f"investigation {investigation.investigation_id} is already "
                f"recorded with different content; an investigation is "
                f"stated once"
            )
        self._write_once(
            _INVESTIGATIONS, investigation.id, _payload(investigation)
        )
        return investigation

    def get(self, investigation_id: str) -> Investigation | None:
        return next(
            (
                found
                for found in self.investigations()
                if found.investigation_id == investigation_id
            ),
            None,
        )

    def investigations(self) -> tuple[Investigation, ...]:
        found: list[Investigation] = []
        for record_id in self._ids(_INVESTIGATIONS):
            payload = self._load(_INVESTIGATIONS, record_id)
            assert payload is not None
            investigation = _investigation_from(payload)
            if investigation.id != record_id:
                raise InvestigationIntegrityError(
                    f"investigation filed under {record_id} re-derives "
                    f"{investigation.id}; refusing to load a record that no "
                    f"longer matches its name"
                )
            found.append(investigation)
        return tuple(found)

    def log_for(self, investigation_id: str) -> StageLog:
        return StageLog(self._root, investigation_id)


def _payload(investigation: Investigation) -> dict[str, object]:
    return {
        "id": investigation.id,
        "investigation_id": investigation.investigation_id,
        "config_id": investigation.config_id,
        "label": investigation.label,
        "stop_after": investigation.stop_after,
    }


def _investigation_from(payload: Mapping[str, object]) -> Investigation:
    return Investigation(
        investigation_id=_text(payload, "investigation_id"),
        config_id=_text(payload, "config_id"),
        label=_text(payload, "label"),
        stop_after=_text(payload, "stop_after"),
    )


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise InvestigationIntegrityError(f"{key} must be a string")
    return value
