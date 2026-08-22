"""Write-once storage for manuscripts and refused drafts.

The same discipline as every store in this repository: a record's
filename is its content id, an identical re-record is a no-op, different
content under the same name refuses, and loading re-derives the id from
what was read rather than trusting the file. Rejected drafts are
preserved as JSON beside the records — the refusal is evidence too, and
its spend is part of the honest account of what writing cost.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ..core.ids import occurrence_id
from ..core.serialize import to_jsonable
from .manuscript import (
    AuthorCall,
    Manuscript,
    ManuscriptError,
    ProseSections,
)

_RECORD_SUFFIX = ".json"
_REJECTED_DIRNAME = "rejected"


class ManuscriptConflictError(ManuscriptError):
    """A manuscript id is already taken by different content."""


class ManuscriptIntegrityError(ManuscriptError):
    """A stored manuscript no longer matches its own name."""


class ManuscriptStore:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _record_path(self, manuscript_id: str) -> Path:
        return self._root / f"{manuscript_id}{_RECORD_SUFFIX}"

    def record(self, manuscript: Manuscript) -> Manuscript:
        """Store one manuscript, write-once."""
        existing = self.get(manuscript.manuscript_id)
        if existing is not None:
            if existing != manuscript:
                raise ManuscriptConflictError(
                    f"manuscript {manuscript.manuscript_id} is already "
                    f"recorded with different content; records are never "
                    f"rewritten"
                )
            return existing
        path = self._record_path(manuscript.manuscript_id)
        path.write_text(
            json.dumps(to_jsonable(manuscript), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manuscript

    def get(self, manuscript_id: str) -> Manuscript | None:
        path = self._record_path(manuscript_id)
        if not path.exists():
            return None
        # The id is recomputed from what was read, never trusted from the
        # file: Manuscript.__post_init__ re-derives it, so a doctored
        # record raises there; a record filed under the wrong name is
        # caught here.
        try:
            manuscript = _from_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except ManuscriptError as error:
            raise ManuscriptIntegrityError(
                f"manuscript filed under {manuscript_id} does not survive "
                f"loading: {error}"
            ) from error
        if manuscript.manuscript_id != manuscript_id:
            raise ManuscriptIntegrityError(
                f"manuscript filed under {manuscript_id} re-derives id "
                f"{manuscript.manuscript_id}; refusing to load a record "
                f"that no longer matches its name"
            )
        return manuscript

    def records(self) -> tuple[Manuscript, ...]:
        return tuple(
            found
            for path in sorted(self._root.glob(f"mscr_*{_RECORD_SUFFIX}"))
            if (found := self.get(path.stem)) is not None
        )

    def for_packet(self, packet_id: str) -> tuple[Manuscript, ...]:
        """Every manuscript authored from one packet, id-ordered. The
        replay lookup: content identity cannot provide idempotence
        because the response id is an occurrence, so the composition
        root asks this instead of calling the model again."""
        return tuple(
            found
            for found in self.records()
            if found.packet_id == packet_id
        )

    def preserve_rejected(
        self,
        *,
        packet_id: str,
        reasons: tuple[tuple[str, str], ...],
        request_fingerprint: str,
        response_id: str,
        payload: object,
        repair: int,
    ) -> Path:
        """Preserve one refused draft as data, never as a document."""
        directory = self._root / _REJECTED_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{occurrence_id('rej')}{_RECORD_SUFFIX}"
        path.write_text(
            json.dumps(
                {
                    "packet_id": packet_id,
                    "reasons": [list(reason) for reason in reasons],
                    "request_fingerprint": request_fingerprint,
                    "response_id": response_id,
                    "payload": to_jsonable(payload),
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


def _from_payload(payload: Mapping[str, object]) -> Manuscript:
    sections = payload["sections"]
    call = payload["call"]
    assert isinstance(sections, Mapping) and isinstance(call, Mapping)
    return Manuscript(
        packet_id=str(payload["packet_id"]),
        sections=ProseSections(
            **{key: str(value) for key, value in sections.items()}
        ),
        call=AuthorCall(
            request_fingerprint=str(call["request_fingerprint"]),
            response_id=str(call["response_id"]),
            provider=str(call["provider"]),
            requested_model=str(call["requested_model"]),
            served_model=str(call["served_model"]),
            provider_request_id=(
                str(call["provider_request_id"])
                if call["provider_request_id"] is not None
                else None
            ),
            latency_seconds=float(call["latency_seconds"]),
            input_tokens=int(call["input_tokens"]),
            output_tokens=int(call["output_tokens"]),
            repair_count=int(call["repair_count"]),
        ),
        manuscript_id=str(payload["manuscript_id"]),
    )
