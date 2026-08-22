"""Write-once storage for manuscripts, reviews, and refused drafts.

The same discipline as every store in this repository: a record's
filename is its content id, an identical re-record is a no-op, different
content under the same name refuses, and loading re-derives the id from
what was read rather than trusting the file. Rejected payloads are
preserved as JSON beside the records — the refusal is evidence too, and
its spend is part of the honest account of what writing cost.

Succession lives in the review store: a :class:`RevisionRecord` names
the review that demanded a revision and the manuscript that answered
it, so :func:`head_for` can say which draft currently stands for a
packet without the manuscript schema ever changing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

from ..core.ids import occurrence_id
from ..core.serialize import to_jsonable
from .manuscript import (
    AuthorCall,
    Manuscript,
    ManuscriptError,
    ProseSections,
)
from .review import (
    ReviewError,
    ReviewFinding,
    ReviewRecord,
    ReviewVerdict,
    RevisionRecord,
)

_RECORD_SUFFIX = ".json"

_R = TypeVar("_R", "ReviewRecord", "RevisionRecord")
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


class ReviewConflictError(ReviewError):
    """A review or revision id is already taken by different content."""


class ReviewIntegrityError(ReviewError):
    """A stored review record no longer matches its own name."""


class AmbiguousHeadError(ReviewError):
    """More than one manuscript stands for a packet, and the records do
    not say which succeeded which."""


class ReviewStore:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # -- reviews -------------------------------------------------------------

    def record_review(self, review: ReviewRecord) -> ReviewRecord:
        return self._record(
            review, review.review_id, self.get_review, "review"
        )

    def get_review(self, review_id: str) -> ReviewRecord | None:
        payload = self._load(review_id)
        if payload is None:
            return None
        try:
            review = _review_from_payload(payload)
        except (ReviewError, ManuscriptError) as error:
            raise ReviewIntegrityError(
                f"review filed under {review_id} does not survive "
                f"loading: {error}"
            ) from error
        if review.review_id != review_id:
            raise ReviewIntegrityError(
                f"review filed under {review_id} re-derives id "
                f"{review.review_id}; refusing to load a record that no "
                f"longer matches its name"
            )
        return review

    def reviews(self) -> tuple[ReviewRecord, ...]:
        return tuple(
            found
            for path in sorted(self._root.glob(f"rvw_*{_RECORD_SUFFIX}"))
            if (found := self.get_review(path.stem)) is not None
        )

    def for_manuscript(self, manuscript_id: str) -> tuple[ReviewRecord, ...]:
        return tuple(
            found
            for found in self.reviews()
            if found.manuscript_id == manuscript_id
        )

    def for_packet(self, packet_id: str) -> tuple[ReviewRecord, ...]:
        return tuple(
            found
            for found in self.reviews()
            if found.packet_id == packet_id
        )

    # -- revisions -----------------------------------------------------------

    def record_revision(self, revision: RevisionRecord) -> RevisionRecord:
        return self._record(
            revision, revision.revision_id, self.get_revision, "revision"
        )

    def get_revision(self, revision_id: str) -> RevisionRecord | None:
        payload = self._load(revision_id)
        if payload is None:
            return None
        try:
            revision = RevisionRecord(
                packet_id=str(payload["packet_id"]),
                review_id=str(payload["review_id"]),
                superseded_manuscript_id=str(
                    payload["superseded_manuscript_id"]
                ),
                revision_manuscript_id=str(
                    payload["revision_manuscript_id"]
                ),
                revision_id=str(payload["revision_id"]),
            )
        except ReviewError as error:
            raise ReviewIntegrityError(
                f"revision filed under {revision_id} does not survive "
                f"loading: {error}"
            ) from error
        if revision.revision_id != revision_id:
            raise ReviewIntegrityError(
                f"revision filed under {revision_id} re-derives id "
                f"{revision.revision_id}; refusing to load a record that "
                f"no longer matches its name"
            )
        return revision

    def revisions_for(self, packet_id: str) -> tuple[RevisionRecord, ...]:
        return tuple(
            found
            for path in sorted(self._root.glob(f"rvn_*{_RECORD_SUFFIX}"))
            if (found := self.get_revision(path.stem)) is not None
            and found.packet_id == packet_id
        )

    # -- rejected ------------------------------------------------------------

    def preserve_rejected(
        self,
        *,
        manuscript_id: str,
        packet_id: str,
        reasons: tuple[tuple[str, str], ...],
        request_fingerprint: str,
        response_id: str,
        payload: object,
        repair: int,
    ) -> Path:
        directory = self._root / _REJECTED_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{occurrence_id('rej')}{_RECORD_SUFFIX}"
        path.write_text(
            json.dumps(
                {
                    "manuscript_id": manuscript_id,
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

    # -- mechanics -----------------------------------------------------------

    def _path(self, record_id: str) -> Path:
        return self._root / f"{record_id}{_RECORD_SUFFIX}"

    def _load(self, record_id: str) -> dict[str, object] | None:
        path = self._path(record_id)
        if not path.exists():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        return loaded

    def _record(
        self,
        record: _R,
        record_id: str,
        get: Callable[[str], _R | None],
        kind: str,
    ) -> _R:
        existing = get(record_id)
        if existing is not None:
            if existing != record:
                raise ReviewConflictError(
                    f"{kind} {record_id} is already recorded with "
                    f"different content; records are never rewritten"
                )
            return existing
        self._path(record_id).write_text(
            json.dumps(to_jsonable(record), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record


def head_for(
    manuscripts: ManuscriptStore,
    reviews: ReviewStore,
    packet_id: str,
) -> tuple[Manuscript, ...]:
    """The manuscripts currently standing for a packet: those no
    revision record supersedes, id-ordered. One is the ordinary state;
    zero means author first; two or more is an interrupted review cycle
    for the review verb to complete."""
    superseded = set()
    for revision in reviews.revisions_for(packet_id):
        if manuscripts.get(revision.revision_manuscript_id) is None:
            raise ReviewIntegrityError(
                f"revision {revision.revision_id} names manuscript "
                f"{revision.revision_manuscript_id}, which the store "
                f"does not hold"
            )
        superseded.add(revision.superseded_manuscript_id)
    return tuple(
        found
        for found in manuscripts.for_packet(packet_id)
        if found.manuscript_id not in superseded
    )


def _review_from_payload(payload: Mapping[str, object]) -> ReviewRecord:
    call = payload["call"]
    findings = payload["findings"]
    assert isinstance(call, Mapping) and isinstance(findings, list)
    return ReviewRecord(
        manuscript_id=str(payload["manuscript_id"]),
        packet_id=str(payload["packet_id"]),
        verdict=ReviewVerdict(str(payload["verdict"])),
        findings=tuple(
            ReviewFinding(
                section=str(entry["section"]),
                quote=str(entry["quote"]),
                issue=str(entry["issue"]),
                subject_id=str(entry["subject_id"]),
                explanation=str(entry["explanation"]),
                origin=str(entry["origin"]),
            )
            for entry in findings
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
        review_id=str(payload["review_id"]),
    )
