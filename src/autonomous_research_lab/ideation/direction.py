"""The CFP/workshop direction ingress: a supplied call text in, an
immutable snapshot and one gated structured direction out.

The snapshot is the source of record: the supplied public text verbatim,
with its URL and supply timestamp as provenance and a content hash, never
touched by a model. The direction is the model's structured reading of
that snapshot — scope, topics, constraints, relevant dates — and every
reported item must appear verbatim in the snapshot (the gate lives in
:mod:`.gates`), so interpretation can never quietly extend the source.
The two are separate records by construction.

There is deliberately no crawler and no HTTP here: a supplied snapshot
with provenance is the whole ingress. CFP material constrains relevance
for candidate generation; it is not scientific evidence and grants no
authority over any record downstream.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from typing import Final

from ..core.ids import content_id
from ..mapping.records import CallProvenance

MAX_SNAPSHOT_CHARS: Final = 200_000
"""A supplied call text larger than this is not a call for papers."""


@dataclass(frozen=True, slots=True)
class CfpSnapshot:
    """One immutable supplied CFP/workshop text with its provenance. The
    hash is computed from the text (or verified against it when
    supplied), so a snapshot whose text and hash disagree cannot be
    constructed."""

    source_url: str
    supplied_at: str
    text: str
    text_sha256: str = field(default="")
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.source_url.strip():
            raise ValueError("a snapshot must name where its text came from")
        try:
            _dt.datetime.fromisoformat(self.supplied_at)
        except ValueError as exc:
            raise ValueError(
                f"supplied_at must be an ISO timestamp: "
                f"{self.supplied_at!r}"
            ) from exc
        if not self.text.strip():
            raise ValueError("a snapshot must carry the supplied text")
        if len(self.text) > MAX_SNAPSHOT_CHARS:
            raise ValueError(
                f"a snapshot holds at most {MAX_SNAPSHOT_CHARS} characters"
            )
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if not self.text_sha256:
            object.__setattr__(self, "text_sha256", digest)
        elif self.text_sha256 != digest:
            raise ValueError(
                "the snapshot's hash does not match its text; a snapshot "
                "whose two records disagree is tampered, not loadable"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "cfp", self.source_url, self.supplied_at, self.text
                ),
            )


@dataclass(frozen=True, slots=True)
class DirectionRecord:
    """The gated structured reading of one snapshot: what the call is
    about, in fields a later stage can check against. ``scope`` is
    extractor synthesis; topics, constraints, and dates are the call's
    own words, verbatim by gate."""

    run_id: str
    snapshot_id: str
    scope: str
    topics: tuple[str, ...]
    constraints: tuple[str, ...]
    relevant_dates: tuple[str, ...]
    provenance: CallProvenance
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.scope.strip():
            raise ValueError("a direction requires a scope")
        if not self.topics:
            raise ValueError("a direction requires at least one topic")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "dir",
                    self.run_id,
                    self.snapshot_id,
                    self.scope,
                    self.topics,
                    self.constraints,
                    self.relevant_dates,
                    self.provenance.response_id,
                ),
            )

    def rendered_text(self) -> str:
        """Everything the direction reports, as one haystack for the
        candidate gate's CFP-alignment checks."""
        return "\n".join(
            (
                self.scope,
                *self.topics,
                *self.constraints,
                *self.relevant_dates,
            )
        )
