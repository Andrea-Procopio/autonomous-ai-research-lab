"""The faithfulness review: criticism that must ground itself.

The reviewer answers one question: does the manuscript's prose claim
anything the evidence packet does not record? Not style, not impression,
not acceptance-worthiness — a venue simulator is a different instrument.
Two kinds of finding exist, and both are held to evidence:

* **deterministic findings** — trusted code's own reading: a
  forbidden-strength phrase (``statistically significant``, ``proves``,
  ``novel`` …) that no verdict in this system ever licenses, or a
  verdict word appearing in prose when no claim's assessment records
  that verdict;
* **model findings** — a gated call returns structured findings, and
  each must survive the same discipline admission applies to support
  quotes: the quote must appear verbatim (case- and whitespace-folded)
  in the named section, and the cited record id must be one the packet
  prints. A review that cannot ground its criticism is not a review,
  and an ungrounded finding earns a corrective call, not a hearing.

The verdict is derived by trusted code — REVISE iff any finding stands
— and the model's schema has no verdict property, so the one judgment
that matters is never the model's to output. Succession lives in its
own record: a revision is a new manuscript named by a
:class:`RevisionRecord` pointing at the review that demanded it, the
way an assessment supersedes an assessment, so the manuscript's own
schema and identity never change.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from ..core.assessment import AssessmentVerdict
from ..core.ids import content_id
from ..core.serialize import to_jsonable
from ..runtime.providers import OutputSchema
from .manuscript import (
    PROSE_SECTIONS,
    AuthorCall,
    Manuscript,
    ManuscriptError,
    ProseRejection,
    ProseSections,
)
from .packet import EvidencePacket, render_markdown, to_json

MODEL_ISSUES: Final = frozenset(
    {
        "overclaim",
        "unsupported_statement",
        "misattributed_evidence",
        "scope_creep",
        "contradicts_record",
    }
)

DETERMINISTIC_ISSUES: Final = frozenset(
    {"forbidden_strength", "unlicensed_verdict"}
)

ISSUES: Final = MODEL_ISSUES | DETERMINISTIC_ISSUES

#: Strength no verdict in this system ever licenses. Word-bounded, so
#: ``improves`` and ``disprove`` cannot fire.
FORBIDDEN_STRENGTH: Final = re.compile(
    r"\b(?:statistically\s+significant|statistical\s+significance"
    r"|prove[sdn]?|novel(?:ty)?|state[- ]of[- ]the[- ]art|sota"
    r"|breakthrough|groundbreaking|guarantee[sd]?|unprecedented)\b",
    re.IGNORECASE,
)

_IDENTIFIER: Final = re.compile(r"\b[a-z]+_[0-9a-f]{16}\b")

ORIGIN_DETERMINISTIC: Final = "deterministic"
ORIGIN_MODEL: Final = "model"

REVIEWER_INSTRUCTION: Final = (
    "You are the reviewing seat of an autonomous research lab — a "
    "faithfulness reviewer, not an impression scorer. The user message "
    "shows the run's evidence packet — the complete verified record — "
    "followed by the draft's five model-authored prose sections. Judge "
    "one question only: does the prose claim anything the packet does "
    "not record?\n"
    "Each finding must name the section, quote the offending prose "
    "verbatim exactly as written, name the issue, carry the id of the "
    "packet record the prose misstates (or the empty string when no "
    "single record applies), and explain what the packet actually "
    "records. A quote that does not appear in the named section, or an "
    "id the packet does not print, is rejected — a review that cannot "
    "ground its criticism is not a review.\n"
    "An empty findings list is the correct answer for a faithful "
    "draft. Never invent an issue to seem rigorous; never object to "
    "wording, style, or brevity; report an absence only when it makes "
    "a present sentence false."
)


class ReviewError(RuntimeError):
    """The review cannot honestly be produced."""


class NothingToReviewError(ReviewError):
    """No manuscript exists for this packet: author first."""


class ReviewRejectedError(ReviewError):
    """The model's review failed to ground itself after every
    corrective call."""


class ReviewVerdict(StrEnum):
    """A review's outcome — deliberately not an epistemic verdict: a
    review judges a document's faithfulness, never a claim's truth."""

    APPROVED = "approved"
    REVISE = "revise"


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    section: str
    quote: str
    issue: str
    subject_id: str
    explanation: str
    origin: str

    def __post_init__(self) -> None:
        if self.issue not in ISSUES:
            raise ReviewError(f"unknown review issue {self.issue!r}")
        if self.origin not in (ORIGIN_DETERMINISTIC, ORIGIN_MODEL):
            raise ReviewError(f"unknown finding origin {self.origin!r}")


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    manuscript_id: str
    packet_id: str
    verdict: ReviewVerdict
    findings: tuple[ReviewFinding, ...]
    call: AuthorCall
    review_id: str = field(default="")

    def __post_init__(self) -> None:
        derived = content_id("rvw", _identity_payload(self, "review_id"))
        if not self.review_id:
            object.__setattr__(self, "review_id", derived)
        elif self.review_id != derived:
            raise ReviewError(
                f"review carries id {self.review_id}, but its content "
                f"derives {derived}; the record does not survive itself"
            )


@dataclass(frozen=True, slots=True)
class RevisionRecord:
    """Succession as its own write-once fact: this review demanded a
    revision, and that manuscript is it. The manuscript's schema never
    learns about revision — a change of mind is a new record pointing
    at what it replaces, exactly as assessments supersede."""

    packet_id: str
    review_id: str
    superseded_manuscript_id: str
    revision_manuscript_id: str
    revision_id: str = field(default="")

    def __post_init__(self) -> None:
        derived = content_id("rvn", _identity_payload(self, "revision_id"))
        if not self.revision_id:
            object.__setattr__(self, "revision_id", derived)
        elif self.revision_id != derived:
            raise ReviewError(
                f"revision carries id {self.revision_id}, but its content "
                f"derives {derived}; the record does not survive itself"
            )


def _identity_payload(record: object, id_field: str) -> str:
    payload = to_jsonable(record)
    assert isinstance(payload, dict)
    payload.pop(id_field, None)
    return json.dumps(payload, sort_keys=True)


# -- deterministic findings ---------------------------------------------------


def _normalized(text: str) -> str:
    """Casefold and collapse whitespace — admission's quote rule,
    restated here because this package may not import it."""
    return " ".join(text.casefold().split())


def deterministic_findings(
    manuscript: Manuscript, packet: EvidencePacket
) -> tuple[ReviewFinding, ...]:
    """Trusted code's own reading of the prose, before any model runs.

    The unlicensed-verdict gate fires only when the verdict word is
    absent from the packet entirely — "supported by prior work" passes
    whenever some claim IS supported. Zero false positives on faithful
    drafts is chosen over catching every misuse; the model seat's
    ``overclaim`` covers the residue.
    """
    licensed = {
        finding.assessment.verdict.casefold()
        for finding in packet.claims
        if finding.assessment is not None
    }
    unlicensed = [
        str(verdict)
        for verdict in AssessmentVerdict
        if str(verdict) not in licensed
    ]
    findings: list[ReviewFinding] = []
    for name in PROSE_SECTIONS:
        text = getattr(manuscript.sections, name)
        for match in FORBIDDEN_STRENGTH.finditer(text):
            findings.append(
                ReviewFinding(
                    section=name,
                    quote=match.group(0),
                    issue="forbidden_strength",
                    subject_id="",
                    explanation=(
                        f"{match.group(0)!r} claims a strength no "
                        f"verdict in this system licenses"
                    ),
                    origin=ORIGIN_DETERMINISTIC,
                )
            )
        for word in unlicensed:
            for match in re.finditer(
                rf"\b{word}\b", text, flags=re.IGNORECASE
            ):
                findings.append(
                    ReviewFinding(
                        section=name,
                        quote=match.group(0),
                        issue="unlicensed_verdict",
                        subject_id="",
                        explanation=(
                            f"no claim in this packet records the "
                            f"verdict {word!r}"
                        ),
                        origin=ORIGIN_DETERMINISTIC,
                    )
                )
    return tuple(findings)


# -- grounding ----------------------------------------------------------------


def packet_identifiers(packet: EvidencePacket) -> frozenset[str]:
    """Every record id the packet prints, in either rendering — the
    universe a finding's subject may name."""
    return frozenset(
        _IDENTIFIER.findall(render_markdown(packet))
    ) | frozenset(_IDENTIFIER.findall(to_json(packet)))


def ground_findings(
    payload: Mapping[str, object],
    manuscript: Manuscript,
    packet: EvidencePacket,
) -> tuple[tuple[ProseRejection, ...], tuple[ReviewFinding, ...]]:
    """Hold every model finding to its own evidence. All rules fire —
    the corrective call deserves the whole list, not the first item."""
    known_ids = packet_identifiers(packet)
    rejections: list[ProseRejection] = []
    grounded: list[ReviewFinding] = []
    entries = payload.get("findings")
    # The schema guarantees the array; validation may hand it back as a
    # list or a tuple.
    assert isinstance(entries, Sequence) and not isinstance(entries, str)
    for index, entry in enumerate(entries):
        assert isinstance(entry, Mapping)
        section = str(entry.get("section", ""))
        quote = str(entry.get("quote", ""))
        issue = str(entry.get("issue", ""))
        subject_id = str(entry.get("subject_id", ""))
        where = f"finding {index + 1}"
        sound = True
        if section not in PROSE_SECTIONS:
            rejections.append(
                ProseRejection(
                    "unknown_section",
                    f"{where} names section {section!r}, which the "
                    f"manuscript does not have",
                )
            )
            sound = False
        if issue not in MODEL_ISSUES:
            rejections.append(
                ProseRejection(
                    "unknown_issue",
                    f"{where} names issue {issue!r}, which is not a "
                    f"reviewable issue",
                )
            )
            sound = False
        if not quote.strip():
            rejections.append(
                ProseRejection(
                    "empty_quote",
                    f"{where} quotes nothing; a finding is grounded in "
                    f"the prose it criticizes",
                )
            )
            sound = False
        elif section in PROSE_SECTIONS:
            text = getattr(manuscript.sections, section)
            if _normalized(quote) not in _normalized(text):
                rejections.append(
                    ProseRejection(
                        "ungrounded_quote",
                        f"{where} quotes text not present in the "
                        f"{section} section; a review that cannot "
                        f"ground its criticism is not a review",
                    )
                )
                sound = False
        if subject_id and subject_id not in known_ids:
            rejections.append(
                ProseRejection(
                    "unknown_subject",
                    f"{where} cites {subject_id}, which this packet "
                    f"does not print",
                )
            )
            sound = False
        if sound:
            grounded.append(
                ReviewFinding(
                    section=section,
                    quote=quote,
                    issue=issue,
                    subject_id=subject_id,
                    explanation=str(entry.get("explanation", "")),
                    origin=ORIGIN_MODEL,
                )
            )
    return tuple(rejections), tuple(grounded)


def derive_verdict(
    findings: tuple[ReviewFinding, ...],
) -> ReviewVerdict:
    """Trusted code's judgment, from findings alone. The model's schema
    has no verdict property: the one judgment that matters is never the
    model's to output."""
    return ReviewVerdict.REVISE if findings else ReviewVerdict.APPROVED


def review_schema(packet: EvidencePacket) -> OutputSchema:
    """The review's output shape, built per packet so invention is
    unexpressible: the subject enum holds exactly the ids this packet
    prints, and the issue enum holds exactly the model's vocabulary.
    Sorted, so the request fingerprint is deterministic."""
    return OutputSchema(
        name="manuscript_review",
        json_schema={
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section": {
                                "type": "string",
                                "enum": list(PROSE_SECTIONS),
                            },
                            "issue": {
                                "type": "string",
                                "enum": sorted(MODEL_ISSUES),
                            },
                            "quote": {"type": "string"},
                            "subject_id": {
                                "type": "string",
                                "enum": [
                                    "",
                                    *sorted(packet_identifiers(packet)),
                                ],
                            },
                            "explanation": {"type": "string"},
                        },
                        "required": [
                            "section",
                            "issue",
                            "quote",
                            "subject_id",
                            "explanation",
                        ],
                    },
                }
            },
            "required": ["findings"],
        },
    )


def render_prose_for_review(sections: ProseSections) -> str:
    """The draft's prose as labeled blocks, deterministic."""
    blocks = []
    for name in PROSE_SECTIONS:
        blocks.append(f"=== {name} ===")
        blocks.append(getattr(sections, name))
        blocks.append("")
    return "\n".join(blocks)


def require_reviewable(
    packet: EvidencePacket, manuscript: Manuscript
) -> None:
    if manuscript.packet_id != packet.packet_id:
        raise ManuscriptError(
            f"manuscript {manuscript.manuscript_id} was authored from "
            f"packet {manuscript.packet_id}, not {packet.packet_id}"
        )
