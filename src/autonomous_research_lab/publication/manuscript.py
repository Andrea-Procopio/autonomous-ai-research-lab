"""The manuscript: model prose behind gates that refuse invention.

The division of labor is the anti-fabrication rule applied to writing.
Trusted code renders every load-bearing section — the registered
science, the findings with their figures, the result table, the
references — straight from the evidence packet, reusing the packet's own
line renderers byte for byte. A model authors only the five prose
sections named in :data:`PROSE_SECTIONS`, and three deterministic gates
judge the tokens before any prose is kept:

* the **number gate** — a word containing a digit is admissible iff
  trusted code already printed that word for this packet. The allowed
  set is every digit-bearing word in the packet's own markdown and JSON
  renderings, so there is no formatter list to maintain and no
  whitelist to argue about: rounded values, unit conversions, recomputed
  percentages, and obfuscations like ``3x`` are unknown by
  construction, while quoting an id or ``600s`` exactly as the packet
  prints it passes;
* the **citation gate** — every bracketed span must be a well-formed
  source id from the packet's bibliography. Markdown links and prose
  citations (``[Smith 2020]``) are malformed by the same rule;
* the **structure gate** — no prose line may open a heading, and no
  section may be empty. Trusted code owns the document's shape.

What the gates do not judge is meaning: whether the discussion
overclaims is the faithfulness reviewer's question (:mod:`.review`),
asked against the packet with grounded findings. Building a weaker
semantic check here would only train prose to pass it.

A :class:`Manuscript` is one authored call's accepted output, named by a
content id over everything in it — sections, packet id, and the call's
provenance. Two identical drafts from distinct calls are distinct
records (the response id is an occurrence, not content); replay
idempotence is the composition root's job, keyed on the packet id.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Final

from ..core.ids import content_id
from ..core.serialize import to_jsonable
from .packet import (
    Bibliography,
    EvidencePacket,
    finding_lines,
    reference_lines,
    render_markdown,
    science_lines,
    table_lines,
    to_json,
)

PROSE_SECTIONS: Final = (
    "abstract",
    "introduction",
    "method_narrative",
    "discussion",
    "limitations",
)

#: A digit-bearing word: the unit the number gate compares. The class
#: keeps signs, decimals, exponents, percent signs, and identifier
#: characters inside the word, so ``-0.5``, ``1e-3``, ``85%``, ``3x``
#: and ``res_1056647fa343295e`` are each one token — and each must
#: appear verbatim in the packet's own renderings to pass. Sentence
#: punctuation is stripped afterwards, symmetrically on both sides.
DIGIT_WORD: Final = re.compile(r"[\w.%+-]*\d[\w.%+-]*")

_TRAILING_PUNCTUATION: Final = ".,;:"

_BRACKETED: Final = re.compile(r"\[([^\[\]]*)\]")
_SOURCE_ID: Final = re.compile(r"[a-z]+_[0-9a-f]{16}\Z")


class ManuscriptError(RuntimeError):
    """The manuscript cannot honestly be produced."""


class NothingToReportError(ManuscriptError):
    """The packet records no claims: a precondition, not a fault."""


class ManuscriptRejectedError(ManuscriptError):
    """The model's prose failed the gates after every corrective call."""


@dataclass(frozen=True, slots=True)
class ProseRejection:
    """One mechanical rule the prose broke, phrased for the corrective
    call: the rule name is stable, the detail names the section and the
    offending token."""

    rule: str
    detail: str


@dataclass(frozen=True, slots=True)
class ProseSections:
    abstract: str
    introduction: str
    method_narrative: str
    discussion: str
    limitations: str


@dataclass(frozen=True, slots=True)
class AuthorCall:
    """The accepted call's provenance, as flat data. A mirror of the
    analysis chain's call-provenance record, restated here because this
    package may not import it."""

    request_fingerprint: str
    response_id: str
    provider: str
    requested_model: str
    served_model: str
    provider_request_id: str | None
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    repair_count: int


@dataclass(frozen=True, slots=True)
class Manuscript:
    packet_id: str
    sections: ProseSections
    call: AuthorCall
    manuscript_id: str = field(default="")

    def __post_init__(self) -> None:
        derived = content_id("mscr", _identity_payload(self))
        if not self.manuscript_id:
            object.__setattr__(self, "manuscript_id", derived)
        elif self.manuscript_id != derived:
            raise ManuscriptError(
                f"manuscript carries id {self.manuscript_id}, but its "
                f"content derives {derived}; the record does not survive "
                f"itself"
            )


def _identity_payload(manuscript: Manuscript) -> str:
    payload = to_jsonable(manuscript)
    assert isinstance(payload, dict)
    payload.pop("manuscript_id", None)
    return json.dumps(payload, sort_keys=True)


def manuscript_to_json(manuscript: Manuscript) -> str:
    """The canonical serialization, matching the packet's convention."""
    return (
        json.dumps(to_jsonable(manuscript), indent=2, sort_keys=True) + "\n"
    )


def require_reportable(packet: EvidencePacket) -> None:
    """A packet without claims funds no manuscript. Refusing here keeps
    the model from ever being asked to write about a run that measured
    nothing — the empty packet is already the honest export."""
    if not packet.claims:
        raise NothingToReportError(
            f"packet {packet.packet_id} records no claims; there is "
            f"nothing a manuscript may honestly claim"
        )


# -- the gates ----------------------------------------------------------------


def _digit_words(text: str) -> frozenset[str]:
    return frozenset(
        stripped
        for match in DIGIT_WORD.finditer(text)
        if (stripped := match.group(0).rstrip(_TRAILING_PUNCTUATION))
        and any(char.isdigit() for char in stripped)
    )


def known_renderings(packet: EvidencePacket) -> frozenset[str]:
    """Every digit-bearing word trusted code prints for this packet, in
    both the markdown and the JSON renderings. A number is known iff it
    is in this set — the same tokenizer runs on the prose, so the
    comparison is symmetric."""
    return _digit_words(render_markdown(packet)) | _digit_words(
        to_json(packet)
    )


def gate_prose(
    payload: dict[str, str],
    *,
    allowed: frozenset[str],
    bibliography: Bibliography | None,
) -> tuple[ProseRejection, ...]:
    """Every mechanical rule the prose breaks, in section order. The
    schema has already guaranteed the five string keys; this judges
    their content."""
    known_sources = frozenset(
        entry.source_id
        for entry in (bibliography.entries if bibliography else ())
    )
    rejections: list[ProseRejection] = []
    for name in PROSE_SECTIONS:
        text = payload[name]
        if not text.strip():
            rejections.append(
                ProseRejection(
                    rule="empty_section",
                    detail=f"{name}: the section is empty",
                )
            )
            continue
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                rejections.append(
                    ProseRejection(
                        rule="structural_markup",
                        detail=(
                            f"{name}: a line opens a heading "
                            f"({line.strip()[:40]!r}); the document's "
                            f"structure is not the model's to write"
                        ),
                    )
                )
        # Bracketed spans belong to the citation gate; judging their
        # digits twice would name one mistake as two.
        for token in sorted(_digit_words(_BRACKETED.sub(" ", text))):
            if token not in allowed:
                rejections.append(
                    ProseRejection(
                        rule="unknown_number",
                        detail=(
                            f"{name}: {token} is not a number this "
                            f"packet states"
                        ),
                    )
                )
        for span in _BRACKETED.finditer(text):
            key = span.group(1)
            if not _SOURCE_ID.fullmatch(key):
                rejections.append(
                    ProseRejection(
                        rule="malformed_citation",
                        detail=(
                            f"{name}: [{key[:40]}] is not a source id; "
                            f"cite only bracketed ids from the references"
                        ),
                    )
                )
            elif key not in known_sources:
                rejections.append(
                    ProseRejection(
                        rule="unknown_citation",
                        detail=(
                            f"{name}: [{key}] is not in this packet's "
                            f"bibliography"
                        ),
                    )
                )
    return tuple(rejections)


# -- assembly -----------------------------------------------------------------


def assemble(packet: EvidencePacket, manuscript: Manuscript) -> str:
    """The manuscript document: trusted skeleton, prose in its five
    slots, nothing else. Pure — the same inputs give the same bytes."""
    if manuscript.packet_id != packet.packet_id:
        raise ManuscriptError(
            f"manuscript {manuscript.manuscript_id} was authored from "
            f"packet {manuscript.packet_id}, not {packet.packet_id}"
        )
    p = packet.provenance
    title = (
        packet.bibliography.candidate_title
        if packet.bibliography is not None
        else packet.science.question
    )
    sections = manuscript.sections
    lines: list[str] = [
        f"# {title}",
        "",
        f"Manuscript {manuscript.manuscript_id} of evidence packet "
        f"{packet.packet_id}",
        f"**Investigation** {p.investigation_id} — {p.label}",
        f"**Run** {p.run_id} (record {p.run_record_id}) — {p.authority}",
        "",
        "## Abstract",
        "",
        sections.abstract,
        "",
        "## Introduction",
        "",
        sections.introduction,
        "",
        "## Registered science",
        "",
        *science_lines(packet.science),
        "",
        "## Method",
        "",
        sections.method_narrative,
        "",
    ]
    for finding in packet.claims:
        for registration in finding.registrations:
            lines.append(
                f"- Pre-registered: {registration.metric} "
                f"{registration.comparator} {registration.threshold:g}, "
                f"{registration.condition} (prediction "
                f"{registration.prediction_id}, experiment "
                f"{registration.spec_id})"
            )
    lines.append("")
    lines.append("## Results")
    for finding in packet.claims:
        lines.append("")
        lines.extend(finding_lines(finding))
        for figure in packet.figures:
            if figure.claim_id != finding.claim_id:
                continue
            lines.append("")
            lines.append(
                f"![{figure.caption}](../figures/{figure.figure_id}.png)"
            )
    lines.extend(
        [
            "",
            *table_lines(packet.tables),
            "",
            "## Discussion",
            "",
            sections.discussion,
            "",
            "## Limitations",
            "",
            sections.limitations,
            "",
            "## References",
            "",
        ]
    )
    if packet.bibliography is not None:
        lines.extend(reference_lines(packet.bibliography))
    lines.extend(
        [
            "",
            (
                f"Prose authored by {manuscript.call.served_model} via "
                f"{manuscript.call.provider} "
                f"({manuscript.call.input_tokens}/"
                f"{manuscript.call.output_tokens} tokens in/out, "
                f"{manuscript.call.repair_count} corrective call(s)); "
                f"every other section assembled by trusted code from "
                f"packet {packet.packet_id}."
            ),
            "",
        ]
    )
    return "\n".join(lines)
