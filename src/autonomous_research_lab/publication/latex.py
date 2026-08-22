"""Venue rendering: the same record, typeset for whichever conference.

A venue is where a manuscript is typeset, and nothing more: deployment
data, parsed from JSON, validated loudly at composition time, and never
written into any scientific record. Retargeting a paper from NeurIPS to
ICML changes rendering only — the packet, the manuscript, and the review
it rode on are byte-for-byte the same records either way.

Two invariants carry over from the manuscript unchanged:

* **numbers** — the ``.tex`` prints exactly the ``:g`` strings the
  packet's own renderers print, through the same
  :func:`~.packet.metric_text` helper. No ``siunitx``, no thousands
  separators, no re-rounding: trusted code prints only what the packet
  prints, in exactly one spelling, in every format;
* **authorship** — the lab never fabricates human authorship. An
  anonymous venue gets ``Anonymous Authors`` and no attribution; a
  non-anonymous one gets an institutional author name and an
  attribution section carrying exactly the sentence the manuscript's
  own assembly prints.

The built-in venues are a conventional first cut — a venue's package
name tracks the year's kit, and an operator overrides any field with a
venue JSON file. The ``plain`` venue uses the bare ``article`` class
with zero staged files and zero packages, so it compiles anywhere a
TeX exists and keeps the whole path testable without one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from .manuscript import Manuscript, ManuscriptError
from .packet import EvidencePacket, metric_text

_ESCAPES: Final = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_ESCAPE_TABLE: Final = str.maketrans(_ESCAPES)

#: The strict citation shape the manuscript gates admitted. Split on
#: this BEFORE escaping: escaping first would turn the id's underscore
#: into ``\_`` and mangle the cite key.
_CITATION: Final = re.compile(r"\[([a-z]+_[0-9a-f]{16})\]")

ANONYMOUS_AUTHOR: Final = "Anonymous Authors"


class VenueError(ValueError):
    """The venue configuration cannot be used as given."""


def escape(text: str) -> str:
    """LaTeX-escape body text in a single pass. ``str.translate`` maps
    each source character exactly once, so backslash ordering is a
    non-issue by construction. ``[`` and ``]`` stay literal: they are
    not special in body text, and no rendered line begins with one
    where an optional argument could be read."""
    return text.translate(_ESCAPE_TABLE)


def prose_to_latex(text: str) -> str:
    """Prose with its citations converted: split on the strict citation
    pattern first, escape only the non-citation segments, and emit each
    source id verbatim inside ``\\citep``. A bracketed span that is not
    a well-formed source id (impossible in gated prose, defended
    anyway) is escaped as literal text."""
    parts = _CITATION.split(text)
    return "".join(
        rf"\citep{{{part}}}" if index % 2 else escape(part)
        for index, part in enumerate(parts)
    )


# -- the venue ----------------------------------------------------------------

_UNSAFE_NAME: Final = re.compile(r"[/\\]|\.\.")


@dataclass(frozen=True, slots=True)
class VenueSpec:
    name: str
    documentclass: str = "article"
    class_options: tuple[str, ...] = ()
    style_package: str = ""
    style_options: tuple[str, ...] = ()
    kit: str = ""
    bibliography_style: str = "plain"
    anonymous: bool = False
    author: str = "Autonomous Research Lab"

    def validate(self) -> None:
        if not self.name or _UNSAFE_NAME.search(self.name):
            raise VenueError(
                f"venue name must be a plain directory name, got "
                f"{self.name!r}"
            )
        if self.kit and _UNSAFE_NAME.search(self.kit):
            raise VenueError(
                f"kit name must be a plain directory name, got "
                f"{self.kit!r}"
            )
        if self.style_package and not self.kit:
            raise VenueError(
                f"venue {self.name} names style package "
                f"{self.style_package!r} but no kit; the kit is where "
                f"the package's files come from, and a package with no "
                f"staged kit cannot compile"
            )
        if not self.documentclass:
            raise VenueError(f"venue {self.name} names no documentclass")
        if not self.bibliography_style:
            raise VenueError(
                f"venue {self.name} names no bibliography style"
            )
        if not self.anonymous and not self.author.strip():
            raise VenueError(
                f"venue {self.name} is not anonymous and names no author"
            )


#: Conventional first cuts. A package name tracks the year's kit — the
#: operator stages the official kit and overrides any field with a
#: venue JSON file when the year moves on. Full venue-macro fidelity
#: (``\\icmltitle`` and friends) is recorded future work.
VENUES: Final[Mapping[str, VenueSpec]] = {
    "plain": VenueSpec(name="plain"),
    "neurips": VenueSpec(
        name="neurips",
        style_package="neurips_2026",
        style_options=("final",),
        kit="neurips",
        bibliography_style="plainnat",
        anonymous=True,
    ),
    "icml": VenueSpec(
        name="icml",
        style_package="icml2026",
        kit="icml",
        bibliography_style="plainnat",
        anonymous=True,
    ),
    "iclr": VenueSpec(
        name="iclr",
        style_package="iclr2026_conference",
        kit="iclr",
        bibliography_style="plainnat",
        anonymous=True,
    ),
}

_VENUE_KEYS: Final = frozenset(
    {
        "name",
        "documentclass",
        "class_options",
        "style_package",
        "style_options",
        "kit",
        "bibliography_style",
        "anonymous",
        "author",
    }
)


def venue_from(payload: Mapping[str, object]) -> VenueSpec:
    """A venue from deployment data, refused loudly when malformed."""
    unknown = sorted(set(payload) - _VENUE_KEYS)
    if unknown:
        raise VenueError(f"unknown venue key(s): {', '.join(unknown)}")
    if "name" not in payload:
        raise VenueError("a venue names itself: 'name' is required")
    spec = VenueSpec(
        name=str(payload["name"]),
        documentclass=str(payload.get("documentclass", "article")),
        class_options=_strings(payload, "class_options"),
        style_package=str(payload.get("style_package", "")),
        style_options=_strings(payload, "style_options"),
        kit=str(payload.get("kit", "")),
        bibliography_style=str(payload.get("bibliography_style", "plain")),
        anonymous=bool(payload.get("anonymous", False)),
        author=str(payload.get("author", "Autonomous Research Lab")),
    )
    spec.validate()
    return spec


def _strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw = payload.get(key, ())
    if isinstance(raw, str) or not isinstance(raw, list | tuple):
        raise VenueError(f"venue {key} must be a list of strings")
    return tuple(str(item) for item in raw)


# -- bibtex -------------------------------------------------------------------


def bibtex_entries(packet: EvidencePacket) -> str:
    """One ``@misc`` per cited source, keyed by its source id verbatim.

    ``@misc`` is the honest minimal type — the packet does not record an
    entry kind. BibTeX ignores fields a style does not know, so one
    mapping serves every venue. The carried-but-never-printed packet
    fields (arxiv id, access level) surface here.
    """
    if packet.bibliography is None:
        return ""
    entries = []
    for entry in packet.bibliography.entries:
        fields = []
        if entry.authors:
            fields.append(
                f"  author = {{{escape(' and '.join(entry.authors))}}},"
            )
        fields.append(f"  title = {{{{{escape(entry.title)}}}}},")
        if entry.year is not None:
            fields.append(f"  year = {{{entry.year}}},")
        if entry.venue:
            fields.append(f"  howpublished = {{{escape(entry.venue)}}},")
        if entry.doi:
            fields.append(f"  doi = {{{escape(entry.doi)}}},")
        if entry.arxiv_id:
            fields.append(f"  eprint = {{{escape(entry.arxiv_id)}}},")
            fields.append("  archivePrefix = {arXiv},")
        if entry.url:
            fields.append(f"  url = {{{escape(entry.url)}}},")
        if entry.access_level:
            fields.append(
                f"  note = {{access: {escape(entry.access_level)}}},"
            )
        entries.append(
            f"@misc{{{entry.source_id},\n" + "\n".join(fields) + "\n}"
        )
    return "\n\n".join(entries) + "\n"


# -- the document -------------------------------------------------------------


def render_latex(
    packet: EvidencePacket, manuscript: Manuscript, venue: VenueSpec
) -> tuple[str, str]:
    """``(main_tex, references_bib)``, mirroring the manuscript
    assembly's section order one for one. Pure — the same inputs give
    the same bytes; the PDF a toolchain later makes from them is a
    derived artifact, not a record."""
    if manuscript.packet_id != packet.packet_id:
        raise ManuscriptError(
            f"manuscript {manuscript.manuscript_id} was authored from "
            f"packet {manuscript.packet_id}, not {packet.packet_id}"
        )
    p = packet.provenance
    sections = manuscript.sections
    title = (
        packet.bibliography.candidate_title
        if packet.bibliography is not None
        else packet.science.question
    )
    author = ANONYMOUS_AUTHOR if venue.anonymous else venue.author

    class_options = (
        f"[{','.join(venue.class_options)}]" if venue.class_options else ""
    )
    lines: list[str] = [
        f"% Rendered by trusted code from packet {packet.packet_id}.",
        f"\\documentclass{class_options}{{{venue.documentclass}}}",
    ]
    if venue.style_package:
        style_options = (
            f"[{','.join(venue.style_options)}]"
            if venue.style_options
            else ""
        )
        lines.append(
            f"\\usepackage{style_options}{{{venue.style_package}}}"
        )
    lines.extend(
        [
            "\\providecommand{\\citep}{\\cite}",
            f"\\title{{{escape(title)}}}",
            f"\\author{{{escape(author)}}}",
            "\\begin{document}",
            "\\maketitle",
            "",
            "\\begin{center}\\small",
            f"Manuscript {manuscript.manuscript_id} of evidence packet "
            f"{packet.packet_id}\\\\",
            f"Investigation {p.investigation_id} --- "
            f"{escape(p.label)}\\\\",
            f"Run {p.run_id} (record {p.run_record_id}) --- "
            f"{escape(p.authority)}",
            "\\end{center}",
            "",
            "\\begin{abstract}",
            prose_to_latex(sections.abstract),
            "\\end{abstract}",
            "",
            "\\section{Introduction}",
            prose_to_latex(sections.introduction),
            "",
            "\\section{Registered science}",
            f"Question ({packet.science.question_id}): "
            f"{escape(packet.science.question)}\\par",
            f"Hypothesis ({packet.science.hypothesis_id}): "
            f"{escape(packet.science.hypothesis)}\\par",
            f"Mechanical reading: "
            f"{escape(packet.science.mechanical_reading)}",
        ]
    )
    if packet.science.admitted_predictions:
        lines.append("\\begin{itemize}")
        for admitted in packet.science.admitted_predictions:
            lines.append(
                f"\\item \\emph{{{escape(admitted.prediction_text)}}} "
                f"--- {escape(admitted.base_metric)}: "
                f"{escape(admitted.expected_higher_arm)} over "
                f"{escape(admitted.expected_lower_arm)}, "
                f"{escape(admitted.condition)}"
            )
        lines.append("\\end{itemize}")
    lines.extend(
        [
            "",
            "\\section{Method}",
            prose_to_latex(sections.method_narrative),
        ]
    )
    registrations = [
        registration
        for finding in packet.claims
        for registration in finding.registrations
    ]
    if registrations:
        lines.append("\\begin{itemize}")
        for registration in registrations:
            lines.append(
                f"\\item Pre-registered: {escape(registration.metric)} "
                f"{escape(registration.comparator)} "
                f"{registration.threshold:g}, "
                f"{escape(registration.condition)} (prediction "
                f"{registration.prediction_id}, experiment "
                f"{registration.spec_id})"
            )
        lines.append("\\end{itemize}")
    lines.extend(["", "\\section{Results}"])
    for finding in packet.claims:
        lines.append(f"\\subsection{{{escape(finding.statement)}}}")
        lines.append(
            f"Claim {finding.claim_id} --- {finding.figures_check}.\\par"
        )
        if finding.assessment is not None:
            lines.append(
                f"\\textbf{{{finding.assessment.verdict.upper()}}} "
                f"({escape(finding.assessment.method)}, assessment "
                f"{finding.assessment.assessment_id})\\par"
            )
            lines.append(
                f"\\begin{{quote}}"
                f"{escape(finding.assessment.rationale)}"
                f"\\end{{quote}}"
            )
        if finding.evidence_rows:
            lines.append("\\begin{itemize}")
            for row in finding.evidence_rows:
                lines.append(
                    f"\\item {row.evidence_id} ({row.relation}, "
                    f"{row.standing}) $\\leftarrow$ result "
                    f"{row.result_id}, seed {row.seed}: "
                    f"{escape(metric_text(row.metrics))}"
                )
            lines.append("\\end{itemize}")
    if packet.tables:
        lines.extend(
            [
                "",
                "\\begin{center}",
                "\\begin{tabular}{lllll}",
                "spec & seed & result & standing & metrics \\\\ \\hline",
            ]
        )
        for table_row in packet.tables:
            lines.append(
                f"{table_row.spec_id} & {table_row.seed} & "
                f"{table_row.result_id} & {table_row.standing} & "
                f"{escape(metric_text(table_row.metrics))} \\\\"
            )
        lines.extend(["\\end{tabular}", "\\end{center}"])
    lines.extend(
        [
            "",
            "\\section{Discussion}",
            prose_to_latex(sections.discussion),
            "",
            "\\section{Limitations}",
            prose_to_latex(sections.limitations),
            "",
            f"\\bibliographystyle{{{venue.bibliography_style}}}",
            "\\bibliography{references}",
        ]
    )
    if not venue.anonymous:
        call = manuscript.call
        lines.extend(
            [
                "",
                "\\section*{Attribution}",
                (
                    f"Prose authored by {escape(call.served_model)} via "
                    f"{escape(call.provider)} "
                    f"({call.input_tokens}/{call.output_tokens} tokens "
                    f"in/out, {call.repair_count} corrective call(s)); "
                    f"every other section assembled by trusted code "
                    f"from packet {packet.packet_id}."
                ),
            ]
        )
    lines.extend(["\\end{document}", ""])
    return "\n".join(lines), bibtex_entries(packet)
