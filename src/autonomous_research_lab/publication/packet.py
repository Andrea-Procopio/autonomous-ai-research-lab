"""The evidence packet: everything a manuscript may later claim, checked.

A deterministic projection of one completed run into a single durable
document: every claim with its verdict and the figures behind it, every
per-seed number, every citation, every table row — each element carrying
the ids of the records and artifact digests it derives from. Downstream
writing consumes this packet; nothing model-authored enters it.

The packet is *checked, not copied*. For an assessment the deterministic
statistician issued, the figures are re-derived here from the immutable
prediction tests — with the Bonferroni denominator and alpha parsed back
out of the recorded rationale, because the statistician pins them at
assessment time and the head state can legitimately imply larger ones —
and the re-rendered summary must equal the recorded rationale byte for
byte. A mismatch fails the export: it means the arithmetic drifted, or
something stamped the statistician's method on numbers trusted code did
not produce. Assessments by other methods are carried verbatim and
marked as restated.

This module holds the schema, the pure checks, and the renderers; it
deliberately imports nothing from the analysis chain, so every value
arrives as plain data mapped in by the composition root. Orderings are
stated where they are not obvious: claims and evidence rows follow state
order, planner decisions follow record-id order (the store's listing —
no timestamp exists), tables follow (spec id, seed).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

from ..core.assessment import AssessmentVerdict, EpistemicAssessment
from ..core.claim import Claim
from ..core.ids import content_id
from ..core.prediction import Consistency, Prediction, PredictionTest
from ..core.serialize import to_jsonable
from ..core.state import ResearchState
from ..evidence.store import EvidenceStore
from ..runtime.statistics import FamilyStatistics, assess_family

STATISTICIAN_METHOD: Final = "statistician:exact-sign-v1"

RE_DERIVED: Final = "re-derived-and-matched"
RESTATED: Final = "restated-from-record"
NOT_ASSESSED: Final = "not-assessed"

_COMPARISONS: Final = re.compile(r"across (\d+) comparison\(s\)")
_ALPHA: Final = re.compile(r"alpha ([0-9eE.+-]+) Bonferroni")

_SENTINEL_RATIONALE: Final = "no admissible conclusive observations"


class PacketError(RuntimeError):
    """The packet cannot honestly be built from this record."""


class FiguresMismatchError(PacketError):
    """A statistician assessment's rationale does not equal the figures
    re-derived from the record it cites."""


# -- flat mirrors -------------------------------------------------------------
#
# Strings and numbers only: the analysis-chain records these mirror are
# read by the composition root, which alone may import their packages.


@dataclass(frozen=True, slots=True)
class SupportQuote:
    source: str
    field_path: str
    quote: str


@dataclass(frozen=True, slots=True)
class AdmittedPrediction:
    prediction_text: str
    condition: str
    base_metric: str
    expected_higher_arm: str
    expected_lower_arm: str
    contrary_observation: str
    supports: tuple[SupportQuote, ...] = ()


@dataclass(frozen=True, slots=True)
class RequirementQuote:
    source: str
    record_id: str
    field_path: str
    quote: str


@dataclass(frozen=True, slots=True)
class RegisteredScience:
    admission_record_id: str
    question_id: str
    question: str
    hypothesis_id: str
    hypothesis: str
    mechanical_reading: str
    admitted_predictions: tuple[AdmittedPrediction, ...] = ()
    requirements: tuple[RequirementQuote, ...] = ()


@dataclass(frozen=True, slots=True)
class SpendSummary:
    granted_wall_clock_seconds: float
    granted_gpu_hours: float
    granted_usd: float
    granted_model_tokens: int
    remaining_wall_clock_seconds: float
    remaining_gpu_hours: float
    remaining_usd: float
    remaining_model_tokens: int
    stage_model_calls: int
    stage_input_tokens: int
    stage_output_tokens: int


@dataclass(frozen=True, slots=True)
class VerifySummary:
    states_checked: int
    results_checked: int
    evidence_checked: int
    blobs_checked: int


@dataclass(frozen=True, slots=True)
class Provenance:
    investigation_id: str
    label: str
    config_id: str
    brief_topic: str
    run_id: str
    run_record_id: str
    authority: str
    head_state_id: str
    spend: SpendSummary
    verified: VerifySummary


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: str
    digest: str
    size_bytes: int
    media_type: str
    kind: str


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    evidence_id: str
    result_id: str
    seed: int | None
    relation: str
    standing: str
    metrics: tuple[tuple[str, float], ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True, slots=True)
class FigureSet:
    """One re-derived :class:`FamilyStatistics`, flattened, with the
    verdict the arithmetic supports."""

    verdict: str
    metric: str
    comparator: str
    threshold: float
    n: int
    values: tuple[float, ...]
    mean: float | None
    stdev: float | None
    consistent_count: int
    direction: str
    p_value: float | None
    effect: float | None
    alpha: float
    comparisons: int
    alpha_adjusted: float


@dataclass(frozen=True, slots=True)
class RenderedFigure:
    """One rendered figure the store pins: the family's numbers, the
    trusted caption, and the digests of the bytes that drew them. No
    timestamp — provenance stays in the manifest, outside identity."""

    figure_id: str
    claim_id: str
    prediction_id: str
    metric: str
    comparator: str
    threshold: float
    points: tuple[tuple[int | None, float], ...]
    n: int
    mean: float | None
    stdev: float | None
    caption: str
    renderer: str
    files: tuple[tuple[str, str, int], ...]
    """``(name, sha256, size_bytes)``, sorted by name."""


@dataclass(frozen=True, slots=True)
class PredictionRegistration:
    prediction_id: str
    spec_id: str
    metric: str
    comparator: str
    threshold: float
    tolerance: float
    condition: str


@dataclass(frozen=True, slots=True)
class AssessmentRecord:
    assessment_id: str
    verdict: str
    method: str
    rationale: str
    scope: str
    evidence_ids: tuple[str, ...] = ()
    supersedes: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimFinding:
    claim_id: str
    statement: str
    scope: str
    figures_check: str
    assessment: AssessmentRecord | None = None
    figures: tuple[FigureSet, ...] = ()
    registrations: tuple[PredictionRegistration, ...] = ()
    evidence_rows: tuple[EvidenceRow, ...] = ()


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    record_id: str
    action: str
    rationale: str
    spec_id: str
    stop_reason: str


@dataclass(frozen=True, slots=True)
class BibliographyEntry:
    source_id: str
    title: str
    authors: tuple[str, ...]
    venue: str
    year: int | None
    doi: str
    arxiv_id: str
    url: str
    access_level: str


@dataclass(frozen=True, slots=True)
class Bibliography:
    candidate_id: str
    candidate_title: str
    research_question: str
    entries: tuple[BibliographyEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class TableRow:
    spec_id: str
    seed: int | None
    result_id: str
    standing: str
    metrics: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    provenance: Provenance
    science: RegisteredScience
    claims: tuple[ClaimFinding, ...] = ()
    planner_decisions: tuple[PlannerDecision, ...] = ()
    bibliography: Bibliography | None = None
    tables: tuple[TableRow, ...] = ()
    figures: tuple[RenderedFigure, ...] = ()
    """Rendered figures, populated by the composition root from the
    write-once figure store and re-derived against the head state at
    build time. Empty stays the honest statement of absence."""

    packet_id: str = field(default="")

    def __post_init__(self) -> None:
        derived = content_id("epkt", _identity_payload(self))
        if not self.packet_id:
            object.__setattr__(self, "packet_id", derived)
        elif self.packet_id != derived:
            raise PacketError(
                f"packet carries id {self.packet_id}, but its content "
                f"derives {derived}; the record does not survive itself"
            )


def _identity_payload(packet: EvidencePacket) -> str:
    payload = to_jsonable(packet)
    assert isinstance(payload, dict)
    payload.pop("packet_id", None)
    return json.dumps(payload, sort_keys=True)


def to_json(packet: EvidencePacket) -> str:
    """The canonical serialization: sorted keys, newline-terminated,
    byte-identical for byte-identical inputs."""
    payload = to_jsonable(packet)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# -- the pure checks ----------------------------------------------------------


def replication_family(
    state: ResearchState,
    prediction: Prediction,
    *,
    admissible: Callable[[str], bool],
    seed_of: Mapping[str, int | None],
) -> tuple[PredictionTest, ...]:
    """One prediction's admissible conclusive tests, seed-ordered.

    The statistician's family rule, restated here because the lab that
    first wrote it lives outside the package boundary. The duplication
    is self-guarding: if the two ever drift, the render-equality check
    below fails the export loudly.
    """
    tests = [
        test
        for test in state.tests_for(prediction.id)
        if test.consistency is not Consistency.INCONCLUSIVE
        and admissible(test.result_id)
    ]
    tests.sort(
        key=lambda test: (
            seed_of.get(test.result_id) is None,
            seed_of.get(test.result_id) or 0,
        )
    )
    return tuple(tests)


def own_predictions(
    state: ResearchState, claim: Claim, store: EvidenceStore
) -> tuple[Prediction, ...]:
    """The predictions a claim's own evidence tested: links → evidence →
    spec → prediction, in state order. Empty when the claim's links
    resolve to no known prediction — the caller refuses rather than
    guessing, because the statistician's fallback in that case widened
    the family and a silent re-derivation here would falsely mismatch.
    """
    linked_evidence = {
        link.evidence_id
        for link in state.evidence_links
        if link.claim_id == claim.id
    }
    linked_specs = {
        store.get_evidence(evidence_id).spec_id
        for evidence_id in linked_evidence
    }
    prediction_ids = {
        spec.prediction_id
        for spec in state.experiments
        if spec.id in linked_specs
    }
    return tuple(
        found for found in state.predictions if found.id in prediction_ids
    )


def check_statistician_assessment(
    assessment: EpistemicAssessment,
    state: ResearchState,
    store: EvidenceStore,
    *,
    admissible: Callable[[str], bool],
    seed_of: Mapping[str, int | None],
) -> tuple[tuple[AssessmentVerdict, FamilyStatistics], ...]:
    """Re-derive a statistician assessment's figures, and hold them to
    the recorded rationale byte for byte.

    The Bonferroni denominator and alpha are parsed from the rationale —
    the statistician pins them at assessment time, and the head state can
    legitimately imply larger ones. The rationale's format is this
    repository's own (:meth:`FamilyStatistics.render`), so the parse is a
    contract, not a heuristic.
    """
    if assessment.method != STATISTICIAN_METHOD:
        raise PacketError(
            f"assessment {assessment.id} was made by "
            f"{assessment.method!r}; only {STATISTICIAN_METHOD!r} figures "
            f"can be re-derived"
        )
    if assessment.rationale == _SENTINEL_RATIONALE:
        if assessment.verdict is not AssessmentVerdict.UNDETERMINED:
            raise FiguresMismatchError(
                f"assessment {assessment.id} claims "
                f"{assessment.verdict} over no admissible observations"
            )
        return ()
    claim = state.claim(assessment.subject_id)
    if claim is None:
        raise PacketError(
            f"assessment {assessment.id} judges {assessment.subject_id}, "
            f"which is not a claim in the head state"
        )
    comparisons = _parsed_comparisons(assessment)
    alpha = _parsed_alpha(assessment)
    own = own_predictions(state, claim, store)
    if not own:
        raise PacketError(
            f"claim {claim.id} links to no known prediction; its "
            f"statistician figures cannot be re-derived faithfully"
        )
    assessed = tuple(
        assess_family(
            prediction,
            replication_family(
                state, prediction, admissible=admissible, seed_of=seed_of
            ),
            alpha=alpha,
            comparisons=comparisons,
        )
        for prediction in own
    )
    rendered = " | ".join(stats.render() for _, stats in assessed)
    if rendered != assessment.rationale:
        raise FiguresMismatchError(
            f"assessment {assessment.id}: the figures re-derived from the "
            f"record do not match its rationale.\n"
            f"  recorded:   {assessment.rationale}\n"
            f"  re-derived: {rendered}\n"
            f"(exports are always governance-on; an ablated run's "
            f"assessments cannot be re-derived and cannot be packeted)"
        )
    return assessed


def _parsed_comparisons(assessment: EpistemicAssessment) -> int:
    found = _COMPARISONS.findall(assessment.rationale)
    if not found or len(set(found)) != 1:
        raise FiguresMismatchError(
            f"assessment {assessment.id} does not state one Bonferroni "
            f"denominator in its rationale"
        )
    return int(found[0])


def _parsed_alpha(assessment: EpistemicAssessment) -> float:
    found = _ALPHA.findall(assessment.rationale)
    if not found or len(set(found)) != 1:
        raise FiguresMismatchError(
            f"assessment {assessment.id} does not state one alpha in its "
            f"rationale"
        )
    return float(found[0])


def figure_set(verdict: AssessmentVerdict, stats: FamilyStatistics) -> FigureSet:
    return FigureSet(
        verdict=str(verdict),
        metric=stats.metric,
        comparator=str(stats.comparator),
        threshold=stats.threshold,
        n=stats.n,
        values=stats.values,
        mean=stats.mean,
        stdev=stats.stdev,
        consistent_count=stats.consistent_count,
        direction=str(stats.direction),
        p_value=stats.p_value,
        effect=stats.effect,
        alpha=stats.alpha,
        comparisons=stats.comparisons,
        alpha_adjusted=stats.alpha_adjusted,
    )


# -- rendering ----------------------------------------------------------------


def science_lines(science: RegisteredScience) -> list[str]:
    """The registered-science block, minus its heading. Shared with the
    manuscript, byte for byte: previously exported files pin these
    renderings, and a restatement would be the drift this package warns
    against."""
    lines = [
        f"**Question** ({science.question_id}): {science.question}",
        f"**Hypothesis** ({science.hypothesis_id}): {science.hypothesis}",
        f"**Mechanical reading**: {science.mechanical_reading}",
        "",
    ]
    for admitted in science.admitted_predictions:
        lines.append(
            f"- *{admitted.prediction_text}* — {admitted.base_metric}: "
            f"{admitted.expected_higher_arm} over "
            f"{admitted.expected_lower_arm}, {admitted.condition}"
        )
    return lines


def metric_text(metrics: tuple[tuple[str, float], ...]) -> str:
    """One metrics tuple as the packet prints it. Every rendering of a
    metric value — markdown, LaTeX, anything later — goes through this
    one ``:g`` expression, so trusted code prints only what the packet
    prints, in exactly one spelling."""
    return ", ".join(f"{name}={value:g}" for name, value in metrics)


def finding_lines(finding: ClaimFinding) -> list[str]:
    """One claim's finding block: statement, verdict, rationale, rows."""
    lines = [
        f"### {finding.statement}",
        f"Claim {finding.claim_id} — {finding.figures_check}",
    ]
    if finding.assessment is not None:
        lines.append(
            f"**{finding.assessment.verdict.upper()}** "
            f"({finding.assessment.method}, "
            f"assessment {finding.assessment.assessment_id})"
        )
        lines.append(f"> {finding.assessment.rationale}")
    for row in finding.evidence_rows:
        lines.append(
            f"- {row.evidence_id} ({row.relation}, {row.standing}) ← "
            f"result {row.result_id}, seed {row.seed}: "
            f"{metric_text(row.metrics)}"
        )
    return lines


def reference_lines(bibliography: Bibliography) -> list[str]:
    """The bibliography block, minus its heading."""
    lines = [
        f"Cited by candidate {bibliography.candidate_id} — "
        f"*{bibliography.candidate_title}*"
    ]
    for entry in bibliography.entries:
        authors = ", ".join(entry.authors)
        year = entry.year if entry.year is not None else "n.d."
        lines.append(
            f"- {authors} ({year}). {entry.title}. {entry.venue}. "
            f"{entry.doi or entry.url} [{entry.source_id}]"
        )
    return lines


def table_lines(tables: tuple[TableRow, ...]) -> list[str]:
    """The per-seed result table, header included."""
    lines = [
        "| spec | seed | result | standing | metrics |",
        "| --- | --- | --- | --- | --- |",
    ]
    for table_row in tables:
        lines.append(
            f"| {table_row.spec_id} | {table_row.seed} | "
            f"{table_row.result_id} | {table_row.standing} | "
            f"{metric_text(table_row.metrics)} |"
        )
    return lines


def render_markdown(packet: EvidencePacket) -> str:
    """A human-readable statement of the packet. The JSON is the record;
    this is the reading copy, derived from the same object."""
    p = packet.provenance
    lines: list[str] = [
        f"# Evidence packet {packet.packet_id}",
        "",
        f"**Investigation** {p.investigation_id} — {p.label}",
        f"**Run** {p.run_id} (record {p.run_record_id}) — {p.authority}",
        f"**Brief** {p.brief_topic}",
        f"**Head state** {p.head_state_id}",
        (
            f"**Verified from cold**: {p.verified.states_checked} states, "
            f"{p.verified.results_checked} results, "
            f"{p.verified.evidence_checked} evidence records, "
            f"{p.verified.blobs_checked} artifact blobs"
        ),
        (
            f"**Spend**: {p.spend.granted_wall_clock_seconds:g}s wall "
            f"granted, {p.spend.remaining_wall_clock_seconds:g}s remaining; "
            f"{p.spend.stage_model_calls} model call(s), "
            f"{p.spend.stage_input_tokens}/{p.spend.stage_output_tokens} "
            f"tokens in/out"
        ),
        "",
        "## Registered science",
        "",
    ]
    lines.extend(science_lines(packet.science))
    lines.append("")
    lines.append("## Findings")
    for finding in packet.claims:
        lines.append("")
        lines.extend(finding_lines(finding))
    lines.append("")
    lines.append("## Planner decisions")
    for decision in packet.planner_decisions:
        lines.append(
            f"- {decision.record_id}: {decision.action}"
            + (f" ({decision.stop_reason})" if decision.stop_reason else "")
            + f" — {decision.rationale}"
        )
    if packet.bibliography is not None:
        lines.append("")
        lines.append("## References")
        lines.extend(reference_lines(packet.bibliography))
    lines.append("")
    lines.append("## Result table")
    lines.append("")
    lines.extend(table_lines(packet.tables))
    lines.append("")
    if packet.figures:
        lines.append("## Figures")
        for figure in packet.figures:
            lines.append("")
            lines.append(f"### {figure.figure_id} — claim {figure.claim_id}")
            lines.append(figure.caption)
            for name, digest, size in figure.files:
                lines.append(
                    f"- {name} sha256 {digest} ({size} bytes), "
                    f"{figure.renderer}"
                )
    else:
        lines.append(
            "Figures: none — nothing in this run rendered plots, and the "
            "packet states that rather than implying an omission."
        )
    lines.append("")
    return "\n".join(lines)
