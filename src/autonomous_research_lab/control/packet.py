"""Building the evidence packet: the one walk that sees the whole run.

The packet's schema, checks and renderers live in
:mod:`autonomous_research_lab.publication.packet`, which the layering
rules keep away from the analysis chain; this module is the composition
root's half — it reads every store, maps records into the flat mirrors,
and hands back a packet whose id derives from its content.

Order of operations, and why:

1. **Resolve the investigation and its facts.** The stage log is
   self-verifying, and ``Fact.STATE_ID`` is the precondition: a walk
   that stopped before experimentation has nothing to packet, and the
   answer is a refusal, not a failure.
2. **Verify the run from cold.** ``verify_run`` covers snapshots,
   facts, artifacts, the ledger and the journal; a run whose records do
   not survive their own digests is not exported. The analysis-chain
   stores are not in its scope — their own loaders re-derive every
   content id — so every read below that returns nothing or raises is
   itself an export failure naming the record.
3. **Walk, map, check.** The statistician's figures are re-derived and
   held to the recorded rationale; everything else is carried verbatim
   with its ids. Export is always governance-on: admissibility comes
   from the durable verification records under ``verifications/`` (a
   lab-convention path, hardcoded here exactly the way the verifier
   hardcodes ``runs/``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..admission.records import AdmissionRecord
from ..admission.store import AdmissionIntegrityError
from ..core.claim import Claim
from ..core.state import ResearchState
from ..evidence.artifacts import ArtifactIntegrityError
from ..evidence.file_store import EvidenceIntegrityError
from ..ideation.store import IdeationIntegrityError
from ..literature.store import LiteratureIntegrityError
from ..persistence.state_store import SnapshotError
from ..program.integrity import IntegrityReport, verify_run
from ..program.records import ResearchRun
from ..publication.figures import (
    FigureError,
    FigureStore,
    StaleFigureError,
    figure_id_for,
    planned_figures,
)
from ..publication.packet import (
    NOT_ASSESSED,
    RE_DERIVED,
    RESTATED,
    STATISTICIAN_METHOD,
    AdmittedPrediction,
    ArtifactRef,
    AssessmentRecord,
    Bibliography,
    BibliographyEntry,
    ClaimFinding,
    EvidencePacket,
    EvidenceRow,
    FigureSet,
    PacketError,
    PlannerDecision,
    PredictionRegistration,
    Provenance,
    RegisteredScience,
    RenderedFigure,
    RequirementQuote,
    SpendSummary,
    SupportQuote,
    TableRow,
    VerifySummary,
    check_statistician_assessment,
    figure_set,
    own_predictions,
    render_markdown,
    to_json,
)
from ..runtime.planning_store import PlanningIntegrityError, PlanningStore
from ..runtime.verification_store import (
    FileVerificationStore,
    ScientificAdmissibility,
    VerificationIntegrityError,
)
from .chain import Stores
from .events import StageLog
from .investigation import Investigation, InvestigationStore
from .stage import Fact

_CONTROL = "control"
_FIGURES = "figures"
_PROGRAM = "program"
_VERIFICATIONS = "verifications"
_PLANNING = "planning"


@dataclass(frozen=True, slots=True)
class RunReading:
    """Everything a publication export reads from one verified run —
    the shared first half of the packet build, extracted so the figures
    verb reads the record exactly the way the packet does."""

    investigations: InvestigationStore
    investigation: Investigation
    log: StageLog
    run_id: str
    head_state_id: str
    report: IntegrityReport
    stores: Stores
    envelope: ResearchRun
    head: ResearchState
    admission: AdmissionRecord
    verifications: FileVerificationStore
    admissible: ScientificAdmissibility
    seed_of: dict[str, int | None]


def read_run_checked(
    root: Path, investigation_id: str | None = None
) -> RunReading:
    """:func:`read_run` with the analysis-chain integrity refusals
    wrapped the way :func:`build_packet` wraps them."""
    try:
        return read_run(root, investigation_id)
    except _INTEGRITY_ERRORS as error:
        raise PacketError(
            f"a record under {root} does not survive its own digest: "
            f"{error}"
        ) from error


def build_packet(
    root: Path, investigation_id: str | None = None
) -> EvidencePacket:
    """The evidence packet for one investigation under ``root``.

    Raises :class:`MissingFactError` when the walk never reached a
    research state (a refusal), and :class:`PacketError` for everything
    that makes an honest packet impossible: a run that does not verify,
    a record that does not load, figures that do not re-derive.
    """
    try:
        return _build(root, investigation_id)
    except _INTEGRITY_ERRORS as error:
        # The analysis-chain stores are outside ``verify_run``'s scope;
        # their loaders re-derive every content id and refuse a doctored
        # record themselves. That refusal is this export's failure.
        raise PacketError(
            f"a record under {root} does not survive its own digest: "
            f"{error}"
        ) from error


_INTEGRITY_ERRORS = (
    AdmissionIntegrityError,
    ArtifactIntegrityError,
    EvidenceIntegrityError,
    IdeationIntegrityError,
    LiteratureIntegrityError,
    PlanningIntegrityError,
    FigureError,
    SnapshotError,
    VerificationIntegrityError,
)


def read_run(root: Path, investigation_id: str | None = None) -> RunReading:
    investigations = InvestigationStore(root / _CONTROL)
    investigation = _resolved(investigations, investigation_id)
    log = investigations.log_for(investigation.investigation_id)
    facts = log.facts()
    head_state_id = facts.require(Fact.STATE_ID)
    run_id = facts.require(Fact.RUN_ID)

    report = verify_run(root, program_root=root / _PROGRAM)
    if not report.ok:
        listed = "; ".join(
            f"{issue.kind}: {issue.subject_id}: {issue.detail}"
            for issue in report.issues[:5]
        )
        raise PacketError(
            f"the run under {root} does not verify from cold "
            f"({len(report.issues)} issue(s): {listed}); a packet is not "
            f"exported from records that do not survive their own digests"
        )

    stores = Stores.under(root)
    envelope = next(
        (found for found in stores.program.runs() if found.run_id == run_id),
        None,
    )
    if envelope is None:
        raise PacketError(
            f"the stage log names run {run_id}, which the program store "
            f"under {root} does not hold"
        )
    head = stores.states.load(head_state_id)
    admission = stores.admission.get_record(envelope.admission_record_id)
    if admission is None:
        raise PacketError(
            f"admission record {envelope.admission_record_id} is not in "
            f"the admission store"
        )

    verifications = FileVerificationStore(root / _VERIFICATIONS)
    return RunReading(
        investigations=investigations,
        investigation=investigation,
        log=log,
        run_id=run_id,
        head_state_id=head_state_id,
        report=report,
        stores=stores,
        envelope=envelope,
        head=head,
        admission=admission,
        verifications=verifications,
        admissible=ScientificAdmissibility(verifications),
        seed_of={
            result.id: result.seed for result in stores.evidence.results()
        },
    )


def _build(root: Path, investigation_id: str | None) -> EvidencePacket:
    reading = read_run(root, investigation_id)
    investigations = reading.investigations
    investigation = reading.investigation
    run_id = reading.run_id
    head_state_id = reading.head_state_id
    report = reading.report
    stores = reading.stores
    envelope = reading.envelope
    head = reading.head
    admission = reading.admission
    verifications = reading.verifications
    admissible = reading.admissible
    seed_of = reading.seed_of
    spend = reading.log.spend()
    balance = stores.program.ledger_for(run_id).balance()

    figures = _rendered_figures(root, reading)

    question = head.question(envelope.question_id)
    hypothesis = head.hypothesis(envelope.hypothesis_id)
    config = investigations.get_config(investigation.config_id) or {}
    brief = config.get("brief")
    brief_topic = (
        str(brief.get("topic", "")) if isinstance(brief, dict) else ""
    )

    return EvidencePacket(
        provenance=Provenance(
            investigation_id=investigation.investigation_id,
            label=investigation.label,
            config_id=investigation.config_id,
            brief_topic=brief_topic,
            run_id=run_id,
            run_record_id=envelope.id,
            authority=envelope.authority,
            head_state_id=head_state_id,
            spend=SpendSummary(
                granted_wall_clock_seconds=envelope.granted.wall_clock_seconds,
                granted_gpu_hours=envelope.granted.gpu_hours,
                granted_usd=envelope.granted.usd,
                granted_model_tokens=envelope.granted.model_tokens,
                remaining_wall_clock_seconds=balance.wall_clock_seconds,
                remaining_gpu_hours=balance.gpu_hours,
                remaining_usd=balance.usd,
                remaining_model_tokens=balance.model_tokens,
                stage_model_calls=spend.model_calls,
                stage_input_tokens=spend.input_tokens,
                stage_output_tokens=spend.output_tokens,
            ),
            verified=VerifySummary(
                states_checked=report.states_checked,
                results_checked=report.results_checked,
                evidence_checked=report.evidence_checked,
                blobs_checked=report.blobs_checked,
            ),
        ),
        science=RegisteredScience(
            admission_record_id=admission.id,
            question_id=envelope.question_id,
            question=question.text if question is not None else "",
            hypothesis_id=envelope.hypothesis_id,
            hypothesis=(
                hypothesis.statement if hypothesis is not None else ""
            ),
            mechanical_reading=admission.mechanical_reading,
            admitted_predictions=tuple(
                AdmittedPrediction(
                    prediction_text=entry.prediction_text,
                    condition=entry.condition,
                    base_metric=entry.base_metric,
                    expected_higher_arm=entry.expected_higher_arm,
                    expected_lower_arm=entry.expected_lower_arm,
                    contrary_observation=entry.contrary_observation,
                    supports=tuple(
                        SupportQuote(
                            source=str(support.source),
                            field_path=support.field_path,
                            quote=support.quote,
                        )
                        for support in entry.support
                    ),
                )
                for entry in admission.operational_predictions
            ),
            requirements=tuple(
                RequirementQuote(
                    source=str(requirement.source),
                    record_id=requirement.record_id,
                    field_path=requirement.field_path,
                    quote=requirement.quote,
                )
                for requirement in (
                    *admission.inherited_requirements,
                    *admission.operator_requirements,
                )
            ),
        ),
        claims=tuple(
            _claim_finding(claim, head, stores, verifications, admissible, seed_of)
            for claim in head.claims
        ),
        planner_decisions=_planner_decisions(root),
        bibliography=_bibliography(stores, admission.selected_candidate_id),
        tables=_tables(stores, verifications),
        figures=figures,
    )


def _rendered_figures(
    root: Path, reading: RunReading
) -> tuple[RenderedFigure, ...]:
    """The figure store's manifests, held to the record they claim to
    draw. A missing figure is honest absence — the operator never ran
    ``arl figures``; an unexpected or altered one is drift or tampering
    and refuses loudly."""
    store = FigureStore(root / _FIGURES)
    expected = planned_figures(
        reading.head,
        reading.stores.evidence,
        admissible=reading.admissible,
        seed_of=reading.seed_of,
    )
    by_id = {figure_id_for(data): data for data in expected}
    for manifest in store.manifests():
        planned = by_id.get(manifest.figure_id)
        if planned is None or planned != manifest.data:
            raise StaleFigureError(
                f"figure {manifest.figure_id} (claim "
                f"{manifest.data.claim_id}) is recorded, but the record "
                f"no longer derives it; a stale figure does not enter "
                f"the packet silently"
            )
        problems = store.verify(manifest.figure_id)
        if problems:
            raise PacketError(
                f"figure {manifest.figure_id} does not survive its "
                f"digests: " + "; ".join(problems)
            )
    mirrors = []
    for data in expected:
        found = store.get(figure_id_for(data))
        if found is None:
            continue
        mirrors.append(
            RenderedFigure(
                figure_id=found.figure_id,
                claim_id=data.claim_id,
                prediction_id=data.prediction_id,
                metric=data.metric,
                comparator=data.comparator,
                threshold=data.threshold,
                points=data.points,
                n=data.n,
                mean=data.mean,
                stdev=data.stdev,
                caption=data.caption,
                renderer=found.renderer,
                files=found.files,
            )
        )
    return tuple(mirrors)


def write_packet(packet: EvidencePacket, out_dir: Path) -> tuple[Path, Path]:
    """Write the packet beside its run, write-once by content id.

    Re-exporting an unchanged run reproduces the same bytes under the
    same name and is a no-op; a different packet gets a different name.
    Nothing here is ever rewritten.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    as_json = to_json(packet)
    as_markdown = render_markdown(packet)
    json_path = out_dir / f"{packet.packet_id}.json"
    markdown_path = out_dir / f"{packet.packet_id}.md"
    for path, content in (
        (json_path, as_json),
        (markdown_path, as_markdown),
    ):
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise PacketError(
                    f"{path} exists with different content; packet files "
                    f"are never rewritten"
                )
            continue
        path.write_text(content, encoding="utf-8")
    return json_path, markdown_path


def _resolved(
    investigations: InvestigationStore, investigation_id: str | None
) -> Investigation:
    known = investigations.investigations()
    if investigation_id is not None:
        found = investigations.get(investigation_id)
        if found is None:
            raise PacketError(
                f"investigation {investigation_id} is not under this root"
            )
        return found
    if not known:
        raise PacketError("no investigation exists under this root")
    if len(known) > 1:
        listed = ", ".join(found.investigation_id for found in known)
        raise PacketError(
            f"{len(known)} investigations exist under this root ({listed}); "
            f"name the one to export"
        )
    return known[0]


def _claim_finding(
    claim: Claim,
    head: ResearchState,
    stores: Stores,
    verifications: FileVerificationStore,
    admissible: ScientificAdmissibility,
    seed_of: Mapping[str, int | None],
) -> ClaimFinding:
    current = head.current_assessment(claim.id)
    figures: tuple[FigureSet, ...] = ()
    if current is None:
        figures_check = NOT_ASSESSED
        record = None
    else:
        record = AssessmentRecord(
            assessment_id=current.id,
            verdict=str(current.verdict),
            method=current.method,
            rationale=current.rationale,
            scope=current.scope,
            evidence_ids=current.evidence_ids,
            supersedes=current.supersedes,
        )
        if current.method == STATISTICIAN_METHOD:
            assessed = check_statistician_assessment(
                current,
                head,
                stores.evidence,
                admissible=admissible,
                seed_of=seed_of,
            )
            figures = tuple(
                figure_set(verdict, stats) for verdict, stats in assessed
            )
            figures_check = RE_DERIVED
        else:
            figures_check = RESTATED

    own = own_predictions(head, claim, stores.evidence)
    registrations = tuple(
        PredictionRegistration(
            prediction_id=prediction.id,
            spec_id=spec.id,
            metric=prediction.metric,
            comparator=str(prediction.comparator),
            threshold=prediction.threshold,
            tolerance=prediction.tolerance,
            condition=prediction.condition,
        )
        for prediction in own
        for spec in head.experiments_for(prediction.id)
    )
    rows = []
    for link in head.evidence_links:
        if link.claim_id != claim.id:
            continue
        evidence = stores.evidence.get_evidence(link.evidence_id)
        result = stores.evidence.get_result(evidence.result_id)
        verification = verifications.get(evidence.result_id)
        manifest = stores.evidence.artifacts.get(evidence.result_id)
        rows.append(
            EvidenceRow(
                evidence_id=evidence.id,
                result_id=evidence.result_id,
                seed=result.seed,
                relation=str(link.relation),
                standing=(
                    str(verification.standing)
                    if verification is not None
                    else "unrecorded"
                ),
                metrics=tuple(sorted(evidence.metrics.items())),
                artifacts=tuple(
                    ArtifactRef(
                        path=entry.path,
                        digest=entry.digest,
                        size_bytes=entry.size_bytes,
                        media_type=entry.media_type,
                        kind=str(entry.kind),
                    )
                    for entry in (
                        manifest.entries if manifest is not None else ()
                    )
                ),
            )
        )
    return ClaimFinding(
        claim_id=claim.id,
        statement=claim.statement,
        scope=claim.scope,
        figures_check=figures_check,
        assessment=record,
        figures=figures,
        registrations=registrations,
        evidence_rows=tuple(rows),
    )


def _planner_decisions(root: Path) -> tuple[PlannerDecision, ...]:
    plans = PlanningStore(root / _PLANNING)
    return tuple(
        PlannerDecision(
            record_id=record.id,
            action=str(record.action),
            rationale=record.rationale,
            spec_id=record.spec_id,
            stop_reason=(
                str(record.stop_reason)
                if record.stop_reason is not None
                else ""
            ),
        )
        for record in plans.records()
    )


def _bibliography(stores: Stores, candidate_id: str) -> Bibliography:
    idea = stores.ideation.get_idea(candidate_id)
    if idea is None:
        raise PacketError(
            f"candidate {candidate_id} is not in the ideation store"
        )
    entries = []
    for source_id in idea.cited_source_ids:
        source = stores.literature.get_source(source_id)
        if source is None:
            raise PacketError(
                f"cited source {source_id} is not in the literature store"
            )
        entries.append(
            BibliographyEntry(
                source_id=source.id,
                title=source.title or "",
                authors=source.authors,
                venue=source.venue or "",
                year=source.publication_year,
                doi=source.doi or "",
                arxiv_id=source.arxiv_id or "",
                url=source.provider_url,
                access_level=str(source.access_level),
            )
        )
    return Bibliography(
        candidate_id=idea.id,
        candidate_title=idea.title,
        research_question=idea.research_question,
        entries=tuple(entries),
    )


def _tables(
    stores: Stores, verifications: FileVerificationStore
) -> tuple[TableRow, ...]:
    rows = []
    for result in stores.evidence.results():
        verification = verifications.get(result.id)
        rows.append(
            TableRow(
                spec_id=result.spec_id,
                seed=result.seed,
                result_id=result.id,
                standing=(
                    str(verification.standing)
                    if verification is not None
                    else "unrecorded"
                ),
                metrics=tuple(sorted(result.metrics.items())),
            )
        )
    rows.sort(key=lambda row: (row.spec_id, row.seed is None, row.seed or 0))
    return tuple(rows)
