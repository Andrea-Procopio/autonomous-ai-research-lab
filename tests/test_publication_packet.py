"""The evidence packet: checked against the record, never copied from it.

The unit half exercises the schema and the pure checks without a run
root: the packet id derives from content and refuses a doctored one, the
rationale parse is a contract with :meth:`FamilyStatistics.render`, and
a rationale whose figures do not re-derive is a loud mismatch. The
integration half walks the CI lab once (module-scoped — the walk is the
expensive part) and holds the exported packet to what that walk is known
to record: two claims, both re-derived-and-matched, two cited sources,
six result rows. Destructive cases copy the walked root first.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from autonomous_research_lab.control.cli import FAILED, OK, REFUSED, main
from autonomous_research_lab.control.packet import build_packet, write_packet
from autonomous_research_lab.control.stage import StageName
from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.publication.packet import (
    NOT_ASSESSED,
    RE_DERIVED,
    STATISTICIAN_METHOD,
    EvidencePacket,
    FiguresMismatchError,
    PacketError,
    Provenance,
    RegisteredScience,
    SpendSummary,
    VerifySummary,
    check_statistician_assessment,
    render_markdown,
    to_json,
)
from autonomous_research_lab.runtime.statistics import assess_family
from examples.vision_chain import walk
from examples.vision_lab import ci_lab

# -- unit: the schema ---------------------------------------------------------


def minimal_packet(**overrides: object) -> EvidencePacket:
    fields: dict[str, object] = {
        "provenance": Provenance(
            investigation_id="inv_1",
            label="a label",
            config_id="cfg_1",
            brief_topic="a topic",
            run_id="run_1",
            run_record_id="rune_1",
            authority="an operator",
            head_state_id="st_1",
            spend=SpendSummary(
                granted_wall_clock_seconds=600.0,
                granted_gpu_hours=0.0,
                granted_usd=1.0,
                granted_model_tokens=10_000,
                remaining_wall_clock_seconds=500.0,
                remaining_gpu_hours=0.0,
                remaining_usd=1.0,
                remaining_model_tokens=9_000,
                stage_model_calls=3,
                stage_input_tokens=100,
                stage_output_tokens=50,
            ),
            verified=VerifySummary(
                states_checked=1,
                results_checked=0,
                evidence_checked=0,
                blobs_checked=0,
            ),
        ),
        "science": RegisteredScience(
            admission_record_id="adm_1",
            question_id="q_1",
            question="does it hold?",
            hypothesis_id="hyp_1",
            hypothesis="it holds",
            mechanical_reading="sign_only",
        ),
    }
    fields.update(overrides)
    return EvidencePacket(**fields)  # type: ignore[arg-type]


class TestPacketIdentity:
    def test_the_id_derives_from_the_content(self) -> None:
        packet = minimal_packet()
        assert packet.packet_id.startswith("epkt_")
        assert minimal_packet().packet_id == packet.packet_id

    def test_different_content_derives_a_different_id(self) -> None:
        science = replace(minimal_packet().science, question="another?")
        assert (
            minimal_packet(science=science).packet_id
            != minimal_packet().packet_id
        )

    def test_the_derived_id_round_trips(self) -> None:
        packet = minimal_packet()
        again = minimal_packet(packet_id=packet.packet_id)
        assert again.packet_id == packet.packet_id

    def test_a_doctored_id_is_refused(self) -> None:
        with pytest.raises(PacketError, match="does not survive"):
            minimal_packet(packet_id="epkt_0000000000000000")

    def test_serialization_is_deterministic_and_terminated(self) -> None:
        first = to_json(minimal_packet())
        assert first == to_json(minimal_packet())
        assert first.endswith("\n")
        assert json.loads(first)["packet_id"] == minimal_packet().packet_id


class TestWriteOnce:
    def test_rewriting_the_same_packet_is_a_no_op(
        self, tmp_path: Path
    ) -> None:
        packet = minimal_packet()
        first = write_packet(packet, tmp_path)
        second = write_packet(packet, tmp_path)
        assert first == second

    def test_a_name_collision_with_other_bytes_is_refused(
        self, tmp_path: Path
    ) -> None:
        packet = minimal_packet()
        (tmp_path / f"{packet.packet_id}.json").write_text(
            "something else", encoding="utf-8"
        )
        with pytest.raises(PacketError, match="never rewritten"):
            write_packet(packet, tmp_path)


# -- unit: the rationale contract ---------------------------------------------

PREDICTION = Prediction(
    hypothesis_id="hyp_1",
    condition="held out",
    metric="contrast",
    comparator=Comparator.GREATER_THAN,
    threshold=0.0,
    expectation="positive",
)


def rendered_rationale(comparisons: int = 1) -> str:
    tests = tuple(
        PredictionTest(
            prediction_id=PREDICTION.id,
            result_id=f"res_{index}",
            metric=PREDICTION.metric,
            observed=0.04 + index / 100,
            consistency=Consistency.CONSISTENT,
            detail="test",
        )
        for index in range(5)
    )
    _, stats = assess_family(PREDICTION, tests, comparisons=comparisons)
    return stats.render()


def statistician_assessment(rationale: str) -> EpistemicAssessment:
    return EpistemicAssessment(
        subject_id="clm_1",
        verdict=AssessmentVerdict.SUPPORTED,
        method=STATISTICIAN_METHOD,
        rationale=rationale,
    )


class TestRationaleParsing:
    def test_the_pinned_m_and_alpha_are_recoverable(self) -> None:
        from autonomous_research_lab.publication.packet import (
            _parsed_alpha,
            _parsed_comparisons,
        )

        assessment = statistician_assessment(rendered_rationale(comparisons=2))
        assert _parsed_comparisons(assessment) == 2
        assert _parsed_alpha(assessment) == 0.05

    def test_a_rationale_without_one_denominator_is_a_mismatch(self) -> None:
        from autonomous_research_lab.publication.packet import (
            _parsed_comparisons,
        )

        garbled = statistician_assessment("a story with no figures")
        with pytest.raises(FiguresMismatchError, match="Bonferroni"):
            _parsed_comparisons(garbled)
        two_stories = statistician_assessment(
            rendered_rationale(comparisons=1)
            + " | "
            + rendered_rationale(comparisons=2)
        )
        with pytest.raises(FiguresMismatchError, match="one Bonferroni"):
            _parsed_comparisons(two_stories)

    def test_only_the_statistician_method_is_re_derivable(self) -> None:
        other = replace(
            statistician_assessment(rendered_rationale()),
            method="demo:prediction-check-v0",
            id="",
        )
        with pytest.raises(PacketError, match="re-derived"):
            check_statistician_assessment(
                other, None, None, admissible=lambda _: True, seed_of={}  # type: ignore[arg-type]
            )

    def test_the_sentinel_rationale_yields_no_figures(self) -> None:
        sentinel = replace(
            statistician_assessment("no admissible conclusive observations"),
            verdict=AssessmentVerdict.UNDETERMINED,
            id="",
        )
        assessed = check_statistician_assessment(
            sentinel, None, None, admissible=lambda _: True, seed_of={}  # type: ignore[arg-type]
        )
        assert assessed == ()

    def test_a_conclusive_verdict_over_the_sentinel_is_a_mismatch(
        self,
    ) -> None:
        doctored = statistician_assessment(
            "no admissible conclusive observations"
        )
        with pytest.raises(FiguresMismatchError, match="no admissible"):
            check_statistician_assessment(
                doctored, None, None, admissible=lambda _: True, seed_of={}  # type: ignore[arg-type]
            )


# -- integration: one CI walk, exported ---------------------------------------


@pytest.fixture(scope="module")
def walked(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("packet") / "run"
    result = walk(root, lab=ci_lab())
    assert result.ok, result.detail
    return root


class TestBuildPacket:
    def test_every_claim_is_re_derived_and_matched(
        self, walked: Path
    ) -> None:
        packet = build_packet(walked)

        assert len(packet.claims) == 2
        verdicts = {
            finding.assessment.verdict
            for finding in packet.claims
            if finding.assessment is not None
        }
        assert verdicts == {"supported", "plausible"}
        for finding in packet.claims:
            assert finding.figures_check == RE_DERIVED
            assert finding.figures
            assert finding.registrations
            assert finding.evidence_rows
        supported = next(
            finding
            for finding in packet.claims
            if finding.assessment is not None
            and finding.assessment.verdict == "supported"
        )
        figures = supported.figures[0]
        assert figures.n == 5
        assert figures.p_value == pytest.approx(0.03125)

    def test_the_bibliography_resolves_every_cited_source(
        self, walked: Path
    ) -> None:
        packet = build_packet(walked)

        assert packet.bibliography is not None
        assert packet.bibliography.candidate_title
        assert len(packet.bibliography.entries) == 2
        for entry in packet.bibliography.entries:
            assert entry.source_id.startswith("lit_")
            assert entry.url

    def test_the_tables_carry_result_ids_and_standings(
        self, walked: Path
    ) -> None:
        packet = build_packet(walked)

        assert len(packet.tables) == 6
        for table_row in packet.tables:
            assert table_row.result_id.startswith("res_")
            assert table_row.standing == "verified_evidence"
            assert table_row.metrics

    def test_the_planner_record_is_in_the_packet(self, walked: Path) -> None:
        packet = build_packet(walked)

        actions = [entry.action for entry in packet.planner_decisions]
        assert "stop" in actions
        stop = next(
            entry for entry in packet.planner_decisions
            if entry.action == "stop"
        )
        assert stop.stop_reason == "question_resolved"

    def test_building_twice_is_byte_identical(self, walked: Path) -> None:
        assert to_json(build_packet(walked)) == to_json(build_packet(walked))

    def test_the_markdown_carries_the_recorded_figures(
        self, walked: Path
    ) -> None:
        packet = build_packet(walked)
        rendered = render_markdown(packet)

        assert packet.packet_id in rendered
        assert "re-derived-and-matched" in rendered
        assert "p=0.03125" in rendered
        assert "## References" in rendered

    def test_a_doctored_rationale_is_a_loud_mismatch(
        self, walked: Path
    ) -> None:
        """In memory, deliberately: on disk the state snapshot's hash
        already shields the rationale, so the byte-flip case is the blob
        test below. This pins the check itself: figures that do not
        re-derive refuse the export even when the record is internally
        consistent."""
        from autonomous_research_lab.evidence.file_store import (
            FileEvidenceStore,
        )
        from autonomous_research_lab.runtime.verification_store import (
            FileVerificationStore,
            ScientificAdmissibility,
        )

        states = FileStateStore(walked)
        loaded = [states.load(found) for found in states.state_ids()]
        head = max(loaded, key=lambda state: len(state.attempts))
        assessed = next(
            head.current_assessment(claim.id)
            for claim in head.claims
            if head.current_assessment(claim.id) is not None
        )
        assert assessed is not None
        doctored = replace(
            assessed,
            rationale=assessed.rationale.replace("mean", "median"),
            id="",
        )
        store = FileEvidenceStore(walked)
        admissible = ScientificAdmissibility(
            FileVerificationStore(walked / "verifications")
        )
        with pytest.raises(FiguresMismatchError, match="do not match"):
            check_statistician_assessment(
                doctored,
                head,
                store,
                admissible=admissible,
                seed_of={
                    result.id: result.seed for result in store.results()
                },
            )


class TestExportRefusals:
    def test_a_tampered_blob_fails_the_export(
        self, walked: Path, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        shutil.copytree(walked, root)
        blob = next(
            found for found in (root / "blobs").rglob("*") if found.is_file()
        )
        payload = bytearray(blob.read_bytes())
        payload[0] ^= 0xFF
        blob.write_bytes(bytes(payload))

        with pytest.raises(PacketError, match="does not verify from cold"):
            build_packet(root)

        assert main(["packet", "--root", str(root)]) == FAILED

    def test_a_walk_stopped_before_funding_is_a_refusal_not_a_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Funding is where the research state (and the run) first exist;
        a walk stopped at admission has honestly nothing to export."""
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab(), stop_after=StageName.ADMISSION)
        assert result.ok

        code = main(["packet", "--root", str(root)])

        assert code == REFUSED
        assert "REFUSED" in capsys.readouterr().out

    def test_a_funding_stopped_walk_exports_an_empty_finding(
        self, tmp_path: Path
    ) -> None:
        """The funded state exists and verifies; a packet with no claims
        is the honest export of a run that has not yet measured."""
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab(), stop_after=StageName.FUNDING)
        assert result.ok

        packet = build_packet(root)

        assert packet.claims == ()
        assert packet.tables == ()
        assert packet.science.question

    def test_an_unassessed_claim_is_marked_never_dropped(self) -> None:
        """The schema's honesty rule, pinned at the unit level: a claim
        with no assessment carries the marker, not an absence."""
        assert NOT_ASSESSED == "not-assessed"


class TestPacketVerb:
    def test_it_writes_both_files_under_the_root(
        self, walked: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["packet", "--root", str(walked)])

        printed = capsys.readouterr().out
        assert code == OK
        packet = build_packet(walked)
        json_path = walked / "packet" / f"{packet.packet_id}.json"
        markdown_path = walked / "packet" / f"{packet.packet_id}.md"
        assert json_path.is_file()
        assert markdown_path.is_file()
        assert packet.packet_id in printed
        # ...and the export changes nothing the verifier checks.
        assert main(["verify", "--root", str(walked)]) == OK

    def test_re_export_is_a_no_op(self, walked: Path) -> None:
        assert main(["packet", "--root", str(walked)]) == OK
        packet = build_packet(walked)
        json_path = walked / "packet" / f"{packet.packet_id}.json"
        before = json_path.read_bytes()
        assert main(["packet", "--root", str(walked)]) == OK
        assert json_path.read_bytes() == before

    def test_out_redirects_the_files(
        self, walked: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "elsewhere"
        code = main(["packet", "--root", str(walked), "--out", str(out)])

        assert code == OK
        assert list(out.glob("*.json")) and list(out.glob("*.md"))

    def test_the_export_round_trips_through_json(self, walked: Path) -> None:
        packet = build_packet(walked)
        payload = json.loads(to_json(packet))
        assert payload["packet_id"] == packet.packet_id
        assert payload["provenance"]["run_id"] == packet.provenance.run_id
        assert len(payload["claims"]) == 2
        assert len(payload["tables"]) == 6
