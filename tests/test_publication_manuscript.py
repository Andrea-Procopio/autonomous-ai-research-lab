"""The manuscript: prose gated to the packet, assembly owned by code.

The unit half needs no run root: gates judge tokens against a small
reportable packet, the author is exercised against the scripted fake
provider, and the store holds the write-once discipline. The
integration half walks the CI lab once (module-scoped) and authors a
real manuscript through the scripted lab — then proves the replay path
never calls a model again.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from autonomous_research_lab.control.cli import OK, REFUSED, main
from autonomous_research_lab.control.config import ConfigError, parse_config
from autonomous_research_lab.control.lab import Lab
from autonomous_research_lab.control.manuscript import author_manuscript
from autonomous_research_lab.control.stage import StageName
from autonomous_research_lab.publication.author import (
    AUTHOR_INSTRUCTION,
    PROSE_SCHEMA,
    ManuscriptAuthor,
)
from autonomous_research_lab.publication.manuscript import (
    PROSE_SECTIONS,
    AuthorCall,
    Manuscript,
    ManuscriptError,
    ManuscriptRejectedError,
    NothingToReportError,
    ProseRejection,
    ProseSections,
    assemble,
    gate_prose,
    known_renderings,
    require_reportable,
)
from autonomous_research_lab.publication.packet import (
    Bibliography,
    BibliographyEntry,
    ClaimFinding,
    EvidencePacket,
    EvidenceRow,
    Provenance,
    RegisteredScience,
    SpendSummary,
    VerifySummary,
    finding_lines,
    reference_lines,
)
from autonomous_research_lab.publication.store import (
    ManuscriptConflictError,
    ManuscriptIntegrityError,
    ManuscriptStore,
)
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    StructuredOutputError,
    UsageLedger,
)
from examples.vision_chain import walk
from examples.vision_lab import ci_lab

SOURCE_ID = "lits_53b0e6573b48e679"


def minimal_packet(**overrides: object) -> EvidencePacket:
    """A claimless packet — the same shape test_publication_packet
    builds, restated here because test modules are not a package."""
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


def reportable_packet() -> EvidencePacket:
    return minimal_packet(
        claims=(
            ClaimFinding(
                claim_id="clm_1",
                statement="it holds",
                scope="held out",
                figures_check="restated-from-record",
                evidence_rows=(
                    EvidenceRow(
                        evidence_id="ev_1",
                        result_id="res_1",
                        seed=11,
                        relation="supports",
                        standing="verified_evidence",
                        metrics=(("accuracy gap", 0.85),),
                    ),
                ),
            ),
        ),
        bibliography=Bibliography(
            candidate_id="idea_1",
            candidate_title="Trained versus random",
            research_question="does it hold?",
            entries=(
                BibliographyEntry(
                    source_id=SOURCE_ID,
                    title="A cited paper",
                    authors=("Ada Lovelace",),
                    venue="A venue",
                    year=2016,
                    doi="",
                    arxiv_id="",
                    url="https://example.invalid/1",
                    access_level="abstract",
                ),
            ),
        ),
    )


def prose(**overrides: str) -> dict[str, str]:
    sections = {name: "Numberless honest prose." for name in PROSE_SECTIONS}
    sections.update(overrides)
    return sections


def rejections(
    sections: dict[str, str],
) -> tuple[ProseRejection, ...]:
    packet = reportable_packet()
    return gate_prose(
        sections,
        allowed=known_renderings(packet),
        bibliography=packet.bibliography,
    )


class TestNumberGate:
    def test_a_number_the_packet_states_passes(self) -> None:
        assert rejections(prose(abstract="the gap was 0.85 held out")) == ()

    def test_an_invented_number_is_refused_naming_the_token(self) -> None:
        found = rejections(prose(discussion="we reached 99.9 accuracy"))
        assert [r.rule for r in found] == ["unknown_number"]
        assert "discussion: 99.9" in found[0].detail

    def test_a_rounded_variant_is_refused(self) -> None:
        found = rejections(prose(abstract="roughly 0.9 gap"))
        assert [r.rule for r in found] == ["unknown_number"]

    def test_a_recomputed_percentage_is_refused(self) -> None:
        found = rejections(prose(abstract="an 85% gap"))
        assert [r.rule for r in found] == ["unknown_number"]
        assert "85%" in found[0].detail

    def test_a_dropped_sign_changes_the_token(self) -> None:
        packet = reportable_packet()
        allowed = frozenset(["-0.5"]) | known_renderings(packet)
        found = gate_prose(
            prose(abstract="an effect of 0.5"),
            allowed=allowed,
            bibliography=packet.bibliography,
        )
        assert [r.rule for r in found] == ["unknown_number"]

    def test_digits_inside_a_cited_id_do_not_tokenize(self) -> None:
        assert rejections(prose(introduction=f"as shown in [{SOURCE_ID}]")) == ()

    def test_both_markdown_and_json_renderings_are_known(self) -> None:
        allowed = known_renderings(reportable_packet())
        assert "600s" in allowed  # the markdown :g rendering, unit glued
        assert "600.0" in allowed  # the JSON rendering
        assert "2016" in allowed  # the bibliography year
        assert "600" not in allowed  # a word the packet never printed

    def test_an_obfuscated_number_is_refused(self) -> None:
        found = rejections(prose(discussion="a 3x speedup"))
        assert [r.rule for r in found] == ["unknown_number"]
        assert "3x" in found[0].detail

    def test_a_quoted_record_id_passes_whole(self) -> None:
        assert rejections(
            prose(discussion="result res_1 at seed 11 supports it")
        ) == ()


class TestCitationGate:
    def test_a_bibliography_source_passes(self) -> None:
        assert rejections(prose(introduction=f"see [{SOURCE_ID}]")) == ()

    def test_a_well_formed_unknown_id_is_refused(self) -> None:
        found = rejections(
            prose(introduction="see [lits_0000000000000000]")
        )
        assert [r.rule for r in found] == ["unknown_citation"]

    def test_a_prose_citation_is_malformed(self) -> None:
        found = rejections(prose(introduction="see [Smith 2020]"))
        assert "malformed_citation" in [r.rule for r in found]

    def test_a_markdown_link_is_malformed(self) -> None:
        found = rejections(
            prose(introduction="see [the paper](https://example.invalid)")
        )
        assert "malformed_citation" in [r.rule for r in found]

    def test_any_bracket_with_an_empty_bibliography_refuses(self) -> None:
        packet = reportable_packet()
        found = gate_prose(
            prose(introduction=f"see [{SOURCE_ID}]"),
            allowed=known_renderings(packet),
            bibliography=None,
        )
        assert [r.rule for r in found] == ["unknown_citation"]


class TestStructureGate:
    def test_a_heading_line_is_refused(self) -> None:
        found = rejections(prose(discussion="fine\n  ## My Own Section"))
        assert [r.rule for r in found] == ["structural_markup"]

    def test_a_blank_section_is_refused(self) -> None:
        found = rejections(prose(limitations="  \n "))
        assert [r.rule for r in found] == ["empty_section"]


def manuscript_for(packet: EvidencePacket) -> Manuscript:
    return Manuscript(
        packet_id=packet.packet_id,
        sections=ProseSections(**prose()),
        call=AuthorCall(
            request_fingerprint="mreq_1",
            response_id="resp_1",
            provider="fake",
            requested_model="fake-model-1",
            served_model="fake-model-1",
            provider_request_id=None,
            latency_seconds=0.25,
            input_tokens=100,
            output_tokens=50,
            repair_count=0,
        ),
    )


class TestManuscriptRecord:
    def test_the_id_derives_from_the_content(self) -> None:
        packet = reportable_packet()
        first = manuscript_for(packet)
        assert first.manuscript_id.startswith("mscr_")
        assert manuscript_for(packet).manuscript_id == first.manuscript_id

    def test_a_doctored_id_is_refused(self) -> None:
        with pytest.raises(ManuscriptError, match="does not survive"):
            replace(
                manuscript_for(reportable_packet()),
                manuscript_id="mscr_0000000000000000",
            )

    def test_a_packet_without_claims_is_a_refusal(self) -> None:
        with pytest.raises(NothingToReportError, match="no claims"):
            require_reportable(minimal_packet())


class TestAssembly:
    def test_assembly_is_deterministic_with_prose_in_its_slots(self) -> None:
        packet = reportable_packet()
        manuscript = replace(
            manuscript_for(packet),
            sections=ProseSections(**prose(abstract="THE ABSTRACT PROSE")),
            manuscript_id="",
        )
        document = assemble(packet, manuscript)
        assert document == assemble(packet, manuscript)
        assert "THE ABSTRACT PROSE" in document
        assert document.startswith("# Trained versus random\n")
        # The load-bearing sections are the packet's own renderers,
        # byte for byte.
        for line in finding_lines(packet.claims[0]):
            assert line in document
        assert packet.bibliography is not None
        for line in reference_lines(packet.bibliography):
            assert line in document

    def test_a_packet_id_mismatch_is_refused(self) -> None:
        packet = reportable_packet()
        other = minimal_packet()
        manuscript = manuscript_for(packet)
        with pytest.raises(ManuscriptError, match="was authored from"):
            assemble(other, manuscript)


class TestStore:
    def test_record_and_get_round_trip(self, tmp_path: Path) -> None:
        store = ManuscriptStore(tmp_path)
        manuscript = manuscript_for(reportable_packet())
        store.record(manuscript)
        assert store.get(manuscript.manuscript_id) == manuscript
        assert store.record(manuscript) == manuscript  # no-op

    def test_different_content_under_the_same_name_refuses(
        self, tmp_path: Path
    ) -> None:
        store = ManuscriptStore(tmp_path)
        manuscript = manuscript_for(reportable_packet())
        store.record(manuscript)
        path = tmp_path / f"{manuscript.manuscript_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["sections"]["abstract"] = "a nicer story"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ManuscriptIntegrityError):
            store.get(manuscript.manuscript_id)
        with pytest.raises((ManuscriptConflictError, ManuscriptIntegrityError)):
            store.record(manuscript_for(reportable_packet()))

    def test_for_packet_finds_the_replay(self, tmp_path: Path) -> None:
        store = ManuscriptStore(tmp_path)
        packet = reportable_packet()
        manuscript = store.record(manuscript_for(packet))
        assert store.for_packet(packet.packet_id) == (manuscript,)
        assert store.for_packet("epkt_0000000000000000") == ()

    def test_rejected_payloads_are_preserved(self, tmp_path: Path) -> None:
        store = ManuscriptStore(tmp_path)
        store.preserve_rejected(
            packet_id="epkt_1",
            reasons=(("unknown_number", "abstract: 99.9"),),
            request_fingerprint="mreq_1",
            response_id="resp_1",
            payload={"abstract": "99.9"},
            repair=0,
        )
        preserved = store.rejected()
        assert len(preserved) == 1
        assert preserved[0]["reasons"] == [["unknown_number", "abstract: 99.9"]]


Built = tuple[ManuscriptAuthor, FakeModelProvider, ManuscriptStore]


def author_with(
    replies: list[str], *, max_corrective_calls: int = 1
) -> Callable[[Path], Built]:
    def build(tmp_path: Path) -> Built:
        provider = FakeModelProvider(replies)
        store = ManuscriptStore(tmp_path / "manuscript")
        return (
            ManuscriptAuthor(
                provider=provider,
                model="fake-model-1",
                ledger=UsageLedger(),
                store=store,
                max_corrective_calls=max_corrective_calls,
            ),
            provider,
            store,
        )

    return build


GOOD_REPLY = json.dumps(prose(abstract="the gap was 0.85"))
BAD_REPLY = json.dumps(prose(abstract="we reached 99.9"))


class TestAuthor:
    def test_a_clean_draft_passes_in_one_call(self, tmp_path: Path) -> None:
        author, provider, store = author_with([GOOD_REPLY])(tmp_path)
        manuscript = author.author(reportable_packet())

        assert len(provider.calls) == 1
        assert manuscript.call.repair_count == 0
        assert manuscript.call.provider == "fake"
        assert manuscript.sections.abstract == "the gap was 0.85"
        assert store.rejected() == ()
        # The instruction and the packet went with the request.
        request = provider.calls[0]
        assert request.instruction == AUTHOR_INSTRUCTION
        assert request.schema is PROSE_SCHEMA

    def test_a_gated_rejection_earns_one_corrective_call(
        self, tmp_path: Path
    ) -> None:
        author, provider, store = author_with([BAD_REPLY, GOOD_REPLY])(tmp_path)
        manuscript = author.author(reportable_packet())

        assert len(provider.calls) == 2
        assert manuscript.call.repair_count == 1
        preserved = store.rejected()
        assert len(preserved) == 1
        corrective = provider.calls[1].messages[-1].content
        assert "unknown_number" in corrective
        assert "99.9" in corrective

    def test_exhausted_correctives_are_a_typed_refusal(
        self, tmp_path: Path
    ) -> None:
        author, provider, store = author_with([BAD_REPLY, BAD_REPLY])(tmp_path)
        with pytest.raises(ManuscriptRejectedError, match="unknown_number"):
            author.author(reportable_packet())
        assert len(provider.calls) == 2
        assert len(store.rejected()) == 2

    def test_a_non_json_reply_is_corrected_the_same_way(
        self, tmp_path: Path
    ) -> None:
        author, _provider, store = author_with(
            ["not json at all", GOOD_REPLY]
        )(tmp_path)
        manuscript = author.author(reportable_packet())
        assert manuscript.call.repair_count == 1
        assert len(store.rejected()) == 1

    def test_two_non_json_replies_reraise_the_schema_error(
        self, tmp_path: Path
    ) -> None:
        author, provider, _store = author_with(["nope", "still nope"])(tmp_path)
        with pytest.raises(StructuredOutputError):
            author.author(reportable_packet())
        assert len(provider.calls) == 2

    def test_the_author_refuses_an_unreportable_packet_before_spending(
        self, tmp_path: Path
    ) -> None:
        author, provider, _ = author_with([GOOD_REPLY])(tmp_path)
        with pytest.raises(NothingToReportError):
            author.author(minimal_packet())
        assert provider.calls == ()


# -- integration: one CI walk, authored ---------------------------------------


@pytest.fixture(scope="module")
def walked(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("manuscript") / "run"
    result = walk(root, lab=ci_lab())
    assert result.ok, result.detail
    return root


class _PoisonedLab:
    """A lab whose model seat must never be taken: the replay path is
    only honest if it reaches no provider at all."""

    def model_provider(self, stage: StageName, /) -> NoReturn:
        raise AssertionError(f"the replay called the model for {stage}")

    def literature_provider(self) -> NoReturn:
        raise AssertionError("the replay asked for literature")

    def runtime(self, request: object, /) -> NoReturn:
        raise AssertionError("the replay asked for a runtime")


class TestAuthorManuscript:
    def test_the_scripted_lab_authors_a_gated_draft(
        self, walked: Path
    ) -> None:
        result = author_manuscript(walked, lab=ci_lab())

        assert not result.replayed
        assert result.manuscript.call.repair_count == 0
        assert result.json_path.is_file()
        assert result.markdown_path.is_file()
        document = result.markdown_path.read_text(encoding="utf-8")
        assert "## Results" in document
        assert "## References" in document
        # ...and the export changes nothing the verifier checks.
        assert main(["verify", "--root", str(walked)]) == OK

    def test_a_second_run_replays_without_a_model(self, walked: Path) -> None:
        author_manuscript(walked, lab=ci_lab())
        before = None
        first = author_manuscript(walked, lab=_PoisonedLab())
        assert first.replayed
        before = first.markdown_path.read_bytes()
        again = author_manuscript(walked, lab=_PoisonedLab())
        assert again.replayed
        assert again.markdown_path.read_bytes() == before

    def test_the_cli_authors_through_a_named_lab(
        self, walked: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "manuscript",
                "--root",
                str(walked),
                "--lab",
                "examples.vision_lab:ci_lab",
            ]
        )
        printed = capsys.readouterr().out
        assert code == OK
        assert "manuscript    mscr_" in printed
        assert "replayed" in printed

    def test_a_model_override_is_recorded_not_silent(
        self, walked: Path, tmp_path: Path
    ) -> None:
        result = author_manuscript(
            walked,
            lab=ci_lab(),
            out_dir=tmp_path / "overridden",
            model="another-model-1",
        )
        assert result.manuscript.call.requested_model == "another-model-1"

    def test_a_walk_with_nothing_to_report_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab(), stop_after=StageName.FUNDING)
        assert result.ok

        code = main(
            [
                "manuscript",
                "--root",
                str(root),
                "--lab",
                "examples.vision_lab:ci_lab",
            ]
        )
        assert code == REFUSED
        assert "REFUSED" in capsys.readouterr().out


class TestStopAfterStaysAChainConcern:
    def test_the_manuscript_seat_is_not_a_stop_after_choice(self) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "run",
                    "nowhere.json",
                    "--root",
                    "/nowhere",
                    "--stop-after",
                    "manuscript",
                ]
            )

    def test_a_config_naming_the_seat_is_refused(self) -> None:
        payload = json.loads(
            Path("examples/vision.json").read_text(encoding="utf-8")
        )
        payload["stop_after"] = "manuscript"
        with pytest.raises(ConfigError, match="no stage of the chain"):
            parse_config(payload)


def _lab_protocol_check() -> Lab:
    """The poisoned lab satisfies the Lab protocol structurally."""
    return _PoisonedLab()
