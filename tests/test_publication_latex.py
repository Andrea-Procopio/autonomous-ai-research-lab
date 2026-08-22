"""Venue rendering: the record typeset, nothing invented in transit.

The load-bearing tests are the escape and citation ones — prose is
arbitrary text, and the single-pass escape table plus split-first
citation handling is the whole safety argument. The rest pins the venue
config discipline, the kit store's staging, the number invariant (the
``.tex`` prints exactly the ``:g`` strings the packet prints), and the
review gate: an unapproved draft is never rendered for submission.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

from autonomous_research_lab.control.cli import FAILED, OK, REFUSED, main
from autonomous_research_lab.control.manuscript import author_manuscript
from autonomous_research_lab.control.render import (
    NotApprovedError,
    RenderError,
    latex_commands,
    render_submission,
)
from autonomous_research_lab.control.review import review_manuscript
from autonomous_research_lab.control.stage import StageName
from autonomous_research_lab.publication.kits import (
    KitConflictError,
    KitIntegrityError,
    KitManifest,
    KitStore,
)
from autonomous_research_lab.publication.latex import (
    VENUES,
    VenueError,
    VenueSpec,
    bibtex_entries,
    escape,
    prose_to_latex,
    render_latex,
    venue_from,
)
from autonomous_research_lab.publication.manuscript import (
    PROSE_SECTIONS,
    AuthorCall,
    Manuscript,
    ManuscriptError,
    ProseSections,
)
from autonomous_research_lab.publication.packet import (
    AssessmentRecord,
    Bibliography,
    BibliographyEntry,
    ClaimFinding,
    EvidencePacket,
    EvidenceRow,
    Provenance,
    RegisteredScience,
    SpendSummary,
    VerifySummary,
)
from autonomous_research_lab.publication.store import ManuscriptStore
from autonomous_research_lab.runtime.providers import FakeModelProvider
from examples.stage_venue_kit import main as stage_main
from examples.vision_chain import walk
from examples.vision_lab import ci_lab

SOURCE_ID = "lits_53b0e6573b48e679"


def reportable_packet() -> EvidencePacket:
    """The 8C test fixture, restated because test modules are not a
    package."""
    return EvidencePacket(
        provenance=Provenance(
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
        science=RegisteredScience(
            admission_record_id="adm_1",
            question_id="q_1",
            question="does it hold?",
            hypothesis_id="hyp_1",
            hypothesis="it holds",
            mechanical_reading="sign_only",
        ),
        claims=(
            ClaimFinding(
                claim_id="clm_1",
                statement="it holds",
                scope="held out",
                figures_check="restated-from-record",
                assessment=AssessmentRecord(
                    assessment_id="asm_1",
                    verdict="supported",
                    method="statistician:exact-sign-v1",
                    rationale="a recorded rationale",
                    scope="held out",
                ),
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


def manuscript_for(packet: EvidencePacket) -> Manuscript:
    return Manuscript(
        packet_id=packet.packet_id,
        sections=ProseSections(
            **{
                name: "The claim is supported on held out data."
                for name in PROSE_SECTIONS
            }
        ),
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


class _FakeLab:
    def __init__(self, replies: list[str]) -> None:
        self.provider = FakeModelProvider(replies)

    def model_provider(self, stage: StageName, /) -> FakeModelProvider:
        del stage
        return self.provider

    def literature_provider(self) -> NoReturn:
        raise AssertionError("the render path asked for literature")

    def runtime(self, request: object, /) -> NoReturn:
        raise AssertionError("the render path asked for a runtime")


def grounded_reply(root: Path) -> str:
    from autonomous_research_lab.control.packet import build_packet

    packet = build_packet(root)
    store = ManuscriptStore(root / "manuscript")
    manuscript = store.for_packet(packet.packet_id)[0]
    quote = manuscript.sections.abstract.split(".")[0]
    return json.dumps(
        {
            "findings": [
                {
                    "section": "abstract",
                    "issue": "overclaim",
                    "quote": quote,
                    "subject_id": "",
                    "explanation": "claims more than the record states",
                }
            ]
        }
    )


class TestEscape:
    @pytest.mark.parametrize(
        ("raw", "escaped"),
        [
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
            ("\\", r"\textbackslash{}"),
        ],
    )
    def test_each_special_character(self, raw: str, escaped: str) -> None:
        assert escape(raw) == escaped

    def test_a_single_pass_never_re_escapes(self) -> None:
        assert escape(r"\%") == r"\textbackslash{}\%"

    def test_plain_text_and_brackets_pass_through(self) -> None:
        assert escape("plain [text] passes.") == "plain [text] passes."


class TestCitations:
    def test_a_source_id_becomes_citep_with_its_underscore(self) -> None:
        assert (
            prose_to_latex(f"as shown in [{SOURCE_ID}].")
            == rf"as shown in \citep{{{SOURCE_ID}}}."
        )

    def test_segments_around_citations_are_escaped(self) -> None:
        rendered = prose_to_latex(f"a 5% gap [{SOURCE_ID}] & more")
        assert rendered == rf"a 5\% gap \citep{{{SOURCE_ID}}} \& more"

    def test_a_non_id_bracket_is_literal_text_never_a_cite(self) -> None:
        rendered = prose_to_latex("see [Smith 2020]")
        assert rendered == "see [Smith 2020]"
        assert "citep" not in rendered


class TestBibtex:
    def test_the_full_field_mapping(self) -> None:
        rendered = bibtex_entries(reportable_packet())
        assert rendered.startswith(f"@misc{{{SOURCE_ID},")
        assert "  author = {Ada Lovelace}," in rendered
        assert "  title = {{A cited paper}}," in rendered
        assert "  year = {2016}," in rendered
        assert "  howpublished = {A venue}," in rendered
        assert "  url = {https://example.invalid/1}," in rendered
        assert "  note = {access: abstract}," in rendered
        assert "doi" not in rendered  # empty field omitted
        assert "eprint" not in rendered

    def test_bytes_are_deterministic(self) -> None:
        assert bibtex_entries(reportable_packet()) == bibtex_entries(
            reportable_packet()
        )


class TestVenueSpec:
    def test_unknown_keys_refuse(self) -> None:
        with pytest.raises(VenueError, match="unknown venue key"):
            venue_from({"name": "x", "page_limit": 8})

    def test_the_builtin_registry(self) -> None:
        assert VENUES["plain"].kit == ""
        assert VENUES["plain"].style_package == ""
        assert not VENUES["plain"].anonymous
        for name in ("neurips", "icml", "iclr"):
            assert VENUES[name].kit == name
            assert VENUES[name].style_package
            assert VENUES[name].bibliography_style == "plainnat"
            assert VENUES[name].anonymous

    def test_a_style_package_needs_a_kit(self) -> None:
        with pytest.raises(VenueError, match="no kit"):
            VenueSpec(name="x", style_package="neurips_2026").validate()

    def test_an_unsafe_name_refuses(self) -> None:
        with pytest.raises(VenueError, match="plain directory name"):
            VenueSpec(name="../evil").validate()

    def test_json_round_trip_with_defaults(self) -> None:
        spec = venue_from({"name": "workshop", "class_options": ["11pt"]})
        assert spec.documentclass == "article"
        assert spec.class_options == ("11pt",)
        assert not spec.anonymous

    def test_list_fields_refuse_strings(self) -> None:
        with pytest.raises(VenueError, match="list of strings"):
            venue_from({"name": "x", "class_options": "11pt"})


class TestRenderLatex:
    def test_section_order_mirrors_the_assembly(self) -> None:
        packet = reportable_packet()
        tex, bib = render_latex(
            packet, manuscript_for(packet), VENUES["plain"]
        )
        order = [
            "\\begin{abstract}",
            "\\section{Introduction}",
            "\\section{Registered science}",
            "\\section{Method}",
            "\\section{Results}",
            "\\section{Discussion}",
            "\\section{Limitations}",
            "\\bibliography{references}",
            "\\section*{Attribution}",
        ]
        positions = [tex.index(marker) for marker in order]
        assert positions == sorted(positions)
        assert bib.startswith("@misc{")

    def test_the_number_invariant_holds(self) -> None:
        """Every metric prints as its :g string; no reformatted variant
        appears."""
        packet = reportable_packet()
        tex, _ = render_latex(packet, manuscript_for(packet), VENUES["plain"])
        assert "accuracy gap=0.85" in tex
        assert "0.850" not in tex
        assert "\\num{" not in tex

    def test_anonymous_venues_carry_no_attribution(self) -> None:
        packet = reportable_packet()
        spec = VenueSpec(name="anon", anonymous=True)
        tex, _ = render_latex(packet, manuscript_for(packet), spec)
        assert "\\author{Anonymous Authors}" in tex
        assert "Attribution" not in tex
        assert "fake-model-1" not in tex

    def test_attributed_venues_name_the_model(self) -> None:
        packet = reportable_packet()
        tex, _ = render_latex(packet, manuscript_for(packet), VENUES["plain"])
        assert "\\author{Autonomous Research Lab}" in tex
        assert "\\section*{Attribution}" in tex
        assert "Prose authored by fake-model-1 via fake" in tex

    def test_the_style_package_line_forms(self) -> None:
        packet = reportable_packet()
        tex, _ = render_latex(
            packet, manuscript_for(packet), VENUES["neurips"]
        )
        assert "\\usepackage[final]{neurips_2026}" in tex
        assert "\\bibliographystyle{plainnat}" in tex

    def test_rendering_twice_is_byte_identical(self) -> None:
        packet = reportable_packet()
        draft = manuscript_for(packet)
        assert render_latex(packet, draft, VENUES["plain"]) == render_latex(
            packet, draft, VENUES["plain"]
        )

    def test_a_packet_id_mismatch_refuses(self) -> None:
        packet = reportable_packet()
        other = reportable_packet()
        import dataclasses

        other = dataclasses.replace(
            other,
            science=dataclasses.replace(other.science, question="another?"),
            packet_id="",
        )
        with pytest.raises(ManuscriptError, match="was authored from"):
            render_latex(other, manuscript_for(packet), VENUES["plain"])

    def test_the_citep_providecommand_is_always_present(self) -> None:
        packet = reportable_packet()
        tex, _ = render_latex(packet, manuscript_for(packet), VENUES["plain"])
        assert "\\providecommand{\\citep}{\\cite}" in tex


def fill_kit(destination: Path) -> None:
    (destination / "dummy_style.sty").write_text(
        "% a dummy style\n", encoding="utf-8"
    )


class TestKitStore:
    def test_stage_round_trip_manifest_last(self, tmp_path: Path) -> None:
        store = KitStore(tmp_path)
        manifest = store.stage(
            "dummy", fetch=fill_kit, source_url="local"
        )
        assert manifest.id.startswith("vkit_")
        assert store.manifest("dummy") == manifest
        assert store.verify("dummy") == ()

    def test_restaging_is_idempotent_and_never_refetches(
        self, tmp_path: Path
    ) -> None:
        store = KitStore(tmp_path)
        calls = []

        def counting(destination: Path) -> None:
            calls.append(1)
            fill_kit(destination)

        first = store.stage("dummy", fetch=counting, source_url="local")
        again = store.stage("dummy", fetch=counting, source_url="local")
        assert first == again
        assert calls == [1]

    def test_deep_tamper_is_reported_not_raised(
        self, tmp_path: Path
    ) -> None:
        store = KitStore(tmp_path)
        store.stage("dummy", fetch=fill_kit, source_url="local")
        (tmp_path / "dummy" / "dummy_style.sty").write_text(
            "% a DUMMY style\n", encoding="utf-8"  # same size, new bytes
        )
        problems = store.verify("dummy")
        assert problems and "no longer match" in problems[0]

    def test_a_leftover_scratch_refuses(self, tmp_path: Path) -> None:
        store = KitStore(tmp_path)
        (tmp_path / ".staging-dummy").mkdir(parents=True)
        with pytest.raises(KitConflictError, match="previous staging"):
            store.stage("dummy", fetch=fill_kit, source_url="local")

    def test_a_doctored_manifest_refuses_on_load(
        self, tmp_path: Path
    ) -> None:
        store = KitStore(tmp_path)
        store.stage("dummy", fetch=fill_kit, source_url="local")
        path = tmp_path / "manifests" / "dummy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["files"][0][1] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(KitIntegrityError, match="does not survive"):
            store.manifest("dummy")

    def test_escaping_paths_refuse(self) -> None:
        with pytest.raises(ValueError, match="stay relative"):
            KitManifest(
                name="dummy",
                source_url="local",
                archive_sha256="",
                retrieved_at="",
                files=(("../evil.sty", "0" * 64, 1),),
            )


class TestLatexCommands:
    def test_the_compile_plan_is_pure_data(self) -> None:
        assert latex_commands(latexmk=True, pdflatex=True, bibtex=True) == (
            ("latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"),
        )
        run = ("pdflatex", "-interaction=nonstopmode", "main.tex")
        assert latex_commands(
            latexmk=False, pdflatex=True, bibtex=True
        ) == (run, ("bibtex", "main"), run, run)
        assert latex_commands(
            latexmk=False, pdflatex=True, bibtex=False
        ) == (run, run)
        assert latex_commands(
            latexmk=False, pdflatex=False, bibtex=False
        ) == ()


# -- integration: one CI walk, rendered ---------------------------------------


@pytest.fixture(scope="module")
def approved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A walked root whose draft is authored and APPROVED."""
    root = tmp_path_factory.mktemp("render") / "run"
    result = walk(root, lab=ci_lab())
    assert result.ok, result.detail
    author_manuscript(root, lab=ci_lab())
    outcome = review_manuscript(root, lab=ci_lab())
    assert str(outcome.review.verdict) == "approved"
    return root


class TestRenderSubmission:
    def test_the_plain_venue_renders_write_once(
        self, approved: Path
    ) -> None:
        result = render_submission(
            approved, venue="plain", venue_config=None, kits_root=None
        )
        assert result.tex_path.is_file()
        assert result.bib_path.is_file()
        assert result.pdf_path is None
        first = result.tex_path.read_bytes()

        again = render_submission(
            approved, venue="plain", venue_config=None, kits_root=None
        )
        assert again.replayed
        assert again.tex_path.read_bytes() == first
        assert main(["verify", "--root", str(approved)]) == OK

    def test_a_kit_venue_copies_its_verified_files(
        self, approved: Path, tmp_path: Path
    ) -> None:
        kits = tmp_path / "kits"
        KitStore(kits).stage("dummy", fetch=fill_kit, source_url="local")
        config = tmp_path / "venue.json"
        config.write_text(
            json.dumps(
                {
                    "name": "dummyvenue",
                    "style_package": "dummy_style",
                    "kit": "dummy",
                    "anonymous": True,
                }
            ),
            encoding="utf-8",
        )
        result = render_submission(
            approved,
            venue=None,
            venue_config=config,
            kits_root=kits,
        )
        tex = result.tex_path.read_text(encoding="utf-8")
        assert "\\usepackage{dummy_style}" in tex
        assert (result.submission_dir / "dummy_style.sty").is_file()

    def test_a_kit_venue_without_kits_names_the_stage_command(
        self, approved: Path
    ) -> None:
        with pytest.raises(RenderError, match="stage_venue_kit"):
            render_submission(
                approved, venue="neurips", venue_config=None, kits_root=None
            )

    def test_a_tampered_kit_fails(
        self, approved: Path, tmp_path: Path
    ) -> None:
        kits = tmp_path / "kits"
        store = KitStore(kits)
        store.stage("dummy", fetch=fill_kit, source_url="local")
        (kits / "dummy" / "dummy_style.sty").write_text(
            "% doctored\n", encoding="utf-8"
        )
        config = tmp_path / "venue.json"
        config.write_text(
            json.dumps(
                {"name": "dv", "style_package": "dummy_style", "kit": "dummy"}
            ),
            encoding="utf-8",
        )
        with pytest.raises(RenderError, match="does not match its manifest"):
            render_submission(
                approved, venue=None, venue_config=config, kits_root=kits
            )

    def test_a_hand_edited_submission_refuses(
        self, approved: Path, tmp_path: Path
    ) -> None:
        result = render_submission(
            approved,
            venue="plain",
            venue_config=None,
            kits_root=None,
            out_dir=tmp_path / "sub",
        )
        result.tex_path.write_text("% doctored", encoding="utf-8")
        with pytest.raises(RenderError, match="never rewritten"):
            render_submission(
                approved,
                venue="plain",
                venue_config=None,
                kits_root=None,
                out_dir=tmp_path / "sub",
            )

    def test_pdf_without_a_toolchain_fails_but_writes_tex(
        self,
        approved: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(RenderError, match="no LaTeX toolchain"):
            render_submission(
                approved,
                venue="plain",
                venue_config=None,
                kits_root=None,
                out_dir=tmp_path / "sub",
                pdf=True,
            )
        assert (
            tmp_path / "sub" / "plain"
        ).exists()  # the .tex tree was written first


class TestRenderRefusals:
    def test_an_unreviewed_draft_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok
        author_manuscript(root, lab=ci_lab())

        with pytest.raises(NotApprovedError, match="review first"):
            render_submission(
                root, venue="plain", venue_config=None, kits_root=None
            )

    def test_a_standing_revise_is_refused_naming_the_review(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok
        author_manuscript(root, lab=ci_lab())
        outcome = review_manuscript(
            root, lab=_FakeLab([grounded_reply(root)]), revise=False
        )
        assert str(outcome.review.verdict) == "revise"

        with pytest.raises(NotApprovedError, match="is revise"):
            render_submission(
                root, venue="plain", venue_config=None, kits_root=None
            )

    def test_no_manuscript_is_a_refusal_via_the_cli(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok

        code = main(["render", "--root", str(root), "--venue", "plain"])
        assert code == REFUSED
        assert "author first" in capsys.readouterr().out

    def test_an_unknown_venue_is_a_failure_listing_the_registry(
        self, approved: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["render", "--root", str(approved), "--venue", "aaai"]
        )
        assert code == FAILED
        assert "plain" in capsys.readouterr().out


class TestStageVenueKitTool:
    def test_a_wrong_pin_refuses_before_extraction(
        self, tmp_path: Path
    ) -> None:
        import zipfile

        archive = tmp_path / "kit.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("style.sty", "% style\n")

        with pytest.raises(SystemExit, match="refusing to stage"):
            stage_main(
                [
                    "--kits-root",
                    str(tmp_path / "kits"),
                    "--venue",
                    "dummy",
                    "--from-zip",
                    str(archive),
                    "--sha256",
                    "0" * 64,
                ]
            )
        assert not (tmp_path / "kits" / "dummy").exists()

    def test_a_correct_pin_stages_and_prints_the_hash(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import hashlib
        import zipfile

        archive = tmp_path / "kit.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("style.sty", "% style\n")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()

        code = stage_main(
            [
                "--kits-root",
                str(tmp_path / "kits"),
                "--venue",
                "dummy",
                "--from-zip",
                str(archive),
                "--sha256",
                digest,
            ]
        )
        printed = capsys.readouterr().out
        assert code == 0
        assert digest in printed
        assert (tmp_path / "kits" / "dummy" / "style.sty").is_file()
