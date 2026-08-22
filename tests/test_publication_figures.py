"""Rendered figures: identity, the store, and the packet's honesty.

Everything here runs without matplotlib: the store's ``record`` is the
trusted injection seam, so fake bytes stand in for rendered ones and
the discipline under test — data-as-identity, hashed-at-creation,
write-once, stale-figure refusal, the packet and document integration —
is exactly the discipline the real renderer goes through. The renderer
itself is covered in ``test_publication_figures_matplotlib.py``.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from autonomous_research_lab.control.cli import FAILED, OK, REFUSED, main
from autonomous_research_lab.control.figures import render_figures
from autonomous_research_lab.control.manuscript import author_manuscript
from autonomous_research_lab.control.packet import build_packet, read_run
from autonomous_research_lab.control.render import render_submission
from autonomous_research_lab.control.review import review_manuscript
from autonomous_research_lab.publication.figures import (
    FigureConflictError,
    FigureData,
    FigureIntegrityError,
    FigureManifest,
    FigureStore,
    FiguresUnavailableError,
    StaleFigureError,
    compose_caption,
    figure_id_for,
    planned_figures,
)
from autonomous_research_lab.publication.manuscript import (
    PROSE_SECTIONS,
    gate_prose,
    known_renderings,
)
from autonomous_research_lab.publication.packet import PacketError
from examples.vision_chain import walk
from examples.vision_lab import ci_lab

ABSENCE_LINE = (
    "Figures: none — nothing in this run rendered plots, and the "
    "packet states that rather than implying an omission."
)


def data(**overrides: object) -> FigureData:
    fields: dict[str, object] = {
        "claim_id": "clm_0123456789abcdef",
        "prediction_id": "prd_0123456789abcdef",
        "metric": "difference in accuracy",
        "comparator": "gt",
        "threshold": 0.0,
        "points": ((11, 0.04), (23, 0.05), (47, 0.03)),
        "n": 3,
        "mean": 0.04,
        "stdev": 0.01,
    }
    fields.update(overrides)
    if "caption" not in fields:
        fields["caption"] = compose_caption(
            claim_id=str(fields["claim_id"]),
            prediction_id=str(fields["prediction_id"]),
            metric=str(fields["metric"]),
            comparator=str(fields["comparator"]),
            threshold=float(str(fields["threshold"])),
            n=int(str(fields["n"])),
            mean=None if fields["mean"] is None else float(str(fields["mean"])),
            stdev=(
                None if fields["stdev"] is None else float(str(fields["stdev"]))
            ),
        )
    return FigureData(**fields)  # type: ignore[arg-type]


def drawn(figure: FigureData) -> tuple[FigureManifest, dict[str, bytes]]:
    """Fake bytes, real digests — the injection seam."""
    figure_id = figure_id_for(figure)
    files = {
        f"{figure_id}.pdf": b"%PDF fake " + figure_id.encode(),
        f"{figure_id}.png": b"\x89PNG fake " + figure_id.encode(),
    }
    manifest = FigureManifest(
        data=figure,
        files=tuple(
            (name, hashlib.sha256(content).hexdigest(), len(content))
            for name, content in sorted(files.items())
        ),
        renderer="fake-renderer 0 (test)",
        rendered_at="2026-08-22T00:00:00+00:00",
    )
    return manifest, files


def injected(root: Path) -> tuple[FigureData, ...]:
    reading = read_run(root)
    plan = planned_figures(
        reading.head,
        reading.stores.evidence,
        admissible=reading.admissible,
        seed_of=reading.seed_of,
    )
    store = FigureStore(root / "figures")
    for figure in plan:
        manifest, files = drawn(figure)
        store.record(manifest, files)
    return plan


class TestFigureIdentity:
    def test_the_id_derives_from_the_data_alone(self) -> None:
        first = figure_id_for(data())
        assert first == figure_id_for(data())
        assert first.startswith("fig_") and len(first) == 20

    def test_a_changed_point_changes_the_id(self) -> None:
        other = data(points=((11, 0.04), (23, 0.05), (47, 0.031)))
        assert figure_id_for(other) != figure_id_for(data())

    def test_bytes_renderer_and_time_are_outside_the_id(self) -> None:
        manifest, _ = drawn(data())
        other = FigureManifest(
            data=data(),
            files=tuple(
                (name, "f" * 64, size + 1)
                for name, _, size in manifest.files
            ),
            renderer="another-renderer 9 (agg)",
            rendered_at="2031-01-01T00:00:00+00:00",
        )
        assert other.figure_id == manifest.figure_id
        assert other != manifest

    def test_a_doctored_manifest_id_is_refused(self) -> None:
        manifest, _ = drawn(data())
        with pytest.raises(FigureIntegrityError, match="survive"):
            replace(manifest, figure_id="fig_" + "0" * 16)

    def test_the_manifest_names_exactly_its_own_files(self) -> None:
        manifest, _ = drawn(data())
        with pytest.raises(ValueError, match="names exactly"):
            replace(
                manifest,
                files=tuple(
                    ("other.pdf", digest, size)
                    for _, digest, size in manifest.files
                ),
                figure_id="",
            )

    def test_malformed_data_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no points"):
            data(points=())
        with pytest.raises(ValueError, match="cannot yield"):
            data(n=2)
        with pytest.raises(ValueError, match="caption"):
            data(caption="   ")


class TestFigureStore:
    def test_record_get_verify_round_trip(self, tmp_path: Path) -> None:
        store = FigureStore(tmp_path / "figures")
        manifest, files = drawn(data())
        recorded = store.record(manifest, files)
        assert store.get(recorded.figure_id) == recorded
        assert store.verify(recorded.figure_id) == ()
        assert store.manifests() == (recorded,)

    def test_an_identical_re_record_is_a_replay(self, tmp_path: Path) -> None:
        store = FigureStore(tmp_path / "figures")
        manifest, files = drawn(data())
        store.record(manifest, files)
        assert store.record(manifest, files) == manifest

    def test_different_bytes_under_the_same_id_refuse(
        self, tmp_path: Path
    ) -> None:
        store = FigureStore(tmp_path / "figures")
        manifest, files = drawn(data())
        store.record(manifest, files)
        other_files = {
            name: content + b"!" for name, content in files.items()
        }
        other = FigureManifest(
            data=data(),
            files=tuple(
                (name, hashlib.sha256(content).hexdigest(), len(content))
                for name, content in sorted(other_files.items())
            ),
            renderer=manifest.renderer,
            rendered_at=manifest.rendered_at,
        )
        with pytest.raises(FigureConflictError, match="never rewritten"):
            store.record(other, other_files)

    def test_bytes_that_do_not_match_their_manifest_refuse(
        self, tmp_path: Path
    ) -> None:
        store = FigureStore(tmp_path / "figures")
        manifest, files = drawn(data())
        doctored = {name: content + b"!" for name, content in files.items()}
        with pytest.raises(FigureIntegrityError, match="does not hash"):
            store.record(manifest, doctored)

    def test_corruption_after_recording_is_named(self, tmp_path: Path) -> None:
        store = FigureStore(tmp_path / "figures")
        manifest, files = drawn(data())
        store.record(manifest, files)
        target = store.file_path(f"{manifest.figure_id}.png")
        content = target.read_bytes()
        target.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        problems = store.verify(manifest.figure_id)
        assert problems and "no longer match" in problems[0]
        with pytest.raises(FigureIntegrityError, match="no longer"):
            store.record(manifest, files)

    def test_a_missing_file_is_named(self, tmp_path: Path) -> None:
        store = FigureStore(tmp_path / "figures")
        manifest, files = drawn(data())
        store.record(manifest, files)
        store.file_path(f"{manifest.figure_id}.pdf").unlink()
        assert any(
            "missing" in problem
            for problem in store.verify(manifest.figure_id)
        )

    def test_a_record_filed_under_the_wrong_name_is_refused(
        self, tmp_path: Path
    ) -> None:
        store = FigureStore(tmp_path / "figures")
        manifest, files = drawn(data())
        store.record(manifest, files)
        manifests_dir = tmp_path / "figures" / "manifests"
        wrong = manifests_dir / ("fig_" + "0" * 16 + ".json")
        (manifests_dir / f"{manifest.figure_id}.json").rename(wrong)
        with pytest.raises(FigureIntegrityError, match="wrong name"):
            store.get("fig_" + "0" * 16)

    def test_reads_on_a_missing_root_create_nothing(
        self, tmp_path: Path
    ) -> None:
        store = FigureStore(tmp_path / "absent")
        assert store.manifests() == ()
        assert store.get("fig_" + "0" * 16) is None
        assert not (tmp_path / "absent").exists()


class TestPlannedFiguresSkips:
    def test_unassessed_and_foreign_methods_are_skipped(self) -> None:
        claims = (SimpleNamespace(id="clm_1"), SimpleNamespace(id="clm_2"))
        assessments = {
            "clm_1": None,
            "clm_2": SimpleNamespace(method="demo:prediction-check-v0"),
        }
        state = SimpleNamespace(
            claims=claims,
            current_assessment=lambda claim_id: assessments[claim_id],
        )
        assert (
            planned_figures(
                state,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                admissible=lambda _: True,
                seed_of={},
            )
            == ()
        )


# -- integration: one CI walk, figured ----------------------------------------


@pytest.fixture(scope="module")
def walked(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("figures-none") / "run"
    result = walk(root, lab=ci_lab())
    assert result.ok, result.detail
    return root


@pytest.fixture(scope="module")
def figured(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("figures-full") / "run"
    result = walk(root, lab=ci_lab())
    assert result.ok, result.detail
    injected(root)
    return root


class TestPacketWithoutFigures:
    def test_the_absence_is_stated_byte_for_byte(self, walked: Path) -> None:
        from autonomous_research_lab.publication.packet import render_markdown

        packet = build_packet(walked)
        assert packet.figures == ()
        rendered = render_markdown(packet)
        assert ABSENCE_LINE in rendered
        assert "## Figures" not in rendered

    def test_the_documents_carry_no_figure_markup(self, walked: Path) -> None:
        from autonomous_research_lab.publication.latex import (
            VENUES,
            render_latex,
        )
        from autonomous_research_lab.publication.manuscript import assemble

        packet = build_packet(walked)
        result = author_manuscript(walked, lab=ci_lab())
        assert "](../figures/" not in assemble(packet, result.manuscript)
        main_tex, _ = render_latex(packet, result.manuscript, VENUES["plain"])
        assert "graphicx" not in main_tex


class TestPacketWithFigures:
    def test_the_mirrors_enter_in_plan_order(self, figured: Path) -> None:
        from autonomous_research_lab.publication.packet import render_markdown

        packet = build_packet(figured)
        assert len(packet.figures) == 2
        reading = read_run(figured)
        plan = planned_figures(
            reading.head,
            reading.stores.evidence,
            admissible=reading.admissible,
            seed_of=reading.seed_of,
        )
        assert tuple(f.figure_id for f in packet.figures) == tuple(
            figure_id_for(entry) for entry in plan
        )
        assert tuple(f.points for f in packet.figures) == tuple(
            entry.points for entry in plan
        )
        rendered = render_markdown(packet)
        assert "## Figures" in rendered
        assert ABSENCE_LINE not in rendered
        for figure in packet.figures:
            assert figure.caption in rendered
            for name, digest, _ in figure.files:
                assert name in rendered and digest in rendered

    def test_figures_change_the_packet_id(
        self, walked: Path, figured: Path
    ) -> None:
        assert build_packet(figured).packet_id != build_packet(walked).packet_id

    def test_the_number_gate_extends_to_the_figure_digests(
        self, walked: Path, figured: Path
    ) -> None:
        packet = build_packet(figured)
        digest = packet.figures[0].files[0][1]
        sections = {name: "Plain prose without numbers." for name in PROSE_SECTIONS}
        sections["discussion"] = f"The figure bytes hash to {digest}."
        assert (
            gate_prose(
                sections,
                allowed=known_renderings(packet),
                bibliography=packet.bibliography,
            )
            == ()
        )
        rejections = gate_prose(
            sections,
            allowed=known_renderings(build_packet(walked)),
            bibliography=packet.bibliography,
        )
        assert rejections and any(
            digest in rejection.detail for rejection in rejections
        )

    def test_the_packet_verb_exports_the_figured_packet(
        self, figured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["packet", "--root", str(figured)]) == OK
        printed = capsys.readouterr().out
        assert "packet" in printed

    def test_a_stale_figure_fails_the_export(
        self, figured: Path, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        shutil.copytree(figured, root)
        stranger = data()
        manifest, files = drawn(stranger)
        FigureStore(root / "figures").record(manifest, files)
        with pytest.raises(StaleFigureError, match="no longer derives"):
            build_packet(root)
        assert main(["packet", "--root", str(root)]) == FAILED

    def test_a_corrupt_figure_file_fails_the_export(
        self, figured: Path, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        shutil.copytree(figured, root)
        store = FigureStore(root / "figures")
        manifest = store.manifests()[0]
        name = manifest.files[1][0]
        content = store.file_path(name).read_bytes()
        store.file_path(name).write_bytes(
            content[:-1] + bytes([content[-1] ^ 1])
        )
        with pytest.raises(PacketError, match="survive its digests"):
            build_packet(root)


class TestFiguresVerb:
    def test_matplotlib_missing_is_a_refusal(
        self,
        walked: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = tmp_path / "run"
        shutil.copytree(walked, root)

        def unavailable(*args: object, **kwargs: object) -> object:
            raise FiguresUnavailableError("matplotlib is not installed")

        monkeypatch.setattr(
            "autonomous_research_lab.control.figures.render_and_manifest",
            unavailable,
        )
        assert main(["figures", "--root", str(root)]) == REFUSED
        assert "REFUSED" in capsys.readouterr().out

    def test_nothing_to_draw_is_a_refusal(
        self,
        walked: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = tmp_path / "run"
        shutil.copytree(walked, root)
        monkeypatch.setattr(
            "autonomous_research_lab.control.figures.planned_figures",
            lambda *args, **kwargs: (),
        )
        assert main(["figures", "--root", str(root)]) == REFUSED
        assert "nothing trusted code may draw" in capsys.readouterr().out

    def test_the_fake_renderer_draws_once_then_replays(
        self,
        walked: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = tmp_path / "run"
        shutil.copytree(walked, root)

        def fake_render(
            figure: FigureData, *, rendered_at: str = ""
        ) -> tuple[FigureManifest, dict[str, bytes]]:
            return drawn(figure)

        monkeypatch.setattr(
            "autonomous_research_lab.control.figures.render_and_manifest",
            fake_render,
        )
        assert main(["figures", "--root", str(root)]) == OK
        assert "(2 rendered, 0 replayed)" in capsys.readouterr().out
        assert main(["figures", "--root", str(root)]) == OK
        assert "(0 rendered, 2 replayed)" in capsys.readouterr().out

    def test_a_finished_store_replays_without_matplotlib(
        self, figured: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "run"
        shutil.copytree(figured, root)

        def unavailable(*args: object, **kwargs: object) -> object:
            raise FiguresUnavailableError("matplotlib is not installed")

        monkeypatch.setattr(
            "autonomous_research_lab.control.figures.render_and_manifest",
            unavailable,
        )
        result = render_figures(root)
        assert len(result.replayed) == 2 and not result.rendered

    def test_a_tampered_store_is_fatal(
        self,
        figured: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = tmp_path / "run"
        shutil.copytree(figured, root)
        store = FigureStore(root / "figures")
        manifest = store.manifests()[0]
        name = manifest.files[1][0]
        content = store.file_path(name).read_bytes()
        store.file_path(name).write_bytes(
            content[:-1] + bytes([content[-1] ^ 1])
        )
        assert main(["figures", "--root", str(root)]) == FAILED
        assert "FATAL" in capsys.readouterr().out


class TestFiguredDocuments:
    def test_the_manuscript_embeds_the_reading_copy(
        self, figured: Path
    ) -> None:
        result = author_manuscript(figured, lab=ci_lab())
        document = result.markdown_path.read_text(encoding="utf-8")
        packet = build_packet(figured)
        for figure in packet.figures:
            assert f"](../figures/{figure.figure_id}.png)" in document
            assert figure.caption in document

    def test_the_submission_carries_the_verified_pdfs(
        self, figured: Path
    ) -> None:
        author_manuscript(figured, lab=ci_lab())
        outcome = review_manuscript(figured, lab=ci_lab())
        assert str(outcome.review.verdict) == "approved"
        rendered = render_submission(
            figured, venue="plain", venue_config=None, kits_root=None
        )
        main_tex = (rendered.submission_dir / "main.tex").read_text(
            encoding="utf-8"
        )
        assert "\\usepackage{graphicx}" in main_tex
        packet = build_packet(figured)
        for figure in packet.figures:
            name = f"{figure.figure_id}.pdf"
            included = (
                f"\\includegraphics[width=0.7\\linewidth]"
                f"{{figures/{name}}}"
            )
            assert included in main_tex
            staged = rendered.submission_dir / "figures" / name
            assert (
                hashlib.sha256(staged.read_bytes()).hexdigest()
                == dict(
                    (entry[0], entry[1]) for entry in figure.files
                )[name]
            )
        assert main(["verify", "--root", str(figured)]) == OK
