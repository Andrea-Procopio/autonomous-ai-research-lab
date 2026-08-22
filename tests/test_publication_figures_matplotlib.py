"""The real renderer, and the whole publication chain behind it.

CI installs the ``figures`` extra, so these run there; a development
environment without matplotlib skips them, exactly as the torch suite
skips itself — every discipline these bytes travel through is already
covered without matplotlib in ``test_publication_figures.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from autonomous_research_lab.control.cli import OK, main
from autonomous_research_lab.publication.figures import (
    FigureData,
    compose_caption,
    figure_id_for,
    render_and_manifest,
    render_figure,
)
from examples.vision_chain import walk
from examples.vision_lab import ci_lab


def family(
    points: tuple[tuple[int | None, float], ...],
    *,
    mean: float | None = None,
    stdev: float | None = None,
) -> FigureData:
    values = [value for _, value in points]
    resolved = sum(values) / len(values) if mean is None else mean
    return FigureData(
        claim_id="clm_0123456789abcdef",
        prediction_id="prd_0123456789abcdef",
        metric="difference in accuracy",
        comparator="gt",
        threshold=0.0,
        points=points,
        n=len(points),
        mean=resolved,
        stdev=stdev,
        caption=compose_caption(
            claim_id="clm_0123456789abcdef",
            prediction_id="prd_0123456789abcdef",
            metric="difference in accuracy",
            comparator="gt",
            threshold=0.0,
            n=len(points),
            mean=resolved,
            stdev=stdev,
        ),
    )


class TestRenderFigure:
    @pytest.mark.parametrize(
        "points",
        [
            ((11, 0.04), (23, 0.05), (47, 0.03)),
            ((11, 0.012),),
            ((None, 0.5),),
            ((11, -0.2), (23, -0.1)),
            ((11, 0.3), (23, 0.3), (47, 0.3)),
        ],
        ids=["three-seeds", "n-1", "seed-none", "negative", "all-equal"],
    )
    def test_real_bytes_for_every_family_shape(
        self, points: tuple[tuple[int | None, float], ...]
    ) -> None:
        png, pdf = render_figure(family(points))
        assert png.startswith(b"\x89PNG")
        assert pdf.startswith(b"%PDF")

    def test_the_manifest_pins_the_bytes_it_returns(self) -> None:
        data = family(((11, 0.04), (23, 0.05), (47, 0.03)), stdev=0.01)
        manifest, files = render_and_manifest(
            data, rendered_at="2026-08-22T00:00:00+00:00"
        )
        assert manifest.figure_id == figure_id_for(data)
        assert set(files) == {
            f"{manifest.figure_id}.pdf",
            f"{manifest.figure_id}.png",
        }
        for name, digest, size in manifest.files:
            assert hashlib.sha256(files[name]).hexdigest() == digest
            assert len(files[name]) == size
        assert manifest.renderer.startswith("matplotlib ")


class TestScriptedChain:
    def test_the_ten_verbs_carry_the_figures_end_to_end(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok, result.detail

        assert main(["figures", "--root", str(root)]) == OK
        assert "(2 rendered, 0 replayed)" in capsys.readouterr().out
        assert main(["figures", "--root", str(root)]) == OK
        assert "(0 rendered, 2 replayed)" in capsys.readouterr().out

        assert main(["packet", "--root", str(root)]) == OK
        capsys.readouterr()
        assert (
            main(
                [
                    "manuscript",
                    "--root",
                    str(root),
                    "--lab",
                    "examples.vision_lab:ci_lab",
                ]
            )
            == OK
        )
        capsys.readouterr()
        markdown = next((root / "manuscript").glob("*.md")).read_text(
            encoding="utf-8"
        )
        assert "](../figures/fig_" in markdown

        assert (
            main(
                [
                    "review",
                    "--root",
                    str(root),
                    "--lab",
                    "examples.vision_lab:ci_lab",
                ]
            )
            == OK
        )
        capsys.readouterr()
        assert (
            main(["render", "--root", str(root), "--venue", "plain"]) == OK
        )
        capsys.readouterr()
        submission = next(
            (root / "submission" / "plain").glob("mscr_*")
        )
        main_tex = (submission / "main.tex").read_text(encoding="utf-8")
        assert "\\usepackage{graphicx}" in main_tex
        assert "\\includegraphics[width=0.7\\linewidth]{figures/fig_" in main_tex
        staged = sorted((submission / "figures").glob("fig_*.pdf"))
        assert len(staged) == 2
        for path in staged:
            assert path.read_bytes().startswith(b"%PDF")

        assert (
            main(
                [
                    "simulate",
                    "--root",
                    str(root),
                    "--venue",
                    "plain",
                    "--lab",
                    "examples.vision_lab:ci_lab",
                ]
            )
            == OK
        )
        assert "MEETS BAR" in capsys.readouterr().out
        assert main(["verify", "--root", str(root)]) == OK
