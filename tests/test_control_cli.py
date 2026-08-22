"""The ``arl`` command line.

What is pinned here is the part a script depends on: which verb does
what, what the exit codes mean, and that a root holding several
investigations is a question rather than a guess. The verbs that walk
the chain are exercised end to end by the canary, which has a lab to
walk it with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autonomous_research_lab.control.chain import (
    StageContext,
    StageOutcome,
    StagePlan,
)
from autonomous_research_lab.control.cli import FAILED, OK, REFUSED, main
from autonomous_research_lab.control.config import RunConfig
from autonomous_research_lab.control.controller import Controller
from autonomous_research_lab.control.lab import LabError, load_lab
from autonomous_research_lab.control.stage import Fact, StageName

CONFIG: dict[str, Any] = {
    "label": "cli chain",
    "model": "fake-model-1",
    "brief": {
        "topic": "sample efficiency in tiny classifiers",
        "cutoff_date": "2026-01-01",
        "recent_window_start": "2024-01-01",
    },
    "cfp": {
        "source_url": "https://example.invalid/cfp",
        "supplied_at": "2026-02-01T00:00:00+00:00",
        "text": "A workshop on sample efficiency.",
    },
    "selection": {
        "compute_constraint": "One CPU.",
        "data_constraint": "Synthetic only.",
        "time_constraint": "Ten minutes.",
        "experimental_constraint": "Seeded.",
    },
    "admission": {
        "scheduling_requirement": "One at a time.",
        "job_duration_requirement": "Ten minutes.",
        "checkpoint_requirement": "None.",
    },
    "funding": {
        "granted": {"wall_clock_seconds": 600.0, "usd": 1.0,
                    "model_tokens": 10_000},
        "authority": "Lab operator.",
    },
}


class Stub:
    """One stage that does nothing, so a root can be prepared without a
    provider, a network, or a lab."""

    @property
    def name(self) -> StageName:
        return StageName.MAPPING

    def plan(self, context: StageContext) -> StagePlan:
        del context
        return StagePlan(key="mapping:brf_1", subject_id="brf_1")

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None:
        del context, plan
        return None

    def execute(self, context: StageContext) -> StageOutcome:
        del context
        return StageOutcome(
            produced=((str(Fact.ASSESSMENT_ID), "madq_1"),),
            detail="a stubbed map",
        )

    def refusals(self) -> tuple[type[Exception], ...]:
        return ()

    def limit(self, config: RunConfig) -> int:
        del config
        return 1


def prepared(root: Path, label: str = "cli chain") -> str:
    """A root with one walked investigation in it."""
    controller = Controller(root, chain=(Stub(),))
    payload = json.loads(json.dumps(CONFIG))
    payload["label"] = label
    investigation = controller.begin(payload)
    controller.walk(investigation)
    return investigation.investigation_id


def write_config(root: Path, **overrides: object) -> Path:
    payload = json.loads(json.dumps(CONFIG))
    payload.update(overrides)
    path = root / "run.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class TestStatus:
    def test_it_prints_the_stage_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        investigation_id = prepared(tmp_path)

        code = main(["status", investigation_id, "--root", str(tmp_path)])

        printed = capsys.readouterr().out
        assert code == OK
        assert investigation_id in printed
        assert "mapping" in printed
        assert "experimentation" in printed

    def test_one_investigation_needs_no_naming(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        investigation_id = prepared(tmp_path)

        code = main(["status", "--root", str(tmp_path)])

        assert code == OK
        assert investigation_id in capsys.readouterr().out

    def test_several_investigations_are_a_question_not_a_guess(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = prepared(tmp_path, label="first")
        second = prepared(tmp_path, label="second")

        code = main(["status", "--root", str(tmp_path)])

        printed = capsys.readouterr().out
        assert code == FAILED
        assert first in printed
        assert second in printed

    def test_an_empty_root_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["status", "--root", str(tmp_path)])

        assert code == FAILED
        assert "no investigation" in capsys.readouterr().out

    def test_an_unknown_investigation_is_named(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        prepared(tmp_path)

        code = main(["status", "inv_nothing", "--root", str(tmp_path)])

        assert code == FAILED
        assert "inv_nothing" in capsys.readouterr().out


class TestVerify:
    def test_an_empty_root_is_intact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["verify", "--root", str(tmp_path)])

        assert code == OK
        assert "intact" in capsys.readouterr().out

    def test_a_walked_root_is_intact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        prepared(tmp_path)

        code = main(["verify", "--root", str(tmp_path)])

        printed = capsys.readouterr().out
        assert code == OK
        assert "event logs      1" in printed

    def test_an_edited_event_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        investigation_id = prepared(tmp_path)
        path = tmp_path / "control" / "logs" / investigation_id / "000001.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["detail"] = "a nicer story"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        code = main(["verify", "--root", str(tmp_path)])

        printed = capsys.readouterr().out
        assert code == FAILED
        assert "event log" in printed
        assert "was edited" in printed


class TestRun:
    def test_an_unusable_config_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "run"
        config = write_config(tmp_path, label="x" * 200)

        code = main(["run", str(config), "--root", str(root)])

        assert code == FAILED
        assert "FATAL" in capsys.readouterr().out
        assert not (root / "control" / "configs").exists()

    def test_a_missing_config_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["run", str(tmp_path / "nothing.json"), "--root", str(tmp_path)]
        )

        assert code == FAILED
        assert "cannot read" in capsys.readouterr().out

    def test_a_lab_that_is_not_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = write_config(tmp_path)

        code = main(
            [
                "run",
                str(config),
                "--root",
                str(tmp_path / "run"),
                "--lab",
                "json:loads",
            ]
        )

        assert code == FAILED
        assert "is not a lab" in capsys.readouterr().out


class TestLoadingALab:
    def test_a_spec_without_a_factory(self) -> None:
        with pytest.raises(LabError, match="module:factory"):
            load_lab("examples.canary_lab")

    def test_a_module_that_does_not_exist(self) -> None:
        with pytest.raises(LabError, match="cannot import"):
            load_lab("no.such.module:lab")

    def test_a_missing_attribute(self) -> None:
        with pytest.raises(LabError, match="no attribute"):
            load_lab("json:not_a_thing")

    def test_something_that_is_not_a_lab(self) -> None:
        with pytest.raises(LabError, match="is not a lab"):
            load_lab("json:loads")


class TestARootThatIsNotThere:
    def test_verify_refuses_it_rather_than_creating_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every store here makes its own directories, so a mistyped path
        would otherwise be answered with a new empty root and the word
        "intact"."""
        missing = tmp_path / "typo"

        code = main(["verify", "--root", str(missing)])

        assert code == FAILED
        assert "no run root" in capsys.readouterr().out
        assert not missing.exists()

    def test_status_refuses_it_too(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "typo"

        code = main(["status", "--root", str(missing)])

        assert code == FAILED
        assert not missing.exists()


class TestPacket:
    """The verb's conventions only — a root with real science to export
    is walked (once) in test_publication_packet.py."""

    def test_a_walk_without_a_research_state_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        prepared(tmp_path)

        code = main(["packet", "--root", str(tmp_path)])

        assert code == REFUSED
        assert "REFUSED" in capsys.readouterr().out

    def test_several_investigations_are_a_question_not_a_guess(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = prepared(tmp_path, label="first")
        second = prepared(tmp_path, label="second")

        code = main(["packet", "--root", str(tmp_path)])

        printed = capsys.readouterr().out
        assert code == FAILED
        assert first in printed
        assert second in printed

    def test_a_missing_root_is_named(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["packet", "--root", str(tmp_path / "nowhere")])

        assert code == FAILED
        assert "no run root" in capsys.readouterr().out
