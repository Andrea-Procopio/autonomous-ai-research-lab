"""The walk: one stage at a time, once each, resumably.

The stages here are stubs. What is under test is the controller's own
behavior — what it skips, what it reconciles, what it records, and where
it stops — and a real stage would only make that harder to see. The
seven real adapters are exercised end to end by the canary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from autonomous_research_lab.control.chain import (
    StageContext,
    StageOutcome,
    StagePlan,
)
from autonomous_research_lab.control.config import RunConfig
from autonomous_research_lab.control.controller import (
    Controller,
    ControllerError,
    Outcome,
)
from autonomous_research_lab.control.stage import (
    Fact,
    StageName,
    StageSpend,
    StageStatus,
)

CONFIG: dict[str, Any] = {
    "label": "stub chain",
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
    "experimentation": {"max_steps": 3},
}


class BoomError(RuntimeError):
    """A stage broke."""


class DoorSaidNoError(RuntimeError):
    """A stage's preconditions were refused."""


@dataclass
class Stub:
    """A stage that does exactly what a test tells it to."""

    stage: StageName
    subject: str = "sub_1"
    produces: tuple[tuple[str, str], ...] = ()
    raises: Exception | None = None
    ends: str = ""
    already_done: StageOutcome | None = None
    steps: int = 0
    """When positive, the stage repeats that many times, each under a new
    key — the shape experimentation has."""

    calls: list[str] = field(default_factory=list)
    probes: list[str] = field(default_factory=list)

    @property
    def name(self) -> StageName:
        return self.stage

    def plan(self, context: StageContext) -> StagePlan:
        subject = self.subject
        if self.steps:
            subject = f"{self.subject}-{len(self.calls)}"
        del context
        return StagePlan(key=f"{self.stage}:{subject}", subject_id=subject)

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None:
        del context
        self.probes.append(plan.key)
        return self.already_done

    def execute(self, context: StageContext) -> StageOutcome:
        del context
        self.calls.append(self.stage)
        if self.raises is not None:
            raise self.raises
        return StageOutcome(
            produced=self.produces,
            spend=StageSpend(model_calls=1, input_tokens=10, output_tokens=5),
            detail=f"{self.stage} did its work",
            ends_investigation=self.ends,
            repeat=bool(self.steps) and len(self.calls) < self.steps,
        )

    def refusals(self) -> tuple[type[Exception], ...]:
        return (DoorSaidNoError,)

    def limit(self, config: RunConfig) -> int:
        return config.experimentation.max_steps if self.steps else 1


def controller(root: Path, *stages: Stub) -> Controller:
    return Controller(root, chain=tuple(stages))


def mapping(**overrides: object) -> Stub:
    fields: dict[str, object] = {
        "stage": StageName.MAPPING,
        "produces": ((str(Fact.ASSESSMENT_ID), "madq_1"),),
    }
    fields.update(overrides)
    return Stub(**fields)  # type: ignore[arg-type]


def ideation(**overrides: object) -> Stub:
    fields: dict[str, object] = {
        "stage": StageName.IDEATION,
        "subject": "sub_2",
        "produces": ((str(Fact.IDEATION_RUN_RECORD_ID), "irun_1"),),
    }
    fields.update(overrides)
    return Stub(**fields)  # type: ignore[arg-type]


class TestAStraightWalk:
    def test_every_stage_runs_once_in_order(self, tmp_path: Path) -> None:
        first, second = mapping(), ideation()

        result = controller(tmp_path, first, second).run(CONFIG)

        assert result.outcome is Outcome.COMPLETED
        assert result.ok
        assert first.calls and second.calls
        assert result.facts.require(Fact.ASSESSMENT_ID) == "madq_1"

    def test_each_stage_leaves_a_running_and_a_terminal_event(
        self, tmp_path: Path
    ) -> None:
        result = controller(tmp_path, mapping()).run(CONFIG)

        statuses = [event.status for event in result.events]
        assert statuses == [StageStatus.RUNNING, StageStatus.SUCCEEDED]

    def test_the_config_is_recorded_before_anything_runs(
        self, tmp_path: Path
    ) -> None:
        control = controller(tmp_path, mapping())
        investigation = control.begin(CONFIG)

        stored = control.investigations.get_config(investigation.config_id)

        assert stored == CONFIG
        assert control.investigations.log_for(
            investigation.investigation_id
        ).events() == ()

    def test_stop_after_halts_the_walk(self, tmp_path: Path) -> None:
        first, second = mapping(), ideation()

        result = controller(tmp_path, first, second).run(
            CONFIG, stop_after=StageName.MAPPING
        )

        assert result.outcome is Outcome.STOPPED
        assert second.calls == []

    def test_a_resumed_walk_passes_the_stop(self, tmp_path: Path) -> None:
        first, second = mapping(), ideation()
        control = controller(tmp_path, first, second)
        investigation = control.begin(CONFIG, stop_after=StageName.MAPPING)
        control.walk(investigation)

        again = Controller(tmp_path, chain=(mapping(), second))
        result = again.resume(investigation.investigation_id)

        assert result.outcome is Outcome.STOPPED
        assert first.calls == [StageName.MAPPING]
        assert second.calls == []


class TestDoingWorkOnlyOnce:
    def test_a_recorded_stage_is_skipped_entirely(self, tmp_path: Path) -> None:
        control = controller(tmp_path, mapping())
        investigation = control.begin(CONFIG)
        control.walk(investigation)

        second_pass = mapping()
        Controller(tmp_path, chain=(second_pass,)).resume(
            investigation.investigation_id
        )

        assert second_pass.calls == []
        assert second_pass.probes == []

    def test_its_produced_ids_are_adopted_from_the_log(
        self, tmp_path: Path
    ) -> None:
        control = controller(tmp_path, mapping(), ideation())
        investigation = control.begin(CONFIG)
        control.walk(investigation)

        later = ideation()
        result = Controller(tmp_path, chain=(mapping(), later)).resume(
            investigation.investigation_id
        )

        assert result.facts.require(Fact.ASSESSMENT_ID) == "madq_1"

    def test_work_on_disk_without_an_event_is_reconciled(
        self, tmp_path: Path
    ) -> None:
        """The crash case: the side effect landed, the record of it did
        not. The controller writes the missing event rather than paying
        for the work twice."""
        done = StageOutcome(
            produced=((str(Fact.ASSESSMENT_ID), "madq_recovered"),),
            detail="found in the store",
        )
        stage = mapping(already_done=done)

        result = controller(tmp_path, stage).run(CONFIG)

        assert stage.calls == []
        assert result.facts.require(Fact.ASSESSMENT_ID) == "madq_recovered"
        assert "reconciled" in result.events[-1].detail

    def test_a_reconciled_stage_records_no_running_event(
        self, tmp_path: Path
    ) -> None:
        stage = mapping(already_done=StageOutcome(detail="already there"))

        result = controller(tmp_path, stage).run(CONFIG)

        assert [event.status for event in result.events] == [
            StageStatus.SUCCEEDED
        ]


class TestWhenAStageDoesNotSucceed:
    def test_a_refusal_stops_the_walk_and_is_recorded(
        self, tmp_path: Path
    ) -> None:
        stage = mapping(raises=DoorSaidNoError("the map is not adequate"))

        result = controller(tmp_path, stage, ideation()).run(CONFIG)

        assert result.outcome is Outcome.REFUSED
        assert not result.ok
        assert result.events[-1].status is StageStatus.REFUSED
        assert "not adequate" in result.events[-1].detail

    def test_a_failure_stops_the_walk_and_is_recorded(
        self, tmp_path: Path
    ) -> None:
        stage = mapping(raises=BoomError("the provider died"))

        result = controller(tmp_path, stage).run(CONFIG)

        assert result.outcome is Outcome.FAILED
        assert result.events[-1].status is StageStatus.FAILED
        assert "BoomError: the provider died" in result.events[-1].detail

    def test_a_later_stage_never_starts(self, tmp_path: Path) -> None:
        later = ideation()

        controller(tmp_path, mapping(raises=BoomError("no")), later).run(CONFIG)

        assert later.calls == []

    def test_resuming_re_attempts_exactly_the_failed_stage(
        self, tmp_path: Path
    ) -> None:
        control = controller(tmp_path, mapping(raises=BoomError("transient")))
        investigation = control.begin(CONFIG)
        control.walk(investigation)

        retried, later = mapping(), ideation()
        result = Controller(tmp_path, chain=(retried, later)).resume(
            investigation.investigation_id
        )

        assert retried.calls == [StageName.MAPPING]
        assert later.calls == [StageName.IDEATION]
        assert result.outcome is Outcome.COMPLETED


class TestAnHonestNo:
    def test_it_ends_the_investigation_as_a_success(
        self, tmp_path: Path
    ) -> None:
        stage = mapping(ends="no candidate was worth proposing")

        result = controller(tmp_path, stage, ideation()).run(CONFIG)

        assert result.outcome is Outcome.ENDED
        assert result.ok
        assert "worth proposing" in result.detail

    def test_the_stage_itself_succeeded(self, tmp_path: Path) -> None:
        result = controller(tmp_path, mapping(ends="nothing eligible")).run(
            CONFIG
        )

        succeeded = [
            event
            for event in result.events
            if event.status is StageStatus.SUCCEEDED
        ]
        assert len(succeeded) == 1

    def test_the_stages_after_it_are_marked_skipped(
        self, tmp_path: Path
    ) -> None:
        result = controller(tmp_path, mapping(ends="nothing eligible")).run(
            CONFIG
        )

        skipped = [
            event.stage
            for event in result.events
            if event.status is StageStatus.SKIPPED
        ]
        assert StageName.IDEATION in skipped
        assert StageName.EXPERIMENTATION in skipped

    def test_resuming_an_ended_investigation_does_nothing(
        self, tmp_path: Path
    ) -> None:
        control = controller(tmp_path, mapping(ends="nothing eligible"))
        investigation = control.begin(CONFIG)
        control.walk(investigation)

        again = mapping()
        result = Controller(tmp_path, chain=(again,)).resume(
            investigation.investigation_id
        )

        assert again.calls == []
        assert result.outcome is Outcome.ENDED


class TestAStageThatRepeats:
    def test_it_runs_until_it_stops_asking(self, tmp_path: Path) -> None:
        stepper = Stub(stage=StageName.EXPERIMENTATION, steps=2)

        result = controller(tmp_path, stepper).run(CONFIG)

        assert len(stepper.calls) == 2
        assert result.outcome is Outcome.COMPLETED

    def test_its_limit_bounds_one_walk(self, tmp_path: Path) -> None:
        stepper = Stub(stage=StageName.EXPERIMENTATION, steps=99)

        result = controller(tmp_path, stepper).run(CONFIG)

        assert len(stepper.calls) == CONFIG["experimentation"]["max_steps"]
        assert result.outcome is Outcome.STOPPED
        assert "limit" in result.detail

    def test_a_repeat_with_no_new_work_is_a_bug_not_a_loop(
        self, tmp_path: Path
    ) -> None:
        @dataclass
        class Stuck(Stub):
            def plan(self, context: StageContext) -> StagePlan:
                del context
                return StagePlan(key="stuck:always", subject_id="always")

        stuck = Stuck(stage=StageName.EXPERIMENTATION, steps=2)

        with pytest.raises(ControllerError, match="no progress"):
            controller(tmp_path, stuck).run(CONFIG)


class TestWhatTheWalkReadsBack:
    def test_a_resumed_walk_uses_the_recorded_config(
        self, tmp_path: Path
    ) -> None:
        """Editing the file afterwards changes nothing: the run that was
        started is the run that continues."""
        control = controller(tmp_path, mapping(raises=BoomError("stop here")))
        investigation = control.begin(CONFIG)
        control.walk(investigation)
        recorded = control.investigations.get_config(investigation.config_id)
        assert recorded is not None

        resumed = Controller(tmp_path, chain=(mapping(),))
        resumed.resume(investigation.investigation_id)

        assert (
            resumed.investigations.get_config(investigation.config_id)
            == recorded
        )

    def test_an_unknown_investigation_is_named_in_the_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ControllerError, match="inv_nothing"):
            controller(tmp_path, mapping()).resume("inv_nothing")

    def test_a_config_that_vanished_is_named_in_the_error(
        self, tmp_path: Path
    ) -> None:
        control = controller(tmp_path, mapping())
        investigation = control.begin(CONFIG)
        (tmp_path / "control" / "configs" / f"{investigation.config_id}.json").unlink()

        with pytest.raises(ControllerError, match="cannot be resumed"):
            control.resume(investigation.investigation_id)

    def test_a_bad_config_writes_nothing_at_all(self, tmp_path: Path) -> None:
        broken = json.loads(json.dumps(CONFIG))
        broken["brief"]["cutoff_date"] = "not a date"

        with pytest.raises(Exception, match="ISO date"):
            controller(tmp_path, mapping()).run(broken)

        assert not (tmp_path / "control" / "configs").exists()


class TestStatus:
    def test_it_reports_every_stage(self, tmp_path: Path) -> None:
        control = controller(tmp_path, mapping(), ideation())
        investigation = control.begin(CONFIG)
        control.walk(investigation)

        report = control.status(investigation.investigation_id)

        assert len(report.lines) == 7
        assert report.lines[0].status is StageStatus.SUCCEEDED
        assert report.lines[2].status is StageStatus.PENDING

    def test_it_totals_the_spend(self, tmp_path: Path) -> None:
        control = controller(tmp_path, mapping(), ideation())
        investigation = control.begin(CONFIG)
        control.walk(investigation)

        report = control.status(investigation.investigation_id)

        assert report.spend == StageSpend(
            model_calls=2, input_tokens=20, output_tokens=10
        )

    def test_it_carries_the_ids_the_chain_produced(
        self, tmp_path: Path
    ) -> None:
        control = controller(tmp_path, mapping())
        investigation = control.begin(CONFIG)
        control.walk(investigation)

        report = control.status(investigation.investigation_id)

        assert report.facts.require(Fact.ASSESSMENT_ID) == "madq_1"
