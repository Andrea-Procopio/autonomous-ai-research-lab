"""The stage event log.

Three properties are pinned here, because the controller's resume story
rests entirely on them: the log is append-only and tamper-loud, a crash
leaves a visible unfinished event, and replaying the log rebuilds every
id the chain produced without any object surviving the process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.control.events import (
    MAX_DETAIL_CHARS,
    StageEvent,
    StageLog,
    StageLogIntegrityError,
)
from autonomous_research_lab.control.stage import (
    ChainFacts,
    Fact,
    MissingFactError,
    StageName,
    StageSpend,
    StageStatus,
)

INVESTIGATION = "inv_1"
SPEND = StageSpend(model_calls=3, input_tokens=1200, output_tokens=340)


def event(**overrides: object) -> StageEvent:
    fields: dict[str, object] = {
        "investigation_id": INVESTIGATION,
        "sequence": 0,
        "stage": StageName.MAPPING,
        "status": StageStatus.RUNNING,
        "key": "mapping:brf_1",
        "subject_id": "brf_1",
    }
    fields.update(overrides)
    return StageEvent(**fields)  # type: ignore[arg-type]


def walked(log: StageLog, stage: StageName, subject: str, **produced: str) -> None:
    """One stage, run to completion the way the controller runs it."""
    key = f"{stage}:{subject}"
    log.append(
        stage=stage, status=StageStatus.RUNNING, key=key, subject_id=subject
    )
    log.append(
        stage=stage,
        status=StageStatus.SUCCEEDED,
        key=key,
        subject_id=subject,
        produced=tuple(produced.items()),
        spend=SPEND,
    )


class TestTheEventItself:
    def test_pending_is_never_an_event(self) -> None:
        with pytest.raises(ValueError, match="absence of an event"):
            event(status=StageStatus.PENDING)

    def test_a_running_event_cannot_name_what_it_produced(self) -> None:
        with pytest.raises(ValueError, match="predates the work"):
            event(produced=(("assessment_id", "madq_1"),))

    def test_produced_ids_are_stored_sorted(self) -> None:
        with pytest.raises(ValueError, match="stored sorted"):
            event(
                status=StageStatus.SUCCEEDED,
                produced=(("b", "id_2"), ("a", "id_1")),
            )

    def test_one_name_produces_one_id(self) -> None:
        with pytest.raises(ValueError, match="one id per name"):
            event(
                status=StageStatus.SUCCEEDED,
                produced=(("a", "id_1"), ("a", "id_2")),
            )

    def test_the_first_event_names_no_predecessor(self) -> None:
        with pytest.raises(ValueError, match="names its predecessor"):
            event(previous_event_id="sevt_earlier")

    def test_every_later_event_names_one(self) -> None:
        with pytest.raises(ValueError, match="names its predecessor"):
            event(sequence=1)

    def test_detail_is_a_breath_not_an_account(self) -> None:
        with pytest.raises(ValueError, match="at most"):
            event(detail="x" * (MAX_DETAIL_CHARS + 1))

    def test_identity_covers_the_content(self) -> None:
        assert event().id != event(detail="something happened").id
        assert event().id == event().id

    def test_spend_cannot_be_negative(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            StageSpend(model_calls=-1)


class TestTheLog:
    def test_a_fresh_log_is_empty(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        assert log.events() == ()
        assert log.facts() == ChainFacts()
        assert log.spend().is_zero

    def test_events_are_sequenced_and_chained(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")

        first, second = log.events()
        assert (first.sequence, second.sequence) == (0, 1)
        assert first.previous_event_id == ""
        assert second.previous_event_id == first.id

    def test_a_second_process_reads_the_same_log(self, tmp_path: Path) -> None:
        walked(
            StageLog(tmp_path, INVESTIGATION),
            StageName.MAPPING,
            "brf_1",
            assessment_id="madq_1",
        )

        reopened = StageLog(tmp_path, INVESTIGATION)

        assert len(reopened.events()) == 2
        assert reopened.facts().require(Fact.ASSESSMENT_ID) == "madq_1"

    def test_no_scratch_file_is_left_behind(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")

        assert list(log.directory.glob("*.tmp")) == []

    def test_spend_totals_over_the_whole_investigation(
        self, tmp_path: Path
    ) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")
        walked(log, StageName.IDEATION, "idir_1", ideation_run_record_id="irun_1")

        assert log.spend() == StageSpend(
            model_calls=6, input_tokens=2400, output_tokens=680
        )


class TestWhatItCatches:
    def test_an_edited_event(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")
        path = log.directory / "000001.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["produced"]["assessment_id"] = "madq_forged"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(StageLogIntegrityError, match="was edited"):
            log.events()

    def test_a_deleted_event(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")
        walked(log, StageName.IDEATION, "idir_1", ideation_run_record_id="irun_1")
        (log.directory / "000001.json").unlink()

        with pytest.raises(StageLogIntegrityError, match="missing or misnamed"):
            log.events()

    def test_an_event_from_another_investigation(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")
        stranger = StageEvent(
            investigation_id="inv_elsewhere",
            sequence=2,
            stage=StageName.IDEATION,
            status=StageStatus.SUCCEEDED,
            key="ideation:idir_1",
            subject_id="idir_1",
            previous_event_id=log.events()[-1].id,
        )
        (log.directory / "000002.json").write_text(
            json.dumps(
                {
                    "id": stranger.id,
                    "investigation_id": stranger.investigation_id,
                    "sequence": 2,
                    "stage": str(stranger.stage),
                    "status": str(stranger.status),
                    "key": stranger.key,
                    "subject_id": stranger.subject_id,
                    "produced": {},
                    "spend": {
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                    "detail": "",
                    "previous_event_id": stranger.previous_event_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        with pytest.raises(StageLogIntegrityError, match="belongs to"):
            log.events()

    def test_unreadable_json(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")
        (log.directory / "000000.json").write_text("{", encoding="utf-8")

        with pytest.raises(StageLogIntegrityError, match="not valid JSON"):
            log.events()


class TestWhatResumeNeeds:
    def test_a_completed_stage_is_found_by_its_key(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")

        found = log.terminal_for("mapping:brf_1")

        assert found is not None
        assert found.status is StageStatus.SUCCEEDED
        assert log.terminal_for("mapping:brf_other") is None

    def test_a_crash_leaves_one_unfinished_event(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")
        log.append(
            stage=StageName.IDEATION,
            status=StageStatus.RUNNING,
            key="ideation:idir_1",
            subject_id="idir_1",
        )

        unfinished = StageLog(tmp_path, INVESTIGATION).unfinished()

        assert unfinished is not None
        assert unfinished.stage is StageName.IDEATION

    def test_a_finished_stage_leaves_none(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(log, StageName.MAPPING, "brf_1", assessment_id="madq_1")

        assert log.unfinished() is None

    def test_a_failed_stage_answers_its_running_event(
        self, tmp_path: Path
    ) -> None:
        """A failure is a finished attempt, not an interrupted one; only
        the crash signature is worth reconciling."""
        log = StageLog(tmp_path, INVESTIGATION)
        log.append(
            stage=StageName.MAPPING,
            status=StageStatus.RUNNING,
            key="mapping:brf_1",
            subject_id="brf_1",
        )
        log.append(
            stage=StageName.MAPPING,
            status=StageStatus.FAILED,
            key="mapping:brf_1",
            subject_id="brf_1",
            detail="the provider refused every call",
        )

        assert log.unfinished() is None

    def test_replaying_the_log_rebuilds_every_id(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        walked(
            log,
            StageName.MAPPING,
            "brf_1",
            assessment_id="madq_1",
            map_run_record_id="map_1",
        )
        walked(
            log,
            StageName.IDEATION,
            "idir_1",
            ideation_run_record_id="irun_1",
        )

        facts = StageLog(tmp_path, INVESTIGATION).facts()

        assert facts.require(Fact.ASSESSMENT_ID) == "madq_1"
        assert facts.require(Fact.MAP_RUN_RECORD_ID) == "map_1"
        assert facts.require(Fact.IDEATION_RUN_RECORD_ID) == "irun_1"

    def test_a_failed_stage_contributes_no_facts(self, tmp_path: Path) -> None:
        log = StageLog(tmp_path, INVESTIGATION)
        log.append(
            stage=StageName.MAPPING,
            status=StageStatus.FAILED,
            key="mapping:brf_1",
            subject_id="brf_1",
            detail="ran out of model calls",
        )

        with pytest.raises(MissingFactError, match="assessment_id"):
            log.facts().require(Fact.ASSESSMENT_ID)
