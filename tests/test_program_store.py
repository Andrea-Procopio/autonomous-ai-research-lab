"""The run store: write-once, tamper-loud, one envelope per directive."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.persistence.state_store import SnapshotError
from autonomous_research_lab.program.authorization import FundingAuthorization
from autonomous_research_lab.program.directive import RunDirective
from autonomous_research_lab.program.records import ResearchRun
from autonomous_research_lab.program.store import (
    ProgramConflictError,
    ProgramIntegrityError,
    ProgramStore,
)

GRANT = ResearchBudget(wall_clock_seconds=1000.0, usd=100.0, model_tokens=10_000)


def directive(label: str = "first run") -> RunDirective:
    return RunDirective(
        admission_record_id="arun_1", authorization_id="fund_1", label=label
    )


def authorization() -> FundingAuthorization:
    return FundingAuthorization(
        admission_record_id="arun_1", granted=GRANT, authority="Lab operator."
    )


def make_run(**overrides: object) -> ResearchRun:
    fields: dict[str, object] = {
        "run_id": "run_1",
        "directive_id": directive().id,
        "authorization_id": authorization().id,
        "admission_record_id": "arun_1",
        "admitted_state_id": "st_admitted",
        "funded_state_id": "st_funded",
        "granted": GRANT,
        "grant_entry_id": "bent_0",
        "label": "first run",
        "authority": "Lab operator.",
        "question_id": "q_1",
        "hypothesis_id": "hyp_1",
        "prediction_ids": ("pred_1",),
    }
    fields.update(overrides)
    return ResearchRun(**fields)  # type: ignore[arg-type]


class TestWriteOnce:
    def test_records_round_trip(self, tmp_path: Path) -> None:
        store = ProgramStore(tmp_path)

        store.record_directive(directive())
        store.record_authorization(authorization())
        store.record_run(make_run())

        assert store.get_directive(directive().id) == directive()
        assert store.get_authorization(authorization().id) == authorization()
        assert store.get_run(make_run().id) == make_run()

    def test_unknown_ids_are_absent_not_invented(self, tmp_path: Path) -> None:
        store = ProgramStore(tmp_path)
        assert store.get_directive("rdir_nope") is None
        assert store.get_authorization("fund_nope") is None
        assert store.get_run("rune_nope") is None

    def test_identical_re_recording_is_a_no_op(self, tmp_path: Path) -> None:
        store = ProgramStore(tmp_path)
        store.record_run(make_run())
        store.record_run(make_run())
        assert len(store.runs()) == 1

    def test_a_tampered_record_fails_to_load(self, tmp_path: Path) -> None:
        store = ProgramStore(tmp_path)
        run = store.record_run(make_run())
        path = tmp_path / "envelopes" / f"{run.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["label"] = "a different run"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(ProgramIntegrityError, match="re-derives"):
            store.get_run(run.id)

    def test_a_directive_starts_at_most_one_run(self, tmp_path: Path) -> None:
        store = ProgramStore(tmp_path)
        store.record_run(make_run())

        with pytest.raises(ProgramConflictError, match="replays that run"):
            store.record_run(make_run(run_id="run_2"))

    def test_a_run_id_is_enveloped_once(self, tmp_path: Path) -> None:
        store = ProgramStore(tmp_path)
        store.record_run(make_run())

        with pytest.raises(ProgramConflictError, match="enveloped once"):
            store.record_run(make_run(directive_id=directive("second").id))

    def test_one_admission_may_back_several_runs(self, tmp_path: Path) -> None:
        store = ProgramStore(tmp_path)
        store.record_run(make_run())
        store.record_run(
            make_run(run_id="run_2", directive_id=directive("second").id)
        )

        assert len(store.runs_for_admission("arun_1")) == 2
        assert store.run_for_directive(directive().id) is not None
        assert store.run_for_directive("rdir_nope") is None


class TestStatesThroughTheirRun:
    def test_the_funded_state_loads_through_its_envelope(
        self, tmp_path: Path
    ) -> None:
        store = ProgramStore(tmp_path)
        funded = ResearchState(objective="o").fund(GRANT)
        store.persist_state(funded)
        run = store.record_run(make_run(funded_state_id=funded.id))

        loaded_run, loaded_state = store.get_funded_state(run.id)

        assert loaded_run == run
        assert loaded_state == funded

    def test_a_state_without_a_run_is_never_exposed(self, tmp_path: Path) -> None:
        store = ProgramStore(tmp_path)

        with pytest.raises(ProgramIntegrityError, match="never exposed"):
            store.get_funded_state("rune_nope")

    def test_a_missing_snapshot_fails_loudly(self, tmp_path: Path) -> None:
        store = ProgramStore(tmp_path)
        run = store.record_run(make_run())

        with pytest.raises(SnapshotError, match="no snapshot"):
            store.get_funded_state(run.id)

    def test_a_doctored_budget_is_caught_against_the_envelope(
        self, tmp_path: Path
    ) -> None:
        """A state's content id excludes its budget, so a doctored budget
        reloads silently. The envelope records what was granted, which is
        what makes the check exact."""
        store = ProgramStore(tmp_path)
        doctored = ResearchState(objective="o").fund(ResearchBudget(usd=1.0))
        store.persist_state(doctored)
        run = store.record_run(make_run(funded_state_id=doctored.id))

        with pytest.raises(ProgramIntegrityError, match="granted"):
            store.get_funded_state(run.id)

    def test_the_snapshot_is_read_back_before_it_is_referenced(
        self, tmp_path: Path
    ) -> None:
        store = ProgramStore(tmp_path)
        funded = ResearchState(objective="o").fund(GRANT)
        store.persist_state(funded)
        path = tmp_path / "states" / f"{funded.id}.json"
        path.write_text(path.read_text(encoding="utf-8")[:40], encoding="utf-8")

        with pytest.raises(SnapshotError, match="never rewritten"):
            store.persist_state(funded)
