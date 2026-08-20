"""Starting a funded run: the door, the preflight, and the bridge.

The bridge is the point of this package. An admitted state is a genesis
state with no budget, and its content id excludes the budget, so funding
it in place would produce two different snapshots claiming one id. Here
the supported path is pinned end to end: the door refuses everything
that is not a matching admission, and the bridge funds a *successor*,
leaving the admitted snapshot exactly as it found it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.admission.directive import AdmissionDirective
from autonomous_research_lab.admission.records import (
    MECHANICAL_READING,
    AdmissionRecord,
    GroundedSupport,
    OperationalPrediction,
    Requirement,
    RequirementSource,
    SupportSource,
)
from autonomous_research_lab.admission.store import (
    AdmissionIntegrityError,
    AdmissionStore,
)
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.mapping.records import CallProvenance
from autonomous_research_lab.program.authorization import FundingAuthorization
from autonomous_research_lab.program.directive import RunDirective
from autonomous_research_lab.program.door import (
    RunRefusedError,
    require_admitted_state_for_run,
)
from autonomous_research_lab.program.preflight import (
    RunPreflightError,
    check_funding_coherence,
)
from autonomous_research_lab.program.starter import start_run
from autonomous_research_lab.program.store import ProgramStore

GRANT = ResearchBudget(wall_clock_seconds=1000.0, usd=100.0, model_tokens=10_000)


def admitted_state() -> ResearchState:
    question = ResearchQuestion(
        text="Do learned scalars concentrate on induction heads?",
        importance="Serves the adaptation-mechanisms topic.",
    )
    hypothesis = Hypothesis(
        statement="Scalar magnitude correlates with induction-head score.",
        rationale="Reweighting amplifies specialized heads.",
        question_id=question.id,
    )
    prediction = Prediction(
        hypothesis_id=hypothesis.id,
        condition="When evaluating overlap across tasks and seeds.",
        metric="difference in overlap: top-weighted minus bottom-weighted",
        comparator=Comparator.GREATER_THAN,
        threshold=0.0,
        expectation="Top-weighted heads overlap with induction heads.",
    )
    return ResearchState(
        objective="Measure the correlation.",
        questions=(question,),
        hypotheses=(hypothesis,),
        predictions=(prediction,),
    )


def admission_record(state: ResearchState, **overrides: object) -> AdmissionRecord:
    values: dict[str, object] = {
        "run_id": "adm_0000000000000001",
        "directive_id": AdmissionDirective(
            selection_run_record_id="srun_0000000000000001",
            scheduling_requirement="Batch-scheduled execution.",
            job_duration_requirement="Jobs bounded to two days.",
            checkpoint_requirement="Checkpoint and resume required.",
        ).id,
        "selection_run_record_id": "srun_0000000000000001",
        "selection_run_id": "sel_0000000000000001",
        "selection_directive_id": "sdir_0000000000000001",
        "prior_art_run_record_id": "prun_0000000000000001",
        "prior_art_run_id": "pac_0000000000000001",
        "selected_prior_art_assessment_id": "paa_0000000000000001",
        "ideation_run_record_id": "irun_0000000000000001",
        "ideation_run_id": "idg_0000000000000001",
        "direction_id": "dir_0000000000000001",
        "snapshot_id": "cfp_0000000000000001",
        "map_run_id": "map_0000000000000001",
        "map_assessment_id": "madq_000000000000001",
        "selected_candidate_id": "idea_0000000000000001",
        "operational_predictions": (
            OperationalPrediction(
                prediction_text="Top-weighted heads overlap with induction heads.",
                condition="When evaluating overlap across tasks and seeds.",
                base_metric="overlap of important heads",
                expected_higher_arm="top-weighted heads",
                expected_lower_arm="bottom-weighted heads",
                contrary_observation="no difference in overlap",
                support=(
                    GroundedSupport(
                        source=SupportSource.CANDIDATE,
                        field_path="predictions[0].text",
                        quote="overlap strongly with high induction heads",
                    ),
                ),
            ),
        ),
        "measurements": ("overlap of important heads",),
        "controls": ("random head ablation",),
        "comparison_targets": ("LoRA fine-tuning",),
        "evaluation_protocol": "Seeded runs across tasks and seeds.",
        "inherited_requirements": (
            Requirement(
                source=RequirementSource.CANDIDATE_RESOURCES,
                record_id="idea_0000000000000001",
                field_path="resources.compute",
                quote="~100 GPU-hours on a single A100",
            ),
        ),
        "operator_requirements": (
            Requirement(
                source=RequirementSource.ADMISSION_DIRECTIVE,
                record_id="adir_0000000000000001",
                field_path="scheduling_requirement",
                quote="Batch-scheduled execution.",
            ),
        ),
        "mechanical_reading": MECHANICAL_READING,
        "question_id": state.questions[0].id,
        "hypothesis_id": state.hypotheses[0].id,
        "prediction_ids": tuple(p.id for p in state.predictions),
        "state_id": state.id,
        "provenance": CallProvenance(
            request_fingerprint="mreq_0000000000000001",
            response_id="mcall_000000000000001",
            provider="fake",
            requested_model="model-x",
            served_model="model-x",
            provider_request_id="req-1",
            latency_seconds=0.1,
            input_tokens=100,
            output_tokens=50,
            repair_count=0,
        ),
        "model_calls": 1,
        "input_tokens": 100,
        "output_tokens": 50,
    }
    values.update(overrides)
    return AdmissionRecord(**values)  # type: ignore[arg-type]


def seed_admission(
    root: Path, state: ResearchState | None = None
) -> tuple[AdmissionStore, AdmissionRecord, ResearchState]:
    store = AdmissionStore(root / "admission")
    admitted = state if state is not None else admitted_state()
    store.persist_state(admitted)
    record = store.record_admission(admission_record(admitted))
    return store, record, admitted


def authorization(record_id: str, **overrides: object) -> FundingAuthorization:
    values: dict[str, object] = {
        "admission_record_id": record_id,
        "granted": GRANT,
        "authority": "Lab operator, standing August allocation.",
    }
    values.update(overrides)
    return FundingAuthorization(**values)  # type: ignore[arg-type]


def directive(
    record_id: str, authorization_id: str, label: str = "first run"
) -> RunDirective:
    return RunDirective(
        admission_record_id=record_id,
        authorization_id=authorization_id,
        label=label,
    )


class TestTheDoor:
    def test_a_matching_admission_and_grant_pass(self, tmp_path: Path) -> None:
        store, record, admitted = seed_admission(tmp_path)
        grant = authorization(record.id)

        inputs = require_admitted_state_for_run(
            store, directive(record.id, grant.id), grant
        )

        assert inputs.admission == record
        assert inputs.admitted_state == admitted

    def test_an_unknown_admission_is_refused(self, tmp_path: Path) -> None:
        store, _, _ = seed_admission(tmp_path)
        grant = authorization("arun_nope")

        with pytest.raises(RunRefusedError, match="no admission record"):
            require_admitted_state_for_run(
                store, directive("arun_nope", grant.id), grant
            )

    def test_a_grant_for_another_admission_is_refused(
        self, tmp_path: Path
    ) -> None:
        store, record, _ = seed_admission(tmp_path)
        grant = authorization("arun_other")

        with pytest.raises(RunRefusedError, match="funds admission"):
            require_admitted_state_for_run(
                store, directive(record.id, grant.id), grant
            )

    def test_a_directive_naming_another_grant_is_refused(
        self, tmp_path: Path
    ) -> None:
        store, record, _ = seed_admission(tmp_path)
        grant = authorization(record.id)
        other = authorization(record.id, authority="Someone else.")

        with pytest.raises(RunRefusedError, match="but was handed"):
            require_admitted_state_for_run(
                store, directive(record.id, other.id), grant
            )

    def test_a_record_disagreeing_with_its_own_state_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The admission accessor proves the record and the snapshot are
        one artifact set. The door proves the record's stamps describe
        what the snapshot actually holds."""
        store = AdmissionStore(tmp_path / "admission")
        admitted = admitted_state()
        store.persist_state(admitted)
        record = store.record_admission(
            admission_record(admitted, prediction_ids=("pred_someone_else",))
        )
        grant = authorization(record.id)

        with pytest.raises(RunRefusedError, match="disagrees with its own state"):
            require_admitted_state_for_run(
                store, directive(record.id, grant.id), grant
            )

    def test_an_already_funded_seed_is_refused(self, tmp_path: Path) -> None:
        """The admission accessor refuses a non-zero budget before the
        door reaches its own check, and the integrity error is left to
        travel as itself: a doctored admission store is broken, not a
        run that happens to be refused."""
        store = AdmissionStore(tmp_path / "admission")
        funded = admitted_state().fund(GRANT)
        store.persist_state(funded)
        record = store.record_admission(admission_record(funded))
        grant = authorization(record.id)

        with pytest.raises(AdmissionIntegrityError, match="non-zero budget"):
            require_admitted_state_for_run(
                store, directive(record.id, grant.id), grant
            )


class TestPreflight:
    def test_a_grant_that_buys_something_passes(self, tmp_path: Path) -> None:
        grant = authorization("arun_1")

        plan = check_funding_coherence(
            directive=directive("arun_1", grant.id),
            authorization=grant,
            minimum_first_step=ResourceCost(usd=5.0),
        )

        assert plan.authorization_id == grant.id

    def test_an_empty_grant_is_refused(self, tmp_path: Path) -> None:
        grant = authorization("arun_1", granted=ResearchBudget.zero())

        with pytest.raises(RunPreflightError, match="empty"):
            check_funding_coherence(
                directive=directive("arun_1", grant.id), authorization=grant
            )

    def test_a_grant_below_the_caller_floor_is_refused(
        self, tmp_path: Path
    ) -> None:
        grant = authorization("arun_1")

        with pytest.raises(RunPreflightError, match="cheapest first step"):
            check_funding_coherence(
                directive=directive("arun_1", grant.id),
                authorization=grant,
                minimum_first_step=ResourceCost(gpu_hours=4.0),
            )


class TestStartingARun:
    def test_the_funded_state_is_a_successor_of_the_admitted_one(
        self, tmp_path: Path
    ) -> None:
        admission_store, record, admitted = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)

        result = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        assert not result.replayed
        assert result.funded_state.parent_id == admitted.id
        assert result.funded_state.id != admitted.id
        assert result.funded_state.budget == GRANT
        assert result.run.admitted_state_id == admitted.id
        assert result.run.funded_state_id == result.funded_state.id

    def test_the_scientific_content_crosses_unchanged(
        self, tmp_path: Path
    ) -> None:
        admission_store, record, admitted = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)

        result = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        funded = result.funded_state
        assert funded.questions == admitted.questions
        assert funded.hypotheses == admitted.hypotheses
        assert funded.predictions == admitted.predictions
        assert funded.results == () and funded.assessments == ()

    def test_the_admission_artifacts_are_untouched(self, tmp_path: Path) -> None:
        admission_store, record, _ = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)
        before = {
            path: path.read_bytes()
            for path in sorted((tmp_path / "admission").rglob("*"))
            if path.is_file()
        }

        start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        after = {
            path: path.read_bytes()
            for path in sorted((tmp_path / "admission").rglob("*"))
            if path.is_file()
        }
        assert before == after
        _, reloaded = admission_store.get_admitted_state(record.id)
        assert reloaded.budget.is_exhausted

    def test_the_grant_is_ledger_entry_zero(self, tmp_path: Path) -> None:
        admission_store, record, _ = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)

        result = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        ledger = program_store.ledger_for(result.run.run_id)
        entries = ledger.entries()
        assert len(entries) == 1
        assert entries[0].id == result.run.grant_entry_id
        assert ledger.balance() == GRANT
        assert ledger.balance() == result.funded_state.budget

    def test_the_run_root_holds_the_whole_lineage(self, tmp_path: Path) -> None:
        admission_store, record, admitted = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)

        result = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        states = program_store.state_store()
        assert set(states.state_ids()) == {admitted.id, result.funded_state.id}
        assert states.load(admitted.id).budget.is_exhausted

    def test_the_run_reloads_from_a_fresh_store(self, tmp_path: Path) -> None:
        admission_store, record, _ = seed_admission(tmp_path)
        grant = authorization(record.id)
        result = start_run(
            admission_store=admission_store,
            program_store=ProgramStore(tmp_path / "run"),
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        fresh = ProgramStore(tmp_path / "run")
        run, state = fresh.get_funded_state(result.run.id)

        assert run == result.run
        assert state == result.funded_state
        assert fresh.ledger_for(run.run_id).balance() == GRANT

    def test_a_refused_door_writes_nothing(self, tmp_path: Path) -> None:
        admission_store, record, _ = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization("arun_other")

        with pytest.raises(RunRefusedError):
            start_run(
                admission_store=admission_store,
                program_store=program_store,
                directive=directive(record.id, grant.id),
                authorization=grant,
            )

        assert program_store.runs() == ()
        assert program_store.state_store().state_ids() == ()

    def test_a_refused_preflight_writes_nothing(self, tmp_path: Path) -> None:
        admission_store, record, _ = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)

        with pytest.raises(RunPreflightError):
            start_run(
                admission_store=admission_store,
                program_store=program_store,
                directive=directive(record.id, grant.id),
                authorization=grant,
                minimum_first_step=ResourceCost(gpu_hours=99.0),
            )

        assert program_store.runs() == ()
        assert program_store.state_store().state_ids() == ()


class TestReplayAndSecondRuns:
    def test_re_running_a_completed_directive_replays_it(
        self, tmp_path: Path
    ) -> None:
        admission_store, record, _ = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)
        first = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        second = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        assert second.replayed
        assert second.run == first.run
        assert second.funded_state == first.funded_state
        assert len(program_store.runs()) == 1
        assert program_store.ledger_for(first.run.run_id).balance() == GRANT

    def test_a_second_stated_run_is_a_new_run(self, tmp_path: Path) -> None:
        """Two runs over one admission are allowed, and each is stated:
        a second label, or a second grant. What is not allowed is running
        the same command twice and getting two runs by accident."""
        admission_store, record, _ = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)
        first = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        second = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id, label="replication"),
            authorization=grant,
        )

        assert not second.replayed
        assert second.run.run_id != first.run.run_id
        assert second.funded_state == first.funded_state  # same content
        assert len(program_store.runs_for_admission(record.id)) == 2
        assert (
            program_store.ledger_for(second.run.run_id).entries()[0].id
            != program_store.ledger_for(first.run.run_id).entries()[0].id
        )

    def test_an_orphan_snapshot_without_an_envelope_is_not_a_run(
        self, tmp_path: Path
    ) -> None:
        """No envelope means no run: a crash after the snapshot leaves an
        inert file, and the re-run starts honestly."""
        admission_store, record, admitted = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)
        orphan = admitted.fund(GRANT)
        program_store.persist_state(orphan)

        assert program_store.runs() == ()
        assert program_store.run_for_directive(
            directive(record.id, grant.id).id
        ) is None

        result = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        assert not result.replayed
        assert result.funded_state.id == orphan.id  # identical content
        assert program_store.ledger_for(result.run.run_id).balance() == GRANT

    def test_the_envelope_records_the_lineage_it_was_funded_from(
        self, tmp_path: Path
    ) -> None:
        admission_store, record, admitted = seed_admission(tmp_path)
        program_store = ProgramStore(tmp_path / "run")
        grant = authorization(record.id)

        result = start_run(
            admission_store=admission_store,
            program_store=program_store,
            directive=directive(record.id, grant.id),
            authorization=grant,
        )

        run = result.run
        assert run.admission_record_id == record.id
        assert run.question_id == admitted.questions[0].id
        assert run.hypothesis_id == admitted.hypotheses[0].id
        assert run.prediction_ids == tuple(p.id for p in admitted.predictions)
        assert run.authority == grant.authority
        assert json.loads(
            (tmp_path / "run" / "runs" / f"{run.id}.json").read_text()
        )["run_id"] == run.run_id
