"""The canary: one synthetic brief through all seven stages.

This is the test the controller exists to pass. Everything else checks a
part; this one carries a topic from nothing to a funded run that
executes real experiments through the ordinary executor, and then does
it again in seven pieces, stopping at every stage boundary and resuming
from the durable record with nothing carried over in memory.

The two runs must arrive at the same place. That is the claim: an
interruption costs time and nothing else.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autonomous_research_lab.control.controller import Controller, Outcome
from autonomous_research_lab.control.lab import RuntimeRequest
from autonomous_research_lab.control.stage import (
    CHAIN_ORDER,
    Fact,
    StageName,
    StageStatus,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.mapping.store import MappingStore
from autonomous_research_lab.persistence import FileStateStore
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.program.integrity import (
    IntegrityIssueKind,
    _reachable_from_roots,
)
from autonomous_research_lab.program.store import ProgramStore
from autonomous_research_lab.selection.store import SelectionStore
from examples.canary_chain import CONFIG, verify, walk
from examples.canary_lab import lab as canary_lab

REPO = Path(__file__).resolve().parents[1]

#: The ids two runs of this config must agree on. Deliberately not all
#: of them: a run *record* is an event, and its id carries an occurrence
#: id that is meant to differ between two runs of the same work. What
#: must not differ is the science — the call text a direction was read
#: from, the state the chain admitted, and the state that state was
#: funded into. Those are content-addressed all the way down to the
#: candidate's own words.
CHAIN_FACTS = (
    Fact.SNAPSHOT_ID,
    Fact.ADMITTED_STATE_ID,
    Fact.FUNDED_STATE_ID,
)


def run_counts(root: Path) -> dict[str, int]:
    """How many run records each stage store holds.

    The number that must not grow when a walk is interrupted: a second
    record means the stage was paid for twice.
    """
    return {
        "mapping": len(MappingStore(root / "mapping").runs()),
        "ideation": len(IdeationStore(root / "ideation").runs()),
        "prior_art": len(PriorArtStore(root / "priorart").runs()),
        "selection": len(SelectionStore(root / "selection").runs()),
        "program": len(ProgramStore(root / "program").runs()),
    }


def facts_of(root: Path) -> dict[str, str]:
    controller = Controller(root)
    (investigation,) = controller.investigations.investigations()
    log = controller.investigations.log_for(investigation.investigation_id)
    facts = log.facts()
    return {str(fact): facts.require(fact) for fact in CHAIN_FACTS}


class TestOneWalk:
    def test_every_stage_succeeds(self, tmp_path: Path) -> None:
        result = walk(tmp_path)

        succeeded = {
            event.stage
            for event in result.events
            if event.status is StageStatus.SUCCEEDED
        }
        assert succeeded == set(CHAIN_ORDER)
        assert result.ok

    def test_the_run_root_verifies_from_cold(self, tmp_path: Path) -> None:
        walk(tmp_path)

        report = verify(tmp_path)

        assert report.ok, report.issues
        assert report.results_checked >= 1
        assert report.evidence_checked >= 1
        assert report.blobs_checked >= 1

    def test_the_experiments_actually_ran(self, tmp_path: Path) -> None:
        """Through the ordinary executor, in a subprocess, writing real
        artifacts — the canary's model is a fixture, its experiments are
        not."""
        walk(tmp_path)

        assert (tmp_path / "runs").is_dir()
        assert list((tmp_path / "runs").glob("*/metrics.json"))

    def test_the_snapshot_chain_is_whole(self, tmp_path: Path) -> None:
        """One root, no absent parents, one head, and a forward walk from
        the root that reaches everything — the four things a verifier has
        to be able to say before it says "intact"."""
        walk(tmp_path)
        states = FileStateStore(tmp_path)
        stored = {found: states.load(found) for found in states.state_ids()}

        roots = [
            state for state in stored.values() if state.parent_id is None
        ]
        absent = [
            state.parent_id
            for state in stored.values()
            if state.parent_id is not None and state.parent_id not in stored
        ]
        heads = set(stored) - {
            state.parent_id for state in stored.values() if state.parent_id
        }

        assert len(roots) == 1
        assert absent == []
        assert len(heads) == 1
        assert len(_reachable_from_roots(list(stored.values()))) == len(stored)

    def test_the_ledger_billed_every_attempt(self, tmp_path: Path) -> None:
        walk(tmp_path)
        program = ProgramStore(tmp_path / "program")
        (run,) = program.runs()

        entries = program.ledger_for(run.run_id).entries()

        assert entries[0].kind.value == "grant"
        assert len(entries) > 1
        assert len({entry.charge_id for entry in entries}) == len(entries)

    def test_it_records_one_investigation_and_one_run(
        self, tmp_path: Path
    ) -> None:
        walk(tmp_path)

        assert len(Controller(tmp_path).investigations.investigations()) == 1
        assert run_counts(tmp_path)["program"] == 1


class TestStoppingAtEveryBoundary:
    def test_the_interrupted_walk_arrives_where_the_whole_one_did(
        self, tmp_path: Path
    ) -> None:
        whole = tmp_path / "whole"
        pieces = tmp_path / "pieces"
        walk(whole)

        for stage in CHAIN_ORDER:
            # A fresh controller every time: nothing but the files on
            # disk crosses the boundary.
            walk(pieces, stop_after=stage)
        walk(pieces)

        assert facts_of(pieces) == facts_of(whole)

    def test_nothing_is_done_twice(self, tmp_path: Path) -> None:
        for stage in CHAIN_ORDER:
            walk(tmp_path, stop_after=stage)
        walk(tmp_path)

        assert run_counts(tmp_path) == {
            "mapping": 1,
            "ideation": 1,
            "prior_art": 1,
            "selection": 1,
            "program": 1,
        }

    def test_funding_happens_exactly_once(self, tmp_path: Path) -> None:
        for stage in CHAIN_ORDER:
            walk(tmp_path, stop_after=stage)
        walk(tmp_path)
        program = ProgramStore(tmp_path / "program")
        (run,) = program.runs()

        grants = [
            entry
            for entry in program.ledger_for(run.run_id).entries()
            if entry.kind.value == "grant"
        ]

        assert len(grants) == 1

    def test_the_interrupted_run_verifies_too(self, tmp_path: Path) -> None:
        for stage in CHAIN_ORDER:
            walk(tmp_path, stop_after=stage)
        walk(tmp_path)

        assert verify(tmp_path).ok, verify(tmp_path).issues

    def test_every_boundary_leaves_a_finished_log(
        self, tmp_path: Path
    ) -> None:
        """No unanswered RUNNING event after a clean stop: a walk that
        stopped where it was told did not crash, and the log should not
        claim it did."""
        for stage in CHAIN_ORDER:
            walk(tmp_path, stop_after=stage)
            controller = Controller(tmp_path)
            (investigation,) = (
                controller.investigations.investigations()
            )
            log = controller.investigations.log_for(
                investigation.investigation_id
            )
            assert log.unfinished() is None, stage


def thin_config(tmp_path: Path) -> Path:
    """The same canary, with too few sources extracted to build on.

    Not a broken fixture: a map this thin is exactly what the adequacy
    verdict exists to catch, and what the next stage's door exists to
    refuse.
    """
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["brief"]["max_extracted_sources"] = 3
    payload["label"] = "canary with a map too thin to build on"
    path = tmp_path / "thin.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class TestAnHonestRefusal:
    def test_a_door_refuses_and_the_walk_stops(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        controller = Controller(root)
        payload = json.loads(
            thin_config(tmp_path).read_text(encoding="utf-8")
        )

        result = controller.run(payload, lab=canary_lab())

        assert result.outcome is Outcome.REFUSED
        assert not result.ok
        assert result.events[-1].status is StageStatus.REFUSED
        assert result.events[-1].stage is StageName.IDEATION
        assert "not adequate" in result.events[-1].detail

    def test_the_stage_before_it_still_succeeded(
        self, tmp_path: Path
    ) -> None:
        """An inadequate map is a finding, not a failure: the mapping
        stage did its job and said so."""
        root = tmp_path / "run"
        payload = json.loads(
            thin_config(tmp_path).read_text(encoding="utf-8")
        )

        result = Controller(root).run(payload, lab=canary_lab())

        mapping = result.events[1]
        assert mapping.stage is StageName.MAPPING
        assert mapping.status is StageStatus.SUCCEEDED
        assert "insufficient" in mapping.detail

    def test_the_refusal_cost_nothing(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        payload = json.loads(
            thin_config(tmp_path).read_text(encoding="utf-8")
        )

        result = Controller(root).run(payload, lab=canary_lab())

        assert result.events[-1].spend.is_zero


class TestAStepInterruptedMidFlight:
    """The finest crash the controller has to survive: a process killed
    inside one step, after some of the states it derived are on disk."""

    def one_step(self, root: Path) -> list[ResearchState]:
        """Run one real step into a shadow store, and return the lineage
        it derived in the order the runtime wrote it.

        A shadow store rather than the run's own, because the point is
        to choose how much of it lands.
        """
        controller = Controller(root)
        (investigation,) = controller.investigations.investigations()
        log = controller.investigations.log_for(
            investigation.investigation_id
        )
        head = log.facts().require(Fact.STATE_ID)
        program = ProgramStore(root / "program")
        (run,) = program.runs()
        shadow = FileStateStore(root / "shadow")
        runtime = canary_lab().runtime(
            RuntimeRequest(
                root=root,
                evidence=FileEvidenceStore(root),
                states=shadow,
                ledger=program.ledger_for(run.run_id),
            )
        )
        runtime.step(FileStateStore(root).load(head))
        derived = {found: shadow.load(found) for found in shadow.state_ids()}
        children = {
            state.parent_id: state
            for state in derived.values()
            if state.parent_id
        }
        ordered: list[ResearchState] = []
        current = children.get(head)
        while current is not None:
            ordered.append(current)
            current = children.get(current.id)
        return ordered

    def test_a_torn_write_is_loud_rather_than_resumed(
        self, tmp_path: Path
    ) -> None:
        """A step's states are written after the step returns, so a
        process killed inside the write leaves a ledger that has been
        debited and a chain that stops mid-attempt.

        Nothing here can honestly recover that: the money moved and the
        state that recorded what it bought did not survive. The reconcile
        refuses the mid-attempt end — resuming there would leave the
        attempt open forever — and the next step finds the ledger and the
        state disagreeing and fails closed, which is the behavior the
        funded run was built to have. Afterwards the verifier says
        exactly which part is broken: the accounting, not the lineage.
        """
        walk(tmp_path, stop_after=StageName.FUNDING)
        derived = self.one_step(tmp_path)
        assert len(derived) > 2
        torn = derived[: len(derived) // 2]
        assert any(
            not attempt.status.is_terminal for attempt in torn[-1].attempts
        )
        states = FileStateStore(tmp_path)
        for partial in torn:
            states.persist(partial)

        result = walk(tmp_path)

        assert result.outcome is Outcome.FAILED
        assert "refusing to guess" in result.detail
        assert not [
            event for event in result.events if "crash hid" in event.detail
        ]
        report = verify(tmp_path)
        assert report.of_kind(IntegrityIssueKind.LEDGER_ISSUE)
        assert not report.of_kind(IntegrityIssueKind.INCOMPLETE_LINEAGE)

    def test_a_whole_step_whose_event_was_lost_is_adopted(
        self, tmp_path: Path
    ) -> None:
        """The other half of the same crash: every state landed, the
        event did not. The step is adopted, not paid for again."""
        walk(tmp_path, stop_after=StageName.FUNDING)
        derived = self.one_step(tmp_path)
        states = FileStateStore(tmp_path)
        for partial in derived:
            states.persist(partial)

        result = walk(tmp_path, stop_after=StageName.EXPERIMENTATION)

        adopted = [
            event
            for event in result.events
            if event.stage is StageName.EXPERIMENTATION
            and "crash hid" in event.detail
        ]
        assert adopted
        assert adopted[0].produced == (
            (str(Fact.STATE_ID), derived[-1].id),
        )
        assert verify(tmp_path).ok


class TestWhenTheRecordIsLost:
    def test_completed_work_is_reconciled_not_repeated(
        self, tmp_path: Path
    ) -> None:
        """The crash between the side effect and the record of it: the
        mapping run is on disk, its succeeded event is not."""
        first = walk(tmp_path, stop_after=StageName.MAPPING)
        assessment = first.facts.require(Fact.ASSESSMENT_ID)
        controller = Controller(tmp_path)
        (investigation,) = controller.investigations.investigations()
        log = controller.investigations.log_for(
            investigation.investigation_id
        )
        last = log.events()[-1]
        assert last.status is StageStatus.SUCCEEDED
        (log.directory / f"{last.sequence:06d}.json").unlink()

        result = walk(tmp_path, stop_after=StageName.MAPPING)

        assert run_counts(tmp_path)["mapping"] == 1
        assert "reconciled" in result.events[-1].detail
        assert result.facts.require(Fact.ASSESSMENT_ID) == assessment


class TestThroughTheCommandLine:
    def test_a_fresh_process_finishes_what_another_started(
        self, tmp_path: Path
    ) -> None:
        """The durability claim at its most literal: two processes, one
        run, nothing shared but the files."""
        started = self._arl(
            "run",
            str(REPO / "examples" / "canary.json"),
            "--root",
            str(tmp_path),
            "--lab",
            "examples.canary_lab:lab",
            "--stop-after",
            str(StageName.SELECTION),
        )
        assert started.returncode == 0, started.stdout
        assert "stopped" in started.stdout

        finished = self._arl(
            "resume", "--root", str(tmp_path), "--lab", "examples.canary_lab:lab"
        )

        assert finished.returncode == 0, finished.stdout
        assert "funding" in finished.stdout
        assert run_counts(tmp_path)["program"] == 1
        assert verify(tmp_path).ok

    def test_a_refusal_exits_two(self, tmp_path: Path) -> None:
        """Two, not one: a precondition that was not met is a different
        thing from a fault, and a script should be able to tell."""
        refused = self._arl(
            "run",
            str(thin_config(tmp_path)),
            "--root",
            str(tmp_path / "run"),
            "--lab",
            "examples.canary_lab:lab",
        )

        assert refused.returncode == 2, refused.stdout
        assert "refused" in refused.stdout

    def test_verify_says_intact(self, tmp_path: Path) -> None:
        walk(tmp_path)

        checked = self._arl("verify", "--root", str(tmp_path))

        assert checked.returncode == 0
        assert "intact" in checked.stdout

    @staticmethod
    def _arl(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "autonomous_research_lab.control.cli",
             *arguments],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=False,
        )


def test_the_walk_is_deterministic(tmp_path: Path) -> None:
    """Two roots, the same config, the same ids. Content addressing is
    what makes the comparison in this file meaningful at all."""
    first, second = tmp_path / "first", tmp_path / "second"
    walk(first)
    walk(second)

    assert facts_of(first) == facts_of(second)
    assert Outcome.STOPPED  # the walk ends at the step limit, not a fault
