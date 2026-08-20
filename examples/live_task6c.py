"""Task 6C: the preserved chain, replayed through the controller.

Five stages of this project's real research chain were run live, one
task at a time, each into its own root, each bridged to the next by a
record id pasted into the following driver by hand. This puts those
same records under one root and asks the controller to walk them::

    5B.1 mapping -> 5C ideation -> 5D.2 prior art -> 5E selection
      -> 5F admission -> funding

Nothing here calls a model or the network, and the point is that it
cannot: the lab supplies a provider that raises on every call and a
literature provider that raises on every search. If the controller
executed any completed stage, the run would die loudly instead of
quietly paying for work already done.

What is being proven, in order:

1. **The config reproduces the chain.** Five directives that were
   hand-authored in five drivers are derived here from one JSON file,
   and each one's content id equals the id the preserved record was
   filed under. Byte-identical directives are what make the skip
   possible; a drifted config would look like new work.
2. **Completed work is recognised, not repeated.** No stage has an event
   log to consult, so every skip comes from the reconcile probe asking
   the stage's own store — the crash path, exercised for real.
3. **The preserved records are untouched.** Every file under the copied
   upstream stores is hashed before and after. Byte identity as a fact,
   not a claim.
4. **Funding happens exactly once.** It is the only stage with work left
   to do. A second walk adopts it from the log; the ledger still holds
   one entry and the run still has one id.
5. **The whole root verifies from cold.**

Run with::

    python -m examples.live_task6c --run-root live_runs/task6c-<date>

Exits 0 only if every check above holds.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from autonomous_research_lab.admission.store import AdmissionStore
from autonomous_research_lab.control.config import RunConfig, load_config
from autonomous_research_lab.control.controller import Controller, Outcome
from autonomous_research_lab.control.lab import RuntimeRequest
from autonomous_research_lab.control.stage import Fact, StageName, StageStatus
from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.literature.retrieval import (
    LiteratureProvider,
    LiteratureQuery,
    RetrievedSearch,
)
from autonomous_research_lab.mapping.store import MappingStore
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.program.integrity import verify_run
from autonomous_research_lab.runtime.providers import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from autonomous_research_lab.selection.store import SelectionStore

CONFIG = Path(__file__).resolve().parent / "task6c_replay.json"

#: The preserved roots, and which of their stores this replay needs.
#: Two roots carry a ``literature`` store; the corpora were retrieved by
#: different stages of the same investigation and are content-addressed,
#: so the union is a merge and any name that disagrees is a fault.
PRESERVED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("task5b1-2026-08-18", ("literature", "mapping")),
    ("task5c-2026-08-19", ("ideation",)),
    ("task5d2-2026-08-19", ("literature", "priorart")),
    ("task5e-2026-08-20", ("selection",)),
    ("task5f-2026-08-20", ("admission",)),
)

#: Where the hand-bridged chain actually arrived, in August 2026. These
#: are checks, not inputs: the controller is given no id at all, and the
#: claim under test is that it reaches the same records the five drivers
#: reached by pasting ids between themselves.
ATTESTED = {
    Fact.MAP_RUN_RECORD_ID: "mrun_3276a246eea9fbb5",
    Fact.IDEATION_RUN_RECORD_ID: "irun_23e08da6543a3bb6",
    Fact.PRIOR_ART_RUN_RECORD_ID: "prun_095dfdb9a99f4f0f",
    Fact.SELECTION_RUN_RECORD_ID: "srun_fd598b0eb2e3e80b",
    Fact.ADMISSION_RECORD_ID: "arun_aef62566adb793d3",
    Fact.ADMITTED_STATE_ID: "st_bea69ecb9b4e3ac2",
}


class ReplayViolationError(AssertionError):
    """A call that the replay proves is never made was made anyway."""


class RefusingProvider(ModelProvider):
    """Every model call is a bug in this run, so every model call is an
    error with the request in it."""

    @property
    def name(self) -> str:
        return "refusing"

    def invoke(self, request: ModelRequest) -> ModelResponse:
        raise ReplayViolationError(
            f"a model call was attempted ({len(request.messages)} message(s), "
            f"model {request.model}); the replay executes no stage that "
            f"could need one"
        )


class RefusingLiterature(LiteratureProvider):
    """The same, for retrieval."""

    @property
    def name(self) -> str:
        return "refusing"

    def search(self, query: LiteratureQuery) -> RetrievedSearch:
        raise ReplayViolationError(
            f"a literature search was attempted ({query.text!r}); the "
            f"replay retrieves nothing"
        )


class ReplayLab:
    """A lab with no instruments, on purpose."""

    def model_provider(self, _stage: StageName) -> ModelProvider:
        return RefusingProvider()

    def literature_provider(self) -> LiteratureProvider:
        return RefusingLiterature()

    def runtime(self, request: RuntimeRequest) -> ResearchRuntime:
        raise ReplayViolationError(
            f"experimentation was attempted at {request.root}; this replay "
            f"stops at the funded run"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preserved",
        type=Path,
        default=Path("live_runs"),
        help="directory holding the preserved task roots (read-only)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="a fresh run directory (put it under the gitignored live_runs/)",
    )
    arguments = parser.parse_args()
    preserved = arguments.preserved.resolve()
    root = arguments.run_root.resolve()

    if root.exists() and any(root.iterdir()):
        print(
            f"FATAL: {root} already holds a run. This driver proves what a "
            f"cold start does; point it at a fresh directory."
        )
        return 1

    print("== assembling one root from five preserved ones ==")
    try:
        copied = _assemble(preserved, root)
    except FileNotFoundError as error:
        print(f"FATAL: {error}")
        return 1
    except ReplayViolationError as error:
        print(f"FATAL: {error}")
        return 1
    for line in copied:
        print(f"  {line}")
    before = _digests(root)
    print(f"  {len(before)} preserved file(s) hashed")

    config, payload = load_config(CONFIG)
    print()
    print("== the config reproduces the hand-authored directives ==")
    mismatches = _check_directives(config, root)
    for line in mismatches.report:
        print(f"  {line}")
    if not mismatches.ok:
        print(
            "FATAL: the config no longer derives the preserved directives, "
            "so the controller would treat completed work as new."
        )
        return 1

    controller = Controller(root)
    print()
    print("== first walk: five stages recognised, one funded ==")
    investigation = controller.begin(payload)
    first = controller.walk(investigation, lab=ReplayLab())
    for event in first.events:
        print(
            f"  {event.stage:<14} {event.status:<10} {event.detail[:64]}"
        )
    if first.outcome is not Outcome.STOPPED:
        print(f"FATAL: expected to stop after funding, got {first.outcome}: "
              f"{first.detail}")
        return 1

    reconciled = [
        event
        for event in first.events
        if event.status is StageStatus.SUCCEEDED
        and event.detail.startswith("reconciled")
    ]
    executed = [
        event for event in first.events if event.status is StageStatus.RUNNING
    ]
    print()
    print(f"  reconciled: {len(reconciled)} stage(s)")
    print(f"  executed:   {len(executed)} stage(s)")
    if len(reconciled) != 5 or len(executed) != 1:
        print(
            "FATAL: expected five recognised stages and one execution "
            "(the funding), which is the whole claim."
        )
        return 1
    # The reconciled events carry the spend the preserved records
    # already held -- what this work cost when it was really run, months
    # of wall clock ago. What must be zero is what the walk itself paid,
    # and the refusing provider is what makes that unfaked: a single
    # attempted call would have ended the run with a traceback.
    recognised = sum(event.spend.model_calls for event in reconciled)
    paid = sum(
        event.spend.model_calls
        for event in first.events
        if event.status is StageStatus.SUCCEEDED
        and not event.detail.startswith("reconciled")
    )
    print(f"  recognised spend: {recognised} model call(s), already paid")
    print(f"  spent now:        {paid} model call(s)")
    if paid:
        print("FATAL: the replay recorded model spend; nothing should have run.")
        return 1

    after_first = _digests(root)
    changed = _changed(before, after_first)
    print()
    print("== the preserved records are untouched ==")
    if changed:
        for line in changed:
            print(f"  {line}")
        print("FATAL: the replay modified preserved records.")
        return 1
    print(f"  {len(before)} file(s) byte-identical after the walk")

    run_id = first.facts.require(Fact.RUN_ID)
    ledger = controller.stores.program.ledger_for(run_id)
    print()
    print("== second walk: nothing left to do ==")
    second = controller.resume(investigation.investigation_id, lab=ReplayLab())
    print(f"  outcome: {second.outcome}")
    print(f"  events:  {len(first.events)} -> {len(second.events)}")
    if second.outcome is not Outcome.STOPPED:
        print(f"FATAL: the second walk did not stop cleanly: {second.detail}")
        return 1
    if len(second.events) != len(first.events):
        print("FATAL: the second walk wrote events; there was nothing to do.")
        return 1
    if second.facts.require(Fact.RUN_ID) != run_id:
        print("FATAL: the second walk produced a different run.")
        return 1

    entries = ledger.entries()
    print(f"  ledger entries: {len(entries)}")
    print(f"  balance:        {ledger.balance()}")
    if len(entries) != 1:
        print("FATAL: funding happened more than once.")
        return 1
    changed = _changed(before, _digests(root))
    if changed:
        for line in changed:
            print(f"  {line}")
        print("FATAL: the second walk modified preserved records.")
        return 1

    print()
    print("== the whole root verifies from cold ==")
    report = verify_run(root, program_root=root / "program")
    print(f"  states {report.states_checked}, results {report.results_checked}, "
          f"evidence {report.evidence_checked}, blobs {report.blobs_checked}")
    for issue in report.issues:
        print(f"  {issue.kind}: {issue.subject_id}: {issue.detail}")
    if not report.ok:
        print("FATAL: the assembled run does not verify.")
        return 1

    print()
    print("== the walk arrived where the hand-bridged chain did ==")
    wrong: list[str] = []
    for fact, attested in ATTESTED.items():
        reached = first.facts.get(fact)
        agrees = reached == attested
        print(f"  {fact:<24} {reached}  {'==' if agrees else '!='}  {attested}")
        if not agrees:
            wrong.append(str(fact))
    if wrong:
        print(
            f"FATAL: the controller reached different records for "
            f"{', '.join(wrong)} than the drivers did."
        )
        return 1

    print()
    print("== what the walk produced ==")
    for name, value in sorted(first.facts.as_mapping().items()):
        print(f"  {name:<24} {value}")
    print()
    print(f"OK: five preserved stages recognised at {recognised} model "
          f"call(s) already paid and none spent now, one run funded once, "
          f"{len(before)} preserved file(s) unchanged, root verifies.")
    print(f"Investigation {investigation.investigation_id} under {root}")
    return 0


# -- assembling ----------------------------------------------------------------


def _assemble(preserved: Path, root: Path) -> list[str]:
    """Copy the preserved stores under one root, refusing any name that
    two roots disagree about."""
    root.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for task, stores in PRESERVED:
        for store in stores:
            source = preserved / task / store
            if not source.is_dir():
                raise FileNotFoundError(f"no preserved store at {source}")
            target = root / store
            copied = _merge(source, target)
            lines.append(f"{task}/{store}: {copied} file(s)")
    return lines


def _merge(source: Path, target: Path) -> int:
    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        destination = target / path.relative_to(source)
        if destination.exists():
            if destination.read_bytes() != path.read_bytes():
                raise ReplayViolationError(
                    f"two preserved roots disagree about "
                    f"{destination.relative_to(target)}; these stores are "
                    f"content-addressed, so a disagreement is a fault"
                )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
    return copied


def _digests(root: Path) -> dict[str, str]:
    """Every file under the copied upstream stores, hashed."""
    stores = {store for _, names in PRESERVED for store in names}
    digests: dict[str, str] = {}
    for store in sorted(stores):
        directory = root / store
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                digests[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return digests


def _changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    lines = [
        f"added:   {name}" for name in sorted(set(after) - set(before))
    ]
    lines += [f"removed: {name}" for name in sorted(set(before) - set(after))]
    lines += [
        f"edited:  {name}"
        for name in sorted(set(before) & set(after))
        if before[name] != after[name]
    ]
    return lines


# -- the directive check --------------------------------------------------------


class _DirectiveCheck:
    def __init__(self) -> None:
        self.report: list[str] = []
        self.ok = True

    def add(self, stage: StageName, derived: str, found: bool) -> None:
        self.report.append(
            f"{stage:<14} {derived}  {'found' if found else 'NOT FOUND'}"
        )
        self.ok = self.ok and found


def _check_directives(config: RunConfig, root: Path) -> _DirectiveCheck:
    """Derive each stage's directive from the config and look it up in
    the preserved store it should already be recorded in.

    This is the check that would have caught a drifted config before the
    controller treated five completed stages as new work.
    """
    check = _DirectiveCheck()

    mapping = MappingStore(root / "mapping")
    check.add(
        StageName.MAPPING,
        config.brief.id,
        mapping.get_brief(config.brief.id) is not None,
    )
    run = next(
        (r for r in mapping.runs() if r.brief_id == config.brief.id), None
    )
    if run is None:
        return check
    assessment = mapping.adequacy_for_run(run.run_id)
    if assessment is None:
        return check

    ideation = IdeationStore(root / "ideation")
    ideation_directive = config.ideation.directive(
        assessment_id=assessment.id, snapshot_id=config.snapshot.id
    )
    check.add(
        StageName.IDEATION,
        ideation_directive.id,
        ideation.get_directive(ideation_directive.id) is not None,
    )
    ideation_run = next(
        (
            record
            for record in ideation.runs()
            if record.directive_id == ideation_directive.id
        ),
        None,
    )
    if ideation_run is None:
        return check

    prior_art = PriorArtStore(root / "priorart")
    prior_art_directive = config.prior_art.directive(
        ideation_run_record_id=ideation_run.id
    )
    check.add(
        StageName.PRIOR_ART,
        prior_art_directive.id,
        prior_art.get_directive(prior_art_directive.id) is not None,
    )
    prior_art_run = next(
        (
            record
            for record in prior_art.runs()
            if record.directive_id == prior_art_directive.id
        ),
        None,
    )
    if prior_art_run is None:
        return check

    selection = SelectionStore(root / "selection")
    selection_directive = config.selection.directive(
        prior_art_run_record_id=prior_art_run.id
    )
    check.add(
        StageName.SELECTION,
        selection_directive.id,
        selection.get_directive(selection_directive.id) is not None,
    )
    selection_run = next(
        (
            record
            for record in selection.runs()
            if record.directive_id == selection_directive.id
        ),
        None,
    )
    if selection_run is None:
        return check

    admission = AdmissionStore(root / "admission")
    admission_directive = config.admission.directive(
        selection_run_record_id=selection_run.id
    )
    check.add(
        StageName.ADMISSION,
        admission_directive.id,
        admission.get_directive(admission_directive.id) is not None,
    )
    return check


if __name__ == "__main__":
    sys.exit(main())
