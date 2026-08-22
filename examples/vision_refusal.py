"""The honest no, proven on a real admission.

The preserved Task 5F run admitted a candidate about attention-head
overlap and few-shot accuracy. The vision lab cannot measure either —
its templates compute linear-probe contrasts — and this driver proves
what happens when the two meet: a typed refusal naming both admitted
metric strings, raised as the same ``ExperimentationUnavailableError``
family the experimentation stage records as REFUSED (exit 2), before
anything runs or spends::

    python -m examples.vision_refusal

Copies the preserved admission into a scratch root, funds it for real
through ``start_run`` — the refusal must come from the lab, not from a
missing run — builds the exact ``RuntimeRequest`` the chain builds, and
asks the vision lab for a runtime. Needs the preserved ``live_runs/``
records and nothing else: no network, no torch, no model.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from autonomous_research_lab.admission.store import AdmissionStore
from autonomous_research_lab.control.chain import ExperimentationStage
from autonomous_research_lab.control.lab import RuntimeRequest
from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.program.authorization import FundingAuthorization
from autonomous_research_lab.program.directive import RunDirective
from autonomous_research_lab.program.starter import start_run
from autonomous_research_lab.program.store import ProgramStore
from examples.vision_lab.backends import Backend, ExecutionProfile
from examples.vision_lab.composition import VisionLab
from examples.vision_lab.measure import UnmeasurablePredictionsError

PRESERVED = (
    Path(__file__).resolve().parent.parent
    / "live_runs"
    / "task5f-2026-08-20"
    / "admission"
)


def main() -> int:
    if not PRESERVED.is_dir():
        print(f"FAIL: preserved admission missing at {PRESERVED}")
        return 1
    with tempfile.TemporaryDirectory(prefix="vision-refusal-") as scratch:
        root = Path(scratch)
        shutil.copytree(PRESERVED, root / "admission")
        admission = AdmissionStore(root / "admission")
        program = ProgramStore(root / "program")
        (record,) = admission.records()

        authorization = FundingAuthorization(
            admission_record_id=record.id,
            granted=ResearchBudget(
                wall_clock_seconds=3600.0, usd=5.0, model_tokens=100_000
            ),
            authority="Lab operator: refusal-proof funding.",
        )
        directive = RunDirective(
            admission_record_id=record.id,
            authorization_id=authorization.id,
            label="vision-refusal-proof",
        )
        started = start_run(
            admission_store=admission,
            program_store=program,
            directive=directive,
            authorization=authorization,
        )
        states = FileStateStore(root)
        states.persist(
            FileStateStore(root / "admission").load(record.state_id)
        )
        states.persist(started.funded_state)

        lab = VisionLab(
            profile=ExecutionProfile(
                backend=Backend.HOST_CPU, datasets_root=root
            ),
            trainer="real",
            scripted=True,
            live_generation=False,
        )
        request = RuntimeRequest(
            root=root,
            evidence=FileEvidenceStore(root),
            states=states,
            ledger=program.ledger_for(started.run.run_id),
            journal=program.journal_for(started.run.run_id),
            bundles=program.bundles(),
        )
        try:
            lab.runtime(request)
        except UnmeasurablePredictionsError as refusal:
            message = str(refusal)
            named = [
                metric
                for metric in (
                    "difference in overlap of important heads",
                    "difference in few-shot accuracy",
                )
                if metric in message
            ]
            is_refusal = isinstance(
                refusal, ExperimentationStage().refusals()
            )
            print(f"refused: {message[:200]}...")
            print(f"both admitted metrics named: {len(named) == 2}")
            print(f"recognised as a stage refusal (exit 2): {is_refusal}")
            if len(named) == 2 and is_refusal:
                print("PASS: the honest no, before anything ran or spent")
                return 0
            print("FAIL: the refusal did not say what it must")
            return 1
        print("FAIL: the vision lab accepted predictions it cannot measure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
