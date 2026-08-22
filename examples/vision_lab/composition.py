"""The vision lab itself: what ``--lab examples.vision_lab:...`` loads.

``runtime()`` is rebuilt fresh on every experimentation step — the
chain's contract — so everything here is derived from durable records
and deployment data, never from memory of a previous step. The order of
composition is the argument:

1. **Find the funded state.** The journal names the run; the program
   store names the run's funded state; the state store holds it.
2. **Refuse what cannot be measured.** Every admitted prediction must
   be in the lab's declared contrast table, or the whole stage refuses
   with a typed, exit-2 honest no before anything spends.
3. **Build the catalog from the record.** The admitted metric strings
   are substituted into the templates by trusted code; capability
   metrics become the closed set every later declaration is held to.
4. **Resolve the backend.** Deployment data picks host or container,
   CPU or GPU; nothing scientific varies with it — and a deployment
   that refuses live generated code on the host refuses here, loudly.
5. **Wire the runtime** with governance on: deterministic scientist and
   analyst, the model-backed engineer over one journalling runner, the
   structural methodology reviewer, the catalog's positive control, and
   file-backed verification records — the store that makes governance
   survive per-step rebuilds.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from autonomous_research_lab.control.lab import (
    ExperimentationUnavailableError,
    LabError,
    RuntimeRequest,
)
from autonomous_research_lab.control.stage import StageName
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.execution.runner import DirectJobRunner
from autonomous_research_lab.literature.openalex import OpenAlexProvider
from autonomous_research_lab.literature.retrieval import LiteratureProvider
from autonomous_research_lab.orchestration.director import (
    FrontierDirector,
    RuleBasedFrontierDirector,
)
from autonomous_research_lab.orchestration.loop import ResearchRuntime
from autonomous_research_lab.orchestration.trajectory import (
    JsonlTrajectoryLogger,
)
from autonomous_research_lab.program.store import ProgramStore
from autonomous_research_lab.roles.base import ResearchRole, RoleName
from autonomous_research_lab.roles.engineer import (
    ImplementationTemplate,
    ModelBackedEngineer,
)
from autonomous_research_lab.roles.planner import (
    ModelBackedPlanner,
    TemplateCatalog,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.implementation_store import (
    ImplementationStore,
)
from autonomous_research_lab.runtime.journal import JournalingJobRunner
from autonomous_research_lab.runtime.metrics import JsonlRuntimeMetrics
from autonomous_research_lab.runtime.muse import (
    MUSE_SPARK_1_2,
    MuseSparkProvider,
)
from autonomous_research_lab.runtime.planning_store import PlanningStore
from autonomous_research_lab.runtime.preflight import (
    DEFAULT_PREFLIGHT_CHECKS,
    PreflightCheck,
)
from autonomous_research_lab.runtime.providers import (
    ModelProvider,
    UsageLedger,
)
from autonomous_research_lab.runtime.verification_store import (
    FileVerificationStore,
)
from examples.trajectory_campaign import StructuralMethodology

from .backends import (
    Backend,
    DatasetBoundBinding,
    ExecutionProfile,
    profile_from,
    resolve,
)
from .catalog import catalog_for, entry_for_metric, fill_slot
from .datasets import DatasetStaged, DatasetStore
from .direction import VisionDirector, VisionDirectorRole
from .measure import require_measurable
from .science import (
    FixedRegionCheck,
    FixedRegionReview,
    VisionAnalyst,
    VisionControls,
    VisionScientist,
)
from .scripted import VisionScriptedLiterature, VisionScriptedModel

PROFILE_ENV = "ARL_VISION_PROFILE"

DATASET_NAME = "cifar10"

SCRIPTED_MODEL_NAME = "vision-scripted-1"

#: The trusted fixture completions of each trainer's slot — what the
#: scripted engineer "generates". Complete function definitions, because
#: the slot fence encloses the whole def.
STUB_SLOT = '''def build_encoder(rng: random.Random) -> list[list[float]]:
    """A full-rank Gaussian projection: the honest minimal answer."""
    return [
        [rng.gauss(0.0, 1.0) for _ in range(DIM)] for _ in range(DIM)
    ]'''

REAL_SLOT = '''def build_encoder() -> nn.Module:
    """Three conv blocks to a 1024-dim feature: small, seeded, plain."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
    )'''


@dataclass(frozen=True)
class VisionLab:
    """One deployment's vision lab. Frozen: a lab is configuration."""

    profile: ExecutionProfile
    trainer: str = "real"
    """Which template sources the catalog serves: ``real`` (torch) or
    ``stub`` (stdlib, for CI and offline walks)."""

    scripted: bool = False
    """Scripted stage instruments (zero network, zero spend) instead of
    Muse and OpenAlex."""

    live_generation: bool = True
    """Whether the engineer's completions come from a live model. Off,
    the scripted provider serves a trusted fixture completion — the
    engineer's contract stays the production one, its trust story does
    not."""

    planner: bool = True
    """Whether the model-backed planner shares the director seat. On,
    the composite director hands consultations to the planner once
    verified evidence exists (see ``direction.py``); off, the run keeps
    the purely deterministic 7A arc."""

    def model_provider(self, stage: StageName, /) -> ModelProvider:
        del stage
        if self.scripted:
            return VisionScriptedModel()
        return MuseSparkProvider()

    def literature_provider(self) -> LiteratureProvider:
        if self.scripted:
            return VisionScriptedLiterature()
        return OpenAlexProvider()

    def runtime(self, request: RuntimeRequest, /) -> ResearchRuntime:
        state = _funded_state(request)
        require_measurable(state.predictions)
        catalog = catalog_for(
            state.predictions,
            trainer=self.trainer,
            gpu_count=self.profile.effective_gpus,
        )
        uses_dataset = self.trainer == "real"
        resolved = resolve(
            self.profile,
            dataset_names=(DATASET_NAME,) if uses_dataset else (),
        )
        if self.live_generation and not resolved.generated_code_allowed:
            raise ExperimentationUnavailableError(
                f"backend {self.profile.backend} runs on the host, and "
                f"this deployment does not allow live model-generated "
                f"code there; use a container backend, or record the "
                f"decision with allow_generated_code_on_host in the "
                f"deployment profile"
            )

        implementations = ImplementationStore(
            request.root / "implementations"
        )
        runner = JournalingJobRunner(
            inner=DirectJobRunner(resolved.executor_factory(request.root)),
            journal=request.journal,
        )
        binding = resolved.binding
        preflight: tuple[PreflightCheck, ...] = (
            *DEFAULT_PREFLIGHT_CHECKS,
            FixedRegionCheck(store=implementations, catalog=catalog),
        )
        if uses_dataset:
            store = DatasetStore(self.profile.datasets_root)
            binding = DatasetBoundBinding(
                inner=binding,
                store=store,
                dataset_name=DATASET_NAME,
                dataset_root_for_job=resolved.dataset_root_for_job,
                device=resolved.device,
            )
            preflight = (*preflight, DatasetStaged(store))

        usage = UsageLedger()
        engineer = ModelBackedEngineer(
            provider=self._engineer_provider(catalog),
            model=(
                SCRIPTED_MODEL_NAME if self.scripted else MUSE_SPARK_1_2
            ),
            runner=runner,
            ledger=usage,
            store=implementations,
            binding=binding,
            template=catalog.entries[0].template,
            template_resolver=_resolver(catalog),
            completion_review=FixedRegionReview(),
            preflight_checks=preflight,
        )
        scientist = VisionScientist(catalog)
        director: FrontierDirector
        seat: ResearchRole
        if self.planner:
            plans = PlanningStore(request.root / "planning")
            planner_role = ModelBackedPlanner(
                provider=(
                    VisionScriptedModel()
                    if self.scripted
                    else MuseSparkProvider()
                ),
                model=(
                    SCRIPTED_MODEL_NAME if self.scripted else MUSE_SPARK_1_2
                ),
                ledger=usage,
                store=plans,
                catalog=catalog,
                # The live lesson from Task 4: Muse truncates planning
                # replies at the default budget.
                max_output_tokens=8192,
            )
            director = VisionDirector(plans=plans)
            seat = VisionDirectorRole(scientist, planner_role)
        else:
            director = RuleBasedFrontierDirector()
            seat = scientist
        return ResearchRuntime(
            config=RuntimeConfig(),
            director=director,
            roles={
                RoleName.RESEARCH_DIRECTOR: seat,
                RoleName.RESEARCH_ENGINEER: engineer,
                RoleName.RESULT_ANALYST: VisionAnalyst(),
            },
            store=request.evidence,
            states=request.states,
            ledger=request.ledger,
            journal=request.journal,
            bundles=request.bundles,
            usage=usage,
            methodology_reviewer=StructuralMethodology(),
            control_source=VisionControls(catalog),
            verifications=FileVerificationStore(
                request.root / "verifications"
            ),
            trajectory=JsonlTrajectoryLogger(
                request.root / "trajectory.jsonl"
            ),
            metrics=JsonlRuntimeMetrics(request.root / "metrics.jsonl"),
        )

    def _engineer_provider(self, catalog: TemplateCatalog) -> ModelProvider:
        if self.live_generation:
            return MuseSparkProvider()
        slot = REAL_SLOT if self.trainer == "real" else STUB_SLOT
        reply = json.dumps(
            {
                "files": [
                    {
                        "path": "experiment.py",
                        "content": fill_slot(
                            catalog.entries[0].template.source, slot
                        ),
                    }
                ],
                "rationale": (
                    "a trusted fixture completion of the template's one "
                    "slot; every fixed region is preserved byte-for-byte"
                ),
            }
        )
        return VisionScriptedModel(engineer_reply=reply)


def _funded_state(request: RuntimeRequest) -> ResearchState:
    run_id = getattr(request.journal, "run_id", "")
    program = ProgramStore(request.root / "program")
    envelopes = program.runs()
    if not envelopes:
        raise ExperimentationUnavailableError(
            f"no funded run exists under {request.root}; fund one before "
            f"experimenting"
        )
    if run_id:
        matched = [e for e in envelopes if e.run_id == run_id]
        if not matched:
            raise ExperimentationUnavailableError(
                f"the journal names run {run_id}, which the program store "
                f"under {request.root} does not hold"
            )
        (envelope,) = matched
    elif len(envelopes) == 1:
        (envelope,) = envelopes
    else:
        raise ExperimentationUnavailableError(
            f"{len(envelopes)} funded runs exist under {request.root} and "
            f"the journal names none of them; this lab refuses to guess"
        )
    return request.states.load(envelope.funded_state_id)


def _resolver(
    catalog: TemplateCatalog,
) -> Callable[[ExperimentSpec], ImplementationTemplate | None]:
    """The spec-to-template seam: the designer put the admitted metric
    first in every spec it derived from the catalog, so the primary key
    names the entry."""

    def resolve_template(spec: ExperimentSpec) -> ImplementationTemplate | None:
        return entry_for_metric(catalog, spec.metrics[0]).template

    return resolve_template


def _profile() -> ExecutionProfile:
    declared = os.environ.get(PROFILE_ENV)
    if not declared:
        raise LabError(
            f"set {PROFILE_ENV} to a deployment profile JSON file; see "
            f"docs/EXECUTION.md"
        )
    path = Path(declared)
    if not path.is_file():
        raise LabError(f"{PROFILE_ENV}={declared} names no file")
    return profile_from(json.loads(path.read_text()))


def lab() -> VisionLab:
    """Production: Muse, OpenAlex, live generation, real training."""
    return VisionLab(profile=_profile(), trainer="real")


def qualification_lab() -> VisionLab:
    """The zero-network qualification: scripted stages, a trusted
    fixture completion, and *real* torch training through the deployed
    backend."""
    return VisionLab(
        profile=_profile(),
        trainer="real",
        scripted=True,
        live_generation=False,
    )


def ci_lab() -> VisionLab:
    """Offline everything: scripted stages, the stdlib stub trainer, no
    dataset, host execution. What CI walks."""
    return VisionLab(
        profile=ExecutionProfile(
            backend=Backend.HOST_CPU,
            datasets_root=Path(),
            timeout_seconds=120.0,
        ),
        trainer="stub",
        scripted=True,
        live_generation=False,
    )
