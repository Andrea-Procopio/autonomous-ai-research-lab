"""The seam between a configured chain and the instruments it uses.

Three things a run needs that a config file must not describe.

**A model provider.** Named per stage, because a lab may want a cheaper
model for screening than for the comparisons that decide a verdict, and
because a deterministic lab wants a different scripted provider for each
stage.

**A literature provider.** One adapter, live or replayed.

**A runtime.** Roles, an executor, and a trusted template catalog. This
is the one that cannot be data. A template is source code that will run
in a container, and a catalog described in JSON would be a way for a
config file to choose what executes — precisely the authority the
architecture keeps in human hands. So the CLI imports a lab module and
asks it, and a lab that cannot supply one says so.

The default lab covers the six analysis-and-funding stages from the
environment and refuses the seventh. That refusal is a first-class
outcome, not an error: ``arl run`` without ``--lab`` is a legitimate way
to carry a topic to a funded run and stop.

One convention is baked into the chain rather than this seam: recovery's
finished-job collector reads the local executor's layout under
``root/runs`` — the same convention the verifier checks. A lab whose
executor keeps jobs elsewhere loses salvage, never correctness (the
conservative arm answers instead); letting a lab supply its own
collector is the extension point the remote-executor era will need.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from ..evidence.store import EvidenceStore
from ..literature.openalex import OpenAlexProvider
from ..literature.resolution import AccessResolvingProvider
from ..literature.retrieval import LiteratureProvider
from ..orchestration.loop import ResearchRuntime
from ..persistence import FileStateStore
from ..persistence.commit_store import CommitBundleStore
from ..runtime.journal import AttemptJournal
from ..runtime.muse import MuseSparkProvider
from ..runtime.providers import ModelProvider
from ..runtime.spend import SpendLedger
from .stage import StageName

_LAB_METHODS = ("model_provider", "literature_provider", "runtime")


class LabError(RuntimeError):
    """A lab could not be loaded, or is not one."""


class ExperimentationUnavailableError(RuntimeError):
    """No lab supplied a runtime, so the experimentation stage cannot be
    attempted. A refusal, not a failure: the chain reached a funded run
    and stopped for want of an instrument."""


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    """Everything a runtime needs that belongs to the run rather than to
    the lab: where the run lives, where its facts and snapshots go, the
    ledger its spend posts to, and the two records that make a step
    recoverable."""

    root: Path
    evidence: EvidenceStore
    states: FileStateStore
    ledger: SpendLedger
    journal: AttemptJournal
    """Where each attempt's phases are written down as they happen. A lab
    that wires it hands the runtime the ability to be killed mid-step and
    resumed; one that does not gets the pre-existing behavior."""

    bundles: CommitBundleStore
    """Where an attempt's whole effect is stored before it is applied."""


class Lab(Protocol):
    """The instruments one investigation runs on.

    The arguments are positional-only so a lab may name them whatever
    reads best in its own module; the controller always passes them in
    order.
    """

    def model_provider(self, stage: StageName, /) -> ModelProvider: ...

    def literature_provider(self) -> LiteratureProvider: ...

    def runtime(self, request: RuntimeRequest, /) -> ResearchRuntime: ...


@dataclass(frozen=True, slots=True)
class DefaultLab:
    """Muse for the analysis stages, OpenAlex for retrieval, nothing for
    experimentation.

    Both adapters read their credentials from the environment at request
    time, which is why there is nothing to configure here and nothing
    that could be written into a record.
    """

    def model_provider(self, _stage: StageName) -> ModelProvider:
        return MuseSparkProvider()

    def literature_provider(self) -> LiteratureProvider:
        return AccessResolvingProvider(OpenAlexProvider())

    def runtime(self, request: RuntimeRequest) -> ResearchRuntime:
        raise ExperimentationUnavailableError(
            f"no lab is wired, so the funded run at {request.root} has no "
            f"roles, no executor and no trusted template catalog to run "
            f"experiments with; pass --lab module:factory to supply them"
        )


def _make_the_working_directory_importable() -> None:
    """Put the working directory on the import path, as ``python -m``
    does and an installed console script does not.

    An operator naming ``--lab examples.canary_lab:lab`` means the one
    in front of them. The flag already names code that will be imported
    and run, so the trust decision was made when it was typed; what this
    avoids is that decision failing on a technicality.
    """
    here = str(Path.cwd())
    if here not in sys.path:
        sys.path.insert(0, here)


def load_lab(spec: str) -> Lab:
    """Import ``module:factory`` and return the lab it makes.

    Checked structurally at the boundary rather than trusted: a module
    that does not answer the three questions a lab answers is refused
    here, with the spec in the message, instead of failing later inside
    a stage.
    """
    _make_the_working_directory_importable()
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise LabError(
            f"a lab is named module:factory (for example "
            f"examples.canary_lab:lab), got {spec!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise LabError(f"cannot import lab module {module_name!r}: {exc}") from exc
    factory = getattr(module, attribute, None)
    if factory is None:
        raise LabError(
            f"lab module {module_name!r} has no attribute {attribute!r}"
        )
    try:
        lab = factory() if callable(factory) else factory
    except Exception as exc:
        raise LabError(
            f"{spec} is not a lab: calling it raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    missing = [
        method for method in _LAB_METHODS if not callable(getattr(lab, method, None))
    ]
    if missing:
        raise LabError(
            f"{spec} is not a lab: it has no {', '.join(missing)} method(s)"
        )
    return cast(Lab, lab)
