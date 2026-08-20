"""The controller: one investigation, walked stage by stage.

Every stage of the chain already exists, is tested, and stores its own
durable records. What did not exist until now is anything in the package
that *walks* them. The walking was done by hand in ``examples/``, where
each driver pinned the previous stage's record id as a constant and took
the previous stage's root as a command-line flag. That is not a pipeline;
it is a person acting as one.

This package is the composition root::

    brief -> mapping -> ideation -> prior art -> selection
          -> admission -> funding -> experimentation

It is the one package allowed to import every stage, and nothing in the
package is allowed to import it — the position ``program`` held before
it. That asymmetry is the whole point: the stages stay unable to reach
each other sideways, and exactly one place knows the order.

What the controller owns is sequencing and memory, never science. It
derives each stage's directive deterministically from the run config and
the ids the earlier stages produced, records what happened to every
stage before and after the side effect, and refuses to run a stage whose
work is already on disk. What a stage decides is the stage's business.

Nothing here retries. A failed stage is a durable fact and a stop; the
operator resumes when they have dealt with the cause.
"""

from __future__ import annotations

from .config import (
    AdmissionSettings,
    ConfigError,
    ExperimentationSettings,
    FundingSettings,
    IdeationSettings,
    PriorArtSettings,
    RunConfig,
    SelectionSettings,
    load_config,
    parse_config,
)
from .events import (
    StageEvent,
    StageLog,
    StageLogConflictError,
    StageLogContentionError,
    StageLogIntegrityError,
)
from .investigation import (
    Investigation,
    InvestigationConflictError,
    InvestigationIntegrityError,
    InvestigationStore,
)
from .stage import (
    CHAIN_ORDER,
    TERMINAL_STATUSES,
    ZERO_SPEND,
    ChainFacts,
    Fact,
    MissingFactError,
    StageName,
    StageSpend,
    StageStatus,
)

__all__ = [
    "CHAIN_ORDER",
    "TERMINAL_STATUSES",
    "ZERO_SPEND",
    "AdmissionSettings",
    "ChainFacts",
    "ConfigError",
    "ExperimentationSettings",
    "Fact",
    "FundingSettings",
    "IdeationSettings",
    "Investigation",
    "InvestigationConflictError",
    "InvestigationIntegrityError",
    "InvestigationStore",
    "MissingFactError",
    "PriorArtSettings",
    "RunConfig",
    "SelectionSettings",
    "StageEvent",
    "StageLog",
    "StageLogConflictError",
    "StageLogContentionError",
    "StageLogIntegrityError",
    "StageName",
    "StageSpend",
    "StageStatus",
    "load_config",
    "parse_config",
]
