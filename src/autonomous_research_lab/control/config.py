"""One investigation's parameters, in one file.

Everything the live drivers hardcoded as module constants lives here
instead: the brief, the call for papers, each stage's caps, the
constraints and requirements the later stages hold candidates to, the
grant, and the authority sentence behind it. What is deliberately *not*
here is any id. An id names a record some earlier stage produced, and a
config that could name one would let an operator paste the chain
together by hand again — the defect this package exists to remove.

Two properties matter more than the shape.

**It fails before the first call.** Parsing builds every directive the
chain will use, against placeholder upstream ids, purely so each
directive's own ``__post_init__`` runs its ceilings and its date checks
now rather than at stage four with three stages' spend already gone. An
unknown key is an error too: a typo that silently selects a default is
how a run ends up measuring something nobody asked for.

**It is recorded verbatim.** The controller stores the operator's own
document, content-addressed, and the investigation names it. What this
codec understood is a derived thing; what was asked for is the record.

Trusted code, templates, roles and executors are not configurable here.
Those come from a lab module the CLI imports, because a trusted template
is source code and describing it in data would make it something else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..admission.directive import AdmissionDirective
from ..core.budget import ResearchBudget
from ..ideation.direction import CfpSnapshot
from ..ideation.directive import IdeationDirective
from ..mapping.brief import ResearchBrief
from ..priorart.directive import PriorArtDirective
from ..program.authorization import (
    MAX_AUTHORITY_CHARS,
    MAX_GPU_HOURS,
    MAX_MODEL_TOKENS,
    MAX_USD,
    MAX_WALL_CLOCK_SECONDS,
    FundingAuthorization,
)
from ..program.directive import RunDirective
from ..selection.directive import SelectionDirective
from .investigation import MAX_LABEL_CHARS
from .stage import CHAIN_ORDER, StageName

MAX_STEPS_CEILING: Final = 200
"""A run that needs more than two hundred deliberations is not being
bounded by this number; it is being bounded by the budget, and the
operator should say so there."""

_PLACEHOLDER: Final = "0000000000000000"
"""Stands in for an upstream id while a directive is being validated.
The real id arrives when the stage before it has actually run."""


class ConfigError(ValueError):
    """The config cannot be read, or describes a run that cannot be
    started. Raised before anything is written."""


# -- per-stage settings -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdeationSettings:
    max_candidates: int = 5
    max_model_calls: int = 4

    def directive(
        self, *, assessment_id: str, snapshot_id: str
    ) -> IdeationDirective:
        return IdeationDirective(
            assessment_id=assessment_id,
            snapshot_id=snapshot_id,
            max_candidates=self.max_candidates,
            max_model_calls=self.max_model_calls,
        )


@dataclass(frozen=True, slots=True)
class PriorArtSettings:
    cutoff_date: str
    recent_window_start: str
    results_per_query: int = 5
    max_screened_per_candidate: int = 35
    max_compared_works: int = 4
    max_model_calls: int = 36

    def directive(self, *, ideation_run_record_id: str) -> PriorArtDirective:
        return PriorArtDirective(
            ideation_run_record_id=ideation_run_record_id,
            cutoff_date=self.cutoff_date,
            recent_window_start=self.recent_window_start,
            results_per_query=self.results_per_query,
            max_screened_per_candidate=self.max_screened_per_candidate,
            max_compared_works=self.max_compared_works,
            max_model_calls=self.max_model_calls,
        )


@dataclass(frozen=True, slots=True)
class SelectionSettings:
    compute_constraint: str
    data_constraint: str
    time_constraint: str
    experimental_constraint: str
    max_eligible_candidates: int = 5
    max_model_calls: int = 4

    def directive(self, *, prior_art_run_record_id: str) -> SelectionDirective:
        return SelectionDirective(
            prior_art_run_record_id=prior_art_run_record_id,
            compute_constraint=self.compute_constraint,
            data_constraint=self.data_constraint,
            time_constraint=self.time_constraint,
            experimental_constraint=self.experimental_constraint,
            max_eligible_candidates=self.max_eligible_candidates,
            max_model_calls=self.max_model_calls,
        )


@dataclass(frozen=True, slots=True)
class AdmissionSettings:
    scheduling_requirement: str
    job_duration_requirement: str
    checkpoint_requirement: str
    max_model_calls: int = 2

    def directive(self, *, selection_run_record_id: str) -> AdmissionDirective:
        return AdmissionDirective(
            selection_run_record_id=selection_run_record_id,
            scheduling_requirement=self.scheduling_requirement,
            job_duration_requirement=self.job_duration_requirement,
            checkpoint_requirement=self.checkpoint_requirement,
            max_model_calls=self.max_model_calls,
        )


@dataclass(frozen=True, slots=True)
class FundingSettings:
    """The operator's grant, and the authority they claim for it.

    Funding is the one stage no model touches. The numbers here are an
    act of authorization, which is why they are written down in the same
    file as everything else and hashed with it.
    """

    granted: ResearchBudget
    authority: str
    label: str

    def authorization(
        self, *, admission_record_id: str
    ) -> FundingAuthorization:
        return FundingAuthorization(
            admission_record_id=admission_record_id,
            granted=self.granted,
            authority=self.authority,
        )

    def directive(
        self, *, admission_record_id: str, authorization_id: str
    ) -> RunDirective:
        return RunDirective(
            admission_record_id=admission_record_id,
            authorization_id=authorization_id,
            label=self.label,
        )


@dataclass(frozen=True, slots=True)
class ExperimentationSettings:
    max_steps: int = 24


# -- the config ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything one investigation needs that is not an id."""

    label: str
    model: str
    brief: ResearchBrief
    snapshot: CfpSnapshot
    ideation: IdeationSettings
    prior_art: PriorArtSettings
    selection: SelectionSettings
    admission: AdmissionSettings
    funding: FundingSettings
    experimentation: ExperimentationSettings
    request_timeout_seconds: float = 240.0
    stop_after: StageName | None = None

    def validate(self) -> None:
        """Build every directive the chain will use and throw them away.

        The point is the construction, not the result: each directive's
        own validation is the authority on its own ceilings, and running
        it here means a config that cannot produce a legal directive is
        rejected before the first model call rather than after several.
        """
        self.ideation.directive(
            assessment_id=f"madq_{_PLACEHOLDER}",
            snapshot_id=self.snapshot.id,
        )
        self.prior_art.directive(
            ideation_run_record_id=f"irun_{_PLACEHOLDER}"
        )
        self.selection.directive(
            prior_art_run_record_id=f"prun_{_PLACEHOLDER}"
        )
        self.admission.directive(
            selection_run_record_id=f"srun_{_PLACEHOLDER}"
        )
        authorization = self.funding.authorization(
            admission_record_id=f"arun_{_PLACEHOLDER}"
        )
        self.funding.directive(
            admission_record_id=f"arun_{_PLACEHOLDER}",
            authorization_id=authorization.id,
        )


def load_config(path: Path | str) -> tuple[RunConfig, Mapping[str, object]]:
    """Read one config file, returning the parsed config and the exact
    document it came from — the second is what gets recorded."""
    location = Path(path)
    try:
        raw = location.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {location}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{location} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{location} must hold a JSON object")
    return parse_config(payload), payload


def parse_config(payload: Mapping[str, object]) -> RunConfig:
    """Read a config document into typed settings, or refuse it."""
    _known(
        payload,
        {
            "label",
            "model",
            "request_timeout_seconds",
            "stop_after",
            "brief",
            "cfp",
            "ideation",
            "prior_art",
            "selection",
            "admission",
            "funding",
            "experimentation",
        },
        "config",
    )
    label = _text(payload, "label", "config")
    if len(label) > MAX_LABEL_CHARS:
        raise ConfigError(
            f"config.label must be at most {MAX_LABEL_CHARS} characters, "
            f"got {len(label)}"
        )
    brief = _brief(_object(payload, "brief", "config"))
    prior_art = _prior_art(
        _object(payload, "prior_art", "config", default=True), brief
    )
    config = RunConfig(
        label=label,
        model=_text(payload, "model", "config"),
        brief=brief,
        snapshot=_snapshot(_object(payload, "cfp", "config")),
        ideation=_ideation(_object(payload, "ideation", "config", default=True)),
        prior_art=prior_art,
        selection=_selection(_object(payload, "selection", "config")),
        admission=_admission(_object(payload, "admission", "config")),
        funding=_funding(_object(payload, "funding", "config"), label),
        experimentation=_experimentation(
            _object(payload, "experimentation", "config", default=True)
        ),
        request_timeout_seconds=_number(
            payload, "request_timeout_seconds", "config", default=240.0
        ),
        stop_after=_stop_after(payload),
    )
    try:
        config.validate()
    except ValueError as exc:
        raise ConfigError(f"the config describes an illegal run: {exc}") from exc
    return config


# -- section readers ----------------------------------------------------------


def _brief(payload: Mapping[str, object]) -> ResearchBrief:
    _known(
        payload,
        {
            "topic",
            "cutoff_date",
            "recent_window_start",
            "workshop_hints",
            "max_queries_per_family",
            "results_per_query",
            "max_screened_sources",
            "max_extracted_sources",
            "max_model_calls",
            "refinement_rounds",
            "max_refinement_queries",
        },
        "brief",
    )
    try:
        return ResearchBrief(
            topic=_text(payload, "topic", "brief"),
            cutoff_date=_text(payload, "cutoff_date", "brief"),
            recent_window_start=_text(payload, "recent_window_start", "brief"),
            workshop_hints=_strings(payload, "workshop_hints", "brief"),
            max_queries_per_family=_integer(
                payload, "max_queries_per_family", "brief", default=2
            ),
            results_per_query=_integer(
                payload, "results_per_query", "brief", default=25
            ),
            max_screened_sources=_integer(
                payload, "max_screened_sources", "brief", default=120
            ),
            max_extracted_sources=_integer(
                payload, "max_extracted_sources", "brief", default=40
            ),
            max_model_calls=_integer(
                payload, "max_model_calls", "brief", default=60
            ),
            refinement_rounds=_integer(
                payload, "refinement_rounds", "brief", default=1
            ),
            max_refinement_queries=_integer(
                payload, "max_refinement_queries", "brief", default=3
            ),
        )
    except ValueError as exc:
        raise ConfigError(f"brief: {exc}") from exc


def _snapshot(payload: Mapping[str, object]) -> CfpSnapshot:
    _known(payload, {"source_url", "supplied_at", "text"}, "cfp")
    try:
        return CfpSnapshot(
            source_url=_text(payload, "source_url", "cfp"),
            supplied_at=_text(payload, "supplied_at", "cfp"),
            text=_text(payload, "text", "cfp"),
        )
    except ValueError as exc:
        raise ConfigError(f"cfp: {exc}") from exc


def _ideation(payload: Mapping[str, object]) -> IdeationSettings:
    _known(payload, {"max_candidates", "max_model_calls"}, "ideation")
    return IdeationSettings(
        max_candidates=_integer(
            payload, "max_candidates", "ideation", default=5
        ),
        max_model_calls=_integer(
            payload, "max_model_calls", "ideation", default=4
        ),
    )


def _prior_art(
    payload: Mapping[str, object], brief: ResearchBrief
) -> PriorArtSettings:
    """The challenge's retrieval window defaults to the brief's.

    Two windows over one investigation would be a way to challenge a
    candidate against a different literature than the one that suggested
    it, so the operator has to mean it to get it.
    """
    _known(
        payload,
        {
            "cutoff_date",
            "recent_window_start",
            "results_per_query",
            "max_screened_per_candidate",
            "max_compared_works",
            "max_model_calls",
        },
        "prior_art",
    )
    return PriorArtSettings(
        cutoff_date=_text(
            payload, "cutoff_date", "prior_art", default=brief.cutoff_date
        ),
        recent_window_start=_text(
            payload,
            "recent_window_start",
            "prior_art",
            default=brief.recent_window_start,
        ),
        results_per_query=_integer(
            payload, "results_per_query", "prior_art", default=5
        ),
        max_screened_per_candidate=_integer(
            payload, "max_screened_per_candidate", "prior_art", default=35
        ),
        max_compared_works=_integer(
            payload, "max_compared_works", "prior_art", default=4
        ),
        max_model_calls=_integer(
            payload, "max_model_calls", "prior_art", default=36
        ),
    )


def _selection(payload: Mapping[str, object]) -> SelectionSettings:
    _known(
        payload,
        {
            "compute_constraint",
            "data_constraint",
            "time_constraint",
            "experimental_constraint",
            "max_eligible_candidates",
            "max_model_calls",
        },
        "selection",
    )
    return SelectionSettings(
        compute_constraint=_text(payload, "compute_constraint", "selection"),
        data_constraint=_text(payload, "data_constraint", "selection"),
        time_constraint=_text(payload, "time_constraint", "selection"),
        experimental_constraint=_text(
            payload, "experimental_constraint", "selection"
        ),
        max_eligible_candidates=_integer(
            payload, "max_eligible_candidates", "selection", default=5
        ),
        max_model_calls=_integer(
            payload, "max_model_calls", "selection", default=4
        ),
    )


def _admission(payload: Mapping[str, object]) -> AdmissionSettings:
    _known(
        payload,
        {
            "scheduling_requirement",
            "job_duration_requirement",
            "checkpoint_requirement",
            "max_model_calls",
        },
        "admission",
    )
    return AdmissionSettings(
        scheduling_requirement=_text(
            payload, "scheduling_requirement", "admission"
        ),
        job_duration_requirement=_text(
            payload, "job_duration_requirement", "admission"
        ),
        checkpoint_requirement=_text(
            payload, "checkpoint_requirement", "admission"
        ),
        max_model_calls=_integer(
            payload, "max_model_calls", "admission", default=2
        ),
    )


def _funding(payload: Mapping[str, object], label: str) -> FundingSettings:
    _known(payload, {"granted", "authority", "label"}, "funding")
    granted = _grant(_object(payload, "granted", "funding"))
    authority = _text(payload, "authority", "funding")
    if len(authority) > MAX_AUTHORITY_CHARS:
        raise ConfigError(
            f"funding.authority must be at most {MAX_AUTHORITY_CHARS} "
            f"characters, got {len(authority)}"
        )
    return FundingSettings(
        granted=granted,
        authority=authority,
        label=_text(payload, "label", "funding", default=label),
    )


def _grant(payload: Mapping[str, object]) -> ResearchBudget:
    _known(
        payload,
        {"wall_clock_seconds", "gpu_hours", "usd", "model_tokens"},
        "funding.granted",
    )
    try:
        granted = ResearchBudget(
            wall_clock_seconds=_number(
                payload, "wall_clock_seconds", "funding.granted", default=0.0
            ),
            gpu_hours=_number(
                payload, "gpu_hours", "funding.granted", default=0.0
            ),
            usd=_number(payload, "usd", "funding.granted", default=0.0),
            model_tokens=_integer(
                payload, "model_tokens", "funding.granted", default=0
            ),
        )
    except ValueError as exc:
        raise ConfigError(f"funding.granted: {exc}") from exc
    if granted.is_exhausted:
        raise ConfigError(
            "funding.granted buys nothing; a run funded with zero of every "
            "resource cannot take a step"
        )
    for name, value, ceiling in (
        ("wall_clock_seconds", granted.wall_clock_seconds, MAX_WALL_CLOCK_SECONDS),
        ("gpu_hours", granted.gpu_hours, MAX_GPU_HOURS),
        ("usd", granted.usd, MAX_USD),
        ("model_tokens", float(granted.model_tokens), float(MAX_MODEL_TOKENS)),
    ):
        if value > ceiling:
            raise ConfigError(
                f"funding.granted.{name} is {value}, above the authorized "
                f"ceiling of {ceiling}"
            )
    return granted


def _experimentation(payload: Mapping[str, object]) -> ExperimentationSettings:
    _known(payload, {"max_steps"}, "experimentation")
    max_steps = _integer(payload, "max_steps", "experimentation", default=24)
    if not 1 <= max_steps <= MAX_STEPS_CEILING:
        raise ConfigError(
            f"experimentation.max_steps must be in 1..{MAX_STEPS_CEILING}, "
            f"got {max_steps}"
        )
    return ExperimentationSettings(max_steps=max_steps)


def _stop_after(payload: Mapping[str, object]) -> StageName | None:
    raw = payload.get("stop_after")
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise ConfigError("config.stop_after must be a stage name")
    named = ", ".join(str(stage) for stage in CHAIN_ORDER)
    try:
        parsed = StageName(raw)
    except ValueError as exc:
        raise ConfigError(
            f"config.stop_after names no stage of the chain: {raw!r} "
            f"(one of: {named})"
        ) from exc
    if parsed not in CHAIN_ORDER:
        # The enum also holds post-run export seats; a walk cannot stop
        # after something it never walks to.
        raise ConfigError(
            f"config.stop_after names no stage of the chain: {raw!r} "
            f"(one of: {named})"
        )
    return parsed


# -- readers ------------------------------------------------------------------


def _known(
    payload: Mapping[str, object], allowed: set[str], where: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(unknown)}; a key that is "
            f"not read is a typo silently selecting a default"
        )


def _object(
    payload: Mapping[str, object], key: str, where: str, *, default: bool = False
) -> Mapping[str, object]:
    if key not in payload:
        if default:
            return {}
        raise ConfigError(f"{where}: missing required section {key!r}")
    value = payload[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{where}.{key} must be an object")
    return value


def _text(
    payload: Mapping[str, object],
    key: str,
    where: str,
    *,
    default: str | None = None,
) -> str:
    if key not in payload:
        if default is not None:
            return default
        raise ConfigError(f"{where}: missing required key {key!r}")
    value = payload[key]
    if not isinstance(value, str):
        raise ConfigError(f"{where}.{key} must be a string")
    return value


def _integer(
    payload: Mapping[str, object], key: str, where: str, *, default: int
) -> int:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{where}.{key} must be an integer")
    return value


def _number(
    payload: Mapping[str, object], key: str, where: str, *, default: float
) -> float:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number")
    return float(value)


def _strings(
    payload: Mapping[str, object], key: str, where: str
) -> tuple[str, ...]:
    if key not in payload:
        return ()
    value = payload[key]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{where}.{key} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{where}.{key} must be a list of strings")
        items.append(item)
    return tuple(items)
