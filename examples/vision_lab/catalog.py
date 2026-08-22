"""The trusted template catalog, built per run from the admitted record.

A template here is a complete training program with fenced *fixed
regions* — seeding, data loading, splits, the probe, the control, and
every byte that writes ``metrics.json`` — and exactly one *slot*, the
encoder architecture, which is what the engineer's model completes.
Everything that measures is trusted code; the model's contribution is
narrow, real, and checkable.

Two trusted substitutions happen at catalog build, both deterministic:

* the admitted prediction's verbatim metric string replaces the
  ``__ARL_PRIMARY_METRIC__`` placeholder, so the exact-match contract
  between admission, the spec, and the executor's ``metrics.json`` is
  kept by construction rather than by a model transcribing a long key;
* the capability's cost estimate picks up the deployment's GPU
  occupancy, so a reservation covers what the executor will actually
  bill.

Because the template id is a content id over the substituted source,
the per-run substitution is fully recorded in provenance — every
``ImplementationRecord`` carries the template id and sha256 it was
completed from.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from autonomous_research_lab.core.budget import ResourceCost
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.roles.engineer import ImplementationTemplate
from autonomous_research_lab.roles.planner import (
    TemplateCapability,
    TemplateCatalog,
)
from autonomous_research_lab.runtime.verification import PositiveControl

from .measure import UnmeasurablePredictionsError, template_for

TEMPLATES_DIR: Final = Path(__file__).resolve().parent / "templates"

PLACEHOLDER: Final = "__ARL_PRIMARY_METRIC__"

_FIXED: Final = re.compile(
    r"# ARL-FIXED-BEGIN[^\n]*\n(.*?)# ARL-FIXED-END", re.DOTALL
)
_SLOT: Final = re.compile(
    r"(# ARL-SLOT-BEGIN[^\n]*\n)(.*?)(# ARL-SLOT-END)", re.DOTALL
)

#: Which template file serves each catalog entry, per trainer kind.
REAL_SOURCES: Final[Mapping[str, str]] = {
    "vision-encoder-contrast-v1": "vision_encoder_v1.py",
    "vision-augmentation-contrast-v1": "vision_augment_v1.py",
}
STUB_SOURCES: Final[Mapping[str, str]] = {
    "vision-encoder-contrast-v1": "stub_trainer_v1.py",
}

#: The metrics every faithful completion of a template reports beside
#: the substituted primary key.
STATIC_METRICS: Final[Mapping[str, tuple[str, ...]]] = {
    "vision-encoder-contrast-v1": (
        "trained_encoder_probe_top1",
        "random_encoder_probe_top1",
        "encoder_train_loss_final",
        "tiny_subset_overfit_top1",
        "n_probe_train",
        "n_probe_eval",
    ),
    "vision-augmentation-contrast-v1": (
        "augmented_probe_top1",
        "plain_probe_top1",
        "encoder_train_loss_final",
        "tiny_subset_overfit_top1",
        "n_probe_train",
        "n_probe_eval",
    ),
}

OVERFIT_CONTROL: Final = PositiveControl(
    name="tiny_subset_overfit",
    metric="tiny_subset_overfit_top1",
    comparator=Comparator.GREATER_OR_EQUAL,
    threshold=0.95,
    rationale=(
        "a faithful probe pipeline must fit a memorizable, separable "
        "subset scored on itself; missing it indicts the instrument, "
        "never the science"
    ),
)

WALL_CLOCK_ESTIMATE: Final = {"real": 600.0, "stub": 30.0}


def fixed_regions(source: str) -> tuple[str, ...]:
    """The fenced regions a completion must preserve byte-for-byte."""
    return tuple(match.group(1) for match in _FIXED.finditer(source))


def fill_slot(source: str, body: str) -> str:
    """The template with its one slot replaced by ``body`` — how a
    trusted fixture completion is built for scripted runs and tests."""
    filled, count = _SLOT.subn(
        lambda match: f"{match.group(1)}{body}\n{match.group(3)}",
        source,
    )
    if count != 1:
        raise ValueError(f"expected exactly one slot, found {count}")
    return filled


def catalog_for(
    predictions: tuple[Prediction, ...],
    *,
    trainer: str = "real",
    gpu_count: int = 0,
) -> TemplateCatalog:
    """The catalog serving exactly these admitted predictions.

    One entry per template the predictions need, its source substituted
    with the admitted metric string, its capability metrics the closed
    set a planner or designer may declare. Anything unservable — an
    unmeasurable prediction, a trainer without a source for the needed
    template, two distinct primary keys competing for one template —
    refuses with the honest error rather than building a catalog that
    could only produce INCONCLUSIVE runs.
    """
    sources = REAL_SOURCES if trainer == "real" else STUB_SOURCES
    needed: dict[str, list[str]] = {}
    for prediction in predictions:
        name = template_for(prediction.metric)
        if name is None:
            raise UnmeasurablePredictionsError(
                f"no template measures {prediction.metric!r}"
            )
        needed.setdefault(name, [])
        if prediction.metric not in needed[name]:
            needed[name].append(prediction.metric)

    entries = []
    for name, metric_keys in sorted(needed.items()):
        if len(metric_keys) != 1:
            raise UnmeasurablePredictionsError(
                f"template {name} would have to emit {len(metric_keys)} "
                f"distinct primary keys ({', '.join(map(repr, metric_keys))}); "
                f"one template reports one contrast"
            )
        filename = sources.get(name)
        if filename is None:
            raise UnmeasurablePredictionsError(
                f"the {trainer} trainer has no source for template {name}"
            )
        raw = (TEMPLATES_DIR / filename).read_text()
        source = raw.replace(PLACEHOLDER, metric_keys[0])
        wall = WALL_CLOCK_ESTIMATE["real" if trainer == "real" else "stub"]
        entries.append(
            TemplateCapability(
                template=ImplementationTemplate(name=name, source=source),
                metrics=(metric_keys[0], *STATIC_METRICS[name]),
                estimated_cost=ResourceCost(
                    wall_clock_seconds=wall,
                    # The reservation must cover what the executor will
                    # bill, occupancy included, or an ordinary run would
                    # read as a budget breach.
                    gpu_hours=wall / 3600.0 * gpu_count,
                ),
                description=(
                    f"measures {metric_keys[0]!r} with trusted fixed "
                    f"measurement code; the engineer completes only the "
                    f"encoder architecture"
                ),
                control=OVERFIT_CONTROL,
            )
        )
    return TemplateCatalog(entries=tuple(entries))


def entry_for_metric(
    catalog: TemplateCatalog, metric: str
) -> TemplateCapability:
    """The capability whose primary key is ``metric``, which exists by
    construction for every admitted prediction the catalog was built
    from."""
    for entry in catalog.entries:
        if entry.metrics and entry.metrics[0] == metric:
            return entry
    raise UnmeasurablePredictionsError(
        f"the catalog serves no entry for {metric!r}"
    )
