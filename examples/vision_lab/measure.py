"""What this lab can measure, and the honest refusal when it cannot.

An admitted prediction's metric is the verbatim string admission encoded:
``difference in {base}: {higher arm} minus {lower arm}``
(:func:`autonomous_research_lab.admission.admitter.encoded_metric`), and
the mechanical prediction test will read ``result.metrics[that string]``
exactly. So a lab's capability is not a vibe — it is the closed set of
contrasts its trusted templates genuinely compute, and the first thing
composition does with a funded state is ask whether every admitted
prediction is in that set.

When one is not, the answer is a typed refusal, not a doomed run.
:class:`UnmeasurablePredictionsError` subclasses
``ExperimentationUnavailableError``, so the controller records a REFUSED
stage and exits 2 — the same first-class "this instrument cannot do
that" the default lab gives the whole seventh stage. An experiment that
could only ever come back INCONCLUSIVE is not an experiment; refusing to
fund it is the scientific position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from autonomous_research_lab.control.lab import ExperimentationUnavailableError
from autonomous_research_lab.core.prediction import Prediction

_GRAMMAR: Final = re.compile(r"difference in (.+?): (.+) minus (.+)")
"""The exact shape ``encoded_metric`` renders. Mirrored, not imported:
admission owns the encoder, this lab owns a parser, and the round-trip
test in ``tests/test_vision_lab.py`` holds the two together."""


class UnmeasurablePredictionsError(ExperimentationUnavailableError):
    """The funded state asks for observables this lab cannot produce."""


@dataclass(frozen=True, slots=True)
class Contrast:
    """One two-arm comparison, normalized for table lookup."""

    base_metric: str
    higher_arm: str
    lower_arm: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (
            _normalized(self.base_metric),
            _normalized(self.higher_arm),
            _normalized(self.lower_arm),
        )


#: The closed table of contrasts this lab's templates compute, keyed by
#: normalized (base metric, higher arm, lower arm) and valued by the
#: template that measures it. This *is* the lab's capability declaration:
#: nothing outside it is runnable, and extending the lab means extending
#: this table alongside a template that earns the new row.
SUPPORTED_CONTRASTS: Final[dict[tuple[str, str, str], str]] = {
    (
        "linear probe accuracy",
        "trained encoder",
        "randomly initialized encoder",
    ): "vision-encoder-contrast-v1",
    (
        "linear probe accuracy",
        "augmented training",
        "plain training",
    ): "vision-augmentation-contrast-v1",
}


def parse_contrast(metric: str) -> Contrast | None:
    """The contrast a metric string encodes, or ``None`` where it is not
    in admission's grammar at all."""
    matched = _GRAMMAR.fullmatch(metric.strip())
    if matched is None:
        return None
    return Contrast(
        base_metric=matched.group(1),
        higher_arm=matched.group(2),
        lower_arm=matched.group(3),
    )


def template_for(metric: str) -> str | None:
    """Which template measures this admitted metric, or ``None``."""
    contrast = parse_contrast(metric)
    if contrast is None:
        return None
    return SUPPORTED_CONTRASTS.get(contrast.key)


def require_measurable(predictions: tuple[Prediction, ...]) -> None:
    """Refuse, with every unmeasurable string named, or return quietly."""
    if not predictions:
        raise UnmeasurablePredictionsError(
            "the funded state carries no predictions; there is nothing "
            "this lab could measure"
        )
    unmeasurable = [
        prediction.metric
        for prediction in predictions
        if template_for(prediction.metric) is None
    ]
    if unmeasurable:
        supported = "; ".join(
            f"difference in {base}: {high} minus {low}"
            for base, high, low in SUPPORTED_CONTRASTS
        )
        listed = "; ".join(repr(metric) for metric in unmeasurable)
        raise UnmeasurablePredictionsError(
            f"this lab cannot measure {len(unmeasurable)} of the admitted "
            f"prediction(s): {listed}. Its templates measure exactly: "
            f"{supported}. An experiment that cannot report the admitted "
            f"metric verbatim could only ever be INCONCLUSIVE, so the "
            f"honest answer is refusal, not a doomed run."
        )


def _normalized(part: str) -> str:
    return " ".join(part.casefold().split())
