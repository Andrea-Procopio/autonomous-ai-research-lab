"""The venue simulator: an impression instrument, honestly labeled.

The faithfulness reviewer judges the manuscript against the record; this
seat judges how the *submission* reads — to a reviewer who, like a real
one, sees only the document. The ensemble is deliberately blind to the
packet: each lens receives the rendered ``main.tex`` and its
bibliography, nothing else, so what it measures is exactly what a venue
would measure.

What stays out, deliberately: bias prompts (an instruction to lean
accept or reject under uncertainty steers the very reading the
instrument exists to take), reflection rounds (the bounded
corrective-call discipline is the only retry), temperature-diverse
sampling (diversity comes from three deterministic lenses — rigor,
clarity, significance — at temperature zero), and any way for the model
to output a verdict: the review form has no accept/reject property, the
aggregate is an arithmetic median computed by trusted code, and the
outcome is that median against an operator-configured bar.

A score here is an instrument reading, recorded write-once with its
call's provenance. It informs; it is never the objective — a lab that
optimized for it would be optimizing exactly the appearance its whole
architecture exists to distrust.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from ..core.ids import content_id
from ..core.serialize import to_jsonable
from ..runtime.providers import (
    Message,
    MessageRole,
    ModelProvider,
    ModelRequest,
    OutputSchema,
    UsageLedger,
)
from ._calling import GatedCaller
from .manuscript import AuthorCall, ProseRejection

if TYPE_CHECKING:
    from .store import SimulationStore

#: Deterministic diversity: three reviewer perspectives at temperature
#: zero, in place of sampling the same prompt at temperature 0.75.
LENSES: Final[tuple[tuple[str, str], ...]] = (
    (
        "rigor",
        "Weigh most heavily the soundness of the experimental design, "
        "the statistical treatment, and whether every claim is "
        "supported by the evidence the paper itself presents.",
    ),
    (
        "clarity",
        "Weigh most heavily the presentation: whether a reader can "
        "follow the method, reproduce the work from what is written, "
        "and trust the exposition to say neither more nor less than "
        "the work shows.",
    ),
    (
        "significance",
        "Weigh most heavily the contribution: whether the question "
        "matters, what the field learns from the answer, and how the "
        "work relates to what is already known.",
    ),
)

DIMENSIONS: Final = (
    "originality",
    "quality",
    "clarity",
    "significance",
    "soundness",
    "presentation",
    "contribution",
)

_FOUR: Final = {"type": "integer", "enum": [1, 2, 3, 4]}

SIMULATOR_INSTRUCTION: Final = (
    "You are a reviewer for {venue}. The user message is one "
    "submission, shown exactly as submitted: its LaTeX source and its "
    "bibliography. You see nothing else — no supplementary record, no "
    "author correspondence. Review it as a conscientious venue "
    "reviewer would: state what the paper actually shows, be honest "
    "about your uncertainty, and never reward or punish what you "
    "cannot verify from the shown text. {anonymity}Fill every field "
    "of the review form. {focus}"
)


class SimulationError(RuntimeError):
    """The simulation cannot honestly be produced."""


class SimulationRejectedError(SimulationError):
    """A lens review failed its mechanical gate after every corrective
    call."""


@dataclass(frozen=True, slots=True)
class VenueReview:
    """One lens's filled review form, with the call that produced it."""

    manuscript_id: str
    packet_id: str
    venue_name: str
    tex_sha256: str
    lens: str
    summary: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    questions: tuple[str, ...]
    originality: int
    quality: int
    clarity: int
    significance: int
    soundness: int
    presentation: int
    contribution: int
    overall: int
    confidence: int
    call: AuthorCall
    review_id: str = field(default="")

    def __post_init__(self) -> None:
        derived = content_id("vrev", _identity_payload(self, "review_id"))
        if not self.review_id:
            object.__setattr__(self, "review_id", derived)
        elif self.review_id != derived:
            raise SimulationError(
                f"venue review carries id {self.review_id}, but its "
                f"content derives {derived}; the record does not "
                f"survive itself"
            )


@dataclass(frozen=True, slots=True)
class SimulationRecord:
    """One ensemble's aggregate: the reviews it read, the medians
    trusted code took, and the bar the outcome was derived against."""

    manuscript_id: str
    packet_id: str
    venue_name: str
    tex_sha256: str
    bar: int
    review_ids: tuple[str, ...]
    medians: tuple[tuple[str, float], ...]
    meets_bar: bool
    simulation_id: str = field(default="")

    def __post_init__(self) -> None:
        derived = content_id(
            "vsim", _identity_payload(self, "simulation_id")
        )
        if not self.simulation_id:
            object.__setattr__(self, "simulation_id", derived)
        elif self.simulation_id != derived:
            raise SimulationError(
                f"simulation carries id {self.simulation_id}, but its "
                f"content derives {derived}; the record does not "
                f"survive itself"
            )

    def median(self, key: str) -> float:
        for name, value in self.medians:
            if name == key:
                return value
        raise KeyError(key)


@dataclass(frozen=True, slots=True)
class PolishRecord:
    """A polish succession: this below-bar simulation demanded a
    revision, and that manuscript is it. Deliberately not a
    :class:`~.review.RevisionRecord` — a polish must never disable, or
    masquerade as, the faithfulness revise cycle."""

    packet_id: str
    simulation_id: str
    superseded_manuscript_id: str
    revision_manuscript_id: str
    polish_id: str = field(default="")

    def __post_init__(self) -> None:
        derived = content_id("plsh", _identity_payload(self, "polish_id"))
        if not self.polish_id:
            object.__setattr__(self, "polish_id", derived)
        elif self.polish_id != derived:
            raise SimulationError(
                f"polish carries id {self.polish_id}, but its content "
                f"derives {derived}; the record does not survive itself"
            )


def _identity_payload(record: object, id_field: str) -> str:
    payload = to_jsonable(record)
    assert isinstance(payload, dict)
    payload.pop(id_field, None)
    return json.dumps(payload, sort_keys=True)


def review_form_schema() -> Mapping[str, object]:
    """The NeurIPS-form review as a closed schema. Every score is an
    integer enum, and there is no verdict property: accept or reject is
    unexpressible — the outcome belongs to trusted code."""
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "weaknesses": {"type": "array", "items": {"type": "string"}},
            "questions": {"type": "array", "items": {"type": "string"}},
            **{name: dict(_FOUR) for name in DIMENSIONS},
            "overall": {
                "type": "integer",
                "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            "confidence": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        },
        "required": [
            "summary",
            "strengths",
            "weaknesses",
            "questions",
            *DIMENSIONS,
            "overall",
            "confidence",
        ],
    }


def aggregate(
    reviews: Sequence[VenueReview],
) -> tuple[tuple[str, float], ...]:
    """Per-field medians across the ensemble, as floats always — the
    same value must serialize the same way whether the median fell on a
    review's own integer or between two."""
    if not reviews:
        raise SimulationError(
            "a simulation with zero reviews is not a reading"
        )
    return tuple(
        (
            key,
            float(
                statistics.median(
                    getattr(review, key) for review in reviews
                )
            ),
        )
        for key in (*DIMENSIONS, "overall", "confidence")
    )


def meets(bar: int, overall_median: float) -> bool:
    return overall_median >= bar


def render_submission_for_review(main_tex: str, references_bib: str) -> str:
    """What the lens sees: the submission, exactly as submitted."""
    return (
        "=== main.tex ===\n"
        + main_tex
        + "\n=== references.bib ===\n"
        + references_bib
    )


def lens_instruction(
    *, venue_name: str, anonymous: bool, focus: str
) -> str:
    anonymity = (
        "The venue is double-blind; do not speculate about authorship. "
        if anonymous
        else ""
    )
    return SIMULATOR_INSTRUCTION.format(
        venue=venue_name, anonymity=anonymity, focus=focus
    )


class VenueSimulator:
    """One lens review per call, under the shared calling discipline.

    The only mechanical gate is a non-empty summary: every score is
    already bounded by the schema, and the opinions are the model's —
    steering them would defeat the instrument.
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        ledger: UsageLedger,
        store: SimulationStore,
        max_output_tokens: int = 8192,
        temperature: float = 0.0,
        request_timeout_seconds: float = 240.0,
        max_corrective_calls: int = 1,
    ) -> None:
        self._model = model
        self._store = store
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._request_timeout_seconds = request_timeout_seconds
        self._caller = GatedCaller(
            provider=provider,
            ledger=ledger,
            max_corrective_calls=max_corrective_calls,
        )

    def review(
        self,
        *,
        main_tex: str,
        references_bib: str,
        venue_name: str,
        anonymous: bool,
        lens: str,
        focus: str,
        manuscript_id: str,
        packet_id: str,
        tex_sha256: str,
    ) -> VenueReview:
        request = ModelRequest(
            model=self._model,
            instruction=lens_instruction(
                venue_name=venue_name, anonymous=anonymous, focus=focus
            ),
            messages=(
                Message(
                    role=MessageRole.USER,
                    content=render_submission_for_review(
                        main_tex, references_bib
                    ),
                ),
            ),
            schema=OutputSchema(
                name="venue_review", json_schema=review_form_schema()
            ),
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            timeout_seconds=self._request_timeout_seconds,
            metadata={
                "stage": "simulate",
                "manuscript": manuscript_id,
                "venue": venue_name,
                "lens": lens,
            },
        )
        payload, call = self._caller.gated_call(
            request,
            gate=_summary_gate,
            preserve=lambda reasons, fingerprint, response_id, raw, repair: (
                self._store.preserve_rejected(
                    manuscript_id=manuscript_id,
                    packet_id=packet_id,
                    lens=lens,
                    tex_sha256=tex_sha256,
                    reasons=reasons,
                    request_fingerprint=fingerprint,
                    response_id=response_id,
                    payload=raw,
                    repair=repair,
                )
            ),
            feedback=_feedback,
            exhausted=lambda rejections: SimulationRejectedError(
                f"the {lens} lens review failed its mechanical gate: "
                + "; ".join(f"{r.rule}: {r.detail}" for r in rejections)
            ),
            budget_error=lambda limit: SimulationRejectedError(
                f"the {lens} lens call budget ({limit}) is spent; "
                f"refusing the call that would exceed it"
            ),
            requested_model=self._model,
        )
        return VenueReview(
            manuscript_id=manuscript_id,
            packet_id=packet_id,
            venue_name=venue_name,
            tex_sha256=tex_sha256,
            lens=lens,
            summary=str(payload["summary"]),
            strengths=_texts(payload, "strengths"),
            weaknesses=_texts(payload, "weaknesses"),
            questions=_texts(payload, "questions"),
            originality=int(payload["originality"]),  # type: ignore[call-overload]
            quality=int(payload["quality"]),  # type: ignore[call-overload]
            clarity=int(payload["clarity"]),  # type: ignore[call-overload]
            significance=int(payload["significance"]),  # type: ignore[call-overload]
            soundness=int(payload["soundness"]),  # type: ignore[call-overload]
            presentation=int(payload["presentation"]),  # type: ignore[call-overload]
            contribution=int(payload["contribution"]),  # type: ignore[call-overload]
            overall=int(payload["overall"]),  # type: ignore[call-overload]
            confidence=int(payload["confidence"]),  # type: ignore[call-overload]
            call=call,
        )


def _summary_gate(
    payload: Mapping[str, object],
) -> tuple[ProseRejection, ...]:
    if not str(payload.get("summary", "")).strip():
        return (
            ProseRejection(
                "empty_summary",
                "a review with no summary took no reading; state what "
                "the paper shows before scoring it",
            ),
        )
    return ()


def _texts(
    payload: Mapping[str, object], key: str
) -> tuple[str, ...]:
    raw = payload[key]
    assert isinstance(raw, Sequence) and not isinstance(raw, str)
    return tuple(str(found) for found in raw)


def _feedback(rejections: tuple[ProseRejection, ...]) -> str:
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    return (
        f"Your review was rejected by the mechanical form gate. "
        f"Nothing was recorded. The rules that fired:\n{rules}\n"
        f"Return one corrected review now, filling every field of the "
        f"form."
    )
