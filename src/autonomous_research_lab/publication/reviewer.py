"""The reviewing seat: one gated call that must ground its criticism.

The same calling discipline as the author — budget before every call,
accounting exactly once, preserve-then-correct-then-refuse — with the
grounding gate in the author's place: a finding whose quote is not in
the named section, or whose subject the packet does not print, earns a
corrective call, and a review that cannot ground itself after that is a
typed refusal, not a review.

Before any call, two zero-cost checks run: the manuscript must belong
to the packet, and the stored draft must still pass the author's own
gates — a draft that no longer does means the gates drifted since it
was recorded, and reviewing on top of drift would judge the wrong
document. Then trusted code takes its own reading
(:func:`~.review.deterministic_findings`); the model's grounded
findings join it; and the verdict is derived, never asked for.
"""

from __future__ import annotations

from typing import NoReturn

from ..runtime.providers import (
    Message,
    MessageRole,
    ModelProvider,
    ModelRequest,
    UsageLedger,
)
from ._calling import GatedCaller
from .manuscript import (
    Manuscript,
    ProseRejection,
    gate_prose,
    known_renderings,
)
from .packet import EvidencePacket, render_markdown
from .review import (
    REVIEWER_INSTRUCTION,
    ReviewError,
    ReviewRecord,
    ReviewRejectedError,
    derive_verdict,
    deterministic_findings,
    ground_findings,
    render_prose_for_review,
    require_reviewable,
    review_schema,
)
from .store import ReviewStore


class FaithfulnessReviewer:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        ledger: UsageLedger,
        store: ReviewStore,
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
        self, packet: EvidencePacket, manuscript: Manuscript
    ) -> ReviewRecord:
        """One review of one draft, findings grounded or refused."""
        require_reviewable(packet, manuscript)
        self._backstop(packet, manuscript)
        found = list(deterministic_findings(manuscript, packet))
        request = ModelRequest(
            model=self._model,
            instruction=REVIEWER_INSTRUCTION,
            messages=(
                Message(
                    role=MessageRole.USER,
                    content=(
                        render_markdown(packet)
                        + "\n\n--- the draft's prose sections ---\n\n"
                        + render_prose_for_review(manuscript.sections)
                    ),
                ),
            ),
            schema=review_schema(packet),
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            timeout_seconds=self._request_timeout_seconds,
            metadata={
                "packet": packet.packet_id,
                "manuscript": manuscript.manuscript_id,
                "stage": "review",
            },
        )
        payload, call = self._caller.gated_call(
            request,
            gate=lambda reply: ground_findings(
                reply, manuscript, packet
            )[0],
            preserve=lambda reasons, fingerprint, response_id, raw, repair: (
                self._store.preserve_rejected(
                    manuscript_id=manuscript.manuscript_id,
                    packet_id=packet.packet_id,
                    reasons=reasons,
                    request_fingerprint=fingerprint,
                    response_id=response_id,
                    payload=raw,
                    repair=repair,
                )
            ),
            feedback=_feedback,
            exhausted=lambda rejections: ReviewRejectedError(
                "the review failed to ground itself: "
                + "; ".join(f"{r.rule}: {r.detail}" for r in rejections)
            ),
            budget_error=lambda limit: ReviewRejectedError(
                f"the reviewer's call budget ({limit}) is spent; "
                f"refusing the call that would exceed it"
            ),
            requested_model=self._model,
        )
        _, grounded = ground_findings(payload, manuscript, packet)
        found.extend(grounded)
        findings = tuple(found)
        return ReviewRecord(
            manuscript_id=manuscript.manuscript_id,
            packet_id=packet.packet_id,
            verdict=derive_verdict(findings),
            findings=findings,
            call=call,
        )

    def _backstop(
        self, packet: EvidencePacket, manuscript: Manuscript
    ) -> None:
        """The author's gates, re-run on the stored draft. A recorded
        manuscript that no longer passes them means the gate code
        drifted since it was authored — loud, and before any spend."""
        rejections = gate_prose(
            {
                name: getattr(manuscript.sections, name)
                for name in manuscript.sections.__dataclass_fields__
            },
            allowed=known_renderings(packet),
            bibliography=packet.bibliography,
        )
        if rejections:
            self._drift(manuscript, rejections)

    def _drift(
        self,
        manuscript: Manuscript,
        rejections: tuple[ProseRejection, ...],
    ) -> NoReturn:
        listed = "; ".join(f"{r.rule}: {r.detail}" for r in rejections)
        raise ReviewError(
            f"the recorded draft {manuscript.manuscript_id} no longer "
            f"passes the author's gates ({listed}); the gates have "
            f"drifted since it was authored, and a review on top of "
            f"drift would judge the wrong document"
        )


def _feedback(rejections: tuple[ProseRejection, ...]) -> str:
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    return (
        f"Your review was rejected by the deterministic grounding "
        f"gates. Nothing was recorded. The rules that fired:\n{rules}\n"
        f"Return one corrected review now. Every quote must appear "
        f"verbatim in the section it names; every subject id must be "
        f"one the packet prints; an empty findings list is the correct "
        f"answer for a faithful draft."
    )
