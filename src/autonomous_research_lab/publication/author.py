"""One gated model call that writes prose, and nothing else.

The author borrows the admission door's calling discipline wholesale:
the call budget is checked before every call, accounting reaches the
ledger exactly once whether the call succeeds or fails, only a schema
violation is a correctable outcome among provider errors, and a gate
rejection earns the same treatment as a schema violation — the payload
is preserved as evidence, at most ``max_corrective_calls`` retries carry
exactly the mechanical rules that fired, and then the refusal is final.

What is deliberately absent: any way for the call to touch a store the
science lives in, any charge against the run's grant (the run is settled
and its packet already states the balance — this call's spend is its
own, durable in the accepted :class:`AuthorCall` and in every preserved
rejection), and any retry for taste. The gates judge tokens; wording is
the model's.
"""

from __future__ import annotations

import json
from typing import Final

from ..runtime.providers import (
    Message,
    MessageRole,
    ModelProvider,
    ModelRequest,
    OutputSchema,
    UsageLedger,
)
from ._calling import GatedCaller
from .manuscript import (
    PROSE_SECTIONS,
    Manuscript,
    ManuscriptError,
    ManuscriptRejectedError,
    ProseRejection,
    ProseSections,
    gate_prose,
    known_renderings,
    require_reportable,
)
from .packet import EvidencePacket, render_markdown
from .review import ReviewFinding
from .store import ManuscriptStore

PROSE_SCHEMA: Final = OutputSchema(
    name="manuscript_prose",
    json_schema={
        "type": "object",
        "properties": {
            name: {"type": "string"} for name in PROSE_SECTIONS
        },
        "required": list(PROSE_SECTIONS),
    },
)

AUTHOR_INSTRUCTION: Final = (
    "You are the writing seat of an autonomous research lab. The user "
    "message is this run's evidence packet — the complete, verified "
    "record of what was registered, measured, and judged. It is your "
    "only source of truth.\n"
    "Write the five prose sections of the manuscript: abstract, "
    "introduction, method_narrative, discussion, limitations. The "
    "results, figures, tables, and references are assembled by trusted "
    "code from the packet; do not restate the result table, and never "
    "invent a section or a heading — no line may start with '#'.\n"
    "Numbers: every numeric token you write must appear digit for digit "
    "in the shown packet. Never round, convert units, aggregate, or "
    "compute a percentage. Where a number is not needed, write "
    "numberless prose.\n"
    "Citations: cite only the bracketed source ids exactly as printed "
    "in the References section, e.g. [lits_0123456789abcdef]. No other "
    "citation form exists.\n"
    "Verdicts: describe each claim's standing only in the recorded "
    "verdict's own word (supported, plausible, undetermined, contested, "
    "refuted). Never write 'statistically significant' or claim any "
    "strength the recorded verdict does not state. No novelty or impact "
    "claims. State scope and limitations from what the packet shows — "
    "sample sizes, thresholds, what was not measured."
)


class ManuscriptAuthor:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        ledger: UsageLedger,
        store: ManuscriptStore,
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

    def author(
        self,
        packet: EvidencePacket,
        *,
        revision_of: Manuscript | None = None,
        findings: tuple[ReviewFinding, ...] = (),
    ) -> Manuscript:
        """One accepted draft for this packet, or a typed refusal.

        With ``revision_of`` and ``findings``, the call revises: the
        prior draft and the grounded findings travel with the request,
        and the same gates judge the result — a revision may fix a
        finding, never loosen a constraint.
        """
        if (revision_of is None) != (not findings):
            raise ManuscriptError(
                "a revision names the draft it revises and the findings "
                "that demanded it — one without the other is malformed"
            )
        require_reportable(packet)
        allowed = known_renderings(packet)
        messages = [
            Message(
                role=MessageRole.USER,
                content=render_markdown(packet),
            ),
        ]
        metadata = {"packet": packet.packet_id, "stage": "manuscript"}
        if revision_of is not None:
            rendered = "\n".join(
                f"- {f.section}: {f.issue}: quoted {f.quote!r} "
                f"(record {f.subject_id or 'none'}) — {f.explanation}"
                for f in findings
            )
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=json.dumps(
                        {
                            name: getattr(revision_of.sections, name)
                            for name in PROSE_SECTIONS
                        }
                    ),
                )
            )
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        f"A faithfulness review found these grounded "
                        f"issues in your draft:\n{rendered}\n"
                        f"Revise the draft to fix each finding; change "
                        f"nothing else beyond what fixing requires; "
                        f"every original constraint holds."
                    ),
                )
            )
            metadata["revision_of"] = revision_of.manuscript_id
        request = ModelRequest(
            model=self._model,
            instruction=AUTHOR_INSTRUCTION,
            messages=tuple(messages),
            schema=PROSE_SCHEMA,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            timeout_seconds=self._request_timeout_seconds,
            metadata=metadata,
        )
        payload, call = self._caller.gated_call(
            request,
            gate=lambda found: gate_prose(
                {name: str(found[name]) for name in PROSE_SECTIONS},
                allowed=allowed,
                bibliography=packet.bibliography,
            ),
            preserve=lambda reasons, fingerprint, response_id, raw, repair: (
                self._store.preserve_rejected(
                    packet_id=packet.packet_id,
                    reasons=reasons,
                    request_fingerprint=fingerprint,
                    response_id=response_id,
                    payload=raw,
                    repair=repair,
                )
            ),
            feedback=_feedback,
            exhausted=lambda rejections: ManuscriptRejectedError(
                "the draft was rejected by the deterministic gates: "
                + "; ".join(f"{r.rule}: {r.detail}" for r in rejections)
            ),
            budget_error=lambda limit: ManuscriptRejectedError(
                f"the author's call budget ({limit}) is spent; "
                f"refusing the call that would exceed it"
            ),
            requested_model=self._model,
        )
        return Manuscript(
            packet_id=packet.packet_id,
            sections=ProseSections(
                **{name: str(payload[name]) for name in PROSE_SECTIONS}
            ),
            call=call,
        )


def _feedback(rejections: tuple[ProseRejection, ...]) -> str:
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    return (
        f"Your draft was rejected by the deterministic manuscript gates. "
        f"Nothing was recorded. The rules that fired:\n{rules}\n"
        f"Return one corrected draft now, satisfying every original "
        f"constraint. Every number must appear digit for digit in the "
        f"shown packet; every citation must be a bracketed source id "
        f"from the References; no line may start with '#'."
    )
