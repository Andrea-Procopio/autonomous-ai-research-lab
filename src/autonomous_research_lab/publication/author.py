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

from collections.abc import Callable, Mapping
from typing import Final

from ..runtime.providers import (
    Message,
    MessageRole,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    OutputSchema,
    StructuredOutputError,
    UsageLedger,
)
from .manuscript import (
    PROSE_SECTIONS,
    AuthorCall,
    Manuscript,
    ManuscriptRejectedError,
    ProseRejection,
    ProseSections,
    gate_prose,
    known_renderings,
    require_reportable,
)
from .packet import EvidencePacket, render_markdown
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


class _Spend:
    __slots__ = ("calls", "input_tokens", "limit", "output_tokens")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


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
        self._provider = provider
        self._model = model
        self._ledger = ledger
        self._store = store
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._request_timeout_seconds = request_timeout_seconds
        self._max_corrective_calls = max_corrective_calls

    def author(self, packet: EvidencePacket) -> Manuscript:
        """One accepted draft for this packet, or a typed refusal."""
        require_reportable(packet)
        allowed = known_renderings(packet)
        request = ModelRequest(
            model=self._model,
            instruction=AUTHOR_INSTRUCTION,
            messages=(
                Message(
                    role=MessageRole.USER,
                    content=render_markdown(packet),
                ),
            ),
            schema=PROSE_SCHEMA,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            timeout_seconds=self._request_timeout_seconds,
            metadata={"packet": packet.packet_id, "stage": "manuscript"},
        )
        payload, call = self._gated_call(
            request,
            gate=lambda found: gate_prose(
                {name: str(found[name]) for name in PROSE_SECTIONS},
                allowed=allowed,
                bibliography=packet.bibliography,
            ),
            packet_id=packet.packet_id,
        )
        return Manuscript(
            packet_id=packet.packet_id,
            sections=ProseSections(
                **{name: str(payload[name]) for name in PROSE_SECTIONS}
            ),
            call=call,
        )

    # -- the calling discipline, mirrored from the admission door ------------

    def _invoke(self, request: ModelRequest, spend: _Spend) -> ModelResponse:
        if spend.calls >= spend.limit:
            raise ManuscriptRejectedError(
                f"the author's call budget ({spend.limit}) is spent; "
                f"refusing the call that would exceed it"
            )
        spend.calls += 1
        try:
            response = self._provider.invoke(request)
        except ModelProviderError as error:
            if self._ledger.record_failure(error):
                assert error.accounting is not None
                spend.input_tokens += error.accounting.usage.input_tokens
                spend.output_tokens += error.accounting.usage.output_tokens
            raise
        self._ledger.record(response)
        spend.input_tokens += response.usage.input_tokens
        spend.output_tokens += response.usage.output_tokens
        return response

    def _attempt(
        self, request: ModelRequest, spend: _Spend
    ) -> tuple[ModelResponse | None, StructuredOutputError | None]:
        try:
            return self._invoke(request, spend), None
        except StructuredOutputError as error:
            return None, error

    def _gated_call(
        self,
        request: ModelRequest,
        *,
        gate: Callable[[Mapping[str, object]], tuple[ProseRejection, ...]],
        packet_id: str,
    ) -> tuple[Mapping[str, object], AuthorCall]:
        spend = _Spend(1 + self._max_corrective_calls)
        response, schema_error = self._attempt(request, spend)
        repairs = 0
        while True:
            if schema_error is not None:
                payload: Mapping[str, object] | None = None
                rejections: tuple[ProseRejection, ...] = (
                    ProseRejection(
                        "invalid_structured_output",
                        f"the reply violated the output schema: "
                        f"{schema_error}",
                    ),
                )
                fingerprint, response_id = request.fingerprint, ""
                raw: object = str(schema_error)
            else:
                assert response is not None
                payload = response.structured
                if payload is None:
                    rejections = (
                        ProseRejection(
                            "no_structured_payload",
                            "the reply carried no structured payload",
                        ),
                    )
                else:
                    rejections = gate(payload)
                fingerprint = response.request_fingerprint
                response_id = response.id
                raw = response.text
            if not rejections:
                break
            self._store.preserve_rejected(
                packet_id=packet_id,
                reasons=tuple((r.rule, r.detail) for r in rejections),
                request_fingerprint=fingerprint,
                response_id=response_id,
                payload=payload if payload is not None else raw,
                repair=repairs,
            )
            if repairs >= self._max_corrective_calls:
                if schema_error is not None:
                    raise schema_error
                raise ManuscriptRejectedError(
                    "the draft was rejected by the deterministic gates: "
                    + "; ".join(
                        f"{r.rule}: {r.detail}" for r in rejections
                    )
                )
            repairs += 1
            request = _repair_request(request, response, rejections, repairs)
            response, schema_error = self._attempt(request, spend)
        assert response is not None
        assert payload is not None  # an absent payload never passes the gate
        return payload, AuthorCall(
            request_fingerprint=response.request_fingerprint,
            response_id=response.id,
            provider=response.provider,
            requested_model=self._model,
            served_model=response.model,
            provider_request_id=response.request_id,
            latency_seconds=response.latency_seconds,
            input_tokens=spend.input_tokens,
            output_tokens=spend.output_tokens,
            repair_count=repairs,
        )


def _repair_request(
    base: ModelRequest,
    failed: ModelResponse | None,
    rejections: tuple[ProseRejection, ...],
    attempt: int,
) -> ModelRequest:
    """One corrective request: the failed draft plus every deterministic
    rule that fired. Mechanical rules only — never a preferred wording."""
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    feedback = (
        f"Your draft was rejected by the deterministic manuscript gates. "
        f"Nothing was recorded. The rules that fired:\n{rules}\n"
        f"Return one corrected draft now, satisfying every original "
        f"constraint. Every number must appear digit for digit in the "
        f"shown packet; every citation must be a bracketed source id "
        f"from the References; no line may start with '#'."
    )
    previous = (
        failed.text
        if failed is not None and failed.text
        else "(the previous reply did not satisfy the output schema)"
    )
    return ModelRequest(
        model=base.model,
        instruction=base.instruction,
        messages=(
            *base.messages,
            Message(role=MessageRole.ASSISTANT, content=previous),
            Message(role=MessageRole.USER, content=feedback),
        ),
        schema=base.schema,
        max_output_tokens=base.max_output_tokens,
        temperature=base.temperature,
        timeout_seconds=base.timeout_seconds,
        metadata={**dict(base.metadata), "corrective": str(attempt)},
    )
