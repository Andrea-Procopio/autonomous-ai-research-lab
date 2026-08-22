"""The gated-call discipline every publication seat shares.

One structured model call under one deterministic gate, with the
bounded corrective-call rules the admission door established: the call
budget is checked before every call; accounting reaches the ledger
exactly once whether the call succeeds or fails; only a schema
violation is a correctable outcome among provider errors; a gate
rejection earns the same treatment as a schema violation — the payload
is preserved as evidence, at most ``max_corrective_calls`` retries
carry exactly the mechanical rules that fired, and then the refusal is
final.

The seats differ only in what this module takes as parameters: where a
rejected payload is preserved, what the corrective feedback restates,
and which typed error an exhausted refusal raises. Everything else —
including the discipline that mechanical rules, never a preferred
wording, trigger a retry — is one implementation, so the author, the
reviewer, and every later seat refuse identically.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..runtime.providers import (
    Message,
    MessageRole,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    StructuredOutputError,
    UsageLedger,
)
from .manuscript import AuthorCall, ProseRejection

#: Preserves one refused payload: (reasons, request_fingerprint,
#: response_id, payload, repair) -> anything.
PreserveRejected = Callable[
    [tuple[tuple[str, str], ...], str, str, object, int], object
]


class _Spend:
    __slots__ = ("calls", "input_tokens", "limit", "output_tokens")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


class GatedCaller:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        ledger: UsageLedger,
        max_corrective_calls: int,
    ) -> None:
        self._provider = provider
        self._ledger = ledger
        self._max_corrective_calls = max_corrective_calls

    def gated_call(
        self,
        request: ModelRequest,
        *,
        gate: Callable[[Mapping[str, object]], tuple[ProseRejection, ...]],
        preserve: PreserveRejected,
        feedback: Callable[[tuple[ProseRejection, ...]], str],
        exhausted: Callable[[tuple[ProseRejection, ...]], Exception],
        budget_error: Callable[[int], Exception],
        requested_model: str,
    ) -> tuple[Mapping[str, object], AuthorCall]:
        spend = _Spend(1 + self._max_corrective_calls)
        response, schema_error = self._attempt(request, spend, budget_error)
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
            preserve(
                tuple((r.rule, r.detail) for r in rejections),
                fingerprint,
                response_id,
                payload if payload is not None else raw,
                repairs,
            )
            if repairs >= self._max_corrective_calls:
                if schema_error is not None:
                    raise schema_error
                raise exhausted(rejections)
            repairs += 1
            request = _repair_request(
                request, response, rejections, repairs, feedback
            )
            response, schema_error = self._attempt(
                request, spend, budget_error
            )
        assert response is not None
        assert payload is not None  # an absent payload never passes the gate
        return payload, AuthorCall(
            request_fingerprint=response.request_fingerprint,
            response_id=response.id,
            provider=response.provider,
            requested_model=requested_model,
            served_model=response.model,
            provider_request_id=response.request_id,
            latency_seconds=response.latency_seconds,
            input_tokens=spend.input_tokens,
            output_tokens=spend.output_tokens,
            repair_count=repairs,
        )

    def _invoke(
        self,
        request: ModelRequest,
        spend: _Spend,
        budget_error: Callable[[int], Exception],
    ) -> ModelResponse:
        if spend.calls >= spend.limit:
            raise budget_error(spend.limit)
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
        self,
        request: ModelRequest,
        spend: _Spend,
        budget_error: Callable[[int], Exception],
    ) -> tuple[ModelResponse | None, StructuredOutputError | None]:
        try:
            return self._invoke(request, spend, budget_error), None
        except StructuredOutputError as error:
            return None, error


def _repair_request(
    base: ModelRequest,
    failed: ModelResponse | None,
    rejections: tuple[ProseRejection, ...],
    attempt: int,
    feedback: Callable[[tuple[ProseRejection, ...]], str],
) -> ModelRequest:
    """One corrective request: the failed reply plus the seat's feedback
    text carrying every deterministic rule that fired."""
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
            Message(role=MessageRole.USER, content=feedback(rejections)),
        ),
        schema=base.schema,
        max_output_tokens=base.max_output_tokens,
        temperature=base.temperature,
        timeout_seconds=base.timeout_seconds,
        metadata={**dict(base.metadata), "corrective": str(attempt)},
    )
