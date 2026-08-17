"""The Muse Spark adapter, tested against captured and documented shapes.

Fixture provenance, kept honest per payload below: the success envelope,
the truncation reply, the 401 error envelope, and the extra-fields content
are sanitized copies of responses the real endpoint returned on 2026-08-17
(identifiers zeroed, content shortened). The 429 and 503 error payloads
were NOT captured live: they are derived from the official error-handling
documentation (status codes, error codes, the ``Retry-After`` header),
using the envelope shape the observed 401 confirmed.

No test opens a network connection, and no test touches a real credential:
keys in this file are obvious dummies.
"""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from email.message import Message as EmailMessage
from typing import Any, ClassVar

import pytest

from autonomous_research_lab.runtime import muse
from autonomous_research_lab.runtime.muse import (
    KEY_ENV_VARS,
    MUSE_SPARK_1_2,
    MuseSparkProvider,
    _api_key,
    _error_for_status,
    build_payload,
    parse_response,
)
from autonomous_research_lab.runtime.providers import (
    InvalidModelResponseError,
    Message,
    MessageRole,
    ModelRequest,
    OutputSchema,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransportError,
    StructuredOutputError,
)

HYPOTHESIS_SCHEMA = OutputSchema(
    name="hypothesis_v1",
    json_schema={
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "confidence": {"type": "number"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["statement", "confidence"],
    },
)

#: OBSERVED live (ids zeroed, content shortened): the success envelope.
_CONTENT = json.dumps(
    {"confidence": 0.82, "statement": "The stream is biased toward heads."}
)
_SUCCESS_BODY: dict[str, Any] = {
    "id": "chatcmpl-01a011cb-0000-0000-0000-000000000000",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {"content": _CONTENT, "refusal": None, "role": "assistant"},
            "logprobs": None,
        }
    ],
    "created": 1787004933,
    "model": "muse-spark-1.2",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 1537,
        "prompt_tokens": 52,
        "total_tokens": 1589,
        "completion_tokens_details": {"reasoning_tokens": 1400},
        "prompt_tokens_details": {"cached_tokens": 49},
    },
}
_SUCCESS_HEADERS = {
    "content-type": "application/json",
    "x-request-id": "00000000-0000-4000-8000-000000000000",
    "x-ratelimit-limit-requests": "3000",
    "x-ratelimit-remaining-requests": "2999",
    "x-ratelimit-limit-tokens": "4000000",
    "x-ratelimit-remaining-tokens": "4000000",
}

#: OBSERVED live: max_tokens exhausted by reasoning — finish_reason
#: "length", content null.
_TRUNCATED_BODY: dict[str, Any] = {
    "id": "chatcmpl-01a011ca-0000-0000-0000-000000000000",
    "choices": [
        {
            "finish_reason": "length",
            "index": 0,
            "message": {"content": None, "refusal": None, "role": "assistant"},
            "logprobs": None,
        }
    ],
    "model": "muse-spark-1.2",
    "object": "chat.completion",
    "usage": {"completion_tokens": 2048, "prompt_tokens": 52, "total_tokens": 2100},
}

#: OBSERVED live: the 401 error envelope, verbatim.
_AUTH_ERROR = (
    b'{"error":{"code":"invalid_api_key","message":"Unauthorized",'
    b'"param":null,"type":"authentication_error"}}'
)

#: OBSERVED live: without additionalProperties on the wire, the model added
#: undeclared fields alongside the declared ones (shortened here).
_EXTRA_FIELDS_CONTENT = json.dumps(
    {
        "confidence": 0.92,
        "statement": "The generator is biased toward heads.",
        "null_hypothesis": "The generator is fair with p = 0.5.",
        "test_method": "Two-tailed binomial test at alpha=0.05.",
    }
)


def _request(**overrides: object) -> ModelRequest:
    defaults: dict[str, object] = {
        "model": MUSE_SPARK_1_2,
        "instruction": "Propose one testable hypothesis.",
        "messages": (
            Message(role=MessageRole.USER, content="The stream may be biased."),
        ),
        "schema": HYPOTHESIS_SCHEMA,
        "max_output_tokens": 8192,
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)  # type: ignore[arg-type]


# -- request translation ------------------------------------------------------


def test_the_payload_matches_the_documented_wire_format() -> None:
    payload = build_payload(_request(metadata={"attempt_id": "att_1"}))

    assert payload["model"] == "muse-spark-1.2"
    assert payload["messages"] == [
        {"role": "system", "content": "Propose one testable hypothesis."},
        {"role": "user", "content": "The stream may be biased."},
    ]
    assert payload["max_tokens"] == 8192
    assert payload["temperature"] == 0.0
    response_format = payload["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "hypothesis_v1"
    # Caller metadata is provenance; it is not sent over the wire.
    assert "metadata" not in payload
    json.dumps(payload)  # the deep-frozen schema was thawed for the wire


def test_optional_knobs_are_omitted_not_defaulted() -> None:
    payload = build_payload(
        _request(schema=None, max_output_tokens=None, temperature=None)
    )
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert "response_format" not in payload


def test_an_instructionless_request_sends_no_system_message() -> None:
    payload = build_payload(_request(instruction=""))
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert [m["role"] for m in messages] == ["user"]


def test_closed_schemas_are_made_explicit_on_the_wire() -> None:
    """Observed live: without additionalProperties the model added six
    undeclared fields; with it, none. The validator is closed by default,
    so the wire schema must say so at every object level."""
    nested = OutputSchema(
        name="nested_v1",
        json_schema={
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
                "detail": {
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                },
            },
        },
    )
    payload = build_payload(_request(schema=nested))
    wire = payload["response_format"]["json_schema"]["schema"]  # type: ignore[index]

    assert wire["additionalProperties"] is False
    assert wire["properties"]["detail"]["additionalProperties"] is False


def test_an_explicitly_open_schema_is_not_silently_closed() -> None:
    permissive = OutputSchema(
        name="open_v1",
        json_schema={
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "additionalProperties": True,
        },
    )
    payload = build_payload(_request(schema=permissive))
    wire = payload["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    assert wire["additionalProperties"] is True


def test_a_property_named_type_does_not_confuse_the_injection() -> None:
    tricky = OutputSchema(
        name="tricky_v1",
        json_schema={
            "type": "object",
            "properties": {"type": {"type": "string"}},
        },
    )
    payload = build_payload(_request(schema=tricky))
    wire = payload["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    # The properties container maps names to schemas; it is not itself an
    # object schema and must not grow an additionalProperties key.
    assert "additionalProperties" not in wire["properties"]
    assert wire["additionalProperties"] is False


# -- response translation -----------------------------------------------------


def test_the_observed_success_envelope_translates_completely() -> None:
    request = _request(metadata={"purpose": "test"})

    response = parse_response(
        _SUCCESS_BODY, _SUCCESS_HEADERS, request=request, latency=10.256
    )

    assert response.provider == "muse"
    assert response.model == "muse-spark-1.2"
    assert response.text == _CONTENT
    assert response.structured is not None
    assert response.structured["statement"] == (
        "The stream is biased toward heads."
    )
    assert response.structured["confidence"] == 0.82
    # Provider-reported counts, preserved exactly.
    assert response.usage.calls == 1
    assert response.usage.input_tokens == 52
    assert response.usage.output_tokens == 1537
    assert response.usage.model == "muse-spark-1.2"
    assert response.latency_seconds == 10.256
    assert response.nominal_cost is None  # no trustworthy per-account rate
    assert response.request_id == "00000000-0000-4000-8000-000000000000"
    assert response.finish_reason == "stop"
    assert response.schema_name == "hypothesis_v1"
    assert response.request_fingerprint == request.fingerprint
    assert response.metadata["purpose"] == "test"
    # Detail counts with no seat in ProviderUsage are preserved verbatim
    # as namespaced metadata, not dropped.
    assert response.metadata["muse:reasoning_tokens"] == "1400"
    assert response.metadata["muse:cached_tokens"] == "49"


def test_unreported_usage_details_add_no_metadata() -> None:
    body = json.loads(json.dumps(_SUCCESS_BODY))
    del body["usage"]["completion_tokens_details"]
    del body["usage"]["prompt_tokens_details"]

    response = parse_response(
        body, _SUCCESS_HEADERS, request=_request(metadata={"k": "v"}), latency=1.0
    )

    assert "muse:reasoning_tokens" not in response.metadata
    assert "muse:cached_tokens" not in response.metadata
    assert response.metadata["k"] == "v"  # caller provenance untouched


def test_the_completion_id_backs_up_a_missing_request_id_header() -> None:
    response = parse_response(
        _SUCCESS_BODY, {"content-type": "application/json"},
        request=_request(), latency=1.0,
    )
    assert response.request_id == "chatcmpl-01a011cb-0000-0000-0000-000000000000"


def test_a_schemaless_request_returns_plain_text() -> None:
    body = dict(_SUCCESS_BODY)
    response = parse_response(
        body, _SUCCESS_HEADERS, request=_request(schema=None), latency=1.0
    )
    assert response.structured is None
    assert response.schema_name == ""
    assert response.text == _CONTENT


def test_the_observed_truncation_reply_is_an_invalid_response() -> None:
    """finish_reason "length" with content null — observed when reasoning
    exhausted max_tokens — is an unusable reply, not a result."""
    with pytest.raises(InvalidModelResponseError, match="length"):
        parse_response(
            _TRUNCATED_BODY, _SUCCESS_HEADERS, request=_request(), latency=1.0
        )


def test_a_reply_without_choices_is_an_invalid_response() -> None:
    bodies: tuple[dict[str, Any], ...] = ({}, {"choices": []}, {"choices": [{}]})
    for body in bodies:
        with pytest.raises(InvalidModelResponseError):
            parse_response(
                body, _SUCCESS_HEADERS, request=_request(), latency=1.0
            )


def test_undeclared_fields_from_the_provider_fail_local_validation() -> None:
    """Observed live: the provider can return extra fields. Local
    validation, not provider-side enforcement, is the contract."""
    body = json.loads(json.dumps(_SUCCESS_BODY))
    body["choices"][0]["message"]["content"] = _EXTRA_FIELDS_CONTENT

    with pytest.raises(StructuredOutputError) as caught:
        parse_response(body, _SUCCESS_HEADERS, request=_request(), latency=1.0)

    assert "null_hypothesis" in caught.value.detail


def test_missing_usage_reads_as_zero_not_invented() -> None:
    body = json.loads(json.dumps(_SUCCESS_BODY))
    del body["usage"]
    response = parse_response(
        body, _SUCCESS_HEADERS, request=_request(), latency=1.0
    )
    assert response.usage.calls == 1
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0


# -- error translation --------------------------------------------------------


def test_the_observed_auth_envelope_maps_to_a_transport_error() -> None:
    error = _error_for_status(401, {}, _AUTH_ERROR)

    assert isinstance(error, ProviderTransportError)
    assert error.status_code == 401
    assert error.provider_error == "invalid_api_key"
    assert "Unauthorized" in str(error)


def test_a_rate_limit_carries_the_servers_retry_after() -> None:
    # DOCUMENTATION-DERIVED, not captured live: status, error code and
    # Retry-After from the official error table, envelope shape as the
    # observed 401 confirmed it.
    envelope = (
        b'{"error":{"code":"rate_limit_exceeded","message":"Your team has '
        b'exceeded the rate limit","param":null,"type":"rate_limit_error"}}'
    )
    error = _error_for_status(429, {"retry-after": "12"}, envelope)

    assert isinstance(error, ProviderRateLimitError)
    assert error.retry_after_seconds == 12.0

    headerless = _error_for_status(429, {}, envelope)
    assert isinstance(headerless, ProviderRateLimitError)
    assert headerless.retry_after_seconds is None

    garbled = _error_for_status(429, {"retry-after": "soon"}, envelope)
    assert isinstance(garbled, ProviderRateLimitError)
    assert garbled.retry_after_seconds is None


def test_server_errors_keep_the_documented_error_code() -> None:
    # DOCUMENTATION-DERIVED, not captured live (same provenance note as
    # the rate-limit envelope above).
    envelope = (
        b'{"error":{"code":"service_overloaded","message":"Backend at '
        b'capacity","param":null,"type":"server_error"}}'
    )
    error = _error_for_status(503, {"retry-after": "5"}, envelope)
    assert isinstance(error, ProviderTransportError)
    assert error.status_code == 503
    assert error.provider_error == "service_overloaded"


def test_an_undocumented_error_body_still_maps_cleanly() -> None:
    error = _error_for_status(500, {}, b"<html>Internal Server Error</html>")
    assert isinstance(error, ProviderTransportError)
    assert error.status_code == 500
    assert error.provider_error is None
    assert "HTTP 500" in str(error)


# -- the credential -----------------------------------------------------------


def test_the_key_is_read_at_invocation_time_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ProviderTransportError, match="MUSE_API_KEY"):
        _api_key()

    monkeypatch.setenv("MODEL_API_KEY", "dummy-official-name")
    assert _api_key() == "dummy-official-name"

    monkeypatch.setenv("MUSE_API_KEY", "dummy-preferred-name")
    assert _api_key() == "dummy-preferred-name"  # MUSE_API_KEY wins


def test_a_whitespace_key_counts_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MUSE_API_KEY", "   ")
    with pytest.raises(ProviderTransportError, match="no Muse API key"):
        _api_key()


def test_the_credential_is_passed_verbatim_never_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only surrounding whitespace is trimmed. A malformed key — pasted
    angle brackets included — goes through untouched, so the failure it
    causes points at the value to fix rather than being half-hidden by a
    silent rewrite."""
    for name in KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MUSE_API_KEY", " <LLM|0000000000000000|dummy-value> \n")
    assert _api_key() == "<LLM|0000000000000000|dummy-value>"


# -- the HTTP glue, network stubbed -------------------------------------------


class _FakeReply:
    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        self._body = body
        self._offset = 0
        self.headers = headers

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> _FakeReply:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _DrippingReply:
    """A body that never ends: each read yields another chunk."""

    headers: ClassVar[dict[str, str]] = {}

    def read(self, amount: int = -1) -> bytes:
        return b"drip"

    def __enter__(self) -> _DrippingReply:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _stub_urlopen(
    monkeypatch: pytest.MonkeyPatch, outcome: _FakeReply | Exception
) -> list[urllib.request.Request]:
    seen: list[urllib.request.Request] = []

    def fake_urlopen(
        request: urllib.request.Request, *, timeout: float
    ) -> _FakeReply:
        seen.append(request)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("MUSE_API_KEY", "dummy-not-a-real-key")
    return seen


def test_invoke_sends_the_documented_request_and_translates_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = _FakeReply(
        json.dumps(_SUCCESS_BODY).encode(), dict(_SUCCESS_HEADERS)
    )
    seen = _stub_urlopen(monkeypatch, reply)

    response = MuseSparkProvider().invoke(_request())

    (sent,) = seen
    assert sent.full_url == "https://api.meta.ai/v1/chat/completions"
    assert sent.get_header("Authorization") == "Bearer dummy-not-a-real-key"
    assert sent.get_header("Content-type") == "application/json"
    assert isinstance(sent.data, bytes)
    wire = json.loads(sent.data)
    assert wire["model"] == "muse-spark-1.2"
    assert response.structured is not None
    assert response.usage.output_tokens == 1537
    assert response.latency_seconds >= 0.0


def test_a_socket_timeout_maps_to_the_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, urllib.error.URLError(TimeoutError()))

    with pytest.raises(ProviderTimeoutError) as caught:
        MuseSparkProvider().invoke(_request(timeout_seconds=45.0))
    assert caught.value.timeout_seconds == 45.0


def test_an_unreachable_endpoint_maps_to_a_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(
        monkeypatch,
        urllib.error.URLError(ConnectionRefusedError("connection refused")),
    )
    with pytest.raises(ProviderTransportError, match="could not reach"):
        MuseSparkProvider().invoke(_request())


def test_an_http_error_maps_through_the_observed_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http_error = urllib.error.HTTPError(
        "https://api.meta.ai/v1/chat/completions",
        401,
        "Unauthorized",
        EmailMessage(),
        io.BytesIO(_AUTH_ERROR),
    )
    _stub_urlopen(monkeypatch, http_error)

    with pytest.raises(ProviderTransportError) as caught:
        MuseSparkProvider().invoke(_request())
    assert caught.value.status_code == 401
    assert caught.value.provider_error == "invalid_api_key"


def test_a_non_json_success_body_is_an_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, _FakeReply(b"not json at all", {}))

    with pytest.raises(InvalidModelResponseError, match="not valid JSON"):
        MuseSparkProvider().invoke(_request())


# -- audit regressions: adapter-local hardening -------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        http.client.BadStatusLine("garbage status line"),
        http.client.IncompleteRead(b"partial body"),
        http.client.LineTooLong("header line"),
    ],
)
def test_http_client_exceptions_map_to_transport_errors(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    """http.client's exception family is not OSError, so without an explicit
    clause a malformed status line or a connection dropped mid-body escapes
    the seam as a raw stdlib exception."""
    _stub_urlopen(monkeypatch, exc)

    with pytest.raises(ProviderTransportError, match="malformed HTTP"):
        MuseSparkProvider().invoke(_request())


def test_truncated_reply_with_partial_content_is_invalid() -> None:
    """CONSTRUCTED variant of the OBSERVED truncation reply: same envelope,
    the null content replaced by partial text. Truncation must be decided by
    finish_reason — and with a schema it must surface as an invalid
    response, not as a schema violation by the model."""
    body = json.loads(json.dumps(_TRUNCATED_BODY))
    body["choices"][0]["message"]["content"] = '{"statement": "The stream is bia'

    with pytest.raises(InvalidModelResponseError, match="length"):
        parse_response(
            body, _SUCCESS_HEADERS, request=_request(schema=None), latency=1.0
        )
    with pytest.raises(InvalidModelResponseError, match="length"):
        parse_response(body, _SUCCESS_HEADERS, request=_request(), latency=1.0)


def test_a_non_utf8_success_body_is_an_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, _FakeReply(b"\xff\xfe\x01 not utf-8", {}))

    with pytest.raises(InvalidModelResponseError, match="not valid JSON"):
        MuseSparkProvider().invoke(_request())


def test_a_dripping_body_hits_the_overall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """urllib's timeout bounds each socket operation, not the exchange: a
    server dripping chunks forever must still hit ProviderTimeoutError once
    the whole-call deadline passes."""
    from types import SimpleNamespace

    tick = iter(range(100, 100_000, 100))
    monkeypatch.setattr(
        muse, "time", SimpleNamespace(monotonic=lambda: float(next(tick)))
    )
    _stub_urlopen(monkeypatch, _DrippingReply())  # type: ignore[arg-type]

    with pytest.raises(ProviderTimeoutError, match="did not complete"):
        MuseSparkProvider().invoke(_request(timeout_seconds=120.0))


def test_an_oversized_body_is_rejected_not_buffered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(muse, "_MAX_BODY_BYTES", 64)
    _stub_urlopen(monkeypatch, _FakeReply(b"x" * 200, {}))

    with pytest.raises(InvalidModelResponseError, match="exceeds"):
        MuseSparkProvider().invoke(_request())


def test_retry_after_rejects_non_finite_and_negative_values() -> None:
    for bad in ("nan", "inf", "-inf", "-5"):
        assert muse._retry_after({"retry-after": bad}) is None
    assert muse._retry_after({"retry-after": "12.5"}) == 12.5
    assert muse._retry_after({"retry-after": "0"}) == 0.0


def test_a_numeric_error_code_falls_back_to_the_string_type() -> None:
    envelope = (
        b'{"error":{"code":42901,"message":"Too many requests",'
        b'"param":null,"type":"rate_limit_error"}}'
    )
    error = _error_for_status(500, {}, envelope)
    assert isinstance(error, ProviderTransportError)
    assert error.provider_error == "rate_limit_error"


def test_wire_injection_never_touches_enum_literals() -> None:
    """Enum entries are data, not schemas: a mapping-valued literal must
    cross the wire byte-identical, or the wire constraint and the local
    validator stop agreeing on what is permitted."""
    schema = OutputSchema(
        name="enum_v1",
        json_schema={
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"kind": {"type": "string"}},
                    "enum": [{"kind": "a"}, {"type": "object"}],
                }
            },
        },
    )
    payload = build_payload(_request(schema=schema))
    wire = payload["response_format"]["json_schema"]["schema"]  # type: ignore[index]

    enum = wire["properties"]["payload"]["enum"]
    assert {"kind": "a"} in enum
    assert {"type": "object"} in enum  # the literal survives untouched
    assert all(
        "additionalProperties" not in entry
        for entry in enum
        if isinstance(entry, dict)
    )
    # ...while the schema nodes themselves are still closed on the wire.
    assert wire["additionalProperties"] is False
    assert wire["properties"]["payload"]["additionalProperties"] is False


def test_the_adapter_satisfies_the_provider_seam() -> None:
    from autonomous_research_lab.runtime.providers import ModelProvider

    provider = MuseSparkProvider()
    assert isinstance(provider, ModelProvider)
    assert provider.name == "muse"


def test_response_values_stay_provider_neutral() -> None:
    """Nothing urllib-shaped crosses the boundary."""
    response = parse_response(
        _SUCCESS_BODY, _SUCCESS_HEADERS, request=_request(), latency=1.0
    )
    assert isinstance(response.text, str)
    assert response.structured is None or isinstance(response.structured, Mapping)
    assert response.request_id is None or isinstance(response.request_id, str)
