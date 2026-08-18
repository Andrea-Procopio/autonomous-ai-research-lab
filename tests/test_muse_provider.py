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
import json
import socket
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import pytest

from autonomous_research_lab.runtime import muse
from autonomous_research_lab.runtime.metrics import ProviderUsage
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
    ModelProviderError,
    ModelRequest,
    OutputSchema,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransportError,
    StructuredOutputError,
    UsageLedger,
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


def test_a_property_named_type_does_not_confuse_the_normalization() -> None:
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
    with pytest.raises(ProviderConfigurationError, match="MUSE_API_KEY"):
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
    with pytest.raises(ProviderConfigurationError, match="no Muse API key"):
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
    def __init__(
        self, body: bytes, headers: dict[str, str], *, status: int = 200
    ) -> None:
        self._body = body
        self._offset = 0
        self.status = status
        self.headers = headers

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk


class _DrippingReply:
    """A body that never ends: each read yields another chunk."""

    status: ClassVar[int] = 200
    headers: ClassVar[dict[str, str]] = {}

    def read(self, amount: int = -1) -> bytes:
        return b"drip"


class _FakeConnection:
    """The ``_connect`` seam, captured: records the one request and serves
    one scripted reply or raises one scripted exception."""

    def __init__(
        self, base_url: str, outcome: _FakeReply | _DrippingReply | Exception
    ) -> None:
        self.base_url = base_url
        self.requests: list[tuple[str, str, bytes, dict[str, str]]] = []
        self.closed = False
        self._outcome = outcome

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.requests.append((method, url, body or b"", dict(headers or {})))

    def getresponse(self) -> _FakeReply | _DrippingReply:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    def close(self) -> None:
        self.closed = True


def _stub_connection(
    monkeypatch: pytest.MonkeyPatch,
    outcome: _FakeReply | _DrippingReply | Exception,
) -> list[_FakeConnection]:
    seen: list[_FakeConnection] = []

    def fake_connect(base_url: str, timeout: float) -> _FakeConnection:
        connection = _FakeConnection(base_url, outcome)
        seen.append(connection)
        return connection

    monkeypatch.setattr(muse, "_connect", fake_connect)
    monkeypatch.setenv("MUSE_API_KEY", "dummy-not-a-real-key")
    return seen


def test_the_connection_is_scheme_dispatched() -> None:
    """Constructing an http.client connection opens no socket, so the real
    _connect is testable offline."""
    plain = muse._connect("http://127.0.0.1:1", 1.0)
    assert isinstance(plain, http.client.HTTPConnection)
    assert not isinstance(plain, http.client.HTTPSConnection)
    secure = muse._connect("https://api.meta.ai/v1", 1.0)
    assert isinstance(secure, http.client.HTTPSConnection)
    with pytest.raises(ProviderConfigurationError, match="scheme"):
        muse._connect("ftp://api.meta.ai/v1", 1.0)
    assert muse._endpoint_path("https://api.meta.ai/v1") == (
        "/v1/chat/completions"
    )
    assert muse._endpoint_path("http://127.0.0.1:8080") == "/chat/completions"


def test_invoke_sends_the_documented_request_and_translates_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = _FakeReply(
        json.dumps(_SUCCESS_BODY).encode(), dict(_SUCCESS_HEADERS)
    )
    seen = _stub_connection(monkeypatch, reply)

    response = MuseSparkProvider().invoke(_request())

    (connection,) = seen
    assert connection.base_url == "https://api.meta.ai/v1"
    ((method, path, body, headers),) = connection.requests
    assert method == "POST"
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer dummy-not-a-real-key"
    assert headers["Content-Type"] == "application/json"
    wire = json.loads(body)
    assert wire["model"] == "muse-spark-1.2"
    assert response.structured is not None
    assert response.usage.output_tokens == 1537
    assert response.latency_seconds >= 0.0
    assert connection.closed  # the connection never outlives the call


def test_a_socket_timeout_maps_to_the_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_connection(monkeypatch, TimeoutError())

    with pytest.raises(ProviderTimeoutError) as caught:
        MuseSparkProvider().invoke(_request(timeout_seconds=45.0))
    assert caught.value.timeout_seconds == 45.0


def test_an_unreachable_endpoint_maps_to_a_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_connection(monkeypatch, ConnectionRefusedError("connection refused"))
    with pytest.raises(ProviderTransportError, match="could not reach"):
        MuseSparkProvider().invoke(_request())


def test_an_http_error_maps_through_the_observed_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = _FakeReply(
        _AUTH_ERROR, {"content-type": "application/json"}, status=401
    )
    _stub_connection(monkeypatch, reply)

    with pytest.raises(ProviderTransportError) as caught:
        MuseSparkProvider().invoke(_request())
    assert caught.value.status_code == 401
    assert caught.value.provider_error == "invalid_api_key"


def test_a_non_json_success_body_is_an_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_connection(monkeypatch, _FakeReply(b"not json at all", {}))

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
    _stub_connection(monkeypatch, exc)

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
    _stub_connection(monkeypatch, _FakeReply(b"\xff\xfe\x01 not utf-8", {}))

    with pytest.raises(InvalidModelResponseError, match="not valid JSON"):
        MuseSparkProvider().invoke(_request())


def test_a_dripping_body_hits_the_overall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The socket timeout bounds each socket operation, not the exchange: a
    server dripping chunks forever must still hit ProviderTimeoutError once
    the whole-call deadline passes."""
    from types import SimpleNamespace

    tick = iter(range(100, 100_000, 100))
    monkeypatch.setattr(
        muse, "time", SimpleNamespace(monotonic=lambda: float(next(tick)))
    )
    _stub_connection(monkeypatch, _DrippingReply())

    with pytest.raises(ProviderTimeoutError, match="did not complete"):
        MuseSparkProvider().invoke(_request(timeout_seconds=120.0))


def test_an_oversized_body_is_rejected_not_buffered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(muse, "_MAX_BODY_BYTES", 64)
    _stub_connection(monkeypatch, _FakeReply(b"x" * 200, {}))

    with pytest.raises(InvalidModelResponseError, match="exceeds"):
        MuseSparkProvider().invoke(_request())


def _serve_once(handler: Callable[[socket.socket], None]) -> int:
    """A one-connection local HTTP server; returns its port. The handler
    receives the accepted connection after the request has been read."""
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(10.0)
    port: int = server.getsockname()[1]

    def run() -> None:
        try:
            conn, _ = server.accept()
            with conn:
                conn.recv(65536)
                handler(conn)
        except OSError:
            pass
        finally:
            server.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def _http_headers(content_length: int) -> bytes:
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(content_length).encode() + b"\r\n\r\n"
    )


def test_a_genuinely_stalled_read_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real socket, not a fake clock: the server sends headers and then
    nothing. The blocking read must surface as ProviderTimeoutError within
    the deadline's order of magnitude, not hang."""

    def stall(conn: socket.socket) -> None:
        conn.sendall(_http_headers(1_000_000))
        time.sleep(3.0)

    port = _serve_once(stall)
    monkeypatch.setenv("MUSE_API_KEY", "dummy-not-a-real-key")
    started = time.monotonic()

    with pytest.raises(ProviderTimeoutError):
        MuseSparkProvider(base_url=f"http://127.0.0.1:{port}").invoke(
            _request(timeout_seconds=0.5)
        )
    assert time.monotonic() - started < 3.0


def test_a_dripping_status_line_hits_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact class of the >570-second live stall of 2026-08-18,
    reproduced pre-body: every byte of the reply arrives within the
    per-socket-operation window, so no single socket operation ever times
    out, while the wall clock runs far past the requested deadline. Before
    the watchdog, this call ran ~15x its configured timeout and then
    SUCCEEDED; the deadline must end it instead."""

    reply = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2\r\n\r\n{}"
    )

    def drip_status(conn: socket.socket) -> None:
        try:
            for index in range(len(reply)):
                conn.sendall(reply[index : index + 1])
                time.sleep(0.15)
        except OSError:
            pass  # the client gave up, which is the point

    port = _serve_once(drip_status)
    monkeypatch.setenv("MUSE_API_KEY", "dummy-not-a-real-key")
    started = time.monotonic()

    with pytest.raises(ProviderTimeoutError, match="within") as caught:
        MuseSparkProvider(base_url=f"http://127.0.0.1:{port}").invoke(
            _request(timeout_seconds=0.5)
        )

    assert time.monotonic() - started < 2.0  # the deadline is wall-clock real
    assert caught.value.timeout_seconds == 0.5
    # A call that never produced a body carries no accounting: unknown
    # spend stays unknown, and the ledger declines to invent a zero.
    assert caught.value.accounting is None
    ledger = UsageLedger()
    assert ledger.record_failure(caught.value) is False
    assert ledger.drain() == ProviderUsage()


def test_a_server_that_never_sends_a_status_line_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Total pre-body silence: the server accepts, reads the request, and
    sends nothing. Bounded by the socket timeout and the watchdog alike."""

    def silent(conn: socket.socket) -> None:
        time.sleep(3.0)

    port = _serve_once(silent)
    monkeypatch.setenv("MUSE_API_KEY", "dummy-not-a-real-key")
    started = time.monotonic()

    with pytest.raises(ProviderTimeoutError, match="within"):
        MuseSparkProvider(base_url=f"http://127.0.0.1:{port}").invoke(
            _request(timeout_seconds=0.4)
        )
    assert time.monotonic() - started < 2.0


def test_a_genuinely_dripping_body_hits_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sharp case a between-chunks clock check cannot catch on its own:
    each byte arrives fast enough that no single socket operation times
    out, while a buffered read(n) would keep recv-ing forever. The
    whole-call deadline must still end the call."""

    def drip(conn: socket.socket) -> None:
        conn.sendall(_http_headers(1_000_000))
        try:
            for _ in range(30):
                conn.sendall(b"x")
                time.sleep(0.15)
        except OSError:
            pass  # the client gave up, which is the point

    port = _serve_once(drip)
    monkeypatch.setenv("MUSE_API_KEY", "dummy-not-a-real-key")
    started = time.monotonic()

    # Either enforcement path is the deadline working: the between-reads
    # check ("did not complete") or the tightened socket timeout firing
    # once the remaining deadline drops below the drip interval.
    with pytest.raises(ProviderTimeoutError, match="within"):
        MuseSparkProvider(base_url=f"http://127.0.0.1:{port}").invoke(
            _request(timeout_seconds=0.6)
        )
    assert time.monotonic() - started < 3.0


class _ExplodingReply:
    """An error reply whose body read raises — a stalled or dropped
    connection while draining the envelope."""

    def __init__(
        self, error: Exception, *, status: int, headers: dict[str, str]
    ) -> None:
        self._error = error
        self.status = status
        self.headers = headers

    def read(self, amount: int = -1) -> bytes:
        raise self._error


@pytest.mark.parametrize(
    "body_error",
    [
        TimeoutError("stalled while draining the error body"),
        http.client.IncompleteRead(b""),
        ValueError("I/O operation on closed file"),
    ],
)
def test_an_unreadable_error_body_still_yields_a_typed_error(
    monkeypatch: pytest.MonkeyPatch, body_error: Exception
) -> None:
    """The status and headers already classify the failure; a body that
    cannot be read must degrade to that classification, never escape the
    handler as an untyped exception."""
    reply = _ExplodingReply(
        body_error, status=429, headers={"retry-after": "7"}
    )
    _stub_connection(monkeypatch, reply)  # type: ignore[arg-type]

    with pytest.raises(ProviderRateLimitError) as caught:
        MuseSparkProvider().invoke(_request())
    # The already-known headers still carry the server's instruction.
    assert caught.value.retry_after_seconds == 7.0


def test_an_http_error_without_headers_or_body_is_still_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_connection(monkeypatch, _FakeReply(b"", {}, status=500))

    with pytest.raises(ProviderTransportError) as caught:
        MuseSparkProvider().invoke(_request())
    assert caught.value.status_code == 500


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


# -- audit regressions: accounting survives failures --------------------------


def test_a_truncated_billed_reply_raises_and_keeps_its_accounting() -> None:
    """OBSERVED live: the truncation reply billed 2048 completion tokens.
    The raised error must carry that spend, exactly as reported."""
    with pytest.raises(InvalidModelResponseError) as caught:
        parse_response(
            _TRUNCATED_BODY, _SUCCESS_HEADERS, request=_request(), latency=9.5
        )

    accounting = caught.value.accounting
    assert accounting is not None
    assert accounting.usage.calls == 1
    assert accounting.usage.input_tokens == 52
    assert accounting.usage.output_tokens == 2048
    assert accounting.usage.model == "muse-spark-1.2"
    assert accounting.model == "muse-spark-1.2"
    assert accounting.latency_seconds == 9.5
    assert accounting.request_id == "00000000-0000-4000-8000-000000000000"
    assert accounting.nominal_cost is None  # unknown, never invented


def test_a_schema_invalid_billed_reply_keeps_its_accounting() -> None:
    body = json.loads(json.dumps(_SUCCESS_BODY))
    body["choices"][0]["message"]["content"] = _EXTRA_FIELDS_CONTENT

    with pytest.raises(StructuredOutputError) as caught:
        parse_response(body, _SUCCESS_HEADERS, request=_request(), latency=2.0)

    accounting = caught.value.accounting
    assert accounting is not None
    assert accounting.usage.input_tokens == 52
    assert accounting.usage.output_tokens == 1537


def test_failed_call_accounting_reaches_the_ledger_exactly_once() -> None:
    ledger = UsageLedger()
    failures: list[ModelProviderError] = []
    for body, expected in (
        (_TRUNCATED_BODY, InvalidModelResponseError),
        (json.loads(json.dumps(_SUCCESS_BODY)), StructuredOutputError),
    ):
        if expected is StructuredOutputError:
            body["choices"][0]["message"]["content"] = _EXTRA_FIELDS_CONTENT
        with pytest.raises(expected) as caught:
            parse_response(
                body, _SUCCESS_HEADERS, request=_request(), latency=1.0
            )
        failures.append(caught.value)

    for failure in failures:
        assert ledger.record_failure(failure) is True

    drained = ledger.drain()
    assert drained.calls == 2
    assert drained.input_tokens == 52 + 52
    assert drained.output_tokens == 2048 + 1537
    assert ledger.drain() == ProviderUsage()  # nothing left to double-count


def test_a_failure_without_reported_usage_stays_unknown() -> None:
    """A body with no usage block yields an error with no accounting —
    unknown spend is absent, never a reported zero with calls=1."""
    body = json.loads(json.dumps(_TRUNCATED_BODY))
    del body["usage"]

    with pytest.raises(InvalidModelResponseError) as caught:
        parse_response(body, _SUCCESS_HEADERS, request=_request(), latency=1.0)

    assert caught.value.accounting is None
    ledger = UsageLedger()
    assert ledger.record_failure(caught.value) is False
    assert ledger.drain() == ProviderUsage()


def test_missing_configuration_raises_before_any_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _stub_connection(monkeypatch, _FakeReply(b"{}", {}))
    for name in KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ProviderConfigurationError) as caught:
        MuseSparkProvider().invoke(_request())

    assert seen == []  # the wire was never touched
    assert isinstance(caught.value, ModelProviderError)  # still runtime-typed


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
