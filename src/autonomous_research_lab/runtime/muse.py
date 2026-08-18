"""Muse Spark adapter: the Meta Model API behind the ModelProvider seam.

The one concrete :class:`~autonomous_research_lab.runtime.providers.
ModelProvider`. Standard library HTTP only — the package keeps its
zero-dependency promise, and every vendor-shaped object (URLs, wire
payloads, ``http.client`` types) stays inside this module.

The whole exchange runs under one wall-clock deadline. A per-socket-
operation timeout restarts at every blocking primitive — TCP connect, each
TLS-handshake read, each header read, each body read — so it bounds no sum:
a server that dribbles bytes can hold a call open for an unbounded multiple
of the configured timeout, which is exactly the >570-second stall observed
live on 2026-08-18. The adapter therefore owns its connection and arms a
watchdog that closes it at the absolute deadline; the per-operation socket
timeout remains as a first fence. One residual is named rather than hidden:
DNS resolution (``getaddrinfo``) runs before any socket exists, so no
stdlib mechanism can interrupt it; everything after name resolution is
under the deadline.

Wire contract
-------------

Established from the official documentation (dev.meta.ai) and from live
responses captured on 2026-08-17. Observed directly: the success envelope,
the truncation reply, the 401 error envelope, and the
``additionalProperties`` behavior. Taken from the official error-handling
documentation without live capture: the wider error table (429/503/504,
their error codes, and the ``Retry-After`` header).

* ``POST {base_url}/chat/completions`` with ``Authorization: Bearer <key>``
  and ``Content-Type: application/json``.
* Request: ``model``, ``messages`` (``system`` / ``user`` / ``assistant``
  roles), optional ``max_tokens`` and ``temperature``, and for structured
  output ``response_format = {"type": "json_schema", "json_schema":
  {"name": ..., "schema": ...}}``.
* Response: ``choices[0].message.content`` (the structured payload arrives
  as a JSON *string*), ``choices[0].finish_reason``, ``model``, ``usage``
  with ``prompt_tokens`` / ``completion_tokens``, and an ``x-request-id``
  header. A generation that exhausts ``max_tokens`` was observed to return
  ``finish_reason: "length"`` with ``content: null`` — that is an unusable
  reply, not a result.
* Errors: an ``{"error": {"message", "type", "param", "code"}}`` envelope
  with the documented status codes; ``Retry-After`` accompanies 429/503.

Two behaviors worth naming:

* **Closed schemas arrive already explicit.** The provider treats an
  *absent* ``additionalProperties`` as open — observed directly: the model
  added six undeclared fields until ``additionalProperties: false`` was
  sent, and none after. The local validator is closed by default, and
  ``OutputSchema`` makes that explicit at construction, so no schema with
  the keyword absent can reach this adapter. The wire schema is therefore
  the request's schema verbatim (thawed for serialization, nothing
  injected), and the request fingerprint covers exactly what is sent.
* **The provider is never trusted with validation.** Whatever
  ``response_format`` promises, the returned text goes through
  ``OutputSchema.parse`` locally, and fails closed.

The API key is read from the environment at invocation time —
``MUSE_API_KEY`` first, then ``MODEL_API_KEY`` (the name the official
documentation uses) — and appears in nothing but the request header.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import math
import os
import socket
import ssl
import threading
import time
import urllib.parse
from collections.abc import Iterable, Mapping
from typing import Final, Protocol

from .metrics import ProviderUsage
from .providers import (
    CallAccounting,
    InvalidModelResponseError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransportError,
)

DEFAULT_BASE_URL: Final = "https://api.meta.ai/v1"
MUSE_SPARK_1_2: Final = "muse-spark-1.2"
KEY_ENV_VARS: Final = ("MUSE_API_KEY", "MODEL_API_KEY")

_CHUNK_BYTES: Final = 64 * 1024
_MAX_BODY_BYTES: Final = 64 * 1024 * 1024
"""Upper bound on a success body. A reply larger than this is a fault to
classify, not a result to buffer until the process dies."""

_MAX_ERROR_BODY_BYTES: Final = 1024 * 1024
"""Error envelopes are a few hundred bytes; anything past this cap is
truncated, and a truncated envelope degrades to the status-code fallback."""

_TRUNCATING_FINISH_REASONS: Final = frozenset({"length"})
"""finish_reason values that mean the generation did not run to completion.
Only the observed value; a future adapter datum can extend it."""


class MuseSparkProvider(ModelProvider):
    """Meta Model API adapter, synchronous, one request per invoke."""

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "muse"

    def invoke(self, request: ModelRequest) -> ModelResponse:
        api_key = _api_key()  # configuration first: no key, no work
        payload = json.dumps(build_payload(request)).encode("utf-8")
        started = time.monotonic()
        raw, headers = self._post(payload, api_key, request.timeout_seconds)
        latency = time.monotonic() - started
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InvalidModelResponseError(
                f"the reply is not valid JSON: {exc}"
            ) from exc
        if not isinstance(body, Mapping):
            raise InvalidModelResponseError(
                f"the reply is not a JSON object: {type(body).__name__}"
            )
        return parse_response(body, headers, request=request, latency=latency)

    def _post(
        self, payload: bytes, api_key: str, timeout: float
    ) -> tuple[bytes, Mapping[str, str]]:
        """One HTTP exchange under one wall-clock deadline.

        ``http.client`` types enter and never leave. The deadline is
        computed before the connection is opened, so connect, TLS
        handshake, headers, body and any error-body drain all spend from
        the same budget; the :class:`_DeadlineWatchdog` closes the
        connection at the deadline, ending whichever blocking primitive is
        in flight. A complete reply already in hand is returned even if
        the timer fires during teardown — ``fired`` only reclassifies
        exceptions, never a success.
        """
        deadline = time.monotonic() + timeout
        connection = _connect(self._base_url, timeout)
        watchdog = _DeadlineWatchdog(connection, deadline)
        try:
            with watchdog:
                connection.request(
                    "POST",
                    _endpoint_path(self._base_url),
                    body=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                reply = connection.getresponse()
                headers = _lowered(reply.headers.items())
                if not 200 <= reply.status < 300:
                    # Redirects are not followed: the adapter posts to one
                    # fixed endpoint, so a 3xx is as much a fault as a 5xx.
                    raise _error_for_status(
                        reply.status,
                        headers,
                        _drain_error_body(reply, deadline=deadline),
                    )
                body = _read_bounded(reply, deadline=deadline, timeout=timeout)
                return body, headers
        except ModelProviderError:
            raise  # already typed by this seam; never reclassified
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"no reply from the Muse endpoint within {timeout}s",
                timeout_seconds=timeout,
            ) from exc
        except (
            http.client.HTTPException,
            OSError,
            ValueError,
            AttributeError,
        ) as exc:
            # A watchdog close surfaces as whatever the interrupted
            # primitive raises — EBADF, ECONNRESET, BadStatusLine, a
            # "closed file" ValueError, or an AttributeError from a
            # response whose file object was torn down mid-read. When the
            # deadline caused it, the deadline is the diagnosis.
            if watchdog.fired:
                raise ProviderTimeoutError(
                    f"no reply from the Muse endpoint within {timeout}s",
                    timeout_seconds=timeout,
                ) from exc
            if isinstance(exc, http.client.HTTPException):
                # BadStatusLine, IncompleteRead, LineTooLong and kin are
                # not OSError, so without this clause they would escape
                # the seam untyped: a reply that cannot even be framed is
                # a transport failure like any other.
                raise ProviderTransportError(
                    f"malformed HTTP exchange with the Muse endpoint: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if isinstance(exc, OSError):
                raise ProviderTransportError(
                    f"could not reach the Muse endpoint: {exc}"
                ) from exc
            raise
        finally:
            connection.close()


# -- the connection and its deadline ------------------------------------------


def _connect(base_url: str, timeout: float) -> http.client.HTTPConnection:
    """One connection for the adapter's single POST, scheme-dispatched.

    ``https`` gets certificate verification via
    :func:`ssl.create_default_context`; plain ``http`` is kept for loopback
    test servers. Any other scheme is local misconfiguration. The
    ``timeout`` here is the per-socket-operation fence; the absolute
    deadline is the :class:`_DeadlineWatchdog`'s job. Module-level on
    purpose: this is the deterministic test seam.
    """
    split = urllib.parse.urlsplit(base_url)
    if split.scheme == "https":
        return http.client.HTTPSConnection(
            split.netloc, timeout=timeout, context=ssl.create_default_context()
        )
    if split.scheme == "http":
        return http.client.HTTPConnection(split.netloc, timeout=timeout)
    raise ProviderConfigurationError(
        f"unsupported Muse base URL scheme {split.scheme!r}: "
        f"expected http or https"
    )


def _endpoint_path(base_url: str) -> str:
    """The request target: the base URL's path plus the documented route."""
    return f"{urllib.parse.urlsplit(base_url).path}/chat/completions"


class _DeadlineWatchdog:
    """Makes the wall-clock deadline real for the phases the socket timeout
    cannot bound in sum: at the absolute deadline it closes the connection,
    so a blocked TLS read, header read or body read raises immediately
    instead of restarting a fresh per-operation timeout. ``fired`` tells
    the caller whether a raised exception *is* the deadline.

    One blocking primitive stays out of reach: while ``getaddrinfo`` /
    TCP connect are still resolving there is no socket to close, so the
    per-attempt socket timeout is the only bound on that phase.
    """

    def __init__(
        self, connection: http.client.HTTPConnection, deadline: float
    ) -> None:
        self._connection = connection
        self._fired = threading.Event()
        self._timer = threading.Timer(
            max(0.0, deadline - time.monotonic()), self._expire
        )
        self._timer.daemon = True

    def __enter__(self) -> _DeadlineWatchdog:
        self._timer.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._timer.cancel()

    @property
    def fired(self) -> bool:
        return self._fired.is_set()

    def _expire(self) -> None:
        self._fired.set()
        # shutdown() reliably wakes a recv blocked in another thread on
        # every platform; close() alone may leave it blocked on some.
        sock = getattr(self._connection, "sock", None)
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self._connection.close()


# -- request translation ------------------------------------------------------


def build_payload(request: ModelRequest) -> dict[str, object]:
    """The provider-neutral request in the observed wire format."""
    messages: list[dict[str, str]] = []
    if request.instruction:
        messages.append({"role": "system", "content": request.instruction})
    messages.extend(
        {"role": message.role.value, "content": message.content}
        for message in request.messages
    )
    payload: dict[str, object] = {"model": request.model, "messages": messages}
    if request.max_output_tokens is not None:
        payload["max_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.schema.name,
                # Verbatim, thawed for serialization only: closed-by-default
                # was made explicit at OutputSchema construction, so what
                # goes on the wire is exactly what the fingerprint covers.
                "schema": _plain(request.schema.json_schema),
            },
        }
    return payload


def _plain(value: object) -> object:
    """Thaw a data value verbatim: frozen mappings become dicts, tuples
    become lists, nothing is injected."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _api_key() -> str:
    """The credential, read at invocation time and held only on the stack.

    The value is used verbatim except for trimming surrounding whitespace
    (a trailing newline from a careless ``echo`` would otherwise be
    rejected by HTTP header validation). The adapter never rewrites a
    credential beyond that: a malformed key is the caller's to fix, and
    silently mutating one would make the fix invisible.
    """
    for name in KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise ProviderConfigurationError(
        "no Muse API key available: set MUSE_API_KEY (or MODEL_API_KEY, the "
        "name the official documentation uses)"
    )


# -- response translation -----------------------------------------------------


def parse_response(
    body: Mapping[str, object],
    headers: Mapping[str, str],
    *,
    request: ModelRequest,
    latency: float,
) -> ModelResponse:
    """The observed success envelope, translated to neutral terms.

    Token counts are carried over exactly as reported. The detail counts
    that have no seat in ``ProviderUsage`` — ``reasoning_tokens`` billed
    inside the completion, ``cached_tokens`` inside the prompt — are
    preserved verbatim as ``muse:``-prefixed response metadata rather than
    dropped. Nominal cost is ``None``: the account's actual rate (tier,
    cached-token discounts) is not knowable from the response, and a wrong
    price is worse than an honest unknown.

    Accounting is extracted *before* the reply's content is judged: a
    truncated or contract-violating reply was still billed, so the raised
    error carries the observed :class:`CallAccounting` rather than
    discarding it.
    """
    accounting = _call_accounting(body, headers, request=request, latency=latency)
    try:
        finish_reason, content = _first_choice(body)
        if request.schema is not None:
            structured = request.schema.parse(content)
            schema_name = request.schema.name
        else:
            structured, schema_name = None, ""
    except ModelProviderError as exc:
        if accounting is not None:
            exc.with_accounting(accounting)
        raise

    usage = body.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    served_model = str(body.get("model") or request.model)
    completion_id = str(body.get("id") or "") or None
    metadata = dict(request.metadata)
    for key, reported in (
        (
            "muse:reasoning_tokens",
            _detail(usage, "completion_tokens_details", "reasoning_tokens"),
        ),
        (
            "muse:cached_tokens",
            _detail(usage, "prompt_tokens_details", "cached_tokens"),
        ),
    ):
        if reported is not None:
            metadata[key] = str(reported)
    return ModelResponse(
        provider="muse",
        model=served_model,
        text=content,
        structured=structured,
        usage=ProviderUsage(
            calls=1,
            input_tokens=_count(usage, "prompt_tokens"),
            output_tokens=_count(usage, "completion_tokens"),
            model=served_model,
        ),
        latency_seconds=latency,
        nominal_cost=None,
        request_id=headers.get("x-request-id") or completion_id,
        finish_reason=finish_reason,
        request_fingerprint=request.fingerprint,
        schema_name=schema_name,
        metadata=metadata,
    )


def _call_accounting(
    body: Mapping[str, object],
    headers: Mapping[str, str],
    *,
    request: ModelRequest,
    latency: float,
) -> CallAccounting | None:
    """What the provider reported having spent, independent of whether the
    reply is usable. ``None`` when the body carries no usage counts —
    unknown spend stays unknown rather than becoming a reported zero."""
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) or isinstance(prompt, bool):
        return None
    if not isinstance(completion, int) or isinstance(completion, bool):
        return None
    served_model = str(body.get("model") or request.model)
    return CallAccounting(
        usage=ProviderUsage(
            calls=1,
            input_tokens=prompt,
            output_tokens=completion,
            model=served_model,
        ),
        latency_seconds=latency,
        request_id=headers.get("x-request-id") or (str(body.get("id") or "") or None),
        model=served_model,
        nominal_cost=None,
    )


def _first_choice(body: Mapping[str, object]) -> tuple[str, str]:
    """``(finish_reason, content)`` of the first choice, or an
    :class:`InvalidModelResponseError` naming what was missing."""
    choices = body.get("choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise InvalidModelResponseError("the reply carried no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise InvalidModelResponseError("the reply's first choice is not an object")
    finish_reason = str(first.get("finish_reason") or "")
    if finish_reason in _TRUNCATING_FINISH_REASONS:
        # The authoritative truncation signal. Observed live with content
        # null, but a truncated generation that still carries partial text
        # is equally unusable — content emptiness is a symptom, not the
        # test.
        raise InvalidModelResponseError(
            f"the generation was truncated (finish_reason {finish_reason!r})"
        )
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise InvalidModelResponseError("the reply's choice carries no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        refusal = message.get("refusal")
        detail = f" (refusal: {refusal})" if isinstance(refusal, str) else ""
        raise InvalidModelResponseError(
            f"the reply carried no content "
            f"(finish_reason {finish_reason!r}){detail}"
        )
    return finish_reason, content


def _count(usage: Mapping[str, object], field: str) -> int:
    value = usage.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _detail(
    usage: Mapping[str, object], container: str, field: str
) -> int | None:
    """A nested usage detail count, or ``None`` when the provider did not
    report one — absent detail is not zero detail."""
    details = usage.get(container)
    if not isinstance(details, Mapping):
        return None
    value = details.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# -- error translation --------------------------------------------------------


def _error_for_status(
    status: int, headers: Mapping[str, str], raw: bytes
) -> ModelProviderError:
    """The documented error envelope, mapped onto the typed hierarchy.

    429 becomes the rate-limit error with the server's own ``Retry-After``;
    everything else the provider refused or failed — authentication,
    billing, invalid request, server errors, the server-side 504 — is a
    transport-level failure carrying the status and the provider's error
    code, because no usable reply was produced and the caller's own
    deadline was not the cause.
    """
    message, code = _error_fields(raw, status)
    if status == 429:
        return ProviderRateLimitError(
            message, retry_after_seconds=_retry_after(headers)
        )
    return ProviderTransportError(
        message, status_code=status, provider_error=code
    )


def _error_fields(raw: bytes, status: int) -> tuple[str, str | None]:
    """``(message, code)`` from the ``{"error": {...}}`` envelope, with a
    plain fallback for bodies that are not the documented shape."""
    fallback = f"the Muse endpoint returned HTTP {status}"
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return fallback, None
    if not isinstance(payload, Mapping):
        return fallback, None
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return fallback, None
    message = error.get("message")
    # The first usable identifier wins: a non-string "code" (some gateways
    # send numbers) must not short-circuit away a perfectly good "type".
    code = next(
        (
            value
            for value in (error.get("code"), error.get("type"))
            if isinstance(value, str) and value
        ),
        None,
    )
    return (
        f"{fallback}: {message}" if isinstance(message, str) else fallback,
        code,
    )


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The server's wait, accepted only when it is a usable number: finite
    and non-negative. 'nan' would crash a caller's time.sleep, 'inf' would
    sleep forever, and the HTTP-date form degrades to None."""
    try:
        seconds = float(headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _lowered(items: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Header pairs as a plain lower-cased dict — the last stdlib-HTTP-
    shaped object converted before anything leaves the adapter."""
    return {key.lower(): value for key, value in items}


class _Reader(Protocol):
    def read(self, amount: int, /) -> bytes: ...


def _read_bounded(reply: _Reader, *, deadline: float, timeout: float) -> bytes:
    """The whole success body, bounded in both time and size.

    The connection's ``timeout`` is a per-socket-operation limit, and a
    buffered ``read(n)`` issues as many socket reads as it takes to fill
    ``n`` bytes — so a server dripping one byte per interval would never
    trip either the socket timeout or a between-chunks clock check. Two
    measures make the deadline real: each iteration reads with ``read1``
    (at most one underlying socket read, buffer served first), so the
    deadline check runs between every socket operation; and the socket
    timeout is shrunk, best-effort, to the remaining deadline so the one
    blocking read cannot overshoot it either. The size cap converts a
    body no sane completion produces into
    :class:`InvalidModelResponseError` instead of unbounded memory growth.
    """
    read_one = getattr(reply, "read1", None)
    chunks = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderTimeoutError(
                f"the reply body did not complete within {timeout}s",
                timeout_seconds=timeout,
            )
        _tighten_timeout(reply, remaining)
        chunk: bytes = (
            read_one(_CHUNK_BYTES)
            if callable(read_one)
            else reply.read(_CHUNK_BYTES)
        )
        if not chunk:
            return bytes(chunks)
        chunks += chunk
        if len(chunks) > _MAX_BODY_BYTES:
            raise InvalidModelResponseError(
                f"the reply body exceeds {_MAX_BODY_BYTES} bytes"
            )


def _tighten_timeout(reply: object, remaining: float) -> None:
    """Best effort: shrink the socket timeout to the remaining deadline so
    the next blocking read cannot overshoot it. Reaches through stdlib
    internals (``fp.raw._sock``) behind hasattr-style guards; when the
    socket is not reachable this way, the connection's per-operation
    timeout still bounds the read."""
    fp = getattr(reply, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    settimeout = getattr(sock, "settimeout", None)
    if callable(settimeout):
        # A socket mid-close may refuse; the connection timeout still holds.
        with contextlib.suppress(OSError):
            settimeout(remaining)


def _drain_error_body(reply: _Reader, *, deadline: float) -> bytes:
    """The error envelope, best effort and bounded in size *and* time.

    The status and headers already classify the failure, so a body that
    cannot be read — stalled mid-drain, dropped, or absent — degrades to
    the status-code fallback rather than replacing a classified HTTP
    failure with an untyped crash from inside the error handler. The drain
    runs inside the watchdog window and checks the deadline between
    chunks, so a server dripping an error body cannot hold the call open
    past the deadline either.
    """
    read_one = getattr(reply, "read1", None)
    chunks = bytearray()
    try:
        while len(chunks) <= _MAX_ERROR_BODY_BYTES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _tighten_timeout(reply, remaining)
            chunk: bytes = (
                read_one(_CHUNK_BYTES)
                if callable(read_one)
                else reply.read(_CHUNK_BYTES)
            )
            if not chunk:
                break
            chunks += chunk
    except (
        TimeoutError,
        OSError,
        ValueError,
        AttributeError,
        http.client.HTTPException,
    ):
        pass
    return bytes(chunks)
