"""Muse Spark adapter: the Meta Model API behind the ModelProvider seam.

The one concrete :class:`~autonomous_research_lab.runtime.providers.
ModelProvider`. Standard library HTTP only — the package keeps its
zero-dependency promise, and every vendor-shaped object (URLs, wire
payloads, urllib types) stays inside this module.

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

* **Closed schemas are made explicit on the wire.** The local validator
  rejects undeclared properties unless a schema opts out, but the provider
  treats an *absent* ``additionalProperties`` as open — observed directly:
  the model added six undeclared fields until ``additionalProperties:
  false`` was sent, and none after. The adapter therefore injects
  ``additionalProperties: false`` into every object schema that does not
  set it, so the model is constrained to exactly what validation accepts.
* **The provider is never trusted with validation.** Whatever
  ``response_format`` promises, the returned text goes through
  ``OutputSchema.parse`` locally, and fails closed.

The API key is read from the environment at invocation time —
``MUSE_API_KEY`` first, then ``MODEL_API_KEY`` (the name the official
documentation uses) — and appears in nothing but the request header.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Final

from .metrics import ProviderUsage
from .providers import (
    InvalidModelResponseError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransportError,
)

DEFAULT_BASE_URL: Final = "https://api.meta.ai/v1"
MUSE_SPARK_1_2: Final = "muse-spark-1.2"
KEY_ENV_VARS: Final = ("MUSE_API_KEY", "MODEL_API_KEY")


class MuseSparkProvider(ModelProvider):
    """Meta Model API adapter, synchronous, one request per invoke."""

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "muse"

    def invoke(self, request: ModelRequest) -> ModelResponse:
        payload = json.dumps(build_payload(request)).encode("utf-8")
        api_key = _api_key()
        started = time.monotonic()
        raw, headers = self._post(payload, api_key, request.timeout_seconds)
        latency = time.monotonic() - started
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
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
        """One HTTP exchange. urllib types enter and never leave."""
        http_request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=timeout) as reply:
                return reply.read(), _lowered(reply.headers.items())
        except urllib.error.HTTPError as exc:
            raise _error_for_status(
                exc.code, _lowered(exc.headers.items()), exc.read()
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ProviderTimeoutError(
                    f"no reply from the Muse endpoint within {timeout}s",
                    timeout_seconds=timeout,
                ) from exc
            raise ProviderTransportError(
                f"could not reach the Muse endpoint: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"no reply from the Muse endpoint within {timeout}s",
                timeout_seconds=timeout,
            ) from exc
        except OSError as exc:
            raise ProviderTransportError(
                f"could not reach the Muse endpoint: {exc}"
            ) from exc


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
                "schema": _wire_schema(request.schema.json_schema),
            },
        }
    return payload


def _wire_schema(schema: Mapping[str, object]) -> dict[str, object]:
    """Thaw the deep-frozen schema for serialization, making the local
    closed-by-default semantics explicit wherever a schema does not opt
    out — the provider reads an absent ``additionalProperties`` as open,
    the validator reads it as closed, and the model should be constrained
    to what validation will accept."""
    thawed = {key: _thaw(value) for key, value in schema.items()}
    if thawed.get("type") == "object" and "additionalProperties" not in thawed:
        thawed["additionalProperties"] = False
    return thawed


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return _wire_schema(value)
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
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
    raise ProviderTransportError(
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
    """
    finish_reason, content = _first_choice(body)
    if request.schema is not None:
        structured = request.schema.parse(content)
        schema_name = request.schema.name
    else:
        structured, schema_name = None, ""

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
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise InvalidModelResponseError("the reply's choice carries no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        # Observed live: exhausting max_tokens on reasoning yields
        # finish_reason "length" with content null. A reply with nothing
        # in it is unusable, whatever the status code said.
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
    code = error.get("code") or error.get("type")
    return (
        f"{fallback}: {message}" if isinstance(message, str) else fallback,
        code if isinstance(code, str) else None,
    )


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after", "")
    try:
        return float(value)
    except ValueError:
        return None


def _lowered(items: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Header pairs as a plain lower-cased dict — the last urllib-shaped
    object converted before anything leaves the adapter."""
    return {key.lower(): value for key, value in items}
