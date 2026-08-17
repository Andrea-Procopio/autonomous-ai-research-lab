"""The model-provider boundary: one narrow seam between a role and an SDK.

Roles need model calls; nothing else in the system should know that a model
exists, let alone which vendor serves it. This module is the whole contract:

    ModelRequest  ->  ModelProvider.invoke()  ->  ModelResponse

An adapter translates a request into whatever its SDK wants, calls it, and
translates the reply back. **No provider object crosses this boundary.**
Every field of :class:`ModelResponse` is a string, a number, or a mapping of
those, so a caller cannot accidentally depend on a vendor type, and a
recorded response stays readable after the SDK is gone.

Structured output fails closed
------------------------------

An :class:`OutputSchema` is checked twice: once at construction, where a
schema this module cannot enforce itself is rejected, and once on every
reply, where the returned JSON is validated locally. A provider that ignores
the schema, truncates the JSON, or invents a field therefore produces a
:class:`StructuredOutputError` rather than a plausible-looking object. The
guarantee does not depend on the provider honouring anything.

Failures are runtime failures
-----------------------------

Everything raised here derives from :class:`ModelProviderError`, a
``RuntimeError``. A model call that fails is an infrastructure event: it is
not a negative result, not evidence, and not a scientific outcome of any
kind. Nothing in this module can reach ``ResearchState`` — model output
becomes a proposal only after a role builds one and the transition layer
commits it.

The six distinctions callers can act on: local misconfiguration
(:class:`ProviderConfigurationError`), transport (:class:`ProviderTransportError`),
timeout (:class:`ProviderTimeoutError`), rate limit (:class:`ProviderRateLimitError`),
unusable reply (:class:`InvalidModelResponseError`), and schema violation
(:class:`StructuredOutputError`).

A failure observed *after* the provider replied may still have been billed.
Adapters attach the observed :class:`CallAccounting` to such errors, and
:meth:`UsageLedger.record_failure` folds it into the same per-step usage
the loop already drains — so a truncated or contract-violating reply keeps
its cost on the books instead of vanishing from the accounting.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Self

from ..core.budget import ResourceCost
from ..core.ids import content_id, occurrence_id
from ..core.types import freeze_mapping
from .metrics import NO_USAGE, ProviderUsage

DEFAULT_TIMEOUT_SECONDS: Final = 120.0
"""Every model call has a finite timeout, for the same reason every
experiment job does: "wait forever" is not an option the contract offers."""


# -- failures -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallAccounting:
    """What one model call observably cost, independent of whether its
    reply was usable.

    Every field records what the provider reported, exactly: unknown
    values stay unknown (``None`` cost, absent accounting) rather than
    becoming invented zeros. Attached to failures so that a billed call
    that raised keeps its spend on the books.
    """

    usage: ProviderUsage
    latency_seconds: float
    request_id: str | None = None
    model: str = ""
    nominal_cost: ResourceCost | None = None


class ModelProviderError(RuntimeError):
    """Base class for every model-call failure.

    A failed model call is an infrastructure event, never a scientific
    result. Nothing derived from this class may be recorded as evidence.

    A failure observed after the provider replied may still have been
    billed; adapters attach the observed :class:`CallAccounting` via
    :meth:`with_accounting`. ``None`` means no accounting was observed —
    unknown spend, not zero spend.
    """

    accounting: CallAccounting | None = None

    def with_accounting(self, accounting: CallAccounting) -> Self:
        """Attach the observed cost of this failed call; returns self."""
        self.accounting = accounting
        return self


class ProviderConfigurationError(ModelProviderError):
    """The caller's own provider configuration is missing or malformed —
    an absent API key, an unusable endpoint — detected before any request
    is made.

    Distinct from the transport family because it is permanent until a
    human fixes it: no retry or backoff can help. A key the *provider*
    rejects is :class:`ProviderTransportError` territory — that verdict
    came from the remote side, and only it knows.
    """


class ProviderTransportError(ModelProviderError):
    """The provider could not be reached, or refused the request.

    Covers connection failures, server errors, authentication and
    permission refusals: cases where no usable reply was produced and the
    fault is not specifically a timeout or a rate limit.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_error = provider_error
        """The provider's own error identifier, when it supplied one. Kept as
        a plain string: adapters translate, they do not re-export SDK types."""


class ProviderTimeoutError(ModelProviderError):
    """The call exceeded its deadline. Distinguished from transport failure
    because retrying a timeout costs a full request again."""

    def __init__(self, message: str, *, timeout_seconds: float) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class ProviderRateLimitError(ModelProviderError):
    """The provider is throttling. Carries the wait it asked for, when it
    gave one, so a caller can back off on fact rather than on guesswork."""

    def __init__(
        self, message: str, *, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class InvalidModelResponseError(ModelProviderError):
    """The provider replied, but the reply is unusable: no content, a
    truncated generation, or a payload that is not the shape the protocol
    promises."""


class StructuredOutputError(ModelProviderError):
    """The reply did not satisfy the requested schema.

    Deliberately not a subclass of :class:`InvalidModelResponseError`: a
    well-formed reply that violates the contract is a different problem from
    a malformed reply, and the two warrant different handling.
    """

    def __init__(self, message: str, *, schema: str, detail: str = "") -> None:
        super().__init__(message)
        self.schema = schema
        self.detail = detail


class SchemaDefinitionError(ValueError):
    """Raised when an :class:`OutputSchema` uses a construct this module
    cannot validate. Rejecting it at construction is the fail-closed half of
    structured output: we never request a shape we could not check."""


# -- structured output --------------------------------------------------------


#: The JSON Schema keywords this module validates. Anything else is rejected
#: at construction rather than sent and hoped for.
_SUPPORTED_KEYWORDS: Final = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "description",
    }
)

_SUPPORTED_TYPES: Final = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)


def _deep_freeze(mapping: Mapping[str, object]) -> Mapping[str, object]:
    """A recursively read-only view of ``mapping``. Core's ``freeze_mapping``
    is shallow by design; schema bodies and structured payloads nest, and a
    frozen record with mutable insides is not frozen."""
    return MappingProxyType(
        {str(key): _frozen(value) for key, value in mapping.items()}
    )


def _frozen(value: object) -> object:
    if isinstance(value, Mapping):
        return _deep_freeze(value)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class OutputSchema:
    """A JSON shape a reply must satisfy, in a subset of JSON Schema.

    The subset is small on purpose: ``type``, ``properties``, ``required``,
    ``additionalProperties``, ``items``, ``enum``, ``description``. It covers
    the flat, typed records roles actually need, and every one of its
    constructs is enforced locally by :meth:`validate`.
    """

    name: str
    json_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SchemaDefinitionError("schema name must be non-empty")
        object.__setattr__(self, "json_schema", _deep_freeze(self.json_schema))
        _check_schema(self.json_schema, path="$")
        if self.json_schema.get("type") != "object":
            raise SchemaDefinitionError(
                "$: the root schema must describe a JSON object — structured "
                "output is a record, not a bare value"
            )

    def validate(self, payload: object) -> Mapping[str, object]:
        """Return ``payload``, recursively frozen, if it satisfies the
        schema; raise :class:`StructuredOutputError` otherwise."""
        problems: list[str] = []
        _validate(payload, self.json_schema, path="$", problems=problems)
        if problems:
            raise StructuredOutputError(
                f"model output does not satisfy schema {self.name!r}",
                schema=self.name,
                detail="; ".join(problems),
            )
        # The root schema is an object by construction, so a payload that
        # validated is a mapping.
        assert isinstance(payload, Mapping)
        return _deep_freeze(payload)

    def parse(self, text: str) -> Mapping[str, object]:
        """Parse ``text`` as JSON and validate it. Fails closed on both."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"model output for schema {self.name!r} is not valid JSON",
                schema=self.name,
                detail=str(exc),
            ) from exc
        return self.validate(payload)


def _check_schema(schema: Mapping[str, object], *, path: str) -> None:
    """Reject, at construction, any schema this module cannot enforce."""
    unsupported = sorted(set(schema) - _SUPPORTED_KEYWORDS)
    if unsupported:
        raise SchemaDefinitionError(
            f"{path}: unsupported schema keyword(s): {', '.join(unsupported)}"
        )
    declared = schema.get("type")
    if declared is None:
        raise SchemaDefinitionError(f"{path}: schema must declare a type")
    if not isinstance(declared, str) or declared not in _SUPPORTED_TYPES:
        raise SchemaDefinitionError(f"{path}: unsupported type {declared!r}")

    properties = schema.get("properties")
    if declared == "object":
        if not isinstance(properties, Mapping) or not properties:
            raise SchemaDefinitionError(
                f"{path}: an object schema must declare properties"
            )
        for name, child in properties.items():
            if not isinstance(child, Mapping):
                raise SchemaDefinitionError(f"{path}.{name}: property must be a schema")
            _check_schema(child, path=f"{path}.{name}")
        required = schema.get("required", ())
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise SchemaDefinitionError(f"{path}: 'required' must be a list of names")
        missing = [name for name in required if name not in properties]
        if missing:
            raise SchemaDefinitionError(
                f"{path}: 'required' names undeclared propert(ies): "
                f"{', '.join(str(m) for m in missing)}"
            )
    elif properties is not None:
        raise SchemaDefinitionError(f"{path}: only object schemas take properties")

    additional = schema.get("additionalProperties")
    if additional is not None:
        if declared != "object":
            raise SchemaDefinitionError(
                f"{path}: only object schemas take additionalProperties"
            )
        if not isinstance(additional, bool):
            raise SchemaDefinitionError(
                f"{path}: 'additionalProperties' must be a boolean — a "
                f"schema-valued additionalProperties is outside the "
                f"supported subset"
            )

    items = schema.get("items")
    if declared == "array":
        if not isinstance(items, Mapping):
            raise SchemaDefinitionError(f"{path}: an array schema must declare items")
        _check_schema(items, path=f"{path}[]")
    elif items is not None:
        raise SchemaDefinitionError(f"{path}: only array schemas take items")

    enum = schema.get("enum")
    if enum is not None and (
        not isinstance(enum, Sequence)
        or isinstance(enum, (str, bytes))
        or not enum
    ):
        raise SchemaDefinitionError(f"{path}: 'enum' must be a non-empty list")


def _validate(
    value: object,
    schema: Mapping[str, object],
    *,
    path: str,
    problems: list[str],
) -> None:
    declared = schema["type"]
    if not _matches_type(value, str(declared)):
        problems.append(
            f"{path}: expected {declared}, got {type(value).__name__}"
        )
        return

    enum = schema.get("enum")
    if (
        isinstance(enum, Sequence)
        and not isinstance(enum, (str, bytes))
        and value not in enum
    ):
        problems.append(f"{path}: {value!r} is not one of the permitted values")

    if declared == "object":
        assert isinstance(value, Mapping)
        properties = schema.get("properties")
        assert isinstance(properties, Mapping)
        required = schema.get("required", ())
        assert isinstance(required, Sequence)
        for name in required:
            if name not in value:
                problems.append(f"{path}: missing required property {name!r}")
        if schema.get("additionalProperties", False) is False:
            for name in value:
                if name not in properties:
                    problems.append(f"{path}: unexpected property {str(name)!r}")
        for name, child in properties.items():
            if name in value:
                assert isinstance(child, Mapping)
                _validate(
                    value[name], child, path=f"{path}.{name}", problems=problems
                )
    elif declared == "array":
        assert isinstance(value, Sequence)
        items = schema.get("items")
        assert isinstance(items, Mapping)
        for index, element in enumerate(value):
            _validate(
                element, items, path=f"{path}[{index}]", problems=problems
            )


def _matches_type(value: object, declared: str) -> bool:
    match declared:
        case "object":
            return isinstance(value, Mapping)
        case "array":
            return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        case "string":
            return isinstance(value, str)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "null":
            return value is None
    return False  # pragma: no cover - construction rejects unknown types


# -- the request --------------------------------------------------------------


class MessageRole(StrEnum):
    """Who authored a turn. The standing instruction is a separate field on
    the request, not a message role, because it is the invocation's contract
    rather than part of the conversation."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("message content must be non-empty")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One model call, described in provider-neutral terms."""

    model: str
    """The model asked for. Providers record what actually served the call
    separately, so a silent substitution is visible in the record."""

    instruction: str = ""
    """The standing instruction (the "system prompt")."""

    messages: tuple[Message, ...] = ()
    schema: OutputSchema | None = None
    """When set, the reply must be JSON satisfying this schema, validated
    locally before the response is returned."""

    max_output_tokens: int | None = None
    temperature: float | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    metadata: Mapping[str, str] = field(default_factory=dict)
    """Caller-supplied provenance carried through to the response, such as
    the invocation or attempt this call belongs to."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not self.model.strip():
            raise ValueError("request must name a model")
        if not self.messages:
            raise ValueError("request must carry at least one message")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive when set")

    @property
    def fingerprint(self) -> str:
        """A content id over everything that determines the reply — the full
        schema body and the timeout included, since two schemas may share a
        name and a deadline shapes what can be generated. Ties a recorded
        response back to the exact request that produced it without storing
        the prompt twice. Caller ``metadata`` is provenance, not content, so
        it deliberately does not participate."""
        return content_id(
            "mreq",
            self.model,
            self.instruction,
            tuple((m.role.value, m.content) for m in self.messages),
            self.schema.name if self.schema else "",
            self.schema.json_schema if self.schema else "",
            self.max_output_tokens,
            self.temperature,
            self.timeout_seconds,
        )


# -- the response -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One completed model call, in terms nothing downstream has to decode.

    Every field is a primitive or a mapping of primitives. An adapter that
    wants to pass its SDK's response object through has to translate it
    first, which is the point.
    """

    provider: str
    model: str
    """The model that actually served the call, which may differ from the
    model requested."""

    text: str
    structured: Mapping[str, object] | None = None
    """The validated structured payload, when the request carried a schema.
    Never populated without validation having passed."""

    usage: ProviderUsage = NO_USAGE
    latency_seconds: float = 0.0
    nominal_cost: ResourceCost | None = None
    """What the call is priced at, when the adapter knows the rate. ``None``
    means unknown; a zero ``ResourceCost`` is a known zero, exactly as it is
    everywhere in ``core``. Token counts in ``usage`` are the ground truth
    either way."""

    request_id: str | None = None
    """The provider's own id for the call, when it returns one. The handle
    for a support conversation about a specific request."""

    finish_reason: str = ""
    request_fingerprint: str = ""
    schema_name: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    id: str = field(default="")
    """Occurrence id: two identical calls are two events."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if self.structured is not None:
            object.__setattr__(self, "structured", _deep_freeze(self.structured))
        if not self.id:
            object.__setattr__(self, "id", occurrence_id("mcall"))


class ModelProvider(ABC):
    """The one interface a role uses to reach a model.

    Implementations translate to and from a vendor SDK and raise the
    :class:`ModelProviderError` family on failure. They never touch
    ``ResearchState``: a model's output is text until a role turns it into a
    proposal and the transition layer commits it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable provider identifier, recorded on every response."""

    @abstractmethod
    def invoke(self, request: ModelRequest) -> ModelResponse:
        """Perform ``request`` and return a validated response.

        Raises :class:`ProviderConfigurationError`,
        :class:`ProviderTimeoutError`, :class:`ProviderRateLimitError`,
        :class:`ProviderTransportError`, :class:`InvalidModelResponseError`,
        or :class:`StructuredOutputError`.
        """


class UsageLedger:
    """Accumulates provider usage so the runtime loop can drain it per step.

    This is the join between a provider and the existing metrics seam: a
    provider records what a call cost, the loop drains the total into
    ``StepMetrics``. It satisfies ``UsageSource`` structurally.
    """

    def __init__(self) -> None:
        self._pending = NO_USAGE

    def record(self, response: ModelResponse) -> None:
        self._pending = self._pending + response.usage

    def record_failure(self, error: ModelProviderError) -> bool:
        """Add the accounting a failed call carried, when it carried any.

        Returns whether anything was recorded, so a caller can tell
        recorded spend from unknown spend. A failure without accounting
        adds nothing — unknown is not zero.

        Call this once per caught failure. A failure never also produced
        a ``ModelResponse``, so it cannot duplicate a success recorded via
        :meth:`record` — but the ledger does not deduplicate: passing the
        same error twice records its spend twice.
        """
        if error.accounting is None:
            return False
        self._pending = self._pending + error.accounting.usage
        return True

    def drain(self) -> ProviderUsage:
        drained, self._pending = self._pending, NO_USAGE
        return drained


# -- the deterministic test provider ------------------------------------------


@dataclass(frozen=True, slots=True)
class ScriptedReply:
    """One canned reply, or one canned failure, for the fake provider."""

    text: str = ""
    error: ModelProviderError | None = None
    model: str = ""
    """The model to report as having served the call. Empty means "the model
    that was requested"; setting it exercises substitution handling."""

    finish_reason: str = "stop"
    request_id: str | None = None
    nominal_cost: ResourceCost | None = None
    """``None`` means the fake reports no price, mirroring an adapter that
    does not know its rate card."""


class FakeModelProvider(ModelProvider):
    """A provider with no network, no clock, and no randomness.

    Replies are served from a script, in order. Usage and latency are derived
    from the request and the reply by fixed rules, so the same script gives
    byte-identical responses on every run and in every environment. Token
    counts are whitespace word counts: enough to prove usage metadata
    survives the boundary, and deliberately not a real tokenizer.
    """

    def __init__(
        self,
        replies: Sequence[ScriptedReply | str],
        *,
        name: str = "fake",
        latency_seconds: float = 0.25,
    ) -> None:
        self._replies = tuple(
            ScriptedReply(text=r) if isinstance(r, str) else r for r in replies
        )
        self._name = name
        self._latency = latency_seconds
        self._calls: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def calls(self) -> tuple[ModelRequest, ...]:
        """Every request received, in order — the assertion surface for
        tests about what a role actually asked for."""
        return tuple(self._calls)

    def invoke(self, request: ModelRequest) -> ModelResponse:
        index = len(self._calls)
        self._calls.append(request)
        if index >= len(self._replies):
            raise InvalidModelResponseError(
                f"fake provider was scripted for {len(self._replies)} call(s), "
                f"received call {index + 1}"
            )
        reply = self._replies[index]
        if reply.error is not None:
            raise reply.error

        structured = request.schema.parse(reply.text) if request.schema else None
        return ModelResponse(
            provider=self._name,
            model=reply.model or request.model,
            text=reply.text,
            structured=structured,
            usage=ProviderUsage(
                calls=1,
                input_tokens=_word_count(request),
                output_tokens=len(reply.text.split()),
                model=reply.model or request.model,
            ),
            latency_seconds=self._latency,
            nominal_cost=reply.nominal_cost,
            request_id=reply.request_id,
            finish_reason=reply.finish_reason,
            request_fingerprint=request.fingerprint,
            schema_name=request.schema.name if request.schema else "",
            metadata=request.metadata,
        )


def _word_count(request: ModelRequest) -> int:
    return len(request.instruction.split()) + sum(
        len(m.content.split()) for m in request.messages
    )
