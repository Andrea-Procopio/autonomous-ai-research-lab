"""The model-provider boundary: validated output, auditable usage, no SDK
types crossing over, and failures that stay infrastructure failures."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from autonomous_research_lab.core.budget import ResourceCost
from autonomous_research_lab.runtime.metrics import ProviderUsage
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    InvalidModelResponseError,
    Message,
    MessageRole,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    OutputSchema,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransportError,
    SchemaDefinitionError,
    ScriptedReply,
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


def _request(**overrides: object) -> ModelRequest:
    defaults: dict[str, object] = {
        "model": "test-model-1",
        "instruction": "Propose one testable hypothesis.",
        "messages": (
            Message(role=MessageRole.USER, content="The stream may be biased."),
        ),
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)  # type: ignore[arg-type]


# -- a successful invocation --------------------------------------------------


def test_a_successful_invocation_returns_text_and_provenance() -> None:
    provider = FakeModelProvider(["A plain textual answer."])
    request = _request()

    response = provider.invoke(request)

    assert response.text == "A plain textual answer."
    assert response.structured is None
    assert response.provider == "fake"
    assert response.model == "test-model-1"
    assert response.finish_reason == "stop"
    # Enough provenance to audit the call after the fact.
    assert response.id.startswith("mcall_")
    assert response.request_fingerprint == request.fingerprint
    assert response.latency_seconds == 0.25


def test_the_request_carries_caller_provenance_through_to_the_response() -> None:
    provider = FakeModelProvider(["ok"])

    response = provider.invoke(
        _request(metadata={"attempt_id": "att_1", "role": "research_engineer"})
    )

    assert response.metadata["attempt_id"] == "att_1"
    assert response.metadata["role"] == "research_engineer"


def test_a_substituted_model_is_visible_in_the_record() -> None:
    """Asking for one model and being served another is recorded, not hidden."""
    provider = FakeModelProvider(
        [ScriptedReply(text="ok", model="test-model-1-fallback")]
    )

    response = provider.invoke(_request(model="test-model-1"))

    assert response.model == "test-model-1-fallback"
    assert response.usage.model == "test-model-1-fallback"


def test_the_provider_returns_only_primitives() -> None:
    """The boundary condition: nothing a vendor SDK owns crosses it."""
    provider = FakeModelProvider(
        [json.dumps({"statement": "The stream is biased.", "confidence": 0.6})]
    )

    response = provider.invoke(_request(schema=HYPOTHESIS_SCHEMA))

    assert isinstance(response, ModelResponse)
    assert isinstance(response.text, str)
    assert isinstance(response.structured, Mapping)
    for value in response.structured.values():
        assert isinstance(value, (str, int, float, bool, tuple, type(None)))
    assert isinstance(response.usage, ProviderUsage)
    assert response.nominal_cost is None  # unknown, and stated as unknown


# -- structured output --------------------------------------------------------


def test_structured_output_is_parsed_and_validated() -> None:
    payload = {
        "statement": "The stream is biased toward heads.",
        "confidence": 0.62,
        "assumptions": ["draws are independent"],
    }
    provider = FakeModelProvider([json.dumps(payload)])

    response = provider.invoke(_request(schema=HYPOTHESIS_SCHEMA))

    assert response.structured is not None
    assert response.structured["statement"] == payload["statement"]
    assert response.structured["confidence"] == 0.62
    assert response.schema_name == "hypothesis_v1"


def test_output_missing_a_required_field_is_rejected() -> None:
    provider = FakeModelProvider([json.dumps({"statement": "No confidence given."})])

    with pytest.raises(StructuredOutputError) as caught:
        provider.invoke(_request(schema=HYPOTHESIS_SCHEMA))

    assert caught.value.schema == "hypothesis_v1"
    assert "confidence" in caught.value.detail


def test_output_with_a_wrongly_typed_field_is_rejected() -> None:
    provider = FakeModelProvider(
        [json.dumps({"statement": "Biased.", "confidence": "high"})]
    )

    with pytest.raises(StructuredOutputError, match="hypothesis_v1"):
        provider.invoke(_request(schema=HYPOTHESIS_SCHEMA))


def test_unexpected_fields_are_rejected_rather_than_ignored() -> None:
    """Fail closed: a field nobody declared is a contract violation, not a
    bonus. Silently dropping it would let a model smuggle content past the
    schema."""
    provider = FakeModelProvider(
        [
            json.dumps(
                {
                    "statement": "Biased.",
                    "confidence": 0.6,
                    "certainty_boost": "trust me",
                }
            )
        ]
    )

    with pytest.raises(StructuredOutputError) as caught:
        provider.invoke(_request(schema=HYPOTHESIS_SCHEMA))

    assert "certainty_boost" in caught.value.detail


def test_nested_array_elements_are_validated() -> None:
    provider = FakeModelProvider(
        [
            json.dumps(
                {"statement": "Biased.", "confidence": 0.6, "assumptions": [1, 2]}
            )
        ]
    )

    with pytest.raises(StructuredOutputError) as caught:
        provider.invoke(_request(schema=HYPOTHESIS_SCHEMA))

    assert "assumptions[0]" in caught.value.detail


def test_malformed_json_is_rejected() -> None:
    provider = FakeModelProvider(["Sure! Here is your hypothesis: {statement: ..."])

    with pytest.raises(StructuredOutputError, match="not valid JSON"):
        provider.invoke(_request(schema=HYPOTHESIS_SCHEMA))


def test_truncated_json_is_rejected() -> None:
    provider = FakeModelProvider(['{"statement": "The stream is bia'])

    with pytest.raises(StructuredOutputError, match="not valid JSON"):
        provider.invoke(_request(schema=HYPOTHESIS_SCHEMA))


def test_enum_values_are_enforced() -> None:
    schema = OutputSchema(
        name="verdict_v1",
        json_schema={
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["supported", "refuted", "undetermined"],
                }
            },
            "required": ["verdict"],
        },
    )
    provider = FakeModelProvider([json.dumps({"verdict": "probably true"})])

    with pytest.raises(StructuredOutputError, match="verdict_v1"):
        provider.invoke(_request(schema=schema))


def test_structured_output_is_immutable_all_the_way_down() -> None:
    """The shallow freeze was a hole: nested lists and objects inside a
    validated payload could still be edited after the fact."""
    payload = {
        "statement": "Biased.",
        "confidence": 0.6,
        "assumptions": ["draws are independent"],
    }
    response = FakeModelProvider([json.dumps(payload)]).invoke(
        _request(schema=HYPOTHESIS_SCHEMA)
    )

    structured = response.structured
    assert structured is not None
    with pytest.raises(TypeError):
        structured["confidence"] = 0.99  # type: ignore[index]
    assumptions = structured["assumptions"]
    assert isinstance(assumptions, tuple)  # a tuple, not a mutable list


def test_the_schema_body_is_immutable_all_the_way_down() -> None:
    with pytest.raises(TypeError):
        HYPOTHESIS_SCHEMA.json_schema["type"] = "array"  # type: ignore[index]
    properties = HYPOTHESIS_SCHEMA.json_schema["properties"]
    assert isinstance(properties, Mapping)
    with pytest.raises(TypeError):
        properties["escape"] = {"type": "string"}  # type: ignore[index]
    statement = properties["statement"]
    assert isinstance(statement, Mapping)
    with pytest.raises(TypeError):
        statement["type"] = "number"  # type: ignore[index]
    assert isinstance(HYPOTHESIS_SCHEMA.json_schema["required"], tuple)


def test_schema_valued_additional_properties_fails_at_construction() -> None:
    """Full JSON Schema allows a schema there; this subset validates only
    booleans, so anything else is rejected up front rather than silently
    treated as permissive."""
    with pytest.raises(SchemaDefinitionError, match="additionalProperties"):
        OutputSchema(
            name="leaky",
            json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "additionalProperties": {"type": "string"},
            },
        )
    # The boolean forms stay constructible.
    OutputSchema(
        name="open",
        json_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": True,
        },
    )


def test_a_non_object_root_schema_fails_at_construction() -> None:
    """Structured output is a record. A bare-value root is discovered when
    the schema is written, not after a model call has been paid for."""
    for root in (
        {"type": "string"},
        {"type": "number"},
        {"type": "array", "items": {"type": "string"}},
    ):
        with pytest.raises(SchemaDefinitionError, match="JSON object"):
            OutputSchema(name="bare", json_schema=root)


def test_a_schema_this_module_cannot_enforce_is_rejected_at_construction() -> None:
    """The other half of failing closed: we never request a shape we could
    not check ourselves, so the guarantee does not rest on the provider."""
    with pytest.raises(SchemaDefinitionError, match="unsupported schema keyword"):
        OutputSchema(
            name="fancy",
            json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "patternProperties": {"^y": {"type": "string"}},
            },
        )
    with pytest.raises(SchemaDefinitionError, match="must declare a type"):
        OutputSchema(name="typeless", json_schema={"properties": {}})
    with pytest.raises(SchemaDefinitionError, match="undeclared propert"):
        OutputSchema(
            name="mismatched",
            json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["y"],
            },
        )


# -- failure propagation ------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ProviderTransportError(
            "connection reset", status_code=503, provider_error="server_error"
        ),
        ProviderTimeoutError("deadline exceeded", timeout_seconds=30.0),
        ProviderRateLimitError("throttled", retry_after_seconds=12.0),
        InvalidModelResponseError("reply carried no content"),
    ],
)
def test_provider_failures_propagate_to_the_caller(
    error: ModelProviderError,
) -> None:
    provider = FakeModelProvider([ScriptedReply(error=error)])

    with pytest.raises(type(error)):
        provider.invoke(_request())


def test_the_five_failure_kinds_are_distinguishable() -> None:
    """A caller can tell a timeout from a rate limit from a schema violation
    without string matching, and every one of them is a runtime error rather
    than anything scientific."""
    kinds = (
        ProviderTransportError("x"),
        ProviderTimeoutError("x", timeout_seconds=1.0),
        ProviderRateLimitError("x"),
        InvalidModelResponseError("x"),
        StructuredOutputError("x", schema="s"),
    )
    assert len({type(k) for k in kinds}) == 5
    for kind in kinds:
        assert isinstance(kind, ModelProviderError)
        assert isinstance(kind, RuntimeError)
    # A schema violation is not merely an invalid response: different fault,
    # different handling.
    assert not isinstance(
        StructuredOutputError("x", schema="s"), InvalidModelResponseError
    )


def test_failure_details_survive_for_the_caller_to_act_on() -> None:
    rate_limited = ProviderRateLimitError("slow down", retry_after_seconds=7.5)
    timed_out = ProviderTimeoutError("too slow", timeout_seconds=30.0)
    refused = ProviderTransportError(
        "no", status_code=401, provider_error="authentication_error"
    )

    assert rate_limited.retry_after_seconds == 7.5
    assert timed_out.timeout_seconds == 30.0
    assert refused.status_code == 401
    assert refused.provider_error == "authentication_error"


def test_running_past_the_script_is_an_invalid_response_not_a_crash() -> None:
    provider = FakeModelProvider(["only one"])
    provider.invoke(_request())

    with pytest.raises(InvalidModelResponseError, match="scripted for 1"):
        provider.invoke(_request())


# -- usage metadata -----------------------------------------------------------


def test_usage_metadata_is_preserved_across_the_boundary() -> None:
    provider = FakeModelProvider(
        [
            ScriptedReply(
                text="one two three four",
                request_id="req_abc123",
                nominal_cost=ResourceCost(usd=0.004, model_tokens=42),
            )
        ]
    )

    response = provider.invoke(
        _request(instruction="two words", messages=(
            Message(role=MessageRole.USER, content="three little words"),
        ))
    )

    assert response.usage.calls == 1
    assert response.usage.input_tokens == 5  # 2 instruction + 3 message
    assert response.usage.output_tokens == 4
    assert response.usage.model == "test-model-1"
    assert response.request_id == "req_abc123"
    assert response.nominal_cost is not None
    assert response.nominal_cost.usd == 0.004
    assert response.nominal_cost.model_tokens == 42


def test_the_usage_ledger_feeds_the_existing_metrics_seam() -> None:
    """Usage reaches StepMetrics the way provider usage already does: the
    ledger accumulates, the loop drains once per step."""
    provider = FakeModelProvider(["one two", "three four five"])
    ledger = UsageLedger()

    for _ in range(2):
        ledger.record(provider.invoke(_request()))

    drained = ledger.drain()
    assert drained.calls == 2
    assert drained.output_tokens == 5
    assert ledger.drain() == ProviderUsage()  # draining is not idempotent


def test_unknown_cost_is_distinct_from_known_zero_cost() -> None:
    """``None`` means the adapter does not know the price. A zero
    ``ResourceCost`` already means known-free everywhere in core, so it
    cannot double as "unknown"."""
    unknown = FakeModelProvider(["ok"]).invoke(_request())
    assert unknown.nominal_cost is None
    assert unknown.usage.input_tokens > 0  # tokens stay the ground truth

    priced_free = FakeModelProvider(
        [ScriptedReply(text="ok", nominal_cost=ResourceCost())]
    ).invoke(_request())
    assert priced_free.nominal_cost is not None
    assert priced_free.nominal_cost.is_zero
    assert priced_free.nominal_cost != unknown.nominal_cost


# -- the deterministic fake ---------------------------------------------------


def test_the_fake_provider_is_deterministic() -> None:
    """Same script, same request, byte-identical results — no clock, no
    network, no randomness anywhere in a unit test."""
    script = [json.dumps({"statement": "Biased.", "confidence": 0.5})]
    request = _request(schema=HYPOTHESIS_SCHEMA)

    first = FakeModelProvider(script).invoke(request)
    second = FakeModelProvider(script).invoke(request)

    assert first.text == second.text
    assert first.structured == second.structured
    assert first.usage == second.usage
    assert first.latency_seconds == second.latency_seconds
    assert first.request_fingerprint == second.request_fingerprint
    # Only the occurrence id differs: two calls are two events.
    assert first.id != second.id


def test_the_fake_provider_records_what_it_was_asked() -> None:
    provider = FakeModelProvider(["a", "b"])
    provider.invoke(_request(messages=(
        Message(role=MessageRole.USER, content="first question"),
    )))
    provider.invoke(_request(messages=(
        Message(role=MessageRole.USER, content="second question"),
    )))

    assert len(provider.calls) == 2
    assert provider.calls[0].messages[0].content == "first question"
    assert provider.calls[1].messages[0].content == "second question"


def test_the_fake_provider_satisfies_the_interface() -> None:
    assert isinstance(FakeModelProvider([]), ModelProvider)


def test_identical_requests_share_a_fingerprint_and_differing_ones_do_not() -> None:
    assert _request().fingerprint == _request().fingerprint
    assert _request().fingerprint != _request(model="other-model").fingerprint
    assert (
        _request().fingerprint
        != _request(schema=HYPOTHESIS_SCHEMA).fingerprint
    )


def test_a_different_schema_body_changes_the_fingerprint() -> None:
    """Two schemas may share a name; the contract is the body. A fingerprint
    that stopped at the name would call two different contracts the same
    request."""
    loose = OutputSchema(
        name="hypothesis_v1",
        json_schema={
            "type": "object",
            "properties": {"statement": {"type": "string"}},
            "required": ["statement"],
        },
    )
    strict = OutputSchema(
        name="hypothesis_v1",
        json_schema={
            "type": "object",
            "properties": {"statement": {"type": "string"}},
            "required": ["statement"],
            "additionalProperties": True,
        },
    )

    assert loose.name == strict.name
    assert (
        _request(schema=loose).fingerprint
        != _request(schema=strict).fingerprint
    )


def test_the_timeout_changes_the_fingerprint() -> None:
    """The deadline shapes what can be generated; it is invocation content."""
    assert (
        _request(timeout_seconds=120.0).fingerprint
        != _request(timeout_seconds=30.0).fingerprint
    )


# -- request validation -------------------------------------------------------


def test_a_request_must_name_a_model_and_carry_a_message() -> None:
    with pytest.raises(ValueError, match="must name a model"):
        ModelRequest(
            model="  ",
            messages=(Message(role=MessageRole.USER, content="hi"),),
        )
    with pytest.raises(ValueError, match="at least one message"):
        ModelRequest(model="test-model-1")
    with pytest.raises(ValueError, match="timeout_seconds"):
        ModelRequest(
            model="test-model-1",
            messages=(Message(role=MessageRole.USER, content="hi"),),
            timeout_seconds=0.0,
        )
    with pytest.raises(ValueError, match="content must be non-empty"):
        Message(role=MessageRole.USER, content="   ")


def test_a_request_has_a_finite_default_timeout() -> None:
    assert _request().timeout_seconds == 120.0
