"""The admission budget preflight: refuse an incoherent run before any
call.

One gated stage, so the arithmetic is small — but it is the same
contract the prior-art and selection preflights carry: a directive that
cannot finish the work its own settings allow is refused with every
violation named, before any model call, rather than failing closed
midway with spend on the ledger. The reply bound is real only because
the gate caps encodings per prediction: an unbounded array would make
any worst-case token estimate a fiction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .directive import AdmissionDirective
from .records import MAX_ENCODINGS_PER_PREDICTION

STAGE_BASE_OUTPUT_TOKENS: Final = 500
"""Fixed reply overhead: the JSON scaffolding around the encodings."""

TOKENS_PER_ENCODING: Final = 450
"""One encoding's worst case: six short prose fields plus a few support
links, rounded up from the 5E per-candidate observations."""

GATED_STAGES: Final = 1
"""Admission makes exactly one gated call; there is no second stage."""


class AdmissionPreflightError(RuntimeError):
    """The directive cannot complete the work its own settings allow.
    Raised before any model call, naming every violation at once."""


@dataclass(frozen=True, slots=True)
class AdmissionCallPlan:
    """What a coherent run may spend, computed before it starts."""

    prediction_count: int
    max_encodings: int
    worst_case_calls: int
    worst_case_output_tokens: int


def check_admission_coherence(
    *,
    directive: AdmissionDirective,
    prediction_count: int,
    max_output_tokens: int,
    max_corrective_calls: int,
) -> AdmissionCallPlan:
    """Refuse a run that could not finish. Violations are collected, not
    short-circuited: the operator sees every problem at once."""
    max_encodings = prediction_count * MAX_ENCODINGS_PER_PREDICTION
    plan = AdmissionCallPlan(
        prediction_count=prediction_count,
        max_encodings=max_encodings,
        worst_case_calls=GATED_STAGES * (1 + max_corrective_calls),
        worst_case_output_tokens=(
            STAGE_BASE_OUTPUT_TOKENS + max_encodings * TOKENS_PER_ENCODING
        ),
    )
    violations: list[str] = []
    if prediction_count < 1:
        violations.append(
            "the selected candidate records no predictions; there is "
            "nothing to operationalize and nothing admissible"
        )
    if plan.worst_case_calls > directive.max_model_calls:
        violations.append(
            f"the run may need {plan.worst_case_calls} calls (one gated "
            f"stage with {max_corrective_calls} corrective call(s)) but "
            f"the directive allows {directive.max_model_calls}"
        )
    if plan.worst_case_output_tokens > max_output_tokens:
        violations.append(
            f"a full reply may need {plan.worst_case_output_tokens} "
            f"output tokens ({prediction_count} prediction(s) at up to "
            f"{MAX_ENCODINGS_PER_PREDICTION} encodings each) but the "
            f"reply window is {max_output_tokens}"
        )
    if violations:
        raise AdmissionPreflightError(
            "the admission directive cannot complete the work its own "
            "settings allow: " + "; ".join(violations)
        )
    return plan
