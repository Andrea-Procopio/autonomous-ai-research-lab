"""Budget coherence for a selection run, checked before any model call.

The Task 5D.2 discipline applied here: a directive that permits work its
own settings cannot finish is refused up front, with every violation
named — never discovered mid-run as truncation, and never reported
afterwards as a scientific deficiency. The check runs after the door and
the trusted partition, and after the zero-eligible short-circuit: a
portfolio with nothing eligible makes zero calls, and budget incoherence
about calls that will never happen must not deny the durable record of
an honest deterministic outcome.

The arithmetic is worst-case and static. Stage 1 is one reply covering
every eligible candidate and every pair; stage 2 covers the winner and
one entry per alternative. Both must fit the output envelope the run
actually wires (16384 tokens, the Task 5C lesson), and both stages with
their corrective calls must fit the directive's call budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .directive import SelectionDirective

STAGE1_BASE_OUTPUT_TOKENS: Final = 400
"""JSON envelope, array brackets, and key overhead of an empty reply."""

STAGE1_TOKENS_PER_CANDIDATE: Final = 1800
"""Ten review fields at the instructed one-to-two short sentences plus
JSON key overhead (~850 tokens), the verdict enum, and the structural
worst case of five disqualifiers — one per ground, duplicates rejected —
each carrying two bounded verbatim quotes and a rationale (~800 tokens).
Rounded up from ~1700."""

STAGE1_TOKENS_PER_PAIR: Final = 250
"""Two ids, keys, and a two-sentence comparison: about twice what the
Task 5D comparison calls actually produced per dimension entry."""

STAGE2_BASE_OUTPUT_TOKENS: Final = 700
"""The winner's id, the decisive tradeoff, the first objective, and the
capability and risk lists."""

STAGE2_TOKENS_PER_ALTERNATIVE: Final = 250
"""One why-selected-over entry: an id, keys, two short sentences."""

GATED_STAGES: Final = 2
"""The comparative review and the final decision. There is no third."""


class SelectionPreflightError(RuntimeError):
    """The directive cannot complete the work it permits. Raised before
    any model call, naming every violation at once."""


@dataclass(frozen=True, slots=True)
class SelectionCallPlan:
    """The worst case a coherent directive reserves. Diagnostic output
    of the preflight, recorded nowhere."""

    eligible: int
    pairs: int
    worst_stage1_output_tokens: int
    worst_stage2_output_tokens: int
    output_token_envelope: int
    worst_calls_total: int


def check_selection_coherence(
    *,
    directive: SelectionDirective,
    eligible_count: int,
    max_output_tokens: int,
    max_corrective_calls: int,
) -> SelectionCallPlan:
    """Refuse a selection run that cannot finish the work its own
    settings allow. Every violation is collected; the refusal names them
    all."""
    pairs = eligible_count * (eligible_count - 1) // 2
    worst_stage1 = (
        STAGE1_BASE_OUTPUT_TOKENS
        + eligible_count * STAGE1_TOKENS_PER_CANDIDATE
        + pairs * STAGE1_TOKENS_PER_PAIR
    )
    worst_stage2 = STAGE2_BASE_OUTPUT_TOKENS + (
        max(eligible_count - 1, 0) * STAGE2_TOKENS_PER_ALTERNATIVE
    )
    worst_calls = GATED_STAGES * (1 + max_corrective_calls)
    plan = SelectionCallPlan(
        eligible=eligible_count,
        pairs=pairs,
        worst_stage1_output_tokens=worst_stage1,
        worst_stage2_output_tokens=worst_stage2,
        output_token_envelope=max_output_tokens,
        worst_calls_total=worst_calls,
    )

    violations: list[str] = []
    if eligible_count > directive.max_eligible_candidates:
        violations.append(
            f"{eligible_count} eligible candidates exceed the directive's "
            f"cap of {directive.max_eligible_candidates}; the comparative "
            f"review judges every pair in one reply"
        )
    if worst_stage1 > max_output_tokens:
        violations.append(
            f"the worst-case comparative review needs "
            f"{worst_stage1} output tokens ({eligible_count} candidates "
            f"at {STAGE1_TOKENS_PER_CANDIDATE} plus {pairs} pairs at "
            f"{STAGE1_TOKENS_PER_PAIR} plus {STAGE1_BASE_OUTPUT_TOKENS} "
            f"base) against an envelope of {max_output_tokens}; a "
            f"truncated reply is a lost call, not a result"
        )
    if worst_stage2 > max_output_tokens:
        violations.append(
            f"the worst-case decision needs {worst_stage2} output tokens "
            f"against an envelope of {max_output_tokens}"
        )
    if worst_calls > directive.max_model_calls:
        violations.append(
            f"worst-case calls {worst_calls} ({GATED_STAGES} gated stages "
            f"with {max_corrective_calls} corrective call"
            f"{'s' if max_corrective_calls != 1 else ''} each) exceed "
            f"max_model_calls {directive.max_model_calls}"
        )
    if violations:
        raise SelectionPreflightError(
            "the directive cannot complete the work it permits: "
            + "; ".join(violations)
        )
    return plan
