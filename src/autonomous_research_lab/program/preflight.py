"""The funding preflight: refuse an incoherent run before anything is
written.

The same contract the prior-art, selection, and admission preflights
carry, one stage down and with no model calls to protect: a directive
whose own settings cannot buy the work it promises is refused with every
violation named, before the grant reaches the ledger, rather than
halting on the first step with an envelope already on disk.

What "cannot buy the work" means here is deliberately modest. This
package does not know what an experiment costs — that is the planner's
catalog and, later, the campaign's estimate. It knows only what is
incoherent on its face: a grant that buys nothing at all, or one whose
resources cannot cover the floor the caller says a first step needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.budget import NO_COST, ResourceCost
from .authorization import FundingAuthorization
from .directive import RunDirective


class RunPreflightError(RuntimeError):
    """The run cannot complete the work its own settings allow. Raised
    before the grant, naming every violation at once."""


@dataclass(frozen=True, slots=True)
class RunPlan:
    """What a coherent run may spend, computed before it starts."""

    directive_id: str
    authorization_id: str
    minimum_first_step: ResourceCost


def check_funding_coherence(
    *,
    directive: RunDirective,
    authorization: FundingAuthorization,
    minimum_first_step: ResourceCost = NO_COST,
) -> RunPlan:
    """Refuse a run that could not take a step. Violations are collected,
    not short-circuited: the operator sees every problem at once.

    ``minimum_first_step`` is the caller's floor — the cheapest action it
    intends to dispatch. Left at zero, the check reduces to "this grant
    buys something", which is the honest bound when the caller has not
    priced anything yet.
    """
    plan = RunPlan(
        directive_id=directive.id,
        authorization_id=authorization.id,
        minimum_first_step=minimum_first_step,
    )
    violations: list[str] = []
    if authorization.granted.is_exhausted:
        violations.append(
            "the authorized grant is empty; a run funded with nothing "
            "cannot take a step, and would halt on its first budget check"
        )
    elif not authorization.granted.can_afford(minimum_first_step):
        violations.append(
            f"the authorized grant {authorization.granted} cannot cover "
            f"the caller's cheapest first step {minimum_first_step}"
        )
    if violations:
        raise RunPreflightError(
            "the run directive cannot complete the work its own settings "
            "allow: " + "; ".join(violations)
        )
    return plan
