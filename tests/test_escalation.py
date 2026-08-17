"""The escalation policy chooses the cheapest sufficient reasoning tier."""

from __future__ import annotations

from autonomous_research_lab.core.budget import ResourceCost
from autonomous_research_lab.runtime.escalation import (
    COMPACT_METHOD,
    CompactValuation,
    EscalationPolicy,
    EscalationSignals,
    Level,
    ReasoningTier,
    as_action_utility,
)

POLICY = EscalationPolicy()


def test_tiers_are_ordered_by_cost() -> None:
    assert (
        ReasoningTier.DETERMINISTIC
        < ReasoningTier.ROUTINE
        < ReasoningTier.STRONG
        < ReasoningTier.DELIBERATE
    )


def test_mechanically_checkable_questions_cost_no_model_call() -> None:
    signals = EscalationSignals(
        importance=Level.HIGH,  # even importance does not buy a model call
        mechanically_checkable=True,
    )
    assert POLICY.tier_for(signals) is ReasoningTier.DETERMINISTIC


def test_routine_decisions_stay_on_the_cheap_tier() -> None:
    # An obvious fix: unimportant, unambiguous. No strong model.
    signals = EscalationSignals(
        importance=Level.LOW, uncertainty=Level.LOW
    )
    assert POLICY.tier_for(signals) is ReasoningTier.ROUTINE
    assert POLICY.tier_for(EscalationSignals()) is ReasoningTier.ROUTINE


def test_expensive_downstream_commitments_escalate() -> None:
    # Choosing between two $500 experiment branches deserves a strong model.
    signals = EscalationSignals(downstream_cost=ResourceCost(usd=500.0))
    assert POLICY.tier_for(signals) is ReasoningTier.STRONG


def test_high_importance_or_uncertainty_escalates() -> None:
    assert (
        POLICY.tier_for(EscalationSignals(importance=Level.HIGH))
        is ReasoningTier.STRONG
    )
    assert (
        POLICY.tier_for(EscalationSignals(uncertainty=Level.HIGH))
        is ReasoningTier.STRONG
    )


def test_conflicting_evidence_around_a_central_result_deliberates() -> None:
    signals = EscalationSignals(
        importance=Level.HIGH, conflicting_evidence=True
    )
    assert POLICY.tier_for(signals) is ReasoningTier.DELIBERATE
    # Conflict alone, on an unimportant matter, does not justify multi-sample.
    assert (
        POLICY.tier_for(EscalationSignals(conflicting_evidence=True))
        is ReasoningTier.ROUTINE
    )


def test_compact_valuation_embeds_without_fake_precision() -> None:
    utility = as_action_utility(
        CompactValuation(
            scientific_value=Level.HIGH,
            expected_cost=ResourceCost(usd=1.0),
            uncertainty=Level.HIGH,
            rationale="worth running",
        )
    )
    assert utility.method == COMPACT_METHOD
    assert utility.expected_information_gain == 1.0
    assert utility.estimate_uncertainty == 1.0
    assert utility.expected_cost.usd == 1.0
    # Dimensions the compact form does not estimate stay None — never zero.
    assert utility.novelty is None
    assert utility.importance is None
    assert utility.discrimination_value is None
