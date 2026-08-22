"""Exact inference: every verdict reachable, every figure from the record."""

from __future__ import annotations

import pytest

from autonomous_research_lab.core.assessment import AssessmentVerdict
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.runtime.statistics import (
    SignDirection,
    assess_family,
)

PREDICTION = Prediction(
    hypothesis_id="hyp_1",
    condition="held out",
    metric="contrast",
    comparator=Comparator.GREATER_THAN,
    threshold=0.0,
    expectation="positive",
)


def observation(
    observed: float | None,
    consistency: Consistency,
    result_id: str = "res_1",
) -> PredictionTest:
    return PredictionTest(
        prediction_id=PREDICTION.id,
        result_id=result_id,
        metric=PREDICTION.metric,
        observed=observed,
        consistency=consistency,
        detail="test",
    )


def consistent(value: float, result_id: str) -> PredictionTest:
    return observation(value, Consistency.CONSISTENT, result_id)


def contrary(value: float, result_id: str) -> PredictionTest:
    return observation(value, Consistency.INCONSISTENT, result_id)


class TestVerdicts:
    def test_an_empty_family_is_undetermined(self) -> None:
        verdict, stats = assess_family(PREDICTION, ())
        assert verdict is AssessmentVerdict.UNDETERMINED
        assert stats.n == 0
        assert stats.p_value is None
        assert stats.direction is SignDirection.NONE

    def test_inconclusive_tests_do_not_count(self) -> None:
        verdict, stats = assess_family(
            PREDICTION, (observation(None, Consistency.INCONCLUSIVE),)
        )
        assert verdict is AssessmentVerdict.UNDETERMINED
        assert stats.n == 0

    def test_a_mixed_family_is_contested(self) -> None:
        verdict, stats = assess_family(
            PREDICTION,
            (consistent(0.05, "res_1"), contrary(-0.02, "res_2")),
        )
        assert verdict is AssessmentVerdict.CONTESTED
        assert stats.direction is SignDirection.NONE
        assert stats.p_value is None

    def test_five_consistent_reach_supported_at_the_exact_tail(self) -> None:
        tests = tuple(
            consistent(0.04 + i / 100, f"res_{i}") for i in range(5)
        )
        verdict, stats = assess_family(PREDICTION, tests)
        assert verdict is AssessmentVerdict.SUPPORTED
        assert stats.p_value == pytest.approx(1 / 32)
        assert stats.direction is SignDirection.SUPPORTING

    def test_three_consistent_are_only_plausible(self) -> None:
        """The honest small-n outcome: unanimous, and underpowered —
        (1/2)^3 = 0.125 cannot clear 0.05."""
        tests = tuple(consistent(0.05, f"res_{i}") for i in range(3))
        verdict, stats = assess_family(PREDICTION, tests)
        assert verdict is AssessmentVerdict.PLAUSIBLE
        assert stats.p_value == pytest.approx(1 / 8)

    def test_one_observation_is_plausible_never_supported(self) -> None:
        verdict, stats = assess_family(
            PREDICTION, (consistent(0.07, "res_1"),)
        )
        assert verdict is AssessmentVerdict.PLAUSIBLE
        assert stats.p_value == pytest.approx(0.5)
        assert stats.stdev is None  # a spread of one number is nothing

    def test_refuted_is_reachable(self) -> None:
        """The directional tail: an all-contrary family tests the
        contrary direction — the naive always-supporting tail would
        make refutation impossible (P[X >= 0] = 1)."""
        tests = tuple(contrary(-0.02, f"res_{i}") for i in range(5))
        verdict, stats = assess_family(PREDICTION, tests)
        assert verdict is AssessmentVerdict.REFUTED
        assert stats.direction is SignDirection.CONTRARY
        assert stats.p_value == pytest.approx(1 / 32)
        assert stats.consistent_count == 0

    def test_a_short_contrary_family_stays_undetermined(self) -> None:
        tests = tuple(contrary(-0.02, f"res_{i}") for i in range(3))
        verdict, _stats = assess_family(PREDICTION, tests)
        assert verdict is AssessmentVerdict.UNDETERMINED

    def test_bonferroni_raises_the_bar(self) -> None:
        """Five unanimous observations clear alpha alone (p=1/32<0.05)
        and fail it across two comparisons (alpha' = 0.025)."""
        tests = tuple(consistent(0.05, f"res_{i}") for i in range(5))
        alone, _ = assess_family(PREDICTION, tests, comparisons=1)
        across, stats = assess_family(PREDICTION, tests, comparisons=2)
        assert alone is AssessmentVerdict.SUPPORTED
        assert across is AssessmentVerdict.PLAUSIBLE
        assert stats.alpha_adjusted == pytest.approx(0.025)


class TestFigures:
    def test_effect_is_signed_toward_the_prediction(self) -> None:
        lesser = Prediction(
            hypothesis_id="hyp_1",
            condition="c",
            metric="loss",
            comparator=Comparator.LESS_THAN,
            threshold=1.0,
            expectation="small",
        )
        _, stats = assess_family(
            lesser,
            (
                PredictionTest(
                    prediction_id=lesser.id,
                    result_id="res_1",
                    metric="loss",
                    observed=0.4,
                    consistency=Consistency.CONSISTENT,
                    detail="t",
                ),
            ),
        )
        assert stats.effect == pytest.approx(0.6)  # threshold - mean

    def test_approximately_has_no_signed_effect(self) -> None:
        approx = Prediction(
            hypothesis_id="hyp_1",
            condition="c",
            metric="value",
            comparator=Comparator.APPROXIMATELY,
            threshold=1.0,
            tolerance=0.1,
            expectation="near one",
        )
        _, stats = assess_family(
            approx,
            (
                PredictionTest(
                    prediction_id=approx.id,
                    result_id="res_1",
                    metric="value",
                    observed=1.05,
                    consistency=Consistency.CONSISTENT,
                    detail="t",
                ),
            ),
        )
        assert stats.effect is None

    def test_render_states_the_whole_computation(self) -> None:
        tests = tuple(
            consistent(value, f"res_{i}")
            for i, value in enumerate((0.04, 0.05, 0.06, 0.05, 0.045))
        )
        _, stats = assess_family(PREDICTION, tests, comparisons=2)
        line = stats.render()
        for piece in (
            "n=5",
            "consistent 5/5",
            "gt 0.0",
            "one-sided p=0.03125",
            "0.025",
            "2 comparison(s)",
        ):
            assert piece in line

    def test_bad_parameters_refuse(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            assess_family(PREDICTION, (), alpha=0.0)
        with pytest.raises(ValueError, match="comparisons"):
            assess_family(PREDICTION, (), comparisons=0)
