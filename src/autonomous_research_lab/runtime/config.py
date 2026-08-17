"""Runtime configuration: optional machinery is switchable, on purpose.

The eventual research contribution of this project is a measurement of which
architectural components actually help. That requires being able to run the
same research program with a component on and off — so every optional runtime
mechanism hangs off a flag here, and the flags are plain typed fields rather
than a configuration framework.

What is deliberately *not* configurable: deterministic validation of results
(it is Tier 0 and load-bearing — turning it off reintroduces silent success),
and the atomic commit discipline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .escalation import ReasoningTier


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    critic_enabled: bool = True
    """When off, critic triggers are still evaluated and recorded (the data
    matters for the ablation) but no critic invocation happens."""

    playbook_enabled: bool = True
    """When off, the director deliberates without a playbook prior."""

    synthesis_enabled: bool = True
    """The slow synthesis loop. When off, only the fast loop runs."""

    synthesis_every: int = 5
    """Meaningful (committed) experiment results between scheduled synthesis
    reviews. Contradictions and stop decisions trigger synthesis regardless."""

    max_candidates: int = 3
    """The director proposes at most this many candidate actions per
    deliberation. Small on purpose: a long menu is a prompt-size cost with no
    demonstrated selection benefit."""

    recent_results: int = 5
    """How many recent results the frontier projection carries."""

    director_tier_floor: ReasoningTier = ReasoningTier.ROUTINE
    """The cheapest tier the director may run at — the cheap/strong director
    switch. Escalation can raise a deliberation above the floor, never below."""

    repeated_failure_threshold: int = 2
    """Failed or cancelled executions of one experiment before the runtime
    raises a deterministic engineering note for the director. An engineering
    signal, not a critic trigger: debugging is not scientific critique."""

    debug_enabled: bool = True
    """When off, engineering failures are diagnosed and noted but the
    bounded repair loop never runs (it also never runs without a debugger
    wired into the runtime)."""

    max_debug_attempts: int = 3
    """Hard ceiling on repair attempts per failed execution. Debugging
    optimizes for obtaining a *valid* experiment, never a positive result;
    a run that is still failing after this many repairs stops."""

    preflight_enabled: bool = True
    """When off, executor-side wiring skips deterministic pre-execution
    checks. The runtime itself only records preflight rejections."""

    methodology_review_enabled: bool = True
    """When on (and a reviewer is wired in), each experiment design is
    reviewed once before its first execution; a rejected design is never
    run — the response is redesign, not debugging."""

    implementation_verification_enabled: bool = True
    """When on (and a verifier is wired in), implementation faithfulness is
    checked selectively — on failed positive controls and on conclusive
    negatives lacking control coverage — never on every run."""

    positive_controls_enabled: bool = True
    """When off, experiment-specific positive controls are not evaluated
    even if a control source is wired in."""

    def __post_init__(self) -> None:
        if self.synthesis_every < 1:
            raise ValueError("synthesis_every must be at least 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if self.recent_results < 0:
            raise ValueError("recent_results must be non-negative")
        if self.repeated_failure_threshold < 1:
            raise ValueError("repeated_failure_threshold must be at least 1")
        if self.max_debug_attempts < 1:
            raise ValueError("max_debug_attempts must be at least 1")
