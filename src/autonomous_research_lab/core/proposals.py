"""Proposals: how roles change the world without touching the state.

Architectural invariant: **no role mutates ResearchState.** A role reads state
and produces a proposal — a typed, attributable request to change it. The
transition layer (:mod:`autonomous_research_lab.orchestration.transitions`)
validates the proposal against the current state and evidence store and commits
it, producing the successor state.

::

    role -> proposal -> validation / transition layer -> ResearchState'

This buys provenance (every state change names its proposer), auditability
(invalid proposals are rejected with reasons, not silently absorbed), safe
search branching (the same proposal can be committed onto different branches),
and a single place to put conflict resolution when multiple agents propose
concurrently.

Every proposal carries ``proposer`` — the role, executor, or method that
produced it. The payloads are ordinary core objects; a proposal adds only
attribution and intent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .assessment import EpistemicAssessment
from .claim import Claim, EvidenceLink
from .evidence import Evidence
from .experiment import ExperimentResult, ExperimentSpec
from .hypothesis import Hypothesis
from .prediction import Prediction


@dataclass(frozen=True, slots=True)
class HypothesisProposal:
    hypothesis: Hypothesis
    proposer: str
    motivation: str = ""


@dataclass(frozen=True, slots=True)
class PredictionProposal:
    prediction: Prediction
    proposer: str
    motivation: str = ""


@dataclass(frozen=True, slots=True)
class ExperimentProposal:
    spec: ExperimentSpec
    proposer: str
    motivation: str = ""


@dataclass(frozen=True, slots=True)
class ResultProposal:
    """An executor reporting a finished run. The one proposal whose payload no
    role can construct: results come from processes, and the transition layer
    records them in the evidence store as facts."""

    result: ExperimentResult
    proposer: str


@dataclass(frozen=True, slots=True)
class EvidenceProposal:
    """A factual reading of a recorded result. Committing it also triggers the
    mechanical check of the prediction the result's experiment tests."""

    evidence: Evidence
    proposer: str


@dataclass(frozen=True, slots=True)
class ClaimProposal:
    claim: Claim
    links: tuple[EvidenceLink, ...] = ()
    proposer: str = ""


@dataclass(frozen=True, slots=True)
class AssessmentProposal:
    assessment: EpistemicAssessment
    proposer: str


Proposal = (
    HypothesisProposal
    | PredictionProposal
    | ExperimentProposal
    | ResultProposal
    | EvidenceProposal
    | ClaimProposal
    | AssessmentProposal
)
