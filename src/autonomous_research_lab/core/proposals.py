"""Proposals: how roles change the world without touching the state.

Architectural invariant: **no role mutates ResearchState.** A role reads a
view of the state and produces a proposal — a typed, attributable request to
change it. The transition layer
(:mod:`autonomous_research_lab.orchestration.transitions`) validates the
proposal against the current state and evidence store and commits it,
producing the successor state.

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

:class:`ProposalKind` names each proposal type as data, so an invocation's
output contract ("this role may return evidence and claims, nothing else")
can be stated and checked without reflection on Python types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .assessment import EpistemicAssessment
from .claim import Claim, EvidenceLink
from .evidence import Evidence
from .experiment import ExperimentResult, ExperimentSpec
from .hypothesis import Hypothesis
from .prediction import Prediction
from .question import ResearchQuestion


@dataclass(frozen=True, slots=True)
class QuestionProposal:
    question: ResearchQuestion
    proposer: str
    motivation: str = ""


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
    """A factual reading of a recorded result."""

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
    QuestionProposal
    | HypothesisProposal
    | PredictionProposal
    | ExperimentProposal
    | ResultProposal
    | EvidenceProposal
    | ClaimProposal
    | AssessmentProposal
)


class ProposalKind(StrEnum):
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    PREDICTION = "prediction"
    EXPERIMENT = "experiment"
    RESULT = "result"
    EVIDENCE = "evidence"
    CLAIM = "claim"
    ASSESSMENT = "assessment"


def kind_of(proposal: Proposal) -> ProposalKind:
    match proposal:
        case QuestionProposal():
            return ProposalKind.QUESTION
        case HypothesisProposal():
            return ProposalKind.HYPOTHESIS
        case PredictionProposal():
            return ProposalKind.PREDICTION
        case ExperimentProposal():
            return ProposalKind.EXPERIMENT
        case ResultProposal():
            return ProposalKind.RESULT
        case EvidenceProposal():
            return ProposalKind.EVIDENCE
        case ClaimProposal():
            return ProposalKind.CLAIM
        case AssessmentProposal():
            return ProposalKind.ASSESSMENT


def payload_ids(proposal: Proposal) -> tuple[str, ...]:
    """The ids of the domain objects this proposal would bring into the state.

    This is what an :class:`~.attempt.ActionOutcome` may legitimately claim as
    ``produced`` for the attempt that generated the proposal.
    """
    match proposal:
        case QuestionProposal():
            return (proposal.question.id,)
        case HypothesisProposal():
            return (proposal.hypothesis.id,)
        case PredictionProposal():
            return (proposal.prediction.id,)
        case ExperimentProposal():
            return (proposal.spec.id,)
        case ResultProposal():
            return (proposal.result.id,)
        case EvidenceProposal():
            return (proposal.evidence.id,)
        case ClaimProposal():
            return (proposal.claim.id, *(link.id for link in proposal.links))
        case AssessmentProposal():
            return (proposal.assessment.id,)
