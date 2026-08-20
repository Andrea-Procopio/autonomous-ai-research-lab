"""Content-addressed persistence of commit bundles.

Layout, under a program root::

    <root>/
    └── bundles/
        └── <bundle_id>.json     one attempt's whole effect, before it lands

A :class:`~..core.commit.CommitBundle` is everything one attempt asks the
state to accept, atomically. Until now it existed only in memory, which
made the last few instructions of a step the most expensive thing in the
system to lose: the experiment had run and been paid for, the result was
in the store, and the only thing standing between that and a committed
successor was a Python object in a process that no longer exists.

Writing the bundle down first turns those instructions into a replay.
The bundle is content-addressed, so applying it again reaches the same
successor with the same id, and a recovering process can finish a step
it never started — without the runtime, without a model, and without
paying for anything twice.

**Results and evidence are stored by reference.** A bundle is written
only after its outputs are durable, so the payloads are already in the
evidence store and copying them here would be a second copy that could
disagree with the first. Loading a bundle therefore needs a
:class:`FactSource` to resolve them, and a bundle whose facts have gone
missing fails loudly instead of reconstructing a smaller bundle than the
one that was written.

Everything else — questions, hypotheses, predictions, specs, claims,
links, assessments — is written in full, because at the moment the
bundle is stored those objects exist nowhere else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol

from ..core.assessment import EpistemicAssessment
from ..core.attempt import ActionOutcome
from ..core.claim import Claim, EvidenceLink
from ..core.commit import CommitBundle
from ..core.evidence import Evidence
from ..core.experiment import ExperimentResult, ExperimentSpec
from ..core.hypothesis import Hypothesis
from ..core.ids import content_id
from ..core.prediction import Prediction
from ..core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    EvidenceProposal,
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    Proposal,
    ProposalKind,
    QuestionProposal,
    ResultProposal,
    kind_of,
)
from ..core.question import ResearchQuestion
from ..core.serialize import to_jsonable
from .state_store import (
    read_assessment,
    read_claim,
    read_hypothesis,
    read_link,
    read_outcome,
    read_prediction,
    read_question,
    read_spec,
)

_BUNDLES: Final = "bundles"
_SUFFIX: Final = ".json"


class BundleError(RuntimeError):
    """A stored bundle is missing, malformed, or no longer hashes to its
    own name."""


class BundleConflictError(RuntimeError):
    """One bundle id would be rewritten with different content."""


class FactSource(Protocol):
    """The little of an evidence store a bundle needs to rehydrate.

    A protocol rather than an import so that persistence keeps depending
    on the scientific vocabulary alone.
    """

    def get_result(self, result_id: str) -> ExperimentResult: ...

    def get_evidence(self, evidence_id: str) -> Evidence: ...


def bundle_id_of(bundle: CommitBundle) -> str:
    """The content id of ``bundle`` — the same in every process, which is
    what makes re-applying one idempotent."""
    return content_id("bun", _bundle_payload(bundle))


class CommitBundleStore:
    """Write-once storage for commit bundles under one root."""

    def __init__(self, root: Path | str) -> None:
        self._directory = Path(root) / _BUNDLES
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._directory

    def record(self, bundle: CommitBundle) -> str:
        """Store ``bundle`` and return its id. Storing the same bundle
        twice writes once."""
        payload = _bundle_payload(bundle)
        bundle_id = content_id("bun", payload)
        path = self._path(bundle_id)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise BundleConflictError(
                    f"bundle {bundle_id} is already stored with different "
                    f"content; bundles are never rewritten"
                )
            return bundle_id
        scratch = path.with_suffix(".tmp")
        scratch.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        scratch.replace(path)
        return bundle_id

    def has(self, bundle_id: str) -> bool:
        return self._path(bundle_id).is_file()

    def bundle_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(path.stem for path in self._directory.glob(f"*{_SUFFIX}"))
        )

    def load(self, bundle_id: str, *, facts: FactSource) -> CommitBundle:
        """Rebuild the bundle filed under ``bundle_id``.

        The id is recomputed from what was read, so a bundle edited after
        it was written fails here rather than committing something nobody
        approved.
        """
        path = self._path(bundle_id)
        if not path.is_file():
            raise BundleError(f"no bundle is stored under {bundle_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BundleError(
                f"bundle {bundle_id} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise BundleError(f"bundle {bundle_id} is not an object")
        try:
            bundle = _bundle_from(payload, facts)
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleError(
                f"bundle {bundle_id} cannot be read: {exc}"
            ) from exc
        rederived = content_id("bun", _bundle_payload(bundle))
        if rederived != bundle_id:
            raise BundleError(
                f"bundle filed under {bundle_id} re-derives {rederived}; "
                f"the file was edited after it was written"
            )
        return bundle

    def _path(self, bundle_id: str) -> Path:
        return self._directory / f"{bundle_id}{_SUFFIX}"


# -- payloads ------------------------------------------------------------------


def _bundle_payload(bundle: CommitBundle) -> dict[str, object]:
    return {
        "attempt_id": bundle.attempt_id,
        "outcome": to_jsonable(bundle.outcome),
        "proposals": [_proposal_payload(p) for p in bundle.proposals],
    }


def _proposal_payload(proposal: Proposal) -> dict[str, object]:
    kind = kind_of(proposal)
    body: dict[str, object] = {"kind": str(kind)}
    match proposal:
        case QuestionProposal():
            body["question"] = to_jsonable(proposal.question)
            body["motivation"] = proposal.motivation
        case HypothesisProposal():
            body["hypothesis"] = to_jsonable(proposal.hypothesis)
            body["motivation"] = proposal.motivation
        case PredictionProposal():
            body["prediction"] = to_jsonable(proposal.prediction)
            body["motivation"] = proposal.motivation
        case ExperimentProposal():
            body["spec"] = to_jsonable(proposal.spec)
            body["motivation"] = proposal.motivation
        case ResultProposal():
            # By reference: the result is already in the evidence store,
            # and a second copy is a second thing to keep in agreement.
            body["result_id"] = proposal.result.id
        case EvidenceProposal():
            body["evidence_id"] = proposal.evidence.id
        case ClaimProposal():
            body["claim"] = to_jsonable(proposal.claim)
            body["links"] = [to_jsonable(link) for link in proposal.links]
        case AssessmentProposal():
            body["assessment"] = to_jsonable(proposal.assessment)
    body["proposer"] = proposal.proposer
    return body


def _bundle_from(
    payload: Mapping[str, object], facts: FactSource
) -> CommitBundle:
    proposals = payload["proposals"]
    if not isinstance(proposals, list):
        raise TypeError("proposals must be a list")
    return CommitBundle(
        attempt_id=_text(payload, "attempt_id"),
        outcome=_outcome_from(payload),
        proposals=tuple(
            _proposal_from(_object(entry), facts) for entry in proposals
        ),
    )


def _outcome_from(payload: Mapping[str, object]) -> ActionOutcome:
    return read_outcome(_object(payload["outcome"]))


def _proposal_from(
    body: Mapping[str, object], facts: FactSource
) -> Proposal:
    kind = ProposalKind(_text(body, "kind"))
    proposer = _text(body, "proposer")
    motivation = str(body.get("motivation", ""))
    match kind:
        case ProposalKind.QUESTION:
            return QuestionProposal(
                question=_question(body), proposer=proposer, motivation=motivation
            )
        case ProposalKind.HYPOTHESIS:
            return HypothesisProposal(
                hypothesis=_hypothesis(body),
                proposer=proposer,
                motivation=motivation,
            )
        case ProposalKind.PREDICTION:
            return PredictionProposal(
                prediction=_prediction(body),
                proposer=proposer,
                motivation=motivation,
            )
        case ProposalKind.EXPERIMENT:
            return ExperimentProposal(
                spec=_spec(body), proposer=proposer, motivation=motivation
            )
        case ProposalKind.RESULT:
            return ResultProposal(
                result=facts.get_result(_text(body, "result_id")),
                proposer=proposer,
            )
        case ProposalKind.EVIDENCE:
            return EvidenceProposal(
                evidence=facts.get_evidence(_text(body, "evidence_id")),
                proposer=proposer,
            )
        case ProposalKind.CLAIM:
            return ClaimProposal(
                claim=_claim(body), links=_links(body), proposer=proposer
            )
        case ProposalKind.ASSESSMENT:
            return AssessmentProposal(
                assessment=_assessment(body), proposer=proposer
            )
    raise TypeError(f"{kind} is not a proposal a bundle can carry")


def _question(body: Mapping[str, object]) -> ResearchQuestion:
    return read_question(_object(body["question"]))


def _hypothesis(body: Mapping[str, object]) -> Hypothesis:
    return read_hypothesis(_object(body["hypothesis"]))


def _prediction(body: Mapping[str, object]) -> Prediction:
    return read_prediction(_object(body["prediction"]))


def _spec(body: Mapping[str, object]) -> ExperimentSpec:
    return read_spec(_object(body["spec"]))


def _claim(body: Mapping[str, object]) -> Claim:
    return read_claim(_object(body["claim"]))


def _assessment(body: Mapping[str, object]) -> EpistemicAssessment:
    return read_assessment(_object(body["assessment"]))


def _links(body: Mapping[str, object]) -> tuple[EvidenceLink, ...]:
    raw = body.get("links", [])
    if not isinstance(raw, list):
        raise TypeError("links must be a list")
    return tuple(read_link(_object(entry)) for entry in raw)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected an object")
    return {str(key): item for key, item in value.items()}


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value
