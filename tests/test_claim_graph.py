from __future__ import annotations

from autonomous_research_lab.core.claim import (
    Claim,
    ClaimStatus,
    EvidenceLink,
    EvidenceRelation,
)
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentStatus,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.knowledge.graph import ClaimEvidenceGraph


def build() -> tuple[ResearchState, InMemoryEvidenceStore, Evidence]:
    store = InMemoryEvidenceStore()
    result = store.record_result(
        ExperimentResult(
            spec_id="exp_1",
            job_id="job_1",
            status=ExperimentStatus.COMPLETED,
            command=("python", "run.py"),
            environment=Environment(python_version="3.11.0", platform="test"),
            metrics={"effect": 0.3},
        )
    )
    evidence = store.record_evidence(
        Evidence(
            result_id=result.id,
            spec_id="exp_1",
            kind=EvidenceKind.MEASUREMENT,
            observation="effect was 0.3",
            metrics={"effect": 0.3},
        )
    )
    return ResearchState(objective="o"), store, evidence


def test_a_claim_with_no_evidence_reads_as_weakly_supported() -> None:
    """The failure mode this guards against: a fluent system emitting confident
    claims that nothing measured ever touched."""
    state, store, _ = build()
    claim = Claim(statement="The effect is real.")
    state = state.upsert_claim(claim)

    graph = ClaimEvidenceGraph.from_state(state, store)
    support = graph.support_for(claim.id)

    assert support.is_unsupported
    assert support.suggested_status() is ClaimStatus.PROPOSED
    assert [s.claim.id for s in graph.weakly_supported()] == [claim.id]


def test_supporting_evidence_raises_the_claim_out_of_weak_support() -> None:
    state, store, evidence = build()
    claim = Claim(statement="The effect is real.")
    state = state.upsert_claim(claim).link_evidence(
        EvidenceLink(
            claim_id=claim.id,
            evidence_id=evidence.id,
            relation=EvidenceRelation.SUPPORTS,
        )
    )

    graph = ClaimEvidenceGraph.from_state(state, store)
    support = graph.support_for(claim.id)

    assert support.supporting[0].id == evidence.id
    assert support.suggested_status() is ClaimStatus.SUPPORTED
    assert graph.weakly_supported() == ()


def test_conflicting_evidence_makes_a_claim_contested_not_settled() -> None:
    state, store, supporting = build()
    contradicting = store.record_evidence(
        Evidence(
            result_id=supporting.result_id,
            spec_id="exp_1",
            kind=EvidenceKind.NULL_RESULT,
            observation="effect was 0.0 under the stricter control",
        )
    )
    claim = Claim(statement="The effect is real.")
    state = (
        state.upsert_claim(claim)
        .link_evidence(
            EvidenceLink(
                claim_id=claim.id,
                evidence_id=supporting.id,
                relation=EvidenceRelation.SUPPORTS,
            )
        )
        .link_evidence(
            EvidenceLink(
                claim_id=claim.id,
                evidence_id=contradicting.id,
                relation=EvidenceRelation.CONTRADICTS,
            )
        )
    )

    graph = ClaimEvidenceGraph.from_state(state, store)
    support = graph.support_for(claim.id)

    assert support.suggested_status() is ClaimStatus.CONTESTED
    assert [s.claim.id for s in graph.contradicted()] == [claim.id]
    assert support.net_weight == 0.0
