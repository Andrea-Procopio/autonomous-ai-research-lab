"""The claim-evidence graph is factual; assessment is separate.

The tests pin both halves: the graph reports what evidence bears on a claim
and how, and nothing anywhere converts that structure into a verdict — the
verdict comes from an EpistemicAssessment with a named method.
"""

from __future__ import annotations

from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.claim import (
    Claim,
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


def linked(
    state: ResearchState, claim: Claim, evidence: Evidence, relation: EvidenceRelation
) -> ResearchState:
    return state.upsert_claim(claim).link_evidence(
        EvidenceLink(claim_id=claim.id, evidence_id=evidence.id, relation=relation)
    )


def test_a_claim_with_no_evidence_is_surfaced() -> None:
    """The failure mode this guards against: a fluent system emitting
    confident claims that nothing measured ever touched."""
    state, store, _ = build()
    claim = Claim(statement="The effect is real.")
    state = state.upsert_claim(claim)

    graph = ClaimEvidenceGraph.from_state(state, store)
    entry = graph.evidence_for(claim.id)

    assert not entry.has_evidence
    assert graph.without_evidence() == (claim,)


def test_the_graph_reports_relations_without_judging_them() -> None:
    state, store, evidence = build()
    claim = Claim(statement="The effect is real.")
    state = linked(state, claim, evidence, EvidenceRelation.SUPPORTS)

    entry = ClaimEvidenceGraph.from_state(state, store).evidence_for(claim.id)
    assert entry.supporting == (evidence,)
    assert entry.contradicting == ()
    # No verdict anywhere on the graph's output: ClaimEvidence carries
    # evidence tuples and nothing status-shaped.
    assert not hasattr(entry, "suggested_status")


def test_contradicted_claims_are_found() -> None:
    state, store, evidence = build()
    claim = Claim(statement="The effect is real.")
    state = linked(state, claim, evidence, EvidenceRelation.CONTRADICTS)

    graph = ClaimEvidenceGraph.from_state(state, store)
    assert [c.claim.id for c in graph.contradicted()] == [claim.id]


def test_inconclusive_evidence_is_kept_distinct() -> None:
    """Underpowered or confounded evidence is neither support nor
    contradiction, and flattening it into either would misstate the record."""
    state, store, evidence = build()
    claim = Claim(statement="The effect is real.")
    state = linked(state, claim, evidence, EvidenceRelation.INCONCLUSIVE)

    entry = ClaimEvidenceGraph.from_state(state, store).evidence_for(claim.id)
    assert entry.inconclusive == (evidence,)
    assert entry.has_evidence


def test_evidence_alone_settles_nothing() -> None:
    """A claim with contradicting evidence is *unassessed*, not refuted: no
    count of edges produces a verdict. Only an assessment changes standing,
    and it names its method."""
    state, store, evidence = build()
    claim = Claim(statement="The effect is real.")
    state = linked(state, claim, evidence, EvidenceRelation.CONTRADICTS)

    graph = ClaimEvidenceGraph.from_state(state, store)
    assert graph.unassessed() == (claim,)
    assert state.current_assessment(claim.id) is None

    assessment = EpistemicAssessment(
        subject_id=claim.id,
        verdict=AssessmentVerdict.CONTESTED,
        method="test:judgment-v0",
        evidence_ids=(evidence.id,),
        rationale="One contradicting reading; power unassessed.",
    )
    state = state.record_assessment(assessment)

    current = state.current_assessment(claim.id)
    assert current is not None
    assert current.verdict is AssessmentVerdict.CONTESTED
    assert ClaimEvidenceGraph.from_state(state, store).unassessed() == ()
