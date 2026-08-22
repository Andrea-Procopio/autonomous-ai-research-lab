"""Evidence-chain validation: can every belief be walked back to a process?

A derived, deterministic checker over the existing objects — no new state, no
graph database. It traces the chain

::

    ResearchQuestion -> Hypothesis -> Prediction -> ExperimentSpec
    -> ExperimentResult -> PredictionTest / Evidence -> Assessment / Claim

and reports where it is broken:

* references to objects that do not exist (dangling provenance);
* state references to facts absent from the store (missing provenance);
* prediction tests whose recorded observation or verdict disagrees with the
  stored result and the pre-registered check (tampered provenance — the one
  place belief could quietly rewrite fact);
* claims no evidence has ever touched;
* conclusive assessments that cite no evidence at all;
* contradictions on the record, surfaced as facts.

The transition layer already refuses to *commit* most of these. This checker
exists for everything the transition layer never saw: reloaded snapshots,
hand-migrated states, foreign stores, and drift between codec versions. It
answers with issues, never with verdicts — deciding what a broken chain means
is critic business.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..core.assessment import AssessmentVerdict
from ..core.claim import EvidenceRelation
from ..core.prediction import Consistency
from ..core.state import ResearchState
from .store import EvidenceStore, UnknownRecordError

#: The verdicts that require grounds: every one of these leans somewhere,
#: and a lean that cites nothing is exactly the ungrounded judgment this
#: check exists to surface. Only UNDETERMINED may stand on no evidence.
_REQUIRES_GROUNDS = frozenset(
    {
        AssessmentVerdict.PLAUSIBLE,
        AssessmentVerdict.SUPPORTED,
        AssessmentVerdict.REFUTED,
        AssessmentVerdict.CONTESTED,
    }
)


class ChainIssueKind(StrEnum):
    DANGLING_REFERENCE = "dangling_reference"
    """A state object references another state object that is absent."""

    MISSING_FACT = "missing_fact"
    """The state references a result or evidence the store does not hold."""

    TAMPERED_TEST = "tampered_test"
    """A prediction test disagrees with the stored result or with the
    mechanical re-check of the pre-registered prediction."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    """A claim with no evidence links at all."""

    UNGROUNDED_ASSESSMENT = "ungrounded_assessment"
    """A conclusive assessment citing no evidence."""

    CONTRADICTION = "contradiction"
    """Conclusive records point both ways. Not broken provenance — a fact
    that deserves attention."""


@dataclass(frozen=True, slots=True)
class ChainIssue:
    kind: ChainIssueKind
    subject_id: str
    detail: str


def validate_evidence_chain(
    state: ResearchState, store: EvidenceStore
) -> tuple[ChainIssue, ...]:
    """Every chain issue in ``state`` against ``store``, deterministically."""
    issues: list[ChainIssue] = []

    def dangling(subject_id: str, detail: str) -> None:
        issues.append(
            ChainIssue(
                kind=ChainIssueKind.DANGLING_REFERENCE,
                subject_id=subject_id,
                detail=detail,
            )
        )

    for hypothesis in state.hypotheses:
        if (
            hypothesis.question_id is not None
            and state.question(hypothesis.question_id) is None
        ):
            dangling(
                hypothesis.id,
                f"hypothesis names unknown question {hypothesis.question_id}",
            )

    for prediction in state.predictions:
        if state.hypothesis(prediction.hypothesis_id) is None:
            dangling(
                prediction.id,
                f"prediction names unknown hypothesis {prediction.hypothesis_id}",
            )

    for spec in state.experiments:
        if state.prediction(spec.prediction_id) is None:
            dangling(
                spec.id, f"experiment names unknown prediction {spec.prediction_id}"
            )

    known_result_ids = set()
    for ref in state.results:
        if state.experiment(ref.spec_id) is None:
            dangling(
                ref.result_id, f"result reference names unknown spec {ref.spec_id}"
            )
        try:
            store.get_result(ref.result_id)
        except UnknownRecordError:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.MISSING_FACT,
                    subject_id=ref.result_id,
                    detail="state references a result the store does not hold",
                )
            )
        else:
            known_result_ids.add(ref.result_id)

    issues.extend(_check_tests(state, store, known_result_ids))

    known_evidence_ids = set()
    for evidence_id in state.evidence_ids:
        try:
            evidence = store.get_evidence(evidence_id)
        except UnknownRecordError:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.MISSING_FACT,
                    subject_id=evidence_id,
                    detail="state references evidence the store does not hold",
                )
            )
            continue
        known_evidence_ids.add(evidence_id)
        if evidence.result_id not in known_result_ids:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.MISSING_FACT,
                    subject_id=evidence_id,
                    detail=(
                        f"evidence reads result {evidence.result_id}, which "
                        f"this state does not reference"
                    ),
                )
            )

    linked: set[str] = set()
    for link in state.evidence_links:
        linked.add(link.claim_id)
        if state.claim(link.claim_id) is None:
            dangling(link.id, f"link names unknown claim {link.claim_id}")
        if link.evidence_id not in known_evidence_ids:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.MISSING_FACT,
                    subject_id=link.id,
                    detail=f"link names unavailable evidence {link.evidence_id}",
                )
            )

    for claim in state.claims:
        if claim.id not in linked:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.UNSUPPORTED_CLAIM,
                    subject_id=claim.id,
                    detail="no evidence has ever touched this claim",
                )
            )

    for assessment in state.assessments:
        subject_known = (
            state.claim(assessment.subject_id) is not None
            or state.hypothesis(assessment.subject_id) is not None
        )
        if not subject_known:
            dangling(
                assessment.id,
                f"assessment targets unknown subject {assessment.subject_id}",
            )
        for evidence_id in assessment.evidence_ids:
            if evidence_id not in known_evidence_ids:
                issues.append(
                    ChainIssue(
                        kind=ChainIssueKind.MISSING_FACT,
                        subject_id=assessment.id,
                        detail=f"assessment cites unavailable evidence {evidence_id}",
                    )
                )
        if (
            assessment.verdict in _REQUIRES_GROUNDS
            and not assessment.evidence_ids
        ):
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.UNGROUNDED_ASSESSMENT,
                    subject_id=assessment.id,
                    detail=(
                        f"verdict {assessment.verdict} rests on no cited evidence"
                    ),
                )
            )

    issues.extend(_contradictions(state))
    return tuple(issues)


def _check_tests(
    state: ResearchState,
    store: EvidenceStore,
    known_result_ids: set[str],
) -> list[ChainIssue]:
    """Re-derive every prediction test from its inputs. A test is the one
    mechanical record in the chain, so it is fully re-checkable."""
    issues: list[ChainIssue] = []
    for test in state.prediction_tests:
        prediction = state.prediction(test.prediction_id)
        if prediction is None:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.DANGLING_REFERENCE,
                    subject_id=test.id,
                    detail=f"test names unknown prediction {test.prediction_id}",
                )
            )
            continue
        if test.result_id not in known_result_ids:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.DANGLING_REFERENCE,
                    subject_id=test.id,
                    detail=f"test names unavailable result {test.result_id}",
                )
            )
            continue
        result = store.get_result(test.result_id)
        observed = result.metrics.get(prediction.metric)
        if test.observed is not None and test.observed != observed:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.TAMPERED_TEST,
                    subject_id=test.id,
                    detail=(
                        f"test records observed {test.observed}, but the stored "
                        f"result reports {observed} for {prediction.metric!r}"
                    ),
                )
            )
            continue
        if test.observed is not None:
            held = prediction.check(test.observed)
            expected = "consistent" if held else "inconsistent"
            if test.consistency.value != expected:
                issues.append(
                    ChainIssue(
                        kind=ChainIssueKind.TAMPERED_TEST,
                        subject_id=test.id,
                        detail=(
                            f"mechanical re-check says {expected}, test "
                            f"records {test.consistency}"
                        ),
                    )
                )
    return issues


def _contradictions(state: ResearchState) -> list[ChainIssue]:
    issues: list[ChainIssue] = []
    for prediction in state.predictions:
        tests = state.tests_for(prediction.id)
        consistent = any(
            t.consistency is Consistency.CONSISTENT for t in tests
        )
        inconsistent = any(
            t.consistency is Consistency.INCONSISTENT for t in tests
        )
        if consistent and inconsistent:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.CONTRADICTION,
                    subject_id=prediction.id,
                    detail="conclusive tests of this prediction disagree",
                )
            )
    for claim in state.claims:
        relations = {
            link.relation
            for link in state.evidence_links
            if link.claim_id == claim.id
        }
        if {
            EvidenceRelation.SUPPORTS,
            EvidenceRelation.CONTRADICTS,
        } <= relations:
            issues.append(
                ChainIssue(
                    kind=ChainIssueKind.CONTRADICTION,
                    subject_id=claim.id,
                    detail="evidence links both support and contradict this claim",
                )
            )
    return issues
