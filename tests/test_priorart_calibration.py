"""Verdict reachability under calibrated blockers: every terminal
verdict demonstrated end to end through the real challenger on closed
corpora, with the default thresholds and no test-only exemptions — plus
the pure boundary and monotonicity pins the Task 5D.2 audit demands.

The suite exists to keep one sentence true: a high refusal rate is not
evidence of scientific rigor unless the acceptance path is demonstrably
reachable under evidence that should satisfy it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.ideation.records import (
    AddressedProblem,
    CandidateIdea,
    DataRequirement,
    DataStatus,
    IdeationRunRecord,
    PortfolioReport,
    Prediction,
    ResourceEstimate,
    TargetedTheme,
    problem_key,
    theme_key,
)
from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.literature.corpus import LiteratureCorpus
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureSource,
    RetrievedSearch,
    ScriptedLiteratureProvider,
)
from autonomous_research_lab.literature.store import LiteratureStore
from autonomous_research_lab.mapping.adequacy import SupportTier
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    ProblemKind,
    SupportLocation,
    ThemeEra,
)
from autonomous_research_lab.priorart.assessment import (
    PriorArtAssessment,
    PriorArtReasonCode,
    PriorArtThresholds,
    PriorArtVerdict,
    assess_prior_art,
)
from autonomous_research_lab.priorart.challenger import (
    PriorArtChallenger,
    PriorArtRejectedError,
)
from autonomous_research_lab.priorart.directive import PriorArtDirective
from autonomous_research_lab.priorart.records import (
    DIMENSIONS,
    DimensionComparison,
    PriorArtCoverage,
    PriorArtQueryFamily,
    PriorArtScreeningRecord,
    SimilarityDecision,
    SimilarityLabel,
    WorkComparison,
)
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    UsageLedger,
)

CUTOFF = "2026-08-18"
WINDOW = "2025-08-18"


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_0",
        response_id="mcall_0",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id=None,
        latency_seconds=0.25,
        input_tokens=10,
        output_tokens=10,
        repair_count=0,
    )


def _source(
    provider_id: str, title: str, abstract: str | None
) -> LiteratureSource:
    return LiteratureSource(
        provider="scripted",
        provider_id=provider_id,
        title=title,
        authors=("Ada Lovelace",),
        publication_date="2026-01-15",
        publication_year=2026,
        venue="Journal of Examples",
        work_type="article",
        abstract=abstract,
        doi=None,
        arxiv_id=None,
        provider_url=f"https://example.org/{provider_id}",
        landing_page_url=None,
        pdf_url=None,
        cited_by_count=10,
        referenced_work_ids=(),
        access_level=(
            AccessLevel.ABSTRACT
            if abstract is not None
            else AccessLevel.METADATA
        ),
    )


def _candidate(cited_ids: tuple[str, ...]) -> CandidateIdea:
    statement = "head-level mechanisms of in-context learning are untested"
    return CandidateIdea(
        run_id="idg_1",
        title="Do Reweighting Scalars Select Induction Heads?",
        research_question="Does head reweighting select induction heads?",
        proposed_contribution="a causal test of head reweighting",
        mechanism="reweighting amplifies specialized induction heads",
        hypothesis="reweighted heads carry the in-context ability",
        grounding="the cited records report head specialization",
        predictions=(
            Prediction(
                text="ablating reweighted heads drops accuracy",
                falsifier="accuracy unchanged after ablation",
            ),
        ),
        datasets=(
            DataRequirement(
                name="synthetic sequences",
                status=DataStatus.NEW_REQUIREMENT,
                role="probe tasks",
            ),
        ),
        metrics=("accuracy",),
        evaluation_protocol="held-out probes",
        baselines=("dense fine-tuning",),
        ablations=("random head subsets",),
        resources=ResourceEstimate(
            compute="one GPU-day", data="synthetic", implementation="small"
        ),
        risks=("effects may not localize",),
        cfp_alignment="matches the adaptation topic",
        aligned_topics=("adaptation",),
        uncertainty="the mechanism may be diffuse",
        search_terms=("attention head reweighting",),
        addressed_problems=(
            AddressedProblem(
                key=problem_key(statement),
                statement=statement,
                kind=ProblemKind.OPEN_PROBLEM,
                tier=SupportTier.TENTATIVE,
            ),
        ),
        targeted_themes=(
            TargetedTheme(
                key=theme_key("adaptation"),
                name="adaptation",
                era=ThemeEra.RECENT,
            ),
        ),
        cited_source_ids=cited_ids,
        cited_recent=len(cited_ids),
        cited_foundational=0,
        cited_undated=0,
        provenance=_provenance(),
    )


def _ideation_run(candidate_ids: tuple[str, ...]) -> IdeationRunRecord:
    return IdeationRunRecord(
        run_id="idg_1",
        directive_id="idir_1",
        assessment_id="madq_1",
        map_run_id="map_1",
        snapshot_id="cfp_1",
        direction_id="dir_1",
        candidate_ids=candidate_ids,
        refusal_justification="",
        diversity_rationale="one candidate",
        model_calls=2,
        input_tokens=100,
        output_tokens=50,
        portfolio=PortfolioReport(
            problems_total=1,
            problems_addressed=1,
            problems_unaddressed=0,
            unaddressed_statements=(),
            addressed_multi_source=0,
            addressed_tentative=1,
            addressed_single_source_limitation=0,
            addressed_contradicted=0,
            candidates=len(candidate_ids),
            distinct_sources_cited=1,
            themes_targeted=1,
            distinct_problem_sets=1,
            distinct_theme_sets=1,
            distinct_dataset_sets=1,
            distinct_metric_sets=1,
        ),
    )


def _retrieved(*sources: LiteratureSource) -> RetrievedSearch:
    return RetrievedSearch(
        provider="scripted",
        retrieved_at="2026-08-19T12:00:00+00:00",
        request_params={"filter": "scripted"},
        total_count=len(sources),
        pages_fetched=1,
        page_identifiers=("",),
        rate_limit={},
        truncated=False,
        sources=sources,
    )


PLAN_GROUPS: dict[str, list[list[str]]] = {
    "mechanism": [["attention head reweighting", "head gating"]],
    "problem_mechanism": [
        ["in-context learning"],
        ["attention head reweighting"],
    ],
    "evaluation_setup": [
        ["held-out probes", "probe tasks"],
        ["induction heads"],
    ],
    "synonyms_legacy": [
        ["head gating", "head masking", "head pruning"],
        ["induction heads"],
    ],
    "competing_approaches": [
        ["LoRA", "adapters"],
        ["attention head reweighting"],
    ],
    "recent": [["attention head reweighting"]],
}

QUERY_REPLY = json.dumps(
    {
        "queries": [
            {
                "family": family.value,
                "groups": [
                    {"alternatives": list(group)}
                    for group in PLAN_GROUPS[family.value]
                ],
            }
            for family in PriorArtQueryFamily
        ]
    }
)


def _screen_reply(*entries: dict[str, object]) -> str:
    return json.dumps({"screens": list(entries)})


def _screen(
    source_id: str, decision: str, reason: str, **extra: object
) -> dict[str, object]:
    entry: dict[str, object] = {
        "source_id": source_id,
        "decision": decision,
        "reason": reason,
    }
    entry.update(extra)
    return entry


def _comparison_entry(
    source_id: str,
    snippet: str,
    prior_position: str,
    similarity: str = "related",
    material_differences: tuple[str, ...] = (
        "a causal ablation test versus an observational report",
    ),
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "similarity": similarity,
        "overlap_features": ["both act on attention heads"],
        "material_differences": list(material_differences),
        "dimensions": [
            {
                "dimension": dimension.value,
                "candidate_position": (
                    "proposes a causal test of head reweighting"
                ),
                "prior_work_position": prior_position,
                "support_location": "abstract",
                "support_snippet": snippet,
            }
            for dimension in DIMENSIONS
        ],
    }


def _comparison_reply(*entries: dict[str, object]) -> str:
    return json.dumps({"comparisons": list(entries)})


#: The closed control corpus: real apparent similarities around the
#: candidate's own topic — two works genuinely close enough to screen
#: as potential overlaps, three nearby, five unrelated, one cited.
CITED = _source(
    "W_cited",
    "Attention Head Specialization",
    "We show that attention heads specialize during in-context learning "
    "and that ablating them removes the ability.",
)
OVERLAP_1 = _source(
    "W_o1",
    "Gating Attention Heads for In-Context Adaptation",
    "We learn per-head gating scalars that amplify induction heads and "
    "improve in-context learning on synthetic probes.",
)
OVERLAP_2 = _source(
    "W_o2",
    "Reweighting Attention for Few-Shot Learning",
    "We reweight attention modules during fine-tuning and report gains "
    "on few-shot benchmarks.",
)
NEARBY_1 = _source(
    "W_r1",
    "Induction Heads Drive In-Context Learning",
    "We characterize induction heads through prefix matching and "
    "copying behaviour in trained transformers.",
)
NEARBY_2 = _source(
    "W_r2",
    "Pruning Attention Heads",
    "We prune attention heads after training and evaluate the pruned "
    "model on language benchmarks.",
)
NEARBY_3 = _source(
    "W_r3",
    "Head Importance Scores in Transformers",
    "We estimate the importance of attention heads with gradient-based "
    "scores.",
)
UNRELATED = tuple(
    _source(f"W_u{index}", title, abstract)
    for index, (title, abstract) in enumerate(
        (
            (
                "A Survey of Optimizers",
                "We survey optimizers for large-scale vision training.",
            ),
            (
                "Protein Structure Prediction",
                "We predict protein structures from sequence data.",
            ),
            (
                "Streaming Speech Recognition",
                "We build a streaming recognizer for conversational "
                "speech.",
            ),
            (
                "Learned Database Indexes",
                "We replace classical database indexes with learned "
                "models.",
            ),
            (
                "Semantic Image Segmentation",
                "We segment natural images into semantic regions.",
            ),
        ),
        start=1,
    )
)

FRESH = (OVERLAP_1, OVERLAP_2, NEARBY_1, NEARBY_2, NEARBY_3, *UNRELATED)

#: Retrievals per family, in proposal order: the control corpus split
#: across the first two families, the rest empty.
CONTROL_RETRIEVALS = (
    _retrieved(*FRESH[:5]),
    _retrieved(*FRESH[5:]),
    _retrieved(),
    _retrieved(),
    _retrieved(),
    _retrieved(),
)

#: The abstract screening reply for the control pool, in pool order —
#: the cited work heads it.
CONTROL_SCREENING = _screen_reply(
    _screen(CITED.id, "related", "nearby head specialization work"),
    _screen(
        OVERLAP_1.id,
        "potential_overlap",
        "gating scalars that amplify induction heads sit close to the "
        "candidate's mechanism",
    ),
    _screen(
        OVERLAP_2.id,
        "potential_overlap",
        "reweights attention modules during adaptation",
    ),
    _screen(NEARBY_1.id, "related", "characterizes induction heads"),
    _screen(NEARBY_2.id, "related", "intervenes on attention heads"),
    _screen(NEARBY_3.id, "related", "scores attention heads"),
    *(
        _screen(
            source.id,
            "unrelated",
            "not about attention heads or in-context learning",
        )
        for source in UNRELATED
    ),
)

#: Comparison of the four closest works, each grounded in a verbatim
#: quote of its own abstract.
CONTROL_COMPARISON = _comparison_reply(
    _comparison_entry(
        OVERLAP_1.id,
        "amplify induction heads",
        "learns gating scalars that amplify induction heads",
        material_differences=(
            "a causal ablation test versus gating during training",
        ),
    ),
    _comparison_entry(
        OVERLAP_2.id,
        "reweight attention modules",
        "reweights whole attention modules during fine-tuning",
        material_differences=(
            "head-level selection versus module-level adaptation",
        ),
    ),
    _comparison_entry(
        CITED.id,
        "attention heads specialize",
        "reports that attention heads specialize",
    ),
    _comparison_entry(
        NEARBY_1.id,
        "prefix matching and copying",
        "characterizes induction heads through prefix matching",
    ),
)


def _wire(
    tmp_path: Path,
    replies: tuple[str, ...],
    retrievals: tuple[RetrievedSearch, ...],
    cited: tuple[LiteratureSource, ...] = (CITED,),
) -> tuple[PriorArtChallenger, FakeModelProvider, PriorArtStore, str]:
    ideation_store = IdeationStore(tmp_path / "ideation")
    known = LiteratureStore(tmp_path / "known_literature")
    for source in cited:
        known.record_source(source)
    idea = ideation_store.record_idea(
        _candidate(tuple(source.id for source in cited))
    )
    run_record = ideation_store.record_run(_ideation_run((idea.id,)))
    provider = FakeModelProvider(replies)
    store = PriorArtStore(tmp_path / "priorart")
    challenger = PriorArtChallenger(
        provider=provider,
        model="model-x",
        ledger=UsageLedger(),
        ideation_store=ideation_store,
        known_literature=known,
        corpus=LiteratureCorpus(
            LiteratureStore(tmp_path / "fresh_literature"),
            ScriptedLiteratureProvider(retrievals),
        ),
        store=store,
    )
    return challenger, provider, store, run_record.id


def _directive(record_id: str, **overrides: object) -> PriorArtDirective:
    defaults: dict[str, object] = {
        "ideation_run_record_id": record_id,
        "cutoff_date": CUTOFF,
        "recent_window_start": WINDOW,
    }
    defaults.update(overrides)
    return PriorArtDirective(**defaults)  # type: ignore[arg-type]


# -- DISTINGUISHED is reachable ------------------------------------------------


def test_distinguished_is_reachable_on_a_closed_control_corpus(
    tmp_path: Path,
) -> None:
    # Eleven in-cutoff abstract-level sources with real apparent
    # similarities, two of them screened as potential overlaps and both
    # compared with grounded material differences: the acceptance path,
    # at the default thresholds, with no exemptions anywhere.
    challenger, provider, _, record_id = _wire(
        tmp_path,
        (QUERY_REPLY, CONTROL_SCREENING, CONTROL_COMPARISON),
        CONTROL_RETRIEVALS,
    )
    result = challenger.run(_directive(record_id))
    (assessment,) = result.assessments
    assert assessment.thresholds == PriorArtThresholds()
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED
    assert assessment.reasons == ()
    coverage = assessment.coverage
    assert coverage.unique_sources == 11
    assert coverage.potential_overlap == 2
    assert coverage.compared_works == 4
    assert coverage.screening_truncated == 0
    assert len(provider.calls) == 3


def test_distinguished_survives_metadata_noise(tmp_path: Path) -> None:
    # The same control corpus plus two generically titled metadata-only
    # sources, honestly screened undecidable in their own gated call:
    # the pre-5D.2 rules blocked here; the calibrated rules do not.
    noise = (
        _source("W_m1", "Adaptive Methods for Sequence Models", None),
        _source("W_m2", "Efficient Fine-Tuning Approaches", None),
    )
    retrievals = (
        CONTROL_RETRIEVALS[0],
        _retrieved(*FRESH[5:], *noise),
        *CONTROL_RETRIEVALS[2:],
    )
    metadata_reply = _screen_reply(
        *(
            _screen(
                source.id,
                "undecidable",
                "a generic title; the metadata cannot settle similarity",
            )
            for source in noise
        )
    )
    challenger, _, _, record_id = _wire(
        tmp_path,
        (QUERY_REPLY, CONTROL_SCREENING, metadata_reply, CONTROL_COMPARISON),
        retrievals,
    )
    result = challenger.run(_directive(record_id))
    (assessment,) = result.assessments
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED
    assert assessment.reasons == ()
    assert assessment.coverage.metadata_level == 2
    assert assessment.coverage.undecidable == 2
    assert assessment.coverage.metadata_ambiguous == 0


# -- the blockers still block --------------------------------------------------


def test_a_material_metadata_ambiguity_blocks_differentiation(
    tmp_path: Path,
) -> None:
    # A metadata-only title that directly claims the candidate's core
    # contribution, attested end to end: the one metadata condition
    # that must survive calibration.
    shadow = _source(
        "W_shadow",
        "A Causal Test of Head Reweighting for Induction Heads",
        None,
    )
    retrievals = (
        CONTROL_RETRIEVALS[0],
        _retrieved(*FRESH[5:], shadow),
        *CONTROL_RETRIEVALS[2:],
    )
    metadata_reply = _screen_reply(
        _screen(
            shadow.id,
            "potential_overlap",
            "the title claims the candidate's proposed causal test",
            overlap_hypothesis={
                "candidate_claim": "a causal test of head reweighting",
                "source_text": "a causal test of head reweighting",
                "support_location": "title",
                "dimension": "claimed_contribution",
                "rationale": (
                    "the title claims the same causal test the candidate "
                    "proposes as its contribution"
                ),
            },
        )
    )
    challenger, _, _, record_id = _wire(
        tmp_path,
        (QUERY_REPLY, CONTROL_SCREENING, metadata_reply, CONTROL_COMPARISON),
        retrievals,
    )
    result = challenger.run(_directive(record_id))
    (assessment,) = result.assessments
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    codes = {reason.code for reason in assessment.reasons}
    assert codes == {PriorArtReasonCode.METADATA_AMBIGUITY}
    (reason,) = assessment.reasons
    assert "a causal test of head reweighting" in reason.detail
    assert "claimed_contribution" in reason.detail


def test_unattested_overlap_speculation_is_rejected(tmp_path: Path) -> None:
    # A hypothesis whose quote is not in the title fails the gate on
    # both attempts; the run fails closed with the payloads preserved,
    # and no verdict of any kind is recorded.
    shadow = _source(
        "W_shadow",
        "A Causal Test of Head Reweighting for Induction Heads",
        None,
    )
    retrievals = (
        CONTROL_RETRIEVALS[0],
        _retrieved(*FRESH[5:], shadow),
        *CONTROL_RETRIEVALS[2:],
    )
    speculation = _screen_reply(
        _screen(
            shadow.id,
            "potential_overlap",
            "the title sits near the candidate's topic",
            overlap_hypothesis={
                "candidate_claim": "a causal test of head reweighting",
                "source_text": "adaptive routing of experts",
                "support_location": "title",
                "dimension": "mechanism",
                "rationale": "the routing could subsume head reweighting",
            },
        )
    )
    challenger, _, store, record_id = _wire(
        tmp_path,
        (QUERY_REPLY, CONTROL_SCREENING, speculation, speculation),
        retrievals,
    )
    with pytest.raises(PriorArtRejectedError, match="unsupported_claim"):
        challenger.run(_directive(record_id))
    rejected = store.rejected()
    assert len(rejected) == 2
    assert all(
        entry["stage"] == "metadata_screening" for entry in rejected
    )
    assert store.prior_art_assessments() == ()


def test_overlapping_is_reachable_and_dominates_thin_coverage(
    tmp_path: Path,
) -> None:
    # One grounded substantial match falsifies regardless of how thin
    # the search was — and the thin coverage stays on the record.
    prior = _source(
        "W_prior",
        "A Causal Test of Attention Head Reweighting",
        "We causally test head reweighting and show that reweighted "
        "heads carry in-context ability on synthetic probes.",
    )
    retrievals = (
        _retrieved(prior),
        _retrieved(),
        _retrieved(),
        _retrieved(),
        _retrieved(),
        _retrieved(),
    )
    screening = _screen_reply(
        _screen(CITED.id, "related", "nearby head specialization work"),
        _screen(
            prior.id,
            "potential_overlap",
            "causally tests head reweighting like the candidate",
        ),
    )
    comparison = _comparison_reply(
        _comparison_entry(
            prior.id,
            "causally test head reweighting",
            "causally tests head reweighting on synthetic probes",
            similarity="substantial_match",
            material_differences=(),
        ),
        _comparison_entry(
            CITED.id,
            "attention heads specialize",
            "reports that attention heads specialize",
        ),
    )
    challenger, _, _, record_id = _wire(
        tmp_path, (QUERY_REPLY, screening, comparison), retrievals
    )
    result = challenger.run(_directive(record_id))
    (assessment,) = result.assessments
    assert assessment.verdict is PriorArtVerdict.OVERLAPPING
    assert assessment.overlapping_work_ids == (prior.id,)
    codes = {reason.code for reason in assessment.reasons}
    assert PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES in codes


def test_a_preflighted_pool_at_the_bound_never_truncates(
    tmp_path: Path,
) -> None:
    # Thirty distinct fresh works plus five cited — the exact worst
    # case the default directive reserves — screens completely: zero
    # truncation, and raw volume alone still buys no differentiation.
    cited = tuple(
        _source(
            f"W_cited{index}",
            f"Attention Head Specialization Part {index}",
            "We show that attention heads specialize during in-context "
            "learning.",
        )
        for index in range(5)
    )
    fresh = tuple(
        _source(
            f"W_f{index}",
            f"Protein Folding Study {index}",
            "We study a protein folding variant from sequence data.",
        )
        for index in range(30)
    )
    retrievals = tuple(
        _retrieved(*fresh[start : start + 5]) for start in range(0, 30, 5)
    )
    pool_order = (*cited, *fresh)
    screening_replies = tuple(
        _screen_reply(
            *(
                _screen(
                    source.id,
                    "unrelated",
                    "not about attention head reweighting",
                )
                for source in pool_order[start : start + 12]
            )
        )
        for start in range(0, 35, 12)
    )
    challenger, provider, _, record_id = _wire(
        tmp_path,
        (QUERY_REPLY, *screening_replies),
        retrievals,
        cited=cited,
    )
    result = challenger.run(_directive(record_id))
    (assessment,) = result.assessments
    coverage = assessment.coverage
    assert coverage.unique_sources == 35
    assert coverage.screened == 35
    assert coverage.screening_truncated == 0
    assert len(provider.calls) == 4  # one query call, three batches
    # Thirty-five sources and a complete family sweep, yet nothing
    # comparable: volume is not differentiation.
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    codes = {reason.code for reason in assessment.reasons}
    assert codes == {PriorArtReasonCode.NO_COMPARABLE_WORK}


# -- pure boundary and monotonicity pins ---------------------------------------


def _screening_record(
    source_id: str, decision: SimilarityDecision
) -> PriorArtScreeningRecord:
    return PriorArtScreeningRecord(
        run_id="pac_1",
        candidate_id="idea_1",
        source_id=source_id,
        known_prior_art=False,
        decision=decision,
        reason="screened against the candidate's mechanism",
        provenance=_provenance(),
    )


def _comparison_record(source_id: str) -> WorkComparison:
    return WorkComparison(
        run_id="pac_1",
        candidate_id="idea_1",
        source_id=source_id,
        known_prior_art=False,
        dimensions=tuple(
            DimensionComparison(
                dimension=dimension,
                candidate_position="reweights attention heads",
                prior_work_position="prunes attention heads",
                support_location=SupportLocation.ABSTRACT,
                support_snippet="prunes attention heads",
            )
            for dimension in DIMENSIONS
        ),
        overlap_features=("both intervene on attention heads",),
        material_differences=("reweighting versus pruning",),
        similarity=SimilarityLabel.RELATED,
        provenance=_provenance(),
    )


def _pure_assessment(
    *, unique_sources: int, screening_truncated: int = 0
) -> PriorArtAssessment:
    screened = unique_sources - screening_truncated
    decisions = {
        "lit_1": SimilarityDecision.POTENTIAL_OVERLAP,
        **{
            f"lit_{index}": SimilarityDecision.UNRELATED
            for index in range(2, screened + 1)
        },
    }
    return assess_prior_art(
        run_id="pac_1",
        candidate_id="idea_1",
        directive_id="pdir_1",
        screenings=tuple(
            _screening_record(source_id, decision)
            for source_id, decision in decisions.items()
        ),
        comparisons=(_comparison_record("lit_1"),),
        coverage=PriorArtCoverage(
            families_executed=tuple(
                family.value for family in PriorArtQueryFamily
            ),
            queries_executed=6,
            total_retrieved=unique_sources,
            unique_sources=unique_sources,
            overlap=0,
            saturation=0.5,
            post_cutoff_excluded=0,
            undated_sources=0,
            abstract_level=unique_sources,
            metadata_level=0,
            known_prior_art_listed=0,
            known_prior_art_recovered=0,
            screened=screened,
            potential_overlap=1,
            related=0,
            unrelated=screened - 1,
            undecidable=0,
            metadata_ambiguous=0,
            screening_truncated=screening_truncated,
            compared_works=1,
        ),
        metadata_source_ids=frozenset(),
        thresholds=PriorArtThresholds(),
    )


def test_the_source_threshold_boundary_is_exact() -> None:
    # Nine screenable sources refuse; ten distinguish. The cliff is
    # deliberate and pinned on both sides.
    below = _pure_assessment(unique_sources=9)
    assert below.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert {reason.code for reason in below.reasons} == {
        PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES
    }
    at = _pure_assessment(unique_sources=10)
    assert at.verdict is PriorArtVerdict.DISTINGUISHED
    assert at.reasons == ()


def test_padding_the_pool_cannot_buy_the_verdict() -> None:
    # Forty sources do not erase a truncation: crossing the count
    # threshold repairs exactly one reason, never the others.
    padded = _pure_assessment(unique_sources=40, screening_truncated=2)
    assert padded.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert {reason.code for reason in padded.reasons} == {
        PriorArtReasonCode.SCREENING_TRUNCATED
    }
