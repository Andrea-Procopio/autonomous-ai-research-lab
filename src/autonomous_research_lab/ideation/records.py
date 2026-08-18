"""The candidate-generation records: everything an ideation run may
durably claim.

A candidate idea is a conjecture about what might be worth investigating —
never a finding, never a proposal, and never scientific state. Nothing
here is ``Evidence``, a ``ResearchQuestion``, a ``Hypothesis``, or any
other scientific-state proposition; a candidate reaches the governed
commit only when a later task deliberately builds that path.

Epistemic labeling is structural, stamped by trusted code, never chosen
by a model (see :data:`CLAIM_KINDS`): what the CFP text reports is kept
apart from what the extractor synthesized from it; a candidate's grounding
narrative (which must trace to cited mapping records) is kept apart from
its conjectures (mechanism, hypothesis) and from its design targets
(predictions, falsifiers, resource estimates — where *new* numbers are
the whole point). Every candidate's novelty status is structurally
``UNASSESSED``: no other value exists until the prior-art challenge
(Task 5D) defines what assessment would even mean.

Cross-references are canonical handles, never echoed prose: trusted code
derives a content key for every inventory problem and field-map theme
(:func:`problem_key`, :func:`theme_key`), the model cites keys, and
trusted code stamps the resolved statement, kind, and computed support
tier onto the record — so a mismatched pair is unconstructible, and a
single-paper limitation carries its tier wherever the candidate goes.

Identity follows the house rules: run ids are occurrences, every stored
record is content-addressed over all of its fields, provider provenance
included.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Final

from ..core.ids import content_id
from ..mapping.adequacy import SupportTier
from ..mapping.records import CallProvenance, ProblemKind, ThemeEra

#: The structural epistemic label of each model-authored record category.
#: Literature facts are never re-authored here — candidates cite the
#: mapping run's source ids; the records themselves stay in the mapping
#: store.
CLAIM_KINDS: Final = {
    "direction.scope": "extractor_synthesis",
    "direction.topics": "cfp_reported",
    "direction.constraints": "cfp_reported",
    "direction.relevant_dates": "cfp_reported",
    "idea.title": "candidate_conjecture",
    "idea.research_question": "candidate_conjecture",
    "idea.proposed_contribution": "candidate_conjecture",
    "idea.mechanism": "candidate_conjecture",
    "idea.hypothesis": "candidate_conjecture",
    "idea.grounding": "grounding_synthesis",
    "idea.predictions": "candidate_prediction",
    "idea.falsifiers": "design_target",
    "idea.datasets": "design_target",
    "idea.metrics": "design_target",
    "idea.evaluation_protocol": "design_target",
    "idea.baselines": "design_target",
    "idea.ablations": "design_target",
    "idea.resources": "design_target",
    "idea.risks": "candidate_conjecture",
    "idea.cfp_alignment": "grounding_synthesis",
    "idea.uncertainty": "grounding_synthesis",
    "idea.search_terms": "design_target",
    "run.diversity_rationale": "grounding_synthesis",
    "run.refusal_justification": "grounding_synthesis",
}


def problem_key(statement: str) -> str:
    """The deterministic content identity of one inventory problem
    statement: the same problem is the same problem wherever it is
    cited. The model cites these keys; it never echoes the sentence."""
    return content_id("prob", statement)


def theme_key(name: str) -> str:
    """The deterministic content identity of one field-map theme name."""
    return content_id("thm", name)


class NoveltyStatus(StrEnum):
    """Structurally the only value a candidate can carry: novelty is a
    claim about the world's literature, and a bounded generation run has
    no standing to assess it. Task 5D's prior-art challenge will define
    the other values when it defines the assessment."""

    UNASSESSED = "unassessed"


@dataclass(frozen=True, slots=True)
class AddressedProblem:
    """One inventory problem a candidate addresses: the model-cited key,
    with the statement, kind, and computed support tier stamped by
    trusted code. A key that does not derive from its statement is
    unconstructible — drift cannot be recorded."""

    key: str
    statement: str
    kind: ProblemKind
    tier: SupportTier

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("an addressed problem requires a statement")
        if self.key != problem_key(self.statement):
            raise ValueError(
                f"key {self.key} does not derive from the statement; an "
                f"addressed problem binds the two by construction"
            )


@dataclass(frozen=True, slots=True)
class TargetedTheme:
    """One field-map theme a candidate targets, bound the same way."""

    key: str
    name: str
    era: ThemeEra

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a targeted theme requires a name")
        if self.key != theme_key(self.name):
            raise ValueError(
                f"key {self.key} does not derive from the theme name; a "
                f"targeted theme binds the two by construction"
            )


@dataclass(frozen=True, slots=True)
class Prediction:
    """One measurable prediction with its explicit falsifier: what would
    be observed, and what observation would refute it. The falsifier is
    structural — a prediction without one cannot be expressed."""

    text: str
    falsifier: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a prediction requires text")
        if not self.falsifier.strip():
            raise ValueError(
                "a prediction requires an explicit falsifier: what "
                "observation would refute it"
            )


class DataStatus(StrEnum):
    """Whether a named dataset exists in the cited records or is a new
    requirement the candidate creates. The distinction is gate-enforced:
    an 'existing' dataset must be reported by a cited record; a new
    requirement is honestly a need, never presented as available."""

    EXISTING = "existing"
    NEW_REQUIREMENT = "new_requirement"


@dataclass(frozen=True, slots=True)
class DataRequirement:
    """One dataset a candidate proposes to use or to create."""

    name: str
    status: DataStatus
    role: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a data requirement must be named")
        if not self.role.strip():
            raise ValueError(
                "a data requirement must say what role the data plays"
            )


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Approximate compute, data, and implementation requirements —
    design estimates, never measurements."""

    compute: str
    data: str
    implementation: str

    def __post_init__(self) -> None:
        for label, value in (
            ("compute", self.compute),
            ("data", self.data),
            ("implementation", self.implementation),
        ):
            if not value.strip():
                raise ValueError(f"a resource estimate requires {label}")


@dataclass(frozen=True, slots=True)
class CandidateIdea:
    """One gated, source-grounded research candidate. Model-authored
    texts arrive through the deterministic gate; the addressed problems'
    statements/kinds/tiers, the targeted themes' names/eras, the cited
    sources' era mix, and the novelty status are stamped by trusted
    code."""

    run_id: str
    title: str
    research_question: str
    proposed_contribution: str
    mechanism: str
    hypothesis: str
    grounding: str
    predictions: tuple[Prediction, ...]
    datasets: tuple[DataRequirement, ...]
    metrics: tuple[str, ...]
    evaluation_protocol: str
    baselines: tuple[str, ...]
    ablations: tuple[str, ...]
    resources: ResourceEstimate
    risks: tuple[str, ...]
    cfp_alignment: str
    aligned_topics: tuple[str, ...]
    uncertainty: str
    search_terms: tuple[str, ...]
    addressed_problems: tuple[AddressedProblem, ...]
    targeted_themes: tuple[TargetedTheme, ...]
    cited_source_ids: tuple[str, ...]
    cited_recent: int
    cited_foundational: int
    cited_undated: int
    provenance: CallProvenance
    novelty_status: NoveltyStatus = NoveltyStatus.UNASSESSED
    id: str = field(default="")

    def __post_init__(self) -> None:
        for label, value in (
            ("title", self.title),
            ("research_question", self.research_question),
            ("proposed_contribution", self.proposed_contribution),
            ("mechanism", self.mechanism),
            ("hypothesis", self.hypothesis),
            ("grounding", self.grounding),
            ("evaluation_protocol", self.evaluation_protocol),
            ("cfp_alignment", self.cfp_alignment),
            ("uncertainty", self.uncertainty),
        ):
            if not value.strip():
                raise ValueError(f"a candidate requires {label}")
        for label, items in (
            ("predictions", self.predictions),
            ("datasets", self.datasets),
            ("metrics", self.metrics),
            ("baselines", self.baselines),
            ("ablations", self.ablations),
            ("risks", self.risks),
            ("aligned_topics", self.aligned_topics),
            ("search_terms", self.search_terms),
            ("addressed_problems", self.addressed_problems),
            ("targeted_themes", self.targeted_themes),
            ("cited_source_ids", self.cited_source_ids),
        ):
            if not items:
                raise ValueError(f"a candidate requires at least one of {label}")
        if len(set(self.cited_source_ids)) != len(self.cited_source_ids):
            raise ValueError("a candidate cites each source at most once")
        keys = [problem.key for problem in self.addressed_problems]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "a candidate addresses each problem at most once"
            )
        cited = self.cited_recent + self.cited_foundational + self.cited_undated
        if (
            min(self.cited_recent, self.cited_foundational, self.cited_undated)
            < 0
            or cited != len(self.cited_source_ids)
        ):
            raise ValueError(
                "the era mix must partition the cited sources exactly"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "idea",
                    self.run_id,
                    self.title,
                    self.research_question,
                    self.proposed_contribution,
                    self.mechanism,
                    self.hypothesis,
                    self.grounding,
                    tuple((p.text, p.falsifier) for p in self.predictions),
                    tuple(
                        (d.name, d.status, d.role) for d in self.datasets
                    ),
                    self.metrics,
                    self.evaluation_protocol,
                    self.baselines,
                    self.ablations,
                    (
                        self.resources.compute,
                        self.resources.data,
                        self.resources.implementation,
                    ),
                    self.risks,
                    self.cfp_alignment,
                    self.aligned_topics,
                    self.uncertainty,
                    self.search_terms,
                    tuple(
                        (p.key, p.statement, p.kind, p.tier)
                        for p in self.addressed_problems
                    ),
                    tuple(
                        (t.key, t.name, t.era) for t in self.targeted_themes
                    ),
                    self.cited_source_ids,
                    self.cited_recent,
                    self.cited_foundational,
                    self.cited_undated,
                    self.novelty_status,
                    self.provenance.response_id,
                ),
            )


@dataclass(frozen=True, slots=True)
class PortfolioReport:
    """Honest portfolio accounting, computed by trusted code alone: which
    inventory problems the candidates address and — named, not just
    counted — which they do not, the support-tier profile of what was
    addressed, and how diverse the portfolio mechanically is (distinct
    values per dimension; semantic diversity is not pretended to be
    checkable)."""

    problems_total: int
    problems_addressed: int
    problems_unaddressed: int
    unaddressed_statements: tuple[str, ...]
    addressed_multi_source: int
    addressed_tentative: int
    addressed_single_source_limitation: int
    addressed_contradicted: int
    candidates: int
    distinct_sources_cited: int
    themes_targeted: int
    distinct_problem_sets: int
    distinct_theme_sets: int
    distinct_dataset_sets: int
    distinct_metric_sets: int

    def __post_init__(self) -> None:
        if self.problems_addressed + self.problems_unaddressed != (
            self.problems_total
        ):
            raise ValueError(
                "addressed and unaddressed problems must partition the "
                "inventory"
            )
        if len(self.unaddressed_statements) != self.problems_unaddressed:
            raise ValueError(
                "every unaddressed problem is named, not just counted"
            )
        tiers = (
            self.addressed_multi_source
            + self.addressed_tentative
            + self.addressed_single_source_limitation
            + self.addressed_contradicted
        )
        if tiers != self.problems_addressed:
            raise ValueError(
                "the tier profile must partition the addressed problems"
            )
        for label in (
            "distinct_problem_sets",
            "distinct_theme_sets",
            "distinct_dataset_sets",
            "distinct_metric_sets",
        ):
            value = int(getattr(self, label))
            if value > self.candidates or (self.candidates and value < 1):
                raise ValueError(
                    f"{label} must be in 1..candidates for a non-empty "
                    f"portfolio, 0 for a refusal"
                )
            if not self.candidates and value:
                raise ValueError(
                    f"{label} must be 0 when there are no candidates"
                )


@dataclass(frozen=True, slots=True)
class IdeationRunRecord:
    """The completed run: what was asked, what was spent, the ids of
    everything produced — or the preserved justification of an honest
    refusal. Written once, after the portfolio. A refusal and a
    portfolio are mutually exclusive by construction."""

    run_id: str
    directive_id: str
    assessment_id: str
    map_run_id: str
    snapshot_id: str
    direction_id: str
    candidate_ids: tuple[str, ...]
    refusal_justification: str
    diversity_rationale: str
    model_calls: int
    input_tokens: int
    output_tokens: int
    portfolio: PortfolioReport
    id: str = field(default="")

    def __post_init__(self) -> None:
        if bool(self.candidate_ids) == bool(self.refusal_justification.strip()):
            raise ValueError(
                "exactly one of a candidate portfolio and a refusal "
                "justification must be present"
            )
        if not self.candidate_ids:
            if self.diversity_rationale.strip():
                raise ValueError(
                    "a refusal carries no diversity rationale"
                )
            if self.portfolio.candidates:
                raise ValueError(
                    "a refusal's portfolio must count zero candidates"
                )
        else:
            if not self.diversity_rationale.strip():
                raise ValueError(
                    "a portfolio requires its diversity rationale"
                )
            if self.portfolio.candidates != len(self.candidate_ids):
                raise ValueError(
                    "the portfolio must count exactly the recorded "
                    "candidates"
                )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "irun",
                    self.run_id,
                    self.directive_id,
                    self.assessment_id,
                    self.map_run_id,
                    self.snapshot_id,
                    self.direction_id,
                    self.candidate_ids,
                    self.refusal_justification,
                    self.diversity_rationale,
                    self.model_calls,
                    self.input_tokens,
                    self.output_tokens,
                    tuple(
                        getattr(self.portfolio, entry.name)
                        for entry in fields(PortfolioReport)
                    ),
                ),
            )
