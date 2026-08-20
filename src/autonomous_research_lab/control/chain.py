"""The seven stages, each wrapped so the controller can treat them alike.

Every adapter here answers the same four questions and nothing else:

``plan``
    What is this stage's work, and what is its idempotency key? The key
    is the stage name and the content id of whatever determines the work
    — a directive for the analysis stages, the state a step begins from
    for experimentation. Directives are content-addressed and derived
    from the config plus the ids upstream, so the key is identical in
    every process. That is what makes the question below answerable.

``completed``
    Is this work already on disk? Asked of the stage's *own* store, not
    of the event log, because the case it exists for is a crash between
    the side effect and the record of it. A hit means the controller
    writes the missing event instead of paying for the work twice.

``execute``
    Do the work, once, and say what it produced.

``refusals``
    Which exceptions mean "a door said no" rather than "this broke".

What no adapter does is decide anything. It builds a directive from
values the operator wrote down, calls the stage, and reads ids off the
record the stage returned. Every scientific judgment -- adequacy,
novelty, eligibility, admissibility -- stays inside the stage that owns
it, and an honest no comes back as an ordinary success that happens to
end the investigation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Protocol

from ..admission.admitter import CandidateAdmitter
from ..admission.door import AdmissionRefusedError
from ..admission.store import AdmissionStore
from ..evidence.file_store import FileEvidenceStore
from ..ideation.generator import IdeaGenerator
from ..ideation.store import IdeationStore
from ..literature.corpus import LiteratureCorpus
from ..literature.store import LiteratureStore
from ..mapping.adequacy import InadequateFieldMapError
from ..mapping.mapper import FieldMapper
from ..mapping.store import MappingStore
from ..persistence import FileStateStore
from ..priorart.assessment import MissingCandidatePortfolioError
from ..priorart.challenger import PriorArtChallenger
from ..priorart.store import PriorArtStore
from ..program.directive import RunDirective
from ..program.door import RunRefusedError
from ..program.preflight import RunPreflightError
from ..program.starter import start_run
from ..program.store import ProgramStore
from ..runtime.providers import UsageLedger
from ..selection.eligibility import MissingChallengedPortfolioError
from ..selection.records import SelectionOutcome
from ..selection.selector import CandidateSelector
from ..selection.store import SelectionStore
from .config import RunConfig
from .lab import ExperimentationUnavailableError, Lab, RuntimeRequest
from .recovery import recover
from .stage import (
    NO_FACTS,
    ZERO_SPEND,
    ChainFacts,
    Fact,
    StageName,
    StageSpend,
)

_LITERATURE: Final = "literature"
_MAPPING: Final = "mapping"
_IDEATION: Final = "ideation"
_PRIORART: Final = "priorart"
_SELECTION: Final = "selection"
_ADMISSION: Final = "admission"
_PROGRAM: Final = "program"


@dataclass(frozen=True, slots=True)
class Stores:
    """Every durable store one investigation writes to, under one root.

    One root per investigation, one directory per stage inside it. The
    facts, the artifacts and the snapshot chain sit at the top, where
    :func:`~..program.integrity.verify_run` expects them.
    """

    root: Path
    literature: LiteratureStore
    mapping: MappingStore
    ideation: IdeationStore
    prior_art: PriorArtStore
    selection: SelectionStore
    admission: AdmissionStore
    program: ProgramStore
    evidence: FileEvidenceStore
    states: FileStateStore

    @classmethod
    def under(cls, root: Path | str) -> Stores:
        base = Path(root)
        return cls(
            root=base,
            literature=LiteratureStore(base / _LITERATURE),
            mapping=MappingStore(base / _MAPPING),
            ideation=IdeationStore(base / _IDEATION),
            prior_art=PriorArtStore(base / _PRIORART),
            selection=SelectionStore(base / _SELECTION),
            admission=AdmissionStore(base / _ADMISSION),
            program=ProgramStore(base / _PROGRAM),
            evidence=FileEvidenceStore(base),
            states=FileStateStore(base),
        )


@dataclass(frozen=True, slots=True)
class StageContext:
    """What a stage is given: where to write, what was asked for, what to
    run on, and what the stages before it produced."""

    stores: Stores
    config: RunConfig
    lab: Lab
    facts: ChainFacts = NO_FACTS

    def with_produced(
        self, produced: tuple[tuple[str, str], ...]
    ) -> StageContext:
        return replace(self, facts=self.facts.updated(produced))


@dataclass(frozen=True, slots=True)
class StagePlan:
    """One stage's work, decided before anything is written."""

    key: str
    subject_id: str


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What one stage did."""

    produced: tuple[tuple[str, str], ...] = ()
    spend: StageSpend = ZERO_SPEND
    detail: str = ""

    ends_investigation: str = ""
    """A reason the chain stops here with nothing left to try — an
    honest scientific no. The stage still succeeded; the investigation
    is what ends."""

    repeat: bool = False
    """The stage has more work under a new key. Only experimentation
    uses it: one step at a time, each keyed by the state it starts
    from, so a crash costs at most the step in flight."""


class Stage(Protocol):
    """One stage of the chain, as the controller sees it."""

    @property
    def name(self) -> StageName: ...

    def plan(self, context: StageContext) -> StagePlan: ...

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None: ...

    def execute(self, context: StageContext) -> StageOutcome: ...

    def refusals(self) -> tuple[type[Exception], ...]: ...

    def limit(self, config: RunConfig, /) -> int:
        """How many times this stage may execute in one walk."""
        ...


class _SingleAttempt:
    """Most stages do one thing once. Only experimentation repeats, and
    it says so."""

    def limit(self, _config: RunConfig) -> int:
        return 1


# -- the stages ---------------------------------------------------------------


class MappingStage(_SingleAttempt):
    """Literature in, an assessed field map out. The one stage with no
    door: it starts from the brief the operator wrote."""

    @property
    def name(self) -> StageName:
        return StageName.MAPPING

    def plan(self, context: StageContext) -> StagePlan:
        brief = context.config.brief
        return StagePlan(key=f"{self.name}:{brief.id}", subject_id=brief.id)

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None:
        record = next(
            (
                found
                for found in context.stores.mapping.runs()
                if found.brief_id == plan.subject_id
            ),
            None,
        )
        if record is None:
            return None
        assessment = context.stores.mapping.adequacy_for_run(record.run_id)
        if assessment is None:
            # The run record is written last, so this should not happen;
            # treating it as incomplete re-runs rather than guessing.
            return None
        return StageOutcome(
            produced=_sorted(
                {
                    Fact.MAP_RUN_RECORD_ID: record.id,
                    Fact.ASSESSMENT_ID: assessment.id,
                }
            ),
            spend=_spend(
                record.model_calls, record.input_tokens, record.output_tokens
            ),
            detail=f"map run {record.run_id}",
        )

    def execute(self, context: StageContext) -> StageOutcome:
        corpus = LiteratureCorpus(
            store=context.stores.literature,
            provider=context.lab.literature_provider(),
        )
        mapper = FieldMapper(
            provider=context.lab.model_provider(self.name),
            model=context.config.model,
            ledger=UsageLedger(),
            corpus=corpus,
            store=context.stores.mapping,
            request_timeout_seconds=context.config.request_timeout_seconds,
        )
        result = mapper.run(context.config.brief)
        record = result.run_record
        return StageOutcome(
            produced=_sorted(
                {
                    Fact.MAP_RUN_RECORD_ID: record.id,
                    Fact.ASSESSMENT_ID: result.assessment.id,
                }
            ),
            spend=_spend(
                record.model_calls, record.input_tokens, record.output_tokens
            ),
            detail=(
                f"{len(result.extractions)} extraction(s), map "
                f"{result.assessment.status}"
            ),
        )

    def refusals(self) -> tuple[type[Exception], ...]:
        return ()


class IdeationStage(_SingleAttempt):
    """An assessed map and a call for papers in, a candidate portfolio
    out — or an honest refusal to propose anything."""

    @property
    def name(self) -> StageName:
        return StageName.IDEATION

    def plan(self, context: StageContext) -> StagePlan:
        directive = context.config.ideation.directive(
            assessment_id=context.facts.require(Fact.ASSESSMENT_ID),
            snapshot_id=context.config.snapshot.id,
        )
        return StagePlan(
            key=f"{self.name}:{directive.id}", subject_id=directive.id
        )

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None:
        record = next(
            (
                found
                for found in context.stores.ideation.runs()
                if found.directive_id == plan.subject_id
            ),
            None,
        )
        if record is None:
            return None
        return StageOutcome(
            produced=_sorted(
                {
                    Fact.IDEATION_RUN_RECORD_ID: record.id,
                    Fact.SNAPSHOT_ID: record.snapshot_id,
                }
            ),
            spend=_spend(
                record.model_calls, record.input_tokens, record.output_tokens
            ),
            detail=f"{len(record.candidate_ids)} candidate(s)",
            ends_investigation=(
                ""
                if record.candidate_ids
                else f"ideation proposed nothing: {record.refusal_justification}"
            ),
        )

    def execute(self, context: StageContext) -> StageOutcome:
        generator = IdeaGenerator(
            provider=context.lab.model_provider(self.name),
            model=context.config.model,
            ledger=UsageLedger(),
            map_store=context.stores.mapping,
            store=context.stores.ideation,
            request_timeout_seconds=context.config.request_timeout_seconds,
        )
        directive = context.config.ideation.directive(
            assessment_id=context.facts.require(Fact.ASSESSMENT_ID),
            snapshot_id=context.config.snapshot.id,
        )
        result = generator.run(directive, context.config.snapshot)
        record = result.run_record
        return StageOutcome(
            produced=_sorted(
                {
                    Fact.IDEATION_RUN_RECORD_ID: record.id,
                    Fact.SNAPSHOT_ID: record.snapshot_id,
                }
            ),
            spend=_spend(
                record.model_calls, record.input_tokens, record.output_tokens
            ),
            detail=f"{len(result.ideas)} candidate(s)",
            ends_investigation=(
                ""
                if result.ideas
                else f"ideation proposed nothing: {record.refusal_justification}"
            ),
        )

    def refusals(self) -> tuple[type[Exception], ...]:
        return (InadequateFieldMapError,)


class PriorArtStage(_SingleAttempt):
    """Every candidate challenged against fresh bounded retrieval."""

    @property
    def name(self) -> StageName:
        return StageName.PRIOR_ART

    def plan(self, context: StageContext) -> StagePlan:
        directive = context.config.prior_art.directive(
            ideation_run_record_id=context.facts.require(
                Fact.IDEATION_RUN_RECORD_ID
            )
        )
        return StagePlan(
            key=f"{self.name}:{directive.id}", subject_id=directive.id
        )

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None:
        record = next(
            (
                found
                for found in context.stores.prior_art.runs()
                if found.directive_id == plan.subject_id
            ),
            None,
        )
        if record is None:
            return None
        return StageOutcome(
            produced=_sorted({Fact.PRIOR_ART_RUN_RECORD_ID: record.id}),
            spend=_spend(
                record.model_calls, record.input_tokens, record.output_tokens
            ),
            detail=f"{len(record.prior_art_assessment_ids)} assessment(s)",
        )

    def execute(self, context: StageContext) -> StageOutcome:
        corpus = LiteratureCorpus(
            store=context.stores.literature,
            provider=context.lab.literature_provider(),
        )
        challenger = PriorArtChallenger(
            provider=context.lab.model_provider(self.name),
            model=context.config.model,
            ledger=UsageLedger(),
            ideation_store=context.stores.ideation,
            known_literature=context.stores.literature,
            corpus=corpus,
            store=context.stores.prior_art,
            request_timeout_seconds=context.config.request_timeout_seconds,
        )
        directive = context.config.prior_art.directive(
            ideation_run_record_id=context.facts.require(
                Fact.IDEATION_RUN_RECORD_ID
            )
        )
        result = challenger.run(directive)
        record = result.run_record
        return StageOutcome(
            produced=_sorted({Fact.PRIOR_ART_RUN_RECORD_ID: record.id}),
            spend=_spend(
                record.model_calls, record.input_tokens, record.output_tokens
            ),
            detail=f"{len(record.prior_art_assessment_ids)} assessment(s)",
        )

    def refusals(self) -> tuple[type[Exception], ...]:
        return (MissingCandidatePortfolioError,)


class SelectionStage(_SingleAttempt):
    """One challenged portfolio in, at most one candidate out.

    The two stops are successes: a selection that finds nothing eligible
    or nothing defensible has done its job exactly, and the
    investigation ends with that on the record.
    """

    @property
    def name(self) -> StageName:
        return StageName.SELECTION

    def plan(self, context: StageContext) -> StagePlan:
        directive = context.config.selection.directive(
            prior_art_run_record_id=context.facts.require(
                Fact.PRIOR_ART_RUN_RECORD_ID
            )
        )
        return StagePlan(
            key=f"{self.name}:{directive.id}", subject_id=directive.id
        )

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None:
        record = next(
            (
                found
                for found in context.stores.selection.runs()
                if found.directive_id == plan.subject_id
            ),
            None,
        )
        if record is None:
            return None
        return _selection_outcome(
            record.id,
            record.outcome,
            _spend(
                record.model_calls, record.input_tokens, record.output_tokens
            ),
        )

    def execute(self, context: StageContext) -> StageOutcome:
        selector = CandidateSelector(
            provider=context.lab.model_provider(self.name),
            model=context.config.model,
            ledger=UsageLedger(),
            ideation_store=context.stores.ideation,
            prior_art_store=context.stores.prior_art,
            store=context.stores.selection,
            request_timeout_seconds=context.config.request_timeout_seconds,
        )
        directive = context.config.selection.directive(
            prior_art_run_record_id=context.facts.require(
                Fact.PRIOR_ART_RUN_RECORD_ID
            )
        )
        result = selector.run(directive)
        record = result.run_record
        return _selection_outcome(
            record.id,
            record.outcome,
            _spend(
                record.model_calls, record.input_tokens, record.output_tokens
            ),
        )

    def refusals(self) -> tuple[type[Exception], ...]:
        return (MissingChallengedPortfolioError,)


class AdmissionStage(_SingleAttempt):
    """The selected candidate, translated into an initial state."""

    @property
    def name(self) -> StageName:
        return StageName.ADMISSION

    def plan(self, context: StageContext) -> StagePlan:
        directive = context.config.admission.directive(
            selection_run_record_id=context.facts.require(
                Fact.SELECTION_RUN_RECORD_ID
            )
        )
        return StagePlan(
            key=f"{self.name}:{directive.id}", subject_id=directive.id
        )

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None:
        record = context.stores.admission.record_for_directive(plan.subject_id)
        if record is None:
            return None
        return StageOutcome(
            produced=_sorted(
                {
                    Fact.ADMISSION_RECORD_ID: record.id,
                    Fact.ADMITTED_STATE_ID: record.state_id,
                }
            ),
            spend=_spend(
                record.model_calls, record.input_tokens, record.output_tokens
            ),
            detail=f"admitted state {record.state_id}",
        )

    def execute(self, context: StageContext) -> StageOutcome:
        admitter = CandidateAdmitter(
            provider=context.lab.model_provider(self.name),
            model=context.config.model,
            ledger=UsageLedger(),
            ideation_store=context.stores.ideation,
            prior_art_store=context.stores.prior_art,
            selection_store=context.stores.selection,
            store=context.stores.admission,
            request_timeout_seconds=context.config.request_timeout_seconds,
        )
        directive = context.config.admission.directive(
            selection_run_record_id=context.facts.require(
                Fact.SELECTION_RUN_RECORD_ID
            )
        )
        result = admitter.run(directive)
        return StageOutcome(
            produced=_sorted(
                {
                    Fact.ADMISSION_RECORD_ID: result.record.id,
                    Fact.ADMITTED_STATE_ID: result.state.id,
                }
            ),
            spend=_spend(
                result.record.model_calls,
                result.record.input_tokens,
                result.record.output_tokens,
            ),
            detail=(
                "replayed a completed admission"
                if result.replayed
                else f"admitted state {result.state.id}"
            ),
        )

    def refusals(self) -> tuple[type[Exception], ...]:
        return (AdmissionRefusedError,)


class FundingStage(_SingleAttempt):
    """The one stage no model touches: an operator grant becomes a run.

    It also copies the admitted seed and its funded successor into the
    run root's own state store. Both, not just the funded one: the
    funded state names the admitted state as its parent, and a run root
    whose oldest snapshot points at a state nobody can find has a
    lineage that does not terminate.
    """

    @property
    def name(self) -> StageName:
        return StageName.FUNDING

    def plan(self, context: StageContext) -> StagePlan:
        directive = self._directive(context)
        return StagePlan(
            key=f"{self.name}:{directive.id}", subject_id=directive.id
        )

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None:
        run = context.stores.program.run_for_directive(plan.subject_id)
        if run is None:
            return None
        return self._outcome(context, run.run_id, run.funded_state_id)

    def execute(self, context: StageContext) -> StageOutcome:
        admission_record_id = context.facts.require(Fact.ADMISSION_RECORD_ID)
        authorization = context.config.funding.authorization(
            admission_record_id=admission_record_id
        )
        result = start_run(
            admission_store=context.stores.admission,
            program_store=context.stores.program,
            directive=self._directive(context),
            authorization=authorization,
        )
        return self._outcome(
            context,
            result.run.run_id,
            result.funded_state.id,
            replayed=result.replayed,
        )

    def refusals(self) -> tuple[type[Exception], ...]:
        return (RunRefusedError, RunPreflightError)

    def _directive(self, context: StageContext) -> RunDirective:
        admission_record_id = context.facts.require(Fact.ADMISSION_RECORD_ID)
        authorization = context.config.funding.authorization(
            admission_record_id=admission_record_id
        )
        return context.config.funding.directive(
            admission_record_id=admission_record_id,
            authorization_id=authorization.id,
        )

    def _outcome(
        self,
        context: StageContext,
        run_id: str,
        funded_state_id: str,
        *,
        replayed: bool = False,
    ) -> StageOutcome:
        program_states = context.stores.program.state_store()
        funded = program_states.load(funded_state_id)
        if funded.parent_id is not None:
            context.stores.states.persist(
                program_states.load(funded.parent_id)
            )
        context.stores.states.persist(funded)
        return StageOutcome(
            produced=_sorted(
                {
                    Fact.RUN_ID: run_id,
                    Fact.FUNDED_STATE_ID: funded_state_id,
                    Fact.STATE_ID: funded_state_id,
                }
            ),
            detail=(
                f"replayed run {run_id}" if replayed else f"funded run {run_id}"
            ),
        )


class ExperimentationStage:
    """One deliberation at a time, each keyed by the state it starts from.

    A step is the unit because it is the smallest thing that is durable
    on its own: the runtime persists the successor snapshot before the
    step returns, the ledger keys its debit by the attempt, and both
    survive the process. So a crash costs the step in flight and nothing
    else, and the reconcile is a question about snapshots rather than a
    guess about intent.
    """

    @property
    def name(self) -> StageName:
        return StageName.EXPERIMENTATION

    def plan(self, context: StageContext) -> StagePlan:
        state_id = context.facts.require(Fact.STATE_ID)
        return StagePlan(key=f"{self.name}:{state_id}", subject_id=state_id)

    def completed(
        self, context: StageContext, plan: StagePlan
    ) -> StageOutcome | None:
        """Ask the attempt journal what the last process left owing.

        Task 6C answered this by inference — walk the lineage down and
        adopt the deepest descendant whose attempts were all resolved —
        which is a guess about a shape rather than a fact about a run. It
        could not tell an attempt that never started from one that ran an
        experiment and died before recording it, and those owe very
        different things.

        Now the journal says. Every attempt it left open is settled and
        closed before the chain steps again, so what this returns is a
        state the run genuinely owes nothing from.
        """
        run_id = context.facts.require(Fact.RUN_ID)
        report = recover(
            journal=context.stores.program.journal_for(run_id),
            ledger=context.stores.program.ledger_for(run_id),
            bundles=context.stores.program.bundles(),
            states=context.stores.states,
            evidence=context.stores.evidence,
            fallback_state_id=plan.subject_id,
        )
        if not report.anything_to_do:
            return None
        return StageOutcome(
            produced=_sorted({Fact.STATE_ID: report.state_id}),
            detail=report.summary(),
            ends_investigation=(
                "an attempt spent more than it was authorized to"
                if report.breached
                else ""
            ),
            repeat=not report.breached,
        )

    def execute(self, context: StageContext) -> StageOutcome:
        run_id = context.facts.require(Fact.RUN_ID)
        runtime = context.lab.runtime(
            RuntimeRequest(
                root=context.stores.root,
                evidence=context.stores.evidence,
                states=context.stores.states,
                ledger=context.stores.program.ledger_for(run_id),
                journal=context.stores.program.journal_for(run_id),
                bundles=context.stores.program.bundles(),
            )
        )
        state = context.stores.states.load(context.facts.require(Fact.STATE_ID))
        report = runtime.step(state)
        halt = report.halt_reason
        return StageOutcome(
            produced=_sorted({Fact.STATE_ID: report.state.id}),
            spend=_spend(
                report.provider_usage.calls,
                report.provider_usage.input_tokens,
                report.provider_usage.output_tokens,
            ),
            detail=halt if halt is not None else f"step to {report.state.id}",
            ends_investigation=f"the runtime halted: {halt}" if halt else "",
            repeat=halt is None,
        )

    def refusals(self) -> tuple[type[Exception], ...]:
        return (ExperimentationUnavailableError,)

    def limit(self, config: RunConfig) -> int:
        """``max_steps`` bounds one walk, exactly as it bounds one call
        to :meth:`ResearchRuntime.run`. Resuming grants another walk;
        what bounds the investigation as a whole is the budget."""
        return config.experimentation.max_steps


def default_chain() -> tuple[Stage, ...]:
    """The chain, in the one order it can happen in."""
    return (
        MappingStage(),
        IdeationStage(),
        PriorArtStage(),
        SelectionStage(),
        AdmissionStage(),
        FundingStage(),
        ExperimentationStage(),
    )


# -- helpers ------------------------------------------------------------------


def _selection_outcome(
    record_id: str, outcome: SelectionOutcome, spend: StageSpend
) -> StageOutcome:
    ends = (
        ""
        if outcome is SelectionOutcome.SELECTED
        else f"selection ended in {outcome}"
    )
    return StageOutcome(
        produced=_sorted({Fact.SELECTION_RUN_RECORD_ID: record_id}),
        spend=spend,
        detail=str(outcome),
        ends_investigation=ends,
    )


def _spend(
    model_calls: int, input_tokens: int, output_tokens: int
) -> StageSpend:
    return StageSpend(
        model_calls=model_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _sorted(produced: dict[Fact, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(fact), value) for fact, value in produced.items()))
