# Architecture

## Design philosophy

A language model is fluent enough to produce everything that *looks* like
research — a crisp hypothesis, a plausible method, a confident result, a tidy
conclusion — without any of it being anchored to something that was measured.
That is the dominant failure mode of an autonomous research system, and it is
not fixed by better prompting. It is fixed by making unsupported output
structurally difficult to produce.

Six consequences shape everything below.

**The state of the research is data, not conversation.** Anything a decision
depends on lives in a structured `ResearchState`. Conversation history is a
working medium, not a system of record.

**Facts and beliefs are stored differently.** An `ExperimentResult` can only be
created by an executor that ran a process, and is append-only. Beliefs —
assessments of hypotheses and claims — are versioned, revisable, and always
link back to the facts they rest on.

**Propositions are not beliefs about them.** A hypothesis, a prediction, a
claim — these are scientific propositions, and they carry no truth status.
What is currently believed about a proposition is a separate, versioned
judgment (`EpistemicAssessment`); what an execution observed about a
prediction is a separate mechanical record (`PredictionTest`). Nothing ever
writes a verdict onto the thing being judged.

**Research is a search, not a pipeline.** Typed actions over a state space,
chosen by a policy, rather than stages hard-coded in sequence. Real research
backtracks, replicates, abandons, and re-scopes.

**Negative results must survive.** Failed hypotheses, inconsistent prediction
tests, failed *attempts* — each is recorded with the same fidelity as
success. A system whose every outcome converts into apparent progress is not
doing science.

**Evidence is not interpretation.** How an observation relates to a claim is a
factual annotation; whether the claim should be believed is a judgment with its
own object, author, method, and version history. No count of evidence edges
produces a verdict anywhere in this codebase.

## The scientific chain

```
ResearchQuestion      what is being asked, and why it matters
    ↓
Hypothesis            a general statement answering a question   (no status)
    ↓
Prediction            what it commits to observably: metric, comparator,
    ↓                 threshold, condition — fixed before any run (no status)
ExperimentSpec        the design that will test the prediction
    ↓
ExperimentJob         one execution event, bound to a backend
    ↓
ExperimentResult      the immutable record of what happened
    ↓
PredictionTest        what THIS execution observed w.r.t. the prediction:
    ↓                 consistent / inconsistent / inconclusive, mechanically
Evidence              a factual reading of the result
    ↓
EvidenceRelation      supports / contradicts / inconclusive, per claim
    ↓
EpistemicAssessment   the judgment: verdict, confidence, method, scope
```

The load-bearing separation is between the middle and the ends. A
`Prediction` is a *proposition*: it never mutates, and there is deliberately
no `PredictionStatus`. When a result commits, the transition layer compares
the observed metric against the pre-registered comparator and threshold and
records a `PredictionTest` — one per (prediction, result) pair. Four runs
yield four tests, and a mixed record

```
Run 1 → consistent
Run 2 → consistent
Run 3 → inconsistent
Run 4 → inconclusive
```

is preserved as exactly that: four coexisting facts. The check is arithmetic,
fixed before the run — it cannot be adjusted to fit the outcome, and no
role's opinion enters into it.

What those tests *mean* for the hypothesis — wrong theory, broken auxiliary
assumption, bad instrument — is a different question, answered only by an
`EpistemicAssessment` that names its method and the evidence it considered.
Current standing is always a query (`ResearchState.current_assessment`,
`ResearchState.tests_for`), never a field on the proposition. An earlier
design cached lifecycle statuses on `Hypothesis` and `Prediction`; both were
removed, because a cached standing on the proposition is exactly the shortcut
— belief mutating the fact it is about — that this ontology exists to forbid.

`ResearchQuestion` sits above hypotheses as the unit of scientific intent: a
hypothesis names the question it attempts to answer, which gives utility an
explicit conditioning target — `U(action | state, question)` — without a
heavier `ResearchProgram` abstraction that Horizon 1 does not need.

## The action lifecycle and the commit boundary

```
ResearchAction        scientific intent          (semantic identity)
    ↓
ActionAttempt         one try at executing it    (occurrence identity)
    ↓
CommitBundle          the attempt's entire effect: proposals + outcome
    ↓
atomic transition     all of it commits, or none of it
```

Lifecycle: `QUEUED → RUNNING → SUCCEEDED | FAILED | CANCELLED | TIMED_OUT`.

Two invariants, each enforced rather than documented:

**A failed attempt never makes work look done.** Whether an action is complete
is a question about attempts with *succeeded* outcomes
(`ResearchState.has_succeeded`); a failed attempt leaves the work open, and a
retry is a new attempt carrying the same action. All attempts — failed ones
included — stay in the state.

**A successful action cannot claim outputs that do not exist.** An attempt's
proposals and its outcome travel together in a `CommitBundle`, and
`commit_bundle` is the only way to resolve an attempt through the transition
layer: it validates the attempt's lifecycle, commits every proposal (or
rejects the lot — a bundle that half-validates changes nothing), checks that
every id in `outcome.produced` was actually committed by the bundle
(mechanically created prediction tests included), and resolves the attempt —
in one step. A bundle whose outcome is not `SUCCEEDED` carries no proposals
and claims no products, by construction.

This is a domain-level commit boundary over an immutable state, not a
distributed transaction system. One consequence is documented rather than
hidden: the evidence store is append-only and idempotent, so a result recorded
during a transition that subsequently fails remains recorded — a fact without
a referencing state, which is honest (the process really ran). Atomicity is a
guarantee about *state membership*, where beliefs could otherwise be
corrupted, not about the existence of facts.

`ResearchState.history` still exists, as an audit trail of selected actions.
Nothing operational reads it, and a test pins that candidate generation
re-offers work whose action appears in history but never succeeded.

## Identity semantics

Two kinds of identifier, for two kinds of question (`core/ids.py`):

* **Content ids** (`content_id`) for semantic objects — questions, hypotheses,
  predictions, specs, claims, evidence, replication groups. Identical content
  → identical id, so the same hypothesis constructed on two branches or in two
  runs is the same hypothesis, and trajectories are comparable object by
  object.
* **Occurrence ids** (`occurrence_id`) for events — attempts, jobs, decisions,
  role invocations. Identical construction → distinct ids. Two identically
  configured executions of one spec are two events; a replication is not its
  original. Executors refuse to submit the same job object twice.

Objects downstream of an execution inherit its event identity: a result's id
derives from its job's, evidence derives from its result, and a
`PredictionTest` embeds the result id it tested against. Two identical runs
therefore agree on every purely semantic object and disagree on every
event-derived one — which is the correct statement of what happened.

## Replication semantics

Independent executions are independent evidence — identity rules make merging
them impossible. The converse question, *which results were testing the same
thing*, is answered by `ReplicationGroup` (`core/replication.py`): a
content-addressed grouping key derived from the experiment spec and the run
configuration, and from nothing else.

Deliberate exclusions from the grouping key:

* **observed metrics** — grouping by outcome would sort contradictory
  replications into different families, which is precisely the moment they
  most need to be seen together;
* **seed** — runs of one protocol under different seeds are statistical
  replications of the same thing (each member keeps its seed);
* **environment** — platform and commit are provenance, recorded per result;
  splitting families by machine would hide cross-machine disagreement, which
  is a finding, not a grouping error.

`group_replications(results)` buckets results by family. A future
statistician role receives the family, not a pre-aggregated summary.

## Core domain objects

All of these live in `core/`, which imports nothing from its sibling packages
(enforced by `tests/test_layering.py`).

| Object | Role |
| --- | --- |
| `ResearchState` | The authoritative state of a research program |
| `ResearchQuestion` | What is being asked, and why it matters |
| `Hypothesis` | A proposition answering a question; no status |
| `Prediction` | A pre-registered, machine-checkable commitment; no status |
| `PredictionTest` | What one execution observed w.r.t. one prediction |
| `ResearchAction` / `ResearchActionType` | A typed scientific intent |
| `ActionAttempt` / `ActionOutcome` | One execution of an intent, and how it ended |
| `CommitBundle` | An attempt's entire effect, committed atomically |
| `ExperimentSpec` | The scientific design testing one prediction |
| `ExperimentResult` / `Environment` | Immutable execution record + provenance |
| `ReplicationGroup` | Protocol identity shared by replicated executions |
| `Evidence` / `EvidenceLink` | Factual reading of a result; its bearing on a claim |
| `Claim` | A scoped assertion; carries no status and no numbers |
| `EpistemicAssessment` | A versioned judgment: verdict, confidence, method |
| `ActionCandidate` / `ActionUtility` / `EvaluatedCandidate` | The decision vocabulary |
| `DecisionRecord` | One orchestration decision, preserved end to end |
| Proposals (`HypothesisProposal`, …) + `ProposalKind` | Attributable requests to change state |
| `ResearchBudget` / `ResourceCost` | What may be spent; what things cost |

`ResearchState` is immutable and lineage-carrying: every mutation returns a new
state whose `parent_id` points at its predecessor. States hold propositions
and judgments, and hold *references* to facts — results and evidence live in
the append-only store, shared across branches, because a fact does not become
a different fact on a different branch of the search.

The core is plain frozen dataclasses with no runtime dependencies. Validation
libraries stay at the future LLM boundary: provider output will be validated
against provider-specific schemas and *translated* into these types, not
replace them.

### On decomposing `ResearchState`

The state is deliberately still one object. Splitting it now (scientific /
execution / epistemic sub-states) would add indirection with no invariant
behind it. The split should happen when the first of these becomes true:

1. **concurrent programs** — multiple research programs share a process and
   need independent lifecycles over common facts;
2. **role-context builders duplicate projections** — if per-role views end up
   re-deriving the same sub-state slices, those slices have earned names;
3. **state hashing or copying shows up in profiles** — content-id computation
   over the full state stops being negligible;
4. **field count obscures the commit surface** — mutators stop fitting on one
   screen and reviewers can no longer see what a transition may touch.

Until then, one dataclass with disciplined mutators is easier to audit than
three objects with a coordinator.

## The decision loop

Three questions, three separated functions, one wiring class:

```
What could we do?            CandidateGenerator  -> ActionCandidate[]
How valuable might each be?  UtilityEvaluator    -> ActionUtility[]
What do we take, given
uncertainty and resources?   SearchPolicy        -> selected candidate
```

`ResearchDirector` composes the three and emits a `Decision` whose
`DecisionRecord` preserves the full tuple — state before, every candidate with
its utility, generator/evaluator/policy names, selection — completed later with
the attempt, its outcome, the assigned role when routing happened, actual
cost, and the state after.

The invariant: **scientific utility is not search policy.** `ActionUtility` is
multi-dimensional (information gain, discrimination value, importance, novelty,
replication value, success probability, expected cost, and the estimate's own
uncertainty) and collapses to a scalar only inside a policy, explicitly, where
the collapse can be seen and criticised. A bandit and a greedy policy consuming
identical utilities are different explorers, not different opinions about
science. `None` in any dimension means *not estimated*, and no consumer may
read it as zero.

Current implementations are deliberately modest: a rule-based generator that
closes structural gaps (a hypothesis without predictions, a prediction without
an experiment, a result nobody analyzed — completion read from succeeded
attempts, never from history); a heuristic evaluator with fixed per-action-type
profiles that flag themselves as maximally uncertain; a greedy value-per-cost
policy. Each is named in every decision record, so their estimates remain
attributable — and dismissible — later.

Because propositions carry no status, the generator *consumes* epistemic
judgments where it needs standing: a hypothesis whose current assessment is
SUPPORTED or REFUTED is treated as settled and not offered further work. That
is a generation policy reading assessments made elsewhere — not the generator
doing epistemology.

The generator offers `stop_investigation` only when nothing else is open:
halting is a legitimate outcome, and a free stop action would dominate any
value-per-cost ranking as a standing candidate.

## Runtime philosophy

> The framework intentionally separates rich internal scientific
> representation from runtime agent count. A domain abstraction does not
> imply an LLM call or autonomous agent.

> Complexity should be added only when it addresses an observed failure mode
> or produces measurable improvements in scientific quality, reliability, or
> efficiency.

> Ceteris paribus, simpler mechanisms are preferred.

The ontology above is deliberately rich — seventeen action types, four
kinds of proposition, versioned judgments — because cheap representation is
what makes honest science checkable. None of that entitles the runtime to
spend. The runtime optimizes, approximately,

```
meaningful scientific progress / (wall-clock + compute + inference cost)
```

and it does so by observing a ground-truth hierarchy:

```
executed result / artifact
  > deterministic validation
    > artifact-grounded independent judgment
      > LLM opinion
```

An LLM is never asked to infer what code can determine. Concretely, per
ordinary experiment iteration the **reasoning-invocation** count is a
design invariant the loop enforces:

```
Director: 1        one deliberate() — candidates, valuations, selection
Executor: 1        one perform() implements/runs the assignment
Critic:   0        deterministic checks stand in for routine critique
```

A *scientifically consequential* result (contradictory replication,
challenged standing, unexpectedly large effect, or an explicit director
request) adds exactly one critic invocation — decided by a deterministic
trigger, never by a model.

Invocations are not model calls, and the accounting refuses to conflate
them. What the loop can enforce is how many times it invokes a reasoning
seat; how many provider calls a model-backed role makes *inside* one
invocation is the adapter's affair, recorded separately from
provider-reported usage (``UsageSource`` → ``provider_calls`` / tokens in
the step metrics) and honestly zero for today's rule-based roles. The
runtime makes no claim to constrain provider-internal calls.

The runtime is equally strict about paying for work: an action whose
estimated cost exceeds the remaining budget is never started, and actual
cost is reconciled after commit — an overrun is billed as far as the budget
reaches, recorded explicitly, and halts the program. Committed-but-unbilled
work cannot exist.

### The frontier: what the director actually sees

`runtime/frontier.py` derives a `ResearchFrontier` — open questions, active
hypotheses, work queues read from facts and succeeded attempts,
contradictions, failed attempts worth revisiting, current best findings,
remaining budget — from one `ResearchState`. It is a **view**: pure
function of the state, no mutators, never persisted as authority, carrying
the id of the state it projects. It exists to keep director prompts small
and stable, and to give context selection one seam to grow behind.

### The director fast path

The runtime default is one reasoning seat, `FrontierDirector`: a single
deliberation performs candidate generation, coarse valuation and selection
together, and `deliberation_record` preserves the intermediate candidate
set in the standard `DecisionRecord` — with the director named as
generator, evaluator *and* policy, which is what happened. The decomposed
`ResearchDirector` (generator → evaluator → policy) remains in the
architecture for the ablation; the runtime never requires its three
potential model calls per step.

Valuations at runtime are deliberately coarse: HIGH / MEDIUM / LOW value
and uncertainty plus an expected `ResourceCost` (`CompactValuation`),
embedded into the rich `ActionUtility` as a named ordinal mapping. The
runtime does not manufacture `novelty = 0.73` until calibration data says
such numbers mean something. Where ranking is ambiguous the rule-based
director records pairwise "prefer A over B because ..." rationale rather
than absolute scores, and the raw reasoning is preserved in the runtime
metrics.

### Default search: directed refinement

The loop's default is diagnose → modify → execute → observe, one branch at
a time. There is no MCTS, no best-first tree search, and no default
branching; the tree exists as *memory* (state lineage, content-addressed
snapshots) rather than as a mandatory algorithm. Branching should return
when there is a scientific reason — competing hypotheses, materially
distinct strategies — and calibrated value estimates to steer it.

### The deterministic validation gate

`runtime/validation.py` checks what machines can check: the process exited
zero, the result names its spec and matches the director's assignment,
declared metrics are present and finite, the seed is recorded, artifacts
hash to the manifest their run wrote, and replications agree within stated
tolerance. The runtime applies these checks — artifact integrity included —
as a **pre-commit gate**: a completed result that fails any of them never
enters `state.results`, never produces a prediction test, and never becomes
evidence. The attempt fails as an *engineering failure*, the run directory
is preserved for diagnosis, and the director sees a deterministic note on
the next frontier. No critic is consulted and no model may overrule the
gate: arithmetic is not a matter of opinion. Cardinality is part of the
gate too — a run-type assignment must return exactly one result.

Failed or cancelled *executions* are different: they commit as honest
execution-failure records with mechanically inconclusive standing, are
deterministically diagnosed, and — when a debugger is wired in — enter the
bounded repair loop (below); repeated failures of one experiment (counted
from the results themselves) raise the same kind of deterministic
engineering note — a debugging signal for the director, not scientific
critique.

The executor itself also refuses silent success: exit-zero without metrics,
a non-finite metric, or a missing declared artifact is a recorded failure
with the run directory preserved. Reading a gate-valid completed result
into `Evidence` is a transcription, not a judgment, so
`evidence_from_result` does it in code — reusing the mechanical
`PredictionTest` the commit already produced.

### Scientific debugging and experiment verification

The runtime distinguishes five ways an experiment can end, and gives each
its own detection and its own response:

```
ENGINEERING FAILURE       crash, timeout, OOM, contract breach → repair execution
IMPLEMENTATION FAILURE    runs, but is not the intended        → verify / debug /
                          experiment (silent bug)                reimplement
METHODOLOGICAL FAILURE    correct code, wrong experiment       → redesign experiment
ANALYTICAL FAILURE        valid experiment, wrong inference    → redo analysis
VERIFIED                  all four dimensions resolved         → outcome is evidence
```

The core principle: **a bad result is not a bug, a successful process is
not a correct experiment, and a reproducible result is not automatically a
scientifically valid one.** Debugging optimizes for obtaining a *valid*
experiment, never for obtaining a positive result — the lab aggressively
repairs invalid experiments and is equally aggressive about preserving
disappointing results when the experiment itself is valid.

The pieces, all removable for ablation:

* **Validity model** (`runtime/verification.py`): every check carries an
  explicit state — `PASS` / `FAIL` / `UNCERTAIN` / `NOT_APPLICABLE`, never
  a manufactured confidence float — on one of four dimensions (execution,
  implementation, methodology, analysis). A `VerificationReport` collapses
  to an `ExperimentValidityStatus` with worst-dimension precedence, and
  `VERIFIED` requires a positive determination on *every* dimension; an
  axis nobody checked yields the explicit intermediate `UNVERIFIED` —
  outcome observed, validity unresolved. Validity is orthogonal to
  scientific outcome: `VERIFIED` plus an inconsistent prediction test is a
  valid scientific negative, while `IMPLEMENTATION_UNCERTAIN` plus the same
  test is a debugging question, not negative evidence.
* **Failure classifier** (`execution/failure_classifier.py`): deterministic
  first-pass diagnosis of failed executions from the executor's structured
  failure reason and preserved stderr — timeout, launch, OOM, import,
  missing path, missing/malformed metrics, missing artifact. Conservative
  (`UNKNOWN`/`UNCERTAIN` when signals are ambiguous) and structurally blind
  to science: a completed run is `NONE` no matter what its metrics say.
* **Bounded repair loops** (`orchestration/debug_loop.py`), two
  structurally separate entries over shared machinery. *Execution repair*:
  diagnose a failed process → propose repair (with its rationale) → rerun
  as a *new* job → stop at `max_debug_attempts`; entry is by failure
  diagnosis only — the debugger *raises* on a completed result.
  *Implementation repair*: a **completed** run may be repaired, but only
  through an `ImplementationRepairTrigger`, whose constructor accepts
  nothing but implementation-dimension verification checks with at least
  one FAIL (a failed positive control, a verifier FAIL, a deterministic
  invariant violation). A prediction test, a small effect, or an
  underperforming baseline cannot be expressed as such a trigger, so
  `while result_is_scientifically_bad: debug()` cannot be written against
  either entry. Every retry, on either path, is a separate auditable
  `DEBUG` attempt committed through the same validation gate and billed at
  its actual cost; the original invalid result and its verification record
  are never deleted or rewritten, and a reimplementation must *earn* its
  own fresh verification — repair resolves only when the new run's
  implementation dimension no longer fails, which still says nothing about
  its scientific outcome.
* **Preflight** (`runtime/preflight.py`): cheap deterministic pre-execution
  checks (command resolves, declared input paths exist, seed propagated)
  behind a small extensible interface; a failed check prevents the launch
  and bills nothing. Checks decide their own applicability — nothing is
  assumed universal.
* **Positive controls** (`PositiveControl`): experiment-specific invariants
  a faithful implementation must satisfy (tiny set overfits, zero learning
  rate changes nothing, a known probe reads exactly right), evaluated
  deterministically against reported metrics. A failed control makes the
  result implementation-*uncertain*; it is never read as a scientific
  negative, because it tests the instrument, not the hypothesis. Controls
  live outside `core/` and are supplied per spec.
* **Selective implementation verification**: a semantic hunt for silent
  bugs (wrong loss, leaked data, bad split, wrong baseline), event-
  triggered — on a failed/uncertain control, or on a conclusive negative
  with no control coverage — never on every run. The hook is a protocol;
  `orchestration/review.py` adapts an existing role to it via a `FALSIFY`
  invocation (no new agent, no new action type), and the review verdict
  feeds a check state without ever committing to scientific state.
* **Methodology gate**: each design is reviewed once, before its first
  execution — *even perfectly implemented, would this experiment answer the
  question?* A rejected design never runs; the director sees `REDESIGN
  EXPERIMENT`, explicitly not "debug" and not a recorded negative.
* **Analysis validity**: raw results are distinguished from downstream
  inference. A deterministic coverage guard rejects judgments citing only
  part of the conclusive evidence available to them (post-hoc run
  selection); the response is *redo the analysis*, never rerun the valid
  experiments beneath it.
* **Negative-result gate**: a conclusive negative becomes strong scientific
  evidence only when execution, implementation, methodology and analysis
  are all positively resolved. Anything less preserves the observation in
  the explicit observed-but-unresolved state — and under no status is a
  result routed to debugging merely for being negative (pinned by test).
* **Durable verification records** (`runtime/verification_store.py`): every
  verified result's report, validity and standing become a record keyed by
  result id — id never maps to different content, verdicts are never
  rewritten, and repair produces a new result with a new record. In-memory
  and one-JSON-file-per-record implementations; absence of a record marks
  a result that was run with verification ablated.
* **The scientific-promotion gate** (in the runtime loop, pre-commit):
  raw observation ≠ verified scientific support. A SUPPORTS/CONTRADICTS
  evidence link or a conclusive assessment may cite evidence only when the
  durable record behind it stands at `VERIFIED_EVIDENCE`; anything less is
  rejected before commit — deterministically, from the record, with no
  model consulted. Inspection is never blocked: unresolved observations
  stay in the evidence store, may be cited as INCONCLUSIVE, may ground
  UNDETERMINED assessments, and reach reasoning seats annotated with their
  standing through context notes. Results with no record (the ablated lab)
  keep legacy semantics.

Deterministic checks always outrank semantic review: dimension aggregation
treats any deterministic `FAIL` as final, so no model verdict can wash out
a failed control or a broken artifact hash.

An honest limit, stated rather than implied: no general system can prove
the total absence of silent scientific bugs. Reliability here comes from
layered deterministic checks, experiment-specific controls, selective
independent judgment, and — later — replication; the verification record
says which layers actually ran, so an unverified result is never mistaken
for a verified one.

### Deterministic routing

Three runtime seats — scientist (`RESEARCH_DIRECTOR`), executor
(`RESEARCH_ENGINEER`), critic/analyst (`RESULT_ANALYST`) — and a static
table (`orchestration/routing.py`) mapping every action type to one of
them, plus the proposal kinds each action may return. Nothing about the
mapping is uncertain today, so nothing about it is inferred.
`RoleSuitability` and `RegistryAssigner` stay for the day routing has
empirical calibration behind it.

Worker lifetime is separated from lab lifetime: the director is
long-lived; every role invocation carries an explicit `RoleContext`
projection built per assignment — an executor sees a spec and its prior
runs, never the research history.

### Two timescales, one director

The fast loop optimizes throughput. The slow loop —
`FrontierDirector.synthesize`, the same seat at a stronger reasoning tier,
not a second agent — reviews what has actually been learned and recommends
continue / replicate / pivot / branch / stop. Its cadence is deterministic
(`SynthesisTrigger`): every N committed results, when a contradiction
appears, and before stopping. Its recommendation reaches the next
deliberation through the frontier's `open_decisions`.

### Cost-aware escalation

`runtime/escalation.py` encodes the spending ladder —

```
Tier 0  deterministic code
Tier 1  cheap model / routine reasoning
Tier 2  strongest model for difficult decisions
Tier 3  multi-sample / debate, only when justified
```

— and a small rule table (`EscalationPolicy`) that picks the cheapest
sufficient tier from decision importance, uncertainty, downstream resource
commitment, and evidence conflict. It is not a model router; it is the
place the principle lives so a router could one day replace it measurably.

### Evidence-chain validation

`evidence/validation.py` re-derives the whole chain — question →
hypothesis → prediction → spec → result → test / evidence → assessment —
against the state and store, deterministically: dangling references,
facts missing from the store, prediction tests whose recorded observation
or verdict disagrees with a mechanical re-check (the one place belief
could quietly rewrite fact), claims without evidence, conclusive
assessments citing none, and contradictions surfaced as facts. It is a
query layer over existing objects; there is no graph database.

### Held-out evaluation

`runtime/evaluators.py` is only a seam, but a guarded one: a development
evaluator the loop may consult freely, and a held-out evaluator that
demands an explicit, recorded release — the autonomous loop holds no
credential, which is how evaluator overfitting is prevented structurally
rather than by policy.

### Measured, removable complexity

Every optional mechanism hangs off a typed flag in `RuntimeConfig` —
critic on/off, playbook on/off, synthesis on/off, cheap/strong director
floor, the repeated-failure threshold, and the verification family
(`debug_enabled` + `max_debug_attempts`, `preflight_enabled`,
`methodology_review_enabled`, `implementation_verification_enabled`,
`positive_controls_enabled`) — and every step writes a
`StepMetrics` record: reasoning invocations, provider-reported calls and
tokens (zero without a provider), wall-clock, experiment compute, whether
the critic fired and why, the reasoning tier, the outcome, deterministic
runtime notes, the raw decision rationale — plus the verification record:
failure category, debug attempts and whether debugging recovered a valid
execution (never conflated with scientific success), validity status,
preflight/control/methodology/implementation/analysis rejections, and
whether a conclusive negative was accepted as evidence or deferred. The eventual research
contribution is a measurement of
which components earn their cost; the flags and the records are how that
measurement stays possible. Playbooks (`runtime/playbook.py`) follow the
same rule: an advisory prior over what usually comes next in empirical ML
— never a stage machine, never checked for compliance.

## Roles: suitability, invocation, and the proposal invariant

A role is a quadruple: **objective**, **information set**, **allowed
actions/tools**, and **output contract** — not a system prompt. Two roles on
the same foundation model are different agents if those differ.

Two value concepts, deliberately not one:

```
ActionUtility     U(a | state)                        "Should the lab perform
                                                       this action?"
RoleSuitability   ≈ P(role succeeds | action, state)  "Who should perform this
                                                       selected action?"
```

`RoleSuitability` (in `roles/`) deliberately does not use the word *utility*:
it expresses no opinion about whether an action is worth performing.
Assignment (`RegistryAssigner`) runs strictly *after* the search policy has
selected an action, and suitability never feeds back into selection — so
"what is scientifically valuable" cannot quietly become "what our current
roles happen to be good at".

Work reaches a role as a `RoleInvocation` (occurrence identity):

```
RoleInvocation
    role              who
    assignment        the selected ResearchAction
    context           RoleContext — an explicit projection, never raw state
    allowed_actions   what this invocation may do
    expected_output   which ProposalKinds may come back (checkable per
                      proposal via invocation.permits)
    budget            the most this invocation may spend
```

`RoleContext` is a typed container of selected domain objects with every field
defaulting to empty: the orchestrator includes exactly what the invocation
needs, and what a role was shown is thereby *recorded* rather than implied. A
hypothesis researcher sees the question, hypotheses, and negative findings; a
skeptic sees one hypothesis, its predictions, its tests, and alternatives; an
engineer sees a spec and execution constraints; a statistician sees raw
results, the design, and the replication group. Rendering a context for a
model is a provider-boundary concern that does not exist yet — deliberately.

**Roles never mutate `ResearchState`.** A role reads its context and produces
typed proposals — `QuestionProposal`, `HypothesisProposal`,
`PredictionProposal`, `ExperimentProposal`, `EvidenceProposal`,
`ClaimProposal`, `AssessmentProposal`, and (executors only) `ResultProposal` —
each naming its proposer. Only the transition layer commits:

```
role -> proposals -> CommitBundle -> validation / atomic commit -> ResearchState'
```

This is enforced twice: structurally (an AST test forbids any module in
`roles/` from calling a state mutator or importing the transition layer) and
behaviourally (transition tests reject orphaned references and unproduced
claims). The payoff is provenance, auditability, safe search branching, and
one place for conflict resolution when multiple agents propose concurrently.

## Evidence model

The `EvidenceStore` invariant is one sentence: **an id never maps to different
content.** Re-recording identical content is a no-op; re-recording different
content raises. Evidence referencing an unrecorded result is rejected.

Metrics enter the system exactly one way: an experiment process writes
`metrics.json`, and the executor reads it. A process that exits zero without
writing metrics is recorded as a failure, because treating a silent run as
success is precisely how empty experiments become reported findings.

## Claims and assessment

A `Claim` carries no status. Its factual support is the set of `EvidenceLink`
edges (supports / contradicts / inconclusive); its standing is the latest
`EpistemicAssessment` targeting it. Assessments are versioned by supersession —
a change of mind is a new assessment naming the one it replaces — and every
assessment records the evidence it actually considered and the **method** that
produced it, because a judgment that cannot say how it was reached cannot be
challenged.

`ClaimEvidenceGraph` (in `knowledge/`) is the factual read model: evidence per
claim by relation, claims without evidence, contradicted claims, unassessed
claims. It offers no verdicts. An earlier draft had an "advisory"
status-suggestion helper; it was removed, because anything that maps edge
counts to a status will eventually be treated as authoritative epistemology
regardless of its docstring.

Not yet built: *which experiment would most reduce uncertainty around this
claim?* That needs calibrated uncertainty, and there is nothing to calibrate
against until real trajectories exist.

## Persistence and trajectory logging

Every run persists two artifacts side by side, both local files, no database:

```
<run_root>/
├── states/
│   └── <state_id>.json      content-addressed ResearchState snapshots
├── trajectory.jsonl          one DecisionRecord per line
└── runs/                     executor run directories (logs, metrics, artifacts)
```

`FileStateStore` (in `persistence/`) serializes states deterministically —
same content, same bytes — so state ids double as snapshot filenames and
identical states deduplicate by construction. Loading reconstructs the full
domain object graph and *recomputes the content id from what it read*: a
snapshot that no longer hashes to its filename fails loudly instead of quietly
resurrecting a different state. The parsing code lives in `persistence`, not
`core` — `core.serialize` stays one-way because parsing needs validation, and
validation is a boundary concern.

Every orchestration decision is preserved as a `DecisionRecord`:

```
(state_t, {candidate_i, utility_i}, selected_t, role_t, attempt_t, outcome_t, state_{t+1})
```

plus the names of the generator, evaluator, and policy involved, predicted and
actual cost, and the ids of everything produced. `state_before_id` and
`state_after_id` are keys into the snapshot store, so the full decision tuple
reconstructs offline — a test walks every record of a demo run and reloads
both endpoint states.

This exists *now*, before any real research runs, because the questions this
project ultimately wants to answer — is utility calibrated? do specialized
roles help? does the search policy earn its cost? where does the loop
manufacture apparent progress? — are questions about these tuples, and
trajectories that only exist for later, successful designs cannot answer them.

## Execution

```python
submit(job) -> job_id      # each job submits at most once
status(job_id) -> JobStatus
collect(job_id) -> ExperimentResult
```

Shaped for asynchronous remote work even though the only backend is local and
synchronous: `submit` returns a handle, so no caller can assume the result is
already available. `ExperimentSpec` (science) and `ExperimentJob`
(commands, paths, env — an occurrence) stay separate objects; re-binding a spec
to a different backend touches no science.

The contract with an experiment process:

| Direction | Channel |
| --- | --- |
| lab → process | `ARL_RUN_DIR`, `ARL_CONFIG` (path to JSON), `ARL_SEED` |
| process → lab | `$ARL_RUN_DIR/metrics.json`, a flat JSON object of numbers |
| collected | anything else in the run directory, as artifacts |

`LocalExecutor` captures git commit, tree cleanliness, Python version,
platform, command, config, seed, logs, runtime and exit code on every run,
failures included.

## Architectural invariants

The list this pass was made against; each is enforced by at least one test.

```
Facts do not mutate with beliefs.
Scientific propositions do not carry truth status.
Evidence does not infer epistemic verdicts.
Roles do not mutate authoritative state.
Search policy does not define scientific utility.
Scientific utility does not determine role assignment.
Execution events have occurrence identity.
Semantic scientific specifications have content identity.
Independent replications remain independent evidence.
Successful outcomes cannot claim nonexistent outputs.
Every important research decision is reconstructible later.
```

## Separation of scientific reasoning from infrastructure

```
core          scientific vocabulary                    (no internal dependencies)
evidence      what happened, append-only + the chain validator
execution     how to make things happen, anywhere; deterministic
              failure classification
knowledge     what it all means, joined — factually; lesson scaffold
persistence   snapshots of states, reconstructible offline
runtime       frontier view, Tier-0 validation, experiment verification
              and preflight, tiers/escalation, metrics, playbooks,
              evaluation seam                         (depends on core only)
search        which move to take
roles         who does the work, under what contract
orchestration director, runtime loop, routing, triggers, bounded debug
              loop, role-backed review, atomic transitions, trajectory
publication   reporting  (empty)
```

Dependencies point downward only; `core` imports nothing from its siblings.
The infrastructure will be rewritten many times — new executors, new storage,
new providers — and the scientific vocabulary should not move when it is.
Equally, when the scientific model turns out to be wrong (it will), fixing it
should not require touching subprocess handling.
