# Architecture

## Design philosophy

A language model is fluent enough to produce everything that looks like
research — a crisp hypothesis, a plausible method, a confident result, a
tidy conclusion — without any of it being anchored to something that was
measured. That is the main failure mode of an autonomous research
system. Better prompting does not fix it. Making unsupported output
structurally hard to produce does.

Six consequences shape everything below.

**The state of the research is data, not conversation.** Anything a
decision depends on lives in a structured `ResearchState`. Conversation
history is a working medium, not a system of record.

**Facts and beliefs are stored differently.** An `ExperimentResult` can
only be created by an executor that ran a process, and is append-only.
Beliefs — assessments of hypotheses and claims — are versioned,
revisable, and always link back to the facts they rest on.

**Propositions are not beliefs about them.** A hypothesis, a
prediction, a claim — these are scientific propositions and carry no
truth status. What is currently believed about a proposition is a
separate, versioned judgment (`EpistemicAssessment`). What an execution
observed about a prediction is a separate mechanical record
(`PredictionTest`). Nothing ever writes a verdict onto the thing being
judged.

**Research is a search, not a pipeline.** Typed actions over a state
space, chosen by a policy — not stages hard-coded in sequence. Real
research backtracks, replicates, abandons, and re-scopes.

**Negative results must survive.** Failed hypotheses, inconsistent
prediction tests, failed attempts — each is recorded with the same
fidelity as success. A system whose every outcome converts into
apparent progress is not doing science.

**Evidence is not interpretation.** How an observation relates to a
claim is a factual annotation. Whether the claim should be believed is
a judgment with its own object, author, method, and version history. No
count of evidence edges produces a verdict anywhere in this codebase.

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
`Prediction` is a proposition: it never mutates, and there is
deliberately no `PredictionStatus`. When a result commits, the
transition layer compares the observed metric against the
pre-registered comparator and threshold and records a `PredictionTest`
— one per (prediction, result) pair. Four runs yield four tests, and a
mixed record

```
Run 1 → consistent
Run 2 → consistent
Run 3 → inconsistent
Run 4 → inconclusive
```

is preserved as exactly that: four coexisting facts. The check is
arithmetic, fixed before the run. It cannot be adjusted to fit the
outcome, and no role's opinion enters it.

What those tests mean for the hypothesis — wrong theory, broken
auxiliary assumption, bad instrument — is a different question,
answered only by an `EpistemicAssessment` that names its method and the
evidence it considered. Current standing is always a query
(`ResearchState.current_assessment`, `ResearchState.tests_for`), never
a field on the proposition. An earlier design cached lifecycle statuses
on `Hypothesis` and `Prediction`; both were removed. A cached standing
on the proposition is exactly the shortcut — belief mutating the fact
it is about — that this ontology exists to forbid.

`ResearchQuestion` sits above hypotheses as the unit of scientific
intent. A hypothesis names the question it tries to answer, which
gives utility an explicit conditioning target — `U(action | state,
question)` — without a heavier `ResearchProgram` abstraction that
Horizon 1 does not need.

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

Lifecycle: `QUEUED → RUNNING → SUCCEEDED | FAILED | CANCELLED |
TIMED_OUT`.

Two invariants, enforced rather than documented:

**A failed attempt never makes work look done.** Whether an action is
complete is a question about attempts with succeeded outcomes
(`ResearchState.has_succeeded`). A failed attempt leaves the work open,
and a retry is a new attempt carrying the same action. All attempts,
failed ones included, stay in the state.

**A successful action cannot claim outputs that do not exist.** An
attempt's proposals and its outcome travel together in a
`CommitBundle`, and `commit_bundle` is the only way to resolve an
attempt through the transition layer. It validates the attempt's
lifecycle, commits every proposal (or rejects the lot — a bundle that
half-validates changes nothing), checks that every id in
`outcome.produced` was actually committed by the bundle (mechanically
created prediction tests included), and resolves the attempt — in one
step. A bundle whose outcome is not `SUCCEEDED` carries no proposals
and claims no products, by construction.

This is a domain-level commit boundary over an immutable state, not a
distributed transaction system. One consequence is documented rather
than hidden: the evidence store is append-only and idempotent, so a
result recorded during a transition that later fails stays recorded — a
fact without a referencing state, which is honest (the process really
ran). Atomicity is a guarantee about state membership, where beliefs
could otherwise be corrupted, not about the existence of facts.

`ResearchState.history` still exists as an audit trail of selected
actions. Nothing operational reads it, and a test pins that candidate
generation re-offers work whose action appears in history but never
succeeded.

## Identity semantics

Two kinds of identifier, for two kinds of question (`core/ids.py`):

* **Content ids** (`content_id`) for semantic objects — questions,
  hypotheses, predictions, specs, claims, evidence, replication
  groups. Identical content → identical id. The same hypothesis built
  on two branches or in two runs is the same hypothesis, and
  trajectories are comparable object by object.
* **Occurrence ids** (`occurrence_id`) for events — attempts, jobs,
  decisions, role invocations. Identical construction → distinct ids.
  Two identically configured executions of one spec are two events; a
  replication is not its original. Executors refuse to submit the same
  job object twice.

Objects downstream of an execution inherit its event identity: a
result's id derives from its job's, evidence derives from its result,
and a `PredictionTest` embeds the result id it tested against. Two
identical runs therefore agree on every purely semantic object and
disagree on every event-derived one — which is the correct statement of
what happened.

## Replication semantics

Independent executions are independent evidence — the identity rules
make merging them impossible. The converse question, *which results
were testing the same thing*, is answered by `ReplicationGroup`
(`core/replication.py`): a content-addressed grouping key derived from
the experiment spec and the run configuration, and nothing else.

Deliberate exclusions from the grouping key:

* **observed metrics** — grouping by outcome would sort contradictory
  replications into different families, which is exactly the moment
  they most need to be seen together;
* **seed** — runs of one protocol under different seeds are statistical
  replications of the same thing (each member keeps its seed);
* **environment** — platform and commit are provenance, recorded per
  result; splitting families by machine would hide cross-machine
  disagreement, which is a finding, not a grouping error.

`group_replications(results)` buckets results by family. A future
statistician role receives the family, not a pre-aggregated summary.

## Core domain objects

All of these live in `core/`, which imports nothing from its sibling
packages (enforced by `tests/test_layering.py`).

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

`ResearchState` is immutable and lineage-carrying: every mutation
returns a new state whose `parent_id` points at its predecessor. States
hold propositions and judgments, and hold *references* to facts —
results and evidence live in the append-only store, shared across
branches, because a fact does not become a different fact on a
different branch of the search.

The core is plain frozen dataclasses with no runtime dependencies.
Validation libraries stay at the future LLM boundary: provider output
is validated against provider-specific schemas and translated into
these types, never made to replace them.

### On decomposing `ResearchState`

The state is deliberately still one object. Splitting it now
(scientific / execution / epistemic sub-states) would add indirection
with no invariant behind it. Split when the first of these becomes
true:

1. **concurrent programs** — multiple research programs share a process
   and need independent lifecycles over common facts;
2. **role-context builders duplicate projections** — per-role views
   keep re-deriving the same sub-state slices, so the slices have
   earned names;
3. **state hashing or copying shows up in profiles** — content-id
   computation over the full state stops being negligible;
4. **field count obscures the commit surface** — mutators stop fitting
   on one screen and reviewers can no longer see what a transition may
   touch.

Until then, one dataclass with disciplined mutators is easier to audit
than three objects with a coordinator.

## The decision loop

Three questions, three separated functions, one wiring class:

```
What could we do?            CandidateGenerator  -> ActionCandidate[]
How valuable might each be?  UtilityEvaluator    -> ActionUtility[]
What do we take, given
uncertainty and resources?   SearchPolicy        -> selected candidate
```

`ResearchDirector` composes the three and emits a `Decision` whose
`DecisionRecord` preserves the full tuple — state before, every
candidate with its utility, generator/evaluator/policy names, the
selection — completed later with the attempt, its outcome, the
assigned role, actual cost, and the state after.

The invariant: **scientific utility is not search policy.**
`ActionUtility` is multi-dimensional (information gain, discrimination
value, importance, novelty, replication value, success probability,
expected cost, and the estimate's own uncertainty) and collapses to a
scalar only inside a policy, explicitly, where the collapse can be seen
and criticized. A bandit and a greedy policy consuming identical
utilities are different explorers, not different opinions about
science. `None` in any dimension means *not estimated*, and no consumer
may read it as zero.

Current implementations are deliberately modest: a rule-based generator
that closes structural gaps (a hypothesis without predictions, a
prediction without an experiment, a result nobody analyzed — completion
read from succeeded attempts, never from history); a heuristic
evaluator with fixed per-action-type profiles that flag themselves as
maximally uncertain; a greedy value-per-cost policy. Each is named in
every decision record, so their estimates stay attributable — and
dismissible — later.

Because propositions carry no status, the generator *consumes*
epistemic judgments where it needs standing: a hypothesis whose current
assessment is SUPPORTED or REFUTED is treated as settled and not
offered further work. That is a generation policy reading assessments
made elsewhere, not the generator doing epistemology.

The generator offers `stop_investigation` only when nothing else is
open. Halting is a legitimate outcome, and a free stop action would
dominate any value-per-cost ranking as a standing candidate.

## Runtime philosophy

> The framework separates rich internal scientific representation from
> runtime agent count. A domain abstraction does not imply an LLM call
> or an autonomous agent.

> Complexity is added only when it addresses an observed failure mode
> or produces measurable improvements in scientific quality,
> reliability, or efficiency.

> All else equal, simpler mechanisms win.

The ontology above is deliberately rich — seventeen action types, four
kinds of proposition, versioned judgments — because cheap
representation is what makes honest science checkable. None of that
entitles the runtime to spend. The runtime optimizes, approximately,

```
meaningful scientific progress / (wall-clock + compute + inference cost)
```

and it observes a ground-truth hierarchy:

```
executed result / artifact
  > deterministic validation
    > artifact-grounded independent judgment
      > LLM opinion
```

An LLM is never asked to infer what code can determine. Per ordinary
experiment iteration, the reasoning-invocation count is a design
invariant the loop enforces:

```
Director: 1        one deliberate() — candidates, valuations, selection
Executor: 1        one perform() implements/runs the assignment
Critic:   0        deterministic checks stand in for routine critique
```

A scientifically consequential result (contradictory replication,
challenged standing, unexpectedly large effect, or an explicit director
request) adds exactly one critic invocation — decided by a
deterministic trigger, never by a model.

Invocations are not model calls, and the accounting refuses to conflate
them. The loop can enforce how many times it invokes a reasoning seat.
How many provider calls a model-backed role makes inside one invocation
is the adapter's affair, recorded separately (`UsageSource` →
`provider_calls` / tokens in the step metrics) and honestly zero for
today's rule-based roles.

The runtime is equally strict about paying for work: an action whose
estimated cost exceeds the remaining budget never starts, and actual
cost is reconciled after commit. An overrun is billed as far as the
budget reaches, recorded explicitly, and halts the program.
Committed-but-unbilled work cannot exist.

### The frontier: what the director actually sees

`runtime/frontier.py` derives a `ResearchFrontier` — open questions,
active hypotheses, work queues read from facts and succeeded attempts,
contradictions, failed attempts worth revisiting, current best
findings, remaining budget — from one `ResearchState`. It is a view: a
pure function of the state (plus an injected admissibility policy),
with no mutators, never persisted as authority, carrying the id of the
state it projects. Scientific standing in the projection is filtered
through `ScientificAdmissibility`; the frontier stores no verdict and
no "resolved" flag anywhere. The permanent truths stay in
`ResearchState` and the verification store. The frontier exists to keep
director prompts small and stable, and to give context selection one
seam to grow behind.

### The director fast path

The runtime default is one reasoning seat, `FrontierDirector`: a single
deliberation performs candidate generation, coarse valuation, and
selection together, and `deliberation_record` preserves the
intermediate candidate set in the standard `DecisionRecord` — with the
director named as generator, evaluator, and policy, which is what
happened. The decomposed `ResearchDirector` stays in the architecture
for the ablation; the runtime never requires its three potential model
calls per step.

Runtime valuations are deliberately coarse: HIGH / MEDIUM / LOW value
and uncertainty plus an expected `ResourceCost` (`CompactValuation`),
embedded into the rich `ActionUtility` as a named ordinal mapping. The
runtime does not manufacture `novelty = 0.73` until calibration data
says such numbers mean something. Where ranking is ambiguous, the
rule-based director records pairwise "prefer A over B because ..."
rationale instead of absolute scores, and the raw reasoning is
preserved in the runtime metrics.

### Default search: directed refinement

The loop's default is diagnose → modify → execute → observe, one branch
at a time. No MCTS, no best-first tree search, no default branching.
The tree exists as memory (state lineage, content-addressed snapshots),
not as a mandatory algorithm. Branching should return when there is a
scientific reason — competing hypotheses, materially distinct
strategies — and calibrated value estimates to steer it.

### The deterministic validation gate

`runtime/validation.py` checks what machines can check: the process
exited zero, the result names its spec and matches the director's
assignment, declared metrics are present and finite, the seed is
recorded, artifacts hash to the manifest their run wrote, and
replications agree within stated tolerance. The runtime applies these
checks — artifact integrity included — as a pre-commit gate. A
completed result that fails any of them never enters `state.results`,
never produces a prediction test, and never becomes evidence. The
attempt fails as an engineering failure, the run directory is preserved
for diagnosis, and the director sees a deterministic note on the next
frontier. No critic is consulted and no model may overrule the gate:
arithmetic is not a matter of opinion. Cardinality is part of the gate
too — a run-type assignment must return exactly one result.

Failed or cancelled executions are different: they commit as honest
execution-failure records with mechanically inconclusive standing, get
a deterministic diagnosis, and — when a debugger is wired in — enter
the bounded repair loop below. Repeated failures of one experiment
(counted from the results themselves) raise the same kind of
deterministic engineering note: a debugging signal for the director,
not scientific critique.

The executor also refuses silent success: exit-zero without metrics, a
non-finite metric, or a missing declared artifact is a recorded failure
with the run directory preserved. Reading a gate-valid completed result
into `Evidence` is a transcription, not a judgment, so
`evidence_from_result` does it in code — reusing the mechanical
`PredictionTest` the commit already produced.

### Scientific debugging and experiment verification

The runtime distinguishes five ways an experiment can end, each with
its own detection and response:

```
ENGINEERING FAILURE       crash, timeout, OOM, contract breach → repair execution
IMPLEMENTATION FAILURE    runs, but is not the intended        → verify / debug /
                          experiment (silent bug)                reimplement
METHODOLOGICAL FAILURE    correct code, wrong experiment       → redesign experiment
ANALYTICAL FAILURE        valid experiment, wrong inference    → redo analysis
VERIFIED                  all four dimensions resolved         → outcome is evidence
```

The core principle: **a bad result is not a bug, a successful process
is not a correct experiment, and a reproducible result is not
automatically a valid one.** Debugging optimizes for obtaining a valid
experiment, never a positive result. The lab aggressively repairs
invalid experiments, and just as aggressively preserves disappointing
results when the experiment itself is valid.

The pieces, all removable for ablation:

* **Validity model** (`runtime/verification.py`): every check carries
  an explicit state — `PASS` / `FAIL` / `UNCERTAIN` /
  `NOT_APPLICABLE`, never a manufactured confidence float — on one of
  four dimensions (execution, implementation, methodology, analysis).
  A `VerificationReport` collapses to an `ExperimentValidityStatus`
  with worst-dimension precedence. `VERIFIED` requires a positive
  determination on every dimension; an axis nobody checked yields the
  explicit intermediate `UNVERIFIED` — outcome observed, validity
  unresolved. Validity is orthogonal to scientific outcome: `VERIFIED`
  plus an inconsistent prediction test is a valid scientific negative,
  while `IMPLEMENTATION_UNCERTAIN` plus the same test is a debugging
  question, not negative evidence.
* **Failure classifier** (`execution/failure_classifier.py`):
  deterministic first-pass diagnosis of failed executions from the
  executor's structured failure reason and preserved stderr — timeout,
  launch, OOM, import, missing path, missing/malformed metrics,
  missing artifact. Conservative (`UNKNOWN`/`UNCERTAIN` when signals
  are ambiguous) and structurally blind to science: a completed run is
  `NONE` no matter what its metrics say.
* **Bounded repair loops** (`orchestration/debug_loop.py`), two
  structurally separate entries over shared machinery. *Execution
  repair*: diagnose a failed process → propose a repair with its
  rationale → rerun as a new job → stop at `max_debug_attempts`; entry
  is by failure diagnosis only — the debugger raises on a completed
  result. *Implementation repair*: a completed run may be repaired,
  but only through an `ImplementationRepairTrigger`, whose constructor
  accepts nothing but implementation-dimension verification checks
  with at least one FAIL (a failed positive control, a verifier FAIL,
  a deterministic invariant violation). A prediction test, a small
  effect, or an underperforming baseline cannot be expressed as such a
  trigger, so `while result_is_scientifically_bad: debug()` cannot be
  written against either entry. Within one bounded episode, each
  iteration responds to the latest attempt's actual state: a completed
  rerun earns fresh verification, a fresh implementation FAIL yields a
  new trigger built from that run's report, and a rerun that crashes
  is diagnosed by the classifier and repaired with execution-repair
  semantics. Every retry is a separate auditable `DEBUG` attempt,
  committed through the same validation gate and billed at its actual
  cost. The original invalid result and its verification record are
  never deleted or rewritten, and repair resolves only when the newest
  run's implementation dimension no longer fails — which still says
  nothing about its scientific outcome.
* **Preflight** (`runtime/preflight.py`): cheap deterministic
  pre-execution checks (command resolves, declared input paths exist,
  seed propagated, the interpreter's `.pth` files are not
  hidden-flagged) behind a small extensible interface. A failed check
  prevents the launch and bills nothing. Checks decide their own
  applicability. `PthFilesVisible` is diagnosis only: it names the
  externally caused condition (CPython ≥ 3.11.9 skips hidden `.pth`
  files, so an editable install vanishes from `sys.path`) and its
  remediation, and mutates nothing itself.
* **Positive controls** (`PositiveControl`): experiment-specific
  invariants a faithful implementation must satisfy (a tiny set
  overfits, zero learning rate changes nothing, a known probe reads
  exactly right), evaluated deterministically against reported
  metrics. A failed control makes the result
  implementation-*uncertain*; it is never read as a scientific
  negative, because it tests the instrument, not the hypothesis.
  Controls live outside `core/` and are supplied per spec.
* **Selective implementation verification**: a semantic hunt for
  silent bugs (wrong loss, leaked data, bad split, wrong baseline),
  event-triggered — on a failed or uncertain control, or on a
  conclusive negative with no control coverage — never on every run.
  The hook is a protocol; `orchestration/review.py` adapts an existing
  role to it via a `FALSIFY` invocation (no new agent, no new action
  type), and the review verdict feeds a check state without ever
  committing to scientific state.
* **Methodology gate**: each design is reviewed once, before its first
  execution — even perfectly implemented, would this experiment answer
  the question? A rejected design never runs; the director sees
  `REDESIGN EXPERIMENT`, explicitly not "debug" and not a recorded
  negative.
* **Analysis validity**: raw results are distinguished from downstream
  inference. A deterministic coverage guard runs before commit: a
  judgment citing only part of the conclusive evidence available to
  its hypothesis (post-hoc run selection) is rejected at the gate and
  never enters authoritative state. The surfaced response is *redo the
  analysis*, never rerun the valid experiments beneath it. Assessor
  contexts carry the full conclusive family, so complete citation is
  always possible.
* **Negative-result gate**: a conclusive negative becomes strong
  scientific evidence only when execution, implementation,
  methodology, and analysis are all positively resolved. Anything less
  keeps the observation in the explicit observed-but-unresolved state
  — and under no status is a result routed to debugging merely for
  being negative (pinned by test).
* **Durable verification records** (`runtime/verification_store.py`):
  every verified result's report becomes a record keyed by result id.
  An id never maps to different content, verdicts are never rewritten,
  and repair produces a new result with a new record. Records are
  internally canonical: the report is the single source of truth,
  validity and standing are derived at construction and cannot be
  supplied, and a serialized record whose stored verdict disagrees
  with its own report fails loudly on load. In-memory and
  one-JSON-file-per-record implementations exist.
* **The scientific-promotion gate** (in the runtime loop, pre-commit):
  raw observation is not verified scientific support. Under enabled
  verification governance (`verification_governance_enabled`, on by
  default) the gate fails closed: a SUPPORTS/CONTRADICTS evidence link
  or a conclusive assessment may cite evidence only when the durable
  record behind it stands at `VERIFIED_EVIDENCE`, and a *missing*
  record blocks exactly like an adverse one — absence can mean a lost
  store, a restart, or mis-wiring, and none of those may silently
  restore trust. Legacy semantics exist only as explicit ablation
  (governance off). Inspection is never blocked: unresolved
  observations stay in the evidence store, may be cited as
  INCONCLUSIVE, may ground UNDETERMINED assessments, and reach
  reasoning seats annotated with their standing. For restart/resume,
  wire a `FileVerificationStore` beside the file state store so
  verdicts reload with the state they govern.
* **Scientific admissibility** (`ScientificAdmissibility` in
  `runtime/verification_store.py`): the one canonical answer to *may
  this recorded result participate in scientific inference?*
  Governance off → everything recorded (explicit ablation). Governance
  on → admissible iff a durable record exists **and** stands at
  `VERIFIED_EVIDENCE`, with missing, unresolved, and invalid failing
  closed alike. Recorded and admissible are deliberately different:
  mechanically conclusive is not scientifically admissible. Every
  scientific control-plane consumer takes this same policy instead of
  re-deriving it — frontier prediction resolution and contradiction
  detection (and therefore escalation and synthesis triggers), critic
  triggering, and analysis coverage (owed to the admissible conclusive
  family only, which keeps the coverage gate and the promotion gate
  from deadlocking over an invalid observation). Inadmissible results
  are never hidden: they stay in the state and stores, remain visible
  with their standing annotations, and simply do not count. So after a
  silent-bug repair, an invalid negative next to its verified repaired
  positive creates no fake contradiction, no false resolution, and no
  spurious escalation — while both runs stay on the record.

Deterministic checks always outrank semantic review: dimension
aggregation treats any deterministic `FAIL` as final, so no model
verdict can wash out a failed control or a broken artifact hash.

An honest limit, stated plainly: no general system can prove the total
absence of silent scientific bugs. Reliability here comes from layered
deterministic checks, experiment-specific controls, selective
independent judgment, and — later — replication. The verification
record says which layers actually ran, so an unverified result is
never mistaken for a verified one.

### Deterministic routing

Three runtime seats — scientist (`RESEARCH_DIRECTOR`), executor
(`RESEARCH_ENGINEER`), critic/analyst (`RESULT_ANALYST`) — and a static
table (`orchestration/routing.py`) mapping every action type to one of
them, plus the proposal kinds each action may return. Nothing about the
mapping is uncertain today, so nothing about it is inferred.
`RoleSuitability` and `RegistryAssigner` stay for the day routing has
empirical calibration behind it.

Worker lifetime is separate from lab lifetime: the director is
long-lived, and every role invocation carries an explicit `RoleContext`
projection built per assignment. An executor sees a spec and its prior
runs, never the research history.

### Two timescales, one director

The fast loop optimizes throughput. The slow loop —
`FrontierDirector.synthesize`, the same seat at a stronger reasoning
tier, not a second agent — reviews what has actually been learned and
recommends continue / replicate / pivot / branch / stop. Its cadence is
deterministic (`SynthesisTrigger`): every N committed results, when a
contradiction appears, and before stopping. Its recommendation reaches
the next deliberation through the frontier's `open_decisions`.

### Cost-aware escalation

`runtime/escalation.py` encodes the spending ladder —

```
Tier 0  deterministic code
Tier 1  cheap model / routine reasoning
Tier 2  strongest model for difficult decisions
Tier 3  multi-sample / debate, only when justified
```

— and a small rule table (`EscalationPolicy`) that picks the cheapest
sufficient tier from decision importance, uncertainty, downstream
resource commitment, and evidence conflict. It is not a model router.
It is the place the principle lives, so a router could one day replace
it measurably.

### Evidence-chain validation

`evidence/validation.py` re-derives the whole chain — question →
hypothesis → prediction → spec → result → test / evidence → assessment
— against the state and store, deterministically: dangling references,
facts missing from the store, prediction tests whose recorded
observation or verdict disagrees with a mechanical re-check (the one
place belief could quietly rewrite fact), claims without evidence,
conclusive assessments citing none, and contradictions surfaced as
facts. It is a query layer over existing objects; there is no graph
database.

### Held-out evaluation

`runtime/evaluators.py` is only a seam, but a guarded one: a
development evaluator the loop may consult freely, and a held-out
evaluator that demands an explicit, recorded release. The autonomous
loop holds no credential, which prevents evaluator overfitting
structurally rather than by policy.

### Measured, removable complexity

Every optional mechanism hangs off a typed flag in `RuntimeConfig` —
critic, playbook, synthesis, cheap/strong director floor, the
repeated-failure threshold, and the verification family
(`debug_enabled` + `max_debug_attempts`, `preflight_enabled`,
`methodology_review_enabled`, `implementation_verification_enabled`,
`positive_controls_enabled`). Every step writes a `StepMetrics` record:
reasoning invocations, provider-reported calls and tokens (zero
without a provider), wall-clock, experiment compute, whether the
critic fired and why, the reasoning tier, the outcome, deterministic
runtime notes, the raw decision rationale — plus the verification
record: failure category, debug attempts and whether debugging
recovered a valid execution (never conflated with scientific success),
validity status, preflight/control/methodology/implementation/analysis
rejections, and whether a conclusive negative was accepted as evidence
or deferred. The eventual research contribution is a measurement of
which components earn their cost; the flags and records are how that
measurement stays possible. Playbooks (`runtime/playbook.py`) follow
the same rule: an advisory prior over what usually comes next in
empirical ML — never a stage machine, never checked for compliance.

## Roles: suitability, invocation, and the proposal invariant

A role is a quadruple — objective, information set, allowed
actions/tools, output contract — not a system prompt. Two roles on the
same foundation model are different agents if those differ.

Two value concepts, deliberately not one:

```
ActionUtility     U(a | state)                        "Should the lab perform
                                                       this action?"
RoleSuitability   ≈ P(role succeeds | action, state)  "Who should perform this
                                                       selected action?"
```

`RoleSuitability` (in `roles/`) deliberately avoids the word *utility*:
it expresses no opinion about whether an action is worth performing.
Assignment (`RegistryAssigner`) runs strictly after the search policy
has selected an action, and suitability never feeds back into
selection — so "what is scientifically valuable" cannot quietly become
"what our current roles happen to be good at".

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

`RoleContext` is a typed container of selected domain objects with
every field defaulting to empty: the orchestrator includes exactly what
the invocation needs, so what a role was shown is recorded rather than
implied. A hypothesis researcher sees the question, hypotheses, and
negative findings; a skeptic sees one hypothesis, its predictions, its
tests, and alternatives; an engineer sees a spec and execution
constraints; a statistician sees raw results, the design, and the
replication group. Rendering a context for a model is a
provider-boundary concern that does not exist yet — deliberately.

**Roles never mutate `ResearchState`.** A role reads its context and
produces typed proposals — `QuestionProposal`, `HypothesisProposal`,
`PredictionProposal`, `ExperimentProposal`, `EvidenceProposal`,
`ClaimProposal`, `AssessmentProposal`, and (executors only)
`ResultProposal` — each naming its proposer. Only the transition layer
commits:

```
role -> proposals -> CommitBundle -> validation / atomic commit -> ResearchState'
```

This is enforced twice: structurally (an AST test forbids any module in
`roles/` from calling a state mutator or importing the transition
layer) and behaviorally (transition tests reject orphaned references
and unproduced claims). The payoff is provenance, auditability, safe
search branching, and one place for conflict resolution when multiple
agents propose concurrently.

### The research engineer: the first model-backed role

The model boundary is one narrow seam (`runtime/providers.py`):
`ModelRequest -> ModelProvider.invoke() -> ModelResponse`, every field
a primitive, no vendor type crossing, every failure a typed
`ModelProviderError` that is an infrastructure event and can never
become evidence. Structured output fails closed — the reply is
validated locally against an `OutputSchema` whose unsupported
constructs were rejected at construction — and closed-by-default is
normalized at construction: an object schema that omits
`additionalProperties` gains `additionalProperties: false` before
anything fingerprints or transmits it, so one contract has one body,
one request fingerprint, and one wire form. The one live adapter
(`runtime/muse.py`, stdlib HTTP only) transmits the schema verbatim
and is never trusted with validation.

`ModelBackedEngineer` (`roles/engineer.py`) is the first role with a
model behind it, deliberately on the executor seat: its output is
checkable by machines. Its authority is narrow by construction. The
model proposes the content of exactly one allowlisted Python file plus
a rationale — the schema has nowhere to put a metric, a result, or a
claim — and trusted code does everything else: deterministic source
validation before any execution (file-set allowlist, path safety,
UTF-8/NUL/size bounds, the source must compile), job construction,
preflight, submission. Metrics enter the lab only through the
`metrics.json` the executed process wrote. A reply that fails
validation is preserved as data (never materialized at model-chosen
paths) and earns at most one bounded generation repair — a corrective
call carrying the exact rejection reason. Provider usage flows through
a `UsageLedger` into the step metrics, failed billed calls included.

Every implementation event leaves a durable record
(`runtime/implementation_store.py`), same invariant as every other
store: an id never maps to different content. Source trees are
content-addressed (identical bytes, one tree); records are per
generation event (template id and hash, source manifest, request
fingerprint, response occurrence id, provider and served model,
rationale, and the exact binding). The job's config carries the
implementation id, which the executor round-trips into
`ExperimentResult.config` — so result → exact executed bytes is a
two-hop lookup through existing infrastructure, and `core/` is
untouched.

### The research planner: one decision, gated deterministically

`ModelBackedPlanner` (`roles/planner.py`) is the second model-backed
role, on the scientist seat, performing exactly one action type:
`PLAN_NEXT_ACTION`. From a deterministic, bounded projection of the
authoritative state — the scientific chain, every piece of evidence
annotated ADMISSIBLE or INADMISSIBLE from the durable verification
records, standing notes and contradictions, the remaining budget, and
an explicit catalog of trusted templates with their measurable metrics
— it selects exactly one next action: a new falsifiable experiment, a
replication at the next unused declared seed, an ablation removing one
named component of an existing procedure, or a typed stop. Never a
plan, never a tree.

Its authority is narrow the same way the engineer's is. The flat
decision schema has no slot for an observed value, a command, a path,
a dependency, or a container setting; experiment costs are stamped
from the catalog, never taken from the reply; and because the
supported schema subset has no `oneOf`, inapplicable fields carry
typed sentinels that a deterministic gate checks mechanically — which
is what makes "a stop hides no experiment" an equality test rather
than prose. The gate (`check_decision`) rejects, with stable rule
names, everything the charter forbids: unknown or mismatched ids,
inadmissible evidence cited as grounds, unsupported templates or
metrics, unfalsifiable predictions, duplicate experiments by content
identity, replications off the deterministic seed policy, ablations
without a valid parent or named component, budget violations, and
internally inconsistent chains. A rejected decision is preserved
durably and earns at most one corrective call carrying every rule that
fired. Scientific disagreement is not a rule, so a valid-but-unwelcome
decision has no route to a second call. Accepted decisions expand
deterministically into ordinary proposals — (hypothesis?, prediction,
experiment), committed atomically by the existing `commit_bundle` — or
into no proposals at all for replicate and stop, whose causal artifact
is the durable `PlanningRecord` (`runtime/planning_store.py`: full
provider provenance, write-once, with rejected attempts and dispatch
markers beside it).

The governance seam is `PlanningDirector` (`orchestration/planning.py`),
a deterministic `FrontierDirector`: an open stop decision becomes
`STOP_INVESTIGATION` through the loop's existing halt path with the
typed reason in its rationale; a pending experiment becomes
`RUN_EXPERIMENT`; an open replicate decision with a live gap becomes
`REPLICATE`; otherwise the planner is invoked. Decisions are dispatched
exactly once, durably. The unmodified `ResearchRuntime` remains the one
orchestration loop, every Tier-0 gate included. The engineer learns
which template the planner chose through a wiring-time resolver reading
the planning store, and stamps it into the implementation record —
decision and implementation cross-check by template id.

## Evidence model

The `EvidenceStore` invariant is one sentence: **an id never maps to
different content.** Re-recording identical content is a no-op;
re-recording different content raises. Evidence referencing an
unrecorded result is rejected.

Metrics enter the system exactly one way: an experiment process writes
`metrics.json`, and the executor reads it. A process that exits zero
without writing metrics is recorded as a failure, because treating a
silent run as success is exactly how empty experiments become reported
findings.

## Claims and assessment

A `Claim` carries no status. Its factual support is the set of
`EvidenceLink` edges (supports / contradicts / inconclusive). Its
standing is the latest `EpistemicAssessment` targeting it. Assessments
are versioned by supersession — a change of mind is a new assessment
naming the one it replaces — and every assessment records the evidence
it actually considered and the method that produced it, because a
judgment that cannot say how it was reached cannot be challenged.

`ClaimEvidenceGraph` (in `knowledge/`) is the factual read model:
evidence per claim by relation, claims without evidence, contradicted
claims, unassessed claims. It offers no verdicts. An earlier draft had
an "advisory" status-suggestion helper; it was removed, because
anything that maps edge counts to a status will eventually be treated
as authoritative epistemology regardless of its docstring.

Not built yet: *which experiment would most reduce uncertainty around
this claim?* That needs calibrated uncertainty, and there is nothing
to calibrate against until real trajectories exist.

## Persistence and trajectory logging

Every run persists two artifacts side by side, both local files, no
database:

```
<run_root>/
├── states/
│   └── <state_id>.json      content-addressed ResearchState snapshots
├── trajectory.jsonl          one DecisionRecord per line
└── runs/                     executor run directories (logs, metrics, artifacts)
```

`FileStateStore` (in `persistence/`) serializes states
deterministically — same content, same bytes — so state ids double as
snapshot filenames and identical states deduplicate by construction.
Loading reconstructs the full domain object graph and recomputes the
content id from what it read: a snapshot that no longer hashes to its
filename fails loudly instead of quietly resurrecting a different
state. The parsing code lives in `persistence`, not `core` —
`core.serialize` stays one-way because parsing needs validation, and
validation is a boundary concern.

Every orchestration decision is preserved as a `DecisionRecord`:

```
(state_t, {candidate_i, utility_i}, selected_t, role_t, attempt_t, outcome_t, state_{t+1})
```

plus the names of the generator, evaluator, and policy involved,
predicted and actual cost, and the ids of everything produced.
`state_before_id` and `state_after_id` are keys into the snapshot
store, so the full decision tuple reconstructs offline — a test walks
every record of a demo run and reloads both endpoint states.

This exists now, before any real research runs, because the questions
this project wants to answer — is utility calibrated? do specialized
roles help? does the search policy earn its cost? where does the loop
manufacture apparent progress? — are questions about these tuples, and
trajectories that only exist for later, successful designs cannot
answer them.

## Execution

```python
submit(job) -> job_id      # each job submits at most once
status(job_id) -> JobStatus
collect(job_id) -> ExperimentResult
```

Shaped for asynchronous remote work even though the only backend is
local and synchronous: `submit` returns a handle, so no caller can
assume the result is already available. `ExperimentSpec` (science) and
`ExperimentJob` (commands, paths, env — an occurrence) stay separate
objects. Re-binding a spec to a different backend touches no science.

The contract with an experiment process:

| Direction | Channel |
| --- | --- |
| lab → process | `ARL_RUN_DIR`, `ARL_CONFIG` (path to JSON), `ARL_SEED` |
| process → lab | `$ARL_RUN_DIR/metrics.json`, a flat JSON object of numbers |
| collected | anything else in the run directory, as artifacts |

`LocalExecutor` captures git commit, tree cleanliness, Python version,
platform, command, config, seed, logs, runtime, and exit code on every
run, failures included.

### Executing generated code

`LocalExecutor`'s isolation is job-private recovery isolation, not a
security sandbox — so live model-generated code never runs directly on
the host. How validated source becomes a runnable job is a `JobBinding`
(`execution/binding.py`): trusted code fixes the command, environment,
timeout, and required artifacts, and generated code chooses none of
them. `HostPythonBinding` exists for trusted fixture source (tests);
`ContainerBinding` is the live path. Its job launches a small trusted
shim (`execution/container_shim.py`) that drives `docker run` with the
policy spelled out as data: network disabled, no pulls (the image is
pinned by digest and must already be present), all capabilities
dropped, read-only root filesystem, the source tree mounted read-only
and only the run directory writable, memory/pids/cpu capped, and a
shim-enforced deadline that kills the container by name. Inside, the
process speaks the ordinary contract above — `ARL_*` in,
`metrics.json` out — so the executor runs, records, hashes, and
collects unchanged. This is a job-binding seam, not a cloud executor:
asynchronous remote backends remain a Horizon 2 concern behind the
same `Executor` interface.

## Literature retrieval

The `literature` package (Task 5A) is the reproducible foundation for
field mapping and idea generation, and deliberately nothing more:

```
bounded literature query
  -> real scholarly API (one adapter: OpenAlex)
  -> normalized source records
  -> deterministic deduplication
  -> durable search provenance
  -> reproducible local corpus
```

The seam mirrors the model-provider boundary. `LiteratureQuery` (text,
an inclusive publication-date range, page size, result budget) is
validated against hard ceilings at construction, so an unbounded crawl
cannot be expressed. `LiteratureProvider.search()` returns a
`RetrievedSearch` whose every field is a primitive, a tuple, or a
mapping of strings — no vendor payload crosses the boundary — and
failures use the same typed taxonomy callers know from model calls
(configuration, authentication, rate limit, timeout, transport,
malformed reply).

**One concrete adapter, no registry.** `OpenAlexProvider` speaks the
documented `/works` contract with stdlib HTTP only, under the same
wall-clock-deadline watchdog as the Muse adapter: cursor paging and at
most one retry per page — only for 429/5xx, the two families the
provider documents backoff for, honoring the server's `Retry-After`
when usable. Two wire choices are observation-driven (2026-08-18):
query text matches in titles and abstracts
(`title_and_abstract.search`), because the plain `search=` parameter
proved to be fulltext matching — 4.63M works matched "in-context
learning" versus 6,339 on-topic ones — the root cause of Task 5B's
poor screening yield; and ordering is always explicit (relevance
ranking is not stable across index updates), mapped from the neutral
`ResultOrdering`: `recency` → `publication_date:desc`, `influence` →
`cited_by_count:desc`. Citation counts order retrieval; they are
discovery signals, never evidence about the papers. The recency
default keeps the exact query fingerprints and record identities the
pre-ordering corpus used — pinned by a golden test against a preserved
live record — so existing Task 5A corpora keep replaying
byte-for-byte. OpenAlex was chosen because it is credential-free for
basic use (CC0 metadata, a documented daily credit budget), carries
stable `W…` ids plus provider-normalized DOIs, and reveals arXiv
identity in observed shapes. The optional `OPENALEX_API_KEY` is read
from the environment at request time, sent only as an `Authorization`
header, and can therefore never appear in a recorded request.

**Sources are snapshots.** A `LiteratureSource` records what the
provider reported at one retrieval — title, authors, dates, venue and
type, abstract when available (reconstructed locally from the inverted
index), canonical identifiers, URLs, citation and reference metadata —
under a content id over every field. Metadata the provider did not
report stays `None`, never a fabricated default, and the
`access_level` field (`metadata` / `abstract` / `full_text`) preserves
how much was actually retrieved, so no later stage can claim to have
read text that was never fetched. Task 5A never sets `full_text`.

**Deduplication reports; it never rewrites.** `deduplicate()` groups
snapshots into works by exact canonical identifiers first (normalized
DOI, normalized arXiv id, the provider's own work id), uses a title
fallback only for snapshots with no canonical identifier at all — and
only when title, year, and first author's family name all agree — and
refuses any merge that would put contradictory identifiers in one
group, surfacing it as a conflict instead. Deterministic in input
order, no similarity scores.

**Provenance is write-once and bounded.** `LiteratureStore` keeps
sources, search records (exact query, provider parameters actually
sent, retrieval timestamp, pagination and rate-limit observations,
returned source ids in provider order), and a replay index from query
fingerprint to completed search — all write-once, ids recomputed on
load exactly like the planning and implementation stores, and capped
in count so a corpus cannot grow without bound. `LiteratureCorpus` is
the cache-or-live rule: an identical completed search replays from
disk with zero network calls; only a miss reaches the provider, and
the retrieval is recorded sources-first, so a search can never cite a
snapshot the store does not hold.

**The scientific boundary.** Literature records describe what external
papers report. They are not `ExperimentResult`, not `Evidence`, and
not proof that a claim is true. The package depends on `core` alone,
and its one deliberate consumer is `mapping` — both directions pinned
by structural tests — so retrieved papers have no path into scientific
state.

## Field mapping (Task 5B)

The `mapping` package turns a broad research direction into a
reproducible, source-grounded map of a field — and deliberately
nothing more:

```
research brief / broad topic
  -> focused literature queries        (model-proposed, code-executed)
  -> Task 5A retrieval and replay
  -> relevance screening               (every verdict preserved)
  -> structured source-grounded extraction
  -> FieldMap
  -> ProblemInventory
```

**A provider-neutral service, not a role.** Its input is a
`ResearchBrief` (topic, hard cutoff date, recent-work window, optional
workshop/CFP hints, and explicit budgets for queries, results,
screened and extracted sources, and model calls — all validated
against ceilings at construction), not `ResearchState`. Its output is
literature analysis, not proposals. It speaks only to the generic
`ModelProvider` seam, and its dependency surface is pinned
structurally: `core`, `literature`, and the provider seam, nothing
else, with nothing else in the package importing it back.

**Authority is split the standard way.** The model proposes query text
per fixed family (recent, foundational, methods, datasets/benchmarks,
metrics/evaluation, baselines, limitations/open-problems), screens
sources as relevant/excluded/uncertain with reasons, extracts what
each source's accessible text reports, clusters findings, and proposes
open problems. Trusted code derives every date range from the brief,
executes every search through the Task 5A corpus (cache-or-live),
stamps every era (recent/foundational/undated — from the brief's
window, never a model opinion) and access level, and holds every
payload to the deterministic gates: unknown or excluded source ids,
missing support, access-level mismatches (abstract-only access cannot
support full-text claims), ungrounded numbers and dataset details
(every number token and every dataset name/version/size/URL/license
must appear verbatim in the cited sources' accessible text),
duplicates, internal contradictions, era mismatches, budget
violations, and coverage language (no "exhaustive", no "systematic
review", no proven-novelty claims) all fail closed. A schema violation
or gate rejection earns at most one corrective call carrying the exact
rules that fired; every rejected payload is preserved under
`rejected/`; a valid but disappointing analysis has no route to a
second call.

**Epistemic labels are structural.** Each record category carries a
fixed claim kind (`CLAIM_KINDS`): bibliographic facts live on Task 5A
sources; extraction lists are author-reported claims or
author-reported limitations (typed
compute/data/generalization/reproducibility); themes and clusters are
mapper synthesis; problem entries are inferred open problems with
supporting *and conflicting* source ids preserved. A source whose
accessible text supports nothing yields an honest
insufficient-support record — deterministically, with no model call,
for metadata-only sources. Dataset extraction records how papers
*report* using datasets; nothing is downloaded or executed.

**Provenance and coverage.** Every accepted record embeds full call
provenance (request fingerprint, response occurrence id, provider,
requested and served model, provider request id, latency, exact token
counts, repair count). Usage reaches the `UsageLedger` exactly once
per call, failures included. The `MappingStore` mirrors the planning
store's write-once, recomputed-id, tamper-loud semantics, with one
extra rule: one verdict and one extraction per source per run. The run
record carries deterministic coverage accounting (per-query
retrieved/new-unique counts, overlap, screening outcomes, access-level
mix, truncations, and a modest saturation indicator), so the map stays
honest about what was not covered.

**Retrieval strategy and bounded refinement (Task 5B.1).** Trusted
code assigns each query family a deterministic retrieval strategy
(`RETRIEVAL_STRATEGIES`): recency ordering for recent work, methods,
evaluation, and the limitations discourse; citation-ranked influence
ordering for foundational work, canonical benchmarks, and established
baselines — recorded on every execution and replayable exactly. When
the initial retrieval screens fewer relevant sources than the adequacy
bar, trusted code triggers up to `refinement_rounds` bounded
refinement passes: one gated model call proposing *new* queries (never
a re-run, capped per round, same topic — narrowing the scientific
question is not refinement), fed by per-family yield counts and sample
exclusion reasons, retrieved and screened into the same budgets.

**The adequacy verdict (Task 5B.1).** After the inventory, trusted
code alone computes a durable `MapAdequacyAssessment`:
`ADEQUATE_FOR_IDEA_GENERATION` or `INSUFFICIENT_COVERAGE`, with typed
reasons. "Adequate" means adequate for bounded candidate generation
under this brief — never exhaustive coverage, a systematic review, or
novelty; absence from a bounded corpus is not novelty; and an honest
insufficiency is a successful outcome. The rules consider relevant and
grounded source counts, query-family coverage, recent/foundational
balance, access-level limitations, uncertainty fraction, theme and
problem support distribution, and cross-paper claims that must
actually span multiple sources — no single metric (source count
included) is sufficient. Every problem carries a computed support tier
(single-source limitation / tentative / multi-source / contradicted),
so one paper's reported limitation can stay in the inventory without
ever being presented as field-wide consensus. Thresholds are explicit,
configurable `AdequacyThresholds`, validated at construction and
embedded verbatim in the assessment, which is content-addressed like
every other record. Task 5C enters through exactly one door,
`require_adequate_for_idea_generation`, which reloads the durable
verdict and refuses anything but an adequate map.

## Candidate idea generation (Task 5C)

The `ideation` package turns the assessed map into a bounded portfolio
of testable research candidates — conjectures carrying their sources,
deliberately not proposals and never scientific state. It is mapping's
only consumer, admitted through the structural tests the way mapping
was admitted to literature, and bounded the same way: it may read
`core`, `mapping`, and the provider seam, and pointedly not
`literature` — sources reach candidates only as the opaque ids mapping
records carry, so what this stage cannot read, it cannot pretend to
have read. One run performs a fixed sequence:

```
IdeationDirective + CfpSnapshot
  -> require_adequate_for_idea_generation      (before any model call)
  -> load and cross-verify the mapping records (deterministic)
  -> one gated direction-extraction call
  -> one gated candidate-portfolio call        (each: at most one
                                                corrective call)
  -> trusted stamping: statements, kinds, support tiers, theme eras,
     cited-source era mix, and the UNASSESSED novelty status
  -> deterministic portfolio accounting and one run record
```

**The CFP ingress.** A run is directed by a real workshop call, and
the ingress keeps source text and interpretation structurally apart.
The `CfpSnapshot` is the supplied public text verbatim — URL, supply
timestamp, and content hash sealed into an immutable record a model
never touches (there is deliberately no crawler: a supplied snapshot
with provenance is the whole ingress). The gated `DirectionRecord` is
the model's structured reading of it — a synthesized scope held to the
full claim-language discipline, plus topics, constraints, and dates
that must appear verbatim in the snapshot: extraction, not invention.
The direction constrains relevance downstream; it is not evidence and
grants no authority.

**Canonical handles, stamped resolution.** Candidates reference
inventory problems and field-map themes by keys trusted code derives
from their content (`prob_…`, `thm_…`), rendered beside the full text.
The model cites keys; trusted code resolves and stamps the statement,
kind, and computed support tier (problems) and name and era (themes)
onto the record, where `AddressedProblem` makes a mismatched pair
unconstructible. This is the source-id precedent applied twice more —
the live Muse shorthand-label hazard is exactly why the model gets
canonical handles rather than sentences to echo.

**What a candidate must carry.** A working title, one research
question, the proposed contribution, a hypothesized mechanism, a
falsifiable hypothesis, and predictions whose explicit falsifiers are
structural (a prediction without one cannot be expressed); datasets
that distinguish existing data (gate-checked against the cited
records) from honest new requirements; metrics, an evaluation
protocol, baselines, ablations, resource estimates, risks with
plausible negative outcomes; CFP alignment naming topics copied
exactly from the extracted direction; the model's own uncertainty
statement; and search terms for the prior-art challenge. Epistemic
labels are structural (`CLAIM_KINDS`): the grounding narrative is
literature-describing and must trace every number to the cited
records' gated claim texts or the addressed problems' own words, while
predictions, falsifiers, and resource estimates are design targets
whose new numbers are the point. Every candidate's novelty status is
`UNASSESSED` — the enum holds no other value, so a generation-time
record cannot even express an assessed one — and novelty-claiming
language (`novel`, `unexplored`, `state of the art`, "the first to",
…) is a gate violation wherever it appears.

**The candidate gate** returns every rule that fired: canonical-handle
checks (`unknown_problem` / `unknown_theme` / `unknown_source` /
`unknown_topic`, each naming the fix), per-problem grounding citations
(addressing a problem while citing none of its sources is
`missing_support`), scoped number grounding, dataset-existence
semantics, banned coverage and novelty language, duplicate and
superficial-variant rejection (`insufficient_diversity`: identical
problem sets with identical mechanisms), the mechanical slice of
falsifiability (`circular_finding`), and control-character rejection
that preserves legitimate Unicode — names and technical terms keep
their spelling; nothing is transliterated. An honest refusal is a
first-class gated outcome: zero candidates with a grounded
justification is a completed run, never retried; zero candidates
without one is `empty_finding`. A valid but disappointing portfolio
has no route to a second call.

**Honest accounting.** Trusted code computes the `PortfolioReport`:
which inventory problems the candidates address and — named, not just
counted — which they do not, the support-tier profile of what was
addressed, and mechanical diversity (distinct
problem/theme/dataset/metric sets across candidates; semantic
diversity is not pretended to be checkable — the model's own diversity
rationale is preserved verbatim beside it). The `IdeationStore`
mirrors the mapping store: write-once, ids recomputed on load,
tamper-loud, `rejected/` preserving every refused payload, one
direction and one run record per run. Provider calls reach the
`UsageLedger` exactly once, failures included, and the directive's
call budget fails closed before the exceeding call.

Proven live 2026-08-18/19 against the preserved Task 5B.1 map and the
real NeurIPS 2026 "Foundations of LLM Post-Training in Changing
Environments" call: three fully structured candidates addressing five
multi-source problems and the contradicted theory-accounts problem,
five unaddressed problems named, ten distinct sources cited, one gate
rejection (source ids pasted into grounding prose read as ungrounded
numbers) repaired by the one corrective call, three model calls in
total.

## The prior-art challenge (Task 5D)

The `priorart` package tries to falsify each candidate's
differentiation against fresh, bounded retrieval — the adversarial
stage the ideation records were built to meet, and the first place
novelty acquires an assessed value. It consumes `ideation` (the
immutable portfolio), `mapping` (the shared gate vocabulary), and
`literature` directly (it runs its own searches through a fresh corpus
root). The one-consumer chain becomes a small DAG whose invariant is
unchanged: retrieved papers and everything derived from them have no
path into scientific state. One run performs a fixed sequence per
candidate:

```
PriorArtDirective
  -> require_candidates_for_prior_art        (before any model call)
  -> check_budget_coherence                  (refuse a directive that
                                              cannot complete its own
                                              promised work)
  -> one gated query-proposal call           (model supplies text only)
  -> trusted retrieval through a fresh corpus (dates, ordering, budgets)
  -> cited-source injection, identifier dedup, cutoff filter,
     cited works ordered first
  -> gated similarity screening in batches   (abstract-level and
                                              metadata-only apart)
  -> one gated nearest-work comparison call   (each call: at most one
                                               corrective call)
  -> trusted coverage + assess_prior_art      (the deterministic verdict)
```

**The one door, again.** A challenge enters through a durable ideation
run record whose portfolio holds loadable candidates —
`require_candidates_for_prior_art` mirrors the adequacy door one stage
down, refusing an unknown record, an honest refusal run, or a partial
portfolio before any model call. The candidate records themselves are
never touched: their novelty status stays structurally `UNASSESSED`,
and the verdicts live beside them in the prior-art store, not on them.

**Adversarial retrieval under trusted dates.** Six fixed query
families per candidate — serving the specification's seven search
intents through the explicit `REQUIRED_INTENTS` mapping — with the
model supplying nothing but *concept groups*: each group the
alternative terms for one concept, groups conjoined, alternatives
OR-joined, every term quoted as an exact phrase by the trusted
`boolean-v1` renderer (order-canonicalized, so the same plan renders
and fingerprints identically). A final query string is not expressible
in the schema at all, and a term carrying its own Boolean, wildcard,
or quoting syntax is a gate rejection — as is a plan conjoining more
than three groups (`excessive_conjunctivity`) or anchored nowhere in
the candidate's own record. Trusted code sets every date range from
the directive's recorded `cutoff_date` (prior art "as of" is a
recorded fact, never a wall-clock accident), fixes the retrieval
strategy per family (influence surfaces canonical prior work a date
sort buries), executes through the corpus, and drops post-cutoff
retrievals before anything is screened. The candidate's own cited
sources join the pool — a candidate overlapping work it itself cites
is the most likely falsifier — deduplicated against fresh snapshots by
exact identifiers, with `known_prior_art` stamped by trusted code.
There are deliberately no refinement rounds: retrieval too thin to
distinguish against is honestly `NOVELTY_UNRESOLVED`, and tuning the
search until a candidate survives is exactly the failure this stage
exists to prevent.

**Comparison held to accessible text.** Screening judges similarity to
*this* candidate (`potential_overlap / related / unrelated /
undecidable` — undecidable is honest), in two gated calls with two
precise instructions. Abstract-level sources go through the similarity
gate. Metadata-only sources — title, year, venue; no abstract was
retrieved — go through the material-ambiguity gate, where a
`potential_overlap` decision must carry an attested
`OverlapHypothesis`: the candidate claim at risk, re-found verbatim in
the candidate's rendered record; the supporting source text, re-found
verbatim in the named accessible part; the overlapping dimension; and
why the concern reaches the core contribution. Generic topical
similarity — a shared broad topic, a common dataset, generic terms —
cannot carry a hypothesis and therefore cannot block. An undecidable
metadata screen is explicitly costless and honest. Cited works screen
first, so any bound ever hit costs the fresh tail, never the most
likely falsifiers. The closest abstract-level works then get one
comparison call covering five explicit dimensions — scientific
question, mechanism, data and setting, evaluation protocol, claimed
contribution — each grounded in a verbatim snippet the gate re-finds
in the named part of the source (title or abstract, exactly what was
retrieved; `full_text` support is expressible and rejectable).
Metadata-only sources are never rendered for comparison. A similarity
label that contradicts its own lists — a match naming no overlaps, a
distinction naming no differences — is a gate violation, and the same
incoherence is unconstructible on the record. Novelty language is
banned in both directions, and the 5C.1 identifier lesson carries
forward verbatim: known ids are names, never numerical claims.

**The deterministic verdict.** `assess_prior_art` is trusted code all
the way down, the adequacy discipline applied to falsification: every
rule evaluated, typed reasons, thresholds traveling inside the
content-addressed assessment, fail-closed aggregation. `OVERLAPPING`
needs one accepted comparison whose substantial match the gate already
forced to ground itself. `DISTINGUISHED` needs a complete family
sweep, an adequate screenable pool, bounded screening uncertainty,
every potentially overlapping abstract actually compared, and nothing
left unscreened. Everything else is `NOVELTY_UNRESOLVED`. The rule
bases are part of the semantics (Task 5D.2): the source threshold
measures the in-cutoff *screenable* pool — a work the cutoff excludes
can never be screened, so it cannot help ground differentiation — and
the uncertainty fraction measures abstract-level screens only, because
a metadata-only source is *expected* to screen undecidable and
counting that here would bill the same missing abstract twice. A
metadata-only source blocks differentiation exactly when it was
screened as a material potential overlap under an attested hypothesis
(a pre-5D.2 metadata potential-overlap screen, which could not carry
one, still blocks fail-closed). A bare undecidable metadata screen is
a coverage fact — counted, recorded, and blocking nothing. Deciding a
material ambiguity "is the same overlap" as some compared work would
still be a model judgment where only trusted code may conclude. Every
verdict describes this bounded corpus alone: `DISTINGUISHED` is never
proof of novelty, absence from the corpus is never novelty, and
citation counts only order retrieval. An `OVERLAPPING` or
`NOVELTY_UNRESOLVED` result is a successful scientific outcome; a
verdict the caller dislikes has no route to a second call.

**Budget coherence.** A directive must not promise work its own
budgets cannot complete. `check_budget_coherence` runs after the door
and before any model or network call. It checks: every candidate's
worst-case pool (six families times `results_per_query`, plus its
cited injection) fits the screening cap; the comparison cap reaches
the threshold's minimum; a pool clearing `min_unique_sources` can be
screened without truncation; and the worst-case gated calls — query
proposal, `ceil(S/b) + 1` screening batches, comparison, each with its
bounded corrective call, per candidate — fit `max_model_calls`. Every
violation is collected into one named refusal, and only the directive
record (an input, not an outcome) is durable. The defaults are the
coherent fixed point for a three-candidate portfolio — 35 screened, 36
calls, batches of 12: exactly `3 x 6 x 2` at the ceiling — so
mechanical `SCREENING_TRUNCATED` is no longer expressible as an
executed run, and the runtime budget guard survives only as defense in
depth. A five-candidate portfolio needs 60 worst-case calls against
the ceiling of 36; the preflight names that mismatch instead of
aborting mid-flight.

**Durability and accounting.** The `PriorArtStore` mirrors the
ideation store — write-once, ids recomputed on load, tamper-loud,
`rejected/` preserving every refused payload — plus one assessment per
candidate per run and one account of each run. Provider calls reach
the `UsageLedger` exactly once, failures included; the directive's
call budget fails closed before the exceeding call; and every executed
query is a durable record that rebuilds its exact `LiteratureQuery`,
so a completed challenge replays from its corpus with zero network
calls.

Proven live 2026-08-19 over the three preserved Task 5C candidates:
eighteen fresh OpenAlex searches (six families each, 180 credits),
nine Muse calls, zero repairs, zero rejections, all eighteen queries
replayed with zero network calls, and three durable
`NOVELTY_UNRESOLVED` verdicts — each on `too_few_unique_sources`. The
thin pools were the live finding: the model proposed ten-plus-term
conjunctive queries, and title/abstract matching ANDs terms, so
sixteen of eighteen searches returned nothing — the same class of
retrieval evidence that turned Task 5B into 5B.1. The verdict
machinery refused, correctly, to certify differentiation on pools of
two to four sources.

Task 5D.1 (proven live 2026-08-19) removed the conjunctivity
mechanically. OpenAlex Boolean semantics were verified through a
controlled probe before the design was fixed, the structured-plan
boundary above replaced the free-text query stage, and the rerun over
the same three candidates retrieved pools of 9, 23, and 17 unique
sources (five zero-result searches of eighteen, against sixteen),
compared twelve nearest works — eight of them fresh discoveries beyond
the candidates' own citations — and again returned three fail-closed
`NOVELTY_UNRESOLVED` verdicts, now on the next honest limits:
metadata-only sources screened as possibly overlapping, one pool one
source short of threshold, and one screening truncation. Two more live
lessons are recorded beside it: prose that describes one source may
quote that source's own banned phrases (a retrieved abstract's literal
"novel compositions of visual concepts" fired `novelty_claim` on an
honest unrelated-screen until source-attested phrases became
quotations; candidate-describing text keeps the strict rule), and
anonymous OpenAlex search 429s under cluster load abort a run into
durable partials, fail-closed (preserved as `task5d1-…partial-2`).
What the evidence demanded next was access-level resolution for
metadata-only works and screening-budget headroom — never a weaker
refusal.

Task 5D.2 (proven live 2026-08-19) audited that demand instead of
obeying it, under one governing principle: **a high refusal rate is
not evidence of scientific rigor unless the acceptance path is
demonstrably reachable under evidence that should satisfy it.** The
blocker audit classified every path that can prevent `DISTINGUISHED`.
Family coverage, no-comparable-work, and uncompared-potential-overlap
are necessary conditions and stand unchanged. The source threshold and
the uncertainty fraction were necessary but mis-based: they counted
unscreenable post-cutoff works, and undecidable screens that merely
restate a missing abstract — in the preserved 5D.1 records, every one
of the six metadata-ambiguity blockers was an undecidable title-only
screen, and each also double-counted into the uncertainty fraction.
The metadata-ambiguity rule as written was a proxy for access level,
not an overlap signal. And mechanical screening truncation was a
configuration incoherence, not a scientific fact — the directive
itself could retrieve more than it could screen. The corrections are
the smallest each defect supports: narrowed and re-based rules with
unchanged threshold *values* (ten stays ten, on the screenable pool,
with the fresh/cited split reported — the cited-injection confound is
documented, not yet separately enforced), the attested-hypothesis bar
for material metadata ambiguity, the coherence preflight, and one
prompt correction (the group/alternative caps were enforced but never
stated — all three 5D.1 corrective calls, about a fifth of that run's
budget, bought exactly that omission). The calibration suite proves
all three verdicts reachable end to end on closed corpora at the
default thresholds, that the one metadata condition which must block
(a title directly claiming the candidate's contribution, attested both
ways) still blocks alone, that unattested speculation fails closed,
and that padding a pool past the count threshold repairs exactly one
reason, never its neighbors. The counterfactual replay of the
preserved 5D.1 records under the calibrated rules — run read-only,
before any new live attempt — changed one verdict of three: two
candidates stayed `NOVELTY_UNRESOLVED` on distinct single causes (nine
screenable sources; three truncated), one became counterfactually
`DISTINGUISHED`.

The live rerun over the same three candidates (18 searches, 180
credits, pools of 11/22/20 unique, 15 Muse calls at 32,295in/70,140out
tokens reconciling exactly with the ledger, one verbatim-quote repair,
zero truncation, zero undecidable screens, two 429-aborted partials
preserved beside it) returned three `DISTINGUISHED` verdicts — twelve
nearest-work comparisons, every snippet re-found, with grounded
material differences against the candidates' own cited works and
against fresh discoveries. All three flipping is exactly the outcome
the 5D.2 directive flags for inspection, and the inspection is on the
record: the counterfactual flipped only one candidate on the old
evidence; the second cleared because the budget now screens the whole
pool it retrieves (a configuration fact); the third cleared because
fresh retrieval crossed the unchanged ten-source bar on its own
(eleven screenable against 5D.1's honest nine). No threshold value
moved, no rule was fit to a candidate, and every rule that released
was shown by controlled calibration to have been restating access
level or configuration — while the conditions that must still refuse
were demonstrated, live and in calibration, to refuse. `DISTINGUISHED`
still means exactly what it meant: materially differentiated from the
closest works *this bounded search surfaced* — never proof of novelty.
What remains honestly absent: access-level resolution for
metadata-only works (an attested material ambiguity still has no
lawful in-repo path to an abstract), and any separate bar on
citation-dominated pools, should live evidence ever show one gamed.

**The Task 5E ingress contract (implemented).** A candidate is
selectable only on a `DISTINGUISHED` assessment. The contract's first
wording said "latest applicable assessment"; the implementation pins
something narrower and reproducible, and says so rather than papering
over the change: no record carries a clock that could define "latest",
so a `SelectionDirective` names one `PriorArtRunRecord` explicitly and
eligibility is computed from that run's assessments alone — an
assessment outside the named run does not exist for that selection,
which is what makes staleness well-defined. `OVERLAPPING` and
`NOVELTY_UNRESOLVED` candidates are ineligible and carry their own
grounded specifics forward, so an empty selection explains itself from
existing records; an evidence-limited candidate is never described as
scientifically indefensible, and no retrieval-retry loop exists to
manufacture a selectable one. The contract's "no new terminal stop
states" holds as stated: selection's three outcomes are record values
inside the `selection` package, nothing entered the scientific state
machine, and admission to `ResearchState` remains future work behind
the same governed commit as every other proposal. Selection exists as
*preference* — deliberately not ranking, since no score is expressible
anywhere in its records. The next chapter describes the layer.

## Candidate selection (Task 5E)

The `selection` package decides which challenged candidate to pursue,
if any. It consumes `priorart` (the verdicts that define eligibility),
`ideation` (the immutable portfolio and the governing direction), and
`mapping` (the shared gate vocabulary) — never `literature`: selection
runs no retrieval and sees sources only through the records upstream
stages froze. Nothing imports it; it is the new leaf of the analysis
DAG, and `priorart` now has exactly one consumer.

One run performs a fixed sequence:

```
SelectionDirective (names one PriorArtRunRecord; four operator
                    resource statements; hard ceilings)
  -> require_challenged_portfolio_for_selection  (before any model call)
  -> partition_by_verdict          (trusted code: eligible iff
                                    DISTINGUISHED in the named run)
  -> zero eligible?                record NO_ELIGIBLE_CANDIDATE —
                                    zero calls, structurally zero
                                    spend, every ineligible candidate
                                    named with its verdict's specifics
  -> check_selection_coherence     (refuse a directive that cannot
                                    complete its own promised work)
  -> stage 1: comparative review   (one gated call: every candidate,
                                    every pair, attested disqualifiers)
  -> all eligible disqualified?    record NO_DEFENSIBLE_CANDIDATE —
                                    no second call
  -> stage 2: final decision       (one gated call over the remaining
                                    contenders)
  -> record SELECTED               (one nested write-once run record)
```

Each stage gets at most one corrective call carrying the exact
mechanical rules that fired, never a preferred candidate; a stage-2
rejection never redoes stage 1. Every rejected payload is preserved.
The comparative review is one inseparable joint judgment over the whole
eligible set, so the run record nests it whole: reviews, pairwise
comparisons, stamps, decision, provenance, and spend are one
content-addressed write, tamper-loud together.

**The authority split.** This is the first layer where a model judgment
is the decision: no deterministic rule can compute "most scientifically
important". The split is explicit. Trusted code decides validity — the
eligible set, the disqualified set, the exact partition, disqualifier
evidence, structural coherence, outcome legality, and spend. The model
decides which non-disqualified eligible candidate wins, and that
preference is labeled `comparative_preference` in the package's
`CLAIM_KINDS` — a model preference validated, never computed, by
trusted code. Task 5F must treat the selection record as exactly that.
The stage schemas carry no numeric field anywhere: score-free
justification is structural, like the engineer schema having nowhere to
put a metric — not a gate detecting score-shaped prose.

**Hard disqualifiers are narrow and attested.** A candidate may be
disqualified only on five typed grounds — not operationally falsifiable
within the directive; minimum resource needs exceeding the directive;
an unmeasurable outcome; no credible baseline or control repairable
without changing the candidate; outside the governing CFP scope — and
every disqualifier is the overlap-hypothesis discipline applied to
resources: it quotes the candidate's own rendered record verbatim and
the named constraint verbatim (the directive statement for
compute/data/time/experimental, the recorded direction for scope), and
the gate re-finds both. Weakness relative to another candidate,
uncertainty between close candidates, current repository limitations,
and implementation difficulty are never disqualifiers. Candidates that
survived the ideation gates carry falsifiers, metrics, baselines, and
CFP alignment as construction invariants, so every ground but the
resource conflict is a near-unreachable fail-closed guard, kept
expressible so a real defect against the directive's stated envelope
still has a name. An unattested disqualification is a gate rejection —
and if any eligible candidate lacks a validated disqualifier,
`NO_DEFENSIBLE_CANDIDATE` is a gate rejection, not an outcome.

**Stops are structural, never a model output.** The decision schema has
no stop shape, and "selecting" a stamped-disqualified or invented
candidate is rejected (`disqualified_selection`, `unknown_candidate`).
The run record's own invariants finish the job:
`NO_DEFENSIBLE_CANDIDATE` is unconstructible unless the disqualified
set equals the eligible set with a validated disqualifier behind every
entry, and `NO_ELIGIBLE_CANDIDATE` is unconstructible with any review,
any call, or any spend on it. A stop with a defensible candidate
remaining cannot be recorded, whatever code path produced it — and a
comparative judgment cannot be disguised as a stop, or a stop as
judgment.

**What a selection means.** A selection is comparative portfolio
judgment under a bounded search: the winner is preferred among the
candidates one named challenge distinguished, under one operator's
stated constraints — never proof of novelty, and never a fact about
which idea is best. The candidate records stay untouched and their
novelty standing stays structurally `UNASSESSED`; the verdicts and the
selection live beside them. Unselected eligible candidates remain
immutable, addressable by id, and available to future selection runs —
not being selected is not a disqualification, and a re-run over the
same portfolio is a new occurrence with its own durable record.

**The live evidence (2026-08-20).** One selection ran over the
preserved Task 5D.2 challenge, whose three candidates were all
`DISTINGUISHED`. The first attempt failed closed on infrastructure —
the Muse endpoint returned HTTP 504 before any reply, a typed
transport error that recorded nothing but the directive — and the
rerun completed as a new occurrence: the preflight passed (worst-case
stage-1 reply 6,550 of 16,384 tokens, 4 of 4 calls reserved), both
gated stages accepted on their first attempt (zero corrective calls,
zero rejected payloads), and the run selected one candidate over two
undisqualified alternatives, with the decisive tradeoff, one rationale
per alternative, a first experimental objective, capabilities, and
risks on the record. Spend was 2 model calls, 7,496 input and 10,833
output tokens, reconciling exactly with the ledger; all 182 preserved
upstream files were byte-identical before and after; the nested record
reloaded identically from a fresh store. No candidate carried a
disqualifier — the operator's stated GPU envelope accommodates all
three — so the honest-stop paths went unexercised live and rest on the
calibration suite, where all three outcomes are proven reachable at
the default ceilings.

## Architectural invariants

The list this pass was made against; each is enforced by at least one
test.

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
Retrieved literature describes; it never becomes evidence.
Candidate ideas carry their sources; generation alone never makes them
scientific state, and their novelty stays unassessed until challenged.
A bounded prior-art search never certifies novelty: its verdicts
describe the searched corpus, fail closed to NOVELTY_UNRESOLVED, and
never modify the candidates they judge.
A high refusal rate is not evidence of rigor: every prior-art verdict,
DISTINGUISHED included, stays demonstrably reachable under evidence
that should satisfy it, and every blocker fires on its own recorded
cause.
A selection is a preference, never a proof: eligibility and
disqualification are stamped by trusted code from one named prior-art
run, an honest empty stop needs a validated verbatim disqualifier for
every eligible candidate, and no numeric score exists anywhere in its
schemas.
Every important research decision is reconstructible later.
```

## Separation of scientific reasoning from infrastructure

```
core          scientific vocabulary                    (no internal dependencies)
evidence      what happened, append-only + the chain validator
execution     how to make things happen, anywhere; deterministic
              failure classification; job bindings and the
              disposable-container launcher for generated code
knowledge     what it all means, joined — factually; lesson scaffold
persistence   snapshots of states, reconstructible offline
runtime       frontier view, Tier-0 validation, experiment verification
              and preflight, tiers/escalation, metrics, playbooks,
              evaluation seam, the model-provider seam + Muse adapter,
              implementation provenance                (depends on core only)
literature    what external papers report: bounded retrieval, one
              scholarly adapter, snapshot records, deduplication,
              write-once search provenance   (depends on core only;
              exactly one consumer — mapping)
mapping       what the literature adds up to: model-backed field
              mapping and problem inventories over the Task 5A corpus,
              deterministically gated, write-once  (depends on core,
              literature, and the provider seam; exactly one
              consumer — ideation)
ideation      what might be worth investigating: CFP-directed, gated
              candidate generation over the assessed map, tier-stamped
              and source-carrying, write-once  (depends on core,
              mapping, and the provider seam; exactly one
              consumer — priorart)
priorart      whether it was already done: the adversarial prior-art
              challenge over the portfolio — budget preflight, fresh
              bounded retrieval, gated screening and comparison, a
              deterministic fail-closed verdict per candidate,
              write-once  (depends on core, literature, mapping,
              ideation, and the provider seam; exactly one
              consumer — selection)
selection     which candidate to pursue, if any: trusted eligibility
              from one named challenge, two gated stages, attested
              disqualifiers, three honest outcomes, one write-once
              nested record  (depends on core, ideation, mapping,
              priorart, and the provider seam; nothing imports it)
search        which move to take
roles         who does the work, under what contract; the model-backed
              engineer and planner
orchestration director, runtime loop, routing, triggers, bounded debug
              loop, role-backed review, atomic transitions, trajectory
publication   reporting  (empty)
```

Dependencies point downward only; `core` imports nothing from its
siblings. The infrastructure will be rewritten many times — new
executors, new storage, new providers — and the scientific vocabulary
should not move when it is. Equally, when the scientific model turns
out to be wrong (it will), fixing it should not require touching
subprocess handling.
