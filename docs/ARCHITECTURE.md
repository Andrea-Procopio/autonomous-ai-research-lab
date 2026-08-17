# Architecture

## Design philosophy

A language model is fluent enough to produce everything that *looks* like
research — a crisp hypothesis, a plausible method, a confident result, a tidy
conclusion — without any of it being anchored to something that was measured.
That is the dominant failure mode of an autonomous research system, and it is
not fixed by better prompting. It is fixed by making unsupported output
structurally difficult to produce.

Five consequences shape everything below.

**The state of the research is data, not conversation.** Anything a decision
depends on lives in a structured `ResearchState`. Conversation history is a
working medium, not a system of record.

**Facts and beliefs are stored differently.** An `ExperimentResult` can only be
created by an executor that ran a process, and is append-only. Beliefs —
hypotheses, claims, assessments — are versioned, revisable, and always link
back to the facts they rest on.

**Research is a search, not a pipeline.** Typed actions over a state space,
chosen by a policy, rather than stages hard-coded in sequence. Real research
backtracks, replicates, abandons, and re-scopes.

**Negative results must survive.** Failed hypotheses, failed predictions,
failed *attempts* — each is recorded with the same fidelity as success. A
system whose every outcome converts into apparent progress is not doing
science.

**Evidence is not interpretation.** How an observation relates to a claim is a
factual annotation; whether the claim should be believed is a judgment with its
own object, author, method, and version history. No count of evidence edges
produces a verdict anywhere in this codebase.

## The scientific chain

```
ResearchQuestion
    ↓
Hypothesis            a general, revisable statement
    ↓
Prediction            what it commits to observably: metric, comparator,
    ↓                 threshold, condition — fixed before any run
ExperimentSpec        the design that will test the prediction
    ↓
ExperimentJob         one execution event, bound to a backend
    ↓
ExperimentResult      the immutable record of what happened
    ↓
Evidence              a factual reading of that result
    ↓
EvidenceRelation      supports / contradicts / inconclusive, per claim
    ↓
EpistemicAssessment   the judgment: verdict, confidence, method, scope
```

The load-bearing object is `Prediction`. A hypothesis is testable only through
the predictions derived from it, and each prediction is machine-checkable: when
evidence is committed, the transition layer compares the observed metric
against the pre-registered comparator and threshold and marks the prediction
`held`, `failed`, or `indeterminate`. That check is arithmetic, fixed before
the run — it cannot be adjusted to fit the outcome, and no role's opinion
enters into it.

What a failed prediction *means* for its hypothesis is a different question —
wrong theory, broken auxiliary assumption, bad instrument — and that question
is answered only by an `EpistemicAssessment`.

## The action lifecycle

```
ResearchAction        scientific intent          (semantic identity)
    ↓
ActionAttempt         one try at executing it    (occurrence identity)
    ↓
ActionOutcome         terminal status + produced ids + actual cost
```

Lifecycle: `QUEUED → RUNNING → SUCCEEDED | FAILED | CANCELLED | TIMED_OUT`.

The invariant: **a failed attempt never makes work look done.** Whether an
action is complete is a question about attempts with *succeeded* outcomes
(`ResearchState.has_succeeded`); a failed attempt leaves the work open, and a
retry is a new attempt carrying the same action. All attempts — failed ones
included — stay in the state.

`ResearchState.history` still exists, as an audit trail of selected actions.
Nothing operational reads it, and a test pins that candidate generation
re-offers work whose action appears in history but never succeeded.

## Identity semantics

Two kinds of identifier, for two kinds of question (`core/ids.py`):

* **Content ids** (`content_id`) for semantic objects — hypotheses,
  predictions, specs, claims, evidence. Identical content → identical id, so
  the same hypothesis constructed on two branches or in two runs is the same
  hypothesis, and trajectories are comparable object by object. Status changes
  preserve identity: a falsified hypothesis is the same hypothesis.
* **Occurrence ids** (`occurrence_id`) for events — attempts, jobs, decisions.
  Identical construction → distinct ids. Two identically configured executions
  of one spec are two events; a replication is not its original. Executors
  refuse to submit the same job object twice.

Objects downstream of an execution inherit its event identity: a result's id
derives from its job's, and evidence derives from its result. Two identical
runs therefore agree on every purely semantic object and disagree on every
event-derived one — which is the correct statement of what happened.

## Core domain objects

All of these live in `core/`, which imports nothing from its sibling packages
(enforced by `tests/test_layering.py`).

| Object | Role |
| --- | --- |
| `ResearchState` | The authoritative state of a research program |
| `ResearchQuestion` | What is being asked, and why it matters |
| `Hypothesis` | A general statement; testable via its predictions |
| `Prediction` | A pre-registered, machine-checkable commitment |
| `ResearchAction` / `ResearchActionType` | A typed scientific intent |
| `ActionAttempt` / `ActionOutcome` | One execution of an intent, and how it ended |
| `ExperimentSpec` | The scientific design testing one prediction |
| `ExperimentResult` / `Environment` | Immutable execution record + provenance |
| `Evidence` / `EvidenceLink` | Factual reading of a result; its bearing on a claim |
| `Claim` | A scoped assertion; carries no status and no numbers |
| `EpistemicAssessment` | A versioned judgment: verdict, confidence, method |
| `ActionCandidate` / `ActionUtility` / `EvaluatedCandidate` | The decision vocabulary |
| `DecisionRecord` | One orchestration decision, preserved end to end |
| Proposals (`HypothesisProposal`, …) | Attributable requests to change state |
| `ResearchBudget` / `ResourceCost` | What may be spent; what things cost |

`ResearchState` is immutable and lineage-carrying: every mutation returns a new
state whose `parent_id` points at its predecessor. States hold *beliefs* and
hold *references* to facts — results and evidence live in the append-only
store, shared across branches, because a fact does not become a different fact
on a different branch of the search.

The core is plain frozen dataclasses with no runtime dependencies. Validation
libraries stay at the future LLM boundary: provider output will be validated
against provider-specific schemas and *translated* into these types, not
replace them.

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
the attempt, its outcome, actual cost, and the state after.

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

The generator offers `stop_investigation` only when nothing else is open:
halting is a legitimate outcome, and a free stop action would dominate any
value-per-cost ranking as a standing candidate.

## Roles and the proposal invariant

A role is a triple: **objective** (its own utility, not a shared reward),
**information access**, and **authority** (which action types it may perform).
Two roles on the same foundation model are different agents if those differ.

**Roles never mutate `ResearchState`.** A role reads state and produces typed
proposals — `HypothesisProposal`, `PredictionProposal`, `ExperimentProposal`,
`EvidenceProposal`, `ClaimProposal`, `AssessmentProposal`, and (executors only)
`ResultProposal` — each naming its proposer. Only the transition layer commits:

```
role -> proposal -> validation / transition layer -> ResearchState'
```

This is enforced twice: structurally (an AST test forbids any module in
`roles/` from calling a state mutator or importing the transition layer) and
behaviourally (transition tests reject orphaned references). The payoff is
provenance, auditability, safe search branching, and one place for conflict
resolution when multiple agents propose concurrently.

The transition layer's validation is referential and mechanical, never
epistemic: predictions must name known hypotheses, experiments must measure
their prediction's metric, evidence must cite recorded results, assessments
must target known subjects. Committing evidence triggers the mechanical
prediction check; committing an assessment updates the subject hypothesis's
lifecycle status (a cache-maintenance rule — the judgment is the assessment).

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

## Trajectory logging

Every orchestration decision is preserved as a `DecisionRecord`:

```
(state_t, {candidate_i, utility_i}, selected_t, attempt_t, outcome_t, state_{t+1})
```

plus the names of the generator, evaluator, and policy involved, predicted and
actual cost, and the ids of everything produced. `JsonlTrajectoryLogger`
appends each record as one JSON line to a local file; consecutive records chain
(`state_after` of one is `state_before` of the next), which tests verify.

This exists *now*, before any real research runs, because the questions this
project ultimately wants to answer — is utility calibrated? do specialized
roles help? does explicit epistemic state earn its cost? — are questions about
these tuples, and trajectories that only exist for later, successful designs
cannot answer them.

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

## Future persistent memory

`knowledge/` is where cross-project institutional memory will go — which
methods worked, which failure modes recur, which questions were already
answered. Deliberately not started: it should be designed against real
trajectories, which is exactly what the trajectory log is accumulating.
Persistence of `ResearchState` and the store to disk is the nearest-term item;
content-addressed ids and immutable records were chosen partly to make that
straightforward.

## Separation of scientific reasoning from infrastructure

```
core          scientific vocabulary                    (no internal dependencies)
evidence      what happened, append-only
execution     how to make things happen, anywhere
knowledge     what it all means, joined — factually
search        which move to take
roles         who does the work, and what each one wants
orchestration candidates, utilities, decisions, transitions, trajectory
publication   reporting  (empty)
```

Dependencies point downward only; `core` imports nothing from its siblings.
The infrastructure will be rewritten many times — new executors, new storage,
new providers — and the scientific vocabulary should not move when it is.
Equally, when the scientific model turns out to be wrong (it will), fixing it
should not require touching subprocess handling.
