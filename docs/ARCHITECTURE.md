# Architecture

## Design philosophy

A language model is fluent enough to produce everything that *looks* like
research — a crisp hypothesis, a plausible method, a confident result, a tidy
conclusion — without any of it being anchored to something that was measured.
That is the dominant failure mode of an autonomous research system, and it is
not fixed by better prompting. It is fixed by making unsupported output
structurally difficult to produce.

Four consequences shape everything below.

**The state of the research is data, not conversation.** Anything a decision
depends on lives in a structured `ResearchState`. Conversation history is a
working medium, not a system of record: it is lossy, unqueryable, and it
rewards whichever framing was most recently persuasive.

**Facts and beliefs are stored differently.** An `ExperimentResult` can only be
created by an executor that ran a process. It is append-only. Beliefs —
hypotheses, claims, interpretations — are versioned, revisable, and always link
back to the facts they rest on. Reinterpretation is expected; revision of the
record is not possible.

**Research is a search, not a pipeline.** Stages hard-coded in sequence encode
the assumption that research proceeds in one direction. Real research
backtracks, replicates, abandons, and re-scopes. Typed actions over a state
space allow that; a fixed `run_stage_3()` does not.

**Negative results must survive.** A system whose every outcome converts into
apparent progress is not doing science. Failure has to be as
representable as success, and equally cheap to record.

## Core domain objects

All of these live in `core/`, which imports nothing from its sibling packages.
The dependency rule is enforced by `tests/test_layering.py`, not by convention.

| Object | Role |
| --- | --- |
| `ResearchState` | The authoritative state of a research program |
| `ResearchQuestion` | What is being asked, and why it matters |
| `Hypothesis` | A falsifiable statement, with the criterion that would refute it |
| `ResearchAction` / `ResearchActionType` | A typed move, with rationale and expected cost |
| `ExperimentSpec` | The scientific design of a test |
| `ExperimentResult` | The immutable record of one execution |
| `Environment` | Provenance sufficient to attempt a re-run |
| `Evidence` | A factual reading of a result |
| `Claim` / `EvidenceLink` | An interpretation, and its links to evidence |
| `ResearchBudget` / `ResourceCost` | What may be spent, and what an action costs |

Three choices in there are worth stating explicitly.

`Hypothesis.falsification_criterion` **is required.** A statement that cannot be
written down alongside the observation that would refute it is not a hypothesis
in this system. Enforcing that at construction makes the rule structural rather
than aspirational — a generator role cannot emit an unfalsifiable hypothesis
even if its prompt drifts.

`ResearchState` **is immutable and carries lineage.** Every mutation returns a
new state whose `parent_id` points at its predecessor. This gives three things
at once: a research trajectory that is inspectable after the fact, safe
branching for search policies, and no defensive copying anywhere.

**Identifiers are content-addressed.** An id is a hash of the object's content,
so the same hypothesis constructed in two runs carries the same id and two
trajectories can be compared object by object. Things that must stay distinct
across identical construction — a replication of the same experiment — carry an
explicit discriminator (`ExperimentJob.attempt`).

### Plain dataclasses, not a validation framework

The domain core uses stdlib frozen dataclasses and has no runtime dependencies.

Pydantic buys validation and serialization, and both will matter — but at the
*boundary*, where model output and stored state enter the system, not in the
vocabulary itself. Putting it in the core would couple every scientific concept
to a third-party object model in the week those concepts are least settled, and
`frozen=True` dataclasses already give the immutability the evidence model
depends on. When the LLM boundary is built, a validation layer belongs there,
and the core will not need to change.

## Orchestration

Three decisions are separated, because they fail in different ways and folding
them together produces the single god-object this design is trying to avoid:

1. **What scientific action to take** — `ResearchDirector`
2. **Which role performs it** — `RoleAssigner`
3. **How to run the resulting work** — `Executor`

`ResearchDirector` splits further. `candidate_actions(state)` enumerates what is
scientifically *available* — the part requiring domain understanding.
`propose(state)` filters those by budget and hands the choice to a
`SearchPolicy`. A subclass supplies judgement about what is possible; the policy
supplies strategy about what is worth doing. Swapping greedy selection for a
bandit later touches no director.

`propose` may return `None`. Halting is a legitimate outcome, and a system that
always finds one more thing to try is spending budget rather than doing
research.

`RuleBasedDirector` is the only implementation today. It closes obvious gaps —
a hypothesis with no experiment, a result nobody has read — and is bookkeeping,
not scientific judgement.

## Roles

A role is not a persona or a prompt. It is a triple:

1. **objective** — the utility it maximises;
2. **information access** — what it may see;
3. **authority** — which action types it may perform.

Two roles backed by the same foundation model are genuinely different agents if
those three differ. A skeptic rewarded for finding confounds behaves differently
from a generator rewarded for novelty, with identical weights behind both.

**Roles do not share a scalar reward.** Each carries its own `UtilityFunction`.
The Research Director maximises expected scientific value of the program under
resource constraints; the Hypothesis Generator maximises novelty, plausibility,
falsifiability and expected information value; the Skeptic maximises the
probability of finding a flaw; the Experiment Designer maximises information
gain per unit cost; the Engineer maximises faithful reproduction of the spec;
the Statistician maximises correctness of inference while minimising
unsupported conclusions; the Verifier minimises unsupported claims and
provenance gaps.

`UtilityFunction` is deliberately a bare protocol — `(state, action) ->
UtilityScore`. Heuristic, explicit, learned and model-evaluated utilities all
satisfy it, because which of those is right is an open research question rather
than a settled design decision. `UtilityScore` keeps named components alongside
the scalar so a decision can be *audited*, not merely ranked.

Only the base contract exists today. Concrete roles need a model provider, and
stub roles that cannot act would make the package look further along than it is.

## Evidence model

```
ExperimentResult   what the machine emitted        (immutable, executor-only)
      ↑
   Evidence        a factual reading of it         ("heads_rate = 0.508, n=4000")
      ↑
    Claim          an interpretation               (links to evidence, holds no numbers)
```

The `EvidenceStore` invariant is one sentence: **an id never maps to different
content.** Re-recording identical content is a no-op, so retries and replays are
safe; re-recording different content raises. Evidence referencing an unrecorded
result is rejected — evidence with nothing behind it is an assertion.

The store is shared across branches of a search. Beliefs fork; observations do
not. This is why `ResearchState` holds *references* (`ResultRef`, evidence ids)
rather than result payloads: a state is a belief about the world, and copying
facts into it would let two branches disagree about what was measured.

Metrics enter the system exactly one way: an experiment process writes
`metrics.json`, and the executor reads it. Nothing else can supply a number.
A process that exits zero without writing metrics is recorded as a **failure**,
because treating a silent run as success is precisely how empty experiments
become reported findings.

## Claim-evidence relationship

`ClaimEvidenceGraph` is a read model joining claims and links (in the state) to
evidence (in the store). It answers:

- Which claims are weakly supported, or supported by nothing at all?
- Which claims does the evidence contradict?

`ClaimSupport.suggested_status()` reads the edges crudely and is **advisory
only** — nothing applies it automatically. Deciding that evidence settles a
claim is the statistician's and verifier's job; encoding a confident rule here
now would quietly become the system's epistemology by default.

There is no graph database. At this size, traversal over tuples is adequate,
and committing to a storage engine would fix the schema before the schema has
earned it. The interface here is what a backing store would eventually
implement.

Not yet built: *which experiment would most reduce uncertainty around this
claim?* That needs a calibrated uncertainty estimate, and there is nothing to
calibrate against until real trajectories exist.

## Search

`SearchPolicy` sees only states and actions. It has no handle on roles, models
or executors, so a search algorithm can be replaced without touching any agent
and vice versa. This is what makes greedy, best-first, beam, bandit, MCTS and
learned policies interchangeable later.

`GreedySearchPolicy` maximises expected information gain per unit cost, one step
at a time. Two details in it are load-bearing:

- An action with no gain estimate scores a **default**, not zero. "Not
  estimated" is not "worthless", and conflating them makes a system
  structurally unable to try anything it has not already learned to value.
- Ties break on action id, not enumeration order, so a choice never depends on
  how candidates happened to be generated.

Its single scalar cost — money plus GPU-hours plus wall-clock — is a known
simplification. Those resources are not fungible, and a real exchange rate is
Horizon 2 work.

## Execution

```python
submit(job) -> job_id
status(job_id) -> JobStatus
collect(job_id) -> ExperimentResult
```

Three methods, shaped for asynchronous remote work even though the only backend
today is local and synchronous. `submit` returns a handle rather than a result,
so no caller can be written in a way that assumes the result is already
available — the code stays correct when a cluster backend replaces the local one.

`ExperimentSpec` (science) and `ExperimentJob` (commands, paths, env) are
separate objects. A spec can be re-bound to a different backend without editing
the science, and the scientific design stays free of infrastructure detail.

The contract with an experiment process:

| Direction | Channel |
| --- | --- |
| lab → process | `ARL_RUN_DIR`, `ARL_CONFIG` (path to JSON), `ARL_SEED` |
| process → lab | `$ARL_RUN_DIR/metrics.json`, a flat JSON object of numbers |
| collected | anything else written into the run directory, as artifacts |

`LocalExecutor` captures git commit, working-tree cleanliness, Python version,
platform, command, config, seed, logs, runtime and exit code on every run,
including failures. A failed run produces a result with provenance, not a gap.

Cloud and cluster backends are deliberately not started. The interface is the
commitment; the implementations follow when there is research worth running on
them.

## Future persistent memory

`knowledge/` currently holds only the claim-evidence read model. It is where
persistent institutional memory will go: which methods worked, which failure
modes recur, which questions were already answered and by what evidence, across
projects rather than within one.

This is deliberately not started. Designing a memory schema before there are
real research trajectories to look at would encode guesses about what is worth
remembering. The trajectories produced by Horizon 1 are the input to that
design.

Persistence more generally — writing `ResearchState` and the evidence store to
disk — is the first thing Horizon 1 needs. Content-addressed ids and immutable
records were chosen partly to make that straightforward when it arrives.

## Separation of scientific reasoning from infrastructure

The layering, bottom to top:

```
core          scientific vocabulary                    (no internal dependencies)
evidence      what happened, append-only
execution     how to make things happen, anywhere
knowledge     what it all means, joined
search        which move to make
roles         who makes it, and what they each want
orchestration putting the above together
publication   reporting  (empty)
```

Dependencies point downward only, and `core` imports nothing from its siblings.

The reason is practical rather than architectural taste. The infrastructure
layer will be rewritten many times — new executors, new backends, new storage,
new providers. The scientific vocabulary should not move when it is. Equally,
when the scientific model turns out to be wrong (it will), that change should
not require touching subprocess handling.

The clearest expression of this rule is the `ExperimentSpec` / `ExperimentJob`
split: what a scientist would write down, and what a machine needs to run it,
are different objects that happen to be related by an id.
