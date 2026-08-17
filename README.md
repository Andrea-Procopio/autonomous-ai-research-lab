# Autonomous AI Research Lab

A research infrastructure project for building autonomous systems capable of
conducting rigorous scientific research end-to-end.

> The system should optimize for producing reliable scientific knowledge, not
> for producing papers that merely look convincing.

That sentence is the design constraint, not a slogan. Language models are
fluent enough to produce research-shaped output — plausible hypotheses, tidy
narratives, confident conclusions — without any of it being anchored to
something measured. Most of the architecture here exists to make that failure
mode structurally hard: experimental results are immutable and can only be
produced by a process that actually ran; claims carry links to the evidence
behind them; and a refuted hypothesis stays refuted on the record — the
refutation is a signed, versioned assessment that cannot be quietly reframed,
because the hypothesis itself carries no status to rewrite.

## Status

**Early architecture. Experimental research project.**

What exists today is a set of domain contracts and one working loop over them.
The system does **not** currently conduct autonomous research: there is no model
provider, no literature access, no cloud execution, and no manuscript
generation. The orchestrator is a rule-based reference implementation that
closes obvious gaps in the state — bookkeeping, not scientific judgement.

The architecture is under active development and interfaces will change.

## Architecture at a glance

The package is a stack of small layers. Dependencies point downward only, and
`core` imports nothing from its siblings — enforced by a test, not by
convention.

| Layer | Responsibility |
| --- | --- |
| `core` | The scientific vocabulary: research state, actions and attempts, questions, hypotheses, predictions and their tests, experiments, evidence, claims, assessments, proposals, commit bundles, decisions, budgets, replication groups. Depends on nothing. |
| `evidence` | Append-only storage of what actually happened, plus the deterministic evidence-chain validator. |
| `execution` | Binding an experiment design to a process, anywhere it runs. Job-private workspace, home directory and environment; artifact-aware; no silent success. |
| `knowledge` | Factual read models: the claim–evidence graph, the lesson shape. |
| `persistence` | Content-addressed state snapshots, reconstructible offline. |
| `runtime` | The cost-aware layer: the frontier view, Tier-0 deterministic validation, reasoning tiers and escalation, runtime metrics, playbooks, the development/held-out evaluation seam. Depends on `core` only. |
| `search` | Selection policies over evaluated candidates. |
| `roles` | Specialized agents: explicit invocations in, typed proposals out. Roles never mutate state. |
| `orchestration` | The director (one deliberation: candidates → valuation → selection), the runtime loop, deterministic routing, critic and synthesis triggers, atomic transitions, the trajectory log. |
| `publication` | Reporting. Deliberately empty for now. |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the reasoning behind each
layer and [docs/ROADMAP.md](docs/ROADMAP.md) for where it is going.

## Design commitments

### The scientific record

- **Scientific state is explicit data.** The authoritative state of a research
  program lives in a structured, immutable `ResearchState`, not in a model's
  conversation history.
- **Propositions carry no truth status.** Hypotheses, predictions and claims
  are immutable propositions; what an execution observed about a prediction is
  a per-run `PredictionTest`, and what is currently believed is a versioned
  `EpistemicAssessment`. Standing is always a query, never a field.
- **Hypotheses commit through predictions.** A hypothesis is testable only via
  pre-registered, machine-checkable predictions (metric, comparator,
  threshold), checked mechanically when each result is committed — never
  adjusted to fit the outcome.
- **Evidence is immutable and machine-produced.** An `ExperimentResult` comes
  only from an executor that ran a process. Interpretations reference evidence;
  they never overwrite it. Independent replications stay independent records,
  grouped into families by protocol identity — never by what they observed.
- **Evidence is not interpretation.** How evidence bears on a claim is a
  factual annotation; whether the claim should be believed is an
  `EpistemicAssessment` with its own author, method, and version history. No
  count of evidence edges produces a verdict anywhere.
- **Negative results are first-class.** Failed runs, inconsistent prediction
  tests, failed attempts and contradicted claims are recorded outcomes with
  provenance.

### How work happens

- **Progress is a search over typed actions**, not a fixed pipeline of stages —
  and an action's *intent* is separate from each *attempt* at executing it. A
  failed attempt leaves the work open; it never makes it look done.
- **Roles propose; commits are atomic.** Roles receive explicit invocations
  (context, allowed actions, output contract) and produce typed, attributable
  proposals. An attempt's proposals and outcome commit together as a
  `CommitBundle` — all or nothing, and a successful attempt cannot claim
  outputs that were not committed. Every decision is trajectory-logged and
  every decision-boundary state is snapshot to disk, from step one.

### Runtime discipline

- **Rich state, sparse reasoning invocations.** A domain abstraction does not
  imply an LLM call or an agent. An ordinary experiment iteration makes
  exactly two reasoning-seat invocations — one director deliberation and one
  executor invocation. A critic is added only when a deterministic trigger
  finds a *scientific* reason, never to review arithmetic. Actual provider
  calls and tokens are recorded separately, from provider reports, and are
  zero for the rule-based roles that exist today.
- **Nothing enters scientific state unchecked.** Every returned result —
  failed executions included — passes a deterministic validation gate *before*
  it can commit: it must name its assigned experiment and verify its artifact
  provenance, and a successful result must additionally carry its declared
  metrics, finite values, and seed. Every role, the critic included, may
  commit only the proposal kinds its invocation's output contract allows.
- **Work is billed at what it actually cost.** The operational execution
  record is kept separate from scientific state: when the gate rejects work
  that already ran, nothing scientific commits, but the actual cost and
  runtime are still metered and billed against the budget. An overrun is
  recorded explicitly and halts the program safely — committed-but-unbilled
  work cannot exist.

## Setup

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The package has no runtime dependencies.

## Running the demos

### The research runtime

```bash
python examples/runtime_loop.py
```

Runs the research runtime twice on a deliberately trivial question — is a
seeded draw stream biased? — and prints both trajectories with their
invocation accounting.

The **normal** scenario walks

```
frontier → director deliberates once → deterministic routing → executor
(job-private workspace, home and environment) → deterministic validation
gate → critic trigger evaluates to false → atomic commit → new frontier
```

at two reasoning invocations — and zero actual model calls, recorded as
such — per experiment iteration. The **escalated** scenario produces a
contradictory replication: the deterministic critic trigger fires, and a
critic review (plus a slow-loop synthesis) is added — and only then.

### The decomposed decision path

```bash
python examples/minimal_loop.py
```

The original demo of the decomposed decision path (generator → utilities →
policy). It walks the same contract at finer action granularity:

```
ResearchState → director (candidates → utilities → policy) → ActionAttempt
→ proposals → transition layer → ExperimentSpec → LocalExecutor
→ ExperimentResult → evidence → prediction checked → assessment → ResearchState'
```

The demo hypothesis is false, and the run says so at every layer. The
`heads_rate` was read out of a `metrics.json` written by a subprocess; when
the result commits, the pre-registered prediction is mechanically tested and
the run's `PredictionTest` records `inconsistent`; and the claim's `refuted`
standing comes from an explicit epistemic assessment that names its method and
the evidence it considered — the hypothesis and prediction objects themselves
never change. Every decision along the way is preserved in a JSONL trajectory
log, and every decision-boundary state is persisted as a content-addressed
snapshot that reconstructs offline.

## Tests and checks

```bash
pytest
ruff check .
mypy
```

## License

MIT — see [LICENSE](LICENSE).
