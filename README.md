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

```
core          scientific vocabulary: state, actions + attempts, questions,
              hypotheses, predictions + their tests, experiments, evidence,
              claims, assessments, proposals, commit bundles, decisions,
              budgets, replication groups              (depends on nothing)
evidence      append-only storage of what actually happened, plus the
              deterministic evidence-chain validator
execution     binding an experiment design to a process, anywhere it runs —
              job-private workspace, home and environment; artifact-aware,
              no silent success
knowledge     factual read models — the claim-evidence graph, the lesson shape
persistence   content-addressed state snapshots, reconstructible offline
runtime       the cost-aware layer: frontier view, Tier-0 validation,
              reasoning tiers + escalation, runtime metrics, playbooks,
              development/held-out evaluation seam     (depends on core only)
search        selection policies over evaluated candidates
roles         specialized agents; explicit invocations in, proposals out
orchestration the director (one deliberation: candidates → valuation →
              selection), the runtime loop, deterministic routing, critic +
              synthesis triggers, atomic transitions, the trajectory log
publication   reporting (deliberately empty)
```

Dependencies point downward only, and `core` imports nothing from its siblings —
enforced by a test, not by convention.

Nine commitments shape the rest:

- **Scientific state is explicit data.** The authoritative state of a research
  program lives in a structured, immutable `ResearchState`, not in a model's
  conversation history.
- **Propositions carry no truth status.** Hypotheses, predictions and claims
  are immutable propositions; what an execution observed about a prediction is
  a per-run `PredictionTest`, and what is currently believed is a versioned
  `EpistemicAssessment`. Standing is always a query, never a field.
- **Progress is a search over typed actions**, not a fixed pipeline of stages —
  and an action's *intent* is separate from each *attempt* at executing it. A
  failed attempt leaves the work open; it never makes it look done.
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
- **Roles propose; commits are atomic.** Roles receive explicit invocations
  (context, allowed actions, output contract) and produce typed, attributable
  proposals. An attempt's proposals and outcome commit together as a
  `CommitBundle` — all or nothing, and a successful attempt cannot claim
  outputs that were not committed. Every decision is trajectory-logged and
  every decision-boundary state is snapshot to disk, from step one.
- **Rich state, sparse reasoning invocations.** A domain abstraction does not
  imply an LLM call or an agent. An ordinary experiment iteration makes one
  director deliberation and one executor invocation — the quantity the loop
  can enforce; actual provider calls and tokens are recorded separately from
  provider reports, and are zero for the rule-based roles that exist today.
  Every returned result — failed executions included — passes a deterministic
  validation gate *before* it can enter scientific state: it must name its
  assigned experiment and verify its artifact provenance, and a successful
  result must additionally carry its declared metrics, finite values, and
  seed. When the gate rejects work that already ran, the scientific commit is
  refused but the operational record survives: actual cost and runtime are
  still metered and billed against the budget. Roles — the critic included —
  commit only proposal kinds within their invocation's output contract, and a
  critic is invoked only when a deterministic trigger finds a scientific
  reason — never to review arithmetic.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the reasoning and
[docs/ROADMAP.md](docs/ROADMAP.md) for where it is going.

## Setup

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The package has no runtime dependencies.

## Running

```bash
python examples/runtime_loop.py
```

This runs the research runtime twice on a deliberately trivial question — is
a seeded draw stream biased? — and prints both trajectories with their
invocation accounting. The normal scenario walks

```
frontier → director deliberates once → deterministic routing → executor
(job-private workspace, home and environment) → deterministic validation
gate → critic trigger evaluates to false → atomic commit → new frontier
```

at two reasoning invocations — and zero actual model calls, recorded as
such — per experiment iteration; the escalated scenario produces a
contradictory replication, the deterministic critic trigger fires, and a
critic review (plus a slow-loop synthesis) is added — and only then.

```bash
python examples/minimal_loop.py
```

The original demo of the decomposed decision path (generator → utilities →
policy). It walks the same contract at finer action granularity and prints
the resulting trajectory:

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
