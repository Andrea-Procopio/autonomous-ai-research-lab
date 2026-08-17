# Autonomous AI Research Lab

Infrastructure for autonomous systems that conduct scientific research:
forming hypotheses, running experiments, and evaluating evidence under an
explicit budget.

The central design problem is that language models can produce
research-shaped output — hypotheses, narratives, conclusions — that is not
anchored to anything measured. The architecture is built to make that
failure hard: experimental results are immutable and can only be produced by
a process that actually ran, claims link to the evidence behind them, and
belief status is recorded as versioned assessments rather than as mutable
fields on hypotheses.

## Status

Early-stage, experimental. The repository contains the domain model and one
working orchestration loop over it. There is no model provider integration,
no literature access, no cloud execution, and no manuscript generation. The
current orchestrator is a rule-based reference implementation used to
exercise the contracts.

Interfaces will change.

## Architecture

The package is organized as layers. Dependencies point downward only, and
`core` imports nothing from the other packages; a test enforces this.

| Layer | Responsibility |
| --- | --- |
| `core` | Domain types: research state, actions and attempts, questions, hypotheses, predictions and prediction tests, experiments, evidence, claims, assessments, proposals, commit bundles, decisions, budgets, replication groups. No dependencies. |
| `evidence` | Append-only stores for results and evidence; the evidence-chain validator. |
| `execution` | Runs experiment jobs as subprocesses with a job-private working directory, home directory, and filtered environment. Collects metrics and artifacts and records their hashes in a manifest. |
| `knowledge` | Read models: the claim–evidence graph, lessons. |
| `persistence` | Content-addressed state snapshots, reconstructible offline. |
| `runtime` | Frontier view, deterministic result validation, reasoning tiers and escalation, runtime metrics, playbooks, evaluation seam. Depends only on `core`. |
| `search` | Selection policies over evaluated candidates. |
| `roles` | Role interfaces: explicit invocations in, typed proposals out. Roles do not mutate state. |
| `orchestration` | The director, the runtime loop, static routing, critic and synthesis triggers, the transition layer, the trajectory log. |
| `publication` | Reporting. Currently empty. |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the reasoning behind
each layer; [docs/ROADMAP.md](docs/ROADMAP.md) describes planned work.

## Design rules

State and evidence:

- The authoritative state of a research program is a structured, immutable
  `ResearchState`, not a model's conversation history.
- Hypotheses, predictions, and claims carry no truth status. What a run
  observed about a prediction is a per-run `PredictionTest`; what is
  currently believed is a versioned `EpistemicAssessment`. Standing is
  computed by query.
- Hypotheses are tested through pre-registered, machine-checkable
  predictions (metric, comparator, threshold), evaluated mechanically when a
  result is committed.
- An `ExperimentResult` is produced only by an executor that ran a process,
  and is never edited afterwards. Replications are stored as separate
  records, grouped by protocol identity.
- Evidence links (how a piece of evidence bears on a claim) are separate
  from assessments (whether the claim should be believed). Verdicts are
  never derived by counting evidence edges.
- Failed runs, inconsistent prediction tests, failed attempts, and
  contradicted claims are recorded outcomes with provenance.

Execution and commits:

- Work is a search over typed actions, not a fixed pipeline. An action's
  intent is separate from each attempt at executing it; a failed attempt
  leaves the action open.
- Roles receive explicit invocations (context, allowed actions, output
  contract) and return typed proposals. An attempt's proposals and outcome
  commit atomically as a `CommitBundle`; a successful attempt cannot claim
  outputs that were not committed. Decisions are logged and every
  decision-boundary state is snapshotted.

Validation and accounting:

- An ordinary experiment step makes two reasoning-seat invocations: one
  director deliberation and one role invocation. A critic runs only when a
  deterministic trigger fires. Provider calls and token counts are recorded
  separately, from provider reports.
- Every returned result, failed executions included, is validated
  deterministically before commit: it must reference its assigned experiment
  and pass artifact-integrity checks, and a successful result must also
  carry its declared metrics, finite values, and seed. Roles, the critic
  included, can commit only the proposal kinds their invocation's output
  contract allows.
- Execution cost is tracked separately from scientific state. Work rejected
  by validation is still billed at its actual cost and runtime. Budget
  overruns are recorded and halt the run.

## Setup

Requires Python 3.11+. The package has no runtime dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Examples

### examples/runtime_loop.py

```bash
python examples/runtime_loop.py
```

Runs the runtime twice on a small question (whether a seeded draw stream is
biased) and prints both trajectories with their invocation accounting.

```
frontier → director deliberates once → static routing → executor
→ validation gate → critic trigger (false) → atomic commit → new frontier
```

The normal scenario completes each experiment iteration in two reasoning
invocations, with zero model calls. The escalated scenario produces a
contradictory replication, which fires the critic trigger and a synthesis
pass.

### examples/minimal_loop.py

```bash
python examples/minimal_loop.py
```

Demonstrates the decomposed decision path (candidate generation → utility
evaluation → selection policy) at finer action granularity:

```
ResearchState → director → ActionAttempt → proposals → transition layer
→ ExperimentSpec → LocalExecutor → ExperimentResult → evidence
→ prediction checked → assessment → ResearchState'
```

The demo's hypothesis is refuted: metrics are read from a `metrics.json`
written by the subprocess, the pre-registered prediction is tested
mechanically at commit, and the refuted standing is recorded as an
assessment. Decisions are written to a JSONL trajectory log and each
decision-boundary state is persisted as a content-addressed snapshot.

## Tests and checks

```bash
pytest
ruff check .
mypy
```

## License

MIT — see [LICENSE](LICENSE).
