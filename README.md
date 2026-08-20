# Autonomous AI Research Lab

Infrastructure for AI systems that do scientific research on their own:
form hypotheses, run experiments, and weigh the evidence, all within a
budget.

The core problem: language models can produce text that looks like
research even when nothing was measured. This codebase makes that
structurally impossible:

- A result can only come from a process that actually ran. No component
  can write one from reasoning.
- Results are never edited after the fact.
- Every claim points to the evidence behind it.
- Belief in a hypothesis is a separate, dated, authored record.
  Changing a verdict means adding a new record, never editing the old
  one.

## Status

Early and experimental, but past the placeholder stage. Proven in live
end-to-end runs:

- **A model connection.** One provider-neutral seam with one concrete
  adapter (Muse Spark 1.2). Replies are validated locally as structured
  JSON. Token usage is billed exactly once per call. A failed call is a
  typed infrastructure event and can never become a scientific result.
- **A model-backed research engineer.** Turns experiment specs into
  code from trusted templates and runs it in a locked-down disposable
  container (no network, pinned image, read-only root). Proven across a
  four-task live campaign, 6/6 implementations verified.
- **A model-backed research planner.** Picks exactly one next
  scientific action from the verified evidence on record, behind a
  deterministic gate. The full loop (planning, engineering, contained
  execution, verification, re-planning) has run live as one autonomous
  trajectory with complete provenance.
- **Bounded literature retrieval** (OpenAlex, behind a provider-neutral
  seam). Normalized source records, conservative deduplication,
  write-once search provenance, and a local corpus that replays
  identical completed searches without touching the network.
- **Evidence-grounded field mapping.** Model-proposed queries executed
  by trusted code, preserved relevance screening, verbatim-grounded
  extraction under deterministic gates, and a durable trusted-code
  adequacy verdict with per-problem support tiers. An honest
  `INSUFFICIENT_COVERAGE` is a successful outcome.
- **CFP-directed candidate idea generation**, behind that adequacy
  guard. An immutable hashed snapshot of a real workshop call, one
  gated portfolio call, and fully structured candidates: falsifiable
  hypothesis, predictions with explicit falsifiers, dataset needs,
  baselines, ablations, risks, and prior-art search terms. Novelty
  starts structurally `UNASSESSED`. An honest refusal is a recorded
  outcome, and the portfolio report names every problem the candidates
  did not address.
- **An adversarial prior-art challenge** over that portfolio. One
  guarded door in. A budget preflight refuses a run that cannot finish
  the work its own settings allow. Per candidate: six trusted-dated
  query families of fresh OpenAlex retrieval, the candidate's own
  cited works injected into the pool and screened first, gated
  screening (metadata-only sources get their own call, where only an
  attested overlap hypothesis can block), nearest-work comparisons
  across five dimensions with every quote re-found verbatim, and a
  deterministic fail-closed verdict: `OVERLAPPING`, `DISTINGUISHED`
  (differentiated from the closest works this bounded search found —
  never proof of novelty), or `NOVELTY_UNRESOLVED`. Proven live three
  times over the same three candidates: the first run exposed its own
  retrieval defect, the second refused on blockers that turned out to
  restate access level and configuration, and the third — after the
  Task 5D.2 calibration — returned three `DISTINGUISHED` verdicts with
  zero truncation and spend that reconciles exactly with the ledger.
  Zero-network replay verified every time. The candidate records stay
  untouched.
- **Candidate selection over the challenged portfolio.** One guarded
  door in: the directive names one prior-art run record, and trusted
  code computes eligibility from that run's `DISTINGUISHED` verdicts
  alone — no "latest assessment" inference exists. Two gated model
  calls (a comparative review of every eligible candidate and every
  pair, then one final choice among the candidates no validated
  disqualifier removed), under score-free schemas with nowhere to put
  a stop. Disqualifiers are narrow and attested: both quotes re-found
  verbatim, in the candidate's own record and in the operator's stated
  constraint. Three honest outcomes — `SELECTED`,
  `NO_ELIGIBLE_CANDIDATE` (decided by trusted code, zero calls, zero
  spend), `NO_DEFENSIBLE_CANDIDATE` (a validated disqualifier per
  eligible candidate) — structurally distinct and never conflated.
  Proven live over the 5D.2 portfolio: three eligible, zero
  disqualifiers, one selection in two calls with zero corrective
  calls, spend reconciling exactly with the ledger, and every
  preserved upstream artifact byte-identical before and after. A
  selection is a model preference validated — never computed — by
  trusted code, and never proof of novelty.
- **Governed admission into research state.** One guarded door in: the
  directive names one selection run record, and trusted code
  re-verifies the whole lineage behind it — cross-checking the records
  against each other, not only each against itself — before requiring
  the outcome `SELECTED`. One gated model call translates the
  candidate's recorded predictions into machine-checkable form under a
  sign-only neutral encoding (the model never authors a number;
  comparator and threshold are structural constants); everything else
  the initial state holds is a deterministic verbatim copy. The state
  snapshot and the write-once admission record are one all-or-nothing
  artifact set, one admission exists per selection run ever, and a
  completed directive replays its stored result at zero model calls.
  Proven live over the 5E selection: one call, zero corrective calls,
  spend reconciling exactly with the ledger, every preserved upstream
  artifact byte-identical, and the replay served through a provider
  that refuses every call. `DISTINGUISHED`, `SELECTED`, and `ADMITTED`
  are rungs of bounded process — none of them means true, novel, or
  empirically supported.
- **A funded research run.** The bridge from an admitted seed to
  something a runtime may spend against. An admitted state carries no
  budget, and a state's identity excludes its budget, so funding one in
  place would leave two different snapshots claiming one id — which the
  append-only snapshot store refuses, correctly. Funding is succession
  instead: the funded state is a child of the admitted one, which keeps
  its zero budget forever. What is spent afterwards is a ledger fact,
  not a field rewrite — sequence-numbered, chained, write-once entries,
  idempotent by charge id, so a balance survives a restart, a repeated
  charge cannot debit twice, and two concurrent debits cannot overspend.
  The runtime loop posts one debit per attempt and fails closed if the
  ledger and the state ever disagree. Proven over the preserved Task 5F
  admission with zero model calls and zero network calls: the grant on
  the ledger, one charge, the same charge again debiting nothing, the
  balance replayed from a fresh store, a disagreement refused, the
  completed directive replaying its run, and every admission file
  byte-identical afterwards. A grant is authorization, never scientific
  standing.

Nothing a model says becomes scientific state until deterministic code
has gated, committed, executed, and verified it. Literature-derived
records describe, conjecture, and prefer, carrying their sources; the
one path into scientific state is admission's governed door, which
turns exactly one validated selection into a bare initial state of
propositions — no result, evidence, or judgment is expressible there.
Funding that seed is a separate operator act with its own record and
its own ledger: it buys the chance to find something out, never a
finding. Orchestration is not model-driven: a fixed-priority director
dispatches work, and every model decision passes a deterministic gate
before it takes effect.

Not built yet: access resolution for metadata-only retrieved works,
real experiment execution over the funded state (the execution
requirements admission carries are stated capabilities, not
implementations, and the admitted predictions' metrics match no
trusted template yet), one command that carries a topic through the
whole chain, a durable evidence store behind the in-memory one,
literature-grounded findings entering research state, cloud execution,
statistician and skeptic roles with real inference, and paper writing.

Expect interfaces to change.

## How the code is organized

| Package | What it does |
| --- | --- |
| `core` | The basic data types: questions, hypotheses, predictions, experiments, results, evidence, claims, assessments, budgets, and the research state that ties them together. Depends on nothing else. |
| `evidence` | Storage for results and evidence. Records can be added but never changed. |
| `execution` | Runs an experiment as a subprocess. Each job gets its own working directory, its own home directory, and a minimal environment. Output files are collected and hashed. |
| `knowledge` | Read-only views over the data, such as the graph linking claims to evidence. |
| `literature` | Bounded search against a real scholarly API (OpenAlex): normalized snapshot records, deterministic deduplication, write-once search provenance, and a replayable local corpus. Depends only on `core`; its one consumer is `mapping`. |
| `mapping` | What the literature adds up to: model-backed field maps and problem inventories, deterministically gated and write-once, judged by a trusted-code adequacy verdict. Depends on `core`, `literature`, and the provider seam; its one consumer is `ideation`. |
| `ideation` | What might be worth investigating: CFP-directed candidate generation over the assessed map. Depends on `core`, `mapping`, and the provider seam; its one consumer is `priorart`. |
| `priorart` | Whether it was already done: the prior-art challenge over the candidate portfolio, with a deterministic fail-closed verdict per candidate. Depends on `core`, `literature`, `mapping`, `ideation`, and the provider seam; its consumers are `selection` and `admission`. |
| `selection` | Which candidate to pursue, if any: gated two-stage selection over the `DISTINGUISHED` survivors of one named prior-art run, with attested disqualifiers and three honest outcomes. Score-free and write-once. Depends on `core`, `ideation`, `mapping`, `priorart`, and the provider seam; its one consumer is `admission`. |
| `admission` | The governed bridge into research state: one named `SELECTED` run verified through its whole lineage, one gated model call encoding the recorded predictions sign-only, deterministic copies for everything else, and an all-or-nothing state snapshot beside a write-once record. Depends on `core` (uniquely including the state it constructs), `ideation`, `mapping`, `priorart`, `selection`, `persistence`, and the provider seam; nothing imports it. |
| `program` | A funded run: the bridge from one admitted state to something the runtime may spend against. One named admission, one authorized grant, a funded successor state, and an append-only budget ledger that is idempotent by charge id and safe under concurrent debits. Depends on `core`, `admission`, and `persistence`; nothing imports it. |
| `persistence` | Saves every state to disk so a run can be inspected or replayed later. |
| `runtime` | Bookkeeping around the loop: open work, validation and verification, cost tracking, metrics, the model-provider seam (with the Muse adapter), and the write-once stores for implementation and planning provenance. |
| `search` | Policies for choosing the next action among candidates. |
| `roles` | Interfaces for the agents, including the model-backed engineer and planner. A role receives a task and returns proposals. It never edits state directly. |
| `orchestration` | The main loop: pick an action, route it to a role, validate what comes back, commit it, log everything. |
| `publication` | Paper writing. Empty for now. |

Dependencies point one way: higher packages import lower ones, never
the reverse, and `core` imports nothing. A test checks this.

More detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/ROADMAP.md](docs/ROADMAP.md).

## The main rules

How data is kept:

- The full state of a research program lives in one immutable data
  structure, not in a model's chat history.
- A hypothesis never stores whether it is true. Each run records
  whether its outcome matched the prediction. Belief in the hypothesis
  is a separate signed record. New judgments are added; old ones stay.
- Predictions are registered before the experiment runs: which metric,
  which comparison, which threshold. Code does the check, so the
  criterion cannot be adjusted afterwards to fit the outcome.
- Only the executor can produce a result, and only from a process that
  ran. Repeat runs are stored as separate records.
- Failed runs and negative results are recorded like any other outcome,
  never dropped.

How the loop works:

- Work is a sequence of typed actions: generate a hypothesis, design an
  experiment, run it, analyze the result, and so on. A failed attempt
  leaves the action open. It never counts as done.
- Roles return proposals. A separate transition layer checks and
  commits them. Everything an attempt produced commits together or not
  at all.
- A normal step makes two reasoning calls: one for the director to pick
  an action, one for the role that performs it. A critic is consulted
  only when a fixed rule says the result warrants it, for example when
  two runs of the same experiment disagree.
- Code checks every result before it enters the record: does it belong
  to the assigned experiment, are the output files intact, are the
  declared metrics present, are the numbers finite. Failed runs are
  checked too.
- Costs are tracked even when a result is rejected: whatever ran is
  billed against the budget. If the budget runs out, the program stops.

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

Runs the loop twice on a toy question: is a seeded random stream
biased?

```
frontier → director picks an action → routed to a role → executor runs it
→ result validated → commit → next step
```

The first run is the normal path: each experiment step costs two
reasoning calls and no critic. The second run makes two runs of the
same experiment disagree, which triggers the critic and a synthesis
pass. The script prints both trajectories and the call counts.

### examples/minimal_loop.py

```bash
python examples/minimal_loop.py
```

An earlier, more granular demo of the same contracts. Its hypothesis
gets refuted: the metric is read from a file written by the
subprocess, the pre-registered prediction check fails, and the
refutation is recorded as an assessment. Every decision is logged and
every intermediate state is saved to disk.

## Tests and checks

```bash
pytest
ruff check .
mypy
```

Known flaky behavior is tracked in
[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md), not in anyone's memory.

## Troubleshooting

### `ModuleNotFoundError: autonomous_research_lab` from `.venv/bin/python` (hidden `.pth`)

**Symptom.** Every process using the venv fails to import the package.
`pytest` cannot collect, and live jobs die at launch with
`Error while finding module specification for
'autonomous_research_lab.execution.container_shim'`. Preflight reports
`preflight:pth_files_visible` FAIL.

**Mechanism.** CPython ≥ 3.11.9 silently skips `.pth` files whose BSD
flags carry `UF_HIDDEN` (`python -v` prints `Skipping hidden .pth
file: ...`). The editable install's
`_editable_impl_autonomous_research_lab.pth` is what puts `src/` on
`sys.path`. When it is flagged, the package vanishes from every venv
interpreter, including executor children, which run with a minimal
environment that excludes `PYTHONPATH`.

**Cause (external).** iCloud Drive "Desktop & Documents" sync re-flags
dot-items under a synced repository. No repository code sets file
flags.

**Fix (operator action; the lab never runs this itself).**

```bash
chflags -R nohidden .venv
.venv/bin/python -c "import autonomous_research_lab"   # verify
```

Expect recurrence while the repository sits inside a synced folder.
The durable fix is moving it out of `~/Desktop` / `~/Documents`, or
excluding it from iCloud sync.

## License

MIT. See [LICENSE](LICENSE).
