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

Nothing a model says becomes scientific state until deterministic code
has gated, committed, executed, and verified it. Literature-derived
records never become scientific state at all: maps, inventories, and
candidate ideas describe and conjecture, carrying their sources, until
a later task admits findings through the same governed commit as every
other proposal. Orchestration is not model-driven: a fixed-priority
director dispatches work, and every model decision passes a
deterministic gate before it takes effect.

Not built yet: access resolution for metadata-only retrieved works,
Task 5E candidate selection (its entry rule is defined: a candidate is
selectable only on a `DISTINGUISHED` assessment), literature-grounded
findings entering research state, cloud execution, statistician and
skeptic roles with real inference, and paper writing.

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
| `priorart` | Whether it was already done: the prior-art challenge over the candidate portfolio, with a deterministic fail-closed verdict per candidate. Depends on `core`, `literature`, `mapping`, `ideation`, and the provider seam; nothing imports it. |
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
