# Autonomous AI Research Lab

Infrastructure for AI systems that do scientific research on their own:
form hypotheses, run experiments, and weigh the evidence, all within a
budget.

The problem this project is built around: language models are good at
producing text that looks like research even when nothing behind it was
actually measured. The codebase is designed so that this cannot happen
quietly.

- A result can only come from a process that actually ran. No component
  can write one from reasoning.
- Results are never edited after the fact.
- Every claim points to the evidence behind it.
- Whether a hypothesis is currently believed is a separate, dated record
  with an author. Changing a verdict means adding a new record, not
  editing the old one.

## Status

Early and experimental. What exists is the data model and one working
loop that runs experiments end to end. What does not exist yet: any
connection to a language model, literature search, cloud execution, or
paper writing. The current decision-making is a simple rule-based
placeholder.

Expect interfaces to change.

## How the code is organized

| Package | What it does |
| --- | --- |
| `core` | The basic data types: questions, hypotheses, predictions, experiments, results, evidence, claims, assessments, budgets, and the research state that ties them together. Depends on nothing else. |
| `evidence` | Storage for results and evidence. Records can be added but never changed. |
| `execution` | Runs an experiment as a subprocess. Each job gets its own working directory, its own home directory, and a minimal environment. Output files are collected and hashed. |
| `knowledge` | Read-only views over the data, such as the graph linking claims to evidence. |
| `persistence` | Saves every state to disk so a run can be inspected or replayed later. |
| `runtime` | Bookkeeping around the loop: what work is open, validation of results, cost tracking, metrics. |
| `search` | Policies for choosing the next action among candidates. |
| `roles` | Interfaces for the agents (director, engineer, critic). A role receives a task and returns proposals. It never edits state directly. |
| `orchestration` | The main loop: pick an action, route it to a role, validate what comes back, commit it, log everything. |
| `publication` | Paper writing. Empty for now. |

Dependencies point one way: higher packages import lower ones, never the
reverse, and `core` imports nothing. A test checks this.

More detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/ROADMAP.md](docs/ROADMAP.md).

## The main rules

How data is kept:

- The full state of a research program lives in one immutable data
  structure, not in a model's chat history.
- A hypothesis never stores whether it is true. Each run records whether
  its outcome matched the prediction, and belief in the hypothesis is a
  separate signed record. New judgments are added; old ones stay.
- Predictions are registered before the experiment runs: which metric,
  which comparison, which threshold. The check against the result is done
  by code, so the criterion cannot be adjusted afterwards to fit the
  outcome.
- Only the executor can produce a result, and only from a process that
  ran. Repeat runs of the same experiment are stored as separate records.
- Failed runs and negative results are recorded like any other outcome,
  not dropped.

How the loop works:

- Work is a sequence of typed actions: generate a hypothesis, design an
  experiment, run it, analyze the result, and so on. A failed attempt
  leaves the action open. It never counts as done.
- Roles return proposals. A separate transition layer checks and commits
  them. Everything an attempt produced is committed together or not at
  all.
- A normal step makes two reasoning calls: one for the director to pick
  an action, one for the role that performs it. A critic is consulted
  only when a fixed rule says the result warrants it, for example when
  two runs of the same experiment disagree.
- Every result is checked by code before it enters the record: does it
  belong to the assigned experiment, are the output files intact, are the
  declared metrics present, are the numbers finite. Failed runs are
  checked too.
- Costs are tracked even when a result is rejected: whatever actually ran
  is billed against the budget. If the budget runs out, the program
  stops.

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

Runs the loop twice on a toy question: is a seeded random stream biased?

```
frontier → director picks an action → routed to a role → executor runs it
→ result validated → commit → next step
```

The first run is the normal path: each experiment step costs two
reasoning calls and no critic. The second run is set up so that two runs
of the same experiment disagree, which triggers the critic and a
synthesis pass. The script prints both trajectories and the call counts.

### examples/minimal_loop.py

```bash
python examples/minimal_loop.py
```

An earlier, more granular demo of the same contracts. Its hypothesis gets
refuted: the metric is read from a file written by the subprocess, the
pre-registered prediction check fails, and the refutation is recorded as
an assessment. Every decision is logged and every intermediate state is
saved to disk.

## Tests and checks

```bash
pytest
ruff check .
mypy
```

## Troubleshooting

### `ModuleNotFoundError: autonomous_research_lab` from `.venv/bin/python` (hidden `.pth`)

**Symptom.** Every process using the venv fails to import the package —
`pytest` cannot collect, and live jobs die at launch with
`Error while finding module specification for
'autonomous_research_lab.execution.container_shim'`. Preflight reports
`preflight:pth_files_visible` FAIL.

**Mechanism.** CPython ≥ 3.11.9 `site.py` silently skips `.pth` files whose
BSD flags carry `UF_HIDDEN` (`python -v` prints `Skipping hidden .pth
file: ...`). The editable install's
`_editable_impl_autonomous_research_lab.pth` is what puts `src/` on
`sys.path`, so when it is flagged the package vanishes from every venv
interpreter — including executor children, which are spawned with a
minimal environment that deliberately excludes `PYTHONPATH`.

**Cause — external.** iCloud Drive "Desktop & Documents" sync re-flags
dot-items under a synced repository (observed re-flagging `.venv`
item-by-item after all repo processes had exited). No repository code sets
file flags.

**Remediation (operator action; the lab never runs this itself).**

```bash
chflags -R nohidden .venv
.venv/bin/python -c "import autonomous_research_lab"   # verify
```

Expect recurrence while the repository stays inside a synced folder. The
durable fix is moving the repository out of `~/Desktop` / `~/Documents`
(or excluding it from iCloud sync).

## License

MIT. See [LICENSE](LICENSE).
