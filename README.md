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

Early and experimental, but past the placeholder stage. What exists and
has been proven in live end-to-end runs:

- **A model connection.** One provider-neutral seam with a single
  concrete adapter (Muse Spark 1.2). Replies are validated locally as
  structured JSON, token usage is billed exactly once per call, and a
  failed call is a typed infrastructure event that can never be recorded
  as a scientific result.
- **A model-backed research engineer** that turns experiment specs into
  code from trusted templates and runs it in a locked-down disposable
  container (no network, pinned image, read-only root) — proven across a
  four-task live campaign, 6/6 implementations verified.
- **A model-backed research planner** that selects exactly one next
  scientific action from the verified evidence on record, behind a
  deterministic gate. The full loop — planning → engineering → contained
  execution → verification → re-planning — has run live as a single
  autonomous trajectory with complete provenance.
- **Bounded literature retrieval** (OpenAlex, behind a provider-neutral
  seam): normalized source records, conservative deduplication,
  write-once search provenance, and a local corpus that replays
  identical completed searches without touching the network. Its one
  consumer is the field mapper below.
- **Evidence-grounded field mapping with a deterministic adequacy
  verdict.** Model-proposed queries executed by trusted code
  (title/abstract matching; recency-ordered recent work beside
  citation-ranked foundational work), preserved relevance screening,
  verbatim-grounded extraction under deterministic gates, a
  source-grounded field map and problem inventory with computed
  per-problem support tiers, and a durable trusted-code
  `MapAdequacyAssessment` — where an honest `INSUFFICIENT_COVERAGE` is
  a successful outcome.
- **CFP-directed candidate idea generation** behind that verdict's
  guard: an immutable hashed snapshot of a real workshop call, a gated
  verbatim direction extraction, and one gated portfolio call producing
  bounded, fully structured research candidates — falsifiable
  hypothesis, predictions with explicit falsifiers, dataset
  requirements, baselines, ablations, risks, prior-art search terms —
  with problem statements, support tiers, and source-era mixes stamped
  by trusted code, novelty structurally `UNASSESSED`, honest refusal as
  a first-class recorded outcome, and portfolio accounting that names
  every problem the candidates did not address.

- **An adversarial prior-art challenge** over that portfolio: one
  guarded door in, a budget-coherence preflight that refuses a
  directive unable to complete its own promised work, six trusted-dated
  query families of fresh OpenAlex retrieval per candidate, the
  candidate's own cited works injected into the pool through an
  identifier dedup bridge and screened first, gated similarity
  screening — metadata-only sources in their own gated call, where
  only an attested material overlap hypothesis (both ends re-found
  verbatim in recorded text) can block — nearest-work comparisons
  across five explicit dimensions with every quoted snippet re-found
  verbatim in accessible text, and a deterministic fail-closed verdict
  per candidate — `OVERLAPPING`, `DISTINGUISHED` (differentiated from
  the closest works this bounded search surfaced — never proof of
  novelty), or `NOVELTY_UNRESOLVED`. Proven live three times over the
  same three preserved candidates: first exposing its own retrieval
  defect (ten-term conjunctive queries, sixteen empty searches of
  eighteen), then — through the trusted concept-group boundary of
  Task 5D.1 — refusing on pools of 9/23/17 with every blocker an
  undecidable title-only screen, a nine-source pool, or a truncation
  the directive's own budgets caused; and then, after the Task 5D.2
  calibration (blocker audit, attested materiality, re-based rules
  with unchanged threshold values, coherence preflight, a calibration
  suite proving all three verdicts reachable, and a read-only
  counterfactual replay that changed one verdict of three), returning
  three `DISTINGUISHED` verdicts on pools of 11/22/20 with zero
  truncation and spend reconciling exactly with the ledger. Full
  zero-network replay verified every time, and the candidate records
  stay untouched, their own novelty status still structurally
  unassessed.

Nothing a model says becomes scientific state until deterministic code
has gated, committed, executed, and verified it — and literature-derived
records never become scientific state at all: maps, inventories, and
candidate ideas describe and conjecture, carrying their sources, until
a later task admits findings through the same governed commit as every
other proposal. Orchestration itself is deliberately not model-driven:
a fixed-priority director dispatches work, and every model decision
passes a deterministic gate before it takes effect.

What does not exist yet: access-level resolution for metadata-only
retrieved works (an attested material ambiguity still has no lawful
in-repo path to an abstract), Task 5E candidate selection over the
challenge verdicts (its ingress contract is defined: selectable only
on a `DISTINGUISHED` assessment, with honest distinguishable stops),
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
| `mapping` | What the literature adds up to: model-backed field mapping and problem inventories over the retrieved corpus, deterministically gated and write-once, judged by a trusted-code adequacy verdict. Depends on `core`, `literature`, and the provider seam; its one consumer is `ideation`. |
| `ideation` | What might be worth investigating: CFP-directed candidate generation over the assessed map, entering through the adequacy guard, tier-stamped and source-carrying, novelty structurally unassessed. Depends on `core`, `mapping`, and the provider seam; its one consumer is `priorart`. |
| `priorart` | Whether it was already done: the adversarial prior-art challenge over the candidate portfolio — fresh bounded retrieval, gated screening and nearest-work comparison, and a deterministic fail-closed verdict per candidate. Depends on `core`, `literature`, `mapping`, `ideation`, and the provider seam; nothing imports it. |
| `persistence` | Saves every state to disk so a run can be inspected or replayed later. |
| `runtime` | Bookkeeping around the loop: what work is open, validation and verification of results, cost tracking, metrics — plus the model-provider seam (with the Muse adapter) and the write-once stores for implementation and planning provenance. |
| `search` | Policies for choosing the next action among candidates. |
| `roles` | Interfaces for the agents, including the model-backed engineer and planner. A role receives a task and returns proposals. It never edits state directly. |
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

Known flaky behavior is tracked in
[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md), not in anyone's memory.

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
