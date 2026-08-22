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
- **Facts that outlive their process.** Results, evidence, and the
  artifact bytes they point at are stored on disk under the same run
  root as the states. Recording a fact stores its bytes first, so a
  state can only reference a result whose outputs are already durable.
  Artifacts are content-addressed, kept once however many results
  produced them, and refused if they escape the run directory, no
  longer match what the run itself recorded, or exceed the size
  ceiling. Each stored record carries its own payload digest, because
  the domain ids of results and evidence deliberately do not cover
  their content — a result's id comes from its job alone — so
  recomputing an id would check nothing. One command verifies a whole
  run from cold: snapshots, payloads, references, artifact bytes, the
  evidence chain, and the budget ledger. Pointed at a demo run before
  this change it found eight state snapshots and no facts at all; after
  it, the same run verifies intact with the executor's whole run
  directory deleted.
- **One command through the chain.** `arl run CONFIG --root DIR` carries
  a topic from a brief to a funded run and, with a lab wired, on into
  experiments. Seven stages, one config that contains no record ids at
  all, and a durable event log: `RUNNING` before each stage's side
  effect, a terminal status after it, so a crash leaves a visible claim
  rather than a gap, and every state a step derives is persisted so the
  snapshot store holds a chain a later process can actually walk. A stage whose work is already on disk is
  recognised, never repeated — from the log if the event is there, from
  the stage's own store if a crash lost it. Nothing retries; a refusal
  or a failure stops the walk, and `arl resume` re-attempts that stage
  and only that stage. Proven twice, deterministically: the preserved
  Task 5B.1–5F records replay under one root, reaching the same records
  five hand-bridged drivers reached, spending nothing and funding once;
  and a synthetic brief walks all seven stages in half a second,
  executing real experiments through the ordinary executor. Walked again
  in seven pieces, stopping and resuming at every boundary, it admits
  the same state, funds the same run once, and still verifies.

- **A production vision lab.** The seventh stage's first real
  instrument: `--lab examples.vision_lab:lab` supplies a trusted
  template catalog for CIFAR-scale representation learning, and a
  funded run trains for real. The lab's capability is a closed table of
  contrasts its templates compute; every admitted prediction is parsed
  against it, and what cannot be measured is refused, typed, exit 2,
  before anything spends — proven against the preserved Task 5F
  admission, whose attention-head observables no vision template
  serves. What can be measured is served by trusted substitution: the
  admitted metric string is written into the template source by catalog
  code, so the exact-match contract holds by construction. Templates
  are fixed programs — seeding, data, splits, probe, control, and every
  byte of `metrics.json` are fenced trusted code a preflight holds
  byte-identical — with one slot, the encoder architecture, for the
  engineer's model. Execution is backend-agnostic by deployment data
  (host CPU/MPS/CUDA or container), datasets are operator-staged bytes
  under content-addressed manifests, and governance stays on. The CI
  walk carries a synthetic vision brief through all seven stages on a
  stdlib stub trainer in seconds: three seeded runs, all
  `VERIFIED_EVIDENCE`, consistent sign tests, `SUPPORTED` assessments,
  and — since Task 7B — a planner ending: once the deterministic
  follow-through is done and verified findings exist, the model-backed
  planner shares the director's seat, pre-registers a sharper
  effect-size claim under the deterministic gate, runs it through the
  same trusted template at a fresh seed, and stops the investigation
  with a typed reason, citing its evidence. And since Task 7C the
  verdicts are exact inference, not sign checks: a deterministic
  statistician runs a one-sided exact sign test over each claim's
  replication family with Bonferroni-adjusted thresholds, and every
  figure — n, mean, spread, effect, the exact p — lives in the
  assessment's own rationale. Unanimous-but-underpowered is honestly
  `PLAUSIBLE`, never inflated to `SUPPORTED`.

Nothing a model says becomes scientific state until deterministic code
has gated, committed, executed, and verified it. Literature-derived
records describe, conjecture, and prefer, carrying their sources; the
one path into scientific state is admission's governed door, which
turns exactly one validated selection into a bare initial state of
propositions — no result, evidence, or judgment is expressible there.
Funding that seed is a separate operator act with its own record and
its own ledger: it buys the chance to find something out, never a
finding. Money is held before an attempt spends it and settled
afterwards, so a process killed mid-step leaves a visible claim on the
budget rather than a silence, and what it actually cost is recorded in
full — past its authorization and past the balance if that is what
happened. Every job runs inside such an attempt, repair reruns included:
the hold, the phase and the job's own id reach disk before the job
exists, so work nobody wrote down first cannot be started. Orchestration is not model-driven: a fixed-priority director
dispatches work, and every model decision passes a deterministic gate
before it takes effect.

Not built yet: access resolution for metadata-only retrieved works,
checkpoint-resume of a half-trained job (7A.1), confidence intervals
and power analysis behind the statistician, re-assessment of claims as
new evidence lands, literature-grounded findings entering research
state, cloud execution, a skeptic role, and paper writing.

Expect interfaces to change.

## How the code is organized

| Package | What it does |
| --- | --- |
| `core` | The basic data types: questions, hypotheses, predictions, experiments, results, evidence, claims, assessments, budgets, and the research state that ties them together. Depends on nothing else. |
| `evidence` | Storage for results and evidence. Records can be added but never changed. Two backends behind one contract: in memory for tests and ablations, and file-backed for a real run — each record under its own payload digest, with the artifact bytes kept in a content-addressed blob store. |
| `execution` | Runs an experiment as a subprocess. Each job gets its own working directory, its own home directory, and a minimal environment. Output files are collected and hashed. |
| `knowledge` | Read-only views over the data, such as the graph linking claims to evidence. |
| `literature` | Bounded search against a real scholarly API (OpenAlex): normalized snapshot records, deterministic deduplication, write-once search provenance, and a replayable local corpus. Depends only on `core`; its one consumer is `mapping`. |
| `mapping` | What the literature adds up to: model-backed field maps and problem inventories, deterministically gated and write-once, judged by a trusted-code adequacy verdict. Depends on `core`, `literature`, and the provider seam; its one consumer is `ideation`. |
| `ideation` | What might be worth investigating: CFP-directed candidate generation over the assessed map. Depends on `core`, `mapping`, and the provider seam; its one consumer is `priorart`. |
| `priorart` | Whether it was already done: the prior-art challenge over the candidate portfolio, with a deterministic fail-closed verdict per candidate. Depends on `core`, `literature`, `mapping`, `ideation`, and the provider seam; its consumers are `selection` and `admission`. |
| `selection` | Which candidate to pursue, if any: gated two-stage selection over the `DISTINGUISHED` survivors of one named prior-art run, with attested disqualifiers and three honest outcomes. Score-free and write-once. Depends on `core`, `ideation`, `mapping`, `priorart`, and the provider seam; its one consumer is `admission`. |
| `admission` | The governed bridge into research state: one named `SELECTED` run verified through its whole lineage, one gated model call encoding the recorded predictions sign-only, deterministic copies for everything else, and an all-or-nothing state snapshot beside a write-once record. Depends on `core` (uniquely including the state it constructs), `ideation`, `mapping`, `priorart`, `selection`, `persistence`, and the provider seam; nothing imports it. |
| `program` | A funded run: the bridge from one admitted state to something the runtime may spend against. One named admission, one authorized grant, a funded successor state, an append-only budget ledger that holds money before an attempt spends it and settles it afterwards, an attempt journal recording how far each attempt got in making itself durable, and the cold verification of a whole run root. Depends on `core`, `admission`, `persistence`, and `evidence`; nothing imports it. |
| `control` | The composition root: one command that walks the seven stages of the chain from a config with no ids in it, recording what happened to each in an append-only event log so an interrupted run resumes without repeating or double-paying for anything — and, since Task 6D, finishing a step a killed process left half done rather than abandoning it — every job it runs, repair reruns included, held and journalled before it is submitted. The one package that may import every stage, and the one nothing imports. |
| `persistence` | Saves every state to disk so a run can be inspected or replayed later. |
| `runtime` | Bookkeeping around the loop: open work, validation and verification, cost tracking, metrics, the model-provider seam (with the Muse adapter), and the write-once stores for implementation and planning provenance. |
| `search` | Policies for choosing the next action among candidates. |
| `roles` | Interfaces for the agents, including the model-backed engineer and planner. A role receives a task and returns proposals. It never edits state directly. |
| `orchestration` | The main loop: pick an action, route it to a role, validate what comes back, commit it, log everything. |
| `publication` | The evidence packet — flat, checked-not-copied mirrors of everything a manuscript may claim — the manuscript (one gated model call writes five prose sections behind deterministic number/citation/structure gates; trusted code assembles everything else), and the faithfulness reviewer (findings grounded in verbatim quotes and record ids or refused; the verdict derived by trusted code; one bounded revise cycle recorded as its own succession fact). Depends on `core`, `evidence`, and `runtime`; its one consumer is `control` (enforced by the layering tests). |

Dependencies point one way: higher packages import lower ones, never
the reverse, and `core` imports nothing. `control` is the one exception
and the reason the rule survives: a composition root may import every
stage precisely because nothing imports it. Tests check all of it.

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

## The command line

```bash
pip install -e .
arl run CONFIG --root DIR [--lab module:factory] [--stop-after STAGE]
arl resume [INVESTIGATION] --root DIR
arl status [INVESTIGATION] --root DIR
arl verify --root DIR
arl packet [INVESTIGATION] --root DIR [--out DIR]
arl manuscript [INVESTIGATION] --root DIR [--lab module:factory] [--model NAME] [--out DIR]
arl review [INVESTIGATION] --root DIR [--lab module:factory] [--model NAME] [--out DIR] [--review-only]
```

`run` records the config and walks as far as it can. `resume` picks up
where a walk stopped, using the config the investigation recorded rather
than whatever the file says now. `status` prints the stage table.
`verify` re-checks every durable claim under the root. `packet` exports
the evidence packet: it verifies the run from cold, re-derives the
statistician's figures against the record, and writes
`packet/<packet_id>.json` and `.md` under the root — refused for a walk
that never reached a research state. `manuscript` authors the workshop
draft from that packet: a model writes prose only, behind deterministic
gates that refuse any number the packet does not state and any citation
outside its bibliography; trusted code assembles everything else.
Re-running replays the recorded draft without a model call. `review`
runs the faithfulness reviewer over that draft: trusted code and one
gated model call judge whether the prose claims anything the packet
does not record — every finding grounded in a verbatim quote and a
record id, or refused. A REVISE verdict triggers at most one
revision-and-re-review cycle; a standing REVISE exits 1 with the
findings printed.

Exit codes are for scripts as much as for people: `0` for a walk that
ended on its own terms — including an honest scientific no, which is a
result and not a fault — `2` for a refusal, `1` for a failure, an
unusable config, or a verification that found something.

Without `--lab`, the chain runs on Muse and OpenAlex from the
environment and stops at the funded run: roles, an executor, and a
trusted template catalog are code, and a lab module supplies them.

## Examples

### examples/canary_chain.py

```bash
python -m examples.canary_chain --run-root /tmp/canary
python -m examples.canary_chain --run-root /tmp/canary --stop-after selection
python -m examples.canary_chain --run-root /tmp/canary   # continues
```

One synthetic brief through all seven stages, with no network, no clock,
and no model — about half a second, and the same answer every time. The
instruments are fixtures; the machinery is not, so the experiments run
through the ordinary executor in real subprocesses and the ledger bills
one debit per attempt. The second and third commands are the point: the
walk stops where it was told, and a later process picks it up from the
durable record with no memory of the first.

### examples/vision_chain.py

```bash
pip install -e ".[dev,vision]"                                # host backends only
python -m examples.vision_lab.stage_cifar10 --datasets-root ~/arl-data
export ARL_VISION_PROFILE=profile.json                        # see docs/EXECUTION.md
python -m examples.vision_chain --run-root /tmp/vision        # real training
python -m examples.vision_chain --run-root /tmp/vision --ci   # stdlib stub, no setup
```

The vision brief through all seven stages, training for real. The
analysis stages run on scripted instruments — zero network, zero model
spend — and the seventh executes genuine representation learning
through whatever backend the deployment profile names. `--ci` swaps the
trainer for a stdlib stub: the exact walk the test suite runs, no
torch, no docker, no dataset.

### examples/vision_refusal.py

```bash
python -m examples.vision_refusal
```

The honest no, on a real record: funds the preserved Task 5F admission
in a scratch root and asks the vision lab for a runtime. The lab
refuses — typed, before anything runs or spends — naming both admitted
attention-head metrics it cannot measure.

### examples/torn_step.py

```bash
python -m examples.torn_step --run-root /tmp/torn --kill-after      # list
python -m examples.torn_step --run-root /tmp/torn --kill-after 10   # die
python -m examples.torn_step --run-root /tmp/torn                   # finish it
python -m examples.torn_step --run-root /tmp/t2 --repair --kill-after 18
```

Two processes, nothing shared but files. The first walks the canary to a
funded run, starts one step, and is killed outright — `os._exit`, no
unwinding, no cleanup — the instant its tenth durable write lands. The
second knows nothing about the first except what is on disk: it reads
the attempt journal, answers whatever was left open, and prints the
ledger, the journal and the verifier side by side. A step makes sixteen
durable writes, and dying after any of them ends the same way — the run
owes nothing, each attempt is charged exactly once, and it verifies from
cold.

`--repair` tears a longer step instead: the one that runs a job, has it
fail, and repairs it inside the same step — fifty writes, and the four
in the middle belong to the rerun's own attempt. Killed at the
eighteenth, the second process finds *two* open attempts rather than
one: the step's, finished from its durable bundle, and the rerun's,
charged its authorization and closed with nothing to show. Before the
rerun had an attempt, the second of those was not on the record at all.

### examples/live_task6c.py

```bash
python -m examples.live_task6c --run-root live_runs/task6c-<date>
```

Assembles the preserved Task 5B.1–5F records under one root and walks
them, with a provider that raises on every call. Every completed stage
is recognised from its own store, funding runs once, the preserved files
are byte-identical afterwards, and the walk arrives at the records five
hand-bridged drivers reached. Needs the preserved `live_runs/` roots.

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

### examples/verify_run.py

```bash
python examples/verify_run.py --root <run_root>
```

Checks a run root written by an earlier process: state snapshots
re-hash to their filenames, the snapshot lineage is whole (every parent
stored, every chain ending at a root, no cycles, and a forward walk from
the roots reaching all of it), result and evidence payloads survive
their digests, every state reference resolves, every stored artifact
still hashes to what its manifest says, the evidence chain holds, and a
funded run's budget ledger replays to the balance its own head carries.
Prints every problem it finds, not just the first, and exits non-zero if
there is one.

A verified run is one whose records survived. Whether its science is
right is a different question.

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
