# Roadmap

Three horizons, then the near-term plan.

The horizons are about capability, not schedule. Move to the next one
only when the previous one produces trajectories good enough to justify
it. Scaling an unreliable research loop just produces unreliable
research faster.

---

## Horizon 1 — Research engine

**Goal: a reliable single-project autonomous research loop.**

One research program, one direction at a time, run end to end without a
human in the loop, producing results that survive scrutiny.

- **Model provider boundary.** A thin interface for structured model
  calls, with token accounting wired into `ResearchBudget`. Validation
  belongs at the boundary, not in the domain core.
- **First concrete roles.** Hypothesis Generator, Experiment Designer,
  Research Engineer, Result Analyst. Each with an explicit utility, not
  a shared reward.
- **The Skeptic.** A role whose job is finding flaws, confounds, and
  alternative explanations in the system's own work. Built early: a
  loop without one drifts toward self-confirmation, and retrofitting
  adversarial review is much harder than building around it.
- **Persistence.** Done for facts (Task 6B, 2026-08-20).
  `ResearchState` snapshots persist to disk content-addressed
  (`persistence/`), and trajectory records reference them. The facts
  they cite now persist beside them: `FileEvidenceStore` writes results
  and evidence under their own payload digests, and a content-addressed
  blob store keeps the artifact bytes, so the executor's run directory
  is expendable once a result is recorded. Recording a fact stores its
  bytes first, which is what makes "a state never references a fact
  that is not durable" true by construction rather than by convention.
  `verify_run` checks a whole run root from cold — snapshots, payloads,
  references, artifact bytes, the evidence chain, and the budget
  ledger — and reports every problem rather than raising on the first.
  Still open: resume-from-snapshot orchestration, and reachability
  rules for anything that would ever delete a stored fact.
- **Literature access.** Substantially done, proven live task by task:
  - *Task 5A (2026-08-18).* Bounded search against OpenAlex behind a
    provider-neutral seam. Normalized snapshot records with preserved
    access levels, deterministic deduplication, write-once search
    provenance, and a local corpus that replays identical completed
    searches with zero network calls.
  - *Task 5B (2026-08-18).* Evidence-grounded field mapping over that
    corpus: model-proposed queries executed by trusted code, preserved
    relevance screening, verbatim-grounded extraction under
    deterministic gates, and a source-grounded `FieldMap` plus
    `ProblemInventory` with honest coverage accounting.
  - *Task 5B.1 (2026-08-18).* The corrective pass the 5B evidence
    demanded. Title/abstract query matching (plain search proved to be
    fulltext matching — the root cause of the 5/60 relevance yield),
    citation-ranked foundational retrieval beside recency-ranked
    recent retrieval, bounded query refinement, and a durable
    `MapAdequacyAssessment` with typed reasons and per-problem support
    tiers, guarding Task 5C behind
    `require_adequate_for_idea_generation`.
  - *Task 5C (2026-08-19).* CFP-directed candidate generation through
    that guard. An immutable hashed snapshot of a real workshop call,
    a gated verbatim direction extraction, one gated portfolio call
    producing fully structured candidates, trusted-code stamping of
    problem statements, kinds, support tiers, and source-era mixes,
    structurally `UNASSESSED` novelty, honest refusal as a recorded
    outcome, and a portfolio report that names every unaddressed
    problem.
  - *Task 5D (2026-08-19).* The prior-art challenge over that
    portfolio: six trusted-dated query families per candidate over a
    fresh corpus, cited-source injection with identifier dedup, gated
    screening and five-dimension comparisons held verbatim to
    accessible text, and a deterministic fail-closed
    `PriorArtAssessment` per candidate that never modifies the
    candidate records. The live run returned three honest
    `NOVELTY_UNRESOLVED` verdicts on thin pools: the model's
    ten-plus-term conjunctive queries left sixteen of eighteen
    searches empty.
  - *Task 5D.1 (2026-08-19).* The retrieval fix. Verified OpenAlex
    Boolean semantics, a structured concept-group plan schema in which
    an opaque query string cannot be expressed, the trusted
    `boolean-v1` renderer with conjunctivity bounds, and the
    source-attested-phrase rule (a described source's own words are
    quotations, not claims). Pools rose from 2-4 to 9/23/17 unique
    sources. All three candidates stayed `NOVELTY_UNRESOLVED`, now on
    metadata-only ambiguity, one source-short pool, and one screening
    truncation.
  - *Task 5D.2 (2026-08-19).* The calibration pass. A blocker audit
    separated necessary conditions from proxies: every 5D.1 metadata
    blocker was an undecidable title-only screen that restated access
    level, and the truncation came from the directive's own
    retrieval/screening mismatch. Changes: material metadata ambiguity
    now needs an attested `OverlapHypothesis`, checked in its own
    gated screening call; verdict rules measure the screenable pool
    and abstract-level screens, with threshold values unchanged; a
    budget preflight refuses an inconsistent directive before any
    call; a calibration suite proves all three verdicts reachable at
    the default thresholds; and a read-only counterfactual replay of
    the preserved 5D.1 records (one verdict of three changes) ran
    before the live rerun. The live rerun returned three
    `DISTINGUISHED` verdicts: pools of 11/22/20, twelve grounded
    comparisons, zero truncation, and spend that reconciles exactly
    with the ledger. `DISTINGUISHED` still means only "differentiated
    from the closest works this bounded search found", never novelty.
  - *Task 5E (2026-08-20).* Candidate selection over the challenged
    portfolio. One door: the directive names one prior-art run record,
    and trusted code computes eligibility from that run's
    `DISTINGUISHED` verdicts alone. Two gated model stages under
    score-free schemas with no stop shape; narrow attested
    disqualifiers quoting the candidate and the operator's stated
    constraints verbatim; three structurally distinct outcomes, all
    proven reachable on closed portfolios. The live run's first
    attempt failed closed on a Muse HTTP 504 (a typed transport
    error; nothing but the directive recorded); the rerun selected one
    candidate over two undisqualified alternatives in two calls with
    zero corrective calls, spend reconciling exactly with the ledger,
    and every preserved upstream artifact byte-identical. A selection
    is a model preference validated — never computed — by trusted
    code, and never proof of novelty.
  - *Task 5F (2026-08-20).* Governed admission of the selected
    candidate. One door: the directive names one selection run record,
    and trusted code re-verifies the complete lineage — selection,
    prior art, ideation, direction, CFP snapshot — cross-checking the
    records against each other before requiring `SELECTED`. One gated
    model call encodes the candidate's recorded predictions under the
    sign-only neutral encoding (comparator and threshold are
    structural constants; the model never authors a number); the
    question, hypothesis, objective, and measurement surface are
    deterministic verbatim copies; execution requirements are quoted
    by provenance, inherited versus operator-stated. The initial state
    is built in one constructor call, persisted content-addressed with
    a read-back check, and recorded all-or-nothing beside a write-once
    admission record — one admission per selection run, ever, with a
    completed directive replaying at zero model calls. The live run
    admitted the 5E winner in one call with zero corrective calls,
    1,415 input and 3,533 output tokens reconciling exactly with the
    ledger, both observables of the single recorded prediction encoded
    as distinct machine-checkable predictions, all 184 preserved
    upstream files byte-identical, and the replay served through a
    provider that refuses every call. An admission is a translation,
    never a promotion: `ADMITTED` does not mean true, novel, or
    empirically supported.
  - *Still open:* access resolution for attested material ambiguities
    on metadata-only works, experiment execution over the funded
    state, and findings entering the state as structured objects
    carrying their sources through the governed commit.
- **Funded runs.** Done (Task 6A, 2026-08-20). An admitted state
  carries no budget, and a state's identity excludes its budget, so
  funding one in place produces different bytes under an unchanged id
  and the append-only snapshot store refuses the write. Funding is
  succession instead: the funded state is a child of the admitted one,
  which keeps its zero budget forever. A run directive names one
  admission and one authorized grant; the run is an event with an
  occurrence id, and every record about it is content-addressed and
  tamper-loud. Spend lives on an append-only ledger — sequence-numbered,
  hash-chained, write-once entries, idempotent by charge id, with the
  exclusive file create as the concurrency primitive — so a balance
  survives a restart, a repeated charge cannot debit twice, and two
  concurrent debits cannot overspend. The runtime posts one debit per
  attempt behind a protocol seam and fails closed when the ledger and
  the state disagree. Proven over the preserved Task 5F admission with
  zero model calls and zero network calls.
- **One command through the chain.** Done (Task 6C, 2026-08-20). A
  controller walks the seven stages from one config that contains no
  record ids, and `arl run/resume/status/verify` is the front door. Each
  stage's identity is its directive's content id, so "has this already
  been done?" is answerable in a process that did none of it: from the
  event log if the event is there, from the stage's own store if a crash
  lost it. `RUNNING` is written before every side effect and a terminal
  status after it, so a crash is a visible claim rather than a gap.
  Nothing retries. An honest scientific no ends the investigation as a
  success and marks the stages that will never run. Proven twice without
  a model or a network: the preserved 5B.1–5F records replay under one
  root and fund once with every file byte-identical, and a synthetic
  brief walks all seven stages — executing real experiments through the
  ordinary executor — then walks again in seven interrupted pieces to
  the same admitted state. Building it exposed two defects, both fixed:
  the verifier passed every run that had spent anything, and the runtime
  persisted only each step's head, leaving committed snapshots pointing
  at parents nobody had written down. A run's whole lineage is stored
  now, and `arl verify` checks that it is whole. Still open: a trusted
  template catalog whose metrics match admitted predictions, without
  which a live run stops at the funded run.
- **Recoverable attempts and budget reservations.** Done (Task 6D,
  2026-08-20). Task 6C made a run resumable between stages; inside a
  step it was not, and a process killed after paying for an experiment
  and before recording what it bought left the ledger and the snapshots
  disagreeing with nothing to decide between them. Now money is *held*
  before an attempt runs and settled afterwards, every attempt writes
  down how far it got, and the whole effect of a step is stored before
  it is applied. Recovery reads that record instead of guessing: a
  durable bundle is applied and the attempt finishes with nothing
  re-run, and anything earlier is charged its authorization in full and
  closed with nothing to show — the conservative direction, because
  releasing money that may be gone is the failure the record exists to
  prevent. The state and the ledger are reconciled against each other
  before the run goes on. Overruns are no longer clamped: the real
  figure is charged, the balance may go negative, and the run stops. A
  step makes sixteen durable writes and the suite kills it after every
  one of them, plus four cross-process kills with nothing shared but
  files. A conservative charge is recorded as one: the closing event
  carries `CONSERVATIVE_MAX` rather than claiming a measurement nobody
  took, which also keeps a crash from reading as a budget breach.
- **Every rerun is an attempt.** Done (Task 6D.1, 2026-08-21). Task 6D
  swept the canary's *design* step, which runs no job, so it never saw
  the one job left outside all of this: the bounded repair loops
  submitted their own reruns, and the attempt that answered for a rerun
  was opened only once the result came back. In between, a job ran with
  no reservation covering it and nothing recording that it existed — a
  crash there charged the spend to nothing and orphaned the outputs,
  while the parent attempt recovered from its own bundle and the run
  came back looking intact. The debugger takes a `JobRunner` now,
  proposing and rerunning are separate calls, and the runtime opens the
  attempt — snapshot, `STARTED`, reservation — before the prepared job
  is handed over, so every rerun carries its own authorization, its own
  phases and its own derived job id, on disk before the job exists. The
  proof is the sweep the old one was missing: a step that executes,
  fails and repairs itself makes fifty durable writes, and the suite
  kills it after every one of them, plus four cross-process kills over
  the rerun's own writes. Running it exposed two defects that had
  nothing to do with repair and everything to do with jobs, both fixed:
  a commit bundle was written before the facts it names were durable, so
  recovery raised rather than finishing the step, and the verifier
  faulted a `SUBMITTED` note whose job never ran — which is the note
  working, not a broken link. The one residue — recovery *answered* an
  interrupted rerun but did not *collect* it — is closed by Task 7A's
  salvage arm below.
- **Collect-finished recovery and GPU accounting.** Done (Task 7A,
  2026-08-21, first slice). A job now declares how many GPUs it
  occupies and the executor bills `gpu_hours` as wall clock times that
  occupancy — a measurement of occupancy, never of utilization, and the
  record says which. The durable job record carries what finishing the
  books needs — start time, pid, timeout, required artifacts, the
  environment captured once before launch — so a cold process can reap
  an orphan whose submitter provably died, deciding success from the
  contract's own evidence. And recovery collects: an attempt whose job
  finished before the crash is completed with the job's measured cost
  instead of conservatively abandoned. The soundness predicate is all
  journal: the submitted job id must equal the one `STARTED`
  pre-registered, the executor's record must be terminal for exactly
  that job, and any `OUTPUTS_DURABLE` event must agree about which
  result it produced — anything unprovable keeps the conservative
  answer. Salvage rebuilds precisely what the live step would have
  built (one result proposal, the deterministic gate, a bundle costing
  what the result cost), so a gate-refused result commits the same
  failed bundle live execution would have. The sweep grew to cover it:
  the simulated crash is a `BaseException` now, because a crash the
  runtime could catch as a role failure was quietly testing the wrong
  thing, and the repairing step's 52 durable writes each get a kill —
  including the two positions where a job finished and nothing had
  recorded it yet.
- **Real experiment execution.** Done (Task 7A, 2026-08-22). The
  seventh stage's first production instrument: `examples/vision_lab`, a
  lab for CIFAR-scale representation learning behind every contract the
  canary proved. Its capability is a closed table of contrasts its
  templates genuinely compute; every admitted prediction is parsed
  against it, and what cannot be measured refuses — typed, exit 2,
  before anything spends — proven against the real preserved Task 5F
  admission, whose attention-head observables no vision template
  serves. What can be measured is served by trusted substitution: the
  admitted metric string is written into the template source by catalog
  code, so the exact-match contract between admission, the spec, and
  `metrics.json` holds by construction. Templates are fixed programs —
  seeding, data, splits, probe, control, and every byte of the metrics
  file fenced as trusted code a preflight holds byte-identical — with
  one slot, the encoder architecture, for the engineer's model.
  Datasets are operator-staged bytes under content-addressed manifests,
  re-verified by preflight before every launch; execution backends
  (host CPU/MPS/CUDA, container CPU/CUDA) are deployment data resolving
  onto the existing binding and executor seams, with the compute device
  an explicit recorded decision — the first live run caught the
  template auto-picking the Apple GPU under a profile that declared
  none, which would have billed a falsehood. Qualified live on
  2026-08-22: a synthetic vision brief walked all seven stages with
  zero network and zero model spend, executing three real seeded
  CIFAR-10 training runs on CPU (~19 s each; probe contrasts +0.040 to
  +0.060, trained ~49% versus random ~45% top-1, overfit control 1.00),
  every result `VERIFIED_EVIDENCE` under default governance, three
  CONSISTENT sign tests, a `SUPPORTED` assessment, an honest stop, and
  a clean cold verify — then again stopped at funding and resumed, and
  again killed outright mid-training: the orphaned trainer finished in
  its own session, the resuming process reaped it from the durable job
  record, salvage rebuilt and committed the very bundle the dead
  process would have (journal: `collected after an interrupted step`,
  `succeeded (rebuilt from the collected job)`, MEASURED 19.8 s), and
  the run ended with three jobs for three seeds — the killed training
  run kept as science, nothing re-run, zero conservative charges.
  Salvage exposed one gate defect the sweep could not: a reaped orphan
  honestly records `exit_code: None`, and the validation gate read the
  missing exit as failure; it now accepts completion by the contract's
  own evidence, with the orphan shape stated in the check's detail.
  Still open, deliberately: the container backends are documented and
  policy-tested but this machine qualified the host path — the first
  Linux/CUDA deployment should re-run the same qualification; the live
  `ModelBackedPlanner` seat is Task 7B (the catalog already fits its
  gate, tested); and checkpoint-resume of a half-trained job is 7A.1,
  designed against a real workload now that one exists.
- **The live planner seat.** Done (Task 7B, 2026-08-22). The hand-off
  7A promised: once the deterministic follow-through is done and
  verified findings exist, the model-backed planner shares the
  director's seat — it may pre-register new science under
  `check_decision`, and it may end the investigation with a typed
  reason. The composite director routes by what the pure rule-based
  director would do: structural and analytical work is returned as-is;
  execution work is delegated to the planning director exactly when a
  planning record owns it (so dispatch bookkeeping lands on planner
  work and stays off bootstrap work, and a bare replication gap never
  becomes an unintended billed consultation); a rule-based stop becomes
  a consultation when findings exist and no earlier consultation ended
  in terminal rejection — a guard read from the planning store, because
  the frontier's failed-attempt view goes blind to a failed
  consultation once any other has succeeded. The planning director is
  called at most once per step and never speculatively. The engineer's
  fixed-region judgment moved into the generation-repair loop as an
  injected `CompletionReview`, so a completion that edits trusted
  measurement code earns one corrective call naming exactly what it
  must not touch; the preflight check remains as backstop. Qualified
  live on 2026-08-22 with real training: after the three bootstrap runs
  (contrasts +0.040/+0.044/+0.060), the scripted planner pre-registered
  an effect-size floor of 0.01 on the admitted contrast and ran it
  through the same trusted template at a fresh seed — observed +0.069,
  CONSISTENT — then stopped with `question_resolved`, citing its
  evidence; four verified results, two SUPPORTED assessments, both
  planning decisions dispatched, zero rejected consultations, and a
  clean cold verify. Building it caught the one reference rule the 7A
  gate-fit test had not covered: a stop decision must still name its
  question. Still open, deliberately: the live Muse-backed planner is
  wired (`lab()`) but needs an operator key; folding the composite's
  structural-work fallback into `PlanningDirector` itself is a later
  src task; richer planner storylines (ablations, replications) await a
  campaign that needs them.
- **Scientific debugging and experiment verification.** Done in its
  Phase 1 form: the five-way failure taxonomy (engineering /
  implementation / methodological / analytical / verified), a
  deterministic failure classifier, bounded execution and
  implementation repair (the latter entered only on typed
  implementation-invalidity evidence), preflight, positive controls,
  selective implementation verification, the methodology gate, analysis
  coverage, the negative-result gate, durable per-result verification
  records, and the scientific-promotion gate that keeps unverified
  observation from becoming trusted support. All flag-gated and
  removable (see ARCHITECTURE). Verification governance fails closed by
  default: a missing record blocks trusted promotion, and the
  ungoverned lab exists only as an explicit config ablation. Still
  open: model-backed repair strategies and reviewers behind the
  existing protocols, richer control libraries, and defaulting the
  runtime to file-backed verification persistence.
- **Statistician and Verifier.** The deterministic statistician is
  done (Task 7C, 2026-08-22). Trusted code runs a one-sided exact sign
  test over the claim's replication family — the raw per-seed
  observations the assessor context now carries, consuming the
  replication-group seam the architecture reserved — with Bonferroni
  adjustment across the hypothesis's tested predictions and every
  figure stated in the assessment's own content-addressed rationale.
  SUPPORTED must clear the adjusted exact tail, so the bootstrap
  declares five seeds (three unanimous observations carry a tail of
  one in eight, and a verdict that cannot clear its own threshold is
  not a verdict); unanimous-but-underpowered is PLAUSIBLE — the
  verdict's first production use — and consistently-contrary-but-
  underpowered stays UNDETERMINED. The assessor context enrichment
  also dissolved a real gate trap: coverage demands the admissible
  conclusive family cited in full while promotion refuses any
  inadmissible citation, and the context now labels the one set that
  satisfies both. Wiring it exposed a second trap the stub lab could
  never show: with a real trainer's cost estimate, synthesis outbids
  replication at equal value, the claim is judged at n=1, and the
  planner gate never opens again — fixed by wiring the empirical-ML
  playbook (replication-gap advice boosts REPLICATE to HIGH at any
  cost estimate; both sides of the tie-break are regression-tested).
  Qualified live on real training, 2026-08-22: five bootstrap
  contrasts (+0.023 to +0.060, mean +0.0419, stdev 0.013) reached
  SUPPORTED at exact p = 0.03125; the planner's fresh-seed claim
  (+0.069 against its 0.01 floor) recorded PLAUSIBLE at n=1 with the
  0.025 Bonferroni-adjusted threshold stated; clean cold verify.
  Still open: confidence intervals and power analysis (a t-quantile
  policy), re-assessment when new conclusive evidence postdates the
  current judgment (frontier re-opening plus `supersedes` wiring), the
  model-backed statistician and reviewer behind the same seat, and
  provenance checking of claims (the Verifier).
- **Publication evidence packet.** Done (Task 8A, 2026-08-22). The
  `publication` package holds its first content — a deterministic
  export, checked rather than copied: `arl packet` verifies the run
  from cold, walks the citation chain back to the cited literature
  (every store re-derives its own content ids on load, so a doctored
  record refuses to load and the export fails naming it), re-derives
  the statistician's figures from the immutable record with the m and
  alpha the assessment pinned, and refuses to write anything a
  manuscript could cite that the record does not support. The packet
  is flat mirrors only — strings and numbers, each element carrying
  the record and artifact ids it points back at — written write-once
  as `packet/<packet_id>.json` and `.md`, named by a content id, so
  re-exporting an unchanged run is a byte-identical no-op. All claims
  in the head state are included; an unassessed claim is marked, never
  dropped. A walk that stopped before funding has no research state
  and gets a typed refusal, not a file. Qualified live on the real
  CIFAR-10 root, 2026-08-22: both claims exported
  re-derived-and-matched (SUPPORTED, n=5, p=0.03125; PLAUSIBLE, n=1,
  adjusted threshold 0.025), two cited sources resolved, six result
  rows each naming its verified result. Still open: rendered figures
  (the packet currently states their absence), manuscript generation
  and the reviewer role (8B/8C), and packet signing.
- **Manuscript generation.** Done (Task 8B, 2026-08-22). The first
  model-authored document, and the model may only phrase: one gated
  call writes five prose sections, and deterministic gates judge the
  tokens before any prose is kept. A word containing a digit passes
  only if the packet's own renderings already contain that exact word
  — so rounded values, unit conversions, recomputed percentages, and
  obfuscations like a "3x" speedup are unknown numbers by
  construction. Every bracketed citation must name a bibliography
  entry; no prose line may open a heading. A rejected draft is
  preserved as evidence and earns one corrective call carrying exactly
  the rules that fired. Trusted code assembles the results and
  references sections from the packet's own renderers byte for byte;
  the call never charges the settled run's grant, and re-running
  replays the recorded draft with zero model calls. Qualified live on
  the real CIFAR-10 root, 2026-08-22: muse-spark-1.2 authored the
  draft in one call with zero correctives — a numberless abstract,
  verdicts stated only as "supported" and "plausible", both cited
  sources resolved — and the root re-verified intact afterwards. Still
  open: the reviewer role that checks a manuscript against the
  claim-evidence graph (8C), rendered figures, LaTeX/submission
  formats, a semantic honest-language gate (deliberately the
  reviewer's job, not the author's), and manuscript signing.
- **Trajectory analysis.** The decision log (`DecisionRecord` + JSONL)
  exists. Horizon 1 adds the analysis code that reads it: utility
  calibration, cost prediction error, failure-mode tallies, feeding
  the ICLR plan below.

**Done when:** the system runs a real ML research question end to end,
and the resulting claims survive independent checking — including the
cases where the honest answer is that the hypothesis was wrong.

---

## Horizon 2 — Research organization

**Goal: multiple specialized agents and multiple concurrent research
projects with resource allocation.**

- **Full role set**, including Methodologist, Evidence Verifier, and
  Scientific Reviewer, with utilities that genuinely differ.
- **Asynchronous execution at scale.** Cloud and SLURM backends behind
  the existing `Executor` interface; many concurrent jobs.
- **Real search policies.** Beam search, bandits, adaptive branching
  over research states. This needs calibrated value estimates, which
  need Horizon 1 trajectories — which is why it is not attempted
  earlier.
- **Non-fungible budgets.** Replacing the greedy policy's single
  scalar cost with a defensible exchange rate between wall-clock,
  GPU-hours, money, and tokens.
- **Portfolio allocation.** Deciding which of several concurrent
  directions deserves more resources, based on expected scientific
  value rather than activity or apparent progress.
- **Replication as routine.** Automatic replication of load-bearing
  results, and detection of contradictions between a program's own
  experiments.

**Done when:** several research programs run concurrently under one
budget, and the allocator's choices beat uniform splitting by a
measurable margin.

---

## Horizon 3 — Autonomous research lab

**Goal: persistent institutional memory, portfolio-level research
strategy, autonomous compute allocation, continuous scientific
production, and self-improvement.**

- **Institutional memory.** Cross-project knowledge in `knowledge/`:
  which methods worked, which failure modes recur, which questions are
  already answered and by what evidence. Designed against real
  trajectories, not guessed in advance.
- **Research strategy at the portfolio level.** Choosing which
  questions are worth asking at all, and when to abandon a direction.
- **Autonomous compute allocation** against expected information gain.
- **Continuous production.** Long-running programs that accumulate
  knowledge rather than terminating at a paper.
- **Self-improvement.** The lab improving its own methodology from its
  own trajectory data — the point at which the architecture becomes
  the object of study rather than only its instrument.

---

## Near-term research plan

**Next 1–2 days — architecture and orchestration.**
Close out Horizon 1's foundation: the provider boundary, the first
concrete roles, persistence, and trajectory instrumentation. Keep the
contracts stable enough that roles slot in without reshaping the core.

**Then — run autonomous ML research projects.**
Real questions on real data, end to end. Expect the first runs to
expose which parts of this architecture were wrong. That is what they
are for.

**Submit strong generated research to suitable non-archival NeurIPS
workshops.** Non-archival deliberately: it gets external expert review
without foreclosing later publication of the same work.

**Preserve complete research trajectories and experiment provenance.**
Every action, decision, utility score, result, and dead end, including
the programs that produced nothing. Trajectories that only cover
successes cannot support any claim about how well the architecture
works.

**Use those trajectories to evaluate the architecture scientifically.**
This is what makes the project research rather than engineering. Open
questions the trajectory data should answer:

- Does separating orchestration from execution measurably improve
  reliability, or is it only cleaner to read?
- Do role-specific utilities produce better research than a single
  shared objective?
- Does an adversarial Skeptic reduce unsupported claims, and by how
  much?
- Which search policies find real findings per unit compute, and where
  does greedy actually fail?
- How often does the system reach a correct negative conclusion, and
  how often does it manufacture apparent progress instead? This is the
  measurement the whole design is built around.

**Develop the architecture itself into a rigorous ICLR submission.**
The contribution must be a scientific evaluation of the architecture:
ablations over the design choices above, measured on preserved
trajectories, with honest reporting of what did not work.

**Workshop acceptance counts are not the contribution.** Acceptance is
a noisy external signal with a small sample size. It measures how
convincing the output looks to a reviewer under time pressure, which
is exactly the objective this system is designed not to optimize. It
is one external check among several. It must never become the utility
function, and a paper whose central claim is "n workshops accepted our
output" would measure the wrong thing.
