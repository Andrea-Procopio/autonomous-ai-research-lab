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
- **Persistence.** Partly done. `ResearchState` snapshots persist to
  disk content-addressed (`persistence/`), and trajectory records
  reference them, so a run is auditable and reconstructible offline.
  Still open: a durable evidence store behind the existing
  `EvidenceStore` protocol, and resume-from-snapshot orchestration.
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
    on metadata-only works, a budgeted planner and experiment
    execution over the admitted state, and findings entering the state
    as structured objects carrying their sources through the governed
    commit.
- **Real experiment execution.** ML training runs under the existing
  executor contract: checkpoints, longer timeouts, GPU accounting.
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
- **Statistician and Verifier.** Real inference behind
  `EpistemicAssessment`. Today's assessments are demo-grade prediction
  checks; these roles bring power, uncertainty, and
  multiple-comparison awareness, plus provenance checking of claims.
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
