# Roadmap

Three horizons, then the near-term plan.

The horizons are about *capability*, not schedule. Each one should be reached
only when the previous one produces trajectories good enough to justify it —
scaling an unreliable research loop just produces unreliable research faster.

---

## Horizon 1 — Research engine

**Goal: a reliable single-project autonomous research loop.**

One research program, one direction at a time, run end to end without a human
in the loop, producing results that survive scrutiny.

- **Model provider boundary.** A thin interface for structured model calls, with
  token accounting wired into `ResearchBudget`. Validation belongs here, at the
  boundary — not in the domain core.
- **First concrete roles.** Hypothesis Generator, Experiment Designer, Research
  Engineer, Result Analyst. Each with an explicit utility, not a shared reward.
- **The Skeptic.** A role whose objective is finding flaws, confounds and
  alternative explanations in the system's own work. Early rather than late: a
  loop without one drifts toward self-confirmation, and retrofitting adversarial
  review is much harder than building around it.
- **Persistence.** Partly done: `ResearchState` snapshots persist to disk
  content-addressed (`persistence/`), and trajectory records reference them,
  so a run is auditable and reconstructible offline. Still open: a durable
  evidence store behind the existing `EvidenceStore` protocol, and
  resume-from-snapshot orchestration.
- **Literature access.** Substantially done. Task 5A (proven live
  2026-08-18): bounded search against a real scholarly API (OpenAlex,
  credential-free, behind a provider-neutral seam), normalized snapshot
  records with preserved access levels, conservative deterministic
  deduplication, write-once search provenance, and a local corpus that
  replays identical completed searches with zero network calls. Task 5B
  (proven live 2026-08-18): evidence-grounded field mapping over that
  corpus — model-proposed queries executed by trusted code, preserved
  relevance screening, verbatim-grounded extraction under deterministic
  gates, and a source-grounded `FieldMap` plus `ProblemInventory` with
  honest bounded-coverage accounting. Still open (Task 5C): candidate
  research questions and idea generation reading these records — and only
  then findings entering the state as structured objects carrying their
  sources, through the same governed commit as every other proposal.
- **Real experiment execution.** ML training runs under the existing executor
  contract — checkpoints, longer timeouts, GPU accounting.
- **Scientific debugging and experiment verification.** Done in its Phase 1
  form: the five-way failure taxonomy (engineering / implementation /
  methodological / analytical / verified), a deterministic failure
  classifier, bounded execution *and* implementation repair (the latter
  entered only on typed implementation-invalidity evidence), preflight,
  positive controls, selective implementation verification, the
  methodology gate, analysis coverage, the negative-result gate, durable
  per-result verification records, and the scientific-promotion gate that
  keeps unverified observation from becoming trusted support — all
  flag-gated and removable (see ARCHITECTURE). Verification governance
  fails closed by default — a missing record blocks trusted promotion, and
  the ungoverned lab exists only as an explicit config ablation. Still
  open: model-backed repair strategies and reviewers behind the existing
  protocols, richer experiment-specific control libraries, and defaulting
  the runtime to file-backed verification persistence (today the durable
  `FileVerificationStore` is opt-in wiring beside the file state store).
- **Statistician and Verifier.** Real inference behind `EpistemicAssessment` —
  today's assessments are demo-grade prediction checks; these roles bring
  power, uncertainty, and multiple-comparison awareness — plus provenance
  checking of claims.
- **Trajectory analysis.** The decision log (`DecisionRecord` + JSONL) exists;
  what Horizon 1 adds is the analysis code that reads it — utility calibration,
  cost prediction error, failure-mode tallies — feeding the ICLR plan below.

**Done when:** the system runs a real ML research question end to end, and the
resulting claims survive independent checking — including the cases where the
honest answer is that the hypothesis was wrong.

---

## Horizon 2 — Research organization

**Goal: multiple specialized agents and multiple concurrent research projects
with resource allocation.**

- **Full role set**, including Methodologist, Evidence Verifier and Scientific
  Reviewer, with utilities that genuinely differ.
- **Asynchronous execution at scale.** Cloud and SLURM backends behind the
  existing `Executor` interface; many concurrent jobs.
- **Real search policies.** Beam search, bandits, adaptive branching over
  research states. This needs calibrated value estimates, which need Horizon 1
  trajectories — which is why it is not attempted earlier.
- **Non-fungible budgets.** Replacing the greedy policy's single scalar cost
  with a defensible exchange rate between wall-clock, GPU-hours, money and
  tokens.
- **Portfolio allocation.** Deciding which of several concurrent directions
  deserves more resources, based on expected scientific value rather than
  activity or apparent progress.
- **Replication as routine.** Automatic replication of load-bearing results,
  and detection of contradictions between a program's own experiments.

**Done when:** several research programs run concurrently under one budget, and
the allocator's choices are better than uniform splitting by a measurable margin.

---

## Horizon 3 — Autonomous research lab

**Goal: persistent institutional memory, portfolio-level research strategy,
autonomous compute allocation, continuous scientific production, and
self-improvement.**

- **Institutional memory.** Cross-project knowledge in `knowledge/`: which
  methods worked, which failure modes recur, which questions are already
  answered and by what evidence. Designed against real trajectories, not
  guessed at in advance.
- **Research strategy at the portfolio level.** Choosing which questions are
  worth asking at all, and when to abandon a direction.
- **Autonomous compute allocation** against expected information gain.
- **Continuous production.** Long-running programs that accumulate knowledge
  rather than terminating at a paper.
- **Self-improvement.** The lab improving its own methodology from its own
  trajectory data — the point at which the architecture becomes the object of
  study rather than only its instrument.

---

## Near-term research plan

**Next 1–2 days — architecture and orchestration.**
Close out Horizon 1's foundation: model provider boundary, the first concrete
roles, persistence, and trajectory instrumentation. Keep the contracts stable
enough that the roles slot in without reshaping the domain core.

**Then — run autonomous ML research projects.**
Real questions on real data, end to end. Expect the first runs to expose which
parts of this architecture were wrong; that is what they are for.

**Submit strong generated research to suitable non-archival NeurIPS workshops.**
Non-archival deliberately: it gets external expert review without foreclosing
later publication of the same work.

**Preserve complete research trajectories and experiment provenance.**
Every action, decision, utility score, result and dead end — including the
programs that produced nothing. Trajectories that only cover successes cannot
support any claim about how well the architecture works.

**Use those trajectories to evaluate the architecture scientifically.**
This is the part that makes the project research rather than engineering.
Open questions the trajectory data should answer:

- Does separating orchestration from execution measurably improve reliability,
  or is it only cleaner to read?
- Do role-specific utilities produce better research than a single shared
  objective?
- Does an adversarial Skeptic reduce unsupported claims, and by how much?
- Which search policies find real findings per unit compute, and where does
  greedy actually fail?
- How often does the system reach a *correct negative* conclusion — and how
  often does it manufacture apparent progress instead? This is the measurement
  the whole design is built around.

**Develop the architecture itself into a rigorous ICLR submission.**

The contribution must be a scientific evaluation of the architecture: ablations
over the design choices above, measured on preserved trajectories, with honest
reporting of what did not work.

**Workshop acceptance counts are not the contribution.** Acceptance is a noisy,
external signal with a small sample size, and it measures how convincing the
output looks to a reviewer under time pressure — which is exactly the objective
this system is designed *not* to optimise. It is useful as one external check
among several; it must never become the utility function, and a paper whose
central claim is "n workshops accepted our output" would be measuring the wrong
thing.
