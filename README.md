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
behind them; a falsified hypothesis is recorded as falsified rather than
quietly reframed.

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
core          scientific vocabulary: state, actions + attempts, hypotheses,
              predictions, experiments, evidence, claims, assessments,
              proposals, decisions, budgets            (depends on nothing)
evidence      append-only storage of what actually happened
execution     binding an experiment design to a process, anywhere it runs
knowledge     factual read models — today, the claim-evidence graph
search        selection policies over evaluated candidates
roles         specialized agents; they propose, never commit
orchestration candidates → utilities → decision, the transition layer,
              and the trajectory log
publication   reporting (deliberately empty)
```

Dependencies point downward only, and `core` imports nothing from its siblings —
enforced by a test, not by convention.

Seven commitments shape the rest:

- **Scientific state is explicit data.** The authoritative state of a research
  program lives in a structured, immutable `ResearchState`, not in a model's
  conversation history.
- **Progress is a search over typed actions**, not a fixed pipeline of stages —
  and an action's *intent* is separate from each *attempt* at executing it. A
  failed attempt leaves the work open; it never makes it look done.
- **Hypotheses commit through predictions.** A hypothesis is testable only via
  pre-registered, machine-checkable predictions (metric, comparator,
  threshold), checked mechanically when evidence is committed — never adjusted
  to fit the outcome.
- **Evidence is immutable and machine-produced.** An `ExperimentResult` comes
  only from an executor that ran a process. Interpretations reference evidence;
  they never overwrite it.
- **Evidence is not interpretation.** How evidence bears on a claim is a
  factual annotation; whether the claim should be believed is an
  `EpistemicAssessment` with its own author, method, and version history. No
  count of evidence edges produces a verdict anywhere.
- **Negative results are first-class.** Failed runs, failed predictions,
  failed attempts and contradicted claims are recorded outcomes with
  provenance.
- **Roles propose; they never commit.** Roles produce typed, attributable
  proposals, validated and applied by a single transition layer — and each
  role has its own objective, not a shared reward. Every orchestration
  decision is preserved in a trajectory log from step one.

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
python examples/minimal_loop.py
```

This walks the full contract on a deliberately trivial question — is a seeded
draw stream biased? — and prints the resulting trajectory:

```
ResearchState → director (candidates → utilities → policy) → ActionAttempt
→ proposals → transition layer → ExperimentSpec → LocalExecutor
→ ExperimentResult → evidence → prediction checked → assessment → ResearchState'
```

The demo hypothesis is false, and the run says so at every layer. The
`heads_rate` was read out of a `metrics.json` written by a subprocess; the
pre-registered prediction is mechanically marked `failed` at commit time; and
the claim's `refuted` standing comes from an explicit epistemic assessment
that names its method and the evidence it considered. Every decision along the
way is preserved in a JSONL trajectory log.

## Tests and checks

```bash
pytest
ruff check .
mypy
```

## License

MIT — see [LICENSE](LICENSE).
