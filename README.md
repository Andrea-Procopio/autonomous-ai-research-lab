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
core          scientific vocabulary: state, actions, hypotheses,
              experiments, evidence, claims, budgets   (depends on nothing)
evidence      append-only storage of what actually happened
execution     binding an experiment design to a process, anywhere it runs
knowledge     read models over results — today, the claim-evidence graph
search        policies over scientific states and actions
roles         specialized agents, each with its own objective and authority
orchestration choosing the next action, and who performs it
publication   reporting (deliberately empty)
```

Dependencies point downward only, and `core` imports nothing from its siblings —
enforced by a test, not by convention.

Five commitments shape the rest:

- **Scientific state is explicit data.** The authoritative state of a research
  program lives in a structured, immutable `ResearchState`, not in a model's
  conversation history.
- **Progress is a search over typed actions**, not a fixed pipeline of stages.
  `generate_hypothesis`, `falsify`, `replicate` and `stop_investigation` are
  peers; which one to take next is a decision, not a line number.
- **Evidence is immutable and machine-produced.** An `ExperimentResult` comes
  only from an executor that ran a process. Interpretations reference evidence;
  they never overwrite it.
- **Negative results are first-class.** Failed runs, null results and
  contradicted claims are recorded outcomes with provenance.
- **Roles do not share a reward.** A skeptic maximising the chance of finding a
  flaw and a generator maximising novelty are different agents even when backed
  by the same model.

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
ResearchState → director proposes an action → ExperimentSpec → LocalExecutor
→ ExperimentResult → evidence recorded → ResearchState updated
```

The demo hypothesis is false, and the run says so. That is the point: the
`heads_rate` in the final state was read out of a `metrics.json` written by a
subprocess, and the claim built on it is marked `refuted` with a link to the
evidence that refuted it.

## Tests and checks

```bash
pytest
ruff check .
mypy
```

## License

MIT — see [LICENSE](LICENSE).
