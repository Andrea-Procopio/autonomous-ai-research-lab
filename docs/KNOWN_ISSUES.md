# Known issues

The durable record of defects and flaky behavior that are known,
diagnosed to some degree, and not yet fixed. An issue leaves this file
by being fixed in a commit that references it — never by being
forgotten.

## The critic's ANALYZE fails coverage on wide hypotheses

- **Where:** `orchestration/loop.py::_critic_context` (single-spec
  projection) versus `_gate_analysis` (hypothesis-wide coverage), reached
  through `_invoke_critic` when a deterministic trigger fires — observed
  live on the vision arc's large-effect trigger at the planner's fresh
  seed.
- **Status:** open; contained by design (the failed analysis commits as a
  failed attempt, nothing scientific enters state, the run continues).
  Recorded so the failing critic path is expected rather than
  rediscovered.
- **First observed:** 2026-08-22, while seating the statistician
  (Task 7C).

The critic is invoked with a single-spec context, but an assessment it
proposes is gated on citing the hypothesis-wide admissible conclusive
family — which the critic cannot see. On any hypothesis with more than
one evidence-bearing spec, the critic's honest attempt is structurally
doomed to an `AnalysisValidityError`. Closing it means giving the critic
context the same family enrichment `_assessment_context` received in 7C,
plus deciding what a critic-authored assessment should cite. Belongs
with the model-backed reviewer work, not in a drive-by.

## Re-assessment machinery does not exist

- **Where:** `runtime/frontier.py::unassessed_claims` (a claim leaves the
  frontier permanently after any assessment) and the unused
  `EpistemicAssessment.supersedes` wiring.
- **Status:** open, deliberate (Task 7C recorded the deferral).
- **First observed:** designed-in; named 2026-08-22.

A claim is assessed exactly once, whenever it first surfaces. Evidence
that lands afterwards never reopens it, and nothing in production sets
`supersedes`. The statistician's records state their `n` and the
Bonferroni `m` pinned at assessment time (sequential-analysis behavior,
disclosed) — but a campaign that keeps replicating after judgment needs
the frontier to re-open claims whose current assessment predates newer
conclusive evidence, and the assessor to supersede. That is its own
task, with its own sweep.

## Timing flakes in the real-deadline Muse tests under heavy load

- **Where:** `tests/test_muse_provider.py`, the tests that use a real
  socket and a real clock:
  `test_a_genuinely_stalled_read_times_out` and
  `test_a_genuinely_dripping_body_hits_the_deadline`.
- **Status:** open. Each observed once; both pass in isolation, in
  normal full runs, and (so far) in CI.
- **First observed:** 2026-08-18, locally, while two copies of the full
  suite ran at the same time in separate git worktrees on one machine.
  The immediate re-run of the same commit passed. The failing assertion
  was not captured.
- **Recurred:** 2026-08-20, in the sibling
  `test_a_genuinely_dripping_body_hits_the_deadline`, during a full
  suite run on a loaded machine. It passed both in isolation and on the
  immediate full re-run. Same mechanism, different test — which is the
  useful part of the sighting: the fragility is in the shared margin
  sizing, not in one test.

### What the tests do

Both are deliberately real deadline tests over a loopback socket, and
both are the same shape. In the stalled-read case the server sends
response headers and then stalls for 3.0 s, while the Muse adapter is
invoked with a 0.5 s deadline. In the dripping-body case the server
sends one byte every 0.15 s instead of stalling outright, so no single
socket operation ever times out and only the whole-call deadline can
end it. Each asserts:

1. the call raises `ProviderTimeoutError` (the watchdog, not the
   stall, ends the exchange), and
2. the call returns within 3.0 s of wall clock (which distinguishes
   "deadline enforced" from "hang until the server relented").

### Mechanism

Both assertions ride on real wall-clock margins, and both margins
shrink under CPU contention:

- If the process is starved for about 2.5 s (GIL contention, many
  concurrent interpreters, timer-thread scheduling delay), the
  elapsed-time bound in assertion 2 can be crossed even though the
  deadline machinery worked.
- If the watchdog timer thread is delayed past the server's 3.0 s
  stall, the server closes the connection first and the failure class
  changes from `ProviderTimeoutError` to a transport error, failing
  assertion 1.

Neither mode indicates a defect in the adapter. The margins are simply
sized for a lightly loaded machine.

### Candidate fixes (deliberately not applied in a drive-by)

- Derive the elapsed bound and the server stall from shared constants
  with a wider ratio (for example: deadline 0.5 s, stall 10 s, elapsed
  bound 5 s), so contention has to be extreme before either assertion
  lies. The second sighting strengthens this one: shared constants
  would widen every deadline test at once.
- Capture and report which assertion failed on the next occurrence
  before choosing the fix.

The test's value is exactly that it uses a real socket and a real
clock — it reproduced the >570 s live stall class that motivated the
watchdog — so the fix should widen margins, not fake the clock.

## Operational: OpenAlex anonymous search returns 429 under cluster load

- **Where:** live prior-art runs through `OpenAlexProvider` without
  `OPENALEX_API_KEY`.
- **Status:** external service behavior, not a defect here. Recorded
  so the failure mode is expected rather than rediscovered.
- **First observed:** 2026-08-19, mid-way through the second Task 5D.1
  live attempt (preserved as `live_runs/task5d1-2026-08-19.partial-2`).
- **Recurred:** 2026-08-19, under sustained cluster load, one
  candidate into each of the first two Task 5D.2 live attempts
  (preserved as `live_runs/task5d2-2026-08-19.partial-1` and
  `.partial-2`, each with its completed candidate durable beside the
  abort). The advertised 30-40 s backoff was not enough between
  candidates; the completing rerun needed a wait of several minutes.

The anonymous search cluster sheds load with HTTP 429 ("Anonymous
search is temporarily rate-limited... retry in Ns, or use a free API
key"). The adapter's bounded single retry honors `Retry-After`, but a
saturated cluster can outlast it. The run then fails closed with
durable partials: coverage stays incomplete, no verdict is recorded,
and a later rerun re-executes from a fresh root (cached queries replay
free). Remediation: wait and rerun, or set `OPENALEX_API_KEY` (free)
for uninterrupted access. The bounded retry policy is deliberate; a
patient retry loop would hide provider degradation inside a scientific
run.
