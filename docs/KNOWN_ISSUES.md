# Known issues

The durable record of defects and flaky behavior that are known,
diagnosed to some degree, and not yet fixed. An issue leaves this file
by being fixed in a commit that references it — never by being
forgotten.

## Repair-loop jobs are not individually recoverable (blocking PR4)

- **Where:** `orchestration/loop.py::_handle_implementation_invalidity`
  and `orchestration/debug_loop.py::_rerun`, which submit jobs inside an
  attempt rather than as attempts of their own.
- **Status:** open, and **blocking for PR4** (real experiment
  execution). Not blocking for Task 6D, which is where it was found.
- **First observed:** 2026-08-20, while scoping Task 6D.

### What happens

Task 6D makes an *attempt* recoverable: the attempt journal records how
far it got, its commit bundle is on disk before it is applied, and a
killed process is finished by the next one. The bounded debug and
implementation-repair loops submit their reruns inside the attempt that
triggered them, and those submissions get no journal entry of their own.

So a crash during a repair rerun leaves an attempt whose last phase is
whatever preceded the repair, and no bundle. Recovery takes the
conservative arm: it charges the attempt's authorization in full, marks
the settlement `CONSERVATIVE_MAX`, and closes the attempt `ABANDONED`.

### Why the accounting is fine and this is still a problem

Nothing is hidden and nothing is paid twice. The charge is recorded as
an unmeasured maximum rather than a cost anyone observed, the ledger and
the state are reconciled, and `verify_run` passes. As *accounting* it is
correct.

As *execution recovery* it is not. The rerun's job may have completed;
its outputs may be sitting in the run directory; and recovery walks past
all of that because nothing wrote down that the job existed. Today that
costs seconds — the deterministic executor's reruns are cheap. Under
PR4 it costs a GPU job, and abandoning one of those is a real loss
however honestly it is billed.

### What would close it

Either of:

1. **Journal repair-loop jobs individually.** Each rerun becomes its own
   attempt: its own reservation, its own `STARTED`/`SUBMITTED`/
   `OUTPUTS_DURABLE`, its own derived job id. Recovery then reattaches
   to the exact job instead of abandoning its parent. This is the real
   fix and it is the direction the journal was shaped for — the phases
   and the derived job id already work per attempt; what is missing is
   that the repair loop opens one.
2. **Refuse repairs above a cost threshold.** If a job is expensive
   enough that losing it matters, do not start a rerun that cannot be
   recovered. Cheaper to build, and it trades a recovery gap for a
   capability gap.

Option 1 unless PR4 is under time pressure, in which case option 2 is an
honest stopgap that must not become permanent.

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
