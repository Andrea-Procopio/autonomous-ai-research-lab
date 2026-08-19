# Known issues

The durable record of defects and flaky behavior that are known,
diagnosed to some degree, and not yet fixed. An issue leaves this file by
being fixed in a commit that references it — never by being forgotten.

## Timing flake: `test_a_genuinely_stalled_read_times_out` under heavy load

- **Where:** `tests/test_muse_provider.py::test_a_genuinely_stalled_read_times_out`
- **Status:** open. Observed once; passes in isolation, in normal full
  runs, and (so far) in CI.
- **First observed:** 2026-08-18, locally, while two copies of the full
  suite ran concurrently in separate git worktrees on the same machine.
  The immediate re-run of the same commit passed. The failing assertion
  was not captured.

### What the test does

It is one of the deliberately *real* deadline tests: a loopback socket
server sends response headers and then stalls for 3.0 s, while the Muse
adapter is invoked with a 0.5 s deadline. The test asserts two things:

1. the call raises `ProviderTimeoutError` (the watchdog, not the stall,
   ends the exchange), and
2. the call returns within 3.0 s of wall clock (the elapsed-time bound
   that distinguishes "deadline enforced" from "hang until the server
   relented").

### Mechanism

Both assertions ride on real wall-clock margins, and both margins shrink
under CPU contention:

- If the process is starved for ~2.5 s (GIL contention, many concurrent
  interpreters, timer-thread scheduling delay), the elapsed-time bound
  in assertion 2 can be crossed even though the deadline machinery
  worked.
- If the watchdog timer thread is delayed past the server's 3.0 s stall,
  the server closes the connection first and the failure class changes
  from `ProviderTimeoutError` to a transport error, failing assertion 1.

Neither mode indicates a defect in the adapter; the margins are simply
sized for a lightly loaded machine.

### Candidate fixes (deliberately not applied in a drive-by)

- Derive the elapsed bound and the server stall from shared constants
  with a wider ratio (e.g. deadline 0.5 s, stall 10 s, elapsed bound
  5 s), so contention has to be extreme before either assertion lies.
- Capture and report which assertion failed on the next occurrence
  before choosing the fix.

The test's value is exactly that it uses a real socket and a real clock
— it reproduced the >570 s live stall class that motivated the watchdog
— so the fix should widen margins, not fake the clock.

## Operational: OpenAlex anonymous search returns 429 under cluster load

- **Where:** live prior-art runs through `OpenAlexProvider` without
  `OPENALEX_API_KEY`.
- **Status:** external service behavior, not a defect here; recorded so
  the failure mode is expected rather than rediscovered.
- **First observed:** 2026-08-19, mid-way through the second Task 5D.1
  live attempt (preserved as `live_runs/task5d1-2026-08-19.partial-2`).

The anonymous search cluster sheds load with HTTP 429 ("Anonymous
search is temporarily rate-limited... retry in Ns, or use a free API
key"). The adapter's bounded single retry honors `Retry-After`, but a
saturated cluster can outlast it; the run then fails closed with
durable partials — coverage stays incomplete, no verdict is recorded,
and a later rerun re-executes from a fresh root (cached queries replay
free). Remediation: wait and rerun, or set `OPENALEX_API_KEY` (free)
for uninterrupted access. The bounded retry policy is deliberate; a
patient retry loop would hide provider degradation inside a
scientific run.
