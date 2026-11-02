# Shared CPU team threshold evaluation (#582)

## Decision

Keep the existing Rayon default and existing work gates. Keep shared Engine-team
dispatch explicitly experimental. On this Xeon, it shortened L32 Opt/PhysCal but
increased aggregate user CPU; L64 Opt did not show a useful wall-time advantage
at 20/40 µs. There is no portable crossover or universal default improvement.
For a controlled L32 experiment, 20/40 µs remain candidates; 80 µs is unsuitable
for the tested L64 Opt configuration. This is an evaluation outcome, not a claim
to have solved the L128 CPU scaling issue #586.

## Matched experiment

Same release binary in every condition: source
`05697d2aae8c50022e6d320513ac3a7846c3c277`, executable SHA-256
`731ca0eb15ce0d5e18aa866094105034cc56d71a5f2f6620eb34c32ab2d8fb98`.
The later existing-worker affinity repair in #620 is not part of this binary.
MPI CPU placement is unset; no MPI launcher is used. The MPI-enabled binary
executes one process/one walker. Source/build/software logs and input hashes
are retained alongside this report.

Linux x86_64, Xeon E5-2699 v3, physical CPU mask 0–3, development image
`sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`,
Rust 1.99.0, MPICH 4.2.0 linkage, pthread OpenBLAS. OpenBLAS/OMP=1,
Rayon capacity=4, tenferro participants=4, held lifetime/default spin,
Engine worker budget unset. Inner participants=1/4, work gate=20/40/80 µs,
shared dispatch off/on. The one-inner-participant control still permits the
four-participant SR Engine; it is not a globally single-threaded process.
The locked providers are tenferro-cpu 0.8.1, threadteam 0.2.0 and spindle 0.2.6;
`engine-dependencies.txt` records their crates.io checksums. The held Engine
owns the reservation; fixed-block dispatch borrows it nonblockingly, with a
serial busy fallback and Rayon unavailable fallback. Existing bounded Spindle
regions take priority. This experiment changes neither those ownership rules
nor the provider's wait implementation.

L32 Opt, fixed-parameter L32 PhysCal and L64 Opt use seed 1, 300 samples,
warmup 10 and interval 1. Opt uses 20 steps/window 1, PhysCal 20 windows.
Each condition receives one excluded warmup and three measured processes.
Worker/gate/shared order reverses on alternating repetitions. This is 144
processes (36 warmups, 108 measurements), followed by six separately profiled
processes. Bash timing includes CLI startup, initialization, calculation and
output, excluding Docker startup. No concurrent local compilation ran during
the sequence. A separate host observer recorded 719 samples over 718.605 s;
one-minute system load ranged 1.41–3.20. This is neither assigned-core utilization
nor proof of isolation. The host was shared and frequency was not controlled.

Four-inner-participant wall medians (seconds):

| Workload | Gate | Rayon | Shared Engine team |
|---|---:|---:|---:|
| L32 Opt | 20 µs | 2.156 | 1.562 |
| L32 Opt | 40 µs | 1.999 | 1.664 |
| L32 Opt | 80 µs | 2.732 | 1.652 |
| L32 PhysCal | 20 µs | 2.613 | 1.591 |
| L32 PhysCal | 40 µs | 2.349 | 1.684 |
| L32 PhysCal | 80 µs | 3.370 | 1.626 |
| L64 Opt | 20 µs | 5.339 | 5.338 |
| L64 Opt | 40 µs | 5.343 | 5.419 |
| L64 Opt | 80 µs | 9.409 | 7.477 |

At 40 µs, Rayon/shared user CPU medians were L32 Opt 4.579/6.288 s,
L32 PhysCal 5.378/6.497 s and L64 Opt 17.283/21.080 s. Wall and CPU medians
are selected independently; they need not refer to the same repetition.
All individual wall/user/system samples and one-inner-participant controls
are in `summary.json`. Their wall medians span 2.382–2.441 s (L32 Opt),
2.121–2.279 s (L32 PhysCal) and 11.581–11.718 s (L64 Opt); short shared-host
samples do not resolve small regressions.

## Separate site observations

These six runs enable per-site scope/gap and shared-dispatch counters at four
inner participants/40 µs. They perturb scheduling and are excluded from timing.
Selected profiled **pool-labelled scope** totals, in milliseconds:

| Workload / site | Calls | Items | Rayon | Shared |
|---|---:|---:|---:|---:|
| L32 Opt Pfaffian | 8098 | 64784 | 922.2 | 569.8 |
| L32 PhysCal Pfaffian | 6450 | 51600 | 967.9 | 464.4 |
| L32 PhysCal Green | 6000 | 1152000 | 1068.6 | 546.9 |
| L64 Opt Pfaffian | 8144 | 65152 | 2172.9 | 2130.2 |
| L64 Opt sampling update | 139978 | 1119824 | 1300.3 | 1552.5 |

L32's sampling-update site stays serial: 69622 Opt calls and 15150 PhysCal
calls in each path. L64's update is pooled but slower in this single shared
observation. Scope times can overlap/nest and cannot be added into a critical
path or interpreted as isolated wake latency. Gap histograms are elapsed gaps
between top-level scopes on each thread, attributed to the next site, not
global serial stalls. `profile-summary.json` retains all scopes and bucket
counts and checks their hot/cold sums.

The shared cases observed four distinct participants with every attempt
successful: 20158 Opt, 18452 PhysCal and 160182 L64 Opt. Busy/unavailable
counters were zero. This establishes activation, not simultaneous CPU use.
Each workload also records 20 **serial** Pfaffian calls in addition to the
pooled calls above; counts must include both modes when comparing to a serial
profile. Pool-labelled counts alone are not the total recomputation count.

## Numerical and ownership limits

All finite formatted non-timing numeric tokens across the 36 measured runs
per workload agree with the first reference run, maximum absolute difference
0 (4 files per Opt, 80 for PhysCal). Diagnostic outputs agree with the matching
timed references too. This is not an event-by-event RNG or C oracle. Independent
fixed-case C/RNG/configuration and five-model move-event comparisons are in
the earlier [shared-team report](../shared-team-20261102/README.md) and
[gap report](../gap-sites-20261102/README.md), including nested dispatch,
bounded-team priority, busy/unavailable fallback and cleanup checks.

Shared dispatch does not remove the driver's hoisted Rayon pool or provide a
global worker cap. Do not infer CPU efficiency, quota compliance, multiwalker
benefit or MPI placement correctness from these single-process measurements.
Those are the separate #583 ownership and budget evaluation.

## Reproduce and inspect

Use the image and source above, a checkout-specific target volume, and build
`cargo build -p mvmc-cli --features mpi --release --locked`. Extract `raw.tar.gz`
into a fresh directory; it supplies the exact `inputs/` copies and full
`mvmc-582-crossover-measured/` outputs, including warmups/diagnostics.
Identical file contents are stored as tar hardlinks without discarding data.
Reanalyse without rerunning:

```sh
tar -xzf raw.tar.gz -C FRESH_DIRECTORY
uv run --no-project python analyze.py FRESH_DIRECTORY/mvmc-582-crossover-measured
uv run --no-project python analyze-profile.py FRESH_DIRECTORY/mvmc-582-crossover-measured
```

`run.sh` and `observe.py` preserve the exact executed local commands. For a new
run, adapt their absolute `/tmp` paths, image/volume and CPU mask, supply the
same inputs, and use fresh output/observer paths. `observe.py` invokes `run.sh`
and captures host `/proc/stat`, load and CPU pressure until the child exits.
`observer.jsonl.gz` preserves all samples and exit status 0. Profiling is
enabled only for the six diagnostic processes. Do not compile concurrently
with measurements or compare this source's timings to later binaries as a
matched before/after experiment.

## #582 acceptance evidence

1. The released threadteam/provider versions and checksums are recorded above.
   The Engine owns the held team; fixed blocks borrow it without blocking and
   nested blocks run inline. The earlier shared-team report records the actual
   API integration and provider-specific waiting experiments.
2. #608 supplies the opt-in driver-lifetime shared dispatch. The paired E2E
   sweep here adds gate and one-inner-participant controls. Earlier
   [spin experiments](../shared-team-20261102/README.md#mpi-wait-policy-comparison)
   and [actual Engine wait diagnostics](../engine-wait-20261102/README.md)
   record default/zero spin, dispatch/lock costs and park events with explicit
   scope limits. These are measurements, not proof of a causal wake mechanism.
3. Sampling updates, Pfaffian recomputation and Green measurement are separated
   in the site observations. L32 Opt/PhysCal and L64 Opt support the decision
   above; three repetitions on one shared host do not justify replacing defaults.
4. The earlier shared-team report's five fixed real/complex/FSZ move-event
   comparisons and 1/2/4-worker repeats check exact RNG/proposal/decision records,
   while C-derived fixtures check numerical/counter/configuration contracts.
   They supplement the finite output comparisons here. No tolerance was widened.
5. Existing bounded-team priority, nested inline work, busy and unavailable
   fallback, error/unwind cleanup and stage/solve leases have focused tests in
   `threading.rs`, `stage_backend.rs` and `cpu_worker_budget.rs`; merged #608,
   #604 and #616 carry their verification. Shared dispatch borrows the existing
   Engine instead of constructing a team per region, and a busy Engine runs
   blocks serially rather than starting another competing dispatch. It does
   not remove the already existing hoisted Rayon pool. Coordinated whole-process
   budgets/rank placement remain #583; this report does not close that issue.

This completes the single-walker experimental evaluation with a decision to
retain the production baseline, rather than adopting a host-specific speedup
as the default. It does not close #584/#585 until their other criteria are met.
