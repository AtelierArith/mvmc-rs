# Rayon inner-kernel diagnostics (#581)

These are two separate diagnostics: call-site instrumentation of actual L32/L64 runs, and a synthetic dispatch test. They do not independently establish an end-to-end speedup or isolate the cause of an application slowdown. `MVMC_RS_INNER_PROFILE=1` times scopes and locks a shared table, perturbing execution; nested scope times can overlap and must not be added as total runtime. The logged user/system CPU times include the entire instrumented program. These measurements are not CPU utilization samples or an idle-spin attribution.

## Instrumented applications

Each case was run once, without an additional warmup, using fixed seed 1, 300 samples, warmup 10 and interval 1. Opt uses 20 steps/window 1; L32 PhysCal uses the supplied fixed parameter file and 20 windows. The one-thread and four-thread runs both use tenferro SR, Rayon capacity 4, OpenBLAS/OMP 1, default gate and wait settings, and CPUs 0–3. L64 PhysCal was not run because the repository has no corresponding fixed-parameter fixture. Raw logs are stored as deterministic gzip files alongside this file, with input hashes. Combined stdout and stderr can interleave even within a line. The binary is the independently built #604 handle-owned-lease candidate, with its default held lifetime: release SHA-256 `15061917d879f99688d59e09adae5832bec9ba1eedb9fef0a4795fce6322bfb1`; its source and build notes are in [solve-lease-experiment.md](../solve-lease-experiment.md). The profiler is a diagnostic build/runtime option; see the log for call-site paths and line numbers.

| Workload | Site | Workers | Calls | Total profiled scope time |
|---|---|---:|---:|---:|
| L32 Opt | Pfaffian recomputation | 1 serial | 8,118 | 1,366 ms |
| L32 Opt | Pfaffian recomputation | 4 pool | 8,098 | 938 ms |
| L32 Opt | sampling update | 1 serial | 69,622 | 395 ms |
| L32 Opt | sampling update | 4 serial | 69,622 | 453 ms |
| L32 PhysCal | Green measurement | 1 serial | 12,000 | 552 ms |
| L32 PhysCal | Green measurement | 4 pool | 6,000 | 1,122 ms |
| L32 PhysCal | Green measurement | 4 serial | 6,000 | 201 ms |
| L64 Opt | Pfaffian recomputation | 1 serial | 8,164 | 6,690 ms |
| L64 Opt | Pfaffian recomputation | 4 pool | 8,144 | 2,414 ms |
| L64 Opt | sampling update | 1 serial | 139,978 | 3,168 ms |
| L64 Opt | sampling update | 4 pool | 139,978 | 1,244 ms |

The shown sites are `pfaffian.rs:222`, `sampling/updates.rs:604`, and `observables/green_measurements.rs:220` in that source. The L32/L64 Opt `zvo_out.dat` and `zqp_opt.dat` and L32 PhysCal `zvo_out_001.dat` were byte-identical between worker counts. Pfaffian recorded-call counts differ by 20, which this report does not explain. Complete RNG states, proposals and decisions were not captured by this diagnostic; equality of a few output files is not a claim of trajectory parity. The measurements prioritize these sites for controlled investigation but do not by themselves justify a new threshold.

To reproduce, prepare `/tmp/mvmc-581-profile/inputs/l32-opt` and `l32-phys` as in [mpi-placement.md](../mpi-placement.md), including the 20-window PhysCal setting; copy `benchmark/hubbard_chain/inputs/hubbard_chain_L64/*.def` into `inputs/l64-opt`. Build the #604 revision in the Linux dev container using `cargo build -p mvmc-cli --features mpi --release --locked` with its own target volume. The saved [run.sh](run.sh) records the exact six Docker commands, image and volume names used; adapt those names and CPU masks to the local host. The output directory must not already contain the generated output paths. The input hashes record the exact copies used.

## Synthetic dispatch

The existing `rayon_dispatch_479` probe was configured with one and four worker threads using `MVMC_DISPATCH_THREADS`, with 1,000 timed regions per variant (`MVMC_DISPATCH_ITERATIONS=1000`), 200 local warmups per variant, then a serial busy-wait gap of 0, 5, 20 or 60 µs before each timed region. Each region had either zero synthetic work or 5 µs per block. The saved [script](dispatch/run.sh) ran one whole-process warmup and three measured processes per worker count, alternating their order, pinned to the same four physical cores. The pool is constructed separately for each process. Binary SHA-256 `72500ff390fc3b0233e9de34ee5cc0ae5cc008effceaba2f75508a42c4019190`; instrumented test source SHA-256 `78654587ca223038ef874b350fc2c76a707326f2f198be5acebb86f550a64e90` on main `176774af8e28bb6e12f37884ef1b25b816c6af53`. An independent untimed broadcast observed the configured participants (masks `0x1` and `0xf`); this is not proof of CPU utilization during each timed call.

The table shows the median of three per-process means for **zero synthetic work**, in microseconds. It is not a distribution of individual call latencies, and the worker-count rows use the same four-CPU affinity, not a single-core isolation experiment. Raw records include the other variants and nonzero-work cases.

| Workers | Dispatch | Gap 0 µs | Gap 5 µs | Gap 20 µs | Gap 60 µs |
|---:|---|---:|---:|---:|---:|
| 1 | Inside hoisted pool | 0.33 | 0.25 | 0.27 | 0.30 |
| 1 | Outside pool | 4.18 | 5.92 | 10.87 | 10.12 |
| 4 | Inside hoisted pool | 1.97 | 1.92 | 1.90 | 7.28 |
| 4 | Outside pool | 8.67 | 8.49 | 11.58 | 12.93 |

Whole-process wall/user/system medians for the complete mix of regions and deliberate busy gaps were respectively 1.406/1.486/0.279 seconds with one worker and 1.551/3.224/1.358 seconds with four. These totals include the deliberately busy caller, so they do not quantify idle spinning of Rayon workers. The slower cold call suggests a wake-up cost in this synthetic setup; actual serial-gap distributions were not measured here.

Reproduce in the dev container with a separate target directory:

```sh
CARGO_TARGET_DIR=/target cargo nextest list -p mvmc-core --test rayon_dispatch_479 --cargo-profile test-fast --locked
MVMC_DISPATCH_THREADS=4 MVMC_DISPATCH_ITERATIONS=1000 \
  /target/test-fast/deps/rayon_dispatch_479-<hash> --ignored --exact rayon_dispatch_variants --nocapture
```

Use the saved [run.sh](dispatch/run.sh) on the host for the exact alternating 1/4-worker sequence, image and container target volume, adjusting CPU affinity and names as needed. The binary is selected in the script; Docker startup is outside the reported Bash `time`.

Both experiments ran on Linux x86_64 6.8.0-147, Xeon E5-2699 v3, on separate physical cores of socket 0, Rust 1.99.0, MPICH 4.2.0, pthread OpenBLAS and dev-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`. No frequency control or host-load isolation was applied. Further direct serial-gap sampling, worker utilization and matched repeated uninstrumented L32/L64 runs are required to complete #581.

## Instrumented gaps and disabled-collector overhead

The experimental collector is opt-in via `MVMC_RS_INNER_GAP_PROFILE=1`. For each calling thread independently, it counts the elapsed wall time from the end of one top-level, nonempty profiled scope to the start of the next such scope on that thread. Nested scopes are excluded; the first scope on each thread has no preceding sample. `hot`/`cold` is the next scope's thread-local `hot_scope` state, not a label on the preceding work. Worker and caller streams are combined, and a scope on a worker may overlap a scope on the driver. Counts can therefore differ between worker settings and do not measure a globally serial phase, idle spin, wake-up latency, or individual call-site gap. Instrumentation overhead and other work/scheduling between scopes are included.

One instrumented L32 process per workload/worker count, with `MVMC_RS_INNER_PROFILE=1` and the gap collector enabled, produced the following counts in buckets `<1`, `<5`, `<20`, `<60`, `<100`, `<1000`, `>=1000` microseconds. [Complete logs and command](gap/) contain call-site totals and wall/user/system times; these instrumented timings are not performance comparisons.

| Workload | Workers | Next scope | Gap histogram |
|---|---:|---|---|
| Opt | 1 | cold | 11906, 12034, 6073, 7, 0, 26, 53 |
| Opt | 1 | hot | 260368, 3984, 294, 22, 0, 1, 0 |
| Opt | 4 | cold | 11339, 12544, 6123, 14, 0, 25, 54 |
| Opt | 4 | hot | 227229, 36888, 532, 8, 11, 1, 0 |
| PhysCal | 1 | cold | 12603, 17309, 67, 2, 0, 19, 1 |
| PhysCal | 1 | hot | 203377, 4851, 299, 2, 0, 16, 4 |
| PhysCal | 4 | cold | 11527, 68968, 4785, 96, 2646, 1420, 504 |
| PhysCal | 4 | hot | 89938, 118273, 316, 2, 0, 1, 19 |

For each workload and worker count, the 4 Opt or 80 PhysCal formatted numeric output files (excluding timing files) agreed with a corresponding uninstrumented run to the printed precision; this does not compare full RNG states or proposals. The source patch is [collector.patch](gap/collector.patch), against main `176774af8e28bb6e12f37884ef1b25b816c6af53`; apply with `git apply --unidiff-zero`. The candidate release binary SHA-256 was `5ee7c581f626932bb88a73cf02f77667e8314f74843f768182f376c8654359cf`.

A separate alternating experiment compared two builds with **both profiling settings unset**, one warmup and three measured processes per workload/worker count, on CPUs 0–3. It used the same fixed L32 inputs as above. The baseline was main-equivalent binary SHA-256 `bc98b9ddcbafdcdd7f4e7d32b5f06979a07fcf63ba72570e47e69bda027d9a62`; the candidate was the binary above. Docker startup was excluded while CLI startup and output were included. [Raw timings, hashes, summary and exact command](overhead/) are preserved.

| Workload | Workers | Baseline median | Collector disabled median |
|---|---:|---:|---:|
| Opt | 1 | 2.418 s | 2.417 s |
| Opt | 4 | 2.105 s | 2.054 s |
| PhysCal | 1 | 2.216 s | 2.168 s |
| PhysCal | 4 | 3.030 s | 2.417 s |

All 4 Opt and 80 PhysCal formatted numeric output files per worker setting matched to printed precision across the six measured runs. The unusually large four-worker PhysCal spread and only three repetitions preclude attributing the difference to this change; these data do not demonstrate a speedup. Full RNG and decision parity is not established by this output check.
