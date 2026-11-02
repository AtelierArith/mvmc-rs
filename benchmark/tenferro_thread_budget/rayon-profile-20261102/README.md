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

## Exploratory Rayon waiting CPU accounting

On the same Xeon/Linux dev container, an independent process repeatedly invoked an empty broadcast for 3 seconds per condition, with one or four Rayon threads. It slept for a *requested* 0, 20, 60 or 1000 microseconds between calls. Each condition was measured three times in fixed order, using CPUs 0–3. The test used `/proc/self/stat` and `/proc/thread-self/stat` user plus system CPU ticks at 100 Hz; `other` is process CPU time minus the calling test thread's CPU time. For inside-pool calls, the caller is itself a Rayon worker; for outside calls, all Rayon workers are counted in `other`. Test-harness and read overhead are also included. Thus this is CPU accounting for the entire repeated workload, including broadcasts and dispatch, **not isolated idle-spin or wake-up time**. Sleep durations are subject to OS scheduling and are not directly measured; calls per second are reported to expose the effective rate.

| Threads | Caller | Requested sleep | Other CPU cores, median | Calls/s, median |
|---:|---|---:|---:|---:|
| 1 | outside | 0 µs | 1.000 | 290579 |
| 1 | inside | 0 µs | 0.003 | 5067555 |
| 4 | outside | 20 µs | 2.000 | 10834 |
| 4 | inside | 20 µs | 1.547 | 12311 |
| 4 | outside | 60 µs | 1.700 | 6749 |
| 4 | inside | 60 µs | 1.687 | 7921 |
| 4 | outside | 1000 µs | 0.380 | 827 |
| 4 | inside | 1000 µs | 0.250 | 846 |

Three-second tick-quantized samples and nonrandomized condition order limit precision. Empty broadcasts and sleeping callers do not reproduce actual kernel work or the measured application gap distributions. The nonzero CPU use at 1 ms suggests further idle/wake investigation but cannot attribute those cycles to spinning. The full [raw data, summary, analysis and command](idle/) include all 16 conditions and three repetitions. The test source is `crates/mvmc-core/tests/rayon_idle_probe_581.rs`. With Rust, Rayon, `jq` and Linux `/proc` available, run on CPUs 0–3 with `taskset -c 0-3 benchmark/tenferro_thread_budget/rayon-profile-20261102/idle/run.sh > idle.log 2>&1`, then `uv run --no-project python benchmark/tenferro_thread_budget/rayon-profile-20261102/idle/analyze.py idle.log`. Keep builds and unrelated workloads off the measurement CPUs during timing. Hardware, image, versions and isolation limitations are as documented above.

The measured source was parent `f915bca5af135372cbc3fa0cc74367a9c09c25a5` plus the new probe source (subsequently rustfmt-formatted without semantic changes). The measured test executable SHA-256 was `3ea7d54a283e55c34e45141b1a2ea018b0b8f08de9e3c724d6b9b81ff2a28b9a`. The probe is Linux-only because it uses `/proc`; it is ignored during normal test runs.

## Four-worker work-gate sweep

A separate same-binary experiment varied `MVMC_RS_INNER_MIN_WORK_NS` at 20000, 40000, 80000 and 160000 ns for L32 Opt, L32 PhysCal and L64 Opt. This setting is an estimated serial-work gate, not a measured actual duration, and applies to both hot and cold regions in this sweep. Each condition had one warmup and three timed runs; the four gate values were alternated in forward and reverse order between repetitions. All used four Rayon/inner workers, OpenBLAS/OMP one thread, tenferro SR, default Engine wait settings, fixed seeds and the inputs/window/sample settings described above. The same collector-capable release binary was used throughout with both profiling settings unset; its SHA-256 is `5ee7c581f626932bb88a73cf02f77667e8314f74843f768182f376c8654359cf` and its source is main `176774af8e28bb6e12f37884ef1b25b816c6af53` plus `gap/collector.patch`. CPUs 0–3 were used; CLI startup and output are included, Docker startup is excluded. The host was not isolated or frequency controlled; load was not recorded.

| Workload | Gate 20 µs | Gate 40 µs | Gate 80 µs | Gate 160 µs |
|---|---:|---:|---:|---:|
| L32 Opt | 2.091 s | 2.226 s | 2.793 s | 2.708 s |
| L32 PhysCal | 2.885 s | 2.426 s | 3.436 s | 2.881 s |
| L64 Opt | 5.539 s | 5.590 s | 11.552 s | 10.678 s |

Values are wall-time medians of three, not phase timings. L32 PhysCal had substantial run-to-run variation (e.g. 40 µs: 2.390, 3.054, 2.426 s); the apparent choice between 20 and 40 µs is therefore not established. L64 Opt was slower in every 80/160 µs sample than every 20/40 µs sample on this host, so keeping a candidate at or below 40 µs deserves further profiling. Do not generalize the optimum or alter defaults from this three-sample sweep. Across the 12 measured runs for each workload, all finite non-timing formatted values in 4 Opt or 80 PhysCal files matched at printed precision; this alone does not compare full RNG states or decisions. Per-run wall/user/system samples, input hashes, logs, numerical output and analysis are in [gates](gates/).

Reproduce by preparing the inputs and binary as described above, adjusting paths and CPU masks in `gates/run.sh`, and running it with a fresh output directory. Then `uv run --no-project python gates/analyze.py /tmp/mvmc-581-gates` recomputes the summary. The saved `raw.tar.gz` can instead be extracted into an empty directory and passed to the analyzer. These measurements do not cover single-worker regressions or L64 PhysCal; keep those as follow-up requirements.

A subsequent one-inner-worker negative control used the same binary, inputs, CPU mask and other settings, with gates 20 and 160 µs. Each received one warmup and three alternating measured runs. Its median wall times were respectively 2.457/2.459 s for L32 Opt, 2.204/2.165 s for L32 PhysCal and 11.852/11.725 s for L64 Opt. L64 20 µs samples were 13.262, 11.809 and 11.852 s; the observed differences do not establish a gate effect. Rayon capacity remained four and the tenferro thread setting was unset, so “one worker” refers specifically to the mvmc inner-worker setting, not a claim that the process had only one thread or core of CPU use. All finite non-timing formatted output values matched at printed precision across the six measured runs per workload. [gates-serial](gates-serial/) contains the corresponding raw data, CPU times, input hashes, analysis and exact command; adjust the fixed paths and use a fresh output directory. The control was run after the four-worker sequence, so cross-table elapsed times are not a matched scaling measurement.

A separate matched one/four inner-worker comparison used the same binary and inputs, explicitly set a 40000 ns gate, and alternated worker counts in one sequence per workload, one warmup and three measured runs each. Rayon capacity was four, OpenBLAS/OMP one thread, tenferro thread count unset; the one-worker label does not imply a single-threaded process. Wall time includes CLI startup/output but not Docker startup. No host-load sampling, frequency control or host isolation was performed.

| Workload | 1 worker wall | 4 workers wall | 1 worker user/sys CPU | 4 workers user/sys CPU |
|---|---:|---:|---:|---:|
| L32 Opt | 2.470 s | 2.087 s | 2.895/0.069 s | 4.799/0.699 s |
| L32 PhysCal | 2.245 s | 2.451 s | 2.223/0.035 s | 5.590/1.175 s |
| L64 Opt | 11.554 s | 5.413 s | 12.433/0.104 s | 17.505/2.576 s |

Each cell is the median of its respective three values; independently selected CPU medians need not correspond to the median-wall run. L32 PhysCal four-worker wall samples were 2.451, 2.279 and 3.054 s, so this short sample shows no reliable parallel speedup. The finite formatted non-timing outputs matched across the six measured runs at printed precision: four files per Opt workload and 80 for PhysCal. No full RNG or decision trace was captured in these timed runs. [scaling](scaling/) preserves individual wall/user/system values, outputs, hashes, exact script and analysis. Reproduce after preparing the same inputs/binary, adjusting the fixed host paths, CPU masks and volume in `scaling/run.sh` and using a fresh output directory; analyze with `uv run --no-project python scaling/analyze.py /tmp/mvmc-581-scaling`. Extract `scaling/raw.tar.gz` to reanalyze the saved runs. These are not C-versus-Rust timings or L128 results.

## Post-broadcast idle observation

To complement the repeated-broadcast accounting, a Linux-only ignored test creates a 1- or 4-worker Rayon pool, completes an empty broadcast, then gives it no further work for one second. The caller sleeps either outside the pool or inside a hoisted `install`. At requested elapsed times 10, 50, 100, 500 and 1000 ms it reads process CPU ticks and subtracts the calling thread's CPU ticks. Each condition used one warmup and five measured processes in alternating order, CPUs 0–3, the same host and development container as above. These are **cumulative other-thread CPU tick differences** since the observation start, not call latency or actual simultaneous utilization; the inside caller is itself one pool worker whereas outside includes all pool workers. `/proc` and test-harness overhead remain included, and sequential reads of process and caller counters can introduce measurement error.

The host reported 100 ticks per second. In all 100 measured observations (four worker/caller combinations × five runs × five time points), the other-thread difference was 0 ticks. This means no increase was observed at 10 ms tick resolution; it does not prove zero CPU consumption, exclude short spins below the resolution, or describe repeated dispatch or an mVMC workload. In particular it does not explain the nonzero whole-process accounting under repeated broadcasts in the preceding diagnostic. The elapsed timestamps in the raw logs show actual sampling times; requested sleep boundaries can be delayed by the OS. No host-load sampling, frequency control or host isolation was applied.

Source is the initial tick-only test revision `c40d65fb` (based on main `1b8ac154e210e51e9c88262622d0b6fd0fdc5604`); the test executable SHA-256 was `006a5bd0c3d32323eeee1f4ae02b3875e1fd903401ce22d466223d84a167e137`. The probe passed targeted Clippy and runs only when explicitly selected. [post-idle](post-idle/) contains logs, per-run summary and scripts. Build in the Linux dev container with a checkout-specific target volume: `cargo test -p mvmc-core --test rayon_idle_probe_581 --locked --profile test-fast --no-run`. Adjust image, target-volume, executable hash, output path and affinity in `post-idle/run.sh`; run it with a fresh output directory, then `uv run --no-project python post-idle/analyze.py /tmp/mvmc-581-post-idle`. The raw archive can be extracted and analyzed independently. Finer-grained per-thread clocks and repeated application-level attribution are still required to quantify sub-tick idle cost.

A follow-up version also reads the nanosecond CPU runtime from `/proc/self/task/TID/schedstat` for every observed thread except the calling test thread. It takes a baseline after the initial broadcast has returned, then samples the same requested 10/50/100/500/1000 ms boundaries. Threads present at the baseline are compared by TID; it records added/missing TIDs, and no additions or disappearances were observed in these runs. The five-run medians of summed other-thread runtime at one second were 0 ns for all four 1/4-worker, inside/outside conditions. The maximum across all 100 time-point observations was 43086 ns (4 workers, inside). This is the kernel's CPU runtime accounting, not elapsed sleep or spin duration; it includes any noncaller test-harness activity and depends on `/proc` availability. The baseline's filesystem reads occur **after** broadcast completion, so activity between broadcast return and baseline is not measured. No work was submitted during the measured interval, and this is not the repeatedly dispatched or application workload.

An independently selected positive-control test spins the calling thread for 20 ms and asserts that its `schedstat` CPU runtime increases; on this host it recorded 19278793 ns. This validates that the counter can advance under a known busy condition, not that unobserved brief activity could not precede the baseline. The final source is main `1b8ac154e210e51e9c88262622d0b6fd0fdc5604` plus the current test changes; measured test executable SHA-256 `c87baf8b45ac0177b7372940b7b5a59731394eda341f2076342420d5a5f25d2f`. One warmup and five measured processes per condition used the same pinned CPUs and host limitations as above. [post-idle-ns](post-idle-ns/) includes the positive-control log, full results, source-independent analyzer and command. Build as above, run the positive control with `/target/test-fast/deps/rayon_idle_probe_581-<hash> --ignored --exact schedstat_busy_positive_control --nocapture`, then adjust paths and run `post-idle-ns/run.sh` with a fresh output directory. `uv run --no-project python post-idle-ns/analyze.py /tmp/mvmc-581-post-idle-ns-final` recreates the summary. The raw archive can be extracted for reanalysis. Kernel scheduling/accounting configuration and permissions may differ on another Linux host.
