# Multi-walker Opt and PhysCal resource experiment (#583)

This experiment runs the Rust in-process multi-walker APIs on the same Hubbard L32 inputs used by [mpi-placement.md](../mpi-placement.md). It compares one and four walkers, and tenferro CPU Engine budgets of one and four threads within each walker count. A four-walker run performs four independent chains and is not the same total work as a one-walker run. Opt uses tenferro SR and 20 optimization steps, with one final averaging step and no initial parameter file. PhysCal uses the supplied fixed parameters and 20 measurement windows; it does not run SR, so the tenferro thread setting is a negative control there. Both use 300 samples per window, warmup 10, interval 1, base seed 1 and C-compatible group-index seed offsets. Inner-kernel workers are fixed at one, Rayon capacity at four and OpenBLAS/OMP at one; this does not mean the process has only one thread.

In a preliminary, untimed check, explicitly selecting tenferro for the measurement Pfaffian returned `unsupported: tenferro 0.8 has no skew-symmetric factorization or Pfaffian` on this path. The measurements below leave `MVMC_RS_MEASURE_PF_BACKEND` unset and use the default C-order measurement backend. This is not a tenferro PhysCal performance result. The experimental shared-team path and its counters are disabled. Opt's Engine is shared through the existing stage backend; the setting is per Engine, not a process-wide cap on walker and other threads.

Each condition used the same Rust executable, one warmup followed by three measured process runs, with Engine settings alternated. Linux x86_64 6.8.0-147, Xeon E5-2699 v3, four physical cores CPUs 0–3, dev-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, Rust 1.99.0 and pthread OpenBLAS. No CPU-frequency control, load recording or host isolation was applied. The executable SHA-256 was `41169114f94d55414e013cb6c350ed2b9cb3f41cb634143e1070cc88f30e6e81`; source was `521df827` plus [`bench_walkers_583.rs`](../../../crates/mvmc-core/examples/bench_walkers_583.rs). Shell elapsed time covers process startup, input and initialization, computation, all output, and the harness's full RNG/configuration/decision record printing. Docker startup is excluded. It is not a kernel-only time and recording overhead is not separately measured.

| Workload | Walkers | Engine threads | Wall median | User CPU median | System CPU median |
|---|---:|---:|---:|---:|---:|
| Opt | 1 | 1 | 2.373 s | 2.331 s | 0.041 s |
| Opt | 1 | 4 | 2.369 s | 2.834 s | 0.045 s |
| Opt | 4 | 1 | 3.175 s | 11.017 s | 0.132 s |
| Opt | 4 | 4 | 3.510 s | 11.723 s | 0.162 s |
| PhysCal | 1 | 1 | 2.165 s | 2.151 s | 0.015 s |
| PhysCal | 1 | 4 | 2.113 s | 2.098 s | 0.015 s |
| PhysCal | 4 | 1 | 2.602 s | 9.889 s | 0.062 s |
| PhysCal | 4 | 4 | 2.584 s | 9.853 s | 0.053 s |

Each CPU median is calculated independently and may not belong to the median-wall run. Four-walker Opt wall samples were 3.825, 3.175, 2.980 s at one Engine thread and 4.443, 3.510, 3.321 s at four; the small three-sample experiment does not isolate a cause or establish a general optimal policy. PhysCal is a control, not evidence that tenferro parallelism accelerates it.

Within each workload/walker count, the uncompressed records were compared exactly across the six measured runs: each walker's complete SFMT state, draw count and integer electron-configuration buffers, plus PhysCal's recorded acceptance booleans and exact integer representations of draws, matched. This is a same-implementation fixed-case comparison, not a full candidate trace or independent C oracle. The root Opt final-energy value and each walker's final PhysCal `<H>` and `<H²>` components were finite and matched to the 17-digit printed precision, with maximum observed absolute difference zero. Opt non-root `RunSummary.final_energy_per_site` is a documented-in-code NaN sentinel because only the output root reads the summary file; the analyzer verifies this sentinel and excludes it from numerical comparisons. Green-function arrays and unprinted floating-point intermediates were not compared. Existing independent multi-walker C and RNG regression tests are complementary.

A **separate observation**, excluded from timings, sampled Linux `/proc/PID/task/TID/stat` approximately every 0.1 s during the same four-walker Opt case. With Engine budget 1 it observed 24 samples, a maximum of 5 OS threads, four distinct threads with CPU ticks and at most four with increased ticks in one interval; budget 4 gave 26 samples, maximum 9 threads, seven distinct CPU-active thread IDs and at most seven with increased ticks in one interval. The counter resolution was 100 Hz. An interval can include different threads at different instants; these numbers do not prove simultaneous use of seven cores, identify thread roles or measure idle spinning. The observed runs' full RNG/configuration records and root final energy matched the corresponding timed runs; observation overhead affects the run.

## Reproduce

Prepare `/tmp/mvmc-583-default/inputs/{opt,phys}` as described in [mpi-placement.md](../mpi-placement.md), including the PhysCal 20-window change and fixed parameter file. Verify `input-sha256.txt`; adjust fixed paths and CPU affinity for the host. In a Linux development container with a checkout-specific target volume:

```sh
CARGO_TARGET_DIR=/target cargo build -p mvmc-core --example bench_walkers_583 --release --locked
```

On the host, adjust image, input and target-volume names in `run.sh` and ensure `/tmp/mvmc-583-walker-matched` is fresh:

```sh
bash benchmark/tenferro_thread_budget/walkers-20261102/run.sh
uv run --no-project python benchmark/tenferro_thread_budget/walkers-20261102/analyze.py /tmp/mvmc-583-walker-matched
uv run --no-project python benchmark/tenferro_thread_budget/walkers-20261102/compact.py /tmp/mvmc-583-walker-matched /tmp/mvmc-583-walker-compact
uv run --no-project python benchmark/tenferro_thread_budget/walkers-20261102/analyze.py /tmp/mvmc-583-walker-compact
```

The compact archive keeps all timing/stderr records and numeric lines, replacing each large discrete line with its SHA-256; its analyzer verifies hashes rather than redoing a byte-for-byte comparison. The reported exact comparison was performed before compaction. Extract `raw-compact.tar.gz` to an empty directory and run the analyzer to reconstruct `summary.json`. The two observation scripts and compressed thread samples and full observation logs are provided; `observe.py IMAGE TARGET INPUTS FRESH_OUTPUT ENGINE_THREADS` runs one diagnostic and `observe-check.py TIMED_ROOT OBSERVATION_PREFIX` compares its records. Adjust its fixed CPU mask for the host.

The matching Linux dev-container tests `cargo nextest run -p mvmc-core --test multiwalker_sr_435 --test multichain --cargo-profile test-fast --locked --no-fail-fast --retries 0` passed 16 tests with `MVMC_RS_SR_BACKEND=tenferro`, `MVMC_RS_TENFERRO_THREADS=4`, `RAYON_NUM_THREADS=4`, BLAS/OMP one; they include independent native-C fixtures, walker RNG ownership, collective behavior and a deadlock gate. This experiment does not resolve global budget enforcement, CPU wake/idle attribution or the broader #583 acceptance criteria.

## Engine team lifetime with four walkers

A subsequent matched four-walker Opt experiment used the same binary, inputs, four CPU cores, four tenferro threads, one inner worker and other settings above. It alternated the default `MVMC_RS_TENFERRO_TEAM_LIFETIME=held` against the experimental `stage` lifetime, with one warmup and three measured runs per condition. The stage lifetime releases the team when no stage/solve lease remains; it does not impose a process-wide budget.

| Lifetime | Wall samples | Wall median | User CPU median | System CPU median |
|---|---|---:|---:|---:|
| held | 3.254, 3.210, 3.267 s | 3.254 s | 11.624 s | 0.159 s |
| stage | 3.209, 3.240, 3.223 s | 3.223 s | 11.974 s | 0.174 s |

The difference is small and does not establish a speedup or an optimal lifetime. Full per-walker RNG, draw-count and integer configuration records matched byte-for-byte across the six measured runs; root final energy matched to printed precision. Non-root final-energy NaN sentinels are handled as above. No full proposal/decision trace or independent C oracle is implied by these comparisons.

Separate `/proc` observation runs, excluded from the timing table, sampled at roughly 0.1 s with 100 Hz CPU ticks. Both had 26 samples and at most nine existing OS threads. Held observed seven distinct CPU-active thread IDs and at most seven with increased ticks in one interval; stage observed nine distinct IDs and at most eight per interval. IDs can span thread creation and exit and activity within an interval need not be simultaneous. These samples do not measure waiting or establish why the counts differ. Each observed run's full discrete records and root final energy matched its corresponding timed run.

The [lifetime directory](lifetime/) retains per-run timings, compressed discrete SHA-256 records, input hashes, summaries, observation samples and logs. Adjust fixed paths, image and CPU mask, use fresh output directories, and run `lifetime/run.sh`; analyze with `uv run --no-project python lifetime/analyze.py /tmp/mvmc-583-walker-lifetime`. Use `compact.py` above to recreate the compact archive form. For an observation, `uv run --no-project python lifetime/observe.py IMAGE TARGET INPUTS FRESH_OUTPUT 4 held` (or `stage`) records one run; `lifetime/observe-check.py TIMED_ROOT OBSERVATION_PREFIX` compares its records. This is still an opt-in experiment, not an automatic policy or completion of #583.

## Waiting policy with four walkers

A third four-walker Opt sequence used the same binary, inputs, CPU mask, four-thread Engine and one inner worker. It compared held/stage lifetime and default/zero `MVMC_RS_TENFERRO_SPIN_POLLS` in all four combinations, with one warmup and three measured runs per combination; condition order was reversed on alternating repetitions. Startup, output and record-printing are included as above.

| Lifetime | Spin polls | Wall samples | Wall median | User CPU median | System CPU median |
|---|---|---|---:|---:|---:|
| held | default | 3.305, 3.313, 3.370 s | 3.313 s | 11.715 s | 0.166 s |
| held | 0 | 2.994, 2.980, 2.973 s | 2.980 s | 10.794 s | 0.170 s |
| stage | default | 3.255, 3.274, 3.363 s | 3.274 s | 12.045 s | 0.166 s |
| stage | 0 | 3.018, 3.032, 3.060 s | 3.032 s | 10.837 s | 0.192 s |

All six default-spin wall samples were slower than all six corresponding zero-spin samples in this particular sequence, and aggregate user CPU medians were lower at zero spin. This does not identify idle spinning as the cause or justify changing the default: the separate MPI4 shared-team experiment in [shared-team-20261102](../shared-team-20261102/) saw zero spin perform worse. Full per-walker SFMT state, draw count and integer configuration records matched byte-for-byte across all 12 measured runs, and root final energy matched at printed precision; this is not a full event trace or C oracle. Non-root Opt energy sentinels are treated as above.

Four separate diagnostic runs sampled `/proc` at approximately 0.1 s and 100 Hz tick resolution. Held/default, held/zero, stage/default and stage/zero respectively observed 26, 23, 26 and 24 samples; each saw at most nine OS threads. The number of distinct CPU-active TIDs was 7, 6, 8 and 8, and the maximum with tick increases within one interval was 7, 6, 8 and 6. TIDs can span creation/exit and interval activity need not be simultaneous; neither CPU spin time nor causation is measured. Each observed run's full discrete record and root energy matched the corresponding timed run.

[spin](spin/) contains compacted timed records, summaries, raw observation samples and full observation logs. Adjust fixed paths, image and affinity and use fresh output directories: `spin/run.sh` runs the sweep; `uv run --no-project python spin/analyze.py /tmp/mvmc-583-walker-spin` analyzes it. `compact.py` above recreates the archived compact form. For a diagnostic, run `uv run --no-project python spin/observe.py IMAGE TARGET INPUTS FRESH_OUTPUT 4 held default` (replace `held`/`default` with `stage`/`0` as needed), and compare outputs with `spin/observe-check.py TIMED_ROOT OBSERVATION_PREFIX`. No global budget or default policy is established by these results.

A separate one-walker Opt sequence used the same four-core mask, four-thread Engine, one inner worker and four lifetime/spin combinations, with one warmup and three alternating measured runs per condition:

| Lifetime | Spin polls | Wall samples | Wall median | User CPU median | System CPU median |
|---|---|---|---:|---:|---:|
| held | default | 2.490, 2.322, 2.451 s | 2.451 s | 2.900 s | 0.050 s |
| held | 0 | 2.371, 2.398, 2.320 s | 2.371 s | 2.362 s | 0.044 s |
| stage | default | 2.372, 2.486, 2.414 s | 2.414 s | 2.596 s | 0.095 s |
| stage | 0 | 2.370, 2.447, 2.382 s | 2.382 s | 2.339 s | 0.103 s |

The samples overlap and do not establish a wall-time advantage; the whole-process user CPU medians were lower with zero spin for these runs, without isolating waiting. The full RNG state, draw count and integer configuration record matched exactly across all 12 measured runs; root final energy matched at printed precision. The one- and four-walker sequences were performed separately and have different total work; do not infer a matched scaling ratio from them. [spin-one](spin-one/) preserves compacted records, samples, input hashes and commands. Adjust the fixed paths and affinity, run `spin-one/run.sh` in a fresh output directory, then `uv run --no-project python spin-one/analyze.py /tmp/mvmc-583-walker-spin-one`; `compact.py` above recreates the archived compact form.

### Exploratory wait-channel and switch sampling

Four additional **single diagnostic runs**, one per lifetime/spin condition in fixed order, sampled each visible Linux thread's state, `/proc/.../wchan`, CPU ticks and voluntary/nonvoluntary context-switch counters at approximately 0.1 s intervals. The counts below sum changes only between two consecutive observations of the same TID; they omit work before first observation, after last observation and from threads never seen. Sampling and repeated `docker top` calls perturb the runs. Different durations and thread lifetimes make totals unsuitable for direct cost attribution or a controlled statistical comparison. A `wchan` value of `0` is recorded as reported by the kernel and is not classified as running or spinning.

| Lifetime / spin | Thread observations | `futex_wait_queue` observations | `0` observations | Observed CPU tick deltas | Voluntary switch deltas | Nonvoluntary switch deltas |
|---|---:|---:|---:|---:|---:|---:|
| held / default | 235 | 135 | 100 | 1223 | 1735 | 484 |
| held / 0 | 203 | 119 | 84 | 1107 | 1864 | 776 |
| stage / default | 230 | 127 | 103 | 1263 | 2903 | 9727 |
| stage / 0 | 208 | 120 | 88 | 1100 | 3705 | 12851 |

The observed stage runs had more sampled switch deltas, but this alone does not establish why, measure spin/wake duration, or prove that the lifetime caused the difference. The 100 Hz CPU tick resolution is coarse; thread names alone do not reliably identify walker versus Engine role because threads may inherit names. All four diagnostic runs' full discrete records and root energy matched the corresponding timed runs. Compressed per-thread samples, output logs, summary and scripts are in [spin-detail](spin-detail/). To reproduce a condition with fresh output, run `uv run --no-project python spin-detail/observe.py IMAGE TARGET INPUTS FRESH_OUTPUT held default` (substitute `stage` and/or `0`); summarize saved JSON with `uv run --no-project python spin-detail/summarize.py FILE...`, and compare uncompressed output records using `spin-detail/observe-check.py TIMED_ROOT OBSERVATION_PREFIX`. Fixed paths and CPU affinity in the scripts must be adapted. Further tracing or controlled repeated observation is needed before drawing a causal conclusion.

### Wait-counter API boundary

The lockfile pins `threadteam 0.2.0` (crates.io checksum `23210e76b7702020e91c6b39b1ea83e3e207153d866b261f354eed71f5dcefa3`) and `tenferro-cpu 0.8.1` (checksum `63b6a5f9d1efd8bb0417d53585580d15b9f621674df42885702f4ce064d253d2`). In that threadteam source, `Pool::wait_counters()` exposes cumulative join sleeps, barrier sleeps and member parks, and on x86-64 its default `spin_polls` is 300000; polls are not a fixed time. These counters are counts, not sleep or spin durations. `CpuEngine::new` takes ownership of its pool and can reserve it as a team, while the inspected tenferro-cpu 0.8.1 public `CpuEngine` API exposes `has_team` and `exec` but no pool or wait-counter accessor. Therefore the present integration cannot read those counters directly from its Engine through the inspected public API; the `/proc` evidence above is a separate approximation, not those counters. Exposing or otherwise measuring Engine wait behavior remains follow-up work. The supplied spin values are explicitly recorded; no inference about actual sleep time is made from the default poll count.

## Standalone threadteam wait counters

The preceding `/proc` work cannot access the Engine-owned pool's direct counters. As a **separate synthetic diagnostic**, [`bench_team_wait_583.rs`](../../../crates/mvmc-core/examples/bench_team_wait_583.rs) creates a `threadteam 0.2.0` Pool directly, reserves one four-participant Team, and lets one or four caller threads serialize access through a mutex. Each caller makes 5,000 calls, each dividing 64 simple integer additions into fixed ranges; the final exact integer sum is checked. The caller optionally sleeps for a *requested* 60 µs between calls. The loop and mutex approximate repeated shared-team dispatch, but do not run `CpuEngine`, mVMC sampling, SR, walkers or MPI; integer work may be optimized and no direct participant count was captured. Do not infer application performance or Engine wait counts from it.

On the same four cores, with one warmup and three alternating measurements for each spin setting, median results were:

| Callers | Requested gap | Spin polls | Wall | User CPU | System CPU | Join sleeps | Member parks |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 0 µs | default | 0.018 s | 0.054 s | 0.007 s | 0 | 2 |
| 1 | 0 µs | 0 | 0.021 s | 0.040 s | 0.034 s | 2 | 13 |
| 1 | 60 µs | default | 0.578 s | 1.752 s | 0.007 s | 0 | 2 |
| 1 | 60 µs | 0 | 0.682 s | 0.305 s | 0.313 s | 3 | 15001 |
| 4 | 0 µs | default | 0.091 s | 0.279 s | 0.015 s | 6 | 10 |
| 4 | 0 µs | 0 | 0.105 s | 0.198 s | 0.167 s | 21 | 29 |
| 4 | 60 µs | default | 0.640 s | 1.942 s | 0.015 s | 9 | 4 |
| 4 | 60 µs | 0 | 0.653 s | 0.455 s | 0.609 s | 178 | 22295 |

The library's counter medians include team startup and acquisition; barrier-sleep medians were zero in all cases. They count events, not their duration or actual spin time. Wall/user/system medians are independent, include process startup, and exclude Docker startup; the 0-gap runs are especially short relative to startup and 1 ms printed timing resolution. Sleep durations can exceed the requested gap. The pattern demonstrates the synthetic pool's park/CPU tradeoff for these settings, not the cause of the application results above. No CPU-frequency control, host-load sampling or host isolation was applied.

Build in the same dev container and dedicated target volume with `cargo build -p mvmc-core --example bench_team_wait_583 --release --locked`; executable SHA-256 was `ddc43c1f6c8162fed107181924a88c8b36180fa773c4d191041b50e14c332af2`, built from `521df827` plus the two benchmark example sources in this PR. Adjust fixed image, volume, output path and CPU mask in [team-wait/run.sh](team-wait/run.sh), use a fresh output directory, and run it; `uv run --no-project python team-wait/analyze.py /tmp/mvmc-583-team-wait` reproduces the summary. [team-wait/raw.tar.gz](team-wait/raw.tar.gz) contains all stdout, stderr, timing and configuration records; extract to an empty directory to analyze without rerunning. The Rust example passed targeted Clippy. Engine-level counters and repeated application-level waiting measurements remain unresolved.
