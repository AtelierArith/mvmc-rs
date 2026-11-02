# CpuEngine lock and dispatch probe (#583)

This synthetic harness uses `tenferro-cpu 0.8.1`'s actual `CpuEngine::new` and `exec().token().for_ranges`, rather than a directly held Pool team. It is not an mVMC workload or a pure worker wake-latency measurement. Source `8efa855a`; executable hash is in `binary-sha256.txt`.

A four-participant Engine is shared under a mutex among 1/4 caller threads. Each caller performs 1000 fixed 64-element integer sums, with requested gaps 0/60 µs before locking. Requested sleep duration is not the actual sleep duration. Every run asserts the exact sum. Internal timers accumulate lock acquisition duration (including the lock call and scheduler delays) and execution-bundle/dispatch duration (including atomic work, scheduling and timer overhead). Totals across callers can overlap: dividing them by call count gives an aggregate per-call average, not critical-path or utilization accounting. No participant identity or Engine wait-counter instrumentation is included.

Each condition received one excluded warmup and three measured processes, alternating default/zero-spin order. CPUs 0–3, Linux x86_64, Xeon E5-2699 v3, Rust 1.99.0, dev-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`. There are no BLAS calls. No concurrent local build was started during measurement; host load was not sampled, frequency was uncontrolled and the shared host was not isolated. Whole-process times include Engine construction, thread start/join and output; Docker startup is excluded. The short runs and millisecond whole-process timing resolution limit interpretation.

| Callers | Gap µs | Spin | Wall median s | User CPU median s | Lock average median µs/call | Dispatch average median µs/call |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 0 | default | 0.007 | 0.014 | 0.073 | 2.998 |
| 1 | 0 | 0 | 0.008 | 0.010 | 0.091 | 3.664 |
| 1 | 60 | default | 0.121 | 0.353 | 0.050 | 3.033 |
| 1 | 60 | 0 | 0.137 | 0.053 | 0.055 | 17.666 |
| 4 | 0 | default | 0.022 | 0.065 | 6.731 | 4.185 |
| 4 | 0 | 0 | 0.036 | 0.055 | 15.728 | 5.907 |
| 4 | 60 | default | 0.124 | 0.377 | 0.045 | 2.967 |
| 4 | 60 | 0 | 0.134 | 0.093 | 12.314 | 8.656 |

All 32 processes completed with the expected integer result. The requested-60-µs cases show lower user CPU with zero spin and longer median elapsed/dispatch times; the four-caller cases also show lock contention. These observations do not separate actual spin duration, pure wake latency, provider work, OS scheduling or instrumentation. They do not justify changing mVMC's default wait policy, and the application-level waiting/ownership criteria remain open. Compare with the separately labelled direct-Pool counter probe and matched application runs.

## Reproduce

Build `cargo build -p mvmc-core --example bench_engine_wait_583 --release --locked` at the source above in the Linux development container using a separate target volume. From the repository root:

```sh
benchmark/tenferro_thread_budget/engine-wait-20261102/run.sh DEVELOPMENT_IMAGE TARGET_VOLUME FRESH_OUTPUT_DIRECTORY
uv run --no-project python benchmark/tenferro_thread_budget/engine-wait-20261102/analyze.py FRESH_OUTPUT_DIRECTORY > summary.json
```

The analyzer checks each sum and call count, excludes warmups, and retains all per-process timings (including system CPU) and per-call averages. `raw.tar.gz` can be extracted and reanalysed without rerunning. The example passed targeted type-checking and Clippy with warnings denied; its integer assertion ran in every measurement. This adds no production execution path.
