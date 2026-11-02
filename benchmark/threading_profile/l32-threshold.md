# L32 inner dispatch work gate (#581)

This is a diagnostic of a single-process Rust CPU path, not a C/Julia or GPU comparison. One release binary at revision `f5324073b756b2491182fa605ea851ac25e30ee2` (SHA-256 `bc98b9ddcbafdcdd7f4e7d32b5f06979a07fcf63ba72570e47e69bda027d9a62`) runs L32 Opt and fixed-parameter PhysCal, with identical inputs within each workload. `MVMC_RS_INNER_THREADS=4`, `RAYON_NUM_THREADS=4`, `MVMC_RS_TENFERRO_THREADS=4`, tenferro SR, and single-thread OpenBLAS are fixed. The two conditions differ only in `MVMC_RS_INNER_MIN_WORK_NS`: unset (current hoisted default) versus `1000000000000`. In separate instrumented diagnostic runs the latter reported no pool-mode inner dispatch calls for either workload; this does not establish that other pools or operations were serial. [Profiles and input hashes](results/l32-threshold/).

Opt runs 20 SR steps with 300 samples and warmup 10; PhysCal loads the fixed parameter file and records 20 windows. Each workload and gate has one warmup followed by four timed repetitions, with gate order alternating. Bash `time` is run inside the container and includes the CLI, initialization, calculation and output, but excludes container startup. The host was shared, frequency was not controlled, and measurements are short. User and system CPU time samples are in the raw logs. No general scaling or causal claim follows.

| Workload | Inner work gate | Measured wall samples (seconds) | Median (seconds) |
|---|---|---|---:|
| Opt | default | 2.090, 2.014, 2.009, 1.963 | 2.0115 |
| Opt | high | 2.415, 2.545, 2.439, 2.428 | 2.4335 |
| PhysCal | default | 2.928, 3.019, 2.362, 3.014 | 2.9710 |
| PhysCal | high | 2.261, 2.380, 2.226, 2.360 | 2.3105 |

For each workload, `diff -qr --exclude='*time*'` reported no differences in the non-timing output files across the eight measured runs. This is byte equality of those formatted files, not a full RNG state, proposal or decision comparison. The observed effect changes direction between workloads, so it does not support a global default change. The default PhysCal samples vary, and the high gate disables multiple dispatch sites rather than isolating Green measurement. Targeted matched measurements and deterministic tests are needed before adjusting a specific site.

Host: Linux x86_64 6.8.0-147, Xeon E5-2699 v3, CPUs 0–3 (four distinct physical cores), devcontainer image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, Rust 1.99.0, MPICH 4.2.0, pthread OpenBLAS. See the [placement report](../tenferro_thread_budget/mpi-placement.md) for input preparation, CPU topology, exact source and compiler context.

## Reproduce

Build the named revision's `mvmc` with MPI support in a dedicated devcontainer target volume as described in the placement report. Prepare its `opt` and `phys` input directories, then use a fresh result directory:

```sh
benchmark/threading_profile/l32-threshold.sh \
  DEVELOPMENT_IMAGE TARGET_VOLUME /tmp/placement-input /tmp/l32-threshold-results
```

Set `MVMC_BENCH_CPU_BASE` to the first of four allowed CPUs if 0–3 are unavailable. The script emits warmup (`0`) and measured (`1`–`4`) logs and output directories; compare non-timing output files separately. The instrumented profiles were collected separately with `MVMC_RS_INNER_PROFILE=1` and are not included in the timing samples.
