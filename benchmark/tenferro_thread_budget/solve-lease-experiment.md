# Solve-scoped CPU team and SR-CG (#583)

The per-stage-handle candidate in [the prior experiment](scoped-engine-experiment.md) recreated the CPU Engine for every SR-CG local product and was 4.1 times slower on one L32 case. This candidate retains the CPU team for the entire solve, with a reference count shared by overlapping solves and stage handles. It releases the shared stage mutex between local products, before walker/MPI collectives, so another walker can reach the same collective. The last lease drops the parallel Engine. Directly constructed stages and CUDA do not use this mechanism. The optional mode does not coordinate all pools or MPI ranks and does not complete #583/#585.

One binary compared `MVMC_RS_TENFERRO_TEAM_LIFETIME=held` and `stage` on four MPICH ranks pinned to disjoint four-CPU masks. The SR-CG input and command are the same as in the prior experiment: L32, 20 Opt steps, 300 samples, warmup 10, interval 1, `NSRCG=1`, seed 1, inner 4, Rayon 4, inferred tenferro 4 and single-thread BLAS. One warmup per condition preceded three measured runs, with order alternating. The timer includes the MPI launcher, initialization, calculation and output, excluding Docker startup.

| Mode | Wall samples (seconds) | Median (seconds) |
|---|---|---:|
| Held | 2.965, 2.999, 2.977 | 2.977 |
| Stage/solve scoped | 3.084, 2.978, 2.945 | 2.978 |

The six measured non-timing output directories were byte-identical using `diff -qr --exclude='*time*'`; full RNG states and intermediate decisions were not recorded in this timing experiment. This short, shared-host experiment indicates that the previous severe regression was not reproduced on this input; it does not establish a speedup, global resource budget, or performance on other models, thread counts and host loads. User/system times and input hashes are in [the raw logs](results/solve-lease-cg/).

Host: Linux x86_64 6.8.0-147, Xeon E5-2699 v3, CPUs 0–15 with four distinct physical cores per rank, development-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, MPICH 4.2.0, Rust 1.99.0 and pthread OpenBLAS. The measured release MPI binary SHA-256 was `3275c4ac02165759e21abb74e09791cfffcae0f599c44cfaaea4267ed4fa3bc9`. After this measurement, the handle was changed to own its team lease independently of the mutable stage; the updated candidate is measured separately below.

## Reproduce

Build the reviewed source with `CARGO_TARGET_DIR=/target cargo build -p mvmc-cli --features mpi --release --locked` in the matching devcontainer and dedicated target volume. Prepare `/tmp/placement-input/opt` as in [the placement report](mpi-placement.md), then from this repository with a fresh output directory:

```sh
benchmark/tenferro_thread_budget/scoped-engine-cg.sh \
  DEVELOPMENT_IMAGE TARGET_VOLUME /tmp/placement-input/opt /tmp/solve-lease-cg-results
```

The script copies inputs, changes `NSRCG` to 1, records hashes and alternates the two modes. `MVMC_BENCH_CPU_BASE` changes the first of 16 allowed CPUs. Verify CPU topology, hashes and non-timing outputs independently.

## Direct SR and PhysCal controls

A separate sequence on the same host and pre-handle-fix candidate binary used direct SR and PhysCal at inner 1 and 4 threads/rank, retaining the other settings and timing boundary above. Each condition had one warmup and three measured runs. The maximum absolute difference between finite formatted numerical values was 0 across the six measured runs within each workload/thread condition; complete RNG states were not compared.

| Workload | Inner threads/rank | Held samples (s) | Stage samples (s) | Median held / stage (s) |
|---|---:|---|---|---|
| Opt, direct SR | 1 | 2.842, 2.890, 3.086 | 2.942, 2.951, 2.969 | 2.890 / 2.951 |
| Opt, direct SR | 4 | 2.778, 2.761, 2.709 | 2.699, 2.803, 2.730 | 2.761 / 2.730 |
| PhysCal | 1 | 2.636, 2.691, 2.595 | 2.733, 2.626, 2.602 | 2.636 / 2.626 |
| PhysCal | 4 | 2.806, 3.218, 2.791 | 3.036, 2.790, 2.758 | 2.806 / 2.790 |

[Raw logs, input hashes and numeric summary](results/solve-lease-mpi/) retain wall, user and system times. These short samples vary and do not establish a speedup. To reproduce, prepare the same Opt and PhysCal inputs and run the following with the same candidate volume in both binary positions:

```sh
benchmark/tenferro_thread_budget/scoped-engine-mpi.sh \
  DEVELOPMENT_IMAGE TARGET_VOLUME TARGET_VOLUME /tmp/placement-input /tmp/solve-lease-mpi-results
uv run --no-project python benchmark/tenferro_thread_budget/analyze-mpi-local-cap.py \
  /tmp/solve-lease-mpi-results > /tmp/solve-lease-mpi-summary.json
```

## Handle-owned lease follow-up

The first candidate released a lease by consulting the stage object at handle drop. Since a caller can replace that object through the public mutable stage reference, this could release the wrong team or leak the original lease. The revised handle owns the exact team lease acquired, independently of later stage replacement. A focused test replaces the stage while a handle is active and checks that the original team is released.

A separate same-binary SR-CG sequence repeated the four-rank L32 setup above after the change, with a fresh warmup and three alternating measured runs per mode:

| Mode | Wall samples (seconds) | Median (seconds) |
|---|---|---:|
| Held | 3.075, 2.977, 2.955 | 2.977 |
| Stage/solve scoped | 3.075, 3.038, 3.038 | 3.038 |

All six non-timing output directories were byte-identical. Full RNG states and intermediate decisions were not recorded. The binary SHA-256 is `15061917d879f99688d59e09adae5832bec9ba1eedb9fef0a4795fce6322bfb1`. This is still one short, shared-host workload and does not establish general performance. [Logs and input hashes](results/solve-lease-guard-cg/) are preserved; use the same reproduction command above with a build containing the handle-owned lease change.

The handle-owned candidate also repeated the direct SR and PhysCal controls as a separate sequence, with the same setup, one warmup and three alternating runs per condition:

| Workload | Inner threads/rank | Held samples (s) | Stage samples (s) | Median held / stage (s) |
|---|---:|---|---|---|
| Opt, direct SR | 1 | 3.162, 2.936, 2.822 | 2.954, 2.898, 3.488 | 2.936 / 2.954 |
| Opt, direct SR | 4 | 2.781, 2.666, 2.668 | 2.777, 2.813, 2.748 | 2.668 / 2.777 |
| PhysCal | 1 | 2.639, 2.648, 2.661 | 2.720, 2.708, 2.656 | 2.648 / 2.708 |
| PhysCal | 4 | 2.991, 3.210, 2.699 | 3.197, 3.029, 3.051 | 2.991 / 3.051 |

Within each workload/thread condition, finite formatted numeric values had maximum absolute difference 0 across the six measured runs; full RNG states were not compared. The observations vary and do not establish a general speedup. The [raw logs, input hashes and summary](results/solve-lease-guard-mpi/) and the preceding reproduction command apply to this revised binary.
