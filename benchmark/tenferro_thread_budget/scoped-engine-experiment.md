# Rejected stage-scoped CPU Engine prototype (#583)

This experiment built the [candidate patch](scoped-engine-candidate.patch) against repository revision `3019a1b15f2268371aa9ddd5d2ef0870071da368`. It added an experimental `MVMC_RS_TENFERRO_TEAM_LIFETIME=stage` to replace the shared CPU stage's Engine with a serial Engine outside a stage handle, constructing a new parallel Engine at each acquisition. `held` retained the Engine. This setting is **not present in production**: the candidate is rejected. Directly constructed stages and CUDA were outside the candidate's scope. The patch SHA-256 is `86282a9dae08b30e3d5b3dbd4a3f9aea101baefa2a572b68670b9a856ea13b36` and the release MPI binary SHA-256 is `d709b816fe9b367f968429274561ab150f88bb3f19671ec8fe7a6d8476788489`. Both compared modes use that same binary.

MPICH runs used four ranks on CPUs 0–15, rank `r` pinned to CPUs `4r..4r+3`, `RAYON_NUM_THREADS=4`, tenferro's inferred four-thread budget, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and L32 fixed inputs described in [the placement report](mpi-placement.md). One warmup per condition preceded three measured runs, with order alternated. Timings inside Docker include the MPI launcher, initialization, computation and output, not container startup. The shared host had no frequency isolation. Raw wall/user/system times and [input hashes](results/scoped-engine/) are provided. The GPU was not used.

| Workload | Inner threads/rank | Held samples (s) | Stage samples (s) | Median held / stage (s) |
|---|---:|---|---|---|
| Opt, direct SR | 1 | 3.109, 2.961, 3.527 | 2.959, 3.539, 3.028 | 3.109 / 3.028 |
| Opt, direct SR | 4 | 2.699, 2.739, 2.766 | 2.698, 2.795, 2.665 | 2.739 / 2.698 |
| PhysCal | 1 | 2.586, 2.645, 2.602 | 2.627, 2.590, 2.604 | 2.602 / 2.604 |
| PhysCal | 4 | 3.100, 2.710, 2.744 | 2.946, 3.212, 2.978 | 2.744 / 2.978 |
| Opt, SR-CG | 4 | 2.963, 2.968, 2.987 | 12.138, 12.389, 12.053 | 2.968 / 12.138 |

The SR-CG sequence is separate, changes `NSRCG` from `0` to `1`, and otherwise uses the same Opt inputs and settings. The stage-scoped candidate is about 4.1 times slower in that case; the frequent Engine construction is a plausible explanation but was not isolated by profiling. The direct and PhysCal samples show variability and do not establish a speedup. This evidence rules out adopting this candidate without a different lifetime strategy, especially around CG's repeated acquisitions.

For direct SR and PhysCal, finite formatted values in the four Opt or 80 PhysCal non-timing files matched with maximum absolute difference 0 across the six runs within each workload/thread condition. For SR-CG, non-timing output directories were byte-identical across all six runs. These are not comparisons of full RNG states, decisions or intermediate numerical operations. A predecessor prototype passed a two-thread team release/reacquisition and cache reuse unit probe, multiwalker C/reference checks, 2/4-rank MPI failure tests and a Linux workspace test run. The final candidate patch was benchmarked after narrowing it to the shared CPU stage; those predecessor results do not prove the final candidate or other providers, workloads, ranks, operating systems or nesting cases correct.

Host: Linux x86_64 6.8.0-147, Xeon E5-2699 v3 (four distinct physical cores per mask), devcontainer image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, MPICH 4.2.0, Rust 1.99.0, pthread OpenBLAS. Historical input and hardware context is detailed in the placement report.

## Reproduce

Use a clean worktree at the revision above and a separate devcontainer target volume. Apply the candidate patch from a checkout containing this report, then build the MPI binary:

```sh
git apply --unidiff-zero /path/to/report-checkout/benchmark/tenferro_thread_budget/scoped-engine-candidate.patch
CARGO_TARGET_DIR=/target cargo build -p mvmc-cli --features mpi --release --locked
```

Prepare `/tmp/placement-input/{opt,phys}` as described in the placement report. From a checkout containing the scripts, use the **same candidate binary volume** for both comparison arguments and fresh output directories:

```sh
benchmark/tenferro_thread_budget/scoped-engine-mpi.sh \
  DEVELOPMENT_IMAGE CANDIDATE_VOLUME CANDIDATE_VOLUME /tmp/placement-input /tmp/scoped-mpi-results
uv run --no-project python benchmark/tenferro_thread_budget/analyze-mpi-local-cap.py \
  /tmp/scoped-mpi-results > /tmp/scoped-summary.json
benchmark/tenferro_thread_budget/scoped-engine-cg.sh \
  DEVELOPMENT_IMAGE CANDIDATE_VOLUME /tmp/placement-input/opt /tmp/scoped-cg-results
```

Check input and binary hashes, CPU placement, numerical outputs and logs. `MVMC_BENCH_CPU_BASE` selects the first of 16 allowed CPUs. No output directory should already contain `results`.
