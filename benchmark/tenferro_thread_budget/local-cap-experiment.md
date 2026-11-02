# Rejected local tenferro thread-cap experiment (#583)

This experiment tested a candidate, **not the current default**. When `MVMC_RS_TENFERRO_THREADS` was unset, the candidate limited the inferred CPU Engine count to the process's `available_parallelism()` minus the configured mVMC inner workers other than the caller, with a minimum of one. An explicit `MVMC_RS_TENFERRO_THREADS` remained authoritative. `RAYON_NUM_THREADS` remained an upper bound for the inferred count. This assumes the normal hoisted driver is an inner worker; it is not a global limit for MPI ranks, cgroups, arbitrary library callers, walkers, BLAS or independently created engines. [Candidate patch](local-cap-candidate.patch), SHA-256 `9f334e3d0b328764872c102028071646186aeb1f0c53f76d743f23893de77176`.

The candidate reduced one measured MPI Opt time, but **more than doubled the measured standalone NPara=128, 300-sample tenferro Gram time**. It was therefore rejected rather than changing the default. These short, host-specific samples do not prove a general speedup, regression, or cause for every workload; they demonstrate a concrete tradeoff that the simple static heuristic does not handle.

## Four-rank Opt and PhysCal

Four MPICH 4.2.0 ranks on CPUs 0–15 were individually pinned to 0–3, 4–7, 8–11 and 12–15. The same Hubbard L=32 Opt/PhysCal input and settings as [mpi-placement.md](mpi-placement.md) were used, except the tenferro thread override was unset. Both binaries used `RAYON_NUM_THREADS=4`, BLAS/OMP 1 and either `MVMC_RS_INNER_THREADS=1` or `4`. The candidate affects only the inferred budget. One warmup and three measured processes per version/mode/inner count were run with alternating order; Bash wall time covers MPI startup through output and excludes Docker startup. The `inner=1` case is a negative control for the proposed cap. There was no CPU frequency control.

| Workload | Inner threads/rank | Before samples (seconds) | Candidate samples (seconds) | Before / candidate median |
|---|---:|---|---|---|
| Opt | 1 | 2.879, 2.872, 2.846 | 3.131, 2.852, 2.846 | 2.872 / 2.852 s |
| Opt | 4 | 2.789, 2.737, 2.724 | 2.607, 2.599, 2.599 | 2.737 / 2.599 s |
| PhysCal | 1 | 2.625, 2.652, 2.625 | 2.589, 2.587, 2.586 | 2.625 / 2.587 s |
| PhysCal | 4 | 3.006, 2.972, 3.064 | 3.083, 3.029, 2.713 | 3.006 / 3.029 s |

For each workload and inner-thread count, all six measured outputs had the same filenames and finite formatted values: 4 Opt or 80 PhysCal files, excluding timing files, had maximum absolute difference 0. This does not validate full RNG state or intermediate decisions. PhysCal does not perform an SR optimization solve. [Raw timings and process CPU times](results/local-cap/mpi/raw-logs.txt), [numeric summary](results/local-cap/mpi/summary.json), [input hashes](results/local-cap/mpi/input-sha256.txt).

## Single-process SR stage

A separate standalone `sr_backend_bench` run was pinned to CPUs 0–3 with `RAYON_NUM_THREADS=4`, BLAS/OMP 1, the tenferro override unset, and inner count 1 or 4. It generates deterministic NPara 64/128 matrices and sample counts 300/`2*NPara`; each process reports a median of seven calls after its own warmup. One additional process warmup and three measured processes were alternated between versions for each inner setting. The table is the median of the three printed per-process Gram medians, rounded to 0.01 ms. The `inner=1` row is a negative control; this standalone example does not run the full hoisted sampling workload.

| Inner threads | NPara / samples | Before Gram | Candidate Gram |
|---:|---|---:|---:|
| 1 | 128 / 300 | 0.47 ms | 0.51 ms |
| 1 | 128 / 256 | 0.42 ms | 0.44 ms |
| 4 | 128 / 300 | 0.53 ms | 1.10 ms |
| 4 | 128 / 256 | 0.48 ms | 0.89 ms |

For the measured 128/300 case, the largest printed relative difference to C-order was 1.3e-15 in both versions and the largest printed scaled solve residual was 8.2e-17 in both. These rounded values are not complete numerical or RNG validation. The actual operation order and detailed per-run results are in [raw logs](results/local-cap/sr/raw-logs.txt); [all cases and operation medians](results/local-cap/sr/summary.json) include Gram, assembly, solve and CG product. Timings of other stages and end-to-end MPI runs need not follow the Gram result.

A separate Linux startup diagnostic used `strace -f -e trace=clone,clone3` on a deliberately missing namelist, four allowed CPUs, inner 4, Rayon 4 and the tenferro SR backend. Successful thread-creation calls were 8 before, 4 with the candidate, and 8 with the candidate plus explicit `MVMC_RS_TENFERRO_THREADS=4`. This is a count of all traced startup thread creations, not a per-pool attribution or CPU utilization measurement. [Counts](results/local-cap/thread-creation-counts.txt). In the same container, `available_parallelism()` on CPUs 0–3 returned 4.

Linux x86_64 6.8.0-147, Xeon E5-2699 v3, development-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, Rust 1.99.0, pthread OpenBLAS. The before source was `f5324073b756b2491182fa605ea851ac25e30ee2`; the candidate used the same Rust code plus the linked patch (its checkout also contained benchmark-only changes from #598). Release `mvmc` binary SHA-256: before `bc98b9ddcbafdcdd7f4e7d32b5f06979a07fcf63ba72570e47e69bda027d9a62`, candidate `1ee99701a1d24b2b7ac3e881968b541e8bf096cdce1774407c571e2f104d0826`. `test-fast` SR example SHA-256: before `6e2b4b1cd342d91070a2aa937a12e4ebedaa5c00e435742d884b9a2861395f38`, candidate `82ae23b059badd02cc8e030827c44da35f30f9a6363e1f5f33d5e336bb91e152`. Version-specific paths and host load can affect binaries and timings.

## Reproduce

Use two separate clean checkouts of `f5324073b756b2491182fa605ea851ac25e30ee2` and separate development-container target volumes. Apply `local-cap-candidate.patch` only in the candidate checkout, then in each container build the MPI CLI and SR example:

```sh
git apply --unidiff-zero /path/to/local-cap-candidate.patch  # candidate checkout only
CARGO_TARGET_DIR=/target cargo build -p mvmc-cli --features mpi --release --locked
CARGO_TARGET_DIR=/target cargo build -p mvmc-core --profile test-fast --example sr_backend_bench --locked
```

Prepare the fixed inputs using the commands in [mpi-placement.md](mpi-placement.md). From a checkout containing these scripts, with 16 consecutive CPUs permitted and fresh output directories:

```sh
benchmark/tenferro_thread_budget/mpi-local-cap.sh \
  DEVELOPMENT_IMAGE BEFORE_VOLUME CANDIDATE_VOLUME /tmp/placement-input /tmp/local-cap-mpi
uv run --no-project python benchmark/tenferro_thread_budget/analyze-mpi-local-cap.py \
  /tmp/local-cap-mpi > /tmp/local-cap-mpi-summary.json
benchmark/tenferro_thread_budget/sr-local-cap.sh \
  DEVELOPMENT_IMAGE BEFORE_VOLUME CANDIDATE_VOLUME /tmp/local-cap-sr
uv run --no-project python benchmark/tenferro_thread_budget/analyze-sr-local-cap.py \
  /tmp/local-cap-sr > /tmp/local-cap-sr-summary.json
```

Set `MVMC_BENCH_CPU_BASE` if CPUs 0–15 are unavailable. The MPI wrapper uses MPICH's `PMI_RANK`; use the matching MPI implementation and verify actual masks with the [affinity audit](affinity-audit.md). The SR-only script uses the first four CPUs from the same base. All measurements include process startup and do not control CPU frequency or other host load.
