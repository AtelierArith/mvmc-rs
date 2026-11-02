# Rejected inner-pool-aware CPU thread cap (#583)

This is a second candidate, **not the current default**. The earlier [static cap](local-cap-experiment.md) also limited standalone SR outside the mVMC inner pool. This candidate instead checked whether Engine construction occurred on that particular pool; only there, and only when `MVMC_RS_TENFERRO_THREADS` was unset, it capped the inferred count by available parallelism minus the other configured inner workers (minimum one). Outside that pool, or with an explicit override, the previous inference was retained. The normal shared Engine still retained its first chosen count, so this was not an adaptive per-call policy or a global MPI/walker/cgroup budget. [Candidate patch](hoisted-cap-candidate.patch), SHA-256 `988c7a882e002f60298172bd9812d5e0ad58f0796d31469885839bfa5075de33`.

The candidate avoided the standalone *outside-pool* regression, but still more than doubled the measured NPara=128 Gram time *inside* the pool. It also increased the median single-process Opt time in this short run. It was rejected; the CPU Engine default remains unchanged. Three samples on one host do not prove general performance or a specific cause.

## Standalone SR, outside and inside the inner pool

The deterministic `sr_backend_bench` generated NPara 64/128 and sample counts 300/`2*NPara`. A small [example patch](sr-hoist-example.patch), applied identically to both binaries, added `MVMC_BENCH_SR_HOIST=1` to call the benchmark through the mVMC inner pool. `0` or unset runs outside. Each process prints the median of seven calls after a warmup; one additional process warmup and three measured processes per condition/version were alternated. The four-CPU affinity, Rayon 4 and inner 4 were the same for both versions. Values below are medians of the three printed Gram medians, rounded to 0.01 ms.

| Context | NPara / samples | Before Gram | Candidate Gram |
|---|---|---:|---:|
| Outside inner pool | 128 / 300 | 0.52 ms | 0.53 ms |
| Outside inner pool | 128 / 256 | 0.46 ms | 0.45 ms |
| On inner-pool worker | 128 / 300 | 0.50 ms | 1.13 ms |
| On inner-pool worker | 128 / 256 | 0.45 ms | 0.92 ms |

The NPara=128/300 printed maximum relative difference to C-order was 1.3e-15, and maximum scaled solve residual was 8.2e-17, for both versions and contexts. These rounded summaries do not validate all numerical states or RNG behavior. Inner 1 was also measured as a negative control. [Outside logs](results/hoisted-cap/sr-outside/raw-logs.txt), [outside summary](results/hoisted-cap/sr-outside/summary.json), [inside logs](results/hoisted-cap/sr-inside/raw-logs.txt), [inside summary](results/hoisted-cap/sr-inside/summary.json).

## End-to-end L=32 Opt and PhysCal

Both runs use the inputs in [mpi-placement.md](mpi-placement.md), 300 samples, 20 Opt SR steps or 20 fixed-parameter PhysCal windows, the tenferro SR backend, `RAYON_NUM_THREADS=4`, BLAS/OMP 1, no explicit tenferro thread override and default spin policy. Four MPI ranks were pinned to disjoint four-CPU masks on CPUs 0–15; the single-process run was pinned to CPUs 0–3. For each workload/inner count/version there was one warmup plus three measured processes with alternating version order. Bash wall time includes initialization, computation and output; MPI timing also includes the launcher. Docker startup is excluded. There was no CPU-frequency control. `inner=1` is a negative control; its timings still vary between binaries and runs.

| Placement | Workload | Inner threads | Before samples (s) | Candidate samples (s) | Before / candidate median |
|---|---|---:|---|---|---|
| MPI 4 ranks | Opt | 1 | 3.448, 2.878, 3.221 | 3.027, 2.909, 2.858 | 3.221 / 2.909 s |
| MPI 4 ranks | Opt | 4 | 2.714, 2.783, 2.709 | 2.609, 2.620, 2.607 | 2.714 / 2.609 s |
| MPI 4 ranks | PhysCal | 1 | 2.596, 2.628, 2.647 | 2.622, 2.693, 2.617 | 2.628 / 2.622 s |
| MPI 4 ranks | PhysCal | 4 | 3.116, 2.985, 2.972 | 2.793, 3.140, 2.781 | 2.985 / 2.793 s |
| Single process | Opt | 1 | 2.386, 2.377, 2.429 | 2.446, 2.417, 2.394 | 2.386 / 2.417 s |
| Single process | Opt | 4 | 2.239, 2.021, 2.321 | 2.346, 2.326, 2.398 | 2.239 / 2.346 s |
| Single process | PhysCal | 1 | 2.217, 2.158, 2.145 | 2.184, 2.181, 2.190 | 2.158 / 2.184 s |
| Single process | PhysCal | 4 | 2.539, 3.054, 2.409 | 2.503, 2.419, 2.463 | 2.539 / 2.463 s |

Within each workload and inner setting, the six measured output sets had identical filenames and finite formatted numeric values: four Opt or 80 PhysCal files, excluding timing files, had maximum absolute difference 0. This does not compare full RNG states or intermediate decisions. PhysCal does not run an SR optimization solve; startup preflight can still construct an SR Engine. The mixed differences and negative-control variability preclude general conclusions. [MPI logs](results/hoisted-cap/mpi/raw-logs.txt), [MPI summary](results/hoisted-cap/mpi/summary.json), [single-process logs](results/hoisted-cap/single/raw-logs.txt), [single-process summary](results/hoisted-cap/single/summary.json).

A separate Linux `strace` startup probe on four allowed CPUs and inner 4 counted 8 successful thread-creation calls before, 4 with the candidate and 8 with the explicit four-thread override. It counted all startup thread creations, not per-pool ownership or CPU utilization. [Counts](results/hoisted-cap/thread-creation-counts.txt).

To reproduce that diagnostic inside the container, point `MVMC_BIN` at each release binary and use the same four-CPU mask. The missing input deliberately ends the program after backend preflight. The third invocation adds the explicit override:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MVMC_RS_INNER_THREADS=4 RAYON_NUM_THREADS=4 \
  MVMC_RS_SR_BACKEND=tenferro taskset -c 0-3 strace -f -e trace=clone,clone3 \
  -o /tmp/threads.trace "$MVMC_BIN" /no-such-namelist || true
rg -c 'clone\(.* = [0-9]+$' /tmp/threads.trace
# Repeat with MVMC_RS_TENFERRO_THREADS=4 set for the candidate.
```

Linux x86_64 6.8.0-147, Xeon E5-2699 v3, Rust 1.99.0, pthread OpenBLAS, MPICH 4.2.0, development-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`. The before production source was `f5324073b756b2491182fa605ea851ac25e30ee2`; the candidate contained the same Rust production code plus the candidate patch (the checkout also had benchmark-only changes from #598/#599). The example patch was applied to both SR binaries. SHA-256: before/after `mvmc` `bc98b9ddcbafdcdd7f4e7d32b5f06979a07fcf63ba72570e47e69bda027d9a62` / `23be54e09d44b2106a082c6cb1054ab6883bdb0e945706a13e66d4179cabafa1`; before/after `test-fast` SR example `b58662e22919a223bd3523532a0d7604083243eca74b12b7b61455712f333530` / `14360ddf5b6c63d71f5d4ffd5a24364529207b6f7f155b1bf87cad1481502cf9`.

## Reproduce

Use two clean checkouts of `f5324073b756b2491182fa605ea851ac25e30ee2` and separate development-container target volumes. Apply `sr-hoist-example.patch` in both, and `hoisted-cap-candidate.patch` only in the candidate checkout. The patches use zero context and must be applied to that exact source revision. In each container:

```sh
git apply --unidiff-zero /path/to/sr-hoist-example.patch
git apply --unidiff-zero /path/to/hoisted-cap-candidate.patch  # candidate only
CARGO_TARGET_DIR=/target cargo build -p mvmc-cli --features mpi --release --locked
CARGO_TARGET_DIR=/target cargo build -p mvmc-core --profile test-fast --example sr_backend_bench --locked
```

Prepare `/tmp/placement-input` using [mpi-placement.md](mpi-placement.md). From a checkout containing these scripts, with fresh output directories and the matching MPI-enabled development image:

```sh
benchmark/tenferro_thread_budget/mpi-local-cap.sh \
  DEVELOPMENT_IMAGE BEFORE_VOLUME CANDIDATE_VOLUME /tmp/placement-input /tmp/hoisted-mpi
benchmark/tenferro_thread_budget/single-local-cap.sh \
  DEVELOPMENT_IMAGE BEFORE_VOLUME CANDIDATE_VOLUME /tmp/placement-input /tmp/hoisted-single
uv run --no-project python benchmark/tenferro_thread_budget/analyze-mpi-local-cap.py /tmp/hoisted-mpi > /tmp/hoisted-mpi-summary.json
uv run --no-project python benchmark/tenferro_thread_budget/analyze-mpi-local-cap.py /tmp/hoisted-single > /tmp/hoisted-single-summary.json
MVMC_BENCH_SR_HOIST=0 benchmark/tenferro_thread_budget/sr-local-cap.sh \
  DEVELOPMENT_IMAGE BEFORE_VOLUME CANDIDATE_VOLUME /tmp/hoisted-sr-outside
MVMC_BENCH_SR_HOIST=1 benchmark/tenferro_thread_budget/sr-local-cap.sh \
  DEVELOPMENT_IMAGE BEFORE_VOLUME CANDIDATE_VOLUME /tmp/hoisted-sr-inside
uv run --no-project python benchmark/tenferro_thread_budget/analyze-sr-local-cap.py /tmp/hoisted-sr-outside > /tmp/hoisted-sr-outside-summary.json
uv run --no-project python benchmark/tenferro_thread_budget/analyze-sr-local-cap.py /tmp/hoisted-sr-inside > /tmp/hoisted-sr-inside-summary.json
```

Use 16 consecutive available CPUs, or set `MVMC_BENCH_CPU_BASE`; the single-process and SR runs use its first four CPUs. The MPI wrapper assumes MPICH's `PMI_RANK`. The [affinity audit](affinity-audit.md) can verify rank masks; it does not measure cgroup quotas or per-worker placement.
