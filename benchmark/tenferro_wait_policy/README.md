# tenferro team wait-policy experiment (#583)

`MVMC_RS_TENFERRO_SPIN_POLLS` is an opt-in per-pool wait-spin count. Unset preserves threadteam's default; `0` skips spin polls and yields before sleeping. This does not reduce the thread budget. tenferro-cpu 0.8.1 uses threadteam 0.2.0; its x86-64 default is 300,000 polls, but the library can cap the effective count under oversubscription/descheduling. This experiment does not observe that effective count. The value is not a time duration.

## Four ranks × four inner threads, L=32

Linux x86_64 6.8.0-147, Xeon E5-2699 v3, CPUs 0–15, MPICH 4.2.0 in development-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, Rust 1.99.0, source `18376d0a…`, optimized release binary SHA-256 `bb01e238d344e0a6e9937a4405331ab09f816f2f06b862b82d7a8fd281dbb2a4`. Four MPI ranks each use `MVMC_RS_INNER_THREADS=4`, `RAYON_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and tenferro SR; tenferro threads are 1 or 4 per rank. Opt uses the repository Hubbard L=32 input with 20 SR steps and window 1; PhysCal uses its fixed checkpoint and 20 windows. Both use 300 samples, warmup 10, interval 1 and seed 1. Input hashes and all raw timings are in [results](results/).

The same binary and input were used for all four conditions; each received one warmup then three timed runs. Within each repetition the condition order was reversed relative to the preceding repetition. `bash time` includes MPI launch, input/initialization, calculation and output, but excludes Docker startup. No frequency control was applied. The serial Engine never creates a team, so the 1-thread rows are a negative control for timing noise rather than a spin-policy experiment.

| Workload | tenferro threads/rank | Policy | Wall samples (s) | Median (s) |
|---|---:|---|---|---:|
| Opt | 1 | default | 2.668, 2.691, 2.677 | 2.677 |
| Opt | 1 | 0 polls | 2.710, 2.692, 2.690 | 2.692 |
| Opt | 4 | default | 2.686, 2.716, 2.774 | 2.716 |
| Opt | 4 | 0 polls | 2.578, 2.763, 2.641 | 2.641 |
| PhysCal | 1 | default | 3.079, 2.853, 3.289 | 3.079 |
| PhysCal | 1 | 0 polls | 3.383, 3.061, 3.376 | 3.376 |
| PhysCal | 4 | default | 3.274, 3.129, 3.040 | 3.129 |
| PhysCal | 4 | 0 polls | 3.241, 3.252, 3.047 | 3.241 |

The small and mixed differences, including movement in the serial negative control, do not justify changing the default. Formatted finite numeric outputs from the twelve measured runs per workload were compared against the first default/1-thread result: four Opt files and 80 PhysCal files, excluding clock-time files, had maximum absolute difference 0. This is not a comparison of complete RNG state or intermediate decisions. Separately, with tenferro threads=4 and zero polls, the existing 2/4-walker C-reference tests passed, including their full per-walker SFMT-state and saved-configuration checks. [Full timing output](results/alternating-logs.txt), [summary](results/alternating-summary.json).

## Post-Gram idle CPU

A separate Linux `test-fast` example (`tenferro_wait_probe`) performed one n=256/samples=300 Gram with tenferro threads=4, then slept. The process was pinned to CPUs 0–3; `idle-probe.py` read aggregate `/proc/PID/stat` CPU ticks for two seconds after its `READY` line. One warmup plus five alternating repetitions per policy. Host tick rate was 100 Hz; process start, Gram and any spin before the READY line is received are excluded.

Default CPU seconds after READY: 0.00, 0.00, 0.01, 0.00, 0.01 (median 0.00). Zero-poll: 0.00 in all five (median 0.00). Tick resolution and observation delay can hide short spin episodes; this is not proof of no spinning or a wait-policy speedup. [Raw samples](results/idle-samples.json).

## Reproduce

Use the Linux development container and an otherwise idle host. Build the MPI-enabled release binary in its dedicated target volume mounted at `/target`. The first script creates a fixed input copy; its extra non-alternating timings are not the table above. Use fresh output paths, a GPU-free CPU workload, and adjust the hard-coded CPU set if 0–15 are unavailable:

```sh
# Inside the container, with /target mounted:
CARGO_TARGET_DIR=/target cargo build -p mvmc-cli --features mpi --release --locked
# On the host; the scripts require Docker and the matching image/target volume:
MVMC_BENCH_CPUSET=0-15 MVMC_BENCH_INNER_THREADS=4 \
  benchmark/tenferro_wait_policy/mpi-l32.sh DEVELOPMENT_IMAGE TARGET_VOLUME /tmp/wait-input
benchmark/tenferro_wait_policy/mpi-l32-alternating.sh \
  DEVELOPMENT_IMAGE TARGET_VOLUME /tmp/wait-input/inputs /tmp/wait-alternating
uv run --no-project python benchmark/tenferro_wait_policy/analyze-alternating.py /tmp/wait-alternating > /tmp/wait-summary.json
cargo build -p mvmc-core --profile test-fast --example tenferro_wait_probe
uv run --no-project python benchmark/tenferro_wait_policy/idle-probe.py \
  target/test-fast/examples/tenferro_wait_probe > /tmp/wait-idle.json
```

The input-creation script also executes a warmup and three non-alternating measurements for each condition; those preliminary values are not used in the table. The `/proc` probe requires Linux and `taskset`; adjust its CPU set as needed.

## Four ranks on four CPUs

The same binary, inputs, warmups and alternating conditions were repeated on CPUs 0–3 with four ranks and `MVMC_RS_INNER_THREADS=1` per rank. Each rank still requests 1 or 4 tenferro threads, so the 4-thread cases can oversubscribe the allocation. Scope, BLAS/Rayon settings and three-repetition limitations are unchanged.

| Workload | tenferro threads/rank | Policy | Wall samples (s) | Median (s) |
|---|---:|---|---|---:|
| Opt | 1 | default | 2.821, 2.768, 2.770 | 2.770 |
| Opt | 1 | 0 polls | 3.386, 2.763, 2.762 | 2.763 |
| Opt | 4 | default | 3.518, 3.614, 3.339 | 3.518 |
| Opt | 4 | 0 polls | 2.856, 2.790, 2.784 | 2.790 |
| PhysCal | 1 | default | 2.579, 2.731, 2.574 | 2.579 |
| PhysCal | 1 | 0 polls | 2.688, 2.636, 2.584 | 2.636 |
| PhysCal | 4 | default | 2.604, 2.633, 2.636 | 2.633 |
| PhysCal | 4 | 0 polls | 2.603, 2.759, 2.574 | 2.603 |

Spin 0 reduced Opt wall time in this particular oversubscribed case; PhysCal was similar. With only three repetitions and a short negative-control outlier, these results do not establish a universal policy or explain the first-order cause. Four Opt and 80 PhysCal files' finite formatted numeric values had maximum absolute difference 0 across all measured conditions, excluding time files. [Raw timings, input hashes and output summary](results/4cpu/).

```sh
MVMC_BENCH_CPUSET=0-3 MVMC_BENCH_INNER_THREADS=1 \
  benchmark/tenferro_wait_policy/mpi-l32-alternating.sh \
  DEVELOPMENT_IMAGE TARGET_VOLUME /tmp/wait-input/inputs /tmp/wait-4cpu
uv run --no-project python benchmark/tenferro_wait_policy/analyze-alternating.py /tmp/wait-4cpu > /tmp/wait-4cpu-summary.json
```

## Live MPI protocol with zero polls

With `MVMC_RS_TENFERRO_SPIN_POLLS=0`, the existing ignored `negative_clock_and_asymmetric_runner_failures` protocol gate passed on MPICH 4.2.0 for 2 and 4 actual ranks, with tenferro threads 1 and 4 per rank, Rayon=4, mVMC inner threads=1 and BLAS/OMP=1. Each gate has a 120-second timeout and validates collective failure agreement; it is not a performance benchmark. [Raw logs](results/mpi-protocol/).

```sh
# Build inside the matching development container with its target volume:
cargo test -p mvmc-core --features mpi --lib --profile test-fast --no-run
for ranks in 2 4; do
  for threads in 1 4; do
    MVMC_RS_SR_BACKEND=tenferro MVMC_RS_INNER_THREADS=1 RAYON_NUM_THREADS=4 \
      MVMC_RS_TENFERRO_THREADS=$threads MVMC_RS_TENFERRO_SPIN_POLLS=0 \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      timeout --kill-after=5s 120s mpirun -n "$ranks" /path/to/mvmc_core_test_binary \
      --ignored --exact run::mpi_runtime_tests::negative_clock_and_asymmetric_runner_failures --nocapture
  done
done
```
