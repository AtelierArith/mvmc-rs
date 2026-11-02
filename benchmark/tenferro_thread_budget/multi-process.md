# Multi-walker and MPI thread-budget checks (#585)

These checks supplement the [SR-stage probe](README.md). They do not establish the best default or close #583/#585: no actual worker participation, parallel regions, nested-pool overlap, waiting policy, or representative end-to-end scaling was measured.

## In-process walkers

On Linux x86_64 6.8.0-147, Xeon E5-2699 v3, source `ce83f8e4…` (product code `068ffc45…`), Rust 1.99.0, `test-fast`, CPUs 0–3 via `taskset`, the existing 2- and 4-walker C-reference suites were run sequentially within each repetition. They cover direct and CG cells, including full per-walker SFMT state and saved configurations, and compare C-derived operands/energies under the existing tolerance. Every run passed. `MVMC_RS_SR_BACKEND=tenferro`, `RAYON_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. One warmup followed by three repetitions; 1-thread measurements were taken before 4-thread measurements, no CPU-frequency control.

| tenferro threads | Test suite time, 3 measurements | Median |
|---:|---|---:|
| 1 | 0.336, 0.324, 0.338 s | 0.336 s |
| 4 | 0.332, 0.329, 0.335 s | 0.332 s |

The test runner's total elapsed time was around 1.09 s per run, including Cargo startup and selection; raw user/system time includes those processes. The short suite does not establish actual parallel participation or a meaningful performance difference. [Raw output](results/multi-process/walker.txt).

```sh
for threads in 1 4; do
  MVMC_RS_SR_BACKEND=tenferro MVMC_RS_TENFERRO_THREADS=$threads \
    RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    taskset -c 0-3 /usr/bin/time -v cargo nextest run -p mvmc-core --cargo-profile test-fast \
    -E 'test(walkers_match_native_c_ungrouped_cells_two_ranks) | test(walkers_match_native_c_ungrouped_cells_four_ranks)' \
    --test-threads 1 --no-fail-fast --retries 0
done
```

## Four real MPI ranks

The existing `negative_clock_and_asymmetric_runner_failures` test was run with 4 MPICH 4.2.0 ranks in the Linux development-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, Rust 1.99.0, MPI-enabled test binary SHA-256 `2396d8f892919789878a17c5c955de85da2c0169832502ca3a7bfda9bdfd9ed6`. Docker `--cpuset-cpus=0-3 --shm-size=1g`; tenferro backend, Rayon=4, inner threads=1, BLAS/OMP=1. Every rank passed all three measured runs. One warmup preceded three alternating 1/4-thread repetitions. `bash time` was measured inside the container around `timeout mpirun ...`, including MPI launcher and process startup but excluding Docker startup. It is not an Opt/PhysCal benchmark.

| tenferro threads per rank | Wall time | Median | User CPU time |
|---:|---|---:|---|
| 1 | 0.261, 0.282, 0.268 s | 0.268 s | 0.542, 0.582, 0.565 s |
| 4 | 0.242, 0.250, 0.303 s | 0.250 s | 0.526, 0.529, 0.671 s |

The injected failures exercise collective/error paths on short workloads; the small observed difference does not establish better MPI performance or lack of oversubscription. [Raw output](results/multi-process/mpi-bench.txt).

```sh
# Run inside the matching development container; build the MPI test binary first.
cargo test -p mvmc-core --features mpi --lib --profile test-fast --no-run
for threads in 1 4; do
  MVMC_RS_SR_BACKEND=tenferro MVMC_RS_INNER_THREADS=1 RAYON_NUM_THREADS=4 \
    MVMC_RS_TENFERRO_THREADS=$threads OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    bash -c 'TIMEFORMAT="wall=%R user=%U sys=%S"; time timeout --kill-after=5s 120s mpirun -n 4 /path/to/mvmc_core_test_binary --ignored --exact run::mpi_runtime_tests::negative_clock_and_asymmetric_runner_failures --nocapture'
done
```

Repeat three times after one warmup in alternating order; limit the container to CPUs 0–3. Launcher and allocation policies may differ across systems.

## Four-rank L=32 Opt and PhysCal

On the same Linux Xeon host and MPICH 4.2.0 image, the MPI-enabled Rust `mvmc` release binary at product source `068ffc45…` (plus test-only commit `93d08148…`), SHA-256 `9d915ecb0de0968bbe27c4985747d5b5f4558ab84a8e2337d1d013e08d9f3f90`, ran the repository Hubbard L=32 inputs. The release profile reports `optimized + debuginfo`. Four ranks share CPUs 0–3 (`--cpuset-cpus=0-3`), `MVMC_RS_INNER_THREADS=1`, `RAYON_NUM_THREADS=4`, BLAS/OMP=1, and `MVMC_RS_SR_BACKEND=tenferro`. Every rank independently sets `MVMC_RS_TENFERRO_THREADS` to 1 or 4. Opt runs 20 SR steps with output window 1 and no initial parameter file. PhysCal uses the repository's fixed `zqp_opt.dat` and 20 output windows. Both retain `NVMCSample=300`, warmup 10, interval 1 and seed 1. Inputs and fixed-parameter SHA-256 values: [input hashes](results/mpi-l32/input-sha256.txt).

An earlier 3-step/2-window/40-sample smoke measurement completed in about 0.2 s and was too short to interpret, so the longer case below was run separately. One warmup was followed by three measurements per condition, alternating thread counts. The `bash time` scope includes `mpirun` startup, input parsing, initialization, full calculation and output, excludes Docker startup, and aggregates child CPU time; no frequency pinning was used.

| Workload | tenferro threads per rank | Wall samples (s) | Median wall (s) | User CPU samples (s) |
|---|---:|---|---:|---|
| Opt | 1 | 2.808, 3.388, 2.872 | 2.872 | 10.820, 13.165, 11.067 |
| Opt | 4 | 3.299, 3.366, 4.124 | 3.366 | 12.733, 12.951, 15.721 |
| PhysCal | 1 | 2.594, 2.603, 2.615 | 2.603 | 10.069, 10.122, 10.130 |
| PhysCal | 4 | 2.623, 2.613, 2.626 | 2.623 | 10.176, 10.137, 10.192 |

For these short, oversubscribed Opt runs, 4 tenferro threads per rank were slower; the PhysCal difference is small relative to the evidence available. This does not isolate worker participation or the cause, establish an optimum, or predict the L=128 scaling tracked by #586. For each workload, the output file sets and all finite numeric tokens (excluding wall-clock time files, ignoring `#` headers) were compared across the six measured runs against the first 1-thread run. Four Opt files and 80 PhysCal files had maximum absolute difference 0. This compares formatted output numbers, not bit patterns, RNG states or intermediate acceptance decisions. Independent C/RNG checks above remain necessary. [Raw timings](results/mpi-l32/timing-logs.txt), [output hashes and numerical summary](results/mpi-l32/summary.json).

Build `mvmc` in the development container with its dedicated target volume, then run from a checkout on an otherwise idle Linux machine with CPUs 0–3 available:

```sh
# In the container, with the same target volume mounted at /target:
CARGO_TARGET_DIR=/target cargo build -p mvmc-cli --features mpi --release --locked
# On the host, provide the container image name, volume name and a fresh output directory:
benchmark/tenferro_thread_budget/mpi-l32.sh DEVELOPMENT_IMAGE TARGET_VOLUME /tmp/mpi-l32-results
uv run --no-project python benchmark/tenferro_thread_budget/analyze-mpi-l32.py /tmp/mpi-l32-results > /tmp/mpi-l32-summary.json
```

The script copies the repository inputs, changes only PhysCal `NDataQtySmp` to 20, records their hashes, uses `--nsteps 20 --nsmp 1` for Opt, and runs the four-rank configurations in separate containers. Adapt the CPU set and `/opt/mpich/bin/mpirun` path for a different image.
