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
