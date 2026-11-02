# tenferro CPU thread-budget probe (#585)

This short synthetic SR-stage experiment compares independent processes, not real MPI ranks or mVMC walkers. Four processes contend on the same four logical CPUs, not four disjoint CPU sets. Each process runs `sr_backend_bench 256` (C-order and tenferro Gram, S/g assembly, solve and CG, with deterministic inputs and two sample counts; one warmup and seven repetitions per stage). Process-group elapsed time includes program startup and all those calls. It does not isolate tenferro-only utilization. Each group was repeated three times. The individual reports include `/usr/bin/time -v` user/system CPU times and numerical differences to C-order.

Linux x86_64 6.8.0-147, Xeon E5-2699 v3; `taskset -c 0-3`, OpenBLAS and OMP each 1 thread, no CPU frequency control. `RAYON_NUM_THREADS` was 1 or 4. Baseline binary was built from `e4240834b0b1ef0271e197931da0b77221662fb6`, SHA-256 `be58a4ab47ed403aaf43afe1dfabce10387e1dd8e188702c3717c725ca86b31b`. The override binary was built from `1c34ad2ca323efef1ac27595dc3d7a3db5f9abd8` plus this change, SHA-256 `d2ae15ed7e06b0b874629fc5f13aa9b3aa429a7ed71cb1c340005d8d9ee94ab5`. Both use `test-fast`, not release. The override run set `MVMC_RS_TENFERRO_THREADS=1`; baseline left it unset. Conditions were sequential, not interleaved; total groups last well below a second, so frequency and scheduling variation matter.

| Rayon threads | Processes | Baseline group wall median | Tenferro=1 group wall median |
|---:|---:|---:|---:|
| 1 | 1 | 0.207 s | 0.218 s |
| 1 | 4 | 0.229 s | 0.223 s |
| 4 | 1 | 0.164 s | 0.193 s |
| 4 | 4 | 0.876 s | 0.221 s |

In this limited oversubscribed probe the explicit cap reduced contention; for one process it also removed useful tenferro parallelism. Each operation's maximum relative error is reported in the per-process output and remains near floating-point roundoff; this is not an independent full-trajectory or RNG test. Shared Rayon execution, true multi-walker behavior, MPI collectives, nested execution, placement and wait-policy behavior are not established. This setting is a manual per-Engine budget, not an automatic global scheduler; #581–#584 and #585 remain open.

The baseline group data and consolidated per-process output are in [results](results/); the override group data are in [results/override](results/override/). Reproduce with an otherwise idle host and adjust the affinity in `probe.sh` if CPUs 0–3 are unavailable:

```sh
cargo build -p mvmc-core --profile test-fast --example sr_backend_bench
benchmark/tenferro_thread_budget/probe.sh target/test-fast/examples/sr_backend_bench /tmp/budget-default
MVMC_RS_TENFERRO_THREADS=1 benchmark/tenferro_thread_budget/probe.sh target/test-fast/examples/sr_backend_bench /tmp/budget-serial
```

Use fresh output paths. The script fixes `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`, controls `RAYON_NUM_THREADS` internally, and records three group times plus per-process raw output.

## MPI protocol check

The existing ignored `run::mpi_runtime_tests::negative_clock_and_asymmetric_runner_failures` test was run in the Linux x86_64 development-container image with MPICH, for 2 and 4 actual ranks. `MVMC_RS_SR_BACKEND` was `c-order` or `tenferro`, `RAYON_NUM_THREADS=4`, and `MVMC_RS_TENFERRO_THREADS` was 1 or 4 for each combination. `MVMC_RS_INNER_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; no CPU affinity was set. All ranks passed in all eight runs. This tests the existing fault-injection and collective protocol only; it does not establish throughput or arbitrary nested/multi-walker behavior. The C-order and tenferro controls both initially exposed the outdated output-file assertion tracked by #590. C `InitFile` creates the empty time file before the PhysCal loop; the corrected test checks exactly that file and no sample output.

Raw outputs: [results/mpi](results/mpi/). Build and execute inside the development container (the test binary hash will differ by build):

```sh
cargo test -p mvmc-core --features mpi --lib --profile test-fast --no-run
# Find the executable printed by the preceding command and substitute it below.
MVMC_RS_SR_BACKEND=tenferro MVMC_RS_INNER_THREADS=1 RAYON_NUM_THREADS=4 \
  MVMC_RS_TENFERRO_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  timeout --kill-after=5s 120s mpirun -n 4 /path/to/mvmc_core_test_binary \
  --ignored --exact run::mpi_runtime_tests::negative_clock_and_asymmetric_runner_failures --nocapture
```

Repeat with `-n 2`, `MVMC_RS_TENFERRO_THREADS=4`, and the `c-order` backend for the complete matrix. The container needs sufficient shared memory (the recorded run used `--shm-size=1g`).

## Single-walker CG smoke timing after the lock fix

As a coarse regression check, the existing `cg_run_with_tenferro_keeps_structure_and_the_pre_update_energy` test was executed with `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1` in separate `test-fast` worktrees, once as warmup then five measured repetitions alternating execution order. Before was `718932f2…` (the source merged as `1c34ad2c…`), after was `068ffc45…`; the latter includes the CPU budget and shared-lock changes. `/usr/bin/time` covered the complete `cargo nextest` command including test selection and process startup, not the CG kernel alone. Median wall times were 0.64 s before (0.64, 0.65, 0.60, 0.59, 0.64 s) and 0.62 s after (0.69, 0.65, 0.62, 0.60, 0.59 s). No affinity or frequency pinning was applied; these coarse, short runs cannot rule out smaller performance regressions. [Full output](results/cg-single-walker.txt).

```sh
OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 /usr/bin/time -f 'wall=%e user=%U system=%S' \
  cargo nextest run -p mvmc-core --cargo-profile test-fast \
  -E 'test(cg_run_with_tenferro_keeps_structure_and_the_pre_update_energy)' --retries 0
```
