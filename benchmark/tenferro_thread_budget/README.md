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
