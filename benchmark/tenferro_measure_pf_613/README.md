# Measurement Pfaffian sources, matched single-thread timing (#613)

`run.sh` times the three measurement Pfaffian sources on the L32 Hubbard chain:

- `calc-m-all` (default): the per-sample C-order `calc_m_all_real` kernels;
- `c-order`: one batched `COrderPfaffian` call per measurement batch (stage A through `PfaffianStages`);
- `tenferro`: the same batch as one run of a prepared tenferro program of the PfaPack extension operation (`mvmc_core::tenferro_pfaffian::TenferroPfaffian`). The executor is the C-order `dsktf2`/`utu2pfa_real`/`utu2inv_real` sequence; tenferro provides the program, its preparation (once per `(n, planes)` shape) and dispatch on a serial `CpuEngine`.

```sh
cargo build --release -p mvmc-cli
benchmark/tenferro_measure_pf_613/run.sh target/release/mvmc /tmp/bench-613 3
```

Workloads: fixed-parameter PhysCal `benchmark/physcal/inputs/hubbard_chain_L32` on its `zqp_opt.dat` (fixed-parameter SHA-256 `2b984a8c1dc01b1e5a306a635a07805c61af3b921498fe863bc5399907d991f5`), and Opt `benchmark/hubbard_chain/inputs/hubbard_chain_L32 --nsteps 20 --nsmp 10 --initial-def none`. Timing scope: whole process wall time (`date` around the executable), including input parsing and output. Threads: `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=RAYON_NUM_THREADS=MVMC_RS_INNER_THREADS=MVMC_RS_TENFERRO_THREADS=1`; `MVMC_RS_SR_BACKEND` and `MVMC_RS_MEASURE_BACKEND` unset (C-order SR and stage B), default measurement batch size.

## Results (2026-11-05)

Linux x86_64 host, Intel Xeon E5-2699 v3, rustc 1.99.0, release profile, OpenBLAS from the workspace lock, base commit `e2a4d06c` plus the #613 change. Three repetitions each, seconds:

| workload | `calc-m-all` | `c-order` | `tenferro` | tenferro / calc-m-all |
| --- | --- | --- | --- | --- |
| PhysCal L32 | 9.936, 9.856, 9.975 | 10.077, 10.086, 10.079 | 10.104, 10.095, 10.111 | 1.016 (median) |
| Opt L32, 20 steps | 2.124, 2.123, 2.142 | 2.176, 2.173, 2.181 | 2.176, 2.200, 2.183 | 1.028 (median) |

The tenferro program adds about 0.2 % over the batched C-order stage on PhysCal and is within run-to-run spread on Opt; both batched stages are 1-3 % slower than the per-sample default because the batch path assembles and copies all planes of a batch (the tenferro executor makes one more copy, `AllFresh` output). There is no host/device transfer: the program runs on the CPU engine and reads the caller's planes zero-copy.

Numerics: in these runs every non-timing output file of `c-order` and `tenferro` was byte-identical to `calc-m-all` (informational; tests compare with explicit bounds, see `crates/mvmc-cli/tests/issue464_backend_selector.rs` and the tenferro cases of `issue181_native_c_physcal.rs`, which also check RNG draw counts, SFMT state and saved configurations exactly against native C).
