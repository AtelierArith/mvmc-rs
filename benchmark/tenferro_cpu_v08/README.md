# CPU tenferro v0.8 migration: SR stage comparison

These are exploratory measurements for #569, not an end-to-end Opt or PhysCal benchmark. The existing `sr_backend_bench` generates deterministic input and reports the median of seven invocations after one warmup for each operation; the operations include explicit tensor conversion and transfers. Tests use 64 and 128 parameters and 300 or twice as many samples. Outputs list C-order and tenferro timings (milliseconds) and maximum relative differences against C-order.

Measured on Linux x86_64, Intel Xeon E5-2699 v3 (36 logical CPUs visible), Rust 1.99.0 (b940084d7), system `/lib/x86_64-linux-gnu/libopenblas.so.0`; `OPENBLAS_NUM_THREADS=1`, `RAYON_NUM_THREADS=1` or `4`. Before: main `5f5a490e3f4a4b77149b147f684ed2bbae252d77`, tenferro 0.7.1 with the repository's security patches. After: the #569 migration working tree, tenferro-ad/tensor/einsum/linalg/runtime 0.8.0 and tenferro-cpu 0.8.1. The exact after commit must be recorded when finalized. The two binaries were run sequentially; no affinity or CPU-frequency control was applied, and each condition was run once. Results describe this run only, not statistically established speedups. The four-thread measurement also does not establish worker utilization or multi-walker behavior.

Build each checkout independently and run after other workloads have stopped:

```sh
cargo build -p mvmc-core --profile test-fast --example sr_backend_bench
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 target/test-fast/examples/sr_backend_bench 64 128
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 target/test-fast/examples/sr_backend_bench 64 128
```

## Raw output

### before, 1 threads

```text
ms, median of 7, C-order / tenferro (cpu-faer)
| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec | rel diff gram,asm,solve,cg |
|---:|---:|---|---|---|---|---|
| 64 | 300 | 0.12 / 0.36 | 0.02 / 0.40 | 0.03 / 0.34 | 0.014 / 0.183 | 7.8e-16 0.0e0 6.3e-16 1.0e-15 |
| 64 | 128 | 0.07 / 0.19 | 0.03 / 0.46 | 0.04 / 0.35 | 0.005 / 0.133 | 0.0e0 0.0e0 6.1e-16 8.5e-16 |
| 128 | 300 | 0.56 / 2.25 | 0.11 / 1.41 | 0.17 / 0.78 | 0.038 / 0.274 | 1.1e-15 0.0e0 1.0e-15 1.3e-15 |
| 128 | 256 | 0.48 / 1.96 | 0.10 / 1.45 | 0.17 / 0.80 | 0.031 / 0.264 | 0.0e0 0.0e0 9.5e-16 1.5e-15 |

```

### before, 4 threads

```text
ms, median of 7, C-order / tenferro (cpu-faer)
| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec | rel diff gram,asm,solve,cg |
|---:|---:|---|---|---|---|---|
| 64 | 300 | 0.11 / 0.43 | 0.02 / 0.45 | 0.04 / 0.34 | 0.013 / 0.166 | 7.8e-16 0.0e0 6.3e-16 1.0e-15 |
| 64 | 128 | 0.06 / 0.16 | 0.02 / 0.45 | 0.04 / 0.32 | 0.005 / 0.134 | 0.0e0 0.0e0 6.1e-16 8.5e-16 |
| 128 | 300 | 0.52 / 1.27 | 0.10 / 1.48 | 0.17 / 0.97 | 0.035 / 0.256 | 1.1e-15 0.0e0 1.0e-15 1.3e-15 |
| 128 | 256 | 0.49 / 1.19 | 0.10 / 1.56 | 0.17 / 0.91 | 0.029 / 0.250 | 0.0e0 0.0e0 9.5e-16 1.5e-15 |

```

### after, 1 threads

```text
ms, median of 7, C-order / tenferro (cpu-faer)
| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec | rel diff gram,asm,solve,cg |
|---:|---:|---|---|---|---|---|
| 64 | 300 | 0.12 / 0.50 | 0.02 / 0.28 | 0.03 / 0.21 | 0.010 / 0.132 | 7.8e-16 0.0e0 6.3e-16 1.0e-15 |
| 64 | 128 | 0.05 / 0.16 | 0.02 / 0.28 | 0.03 / 0.20 | 0.004 / 0.093 | 0.0e0 0.0e0 6.1e-16 8.5e-16 |
| 128 | 300 | 0.40 / 1.23 | 0.08 / 0.50 | 0.13 / 0.42 | 0.028 / 0.180 | 1.1e-15 0.0e0 1.0e-15 1.3e-15 |
| 128 | 256 | 0.35 / 0.93 | 0.08 / 0.48 | 0.12 / 0.41 | 0.022 / 0.147 | 0.0e0 0.0e0 9.5e-16 1.5e-15 |

```

### after, 4 threads

```text
ms, median of 7, C-order / tenferro (cpu-faer)
| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec | rel diff gram,asm,solve,cg |
|---:|---:|---|---|---|---|---|
| 64 | 300 | 0.12 / 0.38 | 0.02 / 0.30 | 0.03 / 0.22 | 0.012 / 0.138 | 7.8e-16 0.0e0 6.3e-16 1.0e-15 |
| 64 | 128 | 0.05 / 0.15 | 0.02 / 0.30 | 0.03 / 0.23 | 0.005 / 0.099 | 0.0e0 0.0e0 6.1e-16 8.5e-16 |
| 128 | 300 | 0.39 / 0.56 | 0.07 / 0.53 | 0.13 / 0.44 | 0.030 / 0.183 | 1.1e-15 0.0e0 1.0e-15 1.3e-15 |
| 128 | 256 | 0.35 / 0.48 | 0.08 / 0.51 | 0.12 / 0.43 | 0.021 / 0.166 | 0.0e0 0.0e0 9.5e-16 1.5e-15 |

```
