# SR solve residuals before/after tenferro v0.8 (#585)

This supplements [the original SR stage comparison](README.md). The measured deterministic `sr_backend_bench` workloads use `NPara=64,128` and sample counts 300 and `2*NPara`. Each process performs one warmup then seven calls per operation; the table in its raw output reports the median operation time. We ran one process warmup plus three measured processes for each version/thread count, alternating before/after order. Measurements are serial in time; no CPU affinity or frequency control was used. Wall and CPU time from `/usr/bin/time -v` include process startup, input generation and all C-order/tenferro operations. The processes are short, so the data do not prove a general speedup.

Each timed Cholesky solve uses the same saved, unmodified C-order `S` and right-hand side `g`. Outside its timer, the example computes the scaled infinity-norm backward error `||S x - g||∞ / (||S||∞ ||x||∞ + ||g||∞)` separately for C-order and tenferro. All four cases in all three measured processes had these maximum reported residuals:

| tenferro revision | Threads | Maximum C-order residual | Maximum tenferro residual |
|---|---:|---:|---:|
| 0.7.1 | 1 | 6.7e-17 | 8.2e-17 |
| 0.7.1 | 4 | 6.7e-17 | 8.2e-17 |
| 0.8 | 1 | 6.7e-17 | 8.2e-17 |
| 0.8 | 4 | 6.7e-17 | 8.2e-17 |

These residuals are rounded to two significant digits and do not by themselves establish conditioning, forward error or other workloads' behavior. The output additionally reports the relative differences to C-order for Gram, assembly, solution and CG product. [All raw logs](results/residual-comparison.txt) include the warmups and process-level `/usr/bin/time -v` values.

Linux x86_64 6.8.0-147, Xeon E5-2699 v3, Rust 1.99.0 (b940084d7), system `/lib/x86_64-linux-gnu/libopenblas.so.0`, OpenBLAS/OMP 1; `RAYON_NUM_THREADS=1` or `4`. Before source `5f5a490e3f4a4b77149b147f684ed2bbae252d77`, tenferro 0.7.1; after source `1aa01f17d2e457395b5c0538e141ebf23a1054fb` plus the example change in this PR, tenferro-ad/tensor/einsum/linalg/runtime 0.8.0 and tenferro-cpu 0.8.1. Both `test-fast` binaries used the same residual instrumentation. Their SHA-256 values were `dce4d484575849f81773a672cc4bf73fbf5b076d16cf050cadca522e6bd1a4bd` (before) and `8eaf661669a91830f3f0a6467444aab9e41326188e331bff8df1c045749d84aa` (after). The numerical computation did not use bitwise comparison.

## Reproduce

In a separate clean checkout of the before revision apply [residual-before.patch](residual-before.patch); the after revision needs this PR's example change. Build each checkout separately:

```sh
git apply /path/to/residual-before.patch   # before checkout only
cargo build -p mvmc-core --profile test-fast --example sr_backend_bench
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/time -v target/test-fast/examples/sr_backend_bench 64 128
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/time -v target/test-fast/examples/sr_backend_bench 64 128
```

For the complete alternating sequence, with no other intentional workload:

```sh
benchmark/tenferro_cpu_v08/compare-residual.sh \
  /path/to/before/sr_backend_bench /path/to/after/sr_backend_bench /tmp/sr-residual-results
```

Use a fresh output directory. The example's residual calculation is checked independently on a diagonal system by `cargo nextest run -p mvmc-core --example sr_backend_bench --cargo-profile test-fast -E 'test(residual_matches_a_diagonal_system_with_known_error)'`.
