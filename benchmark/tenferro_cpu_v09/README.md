# CPU tenferro v0.9 migration: SR stage comparison (#644)

Linux x86_64, Xeon E5-2699 v3 (36 logical CPUs visible), Rust 1.99.0 (b940084d7), `test-fast` profile, `OPENBLAS_NUM_THREADS=1`, `RAYON_NUM_THREADS=1` or `4`. Before: main `8a3004aa`, tenferro 0.8.0 / tenferro-cpu 0.8.1 / threadteam 0.2.0 / t4a-faer 0.27.1 (binary SHA-256 `786de418c17aa6003b9d0c8ef36f2aab63ff28475a55de4ce86d75054b45766c`). After: this change, tenferro 0.9.0 / threadteam 0.3.0 / t4a-faer 0.28.2 (binary SHA-256 `37ca32cfc36971db0ee46487bb8b7d4effc9a9e0cdee4ad373ee6880d1008093`). Before and after were run interleaved, three repetitions per thread count, on a shared host (load average 7 to 19 during the run) without affinity or frequency control. Raw output: [raw-20261104.txt](raw-20261104.txt).

```sh
cargo build -p mvmc-core --profile test-fast --example sr_backend_bench
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 target/test-fast/examples/sr_backend_bench 64 128
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 target/test-fast/examples/sr_backend_bench 64 128
```

## Timing (tenferro column, NPara 128, 256 samples, median of the three per-run medians, ms)

| threads | stage | before | after |
|---:|---|---:|---:|
| 1 | Gram | 0.75 | 0.68 |
| 1 | S,g assembly | 0.33 | 0.31 |
| 1 | Cholesky solve | 0.29 | 0.39 |
| 1 | CG matvec | 0.106 | 0.088 |
| 4 | Gram | 0.41 | 0.32 |
| 4 | S,g assembly | 0.46 | 0.34 |
| 4 | Cholesky solve | 0.41 | 0.30 |
| 4 | CG matvec | 0.143 | 0.088 |

The C-order column (unchanged code) varied by up to about 30% across runs of the same binary, so these differences are within the host noise except possibly the 4-thread CG matvec. This does not establish a speedup; end-to-end Opt timing is unaffected because the default SR path is C-order.

## Numerical differences

The C-order results are unchanged. Gram, assembly and Cholesky relative differences against C-order and both scaled residuals are identical before and after. Only the tenferro CG local product changed, at roundoff level: relative difference to C-order 8.5e-16 to 7.3e-16 (NPara 64, 128 samples) and 1.3e-15 to 1.2e-15 (NPara 128, 300 samples); the other two cases are unchanged. The cause is the GEMM backend under faer: t4a-faer 0.27 used `gemm` 0.19, and t4a-faer 0.28 uses `t4a-gemm` 0.20, which changes the accumulation order for these matrix-vector products. All existing tenferro SR/measurement bounds (`sr_backend_421`, `measurement_tenferro_518`, `multiwalker_sr_435`) pass without tolerance changes.
