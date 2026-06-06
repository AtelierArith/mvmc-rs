# PfaPack comparison benchmark

Compares `extern/PfaPack.jl` with `crates/pfapack` on deterministic
dense skew-symmetric matrices. Each timed iteration clones/copies the
input first because the routines are in-place.

Run Julia:

```sh
julia --project=extern/PfaPack.jl benchmark/pfapack_compare/bench_julia.jl
```

Run Rust scalar backend:

```sh
cargo run --release --manifest-path benchmark/pfapack_compare/Cargo.toml --offline
```

Run Rust SIMD backend. This keeps the scalar code as the reference path
and uses `pulp` runtime dispatch for the complex LTL upper-triangular
rank-2 update when the update is large enough to amortize dispatch cost.

```sh
cargo run --release --manifest-path benchmark/pfapack_compare/Cargo.toml \
  --features pfapack/simd-backend --offline
```

Run Rust BLAS/LAPACK backend. On macOS this links Homebrew OpenBLAS
when `/opt/homebrew/opt/openblas` or `/usr/local/opt/openblas` exists,
and falls back to Accelerate otherwise.

```sh
cargo run --release --manifest-path benchmark/pfapack_compare/Cargo.toml \
  --features pfapack/blas-backend --offline
```
