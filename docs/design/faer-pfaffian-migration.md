# Faer Pfaffian migration (#731)

Status: implementation in progress; production dispatch and CUDA migration are pending.

The owner's requested target is tensor4all/faer for CPU Pfaffians and corresponding
GPU acceleration, measured before PR merge. The starting report is
[pfaffian-benchmark/BENCHMARKS.md](https://github.com/AtelierArith/pfaffian-benchmark/blob/main/BENCHMARKS.md).
Its source revision `14462105` identifies t4a-faer 0.32.0. The adapter currently
pins the registry release 0.32.0; new performance results must identify this
release rather than claim to measure that Git revision.

## Current implementation

`pfapack`'s opt-in `faer-backend` feature supplies real and complex reusable
workspaces, compact column-major input views, sequential execution and explicit
provider errors. Inputs are non-destructive, with strict lower-triangle authority.
The scalar in-place kernels remain independent references. No production caller
has switched yet, and no downstream speedup is claimed.

The dependency adds t4a-faer 0.32.0 alongside tenferro's 0.28.2. Cargo unifies
t4a-faer-traits at 0.24.3 and t4a-private-gemm-x86 at 0.1.28 (the latter is a
minimum requirement of 0.32.0); workspace validation must cover those shared
dependency updates too.

## Call-path findings and remaining work

- `mvmc-core/src/pfaffian.rs` computes Pfaffian and inverse from the same
  `dsktf2`/`zsktf2` decomposition and `utu2pfa`/`utu2inv` pipeline. Replacing
  `pfaffian_ltl_*` alone does not migrate the sampler's rebuilds. The provider
  currently returns only a Pfaffian, so inverse construction needs an explicitly
  measured design and independent residual checks.
- `mvmc-core/src/observables/lanczos_nodal.rs` calls standalone real/complex
  Pfaffians and is a direct candidate for the reusable workspaces.
- `mvmc-gpu/src/scalar.rs` also shares factorization between Pfaffian and inverse;
  `tenferro_native.rs` implements a separate eager pipeline.
- The resident CUDA kernel `gpu/mvmc-gpu-cuda/src/pfaffian_batched.cu` uses one
  block per plane and upper-storage LTL/inverse construction. It is not a faer
  call. Audit its resident fallback callers as well as explicit batched launches
  before adding panel updates; preserve on-device lifetimes and transfer counts.
- Both RTX 3060 devices were visible locally on 2026-11-11. CUDA validation and
  matched before/after measurements are still pending.

Error policies must distinguish structural zeros from nonfinite arithmetic and
underflow. Do not silently map every provider failure to zero. Validate complex
plain-transpose signs, empty/odd/singular inputs, panel crossings, C 1.4.0
fixtures, inverse residuals and unchanged RNG on fixed control paths. Numerical
acceptance divergences require first-divergence evidence, as specified in
`docs/NUMERICAL_COMPARISONS.md`.

## Validation so far

On Linux x86_64, the initial adapter passes `cargo check -p pfapack --features
faer-backend`, `cargo nextest run -p pfapack --features faer-backend` (35 passed;
one existing ignored test), and `cargo clippy -p pfapack --features faer-backend
--all-targets -- -D warnings`. New tests cover boundary/error behavior, an
independent complex 4x4 formula with workspace reuse, and a 64x64 panel-crossing
comparison to the scalar reference. Full workspace, optimized BLAS/SIMD,
doctests, Linux/macOS CI, GPU correctness and E2E performance gates remain open.
