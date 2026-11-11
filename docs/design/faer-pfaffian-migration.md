# Faer Pfaffian migration (#731)

Status: implementation in progress. GPU standalone-Pfaffian panels are implemented;
CPU full rebuild integration is experimental, and final adoption/E2E gates remain open.

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
The scalar in-place kernels remain independent references. Real/complex inverse
workspaces now combine faer's Pfaffian and blocked LU/inverse with shared scratch
and no per-evaluation allocation. The inverse is projected onto plain-transpose
skew symmetry after finite-result validation.

`mvmc-core/faer-pfaffian` is an experimental, non-default feature routing normal
and FSZ rebuilds through those workspaces. The sampler stores the negative of
the true inverse, as before. Non-skew input or provider failure restores the
original plane and executes the C-order path, preserving INFO, partial storage
and FSZ singular behavior. Real/complex scratch is lazy and reused per QP worker.
The current experiment retains C rebuilds below n=256 (real) and n=64
(complex), because the complete-rebuild probe shows regressions below those
regions. The real large-matrix path remains a candidate, not a proven speedup.

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
- GPU standalone helpers now use an eight-column panel with eliminated columns
  as factor storage. It needs the same matrix and two n-vectors, with no added
  transfers or launches. Measurements retain the old kernel below n=128 (real)
  and n=64 (complex). See `benchmark/pfaffian_faer/README.md` for raw probes.
- The main `pfinv_f64`/`pfinv_c64` LTL/inverse rebuild kernels are still pending;
  the standalone-helper speedup must not be presented as their speedup.

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

Additional verification on 2026-11-11:

- `cargo nextest run -p pfapack --features 'faer-backend blas-backend'`:
  39 passed, one pre-existing ignored test. New inverse tests cover real/complex
  n=0/2/4/24/26/64, workspace reuse, independent identity residuals, underflow,
  nonfinite input and inverse overflow. Clippy passes for that configuration.
- `cargo clippy -p mvmc-core --features faer-pfaffian --lib -- -D warnings` passes.
  The direct integration sign/restoration test passes.
- The initial all-size core feature trial found three comparison failures:
  `stage_a_matches_calc_m_all_real_and_applies_the_invm_sign`,
  `stage_backend_c_order_measurement_is_byte_identical_to_calc_m_all`, and
  `stage_backend_tenferro_measurement_matches_calc_m_all`. These compare the new
  LU inverse with explicit old C-order stages. The first observed kernel
  differences were about 6e-15 (before skew projection); after projection, the
  reported n=4 difference is about 1e-15, including zeroed diagonals. One
  three-step SR output difference before projection was 1.6185e-14 in
  `zqp_opt.dat`. The measured size dispatch now retains the faster reference
  implementation at these small sizes. Rerunning the targeted feature tests
  passes all 14 tests, including all three above, without modifying tests or
  tolerances. A separate n=256 integration test verifies actual provider use,
  negative inverse residuals and restored input on failure. Validation for
  eligible-size physical workloads and their discrete controls remains pending.
- CUDA 12.9.2 container `b129bd583bbe`, RTX 3060 device 0: the Rust ignored tests
  `dense_refactor_matches_known_pfaffians_and_inverses` and
  `panel_pfaffians_match_cpu_reference` pass. The latter compares real and
  complex n=2/6/24/26/64/128/130/256 against the independent CPU LTL path.

CPU real Pfaffian+LU inverse has not demonstrated an overall rebuild speedup,
unlike complex n>=64. Five repeated probes with the final skew projection
confirm this (mean timing ratios: 0.57–0.95x real and 1.14–1.69x complex n>=64).
The preliminary report records this limitation explicitly. Real default adoption
requires further inverse optimization or another measured strategy. The feature
remains experimental. Resident E2E performance and the complete GPU gate remain open.

The optimized `pfapack` feature run including `faer-backend`, `blas-backend` and
`simd-backend` passes all 39 enabled tests (one pre-existing ignored test).

## Shared-factor provider prototype (2026-11-11)

The released Pfaffian interface's discarded factors cause duplicate O(n^3)
work when paired with a separate inverse. A local provider extension now
retains L/D/permutation during the original faer reduction and computes the
inverse with faer's triangular inverse and triangular GEMM. Its patch, pinned
dependency lock, isolated reproduction script, domain/layout/identity checks
and five timing samples are in
`benchmark/pfaffian_faer/provider-prototype/`. Mean timing ratios for complete
rebuilds reach about 1.49x real and 2.96x complex at n=512. This resolves the
performance limitation in a prototype, not in a released dependency: no
upstream PR has been submitted and the main workspace still uses the earlier
experimental adapter. An asynchronous scope question asks the owner whether
to include a PR to tensor4all/faer, since that would send a change proposal to
maintainers outside the original mvmc-rs PR scope.

The CUDA standalone-panel gate from source `f3abded0` completed: 115 library,
3 CLI, 1 interval gate, 3 interval parity and 6 resident interval tests passed.
It includes Julia fixtures and 20-step/window repeatability. The standard gate
excludes the separate two-device context rejection test; that test remains to
be run independently. The log is
`benchmark/pfaffian_faer/validation/cuda-standalone-panels-gate.log`.
New complete-rebuild CUDA dispatch is being validated separately; the above
gate is not evidence for changes made after that build.

The new CUDA shared-factor rebuild now passes production-entry tests against
CPU reference/identity residuals for real and complex n=24/26/64/128/130/256,
including zero-pivot status and shared-memory bounds. The complex L128 Opt
stage test passes. The measured dispatch selects new real n=64..126 and
complex n>=64, while preserving the faster existing Julia paired real n>=128
path and its independent C-order cancellation fallback. Rebuild probe details
and the distinction from the real resident baseline are in
`benchmark/pfaffian_faer/README.md`. Complete post-change gates and E2E/transfer
measurements remain pending. The focused follow-up delayed-update/rebuild-boundary
regression passed after the L128 stage. A complete gate for the new rebuild
source (`38f2849b`) is now running, including a separate two-device context test.
