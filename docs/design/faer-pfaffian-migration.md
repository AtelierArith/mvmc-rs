# Faer Pfaffian migration (#731)

CPU default integration, CPU/GPU correctness and matched performance are verified
locally. The final exact-head Linux/macOS CI gate is required before merge.

The owner requested tensor4all/faer's accelerated Pfaffian calculation on CPU
and corresponding GPU work, based on
[pfaffian-benchmark/BENCHMARKS.md](https://github.com/AtelierArith/pfaffian-benchmark/blob/main/BENCHMARKS.md).
Its source `14462105` is the base of t4a-faer 0.32.0. A Pfaffian-only replacement
does not accelerate mVMC's complete rebuild: Pfaffian and inverse originally
share one C-order LTL factorization. Pairing the released faer Pfaffian with a
second LU factorization regressed real rebuilds, so that experiment was replaced.
Historical measurements remain labelled in the report.

## CPU implementation

`pfapack/faer-backend` uses an immutable Git dependency on the public
[AtelierArith/faer fork](https://github.com/AtelierArith/faer/tree/a3dd7b450e81df16065ebf51788414d87bb44171),
revision `a3dd7b450e81df16065ebf51788414d87bb44171`, based on tensor4all/faer.
The branch adds `pfaffian_inverse_scratch` and `pfaffian_inverse`: the original
paired panel reduction retains L/D/permutation, followed by faer's triangular
inverse and triangular GEMM. There is one factorization, no conjugation, and
no inverse skew projection. Private retained factors do not expand the public
API. The fork has not been published to a registry or proposed to external
maintainers; neither action is needed to reproduce this pinned dependency.

Real/complex workspaces retain scratch and select `Par::Seq` inside existing QP
workers. Input has explicit row/column layout; its strict lower triangle is
authoritative, the input remains untouched, and output is a true column-major
inverse. Provider errors distinguish odd dimensions, zero pivots and numerical
failure; inverse overflow is not accepted just because the Pfaffian is finite.

`mvmc-core` and `mvmc-cli` enable `faer-pfaffian` by default. Full sampler and
default `calc-m-all` measurement rebuilds use faer for real n>=128 and complex
n>=64, including FSZ. Smaller matrices retain the faster C-order kernels.
Presenting C's original upper storage as a row-major view represents -A to
faer: its true inverse directly gives the sampler's stored -A^-1, and
`(-1)^(n/2)` corrects the Pfaffian. Ignored lower/diagonal entries need not be
skew. All provider failures restore the original matrix and take the C-order
path, preserving its INFO and partial-storage contracts.

The nonsingular block/full-factor INFO defect is an explicit exception under
the existing clear-C-defect policy, supported by the independent native probe
in `c_toolbox/pfaffian_block_info` and
[Julia-mVMC#98](https://github.com/tmisawa/Julia-mVMC/issues/98).
Genuinely singular inputs retain their errors. See `docs/NUMERICAL_COMPARISONS.md`.

## Call-path audit

- `pfaffian.rs` routes all four normal/FSZ real/complex full rebuild entries
  through the selected workspace before their existing fallback.
- `observables/lanczos_nodal.rs` has the only two standalone Pfaffian callers.
  Its private `expand_terms` appends at most two Hamiltonian operator pairs to
  at most two external pairs; reductions only shorten that list. Thus the
  child matrix order is at most eight. These tiny kernels retain the reference
  implementation rather than paying a large-matrix workspace cost.
- Explicit `c-order` and `tenferro` measurement stages remain independently
  selectable C-order executors. The default `calc-m-all` stage uses the new
  size dispatch. `mvmc-gpu` scalar/native providers also remain references.
- The standalone CUDA workspace disables core's default features explicitly,
  preserving its CPU C-order oracle and existing lock. No CPU faer team/state
  is added to the resident GPU runner.

## Resident GPU implementation

CUDA has no faer GPU Pfaffian API. The Rust/device implementation ports its
delayed panel update organization into the existing resident kernels. Width-eight
panels reuse existing buffers; full rebuilds retain L/D/permutation and share
the reduction with inverse construction. The original input backup survives
through the inverse, so pivot, reciprocal and nonfinite failures restore it
and execute the old reference on device, without a host round trip.

Full rebuild dispatch selects panels for real n=64..126 and complex n>=64.
The existing faster Julia paired factorization remains for real n>=128,
including its independent C-order cancellation fallback. Standalone complex
Pfaffians select panels at n>=64. Standalone real proposal/observable matrices
retain the old loop: dense random kernel gains did not translate to a physical
L128 improvement. Controlled measurements and reverted experiments are retained.

There are no added device allocations, launches, host transfers or explicit
synchronizations. Complex shared storage is 44n bytes, within 48 KiB at n<=1024.

## Evidence and remaining gates

The complete report, raw results, source manifests and reproduction commands
are in `benchmark/pfaffian_faer/README.md` and `e2e/README.md` below that directory.

- Final provider tests cover f32/f64/c32/c64, strided layouts, triangle poison,
  plain-transpose complex signs, empty/odd/singular/nonfinite/extreme scales,
  panel boundaries and independently bounded inverse residuals. Full provider
  tests and doctests pass; held-team allocation tests pass for 4/8/70 workers,
  and its minimal no_std library build passes. Logs identify the fork revision;
  this is downstream validation, not a tensor4all/faer-main benchmark campaign.
- The final CPU adapter's eight focused tests pass, including upper-triangle
  authority, odd-pair sign, negative inverse, failure restoration and residuals.
  The all-features workspace passes 1,455 tests. Doctests, all-target Clippy and
  the explicitly enabled C-referenced threaded runner matrix pass too.
- The selected CUDA source passes all 132 numerical/runner tests, including
  Julia GPU fixtures, supplied tapes, RNG/transfer contracts, 20-step/window
  repeatability, inverse residuals and two-device context rejection.
- Selected complex L128 Opt improves 1.1575x and PhysCal 1.1230x in matched
  alternating three-repetition cohorts. Real L128 is effectively unchanged
  (0.9974x Opt, 0.9942x short PhysCal); no real L128 gain is claimed. Explicit
  transfer calls/bytes and synchronizations are unchanged.
- CPU real L128 Opt/PhysCal improves 1.0418x/1.1027x; complex improves
  1.2153x/1.2711x. All recorded RNG state/counts, configurations and acceptance
  counters match across implementations; same-implementation repetitions pass.
  Five final full-rebuild probes confirm real n>=128 and complex n>=64 gains.
- Exact-head Linux/macOS CI must pass before PR #732 is merged.

Computed floating-point values use explicit numerical bounds, never bitwise
comparison. RNG initialization/state/draw order remain independently exact for
fixed control paths. Any acceptance divergence requires locating its first
numerical cause rather than attributing it to Monte Carlo noise.
