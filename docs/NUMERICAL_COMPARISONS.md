# Portable numerical comparisons

Issue #186 was completed in PR #191 using its original native verification
gates. Issue #190 implements the subsequent user decision: computed
floating-point results, including Rust-to-C results, use explicit numerical
bounds on Linux and macOS. C remains the algorithm, input and numerical
authority; Julia remains the design reference. Reference numerical work runs
on Linux, using the Dev Container on a non-Linux host. Native macOS tests
verify portability and retain labelled historical/platform expectations.

## Comparison rules

`tests/support/numerical_comparison.rs` compares each finite component using

```text
|actual - expected| <= absolute + relative * max(|actual|, |expected|)
```

Callers supply the bounds for their operation. Complex values compare real
and imaginary components separately. A small absolute bound handles
cancellation near zero; tiny range-recovery tests instead use a few minimum
subnormal quanta, so they cannot silently treat every small result as zero.
The helper handles overflow of the difference between finite endpoints.
Range fixtures with coefficients such as `1e-300` use a subnormal floor and
a relative bound; an order-epsilon absolute floor cannot validate losing
those coefficients entirely.

NaN matches only an expected NaN classification; payload/sign are not an
arithmetic contract. Infinity must match infinity with the same sign, never a
finite value. Signed zeros are numerically equivalent. Literal parsing and
copy/restoration contracts can retain exact representations independently of
arithmetic comparisons.

Seed resolution, RNG initialization/state, full 624-word generator blocks,
float conversion and generation order/count for a fixed control path remain
exact and are tested independently. Indices, flags, dimensions and discrete
input contracts remain exact.

General mathematical functions and changes in addition order may introduce
small, justified numerical differences. If such a difference changes a Monte
Carlo acceptance branch, subsequent proposals, saved configurations, solver
termination and final RNG state may differ across implementations/platforms.
These long-run differences are permitted after locating the first numerical
divergence; they do not permit a different RNG algorithm or missing draws on
an otherwise identical control path. Do not reseed or average runs to hide a
kernel defect.

Short controlled runner prefixes retain independent reference checkpoints.
Long prefixes (20 SR steps) instead verify repeatability with the
same input and seed within the implementation: discrete saved state and the
next complete RNG block agree exactly, output shapes and files agree, and
computed output fields use explicit numerical bounds. Independent fixed-input
kernel, solver residual and RNG tests remain the numerical correctness gates.

## Inventory and budgets

The inventory covers parser utility unit tests, parser integration tests,
core internal/runner tests, core integration tests, CLI output tests and
optional Python/Julia reference checks. PfaPack already used numerical golden
bounds; its pivot/layout/status assertions remain exact. xtask has no
floating-point bit comparison gates. Production bit manipulation in SFMT,
range reduction, scaling and cache fingerprints is unchanged.

| Computation | Bound or evidence |
| --- | --- |
| exp/hypot and complex range primitives | 8–32 machine epsilon relative, 2–4 minimum subnormal quanta absolute; a small number of rounded operations and explicit nonfinite recovery |
| trig, small complex products and coefficient initialization/sync | Explicit 8–128 epsilon or `1e-14` absolute/relative budgets near their assertions; parsed literals and copied values remain exact |
| Legendre recurrence/quadrature | Degree-dependent epsilon budgets; endpoint weights include `8 epsilon / (1-z*z)` for denominator cancellation, rather than one uniform loose bound |
| Slater/inverse/rank updates | Dimension/operation budgets, including `512 epsilon` for small supplied systems, plus independent inverse residual checks |
| Individual Green operators | `1e-13` absolute and relative |
| Exhaustive ordered local-energy sums | `1e-12` absolute and relative for up to roughly 4,096 terms; order, signs and duplicates remain covered |
| Three-frame FSZ weighted/factored measurements | `2e-13` absolute and relative; workspace restoration remains exact |
| Sampled SR OO/HO and derivative stores | `1e-12` absolute and relative; Gram reductions also use sample-count-dependent epsilon budgets |
| Fixed direct SR | Matrix/factor/solution regression budgets depend on dimension; independently check the solution against the original unfactored covariance/gradient |
| Fixed CG prefixes | Iteration/GEMV-length epsilon budgets, plus an independently materialized covariance backward-residual check; forced prefix limits and tested termination controls remain exact |
| tenferro SR backend vs C order (issue #421, `crates/mvmc-core/tests/sr_backend_421.rs`) | Real/complex Gram and CG product: `2 gamma_k sum|terms|` per entry, `gamma_k = k eps/(1-k eps)` (`k = samples`, `+3` complex, `n + samples` for CG; the complex Gram sample order is documented in `sr_backend`). S/g assembly: `2 eps` relative (same IEEE operations as C, measured 0). Cholesky solve: `2 * 8 n eps kappa(S)` with kappa from the construction of S, plus an independent residual bound `8 n eps ||S|| ||x||`. Direct-SR short runs: `abs 1e-10 / rel 1e-8`. CG runs are not compared step by step: a one-ulp change of the samples moves the C-order CG solution by ~1e-4..7e-4 (max-iteration CG on the ill-conditioned sampled S), so the test bounds the backend difference by the measured C-order ulp-perturbation spread. |
| Short SR prefixes / long-run repeatability | `1e-11` absolute and relative for computed parameters/energy; short independent checkpoints and long same-implementation discrete/RNG repeatability |
| ComplexUHF Hartree-Fock (`mvmc uhf`) | Iteration counts, indices and headers exact; 10-decimal outputs `abs 2e-10 + rel 1e-10` (print quantum, contraction factor <= 0.9), residual `abs 2e-12 + rel 1e-8`, energy column `abs 1e-10`; orbital files only through eigenvector-gauge-invariant `F F^+` (`abs 3e-5`) and the SFMT noise (`abs 2e-6`); see `docs/COMPLEX_UHF.md` |
| CLI numerical output | `1e-12` absolute and relative, exact indexed coordinates and parameter headers |
| `greenr2k` Fourier transform (vs the compiled Fortran tool) | Printed text layout exact; numeric tokens within one unit of the last printed digit (`E15.5`: five mantissa digits; `F15.10`: ten decimals) plus a `1e-13` absolute floor for values that are zero up to roundoff (sums of at most ~100 terms of magnitude <= 1, about 100 x 2.2e-16). On the Linux fixtures the correlation files agree textually; only the sign of zero in LAPACK's 3x3 inverse differs on the console. Fortran output formats are checked exactly against a gfortran probe. |

These are regression envelopes for the supplied inputs, not forward-error
guarantees for arbitrary ill-conditioned systems or untested BLAS providers.
A new failure requires a first-divergence and residual/conditioning audit
before changing its bound. In particular, multiplying a large condition
number by epsilon and applying that value universally would hide defects.

Independent Mac/Linux historical FSZ fixtures with matching input/control
rows give this evidence:

| Family | Finite components | Maximum absolute difference | Maximum difference divided by `1 + max(abs(actual), abs(expected))` |
| --- | ---: | ---: | ---: |
| Real FSZ setup | 21,312 | `1.42109e-14` | `1.42109e-14` |
| Real FSZ sampling | 544 | `4.75175e-14` | `1.28561e-14` |
| Complex FSZ sampling | 1,496 | `1.06581e-14` | `3.91849e-15` |

The discrete rows, including RNG/configuration/counter rows, agree exactly
for these paired cases. Native Mac scalar helper audits checked 5,013 exp,
10 hypot and 8,292 trig components with zero differences against their
independent fixtures. The nonzero bounds allow roundoff portability without
changing those expected values.

Native C Mac/GNU Green and measurement pairs also have byte-identical inputs
and weights. The complex FSZ one/two-body values differ by at most
`1.77636e-15`, ordered energy sums by `8.88178e-15`, weighted measurements by
`7.10543e-15` and normalized measurements by `3.55271e-15`. Real FSZ outputs
and the factored measurement outputs agree numerically. These independent
pairs support the short-kernel and accumulation budgets in the table.

For 51 archived fixed direct SR systems, the largest infinity-norm condition
number is about `4.03e7`; the largest expected normalized backward error is
`3.99e-17`. Fixed CG examples have condition numbers up to `4.83e9`. Tight
forward regression comparisons therefore accompany residual checks instead
of using a broad `condition * epsilon` allowance. The Rust direct test checks
`||S*x-g||inf / (||S||inf*||x||inf + ||g||inf) <= 128*n*epsilon`.

## Output, fixture storage and optional oracles

Computed output fields are parsed and compared numerically. Row/column
counts, declared integer/index fields, headers and text remain exact.
Dedicated writer-format tests still check formatting. SR diagnostics print
five digits after the decimal point: their numerical comparison allows one
last printed quantum (`1e-5` relative), while dimensions, cuts, indices and
tested iteration counts remain exact.

Hexadecimal float fixture encoding remains lossless storage. It does not
imply a bitwise numerical gate. Source hashes and the SHA-verified reuse of
unchanged fixture artifacts remain exact integrity checks.

Thirty native FSZ matrix/Gram files formerly retained only regenerated-output
SHA checks. Their independent #186 payloads were recovered from retained
macOS/Linux reference outputs, verified against all original digests and
stored as deterministic `.txt.gz` files: 104,144,213 original bytes become
16,375,954 compressed bytes. No numerical values were regenerated by Rust.
Optional checks validate the decompressed stored artifact's SHA and then
compare the newly computed fields numerically, including matrix/solution
residual checks. The digest no longer gates newly computed floating-point
bytes. Optional Julia checks need `gzip`; ordinary Cargo tests do not read
these compressed files or invoke reference runtimes/toolbox programs.

Adversarial comparison tests reject wrong signs, excessive errors, unexpected
nonfinite values, missing components/columns and changed discrete fields.
Both native default and all-feature nextest suites use `test-fast`; optional
ignored tests are included in the all-feature gate. Keep the separate
formatting, Clippy and documentation checks.

## C-order real kernels and CG amplification (#358)

The ordinary real Pfaffian path (`dsktf2`, `utu2inv_real`, every backend)
follows C's operation order: `DSKR2` evaluates `(A + X*t1) - Y*t2`, the
skew-tridiagonal solve divides directly, and for `n > 64` the 64-column
panel product with skew restoration runs before the permutations. Julia's
grouping is kept only as a `#[cfg(test)]` witness of the difference.

**Observation.** After the switch eight families (real CG and direct SR,
hubbard, opttrans, dh2/dh4/dh24, pairhop, rbm real) stopped matching the
Julia-order trajectories at the old 1e-11 bound, by 1e-10..1e-5 in step-1
parameters. A real C `vmc.out` (Linux x86_64, GCC 13.3, OpenBLAS 0.3.26, see
`tests/fixtures/c_order_sr_operands/PROVENANCE.md`) run on the same inputs and
seed shows the converse: C's step-1 parameters also differ from the C-order
Rust parameters (about 1e-4) and from the Julia fixtures (up to 6.4e-3, steps
1/2/3: 6.4e-3, 5.7e-3, 5.4e-3), although step-1 energy, configurations and RNG
are identical.

**First divergence (`heisenberg_chain_real`, step 1, NSRCG=1).**

- Sampled operands agree to the last bit or two. Against the instrumented C,
  `<HO>` and `<O>` differ by at most 6e-16 and the S mean/diagonal/OO array by
  at most 3e-16 in `|d| / (0.1 + |c|)`; e.g. S mean entry 2.08238631462278362e-1
  (C) versus 2.08238631462278390e-1 (Rust). This is the summation order of the
  sampled sums, not a defect.
- The CG solve is the amplifier. `nSmat = 10`, so `max_iter = 10`, and the
  threshold `tol^2 * nSmat^2 = 1e-18` is never met before the iteration cap.
  C's squared residual norm is non-monotone: 5.8e-5, 5.4e-7, 2.4e-16, 8.0e-17,
  1.6e-11, 6.8e-18, 6.2e-15, 4.0e-16, 2.3e-18, 1.0e-18. Cancellation to 1e-16
  magnifies last-bit differences: Rust and C differ by 1e-16 (initial delta),
  5e-15 (iteration 1) and 1e-10 (iteration 2, relative) before the 10-iteration
  stop, giving parameter differences of 1e-4..1e-2.
- Direct SR (Cholesky of the shifted S) is well conditioned for most families:
  hubbard, dh and pairhop direct prefixes still match Julia at 1e-11.

**Policy for these families** (`callback_tests::c_order_amplified_family`):

- Step 1 sampled operands (`<O>`, `<HO>`, S/OO, stored O) against native C at
  `abs = 1e-14`, `rel = 1e-13` (about 100 times the observed 6e-16, covering
  other BLAS/compiler summation orders on macOS ARM), plus step-1 energy
  (`1e-12` against C, `1e-11` against the Julia fixture) and the S-diagonal
  extrema/size columns of `zvo_SRinfo.dat`.
- Configurations, occupancy, projection counts and the next 624 RNG words
  stay exact against the existing fixtures at steps 1, 2 and 3 (they are
  unchanged), including one-step-ahead acceptance decisions. The single
  exception is `pairhop_real` step 3: the amplified step-1/2 updates flip one
  Metropolis acceptance, so that prefix is repeatability-only.
- The parameter, energy (step >= 2), `zvo_var`, `zqp_*` and window outputs of
  these runs are not compared with a reference. They are checked for
  same-implementation repeatability (parameters, energy and configurations
  identical on rerun), for finite values and for a nonzero SR update.
  Window aggregation is covered by `ctest_window_fixtures`.
- The old Julia fixtures in `tests/fixtures/sr_cg`, `sr_direct` and
  `runner_opt_windows` remain as **historical Julia-order references**, used
  only for the exact sampling and step-1 checks above and for families not
  affected by this change.
- The CG solve itself is verified at fixed operands against the C recurrence
  (`c_toolbox/ctest_cg_refresh.c`, `tests/fixtures/sr_cg/c_refresh`,
  `real.txt`/`complex.txt`).
- `rbm_real` has no native C operand reference because native C ignores RBM in
  real mode (#379, tmisawa/Julia-mVMC#59). `vmcmake_real.c` has no RBM code
  (the complex `vmcmake.c` has 27 mentions) and C accepts `FlagRBM=1` for real
  models without a message. On `c_orbital_inputs/namelist_rbm_real.def` (seed 1,
  step 1) C has NPara = 55 (NProj 7, NRBM 36, NSlater 12) and energy
  5.984544891656925, identical for RBM overlay 0.125/-0.25 and for all zeros, and
  bit-identical to Rust with all RBM values zero. Rust applies the RBM weight
  (6.228711216019723); initial parameters agree. Rust keeps the correct math and
  the Julia-derived step-1 energy, sampling and SRinfo checks.
- `opt_real`: C's real-mode OptTrans derivatives are written through a complex
  pointer offset (`vmccal.c`, `calculateOptTransDiff(SROptO + 2*NProj + ...)`),
  so for `NQPOptTrans > 1` derivative 1 is dropped, derivative 2 lands in
  slot 1 and the last slot stays zero. Rust and Julia keep the mathematical
  layout; the last two entries per operand are excluded from the C comparison
  (decision in #370: keep the correct layout; C defect reported as
  tmisawa/Julia-mVMC#55). `real_mode_opttrans_derivatives_use_their_own_slots`
  pins the correct layout via `sum_i w_i O_i = 1`, which C's layout violates.
  Entries before them agree at the bound above. Complex mode was not examined.

**`golden_vs_julia`** is a historical Julia-order comparison. The real LTL
bound is `2 * n * eps * max|entry|` (observed at most `0.6 * n * eps`, for
`n = 4..256`); see the test header for the table and the unchanged complex and
inverse bounds. C-order correctness is covered by the hand-dyadic and analytic
tests (`rank2_real_regression.rs`, `issue176_ordinary_panel.rs`).

## C `-b` varbin parameter blocks (#347)

C's `-b` output (`vmcmain.c:658`) is `fwrite(Para, sizeof(double), NPara, FileVar)` on a
`double complex` array, which writes only `NPara` doubles: the first `ceil(NPara/2)`
parameters (the last cut to its real part for odd `NPara`). Rust deliberately differs and
writes all `2*NPara` doubles per block under the same header. The comparison against the
unmodified C `vmc.out` therefore uses:

- header (`NPara`, step count): bytes exact;
- block data: C's `NPara` doubles against the leading `NPara` doubles of Rust's block. PhysCal
  blocks (file-read or `InitParameter` values, no accumulated arithmetic) and the optimizer's
  step-0 block are compared exactly; later optimizer steps depend on the SR solve (BLAS
  operation order, #358) and use an absolute bound of `1e-8` (observed maximum about `5e-10`,
  parameters O(1)) with RNG-driven control paths identical;
- completeness: the Rust block must contain every parameter of the text `zvo_var` output.

Evidence and the C run: `tests/fixtures/issue347_varbin/PROVENANCE.md`.

## Accelerated-backend validation (#424)

An accelerated implementation of a tensor-shaped stage (tenferro on CPU or CUDA, a batched
Pfaffian kernel) changes summation order. It is validated against the C-order CPU path stage by
stage, never against long Monte Carlo averages, and never bitwise. The harness is
`mvmc_core::accel_validation` (module docs describe the model); the backend is the production `StageBackend` object (issue #437: stage traits `SrStages` and
`PfaffianStages` composed into one object, the same one `stage_backend::acquire` returns). The
C-order oracle is `StageBackend::c_order()` (the production PfaPack sequence and scalar loops).
Variants: tenferro CPU runs in normal CI; the CUDA variant (`open_stage_backend(Cuda(0))`)
runs in the optional gate (`scripts/run_cuda_gate.sh`, see `docs/design/gpu-readiness.md`
section 10). A stage a backend does not implement reports `Unsupported` and is listed as not
compared; it is never replaced by the CPU result. Today tenferro 0.7.1 provides `S` and `g`
through `dot_general` but no Pfaffian, so the Pfaffian stages are exercised against `COrderPfaffian`
and against deliberately perturbed backends that prove the detector works; the batched
Pfaffian API of #423 plugs in by implementing `PfaffianStages`.

**Teacher-forced replay.** The oracle drives a Metropolis trajectory (its decisions advance the
configuration and consume the RNG). At each step the backend under test evaluates the same
candidate and current configurations, so differences never compound through diverging
trajectories. The report gives, for each of `pf`, `invM`, the acceptance weight
`(pf_new / pf_old)^2`, the O store, `S` and `g`: the number of entries compared, the maximum
absolute and relative deviation and the number of entries outside the bound. The bound is
`|a - b| <= abs + rel * max(|a|, |b|)` per quantity (`Tolerances`; defaults `abs = 1e-12`,
`rel = 1e-10`, justified by `n = 6` electrons, O(1) entries, a Pfaffian condition number below
about 1e3 and at most a few hundred summed terms, so reordering errors are O(10) eps with a
margin of about 1e4 that still catches a layout, sign or dtype defect). Tolerances are chosen
from the measured first divergence for a new backend and must not be raised to hide a defect.

**Decision recording and flips.** Each proposal records the oracle weight, the backend weight,
the draw `u`, both decisions (`u < w`), the margin `|w_oracle - u|` and the weight error
`|w_backend - w_oracle|`. A flip needs `u` between the two weights, so its margin cannot exceed
the weight error unless the decision logic itself differs. A flip is a *defect* when its margin
is larger than the measured weight error or larger than the weight bound the tolerance allows
(`abs + rel * w`). A flip inside both is a legitimate consequence of reordering and is only
counted. The report also lists the smallest margin of the run and how many proposals were
inside the allowed weight bound (a flip was possible). Teacher forcing keeps the trajectory
identical after a legitimate flip, so no long-run divergence needs to be explained.

**RNG contract.** Each proposal draws the electron, the empty-site index and the acceptance draw
(always three draws, also for rejected proposals). The draw sequence, the final configuration
and the final RNG state are independent of the backend under test; a test asserts that a backend
returning wrong weights does not change RNG consumption. RNG state is exact, as in the rest of
this document.

**Repeatability.** `repeatability` runs the same backend twice with the same input and seed for
20 steps by default. Moves, decisions, draw bits, the final configuration and the final RNG
state agree exactly; weights, `S` and `g` agree within the bounds; the deviation classification
agrees. This is the same-implementation criterion of the 20-step long-run rule above.

**Benchmark metadata.** Every benchmark or validation result carries a `BenchMetadata` block
(following tenferro-decision-rs `docs/agents/specs/docs/05_TESTING_BENCHMARKS.md` section 5):
source revision, CPU model and available threads, GPU and driver/CUDA/cuBLAS/cuSOLVER versions
(`none` on CPU), OS and architecture, rustc, tenferro version, provider, thread settings
(`RAYON_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`), dtype, batch size, warm-ups,
iterations and whether upload and download are timed. The stage benchmark `bench_stages` reports
the median of 30 iterations after 5 warm-ups (`DEFAULT_ITERATIONS`, `DEFAULT_WARMUPS`) with
upload and download inside the timed region for device backends, and `None` (not 0) for a stage
that is unsupported. Use the same thread count on both sides of a comparison.

Reference environment: Linux x86_64, as above. GPU results are labelled with the metadata block
and are an optional gate; ordinary Rust tests (including the CPU variants of this harness) run
without a GPU, `c_toolbox/` or any C/Julia oracle program.

## Multi-walker optimization reductions (#435)

`run_para_opt_multichain` reproduces the ungrouped (`NSplitSize = 1`) multi-rank C runs of
`tests/fixtures/mpi_matrix_179` with in-process walkers and a `Reducer` that replaces the MPI
communicator. Per walker the counters, saved configurations and the complete SFMT state are
compared exactly (the RNG contract is independent of the reduction); the reduced `<HO>`,
`<OO>`, `<O>` and the step energy use `|a - b| <= 1e-13 + 1e-12 |b|`. The first numerical
divergence from C is the order of the cross-walker sum: the Rust reducer is a rank-order left
fold, `MPI_Allreduce` leaves the order to the MPI library; for two walkers it is a single
two-term sum and agrees with C to the last bit or within roundoff, for four walkers to the
bound above (observed worst case well inside it in every cell). `W = 1` is byte-identical to the
serial optimization. Beyond the first, well-conditioned step the SR solve amplifies reduction
roundoff (#358): trajectories are checked for bitwise repeatability (thread scheduling does not
change the rank-ordered sums), not forced onto C.
