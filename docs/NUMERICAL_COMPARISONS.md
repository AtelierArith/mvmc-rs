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

## C operation order of FMA and lane-split reductions (#449)

C is authoritative for operation order (AGENTS.md). A default C build (x86-64 baseline, no
`-march`, no FMA contraction) evaluates `tmp += a * b` as two separately rounded operations in
sequential order; Julia's `@turbo`/LoopVectorization uses lane-split partial sums with fused
multiply-adds. Rust never contracts `a * b + c`, so a plain loop is bit-identical to default C.

**`two_hop_bilinear_real`** (the `b^T M a` form of the two-electron Pfaffian ratio, used by the
two-body Green functions and the sampler's two-hop proposals). C counterpart:
`calculateNewPfMTwo_child_real`, `extern/mVMC-1.3.0/src/mVMC/pfupdate_two_real.c:167-177`
(`tmp += invM_i[msj] * vec_a[msj]` sequential in `msj`, then `bMa += vec_b[msi] * tmp` in `msi`
order). The Rust default is now exactly this order (it was Julia's four-lane, six-accumulator
FMA tree, plus a #444 hardware-FMA dispatch). Speed is kept without changing a single
association: four rows are evaluated side by side as four independent sequential chains and
`bMa` is then accumulated in row order, so the result is bit-identical to the plain nested loop.
Independent check: `c_toolbox/two_hop_bilinear_449` builds the C loop without FMA and stores the
bits of 132 cases (`tests/fixtures/pfaffian_cg/two_hop_bilinear_c.txt`); the Rust kernel
reproduces all 132 bitwise (`two_hop_bilinear_is_bit_identical_to_the_c_reduction`). A C build
with FMA contraction (`-O3 -march=native`) differs from the no-FMA build in 50 of the 132
cases, so the contract is "default C build". The archived Julia values
(`two_hop_bilinear.txt`) are now only a historical comparison with a reordering bound.

**Audit of the other Julia-style reductions** on the parity path:

| site | C counterpart | Julia style | action |
|---|---|---|---|
| `sampling/updates.rs` `two_hop_bilinear_real` | `pfupdate_two_real.c:167-177` | lane-split, FMA | fixed (above) |
| complex `CalculateMAll` of the sampler's periodic recalculation and the PhysCal refresh (`pfapack` `zsktf2_turbo`: FMA rank-2 update, reciprocal pivot) | `ZSKTRF` order (`zsktf2_c_compat`) | FMA, Julia pivot inverse | default is now the C order (`calc_m_all_complex_production`); Julia's kernel is the opt-in `pfaffian::use_julia_complex_kernel()` (thread-local guard) used only by the archived Julia 1.11 SR prefix tests |
| `sr_backend.rs` real Gram `O O^T` | `calculateOO_Store_real`: `DGEMM('N','T')` for every size (`vmccal.c:685-689`) | `DSYRK` for `max(n, samples) >= 4`, `muladd` loop below (Julia `mul!` dispatch) | now `DGEMM('N','T')`, full matrix; the Julia small-case `mul_add` is gone |
| `c_complex.rs` `divide` (`mul_add` on macOS aarch64) | native ARM compiler-rt contraction | C behaviour on that target | kept (it is the C-faithful emulation) |
| `mvmc-stdface/ccomplex.rs` `x2y2m1` | glibc `__x2y2m1` exact error-free sum | none (C emulation) | kept |
| `mvmc-expert-parsers/utils/julia_{trig,log,exp,hypot}.rs` polynomial `mul_add` | glibc libm | emulation of Julia's libm | replaced by the platform libm in #457 (below); Julia's implementations are an explicit opt-in |
| `pfapack` `simd-backend` (`mul_add_c64s`) | none (SIMD) | opt-in feature, not default | kept, documented opt-in |
| tenferro SR backend (`MVMC_RS_SR_BACKEND=tenferro`) | BLAS/LAPACK | different GEMM/Cholesky order | kept, documented opt-in |

**Effect on the native-C fixtures.** The 69 PhysCal tests of `native_c_physcal_181` (566 output
files) pass with the unchanged bounds before and after all three changes. 137 of the 566 files
(35 scenarios) change at the last bits, in both directions (for example complex-mode
`zvo_out` entries near zero move by about 4e-16 absolute); the Lanczos-sensitive entries,
amplified by the condition number of `alpha`, dominate the changes: the largest absolute
deviation from C over the changed files is 5.1e-9 after against 1.0e-8 before (sum 3.4e-8
against 5.4e-8), so the C order is not worse and slightly closer to C. The tolerances cannot be
tightened on this evidence: the dominant bounds are set by the `alpha`-amplification
(`1e-7`/`1e-6`, measured up to 8e8) and the FSZ one-configuration sensitivity, both unchanged by
this issue. Against Julia the observables now differ at roundoff level instead of being
bit-equal (Hubbard L16/L24/L32 `zvo_cisajscktalt` max|diff| 2.1e-15 / 3.1e-14 / 2.5e-13, energy
|dE| = 0), which is the expected signature of leaving Julia's reduction order.
The `heisenberg_chain_fsz` PhysCal record and the SR-CG/direct-SR prefix tests pass on this
Linux host before and after (OpenBLAS 0.3.26, reference LAPACK 3.12; 1538 of 1538 on `main`);
the 16 failures reported in #444 were not reproducible here (see the PR description).
Measurement aid:
`MVMC_RS_REPORT_MAXDIFF=1 cargo nextest run -p mvmc-cli --test issue181_native_c_physcal
--no-capture` prints the largest absolute and relative difference per output file.

## Platform libm instead of Julia's math functions (#457)

C mVMC calls the platform libm (`exp`, `log`, `sin`, `cos`, `cabs`/`hypot`, `atan2`, ...).
Julia ships its own software implementations, which agree with glibc to within one ulp but not
bit for bit. The parity path now uses the functions C uses: `mvmc_expert_parsers::utils::c_math`
maps each to the Rust `f64` method of the same name, which is the platform libm (glibc on Linux,
the system libm on macOS). Julia's implementations (`julia_exp`, `julia_log`, `julia_trig`,
`julia_hypot`) are kept only as the explicit, process-wide opt-in `c_math::use_julia_libm()`
(guarded and serialized), used by the archived Julia SR prefix tests next to
`pfaffian::use_julia_complex_kernel()`.

| helper | callers on the parity path | C call site |
|---|---|---|
| `exp` | Metropolis weight `sampling/metropolis.rs:48`, FSZ real accept `sampling/fsz_real.rs:452`, projection ratio and Green ratios `observables.rs` (`c_exp`), `observables/fsz_green.rs:353,358`, RBM `cexp` real part `sampling/rbm_math.rs` | `w = exp(2.0*(x + logIpNew - logIpOld))` `vmcmake_real.c:176,251,496,560`, `vmcmake.c:194,278,721,783`, `vmcmake_fsz*.c`; `ProjRatio` `projection.c:56`; `cexp` in `rbm.c` |
| `log` | `LogIP` `sampling/driver.rs:72`, `observables.rs:175`, RBM `clog` real part | `log(fabs(ip))`; `clog` in `rbm.c` |
| `log1p`, `tan`, `atan2`, `sinh` | Julia complex algorithms only under the opt-in (the default RBM path uses `glibc_complex`, #470) | `clog`, `ctanh` in `rbm.c:345-369` |
| `sin`, `cos` | `qp_weight.rs` (Gauss-Legendre nodes and `SPGLCos/SPGLSin`), `parameter_init.rs:105-106` | `gauleg.c:42`, `qp.c:67-73`; RBM initialization `cexp(2 I pi u)` `parameter.c:53` |
| `hypot` | `sync.rs:91` and `parameter_init.rs:252` amplitude normalization | `cabs` `parameter.c:155-167` |

The complex RBM functions were handled in #470 (below): they now follow glibc's algorithms.

**Effect on the fixtures.** The 69 PhysCal tests of `native_c_physcal_181` pass with unchanged
bounds. Only 8 of the 566 output files change at all (scenarios `spin_chain_lanczos1/2`,
`hubbard_chain_dh_rbm_opttrans`, `fsz_dh24_rbm_opttrans_zero_physcal`), at the last bits: the
largest absolute deviation from C over the changed files is 3.525e-12 before and 3.526e-12 after
(the Lanczos-amplified entry), the others move by 1e-17 to 2e-14. Against Julia (Hubbard
L16/L24/L32 PhysCal, `zvo_cisajs`) the observables were bit-equal after #449 and now differ at
1e-17 to 1e-16 (max relative 3e-15 to 2e-14), `zvo_cisajscktalt` unchanged (2.1e-15, 3.1e-14,
2.5e-13), energies |dE| = 0, i.e. the one place where the libm emulation still gave exact Julia
equality. The four `physcal_ref` fixtures stay `ok` against Julia. The archived Julia SR prefix
tests need the opt-in: six of them fail without it (RBM and OptTrans trajectories embed Julia's
`exp` rounding).

**Platform caveat.** glibc and the macOS libm are different implementations; a quantity that
depends on the last bit of a transcendental function (an acceptance decision at a razor-thin
margin, an SR step amplified by ill-conditioning) can differ between Linux and macOS, and the
macOS-versus-Linux difference is of the same size as the Julia-versus-glibc difference this
section removes. Linux (glibc, the platform of the C reference outputs) is the numerical
reference; tests compare computed values within the explicit bounds above on both platforms and
keep RNG state, draw counts and the decisions on identical control paths exact.

## glibc complex functions on the RBM path (#470)

Complex elementary functions differ between libraries in more than the last bit (branch cuts,
overflow scaling, cancellation handling), so the Julia complex algorithms that `rbm_math` used
were a C-parity gap, not just roundoff. The C RBM code calls `cexp`, `clog(ccosh(z))` and
`ctanh` (`rbm.c:41,44,58,94,118,122,155,179,345,357,369`). `mvmc_expert_parsers::utils::
glibc_complex` ports glibc 2.39's `cexp`, `clog`, `ccosh`, `ctanh` and `__x2y2m1` line by line
(templates `math/s_c*_template.c` and `sysdeps/ieee754/dbl-64/x2y2m1.c`; version, tag object,
SHA-256 and extraction boundary in `c_toolbox/glibc_complex_470/README.md`); the RBM weights now
evaluate `clog(ccosh(z))` instead of Julia's sign-flipped `log1p` form. Julia's complex
algorithms remain only behind `c_math::use_julia_libm()` (the archived Julia prefix tests).

* **Check:** `c_toolbox/glibc_complex_470/probe.c` (a standalone kernel check, not a full
  `vmc.out` run) evaluates the four C functions on 1600 inputs (all combinations of signed zeros,
  +-1, +-inf and NaN; large `|Re z|` around the overflow thresholds 354/709/708/2127; the
  negative real axis with tiny, subnormal and signed-zero imaginary parts; every `clog` branch
  including the `x2y2m1` path and the `DBL_MAX`/`DBL_MIN` scalings; RBM-like hidden values).
  On Linux x86_64 with glibc 2.39 all 6400 results of the Rust port are **bit-identical** to the
  C library (`ports_match_the_glibc_probe_fixture`).
* **Native-C RBM fixtures:** the 69 `native_c_physcal_181` tests pass with unchanged bounds; only
  4 of 566 output files change (`hubbard_chain_dh_rbm_opttrans`), at the last bits; the largest
  absolute deviation from C over the changed files drops from 2.1e-14 to 1.1e-14.
* **macOS:** the port shares glibc's algorithm but its building blocks (`exp`, `log`, `log1p`,
  `sincos`, `sinh`, `cosh`, `atan2`, `hypot`) are the macOS libm, which differs from glibc in the
  last bits; Apple's own `cexp`/`clog`/`ctanh` are different algorithms again. On macOS the probe
  fixture is compared to 16 eps per component (plus four subnormal quanta); exact equality is a
  Linux/glibc contract. glibc also selects different `exp`/`log`/`sin`/`cos` code paths (ifunc)
  on CPUs without FMA; the fixture records the generating CPU.

## BLAS provider and kernel matrix (#455)

PR #444 reported 16 local failures (SR-CG and direct-SR prefix tests and the
`heisenberg_chain_fsz` PhysCal record) that CI did not show. They reproduce exactly, and only,
when the linked BLAS does not use the arithmetic of the kernels the archived references were
generated with: the 16 tests below fail on every OpenBLAS kernel except Haswell (and Zen, which
selects the same kernels), on reference BLAS/LAPACK, and on the GitHub Linux runner when
`OPENBLAS_CORETYPE` is not pinned (AMD EPYC 9V74, OpenBLAS 0.3.26 selects `Cooperlake`). The main
CI pins `OPENBLAS_CORETYPE=HASWELL` (Linux) and `NEOVERSEN1` (macOS ARM64), which is why it is
green. Thread count is not a factor.

**Matrix** (Linux x86_64, Xeon E5-2699 v3, Ubuntu OpenBLAS 0.3.26 pthread, LAPACK 3.12;
`mvmc-core` + `mvmc-cli` with `--cargo-profile test-fast`, 1052 tests, same source otherwise).
"before" is `main` at `640213dd` (strict bounds everywhere), "after" is this change.

| provider / kernel | threads | before | after |
|---|---|---|---|
| OpenBLAS Haswell (native) | 1 / 2 / 36 | 1052 pass | 1052 pass (strict bounds) |
| OpenBLAS Zen kernel set (`OPENBLAS_CORETYPE=Zen`) | 1 | 1052 pass | 1052 pass with the strict bounds forced (`MVMC_BLAS_KERNEL_CLASS=reference`) |
| OpenBLAS Sandybridge | 1 / 36 | 16 fail | 1052 pass |
| OpenBLAS Nehalem | 1 | 16 fail | 1052 pass |
| OpenBLAS Prescott | 1 | 16 fail | 1052 pass |
| Reference BLAS + LAPACK (`LD_PRELOAD`, `MVMC_BLAS_KERNEL_CLASS=unverified`) | 1 | 16 fail | 1052 pass (the PhysCal energy record needed its own relaxation, see below) |
| Sandybridge with the strict bounds forced | 1 | 16 fail | 16 fail (the same 16: the gating, not a change of arithmetic, removes them) |
| GitHub Linux x86_64, `OPENBLAS_CORETYPE` unset (EPYC 9V74, core `Cooperlake`) | default | same 16 | see the CI matrix below |
| macOS ARM64 `NEOVERSEN1` (main CI, Homebrew OpenBLAS, overlay references) | 1 | pass | pass (strict bounds) |

The four extra `mvmc-cli::issue348_multidef` failures in the first run of the optional workflow
came from `OPENBLAS_VERBOSE=2` writing the kernel name to the stderr those tests compare; the
workflow no longer sets it.

**Kernel class.** `tests/support/julia_fixture.rs::kernel_class` returns `Reference` for Linux
x86_64 OpenBLAS `Haswell`/`Zen` and for macOS ARM64 cores that have an overlay under
`tests/fixtures/macos_arm_julia/<core>/`, `Unverified` otherwise (including a provider that does
not export `openblas_get_corename`, such as Accelerate; the core name is resolved with `dlsym`,
so test binaries link against such providers). `MVMC_BLAS_KERNEL_CLASS=reference|unverified`
overrides the detection (reference BLAS under `LD_PRELOAD` still reports the OpenBLAS core).
The strict bounds are unchanged on `Reference` kernels; the first divergence of every
`Unverified` case is below.

**First divergence and classification** (all "tolerance too tight for a legitimate provider
difference"; no defect was found, the backward-error and invariant checks pass on every kernel):

- `sr_cg::cg_fixed_input_matches_c_through_residual_refresh`. Operands are dyadic, so the first
  divergence is the GEMV of iteration 2 (limit 2, max |dx|/|x| 3.1e-16, one ulp of the FMA versus
  non-FMA reduction); Haswell reproduces the C fixture with 0 difference at all 41 limits. The
  recurrence then grows the error about 1e4 per two iterations from iteration 5 (limit 5 2e-16,
  6 1.2e-15, 7 1.8e-14, limit 13 2e-2 to 5e-2, up to 0.5 at limit 41; the same on Sandybridge,
  Nehalem, Prescott). This is the #358 amplification of a non-converging CG (`tol = 0`, 41
  iterations on a 32 x 48 operand). On `Unverified` kernels the forward comparison covers limits
  <= 5 (ratio to the budget <= 0.02); the explicit-residual (backward error) check, with its
  own budget, runs at all 41 limits on every kernel.
- 12 `callback_tests` CG/direct prefix tests (complex, FSZ, general, interall, dh2, dh4/dh24,
  pairhop, opttrans, rbm real/complex/fsz/dh24, canonical general RBM). Sampling operands, step-1
  energy and RNG agree; the step-1 parameters differ by 4e-11 (complex, step 1) to 1.9e-5
  (rbm real/complex) after the CG/Cholesky solve. These are the cases #358 already handles with
  the amplified-family policy. On `Unverified` kernels every family now follows that policy
  (sampling checkpoints, step-1 energy, S-diagonal SR diagnostics; parameter trajectory by
  repeatability, finiteness and a nonzero update). The SR iteration count (column 8) is not
  compared there (e.g. pairhop_fsz 142 versus 140 iterations).
- `sr::tests::sampled_direct_sr_matrix_gradient_factor_and_solution_match_julia`. Matrix, gradient
  and Cholesky factor agree at the strict bound; the first divergence is the solve (real/store0
  solution[1] 1.8e-11 versus the 7.1e-13 bound; kappa_1(S) up to 4e7). On `Unverified` kernels the
  solution uses the componentwise Skeel bound `4 n eps (|S^-1| (|S||x| + |b|))_i` built from the
  original matrix; observed worst error / bound over the 49 cases is 4e-3 (Sandybridge), 8e-3
  (Nehalem), 4e-3 (Prescott). The existing backward-error check is unchanged.
- `physcal_issue181::two_sample_runners_match_independent_saved_states_rng_and_ordered_outputs`.
  Only `heisenberg_chain_fsz`: weights, one-body and factored arrays agree at 1e-12/1e-10; the raw
  two-body `direct` array of the sample differs by up to 6e-11 absolute / 1.9e-9 relative
  (Sandybridge 3e-10, Nehalem and Prescott 1.9e-9, reference BLAS 1.3e-9) and, on reference BLAS
  only, the accumulated FSZ energy by 7e-11 / 1.8e-10: ulp-level differences of the complex
  inverse amplified by the cancelling Green sum (the "one-configuration sensitivity" of #449).
  On `Unverified` kernels those two records use `abs 1e-9, rel 1e-7` (at least 16 times the
  observed deviation); Haswell reproduces all of them at the strict bound and the
  independently generated native-C Green fixtures keep the strict bound on every kernel.

Reproduction: `OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=Sandybridge cargo nextest run -p mvmc-core
-p mvmc-cli --cargo-profile test-fast --no-fail-fast`; add `MVMC_BLAS_KERNEL_CLASS=reference` to
see the 16 failures again. The optional `BLAS matrix` workflow (`workflow_dispatch`) runs the
suite on macOS ARM64 with Accelerate, with Homebrew OpenBLAS (native kernel, pinned and default
threads) and on Linux with the native kernel.
