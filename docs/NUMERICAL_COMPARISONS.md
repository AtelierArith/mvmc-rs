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
| Short SR prefixes / long-run repeatability | `1e-11` absolute and relative for computed parameters/energy; short independent checkpoints and long same-implementation discrete/RNG repeatability |
| CLI numerical output | `1e-12` absolute and relative, exact indexed coordinates and parameter headers |

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
- `rbm_real` has no native C operand reference: its Rust model is the
  historical sparse RBM control, which C does not run identically
  (step-1 energy 5.9845 in C, 6.2287 here). It keeps the Julia-derived
  step-1 energy, sampling and SRinfo checks.
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
