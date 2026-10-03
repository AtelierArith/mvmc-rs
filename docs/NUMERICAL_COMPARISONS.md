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
