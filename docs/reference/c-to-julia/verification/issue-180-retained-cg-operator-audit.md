# Retained CG operator audit (diagnostic, not acceptance)

The optional offline program `c_toolbox/ctest_audit_mpi_cg_operator.jl`
materializes the operator from the actual per-rank retained sample arrays,
global mean, stored diagonal, gradient, and weight. It consumes original Julia
and Rust captures; it does not generate expected Rust increments. Normal Cargo
tests neither read nor execute this program.

## Authoritative recurrence

Upstream: `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c`, SHA-256
`41452de5fe766409431c6e1cf73cd2b12485faaeb4bfbcdec1e53444e9af1cf9`.
The source was inspected through CodeGraph, without modifying vendored code.
Lines 258–350 define the original CG main loop; lines 356–425 define its
operator. In particular:

- Default iteration limit is the active dimension (line 262).
- Convergence is strictly `delta < tol*tol*n*n` (265, 306).
- `alpha = delta / xdot(d,q)` has **no tiny-denominator guard** (310).
- Residual refresh occurs every 20 iterations (319–330).
- `beta = xdot(r,r)/delta`, followed by `delta = beta*delta` (333–336).
  Assigning the new dot product directly is not the same floating-point order.
- The operator uses transposed then normal sample GEMVs, global reduction,
  `invW*z - coef*mean`, then `diag*shift*x` (392–420).

The older Julia/Rust guard and residual-norm assignment are not C-parity
expectations. Production migration and the separate C kernel probe belong to
the CG kernel owner; this audit does not change that kernel or any tolerance.

## Actual execution

Session **14439**, terminal exit **0**, output chunk **9436b8**. Linux x86_64
container `73c57e563c61`, Julia 1.13.1; one Julia and OpenBLAS thread.
This is the retained **old-algorithm** world-4 real/store-0/prefix-3/width-1
cell, not a run of the subsequently repaired C-faithful kernel.

Full actual terminal stdout is preserved in
`evidence/issue-180-cg-audit-14439.stdout.txt`, SHA-256
`9aa1472873544a84b19b38b078bda24a2fc0128eb3f35e43e9841e293331f521`.
Exact executed source and helper are separately preserved as
`evidence/issue-180-cg-audit-14439.jl` and
`evidence/issue-180-cg-audit-14439-helper.jl`; their hashes match the executed
files listed below. For reproduction using the archive, preserve the original
include filename `ctest_direct_sr_capture.jl` beside the archived script.
CG captures contain no LAPACK INFO/success status: the operand-only audit
does not synthesize one. Its successful terminal status means the diagnostic
completed, not that the captured CG solves satisfy an acceptance bound.

```sh
docker exec \
  -e JULIA_DEPOT_PATH=/home/vscode/.cache/mvmc/julia-depot \
  -e OPENBLAS_NUM_THREADS=1 -e JULIA_NUM_THREADS=1 \
  73c57e563c61 \
  /home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia \
  --compiled-modules=existing \
  --project=/home/vscode/.cache/mvmc/snapshots/issue179-sr.WbN8B8/extern/Julia-mVMC \
  /workspaces/mvmc-rs/c_toolbox/ctest_audit_mpi_cg_operator.jl \
  /home/vscode/.cache/mvmc/issue179-cgref.yqrDc0/matrix/r4-real-s1-cg1-store0/prefix3/julia \
  /home/vscode/.cache/mvmc/issue179-cgfix-world4.iJRyvh/r4-real-s1-cg1-store0/prefix3/w1/rust \
  /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/r4-real-s1-cg1-store0/inputs/modpara.def
```

Executed script SHA-256:
`1a9b3f64ffb49450100be422200320f284961c08b50884774a040e60e3e63931`.
Shared diagnostic helper SHA-256:
`440cbcb61b58663775fb7e69c52f71464bdd7ee3f4fddce7ebdeb79fe43eb7c1`.
Input modpara SHA-256:
`53a029d196e551ee188bd66f637932ba7b7426482efa695d5c1281f1362610b3`.
The helper's new operand-only diagnostic entry point does not alter original
Julia POTRF/POTRS capture hooks. The older direct-SR audit's executed helper
is separately preserved in its evidence archive; its hash is not this hash.

The audit rejects duplicate keys, invalid shapes/nonfinite operands and
inconsistent rank metadata. It checks this named real cell's input headers,
NSRCG=1, NStore=0, three samples/rank, global Wc=12 and shift=1e-5.
The reconstructed covariance/operator and residual arithmetic use 256-bit
BigFloat. Eigenvalues, condition and covariance-rank estimates use Float64
and are **not certified spectral bounds**. Rank thresholds are diagnostics,
not comparison tolerances. Missing imaginary arrays are allowed only under
the explicitly checked real input contract.

| Step | Iterations J/R | Condition estimate J/R | Backward error J/R | Operator difference | Solution difference |
| --- | --- | --- | --- | --- | --- |
| 1 | 10/10 | 9.35452e6 / 9.35452e6 | 1.49049e-8 / 1.58784e-8 | 0 | 5.30176e-9 |
| 2 | 10/10 | 8.30004e6 / 8.30004e6 | 7.81660e-6 / 2.23071e-7 | 1.12421e-10 | 6.19408e-4 |
| 3 | 7/7 | 2.29227e7 / 2.29804e7 | 3.67719e-8 / 3.74891e-8 | 8.12018e-6 | 6.80573e-5 |

All six estimated covariance ranks are 3, active dimension 10. Maximum
recorded absolute alpha is about 2.00224e7; minimum absolute recorded
denominator is 1.48308e-25. Recursive-residual gaps range from 7.16e-19
to 6.22e-18. Small recursive gaps do not imply a sufficiently converged solve:
the step-2 Julia actual residual infinity norm is 1.29324e-7.

This cell is **not** the aggregate 0.019923 solution-difference maximum in
session 32750. That maximum still needs its own identified cell/operator
audit. The first upstream reduced-product divergence (event 9, about
1.0842e-19) precedes later operator changes. Exact sampler evidence, CG
iteration/algorithm evidence, and the separate 108 direct-SPD solve audit
must not be conflated. No new tolerance, full MPI trajectory acceptance,
fresh C-kernel validation, or milestone completion follows from this report.
CG stopping/iteration counts are numerical-dependent, not RNG invariants.
Different counts may be valid under the portable policy after the same C
algorithm, first numerical divergence, conditioning and actual residuals are
established. The old guard/recurrence differences remain algorithm defects,
independently of those stopping-count considerations.
