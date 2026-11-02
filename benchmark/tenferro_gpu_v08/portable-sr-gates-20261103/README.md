# Portable CUDA arithmetic gates (#623)

This test-only repair replaces legacy arithmetic bit/whole-output equality
with the existing portable numerical policy. Production kernels, fixed problems,
seeds and RNG controls are unchanged. Existing independent kernel/solver and
trajectory C-order bounds are retained; the computed pre-update energy row now
uses the approved short-SR numerical envelope instead of exact text equality.
The exact RNG uniform-conversion/decision check in `sampler_gate` remains exact.

Source reconstruction: `parent-source.txt` plus `git apply --unidiff-zero source.patch`
(the three changed CUDA test files). The numerical-policy inventory is also updated. Environment:
native Linux x86_64, Xeon E5-2699 v3, RTX 3060 GPU 1,
UUID `GPU-ed896702-ca5c-93f6-2537-06fffdac4a91`, driver 580.178.04,
Rust 1.99.0, optimized standalone `test` profile, BLAS/OMP/Rayon=1.
Library paths and exact commands are in `run.sh`; adapt host-specific paths.
These are correctness gates, not new benchmark timings.

## Bounds and evidence

- **S/g:** `abs=0, rel=2 eps`, reusing #190's same-expression assembly contract
  with FMA disabled. Both sides must be finite. All tested assembly discrepancies
  were 0; no failing kernel discrepancy was covered by a wider bound.
- **Direct repeat:** `abs=rel=1e-11`, the existing short-SR repeatability envelope.
  Each solve additionally uses the unchanged C-order solution and independent
  residual checks. Original repeat case n=257, samples=600, seed=77 is preserved.
  Observed repeat difference 0; C-order relative error `1.10e-13`, existing
  conditioned bound `2.49e-10`, device residual `6.08e-14`.
- **CG repeats:** `abs=8 eps`, `rel=2*iterations*gamma_(samples+components)`.
  This is a strict supplied-problem regression budget based on the two GEMV
  reductions, not a general forward-error guarantee for arbitrary CG. The 1/2/4/8/119
  prefixes each had repeat difference 0. Original problem/seed and early C-order
  comparisons remain unchanged; iteration counts are checked exactly.
- **Converged CG:** independently materialize the covariance using C-order Gram
  and mean/diagonal corrections. True residual norm is `2.234e-6`, below the
  `tolerance*components` stopping threshold plus operand-scale rounding allowance,
  `2.990e-6`. Absolute-term scales also retain the original independent CG-product
  bound. This is separate from comparing the recurrence's reported residual.
- **SR routing / resident measurement:** computed repeat fields and pre-update
  energy use `abs=rel=1e-11`; file sets, row counts, headers and integer tokens
  retain exact checks through the common numerical-text helper. Existing C-order
  trajectory/measurement bounds and routing/transfer counters are retained.

## Fixed prefixes are not convergence claims

An initial added assertion incorrectly applied a direct-solver roundoff residual
bound to the zero-tolerance, fixed-119-iteration CG case. The saved initial and
diagnostic logs show device backward residual `6.908e-5` and C-order `7.063e-5`:
both fixed prefixes are unfinished. The final test records these residuals and
checks repeatability/early reference values; it applies the convergence residual
assertion to the actually converged case above. This changes the invalid new
assumption rather than raising an existing tolerance or reseeding the problem.
The observed late device/C-order solution difference `6.73e-4` is the existing
ill-conditioned amplification case (#358), not a new repeat difference.

## Verification

- Updated direct/CG cases: 3/3 passed, with diagnostics (`targeted.log`).
- Broader CUDA correctness group: 34/34 passed, with the transfer microbenchmark
  filtered (`cuda-correctness.log`).
- After updating routing/measurement text checks, their affected group was rerun:
  6/6 passed (`routing-measurement.log`). Nextest isolates each test process;
  the other test cases' source was unchanged by this final text-check update.
- Resident Opt reference, pipeline, failure and 20-step-repeat gates: 6/6 passed
  (`resident-opt.log`); their production/library source is unchanged.
- Standalone CUDA all-target Clippy and formatting passed.

The logs distinguish the earlier failed diagnostic from final passing gates.
Fixtures retain their checked-in provenance; no C/Julia oracle runtime or fixture
regeneration is added to ordinary Rust tests. The existing two-device rejection
proof and device identities are in [the post-ownership validation](../final-validation-20261102/README.md).

Run `run.sh` from the repository root at this PR's source, or reconstruct the
test source from the recorded parent/patch. It includes the targeted diagnostics,
34-case correctness group, resident Opt group and optional two-device precondition
and rejection check. CUDA absence is a failure through `MVMC_RS_CUDA_GATE=1`.
