# Actual canonical direct-SR solve capture (#180)

Current long-run acceptance is20 steps, per the user's2026-10-03 instruction;
the regeneration example below requests1/2/3/20 with matched windows.
The existing52-endpoint archive and recorded1/2/3/50 outcomes remain historical
evidence, not fresh20 validation. To audit that preserved archive explicitly,
pass `1,2,3,50` as the second argument of `ctest_audit_direct_sr_metrics.jl`;
the current default is1/2/3/20. Deliberate failure-boundary and kernel-refresh
regressions are unaffected. No conditioning or comparison budgets change.

`ctest_direct_sr_metrics.jl` is an explicit offline Julia 1.13.1 probe.
It is not a Cargo dependency and does not edit vendored code. It extracts
the complete `stochastic_opt!` function through its top-level `end`, installing
an in-memory copy with two read-only hooks: immediately before `potrf!('U', S)`
and after substitution/catch, before any parameter update or synchronization.
It records **the actual original matrix/RHS and actual overwritten solution**,
not an increment inferred from final parameters or a later re-solve.

The ordinary prefix observer is included unchanged. It preserves canonical
seeds/initial overlays and input ordering, explicit step/window overrides,
and the validated C coefficient/counter shim. FSZ requires the same explicit
native-C energy bridge used by the existing independent prefix references.
No Julia-only Hamiltonian repair or Rust-generated expectation is used.
Only the final solve of each independently restarted 1/2/3/50-step run is
archived; preceding iterations execute the original solver. This is endpoint
conditioning evidence, not a certificate for every intermediate solve.

## Numerical authority and interpretation

Authoritative C sources:

```text
43ed8790cff2715284849f0f8906e4645179dcbb100b51f819918b279d6a36f2  extern/mVMC-1.3.0/src/mVMC/stcopt.c
2bd48d880dcbd95ea1b1b92931178c07fd08f981b8ebf04e7db1709e909e57ca  extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c
```

`StochasticOpt` selects components using exact optimization flags and the
diagonal cutoff. `stcOptInit` constructs the covariance in column-major order,
multiplies its diagonal by `1 + DSROptStaDel`, and forms the gradient with
`-DSROptStepDt * 2`. `stcOptMain` invokes DPOSV with `UPLO=U`. The Julia
capture retains the raw matrix and uses its upper-triangle symmetric completion
for diagnostics, matching that solve contract; raw symmetry discrepancy is
reported rather than silently omitted.

Residuals promote the stored Float64 matrix/RHS/actual solution to 256-bit
BigFloat and compute `r = A*x-b`. Recorded metrics include infinity norms,
normwise backward error `||r||inf/(||A||inf*||x||inf+||b||inf)`, and componentwise
backward error `max_i |r_i|/(|A|*|x|+|b|)_i`. Zero denominators are handled
explicitly, not clamped. Symmetric eigenvalues give a Float64 condition-2
estimate, **not** an interval certificate. Diagnostic factorizations/eigenvalue
calculations happen after the original solution capture and do not mutate the
reference's arrays or consume its RNG.

Use these metrics with the existing #190 backward-residual rule
`128 * dimension * epsilon`, and with exact trajectory checks. Conditioning
does not grant a blanket `condition * epsilon` forward tolerance. These records
do not independently bound accumulated energy or parameter perturbations across
50 steps, and cannot by themselves prove the existing 1e-11 forward budget on
every platform. They supplement the same-input independent prefix comparisons;
no numerical-comparison policy or threshold is changed by this probe.

## Reproduction and provenance

Run from the repository root, with an already independently verified native
FSZ bridge and a new external staging directory:

```sh
stage=$(mktemp -d /tmp/mvmc-ctest-direct-sr.XXXXXX)
MVMC_CTEST_NATIVE_FSZ_BRIDGE=/absolute/path/to/verified-bridge \
  julia +1.13.1 --project=extern/Julia-mVMC \
  c_toolbox/ctest_direct_sr_metrics.jl "$stage" \
  heisenberg_chain_real,hubbard_chain_real,heisenberg_chain_cmp,heisenberg_chain_fsz,hubbard_chain_cmp,hubbard_chain_fsz,kondo_chain_real,kondo_chain_cmp,kondo_chain_stot1_cmp,general_rbm_cmp,hubbard_tetragonal_real,hubbard_tetragonal_momentum_projection_real,kondo_chain_fsz \
  1,2,3,20
```

The bridge's compiler/extraction information and binary hash, Julia/backend
versions, manifest/source hashes, actual seed, solver choices, original and
effective counts, overlay hashes and canonical input hashes accompany the
capture. `direct-sr-provenance.txt` separately hashes this probe, the ordinary
observer and C/Julia solver sources. No C compiler is invoked by this Julia
probe; its actual solve backend is Julia's recorded LAPACK/OpenBLAS, not the
C executable. Standalone C energy validation remains distinct from direct-SR
LAPACK validation. GeneralRBM CG is outside this direct-solve probe.

Reusable MPI/serial dev-only API: include `ctest_direct_sr_capture.jl` in Main;
`CTestDirectSRCapture.install!(before_cb, after_cb)` copies actual operands and
original increments/status into consumer callbacks. Retain pairs with the
harness's actual rank/iteration context, then call `diagnose(before, after)`
after the run. The serial generator and archive auditor share this diagnostic
implementation, so an MPI harness need not duplicate residual formulas.
The original no-active branch produces paired `not_solved="NoActiveComponents"`
callbacks with numerical operands `nothing`, not a fake empty solve. No original
Julia no-parameter success is invented. Calls on absent/root-only ranks must be
labelled by the harness, never supplied with cloned root expectations.

The initial successful 52-case generation used the archived
`ctest_direct_sr_metrics.jl.at-generation` and `ctest_prefix_oracle.jl.at-generation`
sources; their hashes match the generation provenance. The later shared-module
refactoring did not regenerate or replace any expected operands/increments.
