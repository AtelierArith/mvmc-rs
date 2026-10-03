# Scoped actual direct-SR observation

This opt-in Rust observer is diagnostic, never an independent oracle or a
numerical-policy change. API:

```rust
let guard = mvmc_core::sr::observer::capture()?;
// Run the original optimization on this caller thread/rank.
let observations = guard.finish();
```

`CaptureGuard` is thread-bound (`!Send`, `!Sync`), rejects nesting without
disturbing an active scope, and clears thread-local state on Drop/unwind.
Each actual direct-SR call records its original regularized column-major matrix
and gradient immediately before the original `cholesky_solve`; after that same
call it records the overwritten RHS and original success/error status before
parameter mutation. Original POTRF/POTRS INFO is copied after each call without
reinterpreting it. `triangle='U'` is explicit. Failed partial/nonfinite values
are retained, not repaired. Raw component flags and effective input/shift/cut/
step/storage settings accompany each record. Real active parameter indices
are doubled into the same zero-based C component convention as Julia's complex
matrix representation. The input seed is not a claimed MPI rank-offset seed;
the harness must record rank, seed overrides and iteration context separately.

Original no-parameter/all-fixed-or-cut returns produce explicitly tagged
`NotSolved` records, with no matrix/RHS/increment and no invented LAPACK status.
CG does not produce direct-solve observations. If an MPI reference calls its
original solver on only a subset of ranks, other ranks must be labelled
NotSolved by its harness; never copy root expectations into their records.
Disabled hooks return before cloning data, converting indices or allocating
snapshot metadata. They incur only the thread-local enable check; they do not
change arithmetic/order, flags, solver arguments, error handling or RNG.

The independently implemented Julia dev-only equivalent is
`c_toolbox/ctest_direct_sr_capture.jl`. Include it in Main, then call
`CTestDirectSRCapture.install!(before_callback, after_callback)`. Callbacks get
owned copies of actual operands/settings and actual increments/status. Retain
per-rank/per-iteration pairs; call `diagnose(before, after)` **after** the run.
The shared diagnostic implementation computes high-precision residuals and
condition estimates without rerunning the solve or feeding results back.
It extracts the whole original `stochastic_opt!` body through its top-level end,
with read-only hooks immediately before POTRF U and after POTRS/catch but before
updates. The original no-active branch emits paired `not_solved` callbacks with
numerical operands `nothing`. Julia has no corresponding no-parameter early
success branch; its empty-diagonal path errors before solving. Do not invent
Julia parity for Rust's separately labelled no-parameter boundary.

Normal Rust tests require no Julia/C/toolbox reads or execution. Observer tests
check success and nonfinite-update failures, original positive-factor-INFO
followed by substitution, nesting/unwind/re-enable/thread isolation, explicit
no-active records, and actual two-step sampling/configuration/counter/next624
RNG invariance. Computed floats use existing portable comparisons, not bitwise
equality. Exact assertions remain for discrete contracts. An illegal-argument
POTRS failure cannot be safely induced through valid public SR inputs; the
tests do not corrupt LAPACK arguments or manufacture such a reference result.
The original INFO capture nevertheless covers that branch if encountered.

Serial condition/residual fixtures are not an MPI allowance. MPI acceptance
requires the **actual reduced per-rank system and actual original increment**,
plus independent matching-input reference captures and first-divergence evidence.
No new tolerance follows automatically from a condition estimate.

Executed verification: five Rust observer integration tests passed, nextest
`22fd4709-32fa-41fb-8017-38c1e2a69598` (0.095 seconds); the disabled-payload unit
passed separately. `cargo clippy -p mvmc-core --lib --tests -- -D warnings`
passed after public-doc and test-module ordering fixes, without an allowance.
Parent's latest binary-filter run `bf06e2dd-d662-415d-825d-212d1a430b76`
completed with exit 0: five observer integration tests passed, zero skipped,
0.055 seconds. Select this integration target with `binary(sr_observer)`;
the earlier `test(sr_observer)` selection (`87847f48`) selected zero tests and
exited 4, so it is not execution coverage. The failure-path evidence observes
original factorization INFO and subsequent substitution/nonfinite-result
behavior; it does not manufacture an illegal-argument POTRS failure.
Actual Julia 1.13.1 smoke completed with explicit no-active paired absence and
an original POTRF/POTRS solve capture; the latter's measured 256-bit normwise
backward error was `8.673617379884036e-17`. This one-component smoke is API
validation, not an MPI condition bound or additional canonical model run.
All 52 standalone serial captured-system records passed the optional shared
diagnostic audit; their archive hashes were verified independently. Active-MPI
capture generation is owned by Wegener and remains separate.

Independent offline MPI audit completed with exit 0 (handle 22608): 108 actual
Julia/Rust solve pairs across world sizes 2/4, sample width 3, prefixes 1/2/3,
Rust worker counts 1/2/4 and every rank/history entry. Retained captures live in
container `73c57e563c61`, `/home/vscode/.cache/mvmc/issue179-sr-independent`;
the generation snapshot is `/home/vscode/.cache/mvmc/snapshots/issue179-sr.WbN8B8`.
Both original capture processes completed successfully (Julia 43637, Rust 28663).
The audit uses the shared capture helper SHA256
`5e4b11c3110bb6d3e7f28dfb9882954db50bd24766caa688e015f2959ac5671a`, not another
solver or newly reconstructed expected increments. It checks dimension,
not-solved status, original return status, active indices and serialized settings
exactly, then diagnoses each side's retained unfactored matrix/RHS/original
increment. Raw optimization flags are preserved, not compared as if C's
uninitialized imaginary slots were defined zeros; this metadata contract needs
the independently established written-slot mask.

All observed systems were positive definite; maximum estimated condition number
was `2.8515440521065343e7`, maximum 256-bit normwise backward error
`4.9380359474671766e-17`, satisfying the existing `128*n*eps` residual rule.
For root/one-worker world4, solve 1 systems/increments agreed, while solve 2
maximum absolute matrix/RHS/increment differences were respectively
`1.3877787807814457e-17`, `2.168404344971009e-19`, `8.678308072163077e-12`.
Its Julia/Rust condition estimates were `2.0182976432887096e7` /
`2.0182976433063664e7`, backward errors `7.564393603419482e-18` /
`1.1573708167707326e-17`. Solve 3 differences grew to `8.470307788499554e-14`,
`7.795847301039771e-15`, `2.0054014004955434e-11`. These are actual-system
diagnostics, not the first upstream sampling/reduction divergence, a certified
forward-error bound, or permission to widen #190 tolerances.
Wegener's global trace locates an earlier numerical difference in the step 1
reduced accumulator (at most `2.22e-16`). The root/one-worker regularized solve 1
matrix/RHS/increment nevertheless agree in these retained records; solve 2 is
the first differing solver system in this audit, not the global first difference.
The records therefore distinguish reduction-level rounding from later
ill-conditioned solve amplification; they do not prove full trajectory parity.

Reproduction: from a checkout containing the dev-only audit, run
`CTEST_MPI_CAPTURE_ROOT=<retained-root> JULIA_DEPOT_PATH=/home/vscode/.cache/mvmc/julia-depot OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 /home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia --compiled-modules=existing --project=extern/Julia-mVMC c_toolbox/ctest_audit_mpi_sr_capture.jl`.
The executed audit used the same script body via `-e`, the generation snapshot's
project and one Julia/BLAS thread. It writes no capture files or expectations and
is never invoked by normal Rust tests. Full discrete trajectory comparison,
first upstream divergence and any portable policy decision remain separate.

Fail-closed audit revision SHA256
`c70831c836f3ff174e7fa0ad26e24c31c1fd6eb0526110be6d97daa9632d87ac`
was rerun on the same retained captures: handle 34899, terminal exit 0, exactly
108 pairs, unchanged numerical diagnostics. This revision rejects duplicate
record keys, requires each original parsed status to be zero before passing it
to `diagnose`, rejects not-solved records, explicitly verifies positive dimension,
matrix/RHS/increment shapes and finite operands, and asserts the final count is
108 before reporting success. The repeated command used
`--compiled-modules=existing`; original oracle captures were not modified.

The complete stdout was preserved from the terminal tool transcript, not a new
oracle run: [34899 stdout](evidence/issue-180-mpi-sr-audit-34899.stdout.txt), SHA256
`4b96e47dc18ebc9391fb1aec7a98c19f4760aae241fb92e799a447cacd347e62`.
The [executed `-e` source](evidence/issue-180-mpi-sr-audit-34899.jl) has SHA256
`3a064d8e5eafcf80aeac4a8be2138e1e439b82d2f17bdbee8b90b98749259018`.
It matches the current audit revision after the exact two replacements of
`joinpath(@__DIR__, "ctest_direct_sr_capture.jl")` with
`"c_toolbox/ctest_direct_sr_capture.jl"` required for snapshot-relative `-e`
execution. Its bytes/hash are therefore distinct from the reusable file's
`c70831...` hash; no byte-identical execution claim is made for that file.

Exact-environment reproduction from the repository root (reads the archived
executed source; leaves retained captures unchanged):

```bash
docker exec \
  -e JULIA_DEPOT_PATH=/home/vscode/.cache/mvmc/julia-depot \
  -e OPENBLAS_NUM_THREADS=1 -e JULIA_NUM_THREADS=1 \
  -w /home/vscode/.cache/mvmc/snapshots/issue179-sr.WbN8B8 \
  73c57e563c61 \
  /home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia \
  --compiled-modules=existing --project=extern/Julia-mVMC \
  -e "$(cat docs/reference/c-to-julia/verification/evidence/issue-180-mpi-sr-audit-34899.jl)"
```

The container ID, executable/depot paths, generation snapshot and retained-root
are environment-specific reproduction coordinates, not runtime dependencies of
Rust tests. On another host use Julia 1.13.1 with the pinned reference project,
the identical hashed helper, one Julia/BLAS thread and unchanged retained capture
files; set `CTEST_MPI_CAPTURE_ROOT` to their new location. Moving the evidence
does not itself constitute a portability run or native macOS validation.

Subsequent parent-reported issue #179 expanded-matrix results found CG discrete
trajectory failures (real CG NStore 0/1 prefix 3: event 477 Julia rejects while
Rust accepts; complex CG prefixes 2/3 also fail). These are outside this direct
solver audit and require first-divergence/C-contract investigation. The 108
direct-system residual checks neither validate CG trajectory nor justify
relaxing acceptance/configuration/RNG comparisons or computed tolerances.
