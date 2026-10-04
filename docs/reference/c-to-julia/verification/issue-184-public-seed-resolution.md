# A202 public seed integer-resolution API — scoped local validation

Related to open184/185. Base5b0874eb70b72e2a7993662af757d50b25dadc17.
The scoped local gates below passed. Canonical ledger promotion and fresh-head
full Linux CI remain pending; this is not full MPI trajectory validation.

Original Julia8bb1b9e8ae47b1512c00b321be05664ddcac0fd1:
MVMCOptimizers.jl/src/parallel.jl:319–344 public resolve_rnd_seed;
whole SHA a7d72725f193a121f79267be6f931e2439fecdf104de01a0cb1d3a5ca8d91fc1.
MVMCOptimizers.jl/test_unit/test_unit_parallel.jl:162–173 original M0562–M0567,
whole SHA a2b582f61dc981c318f817fc0575cfadfdb8c93e34c5679aeaf7fdae48be5050.
C d73d06bd529d3b2573f38eb5817c4a5f52971006 src/mVMC/vmcmain.c:254–257
initializes RNG with RndSeed+group1. This source order is not a new C runtime check.

API resolve_rnd_seed(input:i64, override:Option<i64>, group1:usize,
&impl Reducer) -> Result<i64,String> is also reexported from mvmc_core.
It exposes the same internal resolver already used by runners, with their
unchanged SystemTime conversion closure factored once. Both existing runners
call this API; internal clock injection remains private. No compatibility
wrapper, scalar/reducer signature, RNG initialization/draw or phase-order change.

Resolution returns the integer BEFORE separate existing SFMT UInt32 conversion:
explicit override first (even negative), else nonnegative input unchanged,
else only output root reads Unix seconds and broadcasts status/base. Then
existing i64 wrapping group addition. Parser default11272 remains supplied
by parser/caller; zero is not replaced with a default. Out-of-UInt32 return values
are not silently repaired: runner seeded_rng still rejects them separately.
This preserves existing defined Rust error/conversion policy, not a claim that
C signed-overflow/invalid conversion is defined or newly widened input support.

Oversized usize group conversion fails collectively before resolution; root
clock/broadcast errors propagate through unchanged collective_result/any_failure.
Callers must invoke on all participating ranks with consistent input/override
policy and correct group offset. This does not synthesize MPI topology or
validate inconsistent per-rank policies. Reducer/ParallelContext architecture
is existing and unchanged. No actual MPI test is claimed by serial mocks.

Existing run::seed_tests cover the six original literal conditions, injected
clock nonconsumption/error, root broadcast before group, SFMT range failures
and deterministic stream fixture. Reuse them instead of duplicating algorithms.
Two ordinary public integration controls verify callability and five
deterministic literal cases, and pre-conversion negative/large/wrapping return
behavior. Negative-time original condition is deterministic via existing private
clock-injected test, not a new flaky wallclock assertion.

Ownership: isolated run.rs/lib.rs hunks only; shared dirty source untouched.
Coordination request sent to parent for Chandra174/181 and Ram run.rs ownership.
No edits to Goodall scalar/reducer/MPI or Ram accumulator paths.
All A202/S117/2342 ledger rows remain unchanged.

## Local execution checkpoint

Receipt: /tmp/mvmc-184-a202-proof.20261004; source base remains the commit above.
Commands used default features, --locked, --cargo-profile test-fast, jobs2 and
BLAS/OMP/MKL threads1. LIST verified exact package #0.0.0, filter-match.status,
named default fingerprint ["default"], source cwd and selected executable.

- Existing run::seed_tests: 8/8 PASS in 0.014s; 236 filter-unselected tests are
  NotRun, not accepted coverage.
- issue184_public_seed_resolution: 2/2 PASS in 0.006s, zero skipped.
- Both RUN commands used --no-tests fail --no-fail-fast --retries 0.
- Scoped core lib/public-test Clippy -D warnings and workspace fmt check passed.
- Eight phases (unit-list, unit-run, public-list, public-run, clippy, fmt,
  unit-post-list, public-post-list) each recorded child.status, command.status,
  adapter.post.status, postempty.status and both stream-writer statuses as 0.
- Pre/post LIST identity comparisons and source, tools, selected toolchain,
  compiler/target sysroot libraries, executable and runtime-provider checks
  passed. Primary, posts and aggregate terminal are all 0.

The owner adapter emitted transient /proc/<pid>/stat disappearance diagnostics;
these are retained, not hidden. Its eight postempty/status records above were
checked separately. Provider inventories bind actual matched ELF runtime paths
and hashes; they are not independent numerical oracle evidence.

The execution froze the document's earlier SOURCE-proposal version. This later
documentation-only checkpoint does not relabel that executed document hash.
Resolver, RNG conversion, tests and expectations are unchanged after execution.
No model, full-workspace, C/Julia runtime or actual MPI run was performed.
