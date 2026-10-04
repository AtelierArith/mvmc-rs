# Public parameter access: validated local milestone (#290)

Related to #184 and #185. Baseline: 87cd786c7086618d6f13ec70110a75377d92c337.
The previous 34b5094 source passed 1082 workspace tests but failed Clippy on
three test-only field reassignments; that receipt remains historical. This
successor uses equivalent struct literals and preserves the #291 exports.
The implementation and ordinary controls have now been compiled and locally
validated as detailed below. Final exact-head Linux CI is still required before
merge. Later main checkpoints are not retroactively assigned to these receipts.
The 2342-row audit is unchanged; #184 and #185 remain open.

## Current validation checkpoint

All commands used default features, `--locked`, jobs2 and BLAS/OMP/MKL threads1.
Nextest used `--cargo-profile test-fast --no-tests fail --no-fail-fast --retries 0`.
No fixture bytes, RNG contracts or numerical tolerances were changed.

- Main87cd frozen source: `cargo nextest run --workspace` passed 1103/1103
  tests, 41 skipped, 221.998s; UUID `9b29e4fa-6438-4a68-86d8-2de1e59cc80b`.
  Workspace strict Clippy passed, but fmt rejected only a reexport line wrap.
  Aggregate1 is preserved, not relabelled as success. Receipt:
  `/tmp/mvmc-290-validation.main87cd-successor`.
- Format-only successor: only that `pub use parameters` line was wrapped;
  API implementation, tests and inputs were byte-identical. Focused nine passed
  0.009s, UUID `232fe382-2edc-489d-a12c-11385dea5da9`; sync/SR 24 passed
  0.443s, UUID `8e0d7b77-2ddf-4a3b-bb9b-ddb3d1ecb3a6` (219 unselected,
  not PASS). Workspace all-target Clippy `--profile test-fast -- -D warnings`,
  `cargo fmt --all --check`, `cargo test --workspace --profile test-fast --doc`,
  focused postLIST, source/tools/compiler-provider posts and aggregate all0.
  Receipt: `/tmp/mvmc-290-validation.format87cd`.

Full workspace was not repeated after this sole formatting transformation.
This publication documentation update is also later than the tested source
freeze; it changes no executable source. Local validation does not claim
MPI, new C/Julia execution, full-model sampling, or complete umbrella acceptance.

## Historical SOURCE checkpoints

At the original SOURCE review these controls had not yet been compiled or run;
that was a historical status, not the current validation state. The earlier
34b5094 source passed 1082 workspace tests but failed Clippy on three test-only
field reassignments. Main87cd's first launch then stopped before Cargo with a
provider-manifest awk syntax error (session25831/exit2). Both failures remain
preserved; neither is assigned to the successful format-only successor.

Public Rust APIs return `Result`: `pack_parameters`, `unpack_parameters`,
`get_parameter_value`, `set_parameter_value`. Indices are zero-based. Invalid
length, range, storage and stale native projection/orbital declarations fail
before writes. Existing `all_complex_flag` binds Gutz/Jast/DH/Orbitals only;
there is no saved RBM ComplexType binding to validate. RBM widths, coefficient
storage and mapping ranges are checked, but detecting a same-width RBM
declaration edit is explicitly outside this API's metadata guarantee.
Unlike Julia's out-of-range getter returning zero/setter doing nothing, Rust
reports a typed range error. There is no compatibility wrapper.

Order: Gutz/Jast/DH2/DH4 projection, nine declared RBM sections, declared
Slater slots, OptTrans. Dense RBM/Slater storage retains unmapped declared
slots; mapped RBM duplicates are updated together. Orbital mappings refer to
one dense Slater coefficient, not independent mutable term values. Truncated
programmatic storage is rejected rather than fabricating coefficients.
BackFlow is not implemented or claimed here.

Access does not initialize, normalize, consume RNG, synchronize ranks or
change optimization flags. OptTrans assignments refresh already-present QP weights
using the existing implementation; absent QP weights remain absent.
Bulk unpack refreshes even unchanged OptTrans coefficients, matching Julia's
touched-slot semantics; it can repair stale derived weights.

## Authorities read

Julia reference revision 8bb1b9e8ae47b1512c00b321be05664ddcac0fd1:

- `MVMCOptimizers.jl/src/parameter_sync.jl:172–282`: public count/set/pack/unpack,
  exact-length rejection, duplicate assignment, OptTrans refresh and separate sync.
  SHA256 `93886834c217f0054b956837a25ee533eb227ad27b8dbd98eb28ec0628ebe7d1`.
- `MVMCOptimizers.jl/src/stochastic_opt.jl:1–278`: parameter sections, mapping
  traversal, getter/direct setter and delta helper.
  SHA256 `d43895a547efb1d8444358b62dc762b481faee58c69afea34fcf28b3f5c9eab7`.
- `MVMCOptimizers.jl/test_unit/test_unit_parallel.jl:233–258`: original S120
  first/last delta, perturbed roundtrip, short-vector rejection.
  SHA256 `a2b582f61dc981c318f817fc0575cfadfdb8c93e34c5679aeaf7fdae48be5050`.

C authority:

- `src/mVMC/parameter.c:95–175`: declared contiguous coefficients in initial
  input and broadcast followed by separate shifts/rescaling/normalization.
  SHA256 `46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0`.
- `src/mVMC/setmemory.c:291–302`: declared Para/Proj/RBM/ProjBF/Slater/OptTrans
  pointer offsets. BackFlow is explicitly outside this Rust API scope.
  SHA256 `573d1fb995de33386e7452f5e2fa34aeb05efe14b5bf4a904eb54d2ac7cf1b9e`.

## Ordinary controls (nine local PASS)

`crates/mvmc-core/tests/issue290_public_parameters.rs` contains nine controls:
original preserved Heisenberg input S120 algebraic roundtrip with independent
dyadic first/last expected updates; short/long vectors and invalid indices with
whole-model nonmutation; independent declared Slater/OptTrans layout and QP
products including signed zero; truncated-storage rejection; declared RBM
holes and duplicate repair with literal independent coefficients.
Additional controls check independent complete block order (Gutz/Jast/6 DH2/
10 DH4/all nine RBM/Slater/OptTrans), malformed mappings and checked width
overflow with nonmutation, stale loaded projection declarations with
nonmutation, and unchanged OptTrans assignments repairing derived QP weights.

Roundtrip is an algebraic API assertion, not an independent numerical model
oracle. No C/Julia invocation, fixture generation, runtime sampling, MPI or
full-model numerical coverage is claimed. Public input fixture bytes are unchanged.
