# Original parameter boundary follow-up (Related #184/#185)

Baseline main14a30849ae4fa9dd78900f113257389d906008ee. Local bounded validation
completed; no publication or full-model/MPI execution.
Two ordinary tests bind original Julia revision8bb1b9e8ae47b1512c00b321be05664ddcac0fd1,
`MVMCOptimizers.jl/test_unit/test_unit_parallel.jl:299–316` (SHA256
`a2b582f61dc981c318f817fc0575cfadfdb8c93e34c5679aeaf7fdae48be5050`).
The unchanged original Heisenberg closure is used without initialization/RNG.
M0588–590 tests both negative-zero components in the same first projection slot
(Julia para_zero[1] equals Rust zero-based values[0]);
S122 compares serial-context synchronization with the local pathway, retaining
Julia's approximate equality interpretation. #290 is closed; this work is scoped
under open #184. Matrix promotions remain separately reviewed.

## Actual bounded local receipt

Receipt `/tmp/mvmc-184-parameter-boundary-proof.20261004`, session2551:
2/2 PASS, zero skipped, 0.009s;
UUID `5c7e0678-8e8e-44bd-ba58-23b25c3d9175`.
`original_first_projection_complex_signed_zero_survives_public_unpack` binds
M0588–M0590; `original_heisenberg_serial_context_sync_matches_local_sync` binds
S122/M0591. The latter is serial-only, not MPI/broadcast parity.
Command: `cargo nextest run --locked --cargo-profile test-fast -p mvmc-core
--test issue184_original_parameter_boundaries --no-tests fail --no-fail-fast
--retries 0`, jobs2 and BLAS/OMP/MKL threads1, default features.
Exact test-target Clippy `--profile test-fast -- -D warnings`, fmt, pre/post LIST,
source/tools/compiler-provider posts and aggregate all0; LIST byte-identical.
Test assertion body SHA at execution:
`679a46ca660f9bb4959c983f93ec67d7f0be04d24c3d915ee165750a3ca25988`
(whole test file, including its historical SOURCE-only comment). Only the comment
and this documentation were updated after that freeze; production/assertions
remain unchanged. Original fixtures and numerical tolerances are unchanged.

## M0587 qualified manual RBM diagnostic: design only

Original Julia `check_duplicate_consistency` is not exported; M0587 calls it
by module qualification and expects an exception for two ChargeRBMPhysLayerTerm
shadows with shared idx0 but values0.1/0.2. It detects, does not repair. The
original compares values with `!=`, not bitwise equality; +0/-0 equality is
separate from signed-zero preservation during unpack. Orbital repair is the
distinct M0585/M0586 contract, not M0587.

C `setmemory.c:291–302` allocates one contiguous declared coefficient per slot;
orbital mappings point to dense Slater storage. Rust follows that representation:
two orbital rows referencing one idx cannot possess conflicting coefficient
values. Julia's independently mutable OrbitalTerm.value error must not be copied
as though that impossible Rust state were observable. M0585–586 remain a
representation distinction requiring review, not automatic full PASS.

RBM terms currently cache values alongside dense rbm_params. Current public
`parameters::checked_count` checks widths/storage/ranges, not shadow consistency.
A prospective optional, manually called read-only typed diagnostic can check
same-section/same-idx shadow equality, matching M0587's detection intent.
Separately, comparing shadows with canonical dense slots detects even two
equally wrong caches; this stronger check must be labelled a Rust cache invariant,
not the original Julia assertion. It can report section, row and idx on mismatch.
Neither diagnostic would change C parameter algorithms, make pack/unpack reject
inconsistent shadows, normalize, repair values, infer undeclared widths or
consume RNG. Existing public pack continues reading canonical dense storage.
Orbital mapping-range validation is meaningful; independent orbital-value
consistency is NotApplicable by representation. For the Julia-style manual check,
value inequality accepts signed-zero equality and rejects NaNs; a stronger dense
cache check needs its separately reviewed equality policy, not an assumed bitwise
parity requirement.

No diagnostic API implementation is proposed here. Acceptance would require
valid duplicates, divergent caches, equally wrong duplicate caches, malformed
mapping ranges and nonmutation controls. Owner #184/#185; no #290 reopening or
newly claimed C runtime/full sampler verification.

## M0581 actual additive operation: bounded local successor

PR293's direct get/set effect does not execute original Julia
`update_parameter_value(data,i,0.25,-0.5)`; that PR293 proof remains
RelatedNotEquivalent for the additive operation. The later actual unit test
`sr::original_parameter_delta_tests::original_m0581_additive_helper_updates_first_and_last_parameters`
parses the untouched original Heisenberg input and calls the actual existing
`sr::update_parameter_value` at zero-based (0,n-1), with the parsed projection
offset. Original before+delta assertions and independent zero→(0.25,-0.5)
endpoint literals are both checked. No getter/setter substitutes the operation.

Rust's existing helper is pub(crate), takes an explicit n_proj and updates C
declared dense storage; Julia's helper is module-qualified, one-based, computes
its layout and adds to mapped term values. On the original complete supported
model, the first projection and last Slater are equivalent targets. Passing the
actual layout n_proj preserves the internal SR API without exporting a new
compatibility wrapper. The added cfg(test) module changes no production algorithm,
visibility, floating-point order, normalization or RNG.

Actual sibling receipt `/tmp/mvmc-184-additive-proof.20261004`, session29610,
baseline main14a308: this unit identity passed1/1 0.011s,
UUID8db81051-8931-48b4-a162-f31ee4ec6da7; 243 unselected are not PASS.
Same-source boundary2 controls passed2/2 0.008s,
UUIDe529a87c-bc39-49e4-a863-2a54126e7404. Exact lib and boundary pre/post LISTs
match; lib/test-target Clippy, fmt, source/tools/compiler-provider posts and
aggregate all0. No full workspace/model/Julia/MPI executed in this receipt.
Executed sr.rs SHA55e820e461f457fba68de04235a560bcd721b9d1293a986017fcf2ffd91a72c6;
boundary test SHA0730581f7e09c2a81110d3fdbd2913677e206efb66157c8dad312813f65212a7.
Original condition joins M0581/M0588–M0591/S122 are now bounded local evidence;
current-head CI and full API/scenario acceptance remain pending. The prior
boundary-only2PASS receipt remains unchanged. This doc checkpoint is later than
the executed source freeze and changes no assertions or production.
