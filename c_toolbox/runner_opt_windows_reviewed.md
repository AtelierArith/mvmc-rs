# Reviewed-fork acquisition: separate lineage

`runner_opt_windows_reviewed.jl` is a new optional generator, not a replacement
for `runner_opt_windows.jl` (historical SHA256
`305934493f2e724cddfdd29274fad6c59eaa6e09b4bc37103d2dda8682286eed`).
The historical file and archived expectations remain unchanged. Reviewed
published PR54 commit62b0f97f076fb55c71c3ab0caa041a9adff94e04 generated the
canonical NPara102 candidate under `/tmp/mvmc-review62b-state.nvDReR-stage`
(prefixes1/2/3) and `/tmp/mvmc-review62b-state.nvDReR-prefix50`. These are new
lineages, not retroactive verification of historical acquisitions.

Use an independent snapshot with the reviewed fork checked out under
`extern/Julia-mVMC`. A full 40-character commit and an externally reviewed
SHA256 manifest are mandatory; every parser/optimizer production `.jl`, the
project and Manifest-v1.13.toml must be included. The generator rejects wrong
hashes, dirty production files, packages loaded from another checkout and an
existing stage path. The operator may not supply arbitrary hashes merely to
bypass pending review. It emits no files into the repository.

It observes the fork's actual `pack_parameters` values and checks all 15 C
section widths against declared NPara. No historical retained-slot shadow,
initializer replacement, delta interception, reseeding or additional RNG
draw is installed. `parameter-audit.tsv` records every declared slot at
initialized, overlaid and synchronized boundaries, its mapped/unmapped marker,
complex value and both component flags. It is an observation, not an
independent C expectation or an automatic assertion of parity.

`compare_reviewed_parameter_audit.jl` compares that full table to separately
generated C records, requiring identical stage/slot sets, contiguous indices,
exact mapping/flags and exactly 624 matching initialization RNG words.
Numerical parameter comparisons use explicit caller-supplied abs/rel budgets
with C-provenance justification; serialized values are not compared bitwise.
No missing stage/slot or unmapped parameter may be silently omitted.

The comparator requires `declared_npara` and `audit_groups` in C provenance,
all three ordered phases per group and contiguous global sequence numbers.
Non-overlay models emit an explicit unchanged overlaid phase before the actual
sync call. Synchronization records are captured immediately after the actual
parser/optimizer sync function returns, not after QP initialization. An events
sidecar names the actual boundary, case, section widths, worker pools and BLAS
thread count; both default worker and actual BLAS counts must be one at runtime.
Exactly one case and a positive unique explicit prefix list are allowed per
stage; existing stages/prefix directories cannot be overwritten.

The C provenance file is key=value, requiring `authority=C`, upstream,
adapter, input, expected and RNG SHA256 fields, command/compiler/extraction
details and numerical_budget_justification. These must describe actual
independent C generation, not transformed Rust output or the historical
Julia slot bridge. The comparator verifies the expected and RNG artifact
hashes and is read-only. Canonical C audit data are checked in separately at
`tests/fixtures/reviewed_parameter_c_audit/canonical_general_rbm`.

Caller budgets must equal `absolute_tolerance`/`relative_tolerance` recorded in
reviewed C provenance. Expected/RNG artifact hashes are verified; input and
upstream/adapter hashes are provenance-only until independently checked by the
acquisition owner, not a claim that this comparator read those original bytes.
The original comparator proves next624 words only. The separate
`diagnose_reviewed_parameter_state.jl ACTUAL_STAGE C_STAGE 102 3` verifies
physical row/phase/slot order,918 mappings,1,782 C-written flags and all nine
actual raw624-word/index192/count192 states against independent C artifacts.
Undefined C flag cells are NOT native zero expectations. User approval scopes
absolute8.7e-19/relative0 solely to initialized/overlaid/synchronized initial
parameter components; no downstream CG or MPI forward-error budget follows.
`test_reviewed_parameter_state.jl ACTUAL_STAGE C_STAGE` passed11/11 (31.6s),
including approved-bound8.7e-19 and outside-bound8.8e-19 regressions, physical
order and state/count/mask failure boundaries. Pauli's first cexp/sincos
divergence is persisted in issue-180-rbm-first-initializer-operation.md.

Planned settings preserve the existing model namelist and all overlays;
prefix lists are explicit and NSROptItrSmp=prefix is an explicit fixture
override, not runtime clamping. Seed=1 except canonical GeneralRBM seed12395.
The existing CG script selects NSRCG=1/NStoreO=0; direct STORE0/1 remain
separate axes. Sampling counts/moves come from each actual model input, not
a blanket substituted count. Each emitted prefix includes modpara, input
file hashes, reviewed production hashes and environment identities.
Failed SR histories remain marked and are not aggregated as final output.
Native FSZ local energy/counter bridges retain their existing mixed-C/Julia
scope, not full C executable/MPI validation.

The regeneration scope includes the independent #182
`general_rbm_cmp_cg` consumer: its five OO/HO stages were independently
verified, but parameter component0 differs by `1.60882e-6` with the new C
recurrence. Keep that parameter assertion; this is not an OO/HO failure or
permission to relax tolerances. Bad historical OptTrans-FSZ CG50 remains a
separate first-discrete-divergence investigation; neither this new generator
nor a numerical kernel repair clears that gap automatically.

After parent authorization and published hash review only:

```sh
julia +1.13.1 --project=extern/Julia-mVMC c_toolbox/runner_opt_windows_reviewed.jl EXTERNAL_NEW_STAGE cg --reviewed-commit=REVIEWED40HEX --reviewed-manifest=REVIEWED_SHA256_MANIFEST --sfmt-state-library=FINAL_OBSERVED_LIB --sfmt-state-library-sha256=REVIEWED_LIB_SHA --sfmt-state-observer-sha256=REVIEWED_HELPER_SHA --case=rbm_reference_cmp --c-kernel-order --steps=1,2,3,50
julia +1.13.1 c_toolbox/compare_reviewed_parameter_audit.jl ACTUAL_TABLE C_TABLE C_PROVENANCE ACTUAL_INITIAL_RNG C_INITIAL_RNG ABS REL
```

Canonical candidate fixtures are imported at
`tests/fixtures/reviewed_cg_62b/canonical_general_rbm`. Actual standalone C
window aggregation is separate from Julia sampling and complete history.
Targeted Rust consumer a82613dc-722f-4632-8fb6-01994de28ba0 passed all prefixes
1/2/3/50 (6.872s), preserving exact sampled configs/RNG and existing numerical
budgets; same-Rust fixedseed/config repeatability passed79bef5c9 (0.424s).
This is not all14-consumer or fullMPI acceptance. Remaining acquisition cases
and native/synthetic distinctions are enumerated in reviewed_cg_62b/README.md.
