# Reviewed C-faithful CG references

New lineage, preserving all historical Julia fixtures unchanged. Published
Julia PR54 commit `62b0f97f076fb55c71c3ab0caa041a9adff94e04`, production CG
SHA `b11d75d9b2baaef31abc59c110c09fedc86a7705b1a3c2ce2bb6c75b8b8a17b3`:
original C recurrence (`delta = beta * delta`) and no Julia-only tiny-denominator
early exit. The independent three-case C refresh fixture checks all limits
1..41, including refresh20/40; this is not a full native C/MPI runner claim.

## Canonical candidate

`canonical_general_rbm` is the complete native-input NPara102 fixture, seed12395,
serial process, Julia1.13.1, one default worker, one OpenBLAS thread. Prefixes
1/2/3 are imported from actual acquisition `/tmp/mvmc-review62b-state.nvDReR-stage`.
Per-prefix provenance records all63 production/environment hashes, input
hashes, options, actual BLAS settings and observer identity. Raw state getters
do not draw/reseed; final library SHA
`7e77954acae2073591edf17b5c9825f0c020a2525026ae450ac31b780d7a7c86`.
Actual 918 initial-slot records, nine full raw624/index192/count192 states,
918 mappings and1,782 C-written flags passed the independent C artifact audit.
Six unwritten C flag cells per phase remain undefined, not native zero goldens.

The user-approved initial numerical budget is **absolute8.7e-19, relative0**,
ONLY initialized/overlaid/synchronized initial parameter components. Maximum
observed8.673617379884035e-19. First initialized slot6 imaginary error
1.0842021724855044e-19 comes from Ccexp vs Julia exp/sincos, not RNG draws:
see `docs/reference/c-to-julia/verification/issue-180-rbm-first-initializer-operation.md`.
No CG-final/MPI tolerance is implied. Primitive stream/state/count/conversion
identity remains strict at equal draw order/count; numerical branch differences
require first-cause/threshold analysis. Same implementation/input/fixed-seed/
worker configuration repeatability is tested separately.

`step-N/c-window-input.txt` contains actual pre-SR Etot/Etot2 and post-SR
parameters, explicitly selected final window. `zqp_*.dat` was computed by the
standalone C avevar probe `/tmp/mvmc-history-layoutprobe.3WpZwp/probe`:
`probe INPUT OUTPUT_HEAD 0 0 0`. Scalar aggregation/schema authority only,
not full native-C sampling. `zvo_var.dat` is actual reviewed Julia complete
declared history. No Rust outputs generated these expectations.

Initial import mistakenly omitted six C block files at each prefix2/3;
targeted97bf83c8 failed the manifest check, not numerical parity. After adding
those12 independent files, targeteda402d5f6 cleared all prefixes1/2/3 (including
exact sampling/RNG checks and existing abs/rel1e-11 numerical budgets), then
failed prefix50 against historical expectations. Prefix50 regeneration is
handle46444 subsequently terminated0,5/5 assertions44.4s. Its complete
independent C windows and actual prefix50 records are imported. The full
canonical consumer passed a82613dc-722f-4632-8fb6-01994de28ba0 (6.872s)
through1/2/3/50 with exact saved sampling/RNG and unchanged numerical budgets.
Rust fixed-seed same-configuration repeatability passed79bef5c9 (0.424s).
No all14-consumer or fullMPI acceptance claim follows.

## All14 failing consumers: exact acquisition mapping

All entries use CG/store0, prefixes1/2/3/50. `general` deliberately consumes
the existing FSZ reference for its equivalent General representation. Direct
subcases in mixed tests keep their already passing direct references.

| Rust callback test stem | Acquisition cases |
| --- | --- |
| real_cg_prefixes | real |
| complex_cg_prefixes | cmp |
| fsz_cg_prefixes | fsz |
| general_cg_prefixes | fsz (General representation) |
| hubbard_cg_prefixes | hubbard |
| pairhop_real_and_fsz_cg_prefixes | pairhop_real, pairhop_fsz |
| interall_fsz_cg_prefixes | interall (existing historical synthetic API regression only) |
| dh2_real_complex_and_fsz_cg_prefixes | dh2_real, dh2_cmp, dh2_fsz |
| dh4_and_dh24_real_complex_and_fsz_cg_prefixes | dh4_real, dh4_cmp, dh4_fsz, dh24_real, dh24_cmp, dh24_fsz |
| rbm_real_and_complex_cg_prefixes | rbm_real, rbm_cmp, rbm_general_cmp |
| rbm_dh24_direct_and_cg_prefixes | rbm_dh24_cmp (CG subcase) |
| rbm_fsz_cg_prefixes | rbm_fsz |
| opttrans_cg_prefixes | opt_real, opt_cmp, opt_fsz, opt_dh24_rbm_cmp |
| canonical_general_rbm_complex_reference | rbm_reference_cmp (CG subcase) |

Archived sparse RBM/OptTrans/InterAll definitions are NOT native C input
acceptance evidence. Rust production strictly rejects incomplete raw RBM
declarations; historical test helpers explicitly construct sparse API models.
Those acquisition cases must be labelled synthetic architecture/kernel checks,
never C-supported namelist goldens or invented expanded flag declarations.
FSZ uses actual unchanged C Hamiltonian kernel bridge, with Julia sampler/SR;
this mixed provenance is distinct from full native C execution.
# Post-integration checkpoint (2026-10-03)

Fresh focused run `f8f34e09-55cb-42d1-b7b2-08afbe3bc4f7` selected
14 CG consumer tests: 14 passed, 195 excluded, 15.030s. This was a mutable
working-tree diagnostic, not a frozen workspace or MPI acceptance result.
The subsequent fail-closed output-file preflight has not yet been rerun.

There are 26 CG case completion markers and 46 direct/store completion
markers. Mechanical imports verified identical artifact bytes; a separate
integrity check validated 55 archive manifests covering 1,687 files.
Older imports without archive manifests remain an integrity-metadata gap.

Basic cases (real, complex, general, Hubbard, FSZ, pair-hop and InterAll)
retain their independently acquired energy/parameter/SR/configuration/RNG
checks. Their acquisition did not retain `step-N-zvo_out.dat`; the full
C-window output comparison added for DH/RBM/OptTrans must not be claimed
for those basic cases. Fresh independent output acquisition remains pending.

Six direct/store bundles remain unadopted: RBM-FSZ store0/store1 failed
with actual status `1` at step `13`; OptTrans-FSZ and InterAll store0/store1
were not acquired after the coordinator stopped. Both failed RBM-FSZ stages
are preserved under `/tmp/mvmc-review62b-runner20.Ctp8Dh-rbm_fsz-directS-julia`
(S=0 or 1). Each has 14 history rows and identical retained configuration,
RNG and parameter artifacts across store modes. The acquisition assertion
still expected the historical 50-step failure boundary; that assertion is
not evidence that production completed 20 steps successfully. Failure-boundary
C-contract review and a separately versioned acquisition remain pending.
