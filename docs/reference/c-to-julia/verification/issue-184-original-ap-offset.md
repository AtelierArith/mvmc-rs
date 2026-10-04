# Original S070/S071 public AP-to-Parallel offset contracts (SOURCE)

Owner/open scope: #184/#185. Candidate base main
`12c5bd83967c12b8d246fa81a102ada360bc19e1`. Two ordinary parser tests have
scoped local execution below, not canonical original-row PASS promotion. No production,
external fixture, golden, numerical tolerance or RNG implementation changes.

## Original conditions and explicit adaptation

Julia revision `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`,
`MVMCExpertModeParsers.jl/test/test_read_input_parameters.jl:605–730`, SHA256
`e807528afdf1bd94abd7b830fd7cb01cfa4fa2c1b935676ea5e816c92a42ef40`.
The complete two testsets were read before implementation.

- S070 declares AP2 with physical rows `0 0 0`, `0 1 1`, `1 0 1`,
  `1 1 0` and flags `0 1`, `1 1`; P1 original rows are `0 1 0`,
  `1 0 0`, flag `0 1`. Expected AP width stays2 after P loading.
- S071 declares AP3 but maps only indices0/1 with original physical rows
  `0 1 0`, `1 0 1`, flags0/1/2 all1. P1 original row `0 1 0`, flag0=1.
  Original assertions require AP width3, total width5, P indices3/4,
  no mapping for AP slot2, and InOrbitalParallel local indices0/1 loaded
  as exactly11/22. The original two overlay rows are `0 11.0 0.0`,
  `1 22.0 0.0` with declared overlay count2.

These literals are source evidence, not a claim that Julia sparse/reverse
physical rows form complete C input. Tests explicitly adapt:

- AP2 keeps all four original pairs/indices, adding explicit sign1.
- AP3 keeps both original off-diagonal pairs/indices and all three flags;
  adds `0 0 0 1` and `1 1 1 1` so all NSite² pairs are initialized.
  Neither added row references slot2, which remains declared/unmapped.
- P1 supplies only the upper pair `0 1 0 1`, removing S070's reverse row.
  C itself creates reverse antisymmetric entries and both spin sectors.
- InP retains independent original11/22, with reversed record order as an
  additional indexed-scatter control. Public namelists cover AP-first and
  P-first order; P-first is supported by C's separate count/body passes,
  not Julia's standalone P-before-AP rejection extension.
- Minimal ModPara NSite2/NMPTrans1 is supplied solely for public loading.
  Dyadic AP coefficient sentinels are independent unchanged-block controls,
  not original initialization values, generated goldens or a runnable model.

## C authority read before implementation

`extern/mVMC-1.3.0/src/mVMC/readdef.c`, SHA256
`6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`:

- `ReadBuffIntCmpFlg`126–144 reads declared integer count/complex header;
  count0 is rejected. Header pass485–500 records AP width and adds twice
  P width to total; count is not inferred from maximum mapped index.
- Body dispatch1010–1035 passes `NProj + FlagRBM*NRBM + NArrayAP` as P
  optimization offset. Whole `GetInfoOrbitalAntiParallel`2462–2533 read:
  it consumes NSite² physical pairs then checks declared flag-row count.
- Whole `GetInfoOrbitalParallel`2583–2640 read: each upper physical pair
  expands spin0/1 to `NArrayAP + 2*fij_org + spn_i`, checks upper ordering,
  and checks Nsite*(Nsite−1) expanded entries and declared P flags.
- `ReadInputParameters`1183 onward, five header reads1207–1217, and complete
  InOrbitalParallel branch1420–1435 read: `(header/2)==Pcount`, consuming
  `2*Pcount` indexed triples and writing `Slater[NArrayAP+idx]`.
  No missing-file C behavior or malformed unchecked scan is treated as safe
  defined parity. This milestone does not run or compile a C program.

## Rust paths and prospective verification

`parse_expert_mode_files` -> AP/P definition parsers -> declared dense
`slater_params` and mappings -> public `read_input_parameters`.
The independent expected five slots are
`[.25, -.5, .125-.25i, 11, 22]`; reserved slot2, projection/RBM storage,
optimization flags and orbital index/sign matrices must stay unchanged.
S070 width assertions and S071 end-to-end offset assertions are distinct
from existing programmatic `input_overlays.rs` coverage.

After parent SOURCE review, proposed bounded verification: whole parser
LIST/RUN, `--locked --cargo-profile test-fast`, jobs2/BLAS1,
`--no-tests fail --no-fail-fast --retries 0`, then parser all-target Clippy
`-D warnings` and fmt/source/input/tool prepost. Two new identities:
`original_s070_public_loader_preserves_ap_width_before_parallel_expansion`
and `original_s071_public_loader_reserves_unmapped_ap_slot_and_scatter_uses_header_offset`.
Original rows/owners/status counts unchanged. No full
API/family/model/RNG/MPI/CLI acceptance claim.

## Local acquisition checkpoint (historical base12c5)

ONE session81018, receipt `/tmp/mvmc-184-ap-offset-proof.20261004`:
focused2/2 PASS, 0 skipped, 0.007s, UUID
`49593254-7834-495f-a151-2ad230341d6e`; full parser275/275 PASS,
0 skipped, 0.375s. Parser all-target Clippy `-D warnings` and workspace
fmt check both0; source-before manifest replay0 and tool hashes replay0;
aggregate0. This is not whole-workspace, all-features, MPI, CLI or native
numerical execution. Tool pins were acquired after launch before final replay,
not a claimed complete pre-launch compiler/linker/provider closure.

Static gitlink inputs were hydrated by archive of main12c5's exact
`c0788c34a6a5753c611633a97cd1ea233203320c`, not by changing the original8bb
literal provenance. Test source SHA256
`24306ea07d3f67e2f6ec774e76d4005ec35c01467d4cfa5384c8301e9b3d95d7`
is unchanged after execution. This documentation checkpoint was appended
after the executed-source manifest; it does not retroassign a new doc hash
to that run. Publication must minimally reconcile to main
`34b5094d45e2e128ad406e21deafd2ed8a6a7ba0`, whose PR287 test/docs delta
does not change this production implementation. Parent review/CI required.
