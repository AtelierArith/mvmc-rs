# Condition pointers — bounded SOURCE audit, no new execution proof

Original documentation revision: 8bb1b9e8ae47b1512c00b321be05664ddcac0fd1.
All 17 original file hashes/line counts remain in the durable
[issue-184-doc-source-index.tsv](issue-184-doc-source-index.tsv).
Rust pointer source resolution is separately pinned in
[issue-184-doc-test-bindings.tsv](issue-184-doc-test-bindings.tsv): all nine
named issue184 test files plus physcal_issue181.rs and ctest_equivalent.rs
exist on exact main3b953a5099a500b1566f0e06107ca0d7a02e685b. Their Git blob IDs,
SHA256 bytes, line counts and explicit in-range line anchors were resolved
from that committed tree, not from dirty shared files. This is MainSourceBound,
not new independent execution proof. Each row retains the settings-equivalence
gap and owner; file-only pointers are not disguised as exact assertion anchors.
Original reviewed302-line audit SHA was
6f3692a124cdeeb9aa8bb56267a41a913be1641987678fc7a4f0ea97f1076c90;
that historical artifact remains preserved in its original owner packet.
This supplement does not promote that entire inventory to verified coverage.
Pointers below identify actual test contracts; fresh/current terminal evidence
and exact input equivalence remain separate, not inferred from matching names.

## docs/manual/02_input_files.md

- Lines3–6: resolve definitions relative to namelist. Public original-input
  entry is `issue184_original_heisenberg_public_loader.rs:81`,
  `original_heisenberg_namelist_exercises_public_loader_conditions`.
  This single16-site input does not cover every keyword in lines12–31.
- Lines23–28: DH/orbital/QP mapping is separate from loader success.
  `issue184_slater_contracts.rs:66`,
  `public_slater_update_composes_opttrans_before_fixed_translation_in_each_plane`
  checks composition; lines134/142 check6/16-site initialization, not trajectories.
- Lines35–51: initial→In* precedence and nine RBM overlay families need individual
  executed contracts. `issue184_optional_opttrans_overlays.rs:44/77/100`
  cover complex G components, OptTrans sign/maps and AP offsets respectively;
  they do not prove every DH/RBM layer or full initial-file lifecycle.
- Lines61–66: SpinJastrow rejection differs from supported TwoBodyGEx. Factored
  public-loader tests are `issue184_green_public_errors.rs:72/95/110` (valid row,
  short declared count, missing file), not runtime Green numerical proof.
- Lines75–77: PairHop bidirectional expansion is C input contract, not energy
  validation. InterAll line78 is individually excluded from #185, not the file.
- Lines79–87: seed resolution/group offsets and phase order remain runner/MPI
  owner evidence; parser acceptance alone is insufficient.

## docs/manual/03_optimization.md

- Lines25–31: nsteps/nsmp/seed/Auto/None/explicit initial-file wrapper contracts
  need actual public prepare/CLI settings. No complete six-argument proof here.
- Lines35–64: world2/4, split1/2, store0/1 and normal/FSZ/standard-QP positive
  matrix belongs #179; serial tests cannot replace those launched cells.
- Lines45–56: actual pre-mutation rejection pointers in
  `issue184_public_rejection_boundaries.rs:144/166/190/234/252`
  cover split3+CG1, CG2, grouped OptTrans, grouped general projection and
  unsupported CG submodes. Individual conditions remain distinct.
- Lines68–75: upstream1e-2 parameter tolerance is historical, not adopted.
- Lines94–103: four prefix models and13 ctest models are separate numerical
  obligations. A parser/layout test proves neither.
- Lines113–117: ParaOpt Lanczos rejection pointer is
  `issue184_public_rejection_boundaries.rs:170`; PhysCal Lanczos is separate.
- Lines125–128: per-step truncate/append and final block outputs require actual
  writer/CLI lifecycle gates, not mere filename construction.
- Lines132–158: environment timer precedence/empty/false values require exact
  process tests; timing parser contracts do not prove process activation.
- Lines162–165: cancellation warning supplies no numerical budget.

## docs/manual/04_physics_calc.md

- Lines15–21: entrypoint, all three Green families, DH, normalization and indexed
  output formats require separate proof. `physcal_issue181.rs` test
  `two_sample_runners_match_independent_saved_states_rng_and_ordered_outputs`
  is the bounded two-frame model reference entry; its historical source/gates
  must not be relabelled as current C near-node/H2 validation.
- Lines25–29: BackFlow rejection needs before-output/runtime boundary evidence.
- Lines30–38: grouped normal mode0 positives and standard QP are #179 launched
  matrix. `issue184_public_rejection_boundaries.rs:212` checks general/Lanczos
  rejection only; it cannot prove positive grouped normal runs.
- Lines39–41: only InterAll spin-metadata section is excluded.
- Lines50–57: independent six-model/CLI/Lanczos1/2 PhysCal remains #181/#174;
  ongoing C near-node repair and unreached cases are not marked complete.
- Lines59–60: fixed-record shape compatibility does not imply matching RNG or
  numerical parameters after optimization.

## examples/README.md

- Lines7–10 identify four16-site/model example callers. Each must map to its
  supported mode/input and actual public execution, not whole-crate commands.
- Lines20–26 document historical50-step default. User long-baseline20 is the
  current acceptance; original source is retained, not silently edited.

## MVMCExpertModeParsers.jl/README.md

- Lines9–15/81–96 distinguish parser/initialization/overlays/QP/validators from
  absent definition writers. Reader-only API difference is intentional; do not
  invent a Rust writer or mark every listed reader proved by a single namelist.
- Lines53–74/102–109: seed123456789, real-only flags, randomized optimized Slater,
  zero Proj, optional RBM and max4 rescale are distinct initialization settings.
  `issue184_slater_contracts.rs:142` checks original16-site seeded wiring only;
  RBM/complex draw order remains initialization owner's exact fixture proof.
- Lines115–120: overlay filepath and offsets map to manual02 contracts above.
- Lines127–132: QP Gauss4/Stot1/NMP1/phase1 is an EXACT documented setting;
  generic quadrature tests are not automatically that integrated QP case.
- Lines157–165: validation result shape is a typed-Rust API difference;
  particular accepted/rejected coordinates still require per-validator tests.
- Lines148–150: precision and SFMT stream identity are independent contracts;
  approximate floating numerical gates do not relax primitive RNG identity.

## MVMCOptimizers.jl/README.md

- Lines9–13: ParaOpt/PhysCal/MPI/threading positive and rejection settings reuse
  the distinct manual03/04 cells; do not promote the whole overview from one run.
- Lines40–46: explicit50/50 real seed11272 Auto example is historical. Current
  user long20 requirement does not authorize silently relabelling this input.
- Lines60–63: lower-level callback step/energy/info architecture is a separate
  hook contract; completed PhysCal #175 does not prove ParaOpt callback steps.
- Lines70–74: high-level driver, owned lower runner, PhysCal and fixed loader
  have distinct entrypoints. `issue184_public_rejection_boundaries.rs:270`
  exercises malformed/missing fixed preparation, not successful full PhysCal.
- Lines79–87: sampler/MainCal/SR/average/sync/QP and BackFlow correspondence
  names are API inventory, not numerical proof. BackFlow rejection unverified here.
- Lines89–91: state composition is represented by Rust typed structs; private
  Julia alias/cache identity is not automatically a required Rust API.

## README.md

- Lines9–14: distinguish all six status rows. Supported MPI is not absent merely
  because an example/benchmark is serial. Lanczos serial ordinary and rejected
  grouped/general modes follow manual04 plus actual public rejection gates.
- Lines18–35: Julia1.13.1+Manifest root/submodules is OPTIONAL reference setup;
  normal Cargo must not execute Julia/C oracle builds. Install/CLI smoke belongs
  #183; no install command executed by this SOURCE audit.
- Lines39–45: actual example exit/summary and mathematical energy need #183/#180
  proof. The illustrative -0.44... is not an independent expected numeric literal.
- Lines66–77: mixed licenses and gitlink pins need packaging proof, not runtime PASS.

## MVMCOptimizers.jl/test/README.md

- Lines18–34 enumerate quadrature, update, SR map, RBM/projection incremental
  comparison, QP/orbital wiring, OO/HO, sync and types. Existing
  `issue184_integration_tools_contracts.rs` quadrature tests and
  `issue184_assertion_contracts.rs` typed/helper tests are bounded leaf pointers.
  Original single-update/full-refactor and all RBM planes need their own gates.
- Lines30–33 explicitly preserve para_qp_trans during sync; normalizing OptTrans
  parameter storage is not permission to normalize phase factors. Preserve this
  condition in any sync-test mapping; current proof pointer pending domain owner.
- Lines38–48: separate C-reference integrations from synthetic unit tests and
  legacy samples. A passing unit suite cannot establish native reference parity.

## MVMCOptimizers.jl/test_unit/summary.md

- Lines15–21: flags/update mapping, nine RBM sections, analytic S/g are original
  specific contracts; current SR implementation/proofs belong #180/#176 owner.
- Lines28–34: historical25 PASS is provenance only, not a current gate count.
- Lines56–63: incremental/full RBM/projection including FSZ, OO/HO, sync and mock
  apply_hop require distinct contracts. `issue184_assertion_contracts.rs` test
  `original_sampling_hop_and_revert_preserve_exact_electron_buffers` is buffer
  lifecycle only, not sampling trajectory or all counter branches.
- Lines68–70: historical118 PASS explicitly skipped integrations; cannot be
  reused as mandatory native model/API proof.

## docs/manual/01_install.md

- Lines5–6/75–80: historical1.11/1.12 environments differ from mandated1.13.1
  reference Manifest. Native macOS portability needs actual CI/native results.
- Lines10–25/95–100: native toolchain/link prerequisites apply to optional Julia
  tooling and Rust BLAS features, not permission to call C/Julia from Cargo tests.
- Lines30–71/99/101: exact gitlink hydration/root-project relative siblings and
  missing dirs/project rejection are #183 setup cells; source inspection is not
  fresh installation proof. Retained generic launch only probes C, not Julia setup.
- Lines85–89: five-step four-example summary/exit gate must be actual #183 run;
  it is not replaced by user long20 numerical acceptance.

## THIRD_PARTY_LICENSES.md

- Lines14–43: Pfa BSD/MPL/Lapack/BLIS and SFMT BSD notices preserved individually.
  Generic native fixtures must retain original mVMC and Pfa license provenance;
  the duplicated private kernel cannot be labelled solely by Julia package GPL.
- Lines49–66: snapshots, input definitions and fixed PhysCal files are separate
  artifacts; no source/runtime oracle dependency in normal Rust tests.
- Lines71–79: master5e7 optimisation, develop66f PhysCal and local622 DH fixture
  lineages cannot be relabelled as our original1.3.0 GNU13 native kernel probe.
- Lines83–99: credit/licensing review is packaging evidence, not numerical PASS.

## docs/manual/README.md

- Lines3–7 are navigation only. Each linked chapter is expanded separately here;
  this file adds no executable scenario or independent test assertion.

## CHANGELOG.md

- Lines12–28/42–52 are v0.5 scope: ordinary serial Lanczos1/2, real/FSZ PairHop,
  grouped standard-QP directSR, grouped normal PhysCal, FSZ factored Green,
  RBM fixed loaders. Reuse manual03/04 distinct numerical/rejection cells;
  #181 native near-node work proves none of the entire grouped matrix.
- Lines68–76/89–95 are v0.4.2 scope: CG split1 and direct split/store0/1 QP1.
  Its historical PhysCal split rejection is superseded by v0.5 lines22–24.
- Lines111–128/132–137 are v0.4.1 MPI failures, timers, real-kernel reuse and
  site-performance work. #179 handles actual widths, #182 timers/threading;
  no performance claim or result is adopted in this mapping.
- Lines154–186 are v0.4 launcher detection, rank0 read/write/warnings, seedgroup,
  CG and duplicate-index overlays. `issue184_optional_opttrans_overlays.rs:100`
  is offset-only; it is not all duplicate-index overlay or MPI warning proof.
- Lines202–215/222–234 are v0.3 normalGreen/DH/offset/local accumulator and
  sequential-chain contracts. Old blanket MPI/Lanczos rejection is historical.
- Lines249–258/271–274 describe statistical12model gates, timer env, platform
  and1.11 manifest history. They do not replace exact primitive/discrete checks
  or authorized1.13.1 reference/runtime policy. Lines281–283 add no new action.

## MVMCOptimizers.jl/test_unit/plan.md

- Lines19–25: parameter flags/writeback nineRBM and analytic S/g remain SR owner
  per-assertion conditions, not full model proof from the planning file.
- Lines30–38: RBM one-hop/two-hop incremental/full and log-value differences
  require all applicable channels. Real native MakeRBMCnt grouping repair is
  specific evidence; no blanket claim all these planned assertions passed.
- Lines43–48/53–59: projection normal/FSZ updates, diff offsets and OO/HO are
  separate sampling/SR leaf contracts, including alias buffers and lengths.
- Lines64–67: max4 rescale and G/J/DH shifts map to bounded
  `issue184_assertion_contracts.rs` literals, not stochastic trajectory.
- Lines74–78: preserve two spin codes vs derived-spin architecture difference;
  `issue184_literal_hamiltonian_parsers.rs` Transfer payload gate only.
- Lines83–97: count/flags/offsets and seed-fixed init require initializer owner
  fixtures. Parsing nine blocks is insufficient for draw-order proof.
- Lines104–107: synthetic mock builders are test architecture, not public
  native-input support. Lines111–116 historical execution/runtime statements
  do not authorize dependency resolve in ordinary Cargo tests.
- Lines131–136: historical blanket split rejection is superseded by actual
  v0.5 contract, while DH/PhysCal/threading remain their distinct current goals.

## MVMCOptimizers.jl/test_unit/INDEX.md

- Lines25–32: sequential xdot, sampled-product-before-global-correction,
  DH2/DH4 writeback and nineRBM offsets map to current SR kernel-specific tests.
  Public source symbol matching alone does not validate each expected cell.
- Lines36–61: RBM/log, DH strides/alias/FSZ/real-part log, hop/revert buffers
  remain sampling leaf tests. Buffer roundtrip is not proposal/acceptance proof.
- Lines65–68: supplied signs/maps/missing weights/identity fallback are bounded
  `issue184_slater_contracts.rs:7/46/66/142` and orbital parser tests; no full PF.
- Lines72–79: diff/OO/HO/BLAS inactive tails/threaded equality are separate
  branches; normalized full arrays and final SR update are not interchangeable.
- Lines83–84: real active-prefix direct/CG and size1 collectives are average
  contracts; empty/small-Wc guard differences remain documented separately.
- Lines88–100: shift flags, DH combined offsets, QP phase preservation, max4
  and typed dimensions reuse bounded tests, not complete runtime scope.
- Lines104–109: prefix copy, energy/SR/Phys merge, counter/timer and local reset
  differ in ownership. `issue184_accumulator_lifecycle_contracts.rs:14/88/105`
  covers buffer lifecycle/factored/all shapes, NOT Julia local-store clear or
  merge/cache identity (intentional architecture differences/source-only).
- Lines113–131: match each accepted/rejected split/CG/Lanczos combination to
  `issue184_public_rejection_boundaries.rs:144/155/166/170/178/190/212/234/252`.
  Positive grouped mode needs #179 launches, not negative boundary PASS.
- Lines135–148: flatten16/conjugation, canonical order/dedup, pair1-based→Rust
  zero-based layout, legacy no-acc, FSZ dispatch, output index/truncate and
  struct wiring are IO134 leaf conditions. `physcal_issue181.rs` original_io134
  cases distinguish synthetic literals from full-model/native six-case proofs.
- Lines152–158: strict fixed loader consumed count/idempotence/DH/OptTrans,
  malformed input and warning behavior are separate. Historical RBM rejection
  is not current support prohibition; modern RBM loader lineage stays explicit.
- Lines162–164: missing fixed before sampling points to public rejection test
  at270; argument-validation mode precedence and no double RNG require #175
  source/gates, not loader alone.
- Lines168–173: comparator layout/index/finite negative controls separate from
  numerical e2e. Listed five models differ from reference README six including
  FSZ; inventory both, do not discard sixth. Old per-quantity tolerances are not
  new bounds for C near-node or Lanczos alpha cancellation.
- Lines179–182: Transfer spin and RBM layer offset parser contracts remain
  separate from optimizer implementation, as in plan lines74–97.

## docs/manual/05_compatibility.md

- Lines7–22: caller correspondence maps to Rust run/sampling/observables/QP/
  sync/average; status icons are not execution proof. Current helper/gate
  pointers are the corresponding manual02/03/04 cells above.
- Lines32–36: seeded Python perturbations are committed INPUTS, not permission
  to run AddRand or generate expected results during Cargo tests.
- Lines46–69: four first10, one CGfirst1,13statistical models are distinct.
  1e-2CG and3std statistical acceptance cannot hide algorithm/discrete drift.
- Lines86–102: exactSFMT/seedgroups vs approximate derived numerics must be
  separated; current190 operation/residual budgets supersede generic BLAS noise.
- Lines109–116: disjoint inner loops vs serial sample/chain outer loop are #182
  source-specific activation/identity proofs, not testname or capacity claims.
- Lines120–134: debug QP threads/lockoff/independent chains are intentionally
  not required supported execution modes; reject/unsupported category, no PASS.
- Lines138–146: old LP64-only environment statement cannot relabel current
  Julia ILP64/runtime captures or native portability CI. Record actual providers.
- Lines150–174: BackFlow/grouped-QP/general-FSZ/CG/Lanczos restrictions map to
  exact public rejections above. Conservative Julia recomputation does not
  override C update/near-node numerical authority (active #181 source repair).
- Lines175–177: only omitted InterAll spin-metadata branch excluded, not file.

## test/integration/reference/README.md

- Lines14–32/38–53: master5e7, gcc15/macOS, strict4model first10 vs13ctest,
  separateCG and real/FSZ PairHop. `ctest_equivalent.rs` entrypoint is a pointer;
  retained current52-run scope/results must be checked individually by #180.
- Lines59–65/141–170: CGfixed1 parameters/tol1e-10/maxiter0/split1 and rank2
  nativeOpenBLAS fixture differs from Accelerate reference. Sameoperand/residual
  and currentC recurrence tests required; old coarsebudget not imported.
- Lines75–89: PairHop input expansion+1step/1sample applies two directions;
  literal parser test is not this C model execution or PhysCal near-node proof.
- Lines95–137: regeneration commands are optional developer workflow only,
  not current execution results and never called by normal Cargo tests.
- Lines174–203: exactsix native PhysCal models, mode1/fixedfile/threeGreen
  filenames, DH layout, FSZ AP+P ordering, separateopt inputs. #181 library and
  #174 positive CLI gates must match each; historicalJulia two-frame models
  are additional independent references, not relabelled as these native six.
- Lines207–219: develop66f vs622DH/FSZ, Clang/gfortran15/Accelerate/singleMPI
  remains historical lineage; generic GNU13 original1.3.0 probe is separate.
- Lines235–263: Hubbard/Spin Lanczos mode1 fixedfile/ls_out/16QQQQ are native
  R1 two-model contracts. Lanczos2 indexed allGreen and near-node branches
  remain additional current requirements, not proved by R1 fixture availability.

All17 documents now have bounded per-section SOURCE classifications. This is
NOT complete exact-assertion/executable coverage: unresolved settings and named
test-to-condition equivalence are explicitly pending owner/runtime evidence.
Original revision/hashes remain unchanged in the17-file inventory. No new Rust
tests/model runs executed and no shared ledger changed.
