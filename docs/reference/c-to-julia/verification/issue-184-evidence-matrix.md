# Julia-mVMC → Rust evidence matrix (issue #184)

Foundation inventory reviewed 2026-10-03 against [#184](https://github.com/AtelierArith/mvmc-rs/issues/184), [#185](https://github.com/AtelierArith/mvmc-rs/issues/185) and its re-audit comments. This is an inventory and evidence ledger, not a completion claim.

## Baselines, scope and status

Issue audit baseline: Rust `f1167c16f23d5929ccf312e56d48a9787e5c6bbd`.
Re-audit/current checkout HEAD: `30d8d69ffc6d2700c56fda81841827a49e72e57c`, with uncommitted runner, CLI, callback, MPI, validation and test changes inspected in this shared tree. These changes are not evidence of a successful execution.
Julia reference J: `extern/Julia-mVMC` at `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`.
C reference C: vendored `extern/mVMC-1.3.0`; individual fixture generator revision/hash/build provenance must be recovered from its metadata before verification.

C is authoritative for input and numerical contracts; Julia supplies API architecture and scenario coverage. New Julia comparisons require Julia 1.13.1 and `Manifest-v1.13.toml`. Historical fixtures, including Julia 1.11 results, are not new 1.13.1 evidence. Follow #186's native validation prerequisite and #190's portable numerical policy; no new tolerance is established by this document. RNG initialization/state/draw order/count/conversions, proposals, acceptance and saved configurations remain exact.

| Axis | Meaning |
| --- | --- |
| executable | Production caller exists for the stated scope; says nothing about parity. |
| partial | Caller or coverage exists, with named missing combinations or unrun checks. |
| rejected | Explicit input/runtime rejection; prove the error boundary separately. |
| excluded / Julia-deferred | Scoped exclusion or an upstream runner limitation; never infer a Rust gap from it. |
| unverified / not-run | No sufficient executed independent evidence is recorded. |
| verified | An actual executed comparison against independent C/Julia expectations has a reproducible command, settings, reference/provenance, result and artifact. A fixture, test source, self-comparison, smoke result, ignored test or feature-disabled return alone does not qualify. |

Implementation state and evidence state are separate. No row below is promoted to verified during this documentation repair.

Owners: #174 CLI PhysCal; #175 PhysCal callback; #176 real-FSZ inverse divergence; #177 root-resolved negative MPI seed; #178 rejection/collective failures; #179 MPI scenario comparisons; #180 native ctest models; #181 serial PhysCal/Green/non-InterAll Lanczos; #182 inner threading. #183 owns optional-gate reporting only. #184 owns inventory/provenance holes; #185 is the umbrella, not the default execution or implementation owner.

Scoped exclusions: InterAll implementation, its Lanczos and validation belong elsewhere (including Kitaev sample material); Julia FFI exports are reference adapters, not Rust runtime requirements. Standard-mode input generation/StdFace, plotting assets, fixture generators and mock helpers are inventoried as aids, not production port promises. Julia-deferred ctest Lanczos wrappers do not exclude Rust non-InterAll PhysCal Lanczos. Julia-only extensions/corrected C semantics require a labelled authority decision, not automatic adoption.

## Evidence record convention

Every row below references J/C and an evidence record E0–E9. These records supply command, settings, result and provenance fields; inheritance applies to every named member of a row, not just the first. Split rows when results differ.

| ID / gate type | Reference / fixture provenance | Command | Settings | Result / missing evidence |
| --- | --- | --- | --- | --- |
| E0 source inventory | J exports/tests/examples and current dirty Rust tree; no comparison fixture | `codegraph explore "extern/Julia-mVMC/src exports test runtests examples; crates/mvmc-core/src/run.rs PhysCal callback runner; crates/mvmc-cli/src/main.rs"`, then export/file inventory reads | HEAD/J above; source inspection only; runtime/BLAS/seed not applicable | Mapping inspected; execution not-run. CodeGraph did not locate Julia packages at the queried src path; actual package paths below were read afterward. |
| E1 reported focused regression | #185 body/comments; Julia/C-backed tests, exact fixture provenance not itemized in report | `cargo nextest run -p mvmc-core --test runner_config --test runtime_contract --test two_body_green --test real_fsz_setup --test mpi_physcal --no-fail-fast --retries 0` | Earlier audit; default profile/features; platform, BLAS, compiler, per-test seeds not recorded | Reported 36 tests: 35 pass, 1 fail; real-FSZ inverse bit comparison failed before retry/RNG assertions. MPI feature-disabled path only. Not rerun here; insufficient for blanket verified status. |
| E2 kernel/contract candidates | C SFMT vectors; historical Julia PfaPack and parser fixtures; recover generator versions from fixture READMEs | Planned `cargo nextest run -p sfmt19937 --test golden_vs_c`; `cargo nextest run -p pfapack --test golden_vs_julia`; `cargo nextest run -p mvmc-expert-parsers` | Seeds/layout/backend cases per test; profiles/features and actual BLAS must be logged | Not-run here; fixture existence is inventory evidence only. |
| E3 runner/output candidates | J integration/reference; Rust phase4/phase5 fixtures and provenance docs | Planned `cargo nextest run -p mvmc-core --cargo-profile test-fast --test run_smoke --test initial_params --no-fail-fast --retries 0` | Record seed, initial/overlay mode, real/cmp/fsz, direct/CG, NStore, nsteps/nsmp, projection flags | Not-run; smoke alone cannot establish independent parity. Optional phase gates require separate explicit selection (#183). |
| E4 CLI contract | Current `crates/mvmc-cli/tests/runtime_contract.rs`; J PhysCal runner/loader contract | Planned `cargo nextest run -p mvmc-cli --test runtime_contract --no-fail-fast --retries 0` | Serial default features; --physcal path, inferred/explicit mode, seed, --out-dir, OptTrans, missing file and contradictory flags | Not-run; independent output/CLI parity and real MPI invocation remain missing. |
| E5 callback contract | Current `crates/mvmc-core/tests/physcal_callback.rs`; J vmc_phys_cal! | Planned `cargo nextest run -p mvmc-core --test physcal_callback --no-fail-fast --retries 0` | Completed-sample index/averaged energy; callback error; mock reducer; files/RNG compared with no-callback run | Not-run; self-comparison/mock reducer is local contract evidence only; independent Julia replay and real rank-local failure remain missing. |
| E6 serial PhysCal/Lanczos | J six-model `physcal_ref/{metadata.txt,zqp_opt.dat,inputs,expected}`; `integration-reference-data.md` | Planned `MVMC_RS_PHYSCAL_181=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test physcal_issue181 --no-fail-fast --retries 0`; `MVMC_RS_LANCZOS_PHYSICAL=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test lanczos_transfer_physcal --run-ignored only` | J fixture mode/seed/fixed record per model; JULIA_MVMC_ROOT when needed; Lanczos selector MVMC_RS_LANCZOS_MODE/MODEL; current #181 tests use atol 1e-10/rtol 1e-8, justification still to audit | Not-run; six-model output, fixed-value/flags, multi-sample and modes 1/2 assertions are candidate evidence, not validated results. New Julia 1.13.1 rerun absent. |
| E7 long ctest | J `ctest_models.jl`, per-model `ctest_ref/ref_mean.dat,ref_std.dat` and metadata | Planned `MVMC_RS_CTEST_MODELS=heisenberg_chain_real,hubbard_chain_real cargo nextest run -p mvmc-core --cargo-profile test-fast --test ctest_equivalent --run-ignored only` | Explicit model selection; native nsteps/nsmp, seeds, mode, direct/CG/storage; record statistical thresholds separately from exact trajectory checks | Not-run; selecting these two is not thirteen-model coverage; unsupported harness cells require #180 work. |
| E8 MPI | J test/mpi scripts and C/J rank references; earlier smoke reports need exact logs/provenance | Pending reproducible launcher commands per executable/rank matrix; candidate gate requires MVMC_RS_MPI_PHYSICAL=1, mpi feature and explicit ignored-test selection | 2/4 ranks; grouped/ungrouped; direct/CG; NStore 0/1; OptTrans; positive/0/negative root seed; output root and communicator; record MPI/BLAS/threads | Not-run; no complete current communication/value/RNG/failure comparison. Feature-disabled pass does not qualify. |
| E9 threading/performance | J threading.jl and test_unit_threading.jl; Rust Pfaffian/QP/helper tests | Pending worker/backend comparison and benchmark command for each mapped call site | Workers 1/2/4, BLAS threads/provider, model/mode and same seed; compiler/platform/revisions required | Not-run; helper coverage does not establish all Julia threaded call sites or performance parity. |

For an executed record add date, exact Rust/J/C revisions (including dirty patch identity), architecture/OS, compiler/Rust/Julia versions, actual BLAS/MPI providers, workers/ranks, all settings/seed overrides, gate selection, comparison/residual metrics, first divergence, tolerance justification, discrete-contract outcome and saved log/artifact. Unknown fields stay explicitly unknown. Normal Rust tests must ultimately use committed independent fixtures without invoking Julia/C or reading toolbox programs.

## Explicit top-level export inventory

Paths in this table are relative to `extern/Julia-mVMC/`; Rust modules are under `crates/`. Grouped symbols share a caller/evidence scope but each symbol is explicitly named. Parser types and qualified helpers are not top-level exports at J.

| Julia source / exported symbols | Rust library / CLI caller | Implementation; evidence | Owner |
| --- | --- | --- | --- |
| `SFMT.jl/src/SFMT.jl: SFMT19937RNG, init_gen_rand` | `sfmt19937::Sfmt19937Rng::{new,seed}; runner seed initialization` | `executable; E2 unverified` | `#184; MPI seed #177` |
| `SFMT.jl/src/SFMT.jl: genrand_real2, gen_rand32, sfmt_dump_rand32` | `Sfmt19937Rng::gen_rand32 and float conversions; draw vectors` | `executable conversion path; dump helper mapping partial; E2` | `#184` |
| `PfaPack.jl/src/PfaPack.jl: pfaffian_ltl!` | `pfapack::pfaffian_ltl_real / pfaffian_ltl_complex` | `executable kernel only; E2` | `#184` |
| `PfaPack.jl/src/PfaPack.jl: julia_zsktf2!, julia_dsktf2!, julia_zsktf2_turbo!` | `pfapack::{zsktf2,dsktf2,zsktf2_turbo}` | `executable kernel only; E2; inverse failure E1` | `#176; thread scope #182` |
| `PfaPack.jl/src/PfaPack.jl: utu2pfa, utu2inv!` | `pfapack UTU2 real/complex functions` | `executable kernel only; E2` | `#176 / #184` |
| `PfaPack.jl/src/PfaPack.jl: fimpl_zsktf2!, fimpl_dsktf2!, cimpl_utu2inv!` | `pure Rust kernels; no FFI-equivalent runtime obligation` | `excluded adapters; E0` | `#184 scope` |
| `MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl: parse_expert_mode_files` | `mvmc_expert_parsers::parse_expert_mode_files* → run/CLI` | `executable; family/error/ordering matrix partial; E2` | `#184; runtime rejection #178` |
| `MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl: init_qp_weight!, update_qp_weight!` | `mvmc-core qp initialization/update → runner` | `executable; QP/sign/default combinations unverified; E2/E3` | `#180 / #181; inventory #184` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: ParallelContext, serial_context, build_parallel_context, is_output_rank` | `parallel/mpi contexts; Reducer::is_output_root; run wrappers/CLI` | `partial serial/grouped scope; E8` | `#178 / #179` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: bcast!, bcast_scalar, allreduce_sum!, reduce_sum_to_root!, barrier` | `Reducer/mpi collective operations via runner (not one-to-one exports)` | `partial helper mapping; E8` | `#179` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: reduce_counter!, allreduce_sum_scalar, allreduce_max_scalar` | `reducer accumulator/counter operations; scalar helper equivalence to audit` | `partial; E8` | `#179; mapping #184` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: abort_parallel, split_loop, split_range, qp_split_range` | `validation/collective failure agreement, parallel partition and QP ranges` | `partial; grouped failure/partition boundaries unverified; E8` | `#178 / #179` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: resolve_rnd_seed` | `runner prepare wrappers + Reducer root seed broadcast` | `new dirty-tree path partial; negative/0/positive actual MPI unrun; E8` | `#177` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: count_total_parameters, pack_parameters, unpack_parameters!` | `parser layout + core sync/state parameter conversion` | `partial architectural mapping; all families/offsets/records E2/E3` | `#180 / #181; inventory #184` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: set_parameter_value!, get_parameter_value` | `core sync/parameter storage access; no claimed one-to-one wrapper` | `partial helper mapping; sparse/shared/unmapped/signed flags E2/E3` | `#184` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: read_initial_def!, read_opt_para_file!` | `mvmc_core::{read_initial_def,read_opt_para_file}; prepare/run/CLI` | `executable; record selection/overlays/fixed values partial; E3/E6` | `#181; inventory #184` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: vmc_para_opt!` | `mvmc_core::vmc_para_opt; optimization callback/sampling-only` | `executable; callback/history/skip-SR combinations unverified; E3/E7/E8` | `#180 / #179; inventory #184` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: run_para_opt_from_namelist` | `run_para_opt_from_namelist[_with_reducer] → CLI optimization` | `executable; full model/mode/store/output scope partial; E3/E7` | `#180 / #179` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: vmc_phys_cal!` | `vmc_phys_cal[_to_dir/_with_reducer]; new *_with_callback and PhysCalCallback` | `partial new callback/error agreement; E5/E6/E8 unrun` | `#175 / #181 / #178 / #179` |
| `MVMCOptimizers.jl/src/MVMCOptimizers.jl: run_phys_cal_from_namelist` | `prepare_phys_cal_from_namelist[_with_reducer_and_opt_trans] → vmc_phys_cal; CLI --physcal PATH` | `partial dirty runner/CLI dispatch; E4/E6/E8 unrun` | `#174 / #181 / #179` |

## Explicit Julia test inventory

All paths below are relative to extern/Julia-mVMC at J. Each script is an individual unverified/not-run cell inheriting E0 and its listed evidence record. This is a file inventory; assertion-by-assertion mapping remains #184 work. Helpers and generators are scoped reference aids.
 
| Julia script | Rust caller scope | Record / owner |
| --- | --- | --- |
| `MVMCExpertModeParsers.jl/test/runtests.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_doublon_holon_parser.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_integration.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_orbital_qptrans_utils.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_parameter_init_complexflag_rbm.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_parameter_initialization.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_parse_expert_mode_files.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_parsers.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_qp_weight.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_read_input_parameters.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_read_input_parameters_rbm_layout.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_sfmt_compatibility.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_trans_parser_spin_indices.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_utils.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCExpertModeParsers.jl/test/test_validation.jl` | parse_expert_mode_files* → validation/init/QP | E2; #184; not-run |
| `MVMCOptimizers.jl/test/runtests.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test/test_slater_update.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/helpers/mock_data.jl` | mock/reference aid; excluded production obligation | E0; #184 scope; not-run |
| `MVMCOptimizers.jl/test_unit/helpers/mock_state.jl` | mock/reference aid; excluded production obligation | E0; #184 scope; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_parallel.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_parameter_sync.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl` | prepare → vmc_phys_cal → Green/output | E6; #181; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_read_opt_para.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_run_phys_cal_runner.jl` | prepare → vmc_phys_cal → Green/output | E6; #181; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_slater_update.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_stochastic_opt.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_threading.jl` | Pfaffian/QP; complete call-site mapping pending | E9; #182; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_types.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl` | validation → runner/CLI runtime_contract | E4/E8; #178; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_vmc_main_cal_sr.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_vmc_sampling_misc.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_vmc_sampling_proj.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_vmc_sampling_qp_split.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_vmc_sampling_rbm.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `MVMCOptimizers.jl/test_unit/test_unit_weight_average.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `PfaPack.jl/test/runtests.jl` | pfapack golden_vs_julia | E2; #176 / #184; not-run |
| `SFMT.jl/test/runtests.jl` | sfmt19937 golden_vs_c | E2; #184; not-run |
| `test/integration/ctest_equivalent.jl` | runner → ctest_equivalent | E7; #180; not-run |
| `test/integration/ctest_models.jl` | runner → ctest_equivalent | E7; #180; not-run |
| `test/integration/lanczos_equivalent.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `test/integration/pairhop_equivalent.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `test/integration/phys_cal_equivalent.jl` | prepare → vmc_phys_cal → Green/output | E6; #181; not-run |
| `test/integration/runtests.jl` | runner/sampling/sync; detailed assertion mapping pending | E3; #180 / #184; not-run |
| `test/integration/test_run_phys_cal_contract.jl` | prepare → vmc_phys_cal → Green/output | E6; #181; not-run |
| `test/integration/tools/generate_ctest_fixtures.jl` | mock/reference aid; excluded production obligation | E0; #184 scope; not-run |
| `test/integration/tools/green_compare.jl` | mock/reference aid; excluded production obligation | E0; #184 scope; not-run |
| `test/integration/tools/test_green_compare.jl` | prepare → vmc_phys_cal → Green/output | E6; #181; not-run |
| `test/mpi/mpi_failure_modes.jl` | validation → runner/CLI runtime_contract | E4/E8; #178; not-run |
| `test/mpi/mpi_hubbard_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `test/mpi/mpi_nsplit_nstore_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `test/mpi/mpi_nsplit_standard_projection_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `test/mpi/mpi_physcal_nsplit_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `test/mpi/mpi_physcal_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `test/mpi/mpi_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `test/mpi/mpi_srcg_e2e_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `test/mpi/mpi_srcg_operate_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `test/mpi/mpi_weight_average_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |
| `test/mpi/run_mpi_smoke.jl` | parallel/mpi/reducer → runner | E8; #177 / #178 / #179; not-run |

J integration/runtests.jl includes lanczos_equivalent.jl, but that file was absent from the on-disk file inventory. This is an #184 inventory hole, not executed Lanczos evidence.

## Individual model and example cells

| Julia inventory / fixture | Rust caller / implementation scope | Evidence / owner |
| --- | --- | --- |
| test/integration/ctest_models.jl: `heisenberg_chain_real` | run_para_opt_from_namelist; ctest_equivalent supported flag | E7 not-run; #180 |
| test/integration/ctest_models.jl: `heisenberg_chain_cmp` | run_para_opt_from_namelist; ctest_equivalent supported flag | E7 not-run; #180 |
| test/integration/ctest_models.jl: `heisenberg_chain_fsz` | run_para_opt_from_namelist; ctest_equivalent supported flag | E7 not-run; #180 |
| test/integration/ctest_models.jl: `hubbard_chain_real` | run_para_opt_from_namelist; ctest_equivalent supported flag | E7 not-run; #180 |
| test/integration/ctest_models.jl: `hubbard_chain_cmp` | run_para_opt_from_namelist; ctest_equivalent unsupported harness flag; actual input rejection vs implementation gap vs unverified execution still to distinguish | E7 not-run; #180 |
| test/integration/ctest_models.jl: `hubbard_chain_fsz` | run_para_opt_from_namelist; ctest_equivalent unsupported harness flag; actual input rejection vs implementation gap vs unverified execution still to distinguish | E7 not-run; #180 |
| test/integration/ctest_models.jl: `kondo_chain_real` | run_para_opt_from_namelist; ctest_equivalent unsupported harness flag; actual input rejection vs implementation gap vs unverified execution still to distinguish | E7 not-run; #180 |
| test/integration/ctest_models.jl: `kondo_chain_cmp` | run_para_opt_from_namelist; ctest_equivalent unsupported harness flag; actual input rejection vs implementation gap vs unverified execution still to distinguish | E7 not-run; #180 |
| test/integration/ctest_models.jl: `kondo_chain_stot1_cmp` | run_para_opt_from_namelist; ctest_equivalent unsupported harness flag; actual input rejection vs implementation gap vs unverified execution still to distinguish | E7 not-run; #180 |
| test/integration/ctest_models.jl: `kondo_chain_fsz` | run_para_opt_from_namelist; ctest_equivalent unsupported harness flag; actual input rejection vs implementation gap vs unverified execution still to distinguish | E7 not-run; #180 |
| test/integration/ctest_models.jl: `general_rbm_cmp` | run_para_opt_from_namelist; ctest_equivalent unsupported harness flag; actual input rejection vs implementation gap vs unverified execution still to distinguish | E7 not-run; #180 |
| test/integration/ctest_models.jl: `hubbard_tetragonal_real` | run_para_opt_from_namelist; ctest_equivalent unsupported harness flag; actual input rejection vs implementation gap vs unverified execution still to distinguish | E7 not-run; #180 |
| test/integration/ctest_models.jl: `hubbard_tetragonal_momentum_projection_real` | run_para_opt_from_namelist; ctest_equivalent unsupported harness flag; actual input rejection vs implementation gap vs unverified execution still to distinguish | E7 not-run; #180 |
| ctest_models.jl: `spin_chain_lanczos` | Julia-deferred para-opt ctest wrapper; Rust non-InterAll measurement scope assessed separately | E6 not-run; #181 |
| ctest_models.jl: `hubbard_chain_lanczos` | Julia-deferred para-opt ctest wrapper; Rust non-InterAll measurement scope assessed separately | E6 not-run; #181 |
| phys_cal_equivalent.jl: `heisenberg_chain_real/physcal_ref` | prepare → measurement → indexed Green/output; fixed-record decode, strict indices/rows and independent trajectory pending | E6 partial/not-run; #181 |
| phys_cal_equivalent.jl: `heisenberg_chain_cmp/physcal_ref` | prepare → measurement → indexed Green/output; fixed-record decode, strict indices/rows and independent trajectory pending | E6 partial/not-run; #181 |
| phys_cal_equivalent.jl: `heisenberg_chain_fsz/physcal_ref` | prepare → measurement → indexed Green/output; fixed-record decode, strict indices/rows and independent trajectory pending | E6 partial/not-run; #181 |
| phys_cal_equivalent.jl: `hubbard_chain_real/physcal_ref` | prepare → measurement → indexed Green/output; fixed-record decode, strict indices/rows and independent trajectory pending | E6 partial/not-run; #181 |
| phys_cal_equivalent.jl: `hubbard_chain_dh_real/physcal_ref` | prepare → measurement → indexed Green/output; fixed-record decode, strict indices/rows and independent trajectory pending | E6 partial/not-run; #181 |
| phys_cal_equivalent.jl: `kondo_chain_real/physcal_ref` | prepare → measurement → indexed Green/output; fixed-record decode, strict indices/rows and independent trajectory pending | E6 partial/not-run; #181 |
| examples/heisenberg_chain_real.jl | run_para_opt_from_namelist; CLI examples/inputs/heisenberg_chain_real/namelist.def | E3 not-run; #180 |
| examples/heisenberg_chain_cmp.jl | run_para_opt_from_namelist; CLI examples/inputs/heisenberg_chain_cmp/namelist.def | E3 not-run; #180 |
| examples/heisenberg_chain_fsz.jl | run_para_opt_from_namelist; CLI examples/inputs/heisenberg_chain_fsz/namelist.def | E3 not-run; #180 / #176 |
| examples/hubbard_chain.jl | run_para_opt_from_namelist; CLI examples/inputs/hubbard_chain_real/namelist.def | E3 not-run; #180 |

## Combination and failure-boundary coverage

All cells inherit J/C, command/settings/result details from the cited records.

| Julia scope / source | Rust library/CLI caller | Implementation / missing evidence | Owner |
| --- | --- | --- | --- |
| Parser src/parsers: ModPara, LocSpin, Trans, CoulombIntra/Inter, Hund, Exchange, PairHop, Gutzwiller, Jastrow, orbital AP/parallel/general, QPTrans, OneBodyG/TwoBodyG/TwoBodyGEx, DH2/DH4, RBM | parse_expert_mode_files* → runner/CLI | E2 partial; C headers/widths/signed flags/errors and full family inventory unverified; InterAll excluded | #184; rejection #178 |
| Declared widths, sparse/shared/unmapped slots, signed flags; parser utils/read_input_parameters.jl/parameter_init.jl | parser layout → init/sync | E2/E3 partial; all family offsets, initialization and exact draws unverified | #184 / #180 / #181 |
| initial.def Auto/None/Path, fixed records, multiple records, In overlays; initial_params.jl | read_initial_def/read_opt_para_file → prepare/run/CLI | E3/E6 partial; ordering, independent record decode and fixed-value combinations missing | #181 / #184 |
| Seed 0/positive/negative, normalization, QP, signs | init → prepare/run/reducer | E2/E3/E8 partial; root seed broadcast and full discrete trajectories unrun | #177 / #180 / #181 |
| Direct SR / SR-CG × NStore 0/1 × real/cmp/FSZ/general | vmc_para_opt → solver/storage | E3/E7/E8 partial; full native model/mode/storage/residual matrix missing | #180 / #179 |
| DH2/DH4, charge/spin/general RBM combinations | sampling/derivatives/measurement runner | E3/E6/E8 partial; combined layouts, measurement and MPI missing | #180 / #181 / #179 |
| OptTrans defaults/enabled × optimization/PhysCal/grouped | parser C OptTrans → runner/CLI validation | E3/E4/E6/E8 partial; supported/rejected contract and independent parity missing | #178 / #180 / #181 / #179 |
| Sampling-only, optimization callbacks, history, summaries, block/final-window output; vmc_para_opt.jl/data_io.jl | vmc_para_opt / run summaries / io | E3/E7/E8 partial; family/flags/nsmp/root and unchanged RNG/files checks missing | #180 / #179; inventory #184 |
| PhysCal sample callback | *_with_callback / PhysCalCallback | E5 partial dirty implementation; independent Julia callback comparison and actual MPI failure agreement unrun | #175 / #178 / #179 |
| Green weights, canonical indices, deduplication, multi-sample output | observables → vmc_phys_cal → io | E6 partial; strict row/discrete index proof missing | #181 |
| Non-InterAll Lanczos 1/2, model/Hamiltonian/output scope | lanczos → PhysCal → io | E6 partial; full supported Hamiltonian matrix and independent values missing | #181 |
| MPI grouped/ungrouped optimization/PhysCal, output roots and rank-local failures | reducer/mpi/runner/CLI backend | E8 partial; current Rust CLI rejects PhysCal NSplitSize > 1; distinguish Rust limitation from Julia's restricted grouped normal-Green support | #178 / #179; CLI #174 |
| Timers, diagnostics, transfer fast path; c_timer.jl/vmc_main_cal.jl | core timer/diagnostic/observable paths | E3/E9 unverified; execution trajectory/performance evidence missing | #182; inventory #184 |
| Inner workers vs BLAS workers; threading.jl | selected Pfaffian/QP helpers | E9 partial; all Julia threaded call sites and 1/2/4 worker matrix missing | #182 |
| Invalid modes/flags/files/ranks, callback/output errors | validation → prepare/run/CLI | E4/E5/E8 partial; exact rejection boundary, caller state/RNG and file/directory snapshots required; callback error follows completed sample/output, so rollback is not presumed | #178 / #175 |

## Current #181 evidence limitations

Working-tree physcal_issue181.rs is candidate coverage, not full #181 verification:

- assert_reference flattens rows and applies tolerance to indices and floats alike. It does not establish exact row count/width/order, discrete integer/index columns, duplicates or file structure. Future repair must validate row structure and discrete columns strictly, using justified tolerances only for numerical columns.
- fixed_values compares Rust-before and Rust-after snapshots. It can detect mutation but does not prove independently expected C record/layout decoding. An independent decoded parameter vector and flags are missing.
- Rerun compares Rust with Rust, without an independent RNG-state/draw-count/proposal/acceptance/saved-configuration oracle. It is local repeatability evidence only.
- Mode 1/2 file existence/self-consistency is not independent Lanczos numerical parity.
- User reports test-fast process **9173** live and compiling during this repair. Exit/result/log remains pending and was not collected here. Compilation is not a passing comparison; a later pass still leaves these oracle/structure gaps open.

Future worker repairs remain under #181 after the slot opens. This repair changes no test source. New runner/CLI/callback/reducer/validation implementations are partial until execution and independent evidence are recorded. #183 gate-reporting changes do not complete #174–#182.

## Maintenance

Split results by API/model/mode/settings; retain failures, explicit skips, missing fixtures, ignored and feature-disabled paths separately. Use #174–#182 for scoped implementation/scenario work, #183 for gate reporting, #184 for inventory/provenance and #185 for umbrella coordination. Record C input-contract differences (including Julia's orbital file ordering restriction), Julia-deferred features and scoped exclusions explicitly. Never substitute self-comparison for independent expected results or tolerate RNG/trajectory drift.
