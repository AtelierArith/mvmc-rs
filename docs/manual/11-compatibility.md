# 11. Compatibility and differences

[Contents](README.md) · Previous: [10. Tutorial](10-tutorial.md) · Next: [Appendix: how the citations were checked](appendix-checks.md)

## 11.1 Authority and function correspondence

The Rust port follows the **Julia design** (public API, runner structure, lifecycle, tests) and the **C numerical contract**
(parameter layout, RNG initialization and draw order, arithmetic order, signs, file formats); when C and Julia disagree, Rust keeps the Julia architecture and adopts the C
behaviour ([AGENTS.md](../../AGENTS.md)). The implementations correspond as follows (C names from `extern/mVMC-1.3.0/src/mVMC`, Julia names from
`extern/Julia-mVMC/docs/src/en/compatibility.md`, Rust paths under `crates/mvmc-core/src`):

| C function | Julia | Rust |
|------------|-------|------|
| `VMCParaOpt` | `vmc_para_opt!` | `run::vmc_para_opt` |
| `VMCPhysCal` | `vmc_phys_cal!` | `run::vmc_phys_cal_in_place_timed` |
| `VMCMainCal`, `VMCMainCal_fsz` | `vmc_main_cal!`, `vmc_main_cal_fsz!` | `run::accumulate_observables_local` |
| `VMCMakeSample`, `_real`, `_fsz` | `vmc_make_sample!` | `sampling::driver::vmc_make_sample*` |
| `InitParameter`, `SyncModifiedParameter` | `init_parameter!`, `sync_modified_parameter!` | `parameter_init::init_parameter` / `sync_modified_parameter` |
| `ReadInitParameter` | `read_initial_def!` | `initial_params::read_initial_def` |
| `ReadInputParameters` | `read_input_parameters!` | `mvmc-expert-parsers` |
| `CalculateGreenFunc` | `calculate_green_func!` | `observables::ordinary_green_values` |
| `StochasticOpt`, `StochasticOptCG` | `stochastic_opt!`, CG | `sr::stochastic_opt_*`, `sr_cg::stochastic_opt_cg_with_reducer` |
| `CalculateMAll_*` | `calculate_m_all.jl` | `pfaffian::calc_m_all_*` |

Chapters 2-6 give the line-level map for every equation.

## 11.2 What Rust takes from Julia, and where Julia-only behaviour is not copied

| Topic | Julia-mVMC | `mvmc-rs` |
|-------|-----------|-----------|
| Optional inner threading | `JULIA_MVMC_INNER_THREADS=1`, `JULIA_NUM_THREADS` | `MVMC_RS_INNER_THREADS` (count) and `MVMC_RS_INNER_THRESHOLD` ([8.5](08-running.md#85-environment-variables)) |
| MPI detection | `JULIA_MVMC_MPI=1` and launcher variables | launcher variables only (`LaunchContext::from_env`); `JULIA_MVMC_MPI` appears only in a test helper |
| Debug dumps | `MVMC_DEBUG_*` | not implemented |
| Timer output | `MVMC_C_TIMER=1` | same variable, same file `zvo_CalcTimer.dat`, plus `MVMC_*_DIAG` families |
| Unsupported options | BackFlow, spin Jastrow, `NSplitSize>1` with CG (undefined in C: stored `O` is read from unwritten memory), `NSRCG>=2`, `useDiagScale`, `RescaleSmat`, FSZ Lanczos (also rejected by C) | the same list is rejected by `validation.rs` ([7.5](07-input-files.md#75-supported-and-rejected-inputs)) |
| High-level runner | `run_para_opt_from_namelist(...; nsteps, mode, nsmp, output_dir, seed, initial_def)` | `run::run_para_opt_from_namelist` with `RunConfig` (same arguments); CLI wraps it |
| `1e-14` Slater amplitude cutoff | applied in the historical Julia table build | **not** applied (C reads the declared value, `slater_update.rs:46`) |
| Output names during optimization | `zvo_out.dat`, `zvo_var.dat` | same (differs from C, [9.1](09-output-files.md#91-energy-per-step-zvo_outdat-opt-and-zvo_out_nnndat-phys)) |
| Julia `exp`, `log`, `hypot`, `sin/cos`, complex division | used by Julia kernels | ported (`julia_exp`, `julia_log`, `julia_hypot`, `julia_trig`, `julia_complex`) where Julia order is kept; C-style variants (`c_complex`, `*_c_compat`) where C order is required |

## 11.3 Known differences from the C reference

Reproduced exactly: the SFMT-19937 generator and its seeding (`RndSeed + group`), the conversion functions (`genrand_real2`), the number and order of draws on every
control path including rejected moves, the file formats of [chapter 9](09-output-files.md) (byte-level for the formats that are written), the parameter layout, and the algorithm
structure of [chapters 2-6](02-theory-vmc-hamiltonian.md).

Differences that a user can observe:

| # | Area | C | Rust |
|---|------|---|------|
| 1 | Driver options | `-b -h -m -o -F -e -s -v` | `-b -h -m -o -F -e -s -v` and the positional `initpara` are implemented (`-m` rejects `N <= 0`, where C divides by `N`, [MultiDef mode](08-running.md#multidef-mode--m)); extra Rust long options ([8.1](08-running.md#81-the-mvmc-command)) |
| 2 | Standard mode / StdFace | built in (`-s`) | `mvmc -s` / `--dry-run` for the lattices in [7.6](07-input-files.md#76-standard-mode-stdface); the others are not yet ported |
| 3 | `zvo_out`/`zvo_var` during optimization | `zvo_out_NNN.dat`, `zvo_var_NNN.dat` | `zvo_out.dat`, `zvo_var.dat` |
| 4 | `zvo_SRinfo.dat` | written for the direct and CG solvers | CG only |
| 5 | `zvo_time_NNN.dat`, `zvo_varbin_NNN.dat` | written (binary with `-b`) | both written (`-b`: C header, complete `2*NPara` blocks instead of C's truncated ones, [8.1](08-running.md#binary-output--b)) |
| 6 | Timer file prefix | `CDataFileHead` | always `zvo` |
| 7 | Parser defaults of `modpara.def` | `SetDefaultValuesModPara` | `ModParaParameters::default` ([7.2](07-input-files.md#72-modparadef)) |
| 8 | `NMPTrans = 0` | not rejected by the reader (the default 0 gives `NQPFix = 0`, `readdef.c:778`) | rejected |
| 9 | `NSROptItrSmp > NSROptItrStep` | leaves rows unwritten | rejected |
| 10 | Lanczos | any of the supported orbital/term sets of C (incl. `InterAll`) | no FSZ (C also rejects it), no `InterAll`, no spin-changing `Trans`; `NSplitSize>1` is supported as in C; a failed $\alpha$ writes `NaN` |
| 11 | `TwoBodyGEx` in optimization | read and ignored by the optimizer | read and ignored (accepted by `validate_para_opt`, as by PhysCal) |
| 12 | BackFlow (`BF`, `NProjBF`) | supported | rejected in optimization; **not checked** in PhysCal ([11.5](#115-open-observations)) |
| 13 | `NSRCG >= 2`; `useDiagScale`, `RescaleSmat` | `NSRCG != 0` selects CG (`vmcmain.c:485`); the other two keys do not exist in C 1.3.0 (Julia-mVMC options) | `NSRCG >= 2` and nonzero `useDiagScale`/`RescaleSmat` are rejected |
| 14 | `exp`, `cosh`, `hypot`, `sin/cos` | libm / C99 complex | Julia-compatible ports (last-bit differences possible) |
| 15 | OpenMP reduction order in the local energy and `NProj` sums | thread-dependent | fixed sequential order |
| 16 | Variance column when $\lvert\langle H\rangle\rvert\le10^{-14}$ | $\pm\infty$/NaN | `0.0` (optimization output only) |
| 17 | Real-mode LTL/inverse operation order | C PFAPACK (`DSKR2` add-then-subtract) | the optimizer uses Julia's order (`dsktf2`, `utu2inv_real`, `crates/mvmc-core/src/pfaffian.rs:693,706`); C-order variants `dsktf2_c_compat` and `utu2inv_real_c_compat` exist in `crates/pfapack` (PR #334) but are not selected by the runner |
| 18 | Hund / Exchange / CoulombInter rows | one term per row | same; `PairHop` rows expanded to both orders in both |
| 19 | MPI Green-function reduction | `MPI_Reduce` to rank 0 | all-reduce, root writes |
| 20 | Negative `DSROptStepDt` | sets `SRFlag = 1` ("Diagonalization Mode" remark, header `sEigenMax sEigenMin`) and replaces `dt` by `-dt` (`readdef.c:746-752`) | not handled: a negative value is used as given, which reverses the direction of the SR step |
| 21 | Key for $N_e$ in `modpara.def` | `Nelectron`, `Ne` | `NElec`, `Nelec` (the C spellings are ignored) |

## 11.4 Numerical comparison policy

The detailed policy is [docs/NUMERICAL_COMPARISONS.md](../NUMERICAL_COMPARISONS.md) (issues #186/#190); the rules a reader needs are:

- **Never compare computed floating-point results bitwise**, including Rust against C. A comparison of `actual` and `expected` passes when
  $|{\rm actual}-{\rm expected}|\le{\rm abs}+{\rm rel}\cdot\max(|{\rm actual}|,|{\rm expected}|)$ with bounds that the test justifies by the operation, the problem scale, the
  conditioning and the solver residual; complex values are compared componentwise. NaN matches only NaN, infinities must match with sign.
- **Exact** (not toleranced): seed resolution, RNG initialization and state, 624-word blocks, draw order and draw count on a fixed control path, indices, flags,
  dimensions and discrete input contracts.
- Small floating-point differences can flip a Metropolis decision; the subsequent trajectory, saved configurations, solver termination and the final RNG state may then
  differ between implementations or platforms. This is allowed *after* the first numerical divergence has been located; it never allows a different RNG algorithm or a missing draw on
  an identical control path. Do not reseed or average runs to hide a defect.
- Long (20-step) runs are checked for **repeatability** (same implementation, same seed), short prefixes against independent reference checkpoints.
- Reference environment: Linux x86_64 (Dev Container on other hosts); macOS is a portability check. Julia references use Julia 1.13.1 with
  `extern/Julia-mVMC/Manifest-v1.13.toml`; record Julia/BLAS versions with generated fixtures. C-derived expected values are generated separately and checked in under
  `tests/fixtures/` with provenance; ordinary Rust tests never invoke C or Julia and never read `c_toolbox/`.

## 11.5 Open observations

Items noticed while writing this manual, to be handled by separate issues. "Observed" means reproduced with a release build of `mvmc`; "code reading" means seen in the sources but not exercised.

1. **Two-electron update, complex normal path.** C defines `rsbOld = raOld + t*Nsite` in all four two-electron update functions (`pfupdate_two_fcmp.c:227`, `pfupdate_two_real.c:227`,
   `pfupdate_two_fsz.c:232`, `pfupdate_two_fsz_real.c:218`), i.e. it uses the *old site of electron a* for the second index of the old matrix element $M^{\rm old}_{ab}$. The Rust *real* update
   preserves this deliberately (`crates/mvmc-core/src/sampling/updates.rs:588`), while the Rust *complex* normal update
   `update_m_all_two_complex_flat` uses the old site of electron b (`crates/mvmc-core/src/sampling/updates.rs:537`), as Julia does. Whether this changes complex exchange updates numerically was **not tested** (code reading).
2. **PhysCal does not check namelist sections.** `validate_phys_cal` omits the keyword check of `validate_para_opt`. With a `BF bf.def` line in `namelist.def`, optimization stops with `unsupported namelist section BF`, whereas
   `--physcal` runs to completion and BackFlow is not applied (observed). Unknown keywords, `SpinJastrow` and unknown `In…` keywords are likewise not rejected in PhysCal.
3. **`TwoBodyGEx` in optimization** is accepted and ignored, as in C (it was rejected with a stale "issue #30" message until #349).
4. **`--help` text** claims the default output directory is "namelist parent dir"; the code uses `<namelist parent>/output` (observed).
5. **`--mode`** is only a validated label; a mismatching label (for example `--mode fsz` on a real model) is silently accepted (observed).
6. **Timer file name** ignores `CDataFileHead` ([9.5](09-output-files.md#95-timers)).
7. **Direct-solver `zvo_SRinfo.dat`** is not written ([9.4](09-output-files.md#94-solver-information-zvo_srinfodat)).
8. **OptTrans derivative layout in real mode** (decision recorded in #370). C `calculateOptTransDiff` (`vmccal.c:639`) is called with a `double complex *` pointer offset (`vmccal.c:466`), so derivative *i* lands at complex slot `offset + i`, not at parameter slot *i*. In a real-mode run with `NQPOptTrans = 3` the native step-1 operands (`tests/fixtures/c_order_sr_operands/opt_real-cg-store0.txt`, observed) show derivative 1 missing, derivative 2 stored in the slot of derivative 1 and the last slot zero. Rust `opt_trans_diff` (`observables.rs:957`) keeps the mathematically correct layout, pinned by `real_mode_opttrans_derivatives_use_their_own_slots` (`sum_i w_i O_i = 1`). The defect is reported as tmisawa/Julia-mVMC#55; the two affected trailing entries of every operand array are excluded from the C comparison ([NUMERICAL_COMPARISONS](../NUMERICAL_COMPARISONS.md)). Complex mode was not examined.
9. **Lanczos failure output** differs ([6.2](06-theory-observables-lanczos.md#62-the-single-step-lanczos-wave-function)).
10. **Rust `modpara.def` defaults** differ from C ([7.2](07-input-files.md#72-modparadef)); results of a run that omits keys are therefore not interchangeable between the two programs.
11. **Negative `DSROptStepDt`** is turned into a positive step with a remark by C (`readdef.c:746-752`) but used unchanged by Rust (code reading, no use of `SRFlag` in `crates/`).
12. **`Nelectron`/`Ne` are ignored by the Rust `modpara.def` parser** (observed: a model with `Nelectron 8` and no `Ncond` reports `Nelec=0` and fails with "normal initialization precondition: ..."); the C manual's own example uses `Nelectron`.
13. **C ignores RBM parameters in real mode** (#379, reported as tmisawa/Julia-mVMC#59). With a real model (`ComplexType 0` orbitals) that declares RBM sections, C accepts `FlagRBM=1` silently, but the real sampler `vmcmake_real.c` contains no RBM code (0 occurrences of `RBM`; the complex `vmcmake.c` has 27), so the RBM weight is never applied. Observed on `tests/fixtures/c_orbital_inputs/namelist_rbm_real.def` (seed 1, step 1): C reports NPara = 55 (NProj 7, NRBM 36, NSlater 12) and a step-1 energy of 5.984544891656925, unchanged when the RBM overlay is changed from 0.125/-0.25 to all zeros and bit-identical to Rust with all RBM values zero. Rust applies the RBM weight (energy 6.228711216019723). Rust keeps the correct RBM math; `rbm_real` therefore has no native C operand reference.

## 11.6 What this manual did not verify

MPI execution (no launcher was available), macOS behaviour, the `simd-backend` build of `mvmc-cli`, the example programs under `crates/mvmc-cli/examples`, OptTrans (`-o`) runs,
FSZ/general-orbital runs, RBM runs, complex-mode runs, and numerical parity of any of the above with C or Julia. See [the appendix](appendix-checks.md).
