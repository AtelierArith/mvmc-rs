# Julia-mVMC → Rust port — planning record

This document is the single source of truth for the port. It captures
(a) the answered / assumed clarifications, (b) the phased plan, (c) the
module-by-module mapping, and (d) the risk register.

## 1. Pre-flight clarifications

| # | Question | Status | Resolution |
|---|---|---|
| 1 | Where does the Rust workspace live? | **Assumed** | `extern/Julia-mVMC-rs/` (this directory). Co-locates fixtures with the Julia source for CI; revisit if the team wants a sibling repo. |
| 2 | Fidelity target: bit-exact C-parity vs. idiomatic rewrite? | **Assumed** | **Bit-exact** — matches Julia-mVMC v0.1's headline guarantee. Floating-point summation order, BLAS call sequence, RNG draw order, and `init_*` phase order all preserved verbatim. |
| 3 | Scope ceiling — v0.1 only or also v0.2+? | **Assumed** | v0.1 only for now (`VMCParaOpt` + partial `VMCPhysCal` + Lanczos step-0). BackFlow / MPI / full Lanczos are listed under Phase 7 (stretch). |
| 4 | BLAS / LAPACK backend? | **Assumed** | Vendored OpenBLAS via `openblas-src` (static). Tolerances upstream were tuned vs. OpenBLAS; MKL / Accelerate paths can be added as features later. |
| 5 | License posture — GPL workspace OK? | **Assumed** | GPL-3.0-or-later workspace, with `sfmt19937` carved out as BSD-3 and `pfapack` carved out as BSD-3 + per-file MPL-2.0 (matches upstream `THIRD_PARTY_LICENSES.md`). |
| 6 | PfaPack scope (user-specified) | **Confirmed** | Port only the **pure-Julia subset** of `PfaPack.jl`: `pfaffian.jl`, `ltl_decomposition.jl`, `utu2.jl`. The FFI shims `c_wrapper.jl` (ccall to `libltl2inv`) and `fortran_wrapper.jl` (ccall to `libzsktf2` / `libdsktf2`) are not ported; their bundled C++ / Fortran sources are out of scope. The optimizer's hot path already calls the pure-Julia routines (`calculate_m_all.jl:202` -> `utu2inv!`, not `cimpl_utu2inv!`), so the FFI surface is dead code there. |

Override #1–5 by editing this table and re-running `xtask` / CI; #6 is
the user's binding decision.

## 2. Phased plan

| Phase | Deliverable | Gate | Status |
|---|---|---|---|
| 0 | Workspace skeleton, CI scaffold, `deny.toml`, planning doc. | `cargo check --workspace` green. | ✅ |
| 0.5 | SFMT fixture / cross-language reference dumper at `extern/Julia-mVMC/tools/dump_sfmt_c_reference.jl`. Loads the same `libsfmt.dylib` that Julia-mVMC uses via `ccall`, so the values it prints are *by construction* the values mVMC C consumes. The dumper is the executable proof for the three-way equivalence `C-mVMC == Julia-mVMC == Rust sfmt19937`. The PfaPack analogue at `extern/Julia-mVMC/tools/dump_pfapack_reference.jl` writes LTL / Pfaffian / inverse text fixtures under `extern/Julia-mVMC-rs/tests/fixtures/pfapack/`, consumed by `crates/pfapack/tests/golden_vs_julia.rs`. The Phase-4 `zvo_out.dat` gate is now covered by `crates/mvmc-core/tests/phase4_zvo_gate.rs`, which compares the bundled C reference for `heisenberg_chain_real`. | Dumpers run offline and emit the values consumed by the matching `tests/golden_vs_*.rs` files. | ✅ (SFMT + PfaPack + Phase-4 `zvo_out.dat` gate). |
| 1 | `sfmt19937` — SFMT19937 state machine, `gen_rand32` / `gen_rand64` / `genrand_real2` / `init_by_array` / biased modulo / non-destructive `dump_rand32`. | Bit-equal to C reference for: 6 init seeds × 8 u32 draws, three `gen_rand_all` refill boundaries (~1999 draws), 8 `gen_rand64` draws, **4 `genrand_real2` IEEE-754 bit patterns** (the VMC sampler hot path), and **3 `init_by_array` key shapes**. Cross-language equivalence proved by `dump_sfmt_c_reference.jl` running against the same `libsfmt` the Julia port loads. | ✅ |
| 2 | `pfapack` — `pfaffian`, `zsktf2` / `dsktf2`, `utu2pfa` / `utu2inv`. | LTL / Pfaffian / inverse match Phase-0.5 goldens to 1e-13 (Real Pfaffian), 1e-12 (Complex Pfaffian), 1e-14 (Real LTL), 1e-13 (Complex LTL), and 1e-11 (inverse, both kinds) on `n in {4, 6, 16}` cases. | ✅ |
| 3 | `mvmc-expert-parsers` — 15 parsers + 8 utils + 23 types + `ExpertModeData`. | Round-trip parse `extern/Julia-mVMC/examples/inputs/*/namelist.def`. | ✅ (12 parsers + 1 util + 11 types covering the four upstream cases; RBM / doublon-holon / OptFlag tracking parked for Phase 4.) |
| 4 | `mvmc-core` in order: 4.1 `state` + layout newtypes + `Reducer` -> 4.2 `pfaffian` -> 4.3 `sampling` -> 4.4 `observables` -> 4.5 `sr` -> 4.6 `sync` / `qp` / `average` / `counter` -> 4.7 `io` -> 4.8 `run`. | 10-step `zvo_out.dat` bit-parity for `heisenberg_chain_real`. | ✅ Complete. 4.1–4.8 are implemented, including QPTrans-aware Slater rebuild, normal-mode real/complex sampling drivers with burn-sample reuse, Heisenberg/Hubbard observables, Slater SR derivative block, direct SR solver, sync/qp/average/counter helpers, `zvo_out.dat`/`zvo_var.dat`/`zqp_opt.dat` writers, `vmc_para_opt`, `run_para_opt_from_namelist`, `read_initial_def`, and the gate test `crates/mvmc-core/tests/phase4_zvo_gate.rs` against the bundled C `heisenberg_chain_real/zvo_out_first10.dat` reference. Complex rank-2 Exchange and FSZ are now wired. The 10-step bit-parity gates `crates/mvmc-core/tests/phase4_zvo_gate_cmp.rs` (`heisenberg_chain_cmp`) and `crates/mvmc-core/tests/phase4_zvo_gate_fsz.rs` (`heisenberg_chain_fsz`) both pass against the bundled C references. BackFlow / full Green-function measurement remain Phase 7 stretch scope. |
| 5 | `mvmc-cli` + sample inputs port. | All 4 upstream models pass 10-step bit-parity; 50-step regression within 1e-8. | ✅ Complete. `green_func1` (non-FSZ 1-body Green function for Transfer/hopping terms) implemented in `observables.rs`; `hubbard_chain_real` 10-step bit-parity gate added (`phase5_zvo_gate_hubbard.rs`). 50-step regression (`phase5_regression_50step.rs`) compares all 4 models against Julia-mVMC reference (`reference/<model>/zvo_out_first50.dat`) within 1e-8. Reference files generated by `tools/dump_zvo_50step_reference.jl`. `mvmc-cli` binary wired: argument parsing (`--nsteps`, `--out-dir`, `--seed`, `MVMC_NSTEPS` env), modpara pre-parse for model info / default nsteps, calls `mvmc_core::run_para_opt_from_namelist`, writes output files, prints final energy. Usage: `cargo run -p mvmc-cli -- <namelist.def> [--nsteps N] [--out-dir DIR] [--seed N]`. |
| 6 | Performance: `rayon` over `qpidx`; SIMD on 2–3 inner loops if profile shows headroom. | Within 1.5x of upstream Julia on `heisenberg_chain_real` 50-step. |
| 7 (stretch) | `mpi` feature via `Reducer`, full `VMCPhysCal` Green-function pipeline, full Lanczos, BackFlow. | Per-feature acceptance test. |

## 3. Module map

### `sfmt19937` (BSD-3-Clause)

| Rust target | Upstream Julia |
|---|---|
| `Sfmt19937Rng` | `SFMT.jl/src/SFMT.jl` + `C_API.jl` |
| `dump_rand32` | `sfmt_dump_rand32` from `SFMT.jl` |
| `gen_real2` | `genrand_real2` from `C_API.jl` |
| `gen_rand_mod` | `gen_rand32() % n` pattern in `vmc_sampling.jl` |
| `FALLBACK_SEED = 11272` | `run_para_opt_from_namelist.jl` |

### `pfapack` (BSD-3-Clause + MPL-2.0)

| Rust target | Upstream Julia | License |
|---|---|---|
| `pfaffian::pfaffian_ltl_{complex,real}` | `PfaPack.jl/src/pfaffian.jl` | BSD-3 |
| `ltl::{zsktf2, dsktf2}` | `PfaPack.jl/src/ltl_decomposition.jl` | BSD-3 |
| `utu2::{utu2pfa_*, utu2inv_*}` | `PfaPack.jl/src/utu2.jl` | MPL-2.0 |
| (intentionally not ported) | `c_wrapper.jl`, `fortran_wrapper.jl` | n/a |

### `mvmc-expert-parsers` (GPL-3.0-or-later)

| Rust target | Upstream Julia |
|---|---|
| `types::ExpertModeData` | `MVMCExpertModeParsers.jl/src/types/expert_types.jl` |
| `parsers::{modpara, locspin, trans, coulomb, hund, exchange, pairhop, interall, gutzwiller, jastrow, orbital, green, qptrans, rbm, doublon_holon}` | `MVMCExpertModeParsers.jl/src/parsers/*.jl` (15 files) |
| `utils::{file, validation, parameter_init, read_input_parameters, qp_weight, opt_flag, orbital_qptrans}` | `MVMCExpertModeParsers.jl/src/utils/*.jl` (8 files) |
| `c_const::{RNG_REAL2_INV, D_AMP_MAX, GAUSSLEG_EPS, RBM_INIT_{REAL,COMPLEX}_SCALE}` | scattered across `parameter_init.jl`, `qp_weight.jl`, `vmc_sampling.jl` |

### `mvmc-core` (GPL-3.0-or-later)

| Rust target | Upstream Julia |
|---|---|
| `state` (+ `SlaterElmFlat`, `InvMColMajor` newtypes) | `MVMCOptimizers.jl/src/{types,workspace}.jl` |
| `reducer::{Reducer, SingleProcessReducer}` | (new — abstraction over `parameter_sync.jl` / `counter.jl` MPI no-ops) |
| `pfaffian` | `MVMCOptimizers.jl/src/calculate_m_all.jl` |
| `sampling` | `MVMCOptimizers.jl/src/{slater_update,vmc_sampling}.jl` (~7.7k LOC) |
| `observables` | `MVMCOptimizers.jl/src/{vmc_main_cal,green_func_calc}.jl` |
| `sr` | `MVMCOptimizers.jl/src/stochastic_opt.jl` (dpotrf + dpotrs, not dposv) |
| `sync`, `qp`, `average`, `counter` | `MVMCOptimizers.jl/src/{parameter_sync,qp_weight_update,weight_average,counter}.jl` |
| `io` | `MVMCOptimizers.jl/src/data_io.jl` |
| `run::{vmc_para_opt, vmc_phys_cal, run_para_opt_from_namelist, read_initial_def}` | `MVMCOptimizers.jl/src/{vmc_para_opt,vmc_phys_cal,run_para_opt_from_namelist,initial_params}.jl` |

## 4. Risk register

| Risk | Mitigation |
|---|---|
| BLAS backend variance (1e-10 tolerance was tuned vs. OpenBLAS only). | Pin OpenBLAS via `openblas-src` static. Document tolerance assumption. |
| `parameter_init.jl` opt-flag index `2*i + 2*NProj + 2*FlagRBM*NRBM` (off-by-one silently desynchronises Slater RNG draws). | Phase-0.5 fixture: first 100 Slater values per model. Catches the bug at Phase 4.1, not Phase 5. |
| `slater_elm` row-major vs. `inv_m` column-major confusion. | `SlaterElmFlat<T>` / `InvMColMajor<T>` newtypes in Phase 4.1; raw `ndarray::Array2` not exposed across module boundaries. |
| C `%.18e` vs Rust `{:.18e}` byte differences (negative zero, denormal padding, exponent width). | Parse-then-compare for `zvo_out.dat`; `format_c_double` helper + golden tests for `zqp_opt.dat`. |
| Pfaffian zero / odd-N return contract — must be a numeric zero, not `Option` / `Result`, because `vmc_sampling` divide-by-zero checks assume it. | API contract documented in `pfapack::pfaffian` rustdoc + an explicit test in Phase 2. |
| MPI gap (Phase 7 may stay unimplemented). | `Reducer` trait exists from Phase 4.1, so today's no-op stays correct; document `NSplitSize > 1` as silently single-process (matches Julia behaviour). |
| Fixture licensing — `test/integration/reference/*` is GPL-3 C-mVMC. | Keep GPL-3 fixtures inside `mvmc-core` integration tests only, never inside `pfapack` (BSD/MPL) or `sfmt19937` (BSD) tests. |

## 5. Tenferro migration status/rules

- `Vec` / slice remains the compatibility boundary first. Convert to explicit col-major `TypedTensor` only at bounded seams, never as a blanket replacement for hot paths.
- `SlaterElmFlat` keeps the legacy row-major logical layout per QP plane. The public offset stays `(qp * n_site2 + row) * n_site2 + col`, even though its tenferro shape is `[n_site2 * n_site2, n_qp_full]`.
- `InvMColMajor` keeps the legacy logical layout `qp * (n_size * n_size + 1) + row + col * n_size`, including the per-QP pad slot, even though its tenferro shape is `[n_size * n_size + 1, n_qp_full]`.
- Production hot paths must not introduce per-sample `Vec -> TypedTensor` conversion or fresh backend creation. The Slater derivative QP reduction uses one scratch tensor for the existing col-major accumulation buffer, a borrowed view over the per-sample QP weights, a reusable `ConcreteEinsumPlan`, and a preallocated weighted output tensor.
- Current tenferro-einsum code covers `finalize_oo_store`, the narrow SR store helper contract, and the Slater derivative QP reduction. Tests cover col-major copy-back, borrowed-weight `einsum_into`, and legacy real/imag output layout.
- Any future conversion must keep the Julia/C parity tests and `phase5_regression_50step` as gates.

### Raw-index audit summary

The `rg -n "\\[[^\\]]+\\*[^\\]]+\\]|\\+ .*\\* n_|\\* n_size \\+|\\* n_site2 \\+|sample \\*|qp \\*" crates/mvmc-core/src` audit still finds raw index math in three categories:

- Intentional typed layout APIs/tests: `state.rs` accessors and assertions for `SlaterElmFlat`, `InvMColMajor`, and sample-backed state buffers; these document the public logical layouts and packed storage.
- Deliberate hot-path scalar loops: `sampling/updates.rs`, `slater_update.rs`, `observables.rs`, `sampling/driver.rs`, and related `Vec`-backed loops; these stay scalar until the corresponding buffers move to tensors.
- Future candidates: non-hot-path layout helpers that already mirror a contiguous plane or sample store and could be tensor-backed later if the regression gates stay green.
- Migration bugs: no remaining audit hit is classified as a migration bug in this pass.

## 6. Quick reference: where to look in upstream

- Optimizer entrypoint and phase order: `MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl:65-100`.
- RNG-draw order critical comments: `MVMCExpertModeParsers.jl/src/utils/parameter_init.jl` (line ~158).
- Pfaffian rank-1/rank-2 update hot loop: `MVMCOptimizers.jl/src/vmc_sampling.jl:1750-1810`.
- Slater-element construction hot loop: `MVMCOptimizers.jl/src/calculate_m_all.jl` (search for `slt_offset`).
- Integration test tolerance policy: `extern/Julia-mVMC/test/integration/runtests.jl:21-30`.
- Storage-order convention for `inv_m`: `MVMCOptimizers.jl/src/vmc_sampling.jl:3236` comment.
