# Resident GPU model-family support matrix (#572)

This page lists the model families and options accepted or rejected by
the resident CUDA APIs: `mvmc_gpu_cuda::opt::run_gpu_para_opt_from_namelist`
([GPU-resident Opt](gpu-resident-opt.md)) and
`mvmc_gpu_cuda::physcal::run_gpu_phys_cal_from_namelist`
([resident GPU PhysCal](gpu-resident-physcal.md)). It compares each family with
the CPU C-order Rust runner and with the upstream Julia GPU extension. Treat it
as a gap analysis. It does not claim that a rejected family is easy to add, or
that Julia GPU supports it.

Sources used (checked 2026-11-04):

- Rust: `main` at `2f5755a5`, standalone workspace `gpu/mvmc-gpu-cuda`.
- Julia: the `extern/Julia-mVMC` pin `73140156ce813efb8b528516b4a9b5640247f23b`.
  "Julia GPU" means the CUDA extension `MVMCOptimizers.jl/ext/*.jl`.
  "Julia CPU" means `MVMCOptimizers.jl/src/*.jl`. The resident numerical
  references remain the recorded revisions `47452f90…` (Opt) and
  `61103114…` (PhysCal). A family taken from a newer pin must record its own
  revision.
- C: `extern/mVMC-1.4.0` (`v1.4.0`, `e83aa6cb`), `src/mVMC/*.c`.

Authority follows AGENTS.md. For a resident GPU path, Julia GPU is the accepted
numerical and RNG reference wherever it implements the feature. The CPU
C-order runner remains C-authoritative for physics. If Julia GPU does not
implement a family, a resident implementation is **new functionality**. Its
device RNG/proposal contract must extend the Julia GPU tape contract. Its
physics (projection factors, ratios, derivatives, energies) must be validated
against the C-order Rust runner, using deterministic per-configuration
quantities rather than sampled trajectories.

## Where rejection happens

Both APIs call `Model::prepare*` (`gpu/mvmc-gpu-cuda/src/opt/model.rs`) before
`with_runtime` opens a CUDA engine (`src/opt.rs:108-119`,
`src/opt/physcal.rs:44-73`). Every rejection in `model.rs` therefore happens on
the host, with no device context or allocation. The order is:

1. Shared CPU input validation: `mvmc_core::validation::validate_para_opt`
   (Opt) or `validate_phys_cal` (PhysCal), at `model.rs:61-68`.
2. The PhysCal Lanczos check, at `model.rs:63-66`.
3. A single resident-scope predicate in `prepare_input`. It returns
   `"resident GPU requires NExUpdatePath<=2 (3 with general orbitals),
   serial execution, and general orbitals for 2Sz != 0"`. Complex Opt with
   `NSRCG≠0` is rejected next (#678), then the FSZ subset (#714): an odd
   `2Sz`, `NSPGaussLeg>1` and Opt with `NSRCG≠0`.
4. Model construction checks in `from_data_with_derivatives`: the FSZ
   spin-sector check, the FSZ family check (complex parameters, RBM, OptTrans, DH2/DH4; #714),
   real/finite parameters (real mode), the complex-mode family check (RBM,
   OptTrans, DH2/DH4, local spins; #678), projection layout, complex QP
   weights and coefficients (real mode), and operator spin conservation for
   normal orbitals.
5. PhysCal measured-operator checks, in `Operators::prepare`
   (`physcal.rs:94-122`). These also run before `with_runtime`.

Capacity checks such as the shared-memory budget (`opt.rs:218-268`) run inside
`State::prepare`. That is after the engine is opened, but before any device
buffer exists.

Host-only tests (`cargo test --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --lib
opt::scope_tests`, no GPU required) cover every rejected family below. The
family tests also assert that the CPU validators accept the same input, so the
error is specific to the resident GPU path. They are in
[`src/opt/scope_tests.rs`](../../gpu/mvmc-gpu-cuda/src/opt/scope_tests.rs),
and `physcal/tests.rs:29` covers Green-function spin conservation.

## Matrix

Abbreviations: **S** supported, **R** rejected before device allocation,
**n/a** not an input of that path.

| Family / option | Resident Opt | Resident PhysCal | CPU C-order Rust | Julia GPU (pin `7314015`) | Authoritative reference for a GPU implementation | Fixture / invocation | Known restrictions |
|---|---|---|---|---|---|---|---|
| Real normal orbitals, Gutzwiller + Jastrow | **S** | **S** | S | **S**: `validate_cuda_data` accepts (`ext/cuda_state.jl:36-51`); projection tables `cuda_state.jl:109-111` | Julia GPU `47452f90` (Opt) / `61103114` (PhysCal) | `gpu/mvmc-gpu-cuda/tests/fixtures/opt_julia_l128.txt`, `opt_julia_operators.txt`, `physcal_julia_l128.txt` (provenance `opt_julia.md`, `physcal_julia.md`); `cargo test --release --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --lib opt:: -- --ignored` on a host with the CUDA libraries | An exactly singular QP plane recovers per plane while the total overlap stays finite and nonzero; a zero or nonfinite total is a sticky typed failure (#573). `2·NElec ≤ 1024` (`model.rs:150-151`, `pfaffian::MAX_N`). Sampler shared memory ≤ 48 KiB (`opt.rs:266`) |
| Hamiltonian: Transfer, CoulombIntra/Inter, Hund, PairHop, Exchange, InterAll | **S** for spin-conserving pairs, real coefficients in real mode and complex Transfer/InterAll coefficients in complex mode (#678); spin-flip Transfer and spin-changing InterAll **S** with general orbitals (#714); **R** spin-flip pairs with normal orbitals, a complex coefficient in real mode | same as Opt | S, including complex values; spin-changing InterAll only with `2Sz=-1` (`validation.rs` `validate_inter_all`, C `readdef.c:2865`) | **S** for the same subset; rejects complex/spin-flip Transfer and InterAll (`ext/cuda_observables.jl:22-23`, `:38-41`) | Julia GPU | `opt_julia_operators.txt` covers all 324 two-pair overlays on three L64 sites (`opt/tests.rs:40`); spin-flip pairs: `tests/fixtures/fsz/hubbard_fsz_L16` (#714) | Spin-flip operators need general orbitals (FSZ `rs` operator encoding) |
| Green functions (OneBodyG, TwoBodyG, TwoBodyGEx) | n/a | **S** for spin-conserving pairs; spin-flip pairs **S** with general orbitals (#714, C `calgrn_fsz.c`); **R** spin-flip pairs with normal orbitals (`Operators::prepare`) | S | **S** for spin-conserving pairs (`ext/cuda_observables.jl:48`) | Julia GPU, except the C-canonical TwoBodyGEx one-body list (see `gpu-resident-physcal.md`) | `physcal_julia_l128.txt`; `physcal/tests.rs:29` (host-only rejection) | Output is ensemble means, not C-format files |
| QP: `NSPGaussLeg`, `NSPStot`, `NMPTrans`, `TransSym` | **S**, `nq = NSPGaussLeg·\|NMPTrans\|` (`model.rs:170-178`); a negative `NMPTrans` is normalized to its absolute value as in Julia GPU (`model.rs:69-76`, `ext/cuda_runners.jl:29`) | **S** | S (C `qp.c:50-80`; anti-periodic sign from `readdef.c:920-925`) | **S** (`cuda_state.jl:98`) | Julia GPU | Only `NSPGaussLeg=8`, `NSPStot=0`, `\|NMPTrans\|=1` (L64/L128) have resident fixtures | **Gap:** no resident fixture for `NSPStot≠0` or `NSPGaussLeg=1`; `NMPTrans=2` is covered only by the complex fixtures (#678). A complex `ParaQPTrans` weight is **S** in complex mode (C `CalculateIP_fcmp`, #678) and **R** in real mode, where C (`qp_real.c:43`) and Julia GPU (`cuda_state.jl:70`, `:102`) take the real part without warning. Rust is deliberately stricter |
| `OptTrans` (variational QP, `NQPOptTrans>1`) | **S** for real weights (#674); complex weights **R** (real-parameter check) | **S** (#674), weights fixed at their normalized input values | S (`crates/mvmc-core/tests/opttrans.rs`, `tests/fixtures/opttrans/`). The CLI activates it with `-o` and optimizes the real weight components (#676; C's `GetInfoOptTrans` flag writes are a defect, tmisawa/Julia-mVMC#94) | **R**: "CUDA variational OptTrans is not supported" (`ext/cuda_state.jl:43`); Julia CPU only | No Julia GPU. C `qp.c` `UpdateQPWeight`, `slater.c` plane composition, `vmccal.c` `calculateOptTransDiff` and `parameter.c` normalization, through the Rust CPU C-order kernels; the sampler keeps the Julia GPU tape contract | `gpu/mvmc-gpu-cuda/tests/fixtures/opttrans/` (L16 doped with exchanges, L128 with the delayed queue; three nonidentity-permutation sectors); tests `opt::opttrans_tests`, `opt::physcal::opttrans_tests`, host-only `scope_tests`; provenance in its `README.md` | An `OptTrans` definition activates the weights and the resident SR optimizes their real components, as in Julia. C's `GetInfoOptTrans` never writes those flags (`readdef.c:2814`). `NQPFull = NQPFix * NQPOptTrans` planes multiply the sampler and inverse cost |
| Doublon-holon DH2 / DH4 | **S** (#665) | **S** (#665) | S (`dh2_runtime.rs`, `dh4_runtime.rs`, `tests/fixtures/dh2`, `dh4`) | **R** (`ext/cuda_state.jl:41-42`, `:48-49`); Julia CPU only | No Julia GPU. Rust CPU C-order (`projection.c`, `parameter.c`) for counters, ratios, derivatives, energies, Green functions and the DH gauge; the sampler keeps the Julia GPU tape contract | `tests/fixtures/dh2/production_real`, `dh4/production_dh4_real`, `gpu/mvmc-gpu-cuda/tests/fixtures/dh/` (L16 doped with exchanges, L128 with the delayed queue); tests `opt::dh_tests`, `opt::physcal::dh_tests`; provenance in its `README.md` | DH counters are recomputed over every (table, center) item per proposal and operator, `O((n2+n4) Nsite)`. SR activity and the GJ/DH2/DH4 gauge use the raw C `OptFlag` exactly-one contract (`stcopt.c:112`, `parameter.c:255-313`): flags `0`, `2`, `-1` and missing flags are fixed, as on the CPU (#700; host `scope_tests`, device `resident_opt_dh_noncanonical_flags_match_cpu_c_order`) |
| RBM (Charge/Spin/General layers) | **S** for real coefficients (#667); complex coefficients **R** (real-parameter check) | **S** (#667) | S (`rbm_production.rs`, `rbm_vs_julia.rs`, `c_rbm_*.rs`). The real path samples without the RBM factor, as C `vmcmake_real.c` does | **R**: "CUDA RBM is not supported" (`ext/cuda_state.jl:40`); Julia CPU only | No Julia GPU. C `rbm.c` on the complex normal path (`vmcmake.c`, `calham.c`, `calgrn.c`, `vmccal.c`) with real coefficients, through the Rust CPU C-order kernels; the sampler keeps the Julia GPU tape contract. C's real path omits the factor (tmisawa/Julia-mVMC#59) | `gpu/mvmc-gpu-cuda/tests/fixtures/rbm/` (L16 doped with exchanges, L128 with the delayed queue; all nine sections, tied indices); tests `opt::rbm_tests`, `opt::physcal::rbm_tests`, host-only `scope_tests`; provenance in its `README.md` | Complete section mappings and `Nneuron=0` (family counts only). Per-walker hidden fields `[walkers][Nneuron]` on the device; each proposal and operator costs `O(Nneuron)`, and derivatives `O(Nneuron Nsite)` per walker. Before allocation, `validate_rbm_tables` checks the family-tag segment against `0..=2` and the coefficient segments against `[-1, NRBM)` separately (#701) |
| SpinJastrow / other generalized Jastrow | n/a | n/a | **R** for every runner (`crates/mvmc-core/src/validation.rs:118-119`); a C 1.4.0 keyword (#84) that is not ported | n/a | n/a | n/a | Out of scope |
| Complex normal orbitals or parameters (`AllComplexFlag≠0`) | **S** (#678): Gutzwiller/Jastrow, complex QP weights and coefficients, `NExUpdatePath` 0-2, direct SR; **R** in complex mode: RBM, OptTrans, DH2/DH4, local spins, `NSRCG≠0` | **S** (#678), same families; `NSRCG` ignored | S | **R**: "CUDA runner requires real parameters" (`ext/cuda_state.jl:39`, `:108`); Julia CPU only | No Julia GPU. C's complex normal path (`CalculateMAll_fcmp`, `pfupdate.c`, `vmcmake.c`, `calham.c`, `locgrn.c`, `calgrn.c`, `SlaterElmDiff_fcmp`, `stcopt.c`/`stcopt_dposv.c`, `parameter.c`) through the Rust CPU C-order complex kernels; the sampler keeps the Julia GPU tape contract | `gpu/mvmc-gpu-cuda/tests/fixtures/complex/` (L16 doped with exchanges, L128; complex orbitals, Transfer phase, InterAll, `TransSym` weight `0.6+0.8i`); tests `opt::complex_tests`, `opt::physcal::complex_tests`, host-only `scope_tests`; provenance in its `README.md` | Immediate C rank-two updates only (no delayed queue or paired factor); complex SR solves C's `2*NPara` real system on stacked `[Re O; Im O]` rows. Sampler shared memory `(nqp+3n)·16 B ≤ 48 KiB` |
| `2Sz ≠ 0` | **S** with general orbitals (#714): even `2Sz`, C `makeInitialSample_fsz` spins (`Ne + 2Sz/2` up) and spin-conserving hops; **R** with normal orbitals (`model.rs`, C `readdef.c:691-696` requires general orbitals) | **S** / **R**, as Opt | S; the CPU validators also accept normal orbitals with `2Sz≠0`, which C rejects | **R**: "CUDA requires 2Sz=0" (`ext/cuda_state.jl:46`) | No Julia GPU. C FSZ path (see the next row) | `gpu/mvmc-gpu-cuda/tests/fixtures/fsz/kondo_fsz_2sz2` (8+8-site Kondo, `2Sz=2`, AP+P orbitals); normal rejection in `scope_tests.rs` | An odd `2Sz` other than -1 is rejected (C `readdef.c:697`) |
| General orbitals / FSZ (`OrbitalGeneral`, `OrbitalParallel`, `2Sz=-1`) | **S** for real parameters (#714): Gutzwiller/Jastrow, any `2Sz`, `NExUpdatePath` 0-3, local spins and local spin flips, spin-flip Transfer/InterAll, direct SR; **R** complex FSZ, RBM, DH2/DH4, OptTrans, `NSPGaussLeg>1`, `NSRCG≠0` | **S** (#714), same families, spin-flip Green functions | S (`real_fsz_*.rs`, `complex_fsz_sampling.rs`, `ctest_general_reference.rs`) | **R**: "CUDA requires sz-conserved orbitals" (`ext/cuda_state.jl:38`); Julia CPU only | No Julia GPU. C 1.4.0 real FSZ path (`UpdateSlaterElm_fsz`, `CalculateMAll_fsz_real`, `vmcmake_fsz_real.c`, `pfupdate_fsz_real.c`, `calham_fsz_real.c`, `locgrn_fsz_real.c`, `SlaterElmDiff_fsz`, `calgrn_fsz.c`) through the Rust CPU C-order FSZ kernels; the sampler extends the Julia GPU tape contract (`interval_kernels.cu`) | `gpu/mvmc-gpu-cuda/tests/fixtures/fsz/` (L16 doped `2Sz=-1` with spin-flip terms and exchanges, `2Sz=2` Kondo, L128 with the delayed queue, KondoGC and Heisenberg chains with local spin flips); tests `opt::fsz_tests`, `opt::physcal::fsz_tests`, host-only `scope_tests`; provenance in its `README.md` | Slots hold `rs = site + spin*Nsite`. Invalid proposals are skipped (symmetric proposals); C redraws until a candidate is valid. C forces `NSPGaussLeg=1` (`readdef.c:703-707`); the Rust parser does not, so the model requires it |
| Local spins + exchange proposals (`NLocalSpin>0`, `NExUpdatePath=1,2`) | **S** (#660); with general orbitals, the FSZ exchange of C `makeCandidate_exchange_fsz` (#714) | **S** (#660, #714) | S | **S**: `NExUpdatePath ∈ 0:2` (`ext/cuda_state.jl:47`), mapped to exchange probability 0 / 0.5 / 1 (`cuda_runners.jl:52-58`); local-spin initialization and validation (`cuda_initial_sampling.jl:13-33`, `:85-87`, `cuda_runners.jl:60-75`); local-spin-aware hopping and exchange proposals in the cached sampler (`cuda_sampling_update.jl:215-230`, `:264-290`). An exchange proposal recomputes the overlap cooperatively (`:327`), and an accepted exchange forces a refactor (`:341`) | Julia GPU `7314015`. The exchange decision uses tape word 3 (zero-based; `u32/2^32 < p`). That is the GPU contract and does not match the C `getUpdateType` `genrand_real2` draw (`vmcmake.c:797-825`) | New Julia GPU fixtures `gpu/mvmc-gpu-cuda/tests/fixtures/localspin/` (Heisenberg L64 all exchanges; Kondo 64+64 sites, `2Ne=128`, mixed moves, delayed queue), Opt and PhysCal flavors, provenance in its `README.md`; tests `opt::localspin_tests`, `opt::physcal::localspin_tests`; host-only `scope_tests` | Infeasible local-spin filling (`nlocal ≤ 2Ne ≤ 2Nsite−nlocal`, `cuda_initial_sampling.jl:87`) is rejected on the host. An accepted exchange costs one in-kernel `O(nq·n³)` refactor per walker |
| `NExUpdatePath=3` (KondoGC, local spin flip) | **S** with real general orbitals (#714 (b)), and path 2 with `2Sz=-1`: C `getUpdateType` as one tape uniform (hop 1/2, exchange 1/4, local spin flip 1/4; path 2 exchange or flip 1/2), C `makeCandidate_LocalSpinFlip_localspin`; **R** with normal orbitals | **S** / **R**, as Opt | S, general orbitals (C `vmcmake.c:811-817`) | **R** (`ext/cuda_state.jl:47`) | No Julia GPU. C FSZ path (`vmcmake_fsz_real.c`) through the Rust CPU C-order FSZ kernels | `gpu/mvmc-gpu-cuda/tests/fixtures/fsz/kondo_fsz_kgc` (8+8 sites), `kondo_fsz_kgc_L128` (64+64, delayed queue), `heisenberg_fsz_L16` (path 2, `2Sz=-1`); `native13_initialization/inputs/KondoChain_fsz` stays rejected (complex orbitals) | A local spin flip is a one-slot move with a projection ratio of exactly one |
| CG SR (`NSRCG≠0`) | **S** (#670, #682): device-resident C 1.4.0 `StochasticOptCG_real` plain CG with the breakdown/limit status and `NSRCGAbortOnFail` (`opt/cg.cu`); DiagScale (`NSRCG=2`, `useDiagScale`), `NSRCGFallback` and `RescaleSmat` **R** (#694) | **S**: accepted and ignored, as in Julia GPU (#572 milestone 0) | S (`sr_cg.rs`, `pfaffian_cg.rs`) | Opt **R** (`ext/cuda_runners.jl:27-28`, `:154`). **PhysCal accepts and ignores `NSRCG`** (`cuda_runners.jl:27`: `physcal \|\| nsrcg == 0`) | Opt: no Julia GPU; C `stcopt_cg_impl.c` through the Rust CPU `SampledSrOperator`, applied to the GPU derivatives. PhysCal: Julia GPU | Benchmark L16/L128 chains and the DH/RBM L16 fixtures with `NSRCG 1` (copied with edited `modpara.def`); tests `opt::cg_tests`, host-only `scope_tests` `resident_accepts_nsrcg_for_physcal_and_opt` | Validation rules and bounds in `gpu-resident-opt.md` (SR-CG section). `NSRCG≠0` with `NStore=0` stays rejected by the shared validator. The host enqueues `NSROptCGMaxIter` (or `NPara`) iterations; after the device stops, each remaining iteration costs three no-op launches |
| Grouped / MPI (`NSplitSize>1`) | **R** `model.rs:85` | **R** | S with an MPI group communicator (`grouped_nsplit_349.rs`) | **R**: "CUDA runner is serial" (`ext/cuda_state.jl:45`) | No Julia GPU | Baseline `NSplitSize 2` (`scope_tests.rs`) | Single-device, single-process design; multi-GPU is a separate design |
| Lanczos (`NLanczosMode>0`) | n/a (Opt is rejected by the shared validator, `validation.rs:218-221`) | **R** `model.rs:63-66` | S, PhysCal (`lanczos.rs`, `issue181_lanczos_native_c.rs`, `lanczos_transfer_physcal.rs`) | **R**: "CUDA Lanczos is not supported" (`ext/cuda_state.jl:44`) | No Julia GPU. C-order `physcal_lanczos.c` | Baseline `NVMCCalMode 1`, `NLanczosMode 1` (`scope_tests.rs`) | Needs H\|ψ⟩ Green-function accumulation on the device |
| Singular / near-singular QP planes | **S** per plane (#573): device validity mask, in-kernel refactor on accept, full-Pfaffian proposals/energy, exact-minor derivatives; zero or nonfinite total overlap is a sticky typed failure; nonfinite proposals rejected (#564) | **S** per plane (full-Pfaffian observables) | follows C: a failed `CalculateMAll` drops the sample (`vmccal.c:135-138`) | **S** per plane: validity (`ext/cuda_batched_inverse.jl:68`), forced refactor (`cuda_sampling_update.jl:343-371`), fallback observables (`cuda_observables.jl:109-122`, `:283-295`), minor derivatives (`cuda_sr.jl:154-187`); rejects zero or nonfinite initial overlaps (`cuda_runners.jl:62-63`) | Julia GPU `7314015` (deliberate divergence from C, `gpu-resident-opt.md`) | `interval/resident/tests.rs` and `opt/tests.rs` #573 tests (ports of `nodal_derivatives.jl`, `batched_inverse.jl`, `paired_sampling_inverse.jl`, `cached_sampling.jl`) | Fallbacks cost `O(n^3)` per term or minor on one block; for exceptional walkers only. See the gate below |

## Recommended milestone order

Each milestone adds one family to both APIs, unless it says otherwise. It
needs its own accepted implementation issue (per #572) and must meet the
#572 test criteria: fixed-tape, initialization, derivative, energy and SR
checks; repeated 20-step results; residual checks; same-implementation
repeatability; and the first numerical divergence. The L128 real baseline and
its zero-copy residency must not regress.

0. **Done — housekeeping (not a family): allow `NSRCG≠0` in resident PhysCal.**
   Julia GPU PhysCal accepts it (`cuda_runners.jl:27`). The shared predicate
   now checks `m.nsrcg` only for Opt, and `scope_tests.rs` covers both routes.
   It is host-only.
1. **Done in #660 — local spins + exchange proposals (`NExUpdatePath=1,2`, real normal
   orbitals; Heisenberg and normal Kondo).** This is the only rejected family
   that Julia GPU already implements, so independent tapes, initial
   configurations and stage values can come from Julia GPU, as for L128. The
   cost is moderate:
   - a local-spin mask buffer;
   - the local-spin initial-configuration rule;
   - empty-destination hopping restricted to sites without a local spin;
   - two-electron exchange proposals scored by full overlap recomputation, the
     Julia contract, which can reuse the existing rebuild/refactor kernels
     instead of a new rank-four update;
   - the exchange occupancy/count update.

   The tape word is already reserved. Open questions:
   - Delayed (Opt) and immediate (PhysCal) queues must flush around an
     exchange.
   - A large local-spin Julia GPU fixture has to be generated, because Julia
     GPU has stage-level coverage only (`parity.jl` 6-site).
2. **Done in #665 — DH2/DH4 (extra real projection terms).** No Julia GPU reference exists.
   The Slater, Pfaffian and sampler kernels are unchanged, so the work is
   limited to projection counters, log-ratios and derivative columns. These
   can be checked deterministically against the C-order projection
   (`dh2_runtime.rs`, `dh4_runtime.rs` fixtures). The sampler keeps the Julia
   GPU tape contract.
3. **Done in #667 — RBM (real).** This is similar to DH, but needs device theta caches and
   more derivative blocks. It has no Julia GPU reference, and C-order RBM is
   the physics reference.
4. **Done in #670 — CG SR for Opt (`NSRCG≠0`).** It has no Julia GPU reference;
   C `stcopt_cg_impl.c` defines the algorithm. The legacy stage-route kernels
   (`sr_device.rs`, `sr_kernels.cu`, removed in #643, kept in `cffc3dc5`) were
   not reused: they read every `ddot` on the host, one synchronization per
   iteration. The new kernels keep the stop decision on the device.
5. **Done in #674 — OptTrans.** QP weights become optimized parameters after
   Slater. The device refreshes the full QP weights after every SR update.
   It has no Julia GPU reference; C `qp.c`/`vmccal.c`/`parameter.c` define the
   physics.
6. **Done in #678 — complex normal orbitals / complex parameters.** Complex
   rebuild Pfaffians/inverses, a complex sampler, complex local energies,
   Green functions and derivatives, and C's `2*NPara` real SR system. It has
   no Julia GPU reference; C's complex normal path defines the physics. RBM,
   OptTrans, DH2/DH4, local spins and SR-CG remain rejected in complex mode.
7. **General orbitals / FSZ / `2Sz≠0` / `NExUpdatePath=3` (#714).** These
   need spin-flip moves and operators. This is the largest milestone, with
   no Julia GPU reference, split into sub-milestones:
   - **(a) Done — real FSZ / general orbitals, any `2Sz`.** `2Sz≠0` is folded
     into (a): with `rs`-encoded slots it only changes the initial spin
     labels, and the spin-conserving FSZ hop serves every even `2Sz`.
   - **(b) Done — local spin flips:** `NExUpdatePath=3` and path 2 with
     `2Sz=-1`.
   - (c) Complex FSZ, if feasible.

There is no milestone for `NSplitSize>1` or Lanczos. The resident design is a
single-process, single-device design, and Lanczos is a separate PhysCal
measurement design. Both stay **R** until an accepted design exists.

This order differs from the evaluation order suggested in #572 (projection,
RBM/DH, complex/FSZ, CG, exchange/local-spin). It puts exchange/local-spin
first because that is the only family with an existing Julia GPU reference. A
measured user demand may reorder milestones 2-7.

### Gate: #573 (singular QP policy)

#573 is a prerequisite gate for every family milestone.

- #573 adopted the Julia GPU per-plane policy. A singular plane is marked
  invalid, and its walker uses full-Pfaffian and exact-minor fallbacks. A zero
  or nonfinite total overlap fails with a typed sticky status
  (`resident_status.cuh`), and nothing is published.
- Exchange moves, local spins and spin projection (`NSPGaussLeg>1`) increase
  the chance of reaching an exactly singular rotated plane.
- FSZ/general orbitals and OptTrans change the plane structure itself.

So no milestone may define its own recovery for singular planes. Each new
update path must keep the validity mask, send invalid walkers to the
fallbacks or the in-kernel refactor, and state with a test how it propagates
the status. The zero-pivot and status semantics that milestone 1 depends on
are now fixed.
