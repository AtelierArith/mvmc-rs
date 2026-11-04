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
- C: `extern/mVMC-1.3.0` (`d73d06bd`), `src/mVMC/*.c`.

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
3. A single resident-scope predicate, at `model.rs:81-94`. It returns
   `"resident GPU requires real normal orbitals, hopping-only sampling, no local
   spins/RBM/DH/OptTrans, NSRCG=0 and serial execution"`.
4. Model construction checks in `from_data_with_derivatives`: real/finite
   parameters `:155`, projection layout `:165`, complex QP weights `:184`,
   operator spin conservation `:343`/`:360`.
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
| Real normal orbitals, Gutzwiller + Jastrow | **S** | **S** | S | **S**: `validate_cuda_data` accepts (`ext/cuda_state.jl:36-51`); projection tables `cuda_state.jl:109-111` | Julia GPU `47452f90` (Opt) / `61103114` (PhysCal) | `gpu/mvmc-gpu-cuda/tests/fixtures/opt_julia_l128.txt`, `opt_julia_operators.txt`, `physcal_julia_l128.txt` (provenance `opt_julia.md`, `physcal_julia.md`); `cargo test --release --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --lib opt:: -- --ignored` on a host with the CUDA libraries | Each QP plane must stay nonsingular; failure is a sticky device status (#573). `2·NElec ≤ 1024` (`model.rs:150-151`, `pfaffian::MAX_N`). Sampler shared memory ≤ 48 KiB (`opt.rs:266`) |
| Hamiltonian: Transfer, CoulombIntra/Inter, Hund, PairHop, Exchange, InterAll | **S** for real, spin-conserving pairs; **R** spin-flip Transfer `model.rs:343`, spin-changing InterAll `model.rs:360`, complex coefficient `model.rs:92`/`:184` | same as Opt | S, including complex values; spin-changing InterAll only with `2Sz=-1` (`validation.rs` `validate_inter_all`, C `readdef.c:2451`) | **S** for the same subset; rejects complex/spin-flip Transfer and InterAll (`ext/cuda_observables.jl:22-23`, `:38-41`) | Julia GPU | `opt_julia_operators.txt` covers all 324 two-pair overlays on three L64 sites (`opt/tests.rs:40`) | Spin-flip operators belong to the general-orbital family |
| Green functions (OneBodyG, TwoBodyG, TwoBodyGEx) | n/a | **S** for spin-conserving pairs; **R** `physcal.rs:101`, `:107`, `:121` | S | **S** for spin-conserving pairs (`ext/cuda_observables.jl:48`) | Julia GPU, except the C-canonical TwoBodyGEx one-body list (see `gpu-resident-physcal.md`) | `physcal_julia_l128.txt`; `physcal/tests.rs:29` (host-only rejection) | Output is ensemble means, not C-format files |
| QP: `NSPGaussLeg`, `NSPStot`, `NMPTrans`, `TransSym` | **S**, `nq = NSPGaussLeg·\|NMPTrans\|` (`model.rs:170-178`); a negative `NMPTrans` is normalized to its absolute value as in Julia GPU (`model.rs:69-76`, `ext/cuda_runners.jl:29`) | **S** | S (C `qp.c:50-80`; anti-periodic sign from `readdef.c:742-747`) | **S** (`cuda_state.jl:98`) | Julia GPU | Only `NSPGaussLeg=8`, `NSPStot=0`, `\|NMPTrans\|=1` (L64/L128) have resident fixtures | **Gap:** no resident fixture for `NSPStot≠0`, `NMPTrans>1` or `NSPGaussLeg=1`. A complex `ParaQPTrans` weight is **R** at `model.rs:184`, while Julia GPU takes the real part without warning (`cuda_state.jl:70`, `:102`). Rust is deliberately stricter |
| `OptTrans` (variational QP, `NQPOptTrans>1`) | **R** `model.rs:89` | **R** `model.rs:89` | S (`crates/mvmc-core/tests/opttrans.rs`, `tests/fixtures/opttrans/`) | **R**: "CUDA variational OptTrans is not supported" (`ext/cuda_state.jl:43`); Julia CPU only | No Julia GPU. Physics from C-order (C `qp.c` `UpdateQPWeight`) | `tests/fixtures/opttrans/run_opt_real` (rejection test); a resident fixture would be new | QP weights become optimized parameters, which changes the SR layout |
| Doublon-holon DH2 / DH4 | **R** `model.rs:90-91` (defensive layout check `:164-165`) | **R** | S (`dh2_runtime.rs`, `dh4_runtime.rs`, `tests/fixtures/dh2`, `dh4`) | **R** (`ext/cuda_state.jl:41-42`, `:48-49`); Julia CPU only | No Julia GPU. C-order projection (`projection.c`) for counters, ratios and derivatives | `tests/fixtures/dh2/production_real`, `dh4/production_dh4_real` (rejection test) | Slater part is unchanged |
| RBM (Charge/Spin/General layers) | **R** `model.rs:88` | **R** | S (`rbm_production.rs`, `rbm_vs_julia.rs`, `c_rbm_*.rs`) | **R**: "CUDA RBM is not supported" (`ext/cuda_state.jl:40`); Julia CPU only | No Julia GPU. C-order (`rbm.c`) | Baseline + `ChargeRBM_PhysLayer` (`scope_tests.rs`) | Needs hidden-unit theta caches on the device |
| SpinJastrow / other generalized Jastrow | n/a | n/a | **R** for every runner (`crates/mvmc-core/src/validation.rs:118-119`); not a C 1.3.0 keyword | n/a | n/a | n/a | Out of scope |
| Complex normal orbitals or parameters | **R** `model.rs:92` (also `:155`, `:184`) | **R** | S | **R**: "CUDA runner requires real parameters" (`ext/cuda_state.jl:39`, `:108`); Julia CPU only | No Julia GPU. C-order complex paths | Baseline with orbital `ComplexType 1` (`scope_tests.rs`) | Needs complex Pfaffian/inverse sampling, complex SR (Hermitian Cholesky) and complex derivatives |
| `2Sz ≠ 0` | **R** `model.rs:82` | **R** | S | **R**: "CUDA requires 2Sz=0" (`ext/cuda_state.jl:46`) | No Julia GPU | Baseline with `2Sz 2` (`scope_tests.rs`) | Belongs with FSZ/general orbitals |
| General orbitals / FSZ (`OrbitalGeneral`, `OrbitalParallel`, `2Sz=-1`) | **R** `model.rs:81` | **R** | S (`real_fsz_*.rs`, `complex_fsz_sampling.rs`, `ctest_general_reference.rs`) | **R**: "CUDA requires sz-conserved orbitals" (`ext/cuda_state.jl:38`); Julia CPU only | No Julia GPU. C-order (`slater_fsz.c`, `vmcmake_fsz.c`) | `tests/fixtures/native13_initialization/inputs/HubbardChain_fsz` (rejection test) | Needs spin-flip moves and operators |
| Local spins + exchange proposals (`NLocalSpin>0`, `NExUpdatePath=1,2`) | **R** `model.rs:83`, `:86-87`; the sampler also guards `interval.rs:240-245` | **R** | S | **S**: `NExUpdatePath ∈ 0:2` (`ext/cuda_state.jl:47`), mapped to exchange probability 0 / 0.5 / 1 (`cuda_runners.jl:52-58`); local-spin initialization and validation (`cuda_initial_sampling.jl:13-33`, `:85-87`, `cuda_runners.jl:60-75`); local-spin-aware hopping and exchange proposals in the cached sampler (`cuda_sampling_update.jl:215-230`, `:264-290`). An exchange proposal recomputes the overlap cooperatively (`:327`), and an accepted exchange forces a refactor (`:341`) | Julia GPU. The exchange decision uses tape word 4 (`u32/2^32 < p`). That is the GPU contract and does not match the C `getUpdateType` `genrand_real2` draw (`vmcmake.c:605-627`) | Julia GPU tests: stage level on the 6-site `heisenberg_chain_real` with exchange 0/0.5/1 (`test/gpu/parity.jl:11`, `:64`); synthetic cached sampler at ns 64/128 (`cached_sampling.jl:22-40`); local-spin initialization (`initial_sampling.jl:50`, `:59`). There is no large local-spin end-to-end Julia GPU runner test (`large_runners.jl` uses Hubbard L128). Rust rejection test: `tests/fixtures/original_heisenberg_parser_184` plus baseline `NExUpdatePath 1` | The Rust 5-word tape already reserves word 3 (zero-based) for the exchange threshold (`interval_kernels.cu:27-28`) |
| `NExUpdatePath=3` (KondoGC, local spin flip) | **R** `model.rs:83` | **R** | S, general orbitals (C `vmcmake.c:620-626`) | **R** (`ext/cuda_state.jl:47`) | No Julia GPU | `native13_initialization/inputs/KondoChain_fsz` (rejection test) | Requires general orbitals |
| CG SR (`NSRCG≠0`) | **R** `model.rs:84` | **R** `model.rs:84` | S (`sr_cg.rs`, `pfaffian_cg.rs`) | Opt **R** (`ext/cuda_runners.jl:27-28`, `:154`). **PhysCal accepts and ignores `NSRCG`** (`cuda_runners.jl:27`: `physcal \|\| nsrcg == 0`) | Opt: no Julia GPU, numerical algorithm from C (`stcopt_cg.c`). PhysCal: Julia GPU | Baseline `NSRCG 1` (`scope_tests.rs`) | **Gap:** resident PhysCal rejects `NSRCG≠0` even though it never runs SR. Julia GPU accepts it. Relaxing the check only for PhysCal is a one-line, host-only change |
| Grouped / MPI (`NSplitSize>1`) | **R** `model.rs:85` | **R** | S with an MPI group communicator (`grouped_nsplit_349.rs`) | **R**: "CUDA runner is serial" (`ext/cuda_state.jl:45`) | No Julia GPU | Baseline `NSplitSize 2` (`scope_tests.rs`) | Single-device, single-process design; multi-GPU is a separate design |
| Lanczos (`NLanczosMode>0`) | n/a (Opt is rejected by the shared validator, `validation.rs:218-221`) | **R** `model.rs:63-66` | S, PhysCal (`lanczos.rs`, `issue181_lanczos_native_c.rs`, `lanczos_transfer_physcal.rs`) | **R**: "CUDA Lanczos is not supported" (`ext/cuda_state.jl:44`) | No Julia GPU. C-order `physcal_lanczos.c` | Baseline `NVMCCalMode 1`, `NLanczosMode 1` (`scope_tests.rs`) | Needs H\|ψ⟩ Green-function accumulation on the device |
| Singular / near-singular QP planes | runtime failure (sticky device status); nonfinite proposals rejected (#564) | runtime failure | follows C; no resident-specific policy | rejects zero or nonfinite initial overlaps (`ext/cuda_runners.jl:62-63`); a zero or nonfinite candidate Pfaffian forces a refactor (`cuda_sampling_update.jl:342-345`); no cofactor path | #573 | #573 | See the gate below |

## Recommended milestone order

Each milestone adds one family to both APIs, unless it says otherwise. It
needs its own accepted implementation issue (per #572) and must meet the
#572 test criteria: fixed-tape, initialization, derivative, energy and SR
checks; repeated 20-step results; residual checks; same-implementation
repeatability; and the first numerical divergence. The L128 real baseline and
its zero-copy residency must not regress.

0. **Housekeeping (not a family): allow `NSRCG≠0` in resident PhysCal.**
   Julia GPU PhysCal accepts it (`cuda_runners.jl:27`). The change moves
   `m.nsrcg != 0` out of the shared predicate for PhysCal and updates
   `scope_tests.rs`. It is host-only.
1. **Local spins + exchange proposals (`NExUpdatePath=1,2`, real normal
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
2. **DH2/DH4 (extra real projection terms).** No Julia GPU reference exists.
   The Slater, Pfaffian and sampler kernels are unchanged, so the work is
   limited to projection counters, log-ratios and derivative columns. These
   can be checked deterministically against the C-order projection
   (`dh2_runtime.rs`, `dh4_runtime.rs` fixtures). The sampler keeps the Julia
   GPU tape contract.
3. **RBM (real).** This is similar to DH, but needs device theta caches and
   more derivative blocks. It has no Julia GPU reference, and C-order RBM is
   the physics reference.
4. **CG SR for Opt (`NSRCG=1`).** It has no Julia GPU reference. C defines the
   algorithm. Device CG kernels exist on the legacy stage route
   (`sr_device.rs`). Removing that route (#643) may delete them, so reuse
   must be decided under #643.
5. **OptTrans.** QP weights become optimized parameters, which changes the
   sparse basis and SR layout. It has no Julia GPU reference.
6. **Complex normal orbitals / complex parameters.** Complex sampling
   Pfaffians, complex SR (Hermitian solve) and derivatives are needed. This is
   large, with no Julia GPU reference.
7. **General orbitals / FSZ / `2Sz≠0` / `NExUpdatePath=3`.** These need
   spin-flip moves and operators. This is the largest milestone, with no
   Julia GPU reference.

There is no milestone for `NSplitSize>1` or Lanczos. The resident design is a
single-process, single-device design, and Lanczos is a separate PhysCal
measurement design. Both stay **R** until an accepted design exists.

This order differs from the evaluation order suggested in #572 (projection,
RBM/DH, complex/FSZ, CG, exchange/local-spin). It puts exchange/local-spin
first because that is the only family with an existing Julia GPU reference. A
measured user demand may reorder milestones 2-7.

### Gate: #573 (singular QP policy)

#573 is a prerequisite gate for every family milestone.

- Today a singular or nonfinite QP plane fails the run with a sticky device
  status, and nothing is published.
- Exchange moves, local spins and spin projection (`NSPGaussLeg>1`) increase
  the chance of reaching an exactly singular rotated plane.
- FSZ/general orbitals and OptTrans change the plane structure itself.

So no milestone may define its own recovery for singular planes. Each one
inherits the current hard-failure semantics until #573 accepts a policy. It
must also state, with a test, how its new update paths propagate the status.
Milestone 1 should start only after #573 has fixed the zero-pivot and
status semantics, because its exchange/refactor path is the first new
consumer of those semantics.
